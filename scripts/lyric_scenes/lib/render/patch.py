"""Restorable scene patches from a small JSON spec — shared by qa/time_ab.py (variants) and render/frames.py
(per-camera still overrides; replaces boss_fight render_stills' hard-coded EXPO dict and per-file time_ab setv() ifs).

Spec (every key optional; the whole thing is a plain dict, e.g. one entry of variants.json):
    hide_collections / show_collections : [collection names]          → collection.hide_render
    hide_objects / show_objects         : [object names]
    hide_prefix / hide_contains         : [str]   objects whose name starts with / contains
    hide_types                          : ["LIGHT", "VOLUME", ...]
        Hidden objects that have KEYED hide_render get that F-curve MUTED (else the render re-evaluates the key and
        un-hides them); muting is restored afterwards, nothing is deleted.
    eevee / render / view_settings / cycles : {attr: value}         → scene.<block>.<attr> = value
    set                                 : {"scene.eevee.volumetric_end": 400, "objects['Terrain'].hide_render": true}
                                          paths starting "scene." resolve from the scene, others from bpy.data
    exposure_offset                     : float, ADDED to scene.view_settings.exposure
    glare_quality                       : "Low"|"Medium"|"High" on every compositor Glare node

apply(scene, spec) -> Undo;  undo() restores every touched value in reverse order. The .blend on disk is never written.
`resolve()`, `parse_path()` and `normalize_spec()` are pure Python (unit-tested without Blender).
"""
import json
import os
import re

KNOWN = {"hide_collections", "show_collections", "hide_objects", "show_objects", "hide_prefix", "hide_contains",
         "hide_types", "eevee", "render", "view_settings", "cycles", "set", "exposure_offset", "glare_quality"}
_TOK = re.compile(r"""\.?([A-Za-z_][A-Za-z0-9_]*)|\[\s*(?:'([^']*)'|"([^"]*)"|(-?\d+))\s*\]""")


def parse_path(path):
    """"objects['A b'].modifiers[0].show_render" → [("attr","objects"),("key","A b"),("attr","modifiers"),
    ("idx",0),("attr","show_render")]. Raises ValueError on anything else (no eval, ever)."""
    out, pos = [], 0
    path = path.strip()
    while pos < len(path):
        m = _TOK.match(path, pos)
        if not m or m.end() == pos:
            raise ValueError(f"bad path {path!r} at {pos}")
        if m.group(1):
            out.append(("attr", m.group(1)))
        elif m.group(4) is not None:
            out.append(("idx", int(m.group(4))))
        else:
            out.append(("key", m.group(2) if m.group(2) is not None else m.group(3)))
        pos = m.end()
    if not out or out[-1][0] != "attr":
        raise ValueError(f"path must end in an attribute: {path!r}")
    return out


def resolve(root, path):
    """Walk all but the last token from `root`; return (owner, attr_name)."""
    toks = parse_path(path)
    o = root
    for kind, v in toks[:-1]:
        o = getattr(o, v) if kind == "attr" else o[v]
    return o, toks[-1][1]


def load_spec(s):
    """JSON text, a path to a JSON file, or already a dict → dict ('' / None → {})."""
    if not s:
        return {}
    if isinstance(s, dict):
        return s
    if os.path.isfile(s):
        with open(s) as f:
            return json.load(f)
    return json.loads(s)


def normalize_spec(spec):
    """Validate keys, turn comma strings into lists. Raises KeyError on an unknown key (typos must not pass silently)."""
    spec = dict(spec or {})
    bad = set(spec) - KNOWN
    if bad:
        raise KeyError(f"unknown patch keys {sorted(bad)}; known: {sorted(KNOWN)}")
    for k in ("hide_collections", "show_collections", "hide_objects", "show_objects", "hide_prefix", "hide_contains",
              "hide_types"):
        v = spec.get(k)
        if isinstance(v, str):
            spec[k] = [x for x in v.split(",") if x]
    return spec


class Undo:
    def __init__(self):
        self._ops = []

    def setattr(self, obj, attr, value):
        old = getattr(obj, attr)
        if hasattr(old, "copy"):              # mathutils Vector/Color/Matrix are LIVE views of the data — snapshot
            old = old.copy()
        elif hasattr(old, "__len__") and not isinstance(old, str) and hasattr(old, "foreach_get"):
            old = tuple(old)                  # bpy_prop_array
        self._ops.append((obj, attr, old))
        setattr(obj, attr, value)

    def __call__(self):
        for obj, attr, old in reversed(self._ops):
            try:
                setattr(obj, attr, old)
            except (ReferenceError, AttributeError):
                pass
        self._ops.clear()

    def __len__(self):
        return len(self._ops)


def _keyed_fcurves(idb, data_path):
    """F-curves animating `data_path` on idb (5.x slotted actions, any layer/strip; falls back to legacy .fcurves)."""
    ad = getattr(idb, "animation_data", None)
    act = ad.action if ad else None
    if act is None:
        return []
    out = []
    layers = getattr(act, "layers", None)
    if layers is not None and len(layers):
        slot = ad.action_slot
        for lay in layers:
            for st in lay.strips:
                for cb in getattr(st, "channelbags", []):
                    if slot is None or cb.slot == slot:
                        out += [fc for fc in cb.fcurves if fc.data_path == data_path]
    elif hasattr(act, "fcurves"):
        out += [fc for fc in act.fcurves if fc.data_path == data_path]
    return out


def _hide(undo, ob, hide=True):
    undo.setattr(ob, "hide_render", hide)
    for fc in _keyed_fcurves(ob, "hide_render"):
        undo.setattr(fc, "mute", True)


def apply(scene, spec):
    """Apply a patch spec to `scene` (in memory). Returns an Undo callable. Unknown names raise KeyError."""
    import bpy
    D = bpy.data
    spec = normalize_spec(load_spec(spec))
    u = Undo()
    for name in spec.get("hide_collections", []):
        u.setattr(D.collections[name], "hide_render", True)
    for name in spec.get("show_collections", []):
        u.setattr(D.collections[name], "hide_render", False)
    for name in spec.get("hide_objects", []):
        _hide(u, D.objects[name])
    for name in spec.get("show_objects", []):
        _hide(u, D.objects[name], False)
    pre, sub, typ = tuple(spec.get("hide_prefix", [])), spec.get("hide_contains", []), set(spec.get("hide_types", []))
    if pre or sub or typ:
        for ob in D.objects:
            if (pre and ob.name.startswith(pre)) or any(s in ob.name for s in sub) or ob.type in typ:
                _hide(u, ob)
    for block in ("eevee", "render", "view_settings", "cycles"):
        for k, v in spec.get(block, {}).items():
            u.setattr(getattr(scene, block), k, v)
    for path, v in spec.get("set", {}).items():
        owner, attr = resolve(scene, path[len("scene."):]) if path.startswith("scene.") else resolve(D, path)
        u.setattr(owner, attr, v)
    if spec.get("exposure_offset"):
        u.setattr(scene.view_settings, "exposure", scene.view_settings.exposure + float(spec["exposure_offset"]))
    if spec.get("glare_quality"):
        tree = getattr(scene, "compositing_node_group", None) or getattr(scene, "node_tree", None)
        for n in (tree.nodes if tree else []):
            if n.bl_idname == "CompositorNodeGlare":
                if "Quality" in n.inputs:                       # 5.x: Quality is an input socket
                    u.setattr(n.inputs["Quality"], "default_value", spec["glare_quality"])
                else:
                    u.setattr(n, "quality", spec["glare_quality"].upper())
    return u


def camera_spec(cam_ob, overrides=None):
    """Per-still patch for a camera: the camera object's own custom props (exposure_offset float, hide_collections /
    show_collections / hide_prefix comma strings, patch = JSON string) merged under `overrides` (dict)."""
    spec = {}
    for k in ("exposure_offset", "hide_collections", "show_collections", "hide_prefix", "hide_objects"):
        if k in cam_ob.keys():
            v = cam_ob[k]
            spec[k] = float(v) if k == "exposure_offset" else str(v)
    if "patch" in cam_ob.keys():
        spec.update(json.loads(cam_ob["patch"]))
    spec.update(overrides or {})
    return normalize_spec(spec)
