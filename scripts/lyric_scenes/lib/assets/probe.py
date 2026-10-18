"""Probe models BEFORE using them: per-mesh bbox in metres, poly counts, materials (KHR_materials_unlit flag, missing /
relative textures), armatures + actions, scale sanity — as JSON. Replaces boss_fight/probe_assets.py + probe_pose.py,
the 9 inspect*.py / insp_*.py copies and v3_tools/probe*.py. Blender script; ALWAYS through the capped slice:

    scripts/server/render.sh t-probe --wait --collect -P ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python /ABS/scripts/lyric_scenes/lib/assets/probe.py -- --out /ABS/probe.json FILE.glb [FILE.gltf FILE.fbx ...] \
        [--strict] [--max-meshes 200]

Each file is imported into an EMPTY factory scene through lib/importer.import_model (same import path the builders use,
so the unlit / bad-image detection is the one that will fire at build time). --strict exits 1 if any file has a flag
(unlit materials, missing/relative textures, implausible size). One summary line per file on stdout.
Facing is NOT inferred: check it with a +Y marker render or lib/checks.direction_check (police cars ×2, 10-12).
"""
import argparse
import json
import os
import sys
import time

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(LIB))

import numpy as np  # noqa: E402

TINY_M, HUGE_M = 0.05, 150.0       # a centimetre-unit human imports at ~180 m; real buildings > 150 m just get a hint


# ------------------------------------------------------------------------------------------------ pure

def scale_flags(size_m):
    """Plausibility of an import's overall size (metres) → list of flag strings (cm/mm-unit exports, empty files)."""
    if not size_m:
        return ["no mesh geometry"]
    L = max(size_m)
    if L <= 0:
        return ["zero-size geometry"]
    if L < TINY_M:
        return [f"tiny: longest side {L:.4g} m — millimetre/metre mix-up? (×1000 → {L * 1000:.3g} m)"]
    if L > HUGE_M:
        return [f"huge: longest side {L:.4g} m — centimetre units? (×0.01 → {L * 0.01:.3g} m)"]
    return []


def summary_line(path, rec):
    s = rec.get("size_m") or [0, 0, 0]
    t = rec.get("totals", {})
    return (f"[probe] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: "
            f"{len(rec.get('meshes', []))} meshes, {t.get('polys', 0)} polys, size {s[0]:.3g}×{s[1]:.3g}×{s[2]:.3g} m "
            f"(longest {rec.get('longest_axis')}), {len(rec.get('materials', []))} mats, "
            f"{len(rec.get('armatures', []))} rigs, {len(rec.get('actions', []))} actions"
            + (f"  FLAGS: {'; '.join(rec['flags'])}" if rec.get("flags") else ""))


# ------------------------------------------------------------------------------------------------ bpy

def _mesh_rec(o):
    from lib import importer
    me = o.data
    w = importer.world_co(o) if len(me.vertices) else np.zeros((0, 3))
    lo, hi = (w.min(0), w.max(0)) if len(w) else (np.zeros(3), np.zeros(3))
    lt = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", lt)
    return dict(name=o.name, parent=o.parent.name if o.parent else None, verts=len(me.vertices), polys=len(me.polygons),
                tris=int((lt - 2).sum()), lo=[round(float(x), 4) for x in lo], hi=[round(float(x), 4) for x in hi],
                dims=[round(float(x), 4) for x in hi - lo], materials=[m.name if m else None for m in me.materials],
                uv_layers=len(me.uv_layers), shape_keys=len(me.shape_keys.key_blocks) if me.shape_keys else 0,
                skinned=any(m.type == "ARMATURE" for m in o.modifiers) or (o.parent is not None and o.parent.type == "ARMATURE"),
                scale=[round(float(x), 4) for x in o.matrix_world.to_scale()])


def _material_rec(m):
    from lib import importer
    ims = importer._images_of([m])
    bad = dict()
    for nm, why in importer.bad_images(ims):
        bad.setdefault(nm, []).append(why)
    types = {n.type for n in m.node_tree.nodes} if m.node_tree else set()
    return dict(name=m.name, unlit=importer.is_unlit(m), principled="BSDF_PRINCIPLED" in types,
                emission="EMISSION" in types, render_method=getattr(m, "surface_render_method", None),
                images=[dict(name=im.name, path=im.filepath, packed=im.packed_file is not None,
                             colorspace=im.colorspace_settings.name, problems=bad.get(im.name, [])) for im in ims])


def _action_rec(a, objs):
    nfc = sum(len(cb.fcurves) for ly in a.layers for st in ly.strips for cb in st.channelbags)
    users = [o.name for o in objs if o.animation_data and o.animation_data.action == a]
    return dict(name=a.name, frame_range=[round(float(x), 2) for x in a.frame_range],
                slots=[s.identifier for s in a.slots], fcurves=nfc, bound_to=users)


def probe_file(path, max_meshes=200):
    """Import `path` into a fresh empty scene → record dict."""
    import bpy
    from lib import importer
    bpy.ops.wm.read_factory_settings(use_empty=True)
    t0 = time.time()
    objs = importer.import_model(path, cache=False, fix_unlit="report", quiet=True)
    log = importer.LOG[-1]
    # bone-display widgets the glTF importer adds (an 'Icosphere' per rig — boss v1 deleted them by name) are not
    # part of the asset: identified as pose-bone custom shapes, listed apart, kept out of size/poly numbers
    widgets = {pb.custom_shape for o in objs if o.type == "ARMATURE" for pb in o.pose.bones if pb.custom_shape}
    meshes = [o for o in objs if o.type == "MESH" and o not in widgets]
    mrec = [_mesh_rec(o) for o in meshes]
    mats = importer._materials_of([o for o in objs if o not in widgets])
    pts = [importer.world_co(o) for o in meshes if len(o.data.vertices)]
    size = None
    if pts:
        P = np.concatenate(pts)
        size = [round(float(v), 3) for v in P.max(0) - P.min(0)]
    axis = "XYZ"[int(np.argmax(size))] if size else None
    rec = dict(path=os.path.abspath(path), format=os.path.splitext(path)[1].lower().lstrip("."),
               import_s=round(time.time() - t0, 2), objects=len(objs), size_m=size, longest_axis=axis,
               longest_m=max(size) if size else None,
               totals=dict(verts=sum(r["verts"] for r in mrec), polys=sum(r["polys"] for r in mrec),
                           tris=sum(r["tris"] for r in mrec)),
               meshes=sorted(mrec, key=lambda r: -r["polys"])[:max_meshes], meshes_truncated=max(0, len(mrec) - max_meshes),
               materials=[_material_rec(m) for m in mats],
               armatures=[dict(name=o.name, bones=len(o.data.bones),
                               roots=[b.name for b in o.data.bones if b.parent is None][:10]) for o in objs if o.type == "ARMATURE"],
               actions=[_action_rec(a, objs) for a in bpy.data.actions],
               bone_widgets=sorted(w.name for w in widgets),
               scaled_objects=log["scaled_objects"], other_types=sorted({o.type for o in objs} - {"MESH", "ARMATURE", "EMPTY"}))
    flags = [f"unlit (KHR_materials_unlit — self-lit, casts no shadow): {log['unlit']}"] if log["unlit"] else []
    flags += [f"image {nm}: {why}" for nm, why in log["bad_images"]]
    flags += scale_flags(size)
    rec["flags"] = flags
    return rec


def main(argv=None):
    import bpy
    argv = sys.argv[sys.argv.index("--") + 1:] if argv is None and "--" in sys.argv else (argv or [])
    ap = argparse.ArgumentParser(prog="probe.py")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true", help="exit 1 if any file has a flag")
    ap.add_argument("--max-meshes", type=int, default=200)
    a = ap.parse_args(argv)
    res = {"blender": bpy.app.version_string, "files": {}}
    for p in a.files:
        p = os.path.abspath(p)
        try:
            rec = probe_file(p, a.max_meshes)
        except Exception as e:                      # one broken file must not lose the others' results
            rec = dict(path=p, error=f"{type(e).__name__}: {e}", flags=[f"import failed: {e}"])
        res["files"][p] = rec
        print(summary_line(p, rec) if "error" not in rec else f"[probe] {p}: IMPORT FAILED {rec['error']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    nflag = sum(1 for r in res["files"].values() if r.get("flags"))
    print(f"[probe] {len(res['files'])} file(s), {nflag} with flags → {a.out}", flush=True)
    return 1 if (a.strict and nflag) else 0


if __name__ == "__main__":
    sys.exit(main())
