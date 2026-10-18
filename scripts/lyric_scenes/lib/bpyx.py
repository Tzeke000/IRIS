"""Core bpy helpers for scene builders (Blender 5.2): keys, node DSL, materials, meshes, collections, lights, math.
One copy instead of rocket's inline block + police/scene_helpers.py + rooftop/rt_util.py + boss_fight's per-file helpers.

    import sys; sys.path.insert(0, "/ABS/scripts/lyric_scenes")
    from lib.bpyx import *          # or: from lib import bpyx as X

KEYS — bake() RECONCILED (2026-10-15). The two old versions were incompatible:
  rooftop  bake(idb, path, vals, index=0, frames=None, interp="LINEAR")   every frame keyed, one interpolation
  rocket   bake(idb, path, vals, index=0, const=False)                    needs module globals N/CUT_SET from init();
           thins redundant keys; forces CONSTANT on the last frame before each cut and keeps the first frame after it
  WON: rooftop's SIGNATURE + rocket's BEHAVIOUR, made explicit — no module globals:
      bake(idb, path, vals, index=0, frames=None, interp="LINEAR", cuts=None, thin=True)
  * cuts = Blender frames where a NEW SHOT STARTS. The key on cut−1 is CONSTANT (the shutter never smears across the
    cut with motion_blur_position START) and keys on cut−1 and cut are always kept. Same result as rocket's
    CUT_SET[c-1] for 0-based cut index c (tests/test_bpyx_pure.py replays rocket's algorithm and compares).
  * thin=True drops only INTERIOR keys of exact plateaus (v[i-1] == v[i] == v[i+1]) — lossless for CONSTANT/LINEAR
    and for BEZIER with auto-clamped handles; rooftop's every-frame behaviour is thin=False.
  * a scalar vals → one key at frame 1 (rocket expanded scalars to N keys, then thinned them to 2: same curve).
  Porting rocket/police code: bake(o, p, v, const=True) → bake(o, p, v, interp="CONSTANT", cuts=CUTS1) where CUTS1 =
  [c + 1 for c in CUTS]; drop the H.init(N, cut_set) call. Porting rooftop code: unchanged (pass cuts= to gain cut safety).
show_mask(ob, mask) keys hide_render (CONSTANT) = ~mask, AND-ed with any hide_render keys already there (rooftop VIS_RULES).

LIGHTS — light() is boss v2's (AREA size, spot cone, aims at target); v3's copy dropped those and ignored target.

ROTATIONS (pure numpy, shared by lib/camera + lib/physics): track_matrix(fwd, up, roll) = to_track_quat("-Z", "Y"),
euler_to_mat, mat_to_euler(R, prev) / euler_seq(R, resets) = to_euler("XYZ", prev) continuity with resets at cuts.
"""
import math

import numpy as np

# ------------------------------------------------------------------------------------------------ pure math

INTERP = ("CONSTANT", "LINEAR", "BEZIER")


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def ramp(i, a, b):
    return np.clip((np.asarray(i, np.float64) - a) / max(b - a, 1e-9), 0.0, 1.0)


def deg(a):
    return math.radians(a)


def hash01(*k):
    """Deterministic 0..1 hash of integers (FNV-1a + murmur finaliser) — rocket's, stable across runs and machines."""
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    return (h ^ (h >> 15)) / 4294967295.0


def gsmooth(x, sigma):
    """Gaussian smooth (edge-padded, same length) — same as lib/song.gsmooth."""
    x = np.asarray(x, np.float64)
    if sigma <= 0:
        return x.copy()
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(x, r, mode="edge"), k / k.sum(), mode="valid")


def track_matrix(fwd, up=(0.0, 0.0, 1.0), roll=0.0):
    """Camera/light rotation (camera −Z along fwd, local Y toward `up`) as (3,3), or (n,3,3) for (n,3) inputs.
    Same result as mathutils `fwd.to_track_quat("-Z", "Y")` for up = +Z (rocket's explicit `up` and bpyx.look_quat's
    roll folded in). Pure numpy. A forward parallel to `up` falls back to up = +Y (no NaN)."""
    f = np.asarray(fwd, np.float64)
    one = f.ndim == 1
    f = np.atleast_2d(f)
    u = np.broadcast_to(np.asarray(up, np.float64), f.shape)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    r = np.cross(f, u)
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    bad = rn[:, 0] < 1e-9
    if bad.any():
        r[bad] = np.cross(f[bad], np.array([0.0, 1.0, 0.0]))
        rn[bad] = np.linalg.norm(r[bad], axis=1, keepdims=True)
    r = r / rn
    y = np.cross(r, f)
    R = np.stack([r, y, -f], axis=2)                       # columns: local X, Y, Z in world
    roll = np.broadcast_to(np.asarray(roll, np.float64), (len(f),))
    if np.any(roll != 0):
        c, s = np.cos(roll), np.sin(roll)
        Rz = np.zeros((len(f), 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1.0
        R = R @ Rz
    return R[0] if one else R


def euler_to_mat(e):
    """XYZ Euler (Blender order: R = Rz·Ry·Rx) → (3,3), or (n,3,3) for (n,3)."""
    e = np.asarray(e, np.float64)
    one = e.ndim == 1
    e = np.atleast_2d(e)
    cx, cy, cz = np.cos(e.T)
    sx, sy, sz = np.sin(e.T)
    R = np.empty((len(e), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -sy, sx * cy, cx * cy
    return R[0] if one else R


def mat_to_euler(R, prev=None):
    """(3,3) rotation → XYZ Euler. With prev: of the two Euler solutions, each unwrapped by 2π per axis toward prev,
    the one closest to prev (what `to_euler("XYZ", prev)` is for: no ±π flips between frames, so keys interpolate)."""
    M = np.asarray(R, np.float64)
    cy = math.hypot(M[0, 0], M[1, 0])
    if cy > 1e-6:
        sols = [(math.atan2(M[2, 1], M[2, 2]), math.atan2(-M[2, 0], cy), math.atan2(M[1, 0], M[0, 0])),
                (math.atan2(-M[2, 1], -M[2, 2]), math.atan2(-M[2, 0], -cy), math.atan2(-M[1, 0], -M[0, 0]))]
    else:                                                   # gimbal lock: y = ±90°, put everything in x
        sols = [(math.atan2(-M[1, 2], M[1, 1]), math.atan2(-M[2, 0], cy), 0.0)]
    if prev is None:
        return np.array(sols[0])
    p = np.asarray(prev, np.float64)
    best = None
    for s in sols:
        s = np.asarray(s)
        s = s + 2 * math.pi * np.round((p - s) / (2 * math.pi))
        d = np.abs(s - p).sum()
        if best is None or d < best[0] - 1e-12:
            best = (d, s)
    return best[1]


def euler_seq(R, resets=()):
    """(n,3,3) rotations → (n,3) continuous XYZ Eulers; continuity restarts at the 0-based indices in `resets`
    (a cut starts its own shot — rocket/police reset there, rooftop didn't)."""
    R = np.asarray(R, np.float64)
    rs = set(int(i) for i in resets)
    out = np.empty((len(R), 3))
    prev = None
    for i in range(len(R)):
        out[i] = prev = mat_to_euler(R[i], None if (prev is None or i in rs) else prev)
    return out


def bake_plan(vals, frames=None, interp="LINEAR", cuts=None, thin=True):
    """Which keys bake() writes: (frames float[k], values float[k], interp names list[k]). Pure numpy.
    vals: per-frame array (frame 1..len) or values at explicit `frames`; cuts: Blender frames where a new shot starts."""
    if interp not in INTERP:
        raise ValueError(f"interp must be one of {INTERP}")
    v = np.asarray(vals, np.float64)
    if v.ndim == 0:
        return np.array([1.0 if frames is None else float(np.ravel(frames)[0])]), v.reshape(1), [interp]
    f = np.arange(1, len(v) + 1, dtype=np.float64) if frames is None else np.asarray(frames, np.float64)
    if len(f) != len(v):
        raise ValueError(f"{len(v)} values for {len(f)} frames")
    n = len(v)
    keep = np.ones(n, bool)
    if thin and n > 2:
        keep[1:-1] = (v[1:-1] != v[:-2]) | (v[1:-1] != v[2:])
    hold = np.zeros(n, bool)
    if cuts is not None and len(cuts):
        fi = np.round(f).astype(np.int64)
        cs = np.asarray(sorted(set(int(c) for c in cuts)), np.int64)
        hold = np.isin(fi, cs - 1)                       # last frame of the old shot
        keep |= hold | np.isin(fi, cs)                   # ... and the first frame of the new one
    ix = np.nonzero(keep)[0]
    names = [("CONSTANT" if hold[i] else interp) for i in ix]
    return f[ix], v[ix], names


# ------------------------------------------------------------------------------------------------ bpy access

def _bpy():
    import bpy
    return bpy


def _interp_codes(names):
    bpy = _bpy()
    items = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items
    table = {k: items[k].value for k in INTERP}
    return np.array([table[x] for x in names], np.int32)


def fcurve(idb, path, index=0):
    """The F-curve for idb.path[index] on its (5.x slotted) action, created if needed."""
    bpy = _bpy()
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idb.name}_act")
    return ad.action.fcurve_ensure_for_datablock(idb, path, index=index)


def find_fcurves(idb, path):
    """Existing F-curves animating `path` on idb (any layer/strip/channelbag of its slot)."""
    ad = getattr(idb, "animation_data", None)
    if not ad or not ad.action:
        return []
    out = []
    for ly in ad.action.layers:
        for st in ly.strips:
            for cb in st.channelbags:
                if ad.action_slot is None or cb.slot == ad.action_slot:
                    out += [fc for fc in cb.fcurves if fc.data_path == path]
    return out


def bake(idb, path, vals, index=0, frames=None, interp="LINEAR", cuts=None, thin=True):
    """Bulk-write keys on idb.<path>[index] (foreach_set; replaces existing keys on that channel). See module doc."""
    fr, v, names = bake_plan(vals, frames, interp, cuts, thin)
    fc = fcurve(idb, path, index)
    kp = fc.keyframe_points
    kp.clear()
    kp.add(len(fr))
    co = np.empty(2 * len(fr), np.float32)
    co[0::2], co[1::2] = fr, v
    kp.foreach_set("co", co)
    kp.foreach_set("interpolation", _interp_codes(names))
    fc.update()
    return fc


def bake_vec(idb, path, arr, **kw):
    """arr (n, k): one bake() per component (location, rotation_euler, color...)."""
    arr = np.asarray(arr, np.float64)
    return [bake(idb, path, arr[:, i], index=i, **kw) for i in range(arr.shape[1])]


def show_mask(ob, mask, combine=True):
    """Key hide_render = ~mask (CONSTANT) for frames 1..len(mask); with combine, frames that existing hide_render keys
    already hide stay hidden (rooftop VIS_RULES). Never use on a SOURCE other objects instance from (gotcha 1)."""
    mask = np.asarray(mask, bool)
    hidden = ~mask
    if combine:
        for fc in find_fcurves(ob, "hide_render"):
            hidden |= np.array([fc.evaluate(i + 1) > 0.5 for i in range(len(mask))])
    return bake(ob, "hide_render", hidden.astype(np.float64), interp="CONSTANT")


# ------------------------------------------------------------------------------------------------ node DSL

class NB:
    """Tiny node-graph DSL (rooftop's) — shader, Geometry Nodes and compositor trees share it.
    Any argument may be a socket (linked) or a constant (default_value; 3-tuples into RGBA get alpha 1)."""

    def __init__(self, nt):
        self.nt = nt

    def n(self, typ, **props):
        nd = self.nt.nodes.new(typ)
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    def set(self, sock, v):
        if v is None:
            return
        if isinstance(v, _bpy().types.NodeSocket):
            self.nt.links.new(v, sock)
        else:
            if isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == "RGBA":
                v = (*v, 1.0)
            sock.default_value = v

    def link(self, a, b):
        self.nt.links.new(a, b)

    def m(self, op, a, b=None, c=None, clamp=False):
        nd = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            self.set(nd.inputs[i], v)
        return nd.outputs[0]

    def v(self, op, a, b=None, c=None):
        nd = self.n("ShaderNodeVectorMath", operation=op)
        if op == "SCALE":
            self.set(nd.inputs[0], a)
            self.set(nd.inputs["Scale"], b)
        else:
            for i, x in enumerate((a, b, c)):
                self.set(nd.inputs[i], x)
        return nd.outputs["Value"] if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else nd.outputs["Vector"]

    def xyz(self, x=0.0, y=0.0, z=0.0):
        nd = self.n("ShaderNodeCombineXYZ")
        for i, s in enumerate((x, y, z)):
            self.set(nd.inputs[i], s)
        return nd.outputs[0]

    def sep(self, vec):
        nd = self.n("ShaderNodeSeparateXYZ")
        self.set(nd.inputs[0], vec)
        return nd.outputs[0], nd.outputs[1], nd.outputs[2]

    def mr(self, val, a, b, c=0.0, d=1.0, interp="LINEAR", clamp=True):
        nd = self.n("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        for name, x in (("Value", val), ("From Min", a), ("From Max", b), ("To Min", c), ("To Max", d)):
            self.set(nd.inputs[name], x)
        return nd.outputs["Result"]

    def mix(self, fac, a, b, blend="MIX", dtype="RGBA"):
        nd = self.n("ShaderNodeMix", data_type=dtype, blend_type=blend)
        self.set(nd.inputs[0], fac)
        if dtype == "RGBA":
            self.set(nd.inputs[6], a)
            self.set(nd.inputs[7], b)
            return nd.outputs[2]
        if dtype == "VECTOR":
            self.set(nd.inputs[4], a)
            self.set(nd.inputs[5], b)
            return nd.outputs[1]
        self.set(nd.inputs[2], a)
        self.set(nd.inputs[3], b)
        return nd.outputs[0]

    def val(self, name, v=0.0):
        """a named Value node (key it with bake(nd_tree, f'nodes["{name}"].outputs[0].default_value', ...))"""
        nd = self.n("ShaderNodeValue")
        nd.name = nd.label = name
        nd.outputs[0].default_value = v
        return nd.outputs[0]

    def tex(self, image, vector=None, interp="Linear", projection="FLAT", blend=0.0):
        nd = self.n("ShaderNodeTexImage", interpolation=interp, projection=projection)
        nd.image = image
        if projection == "BOX":
            nd.projection_blend = blend
        self.set(nd.inputs["Vector"], vector)
        return nd

    # -- geometry-node specific --
    def attr(self, name, dtype="FLOAT"):
        nd = self.n("GeometryNodeInputNamedAttribute", data_type=dtype)
        nd.inputs["Name"].default_value = name
        return nd.outputs["Attribute"]

    def store(self, geo, name, value, dtype="FLOAT", domain="POINT"):
        nd = self.n("GeometryNodeStoreNamedAttribute", data_type=dtype, domain=domain)
        self.set(nd.inputs["Geometry"], geo)
        nd.inputs["Name"].default_value = name
        self.set(nd.inputs["Value"], value)
        return nd.outputs["Geometry"]

    def cmp(self, op, a, b):
        nd = self.n("FunctionNodeCompare", data_type="FLOAT", operation=op)
        self.set(nd.inputs["A"], a)
        self.set(nd.inputs["B"], b)
        return nd.outputs["Result"]

    def band(self, a, b, op="AND"):
        nd = self.n("FunctionNodeBooleanMath", operation=op)
        self.set(nd.inputs[0], a)
        if b is not None:
            self.set(nd.inputs[1], b)
        return nd.outputs[0]

    def rand(self, lo, hi, seed=0, id_=None):
        nd = self.n("FunctionNodeRandomValue", data_type="FLOAT")
        self.set(nd.inputs["Min"], lo)
        self.set(nd.inputs["Max"], hi)
        self.set(nd.inputs["Seed"], seed)
        if id_ is not None:
            self.set(nd.inputs["ID"], id_)
        return nd.outputs["Value"]

    # -- shader specific --
    def sattr(self, name, kind="GEOMETRY"):
        """Attribute node; kind INSTANCER for per-point values on GN instances (gotcha 4)"""
        return self.n("ShaderNodeAttribute", attribute_name=name, attribute_type=kind)

    def geo(self):
        return self.n("ShaderNodeNewGeometry")


def gn_tree(name, inputs=(), with_geo_in=True):
    """Geometry node group with Geometry in/out (+ extra (name, socket_type) inputs) → (group, NB, in, out)."""
    ng = _bpy().data.node_groups.new(name, "GeometryNodeTree")
    if with_geo_in:
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for nm, st in inputs:
        ng.interface.new_socket(nm, in_out="INPUT", socket_type=st)
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nb = NB(ng)
    return ng, nb, nb.n("NodeGroupInput"), nb.n("NodeGroupOutput")


def add_gn(ob, ng, name="GN"):
    mod = ob.modifiers.new(name, "NODES")
    mod.node_group = ng
    return mod


# ------------------------------------------------------------------------------------------------ materials

def new_mat(name):
    """empty node material → (mat, NB, output node)"""
    m = _bpy().data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    nb = NB(nt)
    return m, nb, nb.n("ShaderNodeOutputMaterial")


def principled(nb, base=(0.5, 0.5, 0.5), rough=0.5, metal=0.0, emis=None, estr=0.0):
    """a Principled BSDF node in nb's tree (inputs may be sockets). NOTE: police/rocket's principled(name, ...) made a
    whole material — that is simple_mat() here."""
    b = nb.n("ShaderNodeBsdfPrincipled")
    nb.set(b.inputs["Base Color"], base)
    nb.set(b.inputs["Roughness"], rough)
    nb.set(b.inputs["Metallic"], metal)
    if emis is not None:
        nb.set(b.inputs["Emission Color"], emis)
        nb.set(b.inputs["Emission Strength"], estr)
    return b


def simple_mat(name, base, rough=0.5, metal=0.0, emis=None, estr=0.0):
    m, nb, out = new_mat(name)
    nb.link(principled(nb, base, rough, metal, emis, estr).outputs[0], out.inputs["Surface"])
    return m


def emit_mat(name, color, strength):
    m, nb, out = new_mat(name)
    e = nb.n("ShaderNodeEmission")
    nb.set(e.inputs["Color"], color)
    nb.set(e.inputs["Strength"], strength)
    nb.link(e.outputs[0], out.inputs["Surface"])
    return m


def img(path, color=True):
    """cached image load; absolute existing path asserted (lib/importer.load_img)"""
    from . import importer
    return importer.load_img(path, color)


# ------------------------------------------------------------------------------------------------ collections + objects

def coll(name, parent=None):
    """get-or-create a collection linked under parent (default: scene collection). Never hidden here (gotcha 1)."""
    bpy = _bpy()
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    par = parent or bpy.context.scene.collection
    if c.name not in par.children:
        par.children.link(c)
    return c


def link(ob, coll=None, parent=None):
    (coll or _bpy().context.scene.collection).objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def save_blend(path, compress=True):
    """Save the build with ABSOLUTE paths kept absolute. The default save_as_mainfile(relative_remap=True) rewrites
    absolute image paths to '//…' on save — the arcade magenta scar, reproduced 10-18 by tests/test_verify_saved.sh
    (an in-memory check passes, the saved file fails). Then verify the saved file: lib/render/scene.sh ENV verify."""
    import os
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _bpy().ops.wm.save_as_mainfile(filepath=path, compress=compress, relative_remap=False)
    return path


def empty(name, loc=(0, 0, 0), parent=None, coll=None, kind="PLAIN_AXES"):
    e = _bpy().data.objects.new(name, None)
    e.empty_display_type = kind
    e.location = loc
    return link(e, coll, parent)


def look_quat(loc, target, roll=0.0):
    from mathutils import Quaternion, Vector
    q = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y")
    return q @ Quaternion((0, 0, 1), roll) if roll else q


def look_at(ob, target, roll=0.0):
    """aim a camera/light (−Z) at target"""
    ob.rotation_euler = look_quat(ob.location, target, roll).to_euler()
    return ob


def light(name, typ, loc, target=None, energy=100.0, color=(1, 1, 1), size=0.5, spot_deg=None, spot_blend=0.5,
          coll=None, shadow=True):
    """POINT/SUN/SPOT/AREA light (boss v2's semantics): AREA → size, else shadow_soft_size; SPOT → cone; aimed at target."""
    bpy = _bpy()
    ld = bpy.data.lights.new(name, typ)
    ld.energy, ld.color = energy, color
    if typ == "AREA":
        ld.size = size
    else:
        ld.shadow_soft_size = size
    if spot_deg and typ == "SPOT":
        ld.spot_size, ld.spot_blend = math.radians(spot_deg), spot_blend
    ld.use_shadow = shadow
    o = bpy.data.objects.new(name, ld)
    link(o, coll)
    o.location = loc
    if target is not None:
        look_at(o, target)
    return o


# ------------------------------------------------------------------------------------------------ meshes

def mesh_obj(name, verts, faces=(), coll=None, mat=None, parent=None, loc=None, smooth=False, uvs=None, edges=()):
    """Object from verts/faces. uvs = per-VERTEX (n, 2) array, expanded to loops."""
    bpy = _bpy()
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in np.asarray(verts, np.float64).reshape(-1, 3)], list(edges),
                   [tuple(map(int, f)) for f in faces])
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        lv = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", lv)
        uvl.data.foreach_set("uv", np.asarray(uvs, np.float32)[lv].ravel())
    me.update()
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    if mat is not None:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    if loc is not None:
        ob.location = loc
    return link(ob, coll, parent)


class MB:
    """Accumulate boxes / beams / polys for ONE material, then emit one mesh object (rocket/police)."""

    def __init__(self):
        self.v, self.f = [], []

    def box(self, c, s, rot=None):
        """centre c, full size s, optional 3x3 rotation (mathutils Matrix)"""
        from mathutils import Vector
        sx, sy, sz = s[0] / 2, s[1] / 2, s[2] / 2
        pts = [Vector((x * sx, y * sy, z * sz)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        if rot is not None:
            pts = [rot @ p for p in pts]
        o = len(self.v)
        self.v += [(p.x + c[0], p.y + c[1], p.z + c[2]) for p in pts]
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.f.append(tuple(o + i for i in q))
        return self

    def beam(self, a, b, t):
        from mathutils import Vector
        a, b = Vector(a), Vector(b)
        d = b - a
        return self.box(tuple((a + b) / 2), (t, t, d.length), d.to_track_quat("Z", "Y").to_matrix())

    def poly(self, pts, faces):
        o = len(self.v)
        self.v += [tuple(p) for p in pts]
        self.f += [tuple(o + i for i in f) for f in faces]
        return self

    def obj(self, name, mat=None, parent=None, loc=None, coll=None):
        return mesh_obj(name, self.v, self.f, coll=coll, mat=mat, parent=parent, loc=loc)


def lathe(name, prof, segs=64, mat=None, parent=None, loc=None, a0=0.0, a1=2 * math.pi, cap=False, smooth=True,
          coll=None):
    """Surface of revolution around +Z; prof = [(r, z), ...]. UV u = angle/2π (seam fixed), v = profile arc length."""
    prof = np.asarray(prof, np.float64)
    full = abs((a1 - a0) - 2 * math.pi) < 1e-6
    na = segs if full else segs + 1
    ang = a0 + (a1 - a0) * np.arange(na) / segs
    seg = np.r_[0, np.cumsum(np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1])))]
    tv = seg / max(seg[-1], 1e-9)
    verts, uvs, faces = [], [], []
    for j, (r, z) in enumerate(prof):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
            uvs.append(((a % (2 * math.pi)) / (2 * math.pi), tv[j]))
    for j in range(len(prof) - 1):
        for k in range(segs):
            k2 = (k + 1) % na
            faces.append((j * na + k, j * na + k2, (j + 1) * na + k2, (j + 1) * na + k))
    if cap:
        c = len(verts)
        verts.append((0, 0, prof[0, 1]))
        uvs.append((0.5, 0))
        for k in range(segs):
            faces.append((c, (k + 1) % na, k))
    ob = mesh_obj(name, verts, faces, coll=coll, mat=mat, parent=parent, loc=loc, smooth=smooth)
    me = ob.data
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    uv = np.asarray(uvs, np.float32)[lv].copy()
    ps = np.empty(len(me.polygons), np.int32)
    pt = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_start", ps)
    me.polygons.foreach_get("loop_total", pt)
    for s, t in zip(ps, pt):                       # wrap seam: in a face spanning > 0.5 in u, lift the small ones by 1
        us = uv[s:s + t, 0]
        if us.max() - us.min() > 0.5:
            us[us < 0.5] += 1.0
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", uv.ravel())
    return ob


def uv_sphere(name, r=1.0, seg=32, rings=16, **kw):
    prof = [(r * math.sin(math.pi * k / rings), -r * math.cos(math.pi * k / rings)) for k in range(rings + 1)]
    return lathe(name, prof, seg, **kw)


def points_obj(name, pts, attrs=None, coll=None):
    """Vertex-only mesh with float / vector POINT attributes (bulk foreach_set) — GN instancing input."""
    bpy = _bpy()
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(pts))
    me.vertices.foreach_set("co", pts.ravel())
    for k, v in (attrs or {}).items():
        v = np.asarray(v, np.float32)
        if v.ndim == 2:
            me.attributes.new(k, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", v.ravel())
        else:
            me.attributes.new(k, "FLOAT", "POINT").data.foreach_set("value", v)
    me.update()
    return link(bpy.data.objects.new(name, me), coll)


def boxes_mesh(name, boxes, face_attrs=None, coll=None, bottom=False):
    """boxes (N,6) [x0,y0,z0,x1,y1,z1] → one object, 5 faces each (6 with bottom=True). face_attrs {name: (N,)}."""
    bpy = _bpy()
    b = np.asarray(boxes, np.float64).reshape(-1, 6)
    n = len(b)
    x0, y0, z0, x1, y1, z1 = b.T
    V = np.stack([np.stack(p, 1) for p in ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                                           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))], 1).reshape(-1, 3)
    F = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]] + ([[0, 3, 2, 1]] if bottom else [])
    F = np.array(F)
    nf = len(F)
    faces = (F[None] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 4)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.ravel().astype(np.int32))
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", (np.arange(len(faces)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    for k, v in (face_attrs or {}).items():
        me.attributes.new(k, "FLOAT", "FACE").data.foreach_set("value", np.repeat(np.asarray(v, np.float32), nf))
    me.validate()
    return link(bpy.data.objects.new(name, me), coll)


__all__ = [k for k in dir() if not k.startswith("_") and k not in ("math", "np")] + ["np", "math"]
