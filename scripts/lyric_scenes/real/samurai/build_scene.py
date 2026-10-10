"""NEON SAMURAI DUEL: a photoreal, music-synced Blender scene for "Pull Me Close" (Tzeke000).

    ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/real/samurai/build_scene.py -- \
        [--timeline state/tzeke_songs/pull_me_close/timeline_30fps.npz] \
        [--out scratch/scenes/samurai/samurai.blend] [--samples 32] [--res-pct 100]

This builds the .blend DETERMINISTICALLY. choreo.py plans the whole fight in numpy (who strikes on
which snare, where both blades meet, footwork on the kicks, every camera, every light level). This
script turns that plan into geometry and BAKED keyframes: armature bones (analytic two-bone IK +
spring bones computed here), sword transforms, GN effect inputs, light energies and camera. There
are no handlers, no drivers and no simulations to bake. Frame f (1-based) = timeline index f-1.

Night, heavy rain, a narrow plaza of old wooden townhouses under neon. Two duelists (a ronin in a
straw kasa, and a hooded swordsman in a long coat; faces never shown) circle, step closer on every
"close", draw on "dangerous pose", clash on the drop with sparks, a rain shockwave and lightning.
On the big drop the whole street glows. The final mini-drop is a pass-through cut, and they end
standing back to back while the neon fades.
"""
import argparse
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import choreo as C  # noqa: E402

REPO = C.REPO
SONG = C.SONG_DIR


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=os.path.join(SONG, "timeline_30fps.npz"))
    ap.add_argument("--words", default=os.path.join(SONG, "words.json"))
    ap.add_argument("--audio", default=os.path.join(SONG, "pull_me_close.wav"))
    ap.add_argument("--logo", default=os.path.join(REPO, "state", "tzeke_songs", "_assets",
                                                   "tzeke000_symbol.png"))
    ap.add_argument("--font", default=os.path.join(REPO, "assets", "fonts", "BebasNeue-Regular.ttf"))
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "scenes", "samurai", "samurai.blend"))
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--res-pct", type=int, default=100)
    ap.add_argument("--rain", type=float, default=1.0, help="rain density multiplier")
    return ap.parse_args(argv)


ARGS = parse_args()
DU = C.plan(ARGS.timeline, ARGS.words)
S = DU.S
N, FPS = DU.n, S.fps
IDX = np.arange(N)
print("\n".join("[choreo] " + l for l in DU.log))

# =====================================================================================
# 1. BLENDER HELPERS
# =====================================================================================
SC = bpy.context.scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.armatures,
             bpy.data.curves, bpy.data.worlds, bpy.data.node_groups, bpy.data.images, bpy.data.actions):
    for d in list(coll):
        coll.remove(d)

KEY_ENUM = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items
LIN, CONST = KEY_ENUM["LINEAR"].value, KEY_ENUM["CONSTANT"].value
CUT_SET = DU.cut_before.copy()


def _fcurve(idb, path, index):
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idb.name}_bake")
    return ad.action.fcurve_ensure_for_datablock(idb, path, index=index)


def bake(idb, path, vals, index=0, const=False, cuts=False):
    """One value per timeline frame onto idb.path[index] (bulk foreach_set). cuts=True holds the
    value on the frame before every camera cut (no motion-blur smear across a cut)."""
    v = np.asarray(vals, np.float64)
    if v.ndim == 0:
        v = np.full(N, float(v))
    assert len(v) == N, (path, len(v))
    keep = np.ones(N, bool)
    keep[1:-1] = (v[1:-1] != v[:-2]) | (v[1:-1] != v[2:])
    if cuts:
        keep |= CUT_SET
        keep[np.minimum(np.nonzero(CUT_SET)[0] + 1, N - 1)] = True
    ix = np.nonzero(keep)[0]
    fc = _fcurve(idb, path, index)
    kp = fc.keyframe_points
    kp.clear()
    kp.add(len(ix))
    co = np.empty(2 * len(ix), np.float32)
    co[0::2] = ix + 1
    co[1::2] = v[ix]
    kp.foreach_set("co", co)
    interp = np.full(len(ix), CONST if const else LIN, np.int32)
    if cuts:
        interp[CUT_SET[ix]] = CONST
    kp.foreach_set("interpolation", interp)
    fc.update()
    return fc


def bake_vec(idb, path, arr, const=False, cuts=False):
    arr = np.asarray(arr, np.float64)
    for k in range(arr.shape[1]):
        bake(idb, path, arr[:, k], index=k, const=const, cuts=cuts)


def link(ob, parent=None, coll=None):
    (coll or SC.collection).objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def empty(name, loc=(0, 0, 0), parent=None):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    return link(e, parent)


def mesh_from(name, verts, faces, uvs=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in verts], [], [tuple(map(int, f)) for f in faces])
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        loop_v = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loop_v)
        uvl.data.foreach_set("uv", np.asarray(uvs, np.float32)[loop_v].ravel())
    me.update()
    if smooth and len(me.polygons):
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    return me


def mesh_obj(name, verts, faces, mat=None, parent=None, loc=(0, 0, 0), smooth=True, uvs=None):
    me = mesh_from(name, verts, faces, uvs, smooth)
    if mat is not None:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    return link(ob, parent)


class MB:
    """Accumulate boxes / polys for ONE material, then emit one mesh object."""

    def __init__(self):
        self.v, self.f, self.uv = [], [], []

    def box(self, c, s, rot=None):
        cx, cy, cz = c
        sx, sy, sz = (s[0] / 2, s[1] / 2, s[2] / 2)
        pts = [Vector((x * sx, y * sy, z * sz)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        if rot is not None:
            pts = [rot @ p for p in pts]
        o = len(self.v)
        self.v += [(p.x + cx, p.y + cy, p.z + cz) for p in pts]
        self.uv += [((p.x + cx + p.z + cz) * 0.5, (p.y + cy + p.z + cz) * 0.5) for p in pts]
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.f.append(tuple(o + i for i in q))

    def beam(self, a, b, t, t2=None):
        a, b = Vector(a), Vector(b)
        d = b - a
        rot = d.to_track_quat("Z", "Y").to_matrix()
        self.box(tuple((a + b) / 2), (t, t2 or t, d.length), rot)

    def quad(self, p0, p1, p2, p3, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
        o = len(self.v)
        self.v += [tuple(p) for p in (p0, p1, p2, p3)]
        self.uv += list(uv)
        self.f.append((o, o + 1, o + 2, o + 3))

    def poly(self, pts, faces, uvs=None):
        o = len(self.v)
        self.v += [tuple(p) for p in pts]
        self.uv += list(uvs) if uvs is not None else [(p[0], p[2]) for p in pts]
        self.f += [tuple(o + i for i in f) for f in faces]

    def obj(self, name, mat=None, parent=None, loc=(0, 0, 0), smooth=False):
        return mesh_obj(name, self.v, self.f, mat, parent, loc, smooth=smooth, uvs=self.uv)


def loft(rings, closed=True, cap0=False, cap1=False):
    """rings: list of (n,3) arrays with equal n -> verts, faces, uvs (u around, v along)."""
    rings = [np.asarray(r, np.float64) for r in rings]
    nr, n = len(rings), len(rings[0])
    verts = np.concatenate(rings, 0)
    faces = []
    m = n if closed else n - 1
    for j in range(nr - 1):
        for k in range(m):
            k2 = (k + 1) % n
            faces.append((j * n + k, j * n + k2, (j + 1) * n + k2, (j + 1) * n + k))
    uvs = [((k / n) if closed else k / (n - 1), j / max(nr - 1, 1)) for j in range(nr) for k in range(n)]
    verts = list(map(tuple, verts))
    if cap0:
        c = len(verts)
        verts.append(tuple(rings[0].mean(0)))
        uvs.append((0.5, 0))
        faces += [(c, (k + 1) % n, k) for k in range(m)]
    if cap1:
        c = len(verts)
        verts.append(tuple(rings[-1].mean(0)))
        uvs.append((0.5, 1))
        o = (nr - 1) * n
        faces += [(c, o + k, o + (k + 1) % n) for k in range(m)]
    return verts, faces, uvs


def ring(c, a, b, rx, ry, n=24, a0=0.0, a1=2 * math.pi, closed=True, rfun=None, sq=2.0):
    """Points around centre c in the plane (a, b); superellipse exponent sq (2 = ellipse)."""
    c, a, b = (np.asarray(v, np.float64) for v in (c, a, b))
    cnt = n if closed else n
    ang = a0 + (a1 - a0) * (np.arange(cnt) / (n if closed else n - 1))
    ca, sa = np.cos(ang), np.sin(ang)
    e = 2.0 / sq
    x = np.sign(ca) * np.abs(ca) ** e * rx
    y = np.sign(sa) * np.abs(sa) ** e * ry
    if rfun is not None:
        k = rfun(ang)
        x, y = x * k, y * k
    return c[None] + x[:, None] * a[None] + y[:, None] * b[None]


# ---- node helpers --------------------------------------------------------------------
def nd(nt, typ, **kw):
    n = nt.nodes.new(typ)
    for k, v in kw.items():
        if k == "inp":
            for name, val in v.items():
                n.inputs[name].default_value = val
        else:
            setattr(n, k, v)
    return n


def lk(nt, a, b):
    nt.links.new(a, b)


def M(nt, op, a, b=None, c=None, clamp=False):
    n = nd(nt, "ShaderNodeMath", operation=op, use_clamp=clamp)
    for k, x in enumerate((a, b, c)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = float(x)
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[0]


def V(nt, op, a, b=None, scale=None):
    n = nd(nt, "ShaderNodeVectorMath", operation=op)
    for k, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (tuple, list)):
            n.inputs[k].default_value = x
        else:
            lk(nt, x, n.inputs[k])
    if scale is not None:
        if isinstance(scale, (int, float)):
            n.inputs["Scale"].default_value = scale
        else:
            lk(nt, scale, n.inputs["Scale"])
    return n.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]


def value_node(nt, name, v=0.0):
    n = nd(nt, "ShaderNodeValue", name=name, label=name)
    n.outputs[0].default_value = v
    return n


def value_path(name):
    return f'nodes["{name}"].outputs[0].default_value'


def mix_col(nt, fac, a, b, blend="MIX"):
    n = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type=blend)
    for k, x in ((0, fac), (6, a), (7, b)):
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = x
        elif isinstance(x, (tuple, list)):
            n.inputs[k].default_value = tuple(x) + ((1.0,) if len(x) == 3 else ())
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[2]


def mix_f(nt, fac, a, b):
    n = nd(nt, "ShaderNodeMix", data_type="FLOAT")
    for k, x in ((0, fac), (2, a), (3, b)):
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = x
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[0]


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.node_tree.nodes.clear()
    return m, m.node_tree


def principled(name, color, rough=0.5, metal=0.0, bump=None, coat=0.0, sheen=0.0, spec=0.5,
               emit=None, emit_strength=0.0):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = tuple(color) + (1,)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = 0.05
    b.inputs["Sheen Weight"].default_value = sheen
    if emit is not None:
        b.inputs["Emission Color"].default_value = tuple(emit) + (1,)
        b.inputs["Emission Strength"].default_value = emit_strength
    if bump:
        tc = nd(nt, "ShaderNodeTexCoord")
        nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": bump[0], "Detail": 6.0})
        lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
        bp = nd(nt, "ShaderNodeBump", inp={"Strength": bump[1], "Distance": 0.02})
        lk(nt, nz.outputs["Fac"], bp.inputs["Height"])
        lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
        cr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.3, "From Max": 0.7,
                                               "To Min": rough * 0.6, "To Max": min(1, rough * 1.4)})
        lk(nt, nz.outputs["Fac"], cr.inputs["Value"])
        lk(nt, cr.outputs["Result"], b.inputs["Roughness"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def emissive(name, color, strength, gain_node=None):
    """Emission; gain_node = name of a Value node (keyable) that multiplies the strength."""
    m, nt = new_mat(name)
    e = nd(nt, "ShaderNodeEmission", inp={"Color": tuple(color) + (1,), "Strength": strength})
    if gain_node:
        g = value_node(nt, gain_node, 1.0)
        lk(nt, M(nt, "MULTIPLY", g.outputs[0], strength), e.inputs["Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, e.outputs[0], o.inputs["Surface"])
    return m


def blackbody(k):
    t = k / 100.0
    r = 1.0 if t <= 66 else min(1, max(0, 329.698727446 * (t - 60) ** -0.1332047592 / 255))
    g = (min(1, max(0, (99.4708025861 * math.log(t) - 161.1195681661) / 255)) if t <= 66
         else min(1, max(0, 288.1221695283 * (t - 60) ** -0.0755148492 / 255)))
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else min(1, max(0, (138.5177312231 * math.log(t - 10)
                                                                - 305.0447927307) / 255)))
    return tuple(c ** 2.2 for c in (r, g, b))


def no_shadow(ob):
    ob.visible_shadow = False
    return ob


def collection(name):
    c = bpy.data.collections.new(name)
    SC.collection.children.link(c)
    return c


# =====================================================================================
# 2. CHARACTER MATERIALS
# =====================================================================================
def cloth_material(name, color, rough=0.72, weave=420.0, wet=0.35, sheen=0.12, stripe=None):
    """Wet woven fabric: weave bump + rain-darkened patches that turn glossy."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    tc = nd(nt, "ShaderNodeTexCoord")
    wv = nd(nt, "ShaderNodeTexWave", wave_type="BANDS", bands_direction="X",
            inp={"Scale": weave, "Distortion": 1.5, "Detail": 2.0})
    wv2 = nd(nt, "ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y",
             inp={"Scale": weave, "Distortion": 1.5, "Detail": 2.0})
    lk(nt, tc.outputs["Object"], wv.inputs["Vector"])
    lk(nt, tc.outputs["Object"], wv2.inputs["Vector"])
    wsum = M(nt, "MULTIPLY", M(nt, "ADD", wv.outputs["Fac"], wv2.outputs["Fac"]), 0.5)
    fold = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 7.0, "Detail": 3.0, "Roughness": 0.5})
    lk(nt, tc.outputs["Object"], fold.inputs["Vector"])
    h = M(nt, "ADD", M(nt, "MULTIPLY", wsum, 0.25), fold.outputs["Fac"])
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.35, "Distance": 0.01})
    lk(nt, h, bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    wetn = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 3.5, "Detail": 4.0})
    lk(nt, tc.outputs["Object"], wetn.inputs["Vector"])
    wetm = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.45, "From Max": 0.62, "To Min": 0.0, "To Max": wet})
    lk(nt, wetn.outputs["Fac"], wetm.inputs["Value"])
    col = color
    base = mix_col(nt, wetm.outputs[0], col, tuple(c * 0.55 for c in col))
    if stripe is not None:
        st = nd(nt, "ShaderNodeTexWave", wave_type="BANDS", bands_direction="X",
                inp={"Scale": stripe[0], "Distortion": 0.0})
        lk(nt, tc.outputs["Object"], st.inputs["Vector"])
        sm = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.85, "From Max": 0.95, "To Min": 0, "To Max": 1})
        lk(nt, st.outputs["Fac"], sm.inputs["Value"])
        base = mix_col(nt, sm.outputs[0], base, stripe[1])
    lk(nt, base, b.inputs["Base Color"])
    rr = M(nt, "SUBTRACT", rough, M(nt, "MULTIPLY", wetm.outputs[0], 1.1))
    lk(nt, rr, b.inputs["Roughness"])
    b.inputs["Sheen Weight"].default_value = sheen
    b.inputs["Sheen Roughness"].default_value = 0.4
    b.inputs["Specular IOR Level"].default_value = 0.45
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def straw_material(name):
    """Woven straw kasa: radial weave + wet lacquer coat (rain glints)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    tc = nd(nt, "ShaderNodeTexCoord")
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["Object"], sep.inputs[0])
    ang = M(nt, "ARCTAN2", sep.outputs[1], sep.outputs[0])
    rad = M(nt, "SQRT", M(nt, "ADD", M(nt, "MULTIPLY", sep.outputs[0], sep.outputs[0]),
                          M(nt, "MULTIPLY", sep.outputs[1], sep.outputs[1])))
    a1 = M(nt, "SINE", M(nt, "MULTIPLY", ang, 180.0))
    r1 = M(nt, "SINE", M(nt, "MULTIPLY", rad, 260.0))
    weave = M(nt, "MULTIPLY", a1, r1)
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 40.0, "Detail": 4.0})
    lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
    h = M(nt, "ADD", M(nt, "MULTIPLY", weave, 0.5), M(nt, "MULTIPLY", nz.outputs["Fac"], 0.5))
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.6, "Distance": 0.004})
    lk(nt, h, bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    lk(nt, bp.outputs["Normal"], b.inputs["Coat Normal"])
    base = mix_col(nt, nz.outputs["Fac"], (0.16, 0.115, 0.06), (0.32, 0.24, 0.12))
    lk(nt, base, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.6
    b.inputs["Coat Weight"].default_value = 0.85        # rain film
    b.inputs["Coat Roughness"].default_value = 0.04
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def leather_material(name, color):
    m = principled(name, color, 0.42, 0.0, bump=(90, 0.25), coat=0.35)
    return m


M_KIMONO = cloth_material("Kimono", (0.025, 0.03, 0.05), weave=520)
M_HAKAMA = cloth_material("Hakama", (0.035, 0.033, 0.036), weave=480)
M_COLLAR = cloth_material("Collar", (0.30, 0.31, 0.33), weave=600, wet=0.2)
M_OBI = cloth_material("Obi", (0.16, 0.03, 0.025), weave=700, wet=0.25, sheen=0.25)
M_WRAP = cloth_material("HeadWrap", (0.012, 0.012, 0.014), weave=650, wet=0.2)
M_COAT = cloth_material("Coat", (0.018, 0.017, 0.018), rough=0.55, weave=300, wet=0.55, sheen=0.2)
M_TROUSER = cloth_material("Trouser", (0.03, 0.03, 0.032), weave=500)
M_STRAW = straw_material("Straw")
M_GLOVE = leather_material("Glove", (0.02, 0.016, 0.014))
M_BOOT = leather_material("Boot", (0.018, 0.014, 0.012))
M_SOLE = principled("Sole", (0.05, 0.04, 0.03), 0.8)
M_CORD = cloth_material("Cord", (0.25, 0.04, 0.03), weave=900, wet=0.1)
M_SAYA = principled("Saya", (0.008, 0.008, 0.009), 0.25, 0.0, coat=1.0, spec=0.6)
M_KOIGUCHI = principled("Koiguchi", (0.55, 0.50, 0.42), 0.35)
M_TSUBA = principled("Tsuba", (0.05, 0.045, 0.04), 0.45, 1.0, bump=(150, 0.4))
M_HABAKI = principled("Habaki", (0.9, 0.62, 0.30), 0.28, 1.0)


def blade_material(name, glow_rgb):
    """Polished steel + frosty hamon band along the edge + a keyable edge glow."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    tc = nd(nt, "ShaderNodeTexCoord")
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["UV"], sep.inputs[0])
    # UV: u across the blade (0 = spine, 1 = edge), v along it
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 3.0, "Detail": 5.0})
    nzc = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, M(nt, "MULTIPLY", sep.outputs[1], 14.0), nzc.inputs[0])
    lk(nt, nzc.outputs[0], nz.inputs["Vector"])
    wave = M(nt, "ADD", 0.55, M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz.outputs["Fac"], 0.5), 0.35))
    ham = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.0, "From Max": 0.06, "To Min": 0.0, "To Max": 1.0})
    lk(nt, M(nt, "SUBTRACT", sep.outputs[0], wave), ham.inputs["Value"])
    hm = ham.outputs[0]
    base = mix_col(nt, hm, (0.80, 0.81, 0.83), (0.93, 0.94, 0.95))
    lk(nt, base, b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = 1.0
    lk(nt, mix_f(nt, hm, 0.05, 0.16), b.inputs["Roughness"])
    edge = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.86, "From Max": 0.99, "To Min": 0.0, "To Max": 1.0})
    lk(nt, sep.outputs[0], edge.inputs["Value"])
    g = value_node(nt, "glow", 0.1)
    lk(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", edge.outputs[0], g.outputs[0]), 9.0), b.inputs["Emission Strength"])
    b.inputs["Emission Color"].default_value = tuple(glow_rgb) + (1,)
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def tsuka_material(name):
    """Hilt: diamond (hishigami) silk wrap over ray skin."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    tc = nd(nt, "ShaderNodeTexCoord")
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["UV"], sep.inputs[0])
    u = M(nt, "MULTIPLY", sep.outputs[0], 2.0)
    v = M(nt, "MULTIPLY", sep.outputs[1], 9.0)
    d1 = M(nt, "PINGPONG", M(nt, "ADD", u, v), 0.5)
    d2 = M(nt, "PINGPONG", M(nt, "SUBTRACT", u, v), 0.5)
    dia = M(nt, "MINIMUM", d1, d2)
    win = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.10, "From Max": 0.16, "To Min": 1.0, "To Max": 0.0})
    lk(nt, dia, win.inputs["Value"])
    base = mix_col(nt, win.outputs[0], (0.02, 0.02, 0.025), (0.75, 0.72, 0.62))
    lk(nt, base, b.inputs["Base Color"])
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.8, "Distance": 0.003})
    lk(nt, dia, bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = 0.55
    b.inputs["Sheen Weight"].default_value = 0.6
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


GLOW_A = (0.15, 0.75, 1.0)      # ronin: cyan
GLOW_B = (1.0, 0.16, 0.55)      # hood: magenta
M_BLADE = [blade_material("BladeA", GLOW_A), blade_material("BladeB", GLOW_B)]
M_TSUKA = tsuka_material("Tsuka")


# =====================================================================================
# 3. RIG: armature from choreo.BONES, analytic IK, spring bones, bulk bake
# =====================================================================================
def build_armature(name):
    arm = bpy.data.armatures.new(name)
    ob = link(bpy.data.objects.new(name, arm))
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    for bn, (h, t, p, c) in C.BONES.items():
        b = eb.new(bn)
        b.head, b.tail, b.roll = h, t, 0.0
    for bn, (h, t, p, c) in C.BONES.items():
        if p:
            eb[bn].parent = eb[p]
            eb[bn].use_connect = c
    bpy.ops.object.mode_set(mode="OBJECT")
    for b in arm.bones:
        if b.name.startswith(("fore.", "hand.", "shin.", "foot.")):
            b.inherit_scale = "NONE"
    for pb in ob.pose.bones:
        pb.rotation_mode = "QUATERNION"
    arm.display_type = "STICK"
    return ob


ORDER = ["hips", "spine", "chest", "neck", "head", "upper.R", "fore.R", "hand.R", "upper.L", "fore.L",
         "hand.L", "thigh.R", "shin.R", "foot.R", "thigh.L", "shin.L", "foot.L",
         "skirt.F", "skirt.B", "skirt.R", "skirt.L", "sleeve.R", "sleeve.L", "hood"]


def two_bone(S0, T, P, l1, l2, stretch_max=1.25):
    """Analytic 2-bone IK (vectorised). Returns elbow/knee, end point, stretch factor."""
    d = T - S0
    dist = np.linalg.norm(d, axis=1)
    L = l1 + l2
    st = np.clip(dist / (L * 0.999), 1.0, stretch_max)
    dl = np.minimum(dist, L * 0.999 * st)
    n = d / np.maximum(dist, 1e-9)[:, None]
    l1s, l2s = l1 * st, l2 * st
    a = (l1s ** 2 - l2s ** 2 + dl ** 2) / (2 * np.maximum(dl, 1e-9))
    h = np.sqrt(np.clip(l1s ** 2 - a ** 2, 0, None))
    p = P - S0
    p = p - np.sum(p * n, 1)[:, None] * n
    p = C.nrm(p)
    E = S0 + n * a[:, None] + p * h[:, None]
    End = S0 + n * dl[:, None]
    return E, End, st


class Rig:
    """Turns the choreography of one fighter into pose-bone quaternions (+ arm stretch)."""

    def __init__(self, ob, i):
        self.ob, self.i = ob, i
        self.Rr, self.head_r, self.len = {}, {}, {}
        for b in ob.data.bones:
            mtx = np.array(b.matrix_local)
            self.Rr[b.name] = mtx[:3, :3]
            self.head_r[b.name] = mtx[:3, 3]
            self.len[b.name] = b.length
        self.P, self.H, self.basis, self.scale = {}, {}, {}, {}

    def to_arm(self, pw):
        """world (N,3) -> armature space."""
        i = self.i
        loc = np.concatenate([DU.root[i], (DU.pelvis[i] - C.PELVIS_REST)[:, None]], 1)
        Rinv = np.transpose(C.rotz(DU.yaw[i]), (0, 2, 1))
        return np.einsum("nij,nj->ni", Rinv, pw - loc)

    def rot_to_arm(self, Rw):
        return np.einsum("nij,njk->nik", np.transpose(C.rotz(DU.yaw[self.i]), (0, 2, 1)), Rw)

    def carried(self, b):
        p = C.BONES[b][2]
        if p is None:
            return np.repeat(self.Rr[b][None], N, 0)
        return np.einsum("nij,jk->nik", self.P[p], self.Rr[p].T @ self.Rr[b])

    def head_of(self, b):
        p = C.BONES[b][2]
        if p is None:
            return np.repeat(self.head_r[b][None], N, 0)
        if C.BONES[b][3]:          # connected: parent's tail
            return self.H[p] + self.P[p][:, :, 1] * (self.len[p] * self.scale.get(p, np.ones(N)))[:, None]
        return self.H[p] + np.einsum("nij,j->ni", self.P[p], self.Rr[p].T @ (self.head_r[b] - self.head_r[p]))

    def set(self, b, W, scale=None):
        Cm = self.carried(b)
        self.H[b] = self.head_of(b)
        self.P[b] = W
        self.basis[b] = np.einsum("nji,njk->nik", Cm, W)
        if scale is not None:
            self.scale[b] = scale

    def swing_to(self, b, direction):
        Cm = self.carried(b)
        return np.einsum("nij,njk->nik", C.swing(Cm[:, :, 1], direction), Cm)

    def solve(self):
        i = self.i
        Rs, Rc, ch_h, shR, shL, nk = DU.fk[i]
        I3 = np.repeat(np.eye(3)[None], N, 0)
        self.set("hips", np.repeat(self.Rr["hips"][None], N, 0))
        self.set("spine", np.einsum("nij,jk->nik", Rs, self.Rr["spine"]))
        Rsc = np.einsum("nij,njk->nik", Rs, Rc)
        self.set("chest", np.einsum("nij,jk->nik", Rsc, self.Rr["chest"]))
        # look at the opponent's head (neck 40 %, head the rest), limited
        o = 1 - i
        opp = self.to_arm(np.concatenate([DU.root[o], (DU.pelvis[o] + 0.62)[:, None]], 1))
        hp = self.head_of("neck") + np.array([0, 0, 0.2])
        fwd_c = np.einsum("nij,j->ni", Rsc, np.array([0, 1.0, 0]))
        want = C.nrm(opp - hp)
        want[:, 2] = np.clip(want[:, 2], -0.35, 0.35)
        Q = C.swing(fwd_c, C.nrm(want))
        q = C.mat2quat(Q)
        qn = C.nrm(np.stack([np.ones(N), *(q[:, 1:].T * 0)], 1) * 0.6 + q * 0.4)
        qn = C.nrm(q * 0.4 + np.array([1.0, 0, 0, 0]) * 0.6)
        Rn = C.quat2mat(qn)
        self.set("neck", np.einsum("nij,njk->nik", Rn, np.einsum("nij,jk->nik", Rsc, self.Rr["neck"])))
        self.set("head", np.einsum("nij,njk->nik", Q, np.einsum("nij,jk->nik", Rsc, self.Rr["head"])))
        # arms
        self.ik_err = {}
        for side, si in (("R", 0), ("L", 1)):
            up, fo, ha = f"upper.{side}", f"fore.{side}", f"hand.{side}"
            S0 = self.head_of(up)
            T = self.to_arm(DU.wrist[i][si])
            Pp = self.to_arm(DU.elbow_pole[i][si])
            E, End, st = two_bone(S0, T, Pp, self.len[up], self.len[fo], 1.3)
            self.ik_err[side] = np.linalg.norm(End - T, axis=1)
            self.set(up, self.swing_to(up, E - S0), st)
            self.set(fo, self.swing_to(fo, End - E), st)
            gw = DU.grip_w[i][si]
            Wfree = self.carried(ha)
            Wgrip = self.rot_to_arm(DU.hand_rot[i])
            qf, qg = C.mat2quat(Wfree), C.mat2quat(Wgrip)
            sgn = np.sign(np.sum(qf * qg, 1))[:, None]
            q = C.nrm(qf * (1 - gw)[:, None] + qg * sgn * gw[:, None])
            self.set(ha, C.quat2mat(q))
        # legs
        for side, si in (("R", 0), ("L", 1)):
            th, sh, ft = f"thigh.{side}", f"shin.{side}", f"foot.{side}"
            S0 = self.head_of(th)
            T = self.to_arm(DU.ankle[i][si])
            yaw_rel = DU.foot_yaw[i][si] - DU.yaw[i]
            fwd = np.stack([-np.sin(yaw_rel), np.cos(yaw_rel), np.zeros(N)], 1)
            Pp = T + fwd * 0.8 + np.array([0.0, 0.0, 0.45]) + np.array([0.12 if side == "R" else -0.12, 0, 0])
            E, End, st = two_bone(S0, T, Pp, self.len[th], self.len[sh], 1.06)
            self.ik_err["leg" + side] = np.linalg.norm(End - T, axis=1)
            self.set(th, self.swing_to(th, E - S0), st)
            self.set(sh, self.swing_to(sh, End - E), st)
            Rf = np.einsum("nij,njk->nik", C.rotz(yaw_rel), C.rotx(DU.foot_pitch[i][si]))
            self.set(ft, np.einsum("nij,jk->nik", Rf, self.Rr[ft]))
        self.springs()

    def springs(self):
        """Secondary motion: skirt panels, sleeve bags and the hood lag behind and settle
        (damped springs integrated over the song in world space)."""
        i = self.i
        loc = np.concatenate([DU.root[i], (DU.pelvis[i] - C.PELVIS_REST)[:, None]], 1)
        Rw = C.rotz(DU.yaw[i])
        thighR = C.nrm(self.P["thigh.R"][:, :, 1])
        thighL = C.nrm(self.P["thigh.L"][:, :, 1])
        specs = {  # bone: (stiffness, damping, max angle, leg-follow)
            "skirt.F": (0.10, 0.80, 0.9, "F"), "skirt.B": (0.10, 0.80, 0.9, "B"),
            "skirt.R": (0.10, 0.80, 0.9, "R"), "skirt.L": (0.10, 0.80, 0.9, "L"),
            "sleeve.R": (0.08, 0.84, 1.1, None), "sleeve.L": (0.08, 0.84, 1.1, None),
            "hood": (0.16, 0.75, 0.7, None),
        }
        for b, (k, damp, amax, follow) in specs.items():
            Cm = self.carried(b)
            head = self.head_of(b)
            rest_dir = Cm[:, :, 1].copy()
            if follow is not None:
                lean = {"F": None, "B": None, "R": thighR, "L": thighL}[follow]
                if follow in ("F", "B"):
                    s = 1 if follow == "F" else -1
                    pick = np.where((thighR[:, 1] * s > thighL[:, 1] * s)[:, None], thighR, thighL)
                    lean = pick
                rest_dir = C.nrm(rest_dir * 0.35 + lean * 0.65)
            L = self.len[b]
            hw = np.einsum("nij,nj->ni", Rw, head) + loc
            tw_target = hw + np.einsum("nij,nj->ni", Rw, rest_dir) * L
            x = tw_target[0].copy()
            v = np.zeros(3)
            out = np.empty((N, 3))
            g = np.array([0, 0, -0.0035])
            for t in range(N):
                v = v * damp + (tw_target[t] - x) * k + g
                x = x + v
                d = x - hw[t]
                ln = np.linalg.norm(d)
                x = hw[t] + d / max(ln, 1e-9) * L
                out[t] = x
            dirw = C.nrm(out - hw)
            dir_a = np.einsum("nji,nj->ni", Rw, dirw)
            # limit the swing away from the target direction
            cosang = np.sum(dir_a * rest_dir, 1)
            ang = np.arccos(np.clip(cosang, -1, 1))
            over = ang > amax
            if np.any(over):
                ax = C.nrm(np.cross(rest_dir[over], dir_a[over]))
                R_ = []
                for a_, r_ in zip(ax, rest_dir[over]):
                    ca, sa = math.cos(amax), math.sin(amax)
                    R_.append(r_ * ca + np.cross(a_, r_) * sa + a_ * np.dot(a_, r_) * (1 - ca))
                dir_a[over] = np.array(R_)
            self.set(b, self.swing_to(b, dir_a))

    def bake(self):
        ob = self.ob
        for b in ORDER:
            if b not in self.basis:
                continue
            q = C.qcontinuous(C.mat2quat(self.basis[b]))
            for k in range(4):
                bake(ob, f'pose.bones["{b}"].rotation_quaternion', q[:, k], index=k)
            if b in self.scale:
                bake(ob, f'pose.bones["{b}"].scale', self.scale[b], index=1)
        i = self.i
        loc = np.concatenate([DU.root[i], (DU.pelvis[i] - C.PELVIS_REST)[:, None]], 1)
        bake_vec(ob, "location", loc)
        bake(ob, "rotation_euler", DU.yaw[i], index=2)


# =====================================================================================
# 4. CHARACTER GEOMETRY (rest pose, standing coords, faces +Y) + skin weights
# =====================================================================================
BSEG = {b: (np.array(h, float), np.array(t, float)) for b, (h, t, p, c) in C.BONES.items()}


def seg_dist(P, b):
    a, c = BSEG[b]
    d = c - a
    t = np.clip(((P - a) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)


def auto_weights(P, bones, sharp=4.0, top=3):
    P = np.asarray(P, np.float64)
    W = np.stack([1.0 / (seg_dist(P, b) + 0.015) ** sharp for b in bones], 1)
    if top < len(bones):
        thr = -np.sort(-W, 1)[:, top - 1:top]
        W = np.where(W >= thr, W, 0)
    W /= W.sum(1, keepdims=True)
    return {b: W[:, k] for k, b in enumerate(bones)}


def add_weights(ob, wdict, scale=1.0, acc=None):
    """wdict: bone -> per-vertex weight array. Accumulates when called repeatedly."""
    for b, w in wdict.items():
        vg = ob.vertex_groups.get(b) or ob.vertex_groups.new(name=b)
        for vi in np.nonzero(w > 1e-3)[0]:
            vg.add([int(vi)], float(w[vi] * scale), "ADD")


def skinned(name, verts, faces, uvs, mat, rig, weights, solid=0.0, subsurf=1, smooth=True):
    ob = mesh_obj(name, verts, faces, mat, uvs=uvs, smooth=smooth)
    add_weights(ob, weights)
    am = ob.modifiers.new("Armature", "ARMATURE")     # first: solidify/subsurf see the posed shape
    am.object = rig
    ob.parent = rig
    if solid:
        so = ob.modifiers.new("Solidify", "SOLIDIFY")
        so.thickness = solid
        so.offset = -1.0
        so.use_even_offset = True
    if subsurf:
        ss = ob.modifiers.new("Subsurf", "SUBSURF")
        ss.levels = 0
        ss.render_levels = subsurf
    return ob


def blend_w(*parts):
    """Sum weighted dicts: parts = (factor_array, dict)."""
    out = {}
    for fac, d in parts:
        for b, w in d.items():
            out[b] = out.get(b, 0) + fac * w
    tot = sum(out.values())
    return {b: w / np.maximum(tot, 1e-9) for b, w in out.items()}


def skirt(name, z_top, z_bot, prof, mat, rig, n=40, open_front=0.0, pleats=0.0, back_up=0.0,
          rows=14, solid=0.008):
    """A skirt tube (hakama / long coat) from the waist down; legs drive the hem L/R,
    panel spring bones add lag. prof(t) -> (rx, ry, y_offset)."""
    rings = []
    a0 = math.pi / 2 + open_front
    a1 = math.pi / 2 + 2 * math.pi - open_front
    closed = open_front <= 0
    for j in range(rows + 1):
        t = j / rows
        z = z_top + (z_bot - z_top) * t
        rx, ry, yo = prof(t)
        rf = (lambda ang, t=t: 1 + pleats * t * np.cos(ang * 9) ** 8) if pleats else None
        r = ring((0, yo, z), (1, 0, 0), (0, 1, 0), rx, ry, n, a0, a1, closed=closed, rfun=rf)
        if back_up:
            r[:, 2] += back_up * np.clip(-r[:, 1] / ry, 0, 1) ** 2 * (1 - t) ** 3
        rings.append(r)
    verts, faces, uvs = loft(rings, closed=closed)
    P = np.array(verts)
    depth = np.clip((z_top - P[:, 2]) / (z_top - z_bot), 0, 1)
    side = np.clip(P[:, 0] / 0.12, -1, 1) * 0.5 + 0.5        # 1 = right
    legR = auto_weights(P, ["thigh.R", "shin.R"], 3)
    legL = auto_weights(P, ["thigh.L", "shin.L"], 3)
    ang = np.arctan2(P[:, 1], P[:, 0])
    pan = {"skirt.F": np.clip(np.cos(ang - math.pi / 2), 0, 1) ** 2,
           "skirt.B": np.clip(np.cos(ang + math.pi / 2), 0, 1) ** 2,
           "skirt.R": np.clip(np.cos(ang), 0, 1) ** 2,
           "skirt.L": np.clip(np.cos(ang - math.pi), 0, 1) ** 2}
    ps = sum(pan.values())
    pan = {b: w / ps for b, w in pan.items()}
    hipw = np.clip(1 - depth * 2.2, 0, 1)
    legw = np.clip(depth * 1.6 - 0.15, 0, 1) * 0.75
    panw = np.clip(depth * 1.4, 0, 1) * 0.55
    w = blend_w((hipw * 0.6, {"hips": np.ones(len(P))}), (hipw * 0.4, {"spine": np.ones(len(P))}),
                (legw * side, legR), (legw * (1 - side), legL), (panw, pan))
    return skinned(name, verts, faces, uvs, mat, rig, w, solid=solid, subsurf=1)


def sleeve(name, side, rig, mat, bag=0.20, wide=0.11, n=22, rows=12):
    """Kimono sleeve: a tube along the arm with a bag hanging below the forearm."""
    sg = 1 if side == "R" else -1
    sh = np.array(C.BONES[f"upper.{side}"][0], float)
    el = np.array(C.BONES[f"fore.{side}"][0], float)
    wr = np.array(C.BONES[f"hand.{side}"][0], float)
    rings = []
    down = np.array([0, 0, -1.0])
    for j in range(rows + 1):
        t = j / rows
        p = sh + (el - sh) * (t * 2) if t < 0.5 else el + (wr - el) * ((t - 0.5) * 2)
        ax = C.nrm(el - sh) if t < 0.5 else C.nrm(wr - el)
        a = C.nrm(np.cross(ax, down))          # horizontal across the arm
        b = C.nrm(np.cross(a, ax))             # 'down-ish' perpendicular
        if np.dot(b, down) < 0:
            b = -b
        r0 = 0.07 + 0.03 * t
        bg = bag * np.clip((t - 0.25) / 0.35, 0, 1) ** 0.7

        def rf(angv, bg=bg, r0=r0):
            dn = np.clip(np.sin(angv), 0, 1)
            return 1 + (bg / r0) * dn ** 1.6
        rr = ring(p, a, b, wide * (0.75 + 0.35 * t), r0, n, rfun=rf)
        rings.append(rr)
    verts, faces, uvs = loft(rings, closed=True)
    P = np.array(verts)
    base = auto_weights(P, [f"upper.{side}", f"fore.{side}", "chest"], 4)
    ld = np.array([np.dot(p - (sh + el) / 2, down) for p in P])
    bagw = np.clip((np.array([np.dot(p - wr, -C.nrm(wr - sh)) for p in P]) * 0 + 1), 0, 1)
    # how far each vertex hangs below the arm line
    axs = C.nrm(wr - sh)
    rel = P - sh
    proj = sh + np.outer(rel @ axs, axs)
    hang = np.clip(np.sum((P - proj) * down, 1) / max(bag, 1e-3), 0, 1) ** 1.2
    w = blend_w((1 - hang * 0.8, base), (hang * 0.8, {f"sleeve.{side}": np.ones(len(P))}))
    return skinned(name, verts, faces, uvs, mat, rig, w, solid=0.006, subsurf=1)


def torso(name, rig, mat, rings_spec, n=32, neck_open=True):
    rings = []
    for z, rx, ry, yo, sq in rings_spec:
        r = ring((0, yo, z), (1, 0, 0), (0, 1, 0), rx, ry, n, sq=sq)
        if z > 1.38:                       # sloped shoulders: the outer edge drops toward the arm
            r[:, 2] -= 0.075 * np.clip(np.abs(r[:, 0]) / 0.22, 0, 1) ** 2 * min(1.0, (z - 1.38) / 0.05)
        rings.append(r)
    verts, faces, uvs = loft(rings, closed=True)
    P = np.array(verts)
    w = auto_weights(P, ["chest", "spine", "hips", "neck", "upper.R", "upper.L"], 4)
    return skinned(name, verts, faces, uvs, mat, rig, w, solid=0.006, subsurf=1)


def limb_tube(name, rig, mat, bones, radii, n=16, side_off=None):
    """A tube through bone chain heads/tails with radii per station; weights to the chain."""
    pts = [np.array(C.BONES[bones[0]][0], float)] + [np.array(C.BONES[b][1], float) for b in bones]
    rings = []
    for j, (p, r) in enumerate(zip(pts, radii)):
        ax = C.nrm(pts[min(j + 1, len(pts) - 1)] - pts[max(j - 1, 0)])
        a = C.nrm(np.cross(ax, (0, 1.0, 0)) if abs(ax[1]) < 0.9 else np.cross(ax, (1.0, 0, 0)))
        b = np.cross(ax, a)
        rings.append(ring(p, a, b, r, r, n))
    # densify: add mid rings
    dense = []
    for j in range(len(rings) - 1):
        for s in (0.0, 0.33, 0.66):
            dense.append(rings[j] * (1 - s) + rings[j + 1] * s)
    dense.append(rings[-1])
    verts, faces, uvs = loft(dense, closed=True, cap0=False, cap1=False)
    w = auto_weights(np.array(verts), bones, 4, top=2)
    return skinned(name, verts, faces, uvs, mat, rig, w, solid=0.0, subsurf=1)


def head_mesh(name, rig, mat, c=(0, 0.015, 1.665), r=(0.082, 0.098, 0.118)):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    try:
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=1.0)
        for v in bm.verts:
            x, y, z = v.co
            jaw = 1.0 - 0.18 * max(0.0, -z) * max(0.0, y)
            v.co = (x * r[0] * jaw + c[0], y * r[1] + c[1] + 0.012 * max(0.0, -z), z * r[2] + c[2])
        bm.to_mesh(me)
    finally:
        bm.free()
    verts = [tuple(v.co) for v in me.vertices]
    faces = [tuple(p.vertices) for p in me.polygons]
    bpy.data.meshes.remove(me)
    w = {"head": np.ones(len(verts))}
    return skinned(name, verts, faces, None, mat, rig, w, solid=0.0, subsurf=1)


def kasa(name, rig, mat, cord_mat):
    """Conical straw hat, tilted forward so the brim hides the face."""
    n, rows = 48, 10
    rings = []
    for j in range(rows + 1):
        t = j / rows
        r = 0.005 + 0.335 * t
        z = 1.885 - 0.175 * t ** 1.08 - 0.012 * max(0, t - 0.85) / 0.15
        rings.append(ring((0, 0.0, z), (1, 0, 0), (0, 1, 0), r, r, n))
    verts, faces, uvs = loft(rings, closed=True)
    P = np.array(verts)
    tilt = Matrix.Rotation(math.radians(-9), 3, "X")
    P = np.array([tuple(tilt @ Vector((p[0], p[1], p[2] - 1.70))) for p in P]) + np.array([0, 0.03, 1.70])
    ob = skinned(name, P.tolist(), faces, uvs, mat, rig, {"head": np.ones(len(P))}, solid=0.007, subsurf=1)
    # chin cords
    mb_v, mb_f = [], []
    for s in (1, -1):
        a = np.array([s * 0.075, 0.02, 1.70])
        b = np.array([s * 0.02, 0.085, 1.565])
        for k in range(6):
            pass
        rr = [ring(a + (b - a) * u, (1, 0, 0), C.nrm(np.cross(b - a, (1, 0, 0))), 0.004, 0.004, 6) for u in
              np.linspace(0, 1, 6)]
        v_, f_, _ = loft(rr, closed=True)
        o = len(mb_v)
        mb_v += v_
        mb_f += [tuple(x + o for x in f) for f in f_]
    skinned(name + "Cord", mb_v, mb_f, None, cord_mat, rig, {"head": np.ones(len(mb_v))}, subsurf=0)
    return ob


def hood(name, rig, mat):
    """Deep rounded hood: a shell around the head, open at the face (opening widest at eye
    level), the front edge pulled forward into a peak, the back draping onto the shoulders."""
    c = np.array([0.0, -0.005, 1.665])
    nth, nph = 30, 16
    verts, faces = [], []
    for j in range(nph + 1):
        ph = math.radians(8 + 150 * j / nph)            # from near the crown down past the chin
        alpha = math.radians(62) * math.sin(min(math.pi, (ph - math.radians(22)) / math.radians(118) * math.pi)) \
            if ph > math.radians(22) else 0.0
        alpha = max(alpha, math.radians(48) if ph > math.radians(110) else alpha)
        R = 0.135 + 0.02 * math.sin(ph) + (0.06 * ((ph - math.radians(115)) / math.radians(43)) ** 1.5
                                           if ph > math.radians(115) else 0.0)
        for k in range(nth + 1):
            th = alpha + (2 * math.pi - 2 * alpha) * k / nth          # 0 = forward (+Y)
            x = R * math.sin(ph) * -math.sin(th) * 1.0
            y = R * math.sin(ph) * math.cos(th) * 1.08
            z = R * math.cos(ph) * 1.05
            edge = 1.0 - min(1.0, min(k, nth - k) / 3.0)            # 1 on the face opening
            y += 0.035 * edge * math.sin(ph) + 0.03 * max(0.0, math.cos(ph)) * (1 - min(1.0, abs(th - math.pi) / 2.2))
            z -= 0.012 * edge
            if ph > math.radians(118):                              # drape: flare onto the shoulders
                f = (ph - math.radians(118)) / math.radians(40)
                x *= 1 + 0.9 * f
                y = y * (1 + 0.5 * f) - 0.03 * f
            verts.append(tuple(c + (x, y, z)))
        if j:
            o = j * (nth + 1)
            for k in range(nth):
                faces.append((o - nth - 1 + k, o - nth - 1 + k + 1, o + k + 1, o + k))
    P = np.array(verts)
    w_head = np.clip((P[:, 2] - 1.50) / 0.10, 0, 1)
    w = blend_w((w_head, {"head": np.ones(len(P))}),
                ((1 - w_head) * 0.45, {"hood": np.ones(len(P))}),
                ((1 - w_head) * 0.55, {"chest": np.ones(len(P))}))
    return skinned(name, verts, faces, None, mat, rig, w, solid=0.012, subsurf=1)


def fist(name, side, rig, mat):
    """A gloved fist modelled in SWORD space around the grip, mapped into the hand's rest frame
    (so when the hand bone takes the grip orientation the fist closes around the hilt)."""
    gy = C.GRIP_R_Y if side == "R" else C.GRIP_L_Y
    grip = np.array([0.0, gy, 0.0])
    wrist = grip - C.HAND_DIR_S * C.WRIST_BACK
    G = C.frame_yz(C.HAND_DIR_S, np.array([1.0, 0, 0]))
    hb = f"hand.{side}"
    arm_b = rig.data.bones[hb]
    Rr = np.array(arm_b.matrix_local)[:3, :3]
    head = np.array(arm_b.head_local)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    try:
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=3, use_grid_fill=True)
        for v in bm.verts:
            x, y, z = v.co
            # superellipsoid-ish fist: along the hilt (y) 0.10, across 0.075, depth 0.085
            k = max(abs(x), abs(y), abs(z)) * 2 + 1e-9
            nx, ny, nz = x * 2, y * 2, z * 2
            l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
            s = (abs(nx) ** 4 + abs(ny) ** 4 + abs(nz) ** 4) ** 0.25 / l
            nx, ny, nz = nx * s, ny * s, nz * s
            v.co = (nx * 0.040, ny * 0.050, nz * 0.044 - 0.006)
        bm.to_mesh(me)
    finally:
        bm.free()
    Pg = np.array([v.co for v in me.vertices]) + grip
    # the back of the hand toward the wrist: stretch verts on the wrist side
    toward = (Pg - grip) @ (-C.HAND_DIR_S)
    Pg = Pg + np.outer(np.clip(toward, 0, None) * 0.9, -C.HAND_DIR_S)
    faces = [tuple(p.vertices) for p in me.polygons]
    bpy.data.meshes.remove(me)
    Pr = head + (Rr @ (G.T @ (Pg - wrist).T)).T
    return skinned(name, Pr.tolist(), faces, None, mat, rig, {hb: np.ones(len(Pr))}, subsurf=1)


def foot_mesh(name, side, rig, mat, sole_mat, boot=False):
    sg = 1 if side == "R" else -1
    a = np.array(C.BONES[f"foot.{side}"][0], float)
    rings = []
    stations = [(-0.07, 0.035, 0.045, 0.06), (-0.02, 0.045, 0.05, 0.05), (0.06, 0.048, 0.04, 0.035),
                (0.13, 0.045, 0.03, 0.025), (0.19, 0.03, 0.02, 0.018)]
    for y, rx, rz, zc in stations:
        rings.append(ring(a + (0, y, zc - 0.07), (1, 0, 0), (0, 0, 1), rx, rz, 16, sq=2.6))
    if boot:
        for z in (0.16, 0.26, 0.34):
            rings.insert(0, ring((a[0], a[1] - 0.01, z), (1, 0, 0), (0, 1, 0), 0.058, 0.062, 16))
    verts, faces, uvs = loft(rings, closed=True, cap0=not boot, cap1=True)
    P = np.array(verts)
    w = blend_w((np.clip((P[:, 2] - 0.10) / 0.12, 0, 1), {f"shin.{side}": np.ones(len(P))}),
                (1 - np.clip((P[:, 2] - 0.10) / 0.12, 0, 1), {f"foot.{side}": np.ones(len(P))}))
    ob = skinned(name, verts, faces, uvs, mat, rig, w, subsurf=1)
    # sole
    sv, sf = [], []
    r0 = ring(a + (0, 0.06, -0.085), (1, 0, 0), (0, 1, 0), 0.052, 0.15, 20)
    r1 = r0 + (0, 0, 0.012)
    v_, f_, _ = loft([r0, r1], closed=True, cap0=True, cap1=True)
    skinned(name + "Sole", v_, f_, None, sole_mat, rig, {f"foot.{side}": np.ones(len(v_))}, subsurf=0)
    return ob


def saya_obj(name, rig):
    """Lacquered scabbard on the left hip; a shape key slides it back along its axis (saya-biki)."""
    L = 0.76
    d = C.SAYA_DIR
    up = C.nrm(np.cross(d, np.cross((0, 0, 1.0), d)))
    side = np.cross(d, up)
    rings = []
    for t in np.linspace(0, 1, 14):
        sori = -0.02 * (t ** 2)
        c = C.SAYA_LOCAL + d * (L * t) + up * sori
        rings.append(ring(c, side, up, 0.016 - 0.004 * t, 0.024 - 0.006 * t, 14))
    verts, faces, uvs = loft(rings, closed=True, cap0=False, cap1=True)
    ob = mesh_obj(name, verts, faces, M_SAYA, uvs=uvs)
    ob.data.materials.append(M_KOIGUCHI)
    # koiguchi (mouth band) material on the first ring of faces
    mi = np.zeros(len(ob.data.polygons), np.int32)
    mi[:14] = 1
    ob.data.polygons.foreach_set("material_index", mi)
    vg = ob.vertex_groups.new(name="hips")
    vg.add(list(range(len(verts))), 1.0, "REPLACE")
    ob.shape_key_add(name="Basis")
    sk = ob.shape_key_add(name="pull")
    co = np.array(verts) + d * C.SAYA_PULL
    sk.data.foreach_set("co", co.astype(np.float32).ravel())
    am = ob.modifiers.new("Armature", "ARMATURE")
    am.object = rig
    ob.parent = rig
    return ob


def build_sword(name, blade_mat):
    """Katana in sword space: tsuba at the origin, blade +Y with sori toward -Z, edge +Z."""
    parts = []
    L = C.BLADE_LEN
    # blade cross-section stations (u across: 0 spine (-Z) -> 1 edge (+Z))
    rows = 40
    sec = [(-0.016, 0.0), (-0.012, 0.0032), (0.003, 0.0034), (0.016, 0.0)]  # (z, half-thickness x)
    verts, faces, uvs = [], [], []
    ring_n = 2 * len(sec) - 2
    for j in range(rows + 1):
        t = j / rows
        y = 0.035 + (L - 0.035) * t
        w = 1.0 - 0.28 * t
        tip = max(0.0, (t - 0.9) / 0.1)
        z_curve = -0.018 * t ** 2
        scale_w = w * (1 - tip ** 1.8 * 0.97)
        pts = []
        for z, hx in sec:
            pts.append((hx * (1 - 0.3 * t) * (1 - tip * 0.9), y, z * scale_w + z_curve - 0.004 * tip ** 2))
        for z, hx in reversed(sec[1:-1]):
            pts.append((-hx * (1 - 0.3 * t) * (1 - tip * 0.9), y, z * scale_w + z_curve - 0.004 * tip ** 2))
        o = len(verts)
        verts += pts
        us = [0, 0.15, 0.6, 1.0, 0.6, 0.15]
        uvs += [(us[k], t) for k in range(ring_n)]
        if j:
            for k in range(ring_n):
                k2 = (k + 1) % ring_n
                faces.append((o - ring_n + k, o - ring_n + k2, o + k2, o + k))
    bl = mesh_obj(name + "_blade", verts, faces, blade_mat, uvs=uvs)
    bl.data.polygons.foreach_set("use_smooth", np.zeros(len(bl.data.polygons), bool))
    ws = bl.modifiers.new("WN", "WEIGHTED_NORMAL")
    parts.append(bl)
    # habaki
    hv, hf, hu = loft([ring((0, y, -0.002), (1, 0, 0), (0, 0, 1), 0.0065, 0.018, 16) for y in (0.006, 0.035)],
                      closed=True, cap0=True, cap1=True)
    parts.append(mesh_obj(name + "_habaki", hv, hf, M_HABAKI, uvs=hu))
    # tsuba: oval disc
    tv, tf, tu = loft([ring((0, y, 0), (1, 0, 0), (0, 0, 1), 0.037, 0.041, 32, sq=2.4) for y in (-0.004, 0.004)],
                      closed=True, cap0=True, cap1=True)
    parts.append(mesh_obj(name + "_tsuba", tv, tf, M_TSUBA, uvs=tu))
    # tsuka (hilt)
    tk = []
    for y in np.linspace(-0.006, -C.HILT_LEN, 10):
        t = -y / C.HILT_LEN
        bulge = 1 + 0.06 * math.sin(math.pi * t)
        tk.append(ring((0, y, 0.0), (1, 0, 0), (0, 0, 1), 0.0125 * bulge, 0.0165 * bulge, 16))
    kv, kf, ku = loft(tk, closed=True, cap0=False, cap1=True)
    parts.append(mesh_obj(name + "_tsuka", kv, kf, M_TSUKA, uvs=ku))
    root = empty(name)
    root.rotation_mode = "QUATERNION"
    for p in parts:
        p.parent = root
    return root, parts


# ---------------------------------------------------------------------------------------
def build_fighter(i):
    tag = "A" if i == 0 else "B"
    rig = build_armature("Rig" + tag)
    objs = []
    head_mesh(tag + "_Head", rig, M_WRAP)
    limb_tube(tag + "_Neck", rig, M_WRAP, ["neck"], [0.058, 0.052])
    for s in "RL":
        fist(f"{tag}_Fist.{s}", s, rig, M_GLOVE)
    saya_obj(tag + "_Saya", rig)
    if i == 0:
        # ronin: kimono top, wide sleeves, obi, hakama, kasa, tabi + waraji
        torso(tag + "_Kimono", rig, M_KIMONO, [
            (1.50, 0.07, 0.065, 0.005, 2.0), (1.455, 0.16, 0.10, 0.0, 2.6), (1.415, 0.225, 0.125, 0.0, 3.0),
            (1.33, 0.205, 0.135, 0.005, 2.6), (1.20, 0.18, 0.128, 0.01, 2.4), (1.06, 0.175, 0.125, 0.01, 2.2),
            (0.98, 0.18, 0.13, 0.01, 2.2)])
        for s in "RL":
            sleeve(f"{tag}_Sleeve.{s}", s, rig, M_KIMONO)
            limb_tube(f"{tag}_Forearm.{s}", rig, M_WRAP, [f"fore.{s}"], [0.045, 0.034])
        # collar (eri): the lighter inner kimono crossing left over right
        cv, cf = [], []
        for s, z0 in ((1, 1.50), (-1, 1.50)):
            a = np.array([s * 0.065, 0.02, z0])
            b = np.array([-s * 0.035, 0.135, 1.20])
            strip = []
            for u in np.linspace(0, 1, 10):
                p = a + (b - a) * u
                p[1] = max(p[1], 0.07 + 0.065 * math.sin(math.pi * min(1.0, u * 1.2)) * 0.9) + 0.004
                strip.append([p - np.array([s * 0.018, 0, 0.004]), p + np.array([s * 0.018, 0, 0.004])])
            o = len(cv)
            for q in strip:
                cv += [tuple(q[0]), tuple(q[1])]
            for k in range(len(strip) - 1):
                cf.append((o + 2 * k, o + 2 * k + 1, o + 2 * k + 3, o + 2 * k + 2))
        skinned(tag + "_Collar", cv, cf, None, M_COLLAR, rig,
                auto_weights(np.array(cv), ["chest", "spine", "neck"], 4), solid=0.004, subsurf=1)
        skirt(tag + "_Hakama", 1.06, 0.11,
              lambda t: (0.19 + 0.17 * t ** 0.8, 0.145 + 0.17 * t ** 0.8, 0.01), M_HAKAMA, rig,
              n=48, pleats=0.05, back_up=0.07, rows=16)
        obi_v, obi_f, obi_u = loft([ring((0, 0.01, z), (1, 0, 0), (0, 1, 0), 0.196, 0.15, 32, sq=2.2)
                                    for z in (0.975, 1.03, 1.085)], closed=True)
        skinned(tag + "_Obi", obi_v, obi_f, obi_u, M_OBI, rig,
                auto_weights(np.array(obi_v), ["hips", "spine"], 4), solid=0.01, subsurf=1)
        kasa(tag + "_Kasa", rig, M_STRAW, M_CORD)
        for s in "RL":
            foot_mesh(f"{tag}_Foot.{s}", s, rig, M_WRAP, M_SOLE)
            limb_tube(f"{tag}_Shin.{s}", rig, M_WRAP, [f"shin.{s}"], [0.05, 0.042])
    else:
        # hood: long coat (front slit), slim sleeves, belt, trousers, boots
        torso(tag + "_CoatTop", rig, M_COAT, [
            (1.56, 0.075, 0.07, -0.005, 2.0), (1.47, 0.11, 0.095, 0.0, 2.2), (1.42, 0.225, 0.13, 0.0, 3.0),
            (1.32, 0.21, 0.14, 0.005, 2.6), (1.18, 0.185, 0.132, 0.01, 2.4), (1.03, 0.18, 0.13, 0.01, 2.2),
            (0.95, 0.19, 0.14, 0.01, 2.2)])
        skirt(tag + "_CoatSkirt", 1.0, 0.30,
              lambda t: (0.19 + 0.17 * t, 0.145 + 0.16 * t, 0.01), M_COAT, rig, n=44,
              open_front=0.22, rows=14, solid=0.01)
        for s in "RL":
            limb_tube(f"{tag}_Sleeve.{s}", rig, M_COAT, [f"upper.{s}", f"fore.{s}"], [0.068, 0.062, 0.052])
            limb_tube(f"{tag}_Leg.{s}", rig, M_TROUSER, [f"thigh.{s}", f"shin.{s}"], [0.085, 0.064, 0.05])
            foot_mesh(f"{tag}_Boot.{s}", s, rig, M_BOOT, M_SOLE, boot=True)
        bv, bf, bu = loft([ring((0, 0.01, z), (1, 0, 0), (0, 1, 0), 0.188, 0.142, 32, sq=2.2)
                           for z in (0.985, 1.02, 1.055)], closed=True)
        skinned(tag + "_Belt", bv, bf, bu, M_BOOT, rig, auto_weights(np.array(bv), ["hips", "spine"], 4),
                solid=0.008, subsurf=1)
        hood(tag + "_Hood", rig, M_COAT)
    sword, parts = build_sword(tag + "_Sword", M_BLADE[i])
    return rig, sword, parts


FIGHTERS = [build_fighter(0), build_fighter(1)]
RIGS = []
for i, (rig, sword, parts) in enumerate(FIGHTERS):
    r = Rig(rig, i)
    r.solve()
    r.bake()
    RIGS.append(r)
    for side in ("R", "L"):
        e = r.ik_err[side]
        gw = DU.grip_w[i][0 if side == "R" else 1]
        print(f"[rig] fighter {i} hand.{side}: IK miss while gripping  mean {np.mean(e[gw > 0.99]) * 100:.2f} cm,"
              f" max {np.max(e[gw > 0.99]) * 100:.1f} cm")
    print(f"[rig] fighter {i} legs: miss max {max(r.ik_err['legR'].max(), r.ik_err['legL'].max()) * 100:.1f} cm")
    # sword transform (world)
    q = C.qcontinuous(C.mat2quat(DU.sword_rot[i]))
    bake_vec(sword, "location", DU.sword_pos[i])
    for k in range(4):
        bake(sword, "rotation_quaternion", q[:, k], index=k)
    # saya slide (shape key)
    sy = bpy.data.objects[("A" if i == 0 else "B") + "_Saya"]
    bake(sy.data.shape_keys, 'key_blocks["pull"].value', DU.saya_off / C.SAYA_PULL)
    # blade edge glow
    bake(M_BLADE[i].node_tree, value_path("glow"), DU.blade_glow)

print("[build] characters done")


# =====================================================================================
# 5. ENVIRONMENT MATERIALS
# =====================================================================================
def ground_material():
    """Wet flagstones (plaza) / cobbles (lanes), puddles with live rain ripples, a keyed
    shockwave ring and four keyed footstep ripple slots."""
    m, nt = new_mat("Ground")
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    tc = nd(nt, "ShaderNodeTexCoord")
    pos = tc.outputs["Object"]
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, pos, sep.inputs[0])
    lane = nd(nt, "ShaderNodeMapRange", inp={"From Min": 6.2, "From Max": 6.8, "To Min": 0.0, "To Max": 1.0})
    lk(nt, M(nt, "ABSOLUTE", sep.outputs[1]), lane.inputs["Value"])
    # stones: voronoi distance-to-edge (flagstones 1.6/m, cobbles 5/m)
    vf = nd(nt, "ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE", inp={"Scale": 1.15, "Randomness": 0.7})
    vc = nd(nt, "ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE", inp={"Scale": 5.0, "Randomness": 0.9})
    vfc = nd(nt, "ShaderNodeTexVoronoi", feature="F1", inp={"Scale": 1.15, "Randomness": 0.7})
    vcc = nd(nt, "ShaderNodeTexVoronoi", feature="F1", inp={"Scale": 5.0, "Randomness": 0.9})
    for v in (vf, vc, vfc, vcc):
        lk(nt, pos, v.inputs["Vector"])
    edge = mix_f(nt, lane.outputs[0], vf.outputs["Distance"], vc.outputs["Distance"])
    cellc = mix_col(nt, lane.outputs[0], vfc.outputs["Color"], vcc.outputs["Color"])
    grout = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.0, "From Max": 0.028, "To Min": 1.0, "To Max": 0.0})
    lk(nt, edge, grout.inputs["Value"])
    gr = grout.outputs[0]
    # stone colour: per-stone tint + fine grain
    grain = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 35.0, "Detail": 8.0})
    lk(nt, pos, grain.inputs["Vector"])
    cbw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, cellc, cbw.inputs[0])
    tint = M(nt, "ADD", 0.55, M(nt, "MULTIPLY", cbw.outputs[0], 0.7))
    stone = mix_col(nt, grain.outputs["Fac"], (0.06, 0.058, 0.055), (0.12, 0.115, 0.11))
    stone = mix_col(nt, 1.0, stone, (1, 1, 1), "MULTIPLY")
    tint3 = nd(nt, "ShaderNodeCombineXYZ")
    for k in range(3):
        lk(nt, tint, tint3.inputs[k])
    stone = mix_col(nt, 1.0, stone, tint3.outputs[0], "MULTIPLY")
    base = mix_col(nt, M(nt, "MULTIPLY", gr, 0.7), stone, (0.025, 0.025, 0.025))
    # puddles: large noise, wider in the plaza centre
    pn = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.32, "Detail": 3.0, "Roughness": 0.55})
    lk(nt, pos, pn.inputs["Vector"])
    centre = nd(nt, "ShaderNodeMapRange", inp={"From Min": 2.0, "From Max": 8.0, "To Min": 0.07, "To Max": 0.0})
    lk(nt, V(nt, "LENGTH", V(nt, "MULTIPLY", pos, (1.0, 1.0, 0.0))), centre.inputs["Value"])
    pm = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.50, "From Max": 0.53, "To Min": 0.0, "To Max": 1.0})
    lk(nt, M(nt, "ADD", pn.outputs["Fac"], centre.outputs[0]), pm.inputs["Value"])
    puddle = M(nt, "MAXIMUM", pm.outputs[0], M(nt, "MULTIPLY", gr, 0.6))
    # rain ripples: two voronoi layers, each cell an expanding ring with its own phase
    tval = value_node(nt, "rain_t", 0.0)

    def ripples(scale, rate, seed_off):
        vr = nd(nt, "ShaderNodeTexVoronoi", feature="F1", inp={"Scale": scale, "Randomness": 1.0})
        off = V(nt, "ADD", pos, (seed_off, seed_off * 0.7, 0.0))
        lk(nt, off, vr.inputs["Vector"])
        cs = nd(nt, "ShaderNodeSeparateColor")
        lk(nt, vr.outputs["Color"], cs.inputs[0])
        ph = M(nt, "FRACT", M(nt, "ADD", M(nt, "MULTIPLY", tval.outputs[0], rate), cs.outputs[0]))
        dist = M(nt, "MULTIPLY", vr.outputs["Distance"], 1.0)
        rr = M(nt, "SUBTRACT", dist, M(nt, "MULTIPLY", ph, 0.55))
        band = M(nt, "EXPONENT", M(nt, "MULTIPLY", M(nt, "MULTIPLY", rr, rr), -900.0))
        fade = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, ph), 2.0)
        return M(nt, "MULTIPLY", band, fade)
    rip = M(nt, "ADD", ripples(9.0, 1.7, 0.0), ripples(13.0, 2.1, 3.3))
    # keyed rings: shockwave + footstep slots (centre x, y, radius, strength)
    rings = []
    for nm in ["shock"] + [f"step{k}" for k in range(4)]:
        cx, cy = value_node(nt, nm + "_x", 0.0), value_node(nt, nm + "_y", 0.0)
        rr_, st = value_node(nt, nm + "_r", 0.0), value_node(nt, nm + "_s", 0.0)
        cc = nd(nt, "ShaderNodeCombineXYZ")
        lk(nt, cx.outputs[0], cc.inputs[0])
        lk(nt, cy.outputs[0], cc.inputs[1])
        dd = V(nt, "DISTANCE", V(nt, "MULTIPLY", pos, (1.0, 1.0, 0.0)), cc.outputs[0])
        w = 0.06 if nm == "shock" else 0.025
        x = M(nt, "DIVIDE", M(nt, "SUBTRACT", dd, rr_.outputs[0]), w)
        ring_ = M(nt, "MULTIPLY", M(nt, "EXPONENT", M(nt, "MULTIPLY", M(nt, "MULTIPLY", x, x), -1.0)), st.outputs[0])
        rings.append(ring_)
    rsum = rings[0]
    for r_ in rings[1:]:
        rsum = M(nt, "ADD", rsum, r_)
    height = M(nt, "ADD", M(nt, "MULTIPLY", rip, M(nt, "ADD", M(nt, "MULTIPLY", puddle, 0.8), 0.15)), rsum)
    stone_h = M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, gr), M(nt, "SUBTRACT", 1.0, puddle))
    stone_h = M(nt, "ADD", stone_h, M(nt, "MULTIPLY", grain.outputs["Fac"], 0.15))
    bp1 = nd(nt, "ShaderNodeBump", inp={"Strength": 0.18, "Distance": 0.02})
    lk(nt, M(nt, "MULTIPLY", stone_h, M(nt, "SUBTRACT", 1.0, puddle)), bp1.inputs["Height"])
    bp2 = nd(nt, "ShaderNodeBump", inp={"Strength": 0.22, "Distance": 0.004})
    lk(nt, height, bp2.inputs["Height"])
    lk(nt, bp1.outputs["Normal"], bp2.inputs["Normal"])
    lk(nt, bp2.outputs["Normal"], b.inputs["Normal"])
    dark = mix_col(nt, puddle, base, (0.012, 0.012, 0.013))
    lk(nt, dark, b.inputs["Base Color"])
    wetn = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 2.0, "Detail": 4.0})
    lk(nt, pos, wetn.inputs["Vector"])
    rstone = M(nt, "ADD", 0.16, M(nt, "MULTIPLY", wetn.outputs["Fac"], 0.22))
    lk(nt, mix_f(nt, puddle, rstone, 0.015), b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = 0.55
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def plaster_material():
    m, nt = new_mat("Plaster")
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    tc = nd(nt, "ShaderNodeTexCoord")
    st = nd(nt, "ShaderNodeMapping", inp={"Scale": (6.0, 6.0, 0.35)})
    lk(nt, tc.outputs["Object"], st.inputs[0])
    streak = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 3.0, "Detail": 6.0})
    lk(nt, st.outputs[0], streak.inputs["Vector"])
    blot = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.2, "Detail": 8.0})
    lk(nt, tc.outputs["Object"], blot.inputs["Vector"])
    dirt = M(nt, "MULTIPLY", streak.outputs["Fac"], blot.outputs["Fac"])
    base = mix_col(nt, dirt, (0.30, 0.28, 0.25), (0.07, 0.065, 0.06))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, mix_f(nt, streak.outputs["Fac"], 0.75, 0.30), b.inputs["Roughness"])
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.25, "Distance": 0.02})
    lk(nt, blot.outputs["Fac"], bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def wood_material(name="Wood", col=(0.035, 0.022, 0.014)):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    tc = nd(nt, "ShaderNodeTexCoord")
    mp = nd(nt, "ShaderNodeMapping", inp={"Scale": (1.0, 1.0, 14.0)})
    lk(nt, tc.outputs["Object"], mp.inputs[0])
    wv = nd(nt, "ShaderNodeTexWave", wave_type="RINGS", inp={"Scale": 3.0, "Distortion": 9.0, "Detail": 4.0})
    lk(nt, mp.outputs[0], wv.inputs["Vector"])
    base = mix_col(nt, wv.outputs["Fac"], col, tuple(c * 1.9 for c in col))
    lk(nt, base, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.42
    b.inputs["Coat Weight"].default_value = 0.35
    b.inputs["Coat Roughness"].default_value = 0.08
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.2, "Distance": 0.01})
    lk(nt, wv.outputs["Fac"], bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def paper_glow(name, color, strength, gain="lantern", grid=True):
    """Backlit shoji paper: emission x keyed gain, darker lattice frame lines."""
    m, nt = new_mat(name)
    tc = nd(nt, "ShaderNodeTexCoord")
    e = nd(nt, "ShaderNodeEmission", inp={"Color": tuple(color) + (1,)})
    g = value_node(nt, gain, 1.0)
    fac = 1.0
    if grid:
        sep = nd(nt, "ShaderNodeSeparateXYZ")
        lk(nt, tc.outputs["Object"], sep.inputs[0])
        gx = M(nt, "PINGPONG", M(nt, "MULTIPLY", M(nt, "ADD", sep.outputs[0], sep.outputs[1]), 3.4), 0.5)
        gz = M(nt, "PINGPONG", M(nt, "MULTIPLY", sep.outputs[2], 2.6), 0.5)
        lines = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.02, "From Max": 0.05, "To Min": 0.15, "To Max": 1.0})
        lk(nt, M(nt, "MINIMUM", gx, gz), lines.inputs["Value"])
        nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 4.0, "Detail": 3.0})
        lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
        fac = M(nt, "MULTIPLY", lines.outputs[0], M(nt, "ADD", 0.6, M(nt, "MULTIPLY", nz.outputs["Fac"], 0.8)))
    st = M(nt, "MULTIPLY", g.outputs[0], strength)
    if not isinstance(fac, float):
        st = M(nt, "MULTIPLY", st, fac)
    lk(nt, st, e.inputs["Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, e.outputs[0], o.inputs["Surface"])
    return m


def tile_material():
    m = principled("RoofTile", (0.035, 0.038, 0.042), 0.32, 0.0, bump=(14, 0.2), coat=0.8)
    return m


M_GROUND = ground_material()
M_PLASTER = plaster_material()
M_WOOD = wood_material()
M_WOOD2 = wood_material("WoodLight", (0.07, 0.045, 0.028))
M_TILE = tile_material()
M_FOUND = principled("Foundation", (0.09, 0.09, 0.09), 0.45, bump=(6, 0.5))
M_SHOJI = paper_glow("Shoji", blackbody(2600), 2.2)
M_WINDOW = paper_glow("WindowGlow", blackbody(2300), 1.4)
M_INTERIOR = paper_glow("Interior", blackbody(2400), 0.7, grid=False)
M_NOREN = cloth_material("Noren", (0.02, 0.035, 0.07), weave=300, wet=0.5)
M_METAL = principled("Metal", (0.03, 0.03, 0.032), 0.38, 0.8, bump=(30, 0.1), coat=0.3)
M_SIGNBOX = principled("SignBox", (0.015, 0.015, 0.017), 0.3, 0.2, coat=0.5)
M_TORII = principled("Torii", (0.42, 0.045, 0.02), 0.32, 0.0, bump=(8, 0.15), coat=0.6)
M_TORII_TOP = principled("ToriiTop", (0.01, 0.01, 0.01), 0.25, coat=0.6)
M_STONE = principled("StoneLantern", (0.13, 0.13, 0.125), 0.6, bump=(12, 0.6))
M_CABLE = principled("Cable", (0.01, 0.01, 0.01), 0.3)
M_LAMP_HEAD = emissive("LampHead", (0.75, 0.9, 1.0), 25.0)
M_LAMP_HEAD2 = emissive("LampHead2", blackbody(2500), 25.0)


def neon_mat(name, rgb, strength=14.0, gain="neon"):
    m, nt = new_mat(name)
    e = nd(nt, "ShaderNodeEmission", inp={"Color": tuple(rgb) + (1,)})
    g = value_node(nt, gain, 1.0)
    lk(nt, M(nt, "MULTIPLY", g.outputs[0], strength), e.inputs["Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, e.outputs[0], o.inputs["Surface"])
    return m


NEON_RGB = {"pink": (1.0, 0.10, 0.45), "cyan": (0.10, 0.80, 1.0), "amber": (1.0, 0.45, 0.06),
            "violet": (0.55, 0.15, 1.0), "green": (0.20, 1.0, 0.45), "red": (1.0, 0.06, 0.04),
            "white": (0.85, 0.9, 1.0)}
M_NEON = {k: neon_mat("Neon_" + k, v) for k, v in NEON_RGB.items()}
M_NEON_BAD = neon_mat("Neon_bad", NEON_RGB["cyan"], gain="neon_bad")
NEON_MATS = list(M_NEON.values())


# =====================================================================================
# 6. THE STREET: plaza + two lanes of old townhouses, gate, signs, lanterns
# =====================================================================================
ENV = collection("Street")
GEO = {k: MB() for k in ("plaster", "wood", "wood2", "tile", "found", "shoji", "window", "interior",
                         "noren", "metal", "signbox", "cable", "stone")}
LIGHTS = []                     # (light object, kind, base energy)
LANTERNS = []                   # (location, colour key, has_light)
SIGNS = []                      # (centre, normal, colour, size)


def rng_for(*k):
    return np.random.default_rng(int(hash01(*k) * 2 ** 31))


def hash01(*k):
    return C.hash01(*k)


def tile_strip(mb, a, b, z0, z1, depth, n_out, period=0.27):
    """A corrugated tiled roof slope along the wall a->b: from the wall (z1) out by depth (z0)."""
    a, b, n_out = (np.asarray(v, float) for v in (a, b, n_out))
    L = np.linalg.norm(b - a)
    t = (b - a) / L
    cols = max(2, int(L / period * 4))
    rows = 4
    pts, faces = [], []
    for r in range(rows + 1):
        u = r / rows
        for c in range(cols + 1):
            s = c / cols * L
            corr = 0.025 * abs(math.sin(math.pi * s / period))
            p = a + t * s + n_out * (depth * u)
            pts.append((p[0], p[1], z1 + (z0 - z1) * u + corr))
    for r in range(rows):
        for c in range(cols):
            k = r * (cols + 1) + c
            faces.append((k, k + 1, k + cols + 2, k + cols + 1))
    mb.poly(pts, faces)
    # the eave fascia + round end tiles along the edge
    e0 = a + n_out * depth
    e1 = b + n_out * depth
    mid = (e0 + e1) / 2
    GEO["wood"].box((mid[0], mid[1], z0 - 0.08), (abs(e1[0] - e0[0]) + 0.16 * abs(n_out[0]) + 0.08,
                                                    abs(e1[1] - e0[1]) + 0.16 * abs(n_out[1]) + 0.08, 0.16))


def facade(a, b, n_in, seed, floors=2, lights=True, signs=0, plaza=False):
    """One block of townhouses along the wall line a->b (ground points); n_in points into the street."""
    a, b, n_in = (np.asarray(v, float) for v in (a, b, n_in))
    L = np.linalg.norm(b - a)
    t = (b - a) / L
    rng = rng_for(seed)
    G = GEO

    def P(s, d, z=0.0):
        p = a + t * s + n_in * d
        return (p[0], p[1], z)

    def box_along(s0, s1, d0, d1, z0, z1, mb):
        c = a + t * ((s0 + s1) / 2) + n_in * ((d0 + d1) / 2)
        sx = abs(t[0]) * (s1 - s0) + abs(n_in[0]) * (d1 - d0)
        sy = abs(t[1]) * (s1 - s0) + abs(n_in[1]) * (d1 - d0)
        mb.box((c[0], c[1], (z0 + z1) / 2), (max(sx, 1e-3), max(sy, 1e-3), z1 - z0))

    H1, H2 = 2.95, 5.85
    # wall mass behind
    box_along(0, L, -0.35, 0.0, 0.25, H2 + 0.3, G["plaster"])
    box_along(0, L, -0.02, 0.07, 0.0, 0.26, G["found"])
    # bays
    nb = max(1, int(round(L / 1.9)))
    bw = L / nb
    for k in range(nb + 1):
        s = k * bw
        box_along(s - 0.07, s + 0.07, 0.0, 0.14, 0.0, H1, G["wood"])
        box_along(s - 0.06, s + 0.06, 0.0, 0.10, H1 + 0.2, H2, G["wood"])
    box_along(0, L, 0.0, 0.16, H1 - 0.12, H1 + 0.06, G["wood"])
    box_along(0, L, 0.0, 0.12, H1 + 0.2, H1 + 0.32, G["wood"])
    box_along(0, L, 0.0, 0.12, H2 - 0.1, H2 + 0.05, G["wood"])
    for k in range(nb):
        s0, s1 = k * bw + 0.07, (k + 1) * bw - 0.07
        kind = rng.choice(["lattice", "lattice", "door", "plaster", "shutter"], p=[0.3, 0.2, 0.2, 0.15, 0.15])
        if kind == "lattice":
            box_along(s0, s1, -0.04, -0.02, 0.32, 2.25, G["shoji"])
            box_along(s0, s1, 0.0, 0.08, 0.26, 0.55, G["wood"])
            ns = int((s1 - s0) / 0.06)
            for j in range(ns):
                ss = s0 + (j + 0.5) * (s1 - s0) / ns
                box_along(ss - 0.012, ss + 0.012, 0.02, 0.06, 0.55, 2.25, G["wood2"])
            box_along(s0, s1, 0.0, 0.07, 2.22, 2.30, G["wood"])
            box_along(s0, s1, -0.01, 0.0, 2.3, H1 - 0.12, G["plaster"])
        elif kind == "door":
            box_along(s0, s1, -0.9, -0.85, 0.0, 2.3, G["interior"])
            box_along(s0, s0 + 0.04, -0.85, 0.0, 0.0, 2.3, G["wood"])
            box_along(s1 - 0.04, s1, -0.85, 0.0, 0.0, 2.3, G["wood"])
            box_along(s0, s1, 0.0, 0.07, 2.22, 2.30, G["wood"])
            box_along(s0, s1, -0.01, 0.0, 2.3, H1 - 0.12, G["plaster"])
            # noren: 3-4 cloth panels hanging from a rod
            npnl = 3 if s1 - s0 < 1.6 else 4
            for j in range(npnl):
                p0 = s0 + 0.05 + j * (s1 - s0 - 0.1) / npnl
                p1 = p0 + (s1 - s0 - 0.1) / npnl - 0.02
                box_along(p0, p1, 0.10, 0.115, 1.45 - 0.05 * rng.random(), 2.2, G["noren"])
            box_along(s0, s1, 0.08, 0.13, 2.18, 2.22, G["metal"])
        elif kind == "plaster":
            box_along(s0, s1, -0.02, 0.0, 0.26, H1 - 0.12, G["plaster"])
            box_along(s0 + 0.2, s0 + 0.28, 0.06, 0.14, 0.0, H1, G["metal"])
        else:
            for j in range(18):
                z0 = 0.3 + j * 0.13
                box_along(s0, s1, 0.0, 0.035 + 0.01 * (j % 2), z0, z0 + 0.12, G["wood"])
        # upper floor window
        u = rng.random()
        if u < 0.55:
            ww = min(1.2, s1 - s0 - 0.3)
            c = (s0 + s1) / 2
            box_along(c - ww / 2, c + ww / 2, -0.03, -0.01, 3.7, 4.9, G["window" if rng.random() < 0.75 else "interior"])
            nsl = int(ww / 0.09)
            for j in range(nsl):
                ss = c - ww / 2 + (j + 0.5) * ww / nsl
                box_along(ss - 0.022, ss + 0.022, 0.0, 0.06, 3.7, 4.9, G["plaster" if u < 0.3 else "wood"])
            box_along(c - ww / 2 - 0.05, c + ww / 2 + 0.05, 0.0, 0.08, 3.62, 3.7, G["wood"])
            box_along(c - ww / 2 - 0.05, c + ww / 2 + 0.05, 0.0, 0.08, 4.9, 4.98, G["wood"])
        elif u < 0.75:
            box_along(s0 + 0.1, s1 - 0.1, 0.55, 0.6, 3.35, 4.1, G["wood"])     # balcony rail
            for j in range(int((s1 - s0) / 0.15)):
                ss = s0 + 0.15 + j * 0.15
                box_along(ss - 0.015, ss + 0.015, 0.55, 0.6, 3.3, 4.05, G["wood"])
            box_along(s0 + 0.1, s1 - 0.1, 0.0, 0.6, 3.28, 3.34, G["wood"])
            box_along(s0 + 0.2, s1 - 0.2, -0.03, -0.01, 3.4, 5.3, G["window"])
        # lantern under the eave
        if lights and (k % 2 == (seed % 2)):
            LANTERNS.append((P((s0 + s1) / 2, 0.62, 2.48), "red" if rng.random() < 0.7 else "white", plaza))
    # eaves + tiles
    tile_strip(G["tile"], P(0, 0.0)[:2], P(L, 0.0)[:2], H1 + 0.02, H1 + 0.42, 0.95, n_in)
    tile_strip(G["tile"], P(0, 0.0)[:2], P(L, 0.0)[:2], H2 + 0.08, H2 + 0.45, 0.75, n_in)
    # roof going back
    pts = [P(0, 0.0, H2 + 0.45), P(L, 0.0, H2 + 0.45), P(L, -2.5, H2 + 1.6), P(0, -2.5, H2 + 1.6)]
    G["tile"].poly(pts, [(0, 1, 2, 3)])
    # projecting neon signs
    for j in range(signs):
        s = L * (0.2 + 0.6 * (j + 0.5) / signs) + rng.uniform(-0.3, 0.3)
        col = rng.choice(list(NEON_RGB.keys())[:5])
        SIGNS.append((np.array(P(s, 0.55, 4.25)), t, str(col), (0.5, 1.9)))


# plaza (13 x 13) and lanes (5.2 wide)
W = 6.5
LW = 2.6
facade((-W, -W), (-W, W), (1, 0), 11, signs=2, plaza=True)              # west
facade((W, W), (W, -W), (-1, 0), 12, signs=1, plaza=True)               # east
facade((-W, W), (-LW, W), (0, -1), 13, signs=1, plaza=True)             # north-west
facade((LW, W), (W, W), (0, -1), 14, signs=1, plaza=True)               # north-east
facade((-LW, -W), (-W, -W), (0, 1), 15, signs=1, plaza=True)            # south-west
facade((W, -W), (LW, -W), (0, 1), 16, signs=1, plaza=True)              # south-east
facade((-LW, W), (-LW, 42), (1, 0), 21, signs=4)                        # north lane west
facade((LW, 42), (LW, W), (-1, 0), 22, signs=4)                         # north lane east
facade((-LW, -32), (-LW, -W), (1, 0), 23, signs=2)                      # south lane west
facade((LW, -W), (LW, -32), (-1, 0), 24, signs=2)                       # south lane east
# lane ends: dark blocks far away
for y0 in (42.0, -32.0):
    GEO["plaster"].box((0, y0 + (1.0 if y0 > 0 else -1.0), 4.0), (5.4, 0.4, 8.0))

# ground
gv = [(-7.2, -33, 0), (7.2, -33, 0), (7.2, 43, 0), (-7.2, 43, 0)]
GROUND = mesh_obj("Ground", gv, [(0, 1, 2, 3)], M_GROUND, uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
GROUND.data.polygons.foreach_set("use_smooth", np.zeros(1, bool))

# torii gate in the north lane
TY = 24.0
for sx in (-1.75, 1.75):
    pv, pf, pu = loft([ring((sx, TY, z), (1, 0, 0), (0, 1, 0), 0.22 - 0.02 * z / 5, 0.22 - 0.02 * z / 5, 20)
                       for z in (0.0, 2.5, 5.1)], closed=True, cap1=True)
    mesh_obj(f"ToriiPillar{sx}", pv, pf, M_TORII, uvs=pu)
    GEO["stone"].box((sx, TY, 0.12), (0.62, 0.62, 0.24))
nuki = MB()
nuki.box((0, TY, 4.25), (4.6, 0.18, 0.26))
nuki.box((0, TY, 4.75), (0.3, 0.16, 0.6))
nuki.obj("ToriiNuki", M_TORII)
kv = []
kf = []
for k in range(25):
    x = -3.3 + 6.6 * k / 24
    zc = 5.25 + 0.22 * (abs(x) / 3.3) ** 2.2
    kv += [(x, TY - 0.2, zc - 0.18), (x, TY + 0.2, zc - 0.18), (x, TY + 0.25, zc + 0.12), (x, TY - 0.25, zc + 0.12)]
for k in range(24):
    o = 4 * k
    for q in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        kf.append(tuple(o + i for i in q))
kf += [(0, 3, 2, 1), (96, 97, 98, 99)]
mesh_obj("ToriiKasagi", kv, kf, M_TORII_TOP)

# stone lanterns (toro) at two plaza corners
for (x, y) in ((-5.3, 5.2), (5.2, -5.3)):
    st = GEO["stone"]
    st.box((x, y, 0.15), (0.6, 0.6, 0.3))
    st.box((x, y, 0.8), (0.18, 0.18, 1.0))
    st.box((x, y, 1.35), (0.5, 0.5, 0.12))
    st.box((x, y, 1.62), (0.42, 0.42, 0.42))
    st.box((x, y, 1.9), (0.75, 0.75, 0.1))
    st.box((x, y, 2.02), (0.3, 0.3, 0.18))
    LANTERNS.append(((x, y, 1.62), "toro", True))

# overhead cables (sagging) across the plaza and lanes
cab = GEO["cable"]
for (x0, y0, x1, y1, z) in ((-W, -2.0, W, 1.5, 6.0), (-W, 3.5, W, 2.0, 5.7), (-LW, 12, LW, 13, 5.5),
                            (-LW, 20, LW, 18, 5.9), (-W, -4.5, W, -5.2, 5.6), (-LW, -12, LW, -11, 5.6)):
    p_prev = None
    for k in range(17):
        u = k / 16
        p = (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, z - 0.6 * math.sin(math.pi * u))
        if p_prev is not None:
            cab.beam(p_prev, p, 0.012)
        p_prev = p

# sign boxes + neon glyphs (pseudo-kanji strokes, no real text)
GLYPH_TUBES = {k: [] for k in NEON_RGB}


def tube_poly(pts, r, nseg=8):
    rings = []
    pts = [np.asarray(p, float) for p in pts]
    for j, p in enumerate(pts):
        d = C.nrm(pts[min(j + 1, len(pts) - 1)] - pts[max(j - 1, 0)])
        a = C.nrm(np.cross(d, (0, 0, 1.0)) if abs(d[2]) < 0.9 else np.cross(d, (1.0, 0, 0)))
        rings.append(ring(p, a, np.cross(d, a), r, r, nseg))
    return loft(rings, closed=True, cap0=True, cap1=True)


def glyph_strokes(rng):
    """Random stroke 'characters' on a 1x1 cell: horizontals, verticals, a hook or a slash."""
    st = []
    for _ in range(rng.integers(3, 6)):
        k = rng.random()
        if k < 0.4:
            y = rng.uniform(0.1, 0.9)
            x0, x1 = sorted(rng.uniform(0.05, 0.95, 2))
            st.append([(x0, y), (max(x1, x0 + 0.3), y)])
        elif k < 0.75:
            x = rng.uniform(0.15, 0.85)
            y0, y1 = sorted(rng.uniform(0.05, 0.95, 2))
            st.append([(x, y0), (x, max(y1, y0 + 0.35))])
        elif k < 0.9:
            x = rng.uniform(0.2, 0.8)
            st.append([(x, 0.9), (x, 0.25), (x - 0.15, 0.1)])
        else:
            st.append([(rng.uniform(0.1, 0.4), rng.uniform(0.6, 0.9)), (rng.uniform(0.6, 0.9), rng.uniform(0.1, 0.4))])
    return st


for si, (c, t_along, col, (w, h)) in enumerate(SIGNS):
    rng = rng_for(500 + si)
    out = np.array([-t_along[1], t_along[0], 0.0]) * 0  # not used
    # box: thin slab perpendicular to the wall (faces +-t_along)
    tt = np.array([t_along[0], t_along[1], 0.0])
    nn = np.array([-t_along[1], t_along[0], 0.0])
    GEO["signbox"].box(tuple(c), (abs(tt[0]) * 0.12 + abs(nn[0]) * w, abs(tt[1]) * 0.12 + abs(nn[1]) * w, h))
    GEO["metal"].beam(tuple(c + np.array([0, 0, h / 2 - 0.1]) - nn * w / 2 * np.sign(1)),
                      tuple(c + np.array([0, 0, h / 2 - 0.1]) - nn * (w / 2 + 0.6)), 0.03)
    nch = int(h / (w * 0.95))
    for face in (1, -1):
        for ch in range(nch):
            cell = w * 0.72
            z0 = c[2] + h / 2 - 0.12 - (ch + 1) * (h - 0.24) / nch
            for stroke in glyph_strokes(rng_for(900 + si * 10 + ch)):
                pts = []
                for (gx, gy) in stroke:
                    p = c + tt * (face * 0.075) + nn * ((gx - 0.5) * cell) + np.array([0, 0, z0 + gy * cell * 0.95])
                    pts.append(p)
                GLYPH_TUBES[col].append(tube_poly(pts, 0.011))
    # frame tube
    for face in (1, -1):
        corners = [c + tt * (face * 0.07) + nn * (sx * (w / 2 - 0.04)) + np.array([0, 0, sz * (h / 2 - 0.04)])
                   for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
        GLYPH_TUBES[col].append(tube_poly(corners, 0.008))
    lt = bpy.data.lights.new(f"SignLight{si}", "AREA")
    lt.shape = "RECTANGLE"
    lt.size, lt.size_y = 0.4, h
    lt.color = NEON_RGB[col]
    lt.use_shadow = si < 3
    lt.volume_factor = 1.0
    lo = link(bpy.data.objects.new(f"SignLight{si}", lt), coll=ENV)
    lo.location = tuple(c - nn * 0.0 + np.array([0, 0, 0]))
    lo.rotation_euler = (math.pi / 2, 0, math.atan2(tt[1], tt[0]) + math.pi / 2)
    LIGHTS.append((lo, "neon", 55.0))
    lt2 = lt.copy()
    lo2 = link(bpy.data.objects.new(f"SignLightB{si}", lt2), coll=ENV)
    lo2.location = lo.location
    lo2.rotation_euler = (math.pi / 2, 0, math.atan2(tt[1], tt[0]) - math.pi / 2)
    LIGHTS.append((lo2, "neon", 55.0))
for col, tubes in GLYPH_TUBES.items():
    if not tubes:
        continue
    vv, ff = [], []
    for v_, f_, _ in tubes:
        o = len(vv)
        vv += v_
        ff += [tuple(x + o for x in f) for f in f_]
    ob = mesh_obj("Neon_" + col, vv, ff, M_NEON_BAD if col == "cyan" else M_NEON[col])
    no_shadow(ob)
    ob.users_collection[0].objects.unlink(ob)
    ENV.objects.link(ob)

# "PULL ME CLOSE" neon on the west wall (outline tubes of the letters)
if os.path.exists(ARGS.font):
    fcu = bpy.data.curves.new("TitleNeon", "FONT")
    fcu.body = "PULL ME CLOSE"
    fcu.font = bpy.data.fonts.load(ARGS.font)
    fcu.size = 0.62
    fcu.align_x = "CENTER"
    fcu.fill_mode = "NONE"
    fcu.bevel_depth = 0.011
    fcu.bevel_resolution = 2
    fcu.resolution_u = 6
    tob = link(bpy.data.objects.new("TitleNeon", fcu), coll=ENV)
    tob.location = (-W + 0.32, 1.9, 3.78)
    tob.rotation_euler = (math.pi / 2, 0, math.pi / 2)
    tob.data.materials.append(M_NEON["pink"])
    no_shadow(tob)
    GEO["signbox"].box((-W + 0.22, 1.9, 3.95), (0.06, 4.7, 0.95))
    lt = bpy.data.lights.new("TitleLight", "AREA")
    lt.shape, lt.size, lt.size_y = "RECTANGLE", 0.6, 4.2
    lt.use_shadow = False
    lt.color = NEON_RGB["pink"]
    lo = link(bpy.data.objects.new("TitleLight", lt), coll=ENV)
    lo.location = (-W + 0.45, 1.9, 3.95)
    lo.rotation_euler = (0, -math.pi / 2, 0)
    LIGHTS.append((lo, "neon", 140.0))


# ---- the crest: Tzeke000's emblem printed on two big shop lanterns (east wall) ----------
def crest_image(path):
    """Load the emblem; its white background becomes paper (lit), dark lines block the light."""
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = "sRGB"
    return img


def lantern_mesh(name, r, h, mat, cap_mat, loc, ribs=18):
    prof = []
    for k in range(ribs + 1):
        t = k / ribs
        z = -h / 2 + h * t
        rr = r * (0.55 + 0.45 * math.sin(math.pi * (0.06 + 0.88 * t))) * (1 + 0.025 * math.cos(2 * math.pi * t * ribs) ** 8)
        prof.append((rr, z))
    rings = [ring((0, 0, z), (1, 0, 0), (0, 1, 0), rr, rr, 32) for rr, z in prof]
    v, f, u = loft(rings, closed=True)
    ob = mesh_obj(name, v, f, mat, loc=loc, uvs=u)
    cv, cf, cu = loft([ring((0, 0, z), (1, 0, 0), (0, 1, 0), r * 0.58, r * 0.58, 24) for z in (h / 2 - 0.02, h / 2 + 0.05)],
                      closed=True, cap1=True)
    mesh_obj(name + "_top", cv, cf, cap_mat, loc=loc)
    cv, cf, cu = loft([ring((0, 0, z), (1, 0, 0), (0, 1, 0), r * 0.58, r * 0.58, 24) for z in (-h / 2 - 0.05, -h / 2 + 0.02)],
                      closed=True, cap0=True)
    mesh_obj(name + "_bot", cv, cf, cap_mat, loc=loc)
    return ob


def lantern_paper(name, rgb, strength, crest=None):
    m, nt = new_mat(name)
    tc = nd(nt, "ShaderNodeTexCoord")
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["UV"], sep.inputs[0])
    rib = M(nt, "PINGPONG", M(nt, "MULTIPLY", sep.outputs[1], 18.0), 0.5)
    ribm = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.0, "From Max": 0.06, "To Min": 0.45, "To Max": 1.0})
    lk(nt, rib, ribm.inputs["Value"])
    col = tuple(rgb) + (1,)
    e = nd(nt, "ShaderNodeEmission", inp={"Color": col})
    g = value_node(nt, "lantern", 1.0)
    st = M(nt, "MULTIPLY", M(nt, "MULTIPLY", g.outputs[0], strength), ribm.outputs[0])
    if crest is not None:
        # map the emblem onto the front (u around 0.25 = local +Y ... we rotate the object instead)
        mp = nd(nt, "ShaderNodeMapping", inp={"Location": (-1.0, -0.26, 0.0), "Scale": (3.0, 1.5, 1.0)})
        uvc = nd(nt, "ShaderNodeCombineXYZ")
        lk(nt, sep.outputs[0], uvc.inputs[0])
        lk(nt, sep.outputs[1], uvc.inputs[1])
        lk(nt, uvc.outputs[0], mp.inputs[0])
        tex = nd(nt, "ShaderNodeTexImage", image=crest, extension="CLIP", interpolation="Cubic")
        lk(nt, mp.outputs[0], tex.inputs["Vector"])
        bw = nd(nt, "ShaderNodeRGBToBW")
        lk(nt, tex.outputs["Color"], bw.inputs[0])
        # paper colour where the emblem is white; the emblem's own colours elsewhere; black lines block
        inside = M(nt, "MULTIPLY", tex.outputs["Alpha"], M(nt, "LESS_THAN", bw.outputs[0], 0.93))
        mixc = mix_col(nt, inside, col, tex.outputs["Color"])
        lk(nt, mixc, e.inputs["Color"])
        dark = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.04, "From Max": 0.25, "To Min": 0.03, "To Max": 1.0})
        lk(nt, bw.outputs[0], dark.inputs["Value"])
        st = M(nt, "MULTIPLY", st, mix_f(nt, inside, 1.0, dark.outputs[0]))
    lk(nt, st, e.inputs["Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, e.outputs[0], o.inputs["Surface"])
    return m


CREST = crest_image(ARGS.logo)
M_LANTERN = {"red": lantern_paper("LanternRed", (1.0, 0.16, 0.06), 5.0),
             "white": lantern_paper("LanternWhite", blackbody(2400), 4.0),
             "toro": lantern_paper("ToroGlow", blackbody(2200), 6.0)}
M_CREST = lantern_paper("CrestLantern", blackbody(3000), 1.1, crest=CREST)
M_CAP = principled("LanternCap", (0.01, 0.01, 0.01), 0.3, coat=0.5)

n_light = 0
for k, (loc, col, plaza) in enumerate(LANTERNS):
    if col == "toro":
        GEO["shoji"].box(loc, (0.30, 0.30, 0.28))
    else:
        ob = lantern_mesh(f"Lantern{k}", 0.17, 0.42, M_LANTERN[col], M_CAP, loc)
        GEO["cable"].beam((loc[0], loc[1], loc[2] + 0.26), (loc[0], loc[1], 2.95), 0.008)
    if plaza or col == "toro":
        lt = bpy.data.lights.new(f"LanternLight{k}", "POINT")
        lt.color = M_LANTERN["red" if col == "red" else "white"].node_tree.nodes["Emission"].inputs["Color"].default_value[:3]
        lt.shadow_soft_size = 0.16
        lt.use_shadow = n_light < 4
        lt.volume_factor = 0.6
        lo = link(bpy.data.objects.new(f"LanternLight{k}", lt), coll=ENV)
        lo.location = loc
        LIGHTS.append((lo, "lantern", 28.0 if col != "toro" else 18.0))
        n_light += 1

# the crest lanterns: big, flanking a doorway on the east wall, emblem facing the plaza
for k, y in enumerate((-1.35, 1.35)):
    loc = (W - 0.62, y, 2.28)
    ob = lantern_mesh(f"CrestLantern{k}", 0.30, 0.82, M_CREST, M_CAP, loc, ribs=24)
    # u = 0.5 of the lantern's UV is local -X: the emblem faces the plaza with no rotation
    GEO["cable"].beam((loc[0], loc[1], loc[2] + 0.45), (loc[0], loc[1], 2.95), 0.01)
    lt = bpy.data.lights.new(f"CrestLight{k}", "POINT")
    lt.color = blackbody(2700)
    lt.use_shadow = False
    lt.shadow_soft_size = 0.3
    lt.volume_factor = 0.8
    lo = link(bpy.data.objects.new(f"CrestLight{k}", lt), coll=ENV)
    lo.location = loc
    LIGHTS.append((lo, "lantern", 60.0))
CREST_POS = np.array([W - 0.62, -1.35, 2.28])

# street lamps: two masts with LED spot heads -> light cones full of rain
LAMPS = []
for k, (x, y, rgb, mat) in enumerate(((-5.6, 4.6, (0.70, 0.86, 1.0), M_LAMP_HEAD),
                                      (5.6, -4.4, blackbody(3400), M_LAMP_HEAD2))):
    GEO["metal"].box((x, y, 2.6), (0.12, 0.12, 5.2))
    tx, ty = -x * 0.8, -y * 0.8
    arm_end = (x + (0 - x) * 0.18, y + (0 - y) * 0.18, 5.15)
    GEO["metal"].beam((x, y, 5.1), arm_end, 0.07)
    hv, hf, hu = loft([ring((0, 0, z), (1, 0, 0), (0, 1, 0), 0.16, 0.08, 12) for z in (0.0, 0.05)], closed=True,
                      cap0=True, cap1=True)
    head = mesh_obj(f"LampHead{k}", hv, hf, mat, loc=arm_end)
    head.location.z -= 0.04
    lt = bpy.data.lights.new(f"StreetLamp{k}", "SPOT")
    lt.color = rgb
    lt.spot_size = math.radians(70)
    lt.spot_blend = 0.6
    lt.shadow_soft_size = 0.12
    lt.volume_factor = 2.2
    lo = link(bpy.data.objects.new(f"StreetLamp{k}", lt), coll=ENV)
    lo.location = (arm_end[0], arm_end[1], arm_end[2] - 0.08)
    d = Vector((0 - arm_end[0] * 0.6, 0.25 - arm_end[1] * 0.6, -arm_end[2] + 0.2))
    lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    LIGHTS.append((lo, "lamp", 220.0))
    LAMPS.append(lo)

# the gate backlight: cold light in the fog behind the torii -> silhouettes + shafts
lt = bpy.data.lights.new("GateBack", "AREA")
lt.shape, lt.size, lt.size_y = "RECTANGLE", 5.0, 6.0
lt.color = (0.55, 0.72, 1.0)
lt.volume_factor = 3.0
GATE_LIGHT = link(bpy.data.objects.new("GateBack", lt), coll=ENV)
GATE_LIGHT.location = (0, TY + 6.0, 3.5)
GATE_LIGHT.rotation_euler = (-math.pi / 2, 0, 0)
LIGHTS.append((GATE_LIGHT, "gate", 850.0))
lt = bpy.data.lights.new("SouthGlow", "AREA")
lt.shape, lt.size, lt.size_y = "RECTANGLE", 4.5, 5.0
lt.color = (1.0, 0.35, 0.55)
lt.volume_factor = 2.0
lo = link(bpy.data.objects.new("SouthGlow", lt), coll=ENV)
lo.location = (0, -24.0, 3.5)
lo.rotation_euler = (math.pi / 2, 0, 0)
LIGHTS.append((lo, "gate", 700.0))
# warm spill from the shop fronts (one long low area light per plaza wall)
for k, (x, y, rot, sx) in enumerate(((-W + 0.6, 0, (0, -math.pi / 2, 0), 12.0), (W - 0.6, 0, (0, math.pi / 2, 0), 12.0))):
    lt = bpy.data.lights.new(f"ShopSpill{k}", "AREA")
    lt.shape, lt.size, lt.size_y = "RECTANGLE", 1.2, sx
    lt.color = blackbody(2500)
    lt.use_shadow = False
    lt.specular_factor = 0.0          # fill only: an invisible source must not streak the wet stone
    lt.volume_factor = 0.3
    lo = link(bpy.data.objects.new(f"ShopSpill{k}", lt), coll=ENV)
    lo.location = (x, y, 1.3)
    lo.rotation_euler = rot
    LIGHTS.append((lo, "lantern", 35.0))

# emit the accumulated street geometry
GEO_MATS = {"plaster": M_PLASTER, "wood": M_WOOD, "wood2": M_WOOD2, "tile": M_TILE, "found": M_FOUND,
            "shoji": M_SHOJI, "window": M_WINDOW, "interior": M_INTERIOR, "noren": M_NOREN, "metal": M_METAL,
            "signbox": M_SIGNBOX, "cable": M_CABLE, "stone": M_STONE}
for k, mb in GEO.items():
    if mb.v:
        ob = mb.obj("Street_" + k, GEO_MATS[k])
        ob.users_collection[0].objects.unlink(ob)
        ENV.objects.link(ob)
        if k in ("shoji", "window", "interior", "cable"):
            no_shadow(ob)
print(f"[build] street: {sum(len(m.v) for m in GEO.values())} verts, {len(LIGHTS)} lights, "
      f"{len(LANTERNS)} lanterns, {len(SIGNS)} neon signs")


# =====================================================================================
# 7. EFFECTS (Geometry Nodes, stateless: every input is a baked key)
# =====================================================================================
class GN:
    """Tiny helper around a GeometryNodeTree."""

    def __init__(self, name):
        self.t = bpy.data.node_groups.new(name, "GeometryNodeTree")
        self.t.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        self.t.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        self.gi = self.t.nodes.new("NodeGroupInput")
        self.go = self.t.nodes.new("NodeGroupOutput")
        self.socks = {}

    def inp(self, name, typ, default=None):
        s = self.t.interface.new_socket(name=name, in_out="INPUT", socket_type=typ)
        if default is not None:
            s.default_value = default
        self.socks[name] = s
        return self.gi.outputs[name]

    def n(self, typ, **kw):
        return nd(self.t, typ, **kw)

    def link(self, a, b):
        self.t.links.new(a, b)

    def m(self, op, a, b=None, c=None):
        return M(self.t, op, a, b, c)

    def v(self, op, a, b=None, scale=None):
        return V(self.t, op, a, b, scale)

    def rand(self, lo, hi, seed, vec=False, ident=None):
        r = self.n("FunctionNodeRandomValue", data_type="FLOAT_VECTOR" if vec else "FLOAT")
        want = "VECTOR" if vec else "VALUE"
        ins = [x for x in r.inputs if x.enabled and x.type == want]
        outs = [x for x in r.outputs if x.enabled and x.type == want]
        for sock, val in ((ins[0], lo), (ins[1], hi)):
            if isinstance(val, bpy.types.NodeSocket):
                self.link(val, sock)
            else:
                sock.default_value = val
        if isinstance(seed, bpy.types.NodeSocket):
            self.link(seed, r.inputs["Seed"])
        else:
            r.inputs["Seed"].default_value = int(seed)
        if ident is not None:
            self.link(ident, r.inputs["ID"])
        return outs[0]

    def sep(self, vec):
        s = self.n("ShaderNodeSeparateXYZ")
        self.link(vec, s.inputs[0])
        return s.outputs

    def comb(self, x, y, z):
        c = self.n("ShaderNodeCombineXYZ")
        for k, val in enumerate((x, y, z)):
            if isinstance(val, (int, float)):
                c.inputs[k].default_value = float(val)
            else:
                self.link(val, c.inputs[k])
        return c.outputs[0]

    def smooth(self, x, e0, e1):
        mr = self.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP",
                    inp={"From Min": e0, "From Max": e1, "To Min": 0.0, "To Max": 1.0})
        self.link(x, mr.inputs["Value"])
        return mr.outputs["Result"]

    def finish(self, geo, mat_sock_name="Material"):
        rl = self.n("GeometryNodeRealizeInstances")
        self.link(geo, rl.inputs["Geometry"])
        sm = self.n("GeometryNodeSetMaterial")
        self.link(rl.outputs[0], sm.inputs["Geometry"])
        self.link(self.inp(mat_sock_name, "NodeSocketMaterial"), sm.inputs["Material"])
        self.link(sm.outputs[0], self.go.inputs["Geometry"])
        return self.t


def rain_tree():
    """Falling streaks in a box. Time (s) is a baked input (hit-stop slow motion on drops).
    Each drop wraps from bottom to top at zero scale, so motion blur never smears a wrap."""
    g = GN("Rain")
    cnt = g.inp("Count", "NodeSocketInt", 1000)
    lo = g.inp("Min", "NodeSocketVector")
    hi = g.inp("Max", "NodeSocketVector")
    tm = g.inp("Time", "NodeSocketFloat")
    wind = g.inp("Wind", "NodeSocketVector")
    ln = g.inp("Length", "NodeSocketFloat", 0.1)
    rad = g.inp("Radius", "NodeSocketFloat", 0.0015)
    spd = g.inp("Speed", "NodeSocketFloat", 8.5)
    pts = g.n("GeometryNodePoints")
    g.link(cnt, pts.inputs["Count"])
    base = g.rand((0, 0, 0), (1, 1, 1), 3, vec=True)
    bx = g.sep(base)
    lox, hix = g.sep(lo), g.sep(hi)
    H = g.m("SUBTRACT", hix[2], lox[2])
    sp = g.m("MULTIPLY", spd, g.rand(0.85, 1.15, 7))
    ph = g.m("FRACT", g.m("SUBTRACT", bx[2], g.m("DIVIDE", g.m("MULTIPLY", tm, sp), H)))
    z = g.m("ADD", lox[2], g.m("MULTIPLY", ph, H))
    x = g.m("ADD", lox[0], g.m("MULTIPLY", bx[0], g.m("SUBTRACT", hix[0], lox[0])))
    y = g.m("ADD", lox[1], g.m("MULTIPLY", bx[1], g.m("SUBTRACT", hix[1], lox[1])))
    wx, wy = g.sep(wind)[0], g.sep(wind)[1]
    up = g.m("MULTIPLY", ph, H)
    x = g.m("ADD", x, g.m("MULTIPLY", wx, up))
    y = g.m("ADD", y, g.m("MULTIPLY", wy, up))
    g.link(g.comb(x, y, z), pts.inputs["Position"])
    fade = g.m("MULTIPLY", g.smooth(ph, 0.0, 0.03), g.smooth(g.m("SUBTRACT", 1.0, ph), 0.0, 0.03))
    cyl = g.n("GeometryNodeMeshCylinder", inp={"Vertices": 3, "Radius": 1.0, "Depth": 1.0})
    dirv = g.comb(g.m("MULTIPLY", wx, -1.0), g.m("MULTIPLY", wy, -1.0), -1.0)
    al = g.n("FunctionNodeAlignRotationToVector", axis="Z")
    g.link(dirv, al.inputs["Vector"])
    ins = g.n("GeometryNodeInstanceOnPoints")
    g.link(pts.outputs[0], ins.inputs["Points"])
    g.link(cyl.outputs["Mesh"], ins.inputs["Instance"])
    g.link(al.outputs[0], ins.inputs["Rotation"])
    r_ = g.m("MULTIPLY", rad, fade)
    g.link(g.comb(r_, r_, g.m("MULTIPLY", ln, fade)), ins.inputs["Scale"])
    return g.finish(ins.outputs[0]), g


def splash_tree(crown_obj):
    """Raindrop hits on the ground: each point re-rolls its position every cycle (ID = index +
    cycle * count), scale 0 at the cycle boundary."""
    g = GN("Splash")
    cnt = g.inp("Count", "NodeSocketInt", 1000)
    lo = g.inp("Min", "NodeSocketVector")
    hi = g.inp("Max", "NodeSocketVector")
    tm = g.inp("Time", "NodeSocketFloat")
    size = g.inp("Size", "NodeSocketFloat", 0.03)
    pts = g.n("GeometryNodePoints")
    g.link(cnt, pts.inputs["Count"])
    rate = g.rand(3.0, 5.5, 11)
    off = g.rand(0.0, 1.0, 13)
    tt = g.m("ADD", g.m("MULTIPLY", tm, rate), off)
    cyc = g.m("FLOOR", tt)
    ph = g.m("FRACT", tt)
    idx = g.n("GeometryNodeInputIndex")
    ident = g.n("FunctionNodeFloatToInt")
    g.link(g.m("ADD", idx.outputs[0], g.m("MULTIPLY", cyc, 7919.0)), ident.inputs[0])
    p = g.rand(lo, hi, 17, vec=True, ident=ident.outputs[0])
    g.link(p, pts.inputs["Position"])
    life = g.m("MULTIPLY", g.smooth(ph, 0.0, 0.08), g.smooth(g.m("SUBTRACT", 0.55, ph), 0.0, 0.25))
    oi = g.n("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    oi.inputs["Object"].default_value = crown_obj
    ins = g.n("GeometryNodeInstanceOnPoints")
    g.link(pts.outputs[0], ins.inputs["Points"])
    g.link(oi.outputs["Geometry"], ins.inputs["Instance"])
    grow = g.m("MULTIPLY", size, g.m("ADD", 0.5, g.m("MULTIPLY", ph, 1.6)))
    s_ = g.m("MULTIPLY", grow, life)
    g.link(g.comb(s_, s_, g.m("MULTIPLY", s_, g.m("SUBTRACT", 1.3, ph))), ins.inputs["Scale"])
    rz = g.n("FunctionNodeEulerToRotation") if hasattr(bpy.types, "FunctionNodeEulerToRotation") else None
    if rz is not None:
        g.link(g.comb(0.0, 0.0, g.rand(0.0, 6.28, 19)), rz.inputs[0])
        g.link(rz.outputs[0], ins.inputs["Rotation"])
    return g.finish(ins.outputs[0]), g


def burst_tree(name):
    """Stateless burst (sparks / spray / footstep splashes): Count points leave Origin along
    random directions biased by Normal (and flattened by Flat), fly ballistically, and are drawn
    as streaks along their velocity. Age < 0 or past a point's life -> scale 0."""
    g = GN(name)
    cnt = g.inp("Count", "NodeSocketInt", 60)
    org = g.inp("Origin", "NodeSocketVector")
    nrm_ = g.inp("Normal", "NodeSocketVector")
    age = g.inp("Age", "NodeSocketFloat", 10.0)
    seed = g.inp("Seed", "NodeSocketInt", 0)
    stren = g.inp("Strength", "NodeSocketFloat", 1.0)
    smin = g.inp("SpeedMin", "NodeSocketFloat", 2.0)
    smax = g.inp("SpeedMax", "NodeSocketFloat", 8.0)
    bias = g.inp("Bias", "NodeSocketFloat", 0.0)
    flat = g.inp("Flat", "NodeSocketFloat", 1.0)
    upb = g.inp("Up", "NodeSocketFloat", 0.3)
    grav = g.inp("Gravity", "NodeSocketFloat", 9.8)
    life = g.inp("Life", "NodeSocketFloat", 0.5)
    size = g.inp("Size", "NodeSocketFloat", 0.004)
    stk = g.inp("Streak", "NodeSocketFloat", 0.02)
    pts = g.n("GeometryNodePoints")
    g.link(cnt, pts.inputs["Count"])
    d = g.rand((-1, -1, -1), (1, 1, 1), seed, vec=True)
    dx = g.sep(d)
    d = g.comb(dx[0], dx[1], g.m("MULTIPLY", dx[2], flat))
    d = g.v("ADD", g.v("NORMALIZE", d), g.v("SCALE", nrm_, scale=g.m("MULTIPLY", bias, g.m("SUBTRACT", g.m("MULTIPLY", g.m("GREATER_THAN", g.rand(0.0, 1.0, g.m("ADD", seed, 5)), 0.5), 2.0), 1.0))))
    d = g.v("NORMALIZE", g.v("ADD", g.v("NORMALIZE", d), g.comb(0.0, 0.0, upb)))
    spd = g.m("MULTIPLY", g.rand(smin, smax, g.m("ADD", seed, 9)), g.m("SQRT", stren))
    lf = g.m("MULTIPLY", life, g.rand(0.45, 1.0, g.m("ADD", seed, 13)))
    a = g.m("MAXIMUM", age, 0.0)
    vel0 = g.v("SCALE", d, scale=spd)
    gvec = g.comb(0.0, 0.0, g.m("MULTIPLY", grav, -1.0))
    pos = g.v("ADD", g.v("ADD", org, g.v("SCALE", vel0, scale=a)),
              g.v("SCALE", gvec, scale=g.m("MULTIPLY", 0.5, g.m("MULTIPLY", a, a))))
    ps = g.sep(pos)
    pos = g.comb(ps[0], ps[1], g.m("MAXIMUM", ps[2], 0.005))
    vel = g.v("ADD", vel0, g.v("SCALE", gvec, scale=a))
    g.link(pos, pts.inputs["Position"])
    alive = g.m("MULTIPLY", g.m("GREATER_THAN", age, -0.001), g.m("LESS_THAN", age, lf))
    fade = g.m("MULTIPLY", alive, g.m("POWER", g.m("MAXIMUM", g.m("SUBTRACT", 1.0, g.m("DIVIDE", a, lf)), 0.0), 0.7))
    cyl = g.n("GeometryNodeMeshCylinder", inp={"Vertices": 4, "Radius": 1.0, "Depth": 1.0})
    al = g.n("FunctionNodeAlignRotationToVector", axis="Z")
    g.link(vel, al.inputs["Vector"])
    ins = g.n("GeometryNodeInstanceOnPoints")
    g.link(pts.outputs[0], ins.inputs["Points"])
    g.link(cyl.outputs["Mesh"], ins.inputs["Instance"])
    g.link(al.outputs[0], ins.inputs["Rotation"])
    r_ = g.m("MULTIPLY", size, fade)
    lz = g.m("MULTIPLY", g.m("MAXIMUM", g.m("MULTIPLY", g.v("LENGTH", vel), stk), size), fade)
    g.link(g.comb(r_, r_, lz), ins.inputs["Scale"])
    return g.finish(ins.outputs[0]), g


def gn_object(name, tree, g, values, mat, coll):
    ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    coll.objects.link(ob)
    md = ob.modifiers.new("GN", "NODES")
    md.node_group = tree
    for k, v in values.items():
        getattr(md.properties.inputs, g.socks[k].identifier).value = v
    getattr(md.properties.inputs, g.socks["Material"].identifier).value = mat
    return ob, md


def gn_path(md, g, name):
    return getattr(md.properties.inputs, g.socks[name].identifier).path_from_id("value")


FX = collection("FX")


def water_material(name, emit=0.25, alpha=0.4):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (0.05, 0.055, 0.06, 1)
    b.inputs["Roughness"].default_value = 0.08
    b.inputs["Specular IOR Level"].default_value = 0.8
    b.inputs["Emission Color"].default_value = (0.75, 0.82, 0.95, 1)
    g = value_node(nt, "rain_glow", 1.0)
    lk(nt, M(nt, "MULTIPLY", g.outputs[0], emit), b.inputs["Emission Strength"])
    b.inputs["Alpha"].default_value = alpha
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


M_RAIN = water_material("Rain", 0.03, 0.45)
M_SPLASH = water_material("SplashWater", 0.03, 0.5)
for _m in (M_RAIN, M_SPLASH):
    _m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.5, 0.55, 0.6, 1)
M_SPARK = emissive("Spark", blackbody(2400), 60.0)
M_SPARK_HOT = emissive("SparkHot", blackbody(4200), 90.0)

RAIN_T, RAIN_G = rain_tree()
RAIN_OBJS = []
dens = ARGS.rain
for nm, lo, hi, cnt, ln_, r_ in (
        ("RainPlaza", (-5.5, -5.5, 0.0), (5.5, 5.5, 7.5), 42000, 0.10, 0.0011),
        ("RainNorth", (-1.65, 5.5, 0.0), (1.65, 22.0, 7.5), 13000, 0.10, 0.0013),
        ("RainSouth", (-1.65, -16.0, 0.0), (1.65, -5.5, 7.5), 8000, 0.10, 0.0013)):
    ob, md = gn_object(nm, RAIN_T, RAIN_G, {"Count": int(cnt * dens), "Min": lo, "Max": hi, "Length": ln_,
                                            "Radius": r_, "Wind": (0.035, 0.05, 0.0), "Speed": 8.6}, M_RAIN, FX)
    no_shadow(ob)
    RAIN_OBJS.append((ob, md))
# eave drip curtains (plaza edges): streams of beads along each eave edge
for k, (lo, hi) in enumerate((((-W + 0.95, -5.5, 0.0), (-W + 0.98, 5.5, 2.95)),
                              ((W - 0.98, -5.5, 0.0), (W - 0.95, 5.5, 2.95)),
                              ((-5.5, W - 0.98, 0.0), (-2.7, W - 0.95, 2.95)),
                              ((2.7, W - 0.98, 0.0), (5.5, W - 0.95, 2.95)),
                              ((-5.5, -W + 0.95, 0.0), (-2.7, -W + 0.98, 2.95)),
                              ((2.7, -W + 0.95, 0.0), (5.5, -W + 0.98, 2.95)))):
    L_ = max(hi[0] - lo[0], hi[1] - lo[1])
    ob, md = gn_object(f"EaveDrip{k}", RAIN_T, RAIN_G, {"Count": int(260 * L_ * dens), "Min": lo, "Max": hi,
                                                        "Length": 0.16, "Radius": 0.0024, "Wind": (0, 0, 0),
                                                        "Speed": 6.5}, M_RAIN, FX)
    no_shadow(ob)
    RAIN_OBJS.append((ob, md))
# the crown used by the ground splashes
cv, cf = [], []
for k in range(7):
    a = 2 * math.pi * k / 7
    base = np.array([math.cos(a) * 0.5, math.sin(a) * 0.5, 0.0])
    tip = np.array([math.cos(a) * 0.9, math.sin(a) * 0.9, 1.0])
    v_, f_, _ = tube_poly([base, tip * np.array([1, 1, 0.8])], 0.05, 4)
    o = len(cv)
    cv += v_
    cf += [tuple(x + o for x in f) for f in f_]
CROWN = mesh_obj("SplashCrown", cv, cf, M_SPLASH)
CROWN.hide_render = True
CROWN.hide_viewport = True
SPL_T, SPL_G = splash_tree(CROWN)
SPLASH_OBJS = []
for nm, lo, hi, cnt in (("SplashPlaza", (-5.4, -5.4, 0.0), (5.4, 5.4, 0.0), 2600),
                        ("SplashNorth", (-1.6, 5.4, 0.0), (1.6, 16.0, 0.0), 900)):
    ob, md = gn_object(nm, SPL_T, SPL_G, {"Count": int(cnt * dens), "Min": lo, "Max": hi, "Size": 0.011},
                       M_SPLASH, FX)
    no_shadow(ob)
    SPLASH_OBJS.append((ob, md))

# bursts: 4 spark slots (clashes), 4 footstep-splash slots, 1 shockwave spray
BURST_T, BURST_G = burst_tree("Burst")
SPARK_SLOTS, STEP_SLOTS = [], []
for k in range(4):
    ob, md = gn_object(f"Sparks{k}", BURST_T, BURST_G,
                       {"Count": 90, "SpeedMin": 2.5, "SpeedMax": 9.0, "Bias": 0.9, "Flat": 1.0, "Up": 0.25,
                        "Gravity": 9.8, "Life": 0.55, "Size": 0.0035, "Streak": 0.016}, M_SPARK, FX)
    no_shadow(ob)
    lt = bpy.data.lights.new(f"SparkLight{k}", "POINT")
    lt.color = blackbody(2600)
    lt.shadow_soft_size = 0.05
    lt.use_shadow = False
    lt.volume_factor = 1.5
    lo_ = link(bpy.data.objects.new(f"SparkLight{k}", lt), coll=FX)
    SPARK_SLOTS.append((ob, md, lo_))
for k in range(4):
    ob, md = gn_object(f"StepSplash{k}", BURST_T, BURST_G,
                       {"Count": 46, "SpeedMin": 0.6, "SpeedMax": 1.9, "Bias": 0.0, "Flat": 0.35, "Up": 1.1,
                        "Gravity": 9.8, "Life": 0.5, "Size": 0.0045, "Streak": 0.02, "Normal": (0, 0, 1)},
                       M_SPLASH, FX)
    no_shadow(ob)
    STEP_SLOTS.append((ob, md))
SHOCK, SHOCK_MD = gn_object("ShockSpray", BURST_T, BURST_G,
                            {"Count": 900, "SpeedMin": 3.5, "SpeedMax": 7.5, "Bias": 0.0, "Flat": 0.05, "Up": 0.22,
                             "Gravity": 9.8, "Life": 0.7, "Size": 0.0055, "Streak": 0.018, "Normal": (0, 0, 1)},
                            M_SPLASH, FX)
no_shadow(SHOCK)


# lightning bolts (jagged tubes) in three parts of the sky
def bolt_mesh(name, top, bottom, seed):
    rng = np.random.default_rng(seed)

    def walk(a, b, n, amp):
        pts = [np.array(a, float)]
        for k in range(1, n):
            p = np.array(a) + (np.array(b) - np.array(a)) * k / n
            p[:2] += rng.normal(0, amp, 2)
            pts.append(p)
        pts.append(np.array(b, float))
        return pts
    main = walk(top, bottom, 26, 1.6)
    tubes = [tube_poly(main, 0.10, 6)]
    for _ in range(4):
        j = rng.integers(4, 20)
        a = main[j]
        b = a + np.array([rng.normal(0, 6), rng.normal(0, 6), -rng.uniform(6, 14)])
        tubes.append(tube_poly(walk(a, b, 9, 0.9), 0.05, 5))
    vv, ff = [], []
    for v_, f_, _ in tubes:
        o = len(vv)
        vv += v_
        ff += [tuple(x + o for x in f) for f in f_]
    ob = mesh_obj(name, vv, ff, M_BOLT)
    no_shadow(ob)
    ob.users_collection[0].objects.unlink(ob)
    FX.objects.link(ob)
    return ob


M_BOLT = emissive("Bolt", (0.75, 0.82, 1.0), 160.0)
BOLTS = [bolt_mesh("BoltN", (6, 95, 70), (-3, 85, 6), 1), bolt_mesh("BoltS", (-8, -70, 60), (4, -64, 5), 2),
         bolt_mesh("BoltE", (60, 12, 55), (50, -6, 4), 3)]
sun = bpy.data.lights.new("LightningSun", "SUN")
sun.color = (0.75, 0.82, 1.0)
sun.angle = math.radians(3)
sun.volume_factor = 0.12              # a flash on surfaces, not a white veil through 70 m of haze
SUN = link(bpy.data.objects.new("LightningSun", sun), coll=FX)
SUN.rotation_euler = (math.radians(35), 0, math.radians(160))
print("[build] effects done")


# =====================================================================================
# 8. WORLD: storm sky (lit by the lightning) + rain haze
# =====================================================================================
world = bpy.data.worlds.new("Storm")
SC.world = world
wn = world.node_tree
wn.nodes.clear()
tc = nd(wn, "ShaderNodeTexCoord")
sepw = nd(wn, "ShaderNodeSeparateXYZ")
lk(wn, tc.outputs["Generated"], sepw.inputs[0])
cl = nd(wn, "ShaderNodeTexNoise", inp={"Scale": 3.0, "Detail": 8.0, "Roughness": 0.6})
lk(wn, tc.outputs["Generated"], cl.inputs["Vector"])
sky = mix_col(wn, cl.outputs["Fac"], (0.0025, 0.003, 0.006), (0.010, 0.010, 0.018))
skyglow = mix_col(wn, M(wn, "POWER", M(wn, "SUBTRACT", 1.0, M(wn, "ABSOLUTE", sepw.outputs[2])), 6.0),
                  sky, (0.035, 0.012, 0.03), "ADD")
bg = nd(wn, "ShaderNodeBackground")
lk(wn, skyglow, bg.inputs["Color"])
wflash = value_node(wn, "flash", 0.0)
lk(wn, M(wn, "ADD", 1.0, M(wn, "MULTIPLY", M(wn, "MULTIPLY", wflash.outputs[0], cl.outputs["Fac"]), 14.0)),
   bg.inputs["Strength"])
vol = nd(wn, "ShaderNodeVolumeScatter", inp={"Color": (0.78, 0.84, 1.0, 1), "Density": 0.0065, "Anisotropy": 0.6})
wo = nd(wn, "ShaderNodeOutputWorld")
lk(wn, bg.outputs[0], wo.inputs["Surface"])
lk(wn, vol.outputs[0], wo.inputs["Volume"])

# =====================================================================================
# 9. CAMERA
# =====================================================================================
CAM_D = bpy.data.cameras.new("Cam")
CAM_D.sensor_width = 36.0
CAM_D.clip_start = 0.05
CAM_D.clip_end = 400.0
CAM_D.dof.use_dof = True
CAM_D.dof.aperture_blades = 7
CAM = link(bpy.data.objects.new("Cam", CAM_D))
CAM.rotation_mode = "QUATERNION"
SC.camera = CAM
fwd = C.nrm(DU.tgt - DU.cam)
right = C.nrm(np.cross(fwd, np.array([0, 0, 1.0])))
upv = np.cross(right, fwd)
Rcam = np.stack([right, upv, -fwd], -1)
qcam = C.qcontinuous(C.mat2quat(Rcam))
bake_vec(CAM, "location", DU.cam, cuts=True)
for k in range(4):
    bake(CAM, "rotation_quaternion", qcam[:, k], index=k, cuts=True)
bake(CAM_D, "lens", DU.lens, cuts=True)
bake(CAM_D, "dof.focus_distance", DU.focus, cuts=True)
bake(CAM_D, "dof.aperture_fstop", DU.fstop, cuts=True)

# =====================================================================================
# 10. MUSIC BAKES: rain, splashes, sparks, shockwave, steps, lights, neon, lightning
# =====================================================================================
for ob, md in RAIN_OBJS + SPLASH_OBJS:
    bake(ob, gn_path(md, RAIN_G if md.node_group == RAIN_T else SPL_G, "Time"), DU.rain_time)
gt = M_GROUND.node_tree
bake(gt, value_path("rain_t"), DU.rain_time)
for m in (M_RAIN, M_SPLASH):
    bake(m.node_tree, value_path("rain_glow"), 1.0 + 5.0 * DU.flash + 0.4 * DU.neon / 2.0)

# sparks: round-robin slots
ev = DU.sparks
slot_age = [np.full(N, 50.0) for _ in range(4)]
slot_org = [np.zeros((N, 3)) for _ in range(4)]
slot_nrm = [np.tile((0, 0, 1.0), (N, 1)) for _ in range(4)]
slot_seed = [np.zeros(N) for _ in range(4)]
slot_str = [np.zeros(N) for _ in range(4)]
# A slot switches to its new event ONE FRAME EARLY (origin etc. at f-1, age -1 frame = dead): the
# motion-blur subframes then never see the old origin with live sparks (that drew long dotted lines).
for k, (f, c, nn, st, kind, gap) in enumerate(ev):
    sl = k % 4
    seg = np.arange(max(f - 1, 0), N)
    slot_age[sl][seg] = (seg - f) / FPS
    slot_org[sl][seg] = c
    slot_nrm[sl][seg] = nn
    slot_seed[sl][seg] = 101 + 37 * k
    slot_str[sl][seg] = st
for sl, (ob, md, lo_) in enumerate(SPARK_SLOTS):
    bake(ob, gn_path(md, BURST_G, "Age"), slot_age[sl])
    for j in range(3):
        bake(ob, gn_path(md, BURST_G, "Origin"), slot_org[sl][:, j], index=j, const=True)
        bake(ob, gn_path(md, BURST_G, "Normal"), slot_nrm[sl][:, j], index=j, const=True)
    bake(ob, gn_path(md, BURST_G, "Seed"), slot_seed[sl], const=True)
    bake(ob, gn_path(md, BURST_G, "Strength"), slot_str[sl], const=True)
    bake_vec(lo_, "location", slot_org[sl], const=True)
    bake(lo_.data, "energy", 14.0 * slot_str[sl] * np.exp(-np.maximum(slot_age[sl], 0) / 0.07) * (slot_age[sl] >= 0))

# shockwave on every drop entry (and the pass-through): ground ring + radial spray
sh_age = np.full(N, 50.0)
sh_org = np.zeros((N, 3))
for f, c, nn, st, kind, gap in ev:
    if kind in ("entry", "pass"):
        seg = np.arange(max(f - 1, 0), N)
        sh_age[seg] = (seg - f) / FPS
        sh_org[seg] = (c[0], c[1], 0.02)
bake(SHOCK, gn_path(SHOCK_MD, BURST_G, "Age"), sh_age)
for j in range(3):
    bake(SHOCK, gn_path(SHOCK_MD, BURST_G, "Origin"), sh_org[:, j], index=j, const=True)
bake(gt, value_path("shock_x"), sh_org[:, 0], const=True)
bake(gt, value_path("shock_y"), sh_org[:, 1], const=True)
bake(gt, value_path("shock_r"), 0.1 + 6.0 * np.maximum(sh_age, 0))
bake(gt, value_path("shock_s"), 1.6 * np.exp(-np.maximum(sh_age, 0) / 0.45) * (sh_age >= 0))

# footstep splashes + ripple rings (the strong landings: stomps, lunges, the run)
strong = [l for l in DU.landings if l[2] > 0.5]
st_age = [np.full(N, 50.0) for _ in range(4)]
st_org = [np.zeros((N, 3)) for _ in range(4)]
st_str = [np.zeros(N) for _ in range(4)]
st_seed = [np.zeros(N) for _ in range(4)]
for k, (f, p, st, who) in enumerate(strong):
    sl = k % 4
    seg = np.arange(max(f - 1, 0), N)
    st_age[sl][seg] = (seg - f) / FPS
    st_org[sl][seg] = (p[0], p[1], 0.01)
    st_str[sl][seg] = st
    st_seed[sl][seg] = 7 + 13 * k
for sl, (ob, md) in enumerate(STEP_SLOTS):
    bake(ob, gn_path(md, BURST_G, "Age"), st_age[sl])
    for j in range(3):
        bake(ob, gn_path(md, BURST_G, "Origin"), st_org[sl][:, j], index=j, const=True)
    bake(ob, gn_path(md, BURST_G, "Strength"), st_str[sl], const=True)
    bake(ob, gn_path(md, BURST_G, "Seed"), st_seed[sl], const=True)
    bake(gt, value_path(f"step{sl}_x"), st_org[sl][:, 0], const=True)
    bake(gt, value_path(f"step{sl}_y"), st_org[sl][:, 1], const=True)
    bake(gt, value_path(f"step{sl}_r"), 0.04 + 0.7 * np.maximum(st_age[sl], 0))
    bake(gt, value_path(f"step{sl}_s"), 0.9 * st_str[sl] * np.exp(-np.maximum(st_age[sl], 0) / 0.5)
         * (st_age[sl] >= 0))
print(f"[build] {len(ev)} spark bursts, {len(strong)} footstep splashes")

# lights + emissive gains
for m in NEON_MATS + [M_NEON_BAD]:
    nt_ = m.node_tree
    if "neon" in nt_.nodes:
        bake(nt_, value_path("neon"), DU.neon)
    if "neon_bad" in nt_.nodes:
        bake(nt_, value_path("neon_bad"), DU.neon_bad)
for m in [M_SHOJI, M_WINDOW, M_INTERIOR, M_CREST] + list(M_LANTERN.values()):
    bake(m.node_tree, value_path("lantern"), DU.lantern)
rng_l = np.random.default_rng(3)
for lo_, kind, base in LIGHTS:
    if kind == "neon":
        flick = 1.0 - 0.12 * (rng_l.random(N) < 0.03)
        bake(lo_.data, "energy", base * DU.neon * flick)
    elif kind == "lantern":
        bake(lo_.data, "energy", base * DU.lantern)
    elif kind == "lamp":
        bake(lo_.data, "energy", base * (1 - DU.blackout) * (0.92 + 0.08 * np.clip(DU.neon, 0, 2) / 2))
    elif kind == "gate":
        bake(lo_.data, "energy", base * (0.85 + 0.15 * np.clip(DU.neon, 0, 2) / 2) * (1 - 0.9 * DU.blackout)
             + base * 0.5 * DU.flash)
bake(SUN.data, "energy", 1.1 * DU.flash)
bake(wn, value_path("flash"), DU.flash)
vis = np.zeros((3, N), bool)
for f, a in DU.strikes:
    b = int(C.hash01(f, 99) * 3)
    for d in range(0, 7):
        if f + d < N and DU.flash[f + d] > 0.3:
            vis[b, f + d] = True
for b, ob in enumerate(BOLTS):
    bake(ob, "hide_render", (~vis[b]).astype(np.float64), const=True)

# =====================================================================================
# 11. RENDER SETTINGS + COMPOSITOR
# =====================================================================================
SC.frame_start, SC.frame_end = 1, N
SC.render.fps, SC.render.fps_base = FPS, 1.0
SC.render.resolution_x, SC.render.resolution_y = 1920, 1080
SC.render.resolution_percentage = ARGS.res_pct
SC.view_settings.view_transform = "AgX"
SC.view_settings.look = "AgX - Medium High Contrast"
SC.view_settings.exposure = 0.0
SC.render.use_motion_blur = True
SC.render.motion_blur_shutter = 0.5
SC.render.motion_blur_position = "CENTER"
SC.render.image_settings.file_format = "PNG"
SC.render.image_settings.color_mode = "RGB"
SC.render.image_settings.color_depth = "8"
SC.render.engine = "BLENDER_EEVEE"
ee = SC.eevee
ee.taa_render_samples = ARGS.samples
ee.use_raytracing = True
ee.ray_tracing_method = "SCREEN"
ee.ray_tracing_options.resolution_scale = "2"
ee.ray_tracing_options.trace_max_roughness = 0.45
ee.use_fast_gi = True
ee.volumetric_start = 0.3
ee.volumetric_end = 70.0
ee.volumetric_tile_size = "4"
ee.volumetric_samples = 48
ee.volumetric_sample_distribution = 0.8
ee.use_volumetric_shadows = True
ee.volumetric_shadow_samples = 8
ee.volumetric_ray_depth = 4
ee.motion_blur_steps = 1
ee.use_shadows = True
ee.shadow_pool_size = "1024"
ee.shadow_ray_count = 1
ee.shadow_step_count = 6
ee.clamp_volume_indirect = 10.0
ee.clamp_surface_indirect = 10.0

cg = bpy.data.node_groups.new("Post", "CompositorNodeTree")
cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
rl = cg.nodes.new("CompositorNodeRLayers")
gl = cg.nodes.new("CompositorNodeGlare")
gl.inputs["Type"].default_value = "Bloom"
gl.inputs["Quality"].default_value = "Medium"
gl.inputs["Threshold"].default_value = 0.9
gl.inputs["Strength"].default_value = 0.45
gl.inputs["Size"].default_value = 0.7
gs = cg.nodes.new("CompositorNodeGlare")
gs.inputs["Type"].default_value = "Streaks"
gs.inputs["Quality"].default_value = "Medium"
gs.inputs["Threshold"].default_value = 60.0
gs.inputs["Strength"].default_value = 0.18
gs.inputs["Streaks"].default_value = 4
gs.inputs["Streaks Angle"].default_value = math.radians(12)
gs.inputs["Fade"].default_value = 0.88
ld_ = cg.nodes.new("CompositorNodeLensdist")
ld_.inputs["Distortion"].default_value = -0.008
ld_.inputs["Dispersion"].default_value = 0.014
ld_.inputs["Fit"].default_value = True
vg = cg.nodes.new("CompositorNodeEllipseMask")
vg.inputs["Size"].default_value = (1.2, 1.25)
vb = cg.nodes.new("CompositorNodeBlur") if hasattr(bpy.types, "CompositorNodeBlur") else None
mul = cg.nodes.new("ShaderNodeMix")
mul.data_type = "RGBA"
mul.blend_type = "MULTIPLY"
go = cg.nodes.new("NodeGroupOutput")
cg.links.new(rl.outputs["Image"], gs.inputs["Image"])
cg.links.new(gs.outputs["Image"], gl.inputs["Image"])
cg.links.new(gl.outputs["Image"], ld_.inputs["Image"])
mr = cg.nodes.new("ShaderNodeMapRange")
mr.inputs["To Min"].default_value = 0.62
mr.inputs["To Max"].default_value = 1.0
if vb is not None:
    try:
        vb.inputs["Size"].default_value = (300, 300)
    except Exception:
        pass
    cg.links.new(vg.outputs["Mask"], vb.inputs["Image"])
    cg.links.new(vb.outputs["Image"], mr.inputs["Value"])
else:
    cg.links.new(vg.outputs["Mask"], mr.inputs["Value"])
mul.inputs[0].default_value = 1.0
cg.links.new(ld_.outputs["Image"], mul.inputs[6])
cg.links.new(mr.outputs["Result"], mul.inputs[7])
cg.links.new(mul.outputs[2], go.inputs["Image"])
SC.compositing_node_group = cg
SC.render.compositor_device = "GPU"

if os.path.exists(ARGS.audio):
    se = SC.sequence_editor_create()
    se.strips.new_sound("Music", ARGS.audio, channel=1, frame_start=1)
SC.render.use_sequencer = False

# bookkeeping for render.sh / README / tests
SC["shots"] = ";".join(f"{s}:{n}" for s, n in DU.shots)
SC["entries"] = list(DU.E)
SC["clashes"] = ";".join(f"{f}:{k}:{a}" for f, k, a in DU.clashes)


def verify_rig(frames):
    """Self-test: evaluated bone positions vs the numpy plan (catches any convention slip)."""
    worst = 0.0
    for f in frames:
        SC.frame_set(int(f) + 1)
        for i, (rig, sword, parts) in enumerate(FIGHTERS):
            mw = rig.matrix_world
            for side, si in (("R", 0), ("L", 1)):
                got = np.array(mw @ rig.pose.bones[f"hand.{side}"].head)
                gotf = np.array(mw @ rig.pose.bones[f"foot.{side}"].head)
                e1 = np.linalg.norm(got - DU.wrist[i][si][f]) if DU.grip_w[i][si][f] > 0.99 else 0.0
                e1 -= RIGS[i].ik_err[side][f]
                e2 = np.linalg.norm(gotf - DU.ankle[i][si][f]) - RIGS[i].ik_err["leg" + side][f]
                worst = max(worst, e1, e2)
    print(f"[verify] worst evaluated-vs-plan error over {len(frames)} frames: {worst * 100:.2f} cm")


verify_rig([100, 700, 1319, 1500, 2500, 4900, 5437, 5600])
SC.frame_set(1)
os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(ARGS.out), compress=True)
print(f"[samurai] saved {ARGS.out}  frames 1..{N}  objects {len(bpy.data.objects)}")
