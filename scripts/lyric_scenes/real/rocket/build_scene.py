"""ROCKET LAUNCH — a photoreal, music-synced Blender scene for "Hit a Bump" (Tzeke000).

    ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/real/rocket/build_scene.py -- \
        [--timeline state/tzeke_songs/hit_a_bump/timeline_30fps.npz] \
        [--out scratch/scenes/rocket/rocket.blend] [--engine EEVEE|CYCLES] [--res-pct 100]

Builds the .blend DETERMINISTICALLY from the timeline npz (export_timeline.py): every
music reaction is BAKED as keyframes (camera, lights, plumes, LED clock, smoke, exposure),
so rendering needs no handlers and no Python drivers. Frame f (1-based) = timeline index f-1.

Story (drop entries from the npz, so the scene re-times itself to any song):
  calm intro    night pad, floodlit rocket venting vapour, searchlights, LED countdown clock
                that hits T-0 EXACTLY on entry[0]; build (T_pre): vapour thickens, beacons
                strobe faster, camera creeps in; T_black half-beat: pad lights cut.
  drop 1        IGNITION (green TEA-TEB flash) + liftoff, billowing exhaust, tower clear,
                onboard tail-cam.
  calm          climbing out of a cloud deck lit from inside by the plume.
  drop 2        MAX-Q: vapour cone + shock-ring "sonic boom" on the entry.
  drop 3        BOOSTER SEPARATION on the entry; plume balloons in thin air.
  drop 4        FAIRING SEPARATION on the entry; payload revealed.
  calm          MECO + stage separation, coast in silence (pre-drop blackout).
  BIG drop      upper-stage ignition, sunrise over Earth's limb, a push on every bar downbeat.
  outro         engine cutoff, drift over the Earth.
Music: kick = engine pulse + camera jolt, snare = flash / ice-debris burst, sub-bass = plume
and cloud glow, hi-hat = ice-crystal sparkle.
"""
import argparse
import math
import os
import sys

import bmesh  # noqa: F401  (kept for parity with the other scene builders)
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SONG = os.path.join(REPO, "state", "tzeke_songs", "hit_a_bump")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=os.path.join(SONG, "timeline_30fps.npz"))
    ap.add_argument("--audio", default=os.path.join(SONG, "hit_a_bump_half.wav"))
    ap.add_argument("--logo", default=os.path.join(REPO, "state", "tzeke_songs", "_assets",
                                                   "tzeke000_symbol.png"))
    ap.add_argument("--font", default=os.path.join(REPO, "assets", "fonts", "BebasNeue-Regular.ttf"))
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "scenes", "rocket", "rocket.blend"))
    ap.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    ap.add_argument("--samples", type=int, default=0, help="0 = tuned default per engine")
    ap.add_argument("--res-pct", type=int, default=100)
    return ap.parse_args(argv)


ARGS = parse_args()

# =====================================================================================
# 1. MUSIC → per-frame control arrays (pure numpy, stateless over the whole song)
# =====================================================================================
TL = np.load(ARGS.timeline)
FPS = int(TL["fps"])
N = len(TL["rms"])
BPM = float(TL["bpm"])
BAR = 4 * 60.0 * FPS / BPM                                   # frames per bar (~48.6)
IDX = np.arange(N)


def gsmooth(x, sigma):
    if sigma <= 0:
        return x.astype(np.float64)
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x.astype(np.float64), r, mode="edge"), k, mode="valid")


def age_env(age, tau):
    a = np.asarray(age, np.float64)
    return np.where(a < 1e5, np.exp(-a / tau), 0.0)


ENT = sorted(int(e) for e in TL["entries"])
assert len(ENT) >= 5, f"need 5 drop entries, got {ENT}"
IGN, MAXQ, BSEP, FAIRSEP, SES = ENT[:5]
SEGS = [tuple(int(v) for v in s) for s in TL["segs"]]
MECO = max(s0 for s0, s1, h in SEGS if h == 0 and s1 <= SES)          # calm before the big drop
SECO = min(s0 for s0, s1, h in SEGS if h == 0 and s0 > SES)           # outro start

KICK = age_env(TL["T_kage"], 3.0)
SNARE = age_env(TL["T_sage"], 4.0)
HOT = TL["T_hot"].astype(np.float64)
BIG = TL["T_big"].astype(np.float64)
PRE = TL["T_pre"].astype(np.float64)
BLACK = TL["T_black"].astype(np.float64)
ONE = age_env(TL["T_one_age"], 9.0)
STROBE = (TL["T_strobe_age"] == 0).astype(np.float64)
BASS = gsmooth(TL["bass"], 1.5)
_hh = np.clip(TL["high"] - gsmooth(TL["high"], 12), 0, None)
HAT = np.clip(_hh / (np.percentile(_hh, 98) + 1e-6), 0, 1)
BEAT_PH = TL["T_ph"].astype(np.float64)
KIDX = TL["T_kidx"].astype(np.int64)
SIDX = TL["T_sidx"].astype(np.int64)
SAGE = np.minimum(TL["T_sage"].astype(np.float64), 1e5)
DOWNBEATS = np.unique(TL["T_bar_start"])


def snap(f):
    """Nearest bar downbeat to frame index f (cuts land ON the downbeat)."""
    return int(DOWNBEATS[np.argmin(np.abs(DOWNBEATS - f))])


def hash01(*k):
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    return (h ^ (h >> 15)) / 4294967295.0


def ramp(i, a, b):
    return np.clip((np.asarray(i, np.float64) - a) / max(b - a, 1e-9), 0.0, 1.0)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3.0):
    return 1 - (1 - np.clip(x, 0, 1)) ** p


# --- engine states --------------------------------------------------------------------
LIFT = IGN + 9                                                # liftoff ~ a beat after ignition
on_core = ramp(IDX, IGN - 1, IGN + 3) * (1 - ramp(IDX, MECO, MECO + 4))
on_boost = ramp(IDX, IGN - 1, IGN + 3) * (1 - ramp(IDX, BSEP, BSEP + 2))
on_s2 = ramp(IDX, SES - 1, SES + 2) * (1 - ramp(IDX, SECO, SECO + 8))
IGN_FLASH = np.where((IDX >= IGN) & (IDX < IGN + 4), 1.0 - (IDX - IGN) / 4.0, 0.0)  # TEA-TEB green
PULSE = 1.0 + HOT * (0.30 * KICK + 0.10 * BASS + 0.12 * SNARE) + BIG * 0.35 * ONE
T_SEC = np.maximum(IDX - LIFT, 0) / FPS
ALT = 0.5 * 4.6 * T_SEC ** 2                                  # metres above the pad (pad set)

# --- sets (which world is on screen) ---------------------------------------------------
PAD_END = [s1 for s0, s1, h in SEGS if s0 == IGN][0]          # end of drop 1
SET = np.zeros(N, np.int32)                                    # 0 pad, 1 clouds, 2 high, 3 space
SET[PAD_END:] = 1
SET[BSEP:] = 2
SET[SES:] = 3


def in_set(k):
    return (SET == k).astype(np.float64)


# =====================================================================================
# 2. SHOT LIST (cuts on drop entries and bar downbeats)
# =====================================================================================
def seg_after(f):
    return [s1 for s0, s1, h in SEGS if s0 == f][0]


A_CUTS = [0, snap(IGN * 0.30), snap(IGN * 0.56), snap(IGN * 0.80)]
B_END = PAD_END
B_CUTS = [IGN, snap(IGN + 3.0 * BAR), snap(IGN + 5.5 * BAR), snap(IGN + 7.0 * BAR)]
C_CUTS = [PAD_END, snap(PAD_END + (MAXQ - PAD_END) * 0.52)]
D_END = seg_after(MAXQ)
D_CUTS = [MAXQ, snap(MAXQ + 2.5 * BAR), snap(MAXQ + 4.5 * BAR)]
F_END = seg_after(BSEP)
F_CUTS = [BSEP, snap(BSEP + 3 * BAR), snap(BSEP + 5.5 * BAR)]
H_END = seg_after(FAIRSEP)
H_CUTS = [FAIRSEP, snap(FAIRSEP + 2 * BAR)]
J_CUTS = [SES, snap(SES + 3 * BAR), snap(SES + 6 * BAR)]
SHOTS = [  # (start index, name)
    (A_CUTS[0], "A1_wide"), (A_CUTS[1], "A2_clock"), (A_CUTS[2], "A3_logo"), (A_CUTS[3], "A4_low"),
    (B_CUTS[0], "B1_ign"), (B_CUTS[1], "B2_tower"), (B_CUTS[2], "B3_track"), (B_CUTS[3], "B4_tail"),
    (C_CUTS[0], "C1_cloud"), (C_CUTS[1], "C2_side"),
    (D_CUTS[0], "D1_maxq"), (D_CUTS[1], "D2_under"), (D_CUTS[2], "D3_wide"),
    (D_END, "E_calm"),
    (F_CUTS[0], "F1_sep"), (F_CUTS[1], "F2_tail"), (F_CUTS[2], "F3_wide"),
    (F_END, "G_calm"),
    (H_CUTS[0], "H1_fair"), (H_CUTS[1], "H2_side"),
    (MECO, "I_meco"),
    (J_CUTS[0], "J1_limb"), (J_CUTS[1], "J2_side"), (J_CUTS[2], "J3_wide"),
    (SECO, "K_outro"),
]
SHOTS.sort()
CUTS = sorted({s for s, _ in SHOTS if s > 0} | {PAD_END, BSEP, SES})
SHOT_OF = np.zeros(N, np.int32)
for k, (s, _) in enumerate(SHOTS):
    SHOT_OF[s:] = k
print("[rocket] entries", ENT, "MECO", MECO, "SECO", SECO, "PAD_END", PAD_END)
print("[rocket] shots", [(s, n) for s, n in SHOTS])

# =====================================================================================
# 3. BLENDER HELPERS
# =====================================================================================
SC = bpy.context.scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
             bpy.data.curves, bpy.data.worlds, bpy.data.node_groups, bpy.data.images):
    for d in list(coll):
        coll.remove(d)

KEY_ENUM = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items
LIN, CONST = KEY_ENUM["LINEAR"].value, KEY_ENUM["CONSTANT"].value
CUT_SET = np.zeros(N, bool)
for c in CUTS:
    if 0 < c < N:
        CUT_SET[c - 1] = True                       # the last frame BEFORE a cut holds (no smear)


def _fcurve(idb, path, index):
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idb.name}_bake")
    return ad.action.fcurve_ensure_for_datablock(idb, path, index=index)


def bake(idb, path, vals, index=0, const=False):
    """Bake one value per timeline frame onto idb.path[index] (bulk foreach_set)."""
    v = np.asarray(vals, np.float64)
    if v.ndim == 0:
        v = np.full(N, float(v))
    assert len(v) == N, (path, len(v))
    keep = np.ones(N, bool)
    keep[1:-1] = (v[1:-1] != v[:-2]) | (v[1:-1] != v[2:])
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
    interp[CUT_SET[ix]] = CONST
    kp.foreach_set("interpolation", interp)
    fc.update()
    return fc


def bake_vec(idb, path, arr, const=False):
    arr = np.asarray(arr, np.float64)
    for k in range(arr.shape[1]):
        bake(idb, path, arr[:, k], index=k, const=const)


def link(ob, parent=None):
    SC.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def empty(name, loc=(0, 0, 0), parent=None):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    return link(e, parent)


def mesh_obj(name, verts, faces, mat=None, parent=None, loc=(0, 0, 0), smooth=False, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in verts], [], [tuple(map(int, f)) for f in faces])
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        loop_v = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loop_v)
        uvl.data.foreach_set("uv", np.asarray(uvs, np.float32)[loop_v].ravel())
    me.update()
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    if mat is not None:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    return link(ob, parent)


def lathe(name, prof, segs=64, mat=None, parent=None, loc=(0, 0, 0), a0=0.0, a1=2 * math.pi,
          cap=False, smooth=True):
    """Surface of revolution around +Z. prof = [(r, z), ...]. UV: u = angle/2pi, v = profile t."""
    prof = np.asarray(prof, np.float64)
    full = abs((a1 - a0) - 2 * math.pi) < 1e-6
    na = segs if full else segs + 1
    ang = a0 + (a1 - a0) * np.arange(na) / segs
    seglen = np.r_[0, np.cumsum(np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1])))]
    tv = seglen / max(seglen[-1], 1e-9)
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
    # per-vertex UV; seam duplicates avoided by computing per-loop below
    me_uvs = np.asarray(uvs, np.float32)
    ob = mesh_obj(name, verts, faces, mat, parent, loc, smooth=smooth, uvs=None)
    me = ob.data
    uvl = me.uv_layers.new(name="UVMap")
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    uv = me_uvs[loop_v].copy()
    # fix the wrap seam: in a face whose u spans > 0.5, lift the small ones by 1
    poly_start = np.empty(len(me.polygons), np.int32)
    poly_tot = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_start", poly_start)
    me.polygons.foreach_get("loop_total", poly_tot)
    for ps, pt in zip(poly_start, poly_tot):
        us = uv[ps:ps + pt, 0]
        if us.max() - us.min() > 0.5:
            us[us < 0.5] += 1.0
            uv[ps:ps + pt, 0] = us
    uvl.data.foreach_set("uv", uv.ravel())
    return ob


class MB:
    """Accumulate boxes / polys for ONE material, then emit one mesh object."""

    def __init__(self):
        self.v, self.f = [], []

    def box(self, c, s, rot=None):
        cx, cy, cz = c
        sx, sy, sz = (s[0] / 2, s[1] / 2, s[2] / 2)
        pts = [Vector((x * sx, y * sy, z * sz)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        if rot is not None:
            pts = [rot @ p for p in pts]
        o = len(self.v)
        self.v += [(p.x + cx, p.y + cy, p.z + cz) for p in pts]
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.f.append(tuple(o + i for i in q))

    def beam(self, a, b, t):
        a, b = Vector(a), Vector(b)
        d = b - a
        rot = d.to_track_quat("Z", "Y").to_matrix()
        self.box(tuple((a + b) / 2), (t, t, d.length), rot)

    def poly(self, pts, faces):
        o = len(self.v)
        self.v += [tuple(p) for p in pts]
        self.f += [tuple(o + i for i in f) for f in faces]

    def obj(self, name, mat=None, parent=None, loc=(0, 0, 0)):
        return mesh_obj(name, self.v, self.f, mat, parent, loc)


def uv_sphere(name, r=1.0, seg=32, rings=16, mat=None, parent=None, loc=(0, 0, 0)):
    prof = [(r * math.sin(math.pi * k / rings), -r * math.cos(math.pi * k / rings)) for k in range(rings + 1)]
    return lathe(name, prof, seg, mat, parent, loc)


def cube_obj(name, mat=None, parent=None, loc=(0, 0, 0), scale=(1, 1, 1)):
    m = MB()
    m.box((0, 0, 0), (2, 2, 2))
    ob = m.obj(name, mat, parent, loc)
    ob.scale = scale
    return ob


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


def _sock(nt, x, vec=False):
    if isinstance(x, (int, float)):
        v = nd(nt, "ShaderNodeValue")
        v.outputs[0].default_value = float(x)
        return v.outputs[0]
    if isinstance(x, (tuple, list)):
        c = nd(nt, "ShaderNodeCombineXYZ")
        for k in range(3):
            c.inputs[k].default_value = float(x[k])
        return c.outputs[0]
    return x


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


def mix_col(nt, fac, a, b):
    n = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    for k, x in ((0, fac), (6, a), (7, b)):
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = x
        elif isinstance(x, (tuple, list)):
            n.inputs[k].default_value = tuple(x) + ((1.0,) if len(x) == 3 else ())
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[2]


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.node_tree.nodes.clear()
    return m, m.node_tree


def principled(name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0, bump=None):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = tuple(color) + (1,)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = tuple(emit) + (1,)
        b.inputs["Emission Strength"].default_value = emit_strength
    if bump:
        tc = nd(nt, "ShaderNodeTexCoord")
        nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": bump[0], "Detail": 8.0})
        lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
        bp = nd(nt, "ShaderNodeBump", inp={"Strength": bump[1]})
        lk(nt, nz.outputs["Fac"], bp.inputs["Height"])
        lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
        cr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.3, "From Max": 0.7,
                                               "To Min": rough * 0.7, "To Max": min(1, rough * 1.3)})
        lk(nt, nz.outputs["Fac"], cr.inputs["Value"])
        lk(nt, cr.outputs["Result"], b.inputs["Roughness"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def emissive(name, color, strength, objcolor_gain=False):
    """Emission; with objcolor_gain the per-object colour (keyable!) multiplies it."""
    m, nt = new_mat(name)
    e = nd(nt, "ShaderNodeEmission", inp={"Color": tuple(color) + (1,), "Strength": strength})
    if objcolor_gain:
        oi = nd(nt, "ShaderNodeObjectInfo")
        mc = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        mc.inputs[0].default_value = 1.0
        mc.inputs[6].default_value = tuple(color) + (1,)
        lk(nt, oi.outputs["Color"], mc.inputs[7])
        lk(nt, mc.outputs[2], e.inputs["Color"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, e.outputs[0], o.inputs["Surface"])
    return m


def set_color_keys(ob, rgb_arr, const=True):
    """Bake obj.color (RGBA) — read in shaders through Object Info > Color."""
    arr = np.asarray(rgb_arr, np.float64)
    if arr.ndim == 1:
        arr = np.repeat(arr[:, None], 3, 1)
    for k in range(3):
        bake(ob, "color", arr[:, k], index=k, const=const)


def no_shadow(ob):
    ob.visible_shadow = False
    return ob


VIS = {}                                    # object -> per-frame visibility (bool array)


def show(ob, mask):
    VIS[ob.name] = np.asarray(mask, bool)
    return ob


def blackbody(k):
    """Approx sRGB-linear colour of a blackbody at k kelvin (Tanner Helland fit)."""
    t = k / 100.0
    r = 1.0 if t <= 66 else min(1, max(0, 329.698727446 * (t - 60) ** -0.1332047592 / 255))
    g = (min(1, max(0, (99.4708025861 * math.log(t) - 161.1195681661) / 255)) if t <= 66
         else min(1, max(0, 288.1221695283 * (t - 60) ** -0.0755148492 / 255)))
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else min(1, max(0, (138.5177312231 * math.log(t - 10)
                                                                - 305.0447927307) / 255)))
    return tuple(c ** 2.2 for c in (r, g, b))


# =====================================================================================
# 4. MATERIALS
# =====================================================================================
def load_logo(path):
    """The emblem PNG is opaque on white: flood-fill the white that touches the border → alpha."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    white = px[..., :3].min(-1) > 0.86
    reg = np.zeros_like(white)
    reg[0, :] = white[0, :]
    reg[-1, :] = white[-1, :]
    reg[:, 0] = white[:, 0]
    reg[:, -1] = white[:, -1]
    for _ in range(2000):
        g = reg.copy()
        g[1:] |= reg[:-1]
        g[:-1] |= reg[1:]
        g[:, 1:] |= reg[:, :-1]
        g[:, :-1] |= reg[:, 1:]
        g &= white
        if (g == reg).all():
            break
        reg = g
    a = 1.0 - reg.astype(np.float32)
    k = np.array([1, 4, 6, 4, 1], np.float32)
    k /= k.sum()
    for ax in (0, 1):
        a = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 2, mode="edge"), k, "valid"), ax, a)
    out = bpy.data.images.new("TzekeLogoKeyed", w, h, alpha=True, float_buffer=False)
    px[..., 3] = a
    out.pixels.foreach_set(px.ravel())
    out.pack()
    bpy.data.images.remove(img)
    return out


LOGO = load_logo(ARGS.logo)


def paint_material(name, logo=None, logo_rect=None, frost=False, soot=True):
    """White rocket paint: panel/dirt bump, soot toward the base, optional frost + logo decal.
    logo_rect = (u0, u1, v0, v1) in the lathe UV (u around, v along)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    b.inputs["Roughness"].default_value = 0.38
    b.inputs["Coat Weight"].default_value = 0.15
    tc = nd(nt, "ShaderNodeTexCoord")
    uv = tc.outputs["UV"]
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, uv, sep.inputs[0])
    obj = tc.outputs["Object"]
    # grime: streaks along the length (noise stretched in z)
    st = V(nt, "MULTIPLY", obj, (9.0, 9.0, 0.35))
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6})
    lk(nt, st, nz.inputs["Vector"])
    grime = M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz.outputs["Fac"], 0.35, clamp=True), 0.5)
    base = mix_col(nt, grime, (0.80, 0.80, 0.79), (0.55, 0.53, 0.50))
    if soot:
        sootf = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, sep.outputs[1], clamp=True), 6.0)
        base = mix_col(nt, M(nt, "MULTIPLY", sootf, 0.9, clamp=True), base, (0.05, 0.045, 0.04))
    # panel lines (rings every ~3.5% of the length + 2 seams around)
    ring = M(nt, "FRACT", M(nt, "MULTIPLY", sep.outputs[1], 14.0))
    ringl = M(nt, "LESS_THAN", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", ring, 0.5)), 0.012)
    base = mix_col(nt, M(nt, "MULTIPLY", ringl, 0.35), base, (0.3, 0.3, 0.3))
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.12, "Distance": 0.02})
    lk(nt, M(nt, "SUBTRACT", nz.outputs["Fac"], M(nt, "MULTIPLY", ringl, 0.6)), bp.inputs["Height"])
    nrm = bp.outputs["Normal"]
    rough = M(nt, "ADD", 0.32, M(nt, "MULTIPLY", grime, 0.6))
    if frost:
        fr = value_node(nt, "Frost", 1.0)
        fz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 28.0, "Detail": 10.0, "Roughness": 0.7})
        lk(nt, obj, fz.inputs["Vector"])
        band = M(nt, "MULTIPLY", M(nt, "GREATER_THAN", sep.outputs[1], 0.18),
                 M(nt, "LESS_THAN", sep.outputs[1], 0.93))
        ff = M(nt, "MULTIPLY", M(nt, "MULTIPLY", band, fr.outputs[0]),
               M(nt, "MULTIPLY_ADD", fz.outputs["Fac"], 1.4, -0.25), clamp=True)
        base = mix_col(nt, ff, base, (0.93, 0.95, 0.98))
        rough = M(nt, "ADD", rough, M(nt, "MULTIPLY", ff, 0.5))
        bp2 = nd(nt, "ShaderNodeBump", inp={"Strength": 0.35, "Distance": 0.01})
        lk(nt, M(nt, "MULTIPLY", fz.outputs["Fac"], ff), bp2.inputs["Height"])
        lk(nt, nrm, bp2.inputs["Normal"])
        nrm = bp2.outputs["Normal"]
    if logo is not None:
        u0, u1, v0, v1 = logo_rect
        lu = M(nt, "DIVIDE", M(nt, "SUBTRACT", sep.outputs[0], u0), u1 - u0)
        lv_ = M(nt, "DIVIDE", M(nt, "SUBTRACT", sep.outputs[1], v0), v1 - v0)
        cxy = nd(nt, "ShaderNodeCombineXYZ")
        lk(nt, M(nt, "SUBTRACT", 1.0, lu), cxy.inputs[0])       # lathe u runs CCW: flip so it reads
        lk(nt, lv_, cxy.inputs[1])
        it = nd(nt, "ShaderNodeTexImage", image=logo, extension="CLIP", interpolation="Cubic")
        lk(nt, cxy.outputs[0], it.inputs["Vector"])
        la = it.outputs["Alpha"]
        base = mix_col(nt, la, base, it.outputs["Color"])
        rough = mix_col(nt, la, rough, (0.25, 0.25, 0.25))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, rough, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


# core LOGO: 4.4 m tall, centred facing -Y (angle 270° → u 0.75); core length 42 (+skirt)
CORE_LEN = 42.0
LOGO_H = 4.6
LOGO_W_U = LOGO_H / (2 * math.pi * 1.85)
LOGO_Z0 = 27.0
M_PAINT_CORE = paint_material("PaintCore", LOGO, (0.75 - LOGO_W_U / 2, 0.75 + LOGO_W_U / 2,
                                                  LOGO_Z0 / CORE_LEN, (LOGO_Z0 + LOGO_H) / CORE_LEN),
                              frost=True)
M_PAINT_BOOST = paint_material("PaintBooster", frost=True)
M_PAINT_S2 = paint_material("PaintS2", soot=False)
M_CARBON = principled("Carbon", (0.025, 0.025, 0.027), 0.42, 0.0, bump=(60, 0.08))
M_STEEL = principled("SteelDark", (0.12, 0.12, 0.13), 0.38, 1.0, bump=(40, 0.05))
M_STEEL_TOWER = principled("TowerSteel", (0.32, 0.30, 0.28), 0.55, 0.85, bump=(6, 0.15))
M_CONCRETE = principled("Concrete", (0.36, 0.35, 0.33), 0.85, 0.0, bump=(3, 0.25))
M_GOLD = principled("GoldFoil", (1.0, 0.72, 0.30), 0.55, 1.0, bump=(9, 0.9))
M_SOLAR = principled("Solar", (0.02, 0.03, 0.09), 0.18, 0.4, bump=(80, 0.05))
M_BLACKFAB = principled("BannerFabric", (0.02, 0.02, 0.025), 0.8)


def nozzle_material(name):
    """Dark engine bell that glows from its throat when hot (per-object colour = heat)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.09, 0.08, 0.075, 1), "Metallic": 1.0,
                                                 "Roughness": 0.45})
    tc = nd(nt, "ShaderNodeTexCoord")
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["UV"], sep.inputs[0])
    oi = nd(nt, "ShaderNodeObjectInfo")
    heat = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, oi.outputs["Color"], heat.inputs[0])
    g = M(nt, "MULTIPLY", M(nt, "POWER", M(nt, "SUBTRACT", 1.0, sep.outputs[1]), 2.0), heat.outputs[0])
    lk(nt, M(nt, "MULTIPLY", g, 6.0), b.inputs["Emission Strength"])
    b.inputs["Emission Color"].default_value = (1.0, 0.32, 0.08, 1)
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


M_NOZZLE = nozzle_material("Nozzle")


def plume_material(name, mach=True, vacuum=False):
    """Additive exhaust: Transparent + Emission. Object colour R = intensity, G = green
    (TEA-TEB ignition), B = length phase. UV v: 0 at the nozzle → 1 at the tail."""
    m, nt = new_mat(name)
    m.surface_render_method = "BLENDED"
    m.use_transparency_overlap = True
    tc = nd(nt, "ShaderNodeTexCoord")
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["UV"], sep.inputs[0])
    v = sep.outputs[1]
    oi = nd(nt, "ShaderNodeObjectInfo")
    oc = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, oi.outputs["Color"], oc.inputs[0])
    gain, green, scroll = oc.outputs[0], oc.outputs[1], oc.outputs[2]
    # turbulence: noise in object space, scrolled along the flow by the keyed phase (B)
    off = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, M(nt, "MULTIPLY", scroll, -40.0), off.inputs[2])
    p = V(nt, "ADD", tc.outputs["Object"], off.outputs[0])
    p = V(nt, "MULTIPLY", p, (1.0, 1.0, 0.28))
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.55 if not vacuum else 0.15, "Detail": 7.0,
                                           "Roughness": 0.62, "Distortion": 0.4})
    lk(nt, p, nz.inputs["Vector"])
    turb = nz.outputs["Fac"]
    # colour along the flow
    cr = nd(nt, "ShaderNodeValToRGB")
    els = cr.color_ramp.elements
    if vacuum:
        cols = [(0.0, (1.0, 0.85, 0.65)), (0.08, (1.0, 0.55, 0.30)), (0.45, (0.55, 0.30, 0.55)),
                (1.0, (0.12, 0.10, 0.30))]
    else:
        cols = [(0.0, (1.0, 0.97, 0.90)), (0.10, (1.0, 0.86, 0.55)), (0.35, (1.0, 0.55, 0.18)),
                (0.75, (0.85, 0.25, 0.05)), (1.0, (0.35, 0.07, 0.02))]
    els[0].position, els[0].color = cols[0][0], cols[0][1] + (1,)
    els[1].position, els[1].color = cols[-1][0], cols[-1][1] + (1,)
    for pos, c in cols[1:-1]:
        e = els.new(pos)
        e.color = c + (1,)
    lk(nt, v, cr.inputs[0])
    col = mix_col(nt, M(nt, "MULTIPLY", green, 0.95, clamp=True), cr.outputs[0], (0.25, 1.0, 0.35))
    # alpha: fade along length × soft edges (facing) × turbulence
    lw = nd(nt, "ShaderNodeLayerWeight", inp={"Blend": 0.5})
    face = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 2.4)
    along = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, v, clamp=True), 1.5 if not vacuum else 0.9)
    a = M(nt, "MULTIPLY", face, along)
    a = M(nt, "MULTIPLY", a, M(nt, "MULTIPLY_ADD", turb, 1.6, -0.3, clamp=True))
    if mach:   # shock diamonds near the nozzle
        d = M(nt, "MULTIPLY_ADD", M(nt, "COSINE", M(nt, "MULTIPLY", v, 2 * math.pi * 7.0)), 0.35, 0.75)
        nearm = M(nt, "LESS_THAN", v, 0.32)
        d = M(nt, "ADD", M(nt, "MULTIPLY", nearm, M(nt, "SUBTRACT", d, 1.0)), 1.0)
        a = M(nt, "MULTIPLY", a, d)
    strength = M(nt, "MULTIPLY", a, M(nt, "MULTIPLY", gain, 1.0))
    em = nd(nt, "ShaderNodeEmission")
    lk(nt, col, em.inputs["Color"])
    lk(nt, strength, em.inputs["Strength"])
    tr = nd(nt, "ShaderNodeBsdfTransparent")
    add = nd(nt, "ShaderNodeAddShader")
    lk(nt, tr.outputs[0], add.inputs[0])
    lk(nt, em.outputs[0], add.inputs[1])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, add.outputs[0], o.inputs["Surface"])
    return m


def plume_volume_material(name, vacuum=False):
    """Emissive VOLUME plume in a unit cylinder (r<=1, z: 0 nozzle → -1 tail), scaled per
    object. Object colour R = intensity, G = green ignition (TEA-TEB), B = flow phase.
    Soft edges come free (radial falloff + turbulence), unlike a shell mesh."""
    m, nt = new_mat(name)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    oi = nd(nt, "ShaderNodeObjectInfo")
    oc = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, oi.outputs["Color"], oc.inputs[0])
    gain, green, scroll = oc.outputs[0], oc.outputs[1], oc.outputs[2]
    v = M(nt, "MULTIPLY", sp.outputs[2], -1.0, clamp=True)
    r = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    off = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, M(nt, "MULTIPLY", scroll, 1.2), off.inputs[2])
    q = V(nt, "ADD", V(nt, "MULTIPLY", ob, (2.2, 2.2, 7.0) if not vacuum else (1.4, 1.4, 3.0)), off.outputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D",
            inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6, "Distortion": 0.3})
    lk(nt, q, nz.inputs["Vector"])
    lk(nt, M(nt, "MULTIPLY", scroll, 0.7), nz.inputs["W"])
    turb = nz.outputs["Fac"]
    r0, pw = (0.16, 0.5) if not vacuum else (0.08, 0.32)
    R_ = M(nt, "MULTIPLY", M(nt, "ADD", r0, M(nt, "MULTIPLY", M(nt, "POWER", v, pw), 1.0 - r0)), 0.62)
    rn = M(nt, "ADD", M(nt, "DIVIDE", r, R_), M(nt, "MULTIPLY", M(nt, "SUBTRACT", turb, 0.5), 0.6))
    mask = nd(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP",
              inp={"From Min": 1.0, "From Max": 0.25, "To Min": 0.0, "To Max": 1.0})
    lk(nt, rn, mask.inputs["Value"])
    mk = mask.outputs["Result"]
    core = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, rn, clamp=True), 3.0)
    along = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, v, clamp=True), 1.3 if not vacuum else 0.8)
    attach = M(nt, "MULTIPLY", M(nt, "DIVIDE", v, 0.03), 1.0, clamp=True)
    e = M(nt, "MULTIPLY", M(nt, "MULTIPLY", mk, along), attach)
    e = M(nt, "MULTIPLY", e, M(nt, "MULTIPLY_ADD", core, 2.5, 0.35))
    e = M(nt, "MULTIPLY", e, M(nt, "MULTIPLY_ADD", turb, 1.2, 0.4))
    e = M(nt, "MULTIPLY", e, gain)
    cr = nd(nt, "ShaderNodeValToRGB")
    els = cr.color_ramp.elements
    cols = ([(0.0, (1.0, 0.95, 0.85)), (0.12, (1.0, 0.78, 0.42)), (0.35, (1.0, 0.48, 0.14)),
             (0.7, (0.8, 0.22, 0.05)), (1.0, (0.3, 0.06, 0.02))] if not vacuum else
            [(0.0, (1.0, 0.8, 0.6)), (0.1, (1.0, 0.5, 0.3)), (0.5, (0.55, 0.32, 0.6)), (1.0, (0.15, 0.12, 0.35))])
    els[0].position, els[0].color = cols[0][0], cols[0][1] + (1,)
    els[1].position, els[1].color = cols[-1][0], cols[-1][1] + (1,)
    for pos, c in cols[1:-1]:
        el = els.new(pos)
        el.color = c + (1,)
    lk(nt, M(nt, "ADD", v, M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, core), 0.25)), cr.inputs[0])
    col = mix_col(nt, M(nt, "MULTIPLY", green, 0.95, clamp=True), cr.outputs[0], (0.3, 1.0, 0.4))
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": (0.35, 0.3, 0.27, 1), "Anisotropy": 0.3})
    lk(nt, col, pv.inputs["Emission Color"])
    lk(nt, e, pv.inputs["Emission Strength"])
    smoke = 0.0 if vacuum else 0.05
    lk(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", mk, M(nt, "POWER", v, 1.5)), smoke), pv.inputs["Density"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


M_PLUME = plume_material("Plume")
M_PLUME_VOL = plume_volume_material("PlumeVol")
M_PLUME_VVOL = plume_volume_material("PlumeVacVol", vacuum=True)


def volume_material(name, color=(0.9, 0.9, 0.92), shape="SPHERE", nscale=2.0, detail=6.0,
                    emis_color=(1.0, 0.45, 0.12), aniso=0.3, low_glow=True):
    """Billowing smoke/steam/cloud volume. Keyed Value nodes: Density, Evolve (4D W),
    Glow (emission gain, applied to the lower half). shape SPHERE = unit-sphere falloff in
    object space; SLAB = cloud deck (box, puffy top); COLUMN = trail (cylinder)."""
    m, nt = new_mat(name)
    dens = value_node(nt, "Density", 1.0)
    evo = value_node(nt, "Evolve", 0.0)
    glow = value_node(nt, "Glow", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sepo = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sepo.inputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D",
            inp={"Scale": nscale, "Detail": detail, "Roughness": 0.58, "Distortion": 0.25})
    lk(nt, ob, nz.inputs["Vector"])
    lk(nt, evo.outputs[0], nz.inputs["W"])
    n = nz.outputs["Fac"]
    if shape == "SPHERE":
        r = V(nt, "LENGTH", ob)
        mask = M(nt, "SUBTRACT", M(nt, "ADD", M(nt, "SUBTRACT", 1.0, r), M(nt, "MULTIPLY", n, 0.9)),
                 0.55, clamp=True)
        mask = M(nt, "MULTIPLY", mask, 2.2)
        h = M(nt, "MULTIPLY_ADD", sepo.outputs[2], 0.5, 0.5, clamp=True)
    elif shape == "SLAB":
        h = M(nt, "MULTIPLY_ADD", sepo.outputs[2], 0.5, 0.5, clamp=True)
        # coverage: big puffy cells; density falls off toward a noisy top
        iso = V(nt, "MULTIPLY", ob, (13.0, 13.0, 0.8))
        lk(nt, iso, nz.inputs["Vector"])
        nz2 = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.35, "Detail": 3.0})
        lk(nt, iso, nz2.inputs["Vector"])
        top = M(nt, "MULTIPLY_ADD", nz2.outputs["Fac"], 1.3, -0.35)
        mask = M(nt, "SUBTRACT", M(nt, "ADD", M(nt, "SUBTRACT", top, h), M(nt, "MULTIPLY", n, 0.7)),
                 0.3, clamp=True)
        mask = M(nt, "MULTIPLY", mask, 3.0)
        edge = M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, M(nt, "MAXIMUM",
                                                           M(nt, "ABSOLUTE", sepo.outputs[0]),
                                                           M(nt, "ABSOLUTE", sepo.outputs[1])), clamp=True), 6.0)
        mask = M(nt, "MULTIPLY", mask, M(nt, "MINIMUM", edge, 1.0))
    elif shape == "UNIFORM":
        h = M(nt, "MULTIPLY_ADD", sepo.outputs[2], 0.5, 0.5, clamp=True)
        mask = M(nt, "MULTIPLY", M(nt, "POWER", M(nt, "SUBTRACT", 1.0, h), 2.0), M(nt, "MULTIPLY_ADD", n, 0.8, 0.6))
    else:  # COLUMN: unit cylinder r<=1 along z in [0,1]
        rr = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
        h = M(nt, "MAXIMUM", sepo.outputs[2], 0.0)
        mask = M(nt, "SUBTRACT", M(nt, "ADD", M(nt, "SUBTRACT", 1.0, rr), M(nt, "MULTIPLY", n, 0.9)),
                 0.55, clamp=True)
        mask = M(nt, "MULTIPLY", mask, 2.2)
    d = M(nt, "MULTIPLY", mask, dens.outputs[0])
    green = value_node(nt, "Green", 0.0)
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": tuple(color) + (1,), "Anisotropy": aniso})
    lk(nt, mix_col(nt, green.outputs[0], tuple(emis_color), (0.3, 1.0, 0.4)), pv.inputs["Emission Color"])
    lk(nt, d, pv.inputs["Density"])
    low = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, h, clamp=True), 2.0) if low_glow else 1.0
    lk(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", low, glow.outputs[0]), mask), pv.inputs["Emission Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


def vapor_cone_material(name):
    """Prandtl–Glauert condensation collar: white, translucent, flickering, edge-soft."""
    m, nt = new_mat(name)
    m.surface_render_method = "BLENDED"
    m.use_transparency_overlap = True
    tc = nd(nt, "ShaderNodeTexCoord")
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["UV"], sep.inputs[0])
    gain = value_node(nt, "Gain", 0.0)
    evo = value_node(nt, "Evolve", 0.0)
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 0.55, "Detail": 4.0, "Roughness": 0.55})
    lk(nt, V(nt, "MULTIPLY", tc.outputs["Object"], (1.0, 1.0, 0.35)), nz.inputs["Vector"])
    lk(nt, evo.outputs[0], nz.inputs["W"])
    v = sep.outputs[1]
    prof = M(nt, "MULTIPLY", M(nt, "POWER", v, 1.6), M(nt, "SUBTRACT", 1.0, M(nt, "POWER", v, 14.0)))
    lw = nd(nt, "ShaderNodeLayerWeight", inp={"Blend": 0.5})
    face = M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", lw.outputs["Facing"], 0.7))
    a = M(nt, "MULTIPLY", M(nt, "MULTIPLY", prof, face),
          M(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 2.2, -0.65, clamp=True))
    a = M(nt, "MULTIPLY", a, gain.outputs[0], clamp=True)
    df = nd(nt, "ShaderNodeBsdfDiffuse", inp={"Color": (0.95, 0.96, 1.0, 1)})
    em = nd(nt, "ShaderNodeEmission", inp={"Color": (0.85, 0.9, 1.0, 1), "Strength": 0.25})
    sh = nd(nt, "ShaderNodeAddShader")
    lk(nt, df.outputs[0], sh.inputs[0])
    lk(nt, em.outputs[0], sh.inputs[1])
    tr = nd(nt, "ShaderNodeBsdfTransparent")
    mx = nd(nt, "ShaderNodeMixShader")
    lk(nt, a, mx.inputs[0])
    lk(nt, tr.outputs[0], mx.inputs[1])
    lk(nt, sh.outputs[0], mx.inputs[2])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, mx.outputs[0], o.inputs["Surface"])
    return m


M_VCONE = vapor_cone_material("VaporCone")


def shock_ring_material(name):
    m, nt = new_mat(name)
    m.surface_render_method = "BLENDED"
    gain = value_node(nt, "Gain", 0.0)
    lw = nd(nt, "ShaderNodeLayerWeight", inp={"Blend": 0.25})
    a = M(nt, "MULTIPLY", M(nt, "POWER", lw.outputs["Fresnel"], 2.0), gain.outputs[0])
    em = nd(nt, "ShaderNodeEmission", inp={"Color": (0.9, 0.95, 1.0, 1)})
    lk(nt, M(nt, "MULTIPLY", a, 4.0), em.inputs["Strength"])
    tr = nd(nt, "ShaderNodeBsdfTransparent")
    add = nd(nt, "ShaderNodeAddShader")
    lk(nt, tr.outputs[0], add.inputs[0])
    lk(nt, em.outputs[0], add.inputs[1])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, add.outputs[0], o.inputs["Surface"])
    return m


M_SHOCK = shock_ring_material("ShockRing")


def ground_material():
    """Land with a noisy coastline into a glossy dark sea to the north (+Y)."""
    m, nt = new_mat("Ground")
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    tc = nd(nt, "ShaderNodeTexCoord")
    p = tc.outputs["Object"]
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, p, sep.inputs[0])
    cn = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.0016, "Detail": 6.0})
    lk(nt, p, cn.inputs["Vector"])
    coast = M(nt, "ADD", 950.0, M(nt, "MULTIPLY", cn.outputs["Fac"], 700.0))
    sea = M(nt, "GREATER_THAN", sep.outputs[1], coast)
    gn = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.02, "Detail": 10.0, "Roughness": 0.65})
    lk(nt, p, gn.inputs["Vector"])
    land = mix_col(nt, gn.outputs["Fac"], (0.035, 0.040, 0.025), (0.09, 0.08, 0.06))
    rr = V(nt, "LENGTH", V(nt, "MULTIPLY", p, (1, 1, 0)))
    apron = M(nt, "LESS_THAN", M(nt, "ADD", rr, M(nt, "MULTIPLY", gn.outputs["Fac"], 30)), 150.0)
    land = mix_col(nt, apron, land, (0.20, 0.19, 0.17))
    col = mix_col(nt, sea, land, (0.004, 0.008, 0.012))
    lk(nt, col, b.inputs["Base Color"])
    lk(nt, M(nt, "ADD", M(nt, "MULTIPLY", sea, -0.84), 0.9), b.inputs["Roughness"])
    wv = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.09, "Detail": 4.0})
    lk(nt, V(nt, "MULTIPLY", p, (1.0, 3.0, 1.0)), wv.inputs["Vector"])
    bp = nd(nt, "ShaderNodeBump", inp={"Distance": 0.3})
    lk(nt, M(nt, "MULTIPLY", sea, 0.25), bp.inputs["Strength"])
    lk(nt, wv.outputs["Fac"], bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


M_GROUND = ground_material()
M_LED = emissive("LED", (1.0, 1.0, 1.0), 1.0, objcolor_gain=True)
M_LAMP = emissive("Lamp", (1.0, 1.0, 1.0), 1.0, objcolor_gain=True)

# =====================================================================================
# 5. THE ROCKET (metres; rig origin = base of stage 1, +Z = flight axis)
# =====================================================================================
R = 1.85
RIG = empty("RocketRig")
S1 = empty("S1", parent=RIG)
S2 = empty("S2", parent=RIG)
BL = empty("BoosterL", loc=(-4.0, 0, 0), parent=RIG)
BRt = empty("BoosterR", loc=(4.0, 0, 0), parent=RIG)

core = lathe("CoreBody", [(R * 0.97, -0.6), (R, 0.4), (R, CORE_LEN)], 72, M_PAINT_CORE, S1)
lathe("Interstage", [(R, CORE_LEN), (R, CORE_LEN + 4.2)], 72, M_CARBON, S1)
octaweb = MB()
octaweb.poly([(R * 0.96 * math.cos(a), R * 0.96 * math.sin(a), -0.6) for a in np.linspace(0, 2 * math.pi, 33)[:-1]],
             [tuple(range(32))[::-1]])
octaweb.obj("CoreBase", M_STEEL, S1)
lathe("S2Body", [(R, CORE_LEN + 4.2), (R, CORE_LEN + 13.5), (R * 1.02, CORE_LEN + 14.0)], 72, M_PAINT_S2, S2)
FAIR_Z = CORE_LEN + 14.0
fair_prof = [(R * 1.02, 0.0), (2.6, 1.4), (2.6, 7.6), (2.48, 9.4), (2.12, 11.1), (1.55, 12.5),
             (0.85, 13.5), (0.25, 14.0), (0.0, 14.1)]
FL = empty("FairingL", loc=(0, 2.6, FAIR_Z), parent=S2)      # hinges at the base, ±Y
FR = empty("FairingR", loc=(0, -2.6, FAIR_Z), parent=S2)
lathe("FairingHalfL", fair_prof, 36, M_PAINT_S2, FL, loc=(0, -2.6, 0), a0=0.0, a1=math.pi)
lathe("FairingHalfR", fair_prof, 36, M_PAINT_S2, FR, loc=(0, 2.6, 0), a0=math.pi, a1=2 * math.pi)
# payload (revealed at fairing sep): bus + folded solar wings + antenna
pl = MB()
pl.box((0, 0, FAIR_Z + 3.0), (2.4, 2.4, 4.4))
pl.obj("PayloadBus", M_GOLD, S2)
sw = MB()
sw.box((1.32, 0, FAIR_Z + 3.0), (0.12, 2.2, 4.0))
sw.box((-1.32, 0, FAIR_Z + 3.0), (0.12, 2.2, 4.0))
sw.obj("PayloadWings", M_SOLAR, S2)
lathe("PayloadDish", [(0.0, FAIR_Z + 5.2), (0.9, FAIR_Z + 5.5), (1.1, FAIR_Z + 5.9)], 32, M_STEEL, S2)
# upper-stage vacuum nozzle (hidden inside the interstage until staging)
s2noz = lathe("S2Nozzle", [(0.45, CORE_LEN + 4.2), (0.62, CORE_LEN + 3.4), (1.05, CORE_LEN + 1.8),
                           (1.45, CORE_LEN - 0.4), (1.62, CORE_LEN - 1.9)], 48, M_NOZZLE, S2)


def engine_cluster(name, parent, n_ring=8, ring_r=1.22, bell=(0.13, 0.46), length=1.7, z0=-0.6):
    allv, allf = [], []
    pos = [(0.0, 0.0)] + [(ring_r * math.cos(2 * math.pi * k / n_ring), ring_r * math.sin(2 * math.pi * k / n_ring))
                          for k in range(n_ring)]
    seg = 24
    prof = [(bell[0], z0), (bell[0] * 1.15, z0 - 0.25), (bell[0] + (bell[1] - bell[0]) * 0.55, z0 - length * 0.55),
            (bell[1], z0 - length)]
    uvs = []
    for (px, py) in pos:
        o = len(allv)
        for j, (r, z) in enumerate(prof):
            for k in range(seg):
                a = 2 * math.pi * k / seg
                allv.append((px + r * math.cos(a), py + r * math.sin(a), z))
                uvs.append((k / seg, j / (len(prof) - 1)))
        for j in range(len(prof) - 1):
            for k in range(seg):
                k2 = (k + 1) % seg
                allf.append((o + j * seg + k, o + j * seg + k2, o + (j + 1) * seg + k2, o + (j + 1) * seg + k))
    return mesh_obj(name, allv, allf, M_NOZZLE, parent, smooth=True, uvs=uvs), pos


core_eng, CORE_POS = engine_cluster("CoreEngines", S1)
boosters = []
for side, bem in ((-1, BL), (1, BRt)):
    b = lathe(f"BoosterBody{'LR'[side > 0]}", [(R * 0.97, -0.6), (R, 0.4), (R, CORE_LEN - 0.5), (R * 0.93, CORE_LEN + 1.6),
                                                (R * 0.72, CORE_LEN + 3.6), (R * 0.40, CORE_LEN + 5.0),
                                                (R * 0.12, CORE_LEN + 5.7), (0.0, CORE_LEN + 5.8)],
              64, M_PAINT_BOOST, bem)
    lathe(f"BoosterBand{'LR'[side > 0]}", [(R * 1.005, CORE_LEN - 6.0), (R * 1.005, CORE_LEN - 2.0)], 64, M_CARBON, bem)
    eng, _ = engine_cluster(f"BoosterEngines{'LR'[side > 0]}", bem)
    st = MB()                                                       # attach struts + legs
    for z in (4.0, CORE_LEN - 4.0):
        st.box((-side * 1.95, 0, z), (0.5, 0.5, 0.6))
    for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4):
        st.box((R * 1.04 * math.cos(a), R * 1.04 * math.sin(a), 4.6), (0.28, 0.28, 9.0))
    st.obj(f"BoosterStruts{'LR'[side > 0]}", M_CARBON, bem)
    boosters.append(bem)
legs = MB()
for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4):
    legs.box((R * 1.04 * math.cos(a), R * 1.04 * math.sin(a), 4.6), (0.28, 0.28, 9.0))
legs.box((R * 1.02 * math.cos(-0.35), R * 1.02 * math.sin(-0.35), CORE_LEN / 2), (0.16, 0.16, CORE_LEN - 2))  # raceway
legs.obj("CoreLegs", M_CARBON, S1)

# --- plumes: (inner core with shock diamonds + soft outer sheath) per engine cluster -----
PLUMES = []          # (object, kind)  kind: core/boost/s2


PLUME_W, PLUME_L = 7.5, 75.0           # volume sheath radius at the tail / length (sea level)


def plume_pair(tag, parent, kind, base_r=1.9, length=24.0, z0=-2.2):
    inner = lathe(f"Plume_{tag}_in", [(base_r * 0.55, 0.0), (base_r * 0.75, -0.08 * length), (base_r * 0.85, -0.3 * length),
                                      (base_r * 0.70, -0.65 * length), (base_r * 0.25, -length)],
                  48, M_PLUME, parent, loc=(0, 0, z0))
    outer = lathe(f"Plume_{tag}_vol", [(1.0, 0.0), (1.0, -1.0)], 32, M_PLUME_VOL, parent, loc=(0, 0, z0 + 0.3),
                  cap=True)
    for o in (inner, outer):
        no_shadow(o)
        o.visible_glossy = True
        PLUMES.append((o, kind))
    return inner, outer


plume_pair("core", S1, "core")
plume_pair("bl", BL, "boost")
plume_pair("br", BRt, "boost")
vac = lathe("Plume_s2_vol", [(1.0, 0.0), (1.0, -1.0)], 32, M_PLUME_VVOL, S2, loc=(0, 0, CORE_LEN - 1.7), cap=True)
no_shadow(vac)
PLUMES.append((vac, "s2"))
vac_in = lathe("Plume_s2_in", [(1.4, 0.0), (1.9, -3.0), (2.4, -12.0), (1.8, -26.0), (0.4, -34.0)],
               48, M_PLUME, S2, loc=(0, 0, CORE_LEN - 1.9))
no_shadow(vac_in)
PLUMES.append((vac_in, "s2"))

# plume lights (they light the pad, the smoke, the clouds and the rocket's skin)
PLIGHTS = []
for tag, par, z in (("core", S1, -1.2), ("bl", BL, -1.2), ("br", BRt, -1.2), ("s2", S2, CORE_LEN - 8)):
    ld = bpy.data.lights.new(f"PlumeLight_{tag}", "POINT")
    ld.color = blackbody(2100)
    ld.shadow_soft_size = 4.0
    ld.volume_factor = 1.0
    lo = bpy.data.objects.new(f"PlumeLight_{tag}", ld)
    lo.location = (0, 0, z)
    link(lo, par)
    PLIGHTS.append((lo, tag))

# bounce fill: the glowing exhaust cloud lights the body from below/beside
BOUNCE = bpy.data.lights.new("PlumeBounce", "POINT")
BOUNCE.color = blackbody(2300)
BOUNCE.shadow_soft_size = 10.0
BOUNCE.volume_factor = 0.0
BOUNCE.use_shadow = False
BOUNCE_OB = link(bpy.data.objects.new("PlumeBounce", BOUNCE), S1)
BOUNCE_OB.location = (6.0, -24.0, -14.0)
# --- vapour (LOX vents on the pad), vapour cone (max-Q), shock ring, ice debris ---------
M_VENT = volume_material("VentVapor", (0.95, 0.96, 1.0), "SPHERE", nscale=1.6, detail=5.0, aniso=0.45)
VENTS = []
for k, (z, ang, sx, sz) in enumerate([(CORE_LEN + 1.2, -1.45, 2.6, 4.0), (CORE_LEN + 12.8, -1.8, 2.0, 3.0),
                                       (CORE_LEN - 2.0, 2.4, 2.2, 3.5), (8.0, -0.9, 3.5, 6.0)]):
    v = uv_sphere(f"Vent{k}", 1.0, 16, 8, M_VENT, RIG,
                  loc=((R + sx * 0.7) * math.cos(ang), (R + sx * 0.7) * math.sin(ang), z - sz * 0.4))
    v.scale = (sx, sx, sz)
    VENTS.append(v)
vcone = lathe("VaporCone", [(6.4, -12.0), (4.8, -7.5), (3.4, -3.5), (2.75, 0.0)], 64, M_VCONE, RIG,
              loc=(0, 0, FAIR_Z + 1.6), smooth=True)
no_shadow(vcone)
shock = lathe("ShockRing", [(1.0, -0.15), (1.0, 0.15)], 96, M_SHOCK, RIG, loc=(0, 0, CORE_LEN + 4.0))
no_shadow(shock)


def debris_tree():
    """Ice chips shed from the tank on each snare: stateless — points from (seed=snare index),
    positions = v*age + 0.5*g*age^2 in rocket space. Inputs Age (s) and Seed are keyed."""
    t = bpy.data.node_groups.new("IceDebris", "GeometryNodeTree")
    t.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_age = t.interface.new_socket(name="Age", in_out="INPUT", socket_type="NodeSocketFloat")
    s_seed = t.interface.new_socket(name="Seed", in_out="INPUT", socket_type="NodeSocketInt")
    s_cnt = t.interface.new_socket(name="Count", in_out="INPUT", socket_type="NodeSocketInt")
    t.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = t.nodes.new("NodeGroupInput")
    go = t.nodes.new("NodeGroupOutput")
    pts = t.nodes.new("GeometryNodePoints")
    t.links.new(gi.outputs["Count"], pts.inputs["Count"])

    def rnd(lo, hi, sid, vec=False):
        r = t.nodes.new("FunctionNodeRandomValue")
        r.data_type = "FLOAT_VECTOR" if vec else "FLOAT"
        want = "VECTOR" if vec else "VALUE"
        ins = [x for x in r.inputs if x.enabled and x.type == want]
        outs = [x for x in r.outputs if x.enabled and x.type == want]
        ins[0].default_value = lo
        ins[1].default_value = hi
        add = t.nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        add.inputs[1].default_value = sid
        t.links.new(gi.outputs["Seed"], add.inputs[0])
        ti = t.nodes.new("FunctionNodeFloatToInt")
        t.links.new(add.outputs[0], ti.inputs[0])
        t.links.new(ti.outputs[0], r.inputs["Seed"])
        return outs[0]
    ang = rnd(-1.2, 0.6, 11)                         # facing the -Y side mostly (toward the cameras)
    zz = rnd(18.0, CORE_LEN + 2.0, 23)
    rv = rnd((-1.5, -1.5, -6.0), (1.5, 1.5, -1.0), 37, vec=True)
    # p0 = R*(cos a - sin a ...): build with math nodes
    cx = t.nodes.new("ShaderNodeMath")
    cx.operation = "COSINE"
    t.links.new(ang, cx.inputs[0])
    sx = t.nodes.new("ShaderNodeMath")
    sx.operation = "SINE"
    t.links.new(ang, sx.inputs[0])
    # rotate angle so the shed side faces -Y (camera side)
    p0 = t.nodes.new("ShaderNodeCombineXYZ")
    m1 = t.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m1.inputs[1].default_value = R * 1.05
    t.links.new(sx.outputs[0], m1.inputs[0])
    m2 = t.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m2.inputs[1].default_value = -R * 1.05
    t.links.new(cx.outputs[0], m2.inputs[0])
    t.links.new(m1.outputs[0], p0.inputs[0])
    t.links.new(m2.outputs[0], p0.inputs[1])
    t.links.new(zz, p0.inputs[2])
    # outward velocity along the radial + random
    vel = t.nodes.new("ShaderNodeVectorMath")
    vel.operation = "MULTIPLY_ADD"
    t.links.new(p0.outputs[0], vel.inputs[0])
    vel.inputs[1].default_value = (1.2, 1.2, 0.0)
    t.links.new(rv, vel.inputs[2])
    age = gi.outputs["Age"]
    va = t.nodes.new("ShaderNodeVectorMath")
    va.operation = "SCALE"
    t.links.new(vel.outputs[0], va.inputs[0])
    t.links.new(age, va.inputs["Scale"])
    a2 = t.nodes.new("ShaderNodeMath")
    a2.operation = "MULTIPLY"
    t.links.new(age, a2.inputs[0])
    t.links.new(age, a2.inputs[1])
    g = t.nodes.new("ShaderNodeVectorMath")
    g.operation = "SCALE"
    g.inputs[0].default_value = (0.0, 0.0, -9.0)
    t.links.new(a2.outputs[0], g.inputs["Scale"])
    pos = t.nodes.new("ShaderNodeVectorMath")
    pos.operation = "ADD"
    t.links.new(p0.outputs[0], pos.inputs[0])
    t.links.new(va.outputs[0], pos.inputs[1])
    pos2 = t.nodes.new("ShaderNodeVectorMath")
    pos2.operation = "ADD"
    t.links.new(pos.outputs[0], pos2.inputs[0])
    t.links.new(g.outputs[0], pos2.inputs[1])
    sp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], sp.inputs["Geometry"])
    t.links.new(pos2.outputs[0], sp.inputs["Position"])
    ico = t.nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 0.09
    ico.inputs["Subdivisions"].default_value = 1
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(sp.outputs[0], inst.inputs["Points"])
    t.links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    rr = rnd((0, 0, 0), (6.3, 6.3, 6.3), 51, vec=True)
    t.links.new(rr, inst.inputs["Rotation"])
    sc_r = rnd((0.4, 0.3, 0.2), (1.8, 1.2, 0.9), 63, vec=True)
    t.links.new(sc_r, inst.inputs["Scale"])
    sm = t.nodes.new("GeometryNodeSetMaterial")
    t.links.new(inst.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = principled("Ice", (0.85, 0.92, 1.0), 0.15, 0.0,
                                                     emit=(0.8, 0.9, 1.0), emit_strength=0.6)
    t.links.new(sm.outputs[0], go.inputs["Geometry"])
    return t, s_age, s_seed, s_cnt


def sparkle_tree():
    """Hi-hat sparkle: a fixed cloud of ice crystals around the rocket; on each frame a random
    subset (fraction = HAT input) glints. Stateless (seed = frame)."""
    t = bpy.data.node_groups.new("IceSparkle", "GeometryNodeTree")
    t.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_amt = t.interface.new_socket(name="Amount", in_out="INPUT", socket_type="NodeSocketFloat")
    t.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = t.nodes.new("NodeGroupInput")
    go = t.nodes.new("NodeGroupOutput")
    pts = t.nodes.new("GeometryNodePoints")
    pts.inputs["Count"].default_value = 900
    rp = t.nodes.new("FunctionNodeRandomValue")
    rp.data_type = "FLOAT_VECTOR"
    rp.inputs["Min"].default_value = (-1.0, -1.0, 0.0)
    rp.inputs["Max"].default_value = (1.0, 1.0, 1.0)
    rp.inputs["Seed"].default_value = 7
    # a shell around the rocket: scale to 26 m radius × 95 m tall, push out of the body
    sc_ = t.nodes.new("ShaderNodeVectorMath")
    sc_.operation = "MULTIPLY"
    sc_.inputs[1].default_value = (26.0, 26.0, 95.0)
    t.links.new(rp.outputs[0], sc_.inputs[0])
    off = t.nodes.new("ShaderNodeVectorMath")
    off.operation = "ADD"
    off.inputs[1].default_value = (0.0, 0.0, -20.0)
    t.links.new(sc_.outputs[0], off.inputs[0])
    sp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], sp.inputs["Geometry"])
    t.links.new(off.outputs[0], sp.inputs["Position"])
    st = t.nodes.new("GeometryNodeInputSceneTime")
    ti = t.nodes.new("FunctionNodeFloatToInt")
    t.links.new(st.outputs["Frame"], ti.inputs[0])
    rs = t.nodes.new("FunctionNodeRandomValue")
    rs.data_type = "FLOAT"
    t.links.new(ti.outputs[0], rs.inputs["Seed"])
    cmp = t.nodes.new("ShaderNodeMath")
    cmp.operation = "LESS_THAN"
    t.links.new([x for x in rs.outputs if x.enabled and x.type == "VALUE"][0], cmp.inputs[0])
    t.links.new(gi.outputs["Amount"], cmp.inputs[1])
    dl = sp
    ico = t.nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 0.035
    ico.inputs["Subdivisions"].default_value = 1
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(dl.outputs[0], inst.inputs["Points"])
    t.links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    t.links.new(cmp.outputs[0], inst.inputs["Scale"])          # 1 = glint, 0 = hidden (same topology)
    sm = t.nodes.new("GeometryNodeSetMaterial")
    t.links.new(inst.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = emissive("Glint", (0.85, 0.92, 1.0), 60.0)
    t.links.new(sm.outputs[0], go.inputs["Geometry"])
    return t, s_amt


DEB_TREE, DEB_AGE, DEB_SEED, DEB_CNT = debris_tree()
debris = mesh_obj("IceDebris", [], [], None, RIG)
dmod = debris.modifiers.new("Debris", "NODES")
dmod.node_group = DEB_TREE
getattr(dmod.properties.inputs, DEB_CNT.identifier).value = 46
no_shadow(debris)
SPK_TREE, SPK_AMT = sparkle_tree()
sparkle = mesh_obj("IceSparkle", [], [], None, RIG)
smod = sparkle.modifiers.new("Sparkle", "NODES")
smod.node_group = SPK_TREE
no_shadow(sparkle)

# =====================================================================================
# 6. THE PAD (world; pad deck top z=8, rocket base at z=PAD_Z)
# =====================================================================================
PAD_Z = 10.4
ground = MB()
ground.poly([(-9000, -9000, 0), (9000, -9000, 0), (9000, 9000, 0), (-9000, 9000, 0)], [(0, 1, 2, 3)])
g_ob = ground.obj("Ground", M_GROUND)
deck = MB()
deck.box((-26, 0, 4), (40, 60, 8))          # west block (tower side)
deck.box((20, 18, 4), (40, 24, 8))          # north-east block
deck.box((20, -18, 4), (40, 24, 8))         # south-east block (trench between, running +X)
deck.box((0, 0, 0.6), (12, 12, 1.2))        # trench floor under the rocket
deck.box((-2, 0, 3.0), (2, 12, 6))          # flame deflector wedge
# launch mount: ring frame + 4 hold-down columns
for a in np.linspace(0, 2 * math.pi, 5)[:-1] + math.pi / 4:
    deck.box((4.2 * math.cos(a), 4.2 * math.sin(a), 8 + 1.2), (1.0, 1.0, 2.4))
deck_ob = deck.obj("PadDeck", M_CONCRETE)
mount = MB()
for k in range(16):
    a0, a1 = 2 * math.pi * k / 16, 2 * math.pi * (k + 1) / 16
    mount.beam((4.4 * math.cos(a0), 4.4 * math.sin(a0), 9.9), (4.4 * math.cos(a1), 4.4 * math.sin(a1), 9.9), 0.7)
mount_ob = mount.obj("LaunchMount", M_STEEL)

# fixed service tower: lattice truss, 8 m square, 0..104 m
TW_X0, TW_X1, TW_Y = -17.0, -9.0, 4.0
TW_H = 104.0
tw = MB()
cols = [(TW_X0, -TW_Y), (TW_X1, -TW_Y), (TW_X1, TW_Y), (TW_X0, TW_Y)]
lvl = np.arange(8.0, TW_H + 0.1, 6.0)
for (x, y) in cols:
    tw.beam((x, y, 8.0), (x, y, TW_H), 0.75)
for z0, z1 in zip(lvl[:-1], lvl[1:]):
    for k in range(4):
        (xa, ya), (xb, yb) = cols[k], cols[(k + 1) % 4]
        tw.beam((xa, ya, z1), (xb, yb, z1), 0.42)
        if (k + int(z0 / 6)) % 2 == 0:
            tw.beam((xa, ya, z0), (xb, yb, z1), 0.26)
        else:
            tw.beam((xb, yb, z0), (xa, ya, z1), 0.26)
    tw.box(((TW_X0 + TW_X1) / 2, 0, z1 - 0.1), (TW_X1 - TW_X0, 2 * TW_Y, 0.2))   # grating floors
tw.box(((TW_X0 + TW_X1) / 2, 0, TW_H + 2), (9, 9, 4))        # top hut
tw.beam(((TW_X0 + TW_X1) / 2, 0, TW_H + 4), ((TW_X0 + TW_X1) / 2, 0, TW_H + 22), 0.5)   # lightning mast
tower = tw.obj("Tower", M_STEEL_TOWER)
# swing arms (pivot at the tower face; retract a bar before ignition)
ARMS = []
for k, (z, ln) in enumerate(((PAD_Z + CORE_LEN + 7.0, 7.2), (PAD_Z + CORE_LEN - 3.0, 7.0), (PAD_Z + 18.0, 7.0))):
    piv = empty(f"ArmPivot{k}", loc=(TW_X1, -TW_Y + 0.8, z))
    am = MB()
    am.box((ln / 2, 0, 0), (ln, 1.6, 1.4))
    am.box((ln / 2, 0, -1.0), (ln, 0.25, 0.25))
    am.obj(f"Arm{k}", M_STEEL_TOWER, piv)
    ARMS.append(piv)
# banner with the Tzeke000 emblem on the tower's south face
ban_m, ban_nt = new_mat("Banner")
bb = nd(ban_nt, "ShaderNodeBsdfPrincipled", inp={"Roughness": 0.75})
bi = nd(ban_nt, "ShaderNodeTexImage", image=LOGO, extension="CLIP", interpolation="Cubic")
btc = nd(ban_nt, "ShaderNodeTexCoord")
bmap = nd(ban_nt, "ShaderNodeMapping")
bmap.inputs["Location"].default_value = (-0.1, -0.05, 0)
bmap.inputs["Scale"].default_value = (1.2, 0.68, 1)
lk(ban_nt, btc.outputs["UV"], bmap.inputs["Vector"])
lk(ban_nt, bmap.outputs[0], bi.inputs["Vector"])
lk(ban_nt, mix_col(ban_nt, bi.outputs["Alpha"], (0.02, 0.02, 0.025), bi.outputs["Color"]), bb.inputs["Base Color"])
lk(ban_nt, M(ban_nt, "MULTIPLY", bi.outputs["Alpha"], 0.6), bb.inputs["Emission Strength"])
lk(ban_nt, bi.outputs["Color"], bb.inputs["Emission Color"])
bo = nd(ban_nt, "ShaderNodeOutputMaterial")
lk(ban_nt, bb.outputs[0], bo.inputs["Surface"])
banner = mesh_obj("Banner", [(TW_X0 + 0.3, -TW_Y - 0.6, 52), (TW_X1 - 0.3, -TW_Y - 0.6, 52),
                             (TW_X1 - 0.3, -TW_Y - 0.6, 66), (TW_X0 + 0.3, -TW_Y - 0.6, 66)],
                  [(0, 1, 2, 3)], ban_m, uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
# water tower, lightning masts, floodlight masts
wt = MB()
for k in range(6):
    a = 2 * math.pi * k / 6
    wt.beam((-190 + 8 * math.cos(a), 140 + 8 * math.sin(a), 0), (-190 + 4 * math.cos(a), 140 + 4 * math.sin(a), 70), 0.8)
wt.obj("WaterTowerLegs", M_STEEL_TOWER)
uv_sphere("WaterTowerTank", 11.0, 32, 16, principled("TankWhite", (0.7, 0.7, 0.68), 0.5, 0.2), loc=(-190, 140, 80))
masts = MB()
MAST_POS = [(-70, -70), (70, -70), (-70, 70), (70, 70)]
for (x, y) in MAST_POS:
    masts.beam((x, y, 0), (x, y, 150), 1.1)
masts.obj("LightningMasts", M_STEEL_TOWER)
BEACONS = []
lamp_ob = MB()
for k, (x, y) in enumerate(MAST_POS + [((TW_X0 + TW_X1) / 2, 0)]):
    z = 151 if k < 4 else TW_H + 23
    o = no_shadow(uv_sphere(f"Beacon{k}", 0.6, 12, 6, M_LAMP, loc=(x, y, z)))
    BEACONS.append(o)
    ld = bpy.data.lights.new(f"BeaconL{k}", "POINT")
    ld.color = (1.0, 0.05, 0.02)
    ld.shadow_soft_size = 0.5
    ld.use_shadow = False
    ld.volume_factor = 1.0
    lo = link(bpy.data.objects.new(f"BeaconL{k}", ld))
    lo.location = (x, y, z + 1.2)
    BEACONS.append(lo)
# floodlights: 4 masts with light banks, xenon white spots on the rocket
FLOODS = []
FLOOD_POS = [(-110, -120, 32), (120, -110, 32), (130, 90, 32), (-40, -160, 32)]
fl = MB()
for (x, y, z) in FLOOD_POS:
    fl.beam((x, y, 0), (x, y, z), 0.9)
fl.obj("FloodMasts", M_STEEL_TOWER)
for k, (x, y, z) in enumerate(FLOOD_POS):
    panel = no_shadow(cube_obj(f"FloodPanel{k}", M_LAMP, loc=(x, y, z + 1.5), scale=(2.2, 0.3, 1.2)))
    d = Vector((0, 0, PAD_Z + 38)) - Vector((x, y, z))
    panel.rotation_euler = d.to_track_quat("Y", "Z").to_euler()
    FLOODS.append(panel)
    ld = bpy.data.lights.new(f"Flood{k}", "SPOT")
    ld.spot_size = math.radians(16)
    ld.spot_blend = 0.4
    ld.color = (0.85, 0.9, 1.0)
    ld.shadow_soft_size = 1.5
    ld.volume_factor = 1.0
    lo = link(bpy.data.objects.new(f"Flood{k}", ld))
    lo.location = (x, y, z + 1.5)
    lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    FLOODS.append(lo)
# searchlights sweeping the sky
SEARCH = []
for k, (x, y) in enumerate([(-320, -260), (260, -300), (40, -420)]):
    ld = bpy.data.lights.new(f"Search{k}", "SPOT")
    ld.spot_size = math.radians(2.2)
    ld.spot_blend = 0.15
    ld.color = (0.80, 0.86, 1.0)
    ld.shadow_soft_size = 0.2
    ld.use_shadow = False
    ld.volume_factor = 1.0
    lo = link(bpy.data.objects.new(f"Search{k}", ld))
    lo.location = (x, y, 2.0)
    SEARCH.append(lo)
# distant shore lights along the horizon
rng = np.random.default_rng(5)
dl = MB()
for k in range(420):
    a = rng.uniform(-math.pi * 0.95, -math.pi * 0.05) if k % 3 else rng.uniform(0, math.pi)
    rr = rng.uniform(1800, 5200)
    x, y = rr * math.cos(a), rr * math.sin(a)
    s = rng.uniform(0.6, 2.2)
    dl.box((x, y, s / 2), (s, s, s))
dl_ob = dl.obj("ShoreLights", emissive("ShoreLamp", (1.0, 0.62, 0.28), 60.0))
no_shadow(dl_ob)

FIRE = bpy.data.lights.new("TrenchFire", "POINT")
FIRE.color = blackbody(1900)
FIRE.shadow_soft_size = 12.0
FIRE.volume_factor = 1.0
FIRE_OB = link(bpy.data.objects.new("TrenchFire", FIRE))
FIRE_OB.location = (34.0, 0.0, 9.0)
def billow_tree(name, subdiv=4):
    """Cauliflower cloud: a cluster of noise-displaced spheres inside the unit sphere
    (bottom-heavy), displacement evolving with scene time (stateless). Object scale = size.
    Inputs: Seed, Count, Rmin, Rmax, Amp (displacement), Speed (noise W per second)."""
    t = bpy.data.node_groups.new(name, "GeometryNodeTree")
    t.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    socks = {}
    for nm, ty, dv in (("Seed", "NodeSocketInt", 1), ("Count", "NodeSocketInt", 40), ("Rmin", "NodeSocketFloat", 0.22),
                       ("Rmax", "NodeSocketFloat", 0.5), ("Amp", "NodeSocketFloat", 0.12),
                       ("Speed", "NodeSocketFloat", 0.25)):
        sck = t.interface.new_socket(name=nm, in_out="INPUT", socket_type=ty)
        sck.default_value = dv
        socks[nm] = sck
    t.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = t.nodes.new("NodeGroupInput")
    go = t.nodes.new("NodeGroupOutput")
    pts = t.nodes.new("GeometryNodePoints")
    t.links.new(gi.outputs["Count"], pts.inputs["Count"])

    def rv(vec, lo, hi, salt):
        r = t.nodes.new("FunctionNodeRandomValue")
        r.data_type = "FLOAT_VECTOR" if vec else "FLOAT"
        want = "VECTOR" if vec else "VALUE"
        ins = [x for x in r.inputs if x.enabled and x.type == want]
        if isinstance(lo, (int, float, tuple)):
            ins[0].default_value = lo
        else:
            t.links.new(lo, ins[0])
        if isinstance(hi, (int, float, tuple)):
            ins[1].default_value = hi
        else:
            t.links.new(hi, ins[1])
        ad = t.nodes.new("FunctionNodeIntegerMath")
        ad.operation = "ADD"
        t.links.new(gi.outputs["Seed"], ad.inputs[0])
        ad.inputs[1].default_value = salt
        t.links.new(ad.outputs[0], r.inputs["Seed"])
        return [x for x in r.outputs if x.enabled and x.type == want][0]
    pos = rv(True, (-0.62, -0.62, -0.55), (0.62, 0.62, 0.55), 3)
    sq = t.nodes.new("ShaderNodeVectorMath")                  # bottom-heavy, rounder top
    sq.operation = "MULTIPLY"
    t.links.new(pos, sq.inputs[0])
    sq.inputs[1].default_value = (1.0, 1.0, 1.0)
    sp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], sp.inputs["Geometry"])
    t.links.new(sq.outputs[0], sp.inputs["Position"])
    ico = t.nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 1.0
    ico.inputs["Subdivisions"].default_value = subdiv
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(sp.outputs[0], inst.inputs["Points"])
    t.links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    rad = rv(False, gi.outputs["Rmin"], gi.outputs["Rmax"], 7)
    t.links.new(rad, inst.inputs["Scale"])
    t.links.new(rv(True, (0, 0, 0), (6.3, 6.3, 6.3), 11), inst.inputs["Rotation"])
    real = t.nodes.new("GeometryNodeRealizeInstances")
    t.links.new(inst.outputs[0], real.inputs[0])
    # displacement along the normal: 4D noise, W = time * speed (two octave bands)
    stime = t.nodes.new("GeometryNodeInputSceneTime")
    w = t.nodes.new("ShaderNodeMath")
    w.operation = "MULTIPLY"
    t.links.new(stime.outputs["Seconds"], w.inputs[0])
    t.links.new(gi.outputs["Speed"], w.inputs[1])
    ppos = t.nodes.new("GeometryNodeInputPosition")
    nrm = t.nodes.new("GeometryNodeInputNormal")
    nz = t.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = 3.2
    nz.inputs["Detail"].default_value = 5.0
    nz.inputs["Roughness"].default_value = 0.62
    t.links.new(ppos.outputs[0], nz.inputs["Vector"])
    t.links.new(w.outputs[0], nz.inputs["W"])
    c = t.nodes.new("ShaderNodeMath")
    c.operation = "SUBTRACT"
    t.links.new(nz.outputs["Fac"], c.inputs[0])
    c.inputs[1].default_value = 0.42
    a = t.nodes.new("ShaderNodeMath")
    a.operation = "MULTIPLY"
    t.links.new(c.outputs[0], a.inputs[0])
    t.links.new(gi.outputs["Amp"], a.inputs[1])
    off = t.nodes.new("ShaderNodeVectorMath")
    off.operation = "SCALE"
    t.links.new(nrm.outputs[0], off.inputs[0])
    t.links.new(a.outputs[0], off.inputs["Scale"])
    sp2 = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(real.outputs[0], sp2.inputs["Geometry"])
    t.links.new(off.outputs[0], sp2.inputs["Offset"])
    ss = t.nodes.new("GeometryNodeSetShadeSmooth")
    t.links.new(sp2.outputs[0], ss.inputs["Geometry"])
    t.links.new(ss.outputs[0], go.inputs["Geometry"])
    return t, socks


def billow_material(name, base=(0.80, 0.79, 0.77), emis=(1.0, 0.45, 0.12), fire=False):
    """Steam/smoke surface: diffuse + translucency (backlit glow), lower half glows from the
    fire beneath (keyed 'Glow'), 'Green' = TEA-TEB tint. fire=True: an emissive fireball."""
    m, nt = new_mat(name)
    glow = value_node(nt, "Glow", 0.0)
    green = value_node(nt, "Green", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["Object"], sp.inputs[0])
    low = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY_ADD", sp.outputs[2], 0.7, 0.5), clamp=True), 1.6)
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 4.0, "Detail": 4.0})
    lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
    ecol = mix_col(nt, M(nt, "MULTIPLY", green.outputs[0], 0.95, clamp=True), tuple(emis), (0.3, 1.0, 0.4))
    estr = M(nt, "MULTIPLY", M(nt, "MULTIPLY", glow.outputs[0], low if not fire else 1.0),
             M(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 1.4, 0.3))
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Base Color": tuple(base) + (1,), "Roughness": 1.0})
    lk(nt, ecol, b.inputs["Emission Color"])
    lk(nt, estr, b.inputs["Emission Strength"])
    tl = nd(nt, "ShaderNodeBsdfTranslucent", inp={"Color": tuple(c * 0.8 for c in base) + (1,)})
    mx = nd(nt, "ShaderNodeMixShader")
    mx.inputs[0].default_value = 0.35 if not fire else 0.1
    lk(nt, b.outputs[0], mx.inputs[1])
    lk(nt, tl.outputs[0], mx.inputs[2])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, mx.outputs[0], o.inputs["Surface"])
    return m


BILLOW_T, BILLOW_S = billow_tree("Billow", 4)
BILLOW_T3, BILLOW_S3 = billow_tree("BillowLo", 3)
M_STEAMB = billow_material("SteamBillow")
M_FIREB = billow_material("FireBillow", base=(0.25, 0.2, 0.17), emis=(1.0, 0.55, 0.18), fire=True)


def billow_obj(name, mat, loc=(0, 0, 0), parent=None, seed=1, count=40, rmin=0.22, rmax=0.5, amp=0.12,
               speed=0.25, lo=False):
    ob = mesh_obj(name, [], [], mat, parent, loc)
    md = ob.modifiers.new("Billow", "NODES")
    md.node_group = BILLOW_T3 if lo else BILLOW_T
    sk = BILLOW_S3 if lo else BILLOW_S
    for nm, v in (("Seed", seed), ("Count", count), ("Rmin", rmin), ("Rmax", rmax), ("Amp", amp), ("Speed", speed)):
        getattr(md.properties.inputs, sk[nm].identifier).value = v
    sm = ob.modifiers.new("Mat", "NODES")
    smt = bpy.data.node_groups.get("SetMatOnly")
    if smt is None:
        smt = bpy.data.node_groups.new("SetMatOnly", "GeometryNodeTree")
        smt.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ms = smt.interface.new_socket(name="Material", in_out="INPUT", socket_type="NodeSocketMaterial")
        smt.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        a_, b_ = smt.nodes.new("NodeGroupInput"), smt.nodes.new("NodeGroupOutput")
        st_ = smt.nodes.new("GeometryNodeSetMaterial")
        smt.links.new(a_.outputs["Geometry"], st_.inputs["Geometry"])
        smt.links.new(a_.outputs["Material"], st_.inputs["Material"])
        smt.links.new(st_.outputs[0], b_.inputs["Geometry"])
    sm.node_group = smt
    msock = [it for it in smt.interface.items_tree if it.in_out == "INPUT" and it.name == "Material"][0]
    getattr(sm.properties.inputs, msock.identifier).value = mat
    return ob


# --- exhaust clouds (ignition → liftoff) ------------------------------------------------
STEAM = []
_srng = np.random.default_rng(11)
for k in range(9):
    if k < 5:     # out of the trench to the east, rolling outward
        base = Vector((26 + 22 * k, _srng.uniform(-14, 14), 6))
        drift = Vector((1.0, _srng.uniform(-0.4, 0.4), 0.22))
        r_end = 22 + 6 * k
    else:         # wrapping around the pad
        a = math.pi * (0.5 + (k - 5) * 0.5) + 0.3
        base = Vector((28 * math.cos(a), 28 * math.sin(a), 10))
        drift = Vector((math.cos(a), math.sin(a), 0.35))
        r_end = 28
    o = billow_obj(f"Steam{k}", M_STEAMB, loc=tuple(base), seed=17 * k + 3, count=34 if k < 5 else 26,
                   amp=0.14, speed=0.22)
    o.scale = (0.01, 0.01, 0.01)
    STEAM.append((o, base, drift, r_end, k))
basefire = billow_obj("BaseFire", M_FIREB, loc=(0, 0, 9.0), seed=901, count=30, amp=0.16, speed=0.9, lo=True)
trenchfire = billow_obj("TrenchFire", M_FIREB, loc=(30, 0, 8.0), seed=907, count=26, amp=0.16, speed=0.9, lo=True)
M_TRAIL = volume_material("Trail", (0.88, 0.88, 0.9), "COLUMN", nscale=0.9, detail=5.0,
                          emis_color=(1.0, 0.45, 0.12), aniso=0.2)
trail = lathe("TrailPad", [(1.0, 0.0), (1.0, 1.0)], 24, M_TRAIL, cap=True)
# cold fog rolling off the pad before launch
M_FOG = volume_material("PadFog", (0.9, 0.92, 0.95), "SPHERE", nscale=1.2, detail=4.0, aniso=0.4)
padfog = uv_sphere("PadFog", 1.0, 20, 10, M_FOG, loc=(4, -6, 8.5))
padfog.scale = (26, 26, 5)

# =====================================================================================
# 7. COUNTDOWN CLOCK (7-segment LED board, in front of the A2 camera)
# =====================================================================================
FONT = bpy.data.fonts.load(ARGS.font) if os.path.exists(ARGS.font) else None
CLK = empty("CountdownBoard")
brd = MB()
brd.box((0, 0.06, 0.0), (5.6, 0.12, 2.25))
brd.obj("BoardPanel", principled("BoardBlack", (0.012, 0.012, 0.014), 0.55), CLK)
frm = MB()
for x0, x1, z0, z1 in ((-2.85, 2.85, 1.12, 1.2), (-2.85, 2.85, -1.2, -1.12), (-2.85, -2.77, -1.2, 1.2),
                       (2.77, 2.85, -1.2, 1.2)):
    frm.box(((x0 + x1) / 2, 0.0, (z0 + z1) / 2), (x1 - x0, 0.2, z1 - z0))
for x in (-2.1, 2.1):
    frm.box((x, 0.2, -2.6), (0.18, 0.18, 2.9))
frm.obj("BoardFrame", principled("BoardFrame", (0.35, 0.35, 0.36), 0.35, 1.0), CLK)
SEG_DEF = {"a": (0, 1, 1), "b": (1, 0.5, 0), "c": (1, -0.5, 0), "d": (0, -1, 1), "e": (-1, -0.5, 0),
           "f": (-1, 0.5, 0), "g": (0, 0, 1)}
DIGIT_SEGS = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd", 6: "afgedc",
              7: "abc", 8: "abcdefg", 9: "abcdfg"}
DH, DW, ST = 0.95, 0.50, 0.11          # digit height, width, stroke


def seg_poly(cx, cz, horiz, L):
    t = ST / 2
    if horiz:
        pts = [(-L / 2, 0), (-L / 2 + t, t), (L / 2 - t, t), (L / 2, 0), (L / 2 - t, -t), (-L / 2 + t, -t)]
        return [(cx + x, -0.005, cz + z) for x, z in pts]
    pts = [(0, -L / 2), (t, -L / 2 + t), (t, L / 2 - t), (0, L / 2), (-t, L / 2 - t), (-t, -L / 2 + t)]
    return [(cx + x, -0.005, cz + z) for x, z in pts]


LED_SEGS = {}              # (slot, seg) -> object
DIG_X = [-1.15, -0.47, 0.42, 1.10, 1.99, 2.67]
for slot, x0 in enumerate(DIG_X):
    for s, (sx, sz, h) in SEG_DEF.items():
        L = (DW - ST * 0.2) if h else (DH / 2 - ST * 0.2)
        pts = seg_poly(x0 + sx * DW / 2, -0.25 + sz * DH / 2, h, L)
        o = mesh_obj(f"Seg{slot}{s}", pts, [tuple(range(6))[::-1]], M_LED, CLK)
        LED_SEGS[(slot, s)] = o
COLON = []
for x in (0.0 + 0.0, 1.55):
    xx = (DIG_X[1] + DIG_X[2]) / 2 if x == 0 else (DIG_X[3] + DIG_X[4]) / 2
    for z in (-0.25 + 0.2, -0.25 - 0.2):
        c = 0.06
        COLON.append(mesh_obj(f"Colon{len(COLON)}", [(xx - c, -0.005, z - c), (xx + c, -0.005, z - c),
                                                      (xx + c, -0.005, z + c), (xx - c, -0.005, z + c)],
                              [(3, 2, 1, 0)], M_LED, CLK))
SIGN_H = mesh_obj("SignH", seg_poly(-1.86, -0.25, True, 0.50), [tuple(range(6))[::-1]], M_LED, CLK)
SIGN_V = mesh_obj("SignV", seg_poly(-1.88, -0.25, False, 0.42), [tuple(range(6))[::-1]], M_LED, CLK)


def text_obj(name, body, size, loc, mat, parent, align="CENTER", extrude=0.0):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.extrude = extrude
    if FONT is not None:
        cu.font = FONT
    o = bpy.data.objects.new(name, cu)
    o.location = loc
    o.rotation_euler = (math.pi / 2, 0, 0)
    cu.materials.append(mat)
    return link(o, parent)


text_obj("ClockT", "T", 1.15, (-2.42, -0.01, -0.62), emissive("LEDAmber", (1.0, 0.42, 0.05), 9.0), CLK)
text_obj("ClockTitle", "HIT A BUMP", 0.62, (0.55, -0.01, 0.62), emissive("LEDWhite", (1.0, 0.95, 0.88), 6.0), CLK)
text_obj("ClockSub", "MISSION", 0.26, (-2.0, -0.01, 0.62), emissive("LEDWhiteDim", (0.9, 0.85, 0.8), 2.5), CLK)

# =====================================================================================
# 8. WORLD: physical sky (5.x multiple scattering) + Earth surface/city lights by ray-sphere
#    in the world shader + stars + a pad haze volume.
# =====================================================================================
WORLD = bpy.data.worlds.new("World")
SC.world = WORLD
wn = WORLD.node_tree
wn.nodes.clear()
sky = nd(wn, "ShaderNodeTexSky", name="Sky", sky_type="MULTIPLE_SCATTERING")
sky.sun_rotation = math.radians(0.0)
sky.ground_albedo = 0.3
sky.sun_size = math.radians(0.545)
w_gain = value_node(wn, "SkyGain", 1.0)
w_h = value_node(wn, "EarthH", 1.0157)          # (R+h)/R
w_ov = value_node(wn, "EarthOverlay", 0.0)
w_scroll = value_node(wn, "EarthScroll", 0.0)
w_stars = value_node(wn, "Stars", 1.0)
w_haze = value_node(wn, "Haze", 0.0)
tcw = nd(wn, "ShaderNodeTexCoord")
dvec = V(wn, "NORMALIZE", tcw.outputs["Generated"])
dsep = nd(wn, "ShaderNodeSeparateXYZ")
lk(wn, dvec, dsep.inputs[0])
dz = dsep.outputs[2]
H = w_h.outputs[0]
d2 = M(wn, "MULTIPLY", M(wn, "MULTIPLY", H, H), M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", dz, dz)))
hit = M(wn, "MULTIPLY", M(wn, "LESS_THAN", d2, 1.0), M(wn, "LESS_THAN", dz, 0.0))
thc = M(wn, "SQRT", M(wn, "MAXIMUM", M(wn, "SUBTRACT", 1.0, d2), 0.0))
tt = M(wn, "SUBTRACT", M(wn, "MULTIPLY", M(wn, "MULTIPLY", H, dz), -1.0), thc)
hitp = V(wn, "ADD", V(wn, "SCALE", dvec, scale=tt), (0.0, 0.0, 0.0))
hc = nd(wn, "ShaderNodeCombineXYZ")
lk(wn, H, hc.inputs[2])
nrm = V(wn, "ADD", hitp, hc.outputs[0])                          # unit normal at the hit
rot = nd(wn, "ShaderNodeVectorRotate", rotation_type="X_AXIS")
lk(wn, nrm, rot.inputs["Vector"])
lk(wn, w_scroll.outputs[0], rot.inputs["Angle"])
en = rot.outputs[0]
cont = nd(wn, "ShaderNodeTexNoise", inp={"Scale": 2.4, "Detail": 12.0, "Roughness": 0.62, "Lacunarity": 2.1})
lk(wn, en, cont.inputs["Vector"])
landm = M(wn, "MULTIPLY", M(wn, "SUBTRACT", cont.outputs["Fac"], 0.52), 9.0, clamp=True)
cl = nd(wn, "ShaderNodeTexNoise", inp={"Scale": 14.0, "Detail": 12.0, "Roughness": 0.66, "Distortion": 0.7})
lk(wn, V(wn, "MULTIPLY", en, (1.0, 1.0, 2.2)), cl.inputs["Vector"])
cloudm = M(wn, "MULTIPLY", M(wn, "SUBTRACT", cl.outputs["Fac"], 0.47), 3.2, clamp=True)
alb = M(wn, "ADD", 0.06, M(wn, "MULTIPLY", landm, 0.12))
alb = M(wn, "ADD", M(wn, "MULTIPLY", alb, M(wn, "SUBTRACT", 1.0, cloudm)), M(wn, "MULTIPLY", cloudm, 0.75))
fac = M(wn, "DIVIDE", alb, 0.30)
mu = M(wn, "MULTIPLY", V(wn, "DOT_PRODUCT", nrm, dvec), -1.0)
limb = M(wn, "MULTIPLY", M(wn, "POWER", M(wn, "MAXIMUM", mu, 0.0), 0.6), 1.0, clamp=True)
wgt = M(wn, "MULTIPLY", M(wn, "MULTIPLY", hit, limb), w_ov.outputs[0])
facf = M(wn, "ADD", 1.0, M(wn, "MULTIPLY", wgt, M(wn, "SUBTRACT", fac, 1.0)))
skyc = sky.outputs["Color"]
mulc = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
mulc.inputs[0].default_value = 1.0
lk(wn, skyc, mulc.inputs[6])
fcc = nd(wn, "ShaderNodeCombineColor")
for k in range(3):
    lk(wn, facf, fcc.inputs[k])
lk(wn, fcc.outputs[0], mulc.inputs[7])
earth_col = mulc.outputs[2]
# city lights on the night side: where the sky's ground colour is dark
lum = nd(wn, "ShaderNodeRGBToBW")
lk(wn, skyc, lum.inputs[0])
night = M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", lum.outputs[0], 40.0), clamp=True)
city = nd(wn, "ShaderNodeTexVoronoi", inp={"Scale": 900.0, "Randomness": 1.0})
lk(wn, en, city.inputs["Vector"])
cdot = M(wn, "POWER", M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", city.outputs["Distance"], 7.0), clamp=True), 4.0)
cmask = M(wn, "MULTIPLY", M(wn, "MULTIPLY", cdot, landm), M(wn, "SUBTRACT", 1.0, cloudm))
cl_st = M(wn, "MULTIPLY", M(wn, "MULTIPLY", M(wn, "MULTIPLY", cmask, night), wgt), 0.06)
addc = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, cl_st, addc.inputs[0])
lk(wn, earth_col, addc.inputs[6])
addc.inputs[7].default_value = (1.0, 0.6, 0.25, 1)
earth_col = addc.outputs[2]
# stars (only where the sky is dark and the planet isn't)
sv = nd(wn, "ShaderNodeTexVoronoi", inp={"Scale": 420.0, "Randomness": 1.0})
lk(wn, dvec, sv.inputs["Vector"])
sb = nd(wn, "ShaderNodeSeparateColor")
lk(wn, sv.outputs["Color"], sb.inputs[0])
star = M(wn, "POWER", M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", sv.outputs["Distance"], 9.0), clamp=True), 6.0)
star = M(wn, "MULTIPLY", star, M(wn, "POWER", sb.outputs[0], 6.0))
skydark = M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", lum.outputs[0], 12.0), clamp=True)
star = M(wn, "MULTIPLY", M(wn, "MULTIPLY", star, skydark), M(wn, "SUBTRACT", 1.0, hit))
star = M(wn, "MULTIPLY", M(wn, "MULTIPLY", star, w_stars.outputs[0]), 0.35)
adds = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, star, adds.inputs[0])
lk(wn, earth_col, adds.inputs[6])
adds.inputs[7].default_value = (0.9, 0.93, 1.0, 1)
w_sunel = value_node(wn, "SunEl", 0.0)
w_sung = value_node(wn, "SunDisc", 0.0)
sdir = nd(wn, "ShaderNodeCombineXYZ")
lk(wn, M(wn, "COSINE", w_sunel.outputs[0]), sdir.inputs[1])
lk(wn, M(wn, "SINE", w_sunel.outputs[0]), sdir.inputs[2])
cosang = V(wn, "DOT_PRODUCT", dvec, sdir.outputs[0])
disc = nd(wn, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP",
          inp={"From Min": math.cos(math.radians(0.33)), "From Max": math.cos(math.radians(0.24))})
lk(wn, cosang, disc.inputs["Value"])
cor = M(wn, "ADD", M(wn, "MULTIPLY", M(wn, "POWER", M(wn, "MAXIMUM", cosang, 0.0), 900.0), 3.0),
        M(wn, "MULTIPLY", M(wn, "POWER", M(wn, "MAXIMUM", cosang, 0.0), 45.0), 0.12))
sun_e = M(wn, "MULTIPLY", M(wn, "ADD", M(wn, "MULTIPLY", disc.outputs["Result"], 400.0), cor),
          M(wn, "MULTIPLY", w_sung.outputs[0], M(wn, "SUBTRACT", 1.0, hit)))
addsun = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, sun_e, addsun.inputs[0])
lk(wn, adds.outputs[2], addsun.inputs[6])
addsun.inputs[7].default_value = (1.0, 0.93, 0.82, 1)
bgw = nd(wn, "ShaderNodeBackground")
lk(wn, addsun.outputs[2], bgw.inputs["Color"])
lk(wn, w_gain.outputs[0], bgw.inputs["Strength"])
haze = nd(wn, "ShaderNodeVolumePrincipled", inp={"Color": (0.75, 0.8, 0.9, 1), "Anisotropy": 0.55})
lk(wn, w_haze.outputs[0], haze.inputs["Density"])
wo = nd(wn, "ShaderNodeOutputWorld")
lk(wn, bgw.outputs[0], wo.inputs["Surface"])
lk(wn, haze.outputs[0], wo.inputs["Volume"])

SUN = bpy.data.lights.new("Sun", "SUN")
SUN.angle = math.radians(0.53)
SUN.color = (1.0, 0.86, 0.70)
SUN_OB = link(bpy.data.objects.new("Sun", SUN))
MOON = bpy.data.lights.new("Moonfill", "SUN")
MOON.angle = math.radians(2.0)
MOON.color = (0.55, 0.65, 1.0)
MOON_OB = link(bpy.data.objects.new("Moonfill", MOON))
MOON_OB.rotation_euler = (math.radians(55), 0, math.radians(-150))

M_HAZE = volume_material("PadHaze", (0.8, 0.85, 0.95), "UNIFORM", nscale=3.0, detail=2.0, aniso=0.6)
haze_box = cube_obj("PadHaze", M_HAZE, loc=(0, 0, 450), scale=(3200, 3200, 450))
# --- the cloud deck (calm 2 + max-Q backdrop) ------------------------------------------
M_DECK = volume_material("CloudDeck", (0.95, 0.95, 0.97), "SLAB", nscale=2.6, detail=7.0,
                         emis_color=(1.0, 0.45, 0.15), aniso=0.5)
deck_c = cube_obj("CloudDeck", M_DECK, loc=(0, 0, -200), scale=(2600, 2600, 160))
M_TRAILF = volume_material("TrailFlight", (0.85, 0.85, 0.88), "COLUMN", nscale=0.9, detail=5.0,
                           emis_color=(1.0, 0.45, 0.12), aniso=0.2)
trail_f = lathe("TrailFlight", [(1.0, 0.0), (1.0, 1.0)], 24, M_TRAILF, cap=True)

# =====================================================================================
# 9. CAMERA + SHOTS
# =====================================================================================
CAM_D = bpy.data.cameras.new("Cam")
CAM_D.sensor_width = 36.0
CAM_D.clip_start = 0.3
CAM_D.clip_end = 60000.0
CAM_D.dof.use_dof = True
CAM = link(bpy.data.objects.new("Cam", CAM_D))
SC.camera = CAM

# rocket attitude per set: pitch toward +Y (downrange)
PITCH = np.zeros(N)
PITCH[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [7, 18, 26])
PITCH[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, SES], [30, 52, 66])
PITCH[SET == 3] = np.interp(IDX[SET == 3], [SES, N], [74, 82])
PITCH[(SET == 0)] = np.clip((ALT[SET == 0] - 120) / 300, 0, 1) * 3.0          # roll program-ish
RIG_POS = np.zeros((N, 3))
RIG_POS[:, 2] = np.where(SET == 0, PAD_Z + ALT, 0.0)
# wobble/vibration of the vehicle (tiny) on kicks
RIG_POS[:, 0] += 0.04 * KICK * on_core


def rig_matrix(i):
    return Matrix.Translation(Vector(RIG_POS[i])) @ Matrix.Rotation(math.radians(-PITCH[i]), 4, "X")


def wl(i, p):
    """rocket-local → world."""
    return rig_matrix(i) @ Vector(p)


def shake_vec(i, amp, salt=0):
    """Kick jolts (direction hashed per kick) + a continuous rumble (sum of sines)."""
    k = KIDX[i]
    j = Vector((hash01(k, 1, salt) - 0.5, hash01(k, 2, salt) - 0.5, hash01(k, 3, salt) - 0.5)) * 2 * KICK[i]
    x = i + 977 * salt
    r = Vector((math.sin(x * 1.71 + 0.3) + 0.5 * math.sin(x * 3.13), math.sin(x * 1.37 + 1.1) +
                0.5 * math.sin(x * 2.89 + 2.0), math.sin(x * 1.93 + 2.2)))
    return (j * 1.0 + r * 0.18) * amp


SHOT_FN = {}


def shot(name):
    def deco(fn):
        SHOT_FN[name] = fn
        return fn
    return deco


CLOCK_CAM = Vector((34.0, -150.0, 2.0))


@shot("A1_wide")
def _(i, u, n):
    t = u / n
    s = float(smoothstep(t))
    cam = Vector((150 - 40 * s, -560 + 90 * s, 18 - 4 * s))
    return dict(cam=cam, tgt=Vector((-6, 0, 46 + 4 * s)), lens=45 + 8 * s, fstop=11, shake=0.0)


@shot("A2_clock")
def _(i, u, n):
    t = u / n
    cam = CLOCK_CAM + Vector((-3.0 * t, 1.5 * t, 0.15 * t))
    tgt = Vector((2.0, 0.0, 34.0 - 4 * t))
    rack = float(smoothstep((t - 0.45) / 0.3))          # focus: clock → rocket
    fdist = 7.6 * (1 - rack) + (Vector((0, 0, 38)) - cam).length * rack
    return dict(cam=cam, tgt=tgt, lens=30, fstop=4.0, focus=fdist, shake=0.0)


@shot("A3_logo")
def _(i, u, n):
    t = u / n
    a = math.radians(-98 + 26 * t)
    cam = Vector((42 * math.cos(a), 42 * math.sin(a), PAD_Z + 31 + 3 * t))
    tgt = Vector((0, 0, PAD_Z + LOGO_Z0 + 2.3 + 1.5 * t))
    return dict(cam=cam, tgt=tgt, lens=50, fstop=5.6, shake=0.0)


@shot("A4_low")
def _(i, u, n):
    t = u / n
    creep = float(PRE[i])
    cam = Vector((26 - 6 * t - 4 * creep, -34 + 7 * t + 5 * creep, 9.2 + 0.4 * t))
    tgt = Vector((-1.0, 0, PAD_Z + 26 + 6 * t))
    return dict(cam=cam, tgt=tgt, lens=20 + 2 * creep, fstop=8, shake=0.05 * PRE[i])


@shot("B1_ign")
def _(i, u, n):
    t = u / n
    cam = Vector((104 - 8 * t, -128, 12.0))
    tgt = Vector((12.0, 0.0, PAD_Z + 24 + ALT[i] * 0.8))
    return dict(cam=cam, tgt=tgt, lens=30 - 5 * t, fstop=11, shake=0.9)


@shot("B2_tower")
def _(i, u, n):
    cam = Vector((-30.0, -38.0, 98.0))
    tgt = wl(i, (0, 0, 6.0))
    tgt.z = max(tgt.z, 86.0)
    return dict(cam=cam, tgt=tgt, lens=24, fstop=8, shake=0.7)


@shot("B3_track")
def _(i, u, n):
    cam = Vector((420.0, -1250.0, 6.0))
    tgt = wl(i, (0, 0, 10.0))
    return dict(cam=cam, tgt=tgt, lens=70 + 30 * u / n, fstop=11, shake=0.35)


@shot("B4_tail")
def _(i, u, n):
    t = u / n
    cam = wl(i, (34.0 - 6 * t, -62.0, -36.0 + 10 * t))
    tgt = wl(i, (0, 0, 8.0))
    return dict(cam=cam, tgt=tgt, lens=30, fstop=11, shake=0.5)


DECK_TOP = np.full(N, -5000.0)
_c = (IDX >= C_CUTS[0]) & (IDX < MAXQ)
DECK_TOP[_c] = 70.0 - (IDX[_c] - C_CUTS[0]) / FPS * 32.0
_d = (IDX >= MAXQ) & (IDX < BSEP)
DECK_TOP[_d] = -900.0 - (IDX[_d] - MAXQ) / FPS * 15.0


@shot("C1_cloud")
def _(i, u, n):
    t = u / n
    cam = Vector((150.0, -330.0, DECK_TOP[i] + 30.0 + 40 * t))
    tgt = wl(i, (0, 0, 20.0))
    return dict(cam=cam, tgt=tgt, lens=35, fstop=11, shake=0.1)


@shot("C2_side")
def _(i, u, n):
    t = u / n
    creep = float(PRE[i])
    a = math.radians(-70 + 25 * t)
    d = 120 - 40 * creep
    cam = wl(i, (d * math.cos(a), d * math.sin(a), 20 - 10 * t))
    tgt = wl(i, (0, 0, 30.0))
    return dict(cam=cam, tgt=tgt, lens=50, fstop=8, shake=0.08 + 0.1 * creep)


@shot("D1_maxq")
def _(i, u, n):
    t = u / n
    cam = wl(i, (13.0 - 4 * t, -27.0, CORE_LEN + 9 - 6 * t))
    tgt = wl(i, (0, 0, CORE_LEN + 2))
    return dict(cam=cam, tgt=tgt, lens=26, fstop=5.6, shake=1.0)


@shot("D2_under")
def _(i, u, n):
    t = u / n
    cam = wl(i, (-11.0, -17.0 + 3 * t, -8.0))
    tgt = wl(i, (0, 0, 40.0))
    return dict(cam=cam, tgt=tgt, lens=18, fstop=11, shake=0.9)


@shot("D3_wide")
def _(i, u, n):
    t = u / n
    cam = wl(i, (260.0, -520.0, 140.0 - 60 * t))
    tgt = wl(i, (0, 0, 10.0))
    return dict(cam=cam, tgt=tgt, lens=55, fstop=11, shake=0.4)


@shot("E_calm")
def _(i, u, n):
    t = u / n
    cam = wl(i, (70.0, -95.0, -40.0 - 10 * t))
    tgt = wl(i, (0, 0, 0.0))
    return dict(cam=cam, tgt=tgt, lens=28, fstop=11, shake=0.08)


@shot("F1_sep")
def _(i, u, n):
    t = u / n
    cam = wl(i, (8.0, -84.0 - 12 * t, 26.0))
    tgt = wl(i, (0, 0, 22.0))
    return dict(cam=cam, tgt=tgt, lens=30, fstop=11, shake=0.8)


@shot("F2_tail")
def _(i, u, n):
    tb_ = max(i - BSEP, 0) / FPS
    bpos = (-(4.0 + 2.2 * tb_ + 0.5 * tb_ ** 2), 0.0, -(3.0 * tb_ ** 2) - 2.0 * tb_ + 22.0)
    cam = wl(i, (-9.0, -16.0, 30.0))
    tgt = wl(i, bpos) * 0.8 + wl(i, (0, 0, 0)) * 0.2
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.35)


@shot("F3_wide")
def _(i, u, n):
    t = u / n
    cam = wl(i, (150.0, -230.0, 10.0 + 20 * t))
    tgt = wl(i, (0, 0, 0.0))
    return dict(cam=cam, tgt=tgt, lens=42, fstop=11, shake=0.35)


@shot("G_calm")
def _(i, u, n):
    t = u / n
    cam = wl(i, (55.0 - 15 * t, -85.0, 5.0))
    tgt = wl(i, (0, 6, 20.0))
    return dict(cam=cam, tgt=tgt, lens=35, fstop=11, shake=0.05)


@shot("H1_fair")
def _(i, u, n):
    t = u / n
    cam = wl(i, (9.0, -19.0 - 6 * t, FAIR_Z + 9.0))
    tgt = wl(i, (0, 0, FAIR_Z + 4.0))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=8, shake=0.7)


@shot("H2_side")
def _(i, u, n):
    t = u / n
    cam = wl(i, (30.0, -48.0 + 8 * t, FAIR_Z - 4))
    tgt = wl(i, (0, 0, FAIR_Z - 4))
    return dict(cam=cam, tgt=tgt, lens=35, fstop=11, shake=0.5)


@shot("I_meco")
def _(i, u, n):
    t = u / n
    cam = wl(i, (16.0, -34.0, CORE_LEN + 4 - 4 * t))
    tgt = wl(i, (0, 0, CORE_LEN + 1 - 6 * t))
    return dict(cam=cam, tgt=tgt, lens=30, fstop=11, shake=0.0)


@shot("J1_limb")
def _(i, u, n):
    t = u / n
    cam = wl(i, (-23.0 + 3 * t, -6.0, 10.0 + 4 * t))
    tgt = wl(i, (0.0, 1.0, CORE_LEN + 80.0))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.25)


@shot("J2_side")
def _(i, u, n):
    t = u / n
    cam = wl(i, (24.0 - 6 * t, -10.0, CORE_LEN + 7.0))
    tgt = wl(i, (0, 0, CORE_LEN + 6.0))
    return dict(cam=cam, tgt=tgt, lens=28, fstop=8, shake=0.25)


@shot("J3_wide")
def _(i, u, n):
    t = u / n
    s = float(ease_out(t, 2))
    cam = wl(i, (40.0 + 60 * s, -20.0 - 40 * s, CORE_LEN - 10.0 - 60 * s))
    tgt = wl(i, (0, 0, CORE_LEN + 2.0))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.2)


@shot("K_outro")
def _(i, u, n):
    t = u / n
    cam = wl(i, (90.0 + 20 * t, -60.0, CORE_LEN - 70.0))
    tgt = wl(i, (0, 0, CORE_LEN + 10.0 + 10 * t))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.0)


# --- evaluate the shot list per frame ---------------------------------------------------
cam_loc = np.zeros((N, 3))
cam_rot = np.zeros((N, 3))
cam_lens = np.zeros(N)
cam_focus = np.zeros(N)
cam_fstop = np.zeros(N)
SHOT_BOUNDS = [s for s, _ in SHOTS] + [N]
prev_eul = None
for k, (s0, name) in enumerate(SHOTS):
    s1 = SHOT_BOUNDS[k + 1]
    fn = SHOT_FN[name]
    for i in range(s0, s1):
        d = fn(i, i - s0, max(s1 - s0 - 1, 1))
        thr = max(on_core[i], on_boost[i], on_s2[i])
        amp = d.get("shake", 0.0) * (0.15 * thr + 0.85 * KICK[i] * HOT[i] + 0.6 * IGN_FLASH[i])
        if SET[i] == 3:
            amp *= 0.5
        cam = d["cam"] + shake_vec(i, amp * 0.35)
        if BIG[i] > 0:                                        # big drop: push in on every bar downbeat
            cam = cam + (d["tgt"] - cam) * (0.07 * ONE[i])
        tgt = d["tgt"] + shake_vec(i, amp * 0.9, 1)
        fwd = (tgt - cam).normalized()
        up = Vector(d.get("up", (0, 0, 1)))
        q = fwd.to_track_quat("-Z", "Y")
        if "up" in d:
            right = fwd.cross(up).normalized()
            up2 = right.cross(fwd).normalized()
            q = Matrix((right, up2, -fwd)).transposed().to_quaternion()
        e = q.to_euler("XYZ", prev_eul) if (prev_eul is not None and i != s0) else q.to_euler("XYZ")
        prev_eul = e
        cam_loc[i] = cam
        cam_rot[i] = (e.x, e.y, e.z)
        cam_lens[i] = d["lens"]
        cam_fstop[i] = d["fstop"]
        cam_focus[i] = d.get("focus", (tgt - cam).length)
bake_vec(CAM, "location", cam_loc)
bake_vec(CAM, "rotation_euler", cam_rot)
bake(CAM_D, "lens", cam_lens)
bake(CAM_D, "dof.focus_distance", cam_focus)
bake(CAM_D, "dof.aperture_fstop", cam_fstop)

# countdown board placed in front of the A2 camera, angled toward it
_a2 = [s for s, n in SHOTS if n == "A2_clock"][0]
_cpos = Vector(cam_loc[_a2])
_cdir = Vector((2.0, 0.0, 34.0)) - _cpos
_cdir.z = 0
_cdir.normalize()
_left = Vector((-_cdir.y, _cdir.x, 0))
CLK.location = _cpos + _cdir * 7.4 + _left * 2.35 + Vector((0, 0, 1.25))
_face = (-_cdir).copy()
CLK.rotation_euler = (0, 0, math.atan2(_face.x, -_face.y) - math.radians(24))   # local -Y → camera

# =====================================================================================
# 10. ANIMATION BAKES (all music reactions live here)
# =====================================================================================
# --- sets / visibility ---
pad_objs = ["Ground", "PadDeck", "LaunchMount", "Tower", "Banner", "WaterTowerLegs", "WaterTowerTank",
            "LightningMasts", "FloodMasts", "ShoreLights", "TrailPad", "PadFog", "CountdownBoard"]
for o in bpy.data.objects:
    if o.name.startswith(("Arm", "Beacon", "Flood", "Search", "Steam", "Seg", "Colon", "Sign", "Board",
                          "Clock")) or o.name in pad_objs:
        show(o, SET == 0)
show(bpy.data.objects["CountdownBoard"], IDX < IGN + 2 * FPS)
for o in bpy.data.objects:
    if o.parent == CLK:
        show(o, IDX < IGN + 2 * FPS)
show(SIGN_V, (IDX >= IGN) & (IDX < IGN + 2 * FPS))                # "+" only after T-0 (no z-fight)
show(deck_c, (SET == 1) & (DECK_TOP > -3000))
show(trail_f, SET == 1)
show(bpy.data.objects["TrailPad"], (SET == 0) & (IDX >= LIFT))
for st in STEAM:
    show(st[0], (SET == 0) & (IDX >= IGN - 1))
for v in VENTS:
    show(v, IDX < LIFT + 20)
show(padfog, IDX < IGN + 30)
show(vcone, (IDX >= MAXQ - 2) & (IDX < seg_after(MAXQ)))
show(shock, np.zeros(N, bool))          # ring read as a hard ribbon at 1080p; the boom is the cone burst
s1_vis = IDX < SES + 4 * FPS
for o in [o for o in bpy.data.objects if o.parent in (S1,)]:
    show(o, s1_vis)
for bem in boosters:
    for o in [o for o in bpy.data.objects if o.parent == bem]:
        show(o, IDX < BSEP + 8 * FPS)
for o in (FL, FR):
    for c in o.children:
        show(c, IDX < FAIRSEP + 6 * FPS)
for nm in ("PayloadBus", "PayloadWings", "PayloadDish"):
    show(bpy.data.objects[nm], IDX >= FAIRSEP - 1)
show(debris, SAGE < 30)
bake(MOON, "energy", np.where(SET == 0, 0.05, np.where(SET == 1, 0.25, np.where(SET == 2, 1.6, 0.05))))
moon_rx = np.where(SET >= 2, math.radians(30) - math.pi / 2, math.radians(55))
moon_rz = np.where(SET >= 2, 0.0, math.radians(-150))
bake_vec(MOON_OB, "rotation_euler", np.stack([moon_rx, np.zeros(N), moon_rz], 1), const=True)
bake_vec(MOON, "color", np.stack([np.where(SET >= 2, 1.0, 0.55), np.where(SET >= 2, 0.62, 0.65),
                                  np.where(SET >= 2, 0.55, 1.0)], 1), const=True)

# --- rig + stage motions ---
bake_vec(RIG, "location", RIG_POS)
bake_vec(RIG, "rotation_euler", np.stack([np.radians(-PITCH), np.zeros(N), np.zeros(N)], 1))
# boosters: peel outward + tumble after separation (stateless closed form in seconds)
tb = np.maximum(IDX - BSEP, 0) / FPS
for side, bem in ((-1, BL), (1, BRt)):
    ox = side * (4.0 + 2.2 * tb + 0.5 * tb ** 2)
    oz = -(3.0 * tb ** 2) - 2.0 * tb
    ry = side * np.radians(5.5 * tb ** 1.3)
    loc = np.stack([ox, np.zeros(N), oz], 1)
    bake_vec(bem, "location", loc)
    bake_vec(bem, "rotation_euler", np.stack([np.zeros(N), ry, np.zeros(N)], 1))
# fairing halves: hinge open then fly off
tf = np.maximum(IDX - FAIRSEP, 0) / FPS
for side, fo in ((1, FL), (-1, FR)):
    ang = -side * np.radians(np.minimum(tf * 55, 120) + 8 * np.maximum(tf - 2.2, 0) ** 1.5)
    y = side * (2.6 + 1.2 * tf + 0.6 * tf ** 2)
    z = FAIR_Z - 1.4 * tf ** 2
    bake_vec(fo, "location", np.stack([np.zeros(N), y, z], 1))
    bake_vec(fo, "rotation_euler", np.stack([ang, np.zeros(N), np.zeros(N)], 1))
# stage 1 falls back after MECO (stage sep a beat after MECO)
SEP = MECO + int(round(BAR / 2))
ts = np.maximum(IDX - SEP, 0) / FPS
bake_vec(S1, "location", np.stack([np.zeros(N), -0.25 * ts ** 2, -(1.4 * ts + 0.9 * ts ** 2)], 1))
bake_vec(S1, "rotation_euler", np.stack([np.radians(1.4 * ts ** 1.2), np.zeros(N), np.zeros(N)], 1))
# swing arms retract one bar before ignition (anticipation)
ARM_T = IGN - int(round(BAR))
arm_a = np.radians(-100) * ease_out(ramp(IDX, ARM_T, ARM_T + 40), 2.5)
for k, piv in enumerate(ARMS):
    bake_vec(piv, "rotation_euler", np.stack([np.zeros(N), np.zeros(N), arm_a * (1 if k != 1 else 0.85)], 1))

# --- plumes ---
alt_expand = np.ones(N)
alt_expand[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [1.12, 1.25, 1.45])
alt_expand[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, SES], [1.55, 1.95, 2.3])
phase = np.cumsum(0.9 + 0.6 * HOT) / FPS                              # flow scroll
SET_GAIN = np.where(SET == 0, 0.35, np.where(SET == 1, 0.5, 0.45))
for o, kind in PLUMES:
    on = {"core": on_core, "boost": on_boost, "s2": on_s2}[kind]
    ex = alt_expand if kind != "s2" else np.ones(N)
    vol = "_vol" in o.name
    green = IGN_FLASH if kind != "s2" else np.where((IDX >= SES) & (IDX < SES + 4), 1.0 - (IDX - SES) / 4.0, 0.0)
    lng = on * (1.0 + 0.16 * KICK * HOT + 0.12 * BIG * ONE)
    if kind == "s2":
        pulse2 = 1.0 + 0.45 * BIG * ONE + 0.25 * KICK * HOT
        gain = on * pulse2 * (28.0 if not vol else 0.08)
        sx = np.full(N, 17.0) * (0.4 + 0.6 * on) if vol else np.ones(N) * (0.6 + 0.4 * on)
        sz = np.maximum((np.full(N, 150.0) if vol else np.ones(N)) * lng, 1e-3)
    elif vol:
        gain = on * PULSE * np.where(SET == 0, 1.1, np.where(SET == 1, 0.5, 0.2)) / ex ** 1.2
        sx = PLUME_W * ex * (0.5 + 0.5 * on)
        sz = np.maximum(PLUME_L * lng * ex ** 0.7, 1e-3)
    else:
        gain = on * PULSE * 60.0 * SET_GAIN
        sx = np.maximum(ex ** 0.5 * (0.55 + 0.45 * on), 1e-3)
        sz = np.maximum(lng * ex ** 0.5, 1e-3)
    gain = gain * (1.0 + 2.5 * IGN_FLASH)
    set_color_keys(o, np.stack([gain, green, phase], 1), const=False)
    bake_vec(o, "scale", np.stack([sx, sx, sz], 1))
    show(o, on > 0.001)
for lo, tag in PLIGHTS:
    on = {"core": on_core, "bl": on_boost, "br": on_boost, "s2": on_s2}[tag]
    e = on * PULSE * (1.0 + 2.0 * IGN_FLASH)
    if tag == "s2":
        bake(lo.data, "energy", on * 6.0e4 * (1 + 0.5 * BIG * ONE))
    else:
        bake(lo.data, "energy", e * np.where(SET == 0, 1.5e5, 2.5e5) * (1 + 0.5 * BASS * HOT))
    show(lo, on > 0.001)
bake(BOUNCE, "energy", on_core * PULSE * np.where(SET == 0, 6.0e4, np.where(SET == 1, 3.0e4, 1.5e4)))
show(BOUNCE_OB, on_core > 0.001)
# nozzle glow
set_color_keys(core_eng, np.stack([on_core, on_core, on_core], 1), const=False)
for bem in boosters:
    for c in bem.children:
        if c.name.startswith("BoosterEngines"):
            set_color_keys(c, np.stack([on_boost] * 3, 1), const=False)
s2heat = np.clip(gsmooth(on_s2, 6) * 1.2, 0, 1)
set_color_keys(s2noz, np.stack([s2heat] * 3, 1), const=False)

# --- vents / fog / steam / trail ---
def vol_keys(mat, density, evolve, glow=None):
    nt = mat.node_tree
    bake(nt, value_path("Density"), density)
    bake(nt, value_path("Evolve"), evolve)
    if glow is not None:
        bake(nt, value_path("Glow"), glow)


evo = np.cumsum(np.full(N, 1.0 / FPS))
vent_d = (0.9 + 2.2 * PRE + 0.6 * BASS * (IDX < IGN)) * (1 - ramp(IDX, LIFT, LIFT + 20))
vol_keys(M_VENT, vent_d, evo * 0.35 + np.cumsum(PRE) * 0.004)
for k, v in enumerate(VENTS):
    base = Vector(v.location)
    puff = 1.0 + 0.35 * PRE + 0.08 * np.sin(IDX * 0.07 + k)
    bake_vec(v, "scale", np.stack([v.scale[0] * puff, v.scale[1] * puff, v.scale[2] * puff * (1 + 0.3 * PRE)], 1))
vol_keys(M_FOG, (0.35 + 0.8 * PRE) * (1 - ramp(IDX, IGN, IGN + 25)), evo * 0.15)
tig = np.maximum(IDX - IGN, 0) / FPS
for o, base, drift, r_end, k in STEAM:
    delay = 0.06 * k if k < 5 else 0.35
    tt_ = np.maximum(tig - delay, 0)
    grow = (0.3 * ease_out(np.clip(tt_ / 0.8, 0, 1), 2.0) + 0.7 * ease_out(np.clip(tt_ / 9.0, 0, 1), 1.8)) * (IDX >= IGN)
    r = np.maximum(r_end * grow, 0.01) * (1 + 0.05 * KICK * HOT)
    pos = np.array(base)[None, :] + np.array(drift)[None, :] * (tt_ * 9.0)[:, None] * (IDX >= IGN)[:, None]
    pos[:, 2] += r * 0.45
    bake_vec(o, "location", pos)
    bake_vec(o, "scale", np.stack([r, r * 0.9, r * 0.75], 1))
steam_glow = (on_core + on_boost) * 0.5 * np.clip(1 - ALT / 260, 0, 1) * PULSE * 1.2
bake(M_STEAMB.node_tree, value_path("Glow"), steam_glow * (1 + 2.0 * IGN_FLASH) * 0.5)
bake(M_STEAMB.node_tree, value_path("Green"), np.clip(IGN_FLASH * 1.2, 0, 1))
tfz = np.maximum(IDX - IGN + 1, 0) / FPS
bf_on = (IDX >= IGN - 1) & (SET == 0)
bf_r = (6 + 12 * ease_out(np.clip(tfz / 0.6, 0, 1))) * np.clip(1.4 - ALT / 60, 0.25, 1.0) * (1 + 0.1 * KICK * HOT)
bake_vec(basefire, "scale", np.stack([bf_r, bf_r, bf_r * 0.45], 1))
bake_vec(basefire, "location", np.stack([np.zeros(N), np.zeros(N), 8.5 + np.minimum(ALT, 30) * 0.4], 1))
tr_r = 3 + 13 * ease_out(np.clip(tfz / 1.4, 0, 1), 2.0)
bake_vec(trenchfire, "scale", np.stack([tr_r * 1.3, tr_r, tr_r * 0.7], 1))
bake_vec(trenchfire, "location", np.stack([30 + 22 * ease_out(np.clip(tfz / 2.0, 0, 1)), np.zeros(N), 6 + tr_r * 0.5], 1))
fire_glow = (on_core * PULSE * (1 + 3 * IGN_FLASH)) * np.clip(1.2 - ALT / 120, 0, 1) * 2.6
show(basefire, bf_on & (ALT < 120))
bake(M_FIREB.node_tree, value_path("Green"), np.clip(IGN_FLASH * 1.2, 0, 1))
show(trenchfire, bf_on & (ALT < 160))
PUFFS = []
for side, bem in ((-1, BL), (1, BRt)):
    for z in (3.0, CORE_LEN + 2.5):
        o = billow_obj(f"SepPuff{len(PUFFS)}", M_FIREB, (-side * 1.2, -0.8, z), bem, seed=50 + len(PUFFS),
                       count=14, amp=0.18, speed=1.5, lo=True)
        PUFFS.append((o, BSEP))
for y in (-2.0, 2.0):
    o = billow_obj(f"SepPuff{len(PUFFS)}", M_FIREB, (0.0, y, FAIR_Z + 0.6), S2, seed=50 + len(PUFFS), count=14,
                   amp=0.18, speed=1.5, lo=True)
    PUFFS.append((o, FAIRSEP))
o = billow_obj(f"SepPuff{len(PUFFS)}", M_FIREB, (0.0, -1.0, CORE_LEN + 1.0), S2, seed=77, count=14,
               amp=0.18, speed=1.5, lo=True)
PUFFS.append((o, MECO + int(round(BAR / 2))))
for o, f0 in PUFFS:
    ta = (IDX - f0).astype(np.float64)
    r = np.where(ta >= 0, (0.3 + 1.5 * ease_out(np.clip(ta / 5.0, 0, 1))) * (1 - ramp(ta, 6, 15)) + 0.01, 0.01)
    bake_vec(o, "scale", np.stack([r, r, r], 1))
    show(o, (ta >= 0) & (ta < 16))
puff_glow = np.zeros(N)
for _o, f0 in PUFFS:
    ta = (IDX - f0).astype(np.float64)
    puff_glow = np.maximum(puff_glow, np.where((ta >= 0) & (ta < 16), 40.0 * np.exp(-np.maximum(ta, 0) / 4.0), 0.0))
bake(M_FIREB.node_tree, value_path("Glow"),
     np.maximum(fire_glow * (1.0 - 0.6 * ramp(IDX, IGN + 20, IGN + 120)) * (SET == 0), puff_glow * 0.4))
tp = bpy.data.objects["TrailPad"]
tr_len = np.maximum(ALT + 2.0, 0.1)
bake_vec(tp, "location", np.stack([np.zeros(N), np.zeros(N), np.full(N, 8.0)], 1))
tr_r = 6.0 + 0.04 * ALT
bake_vec(tp, "scale", np.stack([tr_r, tr_r, tr_len], 1))
vol_keys(M_TRAIL, np.full(N, 0.25), evo * 0.2, 0.8 * on_core * PULSE)
bake(FIRE, "energy", (on_core + on_boost) * 0.5 * np.clip(1 - ALT / 200, 0, 1) * PULSE * 2.5e5 * (1 + 2 * IGN_FLASH))
show(FIRE_OB, (SET == 0) & (IDX >= IGN - 1))
# flight trail: hangs below the rocket, pointing back along the flight path
tl = np.where(SET == 1, 2400.0, 1600.0)
tf_loc = np.array([wl(i, (0, 0, -6.0)) for i in range(N)])
bake_vec(trail_f, "location", tf_loc)
bake_vec(trail_f, "rotation_euler", np.stack([np.radians(-PITCH) + math.pi, np.zeros(N), np.zeros(N)], 1))
fr_r = np.where(SET == 1, 8.0, 22.0) * alt_expand
bake_vec(trail_f, "scale", np.stack([fr_r, fr_r, tl], 1))
vol_keys(M_TRAILF, np.where(SET == 1, 0.05, 0.0), evo * 0.15 + np.cumsum(np.full(N, 0.05)),
         0.12 * on_core * PULSE)
# cloud deck: moves down past the rocket; glows from the plume (sub-bass)
bake_vec(deck_c, "location", np.stack([np.zeros(N), np.zeros(N), DECK_TOP - 160.0], 1))
vol_keys(M_DECK, np.full(N, 0.03), evo * 0.05, 0.0 * BASS)
# frost: falls away at liftoff
fr_amt = 1.0 - ramp(IDX, LIFT, LIFT + 90)
for mat in (M_PAINT_CORE, M_PAINT_BOOST):
    bake(mat.node_tree, value_path("Frost"), fr_amt)

# --- vapour cone + shock ring (max-Q) ---
tq = (IDX - MAXQ).astype(np.float64)
vc_gain = np.clip(ramp(IDX, MAXQ - 2, MAXQ + 2) * (1 - ramp(IDX, seg_after(MAXQ) - 30, seg_after(MAXQ))), 0, 1)
vc_gain = vc_gain * (0.35 + 0.9 * KICK * HOT + 0.25 * BASS)          # condensation flickers with the kicks
vc_gain = vc_gain + np.where(IDX >= MAXQ, 1.4 * np.exp(-np.maximum(IDX - MAXQ, 0) / 6.0), 0.0)  # "sonic boom" burst
bake(M_VCONE.node_tree, value_path("Gain"), vc_gain * 0.95)
bake(M_VCONE.node_tree, value_path("Evolve"), evo * 3.0)
vcs = 1.0 + 0.06 * KICK
bake_vec(vcone, "scale", np.stack([vcs, vcs, 1.0 + 0.1 * KICK], 1))
rs = np.where(tq >= 0, 2.0 + 9.0 * ease_out(np.clip(tq / 12, 0, 1), 2.0), 1.0)
bake_vec(shock, "scale", np.stack([rs, rs, 1 + 6 * np.clip(tq / 12, 0, 1)], 1))
bake(M_SHOCK.node_tree, value_path("Gain"), np.where(tq >= 0, np.exp(-np.maximum(tq, 0) / 3.5), 0.0) * 1.3)

# --- ice debris (snares) + sparkle (hi-hats) ---
dpath_age = getattr(dmod.properties.inputs, DEB_AGE.identifier).path_from_id("value")
dpath_seed = getattr(dmod.properties.inputs, DEB_SEED.identifier).path_from_id("value")
bake(debris, dpath_age, np.minimum(SAGE, 60) / FPS)
bake(debris, dpath_seed, np.maximum(SIDX, 0).astype(np.float64) * 7.0, const=True)
spk_path = getattr(smod.properties.inputs, SPK_AMT.identifier).path_from_id("value")
spk_amt = HAT * (0.10 + 0.25 * HOT) * np.where(SET >= 1, 1.0, 0.5 * (IDX < LIFT))
bake(sparkle, spk_path, spk_amt)

# --- pad lights: floods, beacons, searchlights; blackout on T_black before ignition ---
pad_black = ((BLACK > 0) & (IDX < IGN + 2)).astype(np.float64)
pad_cut = np.clip(pad_black + ramp(IDX, IGN - 3, IGN), 0, 1) * (IDX < IGN + 1) + (IDX >= IGN)
pad_cut = np.clip(pad_cut, 0, 1)
flood_on = (1 - pad_cut) + (IDX >= IGN + 6) * 0.6          # floods come back dimmer after ignition
snare_flash = SNARE * (IDX < IGN) * 0.35
for o in FLOODS:
    if o.type == "LIGHT":
        bake(o.data, "energy", 1.0e5 * flood_on * (1 + snare_flash))
    else:
        set_color_keys(o, np.stack([flood_on * 60.0 * (1 + snare_flash)] * 3, 1) * np.array([0.9, 0.95, 1.0]),
                       const=False)
# beacons: blink on the beat; build doubles then quadruples the rate (T_pre)
rate = np.where(PRE > 0.66, 4.0, np.where(PRE > 0.2, 2.0, 1.0))
blink = (np.mod(BEAT_PH * rate + (TL["T_bi"] % 2) * (rate < 2), 1.0) < 0.35).astype(np.float64)
blink = blink * (1 - pad_black)
for o in BEACONS:
    if o.type == "LIGHT":
        bake(o.data, "energy", blink * 1.2e4, const=True)
    else:
        set_color_keys(o, np.stack([blink * 300.0, blink * 12.0, blink * 6.0], 1))
for k, o in enumerate(SEARCH):
    sw_ = np.sin(IDX / FPS * (0.21 + 0.05 * k) + k * 2.1)
    tilt = np.radians(18 + 8 * np.sin(IDX / FPS * 0.13 + k))
    rx = math.pi - tilt                     # spot shines along -Z: flip up, then lean
    rz = np.radians(25 * sw_ + 120 * k)
    bake_vec(o, "rotation_euler", np.stack([rx, np.zeros(N), rz], 1))
    bake(o.data, "energy", 1.6e7 * (1 - pad_black) * (1 - 0.6 * ramp(IDX, IGN, IGN + 60)))

# --- countdown clock (LED segments) ---
rem = np.ceil((IGN - IDX) / FPS).astype(int)
ela = np.floor((IDX - IGN) / FPS).astype(int)
secs = np.where(IDX < IGN, rem, ela)
plus = IDX >= IGN
hh, mm, ss = secs // 3600, (secs // 60) % 60, secs % 60
digs = np.stack([hh // 10, hh % 10, mm // 10, mm % 10, ss // 10, ss % 10], 1)
LED_ON = np.array([1.0, 0.36, 0.035]) * 7.0
LED_OFF = np.array([1.0, 0.36, 0.035]) * 0.05
for (slot, s), o in LED_SEGS.items():
    on = np.array([s in DIGIT_SEGS[int(d)] for d in digs[:, slot]], float)
    on = on * (1 - 0.0 * pad_black)
    set_color_keys(o, LED_OFF[None, :] + on[:, None] * (LED_ON - LED_OFF)[None, :])
colon_on = (BEAT_PH < 0.5).astype(float)
for o in COLON:
    set_color_keys(o, LED_OFF[None, :] + colon_on[:, None] * (LED_ON - LED_OFF)[None, :])
set_color_keys(SIGN_H, np.repeat(LED_ON[None, :], N, 0))
set_color_keys(SIGN_V, plus[:, None] * LED_ON[None, :])          # fully dark when off: no ghost "+"

# --- world / sky / exposure ---
nt = WORLD.node_tree
alt_m = np.zeros(N)
alt_m[SET == 0] = 60 + ALT[SET == 0]
alt_m[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [3500, 9000, 16000])
alt_m[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, MECO, SES], [32000, 58000, 76000, 85000])
alt_m[SET == 3] = np.interp(IDX[SET == 3], [SES, N], [92000, 100000])
bake(nt, 'nodes["Sky"].altitude', alt_m)
# sun: pre-dawn (−7°) on the pad; at altitude it rises toward the limb; it BREAKS the limb on
# the big-drop entry. Horizon dip at h: acos(R/(R+h)).
RE = 6360e3
dip = np.degrees(np.arccos(RE / (RE + alt_m)))
sun_el = np.full(N, -4.8)
sun_el[SET >= 2] = -dip[SET >= 2] - np.interp(IDX[SET >= 2], [BSEP, SES - 20, SES, SES + 3 * BAR, N],
                                              [3.0, 1.1, 0.25, -0.9, -2.6])
bake(nt, 'nodes["Sky"].sun_elevation', np.radians(sun_el))
bake(nt, value_path("EarthH"), (RE + alt_m) / RE)
bake(nt, value_path("SunEl"), np.radians(sun_el))
sun_flare = 1.0 + 2.5 * np.where(IDX >= SES, np.exp(-(IDX - SES) / 12.0), 0.0) + 0.6 * BIG * ONE
bake(nt, value_path("SunDisc"), (SET >= 2) * sun_flare)
bake(nt, value_path("EarthOverlay"), np.clip((alt_m - 8000) / 20000, 0, 1))
bake(nt, value_path("EarthScroll"), np.cumsum(np.where(SET >= 2, 0.00012, 0.00002)))
bake(nt, value_path("Haze"), np.zeros(N))
show(haze_box, SET == 0)
vol_keys(M_HAZE, np.full(N, 0.00028) * (1 - 0.3 * ramp(IDX, IGN, PAD_END)), evo * 0.01)
bake(nt, value_path("Stars"), np.where(SET == 0, 0.55, np.where(SET == 1, 0.8, 1.2)))
bake(nt, value_path("SkyGain"), np.where(SET == 0, 1.0, np.where(SET == 1, 1.0, 1.0)))
# sun lamp follows the sky's sun (azimuth +Y); energy fades in as it clears the limb
sun_vis = np.clip((sun_el + dip + 0.3) / 0.6, 0, 1) * (SET >= 2)
el_r = np.radians(sun_el)
bake_vec(SUN_OB, "rotation_euler", np.stack([el_r - math.pi / 2, np.zeros(N), np.zeros(N)], 1))  # -Z → (0,-cos el,-sin el)
bake(SUN, "energy", sun_vis * 6.0)
show(SUN_OB, sun_vis > 0.001)
# exposure: base per set, blackout on T_black (before every drop), ignition kick
expo = np.where(SET == 0, 2.6, np.where(SET == 1, 0.9, np.where(SET == 2, 1.3, -0.2)))
expo = expo - 2.2 * smoothstep(ramp(IDX, IGN + 5, IGN + 45)) * (SET == 0)
expo = expo + 0.8 * IGN_FLASH                                   # ignition pop (decays over 4 frames)
expo = expo + 0.35 * STROBE * HOT                               # budgeted strobe hits: one-frame pop only
blk = (BLACK > 0) & (IDX > IGN + 10)
expo = np.where(blk, expo - 5.0, expo)
expo = expo - 0.4 * ((IDX >= MECO) & (IDX < SES)) * ramp(IDX, MECO, MECO + 12) * (1 - ramp(IDX, SES - 2, SES))
bake(SC, "view_settings.exposure", expo)

# --- bake visibility (hide_render, CONSTANT) ---
for name, mask in VIS.items():
    ob = bpy.data.objects[name]
    bake(ob, "hide_render", (~mask).astype(np.float64), const=True)

# =====================================================================================
# 11. RENDER SETTINGS + COMPOSITOR
# =====================================================================================
SC.frame_start, SC.frame_end = 1, N
SC.render.fps, SC.render.fps_base = FPS, 1.0
SC.render.resolution_x, SC.render.resolution_y = 1920, 1080
SC.render.resolution_percentage = ARGS.res_pct
SC.view_settings.view_transform = "AgX"
SC.view_settings.look = "AgX - Medium High Contrast"
SC.render.use_motion_blur = True
SC.render.motion_blur_shutter = 0.5
SC.render.motion_blur_position = "START"
SC.render.image_settings.file_format = "PNG"
SC.render.image_settings.color_mode = "RGB"
SC.render.image_settings.color_depth = "8"
SC.render.use_overwrite = False
SC.render.use_placeholder = False
if ARGS.engine == "EEVEE":
    SC.render.engine = "BLENDER_EEVEE"
    ee = SC.eevee
    ee.taa_render_samples = ARGS.samples or 48
    ee.use_raytracing = True
    ee.volumetric_start = 0.5
    ee.volumetric_end = 4000.0
    ee.volumetric_tile_size = "2"
    ee.volumetric_samples = 96
    ee.volumetric_sample_distribution = 0.85
    ee.use_volumetric_shadows = True
    ee.volumetric_shadow_samples = 8
    ee.volumetric_ray_depth = 8
    ee.motion_blur_steps = 1
    ee.use_shadows = True
    ee.shadow_pool_size = "1024"
    ee.shadow_ray_count = 1
    ee.shadow_step_count = 6
    ee.use_fast_gi = True
    ee.clamp_volume_indirect = 10.0
else:
    SC.render.engine = "CYCLES"
    cy = SC.cycles
    cy.device = "GPU"
    cy.samples = ARGS.samples or 64
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.05
    cy.use_denoising = True
    cy.denoiser = "OPTIX"
    cy.max_bounces = 6
    cy.volume_bounces = 1
    cy.volume_step_rate = 4.0
    cy.sample_clamp_indirect = 8.0
    cy.blur_glossy = 1.0
    cy.use_light_tree = True
    cy.caustics_reflective = cy.caustics_refractive = False

# compositor: bloom (plume, sun, LEDs) + a touch of lens dispersion + vignette
cg = bpy.data.node_groups.new("Post", "CompositorNodeTree")
cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
rl = cg.nodes.new("CompositorNodeRLayers")
gl = cg.nodes.new("CompositorNodeGlare")
gl.inputs["Type"].default_value = "Bloom"
gl.inputs["Quality"].default_value = "Medium"
gl.inputs["Threshold"].default_value = 1.2
gl.inputs["Strength"].default_value = 0.4
gl.inputs["Size"].default_value = 0.75
ld_ = cg.nodes.new("CompositorNodeLensdist")
ld_.inputs["Distortion"].default_value = -0.006
ld_.inputs["Dispersion"].default_value = 0.012
ld_.inputs["Fit"].default_value = True
vg = cg.nodes.new("CompositorNodeEllipseMask")
vg.inputs["Size"].default_value = (1.25, 1.25)
vb = cg.nodes.new("CompositorNodeBlur") if hasattr(bpy.types, "CompositorNodeBlur") else None
mul = cg.nodes.new("ShaderNodeMix")
mul.data_type = "RGBA"
mul.blend_type = "MULTIPLY"
go = cg.nodes.new("NodeGroupOutput")
gs = cg.nodes.new("CompositorNodeGlare")
gs.inputs["Type"].default_value = "Streaks"
gs.inputs["Quality"].default_value = "Medium"
gs.inputs["Threshold"].default_value = 80.0
gs.inputs["Strength"].default_value = 0.6
gs.inputs["Streaks"].default_value = 6
gs.inputs["Streaks Angle"].default_value = math.radians(15)
gs.inputs["Fade"].default_value = 0.9
cg.links.new(rl.outputs["Image"], gs.inputs["Image"])
cg.links.new(gs.outputs["Image"], gl.inputs["Image"])
cg.links.new(gl.outputs["Image"], ld_.inputs["Image"])
mr = cg.nodes.new("ShaderNodeMapRange")
mr.inputs["To Min"].default_value = 0.72
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

# sound strip (for scrubbing in the GUI; the MP4 mux happens in render.sh)
if os.path.exists(ARGS.audio):
    se = SC.sequence_editor_create()
    se.strips.new_sound("Music", ARGS.audio, channel=1, frame_start=1)
SC.render.use_sequencer = False

# bookkeeping for render.sh / tests
SC["shots"] = ";".join(f"{s}:{n}" for s, n in SHOTS)
SC["set_of_frame"] = "".join(str(int(k)) for k in SET)          # index i → set digit
rc = np.array([wl(i, (0, 0, 30.0)) for i in range(N)])
vdist = np.linalg.norm(cam_loc - rc, axis=1)
vend = np.clip(1.8 * vdist + 160.0, 300.0, 3000.0)
vend = np.where(SET == 1, np.maximum(vend, 2600.0), vend)
vend = np.where((SET == 0) & (IDX < IGN), np.maximum(vend, 1400.0), vend)
SC["vol_end"] = [float(round(v, 1)) for v in vend]
SC["entries"] = list(ENT)
os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(ARGS.out), compress=True)
print(f"[rocket] saved {ARGS.out}  frames 1..{N}  engine {SC.render.engine}")
