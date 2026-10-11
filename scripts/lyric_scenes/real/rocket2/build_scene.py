"""ROCKET LAUNCH v2 — photoreal, music-synced Blender scene for "Hit a Bump" (Tzeke000).

    ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/real/rocket2/build_scene.py -- \
        [--timeline state/tzeke_songs/hit_a_bump/timeline_30fps.npz] \
        [--lib /data/iris_assets/rocket2/rocket2_lib.blend] [--assets /data/iris_assets] \
        [--out scratch/scenes/rocket2/rocket2.blend]

v2 rebuilds v1 (../rocket) for photorealism (docs/research/rocket_photoreal_2026-10-10.md) and
follows Zeke's "look for what is already online first": the vehicle is assembled from NASA's
public-domain SLS Block 1 hardware model + the Shuttle SRB + the Orion capsule, the tower is NASA's
Mobile Launcher, steam/smoke/clouds are JangaFX CC0 VDB simulations, the Earth is NASA Blue/Black
Marble, scrub is Poly Haven CC0. prep_assets.py turns the downloads into one library .blend.

Built DETERMINISTICALLY from the timeline npz: every music reaction is BAKED (camera, lights,
plumes, LED clock, steam, exposure); no handlers, no Python drivers. Frame f = timeline index f-1.

Story (SLS sequence, drop entries from the npz so it re-times itself):
  intro         pre-dawn pad under xenon floods; LED countdown; deluge water starts sheeting off
                the mobile-launcher deck (T-10 s), hydrogen burn-off sparks, the four core engines
                START on the build's bar (anticipation: translucent plumes + Mach disks + first
                steam), pad floods cut on T_black; T-0 = drop entry: SRB IGNITION.
  drop 1        SRB ignition + liftoff: blinding plume, steam blasting out of the flame trench,
                tower clear, long-lens tracking with heat shimmer, onboard booster camera.
  calm          climbing into a cumulus deck lit from inside by the plume.
  drop 2        MAX-Q: condensation (vapour) cone shroud, flickering on the kicks.
  drop 3        BOOSTER SEPARATION (separation-motor puffs, boosters tumble away).
  drop 4        LAUNCH-ABORT-TOWER JETTISON: tower motor fires, ogive + panels fly off, Orion.
  calm          MECO + core separation, coast; the high plume becomes a sunlit "jellyfish".
  BIG drop      upper-stage ignition and the sun breaking Earth's limb (NASA Blue/Black Marble).
  outro         engine cutoff, drift over the Earth.
Music: kick = engine pulse + camera jolt, snare = ice shed / flood flicker, sub-bass = plume and
cloud glow, hi-hat = beacon strobes (no sparkle glints: real ice falls, it doesn't twinkle).
"""
import argparse
import glob
import math
import os
import sys

import bmesh  # noqa: F401
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector  # noqa: F401

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
    ap.add_argument("--assets", default="/data/iris_assets")
    ap.add_argument("--lib", default="/data/iris_assets/rocket2/rocket2_lib.blend")
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "scenes", "rocket2", "rocket2.blend"))
    ap.add_argument("--samples", type=int, default=0, help="0 = tuned default")
    ap.add_argument("--res-pct", type=int, default=100)
    return ap.parse_args(argv)


ARGS = parse_args()
AS = ARGS.assets

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


# --- SLS engine states -------------------------------------------------------------------
# T-0 (SRB ignition) is EXACTLY the first drop entry (IGN, timeline index 987 = frame 988).
# The four core engines start on the build's bar (real: T-6.36 s; here the bar before the build
# peak so the start-up reads as the musical wind-up), the burn-off igniters spark a bar earlier,
# the deluge water starts ~T-10 s.
PRE_START = int(np.nonzero(PRE > 0)[0][np.nonzero(PRE > 0)[0] < IGN][0])
RS25_T = int(DOWNBEATS[DOWNBEATS <= PRE_START][-1])          # bar at/just before the build
SPARK_T = int(DOWNBEATS[DOWNBEATS < RS25_T][-1])             # one bar earlier: sparklers
DELUGE_T = snap(IGN - 10.0 * FPS)                             # water on the deck
LIFT = IGN + 4                                                # SRB ignition = liftoff (~0.13 s)
_eng_start = [RS25_T + int(round(k * 0.12 * FPS)) for k in range(4)]   # staggered 120 ms
on_eng = [ramp(IDX, s, s + 14) * (1 - ramp(IDX, MECO, MECO + 4)) for s in _eng_start]
on_core = np.mean(on_eng, axis=0)
on_boost = ramp(IDX, IGN - 1, IGN + 1) * (1 - ramp(IDX, BSEP, BSEP + 2))
on_s2 = ramp(IDX, SES - 1, SES + 2) * (1 - ramp(IDX, SECO, SECO + 8))
IGN_FLASH = np.where((IDX >= IGN) & (IDX < IGN + 5), 1.0 - (IDX - IGN) / 5.0, 0.0)   # SRB light-up pop
SPARKS = ramp(IDX, SPARK_T, SPARK_T + 3) * (1 - ramp(IDX, IGN + 2, IGN + 8))
DELUGE = ramp(IDX, DELUGE_T, DELUGE_T + 40) * (1 - ramp(IDX, IGN + 30 * FPS, IGN + 32 * FPS))
PULSE = 1.0 + HOT * (0.30 * KICK + 0.10 * BASS + 0.12 * SNARE) + BIG * 0.35 * ONE
T_SEC = np.maximum(IDX - LIFT, 0) / FPS
ALT = 0.5 * 5.0 * T_SEC ** 2                                  # metres above the pad (pad set)

# --- sets (which world is on screen) ---------------------------------------------------
PAD_END = [s1 for s0, s1, h in SEGS if s0 == IGN][0]          # end of drop 1
SET = np.zeros(N, np.int32)                                    # 0 pad, 1 clouds, 2 high, 3 space
SET[PAD_END:] = 1
SET[BSEP:] = 2
SET[SES:] = 3


def in_set(k):
    return (SET == k).astype(np.float64)


# =====================================================================================
# 2. SHOT LIST (cuts on drop entries and bar downbeats). Camera language (research §1.14):
#    long-lens tracking from km away + low wide pad remotes + onboard cameras; chase angles
#    only where an onboard cue (fisheye, vibration, exposure pumping) sells them.
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
ONBOARD = {"B4_tail", "D2_under", "F1_sep", "I_meco"}   # lens/vibration cues
print("[rocket2] entries", ENT, "MECO", MECO, "SECO", SECO, "PAD_END", PAD_END,
      "RS25", RS25_T, "SPARK", SPARK_T, "DELUGE", DELUGE_T)
print("[rocket2] shots", [(s, n) for s, n in SHOTS])

# =====================================================================================
# 3. BLENDER HELPERS
# =====================================================================================
SC = bpy.context.scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
             bpy.data.curves, bpy.data.worlds, bpy.data.node_groups, bpy.data.images, bpy.data.volumes):
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
# 4. MATERIALS — PBR rules (lookdev §2): paint/foam/concrete are DIELECTRIC, albedo never 1.0,
#    roughness from CC0 scans (Non-Color), box projection in metres; metal only where bare.
# =====================================================================================
def load_img(path, data=False):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = "Non-Color" if data else "sRGB"
    return im


def tex_box(nt, path, tile_m, data=False, coord=None, blend=0.25):
    """Box-projected image texture in object space; tile_m = metres per tile. Returns the node."""
    if coord is None:
        tc = nd(nt, "ShaderNodeTexCoord")
        coord = tc.outputs["Object"]
    mp = nd(nt, "ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / tile_m,) * 3
    lk(nt, coord, mp.inputs["Vector"])
    t = nd(nt, "ShaderNodeTexImage", image=load_img(path, data), projection="BOX", projection_blend=blend)
    lk(nt, mp.outputs[0], t.inputs["Vector"])
    return t


def ph(name, kind, res="2k"):
    d = os.path.join(AS, "polyhaven", "tex_" + name)
    return os.path.join(d, f"{name}_{kind}_{res}.jpg")


def acg(name, kind):
    return os.path.join(AS, "ambientcg", name, f"{name}_2K-JPG_{kind}.jpg")


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
PAINT_R = acg("PaintedMetal017", "Roughness")
PAINT_N = acg("PaintedMetal017", "NormalGL")
PAINT_C = acg("PaintedMetal017", "Color")


def scan_rough_normal(nt, tile_m, rough_lo, rough_hi, nstr=0.35, coord=None):
    """(roughness socket, normal socket) from the ambientCG painted-metal scan."""
    r = tex_box(nt, PAINT_R, tile_m, True, coord)
    n = tex_box(nt, PAINT_N, tile_m, True, coord)
    rr = nd(nt, "ShaderNodeMapRange", inp={"To Min": rough_lo, "To Max": rough_hi})
    lk(nt, r.outputs["Color"], rr.inputs["Value"])
    nm = nd(nt, "ShaderNodeNormalMap", inp={"Strength": nstr})
    lk(nt, n.outputs["Color"], nm.inputs["Color"])
    return rr.outputs["Result"], nm.outputs["Normal"]


def wear(nt, tile_m, lo=0.80, coord=None):
    """Compressed luminance of the scan's colour map: wear/grime WITHOUT its blue pattern."""
    c = tex_box(nt, PAINT_C, tile_m, False, coord)
    bw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, c.outputs["Color"], bw.inputs[0])
    mr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.15, "From Max": 0.6, "To Min": lo, "To Max": 1.0})
    lk(nt, bw.outputs[0], mr.inputs["Value"])
    return mr.outputs["Result"]


def col_mul(nt, col, fac):
    n = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    n.inputs[0].default_value = 1.0
    for k, x in ((6, col), (7, fac)):
        if isinstance(x, (tuple, list)):
            n.inputs[k].default_value = tuple(x) + ((1.0,) if len(x) == 3 else ())
        elif isinstance(x, (int, float)):
            n.inputs[k].default_value = (x, x, x, 1.0)
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[2]


def finish(nt, b):
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])


def paint_white(name, albedo=0.74, tile=1.6, soot=None):
    """Vehicle white paint / thermal blanket: dielectric, scan roughness + normal, wear.
    soot = (z0, z1) object-space band that darkens with a keyed 'Soot' value (plume staining)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0, "Coat Weight": 0.08})
    rough, nrm = scan_rough_normal(nt, tile, 0.32, 0.62)
    base = col_mul(nt, (albedo, albedo, albedo * 0.985), wear(nt, tile * 1.7, 0.86))
    if soot:
        sv = value_node(nt, "Soot", 0.0)
        tc = nd(nt, "ShaderNodeTexCoord")
        sz = nd(nt, "ShaderNodeSeparateXYZ")
        lk(nt, tc.outputs["Object"], sz.inputs[0])
        st = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6})
        lk(nt, V(nt, "MULTIPLY", tc.outputs["Object"], (2.5, 2.5, 0.25)), st.inputs["Vector"])
        band = nd(nt, "ShaderNodeMapRange", inp={"From Min": soot[1], "From Max": soot[0]})
        lk(nt, sz.outputs[2], band.inputs["Value"])
        f = M(nt, "MULTIPLY", M(nt, "MULTIPLY", band.outputs["Result"], sv.outputs[0]),
              M(nt, "MULTIPLY_ADD", st.outputs["Fac"], 1.6, -0.15), clamp=True)
        base = mix_col(nt, f, base, (0.05, 0.045, 0.04))
        rough = M(nt, "ADD", rough, M(nt, "MULTIPLY", f, 0.25))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, rough, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    finish(nt, b)
    return m


def frost_layer(nt, base, rough, nrm, zlo, zhi, amount_node, scale=0.9):
    """Cryogenic frost (lookdev §4): white, rough, bumpy, noise-broken band in object Z,
    scaled by a keyed amount (sheds after liftoff). Returns (base, rough, normal)."""
    tc = nd(nt, "ShaderNodeTexCoord")
    sz = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["Object"], sz.inputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": scale, "Detail": 8.0, "Roughness": 0.65})
    lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
    band = nd(nt, "ShaderNodeMapRange", inp={"From Min": zlo, "From Max": zhi, "To Min": 0.0, "To Max": 1.0})
    lk(nt, sz.outputs[2], band.inputs["Value"])
    tri = M(nt, "SUBTRACT", 1.0, M(nt, "ABSOLUTE", M(nt, "MULTIPLY_ADD", band.outputs["Result"], 2.0, -1.0)))
    mixe = M(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 1.2, M(nt, "MULTIPLY", tri, amount_node))
    fr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 1.05, "From Max": 1.3})
    lk(nt, mixe, fr.inputs["Value"])
    f = fr.outputs["Result"]
    fine = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 60.0, "Detail": 4.0})
    lk(nt, tc.outputs["Object"], fine.inputs["Vector"])
    bp = nd(nt, "ShaderNodeBump", inp={"Distance": 0.004})
    lk(nt, fine.outputs["Fac"], bp.inputs["Height"])
    lk(nt, f, bp.inputs["Strength"])
    lk(nt, nrm, bp.inputs["Normal"])
    base = mix_col(nt, f, base, (0.86, 0.89, 0.92))
    rough = M(nt, "ADD", M(nt, "MULTIPLY", rough, M(nt, "SUBTRACT", 1.0, f)), M(nt, "MULTIPLY", f, 0.85))
    return base, rough, bp.outputs["Normal"]


# --- core stage: orange spray-on foam, intertank stringers, emblem decal, soot -----------
CORE_R = 4.2
LOGO_H = 6.2
LOGO_Z0 = 22.0                                   # on the LH2 tank, below the intertank
LOGO_ANG = -math.pi / 2                          # faces -Y (the camera side)


def foam_material(name):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0, "Coat Weight": 0.0})
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sz = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sz.inputs[0])
    # spray passes: large soft mottling + horizontal banding of the spray robot
    mot = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.35, "Detail": 5.0, "Roughness": 0.55})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.45)), mot.inputs["Vector"])
    band = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 3.0})
    lk(nt, V(nt, "MULTIPLY", ob, (0.15, 0.15, 2.2)), band.inputs["Vector"])
    v = M(nt, "ADD", M(nt, "MULTIPLY", M(nt, "SUBTRACT", mot.outputs["Fac"], 0.5), 0.55),
          M(nt, "MULTIPLY", M(nt, "SUBTRACT", band.outputs["Fac"], 0.5), 0.25))
    dark = (0.30, 0.105, 0.030)
    light = (0.50, 0.22, 0.075)
    base = mix_col(nt, M(nt, "ADD", 0.5, v, clamp=True), dark, light)
    # close-out patches (hand-sprayed repairs): lighter, smoother voronoi cells
    vo = nd(nt, "ShaderNodeTexVoronoi", inp={"Scale": 0.55, "Randomness": 1.0})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.6)), vo.inputs["Vector"])
    patch = M(nt, "MULTIPLY", M(nt, "GREATER_THAN", vo.outputs["Distance"], 0.0),
              M(nt, "GREATER_THAN", M(nt, "FRACT", M(nt, "MULTIPLY", vo.outputs["Distance"], 1.0)), 0.93))
    pw = nd(nt, "ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    lk(nt, vo.outputs["Color"], pw.inputs["Vector"])
    patch = M(nt, "MULTIPLY", M(nt, "LESS_THAN", pw.outputs["Value"], 0.18),
              M(nt, "LESS_THAN", vo.outputs["Distance"], 0.42))
    base = mix_col(nt, M(nt, "MULTIPLY", patch, 0.6), base, (0.58, 0.29, 0.10))
    # foam surface: orange-peel micro bump + lumpy macro bump; matte
    fine = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 22.0, "Detail": 6.0, "Roughness": 0.7})
    lk(nt, ob, fine.inputs["Vector"])
    lump = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 2.2, "Detail": 3.0})
    lk(nt, ob, lump.inputs["Vector"])
    h = M(nt, "ADD", M(nt, "MULTIPLY", fine.outputs["Fac"], 0.35), lump.outputs["Fac"])
    # intertank stringers (z 46.3..52.9): vertical ribs
    ang = M(nt, "ARCTAN2", sz.outputs[1], sz.outputs[0])
    rib = M(nt, "POWER", M(nt, "ABSOLUTE", M(nt, "SINE", M(nt, "MULTIPLY", ang, 72.0))), 6.0)
    it = M(nt, "MULTIPLY", M(nt, "GREATER_THAN", sz.outputs[2], 46.3), M(nt, "LESS_THAN", sz.outputs[2], 52.9))
    h = M(nt, "ADD", h, M(nt, "MULTIPLY", M(nt, "MULTIPLY", rib, it), 1.4))
    # flange rings between tanks/sections
    for zr in (6.2, 46.3, 52.9, 61.7):
        ring = M(nt, "LESS_THAN", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", sz.outputs[2], zr)), 0.10)
        h = M(nt, "SUBTRACT", h, M(nt, "MULTIPLY", ring, 0.8))
        base = mix_col(nt, M(nt, "MULTIPLY", ring, 0.35), base, (0.18, 0.07, 0.02))
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.18, "Distance": 0.03})
    lk(nt, h, bp.inputs["Height"])
    rough = M(nt, "ADD", 0.72, M(nt, "MULTIPLY", v, 0.2))
    nrm = bp.outputs["Normal"]
    # the Tzeke000 emblem: cylindrical decal on the LH2 tank facing -Y (painted on the foam)
    u = M(nt, "DIVIDE", M(nt, "SUBTRACT", ang, LOGO_ANG), 1.0)          # radians around from centre
    s_u = M(nt, "MULTIPLY", u, CORE_R)                                  # metres along the arc
    lw = LOGO_H * LOGO.size[0] / LOGO.size[1]
    du = M(nt, "ADD", M(nt, "DIVIDE", s_u, -lw), 0.5)
    dv = M(nt, "DIVIDE", M(nt, "SUBTRACT", sz.outputs[2], LOGO_Z0), LOGO_H)
    cxy = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, du, cxy.inputs[0])
    lk(nt, dv, cxy.inputs[1])
    it_ = nd(nt, "ShaderNodeTexImage", image=LOGO, extension="CLIP", interpolation="Cubic")
    lk(nt, cxy.outputs[0], it_.inputs["Vector"])
    front = M(nt, "LESS_THAN", sz.outputs[1], 0.0)
    la = M(nt, "MULTIPLY", M(nt, "MULTIPLY", it_.outputs["Alpha"], front), 0.96)
    base = mix_col(nt, la, base, col_mul(nt, it_.outputs["Color"], 0.8))
    rough = M(nt, "ADD", rough, M(nt, "MULTIPLY", la, -0.25))
    # frost where the cryogenic feedline brackets and flanges chill the foam (light, sheds)
    fv = value_node(nt, "Frost", 1.0)
    base, rough, nrm = frost_layer(nt, base, rough, nrm, 52.0, 62.5, M(nt, "MULTIPLY", fv.outputs[0], 0.55), 1.2)
    # soot: the booster plume stains the base of the core in flight (keyed 'Soot')
    sv = value_node(nt, "Soot", 0.0)
    st = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", ob, (2.0, 2.0, 0.2)), st.inputs["Vector"])
    sb = nd(nt, "ShaderNodeMapRange", inp={"From Min": 16.0, "From Max": 2.0})
    lk(nt, sz.outputs[2], sb.inputs["Value"])
    sf = M(nt, "MULTIPLY", M(nt, "MULTIPLY", sb.outputs["Result"], sv.outputs[0]),
           M(nt, "MULTIPLY_ADD", st.outputs["Fac"], 1.5, -0.1), clamp=True)
    base = mix_col(nt, sf, base, (0.035, 0.03, 0.025))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, rough, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    finish(nt, b)
    return m


def srb_material(name, tex_img):
    """Booster: white paint × the NASA SRB texture's LUMINANCE (joint bands, marks — no hue),
    scan roughness, soot keyed at the aft skirt after ignition."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0, "Coat Weight": 0.05})
    tc = nd(nt, "ShaderNodeTexCoord")
    t = nd(nt, "ShaderNodeTexImage", image=tex_img, interpolation="Cubic")
    lk(nt, tc.outputs["UV"], t.inputs["Vector"])
    bw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, t.outputs["Color"], bw.inputs[0])
    mr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.05, "From Max": 0.8, "To Min": 0.06, "To Max": 1.0})
    lk(nt, bw.outputs[0], mr.inputs["Value"])
    base = col_mul(nt, (0.73, 0.72, 0.70), mr.outputs["Result"])
    base = col_mul(nt, base, wear(nt, 2.4, 0.84))
    rough, nrm = scan_rough_normal(nt, 1.8, 0.35, 0.6, 0.3)
    sz = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["Object"], sz.inputs[0])
    sv = value_node(nt, "Soot", 0.0)
    sb = nd(nt, "ShaderNodeMapRange", inp={"From Min": 9.0, "From Max": 0.0})
    lk(nt, sz.outputs[2], sb.inputs["Value"])
    st = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", tc.outputs["Object"], (3.0, 3.0, 0.3)), st.inputs["Vector"])
    sf = M(nt, "MULTIPLY", M(nt, "MULTIPLY", sb.outputs["Result"], sv.outputs[0]),
           M(nt, "MULTIPLY_ADD", st.outputs["Fac"], 1.4, 0.0), clamp=True)
    base = mix_col(nt, sf, base, (0.04, 0.035, 0.03))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, rough, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    finish(nt, b)
    return m


def metal(name, color, rough, tile=1.0, nstr=0.25, heat=False, frost=None):
    """Bare/painted metal with scan roughness variation. heat=True: Object Info colour R = glow
    (nozzle interiors when firing). frost=(zlo, zhi): keyed 'Frost' band (cryo lines)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 1.0})
    r, nrm = scan_rough_normal(nt, tile, rough * 0.75, min(1.0, rough * 1.35), nstr)
    base = col_mul(nt, tuple(color), wear(nt, tile * 1.3, 0.78))
    if frost:
        fv = value_node(nt, "Frost", 1.0)
        base, r, nrm = frost_layer(nt, base, r, nrm, frost[0], frost[1], fv.outputs[0], 2.0)
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, r, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    if heat:
        oi = nd(nt, "ShaderNodeObjectInfo")
        sc_ = nd(nt, "ShaderNodeSeparateColor")
        lk(nt, oi.outputs["Color"], sc_.inputs[0])
        lk(nt, M(nt, "MULTIPLY", sc_.outputs[0], 14.0), b.inputs["Emission Strength"])
        b.inputs["Emission Color"].default_value = blackbody(1900) + (1,)
    finish(nt, b)
    return m


def dielectric(name, color, rough, tile=1.5, nstr=0.25):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0})
    r, nrm = scan_rough_normal(nt, tile, rough * 0.8, min(1.0, rough * 1.25), nstr)
    lk(nt, col_mul(nt, tuple(color), wear(nt, tile * 1.5, 0.8)), b.inputs["Base Color"])
    lk(nt, r, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    finish(nt, b)
    return m


def tower_steel(name, color=(0.20, 0.215, 0.23)):
    """Mobile-launcher steel: grey paint + Poly Haven rusty_painted_metal roughness/normal and
    a desaturated grime multiplier (box-mapped, 2.5 m tiles)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0})
    d = tex_box(nt, ph("rusty_painted_metal", "diff"), 2.5)
    r = tex_box(nt, ph("rusty_painted_metal", "rough"), 2.5, True)
    n = tex_box(nt, ph("rusty_painted_metal", "nor_gl"), 2.5, True)
    bw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, d.outputs["Color"], bw.inputs[0])
    g = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.1, "From Max": 0.6, "To Min": 0.55, "To Max": 1.15})
    lk(nt, bw.outputs[0], g.inputs["Value"])
    lk(nt, col_mul(nt, color, g.outputs["Result"]), b.inputs["Base Color"])
    rr = nd(nt, "ShaderNodeMapRange", inp={"To Min": 0.45, "To Max": 0.85})
    lk(nt, r.outputs["Color"], rr.inputs["Value"])
    wet = value_node(nt, "Wet", 0.0)
    lk(nt, M(nt, "SUBTRACT", rr.outputs["Result"], M(nt, "MULTIPLY", wet.outputs[0], 0.35)), b.inputs["Roughness"])
    nm = nd(nt, "ShaderNodeNormalMap", inp={"Strength": 0.6})
    lk(nt, n.outputs["Color"], nm.inputs["Color"])
    lk(nt, nm.outputs["Normal"], b.inputs["Normal"])
    finish(nt, b)
    return m


def concrete(name, tile=4.0, color_gain=1.0):
    """Pad concrete (Poly Haven concrete_floor_02) + keyed 'Wet' (deluge): darker, glossier."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0})
    d = tex_box(nt, ph("concrete_floor_02", "diff"), tile)
    r = tex_box(nt, ph("concrete_floor_02", "rough"), tile, True)
    n = tex_box(nt, ph("concrete_floor_02", "nor_gl"), tile, True)
    wet = value_node(nt, "Wet", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    pud = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.08, "Detail": 3.0})
    lk(nt, tc.outputs["Object"], pud.inputs["Vector"])
    wmask = M(nt, "MULTIPLY", wet.outputs[0], M(nt, "MULTIPLY_ADD", pud.outputs["Fac"], 2.2, -0.35), clamp=True)
    base = col_mul(nt, d.outputs["Color"], M(nt, "MULTIPLY", color_gain, M(nt, "SUBTRACT", 1.0,
                                                                              M(nt, "MULTIPLY", wmask, 0.45))))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, M(nt, "MULTIPLY", r.outputs["Color"], M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", wmask, 0.85))),
       b.inputs["Roughness"])
    nm = nd(nt, "ShaderNodeNormalMap", inp={"Strength": 0.8})
    lk(nt, n.outputs["Color"], nm.inputs["Color"])
    lk(nt, nm.outputs["Normal"], b.inputs["Normal"])
    finish(nt, b)
    return m


M_FOAM = foam_material("CoreFoam")
M_WHITE = paint_white("VehicleWhite", 0.74, soot=(0.0, 6.0))
M_BLANKET = paint_white("ThermalBlanket", 0.66, tile=0.8)
M_GRAY = dielectric("PaintedGrey", (0.30, 0.30, 0.31), 0.5)
M_METAL = metal("BareAluminium", (0.80, 0.80, 0.81), 0.33)
M_DARK = dielectric("DarkComposite", (0.045, 0.045, 0.048), 0.55)
M_NOZ_OUT = metal("NozzleTubes", (0.30, 0.27, 0.25), 0.42, tile=0.4, nstr=0.5, frost=(1.0, 4.2))
M_NOZ_IN = metal("NozzleInner", (0.20, 0.13, 0.09), 0.55, tile=0.5, heat=True)
M_GOLD = metal("GoldFoil", (1.0, 0.72, 0.30), 0.5, tile=0.3, nstr=0.9)
M_CM = metal("OrionBackshell", (0.62, 0.62, 0.63), 0.45, tile=0.6, nstr=0.6)
M_ML = tower_steel("MLSteel")
M_ML_RAIL = tower_steel("MLRail", (0.32, 0.30, 0.10))
M_CONCRETE = concrete("PadConcrete", 4.0)
M_CONCRETE_DK = concrete("TrenchConcrete", 3.0, 0.55)
M_LED = emissive("LED", (1.0, 1.0, 1.0), 1.0, objcolor_gain=True)
M_LAMP = emissive("Lamp", (1.0, 1.0, 1.0), 1.0, objcolor_gain=True)
SEM_MAT = {"SEM_white": M_WHITE, "SEM_foam": M_FOAM, "SEM_metal": M_METAL, "SEM_gray": M_GRAY,
           "SEM_dark": M_DARK, "SEM_nozzle_out": M_NOZ_OUT, "SEM_nozzle_in": M_NOZ_IN, "SEM_gold": M_GOLD,
           "SEM_cm": M_CM, "SEM_ml": M_ML, "SEM_ml_rail": M_ML_RAIL}


# --- exhaust shaders --------------------------------------------------------------------
def color_ramp(nt, fac, stops):
    cr = nd(nt, "ShaderNodeValToRGB")
    els = cr.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], tuple(stops[0][1]) + (1,)
    els[1].position, els[1].color = stops[-1][0], tuple(stops[-1][1]) + (1,)
    for pos, c in stops[1:-1]:
        e = els.new(pos)
        e.color = tuple(c) + (1,)
    lk(nt, fac, cr.inputs[0])
    return cr.outputs[0]


def plume_shell(name, stops, nscale=0.55, along_pow=1.5, diamonds=0.0, n_diam=6.0, diam_reach=0.35,
                edge_pow=2.4, base_alpha=1.0):
    """Additive exhaust shell (Transparent + Emission). Object colour R = intensity, G = flicker
    phase (4D noise W), B = flow scroll. UV v: 0 at the nozzle → 1 at the tail.
    diamonds > 0: bright Mach disks along the first diam_reach of the plume (sea level)."""
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
    gain, flick, scroll = oc.outputs[0], oc.outputs[1], oc.outputs[2]
    off = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, M(nt, "MULTIPLY", scroll, -40.0), off.inputs[2])
    p = V(nt, "MULTIPLY", V(nt, "ADD", tc.outputs["Object"], off.outputs[0]), (1.0, 1.0, 0.28))
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D",
            inp={"Scale": nscale, "Detail": 7.0, "Roughness": 0.62, "Distortion": 0.4})
    lk(nt, p, nz.inputs["Vector"])
    lk(nt, flick, nz.inputs["W"])
    turb = nz.outputs["Fac"]
    col = color_ramp(nt, M(nt, "ADD", v, M(nt, "MULTIPLY", M(nt, "SUBTRACT", turb, 0.5), 0.25), clamp=True), stops)
    lw = nd(nt, "ShaderNodeLayerWeight", inp={"Blend": 0.5})
    face = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), edge_pow)
    along = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, v, clamp=True), along_pow)
    a = M(nt, "MULTIPLY", M(nt, "MULTIPLY", face, along), M(nt, "MULTIPLY_ADD", turb, 1.6, -0.3, clamp=True))
    a = M(nt, "MULTIPLY", a, base_alpha)
    if diamonds > 0:
        dkey = value_node(nt, "Diamonds", 1.0)           # keyed: sea level only (they fade with altitude)
        # Mach disks: narrow bright bands, strongest at the first one, only near the axis (facing)
        ph_ = M(nt, "FRACT", M(nt, "MULTIPLY", v, n_diam / diam_reach))
        disk = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", ph_, 0.5)), 2.0)), 10.0)
        reach = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, M(nt, "DIVIDE", v, diam_reach), clamp=True), 1.2)
        core = M(nt, "POWER", lw.outputs["Facing"], 1.5)
        a = M(nt, "ADD", a, M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "MULTIPLY", disk, reach), core),
                              M(nt, "MULTIPLY", dkey.outputs[0], diamonds)))
    em = nd(nt, "ShaderNodeEmission")
    lk(nt, col, em.inputs["Color"])
    lk(nt, M(nt, "MULTIPLY", a, gain), em.inputs["Strength"])
    tr = nd(nt, "ShaderNodeBsdfTransparent")
    add = nd(nt, "ShaderNodeAddShader")
    lk(nt, tr.outputs[0], add.inputs[0])
    lk(nt, em.outputs[0], add.inputs[1])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, add.outputs[0], o.inputs["Surface"])
    return m


def plume_volume(name, stops, smoke_col=(0.35, 0.3, 0.27), smoke=0.05, r0=0.16, rpow=0.5, sq=(2.2, 2.2, 7.0),
                 along_pow=1.3):
    """Emissive turbulent VOLUME sheath in a unit cylinder (r<=1, z: 0 nozzle → -1 tail), scaled
    per object. Object colour R = intensity, G = flicker W, B = flow scroll. smoke = scattering
    density toward the tail (soot / Al2O3 smoke lit by the plume lights)."""
    m, nt = new_mat(name)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    oi = nd(nt, "ShaderNodeObjectInfo")
    oc = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, oi.outputs["Color"], oc.inputs[0])
    gain, flick, scroll = oc.outputs[0], oc.outputs[1], oc.outputs[2]
    v = M(nt, "MULTIPLY", sp.outputs[2], -1.0, clamp=True)
    r = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    off = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, M(nt, "MULTIPLY", scroll, 1.2), off.inputs[2])
    q = V(nt, "ADD", V(nt, "MULTIPLY", ob, sq), off.outputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D",
            inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6, "Distortion": 0.3})
    lk(nt, q, nz.inputs["Vector"])
    lk(nt, flick, nz.inputs["W"])
    turb = nz.outputs["Fac"]
    R_ = M(nt, "MULTIPLY", M(nt, "ADD", r0, M(nt, "MULTIPLY", M(nt, "POWER", v, rpow), 1.0 - r0)), 0.62)
    rn = M(nt, "ADD", M(nt, "DIVIDE", r, R_), M(nt, "MULTIPLY", M(nt, "SUBTRACT", turb, 0.5), 0.9))
    mask = nd(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP",
              inp={"From Min": 1.0, "From Max": 0.25, "To Min": 0.0, "To Max": 1.0})
    lk(nt, rn, mask.inputs["Value"])
    mk = mask.outputs["Result"]
    core = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, rn, clamp=True), 3.0)
    along = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, v, clamp=True), along_pow)
    attach = M(nt, "MULTIPLY", M(nt, "DIVIDE", v, 0.03), 1.0, clamp=True)
    e = M(nt, "MULTIPLY", M(nt, "MULTIPLY", mk, along), attach)
    e = M(nt, "MULTIPLY", e, M(nt, "MULTIPLY_ADD", core, 2.5, 0.35))
    e = M(nt, "MULTIPLY", e, M(nt, "MULTIPLY_ADD", turb, 1.2, 0.4))
    e = M(nt, "MULTIPLY", e, gain)
    col = color_ramp(nt, M(nt, "ADD", v, M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, core), 0.25)), stops)
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": tuple(smoke_col) + (1,), "Anisotropy": 0.3})
    lk(nt, col, pv.inputs["Emission Color"])
    lk(nt, e, pv.inputs["Emission Strength"])
    lk(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", mk, M(nt, "POWER", v, 1.2)), smoke), pv.inputs["Density"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


# SRB (aluminised APCP): blinding white-yellow core, yellow → orange edges, Al2O3 smoke
SRB_STOPS = [(0.0, (1.0, 0.97, 0.88)), (0.12, (1.0, 0.88, 0.58)), (0.38, (1.0, 0.66, 0.26)),
             (0.75, (0.92, 0.38, 0.10)), (1.0, (0.45, 0.14, 0.04))]
# RS-25 (LH2/LOX): nearly invisible pale blue-violet sheath, the Mach disks carry the light
RS25_STOPS = [(0.0, (0.75, 0.80, 1.0)), (0.2, (0.62, 0.55, 0.95)), (0.6, (0.85, 0.45, 0.45)), (1.0, (0.4, 0.2, 0.3))]
# RL10 (vacuum, LH2): faint, wide, translucent
VAC_STOPS = [(0.0, (1.0, 0.85, 0.7)), (0.1, (0.85, 0.65, 0.9)), (0.5, (0.45, 0.4, 0.8)), (1.0, (0.12, 0.12, 0.35))]
M_SRB_SHELL = plume_shell("SRBPlume", SRB_STOPS, nscale=0.45, along_pow=2.6, edge_pow=1.3)
M_SRB_VOL = plume_volume("SRBPlumeVol", SRB_STOPS, smoke_col=(0.9, 0.88, 0.85), smoke=0.10, along_pow=3.0)
M_RS25_SHELL = plume_shell("RS25Plume", RS25_STOPS, nscale=0.9, along_pow=1.8, diamonds=6.0, n_diam=5.0,
                           diam_reach=0.45, edge_pow=3.0, base_alpha=0.18)
M_VAC_SHELL = plume_shell("VacPlume", VAC_STOPS, nscale=0.15, along_pow=0.9, edge_pow=2.0, base_alpha=0.5)
M_VAC_VOL = plume_volume("VacPlumeVol", VAC_STOPS, smoke=0.0, r0=0.08, rpow=0.32, sq=(1.4, 1.4, 3.0))


def jellyfish_material(name):
    """Twilight plume ('space jellyfish'): a huge translucent bell of exhaust high enough to be
    SUNLIT while the ground is dark. Faked as low-density scattering + an emission rim (blue-
    white outer shell, warm near the axis). Keyed 'Gain' and 'Evolve'."""
    m, nt = new_mat(name)
    gain = value_node(nt, "Gain", 0.0)
    evo = value_node(nt, "Evolve", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    v = M(nt, "MULTIPLY", sp.outputs[2], -1.0, clamp=True)                 # 0 at the vehicle → 1 far
    r = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    bell = M(nt, "ADD", 0.05, M(nt, "MULTIPLY", M(nt, "POWER", v, 0.45), 0.95))   # bell radius
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 2.5, "Detail": 5.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.5)), nz.inputs["Vector"])
    lk(nt, evo.outputs[0], nz.inputs["W"])
    fil = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 9.0, "Detail": 8.0, "Roughness": 0.7,
                                                                   "Distortion": 0.8})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.08)), fil.inputs["Vector"])          # filaments along the flow
    lk(nt, evo.outputs[0], fil.inputs["W"])
    rn = M(nt, "ADD", M(nt, "DIVIDE", r, bell), M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz.outputs["Fac"], 0.5), 0.6))
    shell = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", rn, 0.85)), 3.0), clamp=True), 2.0)
    shell = M(nt, "MULTIPLY", shell, M(nt, "MULTIPLY_ADD", fil.outputs["Fac"], 3.0, -1.0, clamp=True))
    inner = M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, rn, clamp=True), 0.12)
    fade = M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, v, clamp=True), M(nt, "MULTIPLY", M(nt, "DIVIDE", v, 0.05), 1.0, clamp=True))
    d = M(nt, "MULTIPLY", M(nt, "ADD", shell, inner), fade)
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": (0.85, 0.9, 1.0, 1), "Anisotropy": 0.6})
    lk(nt, M(nt, "MULTIPLY", d, M(nt, "MULTIPLY", gain.outputs[0], 0.0004)), pv.inputs["Density"])
    lk(nt, mix_col(nt, M(nt, "SUBTRACT", 1.0, rn, clamp=True), (0.55, 0.7, 1.0), (1.0, 0.75, 0.55)), pv.inputs["Emission Color"])
    lk(nt, M(nt, "MULTIPLY", d, M(nt, "MULTIPLY", gain.outputs[0], 0.004)), pv.inputs["Emission Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


M_JELLY = jellyfish_material("Jellyfish")

# =====================================================================================
# 5. THE VEHICLE (metres; rig origin = SLS station 0 = core-engine exit plane ~z 0.8,
#    +Z = flight axis). NASA PD hardware from the library + a plain core tank lathe (the NASA
#    model ships the tank only as a 3ds Max file), Shuttle SRB stretched to 5 segments.
# =====================================================================================
with bpy.data.libraries.load(ARGS.lib, link=False) as (_src, _dst):
    _dst.objects = list(_src.objects)
LIB = {o.name: o for o in _dst.objects if o is not None}
print("[rocket2] library:", sorted(LIB))


def lib_obj(name, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), copy=False, new_name=None):
    src = LIB[name]
    ob = src.copy() if copy else src
    if copy:
        ob.name = new_name or (name + "_copy")
    for k, s in enumerate(ob.material_slots):
        mat = s.material
        if mat is not None and mat.name in SEM_MAT:
            ob.material_slots[k].material = SEM_MAT[mat.name]
        elif mat is not None and mat.name.split(".")[0] in SEM_MAT:
            ob.material_slots[k].material = SEM_MAT[mat.name.split(".")[0]]
    ob.location = loc
    ob.rotation_euler = rot
    return link(ob, parent)


CORE_TOP = 64.15          # LVSA base
LVSA_TOP = 72.2
ICPS_TOP = 76.8           # MSA top
SAJ_Z0, SAJ_Z1 = 76.7, 81.1
OGIVE_Z0 = 82.1
LAS_TOP = 95.5
SRB_X = 6.2               # booster axes at x = ±6.2 (from the NASA attach hardware)
R = CORE_R

RIG = empty("RocketRig")
S1 = empty("S1", parent=RIG)                 # core stage (+ LVSA, which stays with it)
S2 = empty("S2", parent=RIG)                 # ICPS + Orion
BL = empty("BoosterL", loc=(-SRB_X, 0, 0), parent=RIG)
BRt = empty("BoosterR", loc=(SRB_X, 0, 0), parent=RIG)
LASG = empty("LASGroup", parent=S2)          # launch abort tower + ogive: jettisoned on drop 4
PANELS = [empty(f"SAJPanel{k}", parent=S2) for k in range(3)]

core = lathe("CoreTank", [(3.55, 2.35), (3.9, 3.2), (R, 5.4), (R, CORE_TOP)], 96, M_FOAM, S1)
heatshield = MB()
heatshield.poly([(3.55 * math.cos(a), 3.55 * math.sin(a), 2.35) for a in np.linspace(0, 2 * math.pi, 49)[:-1]],
                [tuple(range(48))[::-1]])
heatshield.obj("CoreHeatShield", M_DARK, S1)
ENG = lib_obj("SLS_Engines", S1)
lib_obj("SLS_CoreHW", S1)
lib_obj("SLS_LVSA", S1)
lib_obj("SLS_ICPS", S2)
lib_obj("SLS_MSA", S2)
# ESM body + folded solar arrays (hidden under the SAJ panels until they jettison)
lathe("ESMBody", [(2.0, ICPS_TOP), (2.0, SAJ_Z1 - 0.2), (2.5, SAJ_Z1 + 0.6), (2.5, OGIVE_Z0)], 48, M_GOLD, S2)
cm = lib_obj("Orion_CM", S2, loc=(0, 0, OGIVE_Z0 - 0.2))
for k, pn in enumerate(PANELS):
    lib_obj(f"SLS_ESM{k}", pn)
lib_obj("SLS_Ogive", LASG)
lib_obj("SLS_LAS", LASG)
srb_tex = load_img(os.path.join(os.path.dirname(ARGS.lib), "SRB_ALL.png"))
M_SRB = srb_material("BoosterPaint", srb_tex)
SEM_MAT["SEM_srb"] = M_SRB
lib_obj("SRB_Body", BL)
lib_obj("SRB_Body", BRt, copy=True, new_name="SRB_BodyR")
lib_obj("SLS_SRBatt_L", BL, loc=(SRB_X, 0, 0))               # hardware is in vehicle coords
lib_obj("SLS_SRBatt_R", BRt, loc=(-SRB_X, 0, 0))
# RL10 vacuum nozzle under the ICPS (hidden in the LVSA until staging)
s2noz = lathe("S2Nozzle", [(0.35, 71.6), (0.5, 70.9), (0.85, 69.6), (1.05, 68.4)], 40, M_NOZ_IN, S2)


def engine_centres():
    me = ENG.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    low = co[co[:, 2] < 1.3]
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            q = low[(np.sign(low[:, 0]) == sx) & (np.sign(low[:, 1]) == sy)]
            out.append((float((q[:, 0].max() + q[:, 0].min()) / 2), float((q[:, 1].max() + q[:, 1].min()) / 2),
                        float(q[:, 2].min()), float((q[:, 0].max() - q[:, 0].min()) / 2)))
    return out


ENG_POS = engine_centres()
print("[rocket2] RS-25 exits", [tuple(round(v, 2) for v in e) for e in ENG_POS])
SRB_EXIT_Z = -0.6
PLUMES = []          # (object, kind, extra)


def plume_pair(tag, parent, kind, loc, base_r, length, shell_mat, vol_mat=None, vol_len=None):
    inner = lathe(f"Plume_{tag}_in", [(base_r * 0.55, 0.0), (base_r * 0.78, -0.08 * length), (base_r * 0.9, -0.3 * length),
                                      (base_r * 0.75, -0.65 * length), (base_r * 0.3, -length)],
                  48, shell_mat, parent, loc=loc)
    no_shadow(inner)
    PLUMES.append((inner, kind, "shell"))
    outer = None
    if vol_mat is not None:
        outer = lathe(f"Plume_{tag}_vol", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 32, vol_mat, parent,
                      loc=(loc[0], loc[1], loc[2] + 0.3))
        no_shadow(outer)
        PLUMES.append((outer, kind, "vol"))
    return inner, outer


for k, (ex, ey, ez, er) in enumerate(ENG_POS):
    plume_pair(f"rs{k}", S1, f"core{k}", (ex, ey, ez), er * 0.95, 22.0, M_RS25_SHELL)
plume_pair("bl", BL, "boost", (0, 0, SRB_EXIT_Z), 2.0, 34.0, M_SRB_SHELL, M_SRB_VOL)
plume_pair("br", BRt, "boost", (0, 0, SRB_EXIT_Z), 2.0, 34.0, M_SRB_SHELL, M_SRB_VOL)
vac = lathe("Plume_s2_vol", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 32, M_VAC_VOL, S2, loc=(0, 0, 68.2))
no_shadow(vac)
PLUMES.append((vac, "s2", "vol"))
vac_in = lathe("Plume_s2_in", [(1.0, 0.0), (1.5, -3.0), (2.3, -12.0), (1.8, -26.0), (0.4, -34.0)],
               48, M_VAC_SHELL, S2, loc=(0, 0, 68.3))
no_shadow(vac_in)
PLUMES.append((vac_in, "s2", "shell"))
LAS_PLUMES = []
for k in range(4):
    a_ = math.pi / 4 + k * math.pi / 2
    lp = lathe(f"LASPlume{k}", [(0.18, 0.0), (0.35, -0.6), (0.6, -3.0), (0.4, -6.5), (0.1, -9.0)], 24,
               M_SRB_SHELL, LASG, loc=(0.9 * math.cos(a_), 0.9 * math.sin(a_), OGIVE_Z0 + 9.2))
    lp.rotation_euler = (math.radians(32) * -math.sin(a_), math.radians(32) * math.cos(a_), 0.0)
    no_shadow(lp)
    LAS_PLUMES.append(lp)
jelly = lathe("Jellyfish", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 40, M_JELLY, RIG, loc=(0, 0, -10.0))
no_shadow(jelly)

# plume lights: a chain of blackbody points down each plume lights the steam, the pad, the
# tower, the vehicle and the clouds (research fix #2: the plume IS the key light)
PLIGHTS = []
for tag, par, zs in (("bl", BL, (-6.0, -26.0, -60.0)), ("br", BRt, (-6.0, -26.0, -60.0)),
                     ("core", S1, (-4.0, -16.0)), ("s2", S2, (60.0,))):
    for j, z in enumerate(zs):
        ld = bpy.data.lights.new(f"PlumeLight_{tag}{j}", "POINT")
        ld.color = blackbody(2600 if tag in ("bl", "br") else 2200)
        ld.shadow_soft_size = 3.0 + 3.0 * j
        ld.volume_factor = 1.0
        ld.use_shadow = tag in ("bl", "br") or j == 0
        lo = bpy.data.objects.new(f"PlumeLight_{tag}{j}", ld)
        lo.location = (0, 0, z)
        link(lo, par)
        PLIGHTS.append((lo, tag, j))
BOUNCE = bpy.data.lights.new("PlumeBounce", "AREA")
BOUNCE.shape = "DISK"
BOUNCE.size = 40.0
BOUNCE.color = blackbody(2400)
BOUNCE.use_shadow = False
BOUNCE_OB = link(bpy.data.objects.new("PlumeBounce", BOUNCE), S1)
BOUNCE_OB.location = (0.0, -30.0, -40.0)
BOUNCE_OB.rotation_euler = (math.radians(-135), 0, 0)     # faces up and toward the vehicle (+Y,+Z)


# --- GN helpers ------------------------------------------------------------------------
def gn_tree(name, inputs):
    """New GN tree with Geometry in/out + the given [(name, socket_type, default)] inputs."""
    t = bpy.data.node_groups.new(name, "GeometryNodeTree")
    t.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    socks = {}
    for nm, ty, dv in inputs:
        s = t.interface.new_socket(name=nm, in_out="INPUT", socket_type=ty)
        if dv is not None:
            s.default_value = dv
        socks[nm] = s
    t.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return t, socks, t.nodes.new("NodeGroupInput"), t.nodes.new("NodeGroupOutput")


def g_rand(t, gi, lo, hi, salt, vec=False, seed_sock="Seed", seed_scale=1):
    """5.2 Random Value: ONE Min/Max pair retyped by data_type; seed = Seed input + salt."""
    r = t.nodes.new("FunctionNodeRandomValue")
    r.data_type = "FLOAT_VECTOR" if vec else "FLOAT"
    want = "VECTOR" if vec else "VALUE"
    ins = [x for x in r.inputs if x.enabled and x.type == want]
    ins[0].default_value = lo
    ins[1].default_value = hi
    if seed_sock is not None:
        ad = t.nodes.new("FunctionNodeIntegerMath")
        ad.operation = "ADD"
        t.links.new(gi.outputs[seed_sock], ad.inputs[0])
        ad.inputs[1].default_value = salt
        t.links.new(ad.outputs[0], r.inputs["Seed"])
    else:
        r.inputs["Seed"].default_value = salt
    return [x for x in r.outputs if x.enabled and x.type == want][0]


def g_math(t, op, a, b=None, c=None, vec=False):
    n = t.nodes.new("ShaderNodeVectorMath" if vec else "ShaderNodeMath")
    n.operation = op
    for k, x in enumerate((a, b, c)):
        if x is None:
            continue
        if isinstance(x, (int, float, tuple)):
            n.inputs[k].default_value = x
        else:
            t.links.new(x, n.inputs[k])
    if vec:
        return n.outputs["Value" if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else "Vector"]
    return n.outputs[0]


def gn_obj(name, tree, parent=None, values=None):
    ob = mesh_obj(name, [], [], None, parent)
    md = ob.modifiers.new(name, "NODES")
    md.node_group = tree
    for nm, v in (values or {}).items():
        it = [i for i in tree.interface.items_tree if i.in_out == "INPUT" and i.name == nm][0]
        getattr(md.properties.inputs, it.identifier).value = v
    return ob, md


def gn_input_path(md, tree, nm):
    it = [i for i in tree.interface.items_tree if i.in_out == "INPUT" and i.name == nm][0]
    return getattr(md.properties.inputs, it.identifier).path_from_id("value")


# --- ice flakes shed from the cryogenic tanks (research fix #7: no point glints) ---------
M_ICE = principled("IceFlake", (0.80, 0.85, 0.90), 0.55, 0.0)
M_ICE.node_tree.nodes["Principled BSDF"].inputs["Subsurface Weight"].default_value = 0.3


def flake_tree():
    """Stateless ice shedding: Count flakes born on the core/ICPS skin (seed = event index),
    position = p0 + v*age + ½g·age², tumbling spin; flat white flakes, not glints."""
    t, s, gi, go = gn_tree("IceFlakes", [("Age", "NodeSocketFloat", 0.0), ("Seed", "NodeSocketInt", 0),
                                         ("Count", "NodeSocketInt", 60), ("Z0", "NodeSocketFloat", 44.0),
                                         ("Z1", "NodeSocketFloat", 63.0),
                                         ("G", "NodeSocketFloat", 4.9)])
    pts = t.nodes.new("GeometryNodePoints")
    t.links.new(gi.outputs["Count"], pts.inputs["Count"])
    ang = g_rand(t, gi, -2.6, -0.5, 11)                               # mostly on the camera side
    zr = g_rand(t, gi, 0.0, 1.0, 23)
    z = g_math(t, "MULTIPLY_ADD", zr, g_math(t, "SUBTRACT", gi.outputs["Z1"], gi.outputs["Z0"]), gi.outputs["Z0"])
    cx = g_math(t, "MULTIPLY", g_math(t, "COSINE", ang), R * 1.02)
    sy = g_math(t, "MULTIPLY", g_math(t, "SINE", ang), R * 1.02)
    p0 = t.nodes.new("ShaderNodeCombineXYZ")
    t.links.new(cx, p0.inputs[0])
    t.links.new(sy, p0.inputs[1])
    t.links.new(z, p0.inputs[2])
    rv = g_rand(t, gi, (-1.0, -1.0, -3.0), (1.0, 1.0, 0.5), 37, vec=True)
    radial = g_math(t, "MULTIPLY", p0.outputs[0], (0.35, 0.35, 0.0), vec=True)
    vel = g_math(t, "ADD", radial, rv, vec=True)
    age = gi.outputs["Age"]
    pos = g_math(t, "ADD", p0.outputs[0], g_math(t, "SCALE", vel, age, vec=True), vec=True)
    fall = t.nodes.new("ShaderNodeVectorMath")
    fall.operation = "SCALE"
    fall.inputs[0].default_value = (0.0, 0.0, -1.0)                    # ½ g (flakes flutter: drag)
    t.links.new(g_math(t, "MULTIPLY", g_math(t, "MULTIPLY", age, age), gi.outputs["G"]), fall.inputs["Scale"])
    pos = g_math(t, "ADD", pos, fall.outputs["Vector"], vec=True)
    sp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], sp.inputs["Geometry"])
    t.links.new(pos, sp.inputs["Position"])
    cube = t.nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (1.0, 0.7, 0.05)
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(sp.outputs[0], inst.inputs["Points"])
    t.links.new(cube.outputs["Mesh"], inst.inputs["Instance"])
    r0 = g_rand(t, gi, (0, 0, 0), (6.3, 6.3, 6.3), 51, vec=True)
    spin = g_rand(t, gi, (-9, -9, -9), (9, 9, 9), 53, vec=True)
    t.links.new(g_math(t, "ADD", r0, g_math(t, "SCALE", spin, age, vec=True), vec=True), inst.inputs["Rotation"])
    t.links.new(g_rand(t, gi, 0.06, 0.45, 63), inst.inputs["Scale"])
    sm = t.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = M_ICE
    t.links.new(inst.outputs[0], sm.inputs["Geometry"])
    t.links.new(sm.outputs[0], go.inputs["Geometry"])
    return t


FLAKE_T = flake_tree()
flakes, flakes_md = gn_obj("IceFlakes", FLAKE_T, RIG, {"Count": 70})
flakes2, flakes2_md = gn_obj("IceFlakesLiftoff", FLAKE_T, RIG, {"Count": 160, "Z0": 20.0, "Z1": 63.0, "G": 7.4})
no_shadow(flakes)
no_shadow(flakes2)


# --- hydrogen burn-off igniters: sparks fired radially under the core engines -------------
def spark_tree():
    t, s, gi, go = gn_tree("BurnoffSparks", [("Time", "NodeSocketFloat", 0.0), ("Count", "NodeSocketInt", 500),
                                             ("Amount", "NodeSocketFloat", 1.0)])
    pts = t.nodes.new("GeometryNodePoints")
    t.links.new(gi.outputs["Count"], pts.inputs["Count"])
    ph0 = g_rand(t, gi, 0.0, 1.0, 3, seed_sock=None)
    life = 0.9
    tt = g_math(t, "FRACT", g_math(t, "ADD", g_math(t, "DIVIDE", gi.outputs["Time"], life), ph0))
    age = g_math(t, "MULTIPLY", tt, life)
    cyc = g_math(t, "FLOOR", g_math(t, "ADD", g_math(t, "DIVIDE", gi.outputs["Time"], life), ph0))
    idx = t.nodes.new("GeometryNodeInputIndex")
    sd = g_math(t, "ADD", g_math(t, "MULTIPLY", cyc, 7919.0), idx.outputs[0])
    def rnd(lo, hi, salt, vec=False):
        r = t.nodes.new("FunctionNodeRandomValue")
        r.data_type = "FLOAT_VECTOR" if vec else "FLOAT"
        want = "VECTOR" if vec else "VALUE"
        ins = [x for x in r.inputs if x.enabled and x.type == want]
        ins[0].default_value, ins[1].default_value = lo, hi
        r.inputs["Seed"].default_value = salt
        t.links.new(sd, r.inputs["ID"])
        return [x for x in r.outputs if x.enabled and x.type == want][0]
    # emitter = one of the 4 engine exits (by index), launched sideways/down with a fan spread
    which = g_math(t, "FLOOR", rnd(0.0, 3.999, 5))
    exits = ENG_POS
    ex = t.nodes.new("ShaderNodeCombineXYZ")
    # piecewise select of the 4 exits via sums of (which == k) * coord
    for axis in range(2):
        acc = None
        for k in range(4):
            sel = g_math(t, "COMPARE", which, float(k), 0.1)
            term = g_math(t, "MULTIPLY", sel, exits[k][axis])
            acc = term if acc is None else g_math(t, "ADD", acc, term)
        t.links.new(acc, ex.inputs[axis])
    ex.inputs[2].default_value = exits[0][2] - 0.3
    az = rnd(0.0, 6.283, 7)
    sp_ = rnd(6.0, 16.0, 9)
    vx = g_math(t, "MULTIPLY", g_math(t, "COSINE", az), sp_)
    vy = g_math(t, "MULTIPLY", g_math(t, "SINE", az), sp_)
    vz = rnd(-6.0, 2.0, 13)
    vel = t.nodes.new("ShaderNodeCombineXYZ")
    t.links.new(vx, vel.inputs[0])
    t.links.new(vy, vel.inputs[1])
    t.links.new(vz, vel.inputs[2])
    pos = g_math(t, "ADD", ex.outputs[0], g_math(t, "SCALE", vel.outputs[0], age, vec=True), vec=True)
    fall = t.nodes.new("ShaderNodeVectorMath")
    fall.operation = "SCALE"
    fall.inputs[0].default_value = (0.0, 0.0, -4.9)
    t.links.new(g_math(t, "MULTIPLY", age, age), fall.inputs["Scale"])
    pos = g_math(t, "ADD", pos, fall.outputs["Vector"], vec=True)
    setp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], setp.inputs["Geometry"])
    t.links.new(pos, setp.inputs["Position"])
    ico = t.nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 0.05
    ico.inputs["Subdivisions"].default_value = 1
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(setp.outputs[0], inst.inputs["Points"])
    t.links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    alive = g_math(t, "MULTIPLY", g_math(t, "LESS_THAN", rnd(0.0, 1.0, 17), gi.outputs["Amount"]),
                   g_math(t, "SUBTRACT", 1.0, g_math(t, "POWER", tt, 3.0)))
    t.links.new(alive, inst.inputs["Scale"])
    sm = t.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = emissive("Spark", blackbody(2300), 120.0)
    t.links.new(inst.outputs[0], sm.inputs["Geometry"])
    t.links.new(sm.outputs[0], go.inputs["Geometry"])
    return t


SPARK_TREE = spark_tree()
sparks, sparks_md = gn_obj("BurnoffSparks", SPARK_TREE, S1)
no_shadow(sparks)


# --- max-Q condensation cone: a conical shroud of fog hugging the upper stack (fix #8) ----
def vcone_material(name):
    m, nt = new_mat(name)
    gain = value_node(nt, "Gain", 0.0)
    evo = value_node(nt, "Evolve", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]                                     # unit cylinder: r<=1, z 0 (front) → -1 (aft)
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    v = M(nt, "MULTIPLY", sp.outputs[2], -1.0, clamp=True)
    r = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    rs = M(nt, "ADD", 0.36, M(nt, "MULTIPLY", M(nt, "POWER", v, 0.45), 0.62))    # cone surface radius
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 4.0, "Detail": 8.0, "Roughness": 0.65})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.12)), nz.inputs["Vector"])           # streaks along the flow
    lk(nt, evo.outputs[0], nz.inputs["W"])
    rn = M(nt, "ADD", r, M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz.outputs["Fac"], 0.5), 0.12))
    outer = M(nt, "MULTIPLY", M(nt, "SUBTRACT", rs, rn), 7.0, clamp=True)          # sharp shock edge
    inner = M(nt, "MULTIPLY", M(nt, "SUBTRACT", rn, 0.28), 5.0, clamp=True)        # hugs the body, not inside it
    front = M(nt, "MULTIPLY", M(nt, "DIVIDE", v, 0.03), 1.0, clamp=True)
    aft = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, v, clamp=True), 2.2)
    d = M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "MULTIPLY", outer, inner), M(nt, "MULTIPLY", front, aft)),
          M(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 3.2, M(nt, "SUBTRACT", -1.0, M(nt, "MULTIPLY", v, 0.8)), clamp=True))
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": (0.95, 0.96, 1.0, 1), "Anisotropy": 0.55,
                                                    "Emission Color": (0.85, 0.9, 1.0, 1)})
    lk(nt, M(nt, "MULTIPLY", d, M(nt, "MULTIPLY", gain.outputs[0], 14.0)), pv.inputs["Density"])
    lk(nt, M(nt, "MULTIPLY", d, M(nt, "MULTIPLY", gain.outputs[0], 0.6)), pv.inputs["Emission Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


M_VCONE = vcone_material("VaporCone")
vcone = lathe("VaporCone", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 48, M_VCONE, RIG, loc=(0, 0, OGIVE_Z0 + 1.0))
vcone.scale = (8.0, 8.0, 34.0)
vcone_b = []
for side, bem in ((-1, BL), (1, BRt)):       # collars on the booster nose cones too
    c = lathe(f"VaporConeB{'LR'[side > 0]}", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 32, M_VCONE, bem, loc=(0, 0, 53.0))
    c.scale = (4.2, 4.2, 14.0)
    vcone_b.append(c)
for o in [vcone] + vcone_b:
    no_shadow(o)


# --- VDB helpers (JangaFX CC0 sims: Z-up, voxel units, frame k = file k) -----------------
GROUND_ANCHOR = Vector((-4.0, 8.0, -160.0))       # the sim's floor/centre in its own voxel units
CHIMNEY_ANCHOR = Vector((-8.0, -80.0, -192.0))   # chimney mouth


def anchored(loc, rot, scale, anchor):
    """Object location that puts the sim's `anchor` point (voxel units) at world `loc`."""
    from mathutils import Euler
    R_ = Euler(rot).to_matrix()
    s_ = Vector(scale) if not isinstance(scale, (int, float)) else Vector((scale,) * 3)
    a = Vector((anchor.x * s_.x, anchor.y * s_.y, anchor.z * s_.z))
    return Vector(loc) - R_ @ a


def vdb_seq(name, folder, scale, loc=(0, 0, 0), rot=(0, 0, 0), parent=None, mat=None, mode="EXTEND"):
    files = sorted(glob.glob(os.path.join(folder, "*.vdb")))
    vol = bpy.data.volumes.new(name)
    vol.filepath = files[0]
    vol.is_sequence = len(files) > 1
    vol.frame_start = 1
    vol.frame_duration = len(files)
    vol.sequence_mode = mode
    if mat is not None:
        vol.materials.append(mat)
    ob = bpy.data.objects.new(name, vol)
    ob.location = loc
    ob.rotation_euler = rot
    ob.scale = (scale,) * 3
    link(ob, parent)
    return ob, len(files)


def vdb_play(ob, nfiles, start, speed=1.0, first=0):
    """Bake frame_offset so file index = first + (f - start) * speed, clamped to the sim.
    MEASURED (5.2.1): the sequence mode clamps the scene frame to [1, duration] BEFORE the offset
    is added (file = clamp(f - frame_start + 1) + offset), so CLIP mode shows nothing past the
    sim length — use EXTEND and offset against the clamped frame."""
    assert ob.data.sequence_mode == "EXTEND"
    f = IDX + 1
    pre = np.clip(f - ob.data.frame_start + 1, 1, ob.data.frame_duration)
    k = np.clip(np.floor(first + (IDX - start) * speed).astype(np.int64), 0, nfiles - 1)
    bake(ob.data, "frame_offset", (k - pre).astype(np.float64), const=True)
    return k


JANGA = os.path.join(AS, "jangafx")
GROUND_X = os.path.join(JANGA, "x_GroundExplosionVDB", "ground_explosion", "ground_explosion_VDB")
CHIMNEY = os.path.join(JANGA, "x_IndustrialChimneySmokeVDB", "industrial_chimney_smoke", "industrial_chimney_smoke_VDB")
CLOUDS = os.path.join(JANGA, "x_CloudPackVDB", "CloudPack", "CloudPackVDB")


def smoke_material(name, color=(0.92, 0.92, 0.93), dens=1.0, aniso=0.35, glow_grid="flames",
                   glow_col=blackbody(1900), plume_lit=True):
    """VDB steam/smoke: density grid × keyed 'Density'; lit FROM INSIDE by the plume:
    keyed 'Glow' × (flames grid + a falloff from the vehicle axis 'PlumeXY'/'PlumeZ' world position),
    so the core of the cloud glows orange and the far side stays grey-blue (research §1.2)."""
    m, nt = new_mat(name)
    dn = value_node(nt, "Density", dens)
    glow = value_node(nt, "Glow", 0.0)
    pz = value_node(nt, "PlumeZ", 0.0)
    at = nd(nt, "ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="density")
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": tuple(color) + (1,), "Anisotropy": aniso})
    pv.inputs["Density Attribute"].default_value = ""
    d = M(nt, "MULTIPLY", at.outputs["Fac"], dn.outputs[0])
    lk(nt, d, pv.inputs["Density"])
    e = None
    if glow_grid:
        fl = nd(nt, "ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name=glow_grid)
        e = M(nt, "MULTIPLY", fl.outputs["Fac"], 1.2)
    if plume_lit:
        gp = nd(nt, "ShaderNodeNewGeometry")
        sp = nd(nt, "ShaderNodeSeparateXYZ")
        lk(nt, gp.outputs["Position"], sp.inputs[0])
        rr = M(nt, "SQRT", M(nt, "ADD", M(nt, "MULTIPLY", sp.outputs[0], sp.outputs[0]),
                             M(nt, "MULTIPLY", sp.outputs[1], sp.outputs[1])))
        dz = M(nt, "ABSOLUTE", M(nt, "SUBTRACT", sp.outputs[2], pz.outputs[0]))
        fall = M(nt, "EXPONENT", M(nt, "MULTIPLY", M(nt, "ADD", rr, M(nt, "MULTIPLY", dz, 0.8)), -1.0 / 20.0))
        fall = M(nt, "MULTIPLY", fall, d)
        e = fall if e is None else M(nt, "ADD", e, fall)
    if e is not None:
        lk(nt, M(nt, "MULTIPLY", e, glow.outputs[0]), pv.inputs["Emission Strength"])
        pv.inputs["Emission Color"].default_value = tuple(glow_col) + (1,)
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


# LOX/LH2 boil-off vapour: cold, dense — it SINKS and pours down the vehicle (CC0 chimney sim
# turned upside down: its drift axis +Y → world -Z), white and thin
M_VENT = smoke_material("VentVapour", (0.95, 0.96, 1.0), dens=0.35, aniso=0.5, glow_grid=None, plume_lit=False)
VENTS = []
for k, (z, ang, s_) in enumerate(((CORE_TOP + 1.0, -1.25, 0.075), (LVSA_TOP + 2.5, -1.9, 0.055),
                                  (52.5, 2.6, 0.07))):
    rr_ = (R + 0.6) if z < CORE_TOP + 2 else 3.0
    vrot = (math.radians(-90), 0, ang + math.pi / 2)             # sim drift (+Y) → down (−Z): cold vapour sinks
    vloc = anchored((rr_ * math.cos(ang), rr_ * math.sin(ang), z), vrot, s_, CHIMNEY_ANCHOR)
    o, nfi = vdb_seq(f"Vent{k}", CHIMNEY, s_, loc=vloc, rot=vrot, parent=RIG, mat=M_VENT, mode="REPEAT")
    VENTS.append((o, nfi))


def volume_material(name, color=(0.9, 0.9, 0.92), shape="SPHERE", nscale=2.0, detail=6.0,
                    emis_color=(1.0, 0.45, 0.12), aniso=0.3, low_glow=True, sharp=2.2, namp=0.9):
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
    else:  # COLUMN: unit cylinder r<=1 along z in [0,1]; pyroclastic: two octaves, sharp edge
        rr = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
        h = M(nt, "MAXIMUM", sepo.outputs[2], 0.0)
        nz2 = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": nscale * 5.5, "Detail": 4.0})
        lk(nt, ob, nz2.inputs["Vector"])
        lk(nt, evo.outputs[0], nz2.inputs["W"])
        fbm = M(nt, "ADD", M(nt, "SUBTRACT", n, 0.5), M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz2.outputs["Fac"], 0.5), 0.6))
        mask = M(nt, "SUBTRACT", M(nt, "ADD", M(nt, "SUBTRACT", 1.0, rr), M(nt, "MULTIPLY", fbm, namp * 2.0)),
                 0.1, clamp=True)
        mask = M(nt, "MINIMUM", M(nt, "MULTIPLY", mask, sharp), 1.0)
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


# =====================================================================================
# 6. THE PAD + WORLD BUILD (research fix #5): raised hardstand with a flame trench, NASA
#    Mobile Launcher, flame deflector, lightning towers + catenary wires, sphere tanks, water
#    tower, xenon floods; numpy heightfield terrain (scrub flats, coastline + Atlantic to the
#    east, a lagoon between the long-lens camera and the pad); Poly Haven scrub scatter.
# =====================================================================================
PAD_TOP = 14.0                                   # hardstand elevation (m)
ML_DECK = PAD_TOP + 6.98                         # deck top (the NASA ML's legs stand on the pad)
PAD_Z = ML_DECK + 1.2                            # vehicle station 0 above the deck
TRENCH_W = 9.5                                   # half width
_rng = np.random.default_rng(1234)
_lat = _rng.random((257, 257)).astype(np.float64)


def vnoise(x, y, scale):
    """Smooth value noise (0..1) from a tiled 256² lattice — numpy, deterministic."""
    u, v = x / scale, y / scale
    i0, j0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - i0, v - j0
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    i0 %= 256
    j0 %= 256
    a, b = _lat[i0, j0], _lat[i0 + 1, j0]
    c, d = _lat[i0, j0 + 1], _lat[i0 + 1, j0 + 1]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def fbm(x, y, scale, octaves=5):
    s, amp, tot = 0.0, 1.0, 0.0
    for k in range(octaves):
        s = s + amp * vnoise(x + 37.1 * k, y - 11.7 * k, scale / (2 ** k))
        tot += amp
        amp *= 0.5
    return s / tot


CAM_DIR = Vector((-0.34, -0.94, 0.0)).normalized()       # from the pad toward the long-lens site


def height(x, y):
    """Terrain height (m) + masks: (h, sand, pad, water)."""
    land = 1.6 + 2.2 * (fbm(x, y, 900.0) - 0.5) + 0.7 * (fbm(x, y, 70.0) - 0.5)
    # Atlantic to the east
    cx = 1350.0 + 420.0 * (fbm(0.0 * x, y, 2600.0, 3) - 0.5)
    sea = np.clip((x - cx) / 90.0, 0.0, 1.0)
    # lagoon band between the pad and the long-lens camera (SSW)
    s = x * CAM_DIR.x + y * CAM_DIR.y
    t = x * -CAM_DIR.y + y * CAM_DIR.x
    sm = s + 160.0 * (fbm(t, 0.0 * t, 1100.0, 3) - 0.5)
    lag = np.clip((sm - 1450.0) / 70.0, 0, 1) * np.clip((2950.0 - sm) / 70.0, 0, 1)
    water = np.maximum(sea, lag)
    h = land * (1 - water) + (-3.0) * water
    sand = np.clip(1.0 - np.abs(x - cx + 30) / 70.0, 0, 1) + np.clip(1.0 - (np.minimum(np.abs(sm - 1450), np.abs(sm - 2950))) / 45.0, 0, 1) * 0.7
    # the pad mound (superellipse flat top ±95 m, slopes out to ~±175 m)
    q = ((np.abs(x) / 95.0) ** 6 + (np.abs(y) / 95.0) ** 6) ** (1 / 6)
    mound = PAD_TOP * (1 - smoothstep((q - 1.0) / 0.85))
    padm = (q < 1.0).astype(np.float64)
    h = np.maximum(h, mound)
    # flame trench: |y| < TRENCH_W from under the hole out of the east slope
    tf = 3.0 - 2.0 * np.clip(x / 170.0, 0, 1)
    wall = np.clip((TRENCH_W - np.abs(y)) / 1.2, 0, 1) * np.clip((x + 13.0) / 1.2, 0, 1)
    h = h * (1 - wall) + np.minimum(h, tf) * wall
    return h, np.clip(sand, 0, 1), padm, water


def heightfield(name, size, n, mat, z_bias=0.0, hole=None):
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    H, sand, padm, water = height(X, Y)
    if hole is not None:                                       # sink under the near tile (no z-fight)
        inside = (np.abs(X) < hole) & (np.abs(Y) < hole)
        H = np.where(inside, H - 1.5, H)
    H = H + z_bias
    me = bpy.data.meshes.new(name)
    nv = n * n
    me.vertices.add(nv)
    me.vertices.foreach_set("co", np.stack([X.ravel(), Y.ravel(), H.ravel()], 1).astype(np.float32).ravel())
    ii = np.arange(n - 1)
    I, J = np.meshgrid(ii, ii, indexing="ij")
    a = (I * n + J).ravel()
    quads = np.stack([a, a + n, a + n + 1, a + 1], 1)
    me.loops.add(quads.size)
    me.polygons.add(len(quads))
    me.loops.foreach_set("vertex_index", quads.ravel().astype(np.int32))
    me.polygons.foreach_set("loop_start", (np.arange(len(quads)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    for nm, arr in (("sand", sand), ("pad", padm), ("water", water)):
        at = me.attributes.new(nm, "FLOAT", "POINT")
        at.data.foreach_set("value", arr.ravel().astype(np.float32))
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me))


def terrain_material():
    """Scrub flats (procedural greens/browns), sand (Poly Haven coast_sand_01), pad top concrete
    (concrete_floor_02), grassy mound slopes; keyed 'Wet' darkens the pad after the deluge."""
    m, nt = new_mat("Terrain")
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0})
    tc = nd(nt, "ShaderNodeTexCoord")
    p = tc.outputs["Object"]
    a_s = nd(nt, "ShaderNodeAttribute", attribute_name="sand")
    a_p = nd(nt, "ShaderNodeAttribute", attribute_name="pad")
    g1 = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.03, "Detail": 8.0, "Roughness": 0.65})
    lk(nt, p, g1.inputs["Vector"])
    g2 = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.4, "Detail": 6.0, "Roughness": 0.7})
    lk(nt, p, g2.inputs["Vector"])
    scrub = mix_col(nt, g1.outputs["Fac"], (0.030, 0.038, 0.016), (0.085, 0.075, 0.040))
    scrub = mix_col(nt, M(nt, "MULTIPLY", M(nt, "SUBTRACT", g2.outputs["Fac"], 0.35, clamp=True), 1.2), scrub,
                    (0.11, 0.10, 0.065))
    sd = tex_box(nt, ph("coast_sand_01", "diff"), 3.0, coord=p)
    col = mix_col(nt, a_s.outputs["Fac"], scrub, col_mul(nt, sd.outputs["Color"], 0.9))
    cd = tex_box(nt, ph("concrete_floor_02", "diff"), 4.0, coord=p)
    wet = value_node(nt, "Wet", 0.0)
    padc = col_mul(nt, cd.outputs["Color"], M(nt, "SUBTRACT", 0.85, M(nt, "MULTIPLY", wet.outputs[0], 0.4)))
    col = mix_col(nt, a_p.outputs["Fac"], col, padc)
    lk(nt, col, b.inputs["Base Color"])
    rgh = M(nt, "ADD", 0.9, M(nt, "MULTIPLY", a_p.outputs["Fac"], M(nt, "MULTIPLY", wet.outputs[0], -0.75)))
    lk(nt, rgh, b.inputs["Roughness"])
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.25, "Distance": 0.2})
    lk(nt, g2.outputs["Fac"], bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    finish(nt, b)
    return m


def water_material():
    """Lagoon + Atlantic: dark, low roughness, wind ripples; reflects the plume and the floods."""
    m, nt = new_mat("Water")
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.006, 0.012, 0.016, 1), "Roughness": 0.06,
                                                 "IOR": 1.33})
    tc = nd(nt, "ShaderNodeTexCoord")
    w1 = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 0.12, "Detail": 6.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", tc.outputs["Object"], (1.0, 2.5, 1.0)), w1.inputs["Vector"])
    w1.name = "Waves"
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.18, "Distance": 0.25})
    lk(nt, w1.outputs["Fac"], bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    finish(nt, b)
    return m


M_TERRAIN = terrain_material()
M_WATER = water_material()
terr_near = heightfield("TerrainNear", 1300.0, 521, M_TERRAIN)
terr_far = heightfield("TerrainFar", 24000.0, 481, M_TERRAIN, hole=640.0)
sea_me = MB()
sea_me.poly([(-30000, -30000, 0.0), (30000, -30000, 0.0), (30000, 30000, 0.0), (-30000, 30000, 0.0)], [(0, 1, 2, 3)])
sea = sea_me.obj("Water", M_WATER)

# Mobile Launcher (NASA, PD): tower to the north (+Y), exhaust hole over the trench
ML = lib_obj("ML", None, loc=(0, 0, ML_DECK), rot=(0, 0, math.radians(90)))
# flame deflector: a wedge under the hole that turns the exhaust east (+X) down the trench
fd = MB()
fd.poly([(-12, -TRENCH_W, 3.0), (-12, TRENCH_W, 3.0), (-12, TRENCH_W, PAD_TOP - 1.0), (-12, -TRENCH_W, PAD_TOP - 1.0),
         (14, -TRENCH_W, 3.0), (14, TRENCH_W, 3.0)],
        [(0, 1, 2, 3), (3, 2, 5, 4), (0, 3, 4), (1, 5, 2), (0, 4, 5, 1)])
fd.obj("FlameDeflector", M_CONCRETE_DK)
tw = MB()                                       # trench walls (dark, sooty concrete) to hide the heightfield slope
for sy in (-1, 1):
    tw.box((75.0, sy * (TRENCH_W + 0.6), 7.0), (180.0, 1.2, 10.0))
tw.obj("TrenchWalls", M_CONCRETE_DK)
# lightning protection: three 180 m lattice towers with catenary wires (kept out of the
# foreground of the key shots: north/west/east of the pad)
LT_POS = [(-150.0, 150.0), (160.0, 140.0), (10.0, 230.0)]
lt = MB()
for (x0, y0) in LT_POS:
    pts = [(x0 + dx, y0 + dy) for dx, dy in ((-2.5, -2.5), (2.5, -2.5), (2.5, 2.5), (-2.5, 2.5))]
    for (x, y) in pts:
        lt.beam((x, y, 0), (x0 + (x - x0) * 0.35, y0 + (y - y0) * 0.35, 180), 0.45)
    for zz in np.arange(6.0, 180.0, 7.0):
        f0, f1 = 1 - 0.65 * zz / 180, 1 - 0.65 * (zz + 7) / 180
        for k in range(4):
            (xa, ya), (xb, yb) = pts[k], pts[(k + 1) % 4]
            lt.beam((x0 + (xa - x0) * f0, y0 + (ya - y0) * f0, zz), (x0 + (xb - x0) * f1, y0 + (yb - y0) * f1, zz + 7), 0.18)
    lt.beam((x0, y0, 180), (x0, y0, 195), 0.35)
lt.obj("LightningTowers", M_ML)
for a_, b_ in ((0, 1), (1, 2), (2, 0)):
    p0, p1 = Vector((*LT_POS[a_], 192.0)), Vector((*LT_POS[b_], 192.0))
    cu = bpy.data.curves.new(f"Catenary{a_}{b_}", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.06
    spl = cu.splines.new("POLY")
    pts_ = [p0.lerp(p1, t) - Vector((0, 0, 38.0 * 4 * t * (1 - t))) for t in np.linspace(0, 1, 24)]
    spl.points.add(len(pts_) - 1)
    for k, pp in enumerate(pts_):
        spl.points[k].co = (pp.x, pp.y, pp.z, 1.0)
    cu.materials.append(M_DARK)
    link(bpy.data.objects.new(f"Catenary{a_}{b_}", cu))
# LH2 + LOX sphere tanks and the water tower (simple shapes, they live at 250+ m)
M_TANKW = dielectric("TankWhite", (0.70, 0.70, 0.69), 0.45, tile=3.0)
uv_sphere("LH2Sphere", 12.5, 48, 24, M_TANKW, loc=(-260.0, 230.0, 14.5))
uv_sphere("LOXSphere", 10.0, 48, 24, M_TANKW, loc=(270.0, 250.0, 12.0))
wt = MB()
for k in range(6):
    a = 2 * math.pi * k / 6
    wt.beam((-330 + 8 * math.cos(a), -60 + 8 * math.sin(a), 0), (-330 + 4 * math.cos(a), -60 + 4 * math.sin(a), 70), 0.8)
wt.obj("WaterTowerLegs", M_ML)
uv_sphere("WaterTowerTank", 11.0, 32, 16, M_TANKW, loc=(-330, -60, 80))
# xenon floods: four banks on 30 m masts ~220 m out, hot white pools on the stack
FLOODS = []
FLOOD_POS = [(-170, -150, 30, 30), (180, -160, 30, 60), (200, 120, 30, 85), (-60, -230, 30, 45),
             (-230, 40, 30, 70), (90, -250, 30, 20)]            # (x, y, mast height, aim height above PAD_Z)
fl = MB()
for (x, y, z, _a) in FLOOD_POS:
    fl.beam((x, y, 0), (x, y, z), 0.9)
fl.obj("FloodMasts", M_ML)
for k, (x, y, z, aim) in enumerate(FLOOD_POS):
    panel = no_shadow(cube_obj(f"FloodPanel{k}", M_LAMP, loc=(x, y, z + 1.5), scale=(2.2, 0.3, 1.2)))
    d = Vector((0, 0, PAD_Z + aim)) - Vector((x, y, z))
    panel.rotation_euler = d.to_track_quat("Y", "Z").to_euler()
    FLOODS.append(panel)
    ld = bpy.data.lights.new(f"Flood{k}", "SPOT")
    ld.spot_size = math.radians(16)
    ld.spot_blend = 0.45
    ld.color = (0.88, 0.92, 1.0)                 # xenon ~6000 K
    ld.shadow_soft_size = 0.8
    ld.volume_factor = 0.6
    lo = link(bpy.data.objects.new(f"Flood{k}", ld))
    lo.location = (x, y, z + 1.5)
    lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    FLOODS.append(lo)
# sodium work lights up the mobile-launcher tower (every few levels) + on the deck
WORKL = []
M_SODIUM = emissive("SodiumLamp", blackbody(2100), 1.0, objcolor_gain=True)
for k, zz in enumerate(np.arange(ML_DECK + 6.0, ML_DECK + 108.0, 8.5)):
    for sx in (-1, 1):
        x, y = 4.7 + sx * 6.5, 14.7 - 7.2
        o = no_shadow(cube_obj(f"WorkLamp{k}_{sx}", M_SODIUM, loc=(x, y, zz), scale=(0.35, 0.35, 0.2)))
        WORKL.append(o)
    ld = bpy.data.lights.new(f"WorkL{k}", "POINT")
    ld.color = blackbody(2100)
    ld.shadow_soft_size = 1.0
    ld.use_shadow = False
    ld.volume_factor = 0.3
    lo = link(bpy.data.objects.new(f"WorkL{k}", ld))
    lo.location = (4.7, 14.7 - 8.5, zz)
    WORKL.append(lo)
for k, (x, y) in enumerate(((-17.0, -12.0), (17.0, -12.0), (-17.0, 10.0), (17.0, 10.0))):
    ld = bpy.data.lights.new(f"DeckL{k}", "POINT")
    ld.color = blackbody(2300)
    ld.shadow_soft_size = 2.0
    ld.use_shadow = False
    lo = link(bpy.data.objects.new(f"DeckL{k}", ld))
    lo.location = (x, y, ML_DECK + 3.0)
    WORKL.append(lo)
# sky searchlights (the Artemis-style light show) + red aviation beacons on the towers
SEARCH = []
for k, (x, y) in enumerate([(-420, -300), (330, -380), (60, -520)]):
    ld = bpy.data.lights.new(f"Search{k}", "SPOT")
    ld.spot_size = math.radians(1.6)
    ld.spot_blend = 0.15
    ld.color = (0.82, 0.88, 1.0)
    ld.shadow_soft_size = 0.2
    ld.use_shadow = False
    ld.volume_factor = 1.0
    lo = link(bpy.data.objects.new(f"Search{k}", ld))
    lo.location = (x, y, 2.0)
    SEARCH.append(lo)
BEACONS = []
for k, (x, y) in enumerate(LT_POS + [(4.7 * 1.0, 14.7 * 1.0)]):
    z = 196.0 if k < 3 else ML_DECK + 112.0
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
# distant town / road lights along the coast and across the lagoon (warm sodium + cool LED)
rng = np.random.default_rng(5)
dl = MB()
for k in range(700):
    a = rng.uniform(-math.pi, math.pi)
    rr = rng.uniform(3200, 9000)
    x, y = rr * math.cos(a), rr * math.sin(a)
    hh, *_ = height(np.array([x]), np.array([y]))
    if hh[0] < 0.2:
        continue
    s = rng.uniform(0.8, 2.4)
    dl.box((x, y, hh[0] + s / 2 + 2.0), (s, s, s))
dl_ob = dl.obj("ShoreLights", emissive("ShoreLamp", (1.0, 0.62, 0.30), 40.0))
no_shadow(dl_ob)


# --- deluge water: curtains pouring off the ML deck edges + 'rainbird' fountains --------
def curtain_material():
    m, nt = new_mat("WaterCurtain")
    m.surface_render_method = "BLENDED"
    m.use_transparency_overlap = True
    gain = value_node(nt, "Gain", 0.0)
    flow = value_node(nt, "Flow", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    off = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, flow.outputs[0], off.inputs[2])
    p = V(nt, "ADD", V(nt, "MULTIPLY", tc.outputs["Object"], (2.2, 2.2, 0.18)), off.outputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 8.0, "Roughness": 0.7})
    lk(nt, p, nz.inputs["Vector"])
    a = M(nt, "MULTIPLY", M(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 2.4, -0.75, clamp=True), gain.outputs[0], clamp=True)
    df = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.80, 0.84, 0.88, 1), "Roughness": 0.25,
                                                  "Transmission Weight": 0.4})
    tr = nd(nt, "ShaderNodeBsdfTransparent")
    mx = nd(nt, "ShaderNodeMixShader")
    lk(nt, a, mx.inputs[0])
    lk(nt, tr.outputs[0], mx.inputs[1])
    lk(nt, df.outputs[0], mx.inputs[2])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, mx.outputs[0], o.inputs["Surface"])
    return m


M_CURTAIN = curtain_material()
cur = MB()
ML_X = 20.5                                       # ML base footprint in world (rotated 90°): x ±20.5,
ML_Y0, ML_Y1 = -14.6, 32.4                        # y -14.6 (south, open side) .. 32.4 (tower side)
for (x0, y0, x1, y1, ox, oy) in ((-ML_X, ML_Y0, ML_X, ML_Y0, 0.0, -1.0), (-ML_X, ML_Y0, -ML_X, 12.0, -1.0, 0.0),
                                 (ML_X, ML_Y0, ML_X, 12.0, 1.0, 0.0)):
    cur.poly([(x0 + 0.4 * ox, y0 + 0.4 * oy, ML_DECK - 0.2), (x1 + 0.4 * ox, y1 + 0.4 * oy, ML_DECK - 0.2),
              (x1 + 1.6 * ox, y1 + 1.6 * oy, PAD_TOP), (x0 + 1.6 * ox, y0 + 1.6 * oy, PAD_TOP)], [(0, 1, 2, 3)])
curtains = cur.obj("DelugeCurtains", M_CURTAIN)
no_shadow(curtains)


def rainbird_tree():
    """Stateless water fountains: each droplet loops on a parabola from one of six deck nozzles."""
    t, s, gi, go = gn_tree("Rainbirds", [("Time", "NodeSocketFloat", 0.0), ("Count", "NodeSocketInt", 2400),
                                         ("Amount", "NodeSocketFloat", 0.0)])
    pts = t.nodes.new("GeometryNodePoints")
    t.links.new(gi.outputs["Count"], pts.inputs["Count"])
    idx = t.nodes.new("GeometryNodeInputIndex")
    def rnd(lo, hi, salt, vec=False):
        r = t.nodes.new("FunctionNodeRandomValue")
        r.data_type = "FLOAT_VECTOR" if vec else "FLOAT"
        want = "VECTOR" if vec else "VALUE"
        ins = [x for x in r.inputs if x.enabled and x.type == want]
        ins[0].default_value, ins[1].default_value = lo, hi
        r.inputs["Seed"].default_value = salt
        t.links.new(idx.outputs[0], r.inputs["ID"])
        return [x for x in r.outputs if x.enabled and x.type == want][0]
    life = 2.2
    tt = g_math(t, "FRACT", g_math(t, "ADD", g_math(t, "DIVIDE", gi.outputs["Time"], life), rnd(0.0, 1.0, 3)))
    age = g_math(t, "MULTIPLY", tt, life)
    nozzles = [(-14, -12), (14, -12), (-14, 12), (14, 12), (0, -16), (0, 16)]
    which = g_math(t, "FLOOR", rnd(0.0, 5.999, 5))
    ex = t.nodes.new("ShaderNodeCombineXYZ")
    for axis in range(2):
        acc = None
        for k in range(6):
            term = g_math(t, "MULTIPLY", g_math(t, "COMPARE", which, float(k), 0.1), nozzles[k][axis])
            acc = term if acc is None else g_math(t, "ADD", acc, term)
        t.links.new(acc, ex.inputs[axis])
    ex.inputs[2].default_value = ML_DECK + 0.5
    # spray toward the vehicle centre + up, with spread
    to_c = g_math(t, "SCALE", g_math(t, "NORMALIZE", g_math(t, "MULTIPLY", ex.outputs[0], (-1.0, -1.0, 0.0), vec=True), vec=True),
                  rnd(5.0, 9.0, 7), vec=True)
    up = rnd((-1.5, -1.5, 8.0), (1.5, 1.5, 13.0), 9, vec=True)
    vel = g_math(t, "ADD", to_c, up, vec=True)
    pos = g_math(t, "ADD", ex.outputs[0], g_math(t, "SCALE", vel, age, vec=True), vec=True)
    fall = t.nodes.new("ShaderNodeVectorMath")
    fall.operation = "SCALE"
    fall.inputs[0].default_value = (0.0, 0.0, -4.9)
    t.links.new(g_math(t, "MULTIPLY", age, age), fall.inputs["Scale"])
    pos = g_math(t, "ADD", pos, fall.outputs["Vector"], vec=True)
    setp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], setp.inputs["Geometry"])
    t.links.new(pos, setp.inputs["Position"])
    ico = t.nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 0.22
    ico.inputs["Subdivisions"].default_value = 1
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(setp.outputs[0], inst.inputs["Points"])
    t.links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    alive = g_math(t, "LESS_THAN", rnd(0.0, 1.0, 17), gi.outputs["Amount"])
    sz = g_math(t, "MULTIPLY", alive, rnd(0.5, 1.6, 19))
    t.links.new(sz, inst.inputs["Scale"])
    sm = t.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = principled("Spray", (0.82, 0.86, 0.9), 0.2, 0.0)
    t.links.new(inst.outputs[0], sm.inputs["Geometry"])
    t.links.new(sm.outputs[0], go.inputs["Geometry"])
    return t


RAIN_T = rainbird_tree()
rainbirds, rain_md = gn_obj("Rainbirds", RAIN_T)
no_shadow(rainbirds)


# --- scrub scatter on the near terrain (Poly Haven shrubs, CC0), off the pad and the sand --
def scatter_tree():
    t, s, gi, go = gn_tree("Scrub", [])
    dist = t.nodes.new("GeometryNodeDistributePointsOnFaces")
    dist.distribute_method = "POISSON"
    dist.inputs["Distance Min"].default_value = 9.0
    dist.inputs["Density Max"].default_value = 0.0016
    t.links.new(gi.outputs["Geometry"], dist.inputs["Mesh"])
    a_p = t.nodes.new("GeometryNodeInputNamedAttribute")
    a_p.data_type = "FLOAT"
    a_p.inputs["Name"].default_value = "pad"
    a_s = t.nodes.new("GeometryNodeInputNamedAttribute")
    a_s.data_type = "FLOAT"
    a_s.inputs["Name"].default_value = "sand"
    a_w = t.nodes.new("GeometryNodeInputNamedAttribute")
    a_w.data_type = "FLOAT"
    a_w.inputs["Name"].default_value = "water"
    pos = t.nodes.new("GeometryNodeInputPosition")
    rad = g_math(t, "LENGTH", pos.outputs[0], vec=True)
    ok = g_math(t, "MULTIPLY", g_math(t, "LESS_THAN", a_p.outputs["Attribute"], 0.5),
                g_math(t, "LESS_THAN", g_math(t, "ADD", a_s.outputs["Attribute"], a_w.outputs["Attribute"]), 0.2))
    ok = g_math(t, "MULTIPLY", ok, g_math(t, "GREATER_THAN", rad, 200.0))
    ok = g_math(t, "MULTIPLY", ok, g_math(t, "LESS_THAN", rad, 620.0))
    t.links.new(ok, dist.inputs["Selection"])
    coll = bpy.data.collections.new("ScrubSources")
    for nm in ("Scatter_shrub_02", "Scatter_shrub_01", "Scatter_shrub_04"):
        if nm in LIB:
            src = LIB[nm]
            tris = sum(len(p_.vertices) - 2 for p_ in src.data.polygons)
            dm = src.modifiers.new("Decimate", "DECIMATE")       # instances at 200+ m: ~2.5k tris is plenty
            dm.ratio = min(1.0, 2500.0 / max(tris, 1))
            coll.objects.link(src)
    ci = t.nodes.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    inst.inputs["Pick Instance"].default_value = True
    t.links.new(dist.outputs["Points"], inst.inputs["Points"])
    t.links.new(ci.outputs[0], inst.inputs["Instance"])
    r1 = t.nodes.new("FunctionNodeRandomValue")
    r1.data_type = "INT"
    [x for x in r1.inputs if x.enabled and x.type == "INT"][1].default_value = 3
    t.links.new([x for x in r1.outputs if x.enabled and x.type == "INT"][0], inst.inputs["Instance Index"])
    rr = t.nodes.new("FunctionNodeRandomValue")
    rr.data_type = "FLOAT_VECTOR"
    [x for x in rr.inputs if x.enabled and x.type == "VECTOR"][1].default_value = (0.0, 0.0, 6.28)
    t.links.new([x for x in rr.outputs if x.enabled and x.type == "VECTOR"][0], inst.inputs["Rotation"])
    rs = t.nodes.new("FunctionNodeRandomValue")
    rs.data_type = "FLOAT"
    ins = [x for x in rs.inputs if x.enabled and x.type == "VALUE"]
    ins[0].default_value, ins[1].default_value = 0.8, 2.2
    t.links.new([x for x in rs.outputs if x.enabled and x.type == "VALUE"][0], inst.inputs["Scale"])
    jg = t.nodes.new("GeometryNodeJoinGeometry")
    t.links.new(gi.outputs["Geometry"], jg.inputs[0])
    t.links.new(inst.outputs[0], jg.inputs[0])
    t.links.new(jg.outputs[0], go.inputs["Geometry"])
    return t


if any(n in LIB for n in ("Scatter_shrub_02", "Scatter_shrub_01")):
    terr_near.modifiers.new("Scrub", "NODES").node_group = scatter_tree()

# --- steam + smoke: JangaFX CC0 VDB sims (research fix #1: real volumes, not cotton balls) --
M_STEAM = smoke_material("Steam", (0.93, 0.93, 0.94), dens=3.0, aniso=0.35, glow_grid="flames")
M_STEAM_EARLY = smoke_material("SteamEarly", (0.94, 0.95, 0.97), dens=2.4, aniso=0.4, glow_grid=None,
                               glow_col=blackbody(5500))
STEAMV = []        # (object, nfiles, start, speed, base loc, drift (m/s), scale, kind)
_vrng = np.random.default_rng(77)
for k, (x, y, sc_, d0, sp_, rz) in enumerate((
        (125.0, 0.0, 1.35, 0, 0.36, 0.0),           # the trench blast, east
        (185.0, -45.0, 1.15, 5, 0.33, 1.3),
        (170.0, 50.0, 1.05, 9, 0.33, 2.4),
        (-34.0, -38.0, 0.75, 3, 0.40, 0.7),          # rising around the mobile launcher base
        (34.0, -44.0, 0.70, 6, 0.42, 3.9))):
    scl = (sc_ * (1.25 if k < 3 else 1.0), sc_, sc_ * (0.72 if k < 3 else 0.9))
    gz = 1.0 if k < 3 else PAD_TOP
    loc = anchored((x, y, gz), (0, 0, rz), scl, GROUND_ANCHOR)
    o, nfi = vdb_seq(f"SteamVDB{k}", GROUND_X, sc_, loc=tuple(loc), rot=(0, 0, rz), mat=M_STEAM)
    o.scale = scl
    STEAMV.append((o, nfi, IGN + d0, sp_, Vector(loc), Vector((9.0, 1.5 * np.sign(y), 0.8)) if k < 3
                   else Vector((0.0, -1.5, 1.6)), sc_, "ign"))
for k, (x, y, sc_) in enumerate(((118.0, 4.0, 0.42), (-10.0, -30.0, 0.30))):  # first steam: RS-25 start
    loc = anchored((x, y, 1.0 if k == 0 else PAD_TOP), (0, 0, 0.9 * k), sc_, GROUND_ANCHOR)
    o, nfi = vdb_seq(f"SteamEarly{k}", GROUND_X, sc_, loc=tuple(loc), rot=(0, 0, 0.9 * k), mat=M_STEAM_EARLY)
    STEAMV.append((o, nfi, RS25_T + 10 + 6 * k, 0.30, Vector(o.location), Vector((3.0, 0.0, 0.6)), sc_, "early"))
for o, *_ in STEAMV:
    o.data.render.precision = "HALF"
def column_material(name):
    """Exhaust column standing over the pad after liftoff, in a unit cylinder (z 0 ground → 1 at
    the vehicle): pyroclastic edge (two noise octaves, research §4), WIDE at the ground and narrow
    where it leaves the nozzles, soft top so its end never reads as a cap; glows only at the top
    where the plume feeds it (keyed Density / Evolve / Glow)."""
    m, nt = new_mat(name)
    dens = value_node(nt, "Density", 1.0)
    evo = value_node(nt, "Evolve", 0.0)
    glow = value_node(nt, "Glow", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    h = M(nt, "MINIMUM", M(nt, "MAXIMUM", sp.outputs[2], 0.0), 1.0)
    rr = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    rad = M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", M(nt, "POWER", h, 0.7), 0.72))
    n1 = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 0.55, "Detail": 5.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.35)), n1.inputs["Vector"])
    lk(nt, evo.outputs[0], n1.inputs["W"])
    n2 = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 3.2, "Detail": 4.0})
    lk(nt, ob, n2.inputs["Vector"])
    lk(nt, evo.outputs[0], n2.inputs["W"])
    fbm = M(nt, "ADD", M(nt, "SUBTRACT", n1.outputs["Fac"], 0.5), M(nt, "MULTIPLY", M(nt, "SUBTRACT", n2.outputs["Fac"], 0.5), 0.6))
    mask = M(nt, "MULTIPLY", M(nt, "ADD", M(nt, "SUBTRACT", 1.0, M(nt, "DIVIDE", rr, rad)), M(nt, "MULTIPLY", fbm, 1.6)), 3.5, clamp=True)
    top = M(nt, "SUBTRACT", 1.0, M(nt, "POWER", h, 3.0))
    d = M(nt, "MULTIPLY", M(nt, "MULTIPLY", mask, top), dens.outputs[0])
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": (0.90, 0.90, 0.91, 1), "Anisotropy": 0.3,
                                                    "Emission Color": blackbody(1900) + (1,)})
    lk(nt, d, pv.inputs["Density"])
    hot = M(nt, "MULTIPLY", M(nt, "POWER", h, 6.0), top)
    axis = M(nt, "SUBTRACT", 1.0, M(nt, "DIVIDE", rr, M(nt, "MULTIPLY", rad, 0.6)), clamp=True)
    lk(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "MULTIPLY", hot, mask), axis), glow.outputs[0]),
       pv.inputs["Emission Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


M_TRAIL = column_material("Trail")
trail = lathe("TrailPad", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 24, M_TRAIL)
M_HAZE = volume_material("PadHaze", (0.8, 0.85, 0.95), "UNIFORM", nscale=3.0, detail=2.0, aniso=0.6)
haze_box = cube_obj("PadHaze", M_HAZE, loc=(0, 0, 400), scale=(4500, 4500, 400))


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
# 8. WORLD: 5.x multiple-scattering sky + the Earth as a ray-sphere hit in the world shader
#    textured with NASA Blue Marble NG (day), Black Marble 2016 (night lights) and the Blue
#    Marble cloud layer (research fix #9), KSC under the pad, east = downrange (+Y). Stars.
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
hitp = V(wn, "SCALE", dvec, scale=tt)
hc = nd(wn, "ShaderNodeCombineXYZ")
lk(wn, H, hc.inputs[2])
nrm = V(wn, "ADD", hitp, hc.outputs[0])                          # unit normal at the hit (world)
rot = nd(wn, "ShaderNodeVectorRotate", rotation_type="X_AXIS")   # downrange travel (keyed)
lk(wn, nrm, rot.inputs["Vector"])
lk(wn, w_scroll.outputs[0], rot.inputs["Angle"])
en = rot.outputs[0]
# world normal → Earth-fixed direction: up=+Z → KSC radial, +Y → east, -X → north
LAT0, LON0 = math.radians(28.6), math.radians(-80.6)
UP_E = (math.cos(LAT0) * math.cos(LON0), math.cos(LAT0) * math.sin(LON0), math.sin(LAT0))
EAST_E = (-math.sin(LON0), math.cos(LON0), 0.0)
NORTH_E = (-math.sin(LAT0) * math.cos(LON0), -math.sin(LAT0) * math.sin(LON0), math.cos(LAT0))
ens = nd(wn, "ShaderNodeSeparateXYZ")
lk(wn, en, ens.inputs[0])
ece = nd(wn, "ShaderNodeCombineXYZ")
for k in range(3):
    acc = M(wn, "ADD", M(wn, "MULTIPLY", ens.outputs[2], UP_E[k]), M(wn, "MULTIPLY", ens.outputs[1], EAST_E[k]))
    lk(wn, M(wn, "ADD", acc, M(wn, "MULTIPLY", ens.outputs[0], -NORTH_E[k])), ece.inputs[k])
es = nd(wn, "ShaderNodeSeparateXYZ")
lk(wn, ece.outputs[0], es.inputs[0])
lon = M(wn, "ARCTAN2", es.outputs[1], es.outputs[0])
lat = M(wn, "ARCSINE", M(wn, "MINIMUM", M(wn, "MAXIMUM", es.outputs[2], -1.0), 1.0))
uv = nd(wn, "ShaderNodeCombineXYZ")
lk(wn, M(wn, "ADD", M(wn, "DIVIDE", lon, 2 * math.pi), 0.5), uv.inputs[0])
lk(wn, M(wn, "ADD", M(wn, "DIVIDE", lat, math.pi), 0.5), uv.inputs[1])
EARTH_D = os.path.join(AS, "nasa_earth", "derived")


def earth_tex(fn, data=False):
    t = nd(wn, "ShaderNodeTexImage", image=load_img(os.path.join(EARTH_D, fn), data), interpolation="Cubic",
           extension="REPEAT")
    lk(wn, uv.outputs[0], t.inputs["Vector"])
    return t


day = earth_tex("bluemarble_5400.jpg")
night = earth_tex("blackmarble_8192.jpg")
cloudt = earth_tex("clouds_8192.jpg", data=True)
cbw = nd(wn, "ShaderNodeRGBToBW")
lk(wn, cloudt.outputs["Color"], cbw.inputs[0])
cloudm = M(wn, "POWER", cbw.outputs[0], 1.4)
earth_alb = mix_col(wn, cloudm, day.outputs["Color"], (0.80, 0.80, 0.82))
skyc = sky.outputs["Color"]
mu = M(wn, "MULTIPLY", V(wn, "DOT_PRODUCT", nrm, dvec), -1.0)
limb = M(wn, "MULTIPLY", M(wn, "POWER", M(wn, "MAXIMUM", mu, 0.0), 0.6), 1.0, clamp=True)
wgt = M(wn, "MULTIPLY", M(wn, "MULTIPLY", hit, limb), w_ov.outputs[0])
facc = col_mul(wn, earth_alb, 1.0 / 0.30)                           # relative to the sky's ground albedo
fac_mix = mix_col(wn, wgt, (1.0, 1.0, 1.0), facc)
earth_col = col_mul(wn, skyc, fac_mix)
lum = nd(wn, "ShaderNodeRGBToBW")
lk(wn, skyc, lum.inputs[0])
nightf = M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", lum.outputs[0], 40.0), clamp=True)
nbw = nd(wn, "ShaderNodeRGBToBW")
lk(wn, night.outputs["Color"], nbw.inputs[0])
lights = M(wn, "POWER", M(wn, "SUBTRACT", nbw.outputs[0], 0.12, clamp=True), 1.6)
cl_st = M(wn, "MULTIPLY", M(wn, "MULTIPLY", M(wn, "MULTIPLY", lights, nightf), wgt),
          M(wn, "MULTIPLY", M(wn, "SUBTRACT", 1.0, cloudm), 0.5))
addc = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, cl_st, addc.inputs[0])
lk(wn, earth_col, addc.inputs[6])
addc.inputs[7].default_value = (1.0, 0.62, 0.28, 1)
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
wo = nd(wn, "ShaderNodeOutputWorld")
lk(wn, bgw.outputs[0], wo.inputs["Surface"])

SUN = bpy.data.lights.new("Sun", "SUN")
SUN.angle = math.radians(0.53)
SUN.color = (1.0, 0.80, 0.62)
SUN_OB = link(bpy.data.objects.new("Sun", SUN))
MOON = bpy.data.lights.new("Moonfill", "SUN")
MOON.angle = math.radians(2.0)
MOON.color = (0.55, 0.65, 1.0)
MOON_OB = link(bpy.data.objects.new("Moonfill", MOON))
MOON_OB.rotation_euler = (math.radians(55), 0, math.radians(-150))

# --- the cumulus deck (research fix #10): JangaFX CC0 cloud VDBs, clustered, lit by the low
#     sun and from inside by the plume. Parented to a keyed 'Deck' empty that sinks past the climb.
DECK = empty("CloudDeck")
M_CLOUD = smoke_material("Cumulus", (0.96, 0.96, 0.97), dens=0.45, aniso=0.55, glow_grid=None, plume_lit=False)
_cfiles = sorted(glob.glob(os.path.join(CLOUDS, "*.vdb")))
_crng = np.random.default_rng(31)
CLOUD_OBS = []
_cvols = []
for f in _cfiles:
    v = bpy.data.volumes.new(os.path.basename(f))
    v.filepath = f
    v.materials.append(M_CLOUD)
    v.render.precision = "HALF"
    _cvols.append(v)
_slots = []
while len(_slots) < 22:
    x, y = _crng.uniform(-3800, 3800), _crng.uniform(-3000, 4200)
    if math.hypot(x, y) < 330:                      # keep a clear shaft around the climb line
        continue
    if all(math.hypot(x - a, y - b) > 650 for a, b in _slots):
        _slots.append((x, y))
for k, (x, y) in enumerate(_slots):
    v = _cvols[k % len(_cvols)]
    o = bpy.data.objects.new(f"Cloud{k}", v)
    s_ = float(_crng.uniform(1.4, 2.4))
    o.scale = (s_, s_, s_ * float(_crng.uniform(0.75, 1.0)))
    o.rotation_euler = (0, 0, float(_crng.uniform(0, 2 * math.pi)))
    o.location = (x, y, float(_crng.uniform(-60, 60)))
    link(o, DECK)
    CLOUD_OBS.append(o)
M_DECKHAZE = volume_material("DeckHaze", (0.92, 0.93, 0.96), "SLAB", nscale=2.6, detail=7.0,
                             emis_color=(1.0, 0.45, 0.15), aniso=0.5)
deck_haze = cube_obj("DeckHaze", M_DECKHAZE, loc=(0, 0, -260), scale=(9000, 9000, 120), parent=DECK)
M_TRAILF = volume_material("TrailFlight", (0.86, 0.86, 0.88), "COLUMN", nscale=0.5, detail=6.0,
                           emis_color=blackbody(1900), aniso=0.3, sharp=5.0, namp=1.0)
trail_f = lathe("TrailFlight", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 24, M_TRAILF)


# =====================================================================================
# 9. CAMERA + SHOTS
# =====================================================================================
CAM_D = bpy.data.cameras.new("Cam")
CAM_D.sensor_width = 36.0
CAM_D.clip_start = 0.3
CAM_D.clip_end = 60000.0
CAM_D.dof.use_dof = True
CAM_D.dof.aperture_blades = 7
CAM = link(bpy.data.objects.new("Cam", CAM_D))
SC.camera = CAM

PITCH = np.zeros(N)
PITCH[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [7, 18, 26])
PITCH[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, SES], [30, 52, 66])
PITCH[SET == 3] = np.interp(IDX[SET == 3], [SES, N], [74, 82])
PITCH[(SET == 0)] = np.clip((ALT[SET == 0] - 150) / 300, 0, 1) * 3.0          # roll/pitch program
RIG_POS = np.zeros((N, 3))
RIG_POS[:, 2] = np.where(SET == 0, PAD_Z + ALT, 0.0)
RIG_POS[:, 0] += 0.04 * KICK * np.maximum(on_core, on_boost)
# tower clear: a tiny lateral drift away from the tower as it rises (SLS does a slight yaw)
RIG_POS[:, 1] += np.where(SET == 0, -np.clip(ALT - 20, 0, None) * 0.03, 0.0)


def rig_matrix(i):
    return Matrix.Translation(Vector(RIG_POS[i])) @ Matrix.Rotation(math.radians(-PITCH[i]), 4, "X")


def wl(i, p):
    """rocket-local → world."""
    return rig_matrix(i) @ Vector(p)


def wl_lag(i, p, lag=6, salt=0, wobble=0.0):
    """Tracking-mount target: the vehicle point as it was `lag` frames ago, plus a slow operator
    wobble (metres at the subject) — real tracking cameras trail the vehicle slightly."""
    j = max(i - lag, 0)
    w = Vector((math.sin(i * 0.071 + salt) + 0.4 * math.sin(i * 0.173 + 2 * salt),
                0.0, math.sin(i * 0.053 + 1.3 + salt) + 0.4 * math.sin(i * 0.191 + salt)))
    return wl(j, p) + w * wobble


def shake_vec(i, amp, salt=0, hf=False):
    """Kick jolts (hashed per kick) + rumble (sum of sines; hf=True: onboard vibration)."""
    k = KIDX[i]
    j = Vector((hash01(k, 1, salt) - 0.5, hash01(k, 2, salt) - 0.5, hash01(k, 3, salt) - 0.5)) * 2 * KICK[i]
    x = i + 977 * salt
    f = 3.7 if hf else 1.0
    r = Vector((math.sin(x * 1.71 * f + 0.3) + 0.5 * math.sin(x * 3.13 * f), math.sin(x * 1.37 * f + 1.1) +
                0.5 * math.sin(x * 2.89 * f + 2.0), math.sin(x * 1.93 * f + 2.2)))
    return (j * 1.0 + r * (0.18 if not hf else 0.45)) * amp


SHOT_FN = {}


def shot(name):
    def deco(fn):
        SHOT_FN[name] = fn
        return fn
    return deco


STACK_C = PAD_Z + 50.0                             # mid-stack height on the pad


def ground_z(x, y):
    return float(max(height(np.array([x]), np.array([y]))[0][0], 0.0))


LAGOON_CAM = CAM_DIR * 3250.0
LAGOON_CAM.z = ground_z(LAGOON_CAM.x, LAGOON_CAM.y) + 1.8
CLOCK_CAM = Vector((-250.0, -560.0, ground_z(-250.0, -560.0) + 1.7))


@shot("A1_wide")             # 400 mm across the lagoon from 3.2 km: compression, haze, shimmer
def _(i, u, n):
    t = u / n
    s = float(smoothstep(t))
    cam = LAGOON_CAM + Vector((14.0 * s, 0.0, 0.0))
    tgt = Vector((0.0, 4.0, STACK_C + 6.0 - 4 * s))
    return dict(cam=cam, tgt=tgt + Vector((0.6 * math.sin(i * 0.031), 0, 0.4 * math.sin(i * 0.047))),
                lens=380 + 70 * s, fstop=8, shake=0.0, haze=1.0)


@shot("A2_clock")            # press-site countdown clock, rack focus to the pad
def _(i, u, n):
    t = u / n
    cam = CLOCK_CAM + Vector((-2.0 * t, 1.0 * t, 0.15 * t))
    tgt = Vector((10.0, 0.0, STACK_C + 2 - 6 * t))
    rack = float(smoothstep((t - 0.45) / 0.3))
    fdist = 13.5 * (1 - rack) + (Vector((0, 0, STACK_C)) - cam).length * rack
    return dict(cam=cam, tgt=tgt, lens=35, fstop=2.8, focus=fdist, shake=0.0, haze=0.4)


@shot("A3_logo")             # 300 mm tilt up the core to the emblem; tower lattice behind
def _(i, u, n):
    t = u / n
    s = float(smoothstep(t))
    cam = Vector((60.0 - 10 * s, -540.0, 16.0))
    tgt = Vector((0.0, 0.0, PAD_Z + 8.0 + (LOGO_Z0 + LOGO_H * 0.5 - 8.0) * s))
    return dict(cam=cam, tgt=tgt, lens=300, fstop=5.6, shake=0.0, haze=0.6)


@shot("A4_low")              # low remote on the hardstand: deluge, sparks, engine start
def _(i, u, n):
    t = u / n
    creep = float(PRE[i])
    cam = Vector((44.0 - 5 * t - 4 * creep, -52.0 + 6 * t + 4 * creep, PAD_TOP + 1.1))
    tgt = Vector((-2.0, 0.0, PAD_Z + 30 + 6 * t))
    return dict(cam=cam, tgt=tgt, lens=17 + 2 * creep, fstop=8, shake=0.05 * PRE[i] + 0.4 * on_core[i])


@shot("B1_ign")              # 1.3 km due south, 135 mm: the trench blast rolls to the right
def _(i, u, n):
    t = u / n
    cam = Vector((25.0, -1300.0, 6.0))
    tgt = Vector((70.0, 0.0, STACK_C + 12 + ALT[i] * 0.55))
    return dict(cam=cam, tgt=tgt, lens=135 - 20 * t, fstop=8, shake=0.6, haze=1.0)


@shot("B2_tower")            # tower remote at mid height looking UP as the stack clears the tower
def _(i, u, n):
    t = u / n
    cam = Vector((-7.5, 9.5, ML_DECK + 62.0))
    tgt = wl(i, (0.0, 0.0, 60.0))
    tgt.z = max(tgt.z, cam.z + 20.0)
    return dict(cam=cam, tgt=tgt, lens=16, fstop=8, shake=0.7)


@shot("B3_track")            # 5 km tracking mount, 800 mm, trailing the vehicle
def _(i, u, n):
    cam = Vector((-1500.0, -4800.0, 6.0))
    tgt = wl_lag(i, (0, 0, 30.0), lag=5, wobble=2.5)
    return dict(cam=cam, tgt=tgt, lens=800, fstop=11, shake=0.15, haze=1.6)


@shot("B4_tail")             # onboard: booster-mounted camera looking aft at the pad
def _(i, u, n):
    cam = wl(i, (-SRB_X - 2.2, -1.0, 51.0))
    tgt = wl(i, (-SRB_X - 1.2, -2.6, -40.0))
    return dict(cam=cam, tgt=tgt, lens=11, fstop=11, shake=0.8, up=wl(i, (0, 0, 1)) - wl(i, (0, 0, 0)))


DECK_TOP = np.full(N, -5000.0)
_c = (IDX >= C_CUTS[0]) & (IDX < MAXQ)
DECK_TOP[_c] = 120.0 - (IDX[_c] - C_CUTS[0]) / FPS * 34.0
_d = (IDX >= MAXQ) & (IDX < BSEP)
DECK_TOP[_d] = -1100.0 - (IDX[_d] - MAXQ) / FPS * 15.0


@shot("C1_cloud")            # air-to-air long lens: the vehicle climbing out of the deck
def _(i, u, n):
    t = u / n
    cam = Vector((1100.0, -2500.0, DECK_TOP[i] + 140.0 + 90 * t))
    tgt = wl_lag(i, (0, 0, 10.0), lag=4, wobble=1.5)
    return dict(cam=cam, tgt=tgt, lens=420, fstop=8, shake=0.1, haze=0.8)


@shot("C2_side")             # chase angle (exposure pumping on the kicks sells it as footage)
def _(i, u, n):
    t = u / n
    creep = float(PRE[i])
    a = math.radians(-70 + 25 * t)
    d = 150 - 40 * creep
    cam = wl(i, (d * math.cos(a), d * math.sin(a), 30 - 10 * t))
    tgt = wl(i, (0, 0, 45.0))
    return dict(cam=cam, tgt=tgt, lens=55, fstop=8, shake=0.08 + 0.1 * creep)


@shot("D1_maxq")             # close chase in profile: the condensation collar wrapping the upper stack
def _(i, u, n):
    t = u / n
    cam = wl(i, (16.0 - 5 * t, -34.0, OGIVE_Z0 - 8.0 - 6 * t))
    tgt = wl(i, (0.0, 0.0, LVSA_TOP - 4.0))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=5.6, shake=1.0)


@shot("D2_under")            # onboard: looking aft past the booster to the plume and clouds
def _(i, u, n):
    cam = wl(i, (2.4, -R - 0.6, 56.0))
    tgt = wl(i, (SRB_X - 1.0, -2.0, -60.0))
    return dict(cam=cam, tgt=tgt, lens=12, fstop=11, shake=0.9, up=wl(i, (0, 1, 0)) - wl(i, (0, 0, 0)))


@shot("D3_wide")             # long-lens air-to-air: the condensation collar
def _(i, u, n):
    t = u / n
    cam = wl(i, (700.0, -1250.0, 260.0 - 90 * t))
    tgt = wl_lag(i, (0, 0, 55.0), lag=3, wobble=0.8)
    return dict(cam=cam, tgt=tgt, lens=240, fstop=11, shake=0.3, haze=0.5)


@shot("E_calm")
def _(i, u, n):
    t = u / n
    cam = wl(i, (80.0, -110.0, -20.0 - 10 * t))
    tgt = wl(i, (0, 0, 20.0))
    return dict(cam=cam, tgt=tgt, lens=28, fstop=11, shake=0.08)


def booster_pos(i, side):
    tb_ = max(i - BSEP, 0) / FPS
    return (side * (SRB_X + 2.2 * tb_ + 0.5 * tb_ ** 2), 0.0, -(3.0 * tb_ ** 2) - 2.0 * tb_)


@shot("F1_sep")              # onboard on the core: the right booster peels away
def _(i, u, n):
    cam = wl(i, (2.6, -R - 0.6, 47.0))
    b = booster_pos(i, 1)
    tgt = wl(i, (b[0], -0.5, b[2] + 30.0))
    return dict(cam=cam, tgt=tgt, lens=13, fstop=11, shake=0.9, up=wl(i, (0, 0, 1)) - wl(i, (0, 0, 0)))


@shot("F2_tail")
def _(i, u, n):
    b = booster_pos(i, -1)
    cam = wl(i, (-9.0, -18.0, 40.0))
    tgt = wl(i, (b[0], 0, b[2] + 25)) * 0.8 + wl(i, (0, 0, 0)) * 0.2
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.35)


@shot("F3_wide")             # far long lens: two bright boosters falling away, the plume balloons
def _(i, u, n):
    t = u / n
    cam = wl(i, (1400.0, -2500.0, -500.0 + 120 * t))
    tgt = wl_lag(i, (0, 0, -40.0), lag=3, wobble=2.0)
    return dict(cam=cam, tgt=tgt, lens=180, fstop=11, shake=0.25)


@shot("G_calm")              # the sunlit 'jellyfish' over a dark Earth
def _(i, u, n):
    t = u / n
    cam = wl(i, (2600.0, -3600.0, -2200.0 - 400 * t))
    tgt = wl(i, (0, 0, -700.0))
    return dict(cam=cam, tgt=tgt, lens=42, fstop=11, shake=0.04)


@shot("H1_fair")             # launch-abort-tower jettison, close
def _(i, u, n):
    t = u / n
    cam = wl(i, (10.0, -20.0 - 6 * t, OGIVE_Z0 + 6.0))
    tgt = wl(i, (0, 0, OGIVE_Z0 + 2.0 + 6 * t))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=8, shake=0.7)


@shot("H2_side")
def _(i, u, n):
    t = u / n
    cam = wl(i, (34.0, -52.0 + 8 * t, SAJ_Z0 - 4))
    tgt = wl(i, (0, 0, SAJ_Z0 - 2))
    return dict(cam=cam, tgt=tgt, lens=35, fstop=11, shake=0.5)


@shot("I_meco")              # onboard (upper stage) looking aft: the core stage falls away
def _(i, u, n):
    cam = wl(i, (2.9, -2.9, LVSA_TOP + 1.5))
    tgt = wl(i, (0.0, 0.0, 10.0))
    return dict(cam=cam, tgt=tgt, lens=12, fstop=11, shake=0.15, up=wl(i, (0, 1, 0)) - wl(i, (0, 0, 0)))


@shot("J1_limb")
def _(i, u, n):
    t = u / n
    cam = wl(i, (-30.0 + 4 * t, -16.0, 48.0 + 4 * t))
    tgt = wl(i, (0.0, 6.0, 125.0))
    return dict(cam=cam, tgt=tgt, lens=20, fstop=16, shake=0.25)


@shot("J2_side")
def _(i, u, n):
    t = u / n
    cam = wl(i, (24.0 - 6 * t, -10.0, 80.0))
    tgt = wl(i, (0, 0, 79.0))
    return dict(cam=cam, tgt=tgt, lens=28, fstop=8, shake=0.25)


@shot("J3_wide")
def _(i, u, n):
    t = u / n
    s = float(ease_out(t, 2))
    cam = wl(i, (40.0 + 60 * s, -20.0 - 40 * s, 70.0 - 60 * s))
    tgt = wl(i, (0, 0, 80.0))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.2)


@shot("K_outro")
def _(i, u, n):
    t = u / n
    cam = wl(i, (90.0 + 20 * t, -60.0, 15.0))
    tgt = wl(i, (0, 0, 92.0 + 10 * t))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.0)


# --- evaluate the shot list per frame ---------------------------------------------------
cam_loc = np.zeros((N, 3))
cam_rot = np.zeros((N, 3))
cam_lens = np.zeros(N)
cam_focus = np.zeros(N)
cam_fstop = np.zeros(N)
SHOT_HAZE = np.zeros(N)
SHOT_ONB = np.zeros(N)
SHOT_BOUNDS = [s for s, _ in SHOTS] + [N]
prev_eul = None
for k, (s0, name) in enumerate(SHOTS):
    s1 = SHOT_BOUNDS[k + 1]
    fn = SHOT_FN[name]
    onb = name in ONBOARD
    for i in range(s0, s1):
        d = fn(i, i - s0, max(s1 - s0 - 1, 1))
        thr = max(on_core[i], on_boost[i], on_s2[i])
        amp = d.get("shake", 0.0) * (0.15 * thr + 0.85 * KICK[i] * HOT[i] + 0.6 * IGN_FLASH[i] + 0.5 * onb * thr)
        if SET[i] == 3:
            amp *= 0.5
        if d["lens"] > 150:                                     # long lens: jolts read as metres at
            amp *= 150.0 / d["lens"]                            # the subject, not at the camera
        cam = d["cam"] + shake_vec(i, amp * (0.06 if onb else 0.35), hf=onb)
        if BIG[i] > 0:
            cam = cam + (d["tgt"] - cam) * (0.07 * ONE[i])
        tgt = d["tgt"] + shake_vec(i, amp * (0.25 if onb else 0.9), 1, hf=onb)
        fwd = (tgt - cam).normalized()
        q = fwd.to_track_quat("-Z", "Y")
        if "up" in d:
            up = Vector(d["up"]).normalized()
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
        SHOT_HAZE[i] = d.get("haze", 0.0)
        SHOT_ONB[i] = 1.0 if onb else 0.0
bake_vec(CAM, "location", cam_loc)
bake_vec(CAM, "rotation_euler", cam_rot)
bake(CAM_D, "lens", cam_lens)
bake(CAM_D, "dof.focus_distance", cam_focus)
bake(CAM_D, "dof.aperture_fstop", cam_fstop)

# countdown board in front of the A2 camera, angled toward it
_a2 = [s for s, n in SHOTS if n == "A2_clock"][0]
_cpos = Vector(cam_loc[_a2])
_cdir = Vector((10.0, 0.0, STACK_C)) - _cpos
_cdir.z = 0
_cdir.normalize()
_left = Vector((-_cdir.y, _cdir.x, 0))
_gz = float(height(np.array([_cpos.x]), np.array([_cpos.y]))[0][0])
CLK.location = _cpos + _cdir * 13.5 + _left * 4.6 + Vector((0, 0, -0.6))
CLK.location.z = max(CLK.location.z, _gz + 2.65)
_face = (-_cdir).copy()
CLK.rotation_euler = (0, 0, math.atan2(_face.x, -_face.y) - math.radians(24))


# =====================================================================================
# 10. ANIMATION BAKES (all music reactions live here)
# =====================================================================================
# --- SAJ panels: pivot each at its own centroid so it tumbles in place as it flies off ---
PANEL_DIR = []
for pn in PANELS:
    ch = pn.children[0]
    ws = [Vector(c) for c in ch.bound_box]
    cen = sum(ws, Vector()) / 8.0
    pn.location = cen
    ch.location = -cen
    dxy = Vector((cen.x, cen.y, 0.0))
    PANEL_DIR.append(dxy.normalized() if dxy.length > 0.1 else Vector((1, 0, 0)))

# --- sets / visibility ---
PAD_ONLY = ["TerrainNear", "TerrainFar", "Water", "ML", "FlameDeflector", "TrenchWalls", "LightningTowers",
            "LH2Sphere", "LOXSphere", "WaterTowerLegs", "WaterTowerTank", "FloodMasts", "ShoreLights",
            "PadHaze", "TrailPad", "DelugeCurtains", "Rainbirds"]
for o in bpy.data.objects:
    if o.name.startswith(("Catenary", "Beacon", "Flood", "Search", "WorkL", "WorkLamp", "DeckL", "Seg", "Colon", "Sign", "Board", "Clock",
                          "SteamVDB", "SteamEarly")) or o.name in PAD_ONLY:
        show(o, SET == 0)
show(bpy.data.objects["CountdownBoard"], IDX < IGN + 2 * FPS)
for o in bpy.data.objects:
    if o.parent == CLK:
        show(o, IDX < IGN + 2 * FPS)
show(SIGN_V, (IDX >= IGN) & (IDX < IGN + 2 * FPS))
for o in CLOUD_OBS + [deck_haze]:
    show(o, (SET == 1) & (DECK_TOP > -3000))
show(trail_f, (SET == 1) | ((SET == 2) & (IDX < FAIRSEP)))
show(trail, (SET == 0) & (IDX >= LIFT))
show(DECK, np.zeros(N, bool))
for o, nfi in VENTS:
    show(o, IDX < LIFT + 20)
for o, nfi, st0, *_ in STEAMV:
    show(o, (SET == 0) & (IDX >= st0 - 1))
show(curtains, (SET == 0) & (DELUGE > 0.01))
show(rainbirds, (SET == 0) & (DELUGE > 0.01))
show(sparks, (SET == 0) & (SPARKS > 0.001))
show(vcone, (IDX >= MAXQ - 2) & (IDX < D_END))
for c in vcone_b:
    show(c, (IDX >= MAXQ - 2) & (IDX < D_END))
SEP = MECO + int(round(BAR / 2))
s1_vis = IDX < SEP + 4 * FPS
for o in [o for o in bpy.data.objects if o.parent == S1]:
    show(o, s1_vis)
for bem in (BL, BRt):
    for o in [o for o in bpy.data.objects if o.parent == bem]:
        show(o, IDX < BSEP + 8 * FPS)
for o in [c for pn in PANELS for c in pn.children] + list(LASG.children):
    show(o, IDX < FAIRSEP + 6 * FPS)
show(flakes, (SAGE < 36) & (SET == 0) & (IDX >= RS25_T))
show(flakes2, (IDX >= LIFT) & (IDX < LIFT + 4 * FPS) & (SET == 0))
_far = np.zeros(N, bool)
for k_, (s0_, nm_) in enumerate(SHOTS):
    if nm_ in ("F3_wide", "G_calm"):
        _far[s0_:SHOT_BOUNDS[k_ + 1]] = True
show(jelly, (SET == 2) & (IDX >= BSEP + 2 * FPS) & _far)

# --- rig + stage motions ---
bake_vec(RIG, "location", RIG_POS)
bake_vec(RIG, "rotation_euler", np.stack([np.radians(-PITCH), np.zeros(N), np.zeros(N)], 1))
tb = np.maximum(IDX - BSEP, 0) / FPS
for side, bem in ((-1, BL), (1, BRt)):
    ox = side * (SRB_X + 2.2 * tb + 0.5 * tb ** 2)
    oz = -(3.0 * tb ** 2) - 2.0 * tb
    ry = side * np.radians(5.5 * tb ** 1.3)
    bake_vec(bem, "location", np.stack([ox, np.zeros(N), oz], 1))
    bake_vec(bem, "rotation_euler", np.stack([np.zeros(N), ry, np.zeros(N)], 1))
# launch abort system: the jettison motor yanks tower + ogive up and away (keyed 'LASMotor')
tf = np.maximum(IDX - FAIRSEP - 6, 0) / FPS
bake_vec(LASG, "location", np.stack([np.zeros(N), -0.8 * tf ** 2, 26.0 * tf ** 2 + 3.0 * tf], 1))
bake_vec(LASG, "rotation_euler", np.stack([np.radians(-4.0 * tf ** 1.5), np.zeros(N), np.zeros(N)], 1))
tpn = np.maximum(IDX - FAIRSEP, 0) / FPS
for pn, dvec_ in zip(PANELS, PANEL_DIR):
    base = Vector(pn.location)
    off = (1.4 * tpn + 2.2 * tpn ** 2)
    loc = np.stack([base.x + dvec_.x * off, base.y + dvec_.y * off, base.z - 1.6 * tpn ** 2], 1)
    bake_vec(pn, "location", loc)
    tang = Vector((-dvec_.y, dvec_.x, 0.0))
    ang = np.radians(np.minimum(70 * tpn, 200))
    eul = [Matrix.Rotation(float(a), 3, tang).to_euler() for a in ang]
    bake_vec(pn, "rotation_euler", np.array([(e.x, e.y, e.z) for e in eul]))
# core stage falls back after MECO (stage sep half a bar later)
ts = np.maximum(IDX - SEP, 0) / FPS
bake_vec(S1, "location", np.stack([np.zeros(N), -0.25 * ts ** 2, -(1.4 * ts + 0.9 * ts ** 2)], 1))
bake_vec(S1, "rotation_euler", np.stack([np.radians(1.4 * ts ** 1.2), np.zeros(N), np.zeros(N)], 1))

# --- plumes ---
alt_expand = np.ones(N)
alt_expand[SET == 0] = 1.0 + np.clip(ALT[SET == 0] / 2500.0, 0, 0.12)
alt_expand[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [1.15, 1.35, 1.6])
alt_expand[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, SES], [1.8, 2.4, 3.0])
phase = np.cumsum(0.9 + 0.6 * HOT) / FPS                             # flow scroll
flick = np.cumsum(1.4 + 1.2 * HOT + 0.8 * KICK) / FPS               # 4D noise W: flicker
for o, kind, part in PLUMES:
    vol = part == "vol"
    if kind.startswith("core"):
        on = on_eng[int(kind[4:])]
        lng = on * (1.0 + 0.10 * KICK * HOT) * np.where(SET == 0, 1.0, alt_expand ** 0.6)
        gain = on * PULSE * np.where(SET == 0, 22.0, 4.0 / alt_expand) * np.where(IDX >= IGN, 0.6, 1.0)
        sx = np.maximum(alt_expand ** 0.8 * (0.6 + 0.4 * on), 1e-3)
        sz = np.maximum(lng, 1e-3)
    elif kind == "boost":
        on = on_boost
        ex = alt_expand
        lng = on * (1.0 + 0.16 * KICK * HOT)
        if vol:
            gain = on * PULSE * np.where(SET == 0, 0.45, np.where(SET == 1, 0.30, 0.12)) / ex ** 1.2
            sx = np.where(SET == 0, 10.0, 16.0) * ex * (0.5 + 0.5 * on)
            sz = np.maximum(np.where(SET == 0, 110.0, 150.0) * lng * ex ** 0.7, 1e-3)
        else:
            gain = on * PULSE * 30.0 * np.where(SET == 0, 1.0, 0.8) / ex ** 0.6
            sx = np.maximum(np.where(SET == 0, 1.0, 1.8) * ex ** 0.6 * (0.55 + 0.45 * on), 1e-3)
            sz = np.maximum(np.where(SET == 0, 1.0, 1.9) * lng * ex ** 0.7, 1e-3)
        gain = gain * (1.0 + 3.0 * IGN_FLASH)
    else:  # s2
        on = on_s2
        pulse2 = 1.0 + 0.45 * BIG * ONE + 0.25 * KICK * HOT
        lng = on * (1.0 + 0.16 * KICK * HOT + 0.12 * BIG * ONE)
        gain = on * pulse2 * (14.0 if not vol else 0.05)
        sx = np.full(N, 22.0) * (0.4 + 0.6 * on) if vol else np.ones(N) * (0.6 + 0.4 * on)
        sz = np.maximum((np.full(N, 190.0) if vol else np.ones(N)) * lng, 1e-3)
    set_color_keys(o, np.stack([gain, flick, phase], 1), const=False)
    bake_vec(o, "scale", np.stack([sx, sx, sz], 1))
    show(o, on > 0.001)
for lo, tag, j in PLIGHTS:
    if tag in ("bl", "br"):
        e = on_boost * PULSE * (1.0 + 2.5 * IGN_FLASH) * (2.4e6, 1.4e6, 9.0e5)[j]
        e = e * np.where(SET == 0, 1.0, 0.7)
        on = on_boost
    elif tag == "core":
        e = on_core * PULSE * (5.0e4, 3.0e4)[j] * np.where(IDX >= IGN, 1.0, 1.0)
        on = on_core
    else:
        e = on_s2 * 8.0e4 * (1 + 0.5 * BIG * ONE)
        on = on_s2
    bake(lo.data, "energy", e)
    show(lo, on > 0.001)
bake(BOUNCE, "energy", on_boost * PULSE * np.where(SET == 0, 2.0e6, np.where(SET == 1, 8.0e5, 3.0e5)))
show(BOUNCE_OB, on_boost > 0.001)
bake(M_RS25_SHELL.node_tree, value_path("Diamonds"), np.where(SET == 0, 1.0, np.where(SET == 1, 0.25, 0.0)))
las_on = np.where((IDX >= FAIRSEP + 6) & (IDX < FAIRSEP + 6 + 36), 1.0, 0.0) * gsmooth(
    ((IDX >= FAIRSEP + 6) & (IDX < FAIRSEP + 42)).astype(float), 1.5)
for lp in LAS_PLUMES:
    set_color_keys(lp, np.stack([las_on * 40.0, flick, phase], 1), const=False)
    show(lp, las_on > 0.01)
# nozzle interiors glow (Object Info colour = heat)
set_color_keys(ENG, np.stack([on_core, on_core, on_core], 1), const=False)
s2heat = np.clip(gsmooth(on_s2, 6) * 1.2, 0, 1)
set_color_keys(s2noz, np.stack([s2heat] * 3, 1), const=False)
# the twilight jellyfish: a sunlit bell trailing kilometres behind in SET 2
jg = np.clip(ramp(IDX, BSEP + 2 * FPS, BSEP + 8 * FPS) * (1 - ramp(IDX, SES - 20, SES)), 0, 1) * (SET == 2)
bake(M_JELLY.node_tree, value_path("Gain"), jg * (1.0 + 0.3 * BASS))
bake(M_JELLY.node_tree, value_path("Evolve"), np.cumsum(np.full(N, 0.02)))
jr = np.interp(IDX, [BSEP, FAIRSEP, SES], [250.0, 700.0, 1100.0])
jl = np.interp(IDX, [BSEP, FAIRSEP, SES], [900.0, 2600.0, 3800.0])
bake_vec(jelly, "scale", np.stack([jr, jr, jl], 1))

# --- steam (VDB playback + drift), vents, trail, deluge water, wetness ---
evo = np.cumsum(np.full(N, 1.0 / FPS))
for o, nfi, st0, speed, base, drift, sc0, kind in STEAMV:
    vdb_play(o, nfi, st0, speed)
    tt_ = np.maximum(IDX - st0, 0) / FPS
    ease = 1 - np.exp(-tt_ / 6.0)
    pos = np.array(base)[None, :] + np.array(drift)[None, :] * (6.0 * ease + 0.35 * tt_)[:, None]
    bake_vec(o, "location", pos)
plume_base_z = RIG_POS[:, 2] - 2.0
steam_glow = on_boost * PULSE * np.clip(1 - ALT / 420, 0.05, 1) * (1 + 1.5 * IGN_FLASH)
bake(M_STEAM.node_tree, value_path("Glow"), steam_glow * 0.35)
bake(M_STEAM.node_tree, value_path("PlumeZ"), np.minimum(plume_base_z, PAD_TOP + 30.0))
bake(M_STEAM_EARLY.node_tree, value_path("Glow"), on_core * 0.10 * (IDX < IGN) + steam_glow * 0.25)
bake(M_STEAM_EARLY.node_tree, value_path("PlumeZ"), np.full(N, PAD_TOP))
bake(M_VENT.node_tree, value_path("Density"), (0.25 + 0.5 * PRE + 0.2 * BASS * (IDX < IGN)) * (1 - ramp(IDX, LIFT, LIFT + 20)))
tr_len = np.maximum(RIG_POS[:, 2] - PAD_TOP - 6.0, 0.1)
tr_r = 13.0 + 0.06 * ALT
bake_vec(trail, "location", np.stack([np.zeros(N), RIG_POS[:, 1] * 0.5, np.full(N, PAD_TOP)], 1))
bake_vec(trail, "scale", np.stack([tr_r, tr_r, tr_len], 1))

def vkeys(mat, density, evolve, glow=None):
    nt = mat.node_tree
    bake(nt, value_path("Density"), density)
    bake(nt, value_path("Evolve"), evolve)
    if glow is not None:
        bake(nt, value_path("Glow"), glow)


vkeys(M_TRAIL, np.full(N, 0.10) * ramp(IDX, LIFT, LIFT + 15), evo * 0.15, 0.25 * on_boost * PULSE)
wet = ramp(IDX, DELUGE_T + 20, DELUGE_T + 120)
for mat in (M_TERRAIN, M_CONCRETE, M_ML, M_ML_RAIL):
    bake(mat.node_tree, value_path("Wet"), wet)
bake(M_CURTAIN.node_tree, value_path("Gain"), DELUGE * 0.95)
bake(M_CURTAIN.node_tree, value_path("Flow"), np.cumsum(DELUGE) / FPS * 2.6)
bake(rainbirds, gn_input_path(rain_md, RAIN_T, "Time"), IDX / FPS)
bake(rainbirds, gn_input_path(rain_md, RAIN_T, "Amount"), DELUGE * (1 - 0.5 * ramp(IDX, IGN, IGN + 60)))
bake(sparks, gn_input_path(sparks_md, SPARK_TREE, "Time"), IDX / FPS)
bake(sparks, gn_input_path(sparks_md, SPARK_TREE, "Amount"), SPARKS * (0.7 + 0.3 * SNARE))
# frost sheds after liftoff; plume soot builds in flight
fr_amt = 1.0 - ramp(IDX, LIFT, LIFT + 90)
for mat in (M_FOAM, M_NOZ_OUT):
    bake(mat.node_tree, value_path("Frost"), fr_amt)
bake(M_SRB.node_tree, value_path("Soot"), 0.9 * ramp(IDX, IGN, IGN + 40))
bake(M_FOAM.node_tree, value_path("Soot"), 0.7 * ramp(IDX, IGN + 30, PAD_END))
bake(M_WHITE.node_tree, value_path("Soot"), 0.8 * ramp(IDX, RS25_T + 10, IGN + 60))
# flight smoke trail: hangs behind the vehicle along the flight path (SET 1 → early SET 2)
tf_loc = np.array([wl(i, (0, 0, -8.0)) for i in range(N)])
bake_vec(trail_f, "location", tf_loc)
bake_vec(trail_f, "rotation_euler", np.stack([np.radians(-PITCH) + math.pi, np.zeros(N), np.zeros(N)], 1))
fr_r = np.where(SET == 1, 14.0, 30.0) * alt_expand
bake_vec(trail_f, "scale", np.stack([fr_r, fr_r, np.where(SET == 1, 2400.0, 1800.0)], 1))
vkeys(M_TRAILF, np.where(SET == 1, 0.06, 0.025) * on_boost, evo * 0.15 + np.cumsum(np.full(N, 0.05)),
      0.15 * on_boost * PULSE)
# cloud deck sinks past the climb
bake_vec(DECK, "location", np.stack([np.zeros(N), np.zeros(N), DECK_TOP - 120.0], 1))
vkeys(M_DECKHAZE, np.full(N, 0.012), evo * 0.05, np.zeros(N))

# --- vapour cone (max-Q): bursts on the drop entry, then flickers with the kicks ---
vc_gain = np.clip(ramp(IDX, MAXQ - 2, MAXQ + 2) * (1 - ramp(IDX, D_END - 30, D_END)), 0, 1)
vc_gain = vc_gain * (0.45 + 0.9 * KICK * HOT + 0.25 * BASS)
vc_gain = vc_gain + np.where(IDX >= MAXQ, 1.2 * np.exp(-np.maximum(IDX - MAXQ, 0) / 6.0), 0.0)
vc_gain = vc_gain * (IDX >= MAXQ - 2)
bake(M_VCONE.node_tree, value_path("Gain"), vc_gain)
bake(M_VCONE.node_tree, value_path("Evolve"), evo * 2.5)
vcs = 1.0 + 0.05 * KICK
bake_vec(vcone, "scale", np.stack([8.0 * vcs, 8.0 * vcs, 34.0 * (1.0 + 0.1 * KICK)], 1))

# --- ice: shed on the snares (stateless bursts) + a sheet of it at liftoff ---
bake(flakes, gn_input_path(flakes_md, FLAKE_T, "Age"), np.minimum(SAGE, 60) / FPS)
bake(flakes, gn_input_path(flakes_md, FLAKE_T, "Seed"), np.maximum(SIDX, 0).astype(np.float64) * 7.0, const=True)
bake(flakes2, gn_input_path(flakes2_md, FLAKE_T, "Age"), np.maximum(IDX - LIFT, 0) / FPS)

# --- pad lights: floods, beacons, searchlights; blackout on T_black before ignition ---
pad_black = ((BLACK > 0) & (IDX < IGN + 2)).astype(np.float64)
pad_cut = np.clip(pad_black + ramp(IDX, IGN - 3, IGN), 0, 1) * (IDX < IGN + 1) + (IDX >= IGN)
pad_cut = np.clip(pad_cut, 0, 1)
flood_on = (1 - pad_cut) + (IDX >= IGN + 6) * 0.5
snare_flash = SNARE * (IDX < IGN) * 0.3
for o in FLOODS:
    if o.type == "LIGHT":
        bake(o.data, "energy", 1.8e6 * flood_on * (1 + snare_flash))
    else:
        set_color_keys(o, np.stack([flood_on * 80.0 * (1 + snare_flash)] * 3, 1) * np.array([0.9, 0.95, 1.0]),
                       const=False)
for o in WORKL:
    if o.type == "LIGHT":
        bake(o.data, "energy", (2.5e3 if o.name.startswith("WorkL") else 5.0e3) * (1 - pad_cut * (IDX < IGN + 6)))
    else:
        set_color_keys(o, np.stack([(1 - pad_cut * (IDX < IGN + 6)) * 40.0] * 3, 1))
rate = np.where(PRE > 0.66, 4.0, np.where(PRE > 0.2, 2.0, 1.0))
blink = (np.mod(BEAT_PH * rate + (TL["T_bi"] % 2) * (rate < 2), 1.0) < 0.35).astype(np.float64)
blink = np.maximum(blink, (HAT > 0.55) * (IDX < IGN)) * (1 - pad_black)        # hi-hats strobe the beacons
for o in BEACONS:
    if o.type == "LIGHT":
        bake(o.data, "energy", blink * 2.0e4, const=True)
    else:
        set_color_keys(o, np.stack([blink * 300.0, blink * 12.0, blink * 6.0], 1))
for k, o in enumerate(SEARCH):
    sw_ = np.sin(IDX / FPS * (0.21 + 0.05 * k) + k * 2.1)
    tilt = np.radians(14 + 7 * np.sin(IDX / FPS * 0.13 + k))
    rx = math.pi - tilt
    rz = np.radians(25 * sw_ + 120 * k)
    bake_vec(o, "rotation_euler", np.stack([rx, np.zeros(N), rz], 1))
    bake(o.data, "energy", 2.4e7 * (1 - pad_black) * (1 - 0.7 * ramp(IDX, IGN, IGN + 60)))

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
    set_color_keys(o, LED_OFF[None, :] + on[:, None] * (LED_ON - LED_OFF)[None, :])
colon_on = (BEAT_PH < 0.5).astype(float)
for o in COLON:
    set_color_keys(o, LED_OFF[None, :] + colon_on[:, None] * (LED_ON - LED_OFF)[None, :])
set_color_keys(SIGN_H, np.repeat(LED_ON[None, :], N, 0))
set_color_keys(SIGN_V, plus[:, None] * LED_ON[None, :])

# --- world / sky / exposure ---
nt = WORLD.node_tree
alt_m = np.zeros(N)
alt_m[SET == 0] = 30 + ALT[SET == 0]
alt_m[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [3500, 9000, 16000])
alt_m[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, MECO, SES], [32000, 58000, 76000, 85000])
alt_m[SET == 3] = np.interp(IDX[SET == 3], [SES, N], [92000, 100000])
bake(nt, 'nodes["Sky"].altitude', alt_m)
RE = 6360e3
dip = np.degrees(np.arccos(RE / (RE + alt_m)))
sun_el = np.full(N, -6.0)                                           # nautical twilight on the pad
sun_el[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, BSEP], [-5.0, -3.0])
sun_el[SET >= 2] = -dip[SET >= 2] - np.interp(IDX[SET >= 2], [BSEP, SES - 20, SES, SES + 3 * BAR, N],
                                              [2.6, 1.1, 0.25, -0.9, -2.6])
bake(nt, 'nodes["Sky"].sun_elevation', np.radians(sun_el))
bake(nt, value_path("EarthH"), (RE + alt_m) / RE)
bake(nt, value_path("SunEl"), np.radians(sun_el))
sun_flare = 1.0 + 2.5 * np.where(IDX >= SES, np.exp(-(IDX - SES) / 12.0), 0.0) + 0.6 * BIG * ONE
bake(nt, value_path("SunDisc"), (SET >= 2) * sun_flare)
bake(nt, value_path("EarthOverlay"), np.clip((alt_m - 8000) / 20000, 0, 1))
bake(nt, value_path("EarthScroll"), -np.cumsum(np.where(SET >= 2, 0.00012, 0.00002)))
show(haze_box, SET == 0)
vkeys(M_HAZE, np.full(N, 0.00022) * (1 - 0.3 * ramp(IDX, IGN, PAD_END)), evo * 0.01)
bake(nt, value_path("Stars"), np.where(SET == 0, 0.5, np.where(SET == 1, 0.8, 1.2)))
# the sun lamp follows the sky's sun (azimuth +Y); grazing light appears once it clears the
# local horizon at altitude (cloud tops at the end of the calm, the jellyfish, the limb)
sun_vis = np.clip((sun_el + dip + 0.3) / 0.6, 0, 1) * (SET >= 1)
el_r = np.radians(sun_el)
bake_vec(SUN_OB, "rotation_euler", np.stack([el_r - math.pi / 2, np.zeros(N), np.zeros(N)], 1))
bake(SUN, "energy", sun_vis * 6.0)
show(SUN_OB, sun_vis > 0.001)
bake(MOON, "energy", np.where(SET == 0, 0.04, np.where(SET == 1, 0.2, np.where(SET == 2, 1.2, 0.05))))
moon_rx = np.where(SET >= 2, math.radians(30) - math.pi / 2, math.radians(55))
moon_rz = np.where(SET >= 2, 0.0, math.radians(-150))
bake_vec(MOON_OB, "rotation_euler", np.stack([moon_rx, np.zeros(N), moon_rz], 1), const=True)
bake_vec(MOON, "color", np.stack([np.where(SET >= 2, 1.0, 0.55), np.where(SET >= 2, 0.62, 0.65),
                                  np.where(SET >= 2, 0.55, 1.0)], 1), const=True)
# exposure ride (research fix #2): night pad exposed for the floods; the RS-25 start lifts the
# scene a little; SRB ignition CLIPS (2-frame overexposure) and the camera stops down ~3 EV
expo = np.where(SET == 0, 0.7, np.where(SET == 1, 0.6, np.where(SET == 2, 1.0, -0.2)))
expo = expo - 0.3 * on_core * (SET == 0) * (IDX < IGN)
expo = expo - 2.2 * smoothstep(ramp(IDX, IGN + 2, IGN + 26)) * (SET == 0)
expo = expo + 0.8 * IGN_FLASH                                    # the overexposed pop (clips the highlights)
expo = expo + 0.35 * STROBE * HOT
pump = SHOT_ONB * (0.25 * KICK * HOT + 0.08 * np.sin(IDX * 0.9))  # onboard auto-exposure pumping
expo = expo + pump
blk = (BLACK > 0) & (IDX > IGN + 10)
expo = np.where(blk, expo - 5.0, expo)
expo = expo - 0.4 * ((IDX >= MECO) & (IDX < SES)) * ramp(IDX, MECO, MECO + 12) * (1 - ramp(IDX, SES - 2, SES))
bake(SC, "view_settings.exposure", expo)

# --- bake visibility (hide_render, CONSTANT) ---
for name, mask in VIS.items():
    ob = bpy.data.objects[name]
    bake(ob, "hide_render", (~mask).astype(np.float64), const=True)


# =====================================================================================
# 11. RENDER SETTINGS + COMPOSITOR (research fix #4: haze, heat shimmer, grain, dispersion;
#     onboard shots get barrel distortion; highlights clip like a real sensor)
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
SC.render.engine = "BLENDER_EEVEE"
ee = SC.eevee
ee.taa_render_samples = ARGS.samples or 32
ee.use_raytracing = True
ee.ray_tracing_options.resolution_scale = "2"
ee.use_volume_custom_range = True
ee.volumetric_start = 0.5
ee.volumetric_end = 4000.0
ee.volumetric_tile_size = "2"
ee.volumetric_samples = 96
ee.volumetric_sample_distribution = 0.8
ee.use_volumetric_shadows = True
ee.volumetric_shadow_samples = 12
ee.volumetric_ray_depth = 8
ee.motion_blur_steps = 1
ee.use_shadows = True
ee.shadow_pool_size = "1024"
ee.shadow_ray_count = 1
ee.shadow_step_count = 6
ee.use_fast_gi = True
ee.clamp_volume_indirect = 10.0
vl = SC.view_layers[0]
vl.use_pass_mist = True
WORLD.mist_settings.start = 40.0
WORLD.mist_settings.depth = 5000.0
WORLD.mist_settings.falloff = "QUADRATIC"

cg = bpy.data.node_groups.new("Post", "CompositorNodeTree")
cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
CN = cg.nodes.new
CL = cg.links.new
rl = CN("CompositorNodeRLayers")
go = CN("NodeGroupOutput")


def cval(name, v):
    n = CN("ShaderNodeValue")
    n.name = n.label = name
    n.outputs[0].default_value = v
    return n.outputs[0]


def cmath(op, a, b):
    n = CN("ShaderNodeMath")
    n.operation = op
    for k, x in enumerate((a, b)):
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = x
        else:
            CL(x, n.inputs[k])
    return n.outputs[0]


# atmospheric haze between camera and subject (mist pass × keyed amount → keyed haze colour)
haze_amt = cval("HazeAmt", 0.0)
hr, hg, hb = cval("HazeR", 0.02), cval("HazeG", 0.025), cval("HazeB", 0.04)
hcol = CN("CompositorNodeCombineColor")
for k, s_ in enumerate((hr, hg, hb)):
    CL(s_, hcol.inputs[k])
hmix = CN("ShaderNodeMix")
hmix.data_type = "RGBA"
CL(cmath("MINIMUM", cmath("MULTIPLY", rl.outputs["Mist"], haze_amt), 0.85), hmix.inputs[0])
CL(rl.outputs["Image"], hmix.inputs[6])
CL(hcol.outputs[0], hmix.inputs[7])
# heat shimmer: animated 4D noise → Displace, strength ∝ distance (mist) × keyed amount
hn = CN("ShaderNodeTexNoise")
hn.noise_dimensions = "4D"
hn.name = "HeatNoise"
hn.inputs["Scale"].default_value = 16.0
hn.inputs["Detail"].default_value = 2.0
sub = CN("ShaderNodeVectorMath")
sub.operation = "SUBTRACT"
CL(hn.outputs["Color"], sub.inputs[0])
sub.inputs[1].default_value = (0.5, 0.5, 0.5)
heat_amt = cval("HeatAmt", 0.0)
hstr = cmath("MULTIPLY", heat_amt, cmath("POWER", rl.outputs["Mist"], 0.5))
scl = CN("ShaderNodeVectorMath")
scl.operation = "MULTIPLY"
CL(sub.outputs[0], scl.inputs[0])
scl.inputs[1].default_value = (2.2, 0.7, 0.0)
scl2 = CN("ShaderNodeVectorMath")
scl2.operation = "SCALE"
CL(scl.outputs[0], scl2.inputs[0])
CL(hstr, scl2.inputs["Scale"])
dsp = CN("CompositorNodeDisplace")
CL(hmix.outputs[2], dsp.inputs["Image"])
CL(scl2.outputs["Vector"], dsp.inputs[1])
# glare: streaks only on true highlights (sun disc, SRB core), then a soft bloom
gs = CN("CompositorNodeGlare")
gs.inputs["Type"].default_value = "Streaks"
gs.inputs["Quality"].default_value = "Medium"
gs.inputs["Threshold"].default_value = 60.0
gs.inputs["Strength"].default_value = 0.5
gs.inputs["Streaks"].default_value = 6
gs.inputs["Streaks Angle"].default_value = math.radians(15)
gs.inputs["Fade"].default_value = 0.9
CL(dsp.outputs["Image"], gs.inputs["Image"])
gl = CN("CompositorNodeGlare")
gl.inputs["Type"].default_value = "Fog Glow"
gl.inputs["Quality"].default_value = "Medium"
gl.inputs["Threshold"].default_value = 1.4
gl.inputs["Strength"].default_value = 0.35
gl.inputs["Size"].default_value = 0.8
CL(gs.outputs["Image"], gl.inputs["Image"])
ld_ = CN("CompositorNodeLensdist")
ld_.name = "Lens"
ld_.inputs["Distortion"].default_value = -0.006
ld_.inputs["Dispersion"].default_value = 0.012
ld_.inputs["Fit"].default_value = True
CL(gl.outputs["Image"], ld_.inputs["Image"])
vg = CN("CompositorNodeEllipseMask")
vg.inputs["Size"].default_value = (1.25, 1.25)
vb = CN("CompositorNodeBlur") if hasattr(bpy.types, "CompositorNodeBlur") else None
mr = CN("ShaderNodeMapRange")
mr.inputs["To Min"].default_value = 0.70
mr.inputs["To Max"].default_value = 1.0
if vb is not None:
    try:
        vb.inputs["Size"].default_value = (300, 300)
    except Exception:
        pass
    CL(vg.outputs["Mask"], vb.inputs["Image"])
    CL(vb.outputs["Image"], mr.inputs["Value"])
else:
    CL(vg.outputs["Mask"], mr.inputs["Value"])
mul = CN("ShaderNodeMix")
mul.data_type = "RGBA"
mul.blend_type = "MULTIPLY"
mul.inputs[0].default_value = 1.0
CL(ld_.outputs["Image"], mul.inputs[6])
CL(mr.outputs["Result"], mul.inputs[7])
# film grain: per-frame 4D white noise, multiplicative (±~2.5 %), a touch stronger onboard
gn_ = CN("ShaderNodeTexWhiteNoise")
gn_.noise_dimensions = "4D"
gn_.name = "Grain"
gamt = cval("GrainAmt", 0.05)
gsub = cmath("MULTIPLY", cmath("SUBTRACT", gn_.outputs["Value"], 0.5), gamt)
gmul = CN("ShaderNodeMix")
gmul.data_type = "RGBA"
gmul.blend_type = "MULTIPLY"
gmul.inputs[0].default_value = 1.0
CL(mul.outputs[2], gmul.inputs[6])
gcol = CN("CompositorNodeCombineColor")
gadd = cmath("ADD", gsub, 1.0)
for k in range(3):
    CL(gadd, gcol.inputs[k])
CL(gcol.outputs[0], gmul.inputs[7])
CL(gmul.outputs[2], go.inputs["Image"])
SC.compositing_node_group = cg
SC.render.compositor_device = "GPU"

# compositor keys: haze per set/shot, heat shimmer near the ground, onboard barrel distortion
haze = np.where(SET == 0, 0.9 * (0.35 + SHOT_HAZE), np.where(SET == 1, 0.35, 0.0))
bake(cg, 'nodes["HazeAmt"].outputs[0].default_value', haze)
hz_col = np.where((SET == 0)[:, None], np.array([0.008, 0.013, 0.030])[None, :],
                  np.array([0.05, 0.065, 0.10])[None, :])
for k, nm in enumerate(("HazeR", "HazeG", "HazeB")):
    bake(cg, f'nodes["{nm}"].outputs[0].default_value', hz_col[:, k])
heat = np.where(SET == 0, SHOT_HAZE * 0.8 + 0.8 * on_boost * np.clip(1 - ALT / 300, 0, 1), 0.0)
bake(cg, 'nodes["HeatAmt"].outputs[0].default_value', heat)
bake(cg, 'nodes["HeatNoise"].inputs["W"].default_value', IDX / FPS * 1.6)
bake(cg, 'nodes["Grain"].inputs["W"].default_value', IDX * 1.0)
bake(cg, 'nodes["GrainAmt"].outputs[0].default_value', 0.045 + 0.03 * SHOT_ONB)
bake(cg, 'nodes["Lens"].inputs["Distortion"].default_value', -0.006 - 0.075 * SHOT_ONB)

# sound strip (for scrubbing in the GUI; the MP4 mux happens in render.sh)
if os.path.exists(ARGS.audio):
    se = SC.sequence_editor_create()
    se.strips.new_sound("Music", ARGS.audio, channel=1, frame_start=1)
SC.render.use_sequencer = False

# bookkeeping for render_frames.py / tests: per-frame EEVEE volume range wrapped around the
# subject (volumetric_start/end are not animatable): long-lens shots get a tight slab at km range
SC["shots"] = ";".join(f"{s}:{n}" for s, n in SHOTS)
SC["set_of_frame"] = "".join(str(int(k)) for k in SET)
subj = np.array([wl(i, (0, 0, 40.0)) for i in range(N)])
subj[SET == 0] = np.stack([np.full((SET == 0).sum(), 40.0), np.zeros((SET == 0).sum()),
                           np.minimum(subj[SET == 0, 2], PAD_Z + 80)], 1)
dist = np.linalg.norm(cam_loc - subj, axis=1)
vstart = np.full(N, 0.5)
vend = np.clip(1.8 * dist + 300.0, 600.0, 4000.0)
far = (SET == 0) & (dist > 700)
vstart[far] = np.maximum(dist[far] - 520.0, 0.5)
vend[far] = dist[far] + 420.0
vdist = np.full(N, 0.8)
s1 = SET == 1
close1 = s1 & (dist < 400)                                 # chase/onboard on the climb: fine slices near
vend[close1], vdist[close1] = 2500.0, 0.95
far1 = s1 & (dist >= 400)                                  # air-to-air long lens: a slab around the subject
vstart[far1] = np.maximum(dist[far1] - 1500.0, 0.5)
vend[far1] = dist[far1] + 1800.0
vdist[far1] = 0.5
vend = np.where(SET == 2, np.maximum(vend, 4.0 * dist + 1500.0), vend)
SC["vol_start"] = [float(round(v, 1)) for v in vstart]
SC["vol_end"] = [float(round(v, 1)) for v in vend]
SC["vol_dist"] = [float(round(v, 3)) for v in vdist]
# per-frame render quality (render_frames.py applies it; EEVEE settings are not animatable).
# MEASURED at 1080 (full res): volume lighting cost ∝ TAA samples; tile 2 → 4 saves ~30 %.
# Fine froxels only where volumes are close to the lens; fewer samples where volumes dominate.
FINE_TILE = {"A4_low", "B2_tower", "B4_tail", "D1_maxq", "D2_under", "C2_side"}
vtile = np.full(N, 4)
spp = np.full(N, 32)
for k_, (s0_, nm_) in enumerate(SHOTS):
    a_, b_ = s0_, SHOT_BOUNDS[k_ + 1]
    if nm_ in FINE_TILE:
        vtile[a_:b_] = 2
    if nm_.startswith(("B", "A4")):
        spp[a_:b_] = 20                                     # steam shots: the volumes carry the frame
    elif nm_.startswith(("C", "D", "F", "G")):
        spp[a_:b_] = 24
SC["vol_tile"] = [int(v) for v in vtile]
SC["spp"] = [int(v) for v in spp]
SC["entries"] = list(ENT)
os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(ARGS.out), compress=True)
print(f"[rocket2] saved {ARGS.out}  frames 1..{N}  engine {SC.render.engine}")
