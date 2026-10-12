"""POLICE — a realistic night police chase down a wet downtown boulevard (Blender 5.2, EEVEE).

Zeke 2026-10-12: rework the car-street scene "so it's a bit more realistic, the car and the buildings",
using the new skills + MCP, to his hardstyle track "police". The 2.5-D carstreet look (beat-locked
street, lights passing on the beat, nitro on the drop) is rebuilt in true 3-D:

  cars       REAL downloaded models (Sketchfab, CC BY 4.0, credits in ASSETS.md): a sports car fleeing,
             a police SUV chasing. Wheels spin at road speed, bodies pitch on kicks and lean in lane
             changes. The police lightbar strobes red/blue to the music (alternating 1/8 notes in drops,
             a white double-flash on every kick in the BIG drop) and throws coloured light on the
             wet road and facades. A helicopter spotlight pool follows the lead car.
  street     procedural, from CC0 Poly Haven scans (asphalt_02, concrete_floor_02): 4-lane avenue,
             wet asphalt (puddle-varying roughness → reflections), painted lane dashes, raised
             sidewalks, streetlights every 30 m, two cross streets per block.
  buildings  ~70 per period, one object each: concrete/stone/glass facades with a real window grid in
             world space, ~25 % of windows lit (warm/cool), lit storefronts + sign bands at street
             level, aviation beacons on the towers. Pattern keyed on a per-building id (custom prop) so
             the periodic copies are identical.
  PERIODIC   the street repeats every L metres (3 copies built). Cars and cameras move in true
             coordinates; every frame they are shifted back by k*L (k = floor(lead/L)), which is
             invisible because the street is identical; the key before each wrap is CONSTANT and the
             shutter opens at the frame start, so motion blur never smears across a wrap or a cut.

Everything is baked as keyframes (no drivers/handlers). Deterministic from the timeline npz.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
import scene_helpers as H  # noqa: E402
from scene_helpers import (MB, bake, bake_vec, blackbody, cube_obj, emissive, link, lk, M, nd,  # noqa: E402
                           new_mat, no_shadow, set_color_keys, show, tex_box, V, value_node)

SONG = os.path.join(REPO, "state", "tzeke_songs", "police")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=os.path.join(SONG, "timeline_30fps.npz"))
    ap.add_argument("--audio", default=os.path.join(SONG, "police.wav"))
    ap.add_argument("--cars", default="/data/iris_assets/cars")
    ap.add_argument("--polyhaven", default="/data/iris_assets/polyhaven")
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "scenes", "police3d", "police.blend"))
    ap.add_argument("--lead-yaw", type=float, default=0.0, help="deg: rotate the sports car so its nose is +Y")
    ap.add_argument("--police-yaw", type=float, default=0.0, help="deg: rotate the police SUV so its nose is +Y")
    return ap.parse_args(argv)


ARGS = parse_args()

# =====================================================================================
# 1. MUSIC → per-frame arrays
# =====================================================================================
TL = np.load(ARGS.timeline)
FPS = int(TL["fps"])
N = len(TL["rms"])
BPM = float(TL["bpm"])
BAR = 4 * 60.0 * FPS / BPM
IDX = np.arange(N)


def gsmooth(x, sigma):
    x = np.asarray(x, np.float64)
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="same")[r:-r]


def age_env(age, tau):
    a = np.asarray(age, np.float64)
    return np.where(a < 1e4, np.exp(-a / tau), 0.0)


def ramp(i, a, b):
    return np.clip((np.asarray(i, np.float64) - a) / max(b - a, 1e-9), 0.0, 1.0)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def hash01(*k):
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    return (h ^ (h >> 15)) / 4294967295.0


KICK = age_env(TL["T_kage"], 3.0)
SNARE = age_env(TL["T_sage"], 4.0)
HOT = TL["T_hot"].astype(np.float64)
BIG = TL["T_big"].astype(np.float64)
BASS = gsmooth(TL["bass"], 1.5)
_hh = np.clip(TL["high"] - gsmooth(TL["high"], 12), 0, None)
HAT = np.clip(_hh / (np.percentile(_hh, 98) + 1e-6), 0, 1)
BEAT_PH = TL["T_ph"].astype(np.float64)
STROBE = (TL["T_strobe_age"] == 0).astype(np.float64)
DOWNBEATS = np.unique(TL["T_bar_start"])
ENT = sorted(int(e) for e in TL["entries"])
SEGS = [tuple(int(v) for v in s) for s in TL["segs"]]


def snap(f):
    return int(DOWNBEATS[np.argmin(np.abs(DOWNBEATS - f))])


# =====================================================================================
# 2. THE CHASE (true coordinates; +Y = down the avenue)
# =====================================================================================
L = 480.0                                  # street period (m)
LANES = np.array([-5.4, -1.8, 1.8, 5.4])
hot_s = gsmooth(HOT, 12)
big_s = gsmooth(BIG, 12)
SPEED = 24.0 + 17.0 * hot_s + 12.0 * big_s + 5.0 * gsmooth(KICK * HOT, 2)   # m/s (~86-200 km/h)
Y_LEAD = 60.0 + np.cumsum(SPEED) / FPS
GAP = gsmooth(26.0 - 9.0 * hot_s - 5.0 * big_s + 3.0 * np.sin(IDX / FPS * 0.37), 8)
Y_POL = Y_LEAD - GAP
# lane changes: a new target lane at drop entries and every 4 bars inside drops (deterministic)
lane_t = np.zeros(N)
cur = 2
marks = sorted(set(ENT) | {snap(s0 + k * 4 * BAR) for s0, s1, h in SEGS if h for k in range(1, 8)
                           if s0 + k * 4 * BAR < s1})
for j, m in enumerate(marks):
    cur = int((cur + 1 + int(hash01(m, 3) * 3)) % 4)
    lane_t[m:] = LANES[cur]
lane_t[:marks[0] if marks else N] = LANES[2]
X_LEAD = gsmooth(lane_t, 9)
X_POL = gsmooth(np.r_[np.full(14, X_LEAD[0]), X_LEAD[:-14]], 6)       # follows ~0.5 s later
DX_LEAD = np.gradient(X_LEAD) * FPS
DX_POL = np.gradient(X_POL) * FPS
K_WRAP = np.floor((Y_LEAD - 140.0) / L) * L                          # shift applied to everything moving
WRAPS = np.nonzero(np.diff(K_WRAP) != 0)[0] + 1                      # first frame after each wrap

# =====================================================================================
# 3. SHOT LIST (cuts on drop entries and bar lines)
# =====================================================================================
SHOTS = []
_cyc_i = [0]
for s0, s1, h in SEGS:
    if h == 0:                                                         # calm: long shots
        kinds = ["heli", "chase_high", "side_low"] if s1 - s0 > 6 * BAR else ["chase_high"]
        step = max((s1 - s0) / len(kinds), 1)
        for k, kd in enumerate(kinds):
            SHOTS.append((snap(s0 + k * step) if k else s0, kd))
    else:
        cyc = ["front_reverse", "side_low", "bumper", "chase_high", "flyby", "heli"]
        per = 1 * BAR if BIG[s0:s1].mean() > 0.5 else 2 * BAR
        k, t = 0, s0
        while t < s1 - per * 0.6:
            SHOTS.append((snap(t) if k else s0, cyc[_cyc_i[0] % len(cyc)]))
            _cyc_i[0] += 1
            k += 1
            t += per
SHOTS = sorted({s: n for s, n in SHOTS}.items())
SHOTS = [(s, n) for j, (s, n) in enumerate(SHOTS) if j == 0 or s > SHOTS[j - 1][0]]
if SHOTS[0][0] != 0:
    SHOTS.insert(0, (0, "heli"))
CUTS = [s for s, _ in SHOTS[1:]]
cut_set = np.zeros(N, bool)
for c in list(CUTS) + list(WRAPS):
    if 0 < c < N:
        cut_set[c - 1] = True                    # hold the key before a cut/wrap (no smear)
H.init(N, cut_set)
print("[police] frames", N, "bpm", round(BPM, 1), "shots", len(SHOTS), "wraps", len(WRAPS))

# =====================================================================================
# 4. SCENE RESET + WORLD
# =====================================================================================
SC = bpy.context.scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.worlds,
             bpy.data.node_groups, bpy.data.images):
    for d in list(coll):
        coll.remove(d)
H.SC = SC
SC.render.fps = FPS
SC.frame_start, SC.frame_end = 1, N
SC.render.resolution_x, SC.render.resolution_y = 1920, 1080

WORLD = bpy.data.worlds.new("Night")
SC.world = WORLD
wn = WORLD.node_tree
wn.nodes.clear()
tc = nd(wn, "ShaderNodeTexCoord")
sep = nd(wn, "ShaderNodeSeparateXYZ")
lk(wn, tc.outputs["Generated"], sep.inputs[0])
grad = nd(wn, "ShaderNodeMapRange", inp={"From Min": -0.05, "From Max": 0.45})
lk(wn, sep.outputs[2], grad.inputs["Value"])
ramp_n = nd(wn, "ShaderNodeValToRGB")
ramp_n.color_ramp.elements[0].color = (0.055, 0.04, 0.035, 1)        # sodium/LED city glow at the horizon
ramp_n.color_ramp.elements[1].color = (0.004, 0.006, 0.012, 1)       # dark overcast sky
lk(wn, grad.outputs["Result"], ramp_n.inputs["Fac"])
bg = nd(wn, "ShaderNodeBackground", inp={"Strength": 1.0})
lk(wn, ramp_n.outputs["Color"], bg.inputs["Color"])
wo = nd(wn, "ShaderNodeOutputWorld")
lk(wn, bg.outputs[0], wo.inputs["Surface"])

# =====================================================================================
# 5. MATERIALS (CC0 scans; PBR rules from blender-photoreal-lookdev)
# =====================================================================================
PH = ARGS.polyhaven


def ph(name, kind):
    return os.path.join(PH, "tex_" + name, f"{name}_{kind}_2k.jpg")


def mat_asphalt():
    m, nt = new_mat("WetAsphalt")
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    col = tex_box(nt, ph("asphalt_02", "diff"), 3.0)
    rgh = tex_box(nt, ph("asphalt_02", "rough"), 3.0, True)
    nrm = tex_box(nt, ph("asphalt_02", "nor_gl"), 3.0, True)
    dark = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    dark.inputs[0].default_value = 1.0
    lk(nt, col.outputs["Color"], dark.inputs[6])
    dark.inputs[7].default_value = (0.42, 0.42, 0.44, 1)             # wet asphalt is darker (albedo ~0.05)
    # puddles: low-frequency noise → mirror-wet patches, the rest damp (rough 0.18-0.45)
    tcn = nd(nt, "ShaderNodeTexCoord")
    pn = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.09, "Detail": 4.0, "Roughness": 0.6})
    lk(nt, tcn.outputs["Object"], pn.inputs["Vector"])
    pud = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.48, "From Max": 0.58, "To Min": 1.0, "To Max": 0.0})
    lk(nt, pn.outputs["Fac"], pud.inputs["Value"])
    rr = nd(nt, "ShaderNodeMapRange", inp={"To Min": 0.16, "To Max": 0.46})
    lk(nt, rgh.outputs["Color"], rr.inputs["Value"])
    rmix = nd(nt, "ShaderNodeMix", data_type="FLOAT")
    lk(nt, pud.outputs["Result"], rmix.inputs[0])
    lk(nt, rr.outputs["Result"], rmix.inputs[2])
    rmix.inputs[3].default_value = 0.03
    lk(nt, dark.outputs[2], b.inputs["Base Color"])
    lk(nt, rmix.outputs[0], b.inputs["Roughness"])
    nm = nd(nt, "ShaderNodeNormalMap", inp={"Strength": 0.6})
    lk(nt, nrm.outputs["Color"], nm.inputs["Color"])
    lk(nt, nm.outputs["Normal"], b.inputs["Normal"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def mat_concrete(name, tint, tile=2.5, rough=(0.5, 0.85)):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    col = tex_box(nt, ph("concrete_floor_02", "diff"), tile)
    rgh = tex_box(nt, ph("concrete_floor_02", "rough"), tile, True)
    bw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, col.outputs["Color"], bw.inputs[0])
    wear = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.15, "From Max": 0.7, "To Min": 0.75, "To Max": 1.05})
    lk(nt, bw.outputs[0], wear.inputs["Value"])
    mul = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    mul.inputs[0].default_value = 1.0
    mul.inputs[6].default_value = tuple(tint) + (1,)
    lk(nt, wear.outputs["Result"], mul.inputs[7])
    lk(nt, mul.outputs[2], b.inputs["Base Color"])
    rr = nd(nt, "ShaderNodeMapRange", inp={"To Min": rough[0], "To Max": rough[1]})
    lk(nt, rgh.outputs["Color"], rr.inputs["Value"])
    lk(nt, rr.outputs["Result"], b.inputs["Roughness"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def mat_facade(name, tint, glassy=False):
    """Building skin: scan-textured wall with a WORLD-SPACE window grid (floor 3.6 m, bay 3.2 m).
    ~25 % of windows lit (warm or cool), keyed on the per-building id (object prop 'bid') so periodic
    copies match. Ground floor (z < 5 m): lit storefront glass + a coloured sign band."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    col = tex_box(nt, ph("concrete_floor_02", "diff"), 4.0, coord=None)
    bw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, col.outputs["Color"], bw.inputs[0])
    wear = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.15, "From Max": 0.7, "To Min": 0.7, "To Max": 1.05})
    lk(nt, bw.outputs[0], wear.inputs["Value"])
    wall = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    wall.inputs[0].default_value = 1.0
    wall.inputs[6].default_value = tuple(tint) + (1,)
    lk(nt, wear.outputs["Result"], wall.inputs[7])
    geo = nd(nt, "ShaderNodeNewGeometry")
    otc = nd(nt, "ShaderNodeTexCoord")                 # OBJECT coords: identical in the periodic copies
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, otc.outputs["Object"], sp.inputs[0])
    sn = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, geo.outputs["Normal"], sn.inputs[0])
    side = M(nt, "GREATER_THAN", M(nt, "ABSOLUTE", sn.outputs[0]), 0.5)          # faces ±X (toward the avenue)
    along = M(nt, "ADD", M(nt, "MULTIPLY", sp.outputs[1], side),
              M(nt, "MULTIPLY", sp.outputs[0], M(nt, "SUBTRACT", 1.0, side)))
    fl = M(nt, "FLOOR", M(nt, "DIVIDE", sp.outputs[2], 3.6))
    bay = M(nt, "FLOOR", M(nt, "DIVIDE", along, 3.2))
    fz = M(nt, "FRACT", M(nt, "DIVIDE", sp.outputs[2], 3.6))
    fa = M(nt, "FRACT", M(nt, "DIVIDE", along, 3.2))
    win = M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "GREATER_THAN", fz, 0.33 if not glassy else 0.10),
                                M(nt, "LESS_THAN", fz, 0.82 if not glassy else 0.95)),
            M(nt, "MULTIPLY", M(nt, "GREATER_THAN", fa, 0.20 if not glassy else 0.04),
              M(nt, "LESS_THAN", fa, 0.80 if not glassy else 0.96)))
    upper = M(nt, "GREATER_THAN", sp.outputs[2], 5.2)
    roof_cut = M(nt, "LESS_THAN", M(nt, "ABSOLUTE", sn.outputs[2]), 0.5)       # no windows on roofs
    win = M(nt, "MULTIPLY", M(nt, "MULTIPLY", win, upper), roof_cut)
    bid = nd(nt, "ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="bid")
    cmb = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, fl, cmb.inputs[0])
    lk(nt, bay, cmb.inputs[1])
    lk(nt, bid.outputs["Fac"], cmb.inputs[2])
    wn_ = nd(nt, "ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    lk(nt, cmb.outputs[0], wn_.inputs["Vector"])
    sepw = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, wn_.outputs["Color"], sepw.inputs[0])
    lit = M(nt, "GREATER_THAN", wn_.outputs["Value"], 0.80)
    warm = M(nt, "GREATER_THAN", sepw.outputs[1], 0.35)
    lcol = nd(nt, "ShaderNodeMix", data_type="RGBA")
    lk(nt, warm, lcol.inputs[0])
    lcol.inputs[6].default_value = (0.75, 0.85, 1.0, 1)
    lcol.inputs[7].default_value = (1.0, 0.70, 0.40, 1)
    dim = M(nt, "ADD", 0.25, M(nt, "MULTIPLY", sepw.outputs[2], 0.75))
    # storefront band (z < 4.6): lit glass in a per-building colour; sign strip 4.0-4.6 m
    # shop windows: one per 3.2 m bay with 0.35 m mullions, z 0.5-3.3; about half the bays lit
    shop_bay = M(nt, "MULTIPLY", M(nt, "GREATER_THAN", fa, 0.06), M(nt, "LESS_THAN", fa, 0.94))
    store = M(nt, "MULTIPLY", M(nt, "LESS_THAN", sp.outputs[2], 3.3), M(nt, "GREATER_THAN", sp.outputs[2], 0.5))
    store = M(nt, "MULTIPLY", M(nt, "MULTIPLY", store, shop_bay), roof_cut)
    shop_lit = M(nt, "GREATER_THAN", sepw.outputs[0], 0.45)
    has_sign = M(nt, "GREATER_THAN", M(nt, "FRACT", M(nt, "MULTIPLY", bid.outputs["Fac"], 0.7071)), 0.45)
    sign = M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "GREATER_THAN", sp.outputs[2], 3.75),
                                M(nt, "LESS_THAN", sp.outputs[2], 4.1)), M(nt, "MULTIPLY", roof_cut, has_sign))
    hue = nd(nt, "ShaderNodeCombineColor", mode="HSV")          # 5.x: CombineHSV is gone
    lk(nt, M(nt, "FRACT", M(nt, "MULTIPLY", bid.outputs["Fac"], 0.618)), hue.inputs[0])
    hue.inputs[1].default_value = 0.85
    hue.inputs[2].default_value = 1.0
    storec = nd(nt, "ShaderNodeMix", data_type="RGBA")
    lk(nt, M(nt, "GREATER_THAN", M(nt, "FRACT", M(nt, "MULTIPLY", bid.outputs["Fac"], 0.37)), 0.5), storec.inputs[0])
    storec.inputs[6].default_value = (1.0, 0.78, 0.55, 1)
    lk(nt, hue.outputs[0], storec.inputs[7])
    # emission = windows (dim) + storefront + sign
    e_win = M(nt, "MULTIPLY", M(nt, "MULTIPLY", win, lit), M(nt, "MULTIPLY", dim, 2.2))
    ecol = nd(nt, "ShaderNodeMix", data_type="RGBA")
    lk(nt, M(nt, "GREATER_THAN", M(nt, "ADD", store, sign), 0.5), ecol.inputs[0])
    lk(nt, lcol.outputs[2], ecol.inputs[6])
    lk(nt, storec.outputs[2], ecol.inputs[7])
    estr = M(nt, "ADD", e_win, M(nt, "ADD", M(nt, "MULTIPLY", M(nt, "MULTIPLY", store, shop_lit),
                                                     M(nt, "MULTIPLY", dim, 1.6)),
                                   M(nt, "MULTIPLY", sign, 4.0)))
    lk(nt, ecol.outputs[2], b.inputs["Emission Color"])
    lk(nt, estr, b.inputs["Emission Strength"])
    # glass where windows/storefronts are, wall elsewhere
    glass = M(nt, "MAXIMUM", win, store)
    bc = nd(nt, "ShaderNodeMix", data_type="RGBA")
    lk(nt, glass, bc.inputs[0])
    lk(nt, wall.outputs[2], bc.inputs[6])
    bc.inputs[7].default_value = (0.012, 0.014, 0.018, 1)
    lk(nt, bc.outputs[2], b.inputs["Base Color"])
    rmix = nd(nt, "ShaderNodeMix", data_type="FLOAT")
    lk(nt, glass, rmix.inputs[0])
    rmix.inputs[2].default_value = 0.78
    rmix.inputs[3].default_value = 0.05
    lk(nt, rmix.outputs[0], b.inputs["Roughness"])
    bmp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.35, "Distance": 0.08})      # window/shop recesses
    lk(nt, M(nt, "MULTIPLY", glass, -1.0), bmp.inputs["Height"])
    lk(nt, bmp.outputs["Normal"], b.inputs["Normal"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


M_ROAD = mat_asphalt()
M_WALK = mat_concrete("Sidewalk", (0.42, 0.41, 0.40), tile=2.0)
M_CURB = mat_concrete("Curb", (0.55, 0.54, 0.52), tile=1.0)
M_PAINT, _pnt = new_mat("LinePaint")
_b = nd(_pnt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.72, 0.72, 0.70, 1), "Roughness": 0.35})
lk(_pnt, _b.outputs[0], nd(_pnt, "ShaderNodeOutputMaterial").inputs["Surface"])
M_POLE, _pt = new_mat("Pole")
_b = nd(_pt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.05, 0.05, 0.055, 1), "Roughness": 0.4, "Metallic": 0.6})
lk(_pt, _b.outputs[0], nd(_pt, "ShaderNodeOutputMaterial").inputs["Surface"])
M_LAMP = emissive("StreetLamp", blackbody(3600), 40.0)
M_BEACON = emissive("Beacon", (1.0, 0.05, 0.02), 30.0, objcolor_gain=True)
FACADES = [mat_facade("FacadeLight", (0.62, 0.60, 0.56)), mat_facade("FacadeDark", (0.22, 0.22, 0.24)),
           mat_facade("FacadeSand", (0.58, 0.48, 0.36)), mat_facade("GlassTower", (0.10, 0.12, 0.14), glassy=True),
           mat_facade("FacadeBrick", (0.30, 0.21, 0.17))]

# =====================================================================================
# 6. THE STREET (one period built procedurally, then copied at -L, 0, +L)
# =====================================================================================
ROAD_W, WALK = 16.0, 4.0
XF = ROAD_W / 2 + WALK                      # building front line (|x|)
CROSS = [(80.0, 16.0), (320.0, 16.0)]       # cross streets: (start y, width) within the period
rng = np.random.default_rng(2026)
STREET = bpy.data.collections.new("StreetPeriod")
SC.collection.children.link(STREET)


def in_cross(y0, y1):
    return any(not (y1 <= c0 or y0 >= c0 + w) for c0, w in CROSS)


def building_runs():
    """[(y0, y1)] facade spans along one side of the period, avoiding the cross streets, summing to L."""
    spans = []
    y = 0.0
    while y < L - 1e-6:
        nxt = min((c0 for c0, w in CROSS if c0 >= y - 1e-6), default=L)
        if abs(nxt - y) < 1e-6:
            y += [w for c0, w in CROSS if abs(c0 - nxt) < 1e-6][0]
            continue
        end = min(nxt, L)
        n = max(1, int(round((end - y) / 24.0)))
        cuts = np.sort(rng.uniform(0.25, 0.75, n - 1)) if n > 1 else np.array([])
        edges = [y] + list(y + (end - y) * (np.arange(1, n) / n + (cuts - 0.5) * 0.35 / n)) + [end]
        for a, b in zip(edges[:-1], edges[1:]):
            spans.append((a, b))
        y = end
    return spans


BUILDINGS = []                               # (object, height)
bid = 0
for sx in (-1, 1):
    for y0, y1 in building_runs():
        bid += 1
        tall = rng.random() < 0.28
        h = float(rng.uniform(48, 125) if tall else rng.uniform(13, 42))
        depth = float(rng.uniform(16, 28))
        inset = float(rng.uniform(0.0, 1.2))
        mb = MB()
        xf = sx * (XF + inset)
        mb.box((xf + sx * depth / 2, (y0 + y1) / 2, h / 2), (depth, (y1 - y0) - 0.3, h))
        if h > 30 and rng.random() < 0.5:                                  # setback crown
            mb.box((xf + sx * (depth / 2 + 2.0), (y0 + y1) / 2, h + 5.0), (depth - 4.0, (y1 - y0) - 4.0, 10.0))
        mat = FACADES[3] if tall and rng.random() < 0.6 else FACADES[int(rng.integers(0, 5)) if not tall else int(rng.integers(0, 3))]
        ob = mb.obj(f"Bldg{bid:03d}", mat)
        ob["bid"] = float(bid)
        STREET.objects.link(ob)
        SC.collection.objects.unlink(ob)
        BUILDINGS.append((ob, h, xf, (y0 + y1) / 2))
# road, sidewalks, curbs, lane paint for the period
road = MB()
road.box((0, L / 2, -0.05), (ROAD_W, L, 0.1))
for c0, w in CROSS:                          # cross-street asphalt through the sidewalk + building line
    for sx in (-1, 1):
        road.box((sx * (XF + 20), c0 + w / 2, -0.05), (40.0 + WALK, w, 0.1))
ro = road.obj("Road", M_ROAD)
walk = MB()
curb = MB()
for sx in (-1, 1):
    y = 0.0
    for c0, w in CROSS + [(L, 0)]:
        if c0 > y:
            walk.box((sx * (ROAD_W / 2 + WALK / 2), (y + c0) / 2, 0.075), (WALK, c0 - y, 0.15))
            curb.box((sx * (ROAD_W / 2 + 0.1), (y + c0) / 2, 0.08), (0.22, c0 - y, 0.17))
        y = c0 + w
wo_ = walk.obj("Sidewalks", M_WALK)
co_ = curb.obj("Curbs", M_CURB)
paint = MB()
for lx in (-3.6, 0.0, 3.6):
    for y in np.arange(0.0, L, 12.0):
        if not in_cross(y, y + 4.0):
            paint.box((lx, y + 2.0, 0.006), (0.15, 4.0, 0.012))
for sx in (-1, 1):
    y = 0.0
    for c0, w in CROSS + [(L, 0)]:
        if c0 > y:
            paint.box((sx * 7.6, (y + c0) / 2, 0.006), (0.15, c0 - y, 0.012))
        y = c0 + w
for c0, w in CROSS:                          # zebra crossings at each cross street
    for k in range(9):
        paint.box((-6.4 + k * 1.6, c0 - 2.0, 0.006), (0.8, 3.0, 0.012))
po = paint.obj("LanePaint", M_PAINT)
for o in (ro, wo_, co_, po):
    STREET.objects.link(o)
    SC.collection.objects.unlink(o)
# streetlights every 30 m, staggered sides, plus beacons on the towers
LAMPS = []
for k, y in enumerate(np.arange(15.0, L, 30.0)):
    if in_cross(y - 3, y + 3):
        continue
    for sx in (-1, 1):
        mb = MB()
        px = sx * (ROAD_W / 2 + 0.7)
        mb.box((px, y, 4.5), (0.22, 0.22, 9.0))
        mb.box((px - sx * 1.2, y, 8.95), (2.4, 0.16, 0.14))
        pole = mb.obj(f"Pole{k}{'LR'[sx > 0]}", M_POLE)
        head = cube_obj(f"LampHead{k}{'LR'[sx > 0]}", M_LAMP, loc=(px - sx * 2.3, y, 8.8), scale=(0.35, 0.18, 0.05))
        no_shadow(head)
        ld = bpy.data.lights.new(f"Street{k}{'LR'[sx > 0]}", "SPOT")
        ld.energy = 2600.0
        ld.color = blackbody(3600)
        ld.spot_size = math.radians(125)
        ld.spot_blend = 0.6
        ld.use_shadow = False
        lo = bpy.data.objects.new(ld.name, ld)
        lo.location = (px - sx * 2.3, y, 8.6)
        for o in (pole, head):
            STREET.objects.link(o)
            SC.collection.objects.unlink(o)
        STREET.objects.link(lo)
        LAMPS.append(lo)
BEACONS = []
for ob, h, xf, yc in BUILDINGS:
    if h > 60:
        bo = cube_obj(f"Beacon{ob.name}", M_BEACON, loc=(xf + np.sign(xf) * 6, yc, h + 0.4), scale=(0.25, 0.25, 0.25))
        no_shadow(bo)
        STREET.objects.link(bo)
        SC.collection.objects.unlink(bo)
        BEACONS.append(bo)
# the period copies: collection instances at -L and +L (identical by construction)
for k in (-1, 1):
    inst = bpy.data.objects.new(f"StreetCopy{k:+d}", None)
    inst.instance_type = "COLLECTION"
    inst.instance_collection = STREET
    inst.location = (0, k * L, 0)
    link(inst)
blink = ((IDX / FPS * 0.5) % 1.0 < 0.18).astype(float)          # aviation beacons ~0.5 Hz
for bo in BEACONS:
    set_color_keys(bo, np.stack([blink, blink * 0.05, blink * 0.02], 1))
print("[police] street:", len(BUILDINGS), "buildings,", len(LAMPS), "streetlights,", len(BEACONS), "beacons / period")

# =====================================================================================
# 7. THE CARS (real models; normalised: nose +Y, centred, wheels on z=0)
# =====================================================================================


def split_by_y(o, y_split, tag):
    """Split a mesh into its rear (y < y_split) and front halves → two objects (axle groups)."""
    import bmesh
    out = []
    for nm, keep_front in (("R", False), ("F", True)):
        bm = bmesh.new()
        try:
            bm.from_mesh(o.data)
            kill = [v for v in bm.verts if (v.co.y > y_split) != keep_front]
            bmesh.ops.delete(bm, geom=kill, context="VERTS")
            if len(bm.verts) == 0:
                continue
            me = bpy.data.meshes.new(f"{o.name}_{nm}")
            bm.to_mesh(me)
        finally:
            bm.free()
        for m in o.data.materials:
            me.materials.append(m)
        ob = bpy.data.objects.new(me.name, me)
        link(ob)
        out.append(ob)
    bpy.data.objects.remove(o, do_unlink=True)
    return out


def import_car(slug, tag, yaw_deg, wheel_mats=None, axle_split=None):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=os.path.join(ARGS.cars, slug, "model.glb"))
    new = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before]
    meshes = [o for o in new if o.type == "MESH"]
    bpy.context.view_layer.update()
    pts = np.array([list(o.matrix_world @ Vector(c)) for o in meshes for c in o.bound_box])
    lo, hi = pts.min(0), pts.max(0)
    centre = Matrix.Translation(Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    R = Matrix.Rotation(math.radians(yaw_deg), 4, "Z")
    rig = bpy.data.objects.new(f"Car{tag}", None)
    link(rig)
    body = bpy.data.objects.new(f"Body{tag}", None)       # pitch/roll pivot
    link(body, rig)
    wheels = []
    for o in meshes:
        mw = R @ centre @ o.matrix_world
        me = o.data.copy()
        me.transform(mw)
        o.data = me
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    # cars whose wheel meshes span both axles: split them per axle first (rotate about each axle)
    if wheel_mats:
        for o in list(meshes):
            if any(m is not None and m.name.split(".")[0] in wheel_mats for m in o.data.materials):
                meshes.remove(o)
                meshes += split_by_y(o, axle_split, tag)
    # wheels: meshes whose bbox sits low at a corner and is about a wheel's size (or a whole axle pair)
    for o in meshes:
        nv = len(o.data.vertices)
        bb = np.empty(nv * 3, np.float32)
        o.data.vertices.foreach_get("co", bb)
        bb = bb.reshape(-1, 3)
        if nv == 0:
            o.parent = body
            continue
        mn, mx = bb.min(0), bb.max(0)
        sz, c = mx - mn, (mn + mx) / 2
        is_corner = 0.45 < sz[2] < 0.9 and 0.45 < sz[1] < 0.9 and sz[0] < 0.5 and abs(c[0]) > 0.55 and c[2] < 0.55
        is_axle = bool(wheel_mats) and any(m is not None and m.name.split(".")[0] in wheel_mats
                                          for m in o.data.materials) and sz[1] < 0.9 and c[2] < 0.55
        if is_corner or is_axle:
            o.data.transform(Matrix.Translation(Vector(-c)))
            o.location = Vector(c)
            o.parent = rig
            wheels.append(o)
        else:
            o.parent = body
    for o in new:
        try:
            if o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        except ReferenceError:                       # split meshes were already removed
            pass
    print(f"[police] car {tag}: {len(meshes)} meshes, {len(wheels)} wheel parts, size {np.round(hi - lo, 2)}")
    return rig, body, wheels


LEAD, LEAD_BODY, LEAD_WHEELS = import_car("sf_sports_car", "Lead", ARGS.lead_yaw,
                                          wheel_mats={"Material_350", "disk", "Material_369", "Material_353"},
                                          axle_split=0.13)
POL, POL_BODY, POL_WHEELS = import_car("sf_police_suv", "Police", ARGS.police_yaw)


def car_lights(rig, tag, nose_y, tail_y, width, head_z, tail_z, head_energy):
    for sx in (-1, 1):
        ld = bpy.data.lights.new(f"Head{tag}{sx}", "SPOT")
        ld.energy = head_energy
        ld.color = blackbody(5600)
        ld.spot_size = math.radians(38)
        ld.spot_blend = 0.45
        ld.use_shadow = False
        lo = link(bpy.data.objects.new(ld.name, ld), rig)
        lo.location = (sx * width, nose_y, head_z)
        lo.rotation_euler = (math.radians(86), 0, 0)               # -Z → +Y: aim down the road, slightly down
        lt = bpy.data.lights.new(f"Tail{tag}{sx}", "POINT")
        lt.energy = 25.0
        lt.color = (1.0, 0.02, 0.01)
        lt.shadow_soft_size = 0.15
        lt.use_shadow = False
        lo2 = link(bpy.data.objects.new(lt.name, lt), rig)
        lo2.location = (sx * width, tail_y - 0.15, tail_z)


car_lights(LEAD, "Lead", 2.25, -2.3, 0.68, 0.62, 0.78, 900.0)
car_lights(POL, "Police", 2.2, -2.2, 0.72, 0.82, 0.95, 1100.0)
# police lightbar: my own emitters + coloured lights (the model's bar is unlit plastic)
M_RED = emissive("BarRed", (1.0, 0.02, 0.02), 60.0, objcolor_gain=True)
M_BLUE = emissive("BarBlue", (0.02, 0.15, 1.0), 60.0, objcolor_gain=True)
BAR_OBJ = {}
for nm, mat, x in (("red", M_RED, -0.38), ("blue", M_BLUE, 0.38)):
    bo = cube_obj(f"Bar{nm}", mat, loc=(x, 0.05, 1.83), scale=(0.24, 0.1, 0.06))
    bo.parent = POL_BODY
    no_shadow(bo)
    ld = bpy.data.lights.new(f"BarLight{nm}", "POINT")
    ld.color = (1.0, 0.05, 0.03) if nm == "red" else (0.08, 0.25, 1.0)
    ld.shadow_soft_size = 0.2
    ld.use_shadow = False
    lo = link(bpy.data.objects.new(ld.name, ld), POL_BODY)
    lo.location = (x * 1.2, 0.05, 2.05)
    BAR_OBJ[nm] = (bo, ld)
# helicopter spotlight following the lead car
HELI = bpy.data.lights.new("HeliSpot", "SPOT")
HELI.energy = 2.4e6
HELI.color = blackbody(6200)
HELI.spot_size = math.radians(7.5)
HELI.spot_blend = 0.35
HELI.use_shadow = True
HELI_OB = link(bpy.data.objects.new("HeliSpot", HELI))

# =====================================================================================
# 8. MOTION (true → rendered: minus K_WRAP)
# =====================================================================================
WHEEL_R = 0.34


def place_car(rig, body, wheels, X, Y, DX, speed, tag):
    yaw = -np.arctan2(DX, np.maximum(speed, 1.0))                     # heading change in lane changes
    loc = np.stack([X, Y - K_WRAP, np.zeros(N)], 1)
    bake_vec(rig, "location", loc)
    bake_vec(rig, "rotation_euler", np.stack([np.zeros(N), np.zeros(N), yaw], 1))
    acc = np.gradient(gsmooth(speed, 3)) * FPS
    pitch = np.radians(np.clip(-acc * 0.12, -2.5, 2.5) + 0.6 * KICK * np.maximum(HOT, 0.4))   # squat/dive + kick bounce
    roll = np.radians(np.clip(np.gradient(DX) * FPS * 0.8, -4, 4))
    bob = 0.012 * np.sin(IDX * 1.9 + (0 if tag == "Lead" else 1.3)) + 0.025 * KICK * np.maximum(HOT, 0.4)
    bake_vec(body, "rotation_euler", np.stack([pitch, roll, np.zeros(N)], 1))
    bake_vec(body, "location", np.stack([np.zeros(N), np.zeros(N), bob], 1))
    ang = np.cumsum(speed) / FPS / WHEEL_R                            # rolling, radians
    for w in wheels:
        bake(w, "rotation_euler", ang, index=0)                      # unwrapped: no backward snap


place_car(LEAD, LEAD_BODY, LEAD_WHEELS, X_LEAD, Y_LEAD, DX_LEAD, SPEED, "Lead")
place_car(POL, POL_BODY, POL_WHEELS, X_POL, Y_POL, DX_POL, SPEED, "Police")
# lightbar strobe: calm = alternate per half beat; drops = alternate per 1/8 + quad bursts; BIG = white kick flashes
ph8 = (BEAT_PH * 4) % 1.0
alt_fast = np.floor(BEAT_PH * 4).astype(int) % 2
alt_slow = np.floor(BEAT_PH * 2).astype(int) % 2
burst = (ph8 < 0.45).astype(float)
red_on = np.where(HOT > 0.5, (alt_fast == 0) * burst, (alt_slow == 0) * 1.0)
blue_on = np.where(HOT > 0.5, (alt_fast == 1) * burst, (alt_slow == 1) * 1.0)
white = BIG * KICK
for nm, on in (("red", red_on), ("blue", blue_on)):
    bo, ld = BAR_OBJ[nm]
    lvl = np.clip(on * (0.8 + 0.6 * HAT) + white * 1.5, 0, 3)
    set_color_keys(bo, np.stack([lvl, lvl, lvl], 1), const=True)
    bake(ld, "energy", lvl * 2600.0, const=True)                 # 9000 flooded the whole street red
hp = np.stack([X_LEAD + 14.0 + 3 * np.sin(IDX * 0.011), Y_LEAD - 26.0 - K_WRAP, np.full(N, 62.0)], 1)
bake_vec(HELI_OB, "location", hp)
tgt = np.stack([X_LEAD, Y_LEAD - K_WRAP + 2.0, np.zeros(N)], 1) + np.stack(
    [1.2 * np.sin(IDX * 0.07), 1.5 * np.sin(IDX * 0.053 + 1), np.zeros(N)], 1)
d = tgt - hp
rot = []
for v in d:
    q = Vector(v).to_track_quat("-Z", "Y").to_euler()
    rot.append((q.x, q.y, q.z))
bake_vec(HELI_OB, "rotation_euler", np.array(rot))
bake(HELI, "energy", 2.4e6 * (0.85 + 0.15 * KICK * HOT))

# =====================================================================================
# 9. CAMERA
# =====================================================================================
CAM_D = bpy.data.cameras.new("Cam")
CAM_D.sensor_width = 36.0
CAM_D.clip_start = 0.2
CAM_D.clip_end = 3000.0
CAM_D.dof.use_dof = True
CAM_D.dof.aperture_blades = 7
CAM = link(bpy.data.objects.new("Cam", CAM_D))
SC.camera = CAM


def lead(i, dy=0.0, dx=0.0, z=0.0):
    return Vector((X_LEAD[i] + dx, Y_LEAD[i] + dy, z))


def pol(i, dy=0.0, dx=0.0, z=0.0):
    return Vector((X_POL[i] + dx, Y_POL[i] + dy, z))


def shot(kind, i, s0, s1):
    t = (i - s0) / max(s1 - s0 - 1, 1)
    side = 1 if hash01(s0, 7) > 0.5 else -1
    if kind == "chase_high":                                   # behind the cruiser, lights in the foreground
        cam = pol(i, -14.0 - 4 * t, 1.5 * side, 4.2)
        return cam, lead(i, 0, 0, 0.8) * 0.65 + pol(i, 0, 0, 1.0) * 0.35, 32, 11
    if kind == "front_reverse":                                # ahead of the fleeing car, looking back
        cam = lead(i, 11.0 + 2 * t, -1.2 * side, 1.15)
        return cam, lead(i, -6.0, 0, 0.7), 45, 5.6
    if kind == "side_low":                                     # tracking alongside, low
        cam = lead(i, 1.5 - 3 * t, 4.6 * side, 0.55)
        return cam, lead(i, -3.0, 0, 0.6), 24, 5.6
    if kind == "bumper":                                       # on the sports car's rear, looking back at the cruiser
        cam = lead(i, -2.55, 0.35 * side, 0.62)
        return cam, pol(i, 0, 0, 0.9), 18, 8
    if kind == "flyby":                                        # static on the corner; they blast past
        mid = (s0 + s1) // 2
        anchor = Vector((XF - 1.5 * 0 + 0.0, Y_LEAD[mid], 0)) if side > 0 else Vector((-XF, Y_LEAD[mid], 0))
        cam = Vector((side * (ROAD_W / 2 + 2.2), Y_LEAD[mid] + 2.0, 1.1))
        return cam, lead(i, 0, 0, 0.7) * 0.7 + pol(i, 0, 0, 0.9) * 0.3, 24, 8
    # heli: 60 m up, trailing and off to the side, slow drift
    cam = lead(i, -60.0 + 10 * t, 3.0 * side, 64.0)          # above the avenue (never inside a tower)
    return cam, lead(i, 6.0, 0, 0), 55, 11


def shake(i, amp):
    k = int(TL["T_kidx"][i])
    j = Vector((hash01(k, 1) - 0.5, hash01(k, 2) - 0.5, hash01(k, 3) - 0.5)) * 2 * KICK[i]
    r = Vector((math.sin(i * 1.7) + 0.5 * math.sin(i * 3.1), math.sin(i * 1.3 + 1), math.sin(i * 1.9 + 2)))
    return (j + 0.15 * r) * amp


cam_loc = np.zeros((N, 3))
cam_rot = np.zeros((N, 3))
cam_lens = np.zeros(N)
cam_focus = np.zeros(N)
cam_fstop = np.zeros(N)
bounds = [s for s, _ in SHOTS] + [N]
prev = None
for k, (s0, kind) in enumerate(SHOTS):
    s1 = bounds[k + 1]
    amp = {"bumper": 0.06, "side_low": 0.05, "front_reverse": 0.05, "chase_high": 0.08, "flyby": 0.03, "heli": 0.25}[kind]
    for i in range(s0, s1):
        cam, tgt, lens, fs = shot(kind, i, s0, s1)
        a = amp * (0.4 + 1.2 * HOT[i] + 1.0 * BIG[i])
        cam = cam + shake(i, a)
        tgt = tgt + shake(i, a * 1.6)
        off = Vector((0, K_WRAP[i], 0))
        cam, tgt = cam - off, tgt - off
        q = (tgt - cam).to_track_quat("-Z", "Y")
        e = q.to_euler("XYZ", prev) if (prev is not None and i != s0) else q.to_euler("XYZ")
        prev = e
        cam_loc[i] = cam
        cam_rot[i] = (e.x, e.y, e.z)
        cam_lens[i] = lens
        cam_focus[i] = (tgt - cam).length
        cam_fstop[i] = fs
bake_vec(CAM, "location", cam_loc)
bake_vec(CAM, "rotation_euler", cam_rot)
bake(CAM_D, "lens", cam_lens)
bake(CAM_D, "dof.focus_distance", cam_focus)
bake(CAM_D, "dof.aperture_fstop", cam_fstop)

# =====================================================================================
# 10. RENDER SETTINGS + COMPOSITOR
# =====================================================================================
SC.render.engine = "BLENDER_EEVEE"
ee = SC.eevee
ee.taa_render_samples = 32
for attr, val in (("use_raytracing", True), ("ray_tracing_method", "SCREEN"), ("use_shadows", True),
                  ("fast_gi_method", "GLOBAL_ILLUMINATION"), ("use_fast_gi", True)):
    try:
        setattr(ee, attr, val)
    except Exception:
        pass
try:
    ee.ray_tracing_options.resolution_scale = "2"
except Exception:
    pass
SC.render.use_motion_blur = True
SC.render.motion_blur_shutter = 0.5
try:
    SC.render.motion_blur_position = "START"
except Exception:
    pass
vs = SC.view_settings
vs.view_transform = "AgX"
vs.look = "AgX - Medium High Contrast"
vs.exposure = 0.0
bake(SC, "view_settings.exposure", np.full(N, 0.0) + 0.35 * STROBE * BIG)
vl = SC.view_layers[0]
vl.use_pass_mist = True
WORLD.mist_settings.start = 30.0
WORLD.mist_settings.depth = 900.0
WORLD.mist_settings.falloff = "QUADRATIC"
cg = bpy.data.node_groups.new("Post", "CompositorNodeTree")
cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
rl = cg.nodes.new("CompositorNodeRLayers")
go = cg.nodes.new("NodeGroupOutput")
fog = cg.nodes.new("ShaderNodeMix")
fog.data_type = "RGBA"
mm = cg.nodes.new("ShaderNodeMath")
mm.operation = "MULTIPLY"
mm.inputs[1].default_value = 0.55
cg.links.new(rl.outputs["Mist"], mm.inputs[0])
cg.links.new(mm.outputs[0], fog.inputs[0])
cg.links.new(rl.outputs["Image"], fog.inputs[6])
fog.inputs[7].default_value = (0.035, 0.03, 0.03, 1)
gl = cg.nodes.new("CompositorNodeGlare")
gl.inputs["Type"].default_value = "Fog Glow"
gl.inputs["Quality"].default_value = "Medium"
gl.inputs["Threshold"].default_value = 1.2
gl.inputs["Strength"].default_value = 0.4
cg.links.new(fog.outputs[2], gl.inputs["Image"])
st = cg.nodes.new("CompositorNodeGlare")
st.inputs["Type"].default_value = "Streaks"
st.inputs["Quality"].default_value = "Medium"
st.inputs["Threshold"].default_value = 8.0
st.inputs["Strength"].default_value = 0.3
st.inputs["Streaks"].default_value = 2
cg.links.new(gl.outputs["Image"], st.inputs["Image"])
ld_ = cg.nodes.new("CompositorNodeLensdist")
ld_.inputs["Distortion"].default_value = -0.008
ld_.inputs["Dispersion"].default_value = 0.012
ld_.inputs["Fit"].default_value = True
cg.links.new(st.outputs["Image"], ld_.inputs["Image"])
cg.links.new(ld_.outputs["Image"], go.inputs["Image"])
SC.compositing_node_group = cg
SC.render.compositor_device = "GPU"

for name, mask in H.VIS.items():
    ob = bpy.data.objects[name]
    bake(ob, "hide_render", (~mask).astype(np.float64), const=True)
if os.path.exists(ARGS.audio):
    se = SC.sequence_editor_create()
    se.strips.new_sound("Music", ARGS.audio, channel=1, frame_start=1)
SC.render.use_sequencer = False
SC["shots"] = ";".join(f"{s}:{n}" for s, n in SHOTS)
os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(ARGS.out), compress=True)
print(f"[police] saved {ARGS.out}  frames 1..{N}  shots {len(SHOTS)}")
