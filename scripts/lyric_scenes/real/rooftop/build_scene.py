"""BE MY ONE — realistic cyberpunk rooftop party (3-D, EEVEE). First pass, 2026-10-13.

    ~/.local/bin/blender -b --factory-startup --disable-autoexec --python-exit-code 1 \
        --python scripts/lyric_scenes/real/rooftop/build_scene.py -- \
        --out scratch/scenes/bemyone_rooftop/bemyone.blend [--assets /data/iris_assets] [--markers]

Everything is baked as keyframes or analytic Geometry Nodes driven by Scene Time (rain, traffic, drones,
fireworks) — no handlers, no simulation caches, so any frame renders standalone (chunk-safe).
The camera never reacts to the music; every cut lands on a downbeat of the fitted 134.095 BPM grid.

Song map (Zeke's ear, snapped to the grid):  intro 0–29.18 · DROP1 29.18 (beat 65) · breakdown 43.50 (97)
· DROP2 57.82 (129) · short break 72.14 (161) · fly-through 75.72 (169) · BRIDGE 86.46 (193)
· LAST DROP 100.77 (225) · dip 111.51 (249) · final burst 118.67 (265) · music ends 125.83 (281) · 128.4 end.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rt_util import (FPS, NF, NB, P, T0, TT, add_gn, bake, bake_vec, boxes_mesh, bt, coll, deg, emit_mat,  # noqa: E402
                     fr, gn_tree, look_quat, mesh_obj, new_mat, points_obj, principled, simple_mat, smootherstep,
                     smoothstep)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--assets", default="/data/iris_assets")
ap.add_argument("--audio", default=os.path.join(HERE, "../../../../scratch/scenes/bemyone_rooftop/audio.json"))
ap.add_argument("--wav", default="/data/iris_assets/bemyone/be_my_one_master.wav")
ap.add_argument("--markers", action="store_true", help="direction-check markers (+Y cube on the flying car)")
A = ap.parse_args(argv)
BMO = os.path.join(A.assets, "bemyone")
AUD = json.load(open(A.audio))
KICK = np.asarray(AUD["kick"], dtype=np.float64)[:NF]
rng = np.random.default_rng(20261013)

# ---------------------------------------------------------------------------------------------------------
# 0. scene
# ---------------------------------------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
SC.name = "BeMyOne"
SC.render.fps, SC.render.fps_base = FPS, 1.0
SC.frame_start, SC.frame_end = 1, NF
SC.render.resolution_x, SC.render.resolution_y = 1920, 1080
SC.render.engine = "BLENDER_EEVEE"
SC.view_settings.view_transform = "AgX"
SC.view_settings.look = "AgX - Medium High Contrast"
SC.view_settings.exposure = 0.0
SC.render.use_motion_blur = False          # GN rain wraps around the camera: deform blur would smear the wraps
ee = SC.eevee
ee.taa_render_samples = 48
ee.use_raytracing = True
ee.ray_tracing_method = "SCREEN"
ee.ray_tracing_options.resolution_scale = "2"
ee.ray_tracing_options.trace_max_roughness = 0.45
ee.ray_tracing_options.screen_trace_quality = 0.5
ee.use_fast_gi = True
ee.use_shadows = True
ee.shadow_ray_count = 1
ee.shadow_step_count = 6
ee.clamp_surface_indirect = 10.0
ee.shadow_pool_size = "1024"
se = SC.sequence_editor_create()
if os.path.exists(A.wav):
    se.strips.new_sound("BeMyOne", A.wav, channel=1, frame_start=1)

C_CITY = coll("City")
C_ROOF = coll("Rooftop")
C_FX = coll("FX")
C_CAM = coll("Camera")

# ---------------------------------------------------------------------------------------------------------
# 1. timeline → per-frame control curves
# ---------------------------------------------------------------------------------------------------------
B_DROP1, B_BREAK1, B_DROP2, B_BREAK2, B_FLY, B_BRIDGE, B_LAST, B_DIP, B_FINAL, B_END = 65, 97, 129, 161, 169, 193, 225, 249, 265, 281
tD1, tBK, tD2, tBR, tFL, tBG, tLD, tDP, tFN, tEND = (bt(b) for b in (B_DROP1, B_BREAK1, B_DROP2, B_BREAK2, B_FLY,
                                                                      B_BRIDGE, B_LAST, B_DIP, B_FINAL, B_END))
T = TT
BEAT = (T - T0) / P
bfrac = BEAT - np.floor(BEAT)
PULSE = np.where(BEAT >= 0, np.exp(-bfrac * P / 0.16), 0.0)          # beat-locked decay 1→0 (grid, not kicks)


def seg(a, b):
    return (T >= a) & (T < b)


def ramp(a, b, v0, v1, ease=smoothstep):
    return v0 + (v1 - v0) * ease((T - a) / max(1e-6, b - a))


SHUT_A, SHUT_B = tEND - 2.4, tEND + 1.9            # outro: buildings shut off far→near, 123.4 → 127.7 s
city = np.zeros(NF)
city[:] = ramp(0.0, tD1, 0.14, 0.34)
city[seg(tD1, tBK)] = 1.0
city[seg(tBK, tD2)] = np.maximum(0.015, 1.0 - smoothstep((T[seg(tBK, tD2)] - tBK) / 0.45))
city[seg(tD2, tBR)] = 1.0
city[seg(tBR, tFL)] = 0.32
city[seg(tFL, tBG)] = 1.0
m = seg(tBG, tLD)
city[m] = ramp(tBG, tBG + 4 * 4 * P, 1.0, 0.16)[m]
city[seg(tLD, tDP)] = 1.0
city[seg(tDP, tFN)] = 0.6
city[T >= tFN] = 1.0
cityB = np.ones(NF)
cityB[seg(tLD, tDP)] = 1.35
strobe = np.zeros(NF)
strobe[seg(tD2, tBR)] = 1.0
strobe[seg(tFN, SHUT_A)] = 0.55
wphase = BEAT * 0.18
shut = smoothstep((T - SHUT_A) / (SHUT_B - SHUT_A))
DROPS = seg(tD1, tBK) | seg(tD2, tBR) | seg(tFL, tBG) | seg(tLD, tDP) | seg(tFN, tEND)
bb = np.zeros(NF)
bb[DROPS] = 1.0
bb[seg(tBR, tFL)] = 0.25
bb[seg(tBG, tLD)] = ramp(tBG, tBG + 8 * P, 1.0, 0.18)[seg(tBG, tLD)]
bb[seg(tDP, tFN)] = 0.45
bb *= 1.0 - shut
bbpulse = np.where(DROPS, PULSE, 0.0)
barno = np.floor((BEAT - B_DROP1) / 4.0)
bbhue = np.where(BEAT >= B_DROP1, (barno * 0.137) % 1.0, 0.0)
holo = np.zeros(NF)
holo[T >= tD1] = 0.55
holo[seg(tBK, tD2)] = 1.0
holo[seg(tBG, tLD)] = 0.35
holo *= 1.0 - smoothstep((T - (tEND - 1.0)) / 0.8)
LASER_ON = seg(tD1, tBK) | seg(tFL, tBG) | seg(bt(B_LAST + 4), tDP) | seg(tFN, tEND)
laser = np.where(LASER_ON, 0.30 + 0.70 * np.clip(KICK * 1.25, 0, 1), 0.0) * (1.0 - shut)
spire_h = np.clip(ramp(tBG, tLD - 0.15, 0.0, 1.0, ease=lambda x: np.clip(x, 0, 1) ** 1.6), 0, 1)
spire_h[T >= tLD] = 1.0
spire_glow = np.where(T >= tLD, 0.55 + 0.45 * PULSE, 0.0) * (1.0 - shut)
spire_h *= 1.0 - shut
drone_lv = np.where(seg(tD2, tFL + 1.6), 1.0, 0.0) * (1.0 - smoothstep((T - tFL) / 1.6)) * smoothstep((T - tD2) / 0.3)
sky = 0.45 + 0.55 * city * (1.0 - shut)
blink = (np.floor(T / 1.0) % 2 == 0).astype(float) * (1.0 - shut)
ctrl_curves = dict(city=city, cityB=cityB, strobe=strobe, wphase=wphase, wpulse=PULSE, shut=shut, bb=bb,
                   bbpulse=bbpulse, bbhue=bbhue, holo=holo, laser=laser, spire_h=spire_h, spire_glow=spire_glow,
                   drone=drone_lv, sky=sky, blink=blink, time=T)

# CTRL: one shader node group whose Value nodes carry every keyed scalar (all materials read it)
CTRL = bpy.data.node_groups.new("CTRL", "ShaderNodeTree")
_cn = NB(CTRL)
_cgo = _cn.n("NodeGroupOutput")
for k in ctrl_curves:
    CTRL.interface.new_socket(k, in_out="OUTPUT", socket_type="NodeSocketFloat")
for k in ctrl_curves:
    s = _cn.val(k, float(ctrl_curves[k][0]))
    _cn.link(s, _cgo.inputs[k])
for k, arr in ctrl_curves.items():
    bake(CTRL, f'nodes["{k}"].outputs[0].default_value', arr)


def ctrl(nb):
    g = nb.n("ShaderNodeGroup")
    g.node_tree = CTRL
    return g.outputs


# ---------------------------------------------------------------------------------------------------------
# 2. materials
# ---------------------------------------------------------------------------------------------------------
def img(path, color=True):
    im = bpy.data.images.load(path, check_existing=True)
    if not color:
        im.colorspace_settings.name = "Non-Color"
    return im


def facade_material(name="Facade", wstr=2.6):
    """Night facade: dark wet walls, glass, procedural lit windows (keyed density / strobe waves / shutoff),
    LED floor bands on some buildings. Attributes: bid (random per building), soff (shutoff order), kind, led, hue."""
    m, nb, out = new_mat(name)
    cg = ctrl(nb)
    g = nb.geo()
    Pp, Nn = g.outputs["Position"], g.outputs["Normal"]
    px, py, pz = nb.sep(Pp)
    nx, ny, nz = nb.sep(Nn)
    bid = nb.sattr("bid").outputs["Fac"]
    soff = nb.sattr("soff").outputs["Fac"]
    kind = nb.sattr("kind").outputs["Fac"]
    led = nb.sattr("led").outputs["Fac"]
    hue = nb.sattr("hue").outputs["Fac"]
    side = nb.m("SUBTRACT", 1.0, nb.mr(nb.m("ABSOLUTE", nz), 0.3, 0.6))
    tcoord = nb.m("ADD", nb.m("MULTIPLY", px, nb.m("MULTIPLY", ny, -1.0)), nb.m("MULTIPLY", py, nx))
    fh = nb.m("ADD", 3.3, nb.m("MULTIPLY", nb.m("FRACT", nb.m("MULTIPLY", bid, 13.7)), 1.0))      # 3.3–4.3 m floors
    bw = nb.m("ADD", 1.7, nb.m("MULTIPLY", nb.m("FRACT", nb.m("MULTIPLY", bid, 7.3)), 1.9))       # 1.7–3.6 m bays
    fz = nb.m("DIVIDE", pz, fh)
    fl = nb.m("FLOOR", fz)
    fzf = nb.m("FRACT", fz)
    bc = nb.m("DIVIDE", tcoord, bw)
    bay = nb.m("FLOOR", bc)
    bcf = nb.m("FRACT", bc)
    win = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("GREATER_THAN", fzf, 0.22), nb.m("LESS_THAN", fzf, 0.86)),
               nb.m("MULTIPLY", nb.m("GREATER_THAN", bcf, 0.10), nb.m("LESS_THAN", bcf, 0.90)))
    win = nb.m("MULTIPLY", nb.m("MULTIPLY", win, side), nb.m("LESS_THAN", kind, 0.5))
    wn = nb.n("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    nb.set(wn.inputs["Vector"], nb.xyz(fl, bay, nb.m("MULTIPLY", bid, 977.0)))
    cell = wn.outputs["Value"]
    cellc = nb.sep(wn.outputs["Color"])
    dark_bld = nb.m("GREATER_THAN", nb.m("FRACT", nb.m("MULTIPLY", bid, 31.1)), 0.9)               # ~10 % mostly dark
    litfrac = nb.m("MULTIPLY", cg["city"], nb.m("MULTIPLY", nb.m("ADD", 0.10, nb.m("MULTIPLY", bid, 0.28)),
                                                     nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", dark_bld, 0.85))))
    lit = nb.m("LESS_THAN", cell, litfrac)
    on = nb.m("GREATER_THAN", soff, cg["shut"])
    # strobe waves: bands sweep across the city, flashing on every beat
    wv = nb.m("FRACT", nb.m("SUBTRACT", nb.m("ADD", nb.m("MULTIPLY", px, 0.0031), nb.m("MULTIPLY", py, 0.0019)),
                            cg["wphase"]))
    band = nb.m("SUBTRACT", 1.0, nb.mr(wv, 0.0, 0.20))
    strb = nb.m("MULTIPLY", nb.m("MULTIPLY", band, cg["wpulse"]), nb.m("MULTIPLY", cg["strobe"], 3.2))
    wcol = nb.mix(nb.m("GREATER_THAN", cellc[0], 0.58), (1.0, 0.60, 0.30), (0.72, 0.84, 1.0))
    wcol = nb.mix(nb.m("GREATER_THAN", cellc[2], 0.93), wcol, (0.25, 0.9, 1.0))
    blinds = nb.mix(nb.m("GREATER_THAN", cellc[2], 0.55), 1.0,
                    nb.m("ADD", 0.45, nb.m("MULTIPLY", 0.55, nb.m("GREATER_THAN", nb.m("FRACT", nb.m("MULTIPLY", fzf, 14.0)), 0.45))), dtype="FLOAT")
    grad = nb.mr(fzf, 0.22, 0.86, 1.15, 0.55)
    wi = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", lit, nb.m("ADD", 0.15, nb.m("POWER", cellc[1], 1.5))),
                                         nb.m("MULTIPLY", blinds, grad)), cg["cityB"])
    e_win = nb.mix(nb.m("DIVIDE", strb, nb.m("ADD", strb, nb.m("ADD", wi, 0.001))), wcol, (0.55, 0.85, 1.0))
    wtot = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("ADD", wi, strb), win), on)
    # LED floor bands (cyberpunk trim) on some buildings
    ledband = nb.m("LESS_THAN", nb.m("FRACT", nb.m("DIVIDE", pz, 21.6)), 0.025)
    ledamt = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", ledband, led), side),
                  nb.m("MULTIPLY", on, nb.m("POWER", cg["city"], 2.0)))
    hsv = nb.n("ShaderNodeHueSaturation")
    nb.set(hsv.inputs["Color"], (1.0, 0.1, 0.1))
    nb.set(hsv.inputs["Hue"], hue)
    ecol = nb.mix(nb.m("DIVIDE", ledamt, nb.m("ADD", ledamt, nb.m("ADD", wtot, 0.0001))), e_win, hsv.outputs[0])
    estr = nb.m("MULTIPLY", nb.m("ADD", wtot, nb.m("MULTIPLY", ledamt, 3.0)), wstr)
    # surface: wet concrete/metal walls, dark glass in windows, wet roofs
    walln = nb.n("ShaderNodeTexNoise")
    nb.set(walln.inputs["Scale"], 0.35)
    nb.set(walln.inputs["Detail"], 6.0)
    wallc = nb.mix(walln.outputs["Fac"], (0.028, 0.028, 0.032), (0.07, 0.068, 0.07))
    base = nb.mix(win, wallc, (0.008, 0.010, 0.013))
    rough = nb.mix(win, nb.mr(walln.outputs["Fac"], 0.3, 0.7, 0.25, 0.6), 0.06, dtype="FLOAT")
    b = principled(nb, base, rough, 0.0)
    nb.set(b.inputs["Emission Color"], ecol)
    nb.set(b.inputs["Emission Strength"], estr)
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def set_face_attrs(me, **vals):
    for k, v in vals.items():
        a = me.attributes.get(k) or me.attributes.new(k, "FLOAT", "FACE")
        a.data.foreach_set("value", np.full(len(me.polygons), v, np.float32))


M_FACADE = facade_material()


def wet_ground_material():
    m, nb, out = new_mat("WetAsphalt")
    tex = os.path.join(A.assets, "polyhaven", "tex_asphalt_02")
    files = {f.split("_")[-2] if "nor" not in f else "nor": f for f in os.listdir(tex) if f.endswith(".jpg")}
    tc = nb.n("ShaderNodeTexCoord")
    mp = nb.n("ShaderNodeMapping")
    nb.set(mp.inputs["Scale"], (1 / 6.0, 1 / 6.0, 1.0))
    nb.link(tc.outputs["Object"], mp.inputs["Vector"])
    diff = nb.n("ShaderNodeTexImage", image=img(os.path.join(tex, next(v for v in os.listdir(tex) if "diff" in v))))
    nb.link(mp.outputs[0], diff.inputs["Vector"])
    pud = nb.n("ShaderNodeTexNoise")
    nb.set(pud.inputs["Scale"], 0.02)
    nb.set(pud.inputs["Detail"], 4.0)
    pm = nb.mr(pud.outputs["Fac"], 0.48, 0.56)
    base = nb.mix(pm, nb.mix(1.0, diff.outputs["Color"], (0.35, 0.35, 0.35), blend="MULTIPLY"), (0.012, 0.012, 0.014))
    b = principled(nb, base, nb.mix(pm, 0.32, 0.03, dtype="FLOAT"), 0.0)
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def roof_material():
    """Rain-wet concrete roof: scanned CC0 concrete + puddle mask (near-mirror) + raindrop ripples."""
    m, nb, out = new_mat("WetRoof")
    cg = ctrl(nb)
    d = os.path.join(BMO, "ph_tex_concrete_floor_worn_001")
    f = {k: os.path.join(d, x) for x in os.listdir(d) for k in ("diff", "rough", "nor_gl") if k in x and x.endswith("jpg")}
    tc = nb.n("ShaderNodeTexCoord")
    mp = nb.n("ShaderNodeMapping")
    nb.set(mp.inputs["Scale"], (1 / 3.0, 1 / 3.0, 1 / 3.0))
    nb.link(tc.outputs["Object"], mp.inputs["Vector"])
    tex = {}
    for k, path in f.items():
        t = nb.n("ShaderNodeTexImage", image=img(path, color=(k == "diff")))
        t.projection = "BOX"
        t.projection_blend = 0.2
        nb.link(mp.outputs[0], t.inputs["Vector"])
        tex[k] = t
    pud = nb.n("ShaderNodeTexNoise")
    nb.set(pud.inputs["Scale"], 0.09)
    nb.set(pud.inputs["Detail"], 6.0)
    nb.set(pud.inputs["Roughness"], 0.6)
    nb.link(tc.outputs["Object"], pud.inputs["Vector"])
    pm = nb.mr(pud.outputs["Fac"], 0.47, 0.515)
    base = nb.mix(1.0, tex["diff"].outputs["Color"], (0.42, 0.42, 0.44), blend="MULTIPLY")
    base = nb.mix(pm, base, (0.02, 0.02, 0.022))
    rough = nb.mix(pm, nb.m("MULTIPLY", tex["rough"].outputs["Color"], 0.32), 0.012, dtype="FLOAT")
    # ripples: animated 4D noise, only in puddles
    rip = nb.n("ShaderNodeTexNoise", noise_dimensions="4D")
    nb.set(rip.inputs["Scale"], 9.0)
    nb.set(rip.inputs["Detail"], 2.0)
    nb.link(tc.outputs["Object"], rip.inputs["Vector"])
    nb.link(nb.m("MULTIPLY", cg["time"], 2.2), rip.inputs["W"])
    nm = nb.n("ShaderNodeNormalMap")
    nb.link(tex["nor_gl"].outputs["Color"], nm.inputs["Color"])
    nb.set(nm.inputs["Strength"], nb.mix(pm, 0.8, 0.0, dtype="FLOAT"))
    bump = nb.n("ShaderNodeBump")
    nb.set(bump.inputs["Strength"], nb.m("MULTIPLY", pm, 0.05))
    nb.set(bump.inputs["Distance"], 0.002)
    nb.link(rip.outputs["Fac"], bump.inputs["Height"])
    nb.link(nm.outputs[0], bump.inputs["Normal"])
    b = principled(nb, base, rough, 0.0)
    nb.link(bump.outputs[0], b.inputs["Normal"])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


M_GROUND = wet_ground_material()
M_ROOF = roof_material()
M_CONC = simple_mat("WetConcrete", (0.09, 0.088, 0.085), 0.35)
M_METAL = simple_mat("DarkMetal", (0.06, 0.062, 0.065), 0.3, 0.9)
M_STEEL = simple_mat("Steel", (0.35, 0.36, 0.38), 0.25, 1.0)


def beacon_material(name="Beacon", color=(1.0, 0.04, 0.02), strength=30.0):
    m, nb, out = new_mat(name)
    cg = ctrl(nb)
    e = nb.n("ShaderNodeEmission")
    nb.set(e.inputs["Color"], color)
    nb.link(nb.m("MULTIPLY", nb.m("ADD", 0.04, cg["blink"]), strength), e.inputs["Strength"])
    nb.link(e.outputs[0], out.inputs["Surface"])
    return m


M_BEACON = beacon_material()

# ---------------------------------------------------------------------------------------------------------
# 3. layout constants
# ---------------------------------------------------------------------------------------------------------
HOME = (-22.0, -17.0, 22.0, 17.0, 72.0)                       # x0 y0 x1 y1 roof-z
ROOFZ = HOME[4]
W1 = (-58.0, 2.0, -30.0, 46.0, 98.0)                          # west neighbour (signs on its east face)
N1 = (30.0, 8.0, 56.0, 48.0, 114.0)                           # east neighbour (signs on its west face)
HAND = [HOME, W1, N1, (-24, -64, 20, -28, 52), (30, -44, 60, -2, 66), (-62, -46, -30, -6, 58),
        (-26, 30, 22, 58, 44)]
TCEN = Vector((80.0, 520.0, 0.0))                              # tallest tower
TDECK = 300.0
DS_C = Vector((30.0, 440.0, 262.0))                            # drone show centre
HOLO_C = Vector((40.0, 212.0, 0.0))                            # holo billboard building (roof 80)
FLY = [Vector(p) for p in [(-205, 196, 96), (-122, 266, 104), (-55, 330, 118), (-12, 378, 148), (18, 428, 212),
                            (44, 474, 284), (63, 503, 321)]]
HEROES = [  # (model, x, y, height m, yaw deg)
    ("sf_skyscraper_a", -178.0, 300.0, 205.0, 20.0),
    ("sf_skyscraper_b", -40.0, 236.0, 200.0, -15.0),
    ("sf_skyscraper_c", -122.0, 392.0, 250.0, 35.0),
    ("sf_skyscraper_a", 58.0, 360.0, 228.0, 200.0),
    ("sf_skyscraper_b", -58.0, 470.0, 240.0, 160.0),
    ("sf_skyscraper_c", 140.0, 430.0, 215.0, -60.0),
    ("sf_skyscraper_b", 210.0, 610.0, 236.0, 90.0),
    ("sf_skyscraper_a", -20.0, 640.0, 220.0, 120.0),
]


def seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
    return (a + ab * t - p).length, a + ab * t


# ---------------------------------------------------------------------------------------------------------
# 4. procedural city
# ---------------------------------------------------------------------------------------------------------
def build_city():
    boxes, bids, soffs, kinds, leds, hues = [], [], [], [], [], []
    tops = []        # (x, y, z, w, d) building tops for beacons / billboards
    PITCH, HALF = 90.0, 36.0
    bid_counter = 0

    def blocked(x0, y0, x1, y1):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if -75 < cx < 75 and -75 < cy < 70:
            return True
        if (Vector((cx, cy, 0)) - TCEN).length < 70:
            return True
        if (Vector((cx, cy, 0)) - HOLO_C).length < 40:
            return True
        for _, hx, hy, _, _ in HEROES:
            if math.hypot(cx - hx, cy - hy) < 62:
                return True
        if (Vector((cx, cy, 0)) - Vector((DS_C.x, DS_C.y, 0))).length < 45:
            return True
        return False

    for gx in range(-16, 17):
        for gy in range(-6, 19):
            cx, cy = gx * PITCH, gy * PITCH
            d_down = math.hypot(cx - 40, cy - 560)
            if d_down > 1500:
                continue
            split = rng.choice([1, 2, 2, 4, 4, 4])
            if split == 1:
                lots = [(cx - HALF, cy - HALF, cx + HALF, cy + HALF)]
            elif split == 2:
                if rng.random() < 0.5:
                    lots = [(cx - HALF, cy - HALF, cx - 3, cy + HALF), (cx + 3, cy - HALF, cx + HALF, cy + HALF)]
                else:
                    lots = [(cx - HALF, cy - HALF, cx + HALF, cy - 3), (cx - HALF, cy + 3, cx + HALF, cy + HALF)]
            else:
                lots = [(cx - HALF, cy - HALF, cx - 3, cy - 3), (cx + 3, cy - HALF, cx + HALF, cy - 3),
                        (cx - HALF, cy + 3, cx - 3, cy + HALF), (cx + 3, cy + 3, cx + HALF, cy + HALF)]
            for (x0, y0, x1, y1) in lots:
                s = rng.uniform(0.5, 4.0)
                x0, y0, x1, y1 = x0 + s, y0 + rng.uniform(0.5, 4), x1 - rng.uniform(0.5, 4), y1 - rng.uniform(0.5, 4)
                if blocked(x0, y0, x1, y1):
                    continue
                lx, ly = (x0 + x1) / 2, (y0 + y1) / 2
                dd = math.hypot(lx - 40, ly - 560)
                u = rng.random()
                h = 16 + 38 * u * u + 205 * math.exp(-(dd / 400) ** 2) * (0.25 + 0.75 * rng.random() ** 1.3)
                if rng.random() < 0.14:
                    h = max(h, rng.uniform(65, 145) * (1.0 - 0.4 * smoothstep((dd - 700) / 600)))
                h *= 1.0 - 0.5 * smoothstep((dd - 900) / 600)
                # keep the home → skyline sightline open (look down onto roofs in the foreground)
                if abs(lx) < 190 and 40 < ly < 360:
                    h = min(h, 22 + (ly - 40) * 0.14)
                # fly-through corridor: stay well under the camera path
                pc = Vector((lx, ly, 0))
                for a, b in zip(FLY[:-1], FLY[1:]):
                    dist, q = seg_dist(pc, Vector((a.x, a.y, 0)), Vector((b.x, b.y, 0)))
                    if dist < 48:
                        za = a.z + (b.z - a.z) * ((q - Vector((a.x, a.y, 0))).length / max(1e-6, (Vector((b.x, b.y, 0)) - Vector((a.x, a.y, 0))).length))
                        h = min(h, za - 40)
                h = max(h, 10.0)
                bid = rng.random()
                dist_home = math.hypot(lx, ly)
                ledf = 1.0 if (rng.random() < 0.13 and h > 40) else 0.0
                hue = rng.choice([0.0, 0.5, 0.83, 0.55, 0.9, 0.5, 0.62])
                tiers = [(x0, y0, x1, y1, 0.0, h)]
                if h > 75 and rng.random() < 0.75:
                    z1 = h * rng.uniform(0.5, 0.72)
                    k = rng.uniform(0.12, 0.22)
                    w, dpt = x1 - x0, y1 - y0
                    tiers = [(x0, y0, x1, y1, 0.0, z1),
                             (x0 + w * k, y0 + dpt * k, x1 - w * k, y1 - dpt * k, z1, h)]
                    if h > 150 and rng.random() < 0.7:
                        z2 = h * rng.uniform(0.8, 0.9)
                        k2 = k + rng.uniform(0.08, 0.14)
                        tiers[-1] = (x0 + w * k, y0 + dpt * k, x1 - w * k, y1 - dpt * k, z1, z2)
                        tiers.append((x0 + w * k2, y0 + dpt * k2, x1 - w * k2, y1 - dpt * k2, z2, h))
                for (a0, b0, a1, b1, za, zb) in tiers:
                    boxes.append((a0, b0, za, a1, b1, zb))
                    bids.append(bid)
                    soffs.append(dist_home)
                    kinds.append(0.0)
                    leds.append(ledf)
                    hues.append(hue)
                ta0, tb0, ta1, tb1, ztb, zt = tiers[-1]
                tops.append((ta0, tb0, ta1, tb1, zt, ztb, x0, y0, x1, y1))
                if rng.random() < 0.45:     # mechanical penthouse
                    pw, pd = (ta1 - ta0) * rng.uniform(0.25, 0.45), (tb1 - tb0) * rng.uniform(0.25, 0.45)
                    px, py = rng.uniform(ta0 + pw / 2 + 1, ta1 - pw / 2 - 1), rng.uniform(tb0 + pd / 2 + 1, tb1 - pd / 2 - 1)
                    ph = rng.uniform(3.5, 6.5)
                    boxes.append((px - pw / 2, py - pd / 2, zt, px + pw / 2, py + pd / 2, zt + ph))
                    bids.append(bid); soffs.append(dist_home); kinds.append(1.0); leds.append(0.0); hues.append(hue)
                bid_counter += 1
    soffs = np.asarray(soffs)
    # far → small soff → shuts off first. rank-normalised to [0.02, 0.9]; hand-placed + heroes get 0.9+
    ranks = np.argsort(np.argsort(-soffs))
    soff_n = 0.02 + 0.88 * (ranks / max(1, len(ranks) - 1))
    ob = boxes_mesh("City", boxes, dict(bid=bids, soff=soff_n, kind=kinds, led=leds, hue=hues), coll=C_CITY)
    ob.data.materials.append(M_FACADE)
    print(f"[city] {bid_counter} buildings, {len(boxes)} boxes")
    return ob, tops


CITY, TOPS = build_city()

# hand-placed blocks around the home roof (signs hang on W1 / N1)
hb, hattrs = [], dict(bid=[], soff=[], kind=[], led=[], hue=[])
for i, (x0, y0, x1, y1, z) in enumerate(HAND):
    hb.append((x0, y0, 0.0, x1, y1, z))
    for k, v in dict(bid=rng.random(), soff=0.92 + 0.01 * i, kind=0.0, led=0.0, hue=0.0).items():
        hattrs[k].append(v)
HANDOB = boxes_mesh("HandBlocks", hb, hattrs, coll=C_CITY)
HANDOB.data.materials.append(M_FACADE)

# ground + street lights + traffic
gr = mesh_obj("Ground", [(-2200, -1200, 0), (2200, -1200, 0), (2200, 2400, 0), (-2200, 2400, 0)], [(0, 1, 2, 3)], coll=C_CITY)
gr.data.materials.append(M_GROUND)

sl_pts, tr = [], dict(a=[], dv=[], L=[], off=[], spd=[], col=[])
for k in range(-16, 17):
    xs = 45.0 + k * 90.0
    for sgn, lane in ((1, 3.0), (-1, -3.0)):
        L = 2400.0
        n = 26
        for j in range(n):
            tr["a"].append((xs + lane, -700.0, 0.9))
            tr["dv"].append((0.0, float(sgn), 0.0))
            tr["L"].append(L); tr["off"].append(rng.uniform(0, L)); tr["spd"].append(rng.uniform(9, 16))
            tr["col"].append(0.0 if sgn > 0 else 1.0)
    for y in np.arange(-700, 1700, 32.0):
        sl_pts.append((xs + 8.5, y, 8.0))
        sl_pts.append((xs - 8.5, y + 16, 8.0))
for k in range(-8, 20):
    ys = 45.0 + k * 90.0
    for sgn, lane in ((1, 3.0), (-1, -3.0)):
        for j in range(30):
            tr["a"].append((-1500.0, ys + lane, 0.9))
            tr["dv"].append((float(sgn), 0.0, 0.0))
            tr["L"].append(3000.0); tr["off"].append(rng.uniform(0, 3000)); tr["spd"].append(rng.uniform(9, 16))
            tr["col"].append(1.0 if sgn > 0 else 0.0)
    for x in np.arange(-1500, 1500, 32.0):
        sl_pts.append((x, ys + 8.5, 8.0))
# remove lights / traffic inside blocks we hand-built


def traffic_tree():
    ng, nb, gi, go = gn_tree("Traffic")
    T_ = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    a = nb.attr("a", "FLOAT_VECTOR")
    dv = nb.attr("dv", "FLOAT_VECTOR")
    L = nb.attr("L")
    off = nb.attr("off")
    spd = nb.attr("spd")
    s = nb.m("MULTIPLY", nb.m("FRACT", nb.m("DIVIDE", nb.m("ADD", off, nb.m("MULTIPLY", spd, T_)), L)), L)
    pos = nb.v("ADD", a, nb.v("SCALE", dv, s))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(gi.outputs[0], sp.inputs["Geometry"])
    nb.link(pos, sp.inputs["Position"])
    ico = nb.n("GeometryNodeMeshIcoSphere")
    nb.set(ico.inputs["Radius"], 0.75)
    nb.set(ico.inputs["Subdivisions"], 1)
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(sp.outputs[0], inst.inputs["Points"])
    nb.link(ico.outputs["Mesh"], inst.inputs["Instance"])
    rl = nb.n("GeometryNodeRealizeInstances")
    nb.link(inst.outputs[0], rl.inputs["Geometry"])
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(rl.outputs[0], sm.inputs["Geometry"])
    m, mb, out = new_mat("TrafficLights")
    cg = ctrl(mb)
    col = mb.sattr("col").outputs["Fac"]
    e = mb.n("ShaderNodeEmission")
    mb.set(e.inputs["Color"], mb.mix(col, (1.0, 0.85, 0.65), (1.0, 0.05, 0.02)))
    mb.set(e.inputs["Strength"], mb.m("MULTIPLY", mb.m("SUBTRACT", 1.0, cg["shut"]), 9.0))
    mb.link(e.outputs[0], out.inputs["Surface"])
    sm.inputs["Material"].default_value = m
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


TRAF = points_obj("Traffic", np.zeros((len(tr["a"]), 3)), {k: np.asarray(v) for k, v in tr.items()}, coll=C_CITY)
add_gn(TRAF, traffic_tree())


def lamps_tree(name, radius, mat):
    ng, nb, gi, go = gn_tree(name)
    ico = nb.n("GeometryNodeMeshIcoSphere")
    nb.set(ico.inputs["Radius"], radius)
    nb.set(ico.inputs["Subdivisions"], 1)
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(gi.outputs[0], inst.inputs["Points"])
    nb.link(ico.outputs["Mesh"], inst.inputs["Instance"])
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(inst.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = mat
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


def keyed_emit(name, color, strength, key="city", offset=0.25):
    m, nb, out = new_mat(name)
    cg = ctrl(nb)
    e = nb.n("ShaderNodeEmission")
    nb.set(e.inputs["Color"], color)
    lv = nb.m("MULTIPLY", nb.m("SUBTRACT", 1.0, cg["shut"]), nb.m("ADD", offset, nb.m("MULTIPLY", cg[key], 1 - offset)))
    nb.link(nb.m("MULTIPLY", lv, strength), e.inputs["Strength"])
    nb.link(e.outputs[0], out.inputs["Surface"])
    return m


SL = points_obj("StreetLights", np.asarray(sl_pts), coll=C_CITY)
add_gn(SL, lamps_tree("StreetLamps", 0.55, keyed_emit("StreetLamp", (1.0, 0.62, 0.3), 14.0, offset=0.6)))

# aviation beacons on tall roofs
bpts = [((a0 + a1) / 2, (b0 + b1) / 2, z + 1.5) for (a0, b0, a1, b1, z, *_r) in TOPS if z > 110]
BEAC = points_obj("Beacons", np.asarray(bpts), coll=C_CITY)
add_gn(BEAC, lamps_tree("BeaconPts", 0.9, M_BEACON))

# ---------------------------------------------------------------------------------------------------------
# 5. hero towers (CC-BY Artemis26 skyscrapers) with the facade shader
# ---------------------------------------------------------------------------------------------------------
_lib_cache = {}


def import_glb(slug):
    """Import a glb once into a hidden library collection; return its mesh objects."""
    if slug in _lib_cache:
        return _lib_cache[slug]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(BMO, slug, "model.glb"))
    new = [o for o in bpy.data.objects if o not in before]
    lib = bpy.data.collections.get("_lib") or coll("_lib")
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        lib.objects.link(o)
    lib.hide_render = True
    lib.hide_viewport = True
    # bake world transforms into meshes so instances need only one matrix
    meshes = []
    for o in new:
        if o.type == "MESH":
            mw = o.matrix_world.copy()
            me = o.data.copy()
            me.transform(mw)
            meshes.append((o.name, me))
    _lib_cache[slug] = meshes
    return meshes


def model_bbox(meshes):
    lo = np.full(3, 1e9)
    hi = np.full(3, -1e9)
    for _, me in meshes:
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        lo, hi = np.minimum(lo, co.min(0)), np.maximum(hi, co.max(0))
    return lo, hi


def accent_from(mat_src, name):
    """keep a model's emissive accent colour, but key it with the city level / shutoff"""
    col = (1.0, 0.05, 0.02)
    if mat_src and mat_src.node_tree:
        bs = next((n for n in mat_src.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bs:
            col = tuple(bs.inputs["Emission Color"].default_value[:3])
    return keyed_emit(name, col, 6.0, offset=0.04)


HERO_OBS = []
for i, (slug, hx, hy, hh, yaw) in enumerate(HEROES):
    meshes = import_glb(slug)
    lo, hi = model_bbox(meshes)
    s = hh / (hi[2] - lo[2])
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    for nm, me0 in meshes:
        me = me0.copy()
        emis = []
        for si, mt in enumerate(me.materials):
            bs = next((n for n in mt.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mt and mt.node_tree else None
            is_em = bs is not None and bs.inputs["Emission Strength"].default_value > 0.5 and \
                max(bs.inputs["Emission Color"].default_value[:3]) > 0.5 and not bs.inputs["Emission Color"].is_linked
            me.materials[si] = accent_from(mt, f"HeroAccent_{i}_{si}") if is_em else M_FACADE
        set_face_attrs(me, bid=float(rng.random()), soff=0.86 + 0.005 * i, kind=0.0, led=float(i % 2), hue=float(rng.choice([0.5, 0.83])))
        ob = bpy.data.objects.new(f"Hero{i}_{nm}", me)
        C_CITY.objects.link(ob)
        ob.matrix_world = (Matrix.Translation((hx, hy, 0.0)) @ Matrix.Rotation(deg(yaw), 4, "Z") @
                           Matrix.Diagonal((s, s, s, 1.0)) @ Matrix.Translation((-cx, -cy, -lo[2])))
        HERO_OBS.append(ob)
print("[heroes]", len(HERO_OBS), "objects")

# ---------------------------------------------------------------------------------------------------------
# 6. the tallest tower T (procedural) + corner spire with the climbing beam
# ---------------------------------------------------------------------------------------------------------
tx, ty = TCEN.x, TCEN.y
tb = [(tx - 30, ty - 30, 0, tx + 30, ty + 30, 128), (tx - 24, ty - 24, 128, tx + 24, ty + 24, 226),
      (tx - 17, ty - 17, 226, tx + 17, ty + 17, TDECK)]
TOWER = boxes_mesh("TallestTower", tb, dict(bid=[0.61] * 3, soff=[0.995] * 3, kind=[0.0] * 3, led=[1.0] * 3, hue=[0.5] * 3), coll=C_CITY)
TOWER.data.materials.append(M_FACADE)
# deck parapet + helipad-ish roof slab (wet)
par = []
for (a0, b0, a1, b1) in [(tx - 17, ty - 17, tx + 17, ty - 16.6), (tx - 17, ty + 16.6, tx + 17, ty + 17),
                         (tx - 17, ty - 17, tx - 16.6, ty + 17), (tx + 16.6, ty - 17, tx + 17, ty + 17)]:
    par.append((a0, b0, TDECK, a1, b1, TDECK + 1.15))
PAR_T = boxes_mesh("TowerParapet", par, coll=C_CITY)
PAR_T.data.materials.append(M_CONC)
deck = mesh_obj("TowerDeck", [(tx - 16.6, ty - 16.6, TDECK + 0.05), (tx + 16.6, ty - 16.6, TDECK + 0.05),
                              (tx + 16.6, ty + 16.6, TDECK + 0.05), (tx - 16.6, ty + 16.6, TDECK + 0.05)], [(0, 1, 2, 3)], coll=C_CITY)
deck.data.materials.append(M_ROOF)
# vertical LED fins at the crown corners
fins = []
for sx in (-1, 1):
    for sy in (-1, 1):
        fins.append((tx + sx * 17.0 - 0.35, ty + sy * 17.0 - 0.35, 236, tx + sx * 17.0 + 0.35, ty + sy * 17.0 + 0.35, TDECK + 1.2))
        fins.append((tx + sx * 24.0 - 0.35, ty + sy * 24.0 - 0.35, 140, tx + sx * 24.0 + 0.35, ty + sy * 24.0 + 0.35, 226))
FINS = boxes_mesh("TowerFins", fins, coll=C_CITY)
FINS.data.materials.append(keyed_emit("TowerFinLED", (0.2, 0.85, 1.0), 9.0, offset=0.5))

SPIRE_BASE = Vector((tx + 10.5, ty + 10.5, TDECK))
SPIRE_H = 34.0


def spire():
    import bmesh
    me = bpy.data.meshes.new("Spire")
    bm = bmesh.new()
    try:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=1.5, radius2=0.22, depth=SPIRE_H)
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, SPIRE_H / 2))
        bm.to_mesh(me)
    finally:
        bm.free()
    ob = bpy.data.objects.new("Spire", me)
    C_CITY.objects.link(ob)
    ob.location = SPIRE_BASE
    m, nb, out = new_mat("SpireBeam")
    cg = ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    _, _, lz = nb.sep(tc.outputs["Object"])
    u = nb.m("DIVIDE", lz, SPIRE_H)
    filled = nb.m("LESS_THAN", u, cg["spire_h"])
    head = nb.m("SUBTRACT", 1.0, nb.mr(nb.m("ABSOLUTE", nb.m("SUBTRACT", u, cg["spire_h"])), 0.0, 0.06))
    head = nb.m("MULTIPLY", head, nb.m("GREATER_THAN", cg["spire_h"], 0.001))
    lv = nb.m("ADD", nb.m("ADD", nb.m("MULTIPLY", filled, 0.35), nb.m("MULTIPLY", head, 5.0)), nb.m("MULTIPLY", cg["spire_glow"], 1.6))
    b = principled(nb, (0.3, 0.32, 0.35), 0.25, 1.0)
    nb.set(b.inputs["Emission Color"], nb.mix(head, (0.15, 0.6, 1.0), (0.85, 0.95, 1.0)))
    nb.link(nb.m("MULTIPLY", lv, 3.0), b.inputs["Emission Strength"])
    nb.link(b.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    # spire tip beacon
    tip = points_obj("SpireTip", [SPIRE_BASE + Vector((0, 0, SPIRE_H + 0.6))], coll=C_CITY)
    add_gn(tip, lamps_tree("SpireTipPt", 0.5, M_BEACON))
    return ob


SPIRE = spire()
# beam light that climbs with the shader (lights the deck from the spire)
bl = bpy.data.objects.new("SpireLight", bpy.data.lights.new("SpireLight", "POINT"))
bl.data.color = (0.75, 0.9, 1.0)
bl.data.shadow_soft_size = 1.5
C_CITY.objects.link(bl)
bake_vec(bl, "location", np.stack([np.full(NF, SPIRE_BASE.x - 0.5), np.full(NF, SPIRE_BASE.y - 0.5),
                                    SPIRE_BASE.z + 1 + spire_h * (SPIRE_H - 2)], 1))
bake(bl.data, "energy", (spire_h > 0.001) * 900.0 + spire_glow * 2500.0)

# ---------------------------------------------------------------------------------------------------------
# 7. the home rooftop: parapet, wet roof, CC0 props, water tank, bulkhead, mast
# ---------------------------------------------------------------------------------------------------------
x0, y0, x1, y1, rz = HOME
roof = mesh_obj("Roof", [(x0 + 0.3, y0 + 0.3, rz + 0.02), (x1 - 0.3, y0 + 0.3, rz + 0.02), (x1 - 0.3, y1 - 0.3, rz + 0.02),
                         (x0 + 0.3, y1 - 0.3, rz + 0.02)], [(0, 1, 2, 3)], coll=C_ROOF)
roof.data.materials.append(M_ROOF)
par = []
for (a0, b0, a1, b1) in [(x0, y0, x1, y0 + 0.35), (x0, y1 - 0.35, x1, y1), (x0, y0, x0 + 0.35, y1), (x1 - 0.35, y0, x1, y1)]:
    par.append((a0, b0, rz, a1, b1, rz + 1.1))
PAR = boxes_mesh("Parapet", par, coll=C_ROOF)
PAR.data.materials.append(M_CONC)
# coping cap on parapet
cop = [(a0 - 0.05, b0 - 0.05, rz + 1.1, a1 + 0.05, b1 + 0.05, rz + 1.16) for (a0, b0, _, a1, b1, _) in par]
COP = boxes_mesh("Coping", cop, coll=C_ROOF)
COP.data.materials.append(M_METAL)


def place(slug_or_meshes, names, loc, yaw=0.0, scale=1.0, collection=C_ROOF, gltf=None):
    """instance an imported Poly Haven glTF (meshes cached, transforms baked)"""
    if gltf:
        key = gltf
        if key not in _lib_cache:
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=gltf)
            new = [o for o in bpy.data.objects if o not in before]
            lib = bpy.data.collections.get("_lib") or coll("_lib")
            lib.hide_render = True
            lib.hide_viewport = True
            for o in new:
                for c in list(o.users_collection):
                    c.objects.unlink(o)
                lib.objects.link(o)
            _lib_cache[key] = {o.name: o for o in new if o.type == "MESH"}
        src = _lib_cache[key]
    else:
        src = slug_or_meshes
    obs = []
    for nm, o in src.items():
        if names and nm.split(".")[0] not in names:
            continue
        ob = bpy.data.objects.new(f"{nm}_{len(bpy.data.objects)}", o.data)
        collection.objects.link(ob)
        ob.matrix_world = Matrix.Translation(loc) @ Matrix.Rotation(deg(yaw), 4, "Z") @ Matrix.Diagonal((scale,) * 3 + (1,)) @ \
            Matrix.Translation(-o.matrix_world.translation) @ o.matrix_world
        obs.append(ob)
    return obs


def gl(name):
    d = os.path.join(BMO, name)
    return os.path.join(d, next(f for f in os.listdir(d) if f.endswith(".gltf")))


AC = gl("ph_exterior_aircon_unit")
for i, (ax, ay, yaw) in enumerate([(15.5, -9, 180), (17.3, -9, 180), (19.1, -9, 180), (15.5, -5.5, 0), (17.3, -5.5, 0),
                                   (19.1, -5.5, 0), (-19.4, -11, 90), (-19.4, -8.6, 90)]):
    nm = "exterior_aircon_unit_rusted" if i % 3 == 1 else "exterior_aircon_unit"
    place(None, [nm], Vector((ax, ay, rz + 0.62)), yaw, gltf=AC)
# stands under the AC rows
stands = [(14.8, -9.5, rz, 19.9, -8.6, rz + 0.3), (14.8, -6.0, rz, 19.9, -5.1, rz + 0.3)]
STA = boxes_mesh("ACStands", stands, coll=C_ROOF)
STA.data.materials.append(M_STEEL)
# rectangular duct run along the west side (Poly Haven modular duct, repeated along its 0.6 m Y length)
DU = gl("ph_modular_airduct_rectangular_01")
for j in range(16):
    place(None, ["modular_airduct_rectangular_01_single_01"], Vector((-17.5, -6.0 + j * 0.6, rz + 0.55)), 0.0, gltf=DU)
place(None, ["modular_airduct_rectangular_01_corner_01"], Vector((-17.5, -6.0 + 16 * 0.6, rz + 0.55)), 0.0, gltf=DU)
PB = gl("ph_power_box_01")
for (px_, py_, yaw) in [(-4.0, 6.0 - 2.05, 0.0), (-2.6, 6.0 - 2.05, 0.0)]:
    place(None, ["power_box_01_box", "power_box_01_door"], Vector((px_, py_, rz + 1.3)), yaw + 180, gltf=PB)
# bulkhead (stair house)
BH = boxes_mesh("Bulkhead", [(-8.0, 4.0, rz, -1.0, 9.5, rz + 3.3)], dict(bid=[0.3], soff=[0.999], kind=[1.0], led=[0.0], hue=[0.0]), coll=C_ROOF)
BH.data.materials.append(M_CONC)
door = mesh_obj("Door", [(-5.4, 3.98, rz), (-4.2, 3.98, rz), (-4.2, 3.98, rz + 2.2), (-5.4, 3.98, rz + 2.2)], [(0, 1, 2, 3)], coll=C_ROOF)
door.data.materials.append(simple_mat("DoorPaint", (0.06, 0.08, 0.07), 0.45))
lamp = mesh_obj("CageLamp", [(-5.0, 3.9, rz + 2.6), (-4.6, 3.9, rz + 2.6), (-4.6, 3.9, rz + 2.85), (-5.0, 3.9, rz + 2.85)], [(0, 1, 2, 3)], coll=C_ROOF)
lamp.data.materials.append(keyed_emit("CageLampE", (1.0, 0.7, 0.4), 18.0, offset=0.9))
cl = bpy.data.objects.new("CageLight", bpy.data.lights.new("CageLight", "POINT"))
cl.data.energy, cl.data.color, cl.data.shadow_soft_size = 120.0, (1.0, 0.68, 0.38), 0.15
cl.location = (-4.8, 3.6, rz + 2.7)
C_ROOF.objects.link(cl)
bake(cl.data, "energy", 120.0 * (1.0 - smoothstep((T - (tEND + 1.6)) / 0.3)))


def cylinder(name, r, h, loc, mat, segs=24, coll_=C_ROOF):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    try:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r, radius2=r, depth=h)
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, h / 2))
        bm.to_mesh(me)
    finally:
        bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    coll_.objects.link(ob)
    ob.location = loc
    me.materials.append(mat)
    return ob


M_TANK = simple_mat("TankPaint", (0.12, 0.11, 0.1), 0.4)
cylinder("WaterTank", 2.3, 4.0, (-14.5, 11.0, rz + 2.4), M_TANK)
for lx, ly in [(-16.0, 9.5), (-13.0, 9.5), (-16.0, 12.5), (-13.0, 12.5)]:
    cylinder("TankLeg", 0.12, 2.4, (lx, ly, rz), M_STEEL, segs=8)
cylinder("Mast", 0.09, 11.0, (18.5, 13.0, rz), M_STEEL, segs=8)
mt = points_obj("MastTip", [(18.5, 13.0, rz + 11.2)], coll=C_ROOF)
add_gn(mt, lamps_tree("MastTipPt", 0.18, M_BEACON))

# ---------------------------------------------------------------------------------------------------------
# 8. neon signs (CC-BY ydvisual Japanese set) as blade signs on W1/N1 + rooftop lights, flicker-on in the intro
# ---------------------------------------------------------------------------------------------------------
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=os.path.join(BMO, "sf_neon_signs_jp", "model.glb"))
SIGNS = sorted([o for o in bpy.data.objects if o not in before and o.type == "MESH"], key=lambda o: o.name)
_lib = bpy.data.collections.get("_lib") or coll("_lib")
for o in bpy.data.objects:
    if o not in before:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        _lib.objects.link(o)


def sign_image(o):
    mt = o.data.materials[0]
    return next(n.image for n in mt.node_tree.nodes if n.type == "TEX_IMAGE" and n.image)


def mean_color(im):
    px = np.empty(len(im.pixels), np.float32)
    im.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    w = px[:, :3].max(1)
    c = (px[:, :3] * w[:, None]).sum(0) / max(1e-6, w.sum())
    return tuple((c / max(1e-6, c.max())).tolist())


# flicker-on beats in the intro (the build): one sign per bar from the first kick section
FLICK_BEATS = [17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]
FLICK = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]       # per-frame stutter after the trigger, then steady
# (sign index in the set, wall, y, z, flip)
PLACE = [(17, "N1", 22.0, 77.5), (9, "W1", 20.0, 78.5), (12, "N1", 30.0, 82.0), (2, "W1", 28.0, 80.0),
         (15, "N1", 38.0, 84.0), (6, "W1", 36.0, 84.0), (10, "N1", 44.0, 90.0), (4, "W1", 42.0, 88.0),
         (13, "N1", 26.0, 96.0), (3, "W1", 24.0, 91.0), (16, "N1", 34.0, 101.0), (7, "W1", 32.0, 94.0)]
SIGN_OBS = []
for k, (si, wall, sy, sz) in enumerate(PLACE):
    src = SIGNS[si % len(SIGNS)]
    im = sign_image(src)
    s = 0.0075
    me = src.data.copy()
    ob = bpy.data.objects.new(f"Neon{k}", me)
    C_ROOF.objects.link(ob)
    wx = N1[0] if wall == "N1" else W1[2]
    out_dir = -1.0 if wall == "N1" else 1.0               # blade sticks out of the wall towards the gap
    w = src.dimensions.x * s
    ob.matrix_world = Matrix.Translation((wx + out_dir * (w / 2 + 0.4), sy, sz)) @ Matrix.Diagonal((s, s, s, 1.0)) @ \
        Matrix.Translation(-src.matrix_world.translation) @ src.matrix_world
    # material: the sign image as emission, keyed level (flicker-on, steady, shut off in the outro)
    m, nb, out = new_mat(f"NeonMat{k}")
    t = nb.n("ShaderNodeTexImage", image=im)
    lvname = f"neon{k}"
    lv = nb.val(lvname, 0.0)
    b = principled(nb, (0.02, 0.02, 0.02), 0.3)
    nb.link(t.outputs["Color"], b.inputs["Emission Color"])
    nb.link(nb.m("MULTIPLY", lv, 7.0), b.inputs["Emission Strength"])
    nb.link(b.outputs[0], out.inputs["Surface"])
    me.materials.clear()
    me.materials.append(m)
    trig = fr(bt(FLICK_BEATS[k]))
    lvl = np.zeros(NF)
    lvl[trig - 1:] = 1.0
    for j, v in enumerate(FLICK):
        if trig - 1 + j < NF:
            lvl[trig - 1 + j] = v * (0.6 + 0.4 * rng.random())
    if k in (3, 8):     # two tired tubes buzz now and then (not on the beat — they're broken)
        for _ in range(14):
            f0 = int(rng.uniform(trig + 60, NF - 200))
            lvl[f0:f0 + int(rng.integers(1, 4))] = rng.uniform(0.0, 0.3)
    off_t = tEND + 0.6 + 0.12 * k                         # the signs go dark last, one by one
    lvl[T >= off_t] = 0.0
    bake(m.node_tree, f'nodes["{lvname}"].outputs[0].default_value', lvl, interp="CONSTANT")
    # an area light per sign so the neon actually lights the wet roof + walls
    col = mean_color(im)
    import colorsys
    hh_, ss_, vv_ = colorsys.rgb_to_hsv(*col)
    col = colorsys.hsv_to_rgb(hh_, min(1.0, ss_ * 1.6 + 0.25), 1.0)
    L = bpy.data.objects.new(f"NeonLight{k}", bpy.data.lights.new(f"NeonLight{k}", "POINT"))
    L.data.shadow_soft_size = max(0.6, 0.5 * ob.dimensions.z)
    L.data.use_shadow = False
    L.data.color = col
    L.location = ob.matrix_world.translation + Vector((out_dir * 2.2, -1.2, -0.5))     # off the wall: a halo, not a hotspot
    C_ROOF.objects.link(L)
    bake(L.data, "energy", lvl * 260.0)
    SIGN_OBS.append(ob)
print("[signs]", len(SIGN_OBS))

# ---------------------------------------------------------------------------------------------------------
# 9. billboards on downtown facades (hit on the beat in the drops) + the BE MY ONE hologram
# ---------------------------------------------------------------------------------------------------------
def billboard_material(idx):
    m, nb, out = new_mat(f"Billboard{idx}")
    cg = ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    uvx, uvy, _ = nb.sep(tc.outputs["UV"])
    # animated content: two-colour gradient + scrolling bars + a sign image patch; hue jumps every bar
    src = SIGNS[(idx * 5) % len(SIGNS)]
    t = nb.n("ShaderNodeTexImage", image=sign_image(src))
    t.extension = "REPEAT"
    mp = nb.n("ShaderNodeMapping")
    nb.set(mp.inputs["Scale"], (2.0 if idx % 2 else 1.0, 1.0, 1.0))
    nb.link(tc.outputs["UV"], mp.inputs["Vector"])
    nb.link(mp.outputs[0], t.inputs["Vector"])
    bars =nb.m("GREATER_THAN", nb.m("FRACT", nb.m("ADD", nb.m("MULTIPLY", uvy, 6.0), nb.m("MULTIPLY", cg["time"], 0.6 + 0.2 * idx))), 0.5)
    hs = nb.n("ShaderNodeHueSaturation")
    nb.set(hs.inputs["Color"], (0.9, 0.1, 0.6) if idx % 3 else (0.1, 0.7, 1.0))
    nb.link(nb.m("FRACT", nb.m("ADD", cg["bbhue"], 0.13 * idx)), hs.inputs["Hue"])
    grad = nb.mix(uvx, hs.outputs[0], (0.08, 0.02, 0.2))
    content = nb.mix(nb.m("MULTIPLY", bars, 0.35), grad, (1.0, 1.0, 1.0), blend="ADD")
    content = nb.mix(0.6 if idx % 2 == 0 else 0.0, content, t.outputs["Color"])
    lv = nb.m("MULTIPLY", cg["bb"], nb.m("ADD", 0.55, nb.m("MULTIPLY", cg["bbpulse"], 0.9)))
    b = principled(nb, (0.02, 0.02, 0.02), 0.2)
    nb.link(content, b.inputs["Emission Color"])
    nb.link(nb.m("MULTIPLY", lv, 5.0), b.inputs["Emission Strength"])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


cands = [t_ for t_ in TOPS if t_[4] > 60 and 150 < (t_[1] + t_[3]) / 2 < 900 and abs((t_[0] + t_[2]) / 2) < 520]
rng.shuffle(cands)
BBS = []
for (a0, b0, a1, b1, zt, ztb, *_r) in cands:
    if len(BBS) >= 16:
        break
    w = (a1 - a0) * 0.8
    h = min(w * 0.55, (zt - ztb) * 0.6)
    if h < 8.0:
        continue
    zc = rng.uniform(ztb + h / 2 + 2.0, max(ztb + h / 2 + 2.0, zt - h / 2 - 2.0))
    xc = (a0 + a1) / 2
    yb = b0 - 0.25                       # on the top tier's south face (faces the home roof)
    ob = mesh_obj(f"BB{len(BBS)}", [(xc - w / 2, yb, zc - h / 2), (xc + w / 2, yb, zc - h / 2), (xc + w / 2, yb, zc + h / 2),
                                    (xc - w / 2, yb, zc + h / 2)], [(0, 1, 2, 3)], coll=C_CITY)
    ob.data.uv_layers.new(name="UVMap").data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    ob.data.materials.append(billboard_material(len(BBS)))
    BBS.append(ob)
print("[billboards]", len(BBS))


def hologram():
    hc = HOLO_C
    # host building (roof at 80) + projector rail
    hostb = boxes_mesh("HoloHost", [(hc.x - 34, hc.y - 18, 0, hc.x + 34, hc.y + 18, 80)],
                       dict(bid=[0.4], soff=[0.9], kind=[0.0], led=[1.0], hue=[0.83]), coll=C_CITY)
    hostb.data.materials.append(M_FACADE)
    rail = boxes_mesh("HoloRail", [(hc.x - 32, hc.y - 1, 80, hc.x + 32, hc.y + 1, 81.2)], coll=C_CITY)
    rail.data.materials.append(M_METAL)
    # material: cyan/magenta emissive with scanlines + flicker, partly transparent
    m, nb, out = new_mat("Holo")
    m.surface_render_method = "BLENDED"
    cg = ctrl(nb)
    g = nb.geo()
    _, _, pz = nb.sep(g.outputs["Position"])
    scan = nb.mr(nb.m("SINE", nb.m("MULTIPLY", pz, 9.0)), -1, 1, 0.55, 1.0)
    fl = nb.n("ShaderNodeTexNoise")
    nb.set(fl.inputs["Scale"], 1.0)
    nb.link(nb.xyz(nb.m("MULTIPLY", cg["time"], 7.0), 0, 0), fl.inputs["Vector"])
    flick = nb.mr(fl.outputs["Fac"], 0.3, 0.7, 0.75, 1.05)
    colg = nb.mix(nb.mr(pz, 84, 118), (0.15, 0.85, 1.0), (1.0, 0.15, 0.75))
    e = nb.n("ShaderNodeEmission")
    nb.link(colg, e.inputs["Color"])
    nb.link(nb.m("MULTIPLY", nb.m("MULTIPLY", scan, flick), nb.m("MULTIPLY", cg["holo"], 9.0)), e.inputs["Strength"])
    tr_ = nb.n("ShaderNodeBsdfTransparent")
    mx = nb.n("ShaderNodeMixShader")
    nb.link(nb.m("MULTIPLY", cg["holo"], 0.92), mx.inputs[0])
    nb.link(tr_.outputs[0], mx.inputs[1])
    nb.link(e.outputs[0], mx.inputs[2])
    nb.link(mx.outputs[0], out.inputs["Surface"])
    # text
    font = bpy.data.fonts.load(os.path.join(A.assets, "fonts", "b612", "B612-Bold.ttf"))
    cu = bpy.data.curves.new("HoloText", "FONT")
    cu.body = "BE MY ONE"
    cu.font = font
    cu.align_x = "CENTER"
    cu.size = 15.0
    cu.extrude = 0.25
    txt = bpy.data.objects.new("HoloText", cu)
    C_CITY.objects.link(txt)
    txt.location = (hc.x, hc.y, 86.0)
    txt.rotation_euler = (deg(90), 0, 0)               # text faces -Y (towards the home roof)
    cu.materials.append(m)
    # logo plane above the text (alpha from distance-to-white)
    lm, lb, lout = new_mat("HoloLogo")
    lm.surface_render_method = "BLENDED"
    lcg = ctrl(lb)
    t = lb.n("ShaderNodeTexImage", image=img(os.path.join(BMO, "derived", "logo_mask.png")))
    t.extension = "CLIP"
    rgb = t.outputs["Color"]
    sat = lb.n("ShaderNodeSeparateColor", mode="HSV")
    lb.link(rgb, sat.inputs[0])
    col = lb.mix(lb.mr(sat.outputs[1], 0.3, 0.6), (0.2, 0.85, 1.0), rgb)
    e = lb.n("ShaderNodeEmission")
    lb.link(col, e.inputs["Color"])
    lb.link(lb.m("MULTIPLY", lcg["holo"], 10.0), e.inputs["Strength"])
    tr2 = lb.n("ShaderNodeBsdfTransparent")
    mx2 = lb.n("ShaderNodeMixShader")
    lb.link(lb.m("MULTIPLY", t.outputs["Alpha"], lb.m("MINIMUM", lcg["holo"], 1.0)), mx2.inputs[0])
    lb.link(tr2.outputs[0], mx2.inputs[1])
    lb.link(e.outputs[0], mx2.inputs[2])
    lb.link(mx2.outputs[0], lout.inputs["Surface"])
    s = 13.0
    logo = mesh_obj("HoloLogo", [(-s, 0, -s), (s, 0, -s), (s, 0, s), (-s, 0, s)], [(0, 1, 2, 3)], coll=C_CITY)
    uv = logo.data.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    t_uv = lb.n("ShaderNodeTexCoord")
    lb.link(t_uv.outputs["UV"], t.inputs["Vector"])
    logo.location = (hc.x, hc.y - 0.2, 117.0)
    logo.data.materials.append(lm)
    # holo light spill onto the host roof
    hl = bpy.data.objects.new("HoloLight", bpy.data.lights.new("HoloLight", "AREA"))
    hl.data.shape = "RECTANGLE"
    hl.data.size, hl.data.size_y = 60, 30
    hl.data.color = (0.5, 0.6, 1.0)
    hl.location = (hc.x, hc.y - 3, 100)
    hl.rotation_euler = (deg(90), 0, 0)
    C_CITY.objects.link(hl)
    bake(hl.data, "energy", holo * 2.0e5)
    return txt


HOLO = hologram()

# ---------------------------------------------------------------------------------------------------------
# 10. lasers from rooftops (on in the drops, intensity rides the kick; sweeps are slow + smooth)
# ---------------------------------------------------------------------------------------------------------
LASER_COLS = [(0.1, 1.0, 0.25), (0.15, 0.6, 1.0), (1.0, 0.1, 0.8), (0.1, 1.0, 0.25), (0.6, 0.2, 1.0), (0.15, 0.6, 1.0)]
lroots = [t_ for t_ in TOPS if t_[4] > 45 and 120 < (t_[1] + t_[3]) / 2 < 700 and abs((t_[0] + t_[2]) / 2) < 400]
rng.shuffle(lroots)
for i, (a0, b0, a1, b1, zt, *_r) in enumerate(lroots[:6]):
    em = bpy.data.objects.new(f"LaserRig{i}", None)
    C_FX.objects.link(em)
    em.location = ((a0 + a1) / 2, (b0 + b1) / 2, zt + 1.0)
    lm, lb, lo_ = new_mat(f"Laser{i}")
    lm.surface_render_method = "BLENDED"
    lcg = ctrl(lb)
    le = lb.n("ShaderNodeEmission")
    lb.set(le.inputs["Color"], LASER_COLS[i])
    lb.link(lb.m("MULTIPLY", lcg["laser"], 60.0), le.inputs["Strength"])
    ltr = lb.n("ShaderNodeBsdfTransparent")
    lmx = lb.n("ShaderNodeMixShader")
    lb.link(lb.m("MULTIPLY", lcg["laser"], 6.0, clamp=True), lmx.inputs[0])     # invisible (not black) when off
    lb.link(ltr.outputs[0], lmx.inputs[1])
    lb.link(le.outputs[0], lmx.inputs[2])
    lb.link(lmx.outputs[0], lo_.inputs["Surface"])
    for j in range(5):
        beam = cylinder(f"Laser{i}_{j}", 0.22, 1400.0, (0, 0, 0), lm, segs=6, coll_=C_FX)
        beam.parent = em
        beam.rotation_euler = (deg(-28 + 14 * j), 0, 0)
    ph = rng.uniform(0, 6.28)
    rx = np.radians(18) * np.sin(T * 0.41 + ph)
    rz_ = np.radians(rng.uniform(0, 360)) + np.radians(55) * np.sin(T * 0.23 + 2 * ph)
    bake_vec(em, "rotation_euler", np.stack([rx, np.zeros(NF), rz_], 1))

# ---------------------------------------------------------------------------------------------------------
# 11. rain (GN, analytic, wraps around the camera)
# ---------------------------------------------------------------------------------------------------------
CAM = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
C_CAM.objects.link(CAM)
SC.camera = CAM


def rain_tree():
    ng, nb, gi, go = gn_tree("Rain")
    T_ = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    oi = nb.n("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    oi.inputs["Object"].default_value = CAM
    cam = oi.outputs["Location"]
    S = (34.0, 34.0, 22.0)
    V = (1.1, 0.5, -9.5)
    p0 = nb.n("GeometryNodeInputPosition").outputs[0]
    q = nb.v("ADD", p0, nb.v("DIVIDE", nb.v("SUBTRACT", nb.v("SCALE", V, T_), cam), S))
    q = nb.v("SUBTRACT", nb.v("FRACTION", nb.v("ADD", q, (0.5, 0.5, 0.5))), (0.5, 0.5, 0.5))
    pos = nb.v("ADD", cam, nb.v("MULTIPLY", q, S))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(gi.outputs[0], sp.inputs["Geometry"])
    nb.link(pos, sp.inputs["Position"])
    near = nb.n("GeometryNodeDeleteGeometry", domain="POINT")
    nb.link(sp.outputs[0], near.inputs["Geometry"])
    nb.link(nb.cmp("LESS_THAN", nb.v("DISTANCE", nb.n("GeometryNodeInputPosition").outputs[0], cam), 1.6), near.inputs["Selection"])
    sp = near
    cyl = nb.n("GeometryNodeMeshCylinder")
    nb.set(cyl.inputs["Vertices"], 3)
    nb.set(cyl.inputs["Radius"], 1.0)
    nb.set(cyl.inputs["Depth"], 1.0)
    al = nb.n("FunctionNodeAlignRotationToVector", axis="Z")
    nb.set(al.inputs["Vector"], V)
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(sp.outputs[0], inst.inputs["Points"])
    nb.link(cyl.outputs["Mesh"], inst.inputs["Instance"])
    nb.link(al.outputs[0], inst.inputs["Rotation"])
    nb.set(inst.inputs["Scale"], (0.0035, 0.0035, 0.42))
    m, mb, out = new_mat("Rain")
    m.surface_render_method = "BLENDED"
    e = mb.n("ShaderNodeEmission")
    mb.set(e.inputs["Color"], (0.75, 0.8, 0.9))
    mb.set(e.inputs["Strength"], 0.45)
    tr_ = mb.n("ShaderNodeBsdfTransparent")
    gls = mb.n("ShaderNodeBsdfGlossy")
    mb.set(gls.inputs["Roughness"], 0.05)
    add = mb.n("ShaderNodeAddShader")
    mb.link(e.outputs[0], add.inputs[0])
    mb.link(gls.outputs[0], add.inputs[1])
    mx = mb.n("ShaderNodeMixShader")
    mb.set(mx.inputs[0], 0.16)
    mb.link(tr_.outputs[0], mx.inputs[1])
    mb.link(add.outputs[0], mx.inputs[2])
    mb.link(mx.outputs[0], out.inputs["Surface"])
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(inst.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = m
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


RAIN = points_obj("Rain", rng.random((9000, 3)), coll=C_FX)
add_gn(RAIN, rain_tree())

# ---------------------------------------------------------------------------------------------------------
# 12. drones (shape-keyed formations; GN instances lights on the verts)
# ---------------------------------------------------------------------------------------------------------
ND = 480
cam5 = Vector((-300.0, 140.0, 230.0))
dvw = (DS_C - cam5)
dvw.z = 0
dvw.normalize()
RIGHT = dvw.cross(Vector((0, 0, 1)))
UP = Vector((0, 0, 1))


def frame_pts(uv, depth=None):
    uv = np.asarray(uv, dtype=np.float64)
    d = np.zeros(len(uv)) if depth is None else depth
    R, U, D = np.array(RIGHT), np.array(UP), np.array(dvw)
    return np.array(DS_C)[None, :] + uv[:, 0:1] * R[None, :] + uv[:, 1:2] * U[None, :] + d[:, None] * D[None, :]


def fib_sphere(n, r):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(th) * np.sin(phi), r * np.sin(th) * np.sin(phi), r * np.cos(phi)], 1)


def heart(n, s):
    t = np.linspace(0, 2 * np.pi, int(n * 0.62), endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    t2 = np.linspace(0, 2 * np.pi, n - len(t), endpoint=False)
    x2 = 0.62 * 16 * np.sin(t2) ** 3
    y2 = 0.62 * (13 * np.cos(t2) - 5 * np.cos(2 * t2) - 2 * np.cos(3 * t2) - np.cos(4 * t2)) + 1.5
    return np.stack([np.concatenate([x, x2]) * s, np.concatenate([y, y2]) * s], 1)


def text_points(n, text, height):
    font = bpy.data.fonts.load(os.path.join(A.assets, "fonts", "b612", "B612-Bold.ttf"), check_existing=True)
    cu = bpy.data.curves.new("DroneTxt", "FONT")
    cu.body = text
    cu.font = font
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.size = height
    ob = bpy.data.objects.new("DroneTxt", cu)
    SC.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = ob.evaluated_get(dg).to_mesh()
    me.calc_loop_triangles()
    tri = np.array([[v for v in lt.vertices] for lt in me.loop_triangles])
    co = np.array([v.co[:] for v in me.vertices])
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    # dart-throwing with a min spacing for an even, readable fill
    pts = []
    cand = 40 * n
    ti = rng.choice(len(tri), size=cand, p=area / area.sum())
    r1, r2 = rng.random(cand), rng.random(cand)
    s1 = np.sqrt(r1)
    P_ = (1 - s1)[:, None] * a[ti] + (s1 * (1 - r2))[:, None] * b[ti] + (s1 * r2)[:, None] * c[ti]
    dmin = height * 0.075
    for p in P_:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > dmin * dmin for q in pts[-400:]):
            pts.append(p)
            if len(pts) >= n:
                break
    while len(pts) < n:
        pts.append(P_[rng.integers(len(P_))])
    ob.to_mesh_clear()
    bpy.data.objects.remove(ob)
    return np.array(pts)[:, :2]


def order_by_angle(uv):
    return np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))


gx_, gy_ = np.meshgrid(np.arange(24), np.arange(20))
launch = np.stack([(gx_.ravel() - 11.5) * 3.0, (gy_.ravel() - 9.5) * 3.0], 1)[:ND]
launch_w = np.array(DS_C)[None, :] + np.stack([launch[:, 0], launch[:, 1], np.full(ND, 0.0)], 1)
launch_w[:, 2] = 96.0
sph = fib_sphere(ND, 58.0)
sph_w = np.array(DS_C)[None, :] + sph
hrt = heart(ND, 4.4)
hrt_w = frame_pts(hrt)
th = np.linspace(0, 2 * np.pi, ND, endpoint=False)
ring = np.stack([66 * np.cos(th), 66 * np.sin(th)], 1)
ring_w = frame_pts(ring, depth=18 * np.sin(th))
txt = text_points(ND, "BE MY ONE", 30.0)
txt_w = frame_pts(txt)
# consistent assignment: drone i takes the i-th point of each formation sorted by polar angle in the view plane
base_order = order_by_angle(launch - launch.mean(0))


def assign(world, uv=None):
    if uv is None:
        rel = world - np.array(DS_C)[None, :]
        uv = np.stack([rel @ np.array(RIGHT), rel @ np.array(UP)], 1)
    o = order_by_angle(uv - uv.mean(0))
    out = np.empty_like(world)
    out[base_order] = world[o]
    return out


forms = [launch_w, assign(sph_w), assign(hrt_w, hrt), assign(ring_w, ring), assign(txt_w, txt)]
DR = points_obj("Drones", forms[0], {"g": (forms[0][:, 2] * 0 + rng.random(ND))}, coll=C_FX)
DR.shape_key_add(name="Basis")
for k, fw in enumerate(forms[1:], 1):
    sk = DR.shape_key_add(name=f"F{k}")
    sk.data.foreach_set("co", fw.astype(np.float32).ravel())
# weights over time (transitions eased, start on downbeats)
trans = [(B_DROP2, B_DROP2 + 8), (B_DROP2 + 8, B_DROP2 + 12), (B_DROP2 + 16, B_DROP2 + 20), (B_DROP2 + 24, B_DROP2 + 28)]
wts = np.zeros((NF, 4))
for k, (b0, b1) in enumerate(trans):
    up = smootherstep((T - bt(b0)) / (bt(b1) - bt(b0)))
    wts[:, k] = up
    if k > 0:
        wts[:, k - 1] *= 1.0 - up
kb = DR.data.shape_keys
for k in range(4):
    bake(kb, f'key_blocks["F{k + 1}"].value', wts[:, k])
dhue = np.zeros(NF)
dsat = np.full(NF, 0.8)
for k, (b0, _b1) in enumerate(trans):
    dhue[T >= bt(b0)] = [0.52, 0.95, 0.75, 0.12][k]
    if k == 3:
        dsat[T >= bt(b0)] = 0.2
CTRL.interface.new_socket("dhue", in_out="OUTPUT", socket_type="NodeSocketFloat")
CTRL.interface.new_socket("dsat", in_out="OUTPUT", socket_type="NodeSocketFloat")
for nm_, arr in (("dhue", dhue), ("dsat", dsat)):
    s_ = _cn.val(nm_, 0.0)
    _cn.link(s_, _cgo.inputs[nm_])
    bake(CTRL, f'nodes["{nm_}"].outputs[0].default_value', arr, interp="CONSTANT")


def drone_tree():
    ng, nb, gi, go = gn_tree("DroneLights")
    ico = nb.n("GeometryNodeMeshIcoSphere")
    nb.set(ico.inputs["Radius"], 1.6)
    nb.set(ico.inputs["Subdivisions"], 1)
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(gi.outputs[0], inst.inputs["Points"])
    nb.link(ico.outputs["Mesh"], inst.inputs["Instance"])
    rl = nb.n("GeometryNodeRealizeInstances")
    nb.link(inst.outputs[0], rl.inputs["Geometry"])
    m, mb, out = new_mat("DroneLight")
    cg = ctrl(mb)
    g = mb.sattr("g").outputs["Fac"]
    hsv = mb.n("ShaderNodeCombineColor", mode="HSV")
    mb.link(mb.m("FRACT", mb.m("ADD", cg["dhue"], mb.m("MULTIPLY", g, 0.12))), hsv.inputs[0])
    mb.link(cg["dsat"], hsv.inputs[1])
    mb.set(hsv.inputs[2], 1.0)
    e = mb.n("ShaderNodeEmission")
    mb.link(hsv.outputs[0], e.inputs["Color"])
    mb.link(mb.m("MULTIPLY", cg["drone"], 40.0), e.inputs["Strength"])
    mb.link(e.outputs[0], out.inputs["Surface"])
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(rl.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = m
    # hide entirely when the show is off (keeps BVH light)
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


add_gn(DR, drone_tree())
bake(DR, "hide_render", (drone_lv < 0.001).astype(float), interp="CONSTANT")

# ---------------------------------------------------------------------------------------------------------
# 13. fireworks — analytic (no sim): rockets (launch → apex) and burst sparks with drag + gravity
#     bridge (86.5 → 100.8): tracer comets climb and fade WITHOUT bursting; a final volley is timed so
#     every shell reaches apex at the LAST DROP (100.774) → all burst at once. Nothing before the bridge.
# ---------------------------------------------------------------------------------------------------------
PAL = [(1.0, 0.12, 0.6), (0.15, 0.75, 1.0), (1.0, 0.62, 0.18), (1.0, 0.92, 0.8), (1.0, 0.15, 0.06), (0.25, 1.0, 0.35),
       (0.62, 0.25, 1.0)]
rockets = dict(L=[], A=[], tl=[], ta=[], dud=[], col=[])
sparks = dict(c=[], v=[], t0=[], life=[], col=[], k=[], g=[], tr=[], tw=[], sz=[])


def burst_center_home():
    # dip + finale are seen from the home roof: shells over the near skyline, big in frame
    return Vector((rng.uniform(-260, 320), rng.uniform(230, 560), rng.uniform(170, 330)))


def burst_center():
    # ring around the tallest tower, biased NE (the bridge camera's view) but spread so the home roof sees them
    for _ in range(100):
        ang = rng.uniform(-0.35, 1.9)
        r = rng.uniform(170, 640)
        c = Vector((TCEN.x + r * math.cos(ang), TCEN.y + r * math.sin(ang), rng.uniform(320, 430)))
        if -700 < c.x < 900 and 250 < c.y < 1300:
            return c
    return Vector((TCEN.x + 200, TCEN.y + 200, 400))


def add_rocket(apex, ta, dud=False, flight=None):
    flight = flight or rng.uniform(2.4, 3.6)
    L = Vector((apex.x + rng.uniform(-25, 25), apex.y + rng.uniform(-25, 25), rng.uniform(20, 90)))
    rockets["L"].append(L[:]); rockets["A"].append(apex[:]); rockets["tl"].append(ta - flight); rockets["ta"].append(ta)
    rockets["dud"].append(1.0 if dud else 0.0); rockets["col"].append((1.0, 0.7, 0.35))


def add_burst(c, t0, kind=None, col=None):
    kind = kind or rng.choice(["peony", "peony", "chrys", "willow", "ring", "palm", "double"])
    col = col or PAL[rng.integers(len(PAL))]
    def push(d, spd, life, k=1.4, g=6.0, tr=0.14, tw=0.0, sz=0.32, cc=None):
        sparks["c"].append(c[:]); sparks["v"].append(tuple(d * spd)); sparks["t0"].append(t0); sparks["life"].append(life)
        sparks["col"].append(cc or col); sparks["k"].append(k); sparks["g"].append(g); sparks["tr"].append(tr)
        sparks["tw"].append(tw); sparks["sz"].append(sz)
    if kind in ("peony", "chrys", "double"):
        n = 170 if kind != "chrys" else 210
        for d in fib_sphere(n, 1.0):
            d = d + rng.normal(0, 0.04, 3)
            spd = rng.uniform(40, 48)
            if kind == "chrys":
                push(d, spd, rng.uniform(2.4, 3.0), k=1.1, tr=0.32, tw=1.0 if rng.random() < 0.5 else 0.0, sz=0.28)
            else:
                push(d, spd, rng.uniform(1.8, 2.3), tr=0.16)
        if kind == "double":
            c2 = PAL[rng.integers(len(PAL))]
            for d in fib_sphere(90, 1.0):
                push(d, rng.uniform(20, 24), 1.8, tr=0.12, cc=c2)
    elif kind == "willow":
        for d in fib_sphere(190, 1.0):
            push(d + rng.normal(0, 0.05, 3), rng.uniform(26, 32), rng.uniform(3.8, 4.8), k=0.75, g=9.0, tr=0.55, tw=1.0,
                 sz=0.24, cc=(1.0, 0.6, 0.18))
    elif kind == "ring":
        ax = Vector(rng.normal(0, 1, 3)).normalized()
        u = ax.orthogonal().normalized()
        w = ax.cross(u)
        for a in np.linspace(0, 2 * np.pi, 110, endpoint=False):
            d = math.cos(a) * u + math.sin(a) * w
            push(np.array(d[:]), 44.0, 2.1, tr=0.16)
    elif kind == "palm":
        for _ in range(9):
            arm = Vector(rng.normal(0, 1, 3)).normalized()
            arm.z = abs(arm.z) * 0.6 + 0.2
            arm.normalize()
            for _j in range(10):
                d = np.array(arm[:]) + rng.normal(0, 0.05, 3)
                push(d, rng.uniform(42, 52), 2.6, k=1.0, tr=0.45, sz=0.45, cc=(1.0, 0.75, 0.35))
        for d in fib_sphere(60, 1.0):
            push(d, 30, 1.6, tr=0.1, cc=col)


# bridge tracers (duds): density builds over the 8 bars
for b in range(B_BRIDGE + 2, B_LAST - 8):
    n = 1 if b < B_BRIDGE + 14 else (2 if b < B_BRIDGE + 24 else 3)
    if b < B_BRIDGE + 14 and b % 2:
        continue
    for _ in range(n):
        c = burst_center()
        add_rocket(c, bt(b) + rng.uniform(2.6, 3.4), dud=True, flight=rng.uniform(2.6, 3.4))
# the volley: all apex at the LAST DROP
tLAST = bt(B_LAST)
for _ in range(38):
    c = burst_center()
    add_rocket(c, tLAST, flight=rng.uniform(2.2, 3.4))
    add_burst(c, tLAST)
# last drop: 2–4 per beat · dip: one willow per bar · final burst: dense · finale on the last downbeat
for b in range(B_LAST + 1, B_DIP):
    for _ in range(int(rng.integers(2, 5))):
        c = burst_center()
        t0 = bt(b) + rng.choice([0.0, 0.0, P / 2])
        add_rocket(c, t0)
        add_burst(c, t0)
for b in range(B_DIP, B_FINAL, 4):
    c = burst_center_home()
    add_rocket(c, bt(b))
    add_burst(c, bt(b), "willow")
for b in range(B_FINAL, B_END - 4):
    for _ in range(int(rng.integers(3, 6))):
        c = burst_center_home()
        t0 = bt(b) + rng.choice([0.0, 0.0, P / 2])
        add_rocket(c, t0)
        add_burst(c, t0)
for _ in range(14):           # finale canopy: gold willows on the last downbeat of the music
    c = burst_center_home()
    add_rocket(c, bt(B_END - 4))
    add_burst(c, bt(B_END - 4), "willow")
print(f"[fireworks] rockets {len(rockets['tl'])}, sparks {len(sparks['t0'])}, earliest launch {min(rockets['tl']):.2f}s")
assert min(rockets["tl"]) > bt(B_BRIDGE), "fireworks must not appear before the bridge"


def streak_instancer(nb, pts_geo, vel, length, radius, store):
    """Instance a tail-cylinder (head at the point, tail along -vel), realize, keep per-point attributes."""
    cyl = nb.n("GeometryNodeMeshCylinder")
    nb.set(cyl.inputs["Vertices"], 4)
    nb.set(cyl.inputs["Radius"], 1.0)
    nb.set(cyl.inputs["Depth"], 1.0)
    tf = nb.n("GeometryNodeTransform")
    nb.link(cyl.outputs["Mesh"], tf.inputs["Geometry"])
    nb.set(tf.inputs["Translation"], (0, 0, -0.5))
    _, _, tz = nb.sep(nb.n("GeometryNodeInputPosition").outputs[0])
    tmpl = nb.store(tf.outputs["Geometry"], "tz", tz)
    for nm, val in store.items():
        pts_geo = nb.store(pts_geo, nm, val, dtype="FLOAT_VECTOR" if nm == "fc" else "FLOAT")
    al = nb.n("FunctionNodeAlignRotationToVector", axis="Z")
    nb.link(vel, al.inputs["Vector"])
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(pts_geo, inst.inputs["Points"])
    nb.link(tmpl, inst.inputs["Instance"])
    nb.link(al.outputs[0], inst.inputs["Rotation"])
    nb.link(nb.xyz(radius, radius, length), inst.inputs["Scale"])
    rl = nb.n("GeometryNodeRealizeInstances")
    nb.link(inst.outputs[0], rl.inputs["Geometry"])
    return rl.outputs[0]


def fw_material():
    m, nb, out = new_mat("Firework")
    fc = nb.sattr("fc").outputs["Vector"]
    fb = nb.sattr("fb").outputs["Fac"]
    tz = nb.sattr("tz").outputs["Fac"]
    tail = nb.m("POWER", nb.m("ADD", tz, 1.0), 2.2)
    e = nb.n("ShaderNodeEmission")
    nb.link(fc, e.inputs["Color"])
    nb.link(nb.m("MULTIPLY", nb.m("MULTIPLY", fb, tail), 70.0), e.inputs["Strength"])
    nb.link(e.outputs[0], out.inputs["Surface"])
    return m


M_FW = fw_material()


def sparks_tree():
    ng, nb, gi, go = gn_tree("FireworkSparks")
    T_ = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    c, v = nb.attr("c", "FLOAT_VECTOR"), nb.attr("v", "FLOAT_VECTOR")
    t0, life, k, g = nb.attr("t0"), nb.attr("life"), nb.attr("k"), nb.attr("g")
    trl, tw, sz, col = nb.attr("tr"), nb.attr("tw"), nb.attr("sz"), nb.attr("col", "FLOAT_VECTOR")
    age = nb.m("SUBTRACT", T_, t0)
    vis = nb.band(nb.cmp("GREATER_THAN", age, 0.0), nb.cmp("LESS_THAN", age, life))
    dl = nb.n("GeometryNodeDeleteGeometry", domain="POINT")
    nb.link(gi.outputs[0], dl.inputs["Geometry"])
    nb.link(nb.band(vis, None, op="NOT"), dl.inputs["Selection"])
    e = nb.m("EXPONENT", nb.m("MULTIPLY", nb.m("MULTIPLY", k, age), -1.0))
    one_e = nb.m("SUBTRACT", 1.0, e)
    disp = nb.v("SCALE", v, nb.m("DIVIDE", one_e, k))
    sag = nb.m("MULTIPLY", nb.m("DIVIDE", g, k), nb.m("SUBTRACT", age, nb.m("DIVIDE", one_e, k)))
    pos = nb.v("SUBTRACT", nb.v("ADD", c, disp), nb.xyz(0, 0, sag))
    vel = nb.v("SUBTRACT", nb.v("SCALE", v, e), nb.xyz(0, 0, nb.m("MULTIPLY", nb.m("DIVIDE", g, k), one_e)))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(dl.outputs[0], sp.inputs["Geometry"])
    nb.link(pos, sp.inputs["Position"])
    speed = nb.v("LENGTH", vel)
    length = nb.m("MAXIMUM", nb.m("MULTIPLY", speed, trl), 0.5)
    fade = nb.m("POWER", nb.m("SUBTRACT", 1.0, nb.m("DIVIDE", age, life)), 1.6)
    flash = nb.m("ADD", 1.0, nb.m("MULTIPLY", 2.5, nb.m("EXPONENT", nb.m("MULTIPLY", age, -14.0))))
    wn = nb.n("ShaderNodeTexWhiteNoise", noise_dimensions="2D")
    idx = nb.n("GeometryNodeInputIndex").outputs[0]
    nb.link(nb.xyz(idx, nb.m("FLOOR", nb.m("MULTIPLY", T_, 22.0)), 0), wn.inputs["Vector"])
    twk = nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", tw, nb.m("GREATER_THAN", wn.outputs["Value"], 0.55)))
    fb = nb.m("MULTIPLY", nb.m("MULTIPLY", fade, flash), twk)
    geo = streak_instancer(nb, sp.outputs[0], vel, length, sz, {"fc": col, "fb": fb})
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(geo, sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = M_FW
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


def rockets_tree():
    ng, nb, gi, go = gn_tree("FireworkRockets")
    T_ = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    L, Ap = nb.attr("L", "FLOAT_VECTOR"), nb.attr("A", "FLOAT_VECTOR")
    tl, ta, dud, col = nb.attr("tl"), nb.attr("ta"), nb.attr("dud"), nb.attr("col", "FLOAT_VECTOR")
    dur = nb.m("SUBTRACT", ta, tl)
    u = nb.m("DIVIDE", nb.m("SUBTRACT", T_, tl), dur)
    vis = nb.band(nb.cmp("GREATER_THAN", u, 0.0), nb.cmp("LESS_THAN", u, 1.0))
    dl = nb.n("GeometryNodeDeleteGeometry", domain="POINT")
    nb.link(gi.outputs[0], dl.inputs["Geometry"])
    nb.link(nb.band(vis, None, op="NOT"), dl.inputs["Selection"])
    om = nb.m("SUBTRACT", 1.0, u)
    ease = nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", om, om))
    d = nb.v("SUBTRACT", Ap, L)
    wob = nb.xyz(nb.m("MULTIPLY", nb.m("SINE", nb.m("MULTIPLY", u, 11.0)), 1.2), nb.m("MULTIPLY", nb.m("COSINE", nb.m("MULTIPLY", u, 9.0)), 1.2), 0)
    pos = nb.v("ADD", nb.v("ADD", L, nb.v("SCALE", d, ease)), nb.v("SCALE", wob, u))
    vel = nb.v("SCALE", d, nb.m("DIVIDE", nb.m("MULTIPLY", om, 2.0), dur))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(dl.outputs[0], sp.inputs["Geometry"])
    nb.link(pos, sp.inputs["Position"])
    length = nb.m("MINIMUM", nb.m("MAXIMUM", nb.m("MULTIPLY", nb.v("LENGTH", vel), 0.42), 3.0), 38.0)
    duds = nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", dud, nb.mr(u, 0.72, 1.0)))
    fb = nb.m("MULTIPLY", duds, nb.mr(u, 0.0, 0.04, 0.3, 1.0))
    geo = streak_instancer(nb, sp.outputs[0], vel, length, 0.42, {"fc": col, "fb": fb})
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(geo, sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = M_FW
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


FWS = points_obj("FireworkSparks", np.zeros((len(sparks["t0"]), 3)), {k: np.asarray(v) for k, v in sparks.items()}, coll=C_FX)
add_gn(FWS, sparks_tree())
FWR = points_obj("FireworkRockets", np.zeros((len(rockets["tl"]), 3)), {k: np.asarray(v) for k, v in rockets.items()}, coll=C_FX)
add_gn(FWR, rockets_tree())
fw_on = (T >= bt(B_BRIDGE)).astype(float)
bake(FWS, "hide_render", (T < tLAST - 0.05).astype(float), interp="CONSTANT")
bake(FWR, "hide_render", 1.0 - fw_on, interp="CONSTANT")
# burst flash lights: a handful of big point lights that follow the bursts (light the clouds, roofs, puddles)
burst_list = sorted(set((round(t0, 3), tuple(np.round(c, 1))) for c, t0 in zip(sparks["c"], sparks["t0"])))
NFL = 8
flash_e = np.zeros((NFL, NF))
flash_loc = np.zeros((NFL, NF, 3))
flash_col = []
for i, (t0, c) in enumerate(burst_list):
    j = i % NFL
    m = (T >= t0) & (T < t0 + 1.6)
    flash_e[j, m] = np.maximum(flash_e[j, m], np.exp(-(T[m] - t0) / 0.35))
    flash_loc[j, T >= t0] = c
for j in range(NFL):
    Lf = bpy.data.objects.new(f"BurstFlash{j}", bpy.data.lights.new(f"BurstFlash{j}", "POINT"))
    Lf.data.color = PAL[j % len(PAL)]
    Lf.data.shadow_soft_size = 45.0
    Lf.data.use_shadow = False
    C_FX.objects.link(Lf)
    first = np.argmax(flash_loc[j, :, 2] > 0)
    flash_loc[j, :first] = flash_loc[j, first]
    bake_vec(Lf, "location", flash_loc[j], interp="CONSTANT")
    bake(Lf.data, "energy", flash_e[j] * 0.7e7)

# ---------------------------------------------------------------------------------------------------------
# 14. flying car (CC-BY "Hover car", Igor.Friedman) — breakdown drift past in the rain
#     model nose is LOCAL -X (hood/emblem/headlights; +X has the spoiler + cyan tail panel) — measured 10-13
# ---------------------------------------------------------------------------------------------------------
car_meshes = import_glb("sf_hover_car")
clo, chi = model_bbox(car_meshes)
CAR_LEN = 5.4
cs = CAR_LEN / (chi[0] - clo[0])
RIG = bpy.data.objects.new("CarRig", None)
C_FX.objects.link(RIG)
MODEL = bpy.data.objects.new("CarModel", None)
C_FX.objects.link(MODEL)
MODEL.parent = RIG
# local -X (nose) → rig +Y : rotate -90° about Z; centre + sit on z=0
MODEL.matrix_parent_inverse = Matrix.Identity(4)
MODEL.matrix_basis = Matrix.Rotation(deg(-90), 4, "Z") @ Matrix.Diagonal((cs, cs, cs, 1.0)) @ \
    Matrix.Translation((-(clo[0] + chi[0]) / 2, -(clo[1] + chi[1]) / 2, -(clo[2] + chi[2]) / 2))
for nm, me in car_meshes:
    o = bpy.data.objects.new(f"Car_{nm}", me)
    C_FX.objects.link(o)
    o.parent = MODEL
    for mt in me.materials:          # the model's baked emissive panels (thrusters, light bars) — make them read at night
        bs = next((n for n in mt.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mt and mt.node_tree else None
        if bs and bs.inputs["Emission Color"].is_linked:
            bs.inputs["Emission Strength"].default_value = 6.0
        if bs:
            bs.inputs["Roughness"].default_value = min(bs.inputs["Roughness"].default_value, 0.25) if not bs.inputs["Roughness"].is_linked else 0.25
fill = bpy.data.objects.new("CarFill", bpy.data.lights.new("CarFill", "AREA"))
fill.data.energy, fill.data.color, fill.data.size = 900.0, (0.75, 0.6, 1.0), 6.0
C_FX.objects.link(fill)
fill.parent = RIG
fill.location = (0.0, -2.0, 5.0)                    # soft top light (the holo glow) so the body shape reads
for side in (-0.55, 0.55):
    hl = bpy.data.objects.new("CarHead", bpy.data.lights.new("CarHead", "SPOT"))
    hl.data.energy, hl.data.color, hl.data.spot_size, hl.data.spot_blend = 2500.0, (0.85, 0.92, 1.0), deg(38), 0.6
    C_FX.objects.link(hl)
    hl.parent = RIG
    hl.location = (side, CAR_LEN / 2 + 0.1, 0.0)
    hl.rotation_euler = (deg(82), 0, 0)                # spot points along -Z; R_x(+82°)·(0,0,-1) = (0,+0.99,-0.14) → forward, slightly down
under = bpy.data.objects.new("CarThrust", bpy.data.lights.new("CarThrust", "POINT"))
under.data.energy, under.data.color, under.data.shadow_soft_size = 1800.0, (0.2, 0.7, 1.0), 1.0
C_FX.objects.link(under)
under.parent = RIG
under.location = (0, 0, -1.0)
tail = bpy.data.objects.new("CarTail", bpy.data.lights.new("CarTail", "POINT"))
tail.data.energy, tail.data.color = 400.0, (1.0, 0.05, 0.03)
C_FX.objects.link(tail)
tail.parent = RIG
tail.location = (0, -CAR_LEN / 2 - 0.3, 0.2)
if A.markers:
    mk = mesh_obj("DirMarkerPlusY", [(-0.3, 4.0, -0.3), (0.3, 4.0, -0.3), (0.3, 4.6, -0.3), (-0.3, 4.6, -0.3),
                                      (-0.3, 4.0, 0.3), (0.3, 4.0, 0.3), (0.3, 4.6, 0.3), (-0.3, 4.6, 0.3)],
                 [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], coll=C_FX)
    mk.parent = RIG
    mk.data.materials.append(emit_mat("MarkerGreen", (0.0, 1.0, 0.0), 30.0))
# path: enters left at 46 s, exits right ~57 s, ~10 m/s, gentle bob + bank into a slight curve
tc0 = bt(B_BREAK1) + 2.2
u = T - tc0
cpos = np.stack([-58.0 + 9.5 * u, 62.0 + 5.0 * np.sin(u * 0.22), 89.3 + 0.4 * np.sin(u * 1.3)], 1)
cvel = np.gradient(cpos, axis=0) * FPS
yaw = np.arctan2(cvel[:, 1], cvel[:, 0]) - np.pi / 2
roll = -np.gradient(yaw) * FPS * 0.6
pitch = np.gradient(cpos[:, 2]) * FPS * -0.02
bake_vec(RIG, "location", cpos)
bake_vec(RIG, "rotation_euler", np.stack([pitch, roll, yaw], 1))
car_vis = ((T > tc0 - 0.5) & (T < bt(B_DROP2))).astype(float)
bake(RIG, "hide_render", 1.0 - car_vis, interp="CONSTANT")
for o in RIG.children_recursive:
    bake(o, "hide_render", 1.0 - car_vis, interp="CONSTANT")
# direction check (numeric): model nose (local -X end) relative to tail must point along the velocity
dg = bpy.context.evaluated_depsgraph_get()
nose_l = Vector((clo[0], (clo[1] + chi[1]) / 2, (clo[2] + chi[2]) / 2))
tail_l = Vector((chi[0], (clo[1] + chi[1]) / 2, (clo[2] + chi[2]) / 2))
for f in (fr(tc0 + 2.0), fr(tc0 + 5.5), fr(tc0 + 9.0)):
    SC.frame_set(f)
    mw = MODEL.matrix_world
    fwd = (mw @ nose_l) - (mw @ tail_l)
    vel = Vector(cvel[f - 1])
    dot = fwd.normalized().dot(vel.normalized())
    print(f"[dircheck] car frame {f}: nose·velocity = {dot:+.3f}")
    assert dot > 0.9, "flying car is not facing its velocity"

# ---------------------------------------------------------------------------------------------------------
# 15. camera shots (never react to the music; cuts on downbeats)
# ---------------------------------------------------------------------------------------------------------
fly = FLY


def catmull(pts, s):
    """centripetal-free uniform Catmull-Rom through pts, s in [0, 1]"""
    n = len(pts) - 1
    x = min(max(s, 0.0), 1.0) * n
    i = min(int(x), n - 1)
    t = x - i
    p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


BRIDGE_LOOK = Vector((TCEN.x + 10, TCEN.y + 10, TDECK + 15))
SHOTS = []          # (t_start, fn(t) -> (loc, target, lens, fstop, focus))


def lerpv(a, b, x):
    return Vector(a).lerp(Vector(b), float(x))


def shot(t0_, fn):
    SHOTS.append((t0_, fn))


shot(0.0, lambda t: (lerpv((-3.0, -15.8, ROOFZ + 1.45), (-2.2, -9.5, ROOFZ + 1.5), smoothstep(t / bt(33))),
                     Vector((2.0, 60.0, ROOFZ + 5.0)), 22.0, 4.0, 30.0))
shot(bt(33), lambda t: (lerpv((9.0, -5.0, ROOFZ + 0.42), (5.5, -2.0, ROOFZ + 0.5), smoothstep((t - bt(33)) / (tD1 - bt(33)))),
                        Vector((26.0, 30.0, ROOFZ + 5.5)), 28.0, 2.8, 32.0))


def crane(t):
    x = smootherstep((t - tD1) / (bt(B_DROP1 + 14) - tD1))
    x2 = smoothstep((t - bt(B_DROP1 + 14)) / (tBK - bt(B_DROP1 + 14)))
    a, b, c = Vector((0.0, 4.0, ROOFZ + 1.6)), Vector((-18.0, -46.0, 168.0)), Vector((-23.0, -58.0, 172.0))
    loc = a.lerp(b, x).lerp(c, x2)
    tg = Vector((0.0, 120.0, ROOFZ + 14.0)).lerp(Vector((70.0, 520.0, 150.0)), x)
    return loc, tg, 24.0 + 4.0 * x, 16.0, 200.0


shot(tD1, crane)
shot(tBK, lambda t: (lerpv((-8.0, 28.0, 92.0), (-5.0, 34.0, 93.0), smoothstep((t - tBK) / (tD2 - tBK))),
                     Vector((HOLO_C.x, HOLO_C.y, 99.0)), 38.0, 5.6, 60.0))
shot(tD2, lambda t: (lerpv(cam5, cam5 + Vector((22.0, 20.0, -2.0)), smoothstep((t - tD2) / (tFL - tD2))),
                     DS_C + Vector((0, 0, -12.0)), 32.0, 16.0, 400.0))
T_SETTLE = bt(B_BRIDGE + 4)
BRIDGE_CAM = FLY[-1]


def flythrough(t):
    if t < T_SETTLE:
        s = (t - tFL) / (T_SETTLE - tFL)
        se_ = 1.0 - (1.0 - s) ** 2.2                   # fast in, settles onto the deck (slow out)
        loc = catmull(FLY, se_)
        ahead = catmull(FLY, min(1.0, se_ + 0.06))
        hd = ahead - loc
        hd.z = 0.0
        hd = hd.normalized() if hd.length > 1e-6 else Vector((0.6, 0.8, 0.0))
        look = loc + hd * 160.0 + Vector((0, 0, -22.0))      # level gaze: the city falls away as we climb
        w = smoothstep((s - 0.6) / 0.4)
        tg = look.lerp(BRIDGE_LOOK, w)
        return loc, tg, 22.0 - 4.0 * w, 16.0, 200.0
    x = smoothstep((t - T_SETTLE) / (bt(B_LAST + 4) - T_SETTLE))
    return BRIDGE_CAM + Vector((1.2, 1.2, -0.6)) * x, BRIDGE_LOOK, 18.0, 16.0, 200.0


shot(tFL, flythrough)
_boxes = []
for ob in HERO_OBS:
    bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    _boxes.append((ob.name, Vector([min(v[i] for v in bb) for i in range(3)]), Vector([max(v[i] for v in bb) for i in range(3)])))
for k_, (a0, b0, z0, a1, b1, z1) in enumerate(tb):
    _boxes.append((f"T_tier{k_}", Vector((a0, b0, z0)), Vector((a1, b1, z1))))
_mind, _who = 1e9, None
for tt in np.linspace(tFL, T_SETTLE, 400):
    lp = flythrough(tt)[0]
    for nm_, lo_, hi_ in _boxes:
        if nm_.startswith("T_tier") and lp.z > hi_.z:      # arriving above the deck is the landing, not a clip
            continue
        q = Vector([min(max(lp[i], lo_[i]), hi_[i]) for i in range(3)])
        if (q - lp).length < _mind:
            _mind, _who = (q - lp).length, (nm_, round(tt, 2), tuple(round(x, 1) for x in lp))
print(f"[flycheck] min clearance camera→tower bbox during fly-through: {_mind:.1f} m  ({_who})")
assert _mind > 4.0, "fly-through path clips a tower"
ORB_C = Vector((TCEN.x + 120, TCEN.y + 150, 230.0))


def orbit(t):
    a = deg(-118) + deg(9) * smoothstep((t - bt(B_LAST + 4)) / (tDP - bt(B_LAST + 4)))
    loc = ORB_C + Vector((math.cos(a), math.sin(a), 0)) * 980.0 + Vector((0, 0, 60.0))
    return loc, ORB_C + Vector((0, 0, 140.0)), 30.0, 16.0, 900.0


shot(bt(B_LAST + 4), orbit)
shot(tDP, lambda t: (lerpv((2.0, -7.0, ROOFZ + 0.55), (2.0, -5.5, ROOFZ + 0.6), smoothstep((t - tDP) / (tFN - tDP))),
                     Vector((60.0, 420.0, 330.0)), 18.0, 16.0, 200.0))
shot(tFN, lambda t: (lerpv((0.0, -13.0, ROOFZ + 1.7), (-5.0, -30.0, ROOFZ + 15.0), smoothstep((t - tFN) / (NF / FPS - tFN))),
                     Vector((45.0, 420.0, 230.0)), 20.0, 16.0, 200.0))
SHOTS.sort(key=lambda s_: s_[0])
loc_a, rot_a, lens_a, fst_a, foc_a = [], [], [], [], []
prev = None
for i in range(NF):
    t = T[i]
    fn = [fn_ for t0_, fn_ in SHOTS if t0_ <= t + 1e-9][-1]
    loc, tg, lens, fst, foc = fn(t)
    q = look_quat(loc, tg)
    e = q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")
    prev = e
    loc_a.append(loc[:]); rot_a.append(e[:]); lens_a.append(lens); fst_a.append(fst); foc_a.append(foc)
bake_vec(CAM, "location", np.array(loc_a))
bake_vec(CAM, "rotation_euler", np.array(rot_a))
bake(CAM.data, "lens", np.array(lens_a))
CAM.data.sensor_width = 36.0
CAM.data.clip_start, CAM.data.clip_end = 0.1, 6000.0
CAM.data.dof.use_dof = True
CAM.data.dof.aperture_blades = 7
bake(CAM.data, "dof.aperture_fstop", np.array(fst_a))
bake(CAM.data, "dof.focus_distance", np.array(foc_a))
# timeline markers at the cuts (for scrubbing)
for t0_, _fn in SHOTS:
    SC.timeline_markers.new(f"cut {t0_:.2f}s", frame=fr(t0_))
print("[camera] cuts at", [round(s_[0], 2) for s_ in SHOTS])

# ---------------------------------------------------------------------------------------------------------
# 16. world: overcast night sky lit from below by the city; city HDRI only for reflections
# ---------------------------------------------------------------------------------------------------------
W = bpy.data.worlds.new("NightRain")
SC.world = W
wn = NB(W.node_tree)
W.node_tree.nodes.clear()
wout = wn.n("ShaderNodeOutputWorld")
wcg = ctrl(wn)
tcw = wn.n("ShaderNodeTexCoord")
dx, dy, dz = wn.sep(tcw.outputs["Generated"])
up = wn.mr(dz, -0.05, 0.6)
grad = wn.mix(up, (0.030, 0.018, 0.040), (0.004, 0.005, 0.011))
cl_ = wn.n("ShaderNodeTexNoise", noise_dimensions="4D")
wn.set(cl_.inputs["Scale"], 3.2)
wn.set(cl_.inputs["Detail"], 8.0)
wn.set(cl_.inputs["Roughness"], 0.62)
wn.link(tcw.outputs["Generated"], cl_.inputs["Vector"])
wn.link(wn.m("MULTIPLY", wcg["time"], 0.012), cl_.inputs["W"])
cden = wn.mr(cl_.outputs["Fac"], 0.42, 0.72)
glow = wn.m("MULTIPLY", wn.m("MULTIPLY", cden, wn.m("SUBTRACT", 1.15, up)), wcg["sky"])
sky_c = wn.mix(wn.m("MULTIPLY", glow, 0.85), grad, (0.11, 0.055, 0.10), blend="ADD")
below = wn.m("LESS_THAN", dz, 0.0)
sky_c = wn.mix(below, sky_c, (0.012, 0.009, 0.014))
bg = wn.n("ShaderNodeBackground")
wn.link(sky_c, bg.inputs["Color"])
wn.set(bg.inputs["Strength"], 1.0)
hdr_p = os.path.join(BMO, "ph_hdri_shanghai_bund", "shanghai_bund_2k.hdr")
if os.path.exists(hdr_p):
    env = wn.n("ShaderNodeTexEnvironment", image=img(hdr_p))
    bg2 = wn.n("ShaderNodeBackground")
    wn.link(env.outputs["Color"], bg2.inputs["Color"])
    wn.set(bg2.inputs["Strength"], 0.05)
    lp = wn.n("ShaderNodeLightPath")
    mx = wn.n("ShaderNodeMixShader")
    wn.link(lp.outputs["Is Camera Ray"], mx.inputs[0])
    add = wn.n("ShaderNodeAddShader")
    wn.link(bg2.outputs[0], add.inputs[0])
    wn.link(bg.outputs[0], add.inputs[1])
    wn.link(add.outputs[0], mx.inputs[1])
    wn.link(bg.outputs[0], mx.inputs[2])
    wn.link(mx.outputs[0], wout.inputs["Surface"])
else:
    wn.link(bg.outputs[0], wout.inputs["Surface"])
# a faint cool moon-through-cloud key so silhouettes read
moon = bpy.data.objects.new("SkyKey", bpy.data.lights.new("SkyKey", "SUN"))
moon.data.energy, moon.data.color, moon.data.angle = 0.025, (0.6, 0.7, 1.0), deg(8)
moon.rotation_euler = (deg(55), 0, deg(-140))
SC.collection.objects.link(moon)

# ---------------------------------------------------------------------------------------------------------
# 17. compositor: rain-haze (mist), bloom, a touch of lens dispersion + vignette (grain added at mux)
# ---------------------------------------------------------------------------------------------------------
vl = SC.view_layers[0]
vl.use_pass_mist = True
W.mist_settings.start = 15.0
W.mist_settings.depth = 1500.0
W.mist_settings.falloff = "LINEAR"
cg = bpy.data.node_groups.new("Post", "CompositorNodeTree")
cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
cb = NB(cg)
rl = cb.n("CompositorNodeRLayers")
go = cb.n("NodeGroupOutput")
haze_c = cb.n("CompositorNodeCombineColor")
for k_, v_ in enumerate((0.030, 0.026, 0.042)):
    haze_c.inputs[k_].default_value = v_
hm = cb.mix(cb.m("MINIMUM", cb.m("MULTIPLY", rl.outputs["Mist"], 0.95), 0.82), rl.outputs["Image"], haze_c.outputs[0])
glr = cb.n("CompositorNodeGlare")
glr.inputs["Type"].default_value = "Fog Glow"
glr.inputs["Quality"].default_value = "High"
glr.inputs["Threshold"].default_value = 0.9
glr.inputs["Strength"].default_value = 0.55
glr.inputs["Size"].default_value = 0.75
cb.link(hm, glr.inputs["Image"])
ld = cb.n("CompositorNodeLensdist")
ld.inputs["Distortion"].default_value = -0.004
ld.inputs["Dispersion"].default_value = 0.010
ld.inputs["Fit"].default_value = True
cb.link(glr.outputs["Image"], ld.inputs["Image"])
vg = cb.n("CompositorNodeEllipseMask")
vg.inputs["Size"].default_value = (1.3, 1.3)
vbl = cb.n("CompositorNodeBlur")
try:
    vbl.inputs["Size"].default_value = (300, 300)
except Exception:
    pass
cb.link(vg.outputs["Mask"], vbl.inputs["Image"])
vmr = cb.mr(vbl.outputs["Image"], 0.0, 1.0, 0.72, 1.0)
fin = cb.mix(1.0, ld.outputs["Image"], vmr, blend="MULTIPLY")
cb.link(fin, go.inputs["Image"])
SC.compositing_node_group = cg
SC.render.compositor_device = "GPU"

# ---------------------------------------------------------------------------------------------------------
# 18. save
# ---------------------------------------------------------------------------------------------------------
SC.frame_set(1)
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(A.out), compress=True)
print(f"[build] saved {A.out}  frames 1..{NF}  objects {len(bpy.data.objects)}")
