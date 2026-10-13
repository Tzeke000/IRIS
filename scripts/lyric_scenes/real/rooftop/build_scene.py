"""BE MY ONE — realistic cyberpunk rooftop party (3-D, EEVEE). v1 2026-10-13, v2 same evening.

    ~/.local/bin/blender -b --factory-startup --disable-autoexec --python-exit-code 1 \
        --python scripts/lyric_scenes/real/rooftop/build_scene.py -- \
        --out scratch/scenes/bemyone_rooftop/bemyone.blend [--assets /data/iris_assets] [--markers]

Everything is baked as keyframes or analytic Geometry Nodes driven by Scene Time (traffic, sky traffic, drones,
fireworks) — no handlers, no simulation caches, so any frame renders standalone (chunk-safe).
The camera never reacts to the music; every cut lands on a downbeat of the fitted 134.095 BPM grid.

v2 (Zeke): NO falling rain — the rain stopped about an hour ago: wet roofs/streets/puddles, damp facades, vent
steam, lingering haze; motion blur back on (cut-safe). Real street layout + footprints from OpenStreetMap
(Mong Kok; © OpenStreetMap contributors, ODbL) with interior-mapped windows; real-world towers replace the Artemis
model that carried a "2023" sign; research-based fireworks (shell sizes, crossettes, willows, colour change,
strobes, smoke lit from within, diffuse-only flash spill); hero flying cars + sky-lane traffic with aviation nav
lights, all direction-checked; drop-2 strobe is a clean radial ripple.

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
# v2: no falling rain (Zeke: "the rain stopped about an hour ago, everything still wet") → motion blur back on.
# Shutter opens AT the frame (START) and the camera key before every cut is CONSTANT, so blur never crosses a cut.
SC.render.use_motion_blur = True
SC.render.motion_blur_shutter = 0.5
SC.render.motion_blur_position = "START"
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
ee.motion_blur_steps = 1
ee.volumetric_start = 0.5
ee.volumetric_end = 2200.0
ee.volumetric_tile_size = "8"
ee.volumetric_samples = 64
ee.volumetric_sample_distribution = 0.85
ee.use_volumetric_shadows = False
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
# drop-2 window strobe v2: a clean RADIAL ripple — on every beat a lit ring expands from under the drone show
RIPPLE_V = 950.0                                       # m/s: crosses the visible city in ~0.7 s
last_beat_t = T0 + np.floor(BEAT) * P
wr = np.where((strobe > 0) & (BEAT >= 0), (T - last_beat_t) * RIPPLE_V, 0.0)
ctrl_curves = dict(wr=wr, city=city, cityB=cityB, strobe=strobe, wphase=wphase, wpulse=PULSE, shut=shut, bb=bb,
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


def facade_material(name="Facade", wstr=2.4):
    """Night facade v2: damp walls with drip streaks, dark glass, and INTERIOR-MAPPED rooms behind every window
    (Van Dongen 2008: cast the view ray into a virtual room box per window cell — back wall / side walls /
    floor / ceiling — and shade the nearest hit; no geometry). Keyed: room-light density (city), radial strobe
    ripple (wr), shutoff (soff vs shut). Attributes: bid (random per building), soff, kind, led, hue."""
    m, nb, out = new_mat(name)
    cg = ctrl(nb)
    g = nb.geo()
    Pp, Nn, Inc = g.outputs["Position"], g.outputs["Normal"], g.outputs["Incoming"]
    px, py, pz = nb.sep(Pp)
    nx, ny, nz = nb.sep(Nn)
    bid = nb.sattr("bid").outputs["Fac"]
    soff = nb.sattr("soff").outputs["Fac"]
    kind = nb.sattr("kind").outputs["Fac"]
    led = nb.sattr("led").outputs["Fac"]
    hue = nb.sattr("hue").outputs["Fac"]
    side = nb.m("SUBTRACT", 1.0, nb.mr(nb.m("ABSOLUTE", nz), 0.3, 0.6))
    tcoord = nb.m("ADD", nb.m("MULTIPLY", px, nb.m("MULTIPLY", ny, -1.0)), nb.m("MULTIPLY", py, nx))
    fh = nb.m("ADD", 3.1, nb.m("MULTIPLY", nb.m("FRACT", nb.m("MULTIPLY", bid, 13.7)), 1.1))      # 3.1–4.2 m floors
    bw = nb.m("ADD", 1.8, nb.m("MULTIPLY", nb.m("FRACT", nb.m("MULTIPLY", bid, 7.3)), 2.2))       # 1.8–4.0 m bays
    fz = nb.m("DIVIDE", pz, fh)
    fl = nb.m("FLOOR", fz)
    fzf = nb.m("FRACT", fz)
    bc = nb.m("DIVIDE", tcoord, bw)
    bay = nb.m("FLOOR", bc)
    bcf = nb.m("FRACT", bc)
    win = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("GREATER_THAN", fzf, 0.20), nb.m("LESS_THAN", fzf, 0.88)),
               nb.m("MULTIPLY", nb.m("GREATER_THAN", bcf, 0.08), nb.m("LESS_THAN", bcf, 0.92)))
    win = nb.m("MULTIPLY", nb.m("MULTIPLY", win, side), nb.m("LESS_THAN", kind, 0.5))
    wn = nb.n("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    nb.set(wn.inputs["Vector"], nb.xyz(fl, bay, nb.m("MULTIPLY", bid, 977.0)))
    cell = wn.outputs["Value"]
    cr, cgc, cb_ = nb.sep(wn.outputs["Color"])
    wn2 = nb.n("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    nb.set(wn2.inputs["Vector"], nb.xyz(bay, fl, nb.m("ADD", nb.m("MULTIPLY", bid, 313.0), 7.0)))
    r1, r2, r3 = nb.sep(wn2.outputs["Color"])
    dark_bld = nb.m("GREATER_THAN", nb.m("FRACT", nb.m("MULTIPLY", bid, 31.1)), 0.9)
    litfrac = nb.m("MULTIPLY", cg["city"], nb.m("MULTIPLY", nb.m("ADD", 0.10, nb.m("MULTIPLY", bid, 0.28)),
                                                     nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", dark_bld, 0.85))))
    lit = nb.m("LESS_THAN", cell, litfrac)
    on = nb.m("GREATER_THAN", soff, cg["shut"])
    # ---- interior mapping (window-cell local frame: T along facade, U up, K into the building) ----
    d = nb.v("SCALE", Inc, -1.0)
    Tv = nb.xyz(nb.m("MULTIPLY", ny, -1.0), nx, 0.0)
    dx = nb.v("DOT_PRODUCT", d, Tv)
    _, _, dy = nb.sep(d)
    dz = nb.m("MAXIMUM", nb.m("MULTIPLY", nb.v("DOT_PRODUCT", d, Nn), -1.0), 0.04)
    hx = nb.m("MULTIPLY", bw, 0.5)
    hy = nb.m("MULTIPLY", fh, 0.5)
    lx = nb.m("MULTIPLY", nb.m("SUBTRACT", bcf, 0.5), bw)
    ly = nb.m("MULTIPLY", nb.m("SUBTRACT", fzf, 0.5), fh)
    D = nb.m("ADD", 2.5, nb.m("MULTIPLY", r1, 3.5))                                               # 2.5–6 m deep rooms
    sx = nb.m("SUBTRACT", nb.m("MULTIPLY", nb.m("GREATER_THAN", dx, 0.0), 2.0), 1.0)
    sy = nb.m("SUBTRACT", nb.m("MULTIPLY", nb.m("GREATER_THAN", dy, 0.0), 2.0), 1.0)
    tx = nb.m("DIVIDE", nb.m("SUBTRACT", nb.m("MULTIPLY", sx, hx), lx), nb.m("ADD", dx, nb.m("MULTIPLY", sx, 1e-4)))
    ty = nb.m("DIVIDE", nb.m("SUBTRACT", nb.m("MULTIPLY", sy, hy), ly), nb.m("ADD", dy, nb.m("MULTIPLY", sy, 1e-4)))
    tz = nb.m("DIVIDE", D, dz)
    txy = nb.m("MINIMUM", tx, ty)
    t_ = nb.m("MINIMUM", txy, tz)
    is_back = nb.m("LESS_THAN", tz, nb.m("ADD", txy, 1e-4))
    is_x = nb.m("MULTIPLY", nb.m("LESS_THAN", tx, nb.m("ADD", ty, 1e-4)), nb.m("SUBTRACT", 1.0, is_back))
    is_y = nb.m("MULTIPLY", nb.m("SUBTRACT", 1.0, is_back), nb.m("SUBTRACT", 1.0, is_x))
    is_ceil = nb.m("MULTIPLY", is_y, nb.m("GREATER_THAN", sy, 0.0))
    is_floor = nb.m("SUBTRACT", is_y, is_ceil)
    hxp = nb.m("ADD", lx, nb.m("MULTIPLY", dx, t_))
    hyp = nb.m("ADD", ly, nb.m("MULTIPLY", dy, t_))
    hzp = nb.m("MULTIPLY", dz, t_)
    lampv = nb.v("SUBTRACT", nb.xyz(hxp, hyp, hzp), nb.xyz(nb.m("MULTIPLY", nb.m("SUBTRACT", r2, 0.5), bw),
                                                            nb.m("SUBTRACT", hy, 0.25), nb.m("MULTIPLY", D, 0.45)))
    lamp = nb.m("ADD", 0.12, nb.m("DIVIDE", 1.0, nb.m("ADD", 1.0, nb.m("DIVIDE", nb.v("DOT_PRODUCT", lampv, lampv), 5.0))))
    paint = nb.mix(nb.m("GREATER_THAN", r3, 0.66), nb.mix(nb.m("GREATER_THAN", r3, 0.33), (0.78, 0.66, 0.52), (0.78, 0.78, 0.76)),
                   (0.36, 0.56, 0.55))
    alb = nb.mix(is_x, paint, nb.mix(1.0, paint, (0.8, 0.8, 0.8), blend="MULTIPLY"))
    alb = nb.mix(is_ceil, alb, (0.86, 0.86, 0.84))
    alb = nb.mix(is_floor, alb, (0.22, 0.16, 0.11))
    # furniture silhouette against the back wall (sofa / shelf / desk) + curtains at the glass
    wn3 = nb.n("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    nb.set(wn3.inputs["Vector"], nb.xyz(nb.m("ADD", bay, 17.0), nb.m("ADD", fl, 3.0), nb.m("MULTIPLY", bid, 91.0)))
    f1, f2, f3 = nb.sep(wn3.outputs["Color"])
    has = nb.m("GREATER_THAN", f3, 0.45)                                         # ~half the rooms show furniture
    fh_ = nb.m("ADD", 0.45, nb.m("MULTIPLY", f2, 1.3))                           # 0.45–1.75 m tall (sofa … shelf)
    fw_ = nb.m("MULTIPLY", bw, nb.m("ADD", 0.12, nb.m("MULTIPLY", f1, 0.3)))     # 12–42 % of the bay
    furn = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", is_back, has),
                                 nb.m("LESS_THAN", hyp, nb.m("ADD", nb.m("MULTIPLY", hy, -1.0), fh_))),
                nb.m("LESS_THAN", nb.m("ABSOLUTE", nb.m("SUBTRACT", hxp, nb.m("MULTIPLY", nb.m("SUBTRACT", r1, 0.5), nb.m("MULTIPLY", bw, 0.8)))), fw_))
    alb = nb.mix(furn, alb, (0.03, 0.03, 0.035))
    cover = nb.m("MAXIMUM", nb.m("SUBTRACT", nb.m("MULTIPLY", cb_, 1.4), 0.55), 0.0)
    curtain = nb.m("GREATER_THAN", nb.m("ABSOLUTE", lx), nb.m("MULTIPLY", hx, nb.m("SUBTRACT", 1.0, nb.m("MINIMUM", cover, 0.75))))
    roomlight = nb.mix(nb.m("GREATER_THAN", cr, 0.58), (1.0, 0.66, 0.36), (0.78, 0.86, 1.0))
    roomlight = nb.mix(nb.m("GREATER_THAN", cb_, 0.94), roomlight, (0.3, 0.85, 1.0))
    room = nb.mix(1.0, nb.mix(1.0, alb, roomlight, blend="MULTIPLY"), nb.xyz(lamp, lamp, lamp), blend="MULTIPLY")
    room = nb.mix(curtain, room, nb.mix(1.0, roomlight, (0.55, 0.5, 0.42), blend="MULTIPLY"))
    wi = nb.m("MULTIPLY", nb.m("MULTIPLY", lit, nb.m("ADD", 0.55, nb.m("POWER", cgc, 1.5))), cg["cityB"])
    # ---- radial strobe ripple (drop 2): ring expands from under the drone show on every beat ----
    dxy = nb.v("LENGTH", nb.v("SUBTRACT", nb.xyz(px, py, 0.0), (DS_C.x, DS_C.y, 0.0)))
    behind = nb.m("SUBTRACT", cg["wr"], dxy)
    ring = nb.m("MULTIPLY", nb.m("GREATER_THAN", behind, 0.0), nb.m("EXPONENT", nb.m("DIVIDE", behind, -95.0)))
    strb = nb.m("MULTIPLY", ring, nb.m("MULTIPLY", cg["strobe"], 4.0))
    e_win = nb.mix(nb.m("DIVIDE", strb, nb.m("ADD", strb, nb.m("ADD", wi, 0.001))), room, (0.6, 0.88, 1.0))
    wtot = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("ADD", wi, strb), win), on)
    # LED floor bands (cyberpunk trim) on some buildings
    ledband = nb.m("LESS_THAN", nb.m("FRACT", nb.m("DIVIDE", pz, 21.6)), 0.022)
    ledamt = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", ledband, led), side), nb.m("MULTIPLY", on, nb.m("POWER", cg["city"], 2.0)))
    hsv = nb.n("ShaderNodeHueSaturation")
    nb.set(hsv.inputs["Color"], (1.0, 0.1, 0.1))
    nb.set(hsv.inputs["Hue"], hue)
    ecol = nb.mix(nb.m("DIVIDE", ledamt, nb.m("ADD", ledamt, nb.m("ADD", wtot, 0.0001))), e_win, hsv.outputs[0])
    estr = nb.m("MULTIPLY", nb.m("ADD", wtot, nb.m("MULTIPLY", ledamt, 3.0)), wstr)
    # ---- surface: damp concrete/tile with vertical drip streaks, wet glass, wet roofs ----
    walln = nb.n("ShaderNodeTexNoise")
    nb.set(walln.inputs["Scale"], 0.35)
    nb.set(walln.inputs["Detail"], 6.0)
    streak = nb.n("ShaderNodeTexNoise")
    nb.link(nb.xyz(nb.m("MULTIPLY", tcoord, 1.6), nb.m("MULTIPLY", pz, 0.05), nb.m("MULTIPLY", bid, 50.0)), streak.inputs["Vector"])
    nb.set(streak.inputs["Scale"], 1.0)
    nb.set(streak.inputs["Detail"], 3.0)
    wet = nb.mr(streak.outputs["Fac"], 0.42, 0.62)                         # 0 = drier, 1 = rain-darkened run-off streak
    tint = nb.mix(nb.m("FRACT", nb.m("MULTIPLY", bid, 5.1)), (0.05, 0.048, 0.046), (0.075, 0.07, 0.065))
    wallc = nb.mix(walln.outputs["Fac"], tint, nb.mix(1.0, tint, (1.45, 1.4, 1.35), blend="MULTIPLY"))
    wallc = nb.mix(nb.m("MULTIPLY", wet, 0.55), wallc, nb.mix(1.0, wallc, (0.45, 0.45, 0.45), blend="MULTIPLY"))
    base = nb.mix(win, wallc, (0.006, 0.008, 0.010))
    wall_rough = nb.mix(wet, 0.55, 0.16, dtype="FLOAT")
    rough = nb.mix(win, wall_rough, 0.04, dtype="FLOAT")
    b = principled(nb, base, rough, 0.0)
    nb.set(b.inputs["Emission Color"], ecol)
    nb.set(b.inputs["Emission Strength"], estr)
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def set_face_attrs(me, **vals):
    for k, v in vals.items():
        a = me.attributes.get(k) or me.attributes.new(k, "FLOAT", "FACE")
        a.data.foreach_set("value", np.full(len(me.polygons), v, np.float32))




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
    rip = nb.n("ShaderNodeTexNoise")                  # v2: rain has stopped — still puddles, only a faint surface wobble
    nb.set(rip.inputs["Scale"], 3.0)
    nb.set(rip.inputs["Detail"], 2.0)
    nb.link(tc.outputs["Object"], rip.inputs["Vector"])
    nm = nb.n("ShaderNodeNormalMap")
    nb.link(tex["nor_gl"].outputs["Color"], nm.inputs["Color"])
    nb.set(nm.inputs["Strength"], nb.mix(pm, 0.8, 0.0, dtype="FLOAT"))
    bump = nb.n("ShaderNodeBump")
    nb.set(bump.inputs["Strength"], nb.m("MULTIPLY", pm, 0.012))
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
HEROES = [  # (model, x, y, height m, yaw deg) — v2: Artemis "b" (carried a baked "2023" sign) → real-world towers
    ("sf_skyscraper_a", -178.0, 300.0, 205.0, 20.0),
    ("sf_bld_atlanta", -40.0, 236.0, 205.0, -15.0),
    ("sf_skyscraper_c", -122.0, 392.0, 250.0, 35.0),
    ("sf_skyscraper_a", 58.0, 360.0, 228.0, 200.0),
    ("sf_bld_atlanta", -58.0, 470.0, 236.0, 160.0),
    ("sf_skyscraper_c", 140.0, 430.0, 215.0, -60.0),
    ("sf_bld_atlanta", 210.0, 610.0, 230.0, 90.0),
    ("sf_skyscraper_a", -20.0, 640.0, 220.0, 120.0),
    ("sf_bld_singapore", 165.0, 300.0, 150.0, 30.0),
    ("sf_bld_singapore", -245.0, 520.0, 140.0, -20.0),
]


M_FACADE = facade_material()


# sky lanes for flying traffic (Blade Runner-style stacked loops; altitudes kept clear of roofs by osm_cap)
LANES = [dict(c=(40.0, 460.0), r=(430.0, 310.0), z=150.0, dir=1, rot=10.0),      # fitted 10-13: ≥22 m from every
         dict(c=(-260.0, 505.0), r=(300.0, 105.0), z=124.0, dir=-1, rot=60.0),   # tower footprint (offline search)
         dict(c=(20.0, 25.0), r=(240.0, 115.0), z=125.0, dir=1, rot=15.0),       # loops round the home block
         dict(c=(120.0, 640.0), r=(620.0, 430.0), z=182.0, dir=-1, rot=40.0)]


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

    for gx in range(-26, 27):
        for gy in range(-11, 32):
            cx, cy = gx * PITCH, gy * PITCH
            d_down = math.hypot(cx - 40, cy - 560)
            if d_down > 2300:
                continue
            if -1080 < cx < 1080 and -650 < cy < 1650:     # v2: inside the OSM coverage the city is REAL footprints
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


CITY, TOPS = build_city()          # far procedural filler only (outside the OSM coverage)

# ---------------------------------------------------------------------------------------------------------
# 4b. REAL city layout: OpenStreetMap footprints + roads (Mong Kok, Kowloon; © OpenStreetMap contributors, ODbL)
# ---------------------------------------------------------------------------------------------------------
import city_osm  # noqa: E402

OSM_B, OSM_R = city_osm.load(os.path.join(BMO, "osm_hk_mongkok", "osm.json"), rng)


def lane_pts(ln, n=160):
    c, (ra, rb), z, sgn, rot = ln["c"], ln["r"], ln["z"], ln["dir"], math.radians(ln.get("rot", 0.0))
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) * sgn
    x, y = ra * np.cos(th), rb * np.sin(th)
    return np.stack([c[0] + x * math.cos(rot) - y * math.sin(rot), c[1] + x * math.sin(rot) + y * math.cos(rot), np.full(n, z)], 1)


LANE_PTS = [lane_pts(ln) for ln in LANES]


def osm_cap(cx, cy, rad, h):
    """the same keep-outs + height caps the procedural city used, applied to real footprints"""
    if -75 < cx < 75 and -75 < cy < 70:
        return None
    if math.hypot(cx - TCEN.x, cy - TCEN.y) < 60 + rad or math.hypot(cx - HOLO_C.x, cy - HOLO_C.y) < 38 + rad:
        return None
    for slug_, hx_, hy_, _, _ in HEROES:
        if math.hypot(cx - hx_, cy - hy_) < {"sf_bld_atlanta": 62, "sf_bld_singapore": 46}.get(slug_, 40) + rad:
            return None
    if math.hypot(cx - DS_C.x, cy - DS_C.y) < 40 + rad:
        return None
    if abs(cx) < 190 and 40 < cy < 360:                    # home → skyline sightline: look down onto roofs
        h = min(h, 22 + (cy - 40) * 0.14)
    pc = Vector((cx, cy, 0))
    for a, b in zip(FLY[:-1], FLY[1:]):                    # fly-through corridor
        dist, q = seg_dist(pc, Vector((a.x, a.y, 0)), Vector((b.x, b.y, 0)))
        if dist < 40 + rad:
            f_ = (q - Vector((a.x, a.y, 0))).length / max(1e-6, (Vector((b.x, b.y, 0)) - Vector((a.x, a.y, 0))).length)
            h = min(h, a.z + (b.z - a.z) * f_ - 40)
    for lp in LANE_PTS:                                     # sky lanes: keep 25 m under the traffic
        dd = np.hypot(lp[:, 0] - cx, lp[:, 1] - cy)
        if dd.min() < 22 + rad:
            h = min(h, lp[0, 2] - 25)
    return max(h, 8.0) if h > 0 else None


keep, attrs = [], []
for bld in OSM_B:
    c = bld["poly"].mean(0)
    rad = float(np.linalg.norm(bld["poly"] - c, axis=1).max())
    h = osm_cap(float(c[0]), float(c[1]), rad, bld["h"])
    if h is None:
        continue
    bld["h"] = h
    bld["crown"] = rng.uniform(0.72, 0.86) if (h > 70 and rng.random() < 0.45) else None
    keep.append(bld)
    hue_ = float(rng.choice([0.0, 0.5, 0.83, 0.55, 0.9, 0.62]))
    attrs.append(dict(bid=float(rng.random()), soff=math.hypot(c[0], c[1]), kind=0.0,
                      led=1.0 if (h > 45 and rng.random() < 0.10) else 0.0, hue=hue_))
OSM_B = keep
dist_home = np.array([a["soff"] for a in attrs])
rk = np.argsort(np.argsort(-dist_home))
for a, r_ in zip(attrs, rk):
    a["soff"] = 0.02 + 0.86 * r_ / max(1, len(rk) - 1)      # far → small → shuts off first
V_, F_, FA_ = city_osm.prism_mesh(OSM_B, attrs)
me = bpy.data.meshes.new("CityOSM")
me.from_pydata(V_, [], F_)
me.update()
for k, v in FA_.items():
    a_ = me.attributes.new(k, "FLOAT", "FACE")
    a_.data.foreach_set("value", np.asarray(v, np.float32))
CITY_OSM = bpy.data.objects.new("CityOSM", me)
C_CITY.objects.link(CITY_OSM)
me.materials.append(M_FACADE)
for bld in OSM_B:                                            # tops for beacons / lasers / billboards
    p = bld["poly"]
    TOPS.append((float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()), bld["h"],
                 bld["h"] * (bld["crown"] or 0.0), 0, 0, 0, 0, p))
print(f"[osm] {len(OSM_B)} real buildings kept (of {len(keep)}), {len(OSM_R)} road ways; max h {max(b['h'] for b in OSM_B):.0f} m")

# rooftop clutter (HVAC boxes, water tanks, vents, masts, stair houses) scattered on every flat roof
C_KIT = coll("_clutter_kit")
C_KIT.hide_render = True
C_KIT.hide_viewport = True
M_CLUT = simple_mat("RoofKit", (0.07, 0.07, 0.072), 0.35, 0.4)
kit = []
for nm_, bx in (("AC", (2.2, 1.3, 1.1)), ("Vent", (1.0, 1.0, 0.9)), ("Stair", (4.0, 3.0, 3.0)), ("AC2", (3.5, 1.6, 1.4))):
    o = boxes_mesh(f"Kit{nm_}", [(-bx[0] / 2, -bx[1] / 2, 0, bx[0] / 2, bx[1] / 2, bx[2])], coll=C_KIT)
    o.data.materials.append(M_CLUT)
    kit.append(o)
kit.append(cylinder("KitTank", 1.4, 3.2, (0, 0, 0), M_CLUT, segs=12, coll_=C_KIT))
kit.append(cylinder("KitMast", 0.07, 7.0, (0, 0, 0), M_CLUT, segs=6, coll_=C_KIT))


def clutter_tree():
    ng, nb, gi, go = gn_tree("RoofClutter")
    oi = nb.n("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    oi.inputs["Object"].default_value = CITY_OSM
    nrm = nb.n("GeometryNodeInputNormal").outputs[0]
    _, _, nzz = nb.sep(nrm)
    _, _, pzz = nb.sep(nb.n("GeometryNodeInputPosition").outputs[0])
    sel = nb.band(nb.cmp("GREATER_THAN", nzz, 0.9), nb.cmp("GREATER_THAN", pzz, 9.0))
    dist = nb.n("GeometryNodeDistributePointsOnFaces", distribute_method="RANDOM")
    nb.link(oi.outputs["Geometry"], dist.inputs["Mesh"])
    nb.link(sel, dist.inputs["Selection"])
    nb.set(dist.inputs["Density"], 1.0 / 140.0)
    ci = nb.n("GeometryNodeCollectionInfo", transform_space="ORIGINAL")
    ci.inputs["Collection"].default_value = C_KIT
    nb.set(ci.inputs["Separate Children"], True)
    nb.set(ci.inputs["Reset Children"], True)
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(dist.outputs["Points"], inst.inputs["Points"])
    nb.link(ci.outputs[0], inst.inputs["Instance"])
    nb.set(inst.inputs["Pick Instance"], True)
    ri = nb.n("FunctionNodeRandomValue", data_type="INT")
    nb.set(ri.inputs["Min"], 0)
    nb.set(ri.inputs["Max"], len(kit) - 1)
    nb.link(ri.outputs["Value"], inst.inputs["Instance Index"])
    rz_ = nb.n("FunctionNodeRandomValue", data_type="FLOAT")
    nb.set(rz_.inputs["Max"], 6.283)
    er = nb.n("FunctionNodeEulerToRotation") if hasattr(bpy.types, "FunctionNodeEulerToRotation") else None
    if er is not None:
        nb.link(nb.xyz(0, 0, rz_.outputs["Value"]), er.inputs[0])
        nb.link(er.outputs[0], inst.inputs["Rotation"])
    sc_ = nb.rand(0.75, 1.35, seed=3)
    nb.link(nb.xyz(sc_, sc_, sc_), inst.inputs["Scale"])
    nb.link(inst.outputs[0], go.inputs[0])
    return ng


CLUT = points_obj("RoofClutter", np.zeros((1, 3)), coll=C_CITY)
add_gn(CLUT, clutter_tree())

# hand-placed blocks around the home roof (signs hang on W1 / N1)
hb, hattrs = [], dict(bid=[], soff=[], kind=[], led=[], hue=[])
for i, (x0, y0, x1, y1, z) in enumerate(HAND):
    hb.append((x0, y0, 0.0, x1, y1, z))
    for k, v in dict(bid=rng.random(), soff=0.92 + 0.01 * i, kind=0.0, led=0.0, hue=0.0).items():
        hattrs[k].append(v)
HANDOB = boxes_mesh("HandBlocks", hb, hattrs, coll=C_CITY)
HANDOB.data.materials.append(M_FACADE)

# ground (wet asphalt everywhere) + REAL roads: lit wet ribbons, street lamps, left-hand traffic (Hong Kong)
gr = mesh_obj("Ground", [(-2600, -1400, 0), (2600, -1400, 0), (2600, 2800, 0), (-2600, 2800, 0)], [(0, 1, 2, 3)], coll=C_CITY)
gr.data.materials.append(M_GROUND)


def curve_obj(name, polylines, z=0.0, cyclic=False):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    for pts in polylines:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        co = np.ones((len(pts), 4), np.float32)
        co[:, :2] = pts[:, :2]
        co[:, 2] = pts[:, 2] if pts.shape[1] > 2 else z
        sp.points.foreach_set("co", co.ravel())
        sp.use_cyclic_u = cyclic
    ob = bpy.data.objects.new(name, cu)
    C_CITY.objects.link(ob)
    return ob


def road_material():
    m, nb, out = new_mat("WetRoad")
    cg = ctrl(nb)
    al = nb.sattr("along").outputs["Fac"]
    pool = nb.m("POWER", nb.m("ADD", 0.5, nb.m("MULTIPLY", 0.5, nb.m("COSINE", nb.m("MULTIPLY", al, 2 * math.pi / 26.0)))), 5.0)
    gn_ = nb.n("ShaderNodeTexNoise")
    nb.set(gn_.inputs["Scale"], 0.08)
    pud = nb.mr(gn_.outputs["Fac"], 0.45, 0.6)
    b = principled(nb, (0.018, 0.018, 0.02), nb.mix(pud, 0.22, 0.03, dtype="FLOAT"), 0.0)
    nb.set(b.inputs["Emission Color"], (1.0, 0.7, 0.42))
    lv = nb.m("MULTIPLY", nb.m("SUBTRACT", 1.0, cg["shut"]), nb.m("ADD", 0.5, nb.m("MULTIPLY", cg["city"], 0.5)))
    nb.link(nb.m("MULTIPLY", nb.m("MULTIPLY", pool, lv), 0.32), b.inputs["Emission Strength"])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


M_ROAD = road_material()


def ribbon_tree(width):
    ng, nb, gi, go = gn_tree(f"RoadRibbon{int(width)}")
    sp_ = nb.n("GeometryNodeSplineParameter")
    stc = nb.store(gi.outputs[0], "along", sp_.outputs["Length"], domain="POINT")
    prof = nb.n("GeometryNodeCurvePrimitiveLine")
    nb.set(prof.inputs["Start"], (-width / 2, 0, 0))
    nb.set(prof.inputs["End"], (width / 2, 0, 0))
    c2m = nb.n("GeometryNodeCurveToMesh")
    nb.link(stc, c2m.inputs["Curve"])
    nb.link(prof.outputs[0], c2m.inputs["Profile Curve"])
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(c2m.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = M_ROAD
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


def road_in_home(p):
    return (-60 < p[:, 0]) & (p[:, 0] < 60) & (-55 < p[:, 1]) & (p[:, 1] < 55)


ROADS_T = []      # traffic roads (main + residential)
for cls_set, w in ((city_osm.MAIN_ROADS, 15.0), (city_osm.RES_ROADS, 8.0), ({"service"}, 5.0)):
    pls = [np.column_stack([r["pts"], np.full(len(r["pts"]), 0.06 + 0.01 * (w > 10))]) for r in OSM_R if r["cls"] in cls_set]
    if not pls:
        continue
    ob = curve_obj(f"Roads{int(w)}", pls)
    add_gn(ob, ribbon_tree(w))
    if w > 6:
        ROADS_T += [(r, w) for r in OSM_R if r["cls"] in cls_set]
sl_pts = []
for r, w in ROADS_T:
    P_, Tn = city_osm.resample(r["pts"], 26.0)
    if len(P_) == 0:
        continue
    nrm_ = np.stack([-Tn[:, 1], Tn[:, 0]], 1)
    sgn = np.where(np.arange(len(P_)) % 2 == 0, 1.0, -1.0)[:, None]
    q = P_ + nrm_ * sgn * (w / 2 + 1.0)
    sl_pts += [(x, y, 7.5) for x, y in q]
# traffic along the real roads (Sample Curve on a curve object; Hong Kong drives on the LEFT)
RC = curve_obj("TrafficRoads", [np.column_stack([r["pts"], np.full(len(r["pts"]), 0.9)]) for r, w in ROADS_T])
RC.hide_render = True
tr = dict(ci=[], L=[], off=[], spd=[], dirn=[], lane=[])
for i, (r, w) in enumerate(ROADS_T):
    Lr = city_osm.length(r["pts"])
    dens = 1 / 45.0 if w > 10 else 1 / 90.0
    dirs = (1.0,) if r["oneway"] else (1.0, -1.0)
    for dn in dirs:
        for _ in range(max(1, int(Lr * dens))):
            tr["ci"].append(float(i)); tr["L"].append(Lr); tr["off"].append(rng.random()); tr["spd"].append(rng.uniform(7, 15))
            tr["dirn"].append(dn); tr["lane"].append((w / 4.0) if not r["oneway"] else rng.uniform(-w / 4, w / 4))


def traffic_tree(curve_ob=None, name="Traffic", scale=1.0):
    ng, nb, gi, go = gn_tree(name)
    T_ = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    ci, L, off, spd, dirn, lane = nb.attr("ci"), nb.attr("L"), nb.attr("off"), nb.attr("spd"), nb.attr("dirn"), nb.attr("lane")
    oi = nb.n("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    oi.inputs["Object"].default_value = curve_ob or RC
    f_ = nb.m("FRACT", nb.m("ADD", off, nb.m("DIVIDE", nb.m("MULTIPLY", spd, T_), L)))
    f_ = nb.mix(nb.m("GREATER_THAN", dirn, 0.0), nb.m("SUBTRACT", 1.0, f_), f_, dtype="FLOAT")
    sc_ = nb.n("GeometryNodeSampleCurve", mode="FACTOR", use_all_curves=False)
    nb.link(oi.outputs["Geometry"], sc_.inputs["Curves"])
    nb.link(f_, sc_.inputs["Factor"])
    fi = nb.n("FunctionNodeFloatToInt", rounding_mode="ROUND")
    nb.link(ci, fi.inputs[0])
    nb.link(fi.outputs[0], sc_.inputs["Curve Index"])
    fwd = nb.v("SCALE", nb.v("NORMALIZE", sc_.outputs["Tangent"]), dirn)
    left = nb.v("NORMALIZE", nb.v("CROSS_PRODUCT", (0.0, 0.0, 1.0), fwd))
    pos = nb.v("ADD", sc_.outputs["Position"], nb.v("SCALE", left, lane))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(gi.outputs[0], sp.inputs["Geometry"])
    nb.link(pos, sp.inputs["Position"])
    # template: two white headlamps at +Y, two red tail lamps at -Y (aligned so +Y = direction of travel)
    tmpl_pts = [(-0.75, 2.2, 0), (0.75, 2.2, 0), (-0.7, -2.2, 0.1), (0.7, -2.2, 0.1)]
    pts_ = nb.n("GeometryNodePoints")
    nb.set(pts_.inputs["Count"], 4)
    idx = nb.n("GeometryNodeInputIndex").outputs[0]
    sel_front = nb.cmp("LESS_THAN", idx, 1.5)
    px_ = nb.mix(nb.m("MODULO", idx, 2.0), -0.75 * scale, 0.75 * scale, dtype="FLOAT")
    py_ = nb.mix(sel_front, -2.2 * scale, 2.2 * scale, dtype="FLOAT")
    nb.link(nb.xyz(px_, py_, 0.0), pts_.inputs["Position"])
    tmpl = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(pts_.outputs[0], tmpl.inputs["Points"])
    ico = nb.n("GeometryNodeMeshIcoSphere")
    nb.set(ico.inputs["Radius"], 0.28 * scale)
    nb.set(ico.inputs["Subdivisions"], 1)
    nb.link(ico.outputs["Mesh"], tmpl.inputs["Instance"])
    tmpl_r = nb.n("GeometryNodeRealizeInstances")
    nb.link(tmpl.outputs[0], tmpl_r.inputs["Geometry"])
    _, ty_, _ = nb.sep(nb.n("GeometryNodeInputPosition").outputs[0])
    tmpl_s = nb.store(tmpl_r.outputs[0], "front", nb.m("GREATER_THAN", ty_, 0.0))
    al = nb.n("FunctionNodeAlignRotationToVector", axis="Y")
    nb.link(fwd, al.inputs["Vector"])
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(sp.outputs[0], inst.inputs["Points"])
    nb.link(tmpl_s, inst.inputs["Instance"])
    nb.link(al.outputs[0], inst.inputs["Rotation"])
    rl = nb.n("GeometryNodeRealizeInstances")
    nb.link(inst.outputs[0], rl.inputs["Geometry"])
    m, mb, out = new_mat(f"{name}Lights")
    cg = ctrl(mb)
    fr_ = mb.sattr("front").outputs["Fac"]
    e = mb.n("ShaderNodeEmission")
    mb.set(e.inputs["Color"], mb.mix(fr_, (1.0, 0.04, 0.02), (1.0, 0.9, 0.75)))
    mb.set(e.inputs["Strength"], mb.m("MULTIPLY", mb.m("SUBTRACT", 1.0, cg["shut"]), mb.mix(fr_, 3.5, 12.0, dtype="FLOAT")))
    mb.link(e.outputs[0], out.inputs["Surface"])
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(rl.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = m
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


TRAF = points_obj("Traffic", np.zeros((len(tr["ci"]), 3)), {k: np.asarray(v) for k, v in tr.items()}, coll=C_CITY)
add_gn(TRAF, traffic_tree())
print(f"[roads] {len(ROADS_T)} traffic roads, {len(tr['ci'])} cars, {len(sl_pts)} street lamps")



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
_em_done = set()


def keyed_existing_emission(mt, offset=0.02):
    """an imported PBR material keeps its look, but its emission follows the city level + outro shutoff"""
    if mt is None or mt.node_tree is None or mt.name in _em_done:
        return
    _em_done.add(mt.name)
    nb_ = NB(mt.node_tree)
    for bs in [n for n in mt.node_tree.nodes if n.type == "BSDF_PRINCIPLED"]:
        sk = bs.inputs["Emission Strength"]
        if sk.is_linked or sk.default_value <= 0.0:
            continue
        cg_ = ctrl(nb_)
        lv = nb_.m("MULTIPLY", nb_.m("SUBTRACT", 1.0, cg_["shut"]), nb_.m("ADD", offset, nb_.m("MULTIPLY", cg_["city"], 1 - offset)))
        nb_.link(nb_.m("MULTIPLY", lv, sk.default_value * 0.7), sk)


for i, (slug, hx, hy, hh, yaw) in enumerate(HEROES):
    meshes = import_glb(slug)
    lo, hi = model_bbox(meshes)
    s = hh / (hi[2] - lo[2])
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    artemis = slug.startswith("sf_skyscraper")
    for nm, me0 in meshes:
        imgs = any(n.type == "TEX_IMAGE" for mt in me0.materials if mt and mt.node_tree for n in mt.node_tree.nodes)
        if artemis and imgs:
            continue                                   # baked sign/decal meshes (the "2023" one) are dropped
        me = me0.copy()
        if artemis:
            for si, mt in enumerate(me.materials):
                bs = next((n for n in mt.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mt and mt.node_tree else None
                is_em = bs is not None and bs.inputs["Emission Strength"].default_value > 0.5 and \
                    max(bs.inputs["Emission Color"].default_value[:3]) > 0.5 and not bs.inputs["Emission Color"].is_linked
                me.materials[si] = accent_from(mt, f"HeroAccent_{i}_{si}") if is_em else M_FACADE
            set_face_attrs(me, bid=float(rng.random()), soff=0.86 + 0.005 * i, kind=0.0, led=float(i % 2), hue=float(rng.choice([0.5, 0.83])))
        else:
            for mt in me.materials:
                keyed_existing_emission(mt)
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


cands = [t_ for t_ in TOPS if t_[4] > 45 and 150 < (t_[1] + t_[3]) / 2 < 900 and abs((t_[0] + t_[2]) / 2) < 520]
rng.shuffle(cands)
BBS = []


def add_billboard(p0, tdir, nrm, w, h, zc):
    """flat ad panel on a facade: p0 = edge midpoint, tdir = edge direction (CCW), nrm = outward normal"""
    tdir, nrm = Vector((*tdir, 0.0)), Vector((*nrm, 0.0))
    c = Vector((*p0, zc)) + nrm * 0.3
    vs = [c - tdir * w / 2 - Vector((0, 0, h / 2)), c + tdir * w / 2 - Vector((0, 0, h / 2)),
          c + tdir * w / 2 + Vector((0, 0, h / 2)), c - tdir * w / 2 + Vector((0, 0, h / 2))]
    ob = mesh_obj(f"BB{len(BBS)}", [v[:] for v in vs], [(0, 1, 2, 3)], coll=C_CITY)
    ob.data.uv_layers.new(name="UVMap").data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    ob.data.materials.append(billboard_material(len(BBS)))
    BBS.append(ob)


for tp in cands:
    if len(BBS) >= 22:
        break
    a0, b0, a1, b1, zt, ztb = tp[:6]
    if len(tp) > 10:                                   # real footprint: the edge that faces the home roof best
        poly = tp[10]
        best = None
        for k_ in range(len(poly)):
            pa, pb = poly[k_], poly[(k_ + 1) % len(poly)]
            e_ = pb - pa
            L_ = float(np.linalg.norm(e_))
            if L_ < 14:
                continue
            td = e_ / L_
            nr = np.array([td[1], -td[0]])
            mid = (pa + pb) / 2
            to_home = -mid / max(1e-6, np.linalg.norm(mid))
            sc_ = float(nr @ to_home)
            if best is None or sc_ > best[0]:
                best = (sc_, mid, td, nr, L_)
        if best is None or best[0] < 0.6:
            continue
        _, mid, td, nr, L_ = best
        w = min(L_ * 0.75, 38.0)
        h = w * 0.55
        top_z = zt * (tp[5] / zt if tp[5] > 0 else 1.0)
        if top_z - h < 14:
            continue
        add_billboard(mid, td, nr, w, h, rng.uniform(max(h / 2 + 8, top_z * 0.45), top_z - h / 2 - 2))
    else:
        w = (a1 - a0) * 0.8
        h = min(w * 0.55, (zt - ztb) * 0.6)
        if h < 8.0:
            continue
        zc = rng.uniform(ztb + h / 2 + 2.0, max(ztb + h / 2 + 2.0, zt - h / 2 - 2.0))
        add_billboard(((a0 + a1) / 2, b0), (1.0, 0.0), (0.0, -1.0), w, h, zc)
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
# 11. NO falling rain (v2, Zeke): the rain stopped ~an hour ago — wet everything, a little steam, lingering haze
# ---------------------------------------------------------------------------------------------------------
CAM = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
C_CAM.objects.link(CAM)
SC.camera = CAM


def steam_material(name, seed):
    """rising vent steam: 4D noise advected upward, shaped to a widening plume, lit by whatever is near (neon)"""
    m, nb, out = new_mat(name)
    cg = ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    lx_, ly_, lz_ = nb.sep(tc.outputs["Object"])                 # plume object: unit cylinder, z 0..1
    rise = nb.xyz(lx_, ly_, nb.m("SUBTRACT", lz_, nb.m("MULTIPLY", cg["time"], 0.22)))
    n1 = nb.n("ShaderNodeTexNoise", noise_dimensions="4D")
    nb.link(rise, n1.inputs["Vector"])
    nb.link(nb.m("ADD", nb.m("MULTIPLY", cg["time"], 0.15), seed), n1.inputs["W"])
    nb.set(n1.inputs["Scale"], 2.0)
    nb.set(n1.inputs["Detail"], 6.0)
    r = nb.v("LENGTH", nb.xyz(lx_, ly_, 0.0))
    width = nb.m("ADD", 0.25, nb.m("MULTIPLY", lz_, 0.75))
    radial = nb.m("SUBTRACT", 1.0, nb.mr(nb.m("DIVIDE", r, width), 0.2, 1.0))
    vert = nb.m("MULTIPLY", nb.mr(lz_, 0.0, 0.08), nb.m("SUBTRACT", 1.0, nb.mr(lz_, 0.45, 1.0)))
    dens = nb.m("MULTIPLY", nb.m("MULTIPLY", radial, vert), nb.mr(n1.outputs["Fac"], 0.35, 0.62))
    pv = nb.n("ShaderNodeVolumePrincipled")
    nb.set(pv.inputs["Color"], (0.85, 0.86, 0.9))
    nb.link(nb.m("MULTIPLY", dens, 10.0), pv.inputs["Density"])
    nb.set(pv.inputs["Anisotropy"], 0.35)
    nb.set(pv.inputs["Emission Color"], (0.75, 0.62, 0.9))                 # picks up the city/neon glow around it
    nb.link(nb.m("MULTIPLY", dens, 2.2), pv.inputs["Emission Strength"])
    nb.link(pv.outputs[0], out.inputs["Volume"])
    return m


STEAM = []
for k, (x, y, z, r, h) in enumerate([(-10.5, 12.5, ROOFZ + 0.9, 0.9, 7.0), (10.5, 13.5, ROOFZ + 0.9, 0.7, 5.5),
                                     (-44.0, 30.0, W1[4] + 0.5, 1.6, 12.0), (44.0, 40.0, N1[4] + 0.5, 1.8, 14.0),
                                     (-34.0, 62.0, 0.2, 2.2, 16.0), (52.0, -60.0, 0.2, 2.0, 15.0)]):
    st = cylinder(f"Steam{k}", 1.0, 1.0, (x, y, z), steam_material(f"SteamMat{k}", float(k * 3.1)), segs=12, coll_=C_FX)
    st.scale = (r * 2.2, r * 2.2, h)
    STEAM.append(st)
# the vent pipes the roof steam comes out of + a little light that makes the plume read (a vent lamp / spill)
for k, (x, y, col) in enumerate([(-10.5, 12.5, (1.0, 0.72, 0.45)), (10.5, 13.5, (0.75, 0.6, 1.0))]):
    cylinder("VentPipe", 0.18, 0.9, (x, y, ROOFZ), M_STEEL, segs=10)
    vl_ = bpy.data.objects.new(f"VentLight{k}", bpy.data.lights.new(f"VentLight{k}", "POINT"))
    vl_.data.energy, vl_.data.color, vl_.data.shadow_soft_size, vl_.data.use_shadow = 700.0, col, 0.3, False
    vl_.data.specular_factor = 0.0                         # lights the plume + roof, no hot blob in the puddles
    vl_.location = (x + 0.6, y - 0.6, ROOFZ + 1.3)
    C_ROOF.objects.link(vl_)
    bake(vl_.data, "energy", 700.0 * (1.0 - smoothstep((T - (tEND + 1.6)) / 0.3)))


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
# v2 shells after real anatomy (research 10-13): 5–8" shells burst 150–280 m wide; stars decelerate hard (drag),
# then droop under gravity; willows burn 6–8 s; crossette stars split into a 4-way cross; strobes blink;
# chemistry palette — Sr red, Ba green, Na gold, Ca orange, Sr+Cu purple, Cu blue (rare + dim), Ti/Mg silver.
RED, GREEN, GOLD, SILVER = (1.0, 0.07, 0.03), (0.2, 1.0, 0.28), (1.0, 0.58, 0.15), (0.92, 0.94, 1.0)
ORANGE, PURPLE, BLUE = (1.0, 0.32, 0.05), (0.62, 0.18, 1.0), (0.25, 0.42, 1.0)
PAL = [RED, GREEN, GOLD, SILVER, ORANGE, PURPLE, RED, GREEN, GOLD, BLUE]
rockets = dict(L=[], A=[], tl=[], ta=[], dud=[], col=[], thick=[])
sparks = dict(c=[], v=[], t0=[], life=[], col=[], col2=[], tc=[], k=[], g=[], tr=[], tw=[], sb=[], sz=[], br=[])
BURSTS = []          # (t0, centre, colour, radius) → flash lights + smoke puffs


def burst_center_home():
    # dip + finale are seen from the home roof: shells over the near skyline, big in frame
    return Vector((rng.uniform(-300, 340), rng.uniform(260, 620), rng.uniform(190, 340)))


def burst_center():
    # ring around the tallest tower, biased NE (the bridge camera's view) but spread so the home roof sees them
    for _ in range(100):
        ang = rng.uniform(-0.35, 1.9)
        r = rng.uniform(190, 680)
        c = Vector((TCEN.x + r * math.cos(ang), TCEN.y + r * math.sin(ang), rng.uniform(330, 440)))
        if -700 < c.x < 950 and 250 < c.y < 1350:
            return c
    return Vector((TCEN.x + 200, TCEN.y + 200, 400))


def add_rocket(apex, ta, dud=False, flight=None, thick=0.0):
    flight = flight or rng.uniform(2.6, 3.7)
    L = Vector((apex.x + rng.uniform(-25, 25), apex.y + rng.uniform(-25, 25), rng.uniform(20, 90)))
    rockets["L"].append(L[:]); rockets["A"].append(apex[:]); rockets["tl"].append(ta - flight); rockets["ta"].append(ta)
    rockets["dud"].append(1.0 if dud else 0.0); rockets["col"].append((1.0, 0.62, 0.3)); rockets["thick"].append(thick)


def add_burst(c, t0, kind=None, col=None, size=None):
    kind = kind or rng.choice(["peony", "peony", "chrys", "chrys", "willow", "ring", "palm", "crossette", "strobe",
                               "brocade", "peony_cc"])
    col = col or PAL[rng.integers(len(PAL))]
    R = size or rng.choice([45.0, 70.0, 70.0, 95.0, 120.0])         # burst radius (m): ~3" … 8" shells
    def push(d, R_, life, k=1.1, g=7.0, tr=0.12, tw=0.0, sb=0.0, sz=0.45, cc=None, c2=None, tc=9.0, br=1.0, at=None, tt=None):
        d = np.asarray(d, float)
        d = d / max(1e-6, np.linalg.norm(d))
        sparks["c"].append((at if at is not None else c)[:]); sparks["v"].append(tuple(d * R_ * k))
        sparks["t0"].append(tt if tt is not None else t0); sparks["life"].append(life)
        cc = cc or col
        sparks["col"].append(cc); sparks["col2"].append(c2 or cc); sparks["tc"].append(tc)
        sparks["k"].append(k); sparks["g"].append(g); sparks["tr"].append(tr); sparks["tw"].append(tw)
        sparks["sb"].append(sb); sparks["sz"].append(sz); sparks["br"].append(br * (0.6 if cc == BLUE else 1.0))
    jit = lambda n_: fib_sphere(n_, 1.0) + rng.normal(0, 0.035, (n_, 3))
    if kind in ("peony", "peony_cc"):
        c2 = PAL[rng.integers(len(PAL))] if kind == "peony_cc" else None
        for d in jit(240):
            push(d, R * rng.uniform(0.92, 1.05), rng.uniform(2.3, 3.0), c2=c2, tc=rng.uniform(0.45, 0.6) if c2 else 9.0)
        if rng.random() < 0.5:                                      # pistil: a smaller inner sphere, other colour
            ci = PAL[rng.integers(len(PAL))]
            for d in jit(90):
                push(d, R * 0.42, 2.0, cc=ci)
    elif kind == "chrys":
        for d in jit(260):
            push(d, R * rng.uniform(0.92, 1.05), rng.uniform(2.8, 3.4), k=1.0, tr=0.34, tw=1.0 if rng.random() < 0.5 else 0.0)
    elif kind == "willow":
        for d in jit(220):
            push(d, R * 0.85, rng.uniform(6.0, 7.5), k=0.7, g=9.5, tr=0.75, tw=1.0, sz=0.38, cc=GOLD)
    elif kind == "brocade":
        for d in jit(200):
            push(d, R * 0.95, rng.uniform(3.6, 4.4), k=0.85, g=8.5, tr=0.6, tw=1.0, sz=0.36, cc=SILVER if rng.random() < 0.5 else GOLD)
    elif kind == "strobe":
        for d in jit(260):
            push(d, R * rng.uniform(0.9, 1.05), rng.uniform(2.8, 3.4), k=1.0, tr=0.04, sb=1.0, cc=SILVER, br=1.4)
    elif kind == "ring":
        ax = Vector(rng.normal(0, 1, 3)).normalized()
        u = ax.orthogonal().normalized()
        w = ax.cross(u)
        for a in np.linspace(0, 2 * np.pi, 150, endpoint=False):
            push(np.array((math.cos(a) * u + math.sin(a) * w)[:]), R, 2.4, tr=0.14)
    elif kind == "palm":
        for _ in range(8):
            arm = Vector(rng.normal(0, 1, 3)).normalized()
            arm.z = abs(arm.z) * 0.6 + 0.25
            for _j in range(12):
                push(np.array(arm.normalized()[:]) + rng.normal(0, 0.035, 3), R * 0.9, 3.4, k=0.8, g=8.0, tr=0.7, sz=0.8,
                     cc=GOLD, tw=1.0)
    elif kind == "crossette":
        k_, life_p = 0.9, 0.85
        for d in jit(18):
            d = d / np.linalg.norm(d)
            v0 = d * R * 0.75 * k_
            push(d, R * 0.75, life_p, k=k_, tr=0.3, sz=0.6, cc=col)
            e_ = math.exp(-k_ * life_p)                                    # analytic split point + velocity
            ps = np.array(c[:]) + v0 * (1 - e_) / k_ - np.array([0, 0, 7.0 / k_ * (life_p - (1 - e_) / k_)])
            vs = v0 * e_ - np.array([0, 0, 7.0 / k_ * (1 - e_)])
            vn = vs / max(1e-6, np.linalg.norm(vs))
            u = np.cross(vn, [0, 0, 1.0]); u /= max(1e-6, np.linalg.norm(u)); w = np.cross(vn, u)
            for dd in (u, -u, w, -w):                                      # the 4-way cross split
                dvec = vs + dd * 30.0
                spd = np.linalg.norm(dvec)
                push(dvec / spd, spd / 1.3, 1.5, k=1.3, tr=0.2, cc=col, at=Vector(ps), tt=t0 + life_p)
    BURSTS.append((t0, Vector(c[:]), col, R))


# bridge tracers (duds): density builds over the 8 bars — they climb and fade, they do NOT burst
for b in range(B_BRIDGE + 2, B_LAST - 8):
    n = 1 if b < B_BRIDGE + 14 else (2 if b < B_BRIDGE + 24 else 3)
    if b < B_BRIDGE + 14 and b % 2:
        continue
    for _ in range(n):
        c = burst_center()
        add_rocket(c, bt(b) + rng.uniform(2.8, 3.6), dud=True, flight=rng.uniform(2.8, 3.6))
# the volley: every shell reaches apex at the LAST DROP → all burst at once
tLAST = bt(B_LAST)
for _ in range(30):
    c = burst_center()
    kind = rng.choice(["peony", "chrys", "peony_cc", "crossette", "palm", "strobe", "brocade"])
    add_rocket(c, tLAST, flight=rng.uniform(2.6, 3.7), thick=1.0 if kind == "palm" else 0.0)
    add_burst(c, tLAST, kind)
for b in range(B_LAST + 1, B_DIP):
    for _ in range(int(rng.integers(2, 4))):
        c = burst_center()
        t0 = bt(b) + rng.choice([0.0, 0.0, P / 2])
        add_rocket(c, t0)
        add_burst(c, t0)
for b in range(B_DIP, B_FINAL, 4):
    c = burst_center_home()
    add_rocket(c, bt(b))
    add_burst(c, bt(b), "willow")
for b in range(B_FINAL, B_END - 4):
    for _ in range(int(rng.integers(2, 5))):
        c = burst_center_home()
        t0 = bt(b) + rng.choice([0.0, 0.0, P / 2])
        add_rocket(c, t0)
        add_burst(c, t0)
for _ in range(12):           # finale canopy: gold willows on the last downbeat of the music
    c = burst_center_home()
    add_rocket(c, bt(B_END - 4))
    add_burst(c, bt(B_END - 4), "willow", size=110.0)
print(f"[fireworks] rockets {len(rockets['tl'])}, bursts {len(BURSTS)}, sparks {len(sparks['t0'])}, earliest launch {min(rockets['tl']):.2f}s")
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
    hot = nb.mix(nb.m("POWER", nb.m("ADD", tz, 1.0), 12.0), fc, (1.0, 0.95, 0.85))      # white-hot star head
    e = nb.n("ShaderNodeEmission")
    nb.link(hot, e.inputs["Color"])
    nb.link(nb.m("MULTIPLY", nb.m("MULTIPLY", fb, tail), 80.0), e.inputs["Strength"])
    nb.link(e.outputs[0], out.inputs["Surface"])
    return m


M_FW = fw_material()


def sparks_tree():
    ng, nb, gi, go = gn_tree("FireworkSparks")
    T_ = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    c, v = nb.attr("c", "FLOAT_VECTOR"), nb.attr("v", "FLOAT_VECTOR")
    t0, life, k, g = nb.attr("t0"), nb.attr("life"), nb.attr("k"), nb.attr("g")
    trl, tw, sz, col = nb.attr("tr"), nb.attr("tw"), nb.attr("sz"), nb.attr("col", "FLOAT_VECTOR")
    col2, tcf, sb, br = nb.attr("col2", "FLOAT_VECTOR"), nb.attr("tc"), nb.attr("sb"), nb.attr("br")
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
    length = nb.m("MAXIMUM", nb.m("MULTIPLY", speed, trl), 0.6)
    u = nb.m("DIVIDE", age, life)
    fade = nb.m("POWER", nb.m("SUBTRACT", 1.0, u), 0.8)
    flash = nb.m("ADD", 1.0, nb.m("MULTIPLY", 2.5, nb.m("EXPONENT", nb.m("MULTIPLY", age, -14.0))))
    idx = nb.n("GeometryNodeInputIndex").outputs[0]
    wn = nb.n("ShaderNodeTexWhiteNoise", noise_dimensions="2D")
    nb.link(nb.xyz(idx, nb.m("FLOOR", nb.m("MULTIPLY", T_, 22.0)), 0), wn.inputs["Vector"])
    twk = nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", tw, nb.m("GREATER_THAN", wn.outputs["Value"], 0.55)))
    # burn-out: the last 15 % of a star's life sputters
    sput = nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", nb.m("GREATER_THAN", u, 0.85), nb.m("GREATER_THAN", wn.outputs["Value"], 0.4)))
    # strobe stars blink ~9 Hz with their own phase
    ph = nb.m("FRACT", nb.m("ADD", nb.m("MULTIPLY", T_, 9.0), nb.m("MULTIPLY", idx, 0.6180339)))
    blink = nb.m("ADD", nb.m("SUBTRACT", 1.0, sb), nb.m("MULTIPLY", sb, nb.m("MULTIPLY", nb.m("GREATER_THAN", ph, 0.55), 1.8)))
    fb = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", fade, flash), nb.m("MULTIPLY", twk, sput)), nb.m("MULTIPLY", blink, br))
    colnow = nb.mix(nb.m("GREATER_THAN", u, tcf), col, col2, dtype="VECTOR")     # colour-change stars
    geo = streak_instancer(nb, sp.outputs[0], vel, length, sz, {"fc": colnow, "fb": fb})
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(geo, sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = M_FW
    nb.link(sm.outputs[0], go.inputs[0])
    return ng


def rockets_tree():
    ng, nb, gi, go = gn_tree("FireworkRockets")
    T_ = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    L, Ap = nb.attr("L", "FLOAT_VECTOR"), nb.attr("A", "FLOAT_VECTOR")
    tl, ta, dud, col, thick = nb.attr("tl"), nb.attr("ta"), nb.attr("dud"), nb.attr("col", "FLOAT_VECTOR"), nb.attr("thick")
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
    length = nb.m("MINIMUM", nb.m("MAXIMUM", nb.m("MULTIPLY", nb.v("LENGTH", vel), 0.45), 3.0), 42.0)
    duds = nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", dud, nb.mr(u, 0.72, 1.0)))
    fb = nb.m("MULTIPLY", duds, nb.mr(u, 0.0, 0.04, 0.3, 1.0))
    geo = streak_instancer(nb, sp.outputs[0], vel, length, nb.m("ADD", 0.45, nb.m("MULTIPLY", thick, 0.7)), {"fc": col, "fb": fb})
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
BURSTS.sort(key=lambda b_: b_[0])

# burst flash lights: light spill on facades, roofs and puddles. specular_factor kept LOW — v1's full-strength
# specular put a hot highlight on every wet roof (the "pink pillow" sheen); real spill reads as diffuse glow.
NFL = 10
flash_e = np.zeros((NFL, NF))
flash_loc = np.zeros((NFL, NF, 3))
flash_col = np.ones((NFL, NF, 3))
for i, (t0, c, col, R) in enumerate(BURSTS):
    j = i % NFL
    m = (T >= t0) & (T < t0 + 1.8)
    flash_e[j, m] = np.maximum(flash_e[j, m], (R / 90.0) * np.exp(-(T[m] - t0) / 0.4))
    flash_loc[j, T >= t0] = c[:]
    flash_col[j, T >= t0] = col
for j in range(NFL):
    Lf = bpy.data.objects.new(f"BurstFlash{j}", bpy.data.lights.new(f"BurstFlash{j}", "POINT"))
    Lf.data.shadow_soft_size = 45.0
    Lf.data.use_shadow = False
    Lf.data.specular_factor = 0.12
    Lf.data.volume_factor = 0.25
    C_FX.objects.link(Lf)
    first = np.argmax(flash_loc[j, :, 2] > 0)
    flash_loc[j, :first] = flash_loc[j, first]
    flash_col[j, :first] = flash_col[j, first]
    bake_vec(Lf, "location", flash_loc[j], interp="CONSTANT")
    bake_vec(Lf.data, "color", flash_col[j], interp="CONSTANT")
    bake(Lf.data, "energy", flash_e[j] * 0.9e7)

# smoke left behind by each shell: a slowly growing, drifting puff, lit from within at the burst, then by later
# flashes (volumes scatter the flash lights). Pooled objects keep the scene light.
NSM = 22
sm_loc = np.zeros((NSM, NF, 3)); sm_scl = np.full((NSM, NF), 0.001); sm_den = np.zeros((NSM, NF))
sm_em = np.zeros((NSM, NF)); sm_col = np.ones((NSM, NF, 3))
for i, (t0, c, col, R) in enumerate(BURSTS):
    j = i % NSM
    m = T >= t0
    age = T[m] - t0
    sm_loc[j, m] = np.array(c[:])[None, :] + np.stack([age * 1.6, age * 0.6, age * 0.8], 1)
    sm_scl[j, m] = R * (0.45 + 0.55 * (1 - np.exp(-age / 1.2))) + 2.5 * age
    sm_den[j, m] = 0.006 * np.exp(-age / 9.0) * (1 - np.exp(-age / 0.35))
    sm_em[j, m] = 0.3 * np.exp(-age / 0.5)
    sm_col[j, m] = col
SMOKE = []
for j in range(NSM):
    m, nb, out = new_mat(f"ShellSmoke{j}")
    den, em = nb.val("den", 0.0), nb.val("em", 0.0)
    cr_, cg_, cb2 = nb.val("cr", 1.0), nb.val("cg", 1.0), nb.val("cb", 1.0)
    tc = nb.n("ShaderNodeTexCoord")
    r = nb.v("LENGTH", tc.outputs["Object"])
    nz_ = nb.n("ShaderNodeTexNoise")
    nb.link(tc.outputs["Object"], nz_.inputs["Vector"])
    nb.set(nz_.inputs["Scale"], 2.2)
    nb.set(nz_.inputs["Detail"], 5.0)
    shape = nb.m("MULTIPLY", nb.m("SUBTRACT", 1.0, nb.mr(r, 0.35, 1.0)), nb.mr(nz_.outputs["Fac"], 0.4, 0.7))
    pv = nb.n("ShaderNodeVolumePrincipled")
    nb.set(pv.inputs["Color"], (0.5, 0.5, 0.52))
    nb.link(nb.m("MULTIPLY", shape, den), pv.inputs["Density"])
    nb.link(nb.xyz(cr_, cg_, cb2), pv.inputs["Emission Color"])
    nb.link(nb.m("MULTIPLY", shape, em), pv.inputs["Emission Strength"])
    nb.link(pv.outputs[0], out.inputs["Volume"])
    import bmesh
    me = bpy.data.meshes.new(f"ShellSmoke{j}")
    bm = bmesh.new()
    try:
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.materials.append(m)
    ob = bpy.data.objects.new(f"ShellSmoke{j}", me)
    C_FX.objects.link(ob)
    bake_vec(ob, "location", sm_loc[j])
    bake_vec(ob, "scale", np.repeat(sm_scl[j][:, None], 3, 1))
    for nm_, arr in (("den", sm_den[j]), ("em", sm_em[j]), ("cr", sm_col[j, :, 0]), ("cg", sm_col[j, :, 1]), ("cb", sm_col[j, :, 2])):
        bake(m.node_tree, f'nodes["{nm_}"].outputs[0].default_value', arr)
    bake(ob, "hide_render", (sm_den[j] < 1e-5).astype(float), interp="CONSTANT")
    SMOKE.append(ob)


# ---------------------------------------------------------------------------------------------------------
# 14. flying cars — no wheels: thrusters/underglow carry the read. Aviation nav lights (research: FAA AC 43-217):
#     RED on the LEFT (port), GREEN on the RIGHT (starboard), WHITE aft + an anti-collision strobe ~60/min.
#     Every model's NOSE axis was MEASURED from end renders on 10-13 and is mapped to rig +Y; every rig's yaw is
#     derived from its velocity, then nose·velocity is ASSERTED numerically (and +Y markers render with --markers).
#       sf_hover_car     nose = local -X  (hood/emblem/headlights; +X = spoiler + cyan tail panel)
#       sf_car_vertascan nose = local -Y  (windshield + dashboard face -Y; engine vents at +Y)
#       sf_car_dylanheyes nose = local +Y (white light bars at +Y; red backlights + cyan engines at -Y)
# ---------------------------------------------------------------------------------------------------------
NOSE = {"sf_hover_car": "-X", "sf_car_vertascan": "-Y", "sf_car_dylanheyes": "+Y"}
NOSE_ROT = {"-X": -90.0, "-Y": 180.0, "+Y": 0.0}               # rotation about Z taking the nose axis onto +Y
NOSE_VEC = {"-X": (-1, 0, 0), "-Y": (0, -1, 0), "+Y": (0, 1, 0)}
navstrobe = ((T % 1.0) < 0.07).astype(float)
CTRL.interface.new_socket("navstrobe", in_out="OUTPUT", socket_type="NodeSocketFloat")
_ns = _cn.val("navstrobe", 0.0)
_cn.link(_ns, _cgo.inputs["navstrobe"])
bake(CTRL, 'nodes["navstrobe"].outputs[0].default_value', navstrobe, interp="CONSTANT")
M_NAV_R = emit_mat("NavRed", (1.0, 0.02, 0.01), 40.0)
M_NAV_G = emit_mat("NavGreen", (0.05, 1.0, 0.15), 40.0)
_sm, _sb, _so = new_mat("NavStrobe")
_scg = ctrl(_sb)
_se = _sb.n("ShaderNodeEmission")
_sb.set(_se.inputs["Color"], (1.0, 1.0, 1.0))
_sb.link(_sb.m("ADD", 2.0, _sb.m("MULTIPLY", _scg["navstrobe"], 160.0)), _se.inputs["Strength"])
_sb.link(_se.outputs[0], _so.inputs["Surface"])
M_NAV_W = _sm
_car_boosted = set()


def ball(name, r, loc, mat, parent):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    try:
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    C_FX.objects.link(o)
    o.parent = parent
    o.location = loc
    return o


def make_car(slug, name, length, glow=(0.2, 0.7, 1.0), headlights=False):
    meshes = import_glb(slug)
    lo, hi = model_bbox(meshes)
    ax = NOSE[slug]
    long_i = 0 if "X" in ax else 1
    cs = length / (hi[long_i] - lo[long_i])
    rig = bpy.data.objects.new(f"{name}Rig", None)
    C_FX.objects.link(rig)
    model = bpy.data.objects.new(f"{name}Model", None)
    C_FX.objects.link(model)
    model.parent = rig
    model.matrix_basis = Matrix.Rotation(deg(NOSE_ROT[ax]), 4, "Z") @ Matrix.Diagonal((cs, cs, cs, 1.0)) @ \
        Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -(lo[2] + hi[2]) / 2))
    for nm, me in meshes:
        o = bpy.data.objects.new(f"{name}_{nm}", me)
        C_FX.objects.link(o)
        o.parent = model
        for mt in me.materials:
            if mt is None or mt.name in _car_boosted or not mt.node_tree:
                continue
            _car_boosted.add(mt.name)
            bs = next((n for n in mt.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bs and (bs.inputs["Emission Color"].is_linked or bs.inputs["Emission Strength"].default_value > 0.5):
                bs.inputs["Emission Strength"].default_value = max(6.0, bs.inputs["Emission Strength"].default_value)
            if bs and not bs.inputs["Roughness"].is_linked:
                bs.inputs["Roughness"].default_value = min(bs.inputs["Roughness"].default_value, 0.3)
    c_ = (lo + hi) / 2
    nose_l = Vector(c_)
    tail_l = Vector(c_)
    nose_l[long_i] = lo[long_i] if ax[0] == "-" else hi[long_i]
    tail_l[long_i] = hi[long_i] if ax[0] == "-" else lo[long_i]
    w = (hi[1 - long_i] - lo[1 - long_i]) * cs
    hgt = (hi[2] - lo[2]) * cs
    ball(f"{name}NavL", 0.09, (-w / 2, 0.0, 0.0), M_NAV_R, rig)            # port = LEFT = rig -X
    ball(f"{name}NavR", 0.09, (w / 2, 0.0, 0.0), M_NAV_G, rig)             # starboard = RIGHT = rig +X
    ball(f"{name}Strobe", 0.08, (0.0, -length / 2, hgt * 0.35), M_NAV_W, rig)
    ug = bpy.data.objects.new(f"{name}Under", bpy.data.lights.new(f"{name}Under", "POINT"))
    ug.data.energy, ug.data.color, ug.data.shadow_soft_size = 1600.0, glow, 1.2
    ug.data.specular_factor = 0.4
    C_FX.objects.link(ug)
    ug.parent = rig
    ug.location = (0, 0, -hgt * 0.6)
    if headlights:
        for side in (-0.3, 0.3):
            hl = bpy.data.objects.new(f"{name}Head", bpy.data.lights.new(f"{name}Head", "SPOT"))
            hl.data.energy, hl.data.color, hl.data.spot_size, hl.data.spot_blend = 2500.0, (0.85, 0.92, 1.0), deg(38), 0.6
            C_FX.objects.link(hl)
            hl.parent = rig
            hl.location = (side * w, length / 2 + 0.1, 0.0)
            hl.rotation_euler = (deg(82), 0, 0)        # R_x(+82°)·(0,0,-1) = (0,+0.99,-0.14) → forward, slightly down
    if A.markers:
        mk = boxes_mesh(f"{name}MarkPlusY", [(-0.4, length / 2 + 2.0, -0.4, 0.4, length / 2 + 2.8, 0.4)], coll=C_FX)
        mk.parent = rig
        mk.data.materials.append(emit_mat(f"{name}MarkerGreen", (0.0, 1.0, 0.0), 30.0))
    return dict(rig=rig, model=model, nose=nose_l, tail=tail_l, length=length)


def drive(car, pos, vis, label):
    """yaw from velocity (+Y forward), bank into turns (left turn → left side down), pitch from climb; then check"""
    vel = np.gradient(pos, axis=0) * FPS
    yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]) - np.pi / 2)
    rate = np.gradient(yaw) * FPS
    spd = np.linalg.norm(vel[:, :2], axis=1)
    roll = np.clip(-np.arctan(rate * spd / 9.81) * 0.8, -0.5, 0.5)
    pitch = np.clip(np.arctan2(vel[:, 2], np.maximum(spd, 1e-3)) * 0.6, -0.3, 0.3)
    bake_vec(car["rig"], "location", pos)
    bake_vec(car["rig"], "rotation_euler", np.stack([pitch, roll, yaw], 1))
    for o in [car["rig"]] + list(car["rig"].children_recursive):
        bake(o, "hide_render", 1.0 - vis, interp="CONSTANT")
    on = np.nonzero(vis > 0.5)[0]
    worst = 1.0
    for i in on[np.linspace(0, len(on) - 1, 5).astype(int)] if len(on) else []:
        SC.frame_set(int(i) + 1)
        mw = car["model"].matrix_world
        fwd = (mw @ car["nose"]) - (mw @ car["tail"])
        dot = fwd.normalized().dot(Vector(vel[i]).normalized())
        worst = min(worst, dot)
    print(f"[dircheck] {label}: worst nose·velocity over 5 frames = {worst:+.3f}")
    assert worst > 0.9, f"{label} is not facing its velocity"
    car["pos"], car["vis"] = pos, vis


# the breakdown car (hero of the 0:44 section): enters left ~46 s, drifts right through the rain-slick air
BRK = make_car("sf_hover_car", "Car", 5.4, glow=(0.2, 0.7, 1.0), headlights=True)
fill = bpy.data.objects.new("CarFill", bpy.data.lights.new("CarFill", "AREA"))
fill.data.energy, fill.data.color, fill.data.size = 900.0, (0.75, 0.6, 1.0), 6.0
C_FX.objects.link(fill)
fill.parent = BRK["rig"]
fill.location = (0.0, -2.0, 5.0)                    # soft top light (the holo glow) so the body shape reads
tc0 = bt(B_BREAK1) + 2.2
u = T - tc0
cpos = np.stack([-58.0 + 9.5 * u, 62.0 + 5.0 * np.sin(u * 0.22), 89.3 + 0.4 * np.sin(u * 1.3)], 1)
drive(BRK, cpos, ((T > tc0 - 0.5) & (T < bt(B_DROP2))).astype(float), "breakdown car (hover car, nose -X)")


def lane_sample(lp, s):
    """position on a closed polyline lane at arc-length fraction s (array)"""
    P_ = np.vstack([lp, lp[:1]])
    seg = np.linalg.norm(np.diff(P_[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = (s % 1.0) * cum[-1]
    i = np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1)
    f = (d - cum[i]) / np.maximum(seg[i], 1e-6)
    return P_[i] + (P_[i + 1] - P_[i]) * f[:, None], cum[-1]


# hero cars on the sky lanes (hidden in the breakdown — "city goes dark" — and gone once the city shuts off)
LANE_CARS = [("sf_hover_car", 0, 0.08, 31.0), ("sf_car_vertascan", 0, 0.47, 28.0), ("sf_car_dylanheyes", 1, 0.30, 33.0),
             ("sf_car_vertascan", 2, 0.02, 21.0), ("sf_car_dylanheyes", 3, 0.66, 37.0), ("sf_hover_car", 2, 0.55, 23.0)]
lane_vis = (1.0 - seg(tBK, tD2).astype(float)) * (T < SHUT_B).astype(float)
HERO_CARS = []
for k, (slug, li, ph, spd) in enumerate(LANE_CARS):
    car = make_car(slug, f"Lane{k}", {"sf_hover_car": 5.4, "sf_car_vertascan": 4.8, "sf_car_dylanheyes": 6.0}[slug],
                   glow=[(0.2, 0.7, 1.0), (1.0, 0.25, 0.8), (0.6, 0.3, 1.0)][k % 3])
    lp = LANE_PTS[li]
    _, Ll = lane_sample(lp, np.zeros(1))
    pos, _ = lane_sample(lp, ph + spd * T / Ll)
    pos = pos.copy()
    pos[:, 2] += 0.35 * np.sin(T * 1.7 + k)                                   # hover bob
    drive(car, pos, lane_vis, f"lane car {k} ({slug}, nose {NOSE[slug]})")
    HERO_CARS.append(car)

# distant sky traffic: light pairs (white heads +Y / red tails) + a coloured underglow, along the same lanes
SKY = curve_obj("SkyLanes", LANE_PTS, cyclic=True)
SKY.hide_render = True
st = dict(ci=[], L=[], off=[], spd=[], dirn=[], lane=[])
for li, lp in enumerate(LANE_PTS):
    Ll = float(np.linalg.norm(np.diff(np.vstack([lp, lp[:1]])[:, :2], axis=0), axis=1).sum())
    for j in range(int(Ll / 55)):
        st["ci"].append(float(li)); st["L"].append(Ll); st["off"].append(rng.random()); st["spd"].append(rng.uniform(22, 40))
        st["dirn"].append(1.0); st["lane"].append(float(rng.choice([-6.0, 0.0, 6.0])))
SKYT = points_obj("SkyTraffic", np.zeros((len(st["ci"]), 3)), {k: np.asarray(v) for k, v in st.items()}, coll=C_FX)
add_gn(SKYT, traffic_tree(SKY, "SkyTraffic", scale=1.6))
bake(SKYT, "hide_render", 1.0 - lane_vis, interp="CONSTANT")
print(f"[sky] {len(HERO_CARS)} hero cars + {len(st['ci'])} distant flying cars on {len(LANES)} lanes")

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
# cut-safe motion blur: the key on the LAST frame of every shot holds (CONSTANT), so the START-aligned shutter
# (frame → frame+0.5) never smears into the next shot's pose
_cut_frames = [fr(t0_) for t0_, _fn in SHOTS if t0_ > 0]
for fc_ in CAM.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
    for cf in _cut_frames:
        kp_ = fc_.keyframe_points[cf - 2]                    # key index = frame - 1 → frame (cut-1)
        assert abs(kp_.co[0] - (cf - 1)) < 1e-3
        kp_.interpolation = "CONSTANT"
print("[camera] cut-safe CONSTANT keys at frames", [c_ - 1 for c_ in _cut_frames])
# clearance: no hero car may pass through the camera, and no car may fly through a tower
LOC_A = np.array(loc_a)
for k, car in enumerate(HERO_CARS + [BRK]):
    on = car["vis"] > 0.5
    dd_ = np.where(on, np.linalg.norm(car["pos"] - LOC_A, axis=1), 1e9)
    dmin = float(dd_.min())
    car["closest_frame"] = int(np.argmin(dd_)) + 1
    tmin = 1e9
    for nm_, lo_, hi_ in _boxes:
        q = np.clip(car["pos"][on][::5], np.array(lo_), np.array(hi_))
        tmin = min(tmin, float(np.min(np.linalg.norm(q - car["pos"][on][::5], axis=1))) if on.any() else 1e9)
    print(f"[carcheck] car {k}: min dist to camera {dmin:.1f} m (frame {car['closest_frame']}), to tower bbox {tmin:.1f} m")
    assert dmin > 6.0 and tmin > 3.0, f"car {k} clips the camera or a tower"
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
    wn.link(wn.m("MULTIPLY", 0.05, wn.m("ADD", 0.15, wn.m("MULTIPLY", wcg["city"], 0.85))), bg2.inputs["Strength"])   # dark city → dark glass
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
# lingering post-rain haze: a BOUNDED volume box 0–450 m over the city (an EEVEE world volume is treated as
# infinitely deep and blacks out the sky — measured 10-13: background 0.25 → 0.002), so lamps, signs, lasers and
# firework flashes glow in the air while the sky above stays visible
_hm, _hb, _ho = new_mat("PostRainHaze")
_wv = _hb.n("ShaderNodeVolumePrincipled")
_hb.set(_wv.inputs["Color"], (0.75, 0.75, 0.82))
_hb.set(_wv.inputs["Density"], 0.00045)
_hb.set(_wv.inputs["Anisotropy"], 0.55)
_hb.link(_wv.outputs[0], _ho.inputs["Volume"])
_x0, _y0, _z0, _x1, _y1, _z1 = -2600, -1400, -1, 2600, 2800, 450
HAZE = mesh_obj("HazeBox", [(_x0, _y0, _z0), (_x1, _y0, _z0), (_x1, _y1, _z0), (_x0, _y1, _z0),
                            (_x0, _y0, _z1), (_x1, _y0, _z1), (_x1, _y1, _z1), (_x0, _y1, _z1)],
                [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], coll=C_FX)   # watertight
HAZE.data.materials.append(_hm)
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
