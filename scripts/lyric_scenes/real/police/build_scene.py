"""POLICE v2 — a night police chase: downtown avenue → exit ramp → elevated expressway → bridge → NITRO jump.

Zeke 2026-10-12 (v1 notes → v2):
  * both cars drove backwards (models' noses are -Y)            → yaw 180 by default
  * Corvette: only the tyre turned                               → whole wheel (rim/disc/hub islands) spins,
                                                                   calipers stay; spin is FORWARD and capped
                                                                   per frame so spokes never alias backwards
  * better buildings + signs as we pass (like carstreet)        → ledges, pilasters, cornices, parapets,
                                                                   roof gear, awnings, tower crown lights,
                                                                   neon blade signs / channel letters / roof
                                                                   billboards (3-D text), background city
  * windows + headlights react to the music                     → per-material Value nodes baked per frame:
                                                                   lit-window pulse, EQ window columns, kick
                                                                   flashes; headlight lenses + spots pulse
  * exit ramp → away from the city onto a bridge                → ramp in the avenue median at the calm
                                                                   before the last drops, elevated deck over
                                                                   the city, out over the river on a cable-
                                                                   stayed bridge (geometry placed FROM the
                                                                   lead's trajectory, so timing is exact)
  * on the drop: nitro flames + off a ramp                      → BRIDGE OUT kicker: launch on frame 6200 (the
                                                                   BIG drop), ballistic flight, landing on the
                                                                   far deck; police brakes at the gap. Short
                                                                   nitro bursts on every other drop entry.

PERIODIC street (v1): the avenue repeats every L m; moving things are shifted back by k*L each wrap. From the
finale cut F_C the shift is FROZEN and the unique finale geometry (ramp/deck/bridge) lives in rendered coords.
Everything is baked as keyframes (no drivers/handlers). Deterministic from the timeline npz.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
import scene_helpers as H  # noqa: E402
from scene_helpers import (MB, bake, bake_vec, blackbody, cube_obj, emissive, lathe, link, lk, M, nd,  # noqa: E402
                           new_mat, no_shadow, set_color_keys, show, tex_box, value_node, value_path)

SONG = os.path.join(REPO, "state", "tzeke_songs", "police")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=os.path.join(SONG, "timeline_30fps.npz"))
    ap.add_argument("--audio", default=os.path.join(SONG, "police.wav"))
    ap.add_argument("--cars", default="/data/iris_assets/cars")
    ap.add_argument("--polyhaven", default="/data/iris_assets/polyhaven")
    ap.add_argument("--font", default=os.path.join(REPO, "assets", "fonts", "Anton-Regular.ttf"))
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "scenes", "police3d", "police.blend"))
    ap.add_argument("--lead-yaw", type=float, default=180.0, help="deg: the models' noses point -Y")
    ap.add_argument("--police-yaw", type=float, default=180.0)
    ap.add_argument("--wheel-step", type=float, default=0.22, help="max visible wheel turn per frame (rad)")
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


def hash01(*k):
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    return (h ^ (h >> 15)) / 4294967295.0


def norm98(x):
    x = np.asarray(x, np.float64)
    return np.clip(x / (np.percentile(x, 98) + 1e-9), 0, 1)


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
HOT_S = gsmooth(HOT, 4)


def snap(f):
    return int(DOWNBEATS[np.argmin(np.abs(DOWNBEATS - f))])


# music levels for the city (EQ window columns, pulses, flashes)
EQ_L = np.clip(gsmooth(norm98(TL["bass"]), 1.2) * (0.25 + 0.75 * HOT_S), 0, 1)
EQ_M = np.clip(gsmooth(norm98(TL["rms"]), 1.2) * (0.25 + 0.75 * HOT_S), 0, 1)
EQ_H = np.clip(gsmooth(norm98(TL["high"]), 1.0) * (0.25 + 0.75 * HOT_S), 0, 1)
PULSE = np.clip(KICK * (0.35 + 0.65 * HOT_S) + 0.25 * BASS / (BASS.max() + 1e-9), 0, 1.5)
FLASH = np.clip(KICK * (0.35 * HOT_S + 0.9 * BIG), 0, 1.5)
HEADL = 1.0 + 0.9 * KICK * np.maximum(HOT_S, 0.25) + 0.4 * BIG * STROBE

# =====================================================================================
# 2. THE CHASE (true coordinates; +Y = down the avenue)
# =====================================================================================
# ---- scene reset FIRST: the vehicle prototypes must exist before the traffic plan needs their sizes
SC = bpy.context.scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.worlds,
             bpy.data.node_groups, bpy.data.images, bpy.data.curves):
    for d in list(coll):
        coll.remove(d)
H.SC = SC
import vehicles as VEH  # noqa: E402
PROTOS = {n: VEH.build(n) for n in VEH.SPECS}
VEH.cleanup_imports()
TRAFFIC_KINDS = [n for n, p in PROTOS.items() if p.role == "traffic"]
_tw = np.array([2.5 if n == "taxi" else 0.35 if n in ("citybus", "boxtruck", "postvan") else 1.0 for n in TRAFFIC_KINDS])
TRAFFIC_P = _tw / _tw.sum()
SMALL_KINDS = [n for n in TRAFFIC_KINDS if PROTOS[n].length < 5.3]

L = 480.0                                  # street period (m)
LANES = np.array([-5.4, -1.8, 1.8, 5.4])
F_C = 5548                                 # FINALE: calm section before the last drops (a shot cut)
F_JUMP = 6200                              # BIG drop: take-off frame
F_X = 4194                                 # red-light intersection: the Corvette crosses it on this drop entry
hot_s = gsmooth(HOT, 12)
big_s = gsmooth(BIG, 12)
# nitro: a spit of flame on EVERY KICK inside the drops (Zeke 10-12), a burst on each drop entry, the full
# burn from the BIG drop on (pumping on the kicks)
SPIT = age_env(TL["T_kage"], 3.2) * (HOT > 0.5)
NITRO = 0.85 * SPIT
for e in ENT:
    if e < F_JUMP - 5:
        NITRO[e:e + 40] = np.maximum(NITRO[e:e + 40], np.exp(-np.arange(min(40, N - e)) / 12.0))
NITRO[F_JUMP:] = 0.72 + 0.4 * KICK[F_JUMP:]
NITRO[6656:] *= np.linspace(1.0, 0.6, N - 6656)
NITRO_BIG = gsmooth((IDX >= F_JUMP).astype(float), 3)
FIN_PUSH = gsmooth(np.clip((IDX - (F_C - 40)) / 100.0, 0, 1), 6)    # the finale runs hotter (room for the bridge)
SPEED = (24.0 + 17.0 * hot_s + 12.0 * big_s + 5.0 * gsmooth(KICK * HOT, 2) + 14.0 * NITRO_BIG
         + 13.0 * FIN_PUSH * (1 - NITRO_BIG))                                                 # m/s
_cum = np.cumsum(SPEED) / FPS
# start offset chosen so the Corvette is in the middle of a cross street exactly on frame F_X
_y = 60.0 + _cum
Y_XING = min((k * L + c for k in range(-1, int(_y[-1] / L) + 2) for c in (88.0, 328.0)), key=lambda c: abs(c - _y[F_X]))
Y_LEAD = (Y_XING - _cum[F_X]) + _cum
GAP = gsmooth(26.0 - 9.0 * hot_s - 5.0 * big_s + 3.0 * np.sin(IDX / FPS * 0.37), 8)
# lane changes: OFTEN in the calm parts (every 1-2 bars, around traffic), every 2 bars in the drops; none
# just before the red light; from just before the finale only the two centre lanes (ramp in the median)
LANE_LOCK = F_C - 70
CHANGES = []                                                         # (frame, x_from, x_to)
mk = []
for s0, s1, h in SEGS:
    if s0 >= LANE_LOCK:
        break
    s1 = min(s1, LANE_LOCK)
    t = float(s0) if h else s0 + BAR
    while t < s1 - 0.5 * BAR:
        mk.append(s0 if t == s0 else snap(t))
        t += 2 * BAR if h else BAR * (1 if hash01(int(t), 9) < 0.55 else 2)
mk = sorted(set(m for m in mk if 20 < m < LANE_LOCK and not (F_X - 75 < m < F_X + 25)))
lane_i = np.full(N, 2)
cur = 2
for m in mk:
    d = 1 if hash01(m, 3) > 0.5 else -1
    nxt = cur + d if 0 <= cur + d <= 3 else cur - d
    CHANGES.append((m, float(LANES[cur]), float(LANES[nxt])))
    cur = nxt
    lane_i[m:] = cur
lane_t = LANES[lane_i].astype(float)
xcur = 1.8 if lane_t[LANE_LOCK - 1] > 0 else -1.8
lane_t[LANE_LOCK:] = xcur
fm = [snap(F_C + 60 + k * 1.5 * BAR) for k in range(6) if F_C + 60 + k * 1.5 * BAR < 5835 - 30]
fm += [snap(5835 + k * 2 * BAR) for k in range(1, 8) if 5835 + k * 2 * BAR < F_JUMP - 60]
for m in sorted(set(fm)):
    CHANGES.append((m, xcur, -xcur))
    xcur = -xcur
    lane_t[m:] = xcur
lane_t[F_JUMP - 60:] = lane_t[F_JUMP - 61]
X_LEAD = gsmooth(lane_t, 7)
K_WRAP = np.floor((Y_LEAD - 140.0) / L) * L                          # shift applied to everything moving
K_WRAP[F_C:] = K_WRAP[F_C]                                           # FROZEN for the finale
WRAPS = np.nonzero(np.diff(K_WRAP) != 0)[0] + 1                      # first frame after each wrap
KF = K_WRAP[F_C]
YR_LEAD = Y_LEAD - K_WRAP                                            # rendered y

# ---- finale geometry, placed from the lead's own trajectory (rendered coords) ----------
G = 9.81
DECK_H = 21.0                                   # expressway deck height (m)
RAMP_Y0 = YR_LEAD[F_C + 50]                     # ramp foot: the lead gets there ~1.7 s after the cut
RAMP_LEN = 230.0
RAMP_Y1 = RAMP_Y0 + RAMP_LEN
CITY_END = math.ceil((YR_LEAD[5990] + 1.0) / L) * L          # last period ends here → river
KICK_Y1 = YR_LEAD[F_JUMP]                       # kicker lip = take-off point, exactly on the drop
KICK_LEN, KICK_RISE = 16.0, 1.9
KICK_Y0 = KICK_Y1 - KICK_LEN
LAND_DROP = 3.0                                 # far deck is a little lower
v_off = SPEED[F_JUMP]
_ang = math.atan2(KICK_RISE, KICK_LEN) * 1.25   # lip curls up a little
VZ0 = v_off * math.sin(_ang)
_zt = DECK_H + KICK_RISE
_t_land = (VZ0 + math.sqrt(VZ0 ** 2 + 2 * G * (_zt - (DECK_H - LAND_DROP)))) / G
F_LAND = F_JUMP + int(round(_t_land * FPS))
LAND_Y = YR_LEAD[min(F_LAND, N - 1)]
LAND_Y0 = LAND_Y - 26.0                         # far deck's broken edge
END_Y = YR_LEAD[-1] + 400.0
print(f"[police] finale: ramp {RAMP_Y0:.0f}-{RAMP_Y1:.0f}  city ends {CITY_END:.0f}  kicker {KICK_Y0:.0f}-{KICK_Y1:.0f}"
      f"  land {LAND_Y0:.0f}/{LAND_Y:.0f} (f{F_LAND}, air {_t_land:.2f}s, v {v_off:.0f} m/s)  end {END_Y:.0f}"
      f"  red light at y {Y_XING:.0f} (f{F_X})")


def smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def zprof(y):
    """Road surface height along the finale route (rendered y)."""
    y = np.asarray(y, np.float64)
    z = DECK_H * smooth01((y - RAMP_Y0) / RAMP_LEN)
    kick = np.clip((y - KICK_Y0) / KICK_LEN, 0, 1)
    z = z + KICK_RISE * (kick ** 1.6) * (y <= KICK_Y1)
    z = np.where(y >= LAND_Y0, DECK_H - LAND_DROP, z)
    return z


# lead height: road profile, ballistic between take-off and landing
Z_LEAD = np.zeros(N)
fin = IDX >= F_C
Z_LEAD[fin] = zprof(YR_LEAD[fin])
for i in range(F_JUMP, N):
    t = (i - F_JUMP) / FPS
    zb = _zt + VZ0 * t - 0.5 * G * t * t
    zr = zprof(YR_LEAD[i]) if YR_LEAD[i] >= LAND_Y0 else -1e9      # nothing under the gap
    if i > F_JUMP + 3 and zb <= zr:
        break
    Z_LEAD[i] = zb
AIR = np.zeros(N, bool)
AIR[F_JUMP:i] = True
F_LAND = i


# ---- POLICE: they drive the Corvette's own TRACK (x as a function of y along its path), so wherever it
# squeezed past traffic they squeeze past too; more cars join as the song builds; all brake at the gap
def x_path(y):
    return np.interp(y, Y_LEAD, X_LEAD)


A_BRAKE = 8.5
y_stop = KICK_Y0 - 22.0 + KF                      # TRUE y where the first cruiser stops


def brake_to(Y, ystop):
    Y = Y.copy()
    vp = np.gradient(Y) * FPS
    for i in range(F_C, N):
        d_left = ystop - Y[i - 1]
        v_prev = (Y[i - 1] - Y[i - 2]) * FPS
        v_need = math.sqrt(2 * A_BRAKE * max(d_left, 0.0))
        v = max(min(vp[i], v_need, v_prev + 0.5), 0.0)
        Y[i] = Y[i - 1] + v / FPS
    return Y


Y_POL = brake_to(Y_LEAD - GAP, y_stop)
X_POL = gsmooth(x_path(Y_POL), 3)
YR_POL = Y_POL - K_WRAP
Z_POL = np.zeros(N)
Z_POL[fin] = zprof(YR_POL[fin])
POL_STOP = int(np.argmax((IDX > F_C) & (np.gradient(YR_POL) * FPS < 0.3))) or N
print(f"[police] police stops at y {YR_POL[-1]:.0f} (frame {POL_STOP})")
POLICE_X = [("charger", 1096, 13.0), ("crownvic", 2554, 26.0), ("hwy", 3465, 39.0), ("pol_sedan", 4559, 52.0)]
COPS = []
for k, (nm, J, extra) in enumerate(POLICE_X):
    arrive = smooth01((IDX - J) / 150.0)
    g = GAP + extra + 2.0 * np.sin(IDX / FPS * 0.29 + 1.7 * k) + 170.0 * (1 - arrive)
    Yk = brake_to(Y_LEAD - g, y_stop - 11.0 * (k + 1))
    Xk = gsmooth(x_path(Yk), 3) + 0.25 * np.sin(IDX / FPS * 0.8 + k)
    Zk = np.zeros(N)
    Zk[fin] = zprof(Yk[fin] - KF)
    COPS.append(dict(name=nm, J=J, X=Xk, Y=Yk, Z=Zk, vis=IDX >= J, L=PROTOS[nm].length, W=PROTOS[nm].width))

DX_LEAD = np.gradient(X_LEAD) * FPS
DX_POL = np.gradient(X_POL) * FPS

# ---- TRAFFIC PLAN (true coordinates). Light traffic, more in the calm parts. Every lane change of the
# Corvette gets a BLOCKER in the lane it is leaving (overtaken a moment after the swerve), plus ambient cars
# in the other lanes; then everything that would touch a chase car is removed (no collisions, ever).
V_LANE = {-5.4: 10.5, -1.8: 13.0, 1.8: 14.5, 5.4: 12.0}
rngT = np.random.default_rng(77)
TRAF = []


def avenue_car(kind, x, v, f_o, tag, deck=False):
    Y = Y_LEAD[f_o] + v * (IDX - f_o) / FPS
    rel = Y - Y_LEAD
    win = (rel > -230) & (rel < 650)
    if deck:
        vis = (IDX >= F_C) & win & (Y - KF > RAMP_Y0 + 8) & (Y - KF < KICK_Y0 - 170)
    else:
        vis = (IDX < F_C) & win
    p = PROTOS[kind]
    return dict(kind=kind, X=np.full(N, float(x)), Y=Y, v=v, vis=vis, L=p.length, W=p.width, tag=tag,
                axis="y", deck=deck, lane=float(x))


def half(c):
    return (c["W"] / 2, c["L"] / 2) if c.get("axis", "y") == "y" else (c["L"] / 2, c["W"] / 2)


def separation(a, b):
    """Smallest rectangle separation (m, <0 = overlap) over the frames both are visible; +inf if never."""
    m = a["vis"] & b["vis"]
    if not m.any():
        return np.inf
    ax, ay = half(a)
    bx, by = half(b)
    gx = np.abs(a["X"][m] - b["X"][m]) - ax - bx
    gy = np.abs(a["Y"][m] - b["Y"][m]) - ay - by
    return float(np.max(np.stack([gx, gy]), axis=0).min())


CHASE = [dict(name="lead", X=X_LEAD, Y=Y_LEAD, vis=np.ones(N, bool), L=4.75, W=2.05),
         dict(name="p1", X=X_POL, Y=Y_POL, vis=np.ones(N, bool), L=4.6, W=2.1)] + COPS
MARGIN = 0.45                                      # minimum bumper/side clearance anyone gets


def pick(small=False):
    ks = SMALL_KINDS if small else TRAFFIC_KINDS
    if small:
        return ks[int(rngT.integers(len(ks)))]
    return TRAFFIC_KINDS[int(rngT.choice(len(TRAFFIC_KINDS), p=TRAFFIC_P))]


n_block = 0
for (m, xa, xb) in CHANGES:
    deck = m >= F_C
    if m > F_JUMP - 150 or (not deck and m > F_C - 100):
        continue
    if hash01(m, 11) > (0.92 if HOT_S[m] < 0.5 else 0.6):
        continue
    kind = pick(small=deck)
    v = 13.0 if deck else V_LANE[xa]
    for fo in range(m + 16, m + 46, 2):
        c = avenue_car(kind, xa, v, fo, "blocker", deck)
        if separation(c, CHASE[0]) > MARGIN:
            TRAF.append(c)
            n_block += 1
            break
f = 30
while True:                                        # ambient cars in the other lanes
    rate = 0.65 if HOT_S[f] < 0.5 else 0.22
    f += int(rngT.exponential(FPS / rate)) + 8
    if f >= F_C - 90:
        break
    others = [x for x in LANES if abs(x - X_LEAD[f]) > 2.6]
    x = float(others[int(rngT.integers(len(others)))])
    TRAF.append(avenue_car(pick(), x, V_LANE[x], f, "amb"))
f = F_C + 40
while True:                                        # a few on the expressway during the calm climb
    f += int(rngT.exponential(FPS / 0.45)) + 10
    if f >= 5835:
        break
    x = -1.8 if X_LEAD[f] > 0 else 1.8
    TRAF.append(avenue_car(pick(small=True), x, 13.0, f, "amb", deck=True))
# the red light: nobody in the avenue approach/box while it is red for them; a couple of cars QUEUED at the
# stop line in the lanes the Corvette isn't using; a stream of CROSS traffic it threads through
XING_RED = (F_X - 150, F_X + 95)
_w = slice(F_X - 210, F_X + 130)
TRAF = [c for c in TRAF if not np.any(c["vis"][_w] & (c["Y"][_w] > Y_XING - 85) & (c["Y"][_w] < Y_XING + 14))]
qvis = (IDX >= F_X - 330) & (IDX <= F_X + 160) & (IDX < F_C)
for x in LANES:
    if abs(x - X_LEAD[F_X]) < 2.6:
        continue
    yq = Y_XING - 12.5
    for j in range(2 if abs(x) > 3 else 1):
        kind = pick(small=True)
        p = PROTOS[kind]
        yq -= p.length / 2 + 0.6
        TRAF.append(dict(kind=kind, X=np.full(N, float(x)), Y=np.full(N, yq), v=0.0, vis=qvis.copy(), L=p.length,
                         W=p.width, tag="queue", axis="y", deck=False, lane=float(x)))
        yq -= p.length / 2 + 1.2
V_CROSS, SP_CROSS = 12.5, 15.0
for yl, dirn in ((Y_XING - 3.6, 1), (Y_XING + 3.6, -1)):
    best = None
    for ph in np.arange(0.0, SP_CROSS, 0.5):
        cars = []
        for j in range(-10, 11):
            f0 = F_X + (ph + SP_CROSS * j) / V_CROSS * FPS
            X = dirn * V_CROSS * (IDX - f0) / FPS
            vis = (np.abs(X) < 80) & (IDX >= F_X - 210) & (IDX <= F_X + 130)
            if not vis.any():
                continue
            inbox = vis & (np.abs(X) < 11)
            if np.any(inbox & ((IDX < XING_RED[0] + 12) | (IDX > XING_RED[1] - 6))):
                continue
            kind = pick(small=True)
            p = PROTOS[kind]
            cars.append(dict(kind=kind, X=X, Y=np.full(N, yl), v=V_CROSS, vis=vis, L=p.length, W=p.width,
                             tag="cross", axis="x", deck=False, dirn=dirn, f0=f0))
        sep = min((separation(c, CHASE[0]) for c in cars), default=np.inf)
        score = abs(sep - 1.0) if sep >= 0.7 else 99 + (0.7 - sep)
        if best is None or score < best[0]:
            best = (score, sep, cars)
    TRAF += best[2]
    print(f"[police] cross lane {dirn:+d}: {len(best[2])} cars, nearest miss to the Corvette {best[1]:.2f} m")
# cull: anything touching a chase car; same-lane bunching
removed = {}
keep = []
for c in TRAF:
    if any(separation(c, ch) < MARGIN for ch in CHASE):
        removed[c["tag"]] = removed.get(c["tag"], 0) + 1
        continue
    keep.append(c)
TRAF = keep
keep = []
for c in TRAF:
    clash = False
    if c["axis"] == "y" and c["tag"] != "queue":
        for d in keep:
            if d["axis"] == "y" and d["tag"] != "queue" and d["lane"] == c["lane"] and d["deck"] == c["deck"]:
                if separation(c, d) < 3.0:
                    clash = True
                    break
    if not clash:
        keep.append(c)
TRAF = keep
print(f"[police] traffic: {len(TRAF)} cars ({n_block} blockers planned; culled {removed}); "
      f"police: P1 + {', '.join(c['name'] for c in COPS)}")

# =====================================================================================
# 3. SHOT LIST (cuts on drop entries and bar lines; scripted finale)
# =====================================================================================
SHOTS = []
_cyc_i = [0]
for s0, s1, h in SEGS:
    if s0 >= F_C:
        break
    s1 = min(s1, F_C)
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
# the red light: a high corner camera sees the queue, the cross traffic and the Corvette blowing through on
# the drop; then low on the cross street as the cruisers squeeze through after it
X0, X1, X2 = snap(F_X - 3 * BAR), F_X + 12, snap(F_X + 2 * BAR)
SHOTS = [(s_, n) for s_, n in SHOTS if not (X0 <= s_ <= X2)] + [(X0, "xing"), (X1, "xing_low"), (X2, "chase_high")]
# finale (F_C → end): ramp climb, expressway, bridge, the jump, the escape
F_EXIT = snap(int(np.argmax(YR_LEAD * (IDX >= F_C) > CITY_END - 45)))   # the deck leaves the city
FIN = [(F_C, "heli_ramp"), (snap(F_C + 2 * BAR), "side_low"), (snap(F_C + 4 * BAR), "chase_high"),
       (5835, "front_reverse"), (snap(5835 + 1 * BAR), "flyby"), (snap(5835 + 2 * BAR), "bumper"),
       (F_EXIT, "heli_bridge"), (snap(F_EXIT + 2 * BAR), "side_low"),
       (6163, "kicker_view"), (F_JUMP, "jump_side"), (F_JUMP + 34, "land_chase"), (F_LAND + 30, "police_edge"),
       (snap(F_LAND + 30 + 1.5 * BAR), "rear_low")]
t = FIN[-1][0] + BAR
cyc2 = ["side_low_lead", "front_reverse", "chase_lead", "rear_low", "heli_lead", "side_low_lead"]
k = 0
while t < 6656 - BAR * 0.6:
    FIN.append((snap(t), cyc2[k % len(cyc2)]))
    k += 1
    t += BAR
FIN.append((6656, "far"))
SHOTS += FIN
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
# 4. WORLD
# =====================================================================================
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


def mat_concrete(name, tint, tile=2.5, rough=(0.5, 0.85), tex="concrete_floor_02"):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    col = tex_box(nt, ph(tex, "diff"), tile)
    rgh = tex_box(nt, ph(tex, "rough"), tile, True)
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


FACADE_NTS = []                                   # node trees whose music Value nodes get baked


def mat_facade(name, tint, tex="concrete_floor_02", tile=4.0, glassy=False, colored_tex=False):
    """Building skin: scan-textured wall with a WORLD-SPACE window grid (floor 3.6 m, bay 3.2 m).
    ~20 % of windows lit (warm or cool), keyed on the per-building id (object prop 'bid') so periodic
    copies match. Ground floor: lit storefront glass + a coloured sign band. MUSIC: lit windows pulse
    (Pulse), a subset of dark windows flash on kicks (Flash), and on EQ buildings some window COLUMNS
    fill up floor by floor with the bass / mids / highs (EqL / EqM / EqH; uses the 'bh' height prop)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    col = tex_box(nt, ph(tex, "diff"), tile, coord=None)
    if colored_tex:                                  # brick: keep the scan's own colour, tint lightly
        wall = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        wall.inputs[0].default_value = 1.0
        lk(nt, col.outputs["Color"], wall.inputs[6])
        wall.inputs[7].default_value = tuple(tint) + (1,)
    else:
        bw = nd(nt, "ShaderNodeRGBToBW")
        lk(nt, col.outputs["Color"], bw.inputs[0])
        wear = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.15, "From Max": 0.7, "To Min": 0.7, "To Max": 1.05})
        lk(nt, bw.outputs[0], wear.inputs["Value"])
        wall = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        wall.inputs[0].default_value = 1.0
        wall.inputs[6].default_value = tuple(tint) + (1,)
        lk(nt, wear.outputs["Result"], wall.inputs[7])
    pulse = value_node(nt, "Pulse").outputs[0]
    flash = value_node(nt, "Flash").outputs[0]
    eqs = [value_node(nt, n).outputs[0] for n in ("EqL", "EqM", "EqH")]
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
    bh = nd(nt, "ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="bh")
    cmb = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, fl, cmb.inputs[0])
    lk(nt, bay, cmb.inputs[1])
    lk(nt, bid.outputs["Fac"], cmb.inputs[2])
    wn_ = nd(nt, "ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    lk(nt, cmb.outputs[0], wn_.inputs["Vector"])
    sepw = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, wn_.outputs["Color"], sepw.inputs[0])
    lit = M(nt, "GREATER_THAN", wn_.outputs["Value"], 0.80)
    flashset = M(nt, "MULTIPLY", M(nt, "GREATER_THAN", wn_.outputs["Value"], 0.66), M(nt, "SUBTRACT", 1.0, lit))
    warm = M(nt, "GREATER_THAN", sepw.outputs[1], 0.35)
    lcol = nd(nt, "ShaderNodeMix", data_type="RGBA")
    lk(nt, warm, lcol.inputs[0])
    lcol.inputs[6].default_value = (0.75, 0.85, 1.0, 1)
    lcol.inputs[7].default_value = (1.0, 0.70, 0.40, 1)
    dim = M(nt, "ADD", 0.25, M(nt, "MULTIPLY", sepw.outputs[2], 0.75))
    # window interiors: a cross mullion (dark frame), venetian blinds on some, curtains half-drawn on
    # others, and a ceiling-light gradient — so a lit window reads as a ROOM, not a flat white tile
    a0, a1, b0, b1 = (0.04, 0.96, 0.10, 0.95) if glassy else (0.20, 0.80, 0.33, 0.82)
    u = M(nt, "DIVIDE", M(nt, "SUBTRACT", fa, a0), a1 - a0)
    v = M(nt, "DIVIDE", M(nt, "SUBTRACT", fz, b0), b1 - b0)
    cmb3 = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, fl, cmb3.inputs[0])
    lk(nt, bay, cmb3.inputs[1])
    lk(nt, M(nt, "ADD", M(nt, "MULTIPLY", bid.outputs["Fac"], 1.37), 3.3), cmb3.inputs[2])
    wn3 = nd(nt, "ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    lk(nt, cmb3.outputs[0], wn3.inputs["Vector"])
    sep3 = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, wn3.outputs["Color"], sep3.inputs[0])
    mull = M(nt, "LESS_THAN", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", u, 0.5)), 0.012 if glassy else 0.022)
    if not glassy:
        mull = M(nt, "MAXIMUM", mull, M(nt, "LESS_THAN", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", v, 0.70)), 0.025))
    slat = M(nt, "ADD", 0.45, M(nt, "MULTIPLY", M(nt, "GREATER_THAN", M(nt, "FRACT", M(nt, "MULTIPLY", v, 16.0)), 0.4), 0.55))
    blinds = M(nt, "GREATER_THAN", sep3.outputs[0], 0.62)
    slatf = M(nt, "ADD", 1.0, M(nt, "MULTIPLY", blinds, M(nt, "SUBTRACT", slat, 1.0)))
    cw = M(nt, "MULTIPLY", sep3.outputs[2], 0.45)
    side_sel = M(nt, "GREATER_THAN", sep3.outputs[1], 0.5)
    curt = M(nt, "ADD", M(nt, "MULTIPLY", M(nt, "LESS_THAN", u, cw), M(nt, "SUBTRACT", 1.0, side_sel)),
             M(nt, "MULTIPLY", M(nt, "GREATER_THAN", u, M(nt, "SUBTRACT", 1.0, cw)), side_sel))
    curtf = M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", curt, 0.7))
    grad = M(nt, "ADD", 0.7, M(nt, "MULTIPLY", v, 0.4))
    nomull = M(nt, "SUBTRACT", 1.0, mull)
    interior = M(nt, "MULTIPLY", M(nt, "MULTIPLY", slatf, curtf), M(nt, "MULTIPLY", grad, nomull))
    # EQ columns: per (bay, building) — ~22 % of bays on EQ buildings; band picked per column
    cmb2 = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, bay, cmb2.inputs[0])
    lk(nt, bid.outputs["Fac"], cmb2.inputs[1])
    cmb2.inputs[2].default_value = 7.0
    wn2 = nd(nt, "ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    lk(nt, cmb2.outputs[0], wn2.inputs["Vector"])
    sep2 = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, wn2.outputs["Color"], sep2.inputs[0])
    eq_bldg = M(nt, "GREATER_THAN", M(nt, "FRACT", M(nt, "MULTIPLY", bid.outputs["Fac"], 0.4142)), 0.45)
    eq_col = M(nt, "MULTIPLY", M(nt, "LESS_THAN", wn2.outputs["Value"], 0.22), eq_bldg)
    bandsel = sep2.outputs[0]
    lvl = M(nt, "ADD", M(nt, "MULTIPLY", eqs[0], M(nt, "LESS_THAN", bandsel, 0.34)),
            M(nt, "ADD", M(nt, "MULTIPLY", eqs[1], M(nt, "MULTIPLY", M(nt, "GREATER_THAN", bandsel, 0.34),
                                                    M(nt, "LESS_THAN", bandsel, 0.67))),
              M(nt, "MULTIPLY", eqs[2], M(nt, "GREATER_THAN", bandsel, 0.67))))
    nfl = M(nt, "SUBTRACT", M(nt, "DIVIDE", bh.outputs["Fac"], 3.6), 1.5)
    eq_on = M(nt, "LESS_THAN", M(nt, "SUBTRACT", fl, 1.0), M(nt, "MULTIPLY", lvl, nfl))
    eqwin = M(nt, "MULTIPLY", M(nt, "MULTIPLY", win, eq_col), eq_on)
    # storefronts: one per 3.2 m bay with mullions, z 0.5-3.3; about half the bays lit
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
    # emission = windows (dim, pulsing) + kick flashes + EQ columns + storefront + sign
    win_pulse = M(nt, "ADD", 0.75, M(nt, "MULTIPLY", pulse, 0.9))
    e_win = M(nt, "MULTIPLY", M(nt, "MULTIPLY", win, lit), M(nt, "MULTIPLY", dim, M(nt, "MULTIPLY", win_pulse, 2.0)))
    e_win = M(nt, "MULTIPLY", e_win, interior)
    e_fl = M(nt, "MULTIPLY", M(nt, "MULTIPLY", win, flashset), M(nt, "MULTIPLY", flash, M(nt, "MULTIPLY", nomull, 2.6)))
    e_eq = M(nt, "MULTIPLY", eqwin, M(nt, "MULTIPLY", nomull, 3.2))
    ecol = nd(nt, "ShaderNodeMix", data_type="RGBA")
    lk(nt, M(nt, "GREATER_THAN", M(nt, "ADD", store, sign), 0.5), ecol.inputs[0])
    lk(nt, lcol.outputs[2], ecol.inputs[6])
    lk(nt, storec.outputs[2], ecol.inputs[7])
    ecol2 = nd(nt, "ShaderNodeMix", data_type="RGBA")
    lk(nt, eqwin, ecol2.inputs[0])
    lk(nt, ecol.outputs[2], ecol2.inputs[6])
    lk(nt, hue.outputs[0], ecol2.inputs[7])
    estr = M(nt, "ADD", M(nt, "ADD", e_win, M(nt, "ADD", e_fl, e_eq)),
             M(nt, "ADD", M(nt, "MULTIPLY", M(nt, "MULTIPLY", store, shop_lit), M(nt, "MULTIPLY", dim, 1.6)),
               M(nt, "MULTIPLY", sign, 4.0)))
    lk(nt, ecol2.outputs[2], b.inputs["Emission Color"])
    lk(nt, estr, b.inputs["Emission Strength"])
    glass = M(nt, "MAXIMUM", M(nt, "MULTIPLY", win, nomull), store)
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
    # wall relief from the scan's normal map + window recesses
    nrm = tex_box(nt, ph(tex, "nor_gl"), tile, True)
    nmap = nd(nt, "ShaderNodeNormalMap", inp={"Strength": 0.8})
    lk(nt, nrm.outputs["Color"], nmap.inputs["Color"])
    bmp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.35, "Distance": 0.08})
    lk(nt, M(nt, "MULTIPLY", glass, -1.0), bmp.inputs["Height"])
    lk(nt, nmap.outputs["Normal"], bmp.inputs["Normal"])
    lk(nt, bmp.outputs["Normal"], b.inputs["Normal"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    FACADE_NTS.append(nt)
    return m


M_ROAD = mat_asphalt()
M_WALK = mat_concrete("Sidewalk", (0.42, 0.41, 0.40), tile=2.0)
M_CURB = mat_concrete("Curb", (0.55, 0.54, 0.52), tile=1.0)
M_DECK = mat_concrete("DeckConcrete", (0.50, 0.49, 0.47), tile=3.0, tex="concrete_wall_006")
M_ROOF = mat_concrete("RoofGear", (0.30, 0.30, 0.31), tile=1.5, rough=(0.35, 0.7), tex="concrete_wall_006")
M_PAINT, _pnt = new_mat("LinePaint")
_b = nd(_pnt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.72, 0.72, 0.70, 1), "Roughness": 0.35})
lk(_pnt, _b.outputs[0], nd(_pnt, "ShaderNodeOutputMaterial").inputs["Surface"])
M_POLE, _pt = new_mat("Pole")
_b = nd(_pt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.05, 0.05, 0.055, 1), "Roughness": 0.4, "Metallic": 0.6})
lk(_pt, _b.outputs[0], nd(_pt, "ShaderNodeOutputMaterial").inputs["Surface"])
M_STEEL, _st = new_mat("Steel")
_b = nd(_st, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.35, 0.36, 0.38, 1), "Roughness": 0.3, "Metallic": 1.0})
lk(_st, _b.outputs[0], nd(_st, "ShaderNodeOutputMaterial").inputs["Surface"])
M_PANEL, _pn = new_mat("SignPanel")
_b = nd(_pn, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.02, 0.02, 0.025, 1), "Roughness": 0.45, "Metallic": 0.3})
lk(_pn, _b.outputs[0], nd(_pn, "ShaderNodeOutputMaterial").inputs["Surface"])
M_LAMP = emissive("StreetLamp", blackbody(3600), 40.0)
M_BEACON = emissive("Beacon", (1.0, 0.05, 0.02), 30.0, objcolor_gain=True)
M_NEON = emissive("Neon", (1.0, 1.0, 1.0), 7.0, objcolor_gain=True)
M_CROWN = emissive("Crown", (0.75, 0.88, 1.0), 5.0, objcolor_gain=True)
M_AMBER = emissive("Amber", (1.0, 0.45, 0.02), 40.0, objcolor_gain=True)
AWN = [H.principled(f"Awning{k}", c, rough=0.7) for k, c in
       enumerate([(0.35, 0.03, 0.03), (0.03, 0.18, 0.08), (0.03, 0.05, 0.22), (0.30, 0.20, 0.02)])]
FACADES = [mat_facade("FacadeLight", (0.62, 0.60, 0.56)),
           mat_facade("FacadeDark", (0.22, 0.22, 0.24), tex="concrete_wall_006"),
           mat_facade("FacadeSand", (0.70, 0.60, 0.48), tex="painted_plaster_wall", tile=3.0),
           mat_facade("GlassTower", (0.10, 0.12, 0.14), glassy=True),
           mat_facade("FacadeBrick", (0.85, 0.80, 0.78), tex="red_brick_03", tile=2.2, colored_tex=True)]
FONT = bpy.data.fonts.load(ARGS.font, check_existing=True)


def text_obj(name, body, size, mat, loc, rot, parent=None, extrude=0.02, align_y="CENTER", spacing=1.0):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = FONT
    cu.size = size
    cu.extrude = extrude
    cu.align_x = "CENTER"
    cu.align_y = align_y
    cu.space_line = spacing
    cu.materials.append(mat)
    ob = bpy.data.objects.new(name, cu)
    ob.location = loc
    ob.rotation_euler = rot
    return link(ob, parent)


def prism(mb, c, r, h, n=10):
    """Vertical n-gon prism (tanks, columns) into an MB."""
    cx, cy, cz = c
    ring = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    pts = [(x, y, cz) for x, y in ring] + [(x, y, cz + h) for x, y in ring]
    faces = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    mb.poly(pts, faces)


# neon level patterns (per sign, keyed through object colour)
def neon_level(mode, seed):
    if mode == 0:                                                     # steady, faint breathing
        return 0.9 + 0.1 * np.sin(IDX * 0.05 + seed)
    if mode == 1:                                                     # kick pulse
        return 0.55 + 0.9 * KICK * np.maximum(HOT_S, 0.2)
    if mode == 2:                                                     # old tube: snare flicker
        fl = (SNARE > 0.6) & (np.array([hash01(i, seed) for i in range(N)]) > 0.4)
        return np.where(fl, 0.08, 1.0)
    on = (np.floor(BEAT_PH * (2 if seed % 2 else 1)).astype(int) % 2 == 0)    # beat chaser
    return np.where(HOT_S > 0.3, np.where(on, 1.3, 0.25), 0.9)


def hsv(h, s=0.9, v=1.0):
    import colorsys
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


NEON_HUES = [0.0, 0.08, 0.13, 0.33, 0.5, 0.55, 0.62, 0.78, 0.86, 0.93]
SIGN_WORDS = ("BAR", "HOTEL", "NOODLE", "DINER", "MOTEL", "TATTOO", "ARCADE", "PAWN", "KARAOKE", "LIQUOR",
              "PIZZA", "CLUB", "RAMEN", "CASINO", "LOANS", "GARAGE", "SUSHI", "VIDEO", "TAXI", "OPEN")
ROOF_WORDS = ("TZEKE000", "HOTEL", "CASINO", "DINER", "MOTEL", "NIGHT OWL", "KARAOKE")
NEON_KEYS = []                                                        # (object, rgb, level array)

# =====================================================================================
# 6. THE STREET (one period built procedurally, then copied)
# =====================================================================================
ROAD_W, WALK = 16.0, 4.0
XF = ROAD_W / 2 + WALK                      # building front line (|x|)
CROSS = [(80.0, 16.0), (320.0, 16.0)]       # cross streets: (start y, width) within the period
rng = np.random.default_rng(2026)
STREET = bpy.data.collections.new("StreetPeriod")
SC.collection.children.link(STREET)


def to_street(*obs):
    for o in obs:
        STREET.objects.link(o)
        if o.name in SC.collection.objects:
            SC.collection.objects.unlink(o)


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


BUILDINGS = []                               # (object, height, front x, centre y)
ROOFG = MB()                                 # parapets, AC units, tanks, masts
CROWN = MB()                                 # tower crown light strips
AWNMB = [MB() for _ in AWN]
bid = 0
n_signs = 0
for sx in (-1, 1):
    for y0, y1 in building_runs():
        bid += 1
        tall = rng.random() < 0.28
        nfl = int(rng.integers(14, 34) if tall else rng.integers(4, 12))
        h = 3.6 * nfl + 1.2
        depth = float(rng.uniform(16, 28))
        inset = float(rng.uniform(0.0, 1.2))
        mb = MB()
        xf = sx * (XF + inset)
        yc, wy = (y0 + y1) / 2, (y1 - y0) - 0.3
        mb.box((xf + sx * depth / 2, yc, h / 2), (depth, wy, h))
        glassy = tall and rng.random() < 0.6
        mat_i = 3 if glassy else (int(rng.integers(0, 5)) if not tall else int(rng.choice([0, 1, 2])))
        if mat_i == 3 and not glassy:
            mat_i = 4
        top = h
        if h > 30 and rng.random() < 0.5:                                  # setback crown
            mb.box((xf + sx * (depth / 2 + 2.0), yc, h + 5.4), (depth - 4.0, wy - 4.0, 10.8))
            top = h + 10.8
        if not glassy:
            # relief: pilasters every 2 bays, floor-line string courses, a canopy band, a cornice
            px = xf - sx * 0.14
            for yy in np.arange(math.ceil((y0 + 0.6) / 6.4) * 6.4, y1 - 0.6, 6.4):
                mb.box((px, yy, (5.2 + h) / 2), (0.28, 0.55, h - 5.2))
            every = int(rng.integers(3, 6))
            for f in range(every, nfl, every):
                mb.box((xf - sx * 0.09, yc, 3.6 * f), (0.18, wy, 0.26))
            mb.box((xf - sx * 0.3, yc, 4.65), (0.6, wy, 0.3))                       # canopy band
            mb.box((xf - sx * 0.35, yc, h - 0.4), (0.7, wy + 0.3, 0.8))            # cornice
            # parapet ring
            for (cx_, cy_, sx_, sy_) in ((xf + sx * 0.15, yc, 0.3, wy), (xf + sx * (depth - 0.15), yc, 0.3, wy),
                                         (xf + sx * depth / 2, y0 + 0.3, depth, 0.3),
                                         (xf + sx * depth / 2, y1 - 0.45, depth, 0.3)):
                ROOFG.box((cx_, cy_, h + 0.5), (sx_, sy_, 1.0))
        else:
            # glass tower: vertical mullion fins every bay + a light strip at the crown
            for yy in np.arange(math.ceil((y0 + 0.3) / 3.2) * 3.2, y1 - 0.3, 3.2):
                mb.box((xf - sx * 0.08, yy, (5.2 + h) / 2), (0.16, 0.12, h - 5.2))
            CROWN.box((xf - sx * 0.06, yc, top - 0.6), (0.1, wy - 0.4, 0.35))
            CROWN.box((xf - sx * 0.06, yc, top - 1.6), (0.1, wy - 0.4, 0.12))
        # roof gear: AC units, tanks, masts
        for _ in range(int(rng.integers(1, 4))):
            ax = xf + sx * float(rng.uniform(3, depth - 3))
            ay = float(rng.uniform(y0 + 3, y1 - 3))
            ROOFG.box((ax, ay, top + 0.7), (float(rng.uniform(1.5, 3.5)), float(rng.uniform(1.5, 3)), 1.4))
        if not tall and rng.random() < 0.45:
            tx = xf + sx * float(rng.uniform(4, depth - 4))
            ty = float(rng.uniform(y0 + 4, y1 - 4))
            for lx_, ly_ in ((-1.2, -1.2), (1.2, -1.2), (-1.2, 1.2), (1.2, 1.2)):
                ROOFG.box((tx + lx_, ty + ly_, top + 1.5), (0.2, 0.2, 3.0))
            prism(ROOFG, (tx, ty, top + 3.0), 2.0, 3.8)
        if tall and rng.random() < 0.6:
            ROOFG.box((xf + sx * depth / 2, yc, top + 7.0), (0.35, 0.35, 14.0))
        # awnings over the shop bays
        if not glassy and rng.random() < 0.55:
            am = AWNMB[int(rng.integers(0, len(AWN)))]
            tilt = Matrix.Rotation(sx * math.radians(-14), 3, "Y")
            for yy in np.arange(math.ceil(y0 / 3.2) * 3.2 + 1.6, y1 - 1.6, 3.2):
                if rng.random() < 0.7:
                    am.box((xf - sx * 0.65, yy, 3.55), (1.3, 2.7, 0.08), rot=tilt)
        ob = mb.obj(f"Bldg{bid:03d}", FACADES[mat_i])
        ob["bid"] = float(bid)
        ob["bh"] = float(h)
        to_street(ob)
        BUILDINGS.append((ob, h, xf, yc))
        # ---- signs ----
        r = rng.random()
        hue = NEON_HUES[int(rng.integers(0, len(NEON_HUES)))]
        rgb = hsv(hue)
        mode = int(rng.integers(0, 4))
        if r < 0.55 and wy > 9:                                            # blade sign (faces traffic)
            word = SIGN_WORDS[int(rng.integers(0, len(SIGN_WORDS)))]
            n = len(word)
            Ls = 0.95 * n + 0.7
            by = float(rng.uniform(y0 + 3.0, y1 - 3.0))
            bx = xf - sx * 1.25
            bz = 5.6 + Ls / 2
            pm = MB()
            pm.box((xf - sx * 0.15, by, bz), (0.3, 0.12, 0.12))                      # brackets
            pm.box((bx, by, bz), (1.5, 0.22, Ls))
            pob = pm.obj(f"BladePanel{bid}", M_PANEL)
            nb = MB()
            for dy in (-0.13, 0.13):
                for (cx_, cz_, sx_, sz_) in ((bx, bz + Ls / 2 - 0.05, 1.5, 0.07), (bx, bz - Ls / 2 + 0.05, 1.5, 0.07),
                                             (bx - 0.72, bz, 0.07, Ls), (bx + 0.72, bz, 0.07, Ls)):
                    nb.box((cx_, by + dy, cz_), (sx_, 0.04, sz_))
            nob = nb.obj(f"BladeNeon{bid}", M_NEON)
            body = "\n".join(word)
            t1 = text_obj(f"BladeTxtA{bid}", body, 0.95, M_NEON, (bx, by - 0.125, bz + 0.12),
                          (math.pi / 2, 0, 0), spacing=0.86)
            t2 = text_obj(f"BladeTxtB{bid}", body, 0.95, M_NEON, (bx, by + 0.125, bz + 0.12),
                          (math.pi / 2, 0, math.pi), spacing=0.86)
            lvl = neon_level(mode, bid)
            for o_ in (nob, t1, t2):
                no_shadow(o_)
                NEON_KEYS.append((o_, rgb, lvl))
            to_street(pob, nob, t1, t2)
            n_signs += 1
        elif r < 0.85 and not glassy:                                      # channel letters over the shop
            word = SIGN_WORDS[int(rng.integers(0, len(SIGN_WORDS)))]
            if rng.random() < 0.5:
                word = word + " " + SIGN_WORDS[int(rng.integers(0, len(SIGN_WORDS)))]
            size = min(1.1, wy / (0.62 * len(word) + 1))
            t1 = text_obj(f"Fascia{bid}", word, size, M_NEON, (xf - sx * 0.5, yc, 4.95),
                          (math.pi / 2, 0, -sx * math.pi / 2), align_y="BOTTOM", extrude=0.05)
            no_shadow(t1)
            NEON_KEYS.append((t1, rgb, neon_level(mode, bid)))
            to_street(t1)
            n_signs += 1
        elif not tall and wy > 14:                                         # roof billboard facing traffic
            word = ROOF_WORDS[int(rng.integers(0, len(ROOF_WORDS)))]
            bw_ = min(wy - 2, 0.9 * 1.6 * len(word) + 2)
            bxc = xf + sx * min(depth / 2, 8.0)
            pm = MB()
            for lx_ in (-bw_ / 2 + 0.5, bw_ / 2 - 0.5):
                pm.box((bxc + lx_, yc, h + 1.5), (0.25, 0.25, 3.0))
            pm.box((bxc, yc, h + 4.6), (bw_, 0.3, 3.4))
            pob = pm.obj(f"BillPanel{bid}", M_PANEL)
            t1 = text_obj(f"BillTxtA{bid}", word, 2.2, M_NEON, (bxc, yc - 0.17, h + 4.6),
                          (math.pi / 2, 0, 0))
            t2 = text_obj(f"BillTxtB{bid}", word, 2.2, M_NEON, (bxc, yc + 0.17, h + 4.6),
                          (math.pi / 2, 0, math.pi))
            lvl = neon_level(1 if word == "TZEKE000" else mode, bid)
            for o_ in (t1, t2):
                no_shadow(o_)
                NEON_KEYS.append((o_, rgb if word != "TZEKE000" else hsv(0.55), lvl))
            to_street(pob, t1, t2)
            n_signs += 1
ro_ = ROOFG.obj("RoofGear", M_ROOF)
cr_ = CROWN.obj("CrownLights", M_CROWN)
no_shadow(cr_)
to_street(ro_, cr_)
for k, am in enumerate(AWNMB):
    if am.v:
        to_street(am.obj(f"Awnings{k}", AWN[k]))
# background city behind the street rows (periodic too): blocks 40 m with 14 m streets
for sx in (-1, 1):
    mbs = [MB() for _ in FACADES]
    for bx in np.arange(62.0, 430.0, 54.0):
        for by in np.arange(27.0, L, 54.0):
            for _ in range(int(rng.integers(1, 3))):
                h = float(12 + 95 * rng.random() ** 2.4 + (25 if bx < 120 and rng.random() < 0.3 else 0))
                h = 3.6 * round(h / 3.6) + 1.2
                w_, d_ = float(rng.uniform(14, 36)), float(rng.uniform(14, 36))
                ox, oy = float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10))
                mi = int(rng.choice([0, 1, 2, 3, 3, 4]))
                mbs[mi].box((sx * (bx + ox), by + oy, h / 2), (w_, d_, h))
    for k, m_ in enumerate(mbs):
        if m_.v:
            ob = m_.obj(f"BgCity{'LR'[sx > 0]}{k}", FACADES[k])
            ob["bid"] = float(500 + k * 7 + (sx > 0))
            ob["bh"] = 60.0
            to_street(ob)
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
to_street(ro, wo_, co_, po)
# streetlights every 30 m, both sides
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
        to_street(pole, head)
        STREET.objects.link(lo)
        LAMPS.append(lo)
# traffic signals at every cross street (periodic): avenue heads on a mast arm before the crossing + one on
# the far side, cross-street heads on the corners. One object per lamp colour per direction, keyed through
# object colour: the avenue goes amber → RED around F_X while the cross street goes green.
M_SIG = {c: emissive(f"Sig{c}", rgb, 26.0, objcolor_gain=True)
         for c, rgb in (("R", (1.0, 0.03, 0.01)), ("A", (1.0, 0.42, 0.0)), ("G", (0.12, 1.0, 0.4)))}
M_HOUS = H.principled("SignalHousing", (0.03, 0.03, 0.03), rough=0.5)
hous = MB()
sig_lamp = {(d, c): MB() for d in ("av", "cr") for c in "RAG"}


def sig_head(d, x, y, z, fx, fy):
    hous.box((x, y, z), (0.36, 0.3, 1.05) if fy else (0.3, 0.36, 1.05))
    for j, c in enumerate("RAG"):
        zc = z + 0.33 - j * 0.33
        if fy:
            sig_lamp[(d, c)].box((x, y + fy * 0.16, zc), (0.22, 0.03, 0.22))
        else:
            sig_lamp[(d, c)].box((x + fx * 0.16, y, zc), (0.03, 0.22, 0.22))


spm = MB()
for c0, w in CROSS:
    ys = c0 - 4.5
    spm.box((9.0, ys, 3.5), (0.28, 0.28, 7.0))
    spm.box((4.6, ys, 6.95), (8.8, 0.18, 0.18))
    for hx in (5.4, 1.8):
        sig_head("av", hx, ys, 6.25, 0, -1)
    spm.box((-9.0, c0 + w + 1.0, 2.2), (0.2, 0.2, 4.4))
    sig_head("av", -9.0, c0 + w + 1.0, 3.9, 0, -1)
    for sx in (-1, 1):
        for yy in (c0 - 1.0, c0 + w + 1.0):
            spm.box((sx * 9.6, yy, 2.2), (0.2, 0.2, 4.4))
            sig_head("cr", sx * 9.6, yy, 3.9, -sx, 0)
to_street(spm.obj("SignalPoles", M_POLE), hous.obj("SignalHeads", M_HOUS))
for c0, w in CROSS:                                   # the crossings were unlit (lamps skip cross streets)
    for k_, (lx, ly) in enumerate(((7.5, c0 + 3.0), (-7.5, c0 + w - 3.0))):
        ld = bpy.data.lights.new(f"XingLamp{int(c0)}{k_}", "SPOT")
        ld.energy = 5200.0
        ld.color = blackbody(3900)
        ld.spot_size = math.radians(130)
        ld.spot_blend = 0.5
        ld.use_shadow = False
        lo = bpy.data.objects.new(ld.name, ld)
        lo.location = (lx, ly, 9.5)
        STREET.objects.link(lo)
    hm = MB()
    for lx, ly in ((7.5, c0 + 3.0), (-7.5, c0 + w - 3.0)):
        hm.box((lx, ly, 9.7), (0.6, 0.6, 0.12))
    ho = hm.obj(f"XingHeads{int(c0)}", M_LAMP)
    no_shadow(ho)
    to_street(ho)
_on = lambda a, b: ((IDX >= a) & (IDX < b)).astype(float)          # noqa: E731
av_A, av_R = _on(F_X - 185, F_X - 150), _on(F_X - 150, F_X + 95)
av_G = 1.0 - av_A - av_R
cr_G, cr_A = _on(F_X - 140, F_X + 85), _on(F_X + 85, F_X + 95)
cr_R = 1.0 - cr_G - cr_A
for (d, c), mb_ in sig_lamp.items():
    o = mb_.obj(f"Sig_{d}_{c}", M_SIG[c])
    no_shadow(o)
    to_street(o)
    lv = {"avR": av_R, "avA": av_A, "avG": av_G, "crR": cr_R, "crA": cr_A, "crG": cr_G}[d + c]
    set_color_keys(o, 0.05 + 0.95 * lv)
BEACONS = []
for ob, h, xf, yc in BUILDINGS:
    if h > 60:
        bo = cube_obj(f"Beacon{ob.name}", M_BEACON, loc=(xf + np.sign(xf) * 6, yc, h + 0.4), scale=(0.25, 0.25, 0.25))
        no_shadow(bo)
        to_street(bo)
        BEACONS.append(bo)
# the period copies: collection instances (identical by construction). -1..+1 always; the finale adds
# copies up to the river (from F_C, when the shift is frozen)
K_EDGE = int(round(CITY_END / L))
for k in range(-1, K_EDGE):
    if k == 0:
        continue
    inst = bpy.data.objects.new(f"StreetCopy{k:+d}", None)
    inst.instance_type = "COLLECTION"
    inst.instance_collection = STREET
    inst.location = (0, k * L, 0)
    link(inst)
    if k > 1:
        show(inst, IDX >= F_C)
blink = ((IDX / FPS * 0.5) % 1.0 < 0.18).astype(float)          # aviation beacons ~0.5 Hz
for bo in BEACONS:
    set_color_keys(bo, np.stack([blink, blink * 0.05, blink * 0.02], 1))
crown_lvl = 0.5 + 0.9 * np.clip(gsmooth(EQ_L, 1.0), 0, 1.2)
set_color_keys(cr_, crown_lvl)
for o_, rgb, lvl in NEON_KEYS:
    set_color_keys(o_, np.stack([lvl * rgb[0], lvl * rgb[1], lvl * rgb[2]], 1))
for nt in FACADE_NTS:
    for nm, arr in (("Pulse", PULSE), ("Flash", FLASH), ("EqL", EQ_L), ("EqM", EQ_M), ("EqH", EQ_H)):
        bake(nt, value_path(nm), arr)
print("[police] street:", len(BUILDINGS), "buildings,", n_signs, "signs,", len(LAMPS), "streetlights,",
      len(BEACONS), "beacons / period; copies to", K_EDGE)

# =====================================================================================
# 7. FINALE GEOMETRY: ramp, expressway deck, river, bridge, BRIDGE OUT kicker, far deck
# =====================================================================================
FINALE = []                                   # objects shown only from F_C on
DECK_HW = 5.2                                 # deck half width
STEP = 2.0
M_WATER, _wt = new_mat("River")
_b = nd(_wt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.004, 0.006, 0.009, 1), "Roughness": 0.04})
_tc = nd(_wt, "ShaderNodeTexCoord")
_nz = nd(_wt, "ShaderNodeTexNoise", inp={"Scale": 0.35, "Detail": 6.0})
lk(_wt, _tc.outputs["Object"], _nz.inputs["Vector"])
_bp = nd(_wt, "ShaderNodeBump", inp={"Strength": 0.12})
lk(_wt, _nz.outputs["Fac"], _bp.inputs["Height"])
lk(_wt, _bp.outputs["Normal"], _b.inputs["Normal"])
lk(_wt, _b.outputs[0], nd(_wt, "ShaderNodeOutputMaterial").inputs["Surface"])
M_HAZ, _hz = new_mat("Hazard")
_b = nd(_hz, "ShaderNodeBsdfPrincipled", inp={"Roughness": 0.5})
_tc = nd(_hz, "ShaderNodeTexCoord")
_sepz = nd(_hz, "ShaderNodeSeparateXYZ")
lk(_hz, _tc.outputs["Object"], _sepz.inputs[0])
_st_ = M(_hz, "GREATER_THAN", M(_hz, "FRACT", M(_hz, "MULTIPLY", M(_hz, "ADD", _sepz.outputs[0], _sepz.outputs[1]), 0.8)), 0.5)
_hc = nd(_hz, "ShaderNodeMix", data_type="RGBA")
lk(_hz, _st_, _hc.inputs[0])
_hc.inputs[6].default_value = (0.02, 0.02, 0.02, 1)
_hc.inputs[7].default_value = (0.75, 0.45, 0.02, 1)
lk(_hz, _hc.outputs[2], _b.inputs["Base Color"])
lk(_hz, _b.outputs[0], nd(_hz, "ShaderNodeOutputMaterial").inputs["Surface"])
M_CONE = H.principled("Cone", (0.9, 0.22, 0.02), rough=0.5, emit=(1.0, 0.3, 0.05), emit_strength=0.3)
WATER_Z = -14.0


def fin(o):
    FINALE.append(o)
    return o


def strip(name, y0, y1, prof, mat, zfun, step=STEP):
    """Extrude a closed 2-D cross-section [(x, dz)] along y, following z = zfun(y)."""
    ys = np.arange(y0, y1 + 1e-6, step)
    zs = zfun(ys)
    n = len(prof)
    verts, faces = [], []
    for j, (y, z) in enumerate(zip(ys, zs)):
        for x, dz in prof:
            verts.append((x, y, z + dz))
    for j in range(len(ys) - 1):
        for k in range(n):
            a, b_ = j * n + k, j * n + (k + 1) % n
            faces.append((a, b_, b_ + n, a + n))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple((len(ys) - 1) * n + k for k in range(n)))
    return fin(H.mesh_obj(name, verts, faces, mat))


def zroad(y):
    return zprof(y)


SEC_DECK = [(-DECK_HW, 0.0), (DECK_HW, 0.0), (DECK_HW, -0.25), (DECK_HW - 0.6, -1.3), (-DECK_HW + 0.6, -1.3),
            (-DECK_HW, -0.25)]
SEC_TOP = [(-DECK_HW + 0.35, 0.02), (DECK_HW - 0.35, 0.02), (DECK_HW - 0.35, 0.0), (-DECK_HW + 0.35, 0.0)]
SEG1 = (RAMP_Y0 - 4.0, KICK_Y0)                 # ramp + deck up to the kicker
SEG3 = (LAND_Y0, END_Y)
for nm, (a, b_) in (("A", SEG1), ("C", SEG3)):
    strip(f"Deck{nm}", a, b_, SEC_DECK, M_DECK, zroad)
    strip(f"DeckTop{nm}", a, b_, SEC_TOP, M_ROAD, zroad)
    for sx in (-1, 1):
        strip(f"Parapet{nm}{sx}", a, b_, [(sx * DECK_HW - 0.2, 0.0), (sx * DECK_HW + 0.0, 0.0),
                                          (sx * DECK_HW + 0.0, 1.0), (sx * DECK_HW - 0.28, 1.0),
                                          (sx * DECK_HW - 0.42, 0.25)][::sx], M_DECK, zroad)
# the ramp foot: retaining walls down to the street while the deck is low
for sx in (-1, 1):
    strip(f"RampWall{sx}", RAMP_Y0 - 4.0, RAMP_Y0 + RAMP_LEN * 0.45,
          [(sx * (DECK_HW - 0.1), -8.0), (sx * DECK_HW, -8.0), (sx * DECK_HW, -0.2), (sx * (DECK_HW - 0.1), -0.2)][::sx],
          M_DECK, zroad)
# kicker: hazard-striped wedge, full width
kick_prof = [(-DECK_HW + 0.4, 0.0), (DECK_HW - 0.4, 0.0), (DECK_HW - 0.4, -1.2), (-DECK_HW + 0.4, -1.2)]
strip("Kicker", KICK_Y0, KICK_Y1, kick_prof, M_HAZ, zroad, step=0.5)
strip("DeckKickBase", KICK_Y0, KICK_Y1, SEC_DECK, M_DECK, lambda y: np.full_like(y, DECK_H))
# lane paint on the deck
dp = MB()
for y in np.arange(SEG1[0] + 6, SEG3[1], 12.0):
    if SEG1[1] - 2 < y < SEG3[0] + 2:
        continue
    z = float(zroad(np.array([y]))[0])
    dp.box((0.0, y, z + 0.035), (0.15, 4.0, 0.02))
    for sx in (-1, 1):
        dp.box((sx * (DECK_HW - 0.8), y, z + 0.035), (0.15, 12.0, 0.02))
fin(dp.obj("DeckPaint", M_PAINT))
# piers: in the median on the street, into the river beyond the city
pm = MB()
for y in np.arange(RAMP_Y0 + 60, END_Y, 34.0):
    if KICK_Y1 + 2 < y < LAND_Y0 - 2:
        continue
    z = float(zroad(np.array([y]))[0])
    if z < 5:
        continue
    base = 0.0 if y < CITY_END else WATER_Z
    pm.box((0, y, (base + z - 1.3) / 2), (2.2, 2.2, z - 1.3 - base))
    pm.box((0, y, z - 1.6), (7.0, 2.6, 0.7))
fin(pm.obj("Piers", M_DECK))
# deck lamps: poles on both parapets, emissive heads; a real light every other pole
lp = MB()
heads = MB()
for k, y in enumerate(np.arange(RAMP_Y0 + 40, END_Y, 36.0)):
    if KICK_Y0 - 6 < y < LAND_Y0 + 4:
        continue
    z = float(zroad(np.array([y]))[0])
    sx = 1 if k % 2 else -1
    lp.box((sx * (DECK_HW + 0.2), y, z + 4.5), (0.2, 0.2, 9.0))
    lp.box((sx * (DECK_HW - 0.8), y, z + 8.95), (2.2, 0.14, 0.12))
    heads.box((sx * (DECK_HW - 1.7), y, z + 8.82), (0.7, 0.36, 0.1))
    if k % 2 == 0:
        ld = bpy.data.lights.new(f"DeckLamp{k}", "SPOT")
        ld.energy = 1900.0
        ld.color = blackbody(4200)
        ld.spot_size = math.radians(120)
        ld.spot_blend = 0.6
        ld.use_shadow = False
        lo = link(bpy.data.objects.new(ld.name, ld))
        lo.location = (sx * (DECK_HW - 1.7), y, z + 8.6)
        fin(lo)
fin(lp.obj("DeckPoles", M_POLE))
hh = fin(heads.obj("DeckLampHeads", M_LAMP))
no_shadow(hh)
# river + embankment + far shore skyline
wm = MB()
wm.box((0, (CITY_END + END_Y + 3000) / 2, WATER_Z), (6000, END_Y + 3000 - CITY_END, 0.2))
fin(wm.obj("River", M_WATER))
em = MB()
em.box((0, CITY_END + 1.0, (WATER_Z + 0.2) / 2), (900, 2.0, 0.2 - WATER_Z))
em.box((0, CITY_END + 0.5, 0.6), (900, 0.25, 1.0))
fin(em.obj("Embankment", M_CURB))
far = [MB() for _ in FACADES]
far_y = YR_LEAD[-1] + 900
for k in range(260):
    x = float(rng.uniform(-1600, 1600))
    y = far_y + float(rng.uniform(0, 500))
    h = float(15 + 150 * rng.random() ** 2.6)
    h = 3.6 * round(h / 3.6) + 1.2
    far[int(rng.choice([0, 1, 2, 3, 3, 4]))].box((x, y, WATER_Z + h / 2 + 8), (rng.uniform(16, 40), rng.uniform(16, 40), h))
for sx in (-1, 1):                                   # river banks left and right: skyline in every direction
    for k in range(320):
        x = sx * float(rng.uniform(420, 1700))
        y = float(rng.uniform(CITY_END - 300, far_y + 200))
        h = float(12 + 120 * rng.random() ** 2.8)
        h = 3.6 * round(h / 3.6) + 1.2
        far[int(rng.choice([0, 1, 2, 3, 3, 4]))].box((x, y, WATER_Z + h / 2 + 6), (rng.uniform(16, 40), rng.uniform(16, 40), h))
for k, m_ in enumerate(far):
    if m_.v:
        ob = fin(m_.obj(f"FarShore{k}", FACADES[k]))
        ob["bid"] = float(900 + k)
        ob["bh"] = 80.0
# cable-stayed pylon over the river (between the city and the kicker)
PYL_Y = CITY_END + 0.55 * (KICK_Y0 - CITY_END)
if KICK_Y0 - CITY_END > 150:
    pyl = MB()
    ptop = DECK_H + 72.0
    for sx in (-1, 1):
        pyl.beam((sx * 9.0, PYL_Y, WATER_Z), (sx * 6.6, PYL_Y, ptop), 2.6)
    pyl.box((0, PYL_Y, DECK_H - 2.0), (19.0, 3.0, 2.0))
    pyl.box((0, PYL_Y, ptop - 6.0), (14.0, 2.6, 2.0))
    fin(pyl.obj("Pylon", M_DECK))
    cab = MB()
    for sx in (-1, 1):
        for d in np.arange(-150, 151, 15):
            if abs(d) < 10:
                continue
            ya = PYL_Y + d
            if not (CITY_END + 5 < ya < KICK_Y0 - 2):
                continue
            zt = ptop - 8 - 40 * (1 - abs(d) / 150)
            cab.beam((sx * 6.9, PYL_Y + np.sign(d) * 1.0, zt), (sx * (DECK_HW - 0.1), ya, DECK_H + 0.6), 0.09)
    fin(cab.obj("Cables", M_STEEL))
    pb = cube_obj("PylonBeacon", M_BEACON, loc=(0, PYL_Y, ptop + 0.6), scale=(0.4, 0.4, 0.4))
    no_shadow(fin(pb))
    set_color_keys(pb, np.stack([blink, blink * 0.05, blink * 0.02], 1))
# BRIDGE OUT: cones, amber flashers, signs, barriers on both broken edges
cm = MB()
for y in np.arange(KICK_Y0 - 140, KICK_Y0 - 4, 6.0):
    for sx in (-1, 1):
        prism(cm, (sx * (DECK_HW - 0.9), y, DECK_H), 0.22, 0.7, n=8)
fin(cm.obj("Cones", M_CONE))
AMBERS = []
for k, y in enumerate(np.arange(KICK_Y0 - 120, KICK_Y0 + 1, 10.0)):
    for sx in (-1, 1):
        a = cube_obj(f"Amber{k}{sx}", M_AMBER, loc=(sx * (DECK_HW - 0.25), y, DECK_H + 1.25), scale=(0.14, 0.14, 0.14))
        no_shadow(fin(a))
        AMBERS.append((a, k))
for sx in (-1, 1):
    sb = MB()
    sb.box((sx * (DECK_HW - 0.2), KICK_Y0 - 60, DECK_H + 1.4), (0.12, 0.12, 2.8))
    sb.box((sx * (DECK_HW - 1.6), KICK_Y0 - 60, DECK_H + 3.2), (3.2, 0.12, 1.4))
    fin(sb.obj(f"OutSignPanel{sx}", M_PANEL))
    t = text_obj(f"OutSign{sx}", "BRIDGE OUT", 0.62, M_AMBER, (sx * (DECK_HW - 1.6), KICK_Y0 - 60.08, DECK_H + 3.0),
                 (math.pi / 2, 0, 0))
    no_shadow(fin(t))
    AMBERS.append((t, 99))
    bar = MB()
    bar.box((sx * (DECK_HW - 1.4), KICK_Y0 - 3, DECK_H + 0.9), (2.4, 0.15, 0.3))
    bar.box((sx * (DECK_HW - 1.4), LAND_Y0 + 1.0, DECK_H - LAND_DROP + 0.9), (2.4, 0.15, 0.3))
    for yb, zb in ((KICK_Y0 - 3, DECK_H), (LAND_Y0 + 1.0, DECK_H - LAND_DROP)):
        for dx in (-0.9, 0.9):
            bar.box((sx * (DECK_HW - 1.4) + dx, yb, zb + 0.45), (0.1, 0.1, 0.9))
    fin(bar.obj(f"Barrier{sx}", M_HAZ))
amb_blink = lambda k: np.where(((BEAT_PH * 2 + k * 0.5) % 1.0) < 0.5, 1.0, 0.08)   # noqa: E731
for a, k in AMBERS:
    lv = np.full(N, 0.9) if k == 99 else amb_blink(k)
    set_color_keys(a, lv)
for o in FINALE:
    show(o, IDX >= F_C)
print(f"[police] finale objects: {len(FINALE)}")

# =====================================================================================
# 8. THE CARS (real models; normalised: nose +Y, centred, wheels on z=0)
# =====================================================================================


def islands(me):
    """Vertex island labels (connected components over edges), numpy union-find by label propagation."""
    nv = len(me.vertices)
    ed = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    lab = np.arange(nv)
    while True:
        a, b_ = lab[ed[:, 0]], lab[ed[:, 1]]
        mn = np.minimum(a, b_)
        new = lab.copy()
        np.minimum.at(new, ed[:, 0], mn)
        np.minimum.at(new, ed[:, 1], mn)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def sub_mesh(o, keep_mask, name):
    """New object from the faces of o whose vertices are all in keep_mask."""
    import bmesh
    bm = bmesh.new()
    try:
        bm.from_mesh(o.data)
        bm.verts.ensure_lookup_table()
        kill = [v for v in bm.verts if not keep_mask[v.index]]
        bmesh.ops.delete(bm, geom=kill, context="VERTS")
        if len(bm.verts) == 0:
            return None
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
    finally:
        bm.free()
    for m in o.data.materials:
        me.materials.append(m)
    return link(bpy.data.objects.new(name, me))


def import_car(slug, tag, yaw_deg, axles, wz, wr, keep_mats=(), lamp_objs=None):
    """Import a car, normalise it (nose +Y), and rig it: rig empty (yaw/position) → body empty (pitch/roll),
    plus one spinning object per AXLE holding every part that lies inside that axle's wheel cylinders
    (tyre, rim, disc, hub, nuts — found per connected island, so meshes spanning both axles are split).
    keep_mats: materials that stay on the body even inside a wheel (brake calipers don't spin)."""
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
    lamp_names = {}
    for o in meshes:
        mw = R @ centre @ o.matrix_world
        me = o.data.copy()
        me.transform(mw)
        o.data = me
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
        if lamp_objs and o.name in lamp_objs:
            lamp_names[o.name] = lamp_objs[o.name]
    bpy.context.view_layer.update()
    pad = 0.05

    def corner_of(mn, mx):
        for ai, ay in enumerate(axles):
            if (mn[1] > ay - wr - pad and mx[1] < ay + wr + pad and mn[2] > wz - wr - pad and mx[2] < wz + wr + pad
                    and min(abs(mn[0]), abs(mx[0])) > 0.35 and np.sign(mn[0]) == np.sign(mx[0])):
                return ai
        return -1

    parts = {ai: [] for ai in range(len(axles))}
    body_parts = []
    for o in meshes:
        nv = len(o.data.vertices)
        if nv == 0:
            body_parts.append(o)
            continue
        co = np.empty(nv * 3, np.float64)
        o.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mn, mx = co.min(0), co.max(0)
        kept = any(m is not None and m.name.split(".")[0] in keep_mats for m in o.data.materials)
        if kept:
            body_parts.append(o)
            continue
        c = corner_of(mn, mx)
        if c >= 0:
            parts[c].append(o)
            continue
        touches = any(mn[1] < ay + wr and mx[1] > ay - wr for ay in axles) and mn[2] < wz + wr
        if not touches or mx[2] - mn[2] > 1.2:
            body_parts.append(o)
            continue
        lab = islands(o.data)
        cls = np.full(nv, -1)
        for u in np.unique(lab):
            sel = lab == u
            cls[sel] = corner_of(co[sel].min(0), co[sel].max(0))
        if (cls < 0).all():
            body_parts.append(o)
            continue
        for ai in range(len(axles)):
            if (cls == ai).any():
                sub = sub_mesh(o, cls == ai, f"{o.name}_ax{ai}")
                if sub:
                    parts[ai].append(sub)
        if (cls < 0).any():
            sub = sub_mesh(o, cls < 0, f"{o.name}_body")
            if sub:
                body_parts.append(sub)
        bpy.data.objects.remove(o, do_unlink=True)
    for o in body_parts:
        o.parent = body
    wheels = []
    for ai, ay in enumerate(axles):
        ax = link(bpy.data.objects.new(f"Axle{tag}{ai}", None), rig)
        ax.location = (0, ay, wz)
        for o in parts[ai]:
            o.data.transform(Matrix.Translation(Vector((0, -ay, -wz))))
            o.parent = ax
        wheels.append(ax)
        print(f"[police] car {tag} axle {ai} @ y {ay:+.2f}: {len(parts[ai])} parts "
              f"({', '.join(sorted(p.name for p in parts[ai])[:8])})")
    for o in new:
        try:
            if o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        except ReferenceError:
            pass
    print(f"[police] car {tag}: size {np.round(hi - lo, 2)}")
    return rig, body, wheels, lamp_names


HL_NTS = []


def lamp_material(o, color, strength, keyed):
    m, nt = new_mat(f"Lamp_{o.name}")
    e = nd(nt, "ShaderNodeEmission", inp={"Color": tuple(color) + (1,)})
    if keyed:
        v = value_node(nt, "HL", 1.0)
        lk(nt, M(nt, "MULTIPLY", v.outputs[0], strength), e.inputs["Strength"])
        HL_NTS.append(nt)
    else:
        e.inputs["Strength"].default_value = strength
    lk(nt, e.outputs[0], nd(nt, "ShaderNodeOutputMaterial").inputs["Surface"])
    o.data.materials.clear()
    o.data.materials.append(m)


# Corvette: axles at model y -1.18 (front) / +1.45 (rear) → after the 180° yaw +1.18 / -1.45
LEAD, LEAD_BODY, LEAD_WHEELS, _ = import_car("sf_sports_car", "Lead", ARGS.lead_yaw, axles=(1.18, -1.45),
                                             wz=0.32, wr=0.335, keep_mats={"Material_369"})
POL, POL_BODY, POL_WHEELS, _ = import_car("sf_police_suv", "Police", ARGS.police_yaw, axles=(1.245, -1.325),
                                          wz=0.36, wr=0.385)
for o in bpy.data.objects:
    base = o.name.split("_ax")[0].split("_body")[0]
    if o.type != "MESH" or o.parent not in (LEAD_BODY, POL_BODY):
        continue
    if o.parent == LEAD_BODY and base == "Object_25":
        lamp_material(o, (0.9, 0.95, 1.0), 40.0, True)                # Corvette DRL / headlight strips
    elif o.parent == LEAD_BODY and base in ("Object_21", "Object_23"):
        lamp_material(o, (1.0, 0.02, 0.01), 25.0, False)              # Corvette tail lights
    elif o.parent == POL_BODY and base == "Object_10":
        lamp_material(o, (0.9, 0.95, 1.0), 35.0, True)                # police headlight lenses


def car_lights(rig, tag, nose_y, tail_y, width, head_z, tail_z, head_energy):
    heads = []
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
        heads.append(ld)
        lt = bpy.data.lights.new(f"Tail{tag}{sx}", "POINT")
        lt.energy = 25.0
        lt.color = (1.0, 0.02, 0.01)
        lt.shadow_soft_size = 0.15
        lt.use_shadow = False
        lo2 = link(bpy.data.objects.new(lt.name, lt), rig)
        lo2.location = (sx * width, tail_y - 0.15, tail_z)
    return heads


HEADS = car_lights(LEAD, "Lead", 2.25, -2.3, 0.68, 0.62, 0.78, 900.0) + \
    car_lights(POL, "Police", 2.2, -2.2, 0.72, 0.82, 0.95, 1100.0)
for ld in HEADS:
    bake(ld, "energy", ld.energy * HEADL)
for nt in HL_NTS:
    bake(nt, value_path("HL"), HEADL)
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
# NITRO flames from the Corvette's quad exhaust (two cones, one per tailpipe pair) + their light
M_FLAME, _fl = new_mat("NitroFlame")
_tc = nd(_fl, "ShaderNodeTexCoord")
_sp = nd(_fl, "ShaderNodeSeparateXYZ")
lk(_fl, _tc.outputs["Object"], _sp.inputs[0])
_t = _sp.outputs[2]                                       # 0 at the nozzle → 1 at the tip
_nz = nd(_fl, "ShaderNodeTexNoise", inp={"Scale": 3.0, "Detail": 3.0})
_w = nd(_fl, "ShaderNodeTexCoord")
_oi = nd(_fl, "ShaderNodeObjectInfo")
_nv = nd(_fl, "ShaderNodeCombineXYZ")
lk(_fl, M(_fl, "MULTIPLY", _sp.outputs[0], 6.0), _nv.inputs[0])
lk(_fl, M(_fl, "SUBTRACT", M(_fl, "MULTIPLY", _t, 3.0), M(_fl, "MULTIPLY", _oi.outputs["Random"], 40.0)), _nv.inputs[1])
lk(_fl, M(_fl, "MULTIPLY", _sp.outputs[1], 6.0), _nv.inputs[2])
lk(_fl, _nv.outputs[0], _nz.inputs["Vector"])
_ramp = nd(_fl, "ShaderNodeValToRGB")
_ramp.color_ramp.elements[0].position = 0.0
_ramp.color_ramp.elements[0].color = (0.45, 0.6, 1.0, 1)
_e2 = _ramp.color_ramp.elements.new(0.22)
_e2.color = (0.9, 0.85, 1.0, 1)
_e3 = _ramp.color_ramp.elements.new(0.45)
_e3.color = (1.0, 0.45, 0.08, 1)
_ramp.color_ramp.elements[-1].position = 1.0
_ramp.color_ramp.elements[-1].color = (0.6, 0.06, 0.01, 1)
lk(_fl, _t, _ramp.inputs["Fac"])
_em = nd(_fl, "ShaderNodeEmission")
lk(_fl, _ramp.outputs["Color"], _em.inputs["Color"])
_str = M(_fl, "MULTIPLY", M(_fl, "POWER", M(_fl, "SUBTRACT", 1.0, _t, clamp=True), 1.3), 24.0)
lk(_fl, _str, _em.inputs["Strength"])
_al = M(_fl, "MULTIPLY", M(_fl, "POWER", M(_fl, "SUBTRACT", 1.0, _t, clamp=True), 0.8),
        M(_fl, "ADD", 0.35, M(_fl, "MULTIPLY", _nz.outputs["Fac"], 1.0)), clamp=True)
_tr = nd(_fl, "ShaderNodeBsdfTransparent")
_mx = nd(_fl, "ShaderNodeMixShader")
lk(_fl, _al, _mx.inputs[0])
lk(_fl, _tr.outputs[0], _mx.inputs[1])
lk(_fl, _em.outputs[0], _mx.inputs[2])
lk(_fl, _mx.outputs[0], nd(_fl, "ShaderNodeOutputMaterial").inputs["Surface"])
try:
    M_FLAME.surface_render_method = "BLENDED"
except Exception:
    pass
FLAMES = []
prof = [(0.0, 0.0)] + [(0.16 * (1 - s) ** 0.55 * (1 + 0.35 * math.sin(math.pi * s * 0.9)), s) for s in np.linspace(0.02, 1.0, 14)]
for k, x in enumerate((-0.13, 0.13)):
    fo = lathe(f"Flame{k}", prof, segs=16, mat=M_FLAME, parent=LEAD_BODY, loc=(x, -2.36, 0.31))
    fo.rotation_euler = (math.radians(90), 0, 0)              # +Z (lathe axis) → -Y (out the back)
    no_shadow(fo)
    fo.visible_shadow = False
    FLAMES.append(fo)
FL_LIGHT = bpy.data.lights.new("NitroLight", "POINT")
FL_LIGHT.color = (1.0, 0.45, 0.15)
FL_LIGHT.shadow_soft_size = 0.9
FL_LIGHT.use_shadow = False
FL_LO = link(bpy.data.objects.new("NitroLight", FL_LIGHT), LEAD_BODY)
FL_LO.location = (0, -3.2, 0.75)
# helicopter spotlight following the lead car
HELI = bpy.data.lights.new("HeliSpot", "SPOT")
HELI.energy = 2.4e6
HELI.color = blackbody(6200)
HELI.spot_size = math.radians(7.5)
HELI.spot_blend = 0.35
HELI.use_shadow = True
HELI_OB = link(bpy.data.objects.new("HeliSpot", HELI))

# =====================================================================================
# 9. MOTION (true → rendered: minus K_WRAP; finale heights from the route profile)
# =====================================================================================


def place_car(rig, body, wheels, X, Y, Z, DX, speed, tag, wheel_r):
    yaw = -np.arctan2(DX, np.maximum(speed, 1.0))                     # heading change in lane changes
    loc = np.stack([X, Y - K_WRAP, Z], 1)
    bake_vec(rig, "location", loc)
    vz = np.gradient(Z) * FPS
    climb = np.arctan2(vz, np.maximum(speed, 1.0))
    if tag == "Lead":                                                  # nose follows the arc in the air,
        climb = np.where(AIR, climb * 0.8, climb)                     # a touch less than the full angle
    climb = gsmooth(climb, 1.5)
    bake_vec(rig, "rotation_euler", np.stack([climb, np.zeros(N), yaw], 1))
    acc = np.gradient(gsmooth(speed, 3)) * FPS
    pitch = np.radians(np.clip(-acc * 0.12, -3.5, 3.5) + 0.6 * KICK * np.maximum(HOT, 0.4))   # squat/dive + kick bounce
    roll = np.radians(np.clip(np.gradient(DX) * FPS * 0.8, -4, 4))
    bob = 0.012 * np.sin(IDX * 1.9 + (0 if tag == "Lead" else 1.3)) + 0.025 * KICK * np.maximum(HOT, 0.4)
    if tag == "Lead":                                                  # landing: compress + rebound
        land = np.zeros(N)
        k = np.arange(N - F_LAND)
        land[F_LAND:] = np.exp(-k / 6.0) * np.cos(k * 0.55)
        bob = np.where(AIR, 0.0, bob) - 0.16 * land
        pitch = pitch - np.radians(3.0) * land
    bake_vec(body, "rotation_euler", np.stack([pitch, roll, np.zeros(N)], 1))
    bake_vec(body, "location", np.stack([np.zeros(N), np.zeros(N), bob], 1))
    # wheels roll FORWARD (−X rotation for +Y travel); the visible step is capped per frame so spokes
    # never alias into a backwards strobe at 200 km/h (the real 5 rad/frame would freeze or reverse them)
    step = np.minimum(speed / FPS / wheel_r, ARGS.wheel_step)
    if tag == "Lead":
        step = np.where(AIR, step * 1.0, step)
    ang = -np.cumsum(step)
    for w in wheels:
        bake(w, "rotation_euler", ang, index=0)


SPEED_POL = np.gradient(Y_POL) * FPS
place_car(LEAD, LEAD_BODY, LEAD_WHEELS, X_LEAD, Y_LEAD, Z_LEAD, DX_LEAD, SPEED, "Lead", 0.33)
place_car(POL, POL_BODY, POL_WHEELS, X_POL, Y_POL, Z_POL, DX_POL, np.maximum(SPEED_POL, 0.0), "Police", 0.375)
# flames: length/width flicker per frame, hidden when the nitro is off
fl_on = NITRO > 0.04
for k, fo in enumerate(FLAMES):
    fk = np.array([hash01(i, 31 + k) for i in range(N)])
    ln = np.minimum(NITRO * (1.0 + 0.4 * fk + 0.3 * HAT + 0.3 * KICK * BIG) * 1.45, 2.2)
    wd = np.sqrt(np.maximum(NITRO, 0)) * (0.85 + 0.3 * fk)
    bake_vec(fo, "scale", np.stack([wd, wd, np.maximum(ln, 0.001)], 1))
    show(fo, fl_on)
bake(FL_LIGHT, "energy", 160.0 * NITRO * (0.8 + 0.4 * HAT))
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
    bake(ld, "energy", lvl * 2000.0, const=True)                 # 9000 flooded the whole street red


def proto_instance(name, tag, coll):
    p = PROTOS[name]
    rig = bpy.data.objects.new(tag, None)
    coll.objects.link(rig)
    body = bpy.data.objects.new(tag + "_body", None)
    body.instance_type = "COLLECTION"
    body.instance_collection = p.body
    coll.objects.link(body)
    body.parent = rig
    axles = []
    for ai, (y, z, r, col) in enumerate(p.axles):
        a = bpy.data.objects.new(f"{tag}_ax{ai}", None)
        a.instance_type = "COLLECTION"
        a.instance_collection = col
        coll.objects.link(a)
        a.parent = rig
        a.location = (0, y, z)
        axles.append(a)
    return rig, body, axles, p


# the rest of the police: their own models, lightbars strobing out of phase with each other
for k, c in enumerate(COPS):
    rig, body, axles, p = proto_instance(c["name"], f"Cop{k}", SC.collection)
    sp_k = np.maximum(np.gradient(c["Y"]) * FPS, 0.0)
    wr = float(np.mean([a[2] for a in p.axles])) if p.axles else 0.35
    place_car(rig, body, axles, c["X"], c["Y"], c["Z"], np.gradient(c["X"]) * FPS, sp_k, f"Cop{k}", wr)
    ph_k = BEAT_PH + 0.25 * (k + 1)
    af = np.floor(ph_k * 4).astype(int) % 2
    asl = np.floor(ph_k * 2).astype(int) % 2
    bu = (((ph_k * 4) % 1.0) < 0.45).astype(float)
    r_on = np.where(HOT > 0.5, (af == 0) * bu, (asl == 0) * 1.0)
    b_on = np.where(HOT > 0.5, (af == 1) * bu, (asl == 1) * 1.0)
    vis = c["vis"]
    for nm_, mat, x, on in (("r", M_RED, -0.3, r_on), ("b", M_BLUE, 0.3, b_on)):
        bo = cube_obj(f"Cop{k}Bar{nm_}", mat, loc=(x, 0.0, p.roof_z + 0.08), scale=(0.2, 0.08, 0.05))
        bo.parent = body
        no_shadow(bo)
        lv = np.clip(on * (0.8 + 0.6 * HAT) + BIG * KICK * 1.5, 0, 3)
        set_color_keys(bo, np.stack([lv, lv, lv], 1), const=True)
        show(bo, vis)
    ld = bpy.data.lights.new(f"Cop{k}Light", "POINT")
    ld.shadow_soft_size = 0.3
    ld.use_shadow = False
    lo = link(bpy.data.objects.new(ld.name, ld), body)
    lo.location = (0, 0, p.roof_z + 0.45)
    red = r_on >= b_on
    bake_vec(ld, "color", np.stack([np.where(red, 1.0, 0.08), np.where(red, 0.05, 0.25), np.where(red, 0.03, 1.0)], 1),
             const=True)
    bake(ld, "energy", 1400.0 * np.maximum(r_on, b_on) * vis, const=True)
    for o in [body] + axles:
        show(o, vis)
hp = np.stack([X_LEAD + 14.0 + 3 * np.sin(IDX * 0.011), YR_LEAD - 26.0, Z_LEAD + 62.0], 1)
bake_vec(HELI_OB, "location", hp)
tgt = np.stack([X_LEAD, YR_LEAD + 2.0, Z_LEAD], 1) + np.stack(
    [1.2 * np.sin(IDX * 0.07), 1.5 * np.sin(IDX * 0.053 + 1), np.zeros(N)], 1)
d = tgt - hp
rot = []
for v in d:
    q = Vector(v).to_track_quat("-Z", "Y").to_euler()
    rot.append((q.x, q.y, q.z))
bake_vec(HELI_OB, "rotation_euler", np.array(rot))
bake(HELI, "energy", 2.4e6 * (0.85 + 0.15 * KICK * HOT))

# =====================================================================================
# 10. CAMERA (shot functions work in TRUE y; the K_WRAP shift is subtracted at the end)
# =====================================================================================
CAM_D = bpy.data.cameras.new("Cam")
CAM_D.sensor_width = 36.0
CAM_D.clip_start = 0.2
CAM_D.clip_end = 4000.0
CAM_D.dof.use_dof = True
CAM_D.dof.aperture_blades = 7
CAM = link(bpy.data.objects.new("Cam", CAM_D))
SC.camera = CAM


def lead(i, dy=0.0, dx=0.0, z=0.0):
    return Vector((X_LEAD[i] + dx, Y_LEAD[i] + dy, Z_LEAD[i] + z))


def pol(i, dy=0.0, dx=0.0, z=0.0):
    return Vector((X_POL[i] + dx, Y_POL[i] + dy, Z_POL[i] + z))


def true_y(yr, i):
    return yr + K_WRAP[i]


def shot(kind, i, s0, s1):
    t = (i - s0) / max(s1 - s0 - 1, 1)
    side = 1 if hash01(s0, 7) > 0.5 else -1
    elevated = s0 >= F_C                     # finale: cameras stay on the deck (ramp walls, parapets)
    if kind == "chase_high":                                   # behind the cruiser, lights in the foreground
        cam = pol(i, -14.0 - 4 * t, 1.5 * side, 4.2)
        return cam, lead(i, 0, 0, 0.8) * 0.65 + pol(i, 0, 0, 1.0) * 0.35, 32, 11
    if kind == "chase_lead":                                   # behind the Corvette alone (flames)
        cam = lead(i, -12.0 - 3 * t, 1.2 * side, 3.2)
        return cam, lead(i, 2.0, 0, 0.6), 35, 8
    if kind == "rear_low":                                     # low on the road right behind the exhaust
        cam = lead(i, -6.5 + 1.0 * t, 1.5 * side, 0.85)
        return cam, lead(i, -0.5, 0, 0.45), 26, 8
    if kind == "front_reverse":                                # ahead of the fleeing car, looking back
        cam = lead(i, 11.0 + 2 * t, -1.2 * side, 1.15)
        return cam, lead(i, -6.0, 0, 0.7), 45, 5.6
    if kind in ("side_low", "side_low_lead"):                  # tracking alongside, low
        if elevated:                                           # on the deck: from the other lane (parapets)
            cam = lead(i, 1.5 - 3 * t, -3.4 * (1 if X_LEAD[s0] >= 0 else -1), 0.9)
            return cam, lead(i, -0.6, 0, 0.6), 22, 5.6
        cam = lead(i, 1.5 - 3 * t, 4.6 * side, 0.55)
        return cam, lead(i, -3.0 if kind == "side_low" else 0.5, 0, 0.6), 24, 5.6
    if kind == "bumper":                                       # on the sports car's rear, looking back at the cruiser
        cam = lead(i, -2.55, 0.35 * side, 0.62)
        return cam, pol(i, 0, 0, 0.9), 18, 8
    if kind == "flyby":                                        # static on the roadside; they blast past
        mid = (s0 + s1) // 2
        if elevated:
            cam = Vector((side * (DECK_HW - 0.7), Y_LEAD[mid] + 2.0, Z_LEAD[mid] + 1.1))
        else:
            cam = Vector((side * (ROAD_W / 2 + 2.2), Y_LEAD[mid] + 2.0, 1.1))
        return cam, lead(i, 0, 0, 0.7) * 0.7 + pol(i, 0, 0, 0.9) * 0.3, 24, 8
    if kind == "heli_ramp":                                    # high and ahead: the ramp rising out of the avenue
        cam = Vector((-6.0, true_y(RAMP_Y0 + 150.0, i), 44.0 - 6 * t)) + Vector((0, 40.0 * t, 0))
        return cam, lead(i, 25.0, 0, 0) * 0.6 + Vector((0, true_y(RAMP_Y0 + 60, i), 8.0)) * 0.4, 30, 11
    if kind == "heli_bridge":                                  # wide: city behind, river and pylon ahead
        cam = lead(i, -34.0 + 8 * t, 7.0, 13.0)
        return cam, lead(i, 30.0, 0, -2.0), 24, 11
    if kind == "kicker_view":                                  # hovering over the gap, the lip in front
        cam = Vector((-2.2, true_y(KICK_Y1 + 9.0, i), DECK_H + 3.0))
        return cam, lead(i, 0, 0, 0.6), 50 - 14 * t, 8
    if kind == "jump_side":                                    # a tracking drone beside the gap, at deck height:
        cam = Vector((X_LEAD[i] + 15.0, Y_LEAD[i] - 7.0, DECK_H + 1.2))   # the arc reads against the skyline
        return cam, lead(i, 2.0, 0, 0.4), 30, 11
    if kind == "land_chase":                                   # drone behind + above: the touchdown
        cam = lead(i, -13.0, 2.5, 4.5)
        return cam, lead(i, 12.0, 0, -0.5), 28, 11
    if kind == "police_edge":                                  # behind the stopped cruiser at the gap
        cam = pol(i, -7.5, 1.6, 3.0)
        return cam, lead(i, 0, 0, 0) * 0.5 + pol(i, 30.0, 0, 0) * 0.5, 30, 11
    if kind == "xing":                                         # high on the corner of the red-light crossing
        cam = Vector((9.5, Y_XING + 5.0, 9.0))               # over the corner, inside the cross street
        tg = Vector((-2.0, Y_XING - 6.0, 0.5)) * 0.55 + lead(i, 0, 0, 0.5) * 0.45
        return cam, tg, 18, 11
    if kind == "xing_low":                                     # low on the cross street: the cruisers follow
        cam = Vector((-10.4, Y_XING - 9.5, 1.9))             # on the far corner's kerb, clear of both lanes
        return cam, Vector((0.0, Y_XING, 1.0)) * 0.5 + pol(i, 0, 0, 0.8) * 0.5, 22, 8
    if kind == "heli_lead":
        cam = lead(i, -50.0 + 10 * t, 12.0 * side, 36.0)
        return cam, lead(i, 10.0, 0, 0), 40, 11
    if kind == "far":                                          # locked off on the deck: it leaves us
        mid = s0
        cam = Vector((1.5, Y_LEAD[mid] - 5.0, Z_LEAD[mid] + 1.6))
        return cam, lead(i, 0, 0, 0.6), 70, 11
    # heli: 60 m up, trailing and off to the side, slow drift
    cam = lead(i, -60.0 + 10 * t, 3.0 * side, 64.0)
    return cam, lead(i, 6.0, 0, 0), 55, 11


def shake(i, amp):
    k = int(TL["T_kidx"][i])
    j = Vector((hash01(k, 1) - 0.5, hash01(k, 2) - 0.5, hash01(k, 3) - 0.5)) * 2 * KICK[i]
    r = Vector((math.sin(i * 1.7) + 0.5 * math.sin(i * 3.1), math.sin(i * 1.3 + 1), math.sin(i * 1.9 + 2)))
    return (j + 0.15 * r) * amp


AMP = {"bumper": 0.06, "side_low": 0.05, "side_low_lead": 0.05, "front_reverse": 0.05, "chase_high": 0.08,
       "chase_lead": 0.08, "rear_low": 0.07, "flyby": 0.03, "heli": 0.25, "heli_ramp": 0.15, "heli_bridge": 0.15,
       "kicker_view": 0.04, "xing": 0.03, "xing_low": 0.04, "jump_side": 0.05, "land_chase": 0.1, "police_edge": 0.06, "heli_lead": 0.2, "far": 0.02}
cam_loc = np.zeros((N, 3))
cam_rot = np.zeros((N, 3))
cam_lens = np.zeros(N)
cam_focus = np.zeros(N)
cam_fstop = np.zeros(N)
bounds = [s for s, _ in SHOTS] + [N]
prev = None
QA = []
land_shake = np.zeros(N)
land_shake[F_LAND:F_LAND + 20] = np.exp(-np.arange(min(20, N - F_LAND)) / 5.0) * 4.0
for k, (s0, kind) in enumerate(SHOTS):
    s1 = bounds[k + 1]
    amp = AMP[kind]
    seen = 0
    for i in range(s0, s1):
        cam, tgt, lens, fs = shot(kind, i, s0, s1)
        a = amp * (0.4 + 1.2 * HOT[i] + 1.0 * BIG[i] + land_shake[i])
        cam = cam + shake(i, a)
        tgt = tgt + shake(i, a * 1.6)
        lens = lens * (1.0 - 0.12 * (NITRO[i] if kind in ("chase_lead", "rear_low") else 0.0))
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
        # QA: is the lead car on screen?
        Rm = e.to_matrix()
        v = Rm.transposed() @ (lead(i, 0, 0, 0.6) - off - cam)
        if v.z < -0.5:
            hx = 18.0 / lens
            if abs(v.x / -v.z) < hx and abs(v.y / -v.z) < hx * 9 / 16:
                seen += 1
    QA.append((s0, s1, kind, seen / max(s1 - s0, 1)))
bake_vec(CAM, "location", cam_loc)
bake_vec(CAM, "rotation_euler", cam_rot)
bake(CAM_D, "lens", cam_lens)
bake(CAM_D, "dof.focus_distance", cam_focus)
bake(CAM_D, "dof.aperture_fstop", cam_fstop)


def sparse_bake(idb, path, index, vals, a, b):
    """Keys only where the motion bends (linear runs collapse to their ends) + held keys at wraps."""
    v = np.asarray(vals, np.float64)[a:b + 1]
    n = len(v)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    if n > 2:
        keep[1:-1] = np.abs(v[2:] - 2 * v[1:-1] + v[:-2]) > 1e-6
    const = np.zeros(n, bool)
    for c_ in WRAPS:
        if a < c_ <= b:
            keep[c_ - 1 - a] = keep[c_ - a] = True
            const[c_ - 1 - a] = True
    ix = np.nonzero(keep)[0]
    fc = H._fcurve(idb, path, index)
    kp = fc.keyframe_points
    kp.clear()
    kp.add(len(ix))
    co = np.empty(2 * len(ix), np.float32)
    co[0::2] = a + ix + 1
    co[1::2] = v[ix]
    kp.foreach_set("co", co)
    kp.foreach_set("interpolation", np.where(const[ix], H.CONST, H.LIN).astype(np.int32))
    fc.update()


# the lens never sits inside a traffic car: drop any car the camera would pass through
cam_true_y = cam_loc[:, 1] + K_WRAP
keep, n_cam = [], 0
for c in TRAF:
    hx, hy = half(c)
    zc = zprof(c["Y"] - KF) if c["deck"] else np.zeros(N)
    hit = c["vis"] & (np.abs(cam_loc[:, 0] - c["X"]) < hx + 1.0) & (np.abs(cam_true_y - c["Y"]) < hy + 1.0) & \
        (cam_loc[:, 2] < zc + 3.5)
    if hit.any():
        n_cam += 1
        continue
    keep.append(c)
TRAF = keep
TRAF_COLL = bpy.data.collections.new("Traffic")
SC.collection.children.link(TRAF_COLL)
for t_i, c in enumerate(TRAF):
    rig, body, axles, p = proto_instance(c["kind"], f"Traf{t_i:03d}", TRAF_COLL)
    vis = c["vis"]
    a = int(np.argmax(vis))
    b = int(N - 1 - np.argmax(vis[::-1]))
    Yr = c["Y"] - K_WRAP
    Z = zprof(Yr) if c["deck"] else np.zeros(N)
    for k_, arr in enumerate((c["X"], Yr, Z)):
        sparse_bake(rig, "location", k_, arr, a, b)
    if c["axis"] == "x":
        rig.rotation_euler = (0.0, 0.0, -c["dirn"] * math.pi / 2)        # nose +Y → ±X
    if c["v"] > 0:
        ang = -ARGS.wheel_step * (IDX - a).astype(np.float64)
        for ax in axles:
            sparse_bake(ax, "rotation_euler", 0, ang, a, b)
    for o in [body] + axles:
        show(o, vis)
kinds = {}
for c in TRAF:
    kinds[c["tag"]] = kinds.get(c["tag"], 0) + 1
print(f"[police] traffic placed: {len(TRAF)} ({kinds}); {n_cam} dropped for the camera")
for s0, s1, kind, frac in QA:
    if frac < 0.6 and kind not in ("police_edge", "bumper"):
        print(f"[police][QA] shot {s0}-{s1} {kind}: lead on screen only {frac:.0%}")
inside = [i for i in range(F_C) if abs(cam_loc[i][0]) > XF - 0.2 and cam_loc[i][2] < 30]
if inside:
    print(f"[police][QA] {len(inside)} frames with the camera inside the building line, e.g. {inside[:5]}")

# =====================================================================================
# 11. RENDER SETTINGS + COMPOSITOR
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
WORLD.mist_settings.depth = 1100.0
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
fog.inputs[7].default_value = (0.022, 0.024, 0.032, 1)
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
SC["finale"] = f"F_C={F_C};F_JUMP={F_JUMP};F_LAND={F_LAND};POL_STOP={POL_STOP}"
os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(ARGS.out), compress=True)
print(f"[police] saved {ARGS.out}  frames 1..{N}  shots {len(SHOTS)}")
