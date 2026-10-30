"""Sands — the shot table (pure numpy; lib/camera.Shot needs no bpy). Cuts land on DOWNBEATS (S.snap_fr); every shot
is a smooth, slow move — the camera NEVER reacts to the music. Cube at the origin, ~18 m up; the sun sits low toward
−X/−Y (az-rot 248°), so cameras on the +X/+Y side look INTO the sunset (backlit dunes, silhouettes), cameras to the
south/north get the long raking side light across the ripples.
"""
import math

import numpy as np

from lib.camera import Shot

import terrain as TR

CUBE = np.array([0.0, 0.0, 18.8])          # = cube.ZC (asserted in the build)


def gz(x, y, h):
    """camera height h above the sand at (x, y)"""
    return float(TR.height(x, y)) + h


def _lerp(a, b, u):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * u


def _ease(u):
    return u * u * (3 - 2 * u)


def orbit(r, h, a0_deg, a1_deg, lens, tgt=CUBE, ground=False, tz=0.0):
    def fn(i, u, n):
        s = _ease(u / n) * 0.35 + (u / n) * 0.65               # mostly linear, soft start/end
        a = math.radians(a0_deg + (a1_deg - a0_deg) * s)
        x, y = tgt[0] + r * math.cos(a), tgt[1] + r * math.sin(a)
        z = gz(x, y, h) if ground else h
        return dict(cam=(x, y, z), tgt=(tgt[0], tgt[1], tgt[2] + tz), lens=lens)
    return fn


def move(c0, c1, lens0, lens1=None, tgt0=CUBE, tgt1=None, ground=False):
    """straight dolly c0 → c1 (ground=True: z is height above the sand), eased"""
    tgt1 = tgt0 if tgt1 is None else tgt1
    lens1 = lens0 if lens1 is None else lens1

    def fn(i, u, n):
        s = _ease(u / n) * 0.5 + (u / n) * 0.5
        c = _lerp(c0, c1, s)
        if ground:
            c[2] = gz(c[0], c[1], c[2])
        return dict(cam=tuple(c), tgt=tuple(_lerp(tgt0, tgt1, s)), lens=lens0 + (lens1 - lens0) * s)
    return fn


def los_clear(c, p, step=2.0, margin=0.3):
    """is the straight line c → p above the terrain everywhere (except the last 4 m)?"""
    c, p = np.asarray(c, float), np.asarray(p, float)
    L = np.linalg.norm(p - c)
    t = np.arange(step, max(L - 4.0, step), step) / L
    q = c[None] + (p - c)[None] * t[:, None]
    return bool(np.all(q[:, 2] > TR.height(q[:, 0], q[:, 1]) + margin))


def camel_setup(cam_az_deg=58.0, cam_r=150.0, cam_h=2.6, beyond=(120, 260), walk_s=16.0, speed=0.87):
    """Find a camel walk line BEYOND the cube as seen from the camel-shot camera, crossing the view, with the camel's
    hump visible (line of sight clear) along the whole walk. Returns (camera xyz, path a→b, the distance)."""
    a = math.radians(cam_az_deg)
    cx, cy = cam_r * math.cos(a), cam_r * math.sin(a)
    cam = np.array([cx, cy, gz(cx, cy, cam_h)])
    view = -np.array([math.cos(a), math.sin(a)])
    side = np.array([-view[1], view[0]])
    best = None
    for dist in np.arange(beyond[0], beyond[1], 5.0):
        for off in np.arange(-30.0, 30.1, 5.0):
            for sgn in (1, -1):
                p0 = np.array([0.0, 0.0]) + view * dist + side * off
                p1 = p0 + sgn * side * speed * walk_s
                ok = True
                for s in np.linspace(0, 1, 9):
                    p = p0 + (p1 - p0) * s
                    hump = np.array([p[0], p[1], float(TR.height(*p)) + 1.6])
                    if not los_clear(cam, hump):
                        ok = False
                        break
                if not ok:
                    continue
                # prefer: on higher ground (silhouette), near the view centre line, gentle slope
                hh = float(TR.height(*p0))
                sl = float(max(TR.slope(*p0), TR.slope(*p1)))
                score = hh - 0.15 * abs(off) - 40 * max(0, sl - 0.2) - 0.02 * dist
                if best is None or score > best[0]:
                    best = (score, p0, p1, dist)
    if best is None:
        raise RuntimeError("no visible camel path found")
    return cam, (best[1], best[2]), best[3]


def frame_both(cam, a, b, fill=0.62, sensor=36.0, aspect=16 / 9):
    """aim + lens so points a and b both sit inside the frame (their angular spread = `fill` of the frame width)"""
    cam, a, b = (np.asarray(v, float) for v in (cam, a, b))
    da, db = (a - cam) / np.linalg.norm(a - cam), (b - cam) / np.linalg.norm(b - cam)
    mid = (da + db) / np.linalg.norm(da + db)
    ang = math.acos(float(np.clip(da @ db, -1, 1)))
    half_w = 0.5 * ang / fill
    lens = 0.5 * sensor / math.tan(half_w)
    return cam + mid * 100.0, float(np.clip(lens, 24.0, 85.0))


def table(S, camel_cam):
    """→ list[Shot]. Section times come from S.sections (ear.json)."""
    sec = {s["label"]: s for s in S.sections}
    f = lambda t: S.snap_fr(t)      # noqa: E731  downbeat frame
    bar = 4 * S.period
    st = lambda k: sec[k]["start"]  # noqa: E731
    cc, camel_mid = camel_cam
    look_camel, lens_camel = frame_both(cc, CUBE, camel_mid)
    # Zeke's sections (ear.json 10-30): intro 0–41 · build 41–55 · dropA 55–79 · gap · dropB 82–103 · breakdown A/B/C
    # (C = the Sahara vocal, camel window) · build2 137–151 · drop3 151–171 (the big one) · breakdown · outro 178–
    shots = [
        Shot(1, "a_establish", move((215, 125, 42), (150, 88, 32), 30, 32, tgt0=(0, 0, 13), tgt1=(0, 0, 15))),
        Shot(f(st("intro") + 12 * bar), "b_far_dunes", move(*_far_dunes(), 70, ground=True, tgt0=CUBE - [0, 0, 4])),
        Shot(f(st("build")), "c_low_rise", move((30, -16, 1.4), (27, -14, 4.5), 20, ground=True,
                                              tgt0=CUBE - [0, 0, 6], tgt1=CUBE - [0, 0, 3])),
        Shot(f(st("build") + 4 * bar), "d_under", move((7, 5, 1.2), (6, 3.5, 1.2), 14, ground=True,
                                                     tgt0=CUBE, tgt1=CUBE + [1.0, 0.5, 0])),
        Shot(f(st("dropA")), "e_hero_low", orbit(19, 1.3, 38, 58, 15, ground=True, tz=-9.0)),
        Shot(f(st("dropA") + 8 * bar), "f_orbit_mid", orbit(36, 12, 30, 75, 32)),
        Shot(f(st("gap")), "g_wide_south", move((-30, -125, 2.2), (-22, -122, 2.2), 70, ground=True,
                                               tgt0=CUBE - [0, 0, 5])),
        Shot(f(st("dropB")), "h_hero_low2", orbit(20, 1.4, 100, 122, 15, ground=True, tz=-8.0)),
        Shot(f(st("dropB") + 6 * bar), "i_face_close", orbit(24, 17, 100, 118, 55)),
        Shot(f(st("breakdownA")), "j_push_in", move((-48, 40, 9), (-34, 28, 12), 32, 35)),
        Shot(f(st("breakdownB")), "k_orbit_high", orbit(46, 24, 240, 290, 30)),
        Shot(f(st("breakdownC_vocal")), "l_camel_far", move(cc, cc, lens_camel, tgt0=look_camel, tgt1=look_camel)),
        Shot(f(st("build2")), "m_low_slide", move((-36, -30, 1.6), (-24, -38, 1.6), 24, ground=True,
                                                tgt0=CUBE - [0, 0, 3], tgt1=CUBE - [0, 0, 2])),
        Shot(f(st("build2") + 4 * bar), "n_sun_tele", move(*_sun_tele(), 200, tgt0=CUBE - [0, 0, 1])),
        Shot(f(st("drop3")), "o_hero_wide", orbit(26, 1.6, 15, 40, 16, ground=True, tz=-9.0)),
        Shot(f(st("drop3") + 6 * bar), "p_orbit_close", orbit(23, 9, 130, 185, 24)),
        Shot(f(st("drop3") + 9 * bar), "q_high_three_q", orbit(42, 40, 200, 235, 26, tgt=np.array([0.0, 0.0, 9.0]))),
        Shot(f(st("breakdown")), "r_side_mid", move((60, -55, 6), (52, -60, 7), 35, ground=True)),
        Shot(f(st("outro")), "s_hero_low3", orbit(18, 1.2, 60, 85, 14, ground=True, tz=-8.0)),
        Shot(f(st("outro") + 4 * bar), "t_wide_pull", move((40, 70, 10), (70, 120, 16), 28, 26)),
        Shot(f(st("outro") + 8 * bar), "u_final", move((110, -140, 26), (190, -240, 44), 30, 30,
                                                       tgt0=CUBE - [0, 0, 2], tgt1=CUBE - [0, 0, 4])),
    ]
    return shots


def _clear_along(dirv, dists, h, side=0.0, heights=(0, 3, 6, 10, 14)):
    """first distance along dirv (from the cube) where a camera h (+ extra) above the sand sees the cube centre over
    the dunes from both ends of a `side`-m drift → (c0, c1) with z absolute"""
    d = np.asarray(dirv, float) / np.linalg.norm(dirv)
    perp = np.array([-d[1], d[0]])
    for extra in heights:
        for r in dists:
            p0 = d * r
            p1 = d * (r - 10.0) + perp * side
            c0 = np.array([p0[0], p0[1], gz(*p0, h + extra)])
            c1 = np.array([p1[0], p1[1], gz(*p1, h + extra)])
            if los_clear(c0, CUBE - [0, 0, 1.5]) and los_clear(c1, CUBE - [0, 0, 1.5]):
                return c0, c1
    raise RuntimeError(f"no clear view along {dirv}")


def _sun_tele():
    """camera ~650 m out on the anti-sun side of the cube (a little off-axis): the cube beside the setting sun"""
    a = math.radians(248.0 + 180.0 + 6.0)
    return _clear_along((math.sin(a), math.cos(a)), np.arange(650, 900, 10), 5.0, side=12.0)


def _far_dunes():
    """~450 m out to the north-west, a slow drift — the cube small over the dune sea"""
    c0, c1 = _clear_along((-0.89, 0.45), np.arange(420, 700, 10), 3.0, side=-15.0)
    c0[2] = c0[2] - float(TR.height(c0[0], c0[1]))          # move(ground=True) wants height above sand
    c1[2] = c1[2] - float(TR.height(c1[0], c1[1]))
    return c0, c1
