"""ROBOT FIGHT — two giant mechs trading blows in an arena, on the beat.

Scene plugin for scripts/lyric_viz.py (contract: lyric_scenes/README.md).
Zeke 2026-10-10 asked for a robot fight to "Hit a Bump"; Iris chose the
brief: side-on 2.5-D, an arena backdrop (grandstands full of people, light
towers, a hanging jumbotron, LED boards), two articulated mechs in the
middle ground, sparks + debris in front, a fighting-game HUD on top.

WHAT DRIVES WHAT (every element has a named musical cause)
  kick            -> the left mech (BRAWLER) throws a punch whose CONTACT lands
                     on the kick frame (the wind-up starts frames earlier — the
                     whole song is known, so anticipation can precede the beat)
  snare           -> the right mech (STRIKER) counters, contact on the snare
  hit             -> receiver flinches (head leads, torso, hips lag: overlapping
                     action), armour flashes, spark burst + white-hot flash +
                     ring shockwave at the contact point, screen shake ~ strength
  calm            -> slow guarded sparring: jabs that get BLOCKED, idle bob and
                     sway locked to the beat phase, wide camera
  build  (pre)    -> both back off and charge their cores (glow grows, energy
                     crackles, crowd lights strobe faster, lamps flicker)
  power cut (black, the half-beat before each drop) -> arena blackout; only the
                     cores (and their crackle) glow while the mechs charge in
  drop entry      -> they COLLIDE fist-to-fist: huge impact, floor shockwave,
                     "HIT A BUMP" on the jumbotron, shake, the one strobe
  drop            -> combos on every kick/snare, camera tight, crowd jumping
  BIG drop        -> + a beam clash between the cores for two bars (the clash
                     point is shoved by kicks/snares), then the flurry resumes;
                     nobody ever wins — the fight loops
  sung "bump"     -> a HEADBUTT + camera push-in + "BUMP" on the jumbotron
  hi-hat sparkle  -> core flicker + servo sparks from the joints
  sub bass        -> core heat, floor grid + emblem glow, health/energy bars

STATELESS: draw(i) is a pure function of i. The choreography (every strike,
its wind-up / recoil windows, body offsets, stance, steps, camera) is built
ONCE from the whole-song analysis into arrays in R.scene_cache; random
choices are seeded by constants or strike indices.

ASSETS (rendered headless in Blender, deterministic; PNGs are git-ignored,
so regenerate them on a fresh checkout):
  scripts/lyric_scenes/blender/robotfight_mechs.py -> assets/scenes/robotfight/mechs/
  scripts/lyric_scenes/blender/robotfight_arena.py -> assets/scenes/robotfight/arena/
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets" / "scenes" / "robotfight"
_BLENDER_HINT = (
    "robotfight assets missing ({}). Render them headless:\n"
    "  nice -n 15 ~/.local/bin/blender -b --factory-startup --python-exit-code 1"
    " --python scripts/lyric_scenes/blender/robotfight_mechs.py --\n"
    "  nice -n 15 ~/.local/bin/blender -b --factory-startup --python-exit-code 1"
    " --python scripts/lyric_scenes/blender/robotfight_arena.py --")

# ---------------------------------------------------------------------------
# palettes — A = left mech (BRAWLER), B = right mech (STRIKER)
# ---------------------------------------------------------------------------
PALETTES: dict[str, dict] = {
    # cyan mech vs magenta mech in a purple arena
    "neon": dict(
        bg_top=(10, 4, 26), bg_hor=(46, 16, 84), haze=(70, 30, 140),
        plate=(150, 120, 225), crowd_tint=(120, 100, 190),
        floor=(16, 10, 32), grid=(150, 70, 255), emblem=(185, 110, 255),
        crowd=[(255, 236, 215), (0, 235, 255), (255, 40, 200), (170, 90, 255)],
        flood=(205, 215, 255), lamp=(230, 235, 255),
        A=dict(paint=(60, 150, 190), frame=(34, 40, 56), glow=(0, 245, 255),
               core=(70, 255, 255)),
        B=dict(paint=(185, 60, 150), frame=(48, 30, 50), glow=(255, 40, 210),
               core=(255, 80, 230)),
        spark=[(255, 255, 255), (255, 240, 190), (255, 190, 90)],
        board=(0, 230, 255), board2=(255, 40, 200), screen=(120, 230, 255),
        hud_A=(0, 235, 255), hud_B=(255, 50, 210), hud=(235, 235, 255)),
    # hazard orange vs steel blue under sodium floodlights
    "industrial": dict(
        bg_top=(12, 8, 4), bg_hor=(70, 40, 14), haze=(120, 70, 25),
        plate=(215, 175, 130), crowd_tint=(170, 140, 110),
        floor=(22, 16, 10), grid=(255, 150, 30), emblem=(255, 170, 40),
        crowd=[(255, 220, 160), (255, 160, 40), (120, 190, 255), (255, 250, 230)],
        flood=(255, 185, 90), lamp=(255, 215, 150),
        A=dict(paint=(225, 120, 30), frame=(42, 36, 32), glow=(255, 200, 40),
               core=(255, 190, 60)),
        B=dict(paint=(95, 130, 165), frame=(32, 38, 46), glow=(110, 200, 255),
               core=(140, 220, 255)),
        spark=[(255, 255, 240), (255, 220, 140), (255, 140, 40)],
        board=(255, 170, 30), board2=(110, 200, 255), screen=(255, 200, 120),
        hud_A=(255, 150, 30), hud_B=(110, 190, 255), hud=(255, 240, 220)),
    # acid green vs blood red, toxic green fog
    "toxic": dict(
        bg_top=(3, 8, 4), bg_hor=(22, 52, 24), haze=(40, 100, 40),
        plate=(140, 200, 140), crowd_tint=(110, 160, 110),
        floor=(8, 16, 8), grid=(120, 255, 40), emblem=(150, 255, 60),
        crowd=[(220, 255, 210), (140, 255, 40), (255, 40, 40), (230, 255, 120)],
        flood=(200, 255, 190), lamp=(225, 255, 215),
        A=dict(paint=(115, 180, 50), frame=(28, 38, 26), glow=(170, 255, 40),
               core=(190, 255, 70)),
        B=dict(paint=(170, 34, 40), frame=(46, 22, 24), glow=(255, 40, 40),
               core=(255, 70, 50)),
        spark=[(255, 255, 240), (240, 255, 170), (200, 255, 80)],
        board=(140, 255, 40), board2=(255, 40, 40), screen=(170, 255, 120),
        hud_A=(150, 255, 40), hud_B=(255, 50, 50), hud=(230, 255, 230)),
}

STYLE = dict(palette=[(0, 245, 255), (255, 40, 210), (170, 90, 255),
                      (255, 240, 190)],
             font_title="Anton-Regular.ttf")
BLOOM = 0.36

# ---------------------------------------------------------------------------
# rig constants. Units: 1.0 ~ a mech's height. Local x = toward the opponent.
# ---------------------------------------------------------------------------
MECH = (
    dict(key="A", name="BRAWLER", hip_h=0.490, feet=(0.15, -0.17), knee=+1,
         guard_n=(0.285, -0.010), guard_f=(0.215, -0.095), facing=+1),
    dict(key="B", name="STRIKER", hip_h=0.505, feet=(0.13, -0.15), knee=-1,
         guard_n=(0.285, -0.075), guard_f=(0.215, -0.150), facing=-1),
)
PARTS = ("torso", "head", "uarm", "farm", "fist", "thigh", "shin", "foot")
GAP_FIGHT, GAP_CALM, GAP_BACK, GAP_CLASH, GAP_BEAM = 0.68, 0.74, 1.10, 0.90, 1.02
GAP_MIN, GAP_MAX, STANCE_MAX = 0.58, 1.16, 0.78   # hard footwork limits
FOOT_L = 0.12                                    # ankle -> toe, for heel pivots

# camera contract (must match blender/robotfight_arena.py)
CAM_D, CAM_H, VIS_H, HOR_Y = 3.0, 0.55, 1.0 / 0.58, 0.60


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _ease(u: float, e: str) -> float:
    u = min(1.0, max(0.0, u))
    if e == "in":
        return u * u
    if e == "in3":
        return u * u * u
    if e == "out":
        return 1.0 - (1.0 - u) ** 2
    if e == "out3":
        return 1.0 - (1.0 - u) ** 3
    if e == "lin":
        return u
    if e == "back":                      # ease-out with a small overshoot
        c1 = 1.4
        return 1.0 + (c1 + 1.0) * (u - 1.0) ** 3 + c1 * (u - 1.0) ** 2
    return u * u * (3.0 - 2.0 * u)       # "io"


def _keys(t: float, K) -> float:
    """Piecewise eased keyframes [(t, v, ease_into_this_key), ...]."""
    if t <= K[0][0]:
        return K[0][1]
    for (t0, v0, _), (t1, v1, e) in zip(K, K[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * _ease((t - t0) / max(1e-6, t1 - t0), e)
    return K[-1][1]


def _resp(t: float, rise: float = 1.0, decay: float = 6.0) -> float:
    """Hit response: snaps up within a frame or two, decays, settles with a
    small overshoot the other way (follow-through)."""
    if t < 0:
        return 0.0
    g = (1.0 - math.exp(-(t + 1.0) / rise)) * math.exp(-t / decay)
    if 5.0 < t < 16.0:
        g -= 0.10 * math.sin(math.pi * (t - 5.0) / 11.0)
    return g


def _rot(v, th):
    """Rotate (x, y) by a FORWARD lean th (x forward, y up): +th tips the top
    of the vector toward +x."""
    c, s = math.cos(th), math.sin(th)
    return (v[0] * c + v[1] * s, -v[0] * s + v[1] * c)


def _ik(A, T, l1, l2, bend):
    """2-bone IK. bend: "down" = the middle joint is the LOWER of the two
    solutions (elbows), +1 / -1 = knee toward +x / -x (local)."""
    dx, dy = T[0] - A[0], T[1] - A[1]
    d = math.hypot(dx, dy)
    d = min(max(d, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    ang = math.atan2(dy, dx)
    c = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    a = math.acos(max(-1.0, min(1.0, c)))
    j1 = (A[0] + l1 * math.cos(ang + a), A[1] + l1 * math.sin(ang + a))
    j2 = (A[0] + l1 * math.cos(ang - a), A[1] + l1 * math.sin(ang - a))
    if bend == "down":
        # lower elbow; on a near-tie prefer the one behind (x smaller)
        j = j1 if (j1[1] - j2[1]) < -1e-6 or (abs(j1[1] - j2[1]) <= 1e-6
                                               and j1[0] < j2[0]) else j2
    else:
        j = j1 if (j1[0] - j2[0]) * bend > 0 else j2
    e = (A[0] + d * math.cos(ang), A[1] + d * math.sin(ang))
    return j, e


def _bez(p0, c, p1, u):
    a, b, cc = (1 - u) ** 2, 2 * (1 - u) * u, u * u
    return (a * p0[0] + b * c[0] + cc * p1[0], a * p0[1] + b * c[1] + cc * p1[1])


def _arc_ctrl(p0, p1, amt):
    """Control point for an arc from p0 to p1, offset perpendicular by amt x
    the distance (amt > 0 = arc ABOVE the straight line)."""
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) + 1e-9
    nx, ny = -dy / L, dx / L
    if ny < 0:
        nx, ny = -nx, -ny
    return (mx + nx * amt * L, my + ny * amt * L)


def _text_mask(R, txt, font, max_w, max_h, pad=4):
    from PIL import Image, ImageDraw
    f = R.font(200, font)
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    bb = probe.textbbox((0, 0), txt, font=f)
    im = Image.new("L", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), txt, font=f, fill=255)
    arr = np.asarray(im, np.float32) / 255.0
    sc = min(max_w / arr.shape[1], max_h / arr.shape[0])
    tw, th = max(1, int(arr.shape[1] * sc)), max(1, int(arr.shape[0] * sc))
    return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)


def _place(m, H, W, cy=0.5, cx=0.5):
    out = np.zeros((H, W), np.float32)
    h, w = m.shape
    y0 = int(max(0, min(H - h, H * cy - h / 2)))
    x0 = int(max(0, min(W - w, W * cx - w / 2)))
    out[y0:y0 + min(h, H), x0:x0 + min(w, W)] = m[:H - y0, :W - x0]
    return out


def _logo_masks(R, size):
    """Tzeke000's logo as (dark lines, fill, red) masks at size x size."""
    if R.logo is None:
        return None
    rgb = np.asarray(R.logo.convert("RGB"), np.float32) / 255.0
    al = np.asarray(R.logo.split()[-1], np.float32) / 255.0
    lum = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    if al.std() < 0.02:                 # white background baked in
        al = 1.0 - ((lum > 0.93) & (sat < 0.10)).astype(np.float32)
    rd = ((rgb[..., 0] > 0.5) & (rgb[..., 1] < 0.35)
          & (rgb[..., 2] < 0.40)).astype(np.float32) * al
    dk = (lum < 0.35).astype(np.float32) * al * (1 - rd)
    fl = np.clip(al - rd - dk, 0, 1)
    lh, lw = rgb.shape[:2]
    sc = size / max(lh, lw)
    ow, oh = max(1, int(lw * sc)), max(1, int(lh * sc))
    return tuple(_place(cv2.resize(m, (ow, oh), interpolation=cv2.INTER_AREA),
                        size, size) for m in (dk, fl, rd))


def _blob(img, x, y, r, col, gain=1.0):
    """Additive gaussian blob (radius r px) into img (any res)."""
    H, W = img.shape[:2]
    R3 = int(3 * r) + 1
    x0, x1 = max(0, int(x) - R3), min(W, int(x) + R3 + 1)
    y0, y1 = max(0, int(y) - R3), min(H, int(y) + R3 + 1)
    if x1 <= x0 or y1 <= y0 or r <= 0.3:
        return
    xs = (np.arange(x0, x1, dtype=np.float32) - x) / r
    ys = (np.arange(y0, y1, dtype=np.float32) - y) / r
    g = np.exp(-0.5 * (ys[:, None] ** 2 + xs[None, :] ** 2)) * gain
    img[y0:y1, x0:x1] += g[..., None] * np.asarray(col, np.float32)


# ---------------------------------------------------------------------------
# 1. TIMELINE + CHOREOGRAPHY (pure function of the whole-song analysis)
# ---------------------------------------------------------------------------
def _timeline(R, a, lv) -> dict:
    C = R.scene_cache
    if "T" in C:
        return C["T"]
    fps = R.fps
    fr = fps / 30.0
    T = dict(lv.stage_timeline(a, fps, "acid", label="robotfight"))
    n = len(a.rms)
    I, hot, big, pre, black = T["I"], T["hot"], T["big"], T["pre"], T["black"]
    sm = lv._gauss_smooth
    E = np.clip(np.maximum(I, 0.8 * pre), 0, 1).astype(np.float32)
    T["E2"] = sm(E, fps, 1.0)
    T["hot_f"] = sm(hot.astype(np.float32), fps, 0.12)
    bass = np.asarray(a.bass, np.float32)
    T["heat"] = np.clip(sm(bass, fps, 0.15) * 1.1, 0, 1)
    hs = sm(a.high, fps, 0.35)
    T["spark"] = np.clip((np.asarray(a.high, np.float32) - hs) * 5.0, 0, 1)
    bars = np.asarray(a.bars, np.float32)
    nb = bars.shape[1]
    T["bars"] = bars
    T["low"] = sm(bars[:, : nb // 5].mean(1), fps, 0.10)
    T["mid"] = sm(bars[:, nb // 4: (3 * nb) // 4].mean(1), fps, 0.10)
    beat_f = fps * 60.0 / max(60.0, float(a.bpm) or 120.0)
    T["beat_f"] = beat_f
    # drop number per frame (ROUND N on the jumbotron)
    rnd = np.zeros(n, np.int32)
    k = 0
    for s0, s1, h in T["segs"]:
        if h:
            k += 1
        rnd[s0:s1] = max(1, k) if h else k + 1
    T["round"] = np.minimum(rnd, 9)
    sec_bar = np.zeros(n, np.int64)
    for s0, s1, h in T["segs"]:
        sec_bar[s0:s1] = T["bar"][s0:s1] - T["bar"][s0]
    T["sec_bar"] = sec_bar
    # ---- sung "bump": onsets of the lyric punch
    punch = np.zeros(n, np.float32)
    if R.lyric_zoom is not None and float(R.lyric_zoom.max()) > 1.0:
        punch = ((np.asarray(R.lyric_zoom, np.float32) - 1.0)
                 / (float(R.lyric_zoom.max()) - 1.0))
    T["punch"] = punch
    pon = (punch > 0.12).astype(np.int8)
    bump_on = list(np.flatnonzero(np.diff(pon) == 1) + 1)
    # ---- BEAM CLASH: bars 4-5 of the big drop (two bars)
    beam = np.zeros(n, np.float32)
    for s0, s1, h in T["segs"]:
        if h and big[s0]:
            sb = sec_bar[s0:s1]
            idx = np.flatnonzero((sb >= 4) & (sb < 6)) + s0
            if len(idx) > fps:
                b0, b1 = int(idx[0]), int(idx[-1]) + 1
                ramp = max(2, int(5 * fr))
                for f in range(b0, b1):
                    beam[f] = min(1.0, (f - b0 + 1) / ramp, (b1 - f) / ramp)
    T["beam"] = beam
    # clash point offset: kicks shove it toward B, snares toward A
    push = np.zeros(n, np.float32)
    push[np.asarray(a.kick, bool) & (beam > 0)] += 1.0
    push[np.asarray(a.snare, bool) & (beam > 0)] -= 1.0
    ker = np.exp(-np.arange(int(14 * fr)) / (5.0 * fr)).astype(np.float32)
    xoff = np.convolve(push, ker)[:n] * 0.05
    T["beam_x"] = (np.clip(xoff, -0.12, 0.12)
                   + 0.03 * np.sin(np.arange(n) / fps * 7.0)).astype(np.float32)
    beam_age, _ = lv._age_since((push != 0))
    T["beam_hit_age"] = beam_age
    # ---- STRIKES
    entries = list(T["entries"])
    kick, snare = np.asarray(a.kick, bool), np.asarray(a.snare, bool)
    cands = []
    for f in np.flatnonzero(kick | snare):
        cands.append((int(f), "kick" if kick[f] else "snare"))
    for f in bump_on:
        cands.append((int(f), "bump"))
    cands.sort(key=lambda c: (c[0], {"bump": 0, "kick": 1, "snare": 2}[c[1]]))
    nblk = int(round(0.5 * beat_f))
    strikes = []
    for e in entries:
        strikes.append(dict(f0=int(e), att=2, kind="clash", s=1.4, W=nblk + 3,
                            R=int(8 * fr), arm=0, blocked=False))
    busy = np.zeros(n, bool)
    for e in entries:
        busy[max(0, e - nblk - 3):min(n, e + int(9 * fr))] = True
    last = -10 ** 9
    last_calm = -10 ** 9
    last_hb = -10 ** 9
    last_m = [-10 ** 9, -10 ** 9]
    bi = 0
    ncalm = 0
    for f, typ in cands:
        if busy[f] or beam[f] > 0 or black[f]:
            continue
        if not hot[f] and pre[f] > 0.12 and typ != "bump":
            continue                      # the build: backing off, charging
        # per-mech spacing: a snare and a kick 1-3 frames apart in a drop are
        # a CROSS-COUNTER (both land); the same mech needs 4 frames between
        # punches, and calm/headbutts keep the global spacing
        cand_att = 0 if typ == "kick" else 1
        if hot[f] and typ != "bump":
            if f - last_m[cand_att] < int(4 * fr) or f - last < 1:
                continue
        elif f - last < int(4 * fr):
            continue
        if f > n - int(1.5 * fps):
            continue
        if typ == "bump":
            if f - last_hb < 1.0 * fps:   # "bump bump bump": one headbutt,
                continue                  # the rest are camera pushes
            last_hb = f
            att = bi % 2
            bi += 1
            strikes.append(dict(f0=f, att=att, kind="headbutt",
                                s=1.1 if hot[f] else 0.8, W=int(8 * fr),
                                R=int(12 * fr), arm=-1, blocked=False))
            last = f
            continue
        att = 0 if typ == "kick" else 1
        if hot[f]:
            s = 0.62 + 0.45 * float(bass[f]) + (0.18 if big[f] else 0.0)
            strikes.append(dict(f0=f, att=att, kind="", s=s, W=0,
                                R=int(9 * fr), arm=0, blocked=False, src=typ))
            last = last_m[att] = f
        else:
            if f - last_calm < 2 * beat_f or f < int(1.5 * fps):
                continue
            att = ncalm % 2               # calm sparring takes turns
            ncalm += 1
            strikes.append(dict(f0=f, att=att, kind="jab", s=0.38,
                                W=int(8 * fr), R=int(11 * fr), arm=0,
                                blocked=True))
            last = last_calm = f
    strikes.sort(key=lambda d: d["f0"])
    # arms, kinds, wind-up lengths (combos alternate arms; a long pause
    # resets to the near arm so a combo always OPENS with the visible jab)
    prev_att = {0: None, 1: None}
    for j, S in enumerate(strikes):
        rng = np.random.default_rng(1000 + j)
        for m in ((0, 1) if S["att"] == 2 else (S["att"],)):
            p = prev_att[m]
            gap = S["f0"] - p["f0"] if p is not None else 10 ** 6
            if S["kind"] in ("", "jab") and not S["blocked"]:
                if p is not None and gap < 1.1 * fps and p["kind"] in ("", "jab"):
                    S["arm"] = 1 - p["arm"]
                else:
                    S["arm"] = 0
            if S["kind"] == "":
                S["W"] = int(min(6 * fr, max(3 * fr, gap - 2 * fr)))
                r = rng.random()
                if S["arm"] == 0:
                    S["kind"] = "jab" if r < 0.45 else ("hook" if r < 0.75 else "body")
                else:
                    S["kind"] = "upper" if (S["s"] > 0.95 and r < 0.5) else "cross"
            prev_att[m] = S
    # recoil must finish before the same arm winds up again (no pops)
    for j, S in enumerate(strikes):
        for S2 in strikes[j + 1:]:
            same = (S2["att"] == S["att"] or 2 in (S["att"], S2["att"]))
            if same and S2["arm"] == S["arm"] and S["arm"] >= 0:
                room = (S2["f0"] - S2["W"]) - (S["f0"] + 1)
                S["R"] = int(max(2, min(S["R"], room)))
                break
    T["strikes"] = strikes
    T["s_f0"] = np.asarray([S["f0"] for S in strikes], np.int64)
    # ---- per-frame body offsets + arm ownership
    dx = np.zeros((2, n), np.float32)
    lean = np.zeros((2, n), np.float32)
    crouch = np.zeros((2, n), np.float32)
    tilt = np.zeros((2, n), np.float32)
    squash = np.zeros((2, n), np.float32)
    flash = np.zeros((2, n), np.float32)
    flail = np.zeros((2, n), np.float32)
    own = np.full((2, 2, n), -1, np.int32)         # [mech, arm] -> strike
    blk = np.full((2, n), -1, np.int32)            # defender block -> strike
    bump_own = np.full((2, n), -1, np.int32)       # headbutt -> strike

    def add(arr, m, f0, t0, t1, fn):
        for f in range(max(0, f0 + t0), min(n, f0 + t1 + 1)):
            arr[m, f] += fn(f - f0)

    for j, S in enumerate(strikes):
        f0, s, W, Rr, kind = S["f0"], S["s"], S["W"], S["R"], S["kind"]
        A_ = max(2, int(round(0.42 * W)))
        atts = (0, 1) if S["att"] == 2 else (S["att"],)
        for m in atts:
            if kind == "headbutt":
                Kl = [(-W, 0, "lin"), (-4 * fr, -0.04, "io"), (0, 0.13 * s, "in"),
                      (2 * fr, 0.13 * s, "lin"), (14 * fr, 0, "io")]
                Kn = [(-W, 0, "lin"), (-4 * fr, -0.30, "io"), (0, 0.58, "in3"),
                      (2 * fr, 0.62, "out"), (14 * fr, 0, "io")]
                Kt = [(-W, 0, "lin"), (-4 * fr, -0.20, "io"), (0, 0.35, "in"),
                      (12 * fr, 0, "io")]
                Kc = [(-W, 0, "lin"), (-4 * fr, 0.03, "io"), (0, -0.01, "in"),
                      (12 * fr, 0, "io")]
                for f in range(max(0, f0 - W), min(n, f0 + int(14 * fr) + 1)):
                    bump_own[m, f] = j
            else:
                big_ = 1.0 if kind != "jab" or not S["blocked"] else 0.6
                lun = {"jab": 0.05, "cross": 0.06, "hook": 0.045, "upper": 0.04,
                       "body": 0.055, "clash": 0.06}[kind] * min(1.3, s) * big_
                ln = {"jab": 0.18, "cross": 0.24, "hook": 0.30, "upper": 0.14,
                      "body": 0.30, "clash": 0.28}[kind] * min(1.3, s) * big_
                cr = {"upper": 0.05, "body": 0.04}.get(kind, 0.015)
                Kl = [(-W, 0, "lin"), (-A_, -0.035 * s, "io"), (0, lun, "in"),
                      (1, lun, "lin"), (1 + Rr + 3, 0, "out")]
                Kn = [(-W, 0, "lin"), (-A_, -0.12 * s, "io"), (0, ln, "in"),
                      (2, ln * 1.18, "out"), (Rr + 5, 0, "io")]       # overshoot
                Kt = [(-W, 0, "lin"), (-A_, 0.08, "io"), (0, -0.06, "in"),
                      (3, 0.05, "out"), (Rr + 6, 0, "io")]            # head lags
                up = -0.03 if kind == "upper" else 0.0
                Kc = [(-W, 0, "lin"), (-A_, cr, "io"), (0, up, "in"),
                      (Rr + 4, 0, "io")]
                arm = S["arm"]
                for f in range(max(0, f0 - W), min(n, f0 + 1 + Rr + 1)):
                    own[m, arm, f] = j
            tl = int(-W)
            add(dx, m, f0, tl, int(20 * fr), lambda t, K=Kl: _keys(t, K))
            add(lean, m, f0, tl, int(20 * fr), lambda t, K=Kn: _keys(t, K))
            add(tilt, m, f0, tl, int(20 * fr), lambda t, K=Kt: _keys(t, K))
            add(crouch, m, f0, tl, int(20 * fr), lambda t, K=Kc: _keys(t, K))
        # receivers
        recv = (0, 1) if S["att"] == 2 else (1 - S["att"],)
        for m in recv:
            ss = s * (0.35 if S["blocked"] else 1.0)
            body = kind == "body"
            rs = 1.0 / fr
            add(dx, m, f0, 0, int(22 * fr),
                lambda t: -0.05 * ss * _resp(t * rs - 1.0))
            add(lean, m, f0, 0, int(22 * fr),
                lambda t: (0.24 if body else -0.30) * ss * _resp(t * rs))
            add(tilt, m, f0, 0, int(22 * fr),
                lambda t: (0.25 if body else -0.55) * ss * _resp(t * rs + 1.0))
            add(crouch, m, f0, 0, int(22 * fr),
                lambda t: (0.045 if body else 0.012) * ss * _resp(t * rs))
            add(squash, m, f0, 0, int(4 * fr),
                lambda t: 0.07 * ss * math.exp(-t * rs / 1.5))
            add(flash, m, f0, 0, int(3 * fr),
                lambda t: (0.0 if S["blocked"] else 1.0) * (1.0, 0.45, 0.15, 0.0)
                [min(3, int(t * rs))])
            add(flail, m, f0, 0, int(22 * fr),
                lambda t: ss * _resp(t * rs - 2.0, 1.5, 7.0))
            if S["blocked"]:
                for f in range(max(0, f0 - int(6 * fr)), min(n, f0 + int(10 * fr))):
                    blk[m, f] = j
    # ---- calm idle: bob on the beat, sway over the bar; build: charge crouch
    bif = T["bi"].astype(np.float32) + T["ph"]
    ph = T["ph"]
    hotf = T["hot_f"]
    bob = np.exp(-ph * 5.0) * (0.012 + 0.016 * hotf)
    for m in (0, 1):
        crouch[m] += bob + 0.055 * pre + 0.04 * T["beam"]
        lean[m] += (0.035 * np.sin(2 * np.pi * bif / 4.0 + m * np.pi) * (1 - hotf)
                    + 0.10 * pre + 0.16 * T["beam"] + 0.09 * hotf)
        crouch[m] += 0.012 * hotf
        dx[m] += 0.012 * np.sin(2 * np.pi * bif / 8.0 + m * 1.7) * (1 - hotf)
        tilt[m] += 0.04 * np.sin(2 * np.pi * bif / 4.0 + 0.8 + m) * (1 - hotf)
    # ---- FOOTWORK (Zeke 10-10: "maybe a little more footwork from them").
    # The FEET are primary: a step planner places every foot, the stance is
    # derived from where the feet are, so a weight-bearing foot never slides.
    fw = _footwork(T, strikes, n, fps, fr, beat_f, lv)
    crouch += fw["crouch"]
    T.update(dx=dx, lean=lean, crouch=crouch, tilt=tilt, squash=squash,
             flash=flash, flail=flail, own=own, blk=blk, bump_own=bump_own,
             feet=fw["feet"], lift=fw["lift"], heel=fw["heel"],
             stance=fw["stance"], cx=fw["cx"], steps=fw["steps"])
    hotb = hot.astype(np.float32)
    ss_pre = pre * pre * (3 - 2 * pre)
    tt = np.arange(n) / fps
    # ---- CAMERA: zoom (dolly), pan, height
    zt = 1.0 - 0.05 * ss_pre * (1 - hotb) + 0.11 * hotf + 0.04 * big.astype(np.float32)
    zt = sm(zt, fps, 0.45).astype(np.float64)
    for e in entries:                       # crash-in on the collision
        for f in range(e, min(n, e + int(40 * fr))):
            zt[f] += 0.06 * math.exp(-(f - e) / (9.0 * fr))
    zt += 0.05 * punch                       # sung "bump" pushes in
    zt += 0.012 * np.sin(2 * np.pi * tt / 9.0) * (1 - hotf)
    camx = sm(T["cx"], fps, 0.5) * 0.85
    nud = np.zeros(n, np.float32)
    for S in strikes:
        if S["att"] == 2:
            continue
        d = 1.0 if S["att"] == 0 else -1.0
        f0 = S["f0"]
        for f in range(f0, min(n, f0 + int(16 * fr))):
            nud[f] += 0.035 * d * S["s"] * math.exp(-(f - f0) / (5.0 * fr))
    camx = camx + sm(nud, fps, 0.05)
    T["zoom"] = zt.astype(np.float32)
    T["camx"] = camx.astype(np.float32)
    T["camh"] = (CAM_H - 0.15 * np.clip(zt - 1.0, 0, None)).astype(np.float32)
    # beam sweep phase for the floodlights
    T["sweep"] = np.cumsum(0.10 + 0.35 * T["E2"] + 0.4 * hotf).astype(np.float64) / fps
    # ---- shake: per strike (contact) + drop entries
    shake = np.zeros(n, np.float32)
    for S in strikes:
        shake[S["f0"]] = max(shake[S["f0"]], 0.25 * S["s"] if not S["blocked"] else 0.0)
    for e in entries:
        shake[e] = 1.0
    sh_age, sh_idx = lv._age_since(shake > 0)
    T.update(shake=shake, sh_age=sh_age, sh_idx=sh_idx)
    C["T"] = T
    # contact points need the rig, which needs the arrays above
    pts = []
    for S in strikes:
        rig = _rig(R, T, S["f0"])
        pts.append(_contact(rig, S))
    T["s_pt"] = np.asarray(pts, np.float32).reshape(-1, 2)
    nk = sum(1 for S in strikes if S["att"] == 0)
    print(f"[robotfight] {len(strikes)} strikes ({nk} brawler, "
          f"{sum(1 for S in strikes if S['att'] == 1)} striker, "
          f"{sum(1 for S in strikes if S['kind'] == 'headbutt')} headbutts, "
          f"{len(entries)} clashes), beam {int((beam > 0).sum())} frames")
    return T


def _footwork(T, strikes, n, fps, fr, beat_f, lv) -> dict:
    """Plan every footstep of the song, then derive the stance from the feet.

    A time-ordered planner (run ONCE over the whole song, so draw() stays a
    pure function of the frame) places the lead/rear foot of each mech.
    Between its steps a foot is PLANTED (constant x): no sliding while it
    bears weight. Step sources, in musical order:
      * strike step-in   lead foot lands with / just before the contact frame,
                         the rear foot drags up after it (weight behind it)
      * hit step-back    rear foot steps away a frame after a hit, lead follows
      * drop-entry       two charge steps through the blackout into the clash,
                         then two knocked-back steps
      * beat steps       on grid beats, toward a section target that drifts
                         across the ring (circling / exchanging ground); in the
                         calm an in-and-out boxer shuffle, in the build a
                         step back each beat
    Hard limits: the fighters' gap stays in [GAP_MIN, GAP_MAX], each stance
    stays inside +-STANCE_MAX, stance width stays sane. Hip height sells the
    weight: it rises through each swing and dips on landing; a heel pivot on
    hooks/crosses and an on-the-toes bounce in the calm drive heel arrays."""
    hot, pre, black, big = T["hot"], T["pre"], T["black"], T["big"]
    beam = T["beam"]
    sm = lv._gauss_smooth
    hotb = hot.astype(np.float32)
    ss_pre = pre * pre * (3 - 2 * pre)
    gap_t = GAP_CALM - (GAP_CALM - GAP_FIGHT) * sm(hotb, fps, 0.3)
    gap_t = gap_t + (GAP_BACK - GAP_CALM) * ss_pre * (1 - hotb)
    gap_t = gap_t * (1 - beam) + GAP_BEAM * beam
    tt = np.arange(n) / fps
    seg_ph = np.zeros(n, np.float32)
    for k, (s0, s1, h) in enumerate(T["segs"]):
        seg_ph[s0:s1] = 1.9 * k
    c_t = np.where(hot, 0.14 * np.sin(2 * np.pi * tt / 9.0 + seg_ph),
                   0.17 * np.sin(2 * np.pi * tt / 13.0 + seg_ph))
    c_t = sm(c_t, fps, 0.8)
    fac = [M["facing"] for M in MECH]
    off = [M["feet"] for M in MECH]                  # (lead, rear) local x
    mid_off = [fac[m] * (off[m][0] + off[m][1]) / 2.0 for m in (0, 1)]
    # initial placement
    g0 = float(gap_t[0])
    st0 = [float(c_t[0]) - g0 / 2, float(c_t[0]) + g0 / 2]
    xf = [[st0[m] + fac[m] * off[m][0], st0[m] + fac[m] * off[m][1]] for m in (0, 1)]
    init = [list(xf[0]), list(xf[1])]
    busy = [[-10 ** 9, -10 ** 9], [-10 ** 9, -10 ** 9]]
    steps = [[[], []], [[], []]]                     # (s, e, x0, x1)

    def stance(m):
        return (xf[m][0] + xf[m][1]) / 2.0 - mid_off[m]

    def place(m, foot, s0, dur, dfwd, tol, wide=False):
        """Schedule foot `foot` of mech m to move dfwd (toward the opponent
        if > 0) starting at s0. Returns the moved distance (0 = dropped).
        The gap is kept in a band around the section's target gap (wide=True:
        the global band, for the entry charge / knock-back)."""
        s1 = max(s0, busy[m][foot], busy[m][1 - foot])   # other foot planted
        if s1 - s0 > tol or s1 >= n - 2:
            return 0.0
        x0 = xf[m][foot]
        # allowed landing interval = stance-width limits (lead stays 0.20-0.42
        # ahead of rear) INTERSECTED with the gap + frame limits; a step that
        # cannot satisfy both is not taken
        other = xf[m][1 - foot]
        sg = fac[m] * (1 if foot == 0 else -1)
        w_a, w_b = other + sg * 0.20, other + sg * 0.42
        if wide:
            g_lo, g_hi = GAP_MIN, GAP_MAX
        else:
            gt = float(gap_t[min(n - 1, s1)])
            g_lo, g_hi = max(GAP_MIN, gt - 0.10), min(GAP_MAX, gt + 0.14)
        st_o = stance(1 - m)
        if m == 0:
            lo, hi = st_o - g_hi, st_o - g_lo
        else:
            lo, hi = st_o + g_lo, st_o + g_hi
        stm = stance(m)
        # outside the band: any step that does not make it WORSE is allowed
        lo, hi = min(lo, stm), max(hi, stm)
        lo, hi = max(lo, -STANCE_MAX), min(hi, STANCE_MAX)
        x_lo = max(min(w_a, w_b), x0 + 2.0 * (lo - stm))
        x_hi = min(max(w_a, w_b), x0 + 2.0 * (hi - stm))
        if x_lo > x_hi:
            return 0.0
        x1 = min(max(x0 + fac[m] * dfwd, x_lo), x_hi)
        # never step AGAINST the requested direction to satisfy a limit
        if (x1 - x0) * fac[m] * dfwd < 0:
            return 0.0
        if abs(x1 - x0) < 0.008:
            return 0.0
        e = min(n - 1, s1 + dur)
        steps[m][foot].append((s1, e, x0, x1))
        xf[m][foot] = x1
        busy[m][foot] = e
        return fac[m] * (x1 - x0)

    # ---- requests: (start, prio, kind, args)
    reqs = []
    act = np.zeros((2, n), bool)                     # strike-step activity
    for j, S in enumerate(strikes):
        f0, s_ = S["f0"], min(1.3, S["s"])
        if S["att"] == 2:
            continue
        a_, d_ = S["att"], 1 - S["att"]
        hb = S["kind"] == "headbutt"
        if S["blocked"]:
            din, dur = 0.035, int(5 * fr)
        else:
            din = (0.085 if hb else 0.075) * s_
            dur = int(max(3 * fr, min(5 * fr, S["W"] - 1)))
        # step-in: lead foot LANDS on f0-1 (with the punch's weight behind it)
        reqs.append((f0 - 1 - dur, 1, "step", (a_, 0, dur, din, int(1 * fr))))
        reqs.append((f0 + int(2 * fr), 2, "step", (a_, 1, int(5 * fr), din, int(6 * fr))))
        # step-back after taking it (blocked jabs: a small give)
        dback = (0.022 if S["blocked"] else 0.065 * s_ * (0.7 if S["kind"] == "body" else 1.0))
        reqs.append((f0 + int(1 * fr), 1, "step", (d_, 1, int(3 * fr), -dback, int(2 * fr))))
        if not S["blocked"]:
            reqs.append((f0 + int(4 * fr), 2, "step", (d_, 0, int(4 * fr), -dback, int(4 * fr))))
        lo = max(0, f0 - int(8 * fr))
        act[a_, lo:f0 + int(9 * fr)] = True
        act[d_, f0:f0 + int(11 * fr)] = True
    ent_busy = np.zeros(n, bool)
    for e in T["entries"]:
        sh = {}
        for m in (0, 1):
            # charge through the blackout: rear foot, then lead foot (lands e-1)
            reqs.append((e - int(11 * fr), 0, "charge", (m, 1, int(4 * fr), e, sh)))
            reqs.append((e - int(6 * fr), 0, "charge", (m, 0, int(5 * fr), e, sh)))
            # knocked back by the collision
            reqs.append((e + int(2 * fr), 0, "kback", (m, 1, int(4 * fr), -0.11, 99)))
            reqs.append((e + int(6 * fr), 0, "kback", (m, 0, int(5 * fr), -0.11, 99)))
        ent_busy[max(0, e - int(16 * fr)):min(n, e + int(16 * fr))] = True
    gb = np.flatnonzero(T["gbeat"])
    bi = T["bi"]
    for b in gb:
        if black[b] or beam[b] > 0 or ent_busy[b]:
            continue
        for m in (0, 1):
            if act[m, b]:
                continue
            if not hot[b] and pre[b] < 0.12 and (int(bi[b]) + m) % 2:
                continue          # calm: the fighters take turns shuffling
            reqs.append((int(b), 3, "auto", (m, int(b))))
    reqs.sort(key=lambda r: (r[0], r[1]))
    for start, _, kind, args in reqs:
        if start < 0:
            start = 0
        if kind == "step":
            m, foot, dur, d, tol = args
            place(m, foot, start, dur, d, tol)
        elif kind == "kback":
            m, foot, dur, d, tol = args
            place(m, foot, start, dur, d, tol, wide=True)
        elif kind == "charge":
            m, foot, dur, e, sh = args
            if m not in sh:      # stance distance to the clash mark
                tgt = (float(c_t[e]) - GAP_CLASH / 2) if m == 0 else \
                    (float(c_t[e]) + GAP_CLASH / 2)
                sh[m] = fac[m] * (tgt - stance(m))
            place(m, foot, start, dur, sh[m], 99, wide=True)
        else:                     # beat step toward the drifting target
            m, b = args
            calm = not hot[b] and pre[b] < 0.12
            tgt = (float(c_t[b]) - float(gap_t[b]) / 2) if m == 0 else \
                (float(c_t[b]) + float(gap_t[b]) / 2)
            err = fac[m] * (tgt - stance(m))           # + = toward opponent
            if calm and abs(err) < 0.025:
                # the boxer shuffle: in on one bar, out on the next
                err = 0.024 * (1.0 if (int(bi[b]) // 4 + m) % 2 == 0 else -1.0)
            cap = 0.05 if calm else 0.075
            err = max(-cap, min(cap, err))
            if abs(err) < 0.012:
                continue
            dur = int((6 if calm else 4) * fr)
            first = 0 if err > 0 else 1                # lead first going in
            if place(m, first, b, dur, err, int(1 * fr)) != 0.0:
                place(m, 1 - first, b + dur + int(1 * fr), dur, err, int(3 * fr))
    # ---- arrays: planted between steps, an arc through the air during one
    feet = np.zeros((2, 2, n), np.float32)
    lift = np.zeros((2, 2, n), np.float32)
    crouch = np.zeros((2, n), np.float32)
    for m in (0, 1):
        for foot in (0, 1):
            cur = init[m][foot]
            pos = 0
            for (s0, e, x0, x1) in sorted(steps[m][foot]):
                feet[m, foot, pos:s0 + 1] = cur
                L = max(1, e - s0)
                u = np.arange(1, L + 1, dtype=np.float32) / L
                ue = u * u * (3 - 2 * u)
                feet[m, foot, s0 + 1:e + 1] = (x0 + (x1 - x0) * ue)[:max(0, min(n, e + 1) - s0 - 1)]
                d = abs(x1 - x0)
                h = min(0.05, 0.018 + 0.30 * d)
                lift[m, foot, s0 + 1:e + 1] = (h * np.sin(np.pi * u))[:max(0, min(n, e + 1) - s0 - 1)]
                # hips: up through the swing, weight LANDS (dip, then recover)
                dip = 0.004 + 0.07 * d
                crouch[m, s0 + 1:e + 1] -= (0.35 * dip * np.sin(np.pi * u))[:max(0, min(n, e + 1) - s0 - 1)]
                k = np.arange(0, int(10 * fr))
                seg = crouch[m, e:e + len(k)]
                seg += (dip * np.exp(-k / (2.5 * fr)))[:len(seg)]
                cur = x1
                pos = e + 1
            feet[m, foot, pos:] = cur
    # ---- heel pivots (hooks: lead foot; crosses/uppercuts: rear foot) and
    # the calm on-the-toes bounce
    heel = np.zeros((2, 2, n), np.float32)
    for S in strikes:
        if S["att"] == 2 or S["kind"] not in ("hook", "cross", "upper"):
            continue
        m = S["att"]
        foot = 0 if S["kind"] == "hook" else 1
        f0, W, Rr = S["f0"], S["W"], S["R"]
        A_ = max(2, int(round(0.42 * W)))
        K = [(-A_, 0.0, "lin"), (0, 0.38, "in"), (2, 0.38, "lin"), (Rr + 4, 0.0, "io")]
        for f in range(max(0, f0 - A_), min(n, f0 + Rr + 5)):
            heel[m, foot, f] = max(heel[m, foot, f], _keys(f - f0, K))
    calm_w = (1 - T["hot_f"]) * (1 - np.clip(pre * 3, 0, 1)) * (~black)
    bounce = (0.5 - 0.5 * np.cos(2 * np.pi * T["ph"])) * calm_w     # 0 on the beat
    for m in (0, 1):
        crouch[m] -= 0.010 * bounce
        for foot in (0, 1):
            planted = lift[m, foot] < 1e-4
            heel[m, foot] = np.maximum(heel[m, foot], 0.16 * bounce * planted)
    # ---- the stance is DERIVED from the feet (weight between them); a light
    # centred smoothing lets the hips start shifting just before a step lands
    stn = np.zeros((2, n), np.float32)
    for m in (0, 1):
        mid = (feet[m, 0] + feet[m, 1]) / 2.0 - mid_off[m]
        stn[m] = sm(mid, fps, 0.07)
    cx = sm((stn[0] + stn[1]) / 2.0, fps, 0.15)
    nst = sum(len(steps[m][k]) for m in (0, 1) for k in (0, 1))
    gapv = stn[1] - stn[0]
    print(f"[robotfight] footwork: {nst} steps, gap {gapv.min():.2f}-{gapv.max():.2f}, "
          f"stance x {stn.min():.2f}..{stn.max():.2f}")
    return dict(feet=feet, lift=lift, heel=heel, crouch=crouch, stance=stn,
                cx=cx.astype(np.float32), steps=steps)


# ---------------------------------------------------------------------------
# 2. THE RIG — pose of both mechs at frame f (pure)
# ---------------------------------------------------------------------------
def _geo_meta():
    p = ASSETS / "mechs" / "meta.json"
    if not p.is_file():
        raise SystemExit(_BLENDER_HINT.format(p))
    return json.loads(p.read_text())


def _bones(R):
    C = R.scene_cache
    if "bones" in C:
        return C["bones"]
    meta = _geo_meta()
    ppu = meta["ppu"]
    B = []
    for M in MECH:
        k = M["key"]
        P = meta["parts"]
        tor = P[f"{k}_torso"]
        piv = tor["pivot"]

        def loc(q, piv=piv):
            return ((q[0] - piv[0]) / ppu, (piv[1] - q[1]) / ppu)
        anc = {n_: loc(q) for n_, q in tor["anchors"].items()}
        hd = P[f"{k}_head"]
        hc = ((hd["anchors"]["center"][0] - hd["pivot"][0]) / ppu,
              (hd["pivot"][1] - hd["anchors"]["center"][1]) / ppu)
        hf = ((hd["anchors"]["face"][0] - hd["pivot"][0]) / ppu,
              (hd["pivot"][1] - hd["anchors"]["face"][1]) / ppu)
        ft = P[f"{k}_foot"]
        B.append(dict(
            neck=anc["neck"], core=anc["core"], sh_n=anc["sh_n"], sh_f=anc["sh_f"],
            hip_n=anc["hip_n"], hip_f=anc["hip_f"], head_c=hc, head_f=hf,
            l_ua=-P[f"{k}_uarm"]["bone"][1], l_fa=-P[f"{k}_farm"]["bone"][1],
            l_fist=-P[f"{k}_fist"]["bone"][1], l_th=-P[f"{k}_thigh"]["bone"][1],
            l_sh=-P[f"{k}_shin"]["bone"][1],
            # ankle height above the floor (sole is BELOW the pivot in image
            # px; this sign was flipped until 10-10 and sank the feet 0.07
            # under the floor, legs locked straight)
            ankle=(ft["anchors"]["sole"][1] - ft["pivot"][1]) / ppu))
    C["bones"] = B
    return B


def _rig(R, T, f):
    """Both mechs' joints in WORLD units (x right, y up) at frame f."""
    B = _bones(R)
    n = len(T["hot"])
    f = int(min(max(f, 0), n - 1))
    fr = R.fps / 30.0
    out = []
    # bodies first (the arms aim at the OTHER mech's current head/core)
    for m, M in enumerate(MECH):
        b = B[m]
        fac = M["facing"]
        st = float(T["stance"][m, f])
        th = float(T["lean"][m, f])
        hip = (float(T["dx"][m, f]), M["hip_h"] - float(T["crouch"][m, f]))
        # charge tremble in the build / blackout
        trem = float(T["pre"][f]) * 0.006
        if trem > 0:
            hip = (hip[0] + trem * math.sin(f * 2.9 + m), hip[1])

        def tor(p, hip=hip, th=th):
            r = _rot(p, th)
            return (hip[0] + r[0], hip[1] + r[1])
        neck = tor(b["neck"])
        hth = th + float(T["tilt"][m, f])
        hc = _rot(b["head_c"], hth)
        hfp = _rot(b["head_f"], hth)
        out.append(dict(m=m, fac=fac, st=st, th=th, hth=hth, hip=hip, neck=neck,
                        core=tor(b["core"]), sh_n=tor(b["sh_n"]),
                        sh_f=tor(b["sh_f"]), hip_n=tor(b["hip_n"]),
                        hip_f=tor(b["hip_f"]),
                        head_c=(neck[0] + hc[0], neck[1] + hc[1]),
                        head_f=(neck[0] + hfp[0], neck[1] + hfp[1])))

    def to_local(rg, w):            # world point -> this mech's local frame
        return ((w[0] - rg["st"]) * rg["fac"], w[1])

    def to_world(rg, p):
        return (rg["st"] + rg["fac"] * p[0], p[1])

    strikes = T["strikes"]
    for m, rg in enumerate(out):
        b = B[m]
        M = MECH[m]
        op = out[1 - m]
        th = rg["th"]
        fl = float(T["flail"][m, f])
        bw = float(T["beam"][f])
        arms = {}
        for arm, (sk, gk) in enumerate((("sh_n", "guard_n"), ("sh_f", "guard_f"))):
            sh = rg[sk]
            g = _rot(M[gk], th)
            guard = (rg["neck"][0] + g[0] - 0.05 * fl, rg["neck"][1] + g[1] + 0.03 * fl)
            if bw > 0:      # beam clash: both hands thrust forward, bracing
                push = (sh[0] + (0.40 if arm == 0 else 0.34), sh[1] - 0.06 - 0.04 * arm)
                guard = (guard[0] + (push[0] - guard[0]) * bw,
                         guard[1] + (push[1] - guard[1]) * bw)
            tgt = guard
            stretch = 1.0
            j = int(T["own"][m, arm, f])
            if j >= 0:
                S = strikes[j]
                t = (f - S["f0"]) / fr
                W, Rr = S["W"] / fr, S["R"] / fr
                A_ = max(2.0, round(0.42 * W))
                kind = S["kind"]
                ch = {"upper": (-0.02, -0.20), "body": (-0.04, -0.16),
                      "hook": (-0.07, -0.02)}.get(kind, (-0.05, -0.08))
                chamber = (sh[0] + ch[0], sh[1] + ch[1])
                if S["att"] == 2:      # the collision: fists meet mid-ring
                    hit = to_local(rg, ((out[0]["st"] + out[1]["st"]) / 2.0, 0.70))
                elif kind == "body":
                    hit = to_local(rg, to_world(op, op["core"]))
                    hit = (hit[0] - 0.02, hit[1] - 0.02)
                elif S["blocked"]:
                    opb = _block_pt(op, MECH[1 - m])     # its forearm
                    hit = to_local(rg, to_world(op, (opb[0] + 0.02, opb[1] - 0.09)))
                else:
                    hw = to_world(op, op["head_c"])
                    hit = to_local(rg, hw)
                    if kind == "upper":
                        hit = (hit[0] - 0.01, hit[1] - 0.05)
                # knuckles meet the target surface: back off by a fist radius
                hit = (hit[0] - 0.035, hit[1])
                arc = {"hook": 0.42, "upper": -0.35, "jab": 0.10,
                       "cross": 0.16, "body": -0.12, "clash": 0.08}[kind]
                if t <= -W:
                    tgt = guard
                elif t < -A_:
                    tgt = _bez(guard, _arc_ctrl(guard, chamber, -0.15), chamber,
                               _ease((t + W) / max(1e-6, W - A_), "io"))
                elif t < 0:
                    u = _ease((t + A_) / A_, "in")
                    tgt = _bez(chamber, _arc_ctrl(chamber, hit, arc), hit, u)
                    stretch = 1.0 + 0.10 * u * (1 - u) * 4
                elif t <= 1:
                    tgt = hit
                else:
                    u = _ease((t - 1) / max(1.0, Rr), "out")
                    tgt = _bez(hit, _arc_ctrl(hit, guard, -0.18), guard, u)
            else:
                kb = int(T["blk"][m, f])
                if kb >= 0 and arm == 0:
                    S = strikes[kb]
                    t = (f - S["f0"]) / fr
                    w = (_ease((t + 6) / 6, "out") if t < 0 else
                         (1.0 if t < 3 else 1.0 - _ease((t - 3) / 7, "io")))
                    bp = _block_pt(rg, M)
                    tgt = (guard[0] + (bp[0] - guard[0]) * w,
                           guard[1] + (bp[1] - guard[1]) * w)
            # reach: if the strike target is out of reach, the hips go with it
            reach = b["l_ua"] + b["l_fa"] + b["l_fist"] * 0.6
            d = math.hypot(tgt[0] - sh[0], tgt[1] - sh[1])
            if j >= 0 and d > reach:
                ex = min(d - reach, 0.10)
                ux = (tgt[0] - sh[0]) / d
                for kk in ("hip", "neck", "core", "sh_n", "sh_f", "hip_n",
                           "hip_f", "head_c", "head_f"):
                    rg[kk] = (rg[kk][0] + ux * ex, rg[kk][1])
                sh = rg[sk]
            u = (tgt[0] - sh[0], tgt[1] - sh[1])
            L = math.hypot(*u) + 1e-9
            wrist_t = (tgt[0] - u[0] / L * b["l_fist"] * 0.85,
                       tgt[1] - u[1] / L * b["l_fist"] * 0.85)
            el, wr = _ik(sh, wrist_t, b["l_ua"], b["l_fa"], "down")
            arms[arm] = dict(sh=sh, el=el, wr=wr, stretch=stretch)
        rg["arms"] = arms
        # LEGS: the feet are PLANTED where the footwork put them (heel
        # pivots lift the ankle about the toe); knees by IK. If a leg cannot
        # reach its planted foot, the HIPS drop (weight sinks into the stance)
        # instead of the foot sliding.
        ank = {}
        for k_ in (0, 1):
            fxw = float(T["feet"][m, k_, f])
            hl = float(T["heel"][m, k_, f])
            ank[k_] = ((fxw - rg["st"]) * rg["fac"],
                       b["ankle"] + float(T["lift"][m, k_, f]) + FOOT_L * math.sin(hl))
        Lmax = (b["l_th"] + b["l_sh"]) * 0.985
        drop = 0.0
        for k_, hk in ((0, "hip_n"), (1, "hip_f")):
            hp = rg[hk]
            dxh = abs(ank[k_][0] - hp[0])
            vmax = math.sqrt(max(1e-6, Lmax * Lmax - dxh * dxh))
            drop = max(drop, (hp[1] - ank[k_][1]) - vmax)
        drop = min(drop, 0.08)
        if drop > 0:
            for kk in ("hip", "neck", "core", "sh_n", "sh_f", "hip_n", "hip_f",
                       "head_c", "head_f"):
                rg[kk] = (rg[kk][0], rg[kk][1] - drop)
            for A_ in arms.values():
                for kk in ("sh", "el", "wr"):
                    A_[kk] = (A_[kk][0], A_[kk][1] - drop)
        legs = {}
        for k_, hk in ((0, "hip_n"), (1, "hip_f")):
            hp = rg[hk]
            kn, an = _ik(hp, ank[k_], b["l_th"], b["l_sh"], M["knee"])
            legs[k_] = dict(hip=hp, kn=kn, an=ank[k_] if math.hypot(
                an[0] - ank[k_][0], an[1] - ank[k_][1]) < 0.02 else an,
                heel=float(T["heel"][m, k_, f]))
        rg["legs"] = legs
    return out


def _block_pt(rg, M):
    """Where a defender raises its near forearm to catch a jab (local)."""
    g = _rot((0.24, 0.08), rg["th"])
    return (rg["neck"][0] + g[0], rg["neck"][1] + g[1])


def _contact(rig, S):
    """World contact point of strike S."""
    if S["kind"] == "headbutt":
        a_ = rig[S["att"]]
        p = a_["head_f"]
        w = (a_["st"] + a_["fac"] * (p[0] + 0.02), p[1])
        return w
    if S["att"] == 2:               # the collision: between the two fists
        w = [(r_["st"] + r_["fac"] * r_["arms"][0]["wr"][0], r_["arms"][0]["wr"][1])
             for r_ in rig]
        return ((w[0][0] + w[1][0]) / 2.0, (w[0][1] + w[1][1]) / 2.0)
    rg = rig[S["att"]]
    A_ = rg["arms"][S["arm"]]
    el, wr = A_["el"], A_["wr"]
    d = (wr[0] - el[0], wr[1] - el[1])
    L = math.hypot(*d) + 1e-9
    k = 0.10
    p = (wr[0] + d[0] / L * k, wr[1] + d[1] / L * k)
    return (rg["st"] + rg["fac"] * p[0], p[1])


# ---------------------------------------------------------------------------
# 3. ASSETS: sprites colourised per palette, arena plate, text, logo
# ---------------------------------------------------------------------------
def _assets(R, P) -> dict:
    C = R.scene_cache
    if "G" in C:
        return C["G"]
    W, H = R.W, R.H
    meta = _geo_meta()
    ppu0 = meta["ppu"]
    S0 = H / VIS_H
    ppu = S0 * 1.12                      # sprites pre-scaled near their use size
    sc = ppu / ppu0
    d = ASSETS / "mechs"
    spr = {}
    for m, M in enumerate(MECH):
        k = M["key"]
        pal = P[k]
        for part in PARTS:
            name = f"{k}_{part}"
            lit = cv2.imread(str(d / f"{name}_lit.png"), cv2.IMREAD_UNCHANGED)
            idm = cv2.imread(str(d / f"{name}_id.png"), cv2.IMREAD_UNCHANGED)
            if lit is None or idm is None:
                raise SystemExit(_BLENDER_HINT.format(d / name))
            pm = meta["parts"][name]
            nw, nh = max(2, int(round(pm["w"] * sc))), max(2, int(round(pm["h"] * sc)))
            lit = cv2.resize(lit, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            idm = cv2.resize(idm, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            al = lit[..., 3]
            L = lit[..., :3].mean(axis=2)
            # ID pass is straight alpha: B, G, R channel order (cv2)
            ib, ig, ir = idm[..., 0], idm[..., 1], idm[..., 2]
            ssum = ib + ig + ir + 1e-4
            wr, wg, wb = ir / ssum, ig / ssum, ib / ssum
            paint = np.asarray(pal["paint"], np.float32)
            frame = np.asarray(pal["frame"], np.float32)
            # painted metal: dark mid-tones, contrasty shading, highlights
            # bleaching toward white (the lit pass is neutral grey)
            Lc = np.clip(L, 0, 1)[..., None]
            shade_p = 0.06 + 0.95 * Lc ** 2.4
            shade_f = 0.25 + 2.4 * Lc ** 1.5
            col = (wr[..., None] * paint * shade_p
                   + wg[..., None] * (frame * shade_f + 6.0)
                   + wb[..., None] * np.float32(14.0))
            col += 255.0 * (np.clip(Lc - 0.86, 0, 1) / 0.14) ** 2 * 0.55
            col = np.clip(col, 0, 255)
            emis = (wb * al).astype(np.float32)
            piv = (pm["pivot"][0] * sc, pm["pivot"][1] * sc)
            bvec = np.asarray((pm["bone"][0], -pm["bone"][1]), np.float32)
            bvec /= (np.linalg.norm(bvec) + 1e-9)
            rgba = np.dstack([col * al[..., None], al]).astype(np.float32)
            if M["facing"] < 0:          # mirror for the right-hand mech
                rgba = rgba[:, ::-1].copy()
                emis = emis[:, ::-1].copy()
                piv = (nw - piv[0], piv[1])
                bvec = np.asarray((-bvec[0], bvec[1]), np.float32)
            spr[(m, part)] = dict(rgba=rgba, emis=emis, piv=piv, b=bvec,
                                  w=nw, h=nh)
    # ---- arena plate
    ad = ASSETS / "arena"
    aj = ad / "anchors.json"
    if not aj.is_file():
        raise SystemExit(_BLENDER_HINT.format(aj))
    anc = json.loads(aj.read_text())
    Wp, Hp = anc["size"]
    tgt_w = int(round(W * anc["cam"]["margin"]))
    ks = tgt_w / Wp                       # plate px -> frame px at zoom 1
    tgt_h = int(round(Hp * ks))

    def load(name):
        im = cv2.imread(str(ad / name), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(_BLENDER_HINT.format(ad / name))
        im = cv2.resize(im, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255.0
        return im[..., [2, 1, 0]], im[..., 3]
    s_rgb, s_a = load("structure.png")
    c0, a0 = load("crowd_0.png")
    c1, a1 = load("crowd_1.png")
    pt = np.asarray(P["plate"], np.float32)
    ct = np.asarray(P["crowd_tint"], np.float32)
    plates = []
    # PNGs are straight alpha; the crowd was rendered with the structure as a
    # holdout, so "crowd OVER structure" is the correct premultiplied stack
    for cr, ca in ((c0, a0), (c1, a1)):
        prem = (s_rgb * pt * (s_a * (1 - ca))[..., None]
                + cr * ct * 1.15 * ca[..., None])
        al = s_a * (1 - ca) + ca
        plates.append(np.dstack([prem, al]).astype(np.float32))
    hor = anc.get("horizon", [Wp / 2, Hp * (0.5 + (HOR_Y - 0.5) / anc["cam"]["margin"])])
    G = dict(spr=spr, ppu=ppu, plates=plates, ks=ks, pw=tgt_w, ph=tgt_h,
             pu0=tgt_w / 2.0, pv0=float(hor[1]) * ks,
             seats=np.asarray(anc["seats"], np.float32)[:, :2] * ks,
             seat_d=np.asarray(anc["seats"], np.float32)[:, 2],
             lamps=np.asarray(anc["lamps"], np.float32),
             jumbo=np.asarray(anc["jumbo"], np.float32)[:, :2] * ks,
             board=np.asarray(anc["board"], np.float32)[:, :2] * ks,
             ribbon=np.asarray(anc["ribbon"], np.float32)[:, :2] * ks,
             barrier_y=anc["cam"]["barrier_y"])
    G["lamps"][:, :2] *= ks
    # crowd lights: ~40% of seats carry a light; colour + phase seeded
    rs = np.random.default_rng(31)
    ns = len(G["seats"])
    has = rs.random(ns) < 0.30
    G["cl_idx"] = np.flatnonzero(has)
    nl = len(G["cl_idx"])
    kind = rs.choice(len(P["crowd"]), nl, p=[0.55, 0.15, 0.15, 0.15])
    G["cl_col"] = np.asarray(P["crowd"], np.float32)[kind]
    G["cl_phase"] = rs.random(nl).astype(np.float32)
    G["cl_amp"] = rs.uniform(0.5, 1.0, nl).astype(np.float32)
    G["cl_jump"] = np.full(nl, 0.028 * S0 * CAM_D / (CAM_D + 4.0), np.float32)
    # ---- jumbotron content at its zoom-1 size
    jw = int(np.ptp(G["jumbo"][:, 0]))
    jh = int(np.ptp(G["jumbo"][:, 1]))
    jw, jh = max(16, jw), max(8, jh)
    G["jw"], G["jh"] = jw, jh
    ttl = (R.title or "HIT A BUMP").upper()
    txt = dict(
        title=_place(_text_mask(R, ttl, "Anton-Regular.ttf", jw * 0.86, jh * 0.52), jh, jw),
        bump=_place(_text_mask(R, "BUMP", "Anton-Regular.ttf", jw * 0.8, jh * 0.70), jh, jw),
        ready=_place(_text_mask(R, "READY?", "Anton-Regular.ttf", jw * 0.7, jh * 0.55), jh, jw),
        fight=_place(_text_mask(R, "FIGHT!", "Anton-Regular.ttf", jw * 0.75, jh * 0.6), jh, jw),
        power=_place(_text_mask(R, "MAX POWER", "Anton-Regular.ttf", jw * 0.86, jh * 0.5), jh, jw),
        sub=_place(_text_mask(R, "TZEKE000", "BebasNeue-Regular.ttf", jw * 0.3, jh * 0.11), jh, jw, cy=0.87),
    )
    for k in range(1, 10):
        txt[f"round{k}"] = _place(_text_mask(R, f"ROUND {k}", "Anton-Regular.ttf",
                                             jw * 0.8, jh * 0.5), jh, jw)
    lm = _logo_masks(R, int(jh * 0.70))
    if lm is not None:
        txt["logo"] = tuple(_place(x, jh, jw, cy=0.43) for x in lm)
    # LED dot grid for the screen
    pitch = max(2, int(round(jh / 70)))
    yy, xx = np.mgrid[0:jh, 0:jw]
    dots = (0.55 + 0.45 * ((yy % pitch) < pitch - 1) * ((xx % pitch) < pitch - 1)).astype(np.float32)
    G["jtxt"], G["jdots"] = txt, dots
    # ---- LED board strip text (barrier) + ribbon (fascia)
    rh = max(4, int(abs(G["ribbon"][2, 1] - G["ribbon"][0, 1])))
    G["ribbon_tx"] = _text_mask(R, ("ROBOT FIGHT   //   " + ttl + "   //   ") * 3,
                                "BebasNeue-Regular.ttf", 10 ** 6, rh * 0.80)
    G["rh"] = rh
    # ---- floor emblem texture (ring + logo), sampled in world units
    es = 512
    em = np.zeros((es, es), np.float32)
    c = es // 2
    cv2.circle(em, (c, c), int(es * 0.47), 1.0, max(2, es // 70), cv2.LINE_AA)
    cv2.circle(em, (c, c), int(es * 0.42), 0.6, max(1, es // 160), cv2.LINE_AA)
    for k in range(48):
        a_ = 2 * math.pi * k / 48
        r0, r1 = es * 0.425, es * (0.455 if k % 4 else 0.468)
        cv2.line(em, (int(c + r0 * math.cos(a_)), int(c + r0 * math.sin(a_))),
                 (int(c + r1 * math.cos(a_)), int(c + r1 * math.sin(a_))), 0.7, 2,
                 cv2.LINE_AA)
    lm2 = _logo_masks(R, int(es * 0.62))
    if lm2 is not None:
        dk, fl, rd = (_place(x, es, es) for x in lm2)
        em = np.maximum(em, dk * 0.9 + fl * 0.25 + rd * 0.8)
    G["emblem"] = em
    G["emblem_r"] = 0.62                  # world radius of the emblem ring
    # ---- HUD
    hud_h = max(8, int(H * 0.030))
    G["hud_names"] = [_text_mask(R, M["name"], "Anton-Regular.ttf", W * 0.2, hud_h * 0.95)
                      for M in MECH]
    # vignette-free gradient backdrop (behind the plate)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    gtop = np.asarray(P["bg_top"], np.float32)
    ghor = np.asarray(P["bg_hor"], np.float32)
    g = np.clip(ys / HOR_Y, 0, 1) ** 1.6
    G["sky"] = (gtop * (1 - g) + ghor * g)[:, None, :].repeat(1, axis=1)
    G["xs"] = np.arange(W, dtype=np.float32)
    C["G"] = G
    return G


# ---------------------------------------------------------------------------
# 4. DRAWING
# ---------------------------------------------------------------------------
def _camera(R, T, i):
    H = R.H
    z = float(T["zoom"][i])
    S0 = H / VIS_H
    Dc = CAM_D / z
    F = S0 * CAM_D
    return dict(z=z, Dc=Dc, F=F, S=F / Dc, camx=float(T["camx"][i]),
                ch=float(T["camh"][i]), hor=HOR_Y * H)


def _w2s(cam, W, x, h, depth=0.0):
    d = cam["Dc"] + depth
    return (W / 2.0 + (x - cam["camx"]) * cam["F"] / d,
            cam["hor"] + (cam["ch"] - h) * cam["F"] / d)


def _plate_affine(G, cam, W):
    """Plate pixels -> frame pixels for this camera. Reference depth = the
    jumbotron's, so the screen stays framed on a push-in; at the barrier the
    camera-height term compensates to within a pixel or two, and the floor
    runs past the barrier so no seam can open."""
    yr = 3.6
    s = (CAM_D + yr) / (cam["Dc"] + yr)
    k = s                                # plate was pre-scaled to zoom-1 px
    tx = -cam["camx"] * cam["F"] / (cam["Dc"] + yr)
    ty = (cam["ch"] - CAM_H) * cam["F"] / (cam["Dc"] + yr)
    # x = W/2 + k (u - u0) + tx ; y = hor + k (v - v0) + ty
    return np.array([[k, 0, W / 2.0 - k * G["pu0"] + tx],
                     [0, k, cam["hor"] - k * G["pv0"] + ty]], np.float32)


def draw(R, img, i, a, lv) -> None:
    P = lv.scene_palette(R, _MOD)
    T = _timeline(R, a, lv)
    G = _assets(R, P)
    W, H = R.W, R.H
    fps = R.fps
    fr = fps / 30.0
    t = i / float(fps)
    cam = _camera(R, T, i)
    hotf = float(T["hot_f"][i])
    hot = bool(T["hot"][i])
    big = bool(T["big"][i])
    pre = float(T["pre"][i])
    black = bool(T["black"][i])
    heat = float(T["heat"][i])
    spark = float(T["spark"][i])
    kage = int(T["kage"][i])
    kenv = 0.70 ** kage if kage < 40 else 0.0
    sage = int(T["sage"][i])
    senv = 0.70 ** sage if sage < 40 else 0.0
    ph = float(T["ph"][i])
    bw = float(T["beam"][i])
    # house lights: blackout kills them; the build makes them flicker faster
    power = 0.0 if black else 1.0
    if pre > 0.3 and not hot and not black:
        rate = 2.0 + 10.0 * pre
        power *= 0.55 + 0.45 * (1.0 if math.sin(2 * math.pi * t * rate) > -0.3 else 0.25)
    flood = power * (0.70 + 0.22 * hotf + 0.16 * kenv * hotf)
    # ---- backdrop gradient + floor (floor first: the plate's barrier covers
    # the floor's far edge, so the two meet without a seam)
    img[:] = G["sky"] * (0.35 + 0.65 * power)
    rig = _rig(R, T, i)
    mech = _render_mechs(R, T, G, P, rig, cam, i)
    _floor(R, img, T, G, P, cam, i, mech, power, heat, kenv, hotf)
    # ---- arena plate (structure + crowd; crowd jumps on the beat in drops)
    jump = 0.0
    if hot:
        jump = _ease(1.0 - min(1.0, ph / 0.45), "io") if ph < 0.45 else 0.0
    elif pre > 0.4 and not black:
        jump = 1.0 if (ph < 0.5) else 0.0
    pl = G["plates"][0] if jump < 0.5 else G["plates"][1]
    M = _plate_affine(G, cam, W)
    warped = cv2.warpAffine(pl, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    lum = (0.25 + 0.95 * flood + 0.18 * heat * power)
    img *= (1.0 - warped[..., 3:4])
    img += warped[..., :3] * lum
    # haze between the stands and the floor
    hz = np.asarray(P["haze"], np.float32)
    band = np.exp(-((np.arange(H, dtype=np.float32) - cam["hor"]) / (0.22 * H)) ** 2)
    img += band[:, None, None] * hz * (0.10 + 0.12 * heat * power + 0.10 * kenv * hotf * power)
    # ---- crowd lights, jumbotron, LED boards, floodlights
    _crowd_lights(R, img, T, G, P, M, i, t, power, hot, pre, black, ph, jump)
    _boards(R, img, T, G, P, M, i, t, power, kenv, hotf)
    _jumbotron(R, img, T, G, P, M, i, t, mech, power, kenv, hotf, black)
    soft = np.zeros((H // 4, W // 4, 3), np.float32)
    _floods(R, img, soft, T, G, P, M, cam, i, t, flood, power, hotf, kenv)
    # ---- the mechs
    _composite_mechs(R, img, soft, T, G, P, rig, cam, mech, i, power, kenv, senv,
                     hotf, spark, pre, black)
    # ---- FX
    sharp = np.zeros((H, W, 3), np.uint8)
    _fx_cores(R, img, sharp, soft, T, P, rig, cam, i, pre, black, spark, heat, hotf, bw)
    _fx_hits(R, img, sharp, soft, T, P, cam, i)
    if bw > 0:
        _fx_beam(R, sharp, soft, T, P, rig, cam, i, bw)
    _fx_embers(R, soft, T, P, i, t, hotf, power)
    img += sharp.astype(np.float32) * 1.15
    img += cv2.resize(soft, (W, H), interpolation=cv2.INTER_LINEAR)[:H, :W]


def _render_mechs(R, T, G, P, rig, cam, i):
    """Both mechs into an RGBA(premult) + emissive buffer over a ROI.
    The attacker of the most recent strike is drawn in FRONT."""
    W, H = R.W, R.H
    S = cam["S"]
    spr = G["spr"]
    sc = S / G["ppu"]
    # ROI = both mechs' extent
    xs, ys = [], []
    for rg in rig:
        for p in (rg["hip"], rg["neck"], rg["head_c"]):
            wx = rg["st"] + rg["fac"] * p[0]
            sx, sy = _w2s(cam, W, wx, p[1])
            xs.append(sx)
            ys.append(sy)
    pad = int(0.75 * S)
    x0 = max(0, int(min(xs)) - pad)
    x1 = min(W, int(max(xs)) + pad)
    y0 = max(0, int(min(ys)) - int(0.35 * S))
    y1 = min(H, int(cam["hor"] + cam["ch"] * S + 0.12 * S))
    roi = (x0, y0, x1, y1)
    rw, rh = x1 - x0, y1 - y0
    buf = np.zeros((rh, rw, 6), np.float32)
    js = [j for j in range(len(T["strikes"])) if T["strikes"][j]["f0"] <= i + 6]
    front = 0
    if js:
        S_ = T["strikes"][js[-1]]
        front = S_["att"] if S_["att"] in (0, 1) else 0
    order = (1 - front, front)
    ol = max(1, int(round(H / 540)))
    kern = np.ones((2 * ol + 1, 2 * ol + 1), np.uint8)
    for m in order:
        rg = rig[m]
        fac = rg["fac"]
        pal = P[MECH[m]["key"]]
        gcol = np.asarray(pal["glow"], np.float32)
        hf = float(T["flash"][m, i])
        sq = float(T["squash"][m, i])

        def W2(p):
            return _w2s(cam, W, rg["st"] + fac * p[0], p[1])
        A = rg["arms"]
        Lg = rg["legs"]
        items = [
            ("uarm", A[1]["sh"], A[1]["el"], 0.52, 1.0),
            ("farm", A[1]["el"], A[1]["wr"], 0.52, A[1]["stretch"]),
            ("fist", A[1]["wr"], None, 0.52, 1.0),
            ("thigh", Lg[1]["hip"], Lg[1]["kn"], 0.55, 1.0),
            ("shin", Lg[1]["kn"], Lg[1]["an"], 0.55, 1.0),
            ("foot", Lg[1]["an"], "flat", 0.55, 1.0),
            ("torso", rg["hip"], "torso", 1.0, 1.0 - sq),
            ("thigh", Lg[0]["hip"], Lg[0]["kn"], 0.92, 1.0),
            ("shin", Lg[0]["kn"], Lg[0]["an"], 0.92, 1.0),
            ("foot", Lg[0]["an"], "flat", 0.92, 1.0),
            ("head", rg["neck"], "head", 1.0, 1.0),
            ("uarm", A[0]["sh"], A[0]["el"], 1.0, 1.0),
            ("farm", A[0]["el"], A[0]["wr"], 1.0, A[0]["stretch"]),
            ("fist", A[0]["wr"], None, 1.0, 1.0),
        ]
        for part, j0, j1, shade, stretch in items:
            sp = spr[(m, part)]
            p0 = W2(j0)
            if j1 == "torso":
                th = rg["th"]
                d = (fac * math.sin(th), -math.cos(th))
            elif j1 == "head":
                th = rg["hth"]
                d = (fac * math.sin(th), -math.cos(th))
            elif j1 == "flat":           # heel pivot: toe down, about the toe
                hl = Lg[0 if shade > 0.9 else 1]["heel"]
                d = (fac * math.cos(hl), math.sin(hl))
            elif j1 is None:            # fist follows the forearm
                arm = A[0] if shade > 0.9 else A[1]
                e, w_ = W2(arm["el"]), W2(arm["wr"])
                d = (w_[0] - e[0], w_[1] - e[1])
            else:
                p1 = W2(j1)
                d = (p1[0] - p0[0], p1[1] - p0[1])
            L = math.hypot(*d) + 1e-9
            d = (d[0] / L, d[1] / L)
            b = sp["b"]
            ang = math.atan2(d[1], d[0]) - math.atan2(float(b[1]), float(b[0]))
            c, s_ = math.cos(ang), math.sin(ang)
            Rm = np.array([[c, -s_], [s_, c]], np.float32) * sc
            if abs(stretch - 1.0) > 1e-3:   # squash/stretch along the bone
                bb = np.outer(b, b)
                Bm = stretch * bb + (1.0 / math.sqrt(stretch)) * (np.eye(2) - bb)
                Rm = Rm @ Bm.astype(np.float32)
            piv = np.asarray(sp["piv"], np.float32)
            tvec = np.asarray((p0[0] - x0, p0[1] - y0), np.float32) - Rm @ piv
            # transformed bounds within the ROI
            cs = np.array([[0, 0], [sp["w"], 0], [0, sp["h"]], [sp["w"], sp["h"]]],
                          np.float32)
            q = cs @ Rm.T + tvec
            bx0 = max(0, int(q[:, 0].min()) - 1 - ol)
            by0 = max(0, int(q[:, 1].min()) - 1 - ol)
            bx1 = min(rw, int(q[:, 0].max()) + 2 + ol)
            by1 = min(rh, int(q[:, 1].max()) + 2 + ol)
            if bx1 <= bx0 or by1 <= by0:
                continue
            Mx = np.hstack([Rm, (tvec - np.array([bx0, by0], np.float32))[:, None]])
            ww, hh = bx1 - bx0, by1 - by0
            wa = cv2.warpAffine(sp["rgba"], Mx, (ww, hh), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            we = cv2.warpAffine(sp["emis"], Mx, (ww, hh), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            # a dark OUTLINE around every part (cel-style): separates a fist
            # from the chest behind it when both are the same colour
            al2 = cv2.dilate(wa[..., 3], kern)
            reg = buf[by0:by1, bx0:bx1]          # [r g b a eA eB], premult
            reg *= (1.0 - al2)[..., None]
            if shade != 1.0:
                wa[..., :3] *= shade
            if hf > 0:                   # hit flash: the armour blanches
                wa[..., :3] += wa[..., 3:4] * (50.0 * hf)
            reg[..., :3] += wa[..., :3]
            reg[..., 3] += al2
            reg[..., 4 + m] += we * (shade ** 0.5)
    em = (buf[..., 4:5] * np.asarray(P["A"]["glow"], np.float32)
          + buf[..., 5:6] * np.asarray(P["B"]["glow"], np.float32))
    return dict(roi=roi, buf=buf[..., :4], em=em, front=front)


def _floor(R, img, T, G, P, cam, i, mech, power, heat, kenv, hotf):
    """Perspective floor (half res): gloss, grid, the logo emblem, light
    pools, impact ripples, mech shadows + reflections."""
    W, H = R.W, R.H
    hor = cam["hor"]
    yb = cam["hor"] + cam["ch"] * cam["F"] / (cam["Dc"] + G["barrier_y"] + 1.2)
    r0 = max(0, int(yb) - 2)
    if r0 >= H:
        return
    h2, w2 = (H - r0 + 1) // 2, W // 2
    ys = r0 + 2.0 * np.arange(h2, dtype=np.float32) + 0.5
    xs = 2.0 * np.arange(w2, dtype=np.float32) + 0.5
    dd = cam["ch"] * cam["F"] / np.maximum(ys - hor, 1e-3)      # camera dist
    yw = (dd - cam["Dc"])[:, None]                                # depth
    xw = cam["camx"] + (xs[None, :] - W / 2.0) * dd[:, None] / cam["F"]
    pix = (dd / cam["F"] * 2.0)[:, None]                          # world/px
    fl = np.asarray(P["floor"], np.float32)
    near = np.clip(1.0 - yw / 3.0, 0.2, 1.4)
    base = fl * (0.6 + 0.5 * near)[..., None] * (0.35 + 0.65 * power)
    # grid (anti-aliased by the pixel footprint)
    gs = 0.25
    gx = np.abs(((xw / gs) % 1.0) - 0.5) * gs
    gy = np.abs(((yw / gs) % 1.0) - 0.5) * gs
    lw = np.maximum(pix * 0.9, 0.004)
    grid = np.clip(1.0 - (gs / 2 - np.minimum(gx, gy)) / lw, 0, 1)
    fade = np.clip(1.0 - (yw + 0.5) / 4.0, 0, 1) * np.clip(yw + 3.0, 0, 1)
    gamt = (0.10 + 0.35 * heat + 0.30 * kenv * hotf) * power + 0.03
    out = base + (grid * fade)[..., None] * np.asarray(P["grid"], np.float32) * gamt
    # emblem under the fighters (world circle centred between them)
    ecx = float(T["cx"][i])
    er = G["emblem_r"]
    emx = 0.0                            # the emblem is PAINTED: world-fixed
    es = G["emblem"].shape[0]
    mu = ((xw - emx) / (2 * er) + 0.5) * es
    mv = np.broadcast_to(((yw - 0.35) / (2 * er) + 0.5) * es, mu.shape)
    emb = cv2.remap(G["emblem"], mu.astype(np.float32), mv.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    eamt = (0.25 + 0.55 * heat + 0.35 * kenv * hotf) * (0.25 + 0.75 * power)
    out += emb[..., None] * np.asarray(P["emblem"], np.float32) * eamt
    # floodlight pools (follow the beams' aim points)
    sw = float(T["sweep"][i])
    for k in range(4):
        ax = ecx + 0.55 * math.sin(2 * math.pi * (sw * 0.5 + k * 0.27)) + (k - 1.5) * 0.35
        ay = 0.6 + 0.5 * math.sin(2 * math.pi * (sw * 0.31 + k * 0.4))
        pool = np.exp(-(((xw - ax) / 0.35) ** 2 + ((yw - ay) / 0.6) ** 2))
        out += pool[..., None] * np.asarray(P["flood"], np.float32) * (0.12 * power * (1 + 0.6 * kenv * hotf))
    # impact ripples on the floor (drop-entry clashes + heavy hits)
    S_ = T["strikes"]
    f0s = T["s_f0"]
    lo = np.searchsorted(f0s, i - int(30 * R.fps / 30))
    hi = np.searchsorted(f0s, i, side="right")
    for j in range(lo, hi):
        Sj = S_[j]
        if Sj["blocked"] or Sj["s"] < 0.9:
            continue
        age = (i - Sj["f0"]) / (R.fps / 30.0)
        px, _ = T["s_pt"][j]
        rr = 0.1 + age * (0.09 if Sj["att"] == 2 else 0.06)
        ring = np.exp(-((np.sqrt((xw - px) ** 2 + ((yw - 0.0) * 1.0) ** 2) - rr) / 0.04) ** 2)
        amp = (1.4 if Sj["att"] == 2 else 0.6) * math.exp(-age / 9.0)
        out += ring[..., None] * np.asarray(P["spark"][1], np.float32) * amp
    # mech shadows (soft, under each mech)
    for m in (0, 1):
        sx = float(T["stance"][m, i])
        sh = np.exp(-(((xw - sx) / 0.30) ** 2 + ((yw - 0.0) / 0.12) ** 2))
        out *= (1.0 - 0.55 * sh)[..., None]
    up = cv2.resize(out, (W, h2 * 2), interpolation=cv2.INTER_LINEAR)[: H - r0]
    img[r0:H] = up
    # reflections of the mechs (flip about each mech's ground line)
    x0, y0, x1, y1 = mech["roi"]
    gy = int(round(cam["hor"] + cam["ch"] * cam["S"]))
    buf, em = mech["buf"], mech["em"]
    rgbp = buf[..., :3] + em * 40.0
    rows = np.arange(gy, H)
    src = 2 * gy - rows - y0
    ok = (src >= 0) & (src < buf.shape[0])
    if ok.any():
        rr_ = rows[ok]
        refl = rgbp[src[ok]]
        al = buf[src[ok], :, 3:4]
        fade = np.exp(-(rr_ - gy) / (0.10 * H)).astype(np.float32)[:, None, None]
        refl = cv2.blur(refl, (5, 3))
        tgt = img[rr_, x0:x1]
        img[rr_, x0:x1] = tgt * (1 - 0.35 * al * fade) + refl * 0.30 * fade


def _crowd_lights(R, img, T, G, P, M, i, t, power, hot, pre, black, ph, jump):
    W, H = R.W, R.H
    idx = G["cl_idx"]
    pts = G["seats"][idx]
    X = pts[:, 0] * M[0, 0] + M[0, 2]
    Y = pts[:, 1] * M[1, 1] + M[1, 2] - 0.012 * H * M[0, 0]
    phs = G["cl_phase"]
    if hot:
        X = X + 3.0 * np.sin(2 * np.pi * (t * 0.6 + phs)) * M[0, 0]
        Y = Y - G["cl_jump"] * M[0, 0] * jump
        on = 0.55 + 0.45 * np.exp(-((ph - phs * 0.15) % 1.0) * 5.0)
    elif pre > 0.05 and not black:
        rate = 1.5 + 9.0 * pre
        on = (np.sin(2 * np.pi * (t * rate + phs)) > 0.0).astype(np.float32) * 0.9 + 0.1
        Y = Y - G["cl_jump"] * M[0, 0] * jump
    else:                                # calm: phone lights swaying
        X = X + 4.0 * np.sin(2 * np.pi * (t * 0.25 + phs)) * M[0, 0]
        on = 0.40 + 0.20 * np.sin(2 * np.pi * (t * 0.4 + phs * 3))
    if black:
        on = on * 0.35
    amp = on * G["cl_amp"]
    h2, w2 = H // 2, W // 2
    xi = (X / 2).astype(np.int32)
    yi = (Y / 2).astype(np.int32)
    ok = (xi >= 0) & (xi < w2) & (yi >= 0) & (yi < h2)
    buf = np.zeros((h2 * w2, 3), np.float32)
    lin = yi[ok] * w2 + xi[ok]
    v = G["cl_col"][ok] * amp[ok, None]
    for c in range(3):
        buf[:, c] = np.bincount(lin, weights=v[:, c], minlength=h2 * w2)
    buf = buf.reshape(h2, w2, 3)
    sig = max(0.6, 0.0011 * H)
    buf = cv2.GaussianBlur(buf, (0, 0), sig) * (2.0 * math.pi * sig * sig) * 0.9
    img += cv2.resize(buf, (W, H), interpolation=cv2.INTER_LINEAR)


def _boards(R, img, T, G, P, M, i, t, power, kenv, hotf):
    """LED boards: the barrier strip and the balcony ribbon, scrolling."""
    W, H = R.W, R.H
    k = float(M[0, 0])
    _eq_board(R, img, T, G, P, M, i, power, kenv, hotf)
    for key, tx, col, col2, spd in (("ribbon", G["ribbon_tx"], P["board2"], P["board"], -70.0),):
        q = G[key]
        ya = q[0, 1] * k + M[1, 2]
        yb = q[2, 1] * k + M[1, 2]
        y0, y1 = int(round(min(ya, yb))), int(round(max(ya, yb)))
        if y1 <= y0 + 1 or y1 <= 0 or y0 >= H:
            continue
        hh = y1 - y0
        th, tw = tx.shape
        txr = cv2.resize(tx, (max(1, int(tw * hh / th * 0.82)), max(1, int(hh * 0.82))),
                         interpolation=cv2.INTER_AREA)
        span = max(1, txr.shape[1] // 3)
        off = int(t * spd * (1 + 1.2 * hotf) * H / 1080) % span
        cols = (off + np.arange(W) + int(M[0, 2])) % txr.shape[1]
        strip = np.zeros((hh, W), np.float32)
        yo = (hh - txr.shape[0]) // 2
        strip[yo:yo + txr.shape[0]] = txr[:, cols]
        # LED cell texture + kick flash of the whole board in drops
        cell = max(2, hh // 6)
        strip *= (0.6 + 0.4 * ((np.arange(W) % cell) < cell - 1))[None, :]
        c1 = np.asarray(col, np.float32)
        c2 = np.asarray(col2, np.float32)
        bgc = c2 * (0.08 + 0.25 * kenv * hotf)
        y0c, y1c = max(0, y0), min(H, y1)
        reg = img[y0c:y1c]
        s_ = strip[y0c - y0:y1c - y0]
        reg *= 0.35
        reg += (s_[..., None] * c1 * 0.30 + bgc * 0.5) * power


def _eq_board(R, img, T, G, P, M, i, power, kenv, hotf):
    """The barrier's LED board as a long EQUALISER: the spectrum, mirrored
    from the centre outward, in LED cells; kicks flash the base row."""
    W, H = R.W, R.H
    k = float(M[0, 0])
    q = G["board"]
    ya = q[0, 1] * k + M[1, 2]
    yb = q[2, 1] * k + M[1, 2]
    y0, y1 = int(round(min(ya, yb))), int(round(max(ya, yb)))
    if y1 <= y0 + 2 or y1 <= 0 or y0 >= H:
        return
    hh = y1 - y0
    bars = T["bars"][i]
    nb = len(bars)
    cell = max(3, hh // 7)
    ncol = W // cell + 1
    xc = (np.arange(ncol) * cell + cell / 2 - W / 2) / (W / 2)      # -1..1
    band = np.clip(np.abs(xc) * (nb - 1) * 0.9, 0, nb - 1).astype(np.int32)
    lvl = np.clip(bars[band] * 1.35, 0, 1)
    nrow = max(2, hh // cell)
    rows = (np.arange(nrow)[::-1] + 0.5) / nrow                    # top -> bottom
    lit = (rows[:, None] <= lvl[None, :]).astype(np.float32)
    c1 = np.asarray(P["board"], np.float32)
    c2 = np.asarray(P["board2"], np.float32)
    colr = c1[None, None, :] * (1 - rows[:, None, None]) + c2[None, None, :] * rows[:, None, None]
    tile = lit[..., None] * colr * (0.30 + 0.35 * kenv * hotf)
    tile[-1] += c1 * 0.25 * kenv * hotf                              # kick row
    up = cv2.resize(tile, (ncol * cell, nrow * cell), interpolation=cv2.INTER_NEAREST)
    gx = (np.arange(ncol * cell) % cell) < cell - 1
    gy = (np.arange(nrow * cell) % cell) < cell - 1
    up *= (gx[None, :] & gy[:, None])[..., None]
    y0c, y1c = max(0, y0), min(H, y0 + up.shape[0])
    reg = img[y0c:y1c]
    reg *= 0.35
    reg += up[y0c - y0:y1c - y0, :W] * power


def _jumbotron(R, img, T, G, P, M, i, t, mech, power, kenv, hotf, black):
    W, H = R.W, R.H
    k = float(M[0, 0])
    q = G["jumbo"]
    x0 = q[:, 0].min() * k + M[0, 2]
    x1 = q[:, 0].max() * k + M[0, 2]
    y0 = q[:, 1].min() * k + M[1, 2]
    y1 = q[:, 1].max() * k + M[1, 2]
    if black:
        return
    jw, jh = G["jw"], G["jh"]
    tx = G["jtxt"]
    hot = bool(T["hot"][i])
    big = bool(T["big"][i])
    pre = float(T["pre"][i])
    bi = int(T["bi"][i])
    sb = int(T["sec_bar"][i])
    ea = int(T["entry_age"][i])
    punch = float(T["punch"][i])
    bw = float(T["beam"][i])
    rnd_ = int(T["round"][i])
    col = np.asarray(P["screen"], np.float32)
    mode = "logo"
    if punch > 0.12:
        mode = "bump"
    elif bw > 0:
        mode = "power"
    elif hot:
        if sb < 2:
            mode = "title"
        else:
            mode = ("cam", "title", "cam", f"round{rnd_}")[(sb // 1) % 4]
    elif pre > 0.25:
        mode = "fight" if pre > 0.82 else ("ready" if bi % 2 == 0 else "logo")
    if mode == "cam":
        bx0, by0, bx1, by1 = mech["roi"]
        buf = mech["buf"]
        # live close-up of the exchange: centre of the two mechs, chest height
        cxs = (_w2s(_camera(R, T, i), W, float(T["cx"][i]), 0.66))
        cw_ = (bx1 - bx0) * 0.55
        ch_ = cw_ * jh / jw
        sx0 = int(cxs[0] - bx0 - cw_ / 2)
        sy0 = int(cxs[1] - by0 - ch_ / 2)
        Mx = np.array([[jw / cw_, 0, -sx0 * jw / cw_], [0, jh / ch_, -sy0 * jh / ch_]],
                      np.float32)
        crop = cv2.warpAffine(buf, Mx, (jw, jh), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_CONSTANT)
        bgc = np.asarray(P["bg_hor"], np.float32) * 1.4
        rgb = crop[..., :3] * 1.1 + bgc * (1 - crop[..., 3:4])
        # REC dot + frame lines
        rgb[int(jh * 0.06):int(jh * 0.14), int(jw * 0.04):int(jw * 0.07)] += \
            np.asarray((255, 40, 40), np.float32) * (1.0 if (i // int(R.fps / 2)) % 2 else 0.3)
    else:
        if mode == "logo" and "logo" in tx:
            dk, fl, rd = tx["logo"]
            lum = dk * 1.0 + fl * 0.30 + tx["sub"] * 0.9
            rgb = lum[..., None] * col + rd[..., None] * np.asarray((255, 40, 40), np.float32)
        else:
            m_ = tx.get(mode, tx["title"])
            c2 = col if mode not in ("title", "bump", "fight") else \
                np.asarray(P["spark"][1], np.float32)
            rgb = m_[..., None] * c2 * (0.85 + 0.15 * kenv)
            rgb += np.asarray(P["bg_hor"], np.float32) * 0.6
    rgb = rgb * G["jdots"][..., None]
    gain = (0.9 + 0.25 * kenv * hotf)
    if ea < 8:                               # entry: flicker on, slam bright
        gain *= (0.3, 1.6, 0.6, 1.4, 1.0, 1.2, 1.0, 1.1)[ea]
    rgb *= gain * (0.4 + 0.6 * power)
    # place (axis-aligned scale) into the frame
    ow, oh = int(round(x1 - x0)), int(round(y1 - y0))
    if ow < 4 or oh < 4:
        return
    sm_ = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LINEAR)
    X0, Y0 = int(round(x0)), int(round(y0))
    ax0, ay0 = max(0, X0), max(0, Y0)
    ax1, ay1 = min(W, X0 + ow), min(H, Y0 + oh)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    img[ay0:ay1, ax0:ax1] = sm_[ay0 - Y0:ay1 - Y0, ax0 - X0:ax1 - X0]
    # screen glow spilling onto the stands
    q4 = cv2.resize(sm_, (max(1, ow // 8), max(1, oh // 8)), interpolation=cv2.INTER_AREA)
    pad = 2
    g = np.zeros((q4.shape[0] + 2 * pad * 2, q4.shape[1] + 2 * pad * 2, 3), np.float32)
    g[pad * 2:pad * 2 + q4.shape[0], pad * 2:pad * 2 + q4.shape[1]] = q4
    g = cv2.GaussianBlur(g, (0, 0), 2.0)
    gw, gh = g.shape[1] * 8, g.shape[0] * 8
    gl = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_LINEAR)
    GX0, GY0 = X0 - pad * 2 * 8, Y0 - pad * 2 * 8
    bx0, by0 = max(0, GX0), max(0, GY0)
    bx1, by1 = min(W, GX0 + gw), min(H, GY0 + gh)
    if bx1 > bx0 and by1 > by0:
        img[by0:by1, bx0:bx1] += gl[by0 - GY0:by1 - GY0, bx0 - GX0:bx1 - GX0] * 0.35


def _floods(R, img, soft, T, G, P, M, cam, i, t, flood, power, hotf, kenv):
    """Light-tower lamp heads + volumetric beams (quarter res, additive)."""
    W, H = R.W, R.H
    if power <= 0.01:
        return
    k = float(M[0, 0])
    L = G["lamps"]
    lx = L[:, 0] * k + M[0, 2]
    ly = L[:, 1] * k + M[1, 2]
    side = L[:, 3]
    bi = int(T["bi"][i])
    lc = np.asarray(P["lamp"], np.float32)
    fc = np.asarray(P["flood"], np.float32)
    hot = bool(T["hot"][i])
    r_l = max(1.5, 0.006 * H)
    for j in range(len(lx)):
        row = j % 3
        lvl = 0.75
        if hot:          # chase: one row per beat flares, every kick lifts all
            lvl = 0.55 + 0.45 * (row == bi % 3) + 0.35 * kenv
        _blob(soft, lx[j] / 4, ly[j] / 4, r_l * 1.8 / 4, lc, 0.9 * lvl * flood)
        cv2.circle(img, (int(lx[j]), int(ly[j])), int(r_l), (lc * 1.3 * lvl * flood).tolist(),
                   -1, cv2.LINE_AA)
    # beams: 3 per tower from the head toward the floor near the fighters
    sw = float(T["sweep"][i])
    qh, qw = soft.shape[:2]
    beams = np.zeros((qh, qw), np.float32)
    ecx = float(T["cx"][i])
    for s_ in (-1.0, 1.0):
        sel = side == s_
        hx, hy = float(lx[sel].mean()), float(ly[sel].mean())
        for b in range(3):
            ax = ecx + 0.55 * math.sin(2 * math.pi * (sw * 0.5 + (b + (s_ > 0) * 3) * 0.27 * 1.33)) \
                + (b - 1) * 0.4
            ay = 0.6 + 0.4 * math.sin(2 * math.pi * (sw * 0.31 + b * 0.4))
            ex, ey = _w2s(cam, W, ax, 0.0, ay)
            dx, dy = ex - hx, ey - hy
            Lb = math.hypot(dx, dy) + 1e-6
            nx, ny = -dy / Lb, dx / Lb
            w0, w1 = 0.012 * W, 0.06 * W
            pts = np.array([[hx + nx * w0, hy + ny * w0], [hx - nx * w0, hy - ny * w0],
                            [ex - nx * w1, ey - ny * w1], [ex + nx * w1, ey + ny * w1]],
                           np.float32) / 4.0
            lay = np.zeros((qh, qw), np.float32)
            cv2.fillPoly(lay, [np.round(pts).astype(np.int32)], 1.0, cv2.LINE_AA)
            # brighter near the source
            yy = np.linspace(0, 1, qh, dtype=np.float32)[:, None]
            beams += lay * (0.55 + 0.45 * (1 - np.clip((yy * H - hy) / (ey - hy + 1e-6), 0, 1)))
    beams = cv2.GaussianBlur(beams, (0, 0), max(1.0, W / 4 / 220))
    amt = (0.08 + 0.05 * hotf + 0.07 * kenv * hotf) * flood
    if bool(T["big"][i]):
        # BIG DROP: the rig goes to a show — the beams take the fighters'
        # core colours, alternating every beat, and burn brighter
        cA = np.asarray(P["A"]["core"], np.float32)
        cB = np.asarray(P["B"]["core"], np.float32)
        fc = 0.35 * fc + 0.65 * (cA if bi % 2 == 0 else cB)
        amt *= 1.35
    soft += beams[..., None] * fc * amt


def _composite_mechs(R, img, soft, T, G, P, rig, cam, mech, i, power, kenv, senv,
                     hotf, spark, pre, black):
    W, H = R.W, R.H
    x0, y0, x1, y1 = mech["roi"]
    buf, em = mech["buf"], mech["em"]
    al = buf[..., 3:4]
    key = 0.08 + 0.92 * power
    key *= 1.0 + 0.12 * kenv * hotf
    reg = img[y0:y1, x0:x1]
    reg *= (1 - al)
    reg += buf[..., :3] * key
    # screen-space RIM light: floodlights from above-behind, the opponent's
    # core glow from the centre (tinted per side)
    a2 = al[..., 0]
    d = max(1, int(0.004 * H))
    top = np.clip(a2 - np.roll(a2, d, axis=0), 0, 1)
    fc = np.asarray(P["flood"], np.float32)
    reg += top[..., None] * fc * (0.35 * power + 0.04)
    xs = np.arange(x1 - x0)
    cxs = _w2s(cam, W, float(T["cx"][i]), 0.6)[0] - x0
    for m, sgn in ((0, 1), (1, -1)):
        side = np.clip(a2 - np.roll(a2, -sgn * d, axis=1), 0, 1)
        mask = (xs < cxs) if m == 0 else (xs >= cxs)
        oc = np.asarray(P[MECH[1 - m]["key"]]["core"], np.float32)
        amt = 0.18 + 0.4 * pre + 0.5 * (kenv if m == 1 else senv) * hotf
        reg += (side * mask[None, :])[..., None] * oc * amt
    # glow strips (emissive): sub heat + hi-hat flicker; blackout keeps them
    fl = 1.0 - 0.5 * spark * float(np.sin(i * 12.9898) * 0.5 + 0.5)
    eg = (0.55 + 0.45 * float(T["heat"][i]) + 0.8 * pre) * fl
    if black:
        eg *= 1.3
    reg += em * eg
    q = cv2.resize(em, (max(1, (x1 - x0) // 4), max(1, (y1 - y0) // 4)),
                   interpolation=cv2.INTER_AREA)
    q = cv2.GaussianBlur(q, (0, 0), 2.0)
    qy0, qx0 = y0 // 4, x0 // 4
    hh = min(q.shape[0], soft.shape[0] - qy0)
    ww = min(q.shape[1], soft.shape[1] - qx0)
    soft[qy0:qy0 + hh, qx0:qx0 + ww] += q[:hh, :ww] * (0.6 * eg)


def _fx_cores(R, img, sharp, soft, T, P, rig, cam, i, pre, black, spark, heat, hotf, bw):
    """The reactor cores: glow with heat, swell in the build, crackle with
    energy arcs, flicker with hi-hat sparkle. Servo sparks at the joints."""
    W, H = R.W, R.H
    S = cam["S"]
    for m, rg in enumerate(rig):
        pal = P[MECH[m]["key"]]
        cc = np.asarray(pal["core"], np.float32)
        p = rg["core"]
        cx, cy = _w2s(cam, W, rg["st"] + rg["fac"] * p[0], p[1])
        fl = 1.0 - 0.55 * spark * float(np.sin((i + m * 7) * 78.233) * 0.5 + 0.5)
        kag = int(T["kage"][i]) if m == 0 else int(T["sage"][i])
        hitp = 0.72 ** kag if kag < 30 else 0.0
        e = (0.55 + 0.45 * heat + 1.0 * pre + 0.6 * bw + 0.35 * hitp * hotf) * fl
        if black:
            e = 1.6 + 0.4 * fl
        r = 0.032 * S * (1.0 + 0.6 * pre + 0.3 * bw)
        cv2.circle(sharp, (int(cx), int(cy)), max(1, int(r)),
                   np.clip(cc * 0.6 * e, 0, 255).tolist(), -1, cv2.LINE_AA)
        cv2.circle(sharp, (int(cx), int(cy)), max(1, int(r * 0.5)),
                   np.clip((cc * 0.3 + 180) * e, 0, 255).tolist(), -1, cv2.LINE_AA)
        _blob(soft, cx / 4, cy / 4, r * 1.3 / 4, cc, 1.2 * e)
        _blob(soft, cx / 4, cy / 4, r * 4.0 / 4, cc, 0.35 * e)
        # energy crackle: build + blackout (+ the beam)
        ce = max(pre, 1.0 if black else 0.0, bw * 0.6)
        if ce > 0.05:
            rng = np.random.default_rng(i * 7 + m * 3 + 11)
            narc = 2 + int(5 * ce)
            for _ in range(narc):
                ang = rng.uniform(0, 2 * math.pi)
                ln = S * rng.uniform(0.10, 0.30) * (0.5 + ce)
                pts = [(cx, cy)]
                segs = 6
                for s_ in range(1, segs + 1):
                    u = s_ / segs
                    jx = rng.normal(0, 0.03 * S)
                    jy = rng.normal(0, 0.03 * S)
                    pts.append((cx + math.cos(ang) * ln * u + jx * (1 - u * 0.3),
                                cy + math.sin(ang) * ln * u + jy))
                arr = np.round(np.asarray(pts)).astype(np.int32)
                c_ = np.clip(cc * 0.5 + 120, 0, 255) * min(1.0, ce * 1.2)
                cv2.polylines(sharp, [arr], False, c_.tolist(), 1, cv2.LINE_AA)
                for (px, py) in pts[::2]:
                    _blob(soft, px / 4, py / 4, 1.2, cc, 0.25 * ce)
        # servo sparks at the joints on hi-hat sparkle (last 5 frames)
        for back in range(5):
            f = i - back
            if f < 0:
                break
            sv = float(T["spark"][f])
            # rising edges only: a hi-hat ACCENT, not a constant spray
            if sv < 0.5 or f == 0 or float(T["spark"][f - 1]) >= 0.5:
                continue
            rng = np.random.default_rng(f * 13 + m * 5 + 3)
            arm = rg["arms"][int(rng.integers(0, 2))]
            jt = arm["el"] if rng.random() < 0.6 else arm["sh"]
            jx, jy = _w2s(cam, W, rg["st"] + rg["fac"] * jt[0], jt[1])
            for _ in range(5):
                vx = rng.normal(0, 0.6) * S
                vy = -rng.uniform(0.2, 1.2) * S
                tt = (back + 0.5) / R.fps
                px = jx + vx * tt
                py = jy + vy * tt + 2.5 * S * tt * tt
                qx = px - vx * 0.02
                qy = py - (vy + 5.0 * S * tt) * 0.02
                c_ = np.asarray(P["spark"][1], np.float32) * (1 - back / 5) * sv
                cv2.line(sharp, (int(qx), int(qy)), (int(px), int(py)),
                         c_.tolist(), 1, cv2.LINE_AA)


def _fx_hits(R, img, sharp, soft, T, P, cam, i):
    """Impacts: white-hot flash, ring shockwave, spark burst, armour chips.
    All a function of (strike index, frames since contact)."""
    W, H = R.W, R.H
    S = cam["S"]
    fr = R.fps / 30.0
    f0s = T["s_f0"]
    lo = np.searchsorted(f0s, i - int(26 * fr))
    hi = np.searchsorted(f0s, i, side="right")
    sp = [np.asarray(c, np.float32) for c in P["spark"]]
    for j in range(lo, hi):
        Sj = T["strikes"][j]
        age = (i - Sj["f0"]) / fr
        st = Sj["s"] * (0.4 if Sj["blocked"] else 1.0)
        wx, wy = T["s_pt"][j]
        cx, cy = _w2s(cam, W, float(wx), float(wy))
        clash = Sj["att"] == 2
        d = 1.0 if Sj["att"] == 0 else (-1.0 if Sj["att"] == 1 else 0.0)
        # flash
        if age < 6:
            # LOCAL white-hot flash: small hot core + a modest coloured halo.
            # (v1 used a 0.4-mech-height sigma at gain ~3 and whited out the
            # whole frame on the drop-entry clash — the dystopia lesson.)
            fa = (1.0, 0.55, 0.28, 0.14, 0.07, 0.03)[int(age)]
            r = S * (0.035 + 0.03 * min(st, 1.2)) * (1.5 if clash else 1.0)
            _blob(soft, cx / 4, cy / 4, r * 0.9 / 4, sp[0], 1.3 * fa)
            _blob(soft, cx / 4, cy / 4, r * 2.6 / 4, sp[2], 0.30 * fa * min(st, 1.2))
            cv2.circle(sharp, (int(cx), int(cy)), max(1, int(r * 0.55 * (1 - age / 6))),
                       (255, 255, 255), -1, cv2.LINE_AA)
        # ring shockwave (a camera-facing ellipse squashed along the punch)
        if age < 12 and st > 0.3:
            rr = S * (0.04 + age * (0.035 if not clash else 0.06) * (0.6 + 0.6 * st))
            fa = (1 - age / 12) ** 1.5
            ax_ = (int(rr * (0.45 if not clash else 0.5)), int(rr))
            c_ = np.clip(sp[1] * fa * (0.9 if not clash else 1.0), 0, 255)
            th = max(1, int(S * 0.008 * (1 - age / 12) * (2 if clash else 1)))
            cv2.ellipse(sharp, (int(cx), int(cy)), ax_, 0, 0, 360, c_.tolist(), th,
                        cv2.LINE_AA)
            if clash:
                cv2.ellipse(sharp, (int(cx), int(cy)), (int(rr * 1.6), int(rr * 0.35)),
                            0, 0, 360, (c_ * 0.7).tolist(), th, cv2.LINE_AA)
        # sparks: seeded per strike, analytic ballistic paths
        rng = np.random.default_rng(5000 + j)
        ns = int((16 + 30 * st) * (3.0 if clash else 1.0))
        ang = rng.normal(0, 1.1 if not clash else 2.2, ns)
        spd = rng.uniform(0.6, 2.4, ns) * (1.4 if clash else 1.0) * S
        life = rng.uniform(8, 22, ns)
        vx = np.cos(ang) * spd * (d if d != 0 else 1.0) * np.where(
            rng.random(ns) < (0.8 if d != 0 else 0.5), 1.0, -1.0)
        vy = -np.abs(np.sin(ang)) * spd * 0.8 - 0.2 * S
        tt = age / 30.0
        alive = age < life
        if alive.any():
            g = 3.2 * S
            px = cx + vx * tt
            py = cy + vy * tt + 0.5 * g * tt * tt
            qx = px - vx * 0.022
            qy = py - (vy + g * tt) * 0.022
            k_ = np.clip(1 - age / life, 0, 1)
            for n_ in np.flatnonzero(alive):
                ci = sp[0] if k_[n_] > 0.6 else (sp[1] if k_[n_] > 0.3 else sp[2])
                c_ = np.clip(ci * (0.4 + 0.8 * k_[n_]), 0, 255)
                cv2.line(sharp, (int(qx[n_]), int(qy[n_])), (int(px[n_]), int(py[n_])),
                         c_.tolist(), 1 + int(S > 500), cv2.LINE_AA)
        # armour chips (heavy hits): dark tumbling polygons with a lit edge
        if st > 0.8 and age < 20:
            nc = 3 + int(4 * st) + (6 if clash else 0)
            for c in range(nc):
                vx_ = rng.normal(0.4 * (d if d else rng.choice([-1, 1])), 0.6) * S
                vy_ = -rng.uniform(0.6, 1.6) * S
                px = cx + vx_ * tt
                py = cy + vy_ * tt + 0.5 * 3.5 * S * tt * tt
                sz = S * rng.uniform(0.010, 0.022)
                rot = rng.uniform(0, 6.28) + age * rng.normal(0, 0.5)
                pts = []
                for v in range(4):
                    a_ = rot + v * 1.5708 + rng.uniform(-0.4, 0.4)
                    rr_ = sz * rng.uniform(0.6, 1.2)
                    pts.append((px + rr_ * math.cos(a_), py + rr_ * math.sin(a_)))
                arr = np.round(np.asarray(pts)).astype(np.int32)
                cv2.fillPoly(img, [arr], (22.0, 22.0, 26.0))
                cv2.polylines(sharp, [arr[:2]], False,
                              np.clip(sp[2] * 0.7, 0, 255).tolist(), 1, cv2.LINE_AA)


def _fx_beam(R, sharp, soft, T, P, rig, cam, i, bw):
    """Big-drop beam clash: each core fires at a clash point the kicks and
    snares shove back and forth."""
    W, H = R.W, R.H
    S = cam["S"]
    mid = (rig[0]["st"] + rig[1]["st"]) / 2.0 + float(T["beam_x"][i])
    pts = []
    for m, rg in enumerate(rig):
        p = rg["core"]
        pts.append(_w2s(cam, W, rg["st"] + rg["fac"] * p[0], p[1]))
    cyw = (rig[0]["core"][1] + rig[1]["core"][1]) / 2.0
    cpx, cpy = _w2s(cam, W, mid, cyw)
    ha = int(T["beam_hit_age"][i])
    hit = 0.75 ** ha if ha < 20 else 0.0
    rng = np.random.default_rng(i * 3 + 1)
    for m, (sx, sy) in enumerate(pts):
        cc = np.asarray(P[MECH[m]["key"]]["core"], np.float32)
        wob = [(sx + (cpx - sx) * u, sy + (cpy - sy) * u + rng.normal(0, 0.006 * S) * math.sin(math.pi * u))
               for u in np.linspace(0, 1, 12)]
        arr = np.round(np.asarray(wob)).astype(np.int32)
        thick = max(2, int(S * (0.013 + 0.008 * float(T["heat"][i])) * bw))
        cv2.polylines(sharp, [arr], False, np.clip(cc * 0.75 * bw, 0, 255).tolist(),
                      thick, cv2.LINE_AA)
        cv2.polylines(sharp, [arr], False, (230 * bw, 230 * bw, 230 * bw),
                      max(1, thick // 3), cv2.LINE_AA)
        for (px, py) in wob[1::2]:
            _blob(soft, px / 4, py / 4, thick * 1.4 / 4 + 0.6, cc, 0.18 * bw)
    sp = np.asarray(P["spark"][0], np.float32)
    r = S * (0.030 + 0.025 * hit) * bw
    _blob(soft, cpx / 4, cpy / 4, r * 0.9 / 4, sp, 1.1 * bw)
    _blob(soft, cpx / 4, cpy / 4, r * 2.8 / 4, np.asarray(P["spark"][2], np.float32),
          0.25 * bw)
    cv2.circle(sharp, (int(cpx), int(cpy)), max(1, int(r * 0.6)), (255, 255, 255), -1,
               cv2.LINE_AA)
    for k in range(10):                      # clash sparks spray out
        a_ = rng.uniform(0, 2 * math.pi)
        ln = S * rng.uniform(0.05, 0.18) * (0.6 + hit)
        x2, y2 = cpx + math.cos(a_) * ln, cpy + math.sin(a_) * ln * 0.7
        cv2.line(sharp, (int(cpx), int(cpy)), (int(x2), int(y2)),
                 np.clip(np.asarray(P["spark"][1], np.float32) * bw, 0, 255).tolist(), 1,
                 cv2.LINE_AA)


def _fx_embers(R, soft, T, P, i, t, hotf, power):
    """Foreground embers drifting up past the camera (defocused)."""
    qh, qw = soft.shape[:2]
    rng = np.random.default_rng(909)
    n_ = 40
    x = rng.random(n_)
    sp = rng.uniform(0.05, 0.16, n_)
    ph = rng.random(n_)
    sz = rng.uniform(0.8, 2.6, n_)
    y = 1.05 - ((ph + t * sp * (1 + 1.5 * hotf)) % 1.2)
    xx = (x + 0.02 * np.sin(t * 0.7 + ph * 6)) * qw
    col = np.asarray(P["spark"][2], np.float32)
    for k in range(n_):
        _blob(soft, xx[k], y[k] * qh, sz[k] * qh / 270, col,
              0.20 * (0.4 + 0.6 * hotf) * (0.3 + 0.7 * power))


# ---------------------------------------------------------------------------
# 5. POST-FX: strobe (budgeted), shake, aberration, bump push, then the HUD
# ---------------------------------------------------------------------------
def fx(R, img, i, a, lv):
    P = lv.scene_palette(R, _MOD)
    T = _timeline(R, a, lv)
    G = _assets(R, P)
    W, H = R.W, R.H
    fr = R.fps / 30.0
    hotf = float(T["hot_f"][i])
    kage = int(T["kage"][i])
    kenv = 0.70 ** kage if kage < 40 else 0.0
    sa = int(T["strobe_age"][i])
    if R.style.strobe and sa < 4:
        # ONE bright frame, then a fast fall — never 2-3 near-white frames
        img += (38.0, 12.0, 3.0, 0.0)[sa]
    if hotf > 0.05:
        off = int(round((1.0 + 3.0 * kenv) * hotf * W / 1920))
        if off > 0:
            img[..., 0] = np.roll(img[..., 0], off, axis=1)
            img[..., 2] = np.roll(img[..., 2], -off, axis=1)
    z = 1.0
    dx = dy = rot = 0.0
    sh_a = int(T["sh_age"][i])
    if sh_a < int(14 * fr):
        s_ = float(T["shake"][int(T["sh_idx"][i])])
        amp = s_ * 0.018 * H * (0.72 ** (sh_a / fr))
        th = 6.2832 * float(lv._hash01(i, 17.0))
        dx, dy = amp * math.cos(th), amp * math.sin(th)
        rot = s_ * 0.9 * (0.75 ** (sh_a / fr)) * (1 if int(T["sh_idx"][i]) % 2 else -1)
        z += 0.025 * s_ * (0.7 ** (sh_a / fr))
    if abs(dx) + abs(dy) > 0.3 or z > 1.0005 or abs(rot) > 0.02:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, z)
        M[0, 2] += dx
        M[1, 2] += dy
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    _hud(R, img, T, G, P, i)
    return img


def _hud(R, img, T, G, P, i):
    """Fighting-game HUD: energy bars that read the music (BRAWLER = low end,
    STRIKER = mids), a hit chips them, they never empty. Logo badge + round."""
    W, H = R.W, R.H
    fr = R.fps / 30.0
    black = bool(T["black"][i])
    y0, y1 = int(H * 0.040), int(H * 0.068)
    xa0, xa1 = int(W * 0.035), int(W * 0.375)
    xb0, xb1 = int(W * 0.625), int(W * 0.965)
    nseg = 24
    vals = (0.30 + 0.70 * np.clip(float(T["low"][i]) * 1.25, 0, 1),
            0.30 + 0.70 * np.clip(float(T["mid"][i]) * 1.9, 0, 1))
    hit = [0.0, 0.0]
    f0s = T["s_f0"]
    lo = np.searchsorted(f0s, i - int(12 * fr))
    hi = np.searchsorted(f0s, i, side="right")
    for j in range(lo, hi):
        Sj = T["strikes"][j]
        if Sj["blocked"]:
            continue
        for m in ((0, 1) if Sj["att"] == 2 else (1 - Sj["att"],)):
            hit[m] = max(hit[m], Sj["s"] * math.exp(-(i - Sj["f0"]) / (5.0 * fr)))
    dim = 0.25 if black else 1.0
    for m, (x0, x1) in enumerate(((xa0, xa1), (xb0, xb1))):
        col = np.asarray(P["hud_A" if m == 0 else "hud_B"], np.float32)
        v = max(0.15, vals[m] - 0.18 * hit[m])
        # frame + dark backing
        img[y0 - 3:y1 + 3, x0 - 3:x1 + 3] *= 0.25
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3),
                      (np.asarray(P["hud"], np.float32) * 0.8 * dim).tolist(), 1, cv2.LINE_AA)
        sw = (x1 - x0) / nseg
        nfill = v * nseg
        for k in range(nseg):
            # bars drain toward the centre badge, like a fighting game
            kk = k if m == 1 else nseg - 1 - k
            sx0 = int(x0 + kk * sw + 1)
            sx1 = int(x0 + (kk + 1) * sw - 1)
            if k < nfill:
                fa = min(1.0, nfill - k)
                c_ = col * (0.75 + 0.25 * (k / nseg)) * fa
                if hit[m] > 0.05 and nfill - 4 < k < nfill:
                    c_ = c_ + 255.0 * min(1.0, hit[m])        # damage chip flash
                img[y0:y1, sx0:sx1] = c_ * dim
            else:
                img[y0:y1, sx0:sx1] = col * 0.10 * dim
        # name under the bar
        nm = G["hud_names"][m]
        nh, nw = nm.shape
        ny = y1 + int(H * 0.008)
        nx = x0 if m == 0 else x1 - nw
        if ny + nh < H and nx + nw < W:
            img[ny:ny + nh, nx:nx + nw] += nm[..., None] * col * 0.95 * dim
    # (no centre badge: the jumbotron hangs there and carries the logo)


import sys as _sys  # noqa: E402

_MOD = _sys.modules[__name__]
