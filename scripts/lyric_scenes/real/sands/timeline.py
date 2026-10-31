"""Sands — everything that reacts to the music, per frame, from the whole-song analysis (pure numpy, no bpy).

The cube is lyric_viz's `--look stage` LED cube (scripts/lyric_viz.py stage_timeline / _stage_face / _stage_cube) in
3-D, and STATELESS the same way: spin is a precomputed cumsum, kick/snare decays are "frames since the onset", section
colours come from the whole-song structure. Every value is baked as keys, so any frame / chunk renders on its own.

KICK and SNARE do DIFFERENT things (Zeke 10-09 liked that about the 2-D cube):
  KICK  → YAW SLAM: a quarter turn about the vertical axis, snapped (ease-out cubic, 0.28 s); the cube DIPS and
          PUNCHES in scale; the LED gain pumps; in the big drop the edge strips strobe on heavy kicks.
  SNARE → FLIP: a full 360° tumble about a horizontal axis (alternating the cube's X and Y), 0.42 s ease-in-out, so it
          always lands logo-upright; the LED face colours swap A↔B; a short lift.
  Calm parts: slow drift + hover bob only; kicks pulse the LEDs gently, nothing flips.
ENERGY (the crackling arcs): a per-section level (0 calm, a tease in builds, strong in drops, max in the finale) with
a spike at each drop entry and a kick boost; GROUND STRIKES fire on kicks in drops (round-robin over STRIKE_SLOTS).
The camera is NOT in here — it never reacts to the music.
"""
import math

import numpy as np

# lyric_viz STAGE_PALETTES["acid"] (0..255 sRGB) — the 2-D cube's colours
CALM = [((0, 70, 255), (0, 200, 255)), ((120, 30, 255), (0, 150, 255))]
HOT = [((120, 255, 0), (255, 0, 170)), ((0, 255, 200), (255, 30, 80)), ((255, 0, 210), (0, 190, 255)),
       ((255, 220, 0), (160, 0, 255))]
BIG = [(255, 0, 170), (120, 255, 0), (0, 200, 255), (255, 220, 0)]

STRIKE_SLOTS = 4
STRIKE_LIFE_S = 2.4          # grains settle within this; a slot is reused only after it (asserted)
KICK_SLAM_S, FLIP_S = 0.28, 0.42


def srgb_to_lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def age_since(flags):
    """frames since the last True (1e6 before the first) and the index of that True (-1)"""
    flags = np.asarray(flags, bool)
    n = len(flags)
    idx = np.where(flags, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    age = np.where(last >= 0, np.arange(n) - last, 10 ** 6)
    return age, last


def ease_out_cubic(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)


def event_curve(n, starts, length, ease, amount):
    """Σ over events of amount·ease((i − start)/length) — monotone, ends at amount per event (a cumulative angle)"""
    out = np.zeros(n)
    i = np.arange(n)
    for s in starts:
        out += amount * ease((i - s) / float(length))
    return out


def sections_hot(S, ear):
    """(drops [(t0, t1, level)], builds [(t0, t1)], calm sections) from ear.json's 'scene' block (my reading; Zeke's
    numbers replace it) — falls back to the analyser's drops."""
    sc = (ear or {}).get("scene") or {}
    drops = [tuple(d) for d in sc.get("drops", [])] or [(d["time"], d["end"], 0.7) for d in S.map.get("drops", [])]
    builds = [tuple(b) for b in sc.get("builds", [])]
    return drops, builds


def build(S, ear=None):
    n, fps = S.n, S.fps
    t = S.t
    drops, builds = sections_hot(S, ear or S.ear)
    big_t0 = max(drops, key=lambda d: d[2])[0]
    hot = np.zeros(n, bool)
    level = np.zeros(n)
    big = np.zeros(n, bool)
    for t0, t1, lv in drops:
        m = (t >= t0) & (t < t1)
        hot |= m
        level[m] = lv
        if t0 == big_t0:
            big |= m
    build_m = np.zeros(n, bool)
    build_ramp = np.zeros(n)
    for t0, t1 in builds:
        m = (t >= t0) & (t < t1)
        build_m |= m
        build_ramp[m] = (t[m] - t0) / max(t1 - t0, 1e-6)

    # intensity (the 2-D I): smoothed full mix + low end, normalised
    from lib.song import gsmooth, pnorm
    I = np.clip(0.6 * gsmooth(pnorm(S.curve("full")), 0.5 * fps) + 0.4 * gsmooth(pnorm(S.curve("low")), 0.5 * fps), 0, 1)
    I = np.where(hot, np.maximum(I, 0.6), I)

    # grid
    period_f = S.period * fps
    bpos = (t - S.beat0) / S.period
    bi = np.floor(bpos).astype(int)
    ph = bpos - bi
    bar_beat = (bi - S.downbeat_mod) % S.bars_per            # 0..3, 0 = downbeat
    bar = (bi - S.downbeat_mod) // S.bars_per

    # onsets → frames (analysed hits), kept only where they make sense
    kick_f = S.hits("kick")
    snare_f = S.hits("snare")
    kick = np.zeros(n, bool)
    kick[kick_f] = True
    snare = np.zeros(n, bool)
    snare[snare_f] = True
    kage, _ = age_since(kick)
    sage, _ = age_since(snare)
    kenv = np.where(kage < 60, 0.72 ** np.minimum(kage, 60), 0.0)
    senv = np.where(sage < 60, 0.70 ** np.minimum(sage, 60), 0.0)

    # KICK SLAMS: kicks in drops, ≥ 1 beat apart
    slam = []
    for f in kick_f:
        if hot[f] and (not slam or f - slam[-1] >= 0.95 * period_f):
            slam.append(int(f))
    # SNARE FLIPS: in drops, one per bar max: the first snare within ±90 ms of beat 3 (the riddim backbeat)
    flips = []
    seen_bars = set()
    for f in snare_f:
        if not hot[f]:
            continue
        b = int(bar[f])
        near3 = abs(((bpos[f] - S.downbeat_mod) % S.bars_per) - 2.0) * S.period < 0.09
        if near3 and b not in seen_bars and (not flips or f - flips[-1] >= FLIP_S * fps + 2):
            flips.append(int(f))
            seen_bars.add(b)
    flips_x, flips_y = flips[0::2], flips[1::2]

    drift = np.cumsum((0.10 + 0.22 * I) / fps)
    yaw = 0.42 + drift + event_curve(n, slam, KICK_SLAM_S * fps, ease_out_cubic, math.pi / 2)
    flip_x = event_curve(n, flips_x, FLIP_S * fps, ease_in_out, 2 * math.pi)
    flip_y = event_curve(n, flips_y, FLIP_S * fps, ease_in_out, 2 * math.pi)
    hov = 0.30 * np.sin(2 * math.pi * t / 5.3) + 0.12 * np.sin(2 * math.pi * t / 2.3 + 1.0)
    fage, _ = age_since(np.isin(np.arange(n), flips))
    lift = np.where(fage < 30, np.sin(np.pi * np.clip(fage / (FLIP_S * fps * 1.3), 0, 1)) * 0.45, 0.0)
    dip = -0.35 * kenv * hot - 0.08 * kenv * ~hot
    z_off = hov + lift + dip
    scale = 1.0 + 0.055 * kenv * hot + 0.02 * kenv * ~hot

    # section colours (linear), the 2-D logic: calm cycles CALM, drops cycle HOT, the big drop rotates BIG per bar
    A = np.zeros((n, 3))
    B = np.zeros((n, 3))
    edges = np.flatnonzero(np.diff(np.r_[0, hot.astype(int), 0]))       # starts/ends of hot runs
    segs, cur = [], 0
    for e in edges:
        if e > cur:
            segs.append((cur, e))
        cur = e
    if cur < n:
        segs.append((cur, n))
    nh = nc = 0
    for s0, s1 in segs:
        if hot[s0] and big[s0]:
            j = (bar[s0:s1] - bar[s0]) % len(BIG)
            A[s0:s1] = srgb_to_lin(np.asarray(BIG)[j])
            B[s0:s1] = srgb_to_lin(np.asarray(BIG)[(j + 1) % len(BIG)])
        elif hot[s0]:
            a, b = HOT[nh % len(HOT)]
            nh += 1
            A[s0:s1], B[s0:s1] = srgb_to_lin(a), srgb_to_lin(b)
        else:
            a, b = CALM[nc % len(CALM)]
            nc += 1
            A[s0:s1], B[s0:s1] = srgb_to_lin(a), srgb_to_lin(b)
    # anticipation: 8 beats before each drop the colours slide to the drop's; half-beat blackout right before it
    pre = np.zeros(n)
    black = np.zeros(n, bool)
    entries = [s0 for s0, s1 in segs if hot[s0] and s0 > 0]
    for s0 in entries:
        p0 = max(0, s0 - int(round(8 * period_f)))
        w = np.linspace(0, 1, s0 - p0, endpoint=False) ** 1.5
        w = np.where(hot[p0:s0], 0, w)
        A[p0:s0] = A[p0:s0] * (1 - w[:, None]) + A[s0] * w[:, None]
        B[p0:s0] = B[p0:s0] * (1 - w[:, None]) + B[s0] * w[:, None]
        pre[p0:s0] = np.maximum(pre[p0:s0], np.where(hot[p0:s0], 0, w))
        b0 = max(0, s0 - max(1, int(round(0.5 * period_f))))
        black[b0:s0] = ~hot[b0:s0]
    # colour swap on every flip (parity of flips so far)
    nflip = np.cumsum(np.isin(np.arange(n), flips))
    swap = (nflip % 2).astype(float)

    # strobes: every drop entry + ≤1 per bar on a heavy kick in the big drop
    strobe = np.zeros(n, bool)
    strobe[entries] = True
    last = -10 ** 9
    for f in slam:
        if big[f] and f - last >= 0.9 * 4 * period_f:
            strobe[f] = True
            last = f
    stage, _ = age_since(strobe)
    strobe_env = np.where(stage < 20, 0.55 ** np.minimum(stage, 20), 0.0)

    gain = (0.55 + 0.45 * I + 0.25 * kenv) * np.where(black, 0.12, 1.0)

    # ENERGY
    eage, _ = age_since(np.isin(np.arange(n), entries))
    spike = np.where(eage < 3 * fps, np.exp(-eage / (0.8 * fps)), 0.0)
    energy = np.clip(level + 0.30 * kenv * hot + spike * hot + 0.22 * build_ramp ** 2 * build_m, 0, 1)
    energy = np.where(black, 0.0, energy)
    # GLITCH (v2 cyberpunk LED faces): row-shift bursts on drop snares, strobes and the pre-drop blackout; a tease on
    # build snares as the build climbs; nothing in calm parts / the outro
    glitch = np.clip(1.0 * senv * hot + 0.8 * strobe_env + 0.6 * black + 0.6 * senv * build_m * build_ramp ** 2, 0, 1)
    glitch = np.where(glitch < 0.04, 0.0, glitch)

    # GROUND STRIKES on kicks in drops (every slam), round-robin slots; a slot is reused only after STRIKE_LIFE_S
    life = int(round(STRIKE_LIFE_S * fps))
    rng = np.random.default_rng(7)
    strikes = []               # (frame index, slot, angle, radius)
    slot_free = [-10 ** 9] * STRIKE_SLOTS
    cand = sorted(set(slam) | set(entries))
    for f in cand:
        if level[f] < 0.5:
            continue
        slot = int(np.argmin(slot_free))
        if f - slot_free[slot] < life:
            continue           # all slots busy — skip this strike
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(9.0, 26.0)
        strikes.append((int(f), slot, float(a), float(r)))
        slot_free[slot] = f
    return dict(n=n, fps=fps, I=I, hot=hot, big=big, level=level, kenv=kenv, senv=senv, yaw=yaw, flip_x=flip_x,
                flip_y=flip_y, z_off=z_off, scale=scale, A=A, B=B, swap=swap, black=black, pre=pre, ph=ph,
                bi=bi.astype(float), bar_beat=bar_beat, strobe=strobe_env, gain=gain, energy=energy, glitch=glitch,
                slams=slam, flips=flips, entries=entries, strikes=strikes, build=build_m)


def strike_tracks(T, ground_fn, centre=(0.0, 0.0)):
    """per-slot per-frame arrays: end (n,3), age_s (n,) (−1 = never yet), seed (n,), flash (n,) — CONSTANT-stepped
    end/seed, linear age. A slot jumps to its next strike only after STRIKE_LIFE_S (asserted)."""
    n, fps = T["n"], T["fps"]
    out = []
    for k in range(STRIKE_SLOTS):
        ev = [s for s in T["strikes"] if s[1] == k]
        end = np.zeros((n, 3))
        end[:, 2] = -50.0
        age = np.full(n, -1.0)
        seed = np.zeros(n)
        for j, (f, _slot, a, r) in enumerate(ev):
            f1 = ev[j + 1][0] if j + 1 < len(ev) else n
            assert f1 - f >= STRIKE_LIFE_S * fps - 1, (k, f, f1)
            x, y = centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)
            end[f:f1] = (x, y, float(ground_fn(x, y)))
            age[f:f1] = (np.arange(f1 - f)) / fps
            seed[f:f1] = f
        flash = np.where((age >= 0) & (age < 0.25), np.exp(-np.maximum(age, 0) / 0.06), 0.0)
        out.append(dict(end=end, age=age, seed=seed, flash=flash, events=ev))
    return out


def aim_yaw(T, cam_loc, cube_c, targets, facing_fn, ramp_max_s=10.0, offset=0.15, slack=0.10):
    """Zeke v2: every face must actually be SEEN. targets = [(f0, f1, face, name)] windows (0-based frames) in which
    `face` should face the camera. For each target in time order pick the yaw shift Δ (72 candidates) maximising the
    face's mean facing over its window (smallest |Δ| on ties), then ease it in over the calm-ish stretch BEFORE the
    window (≤ ramp_max_s, after the previous window) — so the cube turns slowly between shots, never on the music and
    never during a window. offset = a small 3/4 bias (rad) added after the head-on search so the next face peeks in. Returns (yaw, log)."""
    n, fps = T["n"], T["fps"]
    yaw = T["yaw"].copy()
    prev_end = 0
    log = []
    cands = np.radians(np.arange(-180, 180, 5.0))
    for f0, f1, face, name in sorted(targets):
        w = np.arange(f0, f1)
        scores = []
        for d in cands:
            fc = facing_fn(yaw[w] + d, T["flip_x"][w], T["flip_y"][w], cube_c[w], cam_loc[w])[:, face]
            scores.append(float(np.mean(np.clip(fc, 0, 1))))
        scores = np.array(scores)
        ok = np.flatnonzero(scores >= scores.max() - slack)           # near-best → the SMALLEST turn (calm cube)
        j = ok[np.argmin(np.abs(cands[ok]))]
        sc, d = float(scores[j]), float(cands[j])
        a = max(prev_end, f0 - int(ramp_max_s * fps))
        L = max(f0 - a, 1)
        u = np.clip((np.arange(n) - a) / L, 0, 1)
        yaw += (d + offset) * u * u * (3 - 2 * u)
        log.append(dict(shot=name, face=face, frames=[int(f0) + 1, int(f1)], shift_deg=round(float(np.degrees(d + offset)), 1),
                        ramp_s=round(L / fps, 2), rate_deg_s=round(abs(float(np.degrees(d + offset))) / (L / fps) * 1.5, 1),
                        mean_facing=round(sc, 2)))
        prev_end = f1
    return yaw, log
