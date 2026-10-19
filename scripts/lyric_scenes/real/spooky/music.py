"""Spooky v2 "live picture": the song → every reactive channel, per frame. Pure numpy (no bpy) so it is tested and
plotted outside Blender, and the build bakes exactly what was checked.

    import music as MU
    S = MU.song()                       # lib/song.Song on the 150-BPM grid (ear.json next to the map outranks all of it)
    M = MU.channels(S)                  # {name: float[n]}  (index i ↔ Blender frame i+1)
    SH = MU.shake(S, M)                 # {"loc": (n,3) m, "rot": (n,3) rad (pitch, yaw, roll)}
    $PY -I music.py [--plot OUT.png]    # report (+ a channel plot, safe to Read)

WHERE THE ENERGY SHAPE COMES FROM. The analyser's own sections are off here (it locked the grid at 75 BPM and called
0–118 s "intro"). The macro shape is SECTION_READ below (Iris, by curve, 10-18 — Zeke's drop times not in yet); an
analysis/ear.json with "sections" REPLACES it (labels in LEVEL pick the level; unknown labels → 0.6). The fine detail
always comes from the stems/bands in curves.json, so a section boundary a bar off only moves the macro envelope.

ENVELOPES: asymmetric followers — fast attack (1–3 frames), slower release (8–40 frames) — so nothing pops and nothing
flickers. Integrated CLOCKS (∫ rate dt) carry every "moves faster in the drops" effect: positions stay a pure function
of the frame (GN reads the baked clock instead of Scene Time) and never jump when the rate changes.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(REPO, "scripts/lyric_scenes"))

from lib.song import Song, gsmooth  # noqa: E402

SONG_MAP = "/data/iris_assets/spooky_futurebass/analysis/song_map.json"
FPS = 30
BPM_RANGE = (140.0, 160.0)          # the analyser's 75 is the half-time octave; kicks fit 149.98 (comb 0.74) in this range

# (start s, end s, label, level at start, level at end) — Iris's section read by curve, 10-18
SECTION_READ = [
    (0.0, 32.0, "intro", 0.10, 0.14),
    (32.0, 64.0, "build", 0.28, 0.62),
    (64.0, 116.0, "full", 0.80, 0.80),
    (116.0, 128.0, "dip", 0.36, 0.40),
    (128.0, 152.0, "full", 0.80, 0.78),
    (152.0, 163.2, "breakdown", 0.20, 0.30),
    (163.2, 192.0, "biggest", 0.92, 0.98),
    (192.0, 205.0, "peak", 1.00, 0.95),
    (205.0, 220.0, "bass_out", 0.40, 0.45),
    (220.0, 224.0, "rebuild", 0.50, 0.70),
    (224.0, 240.0, "last_hit", 0.92, 0.62),
    (240.0, 254.4, "outro", 0.30, 0.04),
]
LEVEL = {"intro": 0.1, "build": 0.45, "full": 0.8, "drop": 0.9, "dip": 0.35, "breakdown": 0.22, "break": 0.3,
         "bridge": 0.5, "biggest": 0.95, "peak": 1.0, "bass_out": 0.42, "rebuild": 0.6, "last_hit": 0.85, "outro": 0.15}
BIG_HITS = [32.0, 64.0, 128.0, 163.2, 192.0, 224.0, 236.8]     # shooting stars + bursts (snapped to the grid below)

# shake limits (asserted in the build): a vibration, not an earthquake
SHAKE_ROT_DEG = (0.42, 0.30, 0.22)   # noise peak pitch / yaw / roll at drive 1 (rms ≈ 0.4 × peak)
SHAKE_LOC_M = (0.018, 0.010, 0.018)  # noise peak x / y / z at drive 1
SHAKE_MAX_ROT_DEG = 0.60             # hard bound on |rot| per axis (24 mm, 1080p: ≈ 26 px/deg → ≤ 16 px)
SHAKE_MAX_LOC_M = 0.05               # hard bound on |loc| per axis
SHAKE_MAX_STEP_DEG = 0.30            # max per-frame rotation change: 26 px/deg at 1080p, 0.5 shutter → ≤ 3.9 px blur


def song(path=SONG_MAP):
    p = os.path.join(os.path.dirname(path), "ear.json")
    if os.path.isfile(p):
        return Song.load(path, fps=FPS)                               # Zeke's ear outranks everything here
    return Song.load(path, fps=FPS, ear={"bpm_range": list(BPM_RANGE)})


# ------------------------------------------------------------------------------------------------ helpers

def follower(x, attack, release):
    """asymmetric envelope follower (frames): rises with time-constant `attack`, falls with `release`"""
    x = np.asarray(x, np.float64)
    a = 1.0 - math.exp(-1.0 / max(attack, 1e-6))
    r = 1.0 - math.exp(-1.0 / max(release, 1e-6))
    y = np.empty_like(x)
    v = float(x[0])
    for i, xi in enumerate(x):
        v += (xi - v) * (a if xi > v else r)
        y[i] = v
    return y


def spring(impulses, freq_hz, zeta, fps=FPS):
    """second-order damped spring driven by an impulse train (semi-implicit Euler, 8 substeps) → displacement,
    normalised so a unit impulse peaks at 1. Gives the squash → overshoot → settle of a jelly ghost."""
    w = 2 * math.pi * freq_hz
    sub = 8
    dt = 1.0 / (fps * sub)
    x = v = 0.0
    out = np.empty(len(impulses))
    for i, imp in enumerate(impulses):
        v += imp * w                                   # an impulse sets velocity (peak displacement ≈ imp)
        for _ in range(sub):
            acc = -w * w * x - 2 * zeta * w * v
            v += acc * dt
            x += v * dt
        out[i] = x
    unit = np.zeros(int(fps * 2))
    unit[0] = 1.0
    if not hasattr(spring, "_peak"):
        spring._peak = {}
    key = (freq_hz, zeta, fps)
    if key not in spring._peak:
        xx = vv = 0.0
        pk = 0.0
        for k in range(len(unit)):
            vv += unit[k] * w
            for _ in range(sub):
                vv += (-w * w * xx - 2 * zeta * w * vv) * dt
                xx += vv * dt
            pk = max(pk, abs(xx))
        spring._peak[key] = pk
    return out / spring._peak[key]


def band_noise(n, freqs, seed, fps=FPS):
    """deterministic band-limited noise in [-1, 1]: a sum of sines at `freqs` (Hz) with seeded phases + slow
    amplitude wobble, normalised by its own peak"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    y = np.zeros(n)
    for f in freqs:
        ph, wph = rng.uniform(0, 2 * math.pi, 2)
        f2 = f * rng.uniform(0.97, 1.03)
        y += np.sin(2 * math.pi * f2 * t + ph) * (0.75 + 0.25 * np.sin(2 * math.pi * 0.37 * t + wph))
    return y / max(np.abs(y).max(), 1e-9)


def clock(rate, fps=FPS):
    """∫ rate dt (seconds-equivalent), starting at 0 on frame 1"""
    r = np.asarray(rate, np.float64)
    return np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]))]) / fps


def section_rows(S):
    """[(start, end, label, level0, level1)] — ear.json sections (constant level per label) or SECTION_READ"""
    if S.sections_from == "ear":
        return [(s["start"], s["end"], s["label"], LEVEL.get(s["label"], 0.6), LEVEL.get(s["label"], 0.6))
                for s in S.sections]
    return SECTION_READ


def energy(S, smooth_s=0.6):
    """macro section energy E(t) 0..1: boundaries snapped to downbeats, linear inside a section, gaussian-smoothed"""
    E = np.zeros(S.n)
    for (a, b, _lab, l0, l1) in section_rows(S):
        a2 = 0.0 if a <= 0 else S.snap(a, "beat")
        b2 = S.duration if b >= S.duration - 0.5 else S.snap(b, "beat")
        m = (S.t >= a2) & (S.t < b2)
        if m.any():
            u = (S.t[m] - a2) / max(b2 - a2, 1e-6)
            E[m] = l0 + (l1 - l0) * u
    return np.clip(gsmooth(E, smooth_s * FPS), 0, 1)


def kick_events(S, min_gap=2):
    """kick frames: analyser kick hits ∪ low-band transients, merged within min_gap frames. → (idx int[k], strength[k])"""
    low = np.clip(S.curve("low"), 0, 1.3)
    d = np.maximum(np.diff(low, prepend=low[0]), 0)
    tr = [i for i in range(1, S.n - 1) if d[i] > 0.15 and d[i] >= d[i - 1] and d[i] > d[i + 1]]
    idx = np.unique(np.concatenate([S.hits("kick"), np.asarray(tr, int)]))
    keep = []
    for i in idx:
        if not keep or i - keep[-1] > min_gap:
            keep.append(int(i))
    keep = np.asarray(keep, int)
    win = np.clip(keep[:, None] + np.arange(0, 4)[None], 0, S.n - 1)
    strength = np.clip(low[win].max(1), 0.2, 1.2)
    return keep, strength


def refractory(idx, strength, gap):
    """drop events closer than `gap` frames to the previous KEPT one (ghost squash: rolls must not jitter it)"""
    ki, ks = [], []
    for i, s in zip(idx, strength):
        if not ki or i - ki[-1] >= gap:
            ki.append(i)
            ks.append(s)
    return np.asarray(ki, int), np.asarray(ks)


def impulses(n, idx, strength):
    x = np.zeros(n)
    x[np.clip(idx, 0, n - 1)] = strength
    return x


def onset_env(S, thr=0.45, release=10):
    """peaks of the onset curve above thr → fast-attack / slow-release envelope"""
    o = np.clip(S.curve("onset"), 0, 1.5)
    pk = np.zeros(S.n)
    for i in range(1, S.n - 1):
        if o[i] > thr and o[i] >= o[i - 1] and o[i] > o[i + 1]:
            pk[i] = min(1.0, (o[i] - thr) / 0.6 + 0.3)
    return np.clip(follower(pk, 1.0, release), 0, 1)


def hit_times(S):
    """BIG_HITS snapped to the beat grid, inside the song"""
    return [S.snap(t, "beat") for t in BIG_HITS if 0 < t < S.duration]


def drop_burst(S, tau_s=1.6):
    """1 at each big hit, exp decay (tau seconds) — petals / fireflies burst"""
    x = np.zeros(S.n)
    for t in hit_times(S):
        i = S.idx(t)
        age = np.arange(S.n - i) / FPS
        x[i:] = np.maximum(x[i:], np.exp(-age / tau_s))
    return x


# ------------------------------------------------------------------------------------------------ channels

def channels(S):
    n = S.n
    E = energy(S)
    Es = follower(E, 3, 30)                                       # ease into / out of a section
    low = np.clip(S.curve("low"), 0, 1.2)
    bass = np.clip(S.curve("bass"), 0, 1.0)
    high = np.clip(S.curve("high"), 0, 1.3)
    mid = np.clip(S.curve("mid"), 0, 1.3)
    vox = np.clip(S.curve("vocals"), 0, 1.0)

    bass_env = follower(np.maximum(bass - 0.12, 0) / 0.7, 4, 22)
    high_env = follower(np.clip((high - 0.08) / 0.7, 0, 1.2), 1.5, 10)
    vox_env = follower(np.clip(0.65 * np.maximum(vox - 0.15, 0) / 0.55 + 0.35 * mid, 0, 1.2), 2, 9)
    ons = onset_env(S)
    burst = drop_burst(S)

    kidx, kstr = kick_events(S)
    kick_cam = follower(impulses(n, kidx, kstr), 1.0, 5)          # camera: every kick, rolls → a buzz
    gidx, gstr = refractory(kidx, kstr, gap=9)
    g_amt = (0.35 + 0.65 * Es)                                    # intro kicks: small bounce; drops: full
    g_kick = spring(impulses(n, gidx, gstr), 2.6, 0.32) * g_amt
    g_kick = np.clip(g_kick, -1.2, 1.4)
    g_kick_late = np.r_[np.zeros(3), g_kick[:-3]]                 # the hem follows 3 frames later (follow-through)

    M = dict(
        E=E, Es=Es, low=low, bass_env=bass_env, high_env=high_env, vox_env=vox_env, onset=ons, burst=burst,
        kick_cam=kick_cam,
        # ---- shader CTRL (common.CTRL_KEYS)
        glow=0.85 + 0.30 * Es,
        moon_glow=0.86 + 0.26 * bass_env * (0.4 + 0.6 * Es),
        ghost_glow=0.72 + 0.55 * vox_env + 0.25 * Es + 0.25 * burst,
        petal_density=np.clip(follower(0.06 + 0.42 * Es + 0.30 * burst, 6, 45), 0, 0.85),  # smooth: a gate, never jitter
        petal_glow=np.clip(0.8 + 1.6 * ons + 1.2 * burst, 0, 3.5),
        water_shimmer=0.45 + 1.05 * high_env,
        sky_glow=0.88 + 0.18 * Es,
        stars=0.55 + 0.75 * high_env,
        firefly=np.clip(0.55 + 0.55 * Es + 0.7 * ons + 0.6 * burst, 0, 2.4),
        ff_density=np.clip(follower(0.12 + 0.40 * Es + 0.25 * burst, 6, 45), 0, 0.8),
        bloom=0.40 + 0.28 * bass_env * (0.3 + 0.7 * Es) + 0.10 * burst,
        # ---- GN CTRL_GN (common.GN_KEYS)
        g_clock=clock(1.0 + 0.8 * Es),
        g_sway=0.55 + 1.15 * Es,
        g_kick=g_kick,
        g_kick_late=g_kick_late,
        drift_t=clock(0.6 + 1.9 * Es + 0.8 * burst),
        wind_t=clock(0.45 + 1.5 * Es + 1.4 * burst + 0.4 * ons),
        ff_t=clock(0.8 + 0.9 * Es),
    )
    for k, v in M.items():
        if not np.isfinite(v).all():
            raise ValueError(f"channel {k} not finite")
        if len(v) != n:
            raise ValueError(f"channel {k} has {len(v)} frames, want {n}")
    return M


def shake(S, M):
    """camera vibration: (drive × band noise) + a damped thump per kick, smoothed. Deterministic (fixed seeds).
    drive = section gain (≈0 in intro/breakdown/outro) × (kick envelope + sustained sub)."""
    n = S.n
    Es = M["Es"]
    gain = np.clip((Es - 0.32) / 0.62, 0, 1) ** 1.3                      # 0 below E≈0.32 → calm sections stay still
    sub = follower(np.clip((M["low"] - 0.12) / 0.55, 0, 1), 2, 12)     # the pulsing sub: a continuous hum
    drive = gain * np.clip(0.45 * np.clip(M["kick_cam"] / 0.7, 0, 1.3) + 0.65 * sub, 0, 1.15) ** 0.7
    drive = gsmooth(drive, 1.0)
    kidx, kstr = kick_events(S)
    thump = spring(impulses(n, kidx, kstr), 3.4, 0.5) * gain            # the kick pushes the frame down, it rings back
    rot = np.zeros((n, 3))
    loc = np.zeros((n, 3))
    for a in range(3):
        nz = band_noise(n, (2.9, 4.1, 5.3, 6.4), seed=100 + a)
        rot[:, a] = math.radians(SHAKE_ROT_DEG[a]) * drive * nz
        nz2 = band_noise(n, (2.6, 3.8, 4.9, 6.1), seed=200 + a)
        loc[:, a] = SHAKE_LOC_M[a] * drive * nz2
    rot[:, 0] += math.radians(0.14) * thump                              # pitch nod on the kick
    loc[:, 2] += -0.012 * thump                                          # and a small drop
    for a in range(3):
        rot[:, a] = gsmooth(rot[:, a], 0.6)
        loc[:, a] = gsmooth(loc[:, a], 0.6)
    return dict(loc=loc, rot=rot, drive=drive, gain=gain)


def shake_report(S, SH):
    rot_d = np.degrees(SH["rot"])
    step = np.degrees(np.abs(np.diff(SH["rot"], axis=0))).max(1)
    out = dict(max_rot_deg=[round(float(x), 4) for x in np.abs(rot_d).max(0)],
               max_loc_m=[round(float(x), 4) for x in np.abs(SH["loc"]).max(0)],
               max_step_deg=round(float(step.max()), 4))
    for nm, (a, b) in (("intro", (0, 30)), ("build", (34, 62)), ("full", (66, 114)), ("breakdown", (153, 162)),
                       ("biggest", (165, 204)), ("bass_out", (208, 219)), ("outro", (242, 254))):
        m = (S.t >= a) & (S.t < b)
        out[f"rms_rot_deg_{nm}"] = round(float(np.sqrt((rot_d[m] ** 2).mean())), 4)
    return out


def check_shake(S, SH):
    """→ list of failure strings (empty = pass)"""
    r = shake_report(S, SH)
    bad = []
    if max(r["max_rot_deg"]) > SHAKE_MAX_ROT_DEG:
        bad.append(f"rotation {r['max_rot_deg']} deg > {SHAKE_MAX_ROT_DEG}")
    if max(r["max_loc_m"]) > SHAKE_MAX_LOC_M:
        bad.append(f"translation {r['max_loc_m']} m > {SHAKE_MAX_LOC_M}")
    if r["max_step_deg"] > SHAKE_MAX_STEP_DEG:
        bad.append(f"per-frame step {r['max_step_deg']} deg > {SHAKE_MAX_STEP_DEG}")
    for calm in ("intro", "breakdown", "outro"):
        if r[f"rms_rot_deg_{calm}"] > 0.01:
            bad.append(f"{calm} shakes (rms {r[f'rms_rot_deg_{calm}']} deg > 0.01)")
    if r["rms_rot_deg_biggest"] < 3 * max(r["rms_rot_deg_intro"], 1e-4) or r["rms_rot_deg_biggest"] < 0.025:
        bad.append(f"drops barely shake (rms {r['rms_rot_deg_biggest']} deg)")
    for k in ("loc", "rot"):
        if not np.isfinite(SH[k]).all():
            bad.append(f"shake {k} not finite")
    return bad


# ------------------------------------------------------------------------------------------------ CLI

def _plot(S, M, SH, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = S.t
    rows = [("E / Es", ["E", "Es"]), ("moon_glow / bloom", ["moon_glow", "bloom"]), ("ghost_glow", ["ghost_glow"]),
            ("g_kick", ["g_kick"]), ("petal_density / ff_density", ["petal_density", "ff_density"]),
            ("water_shimmer / stars", ["water_shimmer", "stars"]), ("clock rates", ["g_clock", "wind_t"])]
    fig, ax = plt.subplots(len(rows) + 1, 1, figsize=(12, 13), sharex=True)
    for a, (title, keys) in zip(ax, rows):
        for k in keys:
            y = M[k]
            if k in ("g_clock", "wind_t"):
                y = np.gradient(y) * FPS
            a.plot(t, y, lw=0.6, label=k)
        a.set_ylabel(title, fontsize=7)
        a.legend(fontsize=6, loc="upper left")
    ax[-1].plot(t, np.degrees(SH["rot"][:, 0]), lw=0.4, label="pitch deg")
    ax[-1].plot(t, SH["loc"][:, 2] * 10, lw=0.4, label="z ×10 (m)")
    ax[-1].legend(fontsize=6)
    for h in hit_times(S):
        for a in ax:
            a.axvline(h, color="r", lw=0.4, alpha=0.5)
    fig.tight_layout()
    tmp = out + ".tmp.png"
    fig.savefig(tmp, dpi=70)
    sys.path.insert(0, os.path.join(REPO, "scripts/lyric_scenes/lib/qa"))
    import sheet
    sheet.shrink(tmp, out)
    os.remove(tmp)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    S = song()
    M = channels(S)
    SH = shake(S, M)
    print(f"[music] grid {S.bpm:.3f} BPM beat0 {S.beat0:.4f} ({S.fit_from}); sections from "
          f"{'ear.json' if S.sections_from == 'ear' else 'SECTION_READ'}; hits {[round(h, 2) for h in hit_times(S)]}")
    kidx, _ = kick_events(S)
    print(f"[music] kick events {len(kidx)}; ghost bounces {len(refractory(*kick_events(S), gap=9)[0])}")
    for k in ("glow", "moon_glow", "ghost_glow", "petal_density", "water_shimmer", "stars", "firefly", "bloom",
              "g_sway", "g_kick"):
        v = M[k]
        print(f"  {k:14s} min {v.min():6.3f}  p50 {np.median(v):6.3f}  max {v.max():6.3f}")
    for k in ("g_clock", "drift_t", "wind_t", "ff_t"):
        print(f"  {k:14s} end {M[k][-1]:8.1f} s-equiv (song {S.duration:.1f} s)")
    rep = shake_report(S, SH)
    print("[shake]", json.dumps(rep))
    bad = check_shake(S, SH)
    print("[shake] PASS" if not bad else f"[shake] FAIL {bad}")
    if a.plot:
        _plot(S, M, SH, a.plot)
        print("[music] plot", a.plot)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(shake=rep, hits=hit_times(S), bpm=S.bpm), f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
