"""SPOOKY HALLOWEEN — scene plugin for scripts/lyric_viz.py (--look halloween).

A haunted house on a hill under a huge full moon, 2.5-D parallax:
sky -> stars -> MOON (carries the Tzeke000 emblem) -> clouds -> lightning
-> far dead-tree ridge -> horizon fog -> HOUSE LAYER (hill + house + fence,
a Blender plate) -> bats bursting out -> ground fog + ghost wisps -> the big
dead tree -> GRAVEYARD LAYER (stones + jack-o'-lanterns, a Blender plate)
-> ground fog -> close bats -> the burnt-in title.

What drives what (the music has to be legible in it):
  sub-bass (bass) ....... jack-o'-lantern glow, moon heat
  kick .................. moon + emblem pulse, window punch in drops
  snare ................. bat flocks burst from the house / tree; a figure
                          passes a lit window; lightning budget
  hi-hat sparkle ........ (high minus its smooth) lantern flicker, star
                          twinkle, ghost shimmer
  spectrum (bars) ....... the house WINDOWS: each window column is a band,
                          floors light bottom-up like a meter
  build (pre) ........... wind rises (tree sway, fog speed), moon swells and
                          reddens, camera pulls back (anticipation)
  half-beat black ....... every light snuffs out
  drop entry ............ lightning silhouettes the house, shake + zoom
                          slam, a bat swarm crosses the moon, HIT A BUMP
                          burns in
  big drop .............. blood moon, swarms every 8 bars, everything lit
  sung "bump" ........... the lanterns' mouths flare and "sing"

⚠ STATELESS: everything is a pure function of the frame index + whole-song
arrays computed once (cumsums, frames-since-event) and assets loaded once;
every random choice is seeded by a constant or an event index.

Assets (house + graveyard plates, light-separated grey passes) are rendered
by scripts/lyric_scenes/blender/halloween_scene.py into assets/scenes/halloween/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

f32 = np.float32
REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets" / "scenes" / "halloween"
REBUILD = ("~/.local/bin/blender -b --factory-startup --python-exit-code 1 "
           "--python scripts/lyric_scenes/blender/halloween_scene.py -- "
           "--out assets/scenes/halloween")
PLATE_K = 1.3              # plates are 1.3x the frame: room for the push-in

# ---------------------------------------------------------------- palettes
PALETTES: dict[str, dict] = {
    # pumpkin orange / witch purple / slime green on deep blue-black
    "classic": dict(
        sky_top=(3, 4, 15), sky_hor=(40, 26, 70), hor_glow=(70, 44, 96),
        star=(215, 220, 255), moon=(238, 232, 216), moon_red=(255, 96, 34),
        blood=(205, 24, 18), emblem=(255, 140, 30), emblem_big=(120, 255, 70),
        moonlight=(150, 160, 215), fill=(34, 42, 96), rim=(200, 205, 255),
        cloud=(26, 20, 44), ridge=(30, 24, 56),
        fog=(66, 52, 112), fog_hot=[(110, 255, 90), (170, 80, 255),
                                    (255, 130, 30)],
        stone=(150, 150, 175), skin=(255, 110, 8), stem=(80, 105, 40),
        glow=(255, 148, 36), candle=(255, 170, 80),
        meter=[(255, 120, 15), (255, 215, 80), (130, 255, 70)],
        ghost=(150, 255, 180), bat=(5, 3, 10), tree=(7, 6, 14),
        bolt=(215, 205, 255), title=(255, 196, 86), title_edge=(255, 86, 10),
        char=(18, 6, 2)),
    # crimson / black / bone white
    "blood": dict(
        sky_top=(4, 1, 2), sky_hor=(52, 8, 12), hor_glow=(96, 16, 18),
        star=(255, 225, 215), moon=(240, 234, 226), moon_red=(230, 40, 28),
        blood=(180, 8, 8), emblem=(255, 40, 30), emblem_big=(255, 240, 220),
        moonlight=(205, 185, 180), fill=(56, 16, 22), rim=(255, 225, 210),
        cloud=(30, 6, 8), ridge=(36, 8, 12),
        fog=(84, 22, 26), fog_hot=[(230, 24, 30), (245, 232, 215),
                                   (150, 0, 18)],
        stone=(190, 180, 172), skin=(225, 60, 24), stem=(60, 42, 30),
        glow=(255, 70, 32), candle=(255, 120, 70),
        meter=[(200, 16, 20), (255, 80, 40), (250, 238, 222)],
        ghost=(242, 234, 222), bat=(3, 0, 0), tree=(6, 2, 3),
        bolt=(255, 236, 228), title=(250, 238, 222), title_edge=(225, 20, 20),
        char=(12, 0, 0)),
    # cold cyan / ghost white / violet
    "spectral": dict(
        sky_top=(1, 4, 11), sky_hor=(12, 44, 64), hor_glow=(26, 76, 98),
        star=(220, 245, 255), moon=(226, 238, 246), moon_red=(175, 115, 255),
        blood=(120, 50, 255), emblem=(80, 235, 255), emblem_big=(200, 140, 255),
        moonlight=(160, 215, 255), fill=(18, 44, 86), rim=(205, 245, 255),
        cloud=(8, 22, 36), ridge=(12, 34, 50),
        fog=(34, 86, 112), fog_hot=[(60, 240, 255), (175, 95, 255),
                                    (235, 250, 255)],
        stone=(160, 182, 196), skin=(200, 225, 232), stem=(70, 95, 105),
        glow=(90, 236, 255), candle=(130, 230, 255),
        meter=[(60, 210, 255), (175, 120, 255), (240, 250, 255)],
        ghost=(230, 250, 255), bat=(2, 4, 9), tree=(3, 7, 12),
        bolt=(205, 242, 255), title=(232, 254, 255), title_edge=(60, 210, 255),
        char=(0, 8, 18)),
}

STYLE = dict(palette=[(255, 110, 8), (170, 80, 255), (110, 255, 90),
                      (244, 232, 196)],
             font_title="Anton-Regular.ttf",
             font_lyrics="ArchivoBlack-Regular.ttf")
BLOOM = 0.36

# EQ: window column -> spectrum bars (of 40), low on the left
COL_BANDS = {"A": (0, 3), "B": (3, 6), "C": (6, 10), "D": (10, 15),
             "E": (15, 21), "F": (21, 29), "G": (29, 40)}
COL_ORDER = "ABCDEFG"

# frame-space layout (fractions of W / H) at camera Z = 1
MOON_C = (0.705, 0.300)    # centre (u, v)
MOON_R = 0.215             # radius in frame heights
TREE_BASE = (0.055, 0.87)  # big dead tree's foot
TREE_ANCHOR = (0.17, 0.28)  # where tree flocks burst from

# parallax depth factors (1 = the house)
K_MOON, K_RIDGE, K_HOUSE, K_TREE, K_FG = 0.12, 0.40, 1.0, 1.08, 1.12
OVERSCAN = 1.06


def _c(x):
    return np.asarray(x, f32)


def _lerp(a, b, t):
    return _c(a) * (1.0 - t) + _c(b) * t


def _ramp(cols, h):
    c = _c(cols)
    x = float(np.clip(h, 0, 1)) * (len(c) - 1)
    k = min(len(c) - 2, int(x))
    fr = x - k
    return c[k] * (1 - fr) + c[k + 1] * fr


def _vn(x, seed, lv):
    """1-D value noise (stateless): smooth random in [0, 1)."""
    i0 = math.floor(x)
    fr = x - i0
    a_ = float(lv._hash01(i0, seed))
    b_ = float(lv._hash01(i0 + 1, seed))
    u = fr * fr * (3 - 2 * fr)
    return a_ + (b_ - a_) * u


def _ease_io(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- timeline
def _timeline(R, a, lv) -> dict:
    c = R.scene_cache
    if "T" in c:
        return c["T"]
    fps = R.fps
    T = dict(lv.stage_timeline(a, fps, "acid", label="halloween"))
    n = len(a.rms)
    I, hot, big, pre = T["I"], T["hot"], T["big"], T["pre"]
    black = T["black"]
    gs = lv._gauss_smooth
    E = np.clip(np.maximum(I, 0.8 * pre), 0, 1).astype(f32)
    E2 = gs(E, fps, 1.0)
    hot_f = gs(hot.astype(f32), fps, 0.12)
    big_f = gs(big.astype(f32), fps, 0.45)
    heat = np.clip(gs(a.bass, fps, 0.10) * 1.1, 0, 1)
    hs = gs(a.high, fps, 0.35)
    spark = np.clip((np.asarray(a.high, f32) - hs) * 5.0, 0, 1)
    # WIND: a breeze in the calm, gusting up through the build, strong in drops
    wind = (0.25 + 0.45 * E2 + 1.4 * pre ** 1.5 + 0.35 * hot_f
            + 0.30 * big.astype(f32)).astype(f32)
    wph = (np.cumsum(wind.astype(np.float64)) / fps)
    fogx = np.cumsum((0.010 + 0.030 * wind).astype(np.float64)) / fps
    # CAMERA: a push-in toward the house over each section that creeps
    # faster with intensity; a small pull-BACK across the build (the wind-up
    # before the drop), smoothed so section changes ease instead of cut
    Zr = np.zeros(n, np.float64)
    for s0, s1, h in T["segs"]:
        rate = (0.006 + 0.024 * E2[s0:s1].astype(np.float64)) / fps
        cum = np.cumsum(rate)
        Zr[s0:s1] = 0.15 * (1.0 - np.exp(-cum / 0.15))
    Zr -= 0.035 * pre.astype(np.float64) ** 2
    Z = 1.0 + gs(Zr, fps, 0.22)
    oph = np.cumsum((0.020 + 0.035 * E2).astype(np.float64)) / fps
    camx = (0.030 * np.sin(2 * np.pi * 0.5 * oph)).astype(f32)
    camy = (0.008 * np.sin(2 * np.pi * 0.37 * oph + 1.0)).astype(f32)
    # LIGHTNING: every drop entry; every 4th bar in a drop (2nd in the big
    # one); rare distant sheet lightning on calm snares
    gbeat, bi, bar = T["gbeat"], T["bi"], T["bar"]
    ones = gbeat & (bi % 4 == 0)
    strike = np.zeros(n, f32)
    for s0 in T["entries"]:
        strike[s0] = 1.0
    for f in np.flatnonzero(ones & hot):
        if strike[max(0, f - 8):f + 1].any():
            continue
        if big[f] and bar[f] % 2 == 0:
            strike[f] = 0.8
        elif not big[f] and bar[f] % 4 == 2:
            strike[f] = 0.6
    rs = np.random.default_rng(1031)
    last = -10 ** 9
    for f in np.flatnonzero(np.asarray(a.snare, bool) & ~hot & ~black):
        if rs.random() < 0.12 and f - last > 6 * fps:
            strike[f] = max(strike[f], 0.25)
            last = f
    lage, lidx = lv._age_since(strike > 0)
    shake = np.zeros(n, f32)
    shake[ones & big] = 0.40
    shake[strike >= 0.6] = np.maximum(shake[strike >= 0.6], 0.6)
    for s0 in T["entries"]:
        shake[s0] = 1.0
    sh_age, sh_idx = lv._age_since(shake > 0)
    # BAT FLOCKS on snares (every snare in a drop, sparser in the calm)
    snare = np.asarray(a.snare, bool) & ~black
    fl_f, fl_k = [], []
    lastf = -10 ** 9
    for f in np.flatnonzero(snare):
        gap = (f - lastf) / fps
        if hot[f]:
            if gap < 0.45:
                continue
            kind = 2 if big[f] else 1
        else:
            if gap < 1.6 or lv._hash01(f, 3.3) > 0.45 + 0.4 * pre[f]:
                continue
            kind = 0
        fl_f.append(int(f))
        fl_k.append(kind)
        lastf = f
    # SWARMS across the moon: drop entries + every 8th bar in the big drop
    sw_f = list(T["entries"])
    for f in np.flatnonzero(ones & big):
        if bar[f] % 8 == 4 and all(abs(f - s) > 3 * fps for s in sw_f):
            sw_f.append(int(f))
    sw_f = sorted(sw_f)
    # WINDOW PASSERS: a figure crosses a lit window on some snares
    ps_f = []
    lastp = -10 ** 9
    for f in np.flatnonzero(snare):
        gap = (f - lastp) / fps
        if gap < (1.3 if hot[f] else 2.2):
            continue
        if lv._hash01(f, 9.1) > (0.30 if hot[f] else 0.40):
            continue
        ps_f.append(int(f))
        lastp = f
    redness = np.clip(np.maximum.reduce([0.9 * pre ** 1.3, 0.28 * hot_f,
                                         big_f]), 0, 1).astype(f32)
    msize = (1.0 + 0.16 * pre ** 1.5 + 0.08 * big_f + 0.04 * hot_f).astype(f32)
    bs = np.asarray(a.bars, f32)
    bars = bs.copy()
    bars[1:] = 0.5 * (bs[1:] + bs[:-1])
    # per window-column band level, normalised to its own loud range so every
    # column of windows can climb all its floors (the raw highs never would)
    colL = np.stack([bars[:, lo_:hi_].mean(axis=1) for (lo_, hi_) in
                     (COL_BANDS[k_] for k_ in COL_ORDER)], axis=1)
    ref = np.percentile(colL[hot] if hot.any() else colL, 92, axis=0)
    colL = np.clip(colL / np.maximum(ref, 1e-3), 0, 1.3).astype(f32)
    # the entry that each frame's drop began with (title burn), + big flag
    T.update(E=E, E2=E2, hot_f=hot_f, big_f=big_f, heat=heat, spark=spark,
             wind=wind, wph=wph, fogx=fogx, Z=Z.astype(f32), camx=camx,
             camy=camy, strike=strike, lage=lage, lidx=lidx, shake=shake,
             sh_age=sh_age, sh_idx=sh_idx, fl_f=np.asarray(fl_f, np.int64),
             fl_k=np.asarray(fl_k, np.int64), sw_f=np.asarray(sw_f, np.int64),
             ps_f=np.asarray(ps_f, np.int64), redness=redness, msize=msize,
             bars=bars, colL=colL)
    print(f"[halloween] {len(fl_f)} bat flocks, {len(sw_f)} swarms, "
          f"{len(ps_f)} window passers, {int((strike > 0).sum())} strikes")
    c["T"] = T
    return T


# ---------------------------------------------------------------- assets
def _load_png(name: str, PW: int, PH: int):
    p = ASSETS / f"{name}.png"
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"[halloween] missing asset {p} — build them with:\n  {REBUILD}")
    im = im.astype(f32) / (65535.0 if im.dtype == np.uint16 else 255.0)
    if im.shape[1] != PW or im.shape[0] != PH:
        im = cv2.resize(im, (PW, PH), interpolation=cv2.INTER_AREA)
    # RGBA passes are straight alpha; BW / RGB passes (no alpha channel) come
    # out of Blender premultiplied — returned with alpha None
    if im.ndim == 2:
        return im, None
    rgb = np.ascontiguousarray(im[..., 2::-1][..., :3])
    al = np.ascontiguousarray(im[..., 3]) if im.shape[2] == 4 else None
    return rgb, al


def _lum(x, al=None):
    """Grey value of a pass, premultiplied."""
    g = x if x.ndim == 2 else x.mean(axis=2)
    return g * al if al is not None else g


def _rim(al, H, dx=1, dy=-1, px=2.2):
    """Silhouette edge facing the moon (up-right): coverage whose neighbour
    toward the moon is empty."""
    r = max(1, int(round(px * H / 1080.0)))
    sh = np.roll(al, (-dy * r, -dx * r), axis=(0, 1))
    rim = np.clip(al - sh, 0, 1)
    return cv2.GaussianBlur(rim, (0, 0), max(0.6, 0.5 * r))


def _text_mask(R, txt, font, max_w, max_h, pad=8):
    from PIL import Image, ImageDraw
    f = R.font(220, font)
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    bb = probe.textbbox((0, 0), txt, font=f)
    im = Image.new("L", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), txt, font=f, fill=255)
    arr = np.asarray(im, f32) / 255.0
    sc = min(max_w / arr.shape[1], max_h / arr.shape[0])
    tw, th = max(1, int(arr.shape[1] * sc)), max(1, int(arr.shape[0] * sc))
    return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)


def _noise_tile(rng, h, w, octaves):
    """Horizontally tileable multi-octave value noise, 0..1."""
    out = np.zeros((h, w), f32)
    tot = 0.0
    for gh, gw, amp in octaves:
        nz = rng.random((gh, gw)).astype(f32)
        nz = np.hstack([nz, nz[:, :3]])
        big_w = int(round(w * (gw + 3) / gw))
        up = cv2.resize(nz, (big_w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * up[:, :w]
        tot += amp
    return out / tot


def _gen_tree(seed, scale=1.0, lean=0.12, trunk_n=6, depth=3,
              spread=(0.35, 0.95), bias=0.0, kids=0.55, twig=True):
    """A gnarled dead tree as FK segments: [parent, rel angle, length, w0,
    w1, level]. Units of frame height; angle 0 = right, -pi/2 = up.
    Few, thick, crooked limbs (a readable silhouette) — not a hairball."""
    rng = np.random.default_rng(seed)
    segs = []

    def chain(parent, rel0, L, w, lvl, n):
        p = parent
        for k in range(n):
            rel = (rel0 if k == 0 else 0.0) + rng.normal(0, 0.20 + 0.05 * lvl)
            ln = L / n * rng.uniform(0.8, 1.2)
            wa = w * (1.0 - 0.55 * k / n)
            wb = w * (1.0 - 0.55 * (k + 1) / n)
            segs.append([p, rel, ln, wa, wb, lvl])
            p = len(segs) - 1
            if lvl < depth and k >= (2 if lvl == 0 else 1) and k < n - 1 \
                    and rng.random() < kids:
                side = 1 if rng.random() < 0.5 + bias else -1
                ang = side * rng.uniform(*spread)
                chain(p, ang, L * rng.uniform(0.45, 0.70), wb * rng.uniform(0.55, 0.75),
                      lvl + 1, max(2, n - 1))
        if lvl < depth and (lvl > 0 or twig):          # crooked fork at the end
            for side in (-1, 1):
                if rng.random() < 0.8:
                    chain(p, side * rng.uniform(0.25, 0.65), L * 0.45,
                          w * 0.40, lvl + 1, 2)
    chain(-1, -np.pi / 2 + lean, 0.55 * scale, 0.050 * scale, 0, trunk_n)
    return segs


def _gen_main_tree(seed=42):
    """The big framing tree, art-directed: a crooked trunk on the left edge,
    one limb arching over the top toward the house (claw twigs hanging),
    one up-left out of frame, one low and short. Random twigs on top."""
    rng = np.random.default_rng(seed)
    segs = []
    heads = {}

    def limb(parent, headings, lengths, w0, w1, lvl):
        p = parent
        n = len(headings)
        for k, (hd, ln) in enumerate(zip(headings, lengths)):
            pa = heads[p] if p >= 0 else 0.0
            wa = w0 + (w1 - w0) * k / n
            wb = w0 + (w1 - w0) * (k + 1) / n
            segs.append([p, hd - pa, ln, wa, wb, lvl])
            p = len(segs) - 1
            heads[p] = hd
        return list(range(len(segs) - n, len(segs)))

    def twigs(node, lvl, n_max=2):
        for _ in range(n_max):
            if rng.random() > 0.75:
                continue
            hd = heads[node] + rng.choice([-1, 1]) * rng.uniform(0.4, 1.1)
            if rng.random() < 0.35:
                hd = np.pi / 2 + rng.uniform(-0.6, 0.6)        # hanging claw
            m = int(rng.integers(2, 4))
            hs_, ls_ = [], []
            for _k in range(m):
                hs_.append(hd)
                ls_.append(rng.uniform(0.022, 0.040))
                hd += rng.normal(0, 0.35)
            w = segs[node][4] * rng.uniform(0.35, 0.55)
            ids = limb(node, hs_, ls_, w, w * 0.35, lvl)
            if lvl < 3:
                twigs(ids[-1], lvl + 1, 1)
    trunk = limb(-1, [-1.30, -1.52, -1.78, -1.85, -1.58, -1.30, -1.18],
                 [0.075, 0.075, 0.07, 0.07, 0.065, 0.06, 0.055], 0.062, 0.028, 0)
    top = limb(trunk[-1], [-1.05, -0.85, -0.62, -0.42, -0.22, -0.05, 0.12, 0.3],
               [0.08, 0.08, 0.075, 0.075, 0.07, 0.065, 0.055, 0.045], 0.028, 0.006, 1)
    upl = limb(trunk[4], [-2.0, -2.25, -2.35, -2.15], [0.09] * 4, 0.024, 0.008, 1)
    low = limb(trunk[2], [-0.55, -0.25, 0.05, 0.35, 0.7], [0.05, 0.05, 0.045, 0.04, 0.03],
               0.020, 0.005, 1)
    up2 = limb(top[2], [-1.45, -1.6, -1.35, -1.5], [0.06, 0.06, 0.05, 0.05], 0.012, 0.004, 2)
    for nd in top[1:] + upl[1:] + low[1:] + up2[1:]:
        twigs(nd, 2)
    return segs


def _fk(segs, root, sway, flex):
    """Forward kinematics: returns (x0, y0, x1, y1, w0, w1, lvl) per segment."""
    out = np.zeros((len(segs), 7), f32)
    absang = np.zeros(len(segs), np.float64)
    for j, (p, rel, ln, w0, w1, lvl) in enumerate(segs):
        if p < 0:
            a0 = rel + sway * flex[0]
            x0, y0 = root
        else:
            a0 = absang[p] + rel + sway * flex[min(lvl, len(flex) - 1)]
            x0, y0 = out[p, 2], out[p, 3]
        absang[j] = a0
        out[j] = (x0, y0, x0 + ln * math.cos(a0), y0 + ln * math.sin(a0),
                  w0, w1, lvl)
    return out


def _geo(R, lv) -> dict:
    key = ("G", R.W, R.H, R.scene_palette)
    c = R.scene_cache
    if key in c:
        return c[key]
    import sys
    P = lv.scene_palette(R, sys.modules[__name__])
    W, H = R.W, R.H
    PW, PH = int(round(W * PLATE_K)), int(round(H * PLATE_K))
    mp = ASSETS / "meta.json"
    if not mp.is_file():
        raise SystemExit(f"[halloween] missing {mp} — build the assets with:\n  {REBUILD}")
    meta = json.loads(mp.read_text())
    sc = PW / float(meta["size"][0])
    G = dict(P=P, PW=PW, PH=PH, sc=sc)
    # ---------------- house layer: key, fill, spill, rim (+ glass, ids)
    krgb, ka = _load_png("h_key", PW, PH)
    frgb, fa = _load_png("h_fill", PW, PH)
    grgb, ga = _load_png("h_glass", PW, PH)
    srgb, sa = _load_png("h_spill", PW, PH)
    hg = _lum(grgb, ga)
    hk = _lum(krgb, ka)
    hf = _lum(frgb, fa)
    hs = _lum(srgb, sa) * np.clip(1.0 - 3.0 * hg, 0, 1)
    rim = _rim(ka, H)
    opq = ka.min(axis=1) > 0.995
    hfull = PH
    for r in range(PH - 1, -1, -1):
        if not opq[r]:
            break
        hfull = r
    G["h_full_frac"] = hfull / float(PH)
    rows = np.flatnonzero(ka.max(axis=1) > 0.003)
    y0 = max(0, int(rows[0]) - 2)
    G["h_y0"] = y0
    # 4 light channels + coverage: warped together, combined at frame res
    G["h_stack"] = np.ascontiguousarray(np.stack([hk, hf, hs, rim, ka], axis=2)[y0:])
    # window id map from the projected polygons (+ dilate over the AA edge)
    wins = meta["windows"]
    ids = np.zeros((PH, PW), np.int32)
    for j, w in enumerate(wins):
        pts = np.round(np.asarray(w["poly"], np.float64) * sc).astype(np.int32)
        cv2.fillPoly(ids, [pts], j + 1)
    ids = cv2.dilate(ids.astype(np.float32), np.ones((5, 5), np.uint8)).astype(np.int32)
    ys, xs = np.nonzero(ids)
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    G["w_box"] = (by0 - y0, by1 - y0, bx0, bx1)
    G["w_ids"] = ids[by0:by1, bx0:bx1]
    G["w_glass"] = np.ascontiguousarray(hg[by0:by1, bx0:bx1])
    col = np.array([COL_ORDER.index(w["col"]) for w in wins], np.int64)
    lvl = np.array([w["lvl"] for w in wins], np.int64)
    nlev = np.array([1 + lvl[col == c_].max() for c_ in range(7)], np.int64)
    G.update(w_col=col, w_lvl=lvl, w_nlev=nlev[col],
             w_bbox=[(lambda p: (int(p[:, 1].min()) - by0, int(p[:, 1].max()) - by0,
                                 int(p[:, 0].min()) - bx0, int(p[:, 0].max()) - bx0))(
                 np.asarray(w["poly"]) * sc) for w in wins])
    # anchors in plate px
    G["anc"] = {k: (v[0] * sc, v[1] * sc) for k, v in meta["anchors"].items()}
    # ---------------- graveyard layer (pumpkins channel-packed)
    krgb, ka = _load_png("f_key", PW, PH)
    frgb, fa = _load_png("f_fill", PW, PH)
    rows = np.flatnonzero(ka.max(axis=1) > 0.003)
    fy0 = max(0, int(rows[0]) - 2)
    G["f_y0"] = fy0

    def unpack(rgb, al):
        if al is None:
            al = 1.0
        r, g, b = rgb[..., 0] * al, rgb[..., 1] * al, rgb[..., 2] * al
        return g, np.clip(r - g, 0, None), np.clip(b - g, 0, None)
    Kg, Ks, Kt = unpack(krgb, ka)
    Fg, Fs, Ft = unpack(frgb, fa)
    frim = _rim(ka, H)
    G["f_stack"] = np.ascontiguousarray(
        np.stack([Kg, Ks, Kt, Fg, Fs, Ft, frim, ka], axis=2)[fy0:])
    pumps = []
    for k, pm in enumerate(meta["pumpkins"]):
        lrgb, la = _load_png(f"f_lit{k}", PW, PH)
        lit = (lrgb if lrgb.ndim == 2 else lrgb.max(axis=2)) * (la if la is not None else 1.0)
        # normalise so the brightest hole sits at 1, then drop the faint far
        # spill (continuously: subtract the floor) so the sprite stays compact
        pk = float(np.percentile(lit[lit > 0.02], 99.5)) if (lit > 0.02).any() else 1.0
        lit = np.clip(lit / max(1e-4, pk) - 0.015, 0, None)
        ys, xs = np.nonzero(lit > 0)
        if len(ys) == 0:
            continue
        ly0, ly1 = max(fy0, int(ys.min()) - 2), min(PH, int(ys.max()) + 3)
        lx0, lx1 = max(0, int(xs.min()) - 2), min(PW, int(xs.max()) + 3)
        crop = np.ascontiguousarray(lit[ly0:ly1, lx0:lx1])
        mouth = np.zeros_like(crop)
        for pg in pm["polys"]["mouth"]:
            pts = np.round(np.asarray(pg) * sc - [lx0, ly0]).astype(np.int32)
            cv2.fillPoly(mouth, [pts], 1.0)
        rpx = max(2, int(0.006 * H))
        mouth = cv2.GaussianBlur(cv2.dilate(mouth, np.ones((2 * rpx + 1,) * 2, np.uint8)),
                                 (0, 0), rpx)
        mys, mxs = np.nonzero(mouth > 0.05)
        mb = (int(mys.min()), int(mys.max()) + 1, int(mxs.min()), int(mxs.max()) + 1) \
            if len(mys) else None
        pumps.append(dict(box=(ly0 - fy0, ly1 - fy0, lx0, lx1), lit=crop,
                          mouth=mouth, mbox=mb, seed=17.0 + 5.3 * k,
                          center=(pm["center"][0] * sc, pm["center"][1] * sc)))
    G["pumps"] = pumps
    # rows the graveyard ALWAYS covers (opaque edge to edge) need no house /
    # ridge pixels under them: crop those layers there (+ parallax margin)
    opq = ka.min(axis=1) > 0.995
    full = PH
    for r in range(PH - 1, -1, -1):
        if not opq[r]:
            break
        full = r
    h_cut = min(PH, full + int(0.07 * PH)) - G["h_y0"]
    G["h_stack"] = G["h_stack"][:max(8, h_cut)]
    if pumps:                       # one union box: the lanterns' light sprite
        G["p_box"] = (min(p["box"][0] for p in pumps), max(p["box"][1] for p in pumps),
                      min(p["box"][2] for p in pumps), max(p["box"][3] for p in pumps))
    # ---------------- sky (quarter res), stars, clouds, fog fields
    qh, qw = max(8, H // 4), max(8, W // 4)
    G["qh"], G["qw"] = qh, qw
    yq = np.linspace(0, 1, qh, dtype=f32)[:, None, None]
    g = np.clip(yq / 0.66, 0, 1) ** 1.8
    sky = _c(P["sky_top"]) * (1 - g) + _c(P["sky_hor"]) * g
    sky += _c(P["hor_glow"]) * np.exp(-((yq - 0.63) / 0.07) ** 2) * 0.8
    G["sky"] = sky.astype(f32) * np.ones((1, qw, 1), f32)
    rs = np.random.default_rng(616)
    ns = 260
    st = np.stack([rs.random(ns), rs.random(ns) ** 1.4 * 0.62, rs.random(ns),
                   rs.random(ns), rs.uniform(0.3, 1.0, ns)], 1).astype(f32)
    mx, my = MOON_C[0] * W, MOON_C[1] * H
    d = np.hypot(st[:, 0] * W - mx, st[:, 1] * H - my) / (MOON_R * H)
    G["stars"] = st[d > 1.5]
    cw = qw * 3
    G["cw"] = cw
    G["cloud"] = _noise_tile(np.random.default_rng(71), qh, cw,   # long thin streaks
                             ((6, 4, 1.0), (12, 9, 0.5), (24, 20, 0.25), (48, 44, 0.12)))
    G["cloud_prof"] = (np.exp(-((np.linspace(0, 1, qh) - 0.30) / 0.17) ** 2)
                       ).astype(f32)[:, None]
    cr = np.flatnonzero(G["cloud_prof"][:, 0] > 0.02)
    G["cloud_rows"] = (int(cr[0]), int(cr[-1]) + 1)
    G["fog"] = [_noise_tile(np.random.default_rng(s), qh, cw,
                            ((2, 6, 1.0), (4, 15, 0.55), (9, 40, 0.3), (18, 90, 0.15)))
                for s in (201, 202, 203)]
    ys_ = np.linspace(0, 1, qh, dtype=f32)[:, None]
    G["fog_prof"] = [np.exp(-((ys_ - 0.600) / 0.060) ** 2),   # horizon, behind house
                     np.exp(-((ys_ - 0.765) / 0.075) ** 2),   # hill base
                     np.clip((ys_ - 0.83) / 0.12, 0, 1) ** 1.2]  # graveyard floor
    G["qyy"], G["qxx"] = np.mgrid[0:qh, 0:qw].astype(f32)
    # ---------------- moon texture + emblem (at the largest size used)
    Rm = int(math.ceil(MOON_R * H * 1.30 * OVERSCAN))
    D = 2 * Rm + 2
    yy, xx = np.mgrid[0:D, 0:D].astype(f32)
    rr = np.hypot(xx - D / 2, yy - D / 2) / Rm
    disc = np.clip((1.0 - rr) * Rm * 0.9, 0, 1)
    mr = np.random.default_rng(1313)
    maria = _noise_tile(mr, D, D, ((3, 3, 1.0), (6, 6, 0.5), (12, 12, 0.25)))
    tex = 0.88 - 0.26 * np.clip((maria - 0.42) * 2.5, 0, 1)
    fine = _noise_tile(mr, D, D, ((24, 24, 1.0), (60, 60, 0.6), (120, 120, 0.3)))
    tex += 0.10 * (fine - 0.5)
    for _ in range(26):                                 # craters
        cx_, cy_ = mr.uniform(0.1, 0.9) * D, mr.uniform(0.1, 0.9) * D
        cr = Rm * mr.uniform(0.012, 0.06) * (1.0 if mr.random() < 0.85 else 2.0)
        dd = np.hypot(xx - cx_, yy - cy_) / cr
        tex -= 0.05 * np.clip(1.0 - dd, 0, 1) ** 0.6 * (dd < 1)
        tex += 0.05 * np.exp(-((dd - 1.0) / 0.22) ** 2)
    limb = 0.70 + 0.30 * np.sqrt(np.clip(1 - rr ** 2, 0, 1))
    G["moon_tex"] = (np.clip(tex, 0.3, 1.1) * limb * disc).astype(f32)
    G["moon_disc"] = disc.astype(f32)
    G["moon_D"] = D
    line = fill = red = None
    if R.logo is not None:
        rgb = np.asarray(R.logo.convert("RGB"), f32) / 255.0
        al = np.asarray(R.logo.split()[-1], f32) / 255.0
        lum = rgb.mean(axis=2)
        sat = rgb.max(axis=2) - rgb.min(axis=2)
        if al.std() < 0.02:
            al = 1.0 - ((lum > 0.93) & (sat < 0.10)).astype(f32)
        rd = ((rgb[..., 0] > 0.5) & (rgb[..., 1] < 0.35) & (rgb[..., 2] < 0.40)).astype(f32) * al
        dk = (lum < 0.35).astype(f32) * al * (1 - rd)
        fl = np.clip(al - rd - dk, 0, 1)
        side = int(D * 0.60)
        lh, lw = rgb.shape[:2]
        s_ = side / max(lh, lw)
        ow, oh = max(1, int(lw * s_)), max(1, int(lh * s_))

        def place(m):
            out = np.zeros((D, D), f32)
            m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_AREA)
            oy, ox = (D - oh) // 2, (D - ow) // 2
            out[oy:oy + oh, ox:ox + ow] = m
            return out
        line, fill, red = place(dk), place(fl), place(rd)
    G["em_line"], G["em_fill"], G["em_red"] = line, fill, red
    if line is not None:
        G["em_glow"] = cv2.GaussianBlur(line + 0.5 * red, (0, 0), D * 0.012)
        pack = [G["moon_tex"], G["moon_disc"], line, fill, red, G["em_glow"]]
    else:
        pack = [G["moon_tex"], G["moon_disc"]] + [np.zeros_like(G["moon_disc"])] * 4
    G["moon_pack"] = np.ascontiguousarray(np.stack(pack, axis=2))
    # ---------------- far ridge: dead-tree silhouettes on a low ridge (half res)
    RW, RH = int(W * 1.5), H
    ridge = np.zeros((RH, RW), f32)
    rr_ = np.random.default_rng(4242)
    xs_ = np.arange(RW, dtype=f32) / RH
    gl = 0.615 + 0.018 * np.sin(xs_ * 2.1 + 0.4) + 0.012 * np.sin(xs_ * 5.3 + 2.0) \
        + 0.006 * np.sin(xs_ * 13.0)
    gy = (gl * RH).astype(np.int32)
    for x in range(RW):
        ridge[gy[x]:, x] = 1.0
    for _ in range(52):
        x0 = rr_.uniform(0, RW)
        base = (x0 / RH, float(gl[min(RW - 1, int(x0))]) + 0.005)
        segs = _gen_tree(int(rr_.integers(1 << 30)), scale=rr_.uniform(0.07, 0.15),
                         lean=rr_.normal(0, 0.15), trunk_n=4, depth=3)
        fkd = _fk(segs, base, 0.0, (0, 0, 0, 0))
        for x0_, y0_, x1_, y1_, w0, w1, lv_ in fkd:
            cv2.line(ridge, (int(x0_ * RH), int(y0_ * RH)), (int(x1_ * RH), int(y1_ * RH)),
                     1.0, max(1, int(round(0.5 * (w0 + w1) * RH))), cv2.LINE_AA)
    rows = np.flatnonzero(ridge.max(axis=1) > 0)
    G["r_y0"] = int(rows[0]) if len(rows) else 0
    G["ridge"] = np.ascontiguousarray(ridge[G["r_y0"]:])
    rrim = _rim(G["ridge"], RH, px=1.2)
    r_cut = min(RH, int((G["h_full_frac"] + 0.06) * RH)) - G["r_y0"]
    G["r_stack"] = np.ascontiguousarray(np.stack(
        [G["ridge"], rrim, np.zeros_like(rrim), G["ridge"]], axis=2)[:max(8, r_cut)])
    G["RW"], G["RH"] = RW, RH
    # ---------------- the big dead tree (FK, sways per frame)
    G["tree"] = _gen_main_tree()
    # ---------------- ghost sprite (alpha), eyes cut out
    gh_, gw_ = 160, 110
    gm = np.zeros((gh_, gw_), f32)
    pts = []
    for k in range(25):
        a_ = np.pi + np.pi * k / 24
        pts.append((gw_ / 2 + 0.30 * gw_ * math.cos(a_), 0.36 * gh_ + 0.30 * gw_ * math.sin(a_)))
    pts.append((gw_ * 0.84, 0.70 * gh_))
    for k in range(9):                           # scalloped hem, right to left
        xk = gw_ * (0.88 - 0.76 * k / 8)
        yk = gh_ * (0.98 if k % 2 == 0 else 0.86)
        pts.append((xk, yk))
    pts.append((gw_ * 0.16, 0.70 * gh_))
    cv2.fillPoly(gm, [np.round(pts).astype(np.int32)], 1.0, cv2.LINE_AA)
    for ex in (0.40, 0.60):
        cv2.ellipse(gm, (int(gw_ * ex), int(gh_ * 0.32)), (int(gw_ * 0.06), int(gh_ * 0.065)),
                    0, 0, 360, 0.0, -1, cv2.LINE_AA)
    cv2.ellipse(gm, (int(gw_ * 0.5), int(gh_ * 0.47)), (int(gw_ * 0.05), int(gh_ * 0.06)),
                0, 0, 360, 0.15, -1, cv2.LINE_AA)
    vgrad = np.clip(1.15 - np.linspace(0, 1, gh_, dtype=f32) ** 1.5, 0, 1)[:, None]
    gm = cv2.GaussianBlur(gm, (0, 0), 1.6) * vgrad
    G["ghost"] = gm
    G["ghost_halo"] = cv2.GaussianBlur(np.pad(gm, 30), (0, 0), 14)
    # ---------------- title (burn-in): mask, char ring, flame noise
    tm = _text_mask(R, (R.title or "HIT A BUMP").upper(), "Anton-Regular.ttf",
                    int(W * 0.70), int(H * 0.235))
    pad = int(H * 0.05)
    tm = np.pad(tm, pad)
    dil = cv2.dilate(tm, np.ones((max(3, H // 90),) * 2, np.uint8))
    G["t_mask"] = tm
    G["t_char"] = np.clip(cv2.GaussianBlur(dil, (0, 0), H / 300) - tm, 0, 1)
    G["t_glow"] = cv2.GaussianBlur(tm, (0, 0), H / 70)
    G["t_scrim"] = cv2.GaussianBlur(dil, (0, 0), H / 40)
    th_, tw_ = tm.shape
    G["t_noise"] = _noise_tile(np.random.default_rng(55), th_ * 2, tw_,
                               ((4, 12, 1.0), (9, 30, 0.5), (20, 70, 0.25)))
    G["t_burn"] = _noise_tile(np.random.default_rng(56), th_, tw_,
                              ((3, 8, 1.0), (7, 22, 0.5), (16, 60, 0.3)))
    print(f"[halloween] assets: plate {PW}x{PH}, {len(wins)} windows, "
          f"{len(pumps)} lanterns, {len(G['tree'])} tree segments, palette "
          f"{R.scene_palette or next(iter(PALETTES))}")
    c[key] = G
    return G


# ---------------------------------------------------------------- compositing
def _cam(T, i, k, f, W, H):
    """Plate->frame affine for a layer of depth k: (scale z, offset)."""
    Z = float(T["Z"][i])
    z = OVERSCAN + (Z - 1.0) * k
    ox = -float(T["camx"][i]) * k * H
    oy = -float(T["camy"][i]) * k * H
    return z, ox, oy


def _fg_lift(zf, oyf, foc, H):
    """The push-in zooms the graveyard about the house, which drives its
    bottom edge (and the lanterns on it) down out of frame. Lift the layer by
    90 % of that drop: the lanterns stay put while the stones and the hill
    still read the push. The plate's bottom row stays at/below the frame
    bottom ((1 - 0.9) * (zf - 1) * (H - foc) >= 0), so no gap opens."""
    return oyf - 0.9 * (zf - 1.0) * (H - foc[1])


def _rows(M, h, H):
    top = M[1, 2]
    bot = M[1, 1] * h + M[1, 2]
    r0 = max(0, int(math.floor(min(top, bot))) - 1)
    r1 = min(H, int(math.ceil(max(top, bot))) + 1)
    return r0, r1


def _layer(img, stack, M, mats):
    """Warp a channel STACK (h,w,C: light passes ..., coverage last) into the
    frame, combine the light passes into RGB with the 3x4 matrices `mats`
    (one per 4-channel group) at frame res, and composite it (premultiplied)."""
    H, W = img.shape[:2]
    r0, r1 = _rows(M, stack.shape[0], H)
    if r1 <= r0:
        return
    M2 = M.copy()
    M2[1, 2] -= r0
    out = cv2.warpAffine(stack, M2, (W, r1 - r0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    rgb = None
    for g_, m in enumerate(mats):
        part = np.ascontiguousarray(out[..., 4 * g_:4 * g_ + 4])
        if part.shape[2] < 4:
            part = np.dstack([part, np.zeros(part.shape[:2] + (4 - part.shape[2],), f32)])
        t_ = cv2.transform(part, m)
        rgb = t_ if rgb is None else cv2.add(rgb, t_)
    a_ = out[..., -1:]
    sub = img[r0:r1]
    sub *= (1.0 - a_)
    sub += rgb


def _add_sprite(img, spr, M, x0, y0, col=None):
    """Additive light (h,w[,3]) whose top-left sits at plate (x0, y0),
    warped by the layer's plate->frame affine M."""
    H, W = img.shape[:2]
    h, w = spr.shape[:2]
    s = float(M[0, 0])
    fx0 = s * x0 + M[0, 2]
    fy0 = s * y0 + M[1, 2]
    X0, Y0 = max(0, int(math.floor(fx0)) - 1), max(0, int(math.floor(fy0)) - 1)
    X1, Y1 = min(W, int(math.ceil(fx0 + s * w)) + 1), min(H, int(math.ceil(fy0 + s * h)) + 1)
    if X1 <= X0 or Y1 <= Y0:
        return
    M2 = np.float32([[s, 0, fx0 - X0], [0, s, fy0 - Y0]])
    out = cv2.warpAffine(spr, M2, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    if out.ndim == 2:
        out = out[..., None]
    if col is not None:
        out = out * col
    img[Y0:Y1, X0:X1] += out


def _plate_M(z, ox, oy, f, s0, cpl, cfr, y0=0):
    """frame = (p - cpl) * s0 * z + (cfr - f) * z + f + o, p_y = row + y0."""
    s = s0 * z
    tx = (cfr[0] - cpl[0] * s0 - f[0]) * z + f[0] + ox
    ty = (cfr[1] - cpl[1] * s0 - f[1]) * z + f[1] + oy
    return np.float32([[s, 0, tx], [0, s, ty + y0 * s]])


def _to_frame(pt, z, ox, oy, f, s0, cpl, cfr):
    x = ((pt[0] - cpl[0]) * s0 + cfr[0] - f[0]) * z + f[0] + ox
    y = ((pt[1] - cpl[1]) * s0 + cfr[1] - f[1]) * z + f[1] + oy
    return x, y


def _fog(img, G, T, i, which, amt, col, xoff):
    qh, qw, cw = G["qh"], G["qw"], G["cw"]
    nz = G["fog"][which]
    off = int(xoff * qh) % cw
    cols = (off + np.arange(qw)) % cw
    a_ = np.clip((nz[:, cols] - 0.36) * 2.2, 0, 1) * G["fog_prof"][which] * amt
    a_ = np.clip(a_, 0, 0.92).astype(f32)
    H, W = img.shape[:2]
    rows = np.flatnonzero(a_.max(axis=1) > 0.004)
    if not len(rows):
        return
    q0, q1 = max(0, int(rows[0]) - 1), min(qh, int(rows[-1]) + 2)
    r0, r1 = int(round(q0 * H / qh)), int(round(q1 * H / qh))
    Ar = cv2.resize(a_[q0:q1], (W, r1 - r0), interpolation=cv2.INTER_LINEAR)[..., None]
    sub = img[r0:r1]
    sub *= (1.0 - Ar)
    sub += Ar * _c(col)


def _bat_poly(cx, cy, span, ph, roll):
    """A bat silhouette (two scalloped wings + body + ears) as polygons."""
    up = math.sin(ph)
    ty = -0.34 * up                               # wingtip height
    mid = -0.12 * up
    wing = [(0.05, -0.04), (0.20, mid - 0.05), (0.50, ty), (0.43, ty + 0.10),
            (0.36, mid + 0.06), (0.27, mid + 0.12), (0.18, 0.05 + 0.4 * mid),
            (0.10, 0.10), (0.04, 0.07)]
    body = [(0.0, -0.13), (0.035, -0.17), (0.045, -0.09), (0.05, 0.05),
            (0.0, 0.14), (-0.05, 0.05), (-0.045, -0.09), (-0.035, -0.17)]
    cs, sn = math.cos(roll), math.sin(roll)
    out = []
    for poly in ([(x, y) for x, y in wing], [(-x, y) for x, y in wing], body):
        out.append(np.array([[cx + span * (cs * x - sn * y), cy + span * (sn * x + cs * y)]
                             for x, y in poly], np.float64))
    return out


def _bats_over(img, bats, col):
    """Draw bat silhouettes, compositing only inside their bounding box."""
    polys = []
    for (x, y, span, ph, roll) in bats:
        if span < 1.5:
            continue
        polys += _bat_poly(x, y, span, ph, roll)
    if not polys:
        return
    H, W = img.shape[:2]
    allp = np.vstack(polys)
    x0, y0 = max(0, int(allp[:, 0].min()) - 2), max(0, int(allp[:, 1].min()) - 2)
    x1, y1 = min(W, int(allp[:, 0].max()) + 3), min(H, int(allp[:, 1].max()) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), f32)
    cv2.fillPoly(m, [np.round((p - [x0, y0]) * 4).astype(np.int32) for p in polys], 1.0,
                 cv2.LINE_AA, shift=2)
    sub = img[y0:y1, x0:x1]
    sub *= (1.0 - m[..., None])
    sub += m[..., None] * col


def _flock_bats(T, G, i, fps, W, H, origins, lv):
    """Every live flock's bats at frame i: pure function of (age, seed)."""
    fl_f, fl_k = T["fl_f"], T["fl_k"]
    if not len(fl_f):
        return [], []
    life = int(3.6 * fps)
    lo = np.searchsorted(fl_f, i - life, side="left")
    hi = np.searchsorted(fl_f, i, side="right")
    far, near = [], []
    for e in range(lo, hi):
        f0 = int(fl_f[e])
        age = (i - f0) / fps
        kind = int(fl_k[e])
        rng = np.random.default_rng(70001 + f0)
        okeys = ["tower", "chim1", "chim2", "attic", "tower", "attic", "tree"]
        okey = okeys[int(rng.integers(len(okeys)))]
        ox, oy = origins[okey]
        nb = int(rng.integers(*((5, 9), (9, 15), (14, 21))[kind]))
        bias = rng.uniform(-0.6, 0.6)
        for _ in range(nb):
            delay = rng.uniform(0, 0.28)
            L = rng.uniform(2.2, 3.4)
            th = -np.pi / 2 + bias + rng.uniform(-1.25, 1.25)
            v = rng.uniform(0.12, 0.28) * H
            toward = rng.uniform(-0.6, 1.0)
            curl = rng.uniform(-1, 1)
            fq = rng.uniform(6.0, 9.5)
            ph0 = rng.uniform(0, 6.28)
            size = rng.uniform(0.020, 0.032) * H
            t = age - delay
            if t <= 0 or t >= L:
                continue
            u = t / L
            # burst out (ease-out), then flutter on: an arcing, wobbling path
            dist = v * L * (0.45 * (1 - math.exp(-5 * u)) / (1 - math.exp(-5)) + 0.55 * u)
            dx, dy = math.cos(th), math.sin(th)
            arc = curl * 0.25 * dist * math.sin(math.pi * u)
            wob = 0.020 * H * math.sin(2 * math.pi * (1.3 + 0.9 * abs(curl)) * t + ph0)
            x = ox + dx * dist - dy * (arc + wob)
            y = oy + dy * dist + dx * (arc + wob)
            zf = 1.0 + toward * 2.4 * u ** 1.5 if toward > 0 else \
                max(0.25, 1.0 + toward * 1.2 * u)
            span = size * zf * (1.0 if u < 0.85 or toward > 0 else (1 - u) / 0.15)
            roll = 0.5 * math.atan2(dy, abs(dx) + 1e-3) * 0.4 + 0.15 * math.sin(t * 3 + ph0)
            b = (x, y, span, 2 * math.pi * fq * t + ph0, roll)
            (near if zf > 1.7 else far).append(b)
    return far, near


def _swarm_bats(T, G, i, fps, W, H, mx, my, mr):
    sw = T["sw_f"]
    if not len(sw):
        return []
    life = int(4.0 * fps)
    lo = np.searchsorted(sw, i - life, side="left")
    hi = np.searchsorted(sw, i, side="right")
    out = []
    for e in range(lo, hi):
        f0 = int(sw[e])
        age = (i - f0) / fps
        rng = np.random.default_rng(90001 + f0)
        big = bool(T["big"][f0])
        nb = 110 if big else 70
        dirn = 1 if rng.random() < 0.5 else -1
        for _ in range(nb):
            delay = rng.uniform(0, 0.7)
            L = rng.uniform(1.8, 2.9)
            y0 = my + rng.normal(0, 0.55) * mr
            sag = rng.uniform(0.04, 0.10) * H * (1 if rng.random() < 0.5 else -1)
            fq = rng.uniform(8.0, 11.0)
            ph0 = rng.uniform(0, 6.28)
            size = rng.uniform(0.010, 0.026) * H
            t = age - delay
            if t <= 0 or t >= L:
                continue
            u = t / L
            x = mx + dirn * (-0.62 * W + 1.24 * W * u)
            y = y0 + sag * math.sin(math.pi * u) + 0.008 * H * math.sin(2 * math.pi * 2.1 * t + ph0)
            out.append((x, y, size, 2 * math.pi * fq * t + ph0, 0.25 * dirn * math.cos(math.pi * u)))
    return out


def _bolt(img, idx, lflash, P, W, H, lv):
    """Forked lightning, seeded by its event."""
    rng = np.random.default_rng(9100 + idx)
    lay = np.zeros((H, W), f32)
    side = rng.random()
    x = W * (rng.uniform(0.03, 0.22) if side < 0.5 else rng.uniform(0.86, 0.98))
    y = -5.0
    yend = H * rng.uniform(0.50, 0.60)
    pts = [(x, y)]
    steps = 14
    for _ in range(steps):
        y += (yend + 5) / steps
        x += rng.normal(0, W * 0.016)
        pts.append((x, y))
    th = max(1, int(round(H / 360)))
    cv2.polylines(lay, [np.int32(np.round(pts))], False, 1.0, th * 2, cv2.LINE_AA)
    for _ in range(int(rng.integers(2, 4))):
        k0 = int(rng.integers(2, steps - 3))
        bx, by = pts[k0]
        br = [(bx, by)]
        dirn = rng.choice([-1, 1])
        for _k in range(int(rng.integers(3, 6))):
            bx += dirn * abs(rng.normal(W * 0.02, W * 0.01))
            by += H * rng.uniform(0.015, 0.035)
            br.append((bx, by))
        cv2.polylines(lay, [np.int32(np.round(br))], False, 0.7, th, cv2.LINE_AA)
    q = cv2.resize(lay, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    q = cv2.GaussianBlur(q, (0, 0), 3.0)
    glow = cv2.resize(q, (W, H), interpolation=cv2.INTER_LINEAR)
    tot = (lay * 230.0 + glow * 700.0) * min(1.0, lflash * 1.3)
    img += tot[..., None] * (_c(P["bolt"]) / 255.0)


# ---------------------------------------------------------------- draw
def draw(R, img, i, a, lv) -> None:
    T = _timeline(R, a, lv)
    G = _geo(R, lv)
    P = G["P"]
    W, H = R.W, R.H
    fps = R.fps
    t = i / float(fps)
    hot = bool(T["hot"][i])
    hotf = float(T["hot_f"][i])
    big = bool(T["big"][i])
    bigf = float(T["big_f"][i])
    E2 = float(T["E2"][i])
    pre = float(T["pre"][i])
    heat = float(T["heat"][i])
    spark = float(T["spark"][i])
    black = bool(T["black"][i])
    power = 0.0 if black else 1.0
    kage = int(T["kage"][i])
    kenv = 0.72 ** kage if kage < 40 else 0.0
    ea = int(T["entry_age"][i])
    wind = float(T["wind"][i])
    wph = float(T["wph"][i])
    fogx = float(T["fogx"][i])
    redness = float(T["redness"][i])
    la = int(T["lage"][i])
    lstr = float(T["strike"][int(T["lidx"][i])]) if la < 10 else 0.0
    flick = (1.0, 0.35, 0.95, 0.55, 0.30, 0.18, 0.10, 0.06, 0.03, 0.01)
    lflash = lstr * flick[la] if la < 10 else 0.0
    if not R.style.strobe:
        lflash *= 0.45
    pmax = R.scene_cache.get("pmax")
    if pmax is None:
        pmax = (float(R.lyric_zoom.max()) - 1.0) if R.lyric_zoom is not None else 0.0
        R.scene_cache["pmax"] = pmax
    punch = 0.0
    if R.lyric_zoom is not None and pmax > 0:
        punch = max(0.0, (float(R.lyric_zoom[i]) - 1.0) / pmax)
    # the house's focal point (door) is what the camera pushes in on
    PW, PH = G["PW"], G["PH"]
    s0 = 1.0 / PLATE_K
    cpl = (PW / 2.0, PH / 2.0)
    cfr = (W / 2.0, H / 2.0)
    hm, dr = G["anc"]["house_mid"], G["anc"]["door"]
    door = (hm[0], 0.4 * hm[1] + 0.6 * dr[1])
    foc = ((door[0] - cpl[0]) * s0 + cfr[0], (door[1] - cpl[1]) * s0 + cfr[1])
    # colours of the light this frame
    moon_c = _lerp(P["moon"], P["moon_red"], min(1.0, redness * 1.1))
    moon_c = _lerp(moon_c, P["blood"], bigf * 0.9)
    lightc = _lerp(P["moonlight"], moon_c * 0.75, 0.15 + 0.6 * redness) / 255.0
    moon_on = 0.18 if black else 1.0
    fog_hot = _c(P["fog_hot"][int(T["seg_id"][i]) % len(P["fog_hot"])])
    if big:
        fog_hot = _lerp(fog_hot, P["blood"], 0.55)
    fog_c = _lerp(P["fog"], fog_hot * (0.72 + 0.18 * kenv), min(1.0, 0.85 * hotf + 0.2 * pre))
    fog_c = fog_c * (0.45 if black else 1.0)
    fog_back = fog_c + _c(P["bolt"]) * (0.45 * lflash)     # lit from behind
    fog_c = fog_c + _c(P["bolt"]) * (0.08 * lflash)        # in front: stays dark

    # ---------------- SKY (quarter res): gradient, moon halo, lightning flash
    zm, oxm, oym = _cam(T, i, K_MOON, foc, W, H)
    mx = (MOON_C[0] * W - foc[0]) * zm + foc[0] + oxm
    my = (MOON_C[1] * H - foc[1]) * zm + foc[1] + oym
    msz = float(T["msize"][i]) * (1.0 + 0.025 * kenv * (0.4 + 0.6 * hotf))
    mr = MOON_R * H * zm * msz
    qh, qw = G["qh"], G["qw"]
    dq = np.hypot(G["qxx"] - mx / 4.0, G["qyy"] - my / 4.0) / (mr / 4.0)
    halo = (np.exp(-np.maximum(dq - 1.0, 0) * 2.4) * 0.30
            + np.exp(-np.maximum(dq - 1.0, 0) * 0.7) * 0.22) * (dq > 0.9)
    # lightning flashes are added ON TOP of the sky; the moon + its halo give
    # way so the strike frame stays under the bloom's hue-blotching range
    halo_amt = moon_on * (0.85 + 0.35 * redness + 0.25 * kenv * hotf) * (1 - 0.6 * min(1.0, lflash))
    skyq = G["sky"] * (1.0 - 0.6 * (1 - moon_on)) \
        + halo[..., None] * moon_c * halo_amt \
        + _c(P["bolt"]) * (0.30 * lflash)
    img[:] = cv2.resize(skyq.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR)
    # ---------------- STARS: twinkle, sparkle on the hats
    st = G["stars"]
    tw = 0.55 + 0.45 * np.sin(t * (1.5 + 3.0 * st[:, 2]) + 6.28 * st[:, 3])
    tw = tw + spark * 0.8 * (lv._hash01(st[:, 2] * 97.0, float(i // 2)) > 0.6)
    bright = st[:, 4] * tw * (0.35 if black else 1.0) * (1.0 - 0.6 * lflash)
    scol = _c(P["star"])
    for (sx, sy, _r, _p, _b), bv in zip(st, bright):
        if bv < 0.08:
            continue
        x, y = int(sx * W), int(sy * H)
        rad = 1 if _b < 0.8 else 2
        cv2.circle(img, (x, y), max(1, int(rad * H / 1080)), (scol * min(1.0, bv)).tolist(), -1,
                   cv2.LINE_AA)
    # ---------------- MOON (+ the emblem pulsing on the kick)
    D = G["moon_D"]
    dd = max(4, int(round(2 * mr * D / (D - 2))))
    x0, y0 = int(round(mx - dd / 2)), int(round(my - dd / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + dd), min(H, y0 + dd)
    if xb > xa and yb > ya:
        sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        mp_ = cv2.resize(G["moon_pack"], (dd, dd), interpolation=cv2.INTER_LINEAR)[sl]
        tex, disc = mp_[..., 0], mp_[..., 1]
        # kept UNDER the bloom threshold (190): the per-channel bloom turns a
        # textured surface sitting near it into green/salmon blotches
        mcol = moon_c * moon_on * (0.86 + 0.10 * hotf) * (1.0 + 0.06 * kenv * hotf) \
            * (1.0 - 0.4 * min(1.0, lflash))
        reg = img[ya:yb, xa:xb]
        mrgb = tex[..., None] * mcol
        if G["em_line"] is not None:
            eg = (0.10 + 0.55 * kenv * (0.35 + 0.65 * hotf) + 0.25 * bigf
                  + 0.15 * pre) * moon_on
            ecol = _lerp(P["emblem"], P["emblem_big"], bigf)
            ln, fl, rd, gw = mp_[..., 2], mp_[..., 3], mp_[..., 4], mp_[..., 5]
            # engraved in the calm (darker lines, a hint of the plume), lit
            # from within on the kick: the lines and plume glow
            mrgb = mrgb * (1.0 - (0.38 - 0.30 * min(1.0, eg)) * ln[..., None]
                           - 0.10 * rd[..., None]) \
                + fl[..., None] * mcol * 0.05 \
                + (ln + 0.8 * rd)[..., None] * ecol * (0.85 * eg) \
                + gw[..., None] * ecol * (0.55 * eg)
        reg *= (1.0 - disc[..., None])
        reg += mrgb * disc[..., None]
    # ---------------- CLOUDS drifting across the moon, lit by it
    cl = G["cloud"]
    off = int(fogx * 0.6 * qh) % G["cw"]
    cols = (off + np.arange(qw)) % G["cw"]
    q0, q1 = G["cloud_rows"]
    # the drop's wind tears the clouds off the moon
    ca = np.clip((cl[q0:q1, cols] - 0.57) * 3.0, 0, 1) * G["cloud_prof"][q0:q1] \
        * (0.55 * (1.0 - 0.6 * hotf))
    ccol = _c(P["cloud"])[None, None] * (0.6 if black else 1.0) \
        + (halo[q0:q1] * 2.2)[..., None] * moon_c * halo_amt * 0.22 \
        + _c(P["bolt"]) * (0.45 * lflash)
    r0, r1 = int(round(q0 * H / qh)), int(round(q1 * H / qh))
    A = cv2.resize(ca.astype(f32), (W, r1 - r0), interpolation=cv2.INTER_LINEAR)[..., None]
    Cc = cv2.resize((ccol * ca[..., None]).astype(f32), (W, r1 - r0),
                    interpolation=cv2.INTER_LINEAR)
    sub = img[r0:r1]
    sub *= (1.0 - A)
    sub += Cc
    # ---------------- lightning bolt (behind everything on the ground)
    if la < 7 and lstr >= 0.5:
        _bolt(img, int(T["lidx"][i]), lflash, P, W, H, lv)
    # ---------------- SWARM across the moon (far bats)
    sw = _swarm_bats(T, G, i, fps, W, H, mx, my, mr)
    if sw:
        _bats_over(img, sw, _c(P["bat"]))
    # ---------------- FAR RIDGE (half res, parallax 0.4) + horizon fog
    zr, oxr, oyr = _cam(T, i, K_RIDGE, foc, W, H)
    RW, RH = G["RW"], G["RH"]
    rc = _c(P["ridge"]) * (0.55 if black else 1.0) * (1 + 0.6 * hotf * heat) \
        + _c(P["bolt"]) * 0.10 * lflash
    rimc = moon_c * (0.35 * moon_on) + _c(P["bolt"]) * (1.2 * lflash)
    M = _plate_M(zr, oxr, oyr, foc, H / float(RH), (RW / 2.0, RH / 2.0), cfr,
                 G["r_y0"])
    _layer(img, G["r_stack"], M,
           [np.stack([rc, rimc, np.zeros(3, f32), np.zeros(3, f32)], axis=1).astype(f32)])
    fog_amt = (0.55 + 0.35 * E2 + 0.45 * hotf + 0.25 * bigf) * (0.6 if black else 1.0)
    _fog(img, G, T, i, 0, fog_amt * 0.9, fog_back * 1.25, fogx * 0.8)

    # ---------------- HOUSE LAYER
    zh, oxh, oyh = _cam(T, i, K_HOUSE, foc, W, H)
    kI = moon_on * (1.0 - 0.9 * min(1.0, lflash))
    rimI = moon_on * (0.9 + 0.5 * kenv * hotf + 0.4 * redness) + 2.6 * lflash
    C = np.stack([lightc * 255.0 * 1.15 * kI,                   # key
                  _c(P["fill"]) * 0.70 * (0.5 + 0.5 * moon_on) * (1.0 - 0.8 * min(1.0, lflash)),
                  np.zeros(3, f32),                              # spill (below)
                  moon_c * 0.0 + _c(P["rim"]) * 0.75 * rimI + moon_c * 0.25 * rimI
                  ]).astype(f32)
    # WINDOWS = the equaliser: each column is a band, floors light up from
    # the bottom like a meter; candles flicker in the calm
    nwin = len(G["w_col"])
    Lc = T["colL"][i]
    eqg = 1.05 * hotf + 0.70 * pre ** 1.2 + 0.35 * E2 * (1 - hotf)
    meter = np.clip(Lc[G["w_col"]] * eqg, 0, 1) * G["w_nlev"]
    eqlit = np.clip(meter - G["w_lvl"], 0, 1)
    wid = np.arange(nwin)
    slot = np.floor(t * 0.10 + _hash(wid * 3.7, lv))
    inhab = (lv._hash01(wid * 1.37, slot) < 0.62).astype(f32)
    cand = np.array([0.26 + 0.14 * _vn(t * 6.0 + 11.0 * k_, 4.4, lv)
                     + 0.06 * _vn(t * 17.0 + 5.0 * k_, 8.8, lv) for k_ in range(nwin)], f32)
    lvl_w = np.maximum(inhab * cand * (1.0 - 0.4 * hotf), eqlit * (0.55 + 0.45 * hotf))
    lvl_w += 0.30 * kenv * hotf + 0.25 * bigf
    sp = 0.85 ** (ea - 3) if 3 <= ea < 28 else 0.0
    lvl_w += 0.6 * sp
    pos = G["w_lvl"] / np.maximum(1, G["w_nlev"] - 1)
    mcols = np.stack([_ramp(P["meter"], p_ * 0.5 + 0.5 * eqlit[j] * pos[j])
                      for j, p_ in enumerate(pos)])
    wc = _c(P["candle"])[None] * (1 - hotf) + mcols * hotf
    lut = np.zeros((nwin + 1, 3), f32)
    # the power comes back a few frames AFTER the strike (so the strike
    # frame is a black silhouette), flickering on like an old circuit
    wpow = power * ((0.0, 0.0, 0.0, 1.0, 0.15, 1.0, 0.4, 1.0)[ea] if ea < 8 else 1.0)
    lut[1:] = wc * (lvl_w * wpow)[:, None] * (1.20 + 0.45 * hotf)    # blazing in drops
    # figures passing lit windows (on snares)
    passers = []
    ps = T["ps_f"]
    if len(ps) and power > 0:
        lo_ = np.searchsorted(ps, i - int(1.2 * fps), side="left")
        hi_ = np.searchsorted(ps, i, side="right")
        for e in range(lo_, hi_):
            f0 = int(ps[e])
            u = (i - f0) / (1.1 * fps)
            if not 0 <= u < 1:
                continue
            j = int(lv._hash01(f0, 2.2) * nwin) % nwin
            lut[j + 1] = np.maximum(lut[j + 1], wc[j] * 0.85 * 1.25)
            passers.append((j, u, 1 if lv._hash01(f0, 4.4) < 0.5 else -1))
    spill_c = (lut[1:].mean(axis=0) * 1.6).astype(f32)
    C[2] = spill_c
    by0, by1, bx0, bx1 = G["w_box"]
    glass = G["w_glass"][..., None] * lut[G["w_ids"]]
    for (j, u, dirn) in passers:
        y0b, y1b, x0b, x1b = G["w_bbox"][j]
        ww_, wh_ = x1b - x0b, y1b - y0b
        fx = x0b + ww_ * (0.5 + dirn * (-0.75 + 1.5 * _ease_io(u)))
        fig = np.zeros(glass.shape[:2], f32)
        hy = y0b + wh_ * 0.36 + 0.03 * wh_ * math.sin(u * 9.0)
        cv2.circle(fig, (int(fx), int(hy)), max(2, int(0.17 * ww_)), 1.0, -1, cv2.LINE_AA)
        cv2.ellipse(fig, (int(fx), int(hy + 0.42 * wh_)), (max(3, int(0.42 * ww_)), max(3, int(0.32 * wh_))),
                    0, 180, 360, 1.0, -1, cv2.LINE_AA)
        cv2.rectangle(fig, (int(fx - 0.42 * ww_), int(hy + 0.42 * wh_)),
                      (int(fx + 0.42 * ww_), y1b + 2), 1.0, -1)
        glass *= (1.0 - 0.92 * fig[..., None])
    M = _plate_M(zh, oxh, oyh, foc, s0, cpl, cfr, G["h_y0"])
    _layer(img, G["h_stack"], M, [np.ascontiguousarray(C.T)])
    _add_sprite(img, glass.astype(f32), M, bx0, by0)

    def house_pt(name):
        return _to_frame(G["anc"][name], zh, oxh, oyh, foc, s0, cpl, cfr)

    # ---------------- BAT FLOCKS (from the house / chimneys / tree)
    ztr, oxt, oyt = _cam(T, i, K_TREE, foc, W, H)
    tree_o = ((TREE_ANCHOR[0] * W - foc[0]) * ztr + foc[0] + oxt,
              (TREE_ANCHOR[1] * H - foc[1]) * ztr + foc[1] + oyt)
    origins = dict(tower=house_pt("tower_top"), chim1=house_pt("chimney1"),
                   chim2=house_pt("chimney2"), attic=house_pt("attic"), tree=tree_o)
    far, near = _flock_bats(T, G, i, fps, W, H, origins, lv)
    batc = _c(P["bat"])
    if far:
        _bats_over(img, far, batc)
    # ---------------- GROUND FOG over the hill + GHOST WISPS
    _fog(img, G, T, i, 1, fog_amt * 0.85, fog_c, fogx * 1.3)
    _ghosts(img, G, T, i, t, P, power, hotf, spark, wph, W, H, lv)
    # ---------------- THE BIG DEAD TREE (sways with the wind)
    _tree(img, G, T, i, t, P, wind, wph, ea, ztr, oxt, oyt, foc, moon_c, moon_on,
          lflash, W, H)
    # ---------------- GRAVEYARD LAYER (stones + jack-o'-lanterns)
    # the graveyard zooms about the frame's centre line (not the house), so
    # the outer lanterns spread symmetrically instead of off the right edge
    ffoc = (W / 2.0, foc[1])
    zf, oxf, oyf = _cam(T, i, K_FG, ffoc, W, H)
    oyf = _fg_lift(zf, oyf, ffoc, H)
    skin = _c(P["skin"]) / 255.0
    stem = _c(P["stem"]) / 255.0
    stone = _c(P["stone"]) / 255.0
    Lk = lightc * 255.0 * 1.10 * kI
    Lf = _c(P["fill"]) * 0.95 * (0.5 + 0.5 * moon_on) * (1.0 - 0.7 * min(1.0, lflash))
    Cf = np.stack([Lk * stone, Lk * skin * 0.85, Lk * stem,
                   Lf * stone, Lf * skin * 1.0, Lf * stem,
                   _c(P["rim"]) * 0.30 * rimI + moon_c * 0.12 * rimI]).astype(f32)
    M = _plate_M(zf, oxf, oyf, ffoc, s0, cpl, cfr, G["f_y0"])
    _layer(img, G["f_stack"], M,
           [np.ascontiguousarray(Cf[0:4].T),
            np.ascontiguousarray(np.vstack([Cf[4:7], np.zeros((1, 3), f32)]).T)])
    glow_c = _lerp(P["glow"], P["title"], 0.35 * hotf)   # hotter flame in drops
    if G["pumps"]:
        pb = G["p_box"]
        lu = np.zeros((pb[1] - pb[0], pb[3] - pb[2]), f32)
    for k_, pm in enumerate(G["pumps"]):
        sd = pm["seed"]
        g = (0.62 + 0.50 * heat) * (0.84 + 0.32 * _vn(t * 7.0 + sd, sd, lv))
        g += spark * 0.45 * (float(lv._hash01(i, sd)) - 0.35)
        g += 0.25 * kenv * hotf + 0.2 * bigf + 0.4 * sp
        g = max(0.0, g) * power
        if g <= 0.0:
            continue
        lit = pm["lit"]
        y0p, y1p, x0p, x1p = pm["box"]
        if punch > 0.02 and pm["mbox"] is not None:
            # the mouth SINGS: it flares and its glow drops open
            lit = lit * (1.0 + 2.2 * punch * pm["mouth"])
            my0, my1, mx0, mx1 = pm["mbox"]
            patch = lit[my0:my1, mx0:mx1]
            hh = my1 - my0
            nh = int(round(hh * (1.0 + 0.55 * punch)))
            st_ = cv2.resize(patch, (mx1 - mx0, nh), interpolation=cv2.INTER_LINEAR)
            st_ *= np.linspace(1.0, 0.35, nh, dtype=f32)[:, None]    # jaw fades out
            ye = min(lit.shape[0], my0 + nh)
            lit[my0:ye, mx0:mx1] = np.maximum(lit[my0:ye, mx0:mx1], st_[:ye - my0])
        lu[y0p - pb[0]:y1p - pb[0], x0p - pb[2]:x1p - pb[2]] += lit * g
    if G["pumps"]:
        _add_sprite(img, lu, M, pb[2], pb[0], glow_c * 1.15)
    # ---------------- floor fog, close bats
    _fog(img, G, T, i, 2, fog_amt * 0.40, fog_c * 0.8, fogx * 2.0)
    if near:
        _bats_over(img, near, batc)
    # ---------------- HIT A BUMP burns in at every drop entry
    _title(img, G, T, i, P, W, H, fps)


def _hash(x, lv):
    return lv._hash01(np.asarray(x, np.float64), 1.7)


def _ghosts(img, G, T, i, t, P, power, hotf, spark, wph, W, H, lv):
    if power <= 0:
        return
    gm, gh = G["ghost"], G["ghost_halo"]
    col = _c(P["ghost"])
    specs = ((0.12, 0.70, 0.95, 0.050, 1), (0.47, 0.50, 0.75, 0.037, -1),
             (0.71, 0.74, 1.10, 0.060, -1), (0.33, 0.79, 0.80, 0.043, 1),
             (0.90, 0.66, 0.70, 0.031, 1))
    for k, (x0, y0, sc, spd, d) in enumerate(specs):
        ph = k * 1.7
        x = ((x0 + d * spd * (t + 2.0 * wph * 0.25)) % 1.3) - 0.15
        edge = min(1.0, (x + 0.15) / 0.2, (1.15 - x) / 0.2)
        if edge <= 0:
            continue
        y = y0 + 0.022 * math.sin(2 * math.pi * 0.11 * t + ph)
        gsz = int(0.16 * H * sc)
        gw_ = max(4, int(gsz * gm.shape[1] / gm.shape[0]))
        a_ = (0.20 + 0.22 * hotf + 0.18 * spark * float(lv._hash01(i, k)))
        a_ *= edge * (0.75 + 0.25 * math.sin(2 * math.pi * 0.23 * t + ph))
        spr = cv2.resize(gm, (gw_, gsz), interpolation=cv2.INTER_AREA)
        # the hem ripples (sine shear, stronger toward the tail)
        rows = np.arange(gsz, dtype=f32)
        shift = (0.08 * gw_ * (rows / gsz) ** 2
                 * np.sin(rows / gsz * 7.0 - t * 3.0 * (1 + hotf) + ph)).astype(f32)
        mx_ = (np.arange(gw_, dtype=f32)[None, :] - shift[:, None] * d).astype(f32)
        my_ = np.repeat(rows[:, None], gw_, axis=1).astype(f32)
        spr = cv2.remap(spr, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        hal = cv2.resize(gh, (int(gw_ * gh.shape[1] / gm.shape[1]),
                              int(gsz * gh.shape[0] / gm.shape[0])), interpolation=cv2.INTER_AREA)
        for layer, amt in ((hal, 0.55), (spr, 1.0)):
            lh, lw = layer.shape
            cx, cy = int(x * W), int(y * H)
            xa, ya = cx - lw // 2, cy - lh // 2
            x0c, y0c = max(0, xa), max(0, ya)
            x1c, y1c = min(W, xa + lw), min(H, ya + lh)
            if x1c <= x0c or y1c <= y0c:
                continue
            m = layer[y0c - ya:y1c - ya, x0c - xa:x1c - xa]
            img[y0c:y1c, x0c:x1c] += m[..., None] * col * (a_ * amt)


def _tree(img, G, T, i, t, P, wind, wph, ea, z, ox, oy, foc, moon_c, moon_on,
          lflash, W, H):
    segs = G["tree"]
    # sway: two incommensurate sines on the wind phase (so it never loops
    # visibly) + a shove on every drop entry that settles (follow-through)
    sw = (0.010 + 0.022 * wind) * (0.65 * math.sin(2 * math.pi * 0.29 * wph)
                                   + 0.35 * math.sin(2 * math.pi * 0.71 * wph + 1.3))
    if ea < 45:
        sw += 0.05 * math.exp(-ea / 10.0) * math.cos(ea * 0.45)
    flex = (0.25, 0.55, 1.0, 1.6, 2.3)
    root = (TREE_BASE[0] * W / H, TREE_BASE[1])
    fk = _fk(segs, root, sw, flex)
    s = H * z
    xs = (fk[:, [0, 2]] * H - foc[0]) * z + foc[0] + ox
    ys = (fk[:, [1, 3]] * H - foc[1]) * z + foc[1] + oy
    x0, x1 = int(max(0, xs.min() - 60)), int(min(W, xs.max() + 60))
    y0, y1 = int(max(0, ys.min() - 60)), int(min(H, ys.max() + 60))
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), f32)
    mt = np.zeros_like(m)                    # thick limbs only: they get the rim
    for j in range(len(fk)):
        thf = 0.5 * (fk[j, 4] + fk[j, 5]) * s
        th = max(1, int(round(thf)))
        p0 = (int(xs[j, 0] - x0), int(ys[j, 0] - y0))
        p1 = (int(xs[j, 1] - x0), int(ys[j, 1] - y0))
        cv2.line(m, p0, p1, 1.0, th, cv2.LINE_AA)
        if thf >= 3.5:
            cv2.line(mt, p0, p1, 1.0, th, cv2.LINE_AA)
    # root flare
    bx, by = xs[0, 0] - x0, ys[0, 0] - y0
    w0 = fk[0, 4] * s
    flare = [np.int32([(bx - 2.2 * w0, by + 0.6 * w0), (bx - 0.4 * w0, by - 1.5 * w0),
                       (bx + 0.5 * w0, by - 1.5 * w0), (bx + 2.4 * w0, by + 0.6 * w0)])]
    cv2.fillPoly(m, flare, 1.0, cv2.LINE_AA)
    cv2.fillPoly(mt, flare, 1.0, cv2.LINE_AA)
    r = max(1, int(round(1.6 * H / 1080)))
    shf = np.roll(mt, (r, -r), axis=(0, 1))
    rim = np.clip(mt - shf, 0, 1)
    sub = img[y0:y1, x0:x1]
    sub *= (1.0 - m[..., None])
    sub += m[..., None] * _c(P["tree"])
    sub += rim[..., None] * (moon_c * 0.30 * moon_on + _c(P["bolt"]) * 1.4 * lflash)


def _title(img, G, T, i, P, W, H, fps):
    ea = int(T["entry_age"][i])
    if ea >= 10 ** 5:
        return
    e0 = i - ea
    big = bool(T["big"][min(len(T["big"]) - 1, e0)])
    dur = (2.4 if big else 1.6) * fps
    if ea >= dur:
        return
    u = ea / dur
    tm = G["t_mask"]
    th, tw = tm.shape
    # flare in (scale ease-out), hold, then burn away from the bottom up
    sc = 1.0 + 0.22 * (1.0 - min(1.0, ea / 6.0)) ** 3
    nz = G["t_noise"]
    roll = int(ea * th * 0.035) % th
    flame = nz[(np.arange(th) + roll) % (2 * th)]
    burn_thr = max(0.0, (u - 0.55) / 0.45) * 1.15
    bn = G["t_burn"] - burn_thr
    keep = np.clip(bn * 8.0, 0, 1)
    front = np.clip(1.0 - np.abs(bn) * 14.0, 0, 1) * (burn_thr > 0)
    core = tm * keep
    fade_in = 1.0
    lum = core * (0.70 + 0.45 * flame) * fade_in
    rgb = lum[..., None] * _c(P["title"]) \
        + (G["t_glow"] * keep * 0.9 * fade_in)[..., None] * _c(P["title_edge"]) \
        + (front * tm)[..., None] * _c(P["title_edge"]) * 1.4
    scrim = G["t_scrim"] * fade_in * (1.0 - 0.7 * max(0.0, (u - 0.6) / 0.4))
    char = G["t_char"] * keep
    nw, nh = int(tw * sc), int(th * sc)
    rgb = cv2.resize(rgb.astype(f32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    scrim = cv2.resize(scrim.astype(f32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    char = cv2.resize(char.astype(f32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    cx, cy = W // 2, int(H * 0.665)      # over the empty hill, under the house
    xa, ya = cx - nw // 2, cy - nh // 2
    x0, y0 = max(0, xa), max(0, ya)
    x1, y1 = min(W, xa + nw), min(H, ya + nh)
    if x1 <= x0 or y1 <= y0:
        return
    sl = (slice(y0 - ya, y1 - ya), slice(x0 - xa, x1 - xa))
    sub = img[y0:y1, x0:x1]
    sub *= (1.0 - 0.62 * np.clip(scrim[sl], 0, 1))[..., None]
    sub *= (1.0 - 0.85 * np.clip(char[sl], 0, 1))[..., None]
    sub += _c(P["char"]) * char[sl][..., None]
    sub += rgb[sl]


# ---------------------------------------------------------------- post-FX
def fx(R, img, i, a, lv):
    """Strobe on the budgeted hits only (and capped — no near-white frames),
    a zoom slam + shake on drop entries, chromatic split in drops, a push on
    sung 'bump's."""
    T = _timeline(R, a, lv)
    W, H = R.W, R.H
    hotf = float(T["hot_f"][i])
    kage = int(T["kage"][i])
    kenv = 0.72 ** kage if kage < 40 else 0.0
    sa = int(T["strobe_age"][i])
    if R.style.strobe and sa < 4:
        img += _c(geo_col(R, lv, "bolt")) * (0.10 * 0.45 ** sa)
    if hotf > 0.05:
        off = int(round((1.0 + 2.5 * kenv) * hotf * W / 1920))
        if off > 0:
            img[..., 0] = np.roll(img[..., 0], off, axis=1)
            img[..., 2] = np.roll(img[..., 2], -off, axis=1)
    z = 1.0
    pmax = R.scene_cache.get("pmax", 0.0)
    if R.lyric_zoom is not None and pmax > 0:
        z += 0.035 * (float(R.lyric_zoom[i]) - 1.0) / pmax
    z += 0.010 * kenv * hotf
    ea = int(T["entry_age"][i])
    if ea < 14:
        z += 0.07 * (1.0 - ea / 14.0) ** 3            # the slam, eased out
    dx = dy = 0.0
    sh_a = int(T["sh_age"][i])
    if sh_a < 14:
        amp = float(T["shake"][int(T["sh_idx"][i])]) * 0.014 * H * (0.74 ** sh_a)
        th = 6.2832 * float(lv._hash01(i, 13.0))
        dx, dy = amp * math.cos(th), amp * math.sin(th)
    if z > 1.0005 or abs(dx) + abs(dy) > 0.3:
        M = np.array([[z, 0, (1 - z) * W / 2 + dx],
                      [0, z, (1 - z) * H / 2 + dy]], np.float32)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return img


def geo_col(R, lv, name):
    import sys
    return lv.scene_palette(R, sys.modules[__name__])[name]
