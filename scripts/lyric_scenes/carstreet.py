"""carstreet — NIGHT DRIVE: a chase-cam behind a low sports car cruising a
long straight city street (Zeke 2026-10-10, scene for "Hit a Bump").

Street level, in motion, one-point perspective — deliberately the opposite
of the dystopia's rooftop skyline. Everything is a true pinhole projection
of a tiny world (metres; camera on the car's centre-line, Y up, Z forward):

  road        per-pixel ground plane: lane dashes LOCKED TO THE BEAT GRID
              (the distance travelled is an integer number of dash periods
              at every beat, so dashes cross on the beat), motion-blurred
              analytically; streetlight pools; storefront spill; wet
              reflections (a per-column mirror of the bright layer, smeared
              vertically) + explicit lamp reflection streaks.
  facades     per-pixel planes at |X| = 11 m (storefronts, window grids,
              one EQ column per building driven by the spectrum) and a
              taller back plane at |X| = 34 m for skyline depth.
  objects     streetlights (a pair every 4 dashes, passing the car ON the
              beat), blade signs (invented words), skybridges carrying the
              Tzeke000 logo, the "HIT A BUMP" gantry at every drop entry,
              speed humps + BUMP warning signs for every sung "bump", and
              the big-drop TUNNEL (light rings, one per beat).
  car         Blender-rendered sprites (scripts/lyric_scenes/blender/
              carstreet_car.py -> assets/scenes/carstreet/), relit per frame
              from separate lighting passes; body and wheels are separate
              layers so the suspension and the hop can overlap.

What drives what (rule 3 — every element has a named cause):
  kick        tail-light flare + anamorphic streak, suspension dip, tunnel
              ring flash, gantry text pulse
  snare       backfire pops from the exhaust (drops), blade-sign flicker
  sub bass    underglow (car + road), horizon glow, logo screen heat
  hi-hat      heat shimmer behind the exhaust, window twinkle, rain glints
  spectrum    EQ window columns on every building, sign + storefront levels
  build pre   speed climbs, camera drops low + FOV narrows (anticipation),
              lamps smear into streaks, gantry warning lights blink
  black       the half-beat before each drop: streetlights + city cut out,
              only the car's own lights remain
  drop entry  NITRO: the car surges away from the camera, FOV punches wide,
              speed lines, shake, "HIT A BUMP" ignites on the gantry, the
              underglow switches colour
  big drop    into the tunnel (its portal announces it through the build):
              light rings passing on every beat, flashing on the kick
  sung bump   a striped speed hump (BUMP warning signs ahead of each run)
              the car hits ON the word: compress, launch, stretch, land
              wheels-first with chassis sparks, body squash, settle; the
              camera rides over the same hump a few frames later (a real
              camera-height bounce = follow-through) + a small push-in
  bar ONE     (in drops) a light camera shake

STATELESS: draw(i) reads only arrays precomputed over the whole song (cached
in R.scene_cache) and seeded geometry — chunk-parallel safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "scenes" / "carstreet"
_THIS = sys.modules[__name__]

# --------------------------------------------------------------------------
# palettes (first = default)
# --------------------------------------------------------------------------
PALETTES: dict[str, dict] = {
    # sodium-orange streetlights + cyan / teal neon
    "midnight": dict(
        sky_top=(2, 4, 10), sky_hor=(34, 26, 34), glow=(120, 70, 40),
        fog=(30, 24, 28), asphalt=(13, 13, 16), walk=(30, 28, 30),
        facade=(15, 16, 21), facade2=(10, 12, 18), mark=(150, 150, 140),
        lamp=(255, 150, 55),
        neon=[(0, 230, 255), (0, 255, 190), (255, 60, 150), (255, 190, 60)],
        win=[(255, 185, 105), (190, 215, 255), (255, 145, 70), (110, 235, 225)],
        under=(0, 215, 255), tail=(255, 22, 30), flame=(80, 160, 255),
        rings=[(0, 230, 255), (255, 150, 55), (0, 255, 190), (255, 60, 150)],
        gantry=(0, 235, 255), warn=(255, 40, 30), hump=(255, 200, 30),
        paint=(0.62, 0.70, 0.82), wet=0.55, rain=0.0, sun=False),
    # outrun: magenta / violet / teal, a striped sun at the vanishing point
    "synth": dict(
        sky_top=(10, 2, 26), sky_hor=(110, 20, 90), glow=(255, 60, 150),
        fog=(54, 14, 58), asphalt=(13, 9, 22), walk=(32, 22, 44),
        facade=(18, 12, 30), facade2=(14, 8, 28), mark=(150, 120, 170),
        lamp=(255, 70, 205),
        neon=[(160, 70, 255), (0, 240, 220), (255, 50, 180), (255, 140, 240)],
        win=[(255, 120, 220), (150, 110, 255), (80, 240, 230), (255, 190, 240)],
        under=(255, 40, 200), tail=(255, 30, 90), flame=(120, 220, 255),
        rings=[(255, 40, 190), (0, 240, 220), (150, 70, 255), (255, 140, 60)],
        gantry=(255, 60, 200), warn=(255, 40, 120), hump=(0, 240, 220),
        paint=(0.82, 0.62, 1.0), wet=0.62, rain=0.0, sun=True),
    # cold white / blue, heavier wet reflections, light rain
    "rain": dict(
        sky_top=(4, 6, 10), sky_hor=(38, 48, 62), glow=(90, 120, 160),
        fog=(40, 50, 64), asphalt=(10, 12, 16), walk=(28, 32, 38),
        facade=(14, 17, 22), facade2=(11, 14, 20), mark=(150, 160, 170),
        lamp=(215, 232, 255),
        neon=[(70, 150, 255), (150, 230, 255), (255, 60, 70), (200, 210, 255)],
        win=[(205, 222, 255), (150, 190, 255), (255, 215, 180), (120, 200, 255)],
        under=(60, 150, 255), tail=(255, 28, 40), flame=(120, 190, 255),
        rings=[(150, 220, 255), (70, 140, 255), (230, 240, 255), (255, 60, 70)],
        gantry=(120, 200, 255), warn=(255, 40, 40), hump=(255, 210, 60),
        paint=(0.72, 0.78, 0.86), wet=0.95, rain=1.0, sun=False),
}

STYLE = dict(palette=[(0, 230, 255), (255, 60, 150), (255, 150, 55),
                      (255, 22, 30)],
             font_title="Anton-Regular.ttf", strobe=True)
BLOOM = 0.36

# --------------------------------------------------------------------------
# world constants (metres)
# --------------------------------------------------------------------------
F_FRAC, HY_FRAC = 0.75, 0.40      # focal (x W) and horizon row (x H)
L_DASH, DASH_ON = 12.0, 4.0       # dash period / painted length
LAMP_EVERY = 4                    # dashes between streetlight pairs
XF, XB = 11.0, 34.0               # front / back facade planes
XT, YT = 7.6, 7.2                 # tunnel half-width / ceiling
LAMP_X, LAMP_Y, POLE_X = 5.4, 8.8, 7.7
LANES = (1.75, 5.25)              # dashed lane lines (|X|)
EDGE_X, CURB_X = 6.85, 7.15
Z_REAR0 = 4.2                     # camera -> car rear (m) at rest
HALF_LEN = 2.25
HC0 = 1.85                        # camera height in the calm
FOG_D = 260.0
SPEED_IN = {"calm": 1, "pre0": 1, "pre1": 2, "drop": 4, "big": 6}

SIGN_WORDS = ("BAR", "HOTEL", "NOODLE", "DINER", "MOTEL", "TATTOO", "ARCADE",
              "KARAOKE", "PAWN", "GARAGE", "PARTS", "TIRES", "CAFE", "OPEN",
              "24H", "NITRO", "CLUB", "LIQUOR", "RAMEN", "VIDEO", "DRIVE IN",
              "REPAIR", "CHROME", "NEON", "BOWL", "DANCE", "FUEL", "LOANS")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _ihash(*xs) -> np.ndarray:
    """Integer hash of int arrays -> float32 in [0, 1). Vectorised."""
    h = np.uint32(2166136261)
    out = None
    for x in xs:
        v = np.asarray(x).astype(np.int64).astype(np.uint32)
        out = (v * np.uint32(0x9E3779B1)) if out is None else \
            ((out ^ v) * np.uint32(0x85EBCA6B))
        out ^= out >> np.uint32(13)
    out = (out ^ h) * np.uint32(0xC2B2AE35)
    out ^= out >> np.uint32(16)
    return ((out & np.uint32(0xFFFFFF)).astype(np.float32) / float(1 << 24))


def _dash_cov(z, period, on, blur):
    """Coverage of a periodic painted interval [0, on) at position z,
    box-filtered over [z, z + blur] (motion blur + footprint)."""
    def F(x):
        q = np.floor(x / period)
        return q * on + np.minimum(x - q * period, on)
    blur = np.maximum(blur, 1e-3)
    return np.clip((F(z + blur) - F(z)) / blur, 0.0, 1.0)


def _env(age, tau):
    return np.where(age < 10 ** 5, np.exp(-np.asarray(age, np.float32) / tau),
                    0.0).astype(np.float32)


def _col(c) -> np.ndarray:
    return np.asarray(c, np.float32)


def _text_mask(R, txt: str, font: str, px_h: int, pad: int = 6) -> np.ndarray:
    """Text as a float mask whose ink is ~px_h tall."""
    from PIL import Image, ImageDraw
    f = R.font(220, font)
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    bb = probe.textbbox((0, 0), txt, font=f)
    im = Image.new("L", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), txt, font=f, fill=255)
    arr = np.asarray(im, np.float32) / 255.0
    sc = px_h / max(1, bb[3] - bb[1])
    tw, th = max(1, int(arr.shape[1] * sc)), max(1, int(arr.shape[0] * sc))
    return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)


def _fit(m: np.ndarray, w: int, h: int, cw: int = 0, ch: int = 0) -> np.ndarray:
    """Scale mask m to fit inside w x h (aspect kept), centred on a cw x ch
    canvas (default: w x h)."""
    sc = min(w / m.shape[1], h / m.shape[0])
    tw, th = max(1, int(m.shape[1] * sc)), max(1, int(m.shape[0] * sc))
    r = cv2.resize(m, (tw, th), interpolation=cv2.INTER_AREA)
    w, h = (cw or w), (ch or h)
    out = np.zeros((h, w), np.float32)
    y0, x0 = (h - th) // 2, (w - tw) // 2
    out[y0:y0 + th, x0:x0 + tw] = r
    return out


def _logo_layers(R, size: int):
    """(rgb HxWx3 0..1, alpha) of the Tzeke000 symbol, white bg keyed out."""
    if R.logo is None:
        return None
    rgb = np.asarray(R.logo.convert("RGB"), np.float32) / 255.0
    al = np.asarray(R.logo.split()[-1], np.float32) / 255.0
    if al.std() < 0.02:
        lum = rgb.mean(axis=2)
        sat = rgb.max(axis=2) - rgb.min(axis=2)
        al = 1.0 - ((lum > 0.93) & (sat < 0.10)).astype(np.float32)
        al = cv2.GaussianBlur(al, (0, 0), 1.0)
    sc = size / max(rgb.shape[:2])
    w, h = max(1, int(rgb.shape[1] * sc)), max(1, int(rgb.shape[0] * sc))
    return (cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA),
            cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA))


class Tex:
    """A textured upright rectangle: premultiplied base rgb, alpha and an
    emissive mask (coloured + scaled per frame), with a small mip chain so a
    far sign is resized from a near-size level."""

    def __init__(self, rgb, a, emis=None, halo=0.0):
        self.levels = []
        e = emis if emis is not None else np.zeros_like(a)
        if halo > 0:                       # a soft glow around the tubes
            e = np.maximum(e, 0.6 * cv2.GaussianBlur(e, (0, 0), halo))
        rgb, a, e = rgb.astype(np.float32), a.astype(np.float32), \
            e.astype(np.float32)
        while True:
            self.levels.append((rgb, a, e))
            if min(a.shape) < 24:
                break
            h, w = a.shape
            rgb = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            a = cv2.resize(a, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            e = cv2.resize(e, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    def get(self, w: int, h: int):
        lv = self.levels[0]
        for L in self.levels:
            if L[1].shape[1] >= w and L[1].shape[0] >= h:
                lv = L
        rgb, a, e = lv
        it = cv2.INTER_AREA if w < a.shape[1] else cv2.INTER_LINEAR
        return (cv2.resize(rgb, (w, h), interpolation=it),
                cv2.resize(a, (w, h), interpolation=it),
                cv2.resize(e, (w, h), interpolation=it))


def _blit(img, cov, tex: Tex, x0f, y0f, x1f, y1f, ecol, egain=1.0,
          dim=1.0) -> None:
    """Composite an upright textured rect into img (and its alpha into cov)."""
    H, W = img.shape[:2]
    w, h = int(round(x1f - x0f)), int(round(y1f - y0f))
    if w < 2 or h < 2 or x1f < 0 or y1f < 0 or x0f > W or y0f > H:
        return
    if w > 6 * W or h > 6 * H:
        return
    x0, y0 = int(round(x0f)), int(round(y0f))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    rgb, a, e = tex.get(w, h)
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    rgb, a, e = rgb[sl], a[sl], e[sl]
    reg = img[ya:yb, xa:xb]
    reg *= (1.0 - a)[..., None]
    reg += rgb * dim
    if egain > 0:
        reg += e[..., None] * (_col(ecol) * egain)
    if cov is not None:
        np.maximum(cov[ya:yb, xa:xb], a, out=cov[ya:yb, xa:xb])


# --------------------------------------------------------------------------
# the song timeline: speed, camera, events (whole song, cached)
# --------------------------------------------------------------------------
def _timeline(R, a, lv) -> dict:
    fps = int(R.fps)
    T = dict(lv.stage_timeline(a, fps, "acid", label="carstreet"))
    n = len(a.rms)
    hot, big, pre = T["hot"], T["big"], T["pre"]
    gb = np.flatnonzero(T["gbeat"])
    per = float(np.median(np.diff(gb))) if len(gb) > 2 else fps * 0.4
    entries = list(T["entries"])
    big_entry = [e for e in entries if big[e]]
    # ---- 1. per-beat dash steps (integers -> dashes cross ON the beat)
    steps, N = [], [0]
    for k, f in enumerate(gb):
        mid = min(n - 1, int(f + per / 2))
        if big[mid]:
            m = SPEED_IN["big"]
        elif hot[mid]:
            m = SPEED_IN["drop"]
        elif pre[mid] > 0:
            m = SPEED_IN["pre1"] if pre[mid] >= 0.5 else SPEED_IN["pre0"]
        else:
            m = SPEED_IN["calm"]
        cur = N[-1]
        prev = steps[-1] if steps else 1
        if m == 4:            # lamps every 4 dashes pass ON the beat
            r = cur % 4
            s = 4 - r if 4 - r >= 3 else 8 - r
        elif m == 2:
            s = 2 if cur % 2 == 0 else (1 if prev <= 1 else 3)
        else:
            s = m
        steps.append(s)
        N.append(cur + s)
    steps = np.asarray(steps, np.float64)
    N = np.asarray(N[:-1], np.float64)
    # tangents (dashes per beat): mean of neighbours, clamped for monotony;
    # at a drop entry the tangent keeps the OLD speed so the surge happens
    # AFTER the hit (nitro), not in the blackout before it
    tan = np.empty_like(steps)
    tan[0] = steps[0]
    for k in range(1, len(steps)):
        t_ = 0.5 * (steps[k - 1] + steps[k])
        tan[k] = min(t_, 3 * steps[k - 1], 3 * steps[k])
    ent_set = set(entries)
    for k, f in enumerate(gb):
        if k > 0 and any(abs(int(f) - e) <= 1 for e in ent_set):
            tan[k] = steps[k - 1]
    D = np.zeros(n, np.float64)
    for k in range(len(gb)):
        f0 = int(gb[k])
        f1 = int(gb[k + 1]) if k + 1 < len(gb) else n
        if f1 <= f0:
            continue
        t1 = tan[k + 1] if k + 1 < len(gb) else steps[k]
        n1 = N[k] + steps[k]
        L_ = (gb[k + 1] - gb[k]) if k + 1 < len(gb) else per
        p = (np.arange(f0, f1) - f0) / float(L_)
        h00 = 2 * p ** 3 - 3 * p ** 2 + 1
        h10 = p ** 3 - 2 * p ** 2 + p
        h01 = -2 * p ** 3 + 3 * p ** 2
        h11 = p ** 3 - p ** 2
        D[f0:f1] = h00 * N[k] + h10 * tan[k] + h01 * n1 + h11 * t1
    if len(gb) and gb[0] > 0:
        D[:gb[0]] = N[0] - (gb[0] - np.arange(gb[0])) / per
    Dm = D * L_DASH
    v = np.gradient(Dm).astype(np.float32)            # metres per frame
    # ---- 2. event ages + envelopes
    kage, sage = T["kage"], T["sage"]
    hot_f = lv._gauss_smooth(hot.astype(np.float32), fps, 0.12)
    heat = np.clip(lv._gauss_smooth(a.bass, fps, 0.12) * 1.15, 0, 1)
    hs = lv._gauss_smooth(a.high, fps, 0.35)
    spark = np.clip((np.asarray(a.high, np.float32) - hs) * 5.0, 0, 1)
    ent_flag = np.zeros(n, bool)
    ent_flag[entries] = True
    eage, eidx = lv._age_since(ent_flag)
    nitro = _env(eage, 0.55 * fps) * (eage < 3 * fps)
    # ---- 3. camera
    #   height: calm 1.85, drop 1.62, big 1.45; the build drops it to ~1.3
    #   (anticipation) and after the hit it recovers with a little overshoot
    base_h = np.where(big, 1.45, np.where(hot, 1.62, HC0)).astype(np.float32)
    base_h = lv._gauss_smooth(base_h, fps, 0.35)
    low = (pre ** 1.4).astype(np.float32)
    low2 = low.copy()
    for i in range(1, n):                              # decaying carry
        low2[i] = max(low[i], low2[i - 1] * 0.86)
    hc = base_h - 0.50 * low2
    #   focal: the build narrows it (tension), the entry punches it WIDE
    #   with a small overshoot; drops sit wider than the calm
    base_f = np.where(big, 0.86, np.where(hot, 0.92, 1.0)).astype(np.float32)
    base_f = lv._gauss_smooth(base_f, fps, 0.30)
    ea = np.minimum(eage, 10 ** 5).astype(np.float32)
    punch = np.where(eage < 3 * fps,
                     0.17 * np.exp(-ea / 5.0) * np.cos(ea * 0.16)
                     - 0.0, 0.0).astype(np.float32)
    fovm = base_f + 0.11 * pre ** 2 - punch
    #   the car pulls away on NITRO and the camera catches up (follow-through)
    surge = np.where(eage < 3 * fps,
                     (1 - np.exp(-ea / 2.5)) * np.exp(-ea / (0.9 * fps)),
                     0.0).astype(np.float32)
    Zc = (Z_REAR0 - 0.35 * pre + 2.6 * surge
          + 0.5 * lv._gauss_smooth(big.astype(np.float32), fps, 0.5))
    #   lateral drift in the calm, locked in the drops; camera lags (EMA)
    tt = np.arange(n) / float(fps)
    calm_k = 1.0 - 0.75 * hot_f
    Xcar = (0.30 * np.sin(2 * np.pi * tt / 9.5) + 0.12 * np.sin(
        2 * np.pi * tt / 3.7 + 1.3)) * calm_k
    Xcam = np.zeros(n, np.float32)
    al = 1 - np.exp(-1.0 / (0.55 * fps))
    for i in range(1, n):
        Xcam[i] = Xcam[i - 1] + al * (Xcar[i] - Xcam[i - 1])
    # ---- 4. suspension (kick) + the BUMP hop (sung word)
    ka = np.minimum(kage, 10 ** 5).astype(np.float32)
    susp = np.where(kage < 20, -np.exp(-ka / 3.0) * np.cos(ka * 0.75),
                    0.0).astype(np.float32) * (0.020 + 0.022 * hot_f)
    lift = np.zeros(n, np.float32)       # whole car (wheels) off the road
    brel = np.zeros(n, np.float32)       # body relative to wheels
    sqx = np.ones(n, np.float32)
    sqy = np.ones(n, np.float32)
    bumps = []
    lz = getattr(R, "lyric_zoom", None)
    if lz is not None:
        lz = np.asarray(lz, np.float32)
        pk = [f for f in range(1, n - 1) if lz[f] > 1.01 and lz[f] >= lz[f - 1]
              and lz[f] > lz[f + 1]]
        bumps = pk
    land_frames, cam_hits = [], []
    for fb in bumps:
        # 0-1 compress (anticipation the hump forces), 2..9 airborne
        # (stretch going up), 10 land wheels-first, body squash, settle
        seq_len = 22
        for k in range(seq_len):
            f = fb + k
            if f >= n:
                break
            if k <= 1:
                br, li, sx, sy = -0.07 * (k + 1) / 2, 0.0, 1.0 + 0.012 * k, \
                    1.0 - 0.02 * (k + 1)
            elif k <= 9:
                p = (k - 2) / 7.0
                li = 0.68 * 4 * p * (1 - p)
                br = 0.05 * (1 - p)                  # body leads, wheels hang
                st = 0.035 * (1 - 2 * p)              # stretch up, ease out
                sx, sy = 1 - 0.4 * max(st, 0), 1 + max(st, 0)
            else:
                q = k - 10
                li = 0.0
                br = -0.14 * np.exp(-q / 3.2) * np.cos(q * 0.7)
                sq = 0.07 * np.exp(-q / 2.5) * np.cos(q * 0.7)
                sx, sy = 1 + 0.5 * sq, 1 - sq
            lift[f] = max(lift[f], li)
            brel[f] = br if abs(br) > abs(brel[f]) else brel[f]
            sqx[f], sqy[f] = sx, sy
        land_frames.append(fb + 10)
    # the hump on the road: under the REAR wheels at the word
    humps = []
    for fb in bumps:
        zb = Dm[fb] + Zc[fb] + 0.9
        humps.append(zb)
        ch = int(np.searchsorted(Dm, zb))          # camera reaches it
        if ch < n:
            cam_hits.append(ch)
    # the camera rides over the same hump a few frames later: a real bounce
    # of the camera height (follow-through), not just a 2-D shake
    cam_jolt = np.zeros(n, np.float32)
    for f in cam_hits:
        k = np.arange(0, min(18, n - f))
        cam_jolt[f:f + len(k)] += (0.22 * np.exp(-k / 4.0)
                                   * np.sin(k * 0.55 + 0.3)).astype(np.float32)
    hc = hc + cam_jolt
    # ---- 5. shake (frames since event, amplitude decays)
    shake_ev = []
    for e in entries:
        shake_ev.append((e, 1.0, 0))
    ones = T["gbeat"] & (T["bi"] % 4 == 0)
    for f in np.flatnonzero(ones & hot):
        shake_ev.append((int(f), 0.42 if big[f] else 0.25, 0))
    for f in land_frames:
        shake_ev.append((f, 0.55, 1))
    for f in cam_hits:
        shake_ev.append((f, 0.75, 1))
    sh_amp = np.zeros(n, np.float32)
    sh_vert = np.zeros(n, np.float32)
    for f, A, vert in sorted(shake_ev):
        if f >= n:
            continue
        k = np.arange(0, min(16, n - f))
        val = A * 0.72 ** k
        seg = sh_amp[f:f + len(k)]
        upd = val > seg
        seg[upd] = val[upd]
        if vert:
            sh_vert[f:f + len(k)][upd] = 1.0
    # ---- 6. world layout: lamps, tunnel, gantries, bridges, signs
    zmax = float(Dm[-1]) + 1500.0
    Zcar_c = Z_REAR0 + HALF_LEN
    lamp_z = np.arange(-2, int(zmax / (LAMP_EVERY * L_DASH)) + 2) \
        * LAMP_EVERY * L_DASH + Zcar_c
    tunnel = None
    big_segs = [(s0, s1) for s0, s1, h in T["segs"] if h and big[s0]]
    if big_entry and big_segs:
        e0, e1 = big_segs[-1]
        zm = float(Dm[e0] + Zc[e0])
        ze = float(Dm[min(n - 1, e1)] + 3.0)
        tunnel = (zm, ze, e0, e1)
        lamp_z = lamp_z[(lamp_z < zm - 25) | (lamp_z > ze + 30)]
    gantries = []
    for e in entries:
        if big[e]:
            continue
        gantries.append(dict(z=float(Dm[e]) + 58.0, e=int(e)))
    busy = [g["z"] for g in gantries] + list(humps)
    if tunnel:
        busy += [tunnel[0], tunnel[1]]
    bridges = []
    BR = 52 * L_DASH
    zb_ = 340.0
    while zb_ < zmax:
        ok = all(abs(zb_ - z) > 140 for z in busy)
        if tunnel and tunnel[0] - 150 < zb_ < tunnel[1] + 60:
            ok = False
        if ok:
            bridges.append(dict(z=zb_, k=len(bridges)))
        zb_ += BR
    rs = np.random.default_rng(907)
    blades = []
    for side in (-1, 1):
        z = 25.0 + rs.uniform(0, 30)
        while z < zmax:
            if not (tunnel and tunnel[0] - 30 < z < tunnel[1] + 20):
                blades.append(dict(z=z, side=side,
                                   w=int(rs.integers(len(SIGN_WORDS))),
                                   c=int(rs.integers(4)),
                                   band=int(rs.integers(2, 30)),
                                   flick=bool(rs.random() < 0.35),
                                   y0=float(rs.uniform(4.3, 5.4))))
            z += rs.uniform(32, 70)
    blades.sort(key=lambda b: b["z"])
    # one pair of BUMP signs per run of humps (bump, bump, bump = one sign)
    bump_signs = [dict(z=zb - 36.0, zb=zb) for k, zb in enumerate(humps)
                  if k == 0 or zb - humps[k - 1] > 60.0]
    bs = np.asarray(a.bars, np.float32)
    bars = bs.copy()
    bars[1:] = 0.5 * (bs[1:] + bs[:-1])
    # underglow colour: the palette's own in the calm + the big drop, a new
    # neon at every other drop ENTRY (a colour change that lands on the drop)
    ugsel = np.full(n, -1, np.int32)
    k_ = 0
    for s0, s1, h in T["segs"]:
        if h and not big[s0]:
            ugsel[s0:s1] = (2, 1, 3, 0)[k_ % 4]
            k_ += 1
    dv = np.gradient(lv._gauss_smooth(v, fps, 0.15))
    brake = np.clip(-dv * 6.0, 0, 1).astype(np.float32)
    print(f"[carstreet] {len(gb)} beats, {Dm[-1] / 1000:.2f} km driven, "
          f"{len(lamp_z)} lamp pairs, {len(gantries)} gantries, "
          f"{len(bridges)} skybridges, {len(bumps)} bumps, tunnel="
          f"{'%.0f-%.0fm' % tunnel[:2] if tunnel else None}")
    T.update(n=n, Dm=Dm, v=v, hc=hc.astype(np.float32),
             fovm=fovm.astype(np.float32), Zc=Zc.astype(np.float32),
             Xcar=Xcar.astype(np.float32), Xcam=Xcam, susp=susp, lift=lift,
             brel=brel, sqx=sqx, sqy=sqy, bumps=bumps, humps=humps,
             land_frames=land_frames,
             heat=heat, spark=spark, hot_f=hot_f, eage=eage, nitro=nitro,
             sh_amp=sh_amp, sh_vert=sh_vert, lamp_z=lamp_z, tunnel=tunnel,
             gantries=gantries, bridges=bridges, blades=blades,
             bump_signs=bump_signs, bars=bars, brake=brake, per=per,
             surge=surge, ugsel=ugsel)
    return T


# --------------------------------------------------------------------------
# geometry + textures for this render size (cached)
# --------------------------------------------------------------------------
def _buildings(seed: int, zmax: float, back: bool) -> dict:
    rng = np.random.default_rng(seed)
    starts, ends, hs, kinds = [], [], [], []
    z = -80.0
    while z < zmax:
        if not back and rng.random() < 0.07:          # a cross street
            w = rng.uniform(12, 18)
            starts.append(z)
            ends.append(z + w)
            hs.append(0.0)
            kinds.append(1)
            z += w
            continue
        w = rng.uniform(24, 60) if back else rng.uniform(8, 24)
        h = rng.uniform(40, 150) if back else rng.uniform(9, 46)
        starts.append(z)
        ends.append(z + w)
        hs.append(h)
        kinds.append(0)
        z += w + (rng.uniform(0, 2.5) if back else 0.0)
    n = len(starts)
    B = dict(start=np.asarray(starts), end=np.asarray(ends),
             h=np.asarray(hs, np.float32), gap=np.asarray(kinds) == 1,
             fh=rng.uniform(3.0, 3.8, n).astype(np.float32),
             ws=rng.uniform(1.7, 3.0, n).astype(np.float32),
             ww=rng.uniform(0.40, 0.72, n).astype(np.float32),
             lit=rng.uniform(0.10, 0.45, n).astype(np.float32),
             wc=rng.integers(0, 4, n), sc=rng.integers(0, 4, n),
             tone=rng.uniform(0.7, 1.35, n).astype(np.float32),
             eq=rng.random(n) < (0.25 if back else 0.45),
             eqc=rng.integers(1, 6, n), band=rng.integers(1, 30, n),
             trim=rng.random(n) < (0.18 if back else 0.28),
             shop=rng.uniform(0.6, 1.3, n).astype(np.float32),
             key=rng.integers(1, 2 ** 30, n))
    return B


def _geo(R, T, P) -> dict:
    W, H = R.W, R.H
    f32 = np.float32
    zmax = float(T["Dm"][-1]) + 1500.0
    G = dict(W=W, H=H, f0=F_FRAC * W, hy=HY_FRAC * H, cx=W / 2.0,
             ppm=H / 1080.0)
    # sky (full res, static) — gradient + a warm haze band at the horizon
    ys = np.linspace(0.0, 1.0, H, dtype=f32)[:, None]
    hyf = HY_FRAC
    g = np.clip(ys / hyf, 0, 1) ** 2.4
    sky = (_col(P["sky_top"]) * (1 - g)[..., None] + _col(P["sky_hor"]) * g[..., None])
    band = np.exp(-((ys - hyf) / 0.055) ** 2)[..., None]
    G["sky_half"] = np.ascontiguousarray(np.broadcast_to(
        sky[::2][: H // 2], (H // 2, W // 2, 3))).astype(f32)
    G["skyband_half"] = np.ascontiguousarray(band[::2][: H // 2]).astype(f32)
    # buildings: [side][plane]
    G["bld"] = {s: dict(front=_buildings(101 + (s > 0) * 17, zmax, False),
                        back=_buildings(303 + (s > 0) * 29, zmax, True))
                for s in (-1, 1)}
    # textures ------------------------------------------------------------
    ppm = 64.0 * G["ppm"]                  # texture px per metre at level 0
    tex = {}
    # blade signs: vertical stacked letters on a dark panel, neon border
    for wi, word in enumerate(SIGN_WORDS):
        letters = [c for c in word if c != " "]
        wm = 1.5
        hm = 0.95 * len(letters) + 0.7
        tw, th = int(wm * ppm), int(hm * ppm)
        em = np.zeros((th, tw), f32)
        for k, ch in enumerate(letters):
            m = _text_mask(R, ch, "Anton-Regular.ttf", int(0.72 * ppm))
            m = _fit(m, int(0.95 * tw * 0.8), int(0.80 * ppm))
            y0 = int((0.35 + 0.95 * k) * ppm)
            x0 = (tw - m.shape[1]) // 2
            em[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = np.maximum(
                em[y0:y0 + m.shape[0], x0:x0 + m.shape[1]], m)
        t = max(2, int(0.06 * ppm))
        cv2.rectangle(em, (t, t), (tw - 1 - t, th - 1 - t), 0.85, t)
        al = np.ones((th, tw), f32) * 0.94
        rgb = np.full((th, tw, 3), 5.0, f32) * al[..., None]
        tex[("blade", wi)] = Tex(rgb, al, em, halo=2.0 * G["ppm"])
    # BUMP warning sign (diamond, retro-reflective) on a post
    s_ = int(1.6 * ppm)
    dm = np.zeros((s_, s_), f32)
    pts = np.array([[s_ // 2, 2], [s_ - 3, s_ // 2], [s_ // 2, s_ - 3],
                    [2, s_ // 2]], np.int32)
    cv2.fillPoly(dm, [pts], 1.0, lineType=cv2.LINE_AA)
    txt = _fit(_text_mask(R, "BUMP", "Anton-Regular.ttf", 200), int(s_ * 0.56),
               int(s_ * 0.26))
    ink = np.zeros_like(dm)
    y0 = (s_ - txt.shape[0]) // 2
    x0 = (s_ - txt.shape[1]) // 2
    ink[y0:y0 + txt.shape[0], x0:x0 + txt.shape[1]] = txt
    face = np.clip(dm - ink, 0, 1)
    tex["bumpsign"] = Tex(np.zeros((s_, s_, 3), f32), dm, face * 0.9)
    # overhead gantry panel: "HIT A BUMP" (13 x 3 m)
    gw, gh = int(13 * ppm), int(3.0 * ppm)
    title = (R.title or "HIT A BUMP").upper()
    gm = _fit(_text_mask(R, title, "Anton-Regular.ttf", 220),
              int(gw * 0.90), int(gh * 0.70), gw, gh)
    frame = np.zeros((gh, gw), f32)
    t = max(2, int(0.07 * ppm))
    cv2.rectangle(frame, (t, t), (gw - 1 - t, gh - 1 - t), 1.0, t)
    tex["gantry_txt"] = Tex(np.full((gh, gw, 3), 6.0, f32), np.ones((gh, gw), f32),
                            gm, halo=3.0 * G["ppm"])
    tex["gantry_frame"] = Tex(np.zeros((gh, gw, 3), f32), np.zeros((gh, gw), f32),
                              frame, halo=2.0 * G["ppm"])
    # skybridge panel: logo + TZEKE000 (10 x 3.4 m)
    bw, bh = int(10 * ppm), int(3.4 * ppm)
    brgb = np.zeros((bh, bw, 3), f32)
    bal = np.ones((bh, bw), f32)
    bem = np.zeros((bh, bw), f32)
    brgb[:] = 8.0
    lg = _logo_layers(R, int(bh * 0.86))
    if lg is not None:
        lrgb, la = lg
        y0 = (bh - la.shape[0]) // 2
        x0 = int(bw * 0.06)
        sl = (slice(y0, y0 + la.shape[0]), slice(x0, x0 + la.shape[1]))
        brgb[sl] = brgb[sl] * (1 - la[..., None]) + lrgb * 255.0 * la[..., None]
        logo_x1 = x0 + la.shape[1]
    else:
        logo_x1 = int(bw * 0.06)
    tz = _fit(_text_mask(R, "TZEKE000", "Anton-Regular.ttf", 200),
              int(bw - logo_x1 - bw * 0.08), int(bh * 0.5))
    y0 = (bh - tz.shape[0]) // 2
    x0 = logo_x1 + int(bw * 0.04)
    bem[y0:y0 + tz.shape[0], x0:x0 + tz.shape[1]] = tz
    cv2.rectangle(bem, (t, t), (bw - 1 - t, bh - 1 - t), 0.7, t)
    tex["bridge"] = Tex(brgb, bal, bem, halo=2.5 * G["ppm"])
    # tunnel portal sign: HIT A BUMP (24 x 4.5 m)
    pw, ph = int(24 * ppm * 0.5), int(4.5 * ppm * 0.5)
    pm = _fit(_text_mask(R, title, "Anton-Regular.ttf", 220), int(pw * 0.92),
              int(ph * 0.82), pw, ph)
    tex["portal"] = Tex(np.full((ph, pw, 3), 4.0, f32), np.ones((ph, pw), f32),
                        pm, halo=2.0 * G["ppm"])
    # vanishing-point landmark: a tower with a giant logo screen
    Hl = int(0.40 * H)
    Wl = int(Hl * 0.42)
    lrgb = np.zeros((Hl, Wl, 3), f32)
    lal = np.zeros((Hl, Wl), f32)
    lem = np.zeros((Hl, Wl), f32)
    body = np.array([[int(Wl * 0.18), Hl], [int(Wl * 0.18), int(Hl * 0.30)],
                     [int(Wl * 0.30), int(Hl * 0.22)], [int(Wl * 0.50), int(Hl * 0.04)],
                     [int(Wl * 0.70), int(Hl * 0.22)], [int(Wl * 0.82), int(Hl * 0.30)],
                     [int(Wl * 0.82), Hl]], np.int32)
    cv2.fillPoly(lal, [body], 1.0, lineType=cv2.LINE_AA)
    lrgb[:] = _col(P["facade2"]) * 0.8
    lrgb *= lal[..., None]
    rr = np.random.default_rng(5)
    for _ in range(260):
        x = int(rr.uniform(0.22, 0.78) * Wl)
        y = int(rr.uniform(0.62, 0.99) * Hl)
        if lal[y, x] > 0.5:
            lem[y:y + 2, x:x + 2] = rr.uniform(0.2, 0.7)
    cv2.line(lem, (Wl // 2, int(Hl * 0.0)), (Wl // 2, int(Hl * 0.05)), 1.0, 2)
    scr = (int(Wl * 0.12), int(Hl * 0.26), int(Wl * 0.88), int(Hl * 0.58))
    G["lm_screen"] = scr
    lscr = np.zeros((scr[3] - scr[1], scr[2] - scr[0], 3), f32)
    lsa = np.zeros(lscr.shape[:2], f32)
    lg = _logo_layers(R, int(min(lscr.shape[:2]) * 0.9))
    if lg is not None:
        lr, la = lg
        y0 = (lscr.shape[0] - la.shape[0]) // 2
        x0 = (lscr.shape[1] - la.shape[1]) // 2
        lscr[y0:y0 + la.shape[0], x0:x0 + la.shape[1]] = lr * la[..., None]
        lsa[y0:y0 + la.shape[0], x0:x0 + la.shape[1]] = la
    G["lm"] = (lrgb, lal, lem)
    G["lm_logo"] = (lscr, lsa)
    # outrun sun (synth only)
    if P.get("sun"):
        S = int(0.36 * H)
        yy, xx = np.mgrid[0:S, 0:S].astype(f32)
        r = np.hypot(xx - S / 2, yy - S / 2) / (S / 2)
        disc = np.clip((1 - r) * S * 0.25, 0, 1)
        t_ = yy / S
        stripes = ((np.sin(t_ * 60 * (0.3 + t_)) > (1.6 * t_ - 0.75))
                   | (t_ < 0.5)).astype(f32)
        grad = (_col((255, 230, 90)) * (1 - t_[..., None])
                + _col((255, 40, 150)) * t_[..., None])
        G["sun"] = (grad * (disc * stripes)[..., None]).astype(f32)
    # ground row/col coordinates (full res and quarter res)
    G["yrow"] = np.arange(H, dtype=f32)[:, None]
    G["xcol"] = np.arange(W, dtype=f32)[None, :]
    qh, qw = max(8, H // 4), max(8, W // 4)
    G["q"] = (qh, qw)
    # rain drops (screen space, seeded)
    rr = np.random.default_rng(77)
    nd = int(900 * (W * H) / (1920 * 1080)) + 200
    G["rain"] = np.stack([rr.random(nd), rr.random(nd), rr.uniform(0.7, 1.4, nd),
                          rr.uniform(0.4, 1.0, nd)], 1).astype(f32)
    # speed lines (seeded): angle, phase, speed, width
    rs = np.random.default_rng(31)
    ang = rs.uniform(0, 2 * np.pi, 140)
    keep = ~((ang > 1.15 * np.pi) & (ang < 1.85 * np.pi))   # spare the sky/top
    G["slines"] = np.stack([ang[keep], rs.random(keep.sum()),
                            rs.uniform(0.7, 1.3, keep.sum()),
                            rs.uniform(0.6, 1.4, keep.sum())], 1).astype(f32)
    # vignette-free noise for reflections' ripple (half res)
    hh, hw = H // 2, W // 2
    G["half"] = (hh, hw)
    nz = cv2.resize(np.random.default_rng(9).random((hh // 6 + 2, hw // 40 + 2))
                    .astype(f32), (hw, hh), interpolation=cv2.INTER_CUBIC)
    G["ripple"] = (nz - 0.5).astype(f32)
    G["ripple4"] = cv2.resize(G["ripple"], (hw // 2, hh // 2),
                              interpolation=cv2.INTER_AREA)
    G["car"] = _load_car(R)
    G["tex"] = tex
    return G


def _load_car(R) -> dict:
    meta_p = ASSET_DIR / "car_meta.json"
    if not meta_p.is_file():
        raise SystemExit(
            f"carstreet: car sprites missing ({meta_p}). Generate them with:\n"
            "  ~/.local/bin/blender -b --factory-startup --python-exit-code 1 "
            "--python scripts/lyric_scenes/blender/carstreet_car.py -- "
            "--out assets/scenes/carstreet")
    meta = json.loads(meta_p.read_text())
    s0 = R.W / float(meta["W"])
    out = dict(meta=meta, s0=s0, sets=[])
    for tag, hm in sorted(meta["heights"].items(), key=lambda kv: kv[1]["hc"]):
        d = dict(hc=float(hm["hc"]), x0=hm["x0"] * s0, y0=hm["y0"] * s0,
                 cy=0.5 * (hm["contactL"][1] + hm["contactR"][1]) * s0)
        for nm in ("base", "top0", "top1", "top2", "sideL", "sideR", "ids",
                   "wheels"):
            p = ASSET_DIR / f"car_{tag}_{nm}.png"
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None:
                raise SystemExit(f"carstreet: missing car sprite {p}")
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
            w = max(2, int(round(im.shape[1] * s0)))
            h = max(2, int(round(im.shape[0] * s0)))
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            a = im[..., 3:4]
            d[nm] = (im[..., :3] * a).astype(np.float32)       # premultiplied
            if nm in ("base", "wheels"):
                d[nm + "_a"] = im[..., 3].astype(np.float32)
        ids = d["ids"]
        d["m_paint"] = ids[..., 0]
        d["m_glass"] = ids[..., 1]
        d["m_tail"] = ids[..., 2]
        out["sets"].append(d)
    return out


def _cache(R, a, lv):
    c = R.scene_cache.get("carstreet")
    if c is None or c["key"] != (R.W, R.H):
        P = lv.scene_palette(R, _THIS)
        T = c["T"] if c is not None else _timeline(R, a, lv)
        c = dict(key=(R.W, R.H), T=T, P=P, G=_geo(R, T, P))
        R.scene_cache["carstreet"] = c
    return c


# --------------------------------------------------------------------------
# per-frame camera
# --------------------------------------------------------------------------
def _cam(G, T, i) -> dict:
    hc = float(T["hc"][i])
    f = G["f0"] * float(T["fovm"][i])
    return dict(hc=hc, f=f, hy=G["hy"], cx=G["cx"], Dm=float(T["Dm"][i]),
                Xc=float(T["Xcam"][i]), v=float(T["v"][i]))


def _proj(cam, X, Y, Z):
    Z = np.maximum(Z, 0.05)
    return (cam["cx"] + cam["f"] * (X - cam["Xc"]) / Z,
            cam["hy"] + cam["f"] * (cam["hc"] - Y) / Z)


# --------------------------------------------------------------------------
# facades (per pixel)
# --------------------------------------------------------------------------
def _facade_side(G, T, P, i, cam, side, plane, power, scale, levels_bars,
                 lamp_rel, lampcol, out_rgb, out_m):
    """Shade one side's facade plane into (out_rgb, out_m), a buffer on the
    grid x = (k + 0.5) / scale - 0.5 (same for y). Columns are processed in
    chunks, each only over the rows between its tallest building top and its
    lowest base line (below that it is road, above it sky)."""
    W, H = G["W"], G["H"]
    Xp = XF if plane == "front" else XB
    cx, f, hy, hc = cam["cx"], cam["f"], cam["hy"], cam["hc"]
    Xrel = side * Xp - cam["Xc"]
    mh, mw = out_m.shape
    xg = (np.arange(mw, dtype=np.float32) + 0.5) / scale - 0.5
    yg = (np.arange(mh, dtype=np.float32) + 0.5) / scale - 0.5
    cols = np.flatnonzero((xg < cx - 1.0) if side < 0 else (xg > cx + 1.0))
    if len(cols) < 2:
        return
    Z = np.maximum(f * Xrel / (xg[cols] - cx), 0.2)
    B = G["bld"][side][plane]
    b = np.clip(np.searchsorted(B["start"], cam["Dm"] + Z, side="right") - 1,
                0, len(B["start"]) - 1)
    hb = np.where(B["gap"][b], 0.0, B["h"][b])
    ytop = hy - f * (hb - hc) / Z
    ybase = hy + f * hc / Z
    CH = 48
    for k0 in range(0, len(cols), CH):
        cs = cols[k0:k0 + CH]
        sl = slice(k0, k0 + CH)
        if hb[sl].max() <= 0:
            continue
        ra = int(np.searchsorted(yg, ytop[sl][hb[sl] > 0].min() - 2))
        rb = int(np.searchsorted(yg, ybase[sl].max() + 2))
        ra, rb = max(0, ra), min(mh, rb)
        if rb <= ra:
            continue
        out = _facade_grid(G, T, P, i, cam, side, plane, power, levels_bars,
                           lamp_rel, lampcol, xg[cs], yg[ra:rb], 1.0 / scale)
        if out is None:
            continue
        rgb, inb = out
        out_rgb[ra:rb, cs[0]:cs[-1] + 1] = np.where(
            inb[..., None], rgb, out_rgb[ra:rb, cs[0]:cs[-1] + 1])
        out_m[ra:rb, cs[0]:cs[-1] + 1] = np.maximum(
            out_m[ra:rb, cs[0]:cs[-1] + 1], inb)


def _facade_grid(G, T, P, i, cam, side, plane, power, levels_bars, lamp_rel,
                 lampcol, xs, ys, step):
    """Per-pixel facade shading at full-res positions xs (columns, all on
    `side`) x ys (rows). Returns (rgb, inside-building mask) or None."""
    Xp = XF if plane == "front" else XB
    cx = cam["cx"]
    f = cam["f"]
    Xrel = side * Xp - cam["Xc"]
    Z = np.maximum(f * Xrel / (xs - cx), 0.2)
    u = cam["Dm"] + Z
    B = G["bld"][side][plane]
    b = np.clip(np.searchsorted(B["start"], u, side="right") - 1, 0,
                len(B["start"]) - 1)
    uu = u - B["start"][b]
    hb = B["h"][b]
    gap = B["gap"][b]
    k = Z / f                                          # metres per px (Y)
    Y = cam["hc"] - (ys[:, None] - cam["hy"]) * k[None, :]
    inb = (Y >= 0.0) & (Y < hb[None, :]) & (~gap)[None, :]
    if not inb.any():
        return None
    fh, ws, ww = B["fh"][b], B["ws"][b], B["ww"][b]
    du = (Z * Z) / (f * abs(Xrel)) * step             # metres per sample (u)
    dY = k * step
    detx = np.clip(1.8 - du / (0.30 * ws), 0, 1)
    dety = np.clip(1.8 - dY / (0.28 * fh), 0, 1)
    det = np.minimum(detx, dety)[None, :]
    tone = B["tone"][b]
    fac = _col(P["facade"] if plane == "front" else P["facade2"]) \
        * tone[:, None]
    # facade base with a slight upward darkening
    yr = np.clip(Y / np.maximum(hb[None, :], 1), 0, 1)
    rgb = fac[None, :, :] * (1.15 - 0.45 * yr)[..., None]
    up = Y > 4.6
    cfl = (Y - 4.6) / fh[None, :]
    j = np.floor(cfl).astype(np.int32)
    fy = cfl - j
    c = np.floor(uu / ws).astype(np.int32)
    fx = uu / ws - c
    winx = (fx > 0.5 - ww / 2) & (fx < 0.5 + ww / 2)
    win = up & winx[None, :] & (fy > 0.28) & (fy < 0.86)
    hsh = _ihash(B["key"][b][None, :], c[None, :], j)
    lit = hsh < B["lit"][b][None, :]
    # spectrum: one EQ column per (some) building, floors lit up to the level
    lvl = levels_bars[B["band"][b]]                    # 0..1 per column
    nfl = np.maximum(1.0, (hb - 4.6) / fh)
    eqcol = B["eq"][b] & ((c % 7) == B["eqc"][b] % 7)
    eqon = eqcol[None, :] & (j < (lvl * nfl * 1.15)[None, :]) & win
    wcol = np.asarray(P["win"], np.float32)[B["wc"][b]]   # per column
    neon = np.asarray(P["neon"], np.float32)
    twinkle = 0.75 + 0.25 * _ihash(hsh * 1000 + i // 5)
    winrgb = wcol[None, :, :] * (0.35 + 0.45 * hsh)[..., None] \
        * twinkle[..., None]
    win_lit = win & lit & ~eqon
    mean_win = (wcol * (B["lit"][b] * ww * 0.58 * 0.55)[:, None])[None, :, :]
    # inside a lit window: brighter at the top (ceiling light), a mullion
    # cross, and some blinds half drawn
    fxl = (fx[None, :] - (0.5 - ww[None, :] / 2)) / np.maximum(ww[None, :], 1e-3)
    fyl = (fy - 0.28) / 0.58
    cross = (np.abs(fxl - 0.5) < 0.035) | (np.abs(fyl - 0.62) < 0.04)
    blind = fyl > (0.45 + 0.6 * _ihash(hsh * 7919))
    wmod = (0.62 + 0.38 * fyl) * np.where(cross, 0.35, 1.0) \
        * np.where(blind, 0.55, 1.0)
    pat = np.where(win_lit[..., None], winrgb * (wmod * power)[..., None], 0.0)
    pat = np.where((win & ~lit & ~eqon)[..., None], fac[None] * 0.5, pat)
    eqc_rgb = neon[B["sc"][b]][None, :, :] * 0.95
    pat = np.where(eqon[..., None], eqc_rgb * power, pat)
    winpart = win | eqon
    detp = det[..., None]
    rgb = np.where(up[..., None],
                   np.where(winpart[..., None], pat, rgb) * detp
                   + (rgb * (1 - 0.58 * 0.5) + mean_win * power) * (1 - detp),
                   rgb)
    # EQ columns stay visible even far away (they are the point)
    if plane == "front":
        # STOREFRONT (ground floor): lit shop windows + a fascia neon band
        # each building: a lit shop (interior + ceiling light strip + shelf
        # silhouettes behind mullions), a closed shutter, or a dark front
        scol = np.asarray(P["win"], np.float32)[B["sc"][b]] * B["shop"][b][:, None]
        kind = _ihash(B["key"][b], 77)
        pw_ = 1.35
        pi_ = np.floor(uu / pw_).astype(np.int32)
        pane = uu / pw_ - pi_
        mull = (pane < 0.06) | (pane > 0.94)
        zone = (Y > 0.35) & (Y < 3.25)
        lit_shop = (kind < 0.62)[None, :] & zone
        shut = ((kind >= 0.62) & (kind < 0.86))[None, :] & zone
        ph_ = _ihash(B["key"][b], pi_)                       # per pane
        shelf = ((np.abs(Y - 1.05) < 0.10) | (np.abs(Y - 1.75) < 0.10)) \
            & (ph_ > 0.35)[None, :]
        ceil = np.abs(Y - 3.0) < 0.12
        grad = (0.35 + 0.65 * np.clip(Y / 3.2, 0, 1)) ** 1.5
        inter = scol[None] * (grad * (0.26 + 0.30 * ph_[None, :]))[..., None]
        inter = np.where(shelf[..., None], inter * 0.35, inter)
        inter = np.where(ceil[..., None], scol[None] * 1.15 + 30.0, inter)
        inter = np.where(mull[None, :, None], fac[None] * 0.6, inter)
        dz_ = det[..., None]
        inter = inter * dz_ + scol[None] * 0.40 * (1 - dz_)
        rgb = np.where(lit_shop[..., None], inter * power + inter * 0.08,
                       rgb)
        rib = (np.floor(Y / 0.14) % 2 == 0)
        shc = fac[None] * np.where(rib, 2.4, 1.7)[..., None]
        shc = shc * dz_ + fac[None] * 2.05 * (1 - dz_)
        rgb = np.where(shut[..., None], shc, rgb)
        rgb = np.where((Y < 0.35)[..., None], fac[None] * 0.6, rgb)
        sband = (Y > 3.5) & (Y < 4.25)
        sl = _ihash(B["key"][b], np.floor(uu / 0.45).astype(np.int32)) > 0.35
        sgl = levels_bars[(B["band"][b] + 7) % 40]
        scol2 = neon[B["sc"][b]] * (0.45 + 0.9 * sgl)[:, None]
        rgb = np.where((sband & sl[None, :])[..., None],
                       scol2[None] * power + 6.0, rgb)
        # rooftop neon trim on some buildings
        trim = B["trim"][b][None, :] & (Y > hb[None, :] - 0.35)
        rgb = np.where(trim[..., None], neon[(B["sc"][b] + 1) % 4][None]
                       * (0.8 * power + 0.1), rgb)
        # streetlamp light sweeping the lower facade (moves with the road)
        if lamp_rel is not None and len(lamp_rel):
            dz = u[:, None] - lamp_rel[None, :]
            lw = np.exp(-(dz / 9.0) ** 2).sum(axis=1) * power
            rgb += (lw[None, :] * np.exp(-((Y - 5.0) / 6.0) ** 2))[..., None] \
                * (_col(lampcol) * 0.22)
    else:
        trim = B["trim"][b][None, :] & (Y > hb[None, :] - 0.9) & (Y < hb[None, :] - 0.3)
        rgb = np.where(trim[..., None], neon[(B["sc"][b] + 2) % 4][None]
                       * (0.55 * power + 0.08), rgb)
    # fog
    fg = (1.0 - np.exp(-Z / (FOG_D * (0.75 if plane == "back" else 1.0))))
    rgb = rgb * (1 - fg)[None, :, None] + _col(P["fog"])[None, None, :] \
        * fg[None, :, None]
    return rgb.astype(np.float32), inb


# --------------------------------------------------------------------------
# tunnel (walls / ceiling / portal), half res
# --------------------------------------------------------------------------
def _tunnel_layer(G, T, P, i, cam, a, img, cov):
    tun = T["tunnel"]
    if tun is None:
        return None
    zm, ze = tun[0], tun[1]
    Dm = cam["Dm"]
    Zm, Ze = zm - Dm, ze - Dm
    if Zm > 900 or Ze < -2:
        return None
    W, H = G["W"], G["H"]
    hh, hw = G["half"]
    f, hy, cx, hc, Xc = cam["f"], cam["hy"], cam["cx"], cam["hc"], cam["Xc"]
    ys = (np.arange(hh, dtype=np.float32) * 2 + 0.5)[:, None]
    xs = (np.arange(hw, dtype=np.float32) * 2 + 0.5)[None, :]
    xp = (xs - cx) / f
    yp = (ys - hy) / f
    big = 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        Zw = np.where(xp > 1e-6, (XT - Xc) / xp,
                      np.where(xp < -1e-6, (-XT - Xc) / xp, big))
        Zcl = np.where(yp < -1e-6, (YT - hc) / (-yp), big)
        Zg = np.where(yp > 1e-6, hc / yp, big)
    Zw = np.broadcast_to(Zw, (hh, hw))
    Zcl = np.broadcast_to(Zcl, (hh, hw))
    Zg = np.broadcast_to(Zg, (hh, hw))
    Zwc = np.minimum(Zw, Zcl)
    is_wall = Zw <= Zcl
    inside = (Zwc < Zg) & (Zwc >= max(Zm, 0.05)) & (Zwc <= Ze)
    rgb = np.zeros((hh, hw, 3), np.float32)
    mask = np.zeros((hh, hw), np.float32)
    T_ = T
    bi = T["bi"]
    kage = int(T_["kage"][i])
    kenv = 0.70 ** kage if kage < 30 else 0.0
    rings = np.asarray(P["rings"], np.float32)
    RS = SPEED_IN["big"] * L_DASH                # ring spacing: one per beat
    if inside.any():
        Zh = np.where(inside, Zwc, 1.0)
        zw = Dm + Zh
        rel = zw - zm
        ring_k = np.floor(rel / RS + 0.5).astype(np.int32)
        dzr = rel - ring_k * RS
        foot = Zh * Zh / (f * 6.0) + np.abs(cam["v"])
        # box-filtered over the frame's travel: a motion-blurred ring SPREADS
        # its light instead of painting the whole wall (photosensitivity)
        ring_on = _dash_cov(rel + 0.7 - foot / 2, RS, 1.4, foot)
        bar = (int(bi[i]) // 4)
        rc = rings[(ring_k + bar) % len(rings)]
        # concrete + light falling off from the nearest ring
        lit = np.exp(-np.abs(dzr) / 11.0)[..., None] * rc * 0.34
        Y = hc - yp * Zh
        Xh = Xc + xp * Zh
        base = np.full((hh, hw, 3), 13.0, np.float32)
        # panel seams on the walls (every 3 m)
        seam = _dash_cov(zw, 3.0, 0.12, foot) * 0.6
        base *= (1.0 - seam)[..., None]
        # wall light strips (scroll with speed, motion blurred)
        strip = np.where(is_wall, np.exp(-((Y - 1.1) / 0.09) ** 2)
                         + np.exp(-((Y - (YT - 0.4)) / 0.09) ** 2), 0.0)
        strip = strip * (0.3 + 0.7 * _dash_cov(zw, 6.0, 3.0, foot + 0.1))
        ncol = _col(P["neon"][0])
        col = base + lit + strip[..., None] * ncol * 0.9
        # the rings themselves: emissive frames; the one passing the car
        # flashes on the kick
        near = np.exp(-(((ring_k * RS + zm) - (Dm + T_["Zc"][i] + 2)) / 40.0) ** 2)
        rb = 0.85 + 1.2 * kenv * near[..., None] if np.ndim(near) else 1.0
        col = col + ring_on[..., None] * rc * rb
        # thin secondary rings every 2 dashes: the tunnel's structure
        RS2 = 2 * L_DASH
        on2 = _dash_cov(rel + 0.18 - foot / 2, RS2, 0.36, foot)
        col = col + (on2 * (1 - ring_on))[..., None] * (rc * 0.22 + 10.0)
        fg = 1.0 - np.exp(-Zh / 300.0)
        col = col * (1 - fg)[..., None] + _col(P["fog"]) * 0.5 * fg[..., None]
        rgb = np.where(inside[..., None], col, rgb)
        mask = np.maximum(mask, inside.astype(np.float32))
    # portal face (camera still outside)
    if Zm > 0.3:
        Xm = Xc + xp * Zm
        Ym = hc - yp * Zm
        Xm = np.broadcast_to(Xm, (hh, hw))
        Ym = np.broadcast_to(Ym, (hh, hw))
        opening = (np.abs(Xm) < XT) & (Ym < YT) & (Ym > 0)
        Zf = np.where(np.abs(xp) > 1e-6, (np.sign(xp) * XF - Xc) / np.where(
            np.abs(xp) > 1e-6, xp, 1.0), big)
        Zf = np.where(Zf > 0, Zf, big)
        face = (~opening) & (Ym >= 0) & (Ym < 48.0) & (np.abs(Xm) < 60) \
            & (Zm < Zg) & (Zm < np.broadcast_to(Zf, (hh, hw)) + 1e-3)
        if face.any():
            fc = np.full((hh, hw, 3), 1.0, np.float32) * _col(P["facade"]) * 0.9
            # horizontal concrete bands + a neon outline around the opening
            band = (np.floor(Ym / 2.4) % 2 == 0)
            fc = np.where(band[..., None], fc * 1.25, fc)
            edge = (np.abs(np.abs(Xm) - XT) < 0.35) & (Ym < YT + 0.35) | \
                (np.abs(Ym - YT) < 0.35) & (np.abs(Xm) < XT + 0.35)
            rc0 = rings[0] * (0.9 + 0.8 * kenv)
            fc = np.where(edge[..., None], rc0, fc)
            fg = 1.0 - np.exp(-Zm / FOG_D)
            fc = fc * (1 - fg) + _col(P["fog"]) * fg
            rgb = np.where(face[..., None], fc, rgb)
            mask = np.maximum(mask, face.astype(np.float32))
    if mask.max() <= 0:
        return None
    big_rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    big_m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    img += (big_rgb - img) * big_m[..., None]
    # portal sign above the opening
    if Zm > 0.3:
        x0, y0 = _proj(cam, -12.0, YT + 6.0, Zm)
        x1, y1 = _proj(cam, 12.0, YT + 1.5, Zm)
        hotf = float(T["hot_f"][i])
        pre = float(T["pre"][i])
        blink = 1.0 if (int(T["bi"][i]) % 2 == 0 or pre < 0.05) else 0.45
        gain = (0.5 + 0.6 * pre) * blink * (0.0 if T["black"][i] else 1.0) \
            + 0.8 * hotf
        _blit(img, cov, G["tex"]["portal"], x0, y0, x1, y1, P["gantry"], gain)
    return dict(Zm=Zm, Ze=Ze)


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------
def draw(R, img, i, a, lv) -> None:
    c = _cache(R, a, lv)
    T, P, G = c["T"], c["P"], c["G"]
    W, H = G["W"], G["H"]
    f32 = np.float32
    cam = _cam(G, T, i)
    us = int(T["ugsel"][i])
    cam["ug"] = _col(P["under"] if us < 0 else P["neon"][us])
    Dm, hc, f, hy, cx = cam["Dm"], cam["hc"], cam["f"], cam["hy"], cam["cx"]
    v = cam["v"]
    black = bool(T["black"][i])
    power = 0.0 if black else 1.0
    hot = bool(T["hot"][i])
    hotf = float(T["hot_f"][i])
    big = bool(T["big"][i])
    pre = float(T["pre"][i])
    heat = float(T["heat"][i])
    spark = float(T["spark"][i])
    kage = int(T["kage"][i])
    kenv = 0.70 ** kage if kage < 30 else 0.0
    sage = int(T["sage"][i])
    senv = 0.6 ** sage if sage < 20 else 0.0
    bars = T["bars"][i]
    lampcol = _col(P["lamp"])
    tun = T["tunnel"]
    in_tun = tun is not None and tun[0] - 2 <= Dm <= tun[1]
    # in the drops the city's neon pulses on the kick
    neon_k = 1.0 + 0.55 * kenv * hotf
    # streak factor: lamps smear into streaks on the build and in drops
    streak = 1.0 + 5.0 * pre ** 1.5 + 2.5 * hotf + 2.0 * float(T["nitro"][i])
    # ---- SKY + FACADES on a half-res canvas, upsampled once --------------
    hh, hw = G["half"]
    can = G["sky_half"] + G["skyband_half"] * (
        _col(P["glow"]) * (0.35 + 0.25 * heat) * (0.25 + 0.75 * power))
    lamp_z = T["lamp_z"]
    lz_vis = lamp_z[(lamp_z > Dm - 15) & (lamp_z < Dm + 420)]
    qb_rgb = np.zeros((hh // 2, hw // 2, 3), f32)
    qb_m = np.zeros((hh // 2, hw // 2), f32)
    fr_rgb = np.zeros((hh, hw, 3), f32)
    fr_m = np.zeros((hh, hw), f32)
    for side in (-1, 1):
        _facade_side(G, T, P, i, cam, side, "back", power * 0.8, 0.25, bars,
                     None, lampcol, qb_rgb, qb_m)
        _facade_side(G, T, P, i, cam, side, "front", power * neon_k, 0.5,
                     bars, lz_vis, lampcol, fr_rgb, fr_m)
    bm = cv2.resize(qb_m, (hw, hh), interpolation=cv2.INTER_LINEAR)
    can = cv2.blendLinear(cv2.resize(qb_rgb, (hw, hh),
                                     interpolation=cv2.INTER_LINEAR),
                          can, bm, 1.0 - bm)
    can = cv2.blendLinear(fr_rgb, can, fr_m, 1.0 - fr_m)
    fmask = np.maximum(bm, fr_m)
    img[:] = cv2.resize(can, (W, H), interpolation=cv2.INTER_LINEAR)
    # ---- the vanishing point: landmark tower + logo screen (+ outrun sun),
    # drawn at full res BEHIND the facades (masked by their coverage)
    vpx = cx - f * cam["Xc"] / 4000.0
    lrgb, lal, lem = G["lm"]
    lx0 = int(vpx - lal.shape[1] / 2)
    ly0 = int(hy - lal.shape[0] + 0.02 * H)
    rx0, ry0 = max(0, lx0 - int(0.12 * H)), max(0, ly0)
    rx1 = min(W, lx0 + lal.shape[1] + int(0.12 * H))
    ry1 = min(H, int(hy + 0.05 * H))
    before = img[ry0:ry1, rx0:rx1].copy()
    if "sun" in G:
        S = G["sun"]
        _add_clip(img, S * (0.75 + 0.25 * heat), int(vpx - S.shape[1] / 2),
                  int(hy - S.shape[0] * 0.52))
    _over_clip(img, lrgb, lal, lx0, ly0)
    _add_clip(img, lem[..., None] * _col(P["win"][0]) * (0.6 * power + 0.1),
              lx0, ly0)
    scr = G["lm_screen"]
    lscr, lsa = G["lm_logo"]
    gl = (0.55 + 0.35 * heat + 0.3 * kenv * hotf) * (0.15 + 0.85 * power)
    _add_clip(img, (lscr * 255.0 * gl * 0.9).astype(f32), lx0 + scr[0],
              ly0 + scr[1])
    frame_c = _col(P["neon"][0]) * (0.6 + 0.6 * heat) * (0.2 + 0.8 * power)
    cv2.rectangle(img, (lx0 + scr[0], ly0 + scr[1]),
                  (lx0 + scr[2], ly0 + scr[3]),
                  tuple(float(q) for q in frame_c), max(1, int(2 * G["ppm"])))
    if ry1 > ry0 and rx1 > rx0:
        fm = cv2.resize(fmask, (W, H), interpolation=cv2.INTER_LINEAR)[
            ry0:ry1, rx0:rx1]
        after = img[ry0:ry1, rx0:rx1]
        img[ry0:ry1, rx0:rx1] = before + (after - before) * (1.0 - fm)[..., None]
    # ---- TUNNEL (walls, ceiling, portal) ----------------------------------
    cov = np.zeros((H, W), f32)
    tinfo = _tunnel_layer(G, T, P, i, cam, a, img, cov)
    zcull = (tun[0] if (tun is not None and Dm < tun[1]) else 1e12)
    # ---- OBJECTS far -> near (above-ground parts; poles reach the ground)
    objs = []
    for zl in lz_vis:
        objs.append((zl - Dm, "lamp", zl))
    for b in T["blades"]:
        Z = b["z"] - Dm
        if 2.0 < Z < 380:
            objs.append((Z, "blade", b))
    for g in T["gantries"]:
        Z = g["z"] - Dm
        if 0.5 < Z < 700:
            objs.append((Z, "gantry", g))
    for br in T["bridges"]:
        Z = br["z"] - Dm
        if 0.5 < Z < 900:
            objs.append((Z, "bridge", br))
    for bsn in T["bump_signs"]:
        Z = bsn["z"] - Dm
        if 1.0 < Z < 220:
            objs.append((Z, "bumpsign", bsn))
    objs = [o for o in objs if (o[0] + Dm) < zcull - 1]
    objs.sort(key=lambda o: -o[0])
    lamp_heads = []
    for Z, kind, ob in objs:
        if kind == "lamp":
            lamp_heads += _draw_lamp(img, cov, G, P, cam, Z, power, streak, T, i)
        elif kind == "blade":
            _draw_blade(img, cov, G, P, cam, Z, ob, power * neon_k, bars, senv,
                        T, i)
        elif kind == "gantry":
            _draw_gantry(img, cov, G, P, cam, Z, ob, power, kenv, T, i)
        elif kind == "bridge":
            _draw_bridge(img, cov, G, P, cam, Z, ob, power, heat, T, i)
        elif kind == "bumpsign":
            _draw_bumpsign(img, cov, G, P, cam, Z, ob, power)
    # ---- GROUND (road + sidewalks) with wet reflections -------------------
    _ground(img, cov, G, T, P, i, cam, power, lz_vis, lamp_heads, heat, kenv,
            tinfo, in_tun)
    # ---- CAR --------------------------------------------------------------
    _draw_car(img, G, T, P, i, cam, power, kenv, senv, spark, heat, lz_vis,
              in_tun)
    # ---- SPEED LINES + RAIN ----------------------------------------------
    _speed_lines(img, G, T, P, i, cam)
    if P["rain"] > 0:
        _rain(img, G, T, P, i, cam, spark)


def _add_clip(img, src, x0, y0):
    H, W = img.shape[:2]
    h, w = src.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    img[ya:yb, xa:xb] += src[ya - y0:yb - y0, xa - x0:xb - x0]


def _over_clip(img, rgb, al, x0, y0):
    H, W = img.shape[:2]
    h, w = al.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    a_ = al[ya - y0:yb - y0, xa - x0:xb - x0][..., None]
    reg = img[ya:yb, xa:xb]
    reg *= (1 - a_)
    reg += rgb[ya - y0:yb - y0, xa - x0:xb - x0]


# --------------------------------------------------------------------------
# objects
# --------------------------------------------------------------------------
def _glow_dot(img, x, y, r, col, amt):
    """Additive soft glow (gaussian) at (x, y), radius r px."""
    H, W = img.shape[:2]
    r = min(r, 0.035 * W)
    R_ = int(max(2, 2.6 * r))
    xa, xb = int(max(0, x - R_)), int(min(W, x + R_ + 1))
    ya, yb = int(max(0, y - R_)), int(min(H, y + R_ + 1))
    if xb <= xa or yb <= ya:
        return
    xx = np.arange(xa, xb, dtype=np.float32)[None, :] - x
    yy = np.arange(ya, yb, dtype=np.float32)[:, None] - y
    g = np.exp(-(xx * xx + yy * yy) / (2 * r * r)) * amt
    img[ya:yb, xa:xb] += g[..., None] * _col(col)


def _draw_lamp(img, cov, G, P, cam, Z, power, streak, T, i):
    """A pair of streetlights at depth Z. Returns head positions for the
    road reflections."""
    heads = []
    if Z < 0.6:
        return heads
    f = cam["f"]
    pole_c = (10.0, 10.0, 12.0)
    lc = _col(P["lamp"])
    for side in (-1, 1):
        xb, yb = _proj(cam, side * POLE_X, 0.0, Z)
        xt, yt = _proj(cam, side * POLE_X, LAMP_Y + 0.3, Z)
        xh, yh = _proj(cam, side * LAMP_X, LAMP_Y, Z)
        th = max(1, int(round(0.22 * f / Z)))
        cv2.line(img, (int(xb), int(yb)), (int(xt), int(yt)), pole_c, th,
                 cv2.LINE_AA)
        cv2.line(cov, (int(xb), int(yb)), (int(xt), int(yt)), 1.0, th,
                 cv2.LINE_AA)
        cv2.line(img, (int(xt), int(yt)), (int(xh), int(yh)), pole_c,
                 max(1, th // 2 + 1), cv2.LINE_AA)
        cv2.line(cov, (int(xt), int(yt)), (int(xh), int(yh)), 1.0,
                 max(1, th // 2 + 1), cv2.LINE_AA)
        if power <= 0:
            continue
        # head: luminaire + streak toward where it came from (the VP)
        hw_ = max(1.5, 0.9 * f / Z)
        Zt = Z + max(0.0, cam["v"]) * streak
        xs_, ys_ = _proj(cam, side * LAMP_X, LAMP_Y, Zt)
        thk = max(1, int(round(0.30 * f / Z)))
        col = tuple(float(q) for q in np.minimum(
            lc * (0.45 + 0.85 * float(np.exp(-Z / 120.0))) * power, 255.0))
        cv2.line(img, (int(xh), int(yh)), (int(xs_), int(ys_)), col, thk,
                 cv2.LINE_AA)
        # far heads are dimmer than white: a cluster of white dots at the VP
        # blooms into one blob
        fade = float(np.exp(-max(0.0, Z - 30.0) / 110.0))
        hcol = np.array((255.0, 245.0, 225.0)) * (0.35 + 0.65 * fade)
        cv2.ellipse(img, (int(xh), int(yh)), (int(hw_), max(1, int(hw_ * 0.28))),
                    0, 0, 360, tuple(float(q) for q in hcol), -1, cv2.LINE_AA)
        r = max(1.5, 2.2 * f / Z)
        _glow_dot(img, xh, yh, r, lc, 0.85 * power * fade)
        _glow_dot(img, xh, yh, r * 4.0, lc, 0.16 * power * fade ** 2)
        heads.append((xh, yh, xs_, ys_, Z, side, r))
    return heads


def _draw_blade(img, cov, G, P, cam, Z, b, power, bars, senv, T, i):
    tex = G["tex"][("blade", b["w"])]
    rgb, al, em = tex.levels[0]
    hm = al.shape[0] / al.shape[1] * 1.5
    side = b["side"]
    Xin, Xout = side * (XF - 2.0), side * (XF - 0.4)
    y0w, y1w = b["y0"] + hm, b["y0"]
    xa, ya = _proj(cam, min(Xin, Xout), y0w, Z)
    xb, yb = _proj(cam, max(Xin, Xout), y1w, Z)
    lvl = float(bars[b["band"]])
    gain = (0.55 + 0.9 * lvl) * power
    if b["flick"]:                        # a broken tube: snare flicker
        gain *= 1.0 - 0.8 * senv * (int(i) % 2)
    fade = float(np.exp(-Z / 300.0))
    _blit(img, cov, tex, xa, ya, xb, yb, P["neon"][b["c"]], gain * fade + 0.02,
          dim=fade)
    # its bracket to the wall
    xw, yw = _proj(cam, side * XF, y0w - 0.2, Z)
    cv2.line(img, (int(xb if side > 0 else xa), int(ya)), (int(xw), int(yw)),
             (14.0, 14.0, 16.0), max(1, int(0.08 * cam["f"] / Z)), cv2.LINE_AA)


def _draw_gantry(img, cov, G, P, cam, Z, g, power, kenv, T, i):
    """Overhead sign gantry: dark on the approach (warning lights blink on
    the beat during the build), IGNITES with "HIT A BUMP" at the drop."""
    f = cam["f"]
    e = g["e"]
    age = i - e
    steel = (16.0, 17.0, 20.0)
    # posts + truss
    for side in (-1, 1):
        xb, yb = _proj(cam, side * 7.9, 0.0, Z)
        xt, yt = _proj(cam, side * 7.9, 9.9, Z)
        th = max(1, int(0.32 * f / Z))
        cv2.line(img, (int(xb), int(yb)), (int(xt), int(yt)), steel, th, cv2.LINE_AA)
        cv2.line(cov, (int(xb), int(yb)), (int(xt), int(yt)), 1.0, th, cv2.LINE_AA)
    xl, yt0 = _proj(cam, -7.9, 10.0, Z)
    xr, yt1 = _proj(cam, 7.9, 9.7, Z)
    cv2.rectangle(img, (int(xl), int(yt0)), (int(xr), int(yt1)), steel, -1)
    cv2.rectangle(cov, (int(xl), int(yt0)), (int(xr), int(yt1)), 1.0, -1)
    x0, y0 = _proj(cam, -6.5, 9.7, Z)
    x1, y1 = _proj(cam, 6.5, 6.7, Z)
    if age < 0:
        # approach: unlit panel, red warning lamps blinking on the beat
        pre = float(T["pre"][i])
        _blit(img, cov, G["tex"]["gantry_txt"], x0, y0, x1, y1, P["gantry"],
              0.04 * power)
        on = (int(T["bi"][i]) % 2 == 0) and not T["black"][i]
        if on and pre > 0:
            for xw in (-6.2, 6.2):
                xx, yy = _proj(cam, xw, 10.3, Z)
                _glow_dot(img, xx, yy, max(2.0, 0.5 * f / Z), P["warn"],
                          1.6 + 2.0 * pre)
        return
    # ignition: a neon tube flickers on over 3 frames, then pulses with kicks
    ign = (0.35, 0.05, 0.8, 0.3, 1.0)[age] if age < 5 else 1.0
    gain = ign * (0.85 + 0.55 * kenv)
    _blit(img, cov, G["tex"]["gantry_txt"], x0, y0, x1, y1, P["gantry"], gain)
    _blit(img, cov, G["tex"]["gantry_frame"], x0, y0, x1, y1, P["neon"][2],
          0.8 * ign)


def _draw_bridge(img, cov, G, P, cam, Z, br, power, heat, T, i):
    """A skybridge between the facades, its side carrying the Tzeke000 logo."""
    f = cam["f"]
    ytop, ybot, depth = 11.5, 7.6, 6.0
    conc = _col(P["facade"]) * 1.4
    fg = 1.0 - np.exp(-Z / FOG_D)
    conc = conc * (1 - fg) + _col(P["fog"]) * fg
    # underside (visible from below as it approaches)
    Zb = Z + depth
    p = [_proj(cam, -XF, ybot, Z), _proj(cam, XF, ybot, Z),
         _proj(cam, XF, ybot, Zb), _proj(cam, -XF, ybot, Zb)]
    pts = np.array([[int(x), int(y)] for x, y in p], np.int32)
    cv2.fillPoly(img, [pts], tuple(float(q) for q in conc * 0.6), cv2.LINE_AA)
    cv2.fillPoly(cov, [pts], 1.0, cv2.LINE_AA)
    # strip lights along the underside edges (scroll overhead)
    ncol = _col(P["neon"][1]) * (0.3 + 0.7 * power)
    x0, y0 = _proj(cam, -XF, ytop, Z)
    x1, y1 = _proj(cam, XF, ybot, Z)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                  tuple(float(q) for q in conc), -1)
    cv2.rectangle(cov, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, -1)
    # window band on the bridge
    xa, ya = _proj(cam, -XF, ytop - 0.6, Z)
    xb, yb = _proj(cam, XF, ytop - 1.0, Z)
    cv2.rectangle(img, (int(xa), int(ya)), (int(xb), int(yb)),
                  tuple(float(q) for q in _col(P["win"][1]) * 0.35 * (0.2 + 0.8 * power)),
                  -1)
    xa, ya = _proj(cam, -XF, ybot + 0.1, Z)
    xb, yb = _proj(cam, XF, ybot - 0.05, Z)
    cv2.rectangle(img, (int(xa), int(ya)), (int(xb), int(yb)),
                  tuple(float(q) for q in ncol), -1)
    # the logo panel, centred over the road
    px0, py0 = _proj(cam, -5.0, ytop - 0.3, Z - 0.05)
    px1, py1 = _proj(cam, 5.0, ybot + 0.6, Z - 0.05)
    lit = (0.55 + 0.6 * heat) * (0.15 + 0.85 * power)
    _blit(img, cov, G["tex"]["bridge"], px0, py0, px1, py1, P["neon"][0],
          lit * (1 - fg), dim=(0.35 + 0.65 * power) * (1 - fg) * (0.7 + 0.4 * heat))


def _draw_bumpsign(img, cov, G, P, cam, Z, bsn, power):
    tex = G["tex"]["bumpsign"]
    for side in (-1, 1):
        xc = side * 7.7
        xb, yb = _proj(cam, xc, 0.0, Z)
        xt, yt = _proj(cam, xc, 2.6, Z)
        th = max(1, int(0.08 * cam["f"] / Z))
        cv2.line(img, (int(xb), int(yb)), (int(xt), int(yt)), (40.0, 40.0, 44.0),
                 th, cv2.LINE_AA)
        cv2.line(cov, (int(xb), int(yb)), (int(xt), int(yt)), 1.0, th, cv2.LINE_AA)
        x0, y0 = _proj(cam, xc - 1.1, 4.5, Z)
        x1, y1 = _proj(cam, xc + 1.1, 2.3, Z)
        fade = float(np.exp(-Z / 200.0))
        # retro-reflective: lit by the car's lights, even in a blackout
        _blit(img, cov, tex, x0, y0, x1, y1, P["hump"], 0.75 * fade + 0.1)


# --------------------------------------------------------------------------
# the ground (road, sidewalks, markings, light pools, reflections)
# --------------------------------------------------------------------------
def _ground(img, cov, G, T, P, i, cam, power, lz_vis, heads, heat, kenv,
            tinfo, in_tun):
    W, H = G["W"], G["H"]
    f, hy, cx, hc, Xc, Dm = (cam["f"], cam["hy"], cam["cx"], cam["hc"],
                             cam["Xc"], cam["Dm"])
    # the surface is shaded on the HALF-res grid (rows r0.. of the half
    # frame), then upsampled and composited at full res
    hh, hw = G["half"]
    r0 = int(np.floor(hy / 2.0)) + 1
    if r0 >= hh:
        return
    y0 = 2 * r0
    rows = (np.arange(r0, hh, dtype=np.float32) * 2 + 0.5)[:, None]
    xcol = (np.arange(hw, dtype=np.float32) * 2 + 0.5)[None, :]
    Zg = f * hc / (rows - hy)                       # (h,1)
    Xg = Xc + (xcol - cx) * Zg / f                   # (h,W)
    aX = np.abs(Xg)
    zw = Dm + Zg
    pxw = 2.0 * Zg / f                               # metres per px across
    pxz = 2.0 * Zg * Zg / (f * hc)                   # metres per px along
    v = abs(cam["v"])
    ground = aX < XF
    tun = T["tunnel"]
    if tinfo is not None and tun is not None:
        Zm, Ze = tinfo["Zm"], tinfo["Ze"]
        xp = (xcol - cx) / f
        if Zm > 0.3:                                  # portal occludes
            Xm = Xc + xp * Zm
            ground &= ~((Zg > Zm) & (np.abs(Xm) > XT))
        with np.errstate(divide="ignore", invalid="ignore"):
            Zw = np.where(xp > 1e-6, (XT - Xc) / xp,
                          np.where(xp < -1e-6, (-XT - Xc) / xp, 1e9))
        occ = (Zw < Zg) & (Zw >= max(Zm, 0.05)) & (Zw <= Ze)
        ground &= ~occ
    # -------- base materials
    asph = _col(P["asphalt"])
    walk = _col(P["walk"])
    is_walk = aX > CURB_X
    col = np.where(is_walk[..., None], walk, asph)
    # sidewalk joints
    jt = _dash_cov(zw, 2.5, 0.08, np.maximum(pxz, v * 0.5)) * is_walk
    col = col * (1 - 0.35 * jt)[..., None]
    # curb edge highlight
    curb = np.clip(1 - np.abs(aX - CURB_X) / np.maximum(pxw * 1.2, 0.06), 0, 1)
    # markings: dashed lane lines (beat-locked), solid edge lines
    mw = 0.075
    dl = np.minimum(np.abs(aX - LANES[0]), np.abs(aX - LANES[1]))
    cxv = np.clip((mw - dl) / np.maximum(pxw, 1e-3) + 0.5, 0, 1)
    blur = np.maximum(pxz, v)
    cz = _dash_cov(zw, L_DASH, DASH_ON, blur)
    edge = np.clip((mw * 1.4 - np.abs(aX - EDGE_X)) / np.maximum(pxw, 1e-3) + 0.5,
                   0, 1)
    mark = np.clip(cxv * cz + edge, 0, 1)
    # speed humps (yellow chevrons) for every sung "bump"
    hump = np.zeros_like(mark)
    for zb in T["humps"]:
        Zb = zb - Dm
        if -2 < Zb < 260:
            inz = _dash_cov(zw - zb + 0.0, 1e6, 1.0, np.maximum(pxz, v * 0.5))
            if inz.max() > 0:
                stripe = (np.floor((Xg + 0.3) / 0.6) % 2 == 0)
                hump = np.maximum(hump, inz * (0.35 + 0.65 * stripe) * (aX < 6.9))
    # -------- lighting (quarter res): ambient + lamp pools + storefront
    # spill + car underglow/shadow + tail glow + tunnel rings
    qh, qw = max(4, (H - y0) // 4), max(4, W // 4)
    qr = (np.arange(qh, dtype=np.float32) * 4 + 2 + y0)[:, None]
    qc = (np.arange(qw, dtype=np.float32) * 4 + 2)[None, :]
    qZ = f * hc / (qr - hy)
    qX = Xc + (qc - cx) * qZ / f
    qz = Dm + qZ
    Lt = np.zeros((qh, qw, 3), np.float32)
    Lt += 0.55                                        # ambient
    lc = _col(P["lamp"]) / 255.0
    if power > 0 and not in_tun:
        qzc = qz[:, 0]                                 # decreasing with row
        for zl in lz_vis:
            # rows within 3 sigma of this lamp only (qz is row-constant)
            ra = int(np.searchsorted(-qzc, -(zl + 20.0)))
            rb = int(np.searchsorted(-qzc, -(zl - 20.0)))
            if rb <= ra:
                continue
            dz2 = (qzc[ra:rb, None] - zl) ** 2
            Xs = qX[ra:rb]
            pool = (np.exp(-((Xs + (LAMP_X - 0.6)) ** 2 + dz2) / 84.5)
                    + np.exp(-((Xs - (LAMP_X - 0.6)) ** 2 + dz2) / 84.5)) * 3.2
            Lt[ra:rb] += pool[..., None] * lc
        # storefront spill along the sidewalks
        for side in (-1, 1):
            Bf = G["bld"][side]["front"]
            bq = np.clip(np.searchsorted(Bf["start"], qz.ravel(), side="right") - 1,
                         0, len(Bf["start"]) - 1).reshape(qz.shape)
            sc = np.asarray(P["win"], np.float32)[Bf["sc"][bq]] / 255.0
            near = np.exp(-np.maximum(0, XF - side * qX) / 1.8) \
                * (~Bf["gap"][bq]) * (side * qX > 0)
            Lt += near[..., None] * sc * 1.6
    if in_tun or (tun is not None and tun[0] - Dm < 300 and Dm < tun[1]):
        zm, ze = tun[0], tun[1]
        RS = SPEED_IN["big"] * L_DASH
        rel = qz - zm
        rk = np.floor(rel / RS + 0.5)
        dzr = rel - rk * RS
        inside = (qz >= zm) & (qz <= ze)
        rc = np.asarray(P["rings"], np.float32)[((rk.astype(np.int32)
                                                  + int(T["bi"][i]) // 4) % 4)] / 255.0
        Lt += (inside * np.exp(-np.abs(dzr) / 6.0) * (1.6 + 2.0 * kenv))[..., None] * rc
    # the car: shadow under it, underglow around it, tail glow behind
    Zc = float(T["Zc"][i])
    Xcar = float(T["Xcar"][i])
    lift = float(T["lift"][i])
    zc0 = Dm + Zc + HALF_LEN
    dxc = qX - Xcar
    dzc = qz - zc0
    sh = np.exp(-((dxc / 1.05) ** 4 + (dzc / 2.5) ** 4)) * (1 - min(0.8, lift * 1.6))
    ug = cam["ug"] / 255.0
    ugl = np.exp(-((dxc / 2.7) ** 2 + (dzc / 3.4) ** 2)) * \
        (0.55 + 1.5 * heat) * (1 - min(0.5, lift * 0.6))
    Lt = Lt * (1 - 0.82 * sh)[..., None] + ugl[..., None] * ug * 1.6
    tg = np.exp(-((dxc / 1.3) ** 2 + ((qz - (Dm + Zc - 0.6)) / 1.0) ** 2))
    Lt += (tg * (0.6 + 1.2 * kenv))[..., None] * (_col(P["tail"]) / 255.0)
    # the wet road MIRRORS the underglow + tail-lights (additive sheen): a
    # pool of the car's own light that stays on the road when it hops
    Ladd = (ugl * (55.0 + 70.0 * heat))[..., None] * ug \
        + (tg * (14.0 + 30.0 * kenv))[..., None] * (_col(P["tail"]) / 255.0)
    Laddb = cv2.resize(Ladd.astype(np.float32), (hw, hh - r0),
                       interpolation=cv2.INTER_LINEAR)
    Lbig = cv2.resize(Lt, (hw, hh - r0), interpolation=cv2.INTER_LINEAR)
    # -------- compose the surface
    markc = _col(P["mark"])
    col = col * (1 - mark[..., None]) + markc * mark[..., None]
    col = col + curb[..., None] * 11.0
    hc_col = _col(P["hump"])
    col = col * (1 - hump[..., None]) + hc_col * 0.55 * hump[..., None]
    col = col * Lbig + Laddb * (0.5 + 0.5 * P["wet"])
    # -------- wet reflection of the bright layer (half res, per-column mirror)
    # (quarter res: it is smeared vertically anyway)
    q4h, q4w = hh // 2, hw // 2
    src = cv2.resize(img, (q4w, q4h), interpolation=cv2.INTER_AREA)
    src = np.maximum(src - 48.0, 0.0) * 1.15
    Xp = XT if in_tun else XF
    xs_q = (np.arange(q4w, dtype=np.float32) * 4 + 1.5)
    ybm = hy + hc * np.abs(xs_q - cx) / Xp             # mirror line (full px)
    ys_q = (np.arange(q4h, dtype=np.float32) * 4 + 1.5)[:, None]
    rip = G["ripple4"] * (1.0 + 2.0 * (1 - P["wet"]))
    map_y = np.broadcast_to(((2 * ybm[None, :] - ys_q) / 4.0),
                            (q4h, q4w)).astype(np.float32)
    map_x = (np.arange(q4w, dtype=np.float32)[None, :] + rip).astype(np.float32)
    refl = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sy = 1.5 + 4.5 * (1 - P["wet"])
    refl = cv2.GaussianBlur(refl, (0, 0), sigmaX=0.6, sigmaY=sy)
    refl = cv2.resize(refl, (hw, hh), interpolation=cv2.INTER_LINEAR)[r0:]
    fres = (0.30 + 0.70 * np.exp(-(rows - hy) / (0.22 * H)))
    wet = P["wet"]
    col += refl * (fres * wet * 0.9)[..., None]
    # sidewalks reflect less
    col -= refl * (is_walk * 0.5 * fres * wet * 0.9)[..., None]
    # fog
    fg = 1.0 - np.exp(-Zg / FOG_D)
    col = col * (1 - fg)[..., None] + _col(P["fog"]) * fg[..., None]
    # composite under already-drawn objects (poles, sign posts)
    colF = cv2.resize(col.astype(np.float32), (W, H - y0),
                      interpolation=cv2.INTER_LINEAR)
    gm = cv2.resize(ground.astype(np.float32), (W, H - y0),
                    interpolation=cv2.INTER_LINEAR) * (1.0 - cov[y0:])
    gm[: max(0, int(np.ceil(hy)) + 1 - y0)] = 0.0
    img[y0:] = cv2.blendLinear(colF, np.ascontiguousarray(img[y0:]), gm,
                               (1.0 - gm))
    # explicit lamp reflections: long vertical streaks at the true mirror
    if power > 0:
        lc = _col(P["lamp"])
        for (xh, yh, xs_, ys_, Z, side, r) in heads:
            yr = hy + f * (hc + LAMP_Y) / Z
            if yr > H + 50:
                continue
            L_ = (yr - hy) * (0.35 + 0.5 * wet)
            _streak_v(img, xh, yr, max(1.5, r * 0.8), L_,
                      lc * (0.45 * wet + 0.1) * float(np.exp(-Z / 240.0)), y0)


def _streak_v(img, x, y, wx, L, col, ymin):
    """Additive vertical streak centred at (x, y), half-length L."""
    H, W = img.shape[:2]
    xa, xb = int(max(0, x - 3 * wx)), int(min(W, x + 3 * wx + 1))
    ya, yb = int(max(ymin, y - L)), int(min(H, y + L))
    if xb <= xa or yb <= ya:
        return
    xx = np.arange(xa, xb, dtype=np.float32) - x
    yy = np.arange(ya, yb, dtype=np.float32) - y
    g = np.exp(-(xx / wx) ** 2 / 2)[None, :] * np.exp(-(yy / max(1.0, L * 0.5)) ** 2)[:, None]
    img[ya:yb, xa:xb] += g[..., None] * col


# --------------------------------------------------------------------------
# the car
# --------------------------------------------------------------------------
def _car_light(T, P, i, cam, lz_vis, power, in_tun, kenv, bars, G):
    """Weights for the lighting passes: overhead lamp sweep (3 positions),
    left/right storefront neon, ambient."""
    Zcc = float(T["Zc"][i]) + HALF_LEN
    Dm = cam["Dm"]
    w = np.zeros(3, np.float32)
    lcol = np.zeros(3, np.float32)
    srcs = []
    if not in_tun:
        for zl in lz_vis:
            srcs.append((zl - Dm - Zcc, _col(P["lamp"]) / 255.0 * power, 1.0))
    else:
        tun = T["tunnel"]
        RS = SPEED_IN["big"] * L_DASH
        k0 = np.floor((Dm + Zcc - tun[0]) / RS)
        for k in (k0 - 1, k0, k0 + 1, k0 + 2):
            zr = tun[0] + k * RS
            rc = _col(P["rings"][int(k + int(T["bi"][i]) // 4) % 4]) / 255.0
            srcs.append((zr - Dm - Zcc, rc * (1.0 + 1.2 * kenv), 1.6))
    out = []
    for ly, c, g in srcs:
        if ly >= 4.5:
            ws = (np.exp(-((ly - 4.5) / 7.0) ** 2), 0, 0)
        elif ly >= 0:
            t = ly / 4.5
            ws = (t, 1 - t, 0)
        elif ly >= -5:
            t = -ly / 5.0
            ws = (0, 1 - t, t)
        else:
            ws = (0, 0, np.exp(-((ly + 5) / 4.0) ** 2))
        out.append((np.asarray(ws, np.float32) * g, c))
    return out


def _draw_car(img, G, T, P, i, cam, power, kenv, senv, spark, heat, lz_vis,
              in_tun):
    C = G["car"]
    sets = C["sets"]
    W, H = G["W"], G["H"]
    f, hy, cx, hc = cam["f"], cam["hy"], cam["cx"], cam["hc"]
    Zc = float(T["Zc"][i])
    Xrel = float(T["Xcar"][i]) - cam["Xc"]
    lift = float(T["lift"][i])
    brel = float(T["brel"][i]) + float(T["susp"][i])
    sqx, sqy = float(T["sqx"][i]), float(T["sqy"][i])
    hotf = float(T["hot_f"][i])
    f_ref = C["meta"]["f"] * C["s0"]
    s = (f / f_ref) * (Z_REAR0 / Zc)
    Zcontact = Zc + (HALF_LEN + C["meta"]["rear_wheel_y"])
    ax = cx + f * Xrel / Zcontact
    ay = hy + f * hc / Zcontact
    lift_px = f * lift / Zcontact
    body_px = f * brel / Zcontact
    # bracket the camera height between rendered sets
    hcs = [d["hc"] for d in sets]
    hcl = float(np.clip(hc, hcs[0], hcs[-1]))
    k = int(np.clip(np.searchsorted(hcs, hcl) - 1, 0, len(sets) - 2))
    t = (hcl - hcs[k]) / max(1e-6, hcs[k + 1] - hcs[k])
    pairs = [(sets[k], 1 - t), (sets[k + 1], t)]
    pairs = [(d, w) for d, w in pairs if w > 0.02]
    # lighting
    lights = _car_light(T, P, i, cam, lz_vis, power, in_tun, kenv,
                        T["bars"][i], G)
    bars = T["bars"][i]
    nL = _col(P["neon"][int(T["bi"][i] // 8) % 4]) / 255.0 * (0.35 + 0.9 * float(bars[6])) * power
    nR = _col(P["neon"][(int(T["bi"][i] // 8) + 2) % 4]) / 255.0 * (0.35 + 0.9 * float(bars[14])) * power
    if in_tun:
        nL = nL * 0.3 + _col(P["rings"][0]) / 255.0 * 0.5
        nR = nR * 0.3 + _col(P["rings"][1]) / 255.0 * 0.5
    amb = 0.45 + 0.25 * power
    tint = np.asarray(P["paint"], np.float32)
    tail_I = (0.7 + 2.2 * kenv * (0.4 + 0.6 * hotf) + 1.6 * float(T["brake"][i]))
    tcol = _col(P["tail"])
    # the bracketing height sets are blended into ONE premultiplied layer
    # (wheels under body within each set), then composited once — blending
    # them one after the other let the road show through the car
    cxr = 0.5 * C["meta"]["W"] * C["s0"]
    rects = []
    for d, wgt in pairs:
        h_, w_ = d["base_a"].shape
        x0r, y0r, cyr = d["x0"], d["y0"], d["cy"]
        X0 = ax + s * min(sqx, 1.0) * (x0r - cxr) - 4
        X1 = ax + s * max(sqx, 1.0) * (x0r + w_ - cxr) + 4
        Y0 = ay - lift_px - abs(body_px) + s * max(sqy, 1.0) * (y0r - cyr) - 4
        Y1 = ay - lift_px + abs(body_px) + s * max(sqy, 1.0) * (y0r + h_ - cyr) + 4
        rects.append((X0, Y0, X1, Y1))
    rx0 = int(max(0, np.floor(min(r[0] for r in rects))))
    ry0 = int(max(0, np.floor(min(r[1] for r in rects))))
    rx1 = int(min(W, np.ceil(max(r[2] for r in rects))))
    ry1 = int(min(H, np.ceil(max(r[3] for r in rects))))
    if rx1 <= rx0 or ry1 <= ry0:
        return
    lay = np.zeros((ry1 - ry0, rx1 - rx0, 3), np.float32)
    lay_a = np.zeros((ry1 - ry0, rx1 - rx0), np.float32)
    for d, wgt in pairs:
        body = d["base"] * amb
        for ws, lcol in lights:
            for kk in range(3):
                if ws[kk] > 0.01:
                    body = body + d[f"top{kk}"] * (lcol * ws[kk] * 0.85)
        body = body + d["sideL"] * (nL * 0.6) + d["sideR"] * (nR * 0.6)
        mp = d["m_paint"][..., None]
        body = body * (1 + (tint - 1) * mp)
        body *= 255.0
        body += d["m_tail"][..., None] * tcol * tail_I
        ba = d["base_a"]
        wh = d["wheels"] * 255.0 * (amb * 1.1)
        for ws, lcol in lights:
            if ws[1] > 0.01:
                wh = wh + d["wheels"] * 255.0 * (lcol * ws[1] * 0.5)
        wa = d["wheels_a"]
        x0r, y0r, cyr = d["x0"], d["y0"], d["cy"]
        t_rgb = np.zeros_like(lay)
        t_a = np.zeros_like(lay_a)
        # wheels: ground contact + lift only (they lag the body)
        _warp_into(t_rgb, t_a, rx0, ry0, wh, wa, s, s,
                   ax + s * (x0r - cxr), ay - lift_px + s * (y0r - cyr))
        # body: + suspension / hop offset, squash & stretch about the
        # contact line (volume kept: x widens when y squashes)
        _warp_into(t_rgb, t_a, rx0, ry0, body, ba, s * sqx, s * sqy,
                   ax + s * sqx * (x0r - cxr),
                   ay - lift_px - body_px + s * sqy * (y0r - cyr))
        lay += t_rgb * wgt
        lay_a += t_a * wgt
    wsum = sum(w for _, w in pairs)
    if wsum > 0:
        lay /= wsum
        lay_a /= wsum
    reg = img[ry0:ry1, rx0:rx1]
    reg *= (1.0 - lay_a)[..., None]
    reg += lay
    # tail-light anamorphic flare on the kick + its wet reflection
    tz = C["meta"]["tail_z"]
    Zr = Zc - 0.05
    ty = hy + f * (hc - tz - lift - brel) / Zr
    if kenv > 0.05:
        L = (0.25 + 0.55 * hotf) * W * kenv
        _hflare(img, ax, ty, L, max(1.5, 2.5 * G["ppm"]), tcol * 0.9 * kenv)
    for sx in (-0.83, 0.83):
        xx = cx + f * (Xrel + sx) / Zr
        yr = hy + f * (hc + tz) / Zr
        _streak_v(img, xx, yr, max(2.0, 4 * G["ppm"]), (yr - ty) * 0.6,
                  tcol * (0.25 + 0.25 * kenv) * P["wet"], int(hy))
    # underglow LED strip under the body (visible between the tyres)
    ug = cam["ug"]
    uy = hy + f * (hc - 0.10 - lift - brel * 0.5) / (Zc + 0.3)
    xl = cx + f * (Xrel - 0.62) / (Zc + 0.3)
    xr = cx + f * (Xrel + 0.62) / (Zc + 0.3)
    gI = (0.55 + 1.3 * heat)
    cv2.line(img, (int(xl), int(uy)), (int(xr), int(uy)),
             tuple(float(q) for q in ug * min(1.2, gI)), max(1, int(2 * G["ppm"])),
             cv2.LINE_AA)
    _hflare(img, (xl + xr) / 2, uy, (xr - xl) * 0.7, max(2.0, 5 * G["ppm"]),
            ug * 0.35 * gI)
    # sparks: the chassis scrapes the road when the car lands from a bump
    for lf in T["land_frames"]:
        age = i - lf
        if 0 <= age < 12:
            _sparks(img, G, ax, ay, f / Zcontact, age, lf)
    # exhaust: heat shimmer on hi-hat sparkle, backfire pops on snares (drops)
    ez = C["meta"]["exhaust_z"]
    ey = hy + f * (hc - ez - lift - brel) / (Zc - 0.1)
    exs = [cx + f * (Xrel + ex) / (Zc - 0.1) for ex in C["meta"]["exhaust_x"]]
    if spark > 0.08:
        _shimmer(img, exs, ey, f / Zc, spark, i)
    if senv > 0.1 and bool(T["hot"][i]) and power > 0:
        fc = _col(P["flame"])
        for ex in exs:
            r = (0.03 + 0.06 * senv) * f / Zc
            _glow_dot(img, ex, ey + r * 0.3, r, fc, 0.7 * senv)
            _glow_dot(img, ex, ey + r * 0.3, r * 0.4, (255, 230, 200), 0.8 * senv)


def _sparks(img, G, ax, ay, ppm_m, age, seed):
    """A burst of sparks from under the car, stateless: seeded by the
    landing frame, positions are closed-form in the age."""
    rs = np.random.default_rng(1000 + int(seed))
    n = 60
    x0 = ax + rs.uniform(-0.8, 0.8, n) * ppm_m
    y0 = ay - 0.05 * ppm_m + rs.uniform(-0.02, 0.02, n) * ppm_m
    vx = rs.uniform(-1.0, 1.0, n) * 0.11 * ppm_m + np.sign(x0 - ax) * 0.05 * ppm_m
    vy = rs.uniform(-0.10, 0.02, n) * ppm_m
    g = 0.022 * ppm_m
    life = rs.uniform(5, 12, n)
    a0 = np.maximum(0, age - 1)
    px0 = x0 + vx * a0
    py0 = y0 + vy * a0 + 0.5 * g * a0 * a0
    px1 = x0 + vx * age
    py1 = y0 + vy * age + 0.5 * g * age * age
    H, W = img.shape[:2]
    for k in range(n):
        if age > life[k]:
            continue
        fade = 1.0 - age / life[k]
        col = (255.0 * fade + 40, 190.0 * fade ** 1.5 + 30, 80.0 * fade ** 3)
        cv2.line(img, (int(px0[k]), int(py0[k])), (int(px1[k]), int(py1[k])),
                 col, max(1, int(round(1.6 * G["ppm"]))), cv2.LINE_AA)


def _warp_into(dst, dst_a, ox, oy, rgb, al, sx, sy, tx, ty):
    """Premultiplied sprite OVER (dst, dst_a), whose origin is frame pixel
    (ox, oy); sprite scaled by (sx, sy) with its top-left at frame (tx, ty)."""
    h, w = dst_a.shape
    M = np.array([[sx, 0, tx - ox], [0, sy, ty - oy]], np.float32)
    r = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = cv2.warpAffine(al, M, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    dst *= (1.0 - a)[..., None]
    dst += r
    dst_a *= (1.0 - a)
    dst_a += a


def _hflare(img, x, y, L, wy, col):
    H, W = img.shape[:2]
    xa, xb = int(max(0, x - L)), int(min(W, x + L))
    ya, yb = int(max(0, y - 4 * wy)), int(min(H, y + 4 * wy + 1))
    if xb <= xa or yb <= ya:
        return
    xx = (np.arange(xa, xb, dtype=np.float32) - x) / max(1.0, L)
    yy = (np.arange(ya, yb, dtype=np.float32) - y) / wy
    g = (np.exp(-yy ** 2 / 2)[:, None] * (1 - np.abs(xx)) ** 2)
    img[ya:yb, xa:xb] += g[..., None] * _col(col)


def _shimmer(img, exs, ey, ppm_m, spark, i):
    """Heat haze: wobble the pixels just behind the tips (toward the
    camera = below on screen)."""
    H, W = img.shape[:2]
    x0 = int(max(0, min(exs) - 0.25 * ppm_m))
    x1 = int(min(W, max(exs) + 0.25 * ppm_m))
    y0 = int(max(0, ey - 0.05 * ppm_m))
    y1 = int(min(H, ey + 0.55 * ppm_m))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    reg = img[y0:y1, x0:x1].copy()
    hh, ww = reg.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    amp = (1.0 + 3.0 * spark) * (ppm_m / 300.0)
    ph = i * 0.9
    fall = np.clip(yy / hh, 0, 1)
    dx = amp * np.sin(yy * 0.45 + ph) * np.sin(xx * 0.08 + ph * 0.7) * fall
    dy = 0.5 * amp * np.sin(xx * 0.3 - ph * 1.3) * fall
    img[y0:y1, x0:x1] = cv2.remap(reg, xx + dx, yy + dy, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------
# speed lines + rain
# --------------------------------------------------------------------------
def _speed_lines(img, G, T, P, i, cam):
    nit = float(T["nitro"][i])
    hotf = float(T["hot_f"][i])
    pre = float(T["pre"][i])
    amt = max(nit * 1.0, hotf * 0.35 + 0.0, pre ** 2 * 0.3)
    if amt < 0.03:
        return
    W, H = G["W"], G["H"]
    cx, cy = cam["cx"], cam["hy"]
    diag = np.hypot(W, H)
    t = i / 30.0
    L = G["slines"]
    rate = 1.2 + 2.0 * hotf + 2.5 * nit
    ph = (L[:, 1] + t * rate * L[:, 2]) % 1.0
    r0 = diag * (0.16 + 0.55 * ph ** 2)
    r1 = r0 * (1.0 + 0.10 + 0.35 * amt)
    ca, sa = np.cos(L[:, 0]), np.sin(L[:, 0])
    col = _col(P["neon"][0]) * 0.5 + 120.0
    lay = np.zeros((H, W, 3), np.float32)
    for k in range(len(L)):
        if L[k, 3] * amt < 0.25:
            continue
        p0 = (int(cx + ca[k] * r0[k]), int(cy + sa[k] * r0[k] * 0.62))
        p1 = (int(cx + ca[k] * r1[k]), int(cy + sa[k] * r1[k] * 0.62))
        a_ = min(1.0, amt * L[k, 3]) * np.sin(np.pi * ph[k]) * 0.55
        cv2.line(lay, p0, p1, tuple(float(q) for q in col * a_),
                 max(1, int(1.5 * G["ppm"] * L[k, 3])), cv2.LINE_AA)
    img += lay


def _rain(img, G, T, P, i, cam, spark):
    W, H = G["W"], G["H"]
    D = G["rain"]
    t = i / 30.0
    v = abs(cam["v"])
    x = (D[:, 0] * W * 1.3 - 0.15 * W)
    y = ((D[:, 1] + t * 1.6 * D[:, 2]) % 1.0) * H
    # forward motion: drops stream away from the vanishing point
    dx, dy = x - cam["cx"], y - cam["hy"]
    sp = 0.004 + 0.012 * v
    lx = 0.012 * H * D[:, 2]
    x1 = x + dx * sp
    y1 = y + lx + dy * sp
    col = _col((170, 190, 215)) * (0.25 + 0.35 * spark)
    lay = np.zeros((H, W, 3), np.float32)
    for k in range(len(D)):
        a_ = float(D[k, 3])
        cv2.line(lay, (int(x[k]), int(y[k])), (int(x1[k]), int(y1[k])),
                 tuple(float(q) for q in col * a_), 1, cv2.LINE_AA)
    img += lay


# --------------------------------------------------------------------------
# post FX
# --------------------------------------------------------------------------
def fx(R, img, i, a, lv):
    c = R.scene_cache.get("carstreet")
    if c is None:
        return img
    T, G = c["T"], c["G"]
    W, H = R.W, R.H
    # strobe: budgeted hits only, modest, never stacked
    sa = int(T["strobe_age"][i])
    if R.style.strobe and sa < 4:
        img += 70.0 * (0.45 ** sa)
    hotf = float(T["hot_f"][i])
    kage = int(T["kage"][i])
    kenv = 0.72 ** kage if kage < 30 else 0.0
    if hotf > 0.05:
        off = int(round((1.0 + 2.5 * kenv) * hotf * W / 1920))
        if off > 0:
            img[..., 0] = np.roll(img[..., 0], off, axis=1)
            img[..., 2] = np.roll(img[..., 2], -off, axis=1)
    amp = float(T["sh_amp"][i]) * 0.014 * H
    z = 1.0
    lz = getattr(R, "lyric_zoom", None)
    if lz is not None:
        pm = float(np.max(lz)) - 1.0
        if pm > 0:
            z += 0.035 * (float(lz[i]) - 1.0) / pm
    dx = dy = 0.0
    if amp > 0.3:
        th = 6.2832 * float(lv._hash01(i, 7.0))
        dx, dy = amp * np.cos(th), amp * np.sin(th)
        if T["sh_vert"][i] > 0:
            dx *= 0.3
            dy = amp * (1.0 if (i % 2 == 0) else -0.7)
        z = max(z, 1.0 + 2.2 * amp / min(W, H))
    if z > 1.0005 or abs(dx) + abs(dy) > 0.3:
        cxp, cyp = W / 2, H * HY_FRAC
        M = np.array([[z, 0, (1 - z) * cxp + dx], [0, z, (1 - z) * cyp + dy]],
                     np.float32)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return img
