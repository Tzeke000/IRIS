"""STEAMPUNK — a brass-and-iron engine room inside a clocktower, at night
(scene plugin for scripts/lyric_viz.py; contract in README.md). Built for
Zeke's "Hit a Bump" (hardcore dubstep).

2.5-D, seven depth planes, a camera that moves IN DEPTH (it pushes in on the
build, is blown back on the drop, breathes in the calm):
  4.0  back wall: riveted plates, lattice girders, the round CLOCK WINDOW
       (frosted moonlit glass, mirrored numerals, hands that tick a notch
       every bar) — the halo the main gear sits in
  3.0  far gear train in the haze
  2.6  copper steam pipes + their vents
  2.0  the MAIN GEAR TRAIN (Tzeke000 medallion fixed upright in the main
       gear's face) + two vertical steam engines (flywheel, crank, rod,
       crosshead, cylinder)
  1.7  a row of filament bulbs = the spectrum
  1.45 the boiler backhead: firebox, six pressure gauges, a Nixie readout,
       caged warning lamps, two whistles
  1.0  out-of-focus foreground gears framing the shot

What drives what (every element has a NAMED cause):
  kick        escapement TICK of the gear train (snap + settle), pistons
              stroke (half a crank turn per kick, the two engines alternate),
              firebox flare, bulbs sway (pendulum follow-through),
              medallion punch
  snare       steam vents burst (seeded per snare index, more in drops)
  hi-hat      (high minus its smooth) filament flicker + small sparks
              spitting from the gear meshes
  sub-bass    firebox heat (+ the firelight on every brass surface), the
              medallion's red plume glows like a coal
  spectrum    the bulbs (14 bands) and the gauges (6 bands, spring-damped
              needles that overshoot)
  build pre   gauges climb into the red, needles shiver, warning lamps blink
              faster and faster, Nixie pressure counts up, steam leaks hiss,
              camera pushes in
  black       (half beat before a drop) every lamp dies, needles fall
  drop entry  valves blow: steam floods from every vent + whistles + cylinder
              cocks, a spark shower, camera blown back + shake, the
              HIT A BUMP maker's plate slams in
  big drop    furnace-red grade, faster gears (two teeth a tick), more steam,
              rotating alarm wash
  bar ONE     the clock's minute hand ticks a notch
  sung "bump" camera push-in + both whistles blow

Assets are Blender renders of NORMAL / material-weight / AO maps (see
blender/steampunk_assets.py), lit here per frame: static plates through
precomputed per-light images, rotating gears through per-frame matcaps (the
light is rotated into the gear's frame, so highlights stay put while the
teeth turn).

STATELESS: everything per-frame is a function of i over arrays computed once
for the whole song (cumsums, frames-since-event, a spring integrated from
frame 0) and seeded randomness. Nothing carries from frame to frame.
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ASSET_DIR = os.path.join(REPO, "assets", "scenes", "steampunk")
ASSET_CMD = ("~/.local/bin/blender -b --factory-startup --python-exit-code 1 "
             "--python scripts/lyric_scenes/blender/steampunk_assets.py")
ASSET_VERSION = 3

# ---------------------------------------------------------------------------
# palettes
# ---------------------------------------------------------------------------
_BASE = dict(
    # metal albedos (0..255) + specular tints
    brass=(196, 146, 62), copper=(182, 94, 50), steel=(150, 150, 156),
    iron=(44, 39, 35),
    brass_s=(255, 200, 115), copper_s=(255, 150, 100), steel_s=(215, 222, 232),
    iron_s=(45, 40, 36),
    patina=(64, 150, 124), patina_amt=0.0,
    # light colours (multipliers)
    amb=(0.19, 0.16, 0.14), fire=(1.45, 0.66, 0.24), key=(1.05, 0.74, 0.40),
    back=(0.55, 0.72, 0.95), spot=(1.05, 0.90, 0.66), red=(1.6, 0.14, 0.06),
    # the finale's furnace grade (big drop)
    big_amb=(0.30, 0.09, 0.04), big_fire=(2.0, 0.42, 0.10),
    big_key=(1.35, 0.42, 0.14), big_back=(0.95, 0.32, 0.16),
    # window glass (centre, edge) + the room's haze
    glass_c=(168, 188, 206), glass_e=(50, 72, 98), fog=(26, 21, 18),
    steam=(1.0, 0.96, 0.92),
    # enamel on the medallion
    en_fill=(226, 212, 178), en_line=(22, 18, 16), en_red=(178, 30, 22),
    # instruments
    dial=(226, 212, 176), ink=(34, 26, 20), needle=(30, 22, 18),
    red_zone=(190, 34, 26), nixie=(255, 112, 32), nixie_core=(255, 196, 130),
    bulb=(255, 140, 45), bulb_hot=(255, 226, 170), lamp=(255, 34, 20),
    spark=(255, 214, 140), flash=(255, 196, 130),
)
PALETTES = {
    # brass / copper / amber on soot and dark iron, cold moonlit window
    "brass": dict(_BASE),
    # teal-green patina eating the brass, copper, a sea-glass window
    "verdigris": dict(
        _BASE, brass=(168, 140, 76), copper=(160, 92, 58), iron=(32, 38, 37),
        patina=(70, 156, 132), patina_amt=0.80,
        amb=(0.12, 0.18, 0.17), back=(0.42, 0.88, 0.80), key=(0.98, 0.78, 0.46),
        glass_c=(170, 232, 212), glass_e=(28, 86, 80), fog=(14, 27, 25),
        dial=(214, 222, 196), big_amb=(0.12, 0.17, 0.15), big_key=(1.0, 0.62, 0.30),
        big_fire=(1.9, 0.50, 0.14), big_back=(0.35, 0.95, 0.80)),
    # red / orange heat, dark iron, the window is the glow of a burning city
    "furnace": dict(
        _BASE, brass=(176, 112, 50), copper=(166, 72, 36), steel=(124, 112, 106),
        iron=(34, 25, 21), amb=(0.22, 0.10, 0.06), fire=(1.9, 0.58, 0.16),
        key=(1.25, 0.50, 0.18), back=(1.05, 0.40, 0.14), spot=(1.15, 0.72, 0.42),
        glass_c=(255, 156, 70), glass_e=(96, 24, 10), fog=(34, 13, 7),
        dial=(232, 200, 150), big_back=(1.2, 0.30, 0.10)),
}
STYLE = dict(palette=[(232, 172, 72), (255, 120, 34), (200, 92, 44),
                      (120, 200, 190)],
             font_title="Anton-Regular.ttf", strobe=True)
BLOOM = 0.34

f32 = np.float32


def _h01(x, y=0.0):
    """Deterministic hash of (x, y) -> [0, 1) (same as lyric_viz._hash01)."""
    v = math.sin(float(x) * 12.9898 + float(y) * 78.233) * 43758.5453
    return v - math.floor(v)


def _c(v):
    return np.asarray(v, f32)


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------
def _manifest():
    p = os.path.join(ASSET_DIR, "manifest.json")
    if not os.path.isfile(p):
        raise SystemExit(f"steampunk: assets missing ({p}). Generate them with:\n  {ASSET_CMD}")
    m = json.load(open(p))
    if m.get("version") != ASSET_VERSION:
        raise SystemExit(f"steampunk: assets are version {m.get('version')}, scene needs "
                         f"{ASSET_VERSION}. Regenerate:\n  {ASSET_CMD}")
    need = ("main", "g24", "g20", "g12", "flywheel", "medallion", "wall", "pipes",
            "engine", "backhead", "plate")
    miss = [n for n in need if n not in m["assets"]
            or not os.path.isfile(os.path.join(ASSET_DIR, n + ".npz"))]
    if miss:
        raise SystemExit(f"steampunk: assets {miss} missing in {ASSET_DIR}. Regenerate:\n  {ASSET_CMD}")
    return m


def _load_maps(name, scale, mirror=False):
    """Asset maps resampled by `scale` (target ppu / native ppu)."""
    z = np.load(os.path.join(ASSET_DIR, name + ".npz"))
    a = z["alpha"].astype(f32) / 255.0
    n = z["normal"].astype(f32) / 127.5 - 1.0
    w = z["weights"].astype(f32) / 255.0
    ao = z["ao"].astype(f32) / 255.0
    det = z["detail"].astype(f32) / 255.0
    if abs(scale - 1.0) > 1e-3:
        h0, w0 = a.shape
        sz = (max(2, int(round(w0 * scale))), max(2, int(round(h0 * scale))))
        it = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        # premultiply so edges don't drag in the (meaningless) outside values
        stack = np.dstack([a, n * a[..., None], w * a[..., None], ao * a, det * a])
        stack = np.dstack([cv2.resize(np.ascontiguousarray(stack[..., c0:c0 + 3]), sz,
                                      interpolation=it).reshape(sz[1], sz[0], -1)
                           for c0 in range(0, stack.shape[2], 3)])
        a = stack[..., 0]
        inv = 1.0 / np.maximum(a, 1e-3)
        n = stack[..., 1:4] * inv[..., None]
        w = stack[..., 4:7] * inv[..., None]
        ao = stack[..., 7] * inv
        det = stack[..., 8] * inv
    n[..., 1] *= -1.0                      # Blender y-up -> screen y-down
    if mirror:
        a, w, ao, det = a[:, ::-1], w[:, ::-1], ao[:, ::-1], det[:, ::-1]
        n = n[:, ::-1].copy()
        n[..., 0] *= -1.0
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-4)
    n[a < 1e-3] = (0.0, 0.0, 1.0)
    return dict(a=np.ascontiguousarray(a), n=np.ascontiguousarray(n),
                w=np.ascontiguousarray(np.clip(w, 0, 1)),
                ao=np.ascontiguousarray(np.clip(ao, 0, 1)),
                det=np.ascontiguousarray(np.clip(det, 0, 1)))


def _surface(M, P):
    """Albedo, specular tint, AO from the material weights + palette."""
    w, det, ao = M["w"], M["det"], M["ao"]
    wi = np.clip(1.0 - w.sum(-1), 0, 1)
    alb = (w[..., 0:1] * _c(P["brass"]) + w[..., 1:2] * _c(P["copper"])
           + w[..., 2:3] * _c(P["steel"]) + wi[..., None] * _c(P["iron"]))
    alb *= (0.70 + 0.55 * det)[..., None]
    spc = (w[..., 0:1] * _c(P["brass_s"]) + w[..., 1:2] * _c(P["copper_s"])
           + w[..., 2:3] * _c(P["steel_s"]) + wi[..., None] * _c(P["iron_s"]))
    spc *= (0.55 + 0.6 * det)[..., None]
    if P["patina_amt"] > 0:
        pm = np.clip((1.0 - ao) * 1.8 + (det - 0.40) * 1.6, 0, 1) \
            * (w[..., 0] + w[..., 1]) * P["patina_amt"]
        alb = alb * (1 - pm[..., None]) + _c(P["patina"]) * (0.65 + 0.5 * det)[..., None] * pm[..., None]
        spc *= (1.0 - 0.85 * pm)[..., None]
    return alb / 255.0, spc / 255.0, (0.35 + 0.65 * ao)


# lights per depth plane: (channel, x, y, dz toward camera, falloff R, power).
# x/y are screen units at rest (the window is at (0, -0.11)).
def _lights(L):
    wx, wy = L["window"]["c"]
    bulbs = [("key", x, -0.45, 0.0, 0.9, 0.30) for x in (-0.6, -0.2, 0.2, 0.6)]
    return {
        "wall": [("fire", 0.0, 0.85, 1.6, 1.4, 0.9),
                 ("key", -0.55, -0.55, 1.6, 1.4, 0.55), ("key", 0.55, -0.55, 1.6, 1.4, 0.55),
                 ("back", wx, wy, 0.06, 0.42, 1.25)],
        "pipes": [("fire", 0.0, 0.58, 0.60, 0.95, 1.0),
                  *[(c, x, y, 0.80, r, p) for (c, x, y, _, r, p) in bulbs],
                  ("back", wx, wy, -0.28, 0.85, 1.3)],
        "main": [("fire", 0.0, 0.52, 0.42, 0.75, 1.15),
                 *[(c, x, y, 0.32, r * 0.8, p * 0.7) for (c, x, y, _, r, p) in bulbs],
                 ("back", wx, wy, -0.34, 1.0, 1.5),
                 ("spot", 0.12, -0.36, 0.50, 0.45, 1.0)],
        "backhead": [("fire", 0.0, 0.455, 0.07, 0.30, 1.8),
                     ("key", -0.4, -0.15, 0.55, 1.2, 0.55), ("key", 0.4, -0.15, 0.55, 1.2, 0.55),
                     ("red", -0.79, 0.368, 0.05, 0.13, 1.0), ("red", 0.79, 0.368, 0.05, 0.13, 1.0),
                     ("amb2", 0.0, -1.0, 1.0, 3.0, 0.4)],
        "plate": [("key", -0.35, -0.45, 0.55, 1.6, 1.0), ("fire", 0.0, 0.55, 0.35, 1.4, 0.7),
                  ("spot", 0.25, -0.30, 0.40, 1.0, 0.6)],
    }


CHANNELS = ("amb", "fire", "key", "back", "spot", "red", "amb2")
SPEC_P = 26.0


def _plate_S(M, P, lights, x0, y0, ppu):
    """Per-light lit images of a static plate (premultiplied), white light."""
    alb, spc, ao = _surface(M, P)
    n, a = M["n"], M["a"]
    h, w = a.shape
    xs = x0 + (np.arange(w, dtype=f32) + 0.5) / ppu
    ys = y0 + (np.arange(h, dtype=f32) + 0.5) / ppu
    X, Y = np.meshgrid(xs, ys)
    acc = {}
    amb = alb * (ao * (0.55 + 0.45 * n[..., 2]))[..., None]
    acc["amb"] = amb
    for ch, lx, ly, dz, R, pw in lights:
        Lx, Ly, Lz = lx - X, ly - Y, np.full_like(X, dz)
        d = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz) + 1e-6
        Lx, Ly, Lz = Lx / d, Ly / d, Lz / d
        F = pw / (1.0 + (d / R) ** 2)
        nl = np.clip(n[..., 0] * Lx + n[..., 1] * Ly + n[..., 2] * Lz, 0, None)
        Hx, Hy, Hz = Lx, Ly, Lz + 1.0
        hn = np.sqrt(Hx * Hx + Hy * Hy + Hz * Hz) + 1e-6
        nh = np.clip((n[..., 0] * Hx + n[..., 1] * Hy + n[..., 2] * Hz) / hn, 0, None)
        sp = (nh ** SPEC_P) * 1.6 + (nh ** 6) * 0.18
        img = alb * (ao * nl * F)[..., None] + spc * (sp * F * (0.4 + 0.6 * ao))[..., None]
        acc[ch] = acc.get(ch, 0.0) + img
    return _pack(acc, a)


def _pack(acc, a):
    """Per-light images -> one (h, w, 3*nl) stack: a frame's lighting is then
    ONE matmul with a block matrix of the light colours (5x faster than
    summing scaled images)."""
    chs = list(acc)
    S = np.empty(a.shape + (3 * len(chs),), f32)
    for k, ch in enumerate(chs):
        S[..., 3 * k:3 * k + 3] = acc[ch] * (a[..., None] * 255.0)
    return dict(S=S, chs=chs)


class _Matcap:
    """A rotating gear: sprite-space maps + per-frame matcap lighting."""
    G = 48

    def __init__(self, M, P, ppu):
        self.alb, self.spc, ao = _surface(M, P)
        self.alb = (self.alb * ao[..., None]).astype(f32)
        self.ao = ao
        self.a = M["a"]
        self.albA = (self.alb * (M["a"] * 255.0)[..., None]).astype(f32)
        self.spcA = (self.spc * ((0.4 + 0.6 * ao) * M["a"] * 255.0)[..., None]).astype(f32)
        g = self.G - 1
        self.mx = ((M["n"][..., 0] + 1.0) * 0.5 * g).astype(f32)
        self.my = ((M["n"][..., 1] + 1.0) * 0.5 * g).astype(f32)
        self.ppu = ppu
        h, w = self.a.shape
        self.cx, self.cy = (w - 1) / 2.0, (h - 1) / 2.0
        u = np.linspace(-1, 1, self.G, dtype=f32)
        U, V = np.meshgrid(u, u)
        r = np.sqrt(U * U + V * V)
        s = np.where(r > 0.999, 0.999 / np.maximum(r, 1e-6), 1.0)
        U, V = U * s, V * s
        self.N = np.stack([U, V, np.sqrt(np.clip(1 - U * U - V * V, 0, 1))], -1)

    def shade(self, lights, cols, theta):
        """lights: [(channel, Lx, Ly, Lz, F)] in SCREEN space at the gear centre."""
        ct, st = math.cos(-theta), math.sin(-theta)
        N = self.N
        dif = np.zeros((self.G, self.G, 3), f32)
        spe = np.zeros((self.G, self.G, 3), f32)
        dif += cols["amb"] * (0.55 + 0.45 * N[..., 2:3])
        for ch, lx, ly, lz, F in lights:
            c = cols[ch]
            if not c.any():
                continue
            sx, sy = ct * lx - st * ly, st * lx + ct * ly      # into sprite frame
            nl = np.clip(N[..., 0] * sx + N[..., 1] * sy + N[..., 2] * lz, 0, None)
            hx, hy, hz = sx, sy, lz + 1.0
            hn = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-6
            nh = np.clip((N[..., 0] * hx + N[..., 1] * hy + N[..., 2] * hz) / hn, 0, None)
            dif += (nl * F)[..., None] * c
            spe += ((nh ** SPEC_P) * 1.6 + (nh ** 6) * 0.18)[..., None] * (F * c)
        D = cv2.remap(dif, self.mx, self.my, cv2.INTER_LINEAR)
        S = cv2.remap(spe, self.mx, self.my, cv2.INTER_LINEAR)
        D *= self.albA               # albedo * AO * alpha * 255 (premultiplied)
        S *= self.spcA
        D += S
        return D


def _blit(img, rgb, alpha, M, region=None):
    """Composite premultiplied rgb (h,w,3) with alpha (h,w) through affine M
    (sprite px -> screen px). rgb None = a pure shadow (darken only)."""
    H, W = img.shape[:2]
    h, w = (alpha if alpha is not None else rgb).shape[:2]
    cs = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ np.asarray(M, np.float64).T
    x0, y0 = int(np.floor(cs[:, 0].min())) - 1, int(np.floor(cs[:, 1].min())) - 1
    x1, y1 = int(np.ceil(cs[:, 0].max())) + 1, int(np.ceil(cs[:, 1].max())) + 1
    if region is not None:
        x0, y0 = max(x0, region[0]), max(y0, region[1])
        x1, y1 = min(x1, region[2]), min(y1, region[3])
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    Mt = np.array(M, np.float64).copy()
    Mt[0, 2] -= x0
    Mt[1, 2] -= y0
    sz = (x1 - x0, y1 - y0)
    reg = img[y0:y1, x0:x1]
    if alpha is not None:
        al = cv2.warpAffine(alpha, Mt, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        reg *= (1.0 - al)[..., None]
    if rgb is not None:
        reg += cv2.warpAffine(rgb, Mt, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _rot_M(theta, scale, cx, cy, px, py):
    """Affine: sprite px (centre cx,cy) -> rotate theta, scale -> screen (px,py)."""
    c, s = math.cos(theta) * scale, math.sin(theta) * scale
    return np.array([[c, -s, px - c * cx + s * cy],
                     [s, c, py - s * cx - c * cy]], np.float64)


def _ease_back_diff(L, overshoot=1.0):
    """Increments of an ease-out-back step 0 -> 1 over L frames: snaps past the
    mark, settles back (the escapement)."""
    c1 = 1.70158 * overshoot
    c3 = c1 + 1.0
    u = np.linspace(0.0, 1.0, L + 1)
    f = 1.0 + c3 * (u - 1.0) ** 3 + c1 * (u - 1.0) ** 2
    return np.diff(f)


def _ease_out_diff(L, p=3.0):
    u = np.linspace(0.0, 1.0, L + 1)
    return np.diff(1.0 - (1.0 - u) ** p)


# ---------------------------------------------------------------------------
# the whole-song timeline (cached)
# ---------------------------------------------------------------------------
def _timeline(R, a, lv, L):
    T = dict(lv.stage_timeline(a, R.fps, "acid", label="steampunk"))
    fps = R.fps
    n = len(a.rms)
    t = np.arange(n, dtype=f32) / fps
    I, hot, big, pre, black = T["I"], T["hot"], T["big"], T["pre"], T["black"]
    E = np.clip(np.maximum(I, 0.8 * pre), 0, 1).astype(f32)
    E2 = lv._gauss_smooth(E, fps, 1.0)
    hot_f = lv._gauss_smooth(hot.astype(f32), fps, 0.12)
    big_f = lv._gauss_smooth(big.astype(f32), fps, 0.25)
    heat = np.clip(lv._gauss_smooth(a.bass, fps, 0.12) * 1.15, 0, 1).astype(f32)
    hs = lv._gauss_smooth(a.high, fps, 0.35)
    spark = np.clip((np.asarray(a.high, f32) - hs) * 5.0, 0, 1).astype(f32)
    kage = T["kage"]
    kenv = np.where(kage < 40, 0.72 ** np.minimum(kage, 40), 0.0).astype(f32)
    beat_f = fps * 60.0 / max(60.0, float(a.bpm) or 120.0)
    kick = np.asarray(a.kick, bool)
    ea = T["entry_age"]
    # ---- MAIN GEAR: drift that speeds with intensity + escapement ticks
    # drift stays SMALL next to the ticks so the escapement reads as steps
    spd = (0.04 + 0.12 * E2 + 0.08 * hot_f + 0.22 * big_f) / fps
    tick = np.zeros(n, np.float64)
    Lt = max(3, int(round(0.22 * fps)))
    dB = _ease_back_diff(Lt, 1.35)
    pitch = 2 * np.pi / 40
    for f in np.flatnonzero(kick):
        st = pitch * (2.0 if big[f] else (1.0 if hot[f] or pre[f] > 0.3 else 0.5))
        seg = tick[f:f + Lt]
        seg += st * dB[:len(seg)]
    theta = np.cumsum(spd.astype(np.float64) + tick)
    # ---- ENGINES: half a crank turn per kick (eased out: the stroke snaps),
    # plus an idle turn that follows the energy
    phi_k = np.zeros(n, np.float64)
    Lp = max(3, int(round(0.30 * fps)))
    dP = _ease_out_diff(Lp, 3.0)
    for f in np.flatnonzero(kick):
        seg = phi_k[f:f + Lp]
        seg += np.pi * (1.0 if (hot[f] or pre[f] > 0.2) else 0.5) * dP[:len(seg)]
    phi = np.cumsum((0.35 + 1.2 * E2) / fps + phi_k) + np.pi / 2
    # ---- BULBS: spectrum -> 14 bulbs (log-ish band groups)
    bars = np.asarray(a.bars, f32)
    sm = bars.copy()
    sm[1:] = 0.55 * bars[1:] + 0.45 * bars[:-1]
    nb = L["bulbs"]["n"]
    nbar = bars.shape[1]
    edges = np.round(np.linspace(0.0, np.sqrt(nbar), nb + 1) ** 2).astype(int)
    for k in range(1, nb + 1):                     # strictly increasing, ends at nbar
        edges[k] = max(edges[k], edges[k - 1] + 1)
    edges = np.minimum(edges, nbar - (nb - np.arange(nb + 1)))
    edges[-1] = nbar
    bl = np.stack([sm[:, edges[k]:edges[k + 1]].mean(1) for k in range(nb)], 1)
    bl = bl / np.maximum(np.percentile(bl, 97, axis=0), 1e-3)
    bulbs = np.clip(bl * (0.45 + 0.55 * E[:, None] + 0.25 * hot_f[:, None]), 0, 1.2).astype(f32)
    # ---- GAUGES: 6 bands, spring-damped needles (overshoot), red on builds
    gb = [(0, 2), (2, 5), (5, 10), (10, 18), (18, 28), (28, 40)]
    tgt = np.stack([sm[:, lo:hi].mean(1) for lo, hi in gb], 1)
    tgt = tgt / np.maximum(np.percentile(tgt, 96, axis=0), 1e-3)
    tgt = np.clip(0.08 + 0.62 * tgt * (0.55 + 0.45 * E[:, None]) + 0.10 * hot_f[:, None]
                  + 0.62 * (pre[:, None] ** 1.3), 0, 1.10)
    tgt[black] = 0.02                         # power cut: pressure gone
    om, ze = 2 * np.pi * 3.3, 0.28
    x = np.zeros(6)
    v = np.zeros(6)
    gauge = np.zeros((n, 6), f32)
    sub = 4
    dt = 1.0 / (fps * sub)
    for i in range(n):
        for _ in range(sub):
            acc = om * om * (tgt[i] - x) - 2 * ze * om * v
            v += acc * dt
            x += v * dt
            hi = x > 1.12
            v[hi] = -0.35 * v[hi]
            x[hi] = 1.12
            lo = x < 0.0
            v[lo] = -0.3 * v[lo]
            x[lo] = 0.0
        gauge[i] = x
    # ---- WARNING LAMPS: blink rate rises with the build (phase = cumsum)
    freq = np.where(pre > 0, 1.2 + 7.0 * pre ** 1.6, 0.6)
    lphase = np.cumsum(freq / fps)
    # ---- NIXIE pressure: climbs through the build to 9999 at the drop
    psi = (60 + 900 * lv._gauss_smooth(E, fps, 0.6) + 2600 * pre ** 2
           + 6400 * pre ** 6 + 1500 * hot_f * lv._gauss_smooth(heat, fps, 0.08))
    psi = np.clip(psi, 0, 9999).astype(np.int64)
    # ---- CAMERA: depth moves only (push in on builds, blown back on drops)
    pmax = 0.0
    punch = np.zeros(n, f32)
    if R.lyric_zoom is not None:
        pmax = float(np.max(R.lyric_zoom)) - 1.0
        if pmax > 1e-4:
            punch = np.clip((np.asarray(R.lyric_zoom, f32) - 1.0) / pmax, 0, 1)
    blow = np.where(ea < 10 ** 5, np.exp(-np.minimum(ea, 10 ** 5) / (0.35 * fps)), 0.0)
    blow = np.where(hot, blow, 0.0).astype(f32)
    cz = (0.025 * np.sin(2 * np.pi * t / 19.0) + 0.17 * pre ** 1.6
          - 0.15 * blow * (1.4 * big + 1.0 * ~big) + 0.04 * hot_f + 0.03 * big_f
          + 0.018 * kenv * hot_f + 0.10 * punch)
    cz = np.clip(lv._gauss_smooth(cz, fps, 0.04), -0.14, 0.24).astype(f32)
    camx = (0.030 * np.sin(2 * np.pi * t / 23.0 + 1.0)).astype(f32)
    camy = (0.016 * np.sin(2 * np.pi * t / 29.0)).astype(f32)
    # ---- clock minute hand: a notch per bar, eased
    ones = T["gbeat"] & (T["bi"] % 4 == 0)
    mh = np.zeros(n, np.float64)
    Lm = max(2, int(round(0.18 * fps)))
    dM = _ease_back_diff(Lm, 0.6)
    for f in np.flatnonzero(ones):
        seg = mh[f:f + Lm]
        seg += (2 * np.pi / 60) * dM[:len(seg)]
    minute = np.cumsum(mh) - np.pi / 2 + 0.7
    # ---- STEAM EVENTS (frame, vent kind, strength, seed)
    ev = []
    vents = L["vents"]
    rs = np.random.default_rng(1881)
    sn = np.flatnonzero(np.asarray(a.snare, bool))
    for k, f in enumerate(sn):
        if black[f]:
            continue
        nv = 3 if big[f] else (2 if hot[f] else 1)
        if not hot[f] and pre[f] < 0.2 and rs.random() < 0.45:
            continue
        r2 = np.random.default_rng(5000 + k)
        for vi in r2.choice(len(vents), nv, replace=False):
            ev.append((int(f), "v", int(vi), 0.55 + 0.45 * float(hot_f[f]) + 0.2 * big[f],
                       5000 + k * 7 + int(vi)))
    for f in range(0, n, 4):                       # build: hissing leaks
        if pre[f] > 0.15 and not black[f]:
            r2 = np.random.default_rng(9000 + f)
            ev.append((f, "v", int(r2.integers(len(vents))), 0.18 + 0.30 * float(pre[f]),
                       9000 + f))
    for k, s0 in enumerate(T["entries"]):         # drop entry: valves blow
        s = 1.6 if big[s0] else 1.25
        for vi in range(len(vents)):
            ev.append((int(s0), "v", vi, s, 100 * k + vi))
        for wi in range(2):
            ev.append((int(s0), "w", wi, s, 100 * k + 50 + wi))
            ev.append((int(s0), "c", wi, s, 100 * k + 60 + wi))
    if pmax > 1e-4:                                # sung "bump": whistles
        on = np.flatnonzero((punch[1:] > 0.25) & (punch[:-1] <= 0.25)) + 1
        for k, f in enumerate(on):
            for wi in range(2):
                ev.append((int(f), "w", wi, 0.9, 7000 + 3 * k + wi))
    for f in np.flatnonzero(ones & hot):           # in drops: cylinder cocks on the ONE
        ev.append((int(f), "c", int(T["bar"][f] % 2), 0.55 + 0.35 * big[f], 8000 + int(f)))
    ev.sort(key=lambda e: e[0])
    # ---- SPARK EVENTS (frame, kind, strength, seed): hats + entries + big ONEs
    sp_ev = []
    last = -99
    for f in range(n):
        if spark[f] > 0.32 and f - last >= 3 and not black[f]:
            sp_ev.append((f, "h", float(spark[f]), 300 + f))
            last = f
    for k, s0 in enumerate(T["entries"]):
        sp_ev.append((int(s0), "e", 1.6 if big[s0] else 1.0, 40 + k))
    for f in np.flatnonzero(ones & big):
        sp_ev.append((int(f), "m", 1.0, 700 + int(f)))
    sp_ev.sort(key=lambda e: e[0])
    # ---- SHAKE (like the megacity): entries hard, finale ONEs medium
    shake = np.zeros(n, f32)
    shake[ones & hot] = 0.25
    shake[ones & big] = 0.5
    for s0 in T["entries"]:
        shake[s0] = 1.0
    sh_age, sh_idx = lv._age_since(shake > 0)
    T.update(E=E, E2=E2, hot_f=hot_f, big_f=big_f, heat=heat, spark=spark, kenv=kenv,
             theta=theta, phi=phi, bulbs=bulbs, gauge=gauge, lphase=lphase, psi=psi,
             cz=cz, camx=camx, camy=camy, minute=minute, punch=punch, pmax=pmax,
             ev=ev, ev_f=np.array([e[0] for e in ev] or [10 ** 9]),
             sp=sp_ev, sp_f=np.array([e[0] for e in sp_ev] or [10 ** 9]),
             shake=shake, sh_age=sh_age, sh_idx=sh_idx, beat_f=beat_f, blow=blow)
    return T


# ---------------------------------------------------------------------------
# geometry (built once per render size)
# ---------------------------------------------------------------------------
def _gear_chain(L):
    """Centres + phase rule for the meshing trains."""
    m = L["module"]
    out = {}
    for name, spr, N, parent, ang in L["train"]:
        if parent is None:
            out[name] = dict(c=tuple(L["main_c"]), N=N, spr=spr, parent=None, sc=1.0, d="main")
        else:
            p = out[parent]
            dist = m * (p["N"] + N) / 2.0
            a_ = math.radians(ang)
            out[name] = dict(c=(p["c"][0] + dist * math.cos(a_), p["c"][1] + dist * math.sin(a_)),
                             N=N, spr=spr, parent=parent, ang=a_, sc=1.0, d="main")
    for name, spr, N, parent, ang, c, sc in L["far"]:
        if parent is None:
            out[name] = dict(c=tuple(c), N=N, spr=spr, parent=None, sc=sc, d="far")
        else:
            p = out[parent]
            dist = m * sc * (p["N"] + N) / 2.0
            a_ = math.radians(ang)
            out[name] = dict(c=(p["c"][0] + dist * math.cos(a_), p["c"][1] + dist * math.sin(a_)),
                             N=N, spr=spr, parent=parent, ang=a_, sc=sc, d="far")
    return out


def _gear_angles(G, root_theta):
    """theta per gear, from the root angle; meshing is EXACT (no drift): the
    child's tooth phase at the contact is the parent's, mirrored by half a
    pitch, so a tooth always sits in a gap."""
    th = {}
    for name, g in G.items():
        if g["parent"] is None:
            th[name] = root_theta.get(name, 0.0)
        else:
            p = G[g["parent"]]
            tp = th[g["parent"]]
            Np, Nc = p["N"], g["N"]
            fp = ((g["ang"] - tp) / (2 * np.pi / Np)) % 1.0
            th[name] = g["ang"] + np.pi - (2 * np.pi / Nc) * (0.5 - fp)
    return th


def _logo_masks(R, size):
    if R.logo is None:
        return None
    rgb = np.asarray(R.logo.convert("RGB"), f32) / 255.0
    al = np.asarray(R.logo.split()[-1], f32) / 255.0
    lum = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    if al.std() < 0.02:
        al = 1.0 - ((lum > 0.93) & (sat < 0.10)).astype(f32)
    rd = ((rgb[..., 0] > 0.5) & (rgb[..., 1] < 0.35) & (rgb[..., 2] < 0.40)).astype(f32) * al
    dk = (lum < 0.35).astype(f32) * al * (1 - rd)
    fl = np.clip(al - rd - dk, 0, 1)
    h, w = al.shape
    s = size / max(h, w)
    sz = (max(1, int(w * s)), max(1, int(h * s)))
    return [cv2.resize(m, sz, interpolation=cv2.INTER_AREA) for m in (dk, fl, rd)]


def _geo(R, lv):
    C = R.scene_cache
    key = ("sp_geo", R.W, R.H, R.scene_palette)
    if key in C:
        return C[key]
    man = _manifest()
    L = man["layout"]
    A = man["assets"]
    P = lv.scene_palette(R, sys.modules[__name__])
    H = R.H
    LT = _lights(L)
    G = dict(L=L, P=P, LT=LT)
    # ---- static plates at their target resolution
    plates = {}
    for name, depth, ppu_f, lights in (("wall", "wall", 1.0, "wall"),
                                       ("pipes", "pipes", 1.08, "pipes"),
                                       ("engine", "main", 1.12, "main"),
                                       ("backhead", "backhead", 1.15, "backhead"),
                                       ("plate", "backhead", 1.25, "plate")):
        meta = A[name]
        tgt = H * ppu_f
        sc = tgt / meta["ppu"]
        for mir in ((False, True) if name == "engine" else (False,)):
            M = _load_maps(name, sc, mirror=mir)
            h, w = M["a"].shape
            ppu = w / (meta["x"][1] - meta["x"][0])
            x0 = meta["x"][0] if not mir else -meta["x"][1]
            y0 = meta["y"][0]
            S = _plate_S(M, P, LT[lights], x0, y0, ppu)
            plates[name + ("_L" if mir else "")] = dict(S=S["S"], chs=S["chs"], a=M["a"], x0=x0,
                                                         y0=y0, ppu=ppu, d=L["depth"][depth])
    G["plates"] = plates
    # medallion: logo enamel inlaid into the albedo before lighting
    meta = A["medallion"]
    sc = H * 1.15 / meta["ppu"]
    M = _load_maps("medallion", sc)
    h, w = M["a"].shape
    ppu = w / (meta["x"][1] - meta["x"][0])
    alb, spc, ao = _surface(M, P)
    face_r = (L["medallion_r"] - 0.014) * ppu
    lm = _logo_masks(R, face_r * 2 * 0.80)
    emis = np.zeros((h, w), f32)
    if lm is not None:
        dk, fl, rd = lm
        lh, lw = dk.shape
        y0, x0 = (h - lh) // 2, (w - lw) // 2
        full = []
        for m in (dk, fl, rd):
            z = np.zeros((h, w), f32)
            z[y0:y0 + lh, x0:x0 + lw] = m
            full.append(z)
        dk, fl, rd = full
        for m, col, sk in ((fl, P["en_fill"], 0.35), (rd, P["en_red"], 0.30), (dk, P["en_line"], 0.08)):
            alb = alb * (1 - m[..., None]) + (_c(col) / 255.0) * m[..., None]
            spc = spc * (1 - m[..., None] * (1 - sk))
        # the enamel sits in engraved cells: emboss the line work into the normals
        bl = cv2.GaussianBlur(dk + 0.5 * rd, (0, 0), max(0.8, ppu * 0.0012))
        gx = cv2.Sobel(bl, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(bl, cv2.CV_32F, 0, 1, ksize=3)
        n = M["n"].copy()
        n[..., 0] -= gx * 1.6
        n[..., 1] -= gy * 1.6
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        M["n"] = n
        emis = cv2.GaussianBlur(rd, (0, 0), 1.0)
    # bake: reuse _plate_S via a surface override
    M2 = dict(M)
    G["med_alb"] = (alb, spc, ao)
    S = _plate_S_custom(M2, alb, spc, ao, LT["main"], meta["x"][0], meta["y"][0], ppu)
    G["medallion"] = dict(S=S["S"], chs=S["chs"], a=M["a"], ppu=ppu, emis=emis * M["a"], w=w, h=h)
    # ---- gears: matcaps at the needed scale per USE
    gears = {}
    GC = _gear_chain(L)
    for nm in set(g["spr"] for g in GC.values()) | {"flywheel", "g24", "g20"}:
        meta = A[nm]
        gears[nm] = {}
    G["chain"] = GC
    G["gear_sets"] = {}
    for gname, g in GC.items():
        res_f = 1.12 if g["d"] == "main" else 0.55
        key2 = (g["spr"], g["sc"], res_f)
        if key2 not in G["gear_sets"]:
            meta = A[g["spr"]]
            tgt = H * res_f * g["sc"]
            M = _load_maps(g["spr"], tgt / meta["ppu"])
            G["gear_sets"][key2] = _Matcap(M, P, tgt)
        g["mc"] = G["gear_sets"][key2]
        g["res_f"] = res_f
    meta = A["flywheel"]
    G["fly"] = _Matcap(_load_maps("flywheel", H * 1.12 / meta["ppu"]), P, H * 1.12)
    near = []
    for name, spr, N, c, sc, dirn in L["near"]:
        meta = A[spr]
        tgt = H * 0.25 * sc * 1.25                    # quarter res: it gets blurred
        near.append(dict(name=name, mc=_Matcap(_load_maps(spr, tgt / meta["ppu"]), P, tgt),
                         c=c, sc=sc, dir=dirn, N=N, ppu=tgt / sc))
    G["near"] = near
    # ---- window glass (emissive), in wall-sprite pixels
    wl = plates["wall"]
    wx, wy = L["window"]["c"]
    wr = L["window"]["r"]
    ppu = wl["ppu"]
    cxp, cyp = (wx - wl["x0"]) * ppu, (wy - wl["y0"]) * ppu
    rp = (wr + 0.012) * ppu
    gx0, gy0 = int(cxp - rp) - 2, int(cyp - rp) - 2
    gs = int(2 * rp) + 4
    yy, xx = np.mgrid[0:gs, 0:gs].astype(f32)
    dx, dy = xx + gx0 - cxp, yy + gy0 - cyp
    r = np.sqrt(dx * dx + dy * dy) / (wr * ppu)
    ang = np.arctan2(dy, dx)
    ring = np.digitize(r, [0.30, 0.62])
    sect = np.floor(((ang + np.pi) / (2 * np.pi)) * 12).astype(int) % 12
    pane = 0.82 + 0.30 * lv._hash01(ring * 13.0 + sect, 3.0)
    rng = np.random.default_rng(77)
    fro = cv2.resize(rng.random((max(4, gs // 18), max(4, gs // 18))).astype(f32), (gs, gs),
                     interpolation=cv2.INTER_CUBIC)
    moon = np.exp(-(((dx / (wr * ppu)) - 0.30) ** 2 + ((dy / (wr * ppu)) + 0.34) ** 2) / 0.05)
    g = np.clip(1.0 - r ** 1.6, 0, 1)
    glass = (_c(P["glass_e"]) + (_c(P["glass_c"]) - _c(P["glass_e"])) * (g ** 1.2)[..., None])
    glass = glass * (pane * (0.86 + 0.28 * fro) + 0.5 * moon)[..., None]
    glass *= (r < 1.03)[..., None]
    # mirrored roman numerals painted on the outer panes (seen from inside)
    from PIL import Image, ImageDraw
    num = Image.new("L", (gs, gs), 0)
    dr = ImageDraw.Draw(num)
    fnt = R.font(max(8, int(wr * ppu * 0.16)), "Anton-Regular.ttf")
    romans = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    for k, s_ in enumerate(romans):
        a_ = -np.pi / 2 + 2 * np.pi * k / 12
        tx, ty = gs / 2 + 0.80 * wr * ppu * math.cos(a_), gs / 2 + 0.80 * wr * ppu * math.sin(a_)
        bb = dr.textbbox((0, 0), s_, font=fnt)
        dr.text((tx - (bb[2] - bb[0]) / 2 - bb[0], ty - (bb[3] - bb[1]) / 2 - bb[1]), s_,
                font=fnt, fill=255)
    num = np.asarray(num, f32)[:, ::-1] / 255.0
    glass *= (1.0 - 0.80 * num)[..., None]
    G["glass"] = dict(img=glass.astype(f32), x0=gx0, y0=gy0, cx=cxp, cy=cyp, R=wr * ppu)
    # ---- instruments: dial faces + Nixie glyphs
    G["dials"] = _dials(R, L, P, H * 1.15)
    G["nixie"] = _nixie_glyphs(R, L, H * 1.15)
    # ---- screen-space light maps at quarter res (for steam + haze)
    qh, qw = max(4, R.H // 4), max(4, R.W // 4)
    ys = (np.arange(qh, dtype=f32) + 0.5) / qh - 0.5
    xs = ((np.arange(qw, dtype=f32) + 0.5) / qw - 0.5) * (R.W / R.H)
    X, Y = np.meshgrid(xs, ys)
    G["q"] = (qh, qw)
    G["map_fire"] = (1.0 / (1.0 + ((X / 0.55) ** 2 + ((Y - 0.55) / 0.42) ** 2) * 2.0))[..., None]
    G["map_win"] = np.exp(-((X - wx) ** 2 + (Y - wy) ** 2) / (2 * 0.30 ** 2))[..., None]
    G["map_top"] = np.clip(0.5 - Y, 0, 1)[..., None] ** 1.5
    G["XY"] = (X, Y)
    # haze gradients (full res)
    yf = np.linspace(0, 1, R.H, dtype=f32)[:, None, None]
    G["hz"] = (0.55 + 0.45 * np.clip(1 - np.abs(yf - 0.55) / 0.55, 0, 1)).astype(f32)
    G["hz_fire"] = (np.clip((yf - 0.45) / 0.55, 0, 1) ** 1.5).astype(f32)
    # steam turbulence + fire noise textures (tileable-ish)
    rn = np.random.default_rng(31)
    tex = np.zeros((128, 128), f32)
    for gsz, amp in ((4, 1.0), (8, 0.55), (16, 0.3), (32, 0.15)):
        z = rn.random((gsz, gsz)).astype(f32)
        z = np.tile(z, (3, 3))
        up = cv2.resize(z, (384, 384), interpolation=cv2.INTER_CUBIC)[128:256, 128:256]
        tex += amp * up
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    G["tex"] = tex
    # gaussian puff kernels (by radius in quarter-res px)
    G["kern"] = {}
    # bulbs
    B = L["bulbs"]
    bx = np.linspace(B["x0"], B["x1"], B["n"])
    by = np.where(np.arange(B["n"]) % 2 == 0, B["y_lo"], B["y_hi"]) + 0.02 * np.sin(np.arange(B["n"]) * 2.1)
    G["bulb_xy"] = np.stack([bx, by], 1)
    C[key] = G
    return G


def _plate_S_custom(M, alb, spc, ao, lights, x0, y0, ppu):
    """_plate_S with a given surface (the medallion's enamel)."""
    n, a = M["n"], M["a"]
    h, w = a.shape
    xs = x0 + (np.arange(w, dtype=f32) + 0.5) / ppu
    ys = y0 + (np.arange(h, dtype=f32) + 0.5) / ppu
    X, Y = np.meshgrid(xs, ys)
    acc = {"amb": alb * (ao * (0.55 + 0.45 * n[..., 2]))[..., None]}
    for ch, lx, ly, dz, R_, pw in lights:
        # the medallion sits at the main gear centre: lights are relative to it
        Lx, Ly, Lz = lx - X, ly - (Y - 0.08), np.full_like(X, dz)
        d = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz) + 1e-6
        Lx, Ly, Lz = Lx / d, Ly / d, Lz / d
        F = pw / (1.0 + (d / R_) ** 2)
        nl = np.clip(n[..., 0] * Lx + n[..., 1] * Ly + n[..., 2] * Lz, 0, None)
        Hz = Lz + 1.0
        hn = np.sqrt(Lx * Lx + Ly * Ly + Hz * Hz) + 1e-6
        nh = np.clip((n[..., 0] * Lx + n[..., 1] * Ly + n[..., 2] * Hz) / hn, 0, None)
        sp = (nh ** SPEC_P) * 1.6 + (nh ** 6) * 0.18
        img = alb * (ao * nl * F)[..., None] + spc * (sp * F)[..., None]
        acc[ch] = acc.get(ch, 0.0) + img
    return _pack(acc, a)


def _dials(R, L, P, ppu):
    """Gauge faces (RGB, alpha) — cream dial, ticks, red zone, band label."""
    from PIL import Image, ImageDraw
    fr = L["face_r"]
    s = int(round(2 * fr * ppu)) + 2
    labels = ("SUB", "BASS", "LOW", "MID", "HIGH", "AIR")
    out = []
    ss = 3
    S = s * ss
    for k in range(len(L["gauges"])):
        im = Image.new("RGB", (S, S), (0, 0, 0))
        al = Image.new("L", (S, S), 0)
        d = ImageDraw.Draw(im)
        da = ImageDraw.Draw(al)
        c = S / 2
        rr = fr * ppu * ss
        da.ellipse((c - rr, c - rr, c + rr, c + rr), fill=255)
        d.ellipse((c - rr, c - rr, c + rr, c + rr), fill=tuple(P["dial"]))
        # red zone arc 0.78..1.0
        a0 = 135 + 270 * 0.78
        d.arc((c - rr * 0.86, c - rr * 0.86, c + rr * 0.86, c + rr * 0.86), a0, 135 + 270,
              fill=tuple(P["red_zone"]), width=max(2, int(rr * 0.13)))
        for j in range(51):
            v = j / 50.0
            a_ = math.radians(135 + 270 * v)
            major = j % 5 == 0
            r0 = rr * (0.70 if major else 0.78)
            r1 = rr * 0.88
            d.line((c + r0 * math.cos(a_), c + r0 * math.sin(a_), c + r1 * math.cos(a_),
                    c + r1 * math.sin(a_)), fill=tuple(P["ink"]),
                   width=max(1, int(rr * (0.035 if major else 0.016))))
        fnt = R.font(max(6, int(rr * 0.17)), "BebasNeue-Regular.ttf")
        for j in range(0, 11, 2):
            a_ = math.radians(135 + 270 * j / 10)
            tx, ty = c + rr * 0.54 * math.cos(a_), c + rr * 0.54 * math.sin(a_)
            t_ = str(j)
            bb = d.textbbox((0, 0), t_, font=fnt)
            d.text((tx - (bb[2] + bb[0]) / 2, ty - (bb[3] + bb[1]) / 2), t_, font=fnt,
                   fill=tuple(P["ink"]))
        f2 = R.font(max(6, int(rr * 0.24)), "BebasNeue-Regular.ttf")
        bb = d.textbbox((0, 0), labels[k], font=f2)
        d.text((c - (bb[2] + bb[0]) / 2, c + rr * 0.40 - (bb[3] + bb[1]) / 2), labels[k], font=f2,
               fill=tuple(P["ink"]))
        f3 = R.font(max(5, int(rr * 0.11)), "BebasNeue-Regular.ttf")
        bb = d.textbbox((0, 0), "PSI x100", font=f3)
        d.text((c - (bb[2] + bb[0]) / 2, c - rr * 0.36 - (bb[3] + bb[1]) / 2), "PSI x100", font=f3,
               fill=tuple(P["ink"]))
        rgb = np.asarray(im.resize((s, s), Image.LANCZOS), f32)
        a = np.asarray(al.resize((s, s), Image.LANCZOS), f32) / 255.0
        # inner shadow under the bezel + slight vignette (depth)
        yy, xx = np.mgrid[0:s, 0:s].astype(f32)
        rn = np.sqrt((xx - (s - 1) / 2) ** 2 + (yy - (s - 1) / 2) ** 2) / (fr * ppu)
        sh = np.clip(1.0 - np.clip((rn - 0.78) / 0.22, 0, 1) ** 2 * 0.6, 0.35, 1)
        sh *= 1.0 - 0.25 * np.clip((yy - xx) / s, -1, 1) * 0.0
        rgb *= sh[..., None]
        glint = np.clip(1 - np.abs(rn - 0.80) / 0.07, 0, 1) * np.clip(-(xx + yy - s) / s * 2.2, 0, 1)
        out.append(dict(rgb=rgb * a[..., None], a=a, s=s, glint=(glint * a).astype(f32)))
    return out


def _nixie_glyphs(R, L, ppu):
    from PIL import Image, ImageDraw
    Nx = L["nixie"]
    tw = Nx["w"] * 0.92 / Nx["n"]
    th = Nx["h"] * 0.78
    w, h = max(6, int(tw * ppu)), max(8, int(th * ppu))
    fnt = R.font(max(6, int(h * 0.92)), "Poppins-Light.ttf")
    out = []
    for dch in "0123456789":
        im = Image.new("L", (w * 2, h * 2), 0)
        d = ImageDraw.Draw(im)
        f2 = R.font(max(6, int(h * 1.6)), "Poppins-Light.ttf")
        bb = d.textbbox((0, 0), dch, font=f2)
        d.text((w - (bb[2] + bb[0]) / 2, h - (bb[3] + bb[1]) / 2), dch, font=f2, fill=255)
        m = np.asarray(im.resize((w, h), Image.LANCZOS), f32) / 255.0
        out.append(m)
    del fnt
    return dict(g=np.stack(out), w=w, h=h, tw=tw, th=th)


# ---------------------------------------------------------------------------
# per-frame helpers
# ---------------------------------------------------------------------------
def _proj(R, T, i, d):
    """(k, ox, oy): layer units -> screen px for depth d at frame i."""
    cz = float(T["cz"][i])
    z = d / (d - cz)
    k = R.H * z
    par = 2.0 / d
    ox = R.W / 2.0 - k * float(T["camx"][i]) * par
    oy = R.H / 2.0 - k * float(T["camy"][i]) * par
    return k, ox, oy


def _cols(T, P, i):
    """Per-frame light colours for every channel."""
    black = bool(T["black"][i])
    E = float(T["E"][i])
    heat = float(T["heat"][i])
    hot_f = float(T["hot_f"][i])
    big = float(T["big_f"][i])
    kenv = float(T["kenv"][i])
    ea = int(T["entry_age"][i])
    flare = 0.85 ** ea if ea < 30 else 0.0
    bulbs = float(np.mean(T["bulbs"][i]))
    lp = float(T["lphase"][i]) % 1.0
    pre = float(T["pre"][i])

    def mix(a_, b_):
        return _c(P[a_]) * (1 - big) + _c(P[b_]) * big
    c = {}
    c["amb"] = mix("amb", "big_amb") * ((0.35 if black else 1.0) * (0.85 + 0.30 * E))
    fire = 0.26 + 1.00 * heat + 0.45 * kenv * hot_f + 0.45 * flare
    c["fire"] = mix("fire", "big_fire") * (0.12 if black else fire)
    c["key"] = mix("key", "big_key") * (0.0 if black else (0.15 + 0.95 * bulbs + 0.35 * flare))
    c["back"] = mix("back", "big_back") * ((0.30 if black else 1.0) * (0.60 + 0.35 * E + 0.30 * kenv * hot_f))
    c["spot"] = _c(P["spot"]) * ((0.25 if black else 1.0) * (0.85 + 0.45 * kenv))
    lamp = _lamp_level(T, i)
    c["red"] = _c(P["red"]) * lamp
    c["amb2"] = c["amb"] * 1.0
    del lp, pre
    return c


def _lamp_level(T, i):
    if T["black"][i]:
        return 0.0
    pre = float(T["pre"][i])
    hot = bool(T["hot"][i])
    if hot:
        ph = float(T["ph"][i])
        if T["big"][i]:
            ph = (ph * 2.0) % 1.0
        return 1.0 if ph < 0.5 else 0.12
    if pre > 0:
        return 1.0 if (float(T["lphase"][i]) % 1.0) < 0.5 else 0.08
    return 0.06


def _lit_plate(pl, cols, crop=None):
    """sum_l colour_l * S_l as one matmul, over the crop (y0, y1, x0, x1)."""
    chs = pl["chs"]
    B = np.zeros((3 * len(chs), 3), f32)
    for k, ch in enumerate(chs):
        c = cols.get(ch)
        if c is not None:
            B[3 * k, 0], B[3 * k + 1, 1], B[3 * k + 2, 2] = c
    S = pl["S"] if crop is None else pl["S"][crop[0]:crop[1], crop[2]:crop[3]]
    return np.matmul(S, B)


def _draw_plate(img, R, T, i, pl, cols, extra=None, region=None):
    k, ox, oy = _proj(R, T, i, pl["d"])
    s = k / pl["ppu"]
    tx = k * (pl["x0"] + 0.5 / pl["ppu"]) + ox
    ty = k * (pl["y0"] + 0.5 / pl["ppu"]) + oy
    h, w = pl["a"].shape
    # light only what lands in the frame
    u0 = max(0, int((0 - tx) / s) - 2)
    u1 = min(w, int((R.W - tx) / s) + 3)
    v0 = max(0, int((0 - ty) / s) - 2)
    v1 = min(h, int((R.H - ty) / s) + 3)
    if u1 <= u0 or v1 <= v0:
        return k, ox, oy
    rgb = _lit_plate(pl, cols, (v0, v1, u0, u1))
    if extra is not None:
        rgb = extra(rgb, u0, v0)
    M = np.array([[s, 0, tx + s * u0], [0, s, ty + s * v0]], np.float64)
    _blit(img, rgb, pl["a"][v0:v1, u0:u1], M, region)
    return k, ox, oy


def _haze(img, G, P, cols, amt):
    h = G["hz"] * amt                                   # (H, 1, 1)
    fog = _c(P["fog"]) * (0.6 + 0.6 * float(np.mean(cols["back"])))
    add = h * fog + (h * G["hz_fire"]) * (cols["fire"] * 16.0)   # (H, 1, 3)
    img *= (1.0 - h)
    img += add


def _gear_lights(G, which, x, y):
    out = []
    for ch, lx, ly, dz, R_, pw in G["LT"][which]:
        Lx, Ly, Lz = lx - x, ly - y, dz
        d = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz) + 1e-6
        out.append((ch, Lx / d, Ly / d, Lz / d, pw / (1.0 + (d / R_) ** 2)))
    return out


def _draw_gear(img, mc, lights, cols, theta, k, ox, oy, x, y, sprite_ppu, fade=None):
    rgb = mc.shade(lights, cols, theta)
    a = mc.a
    if fade is not None:              # haze: mix toward the fog colour
        amt, fc = fade
        rgb = rgb * (1 - amt) + fc * (a[..., None] * amt)
    M = _rot_M(theta, k / sprite_ppu, mc.cx, mc.cy, k * x + ox, k * y + oy)
    _blit(img, rgb, a, M)


def _kernel(G, r):
    r = max(1, int(round(r)))
    K = G["kern"].get(r)
    if K is None:
        s = 2 * r + 1
        yy, xx = np.mgrid[0:s, 0:s].astype(f32) - r
        K = np.exp(-(xx * xx + yy * yy) / (2 * (0.45 * r) ** 2)).astype(f32)
        G["kern"][r] = K
    return K


def _add_k(D, K, cx, cy, w):
    r = K.shape[0] // 2
    h, W = D.shape
    x0, y0 = int(cx) - r, int(cy) - r
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + K.shape[1]), min(h, y0 + K.shape[0])
    if xb <= xa or yb <= ya:
        return
    D[ya:yb, xa:xb] += K[ya - y0:yb - y0, xa - x0:xb - x0] * w


# ---- steam ---------------------------------------------------------------
_PLUME = {
    #       jet speed, decel tau, burst s, puffs, life s, r0, growth, rise
    "v": (1.00, 0.22, 0.30, 12, 1.5, 0.010, 0.11, 0.10),
    "w": (1.60, 0.18, 0.45, 16, 1.4, 0.008, 0.09, 0.14),
    "c": (1.20, 0.20, 0.35, 12, 1.2, 0.010, 0.10, 0.05),
}


def _comp_q(img, op, col, k_op, k_col):
    """Composite a quarter-res (opacity, premultiplied colour) layer over the
    frame, upsampling only its bounding box."""
    qh, qw = op.shape
    rows = np.flatnonzero(op.max(1) > 0.004)
    cols_ = np.flatnonzero(op.max(0) > 0.004)
    if len(rows) == 0:
        return
    y0, y1 = max(0, rows[0] - 2), min(qh, rows[-1] + 3)
    x0, x1 = max(0, cols_[0] - 2), min(qw, cols_[-1] + 3)
    H, W = img.shape[:2]
    sy, sx = H / qh, W / qw
    Y0, Y1, X0, X1 = int(y0 * sy), min(H, int(y1 * sy)), int(x0 * sx), min(W, int(x1 * sx))
    # resize the box with the same sampling grid as a full-frame resize
    M = np.array([[1.0 / sx, 0, (X0 + 0.5) / sx - 0.5], [0, 1.0 / sy, (Y0 + 0.5) / sy - 0.5]],
                 np.float64)
    q = np.dstack([op, col])
    up = cv2.warpAffine(q, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE)
    reg = img[Y0:Y1, X0:X1]
    reg *= (1.0 - k_op * up[..., 0:1])
    reg += up[..., 1:] * k_col


def _steam_sources(G, kind, idx):
    L = G["L"]
    if kind == "v":
        x, y, dx, dy = L["vents"][idx]
        return x, y, dx, dy, L["depth"]["pipes"], "back"
    if kind == "w":
        x, y = L["whistles"][idx]
        return x, y - 0.09, 0.0, -1.0, L["depth"]["backhead"], "front"
    E = L["engine"]
    sx = -1 if idx == 0 else 1
    return (sx * (E["x"] - E["cyl_r"] * 1.25), E["cyl_bot"] - 0.005, sx * -1.0, 0.25,
            L["depth"]["main"], "front")


def _steam(img, R, G, T, P, cols, i, layer, black):
    if black:
        return
    fps = R.fps
    life = 1.6
    lo = np.searchsorted(T["ev_f"], i - int(life * fps))
    hi = np.searchsorted(T["ev_f"], i, side="right")
    if hi <= lo:
        return
    qh, qw = G["q"]
    D = np.zeros((qh, qw), f32)
    any_ = False
    for e in T["ev"][lo:hi]:
        f0, kind, idx, strength, seed = e
        x, y, dx, dy, d, lay = _steam_sources(G, kind, idx)
        if lay != layer:
            continue
        sp, tau, burst, npf, lf, r0, gr, rise = _PLUME[kind]
        age = (i - f0) / fps
        k, ox, oy = _proj(R, T, i, d)
        ln = math.hypot(dx, dy)
        ux, uy = dx / ln, dy / ln
        rng = np.random.default_rng(seed)
        jit = rng.normal(0, 1, (npf, 3))
        st_ = strength ** 0.5
        for j in range(npf):
            a_ = age - burst * j / npf
            if a_ <= 0 or a_ > lf:
                continue
            s_ = sp * st_ * tau * (1 - math.exp(-a_ / tau)) * (1 + 0.15 * jit[j, 0])
            px = x + ux * s_ - uy * 0.06 * jit[j, 1] * a_ + 0.03 * a_ * math.sin(a_ * 3 + seed)
            py = y + uy * s_ + ux * 0.06 * jit[j, 1] * a_ - rise * a_ * a_
            rad = (r0 + gr * a_ ** 0.8 * (0.8 + 0.4 * st_)) * (1 + 0.2 * jit[j, 2])
            op = strength * 0.55 * math.exp(-a_ / (0.45 * lf)) * min(1.0, a_ / 0.05)
            cx_, cy_ = (k * px + ox) / 4.0, (k * py + oy) / 4.0
            rq = k * rad / 4.0
            if op < 0.01:
                continue
            _add_k(D, _kernel(G, min(rq * 1.6, 90)), cx_, cy_, op)
            any_ = True
    if not any_:
        return
    t = i / fps
    tex = G["tex"]
    ty = (np.arange(qh) + int(t * 22)) % 128
    tx = (np.arange(qw) + int(t * 5)) % 128
    D *= 0.45 + 1.1 * tex[ty][:, tx]
    D = cv2.GaussianBlur(D, (0, 0), 1.2)
    op = 1.0 - np.exp(-1.4 * D)
    # the steam is lit by the room: fire from below, window backlight, bulbs
    Lq = (cols["amb"] * 1.1 + cols["fire"] * G["map_fire"] * 0.9
          + cols["back"] * G["map_win"] * 0.55 + cols["key"] * G["map_top"] * 0.55)
    col = np.clip(Lq * 255.0 * _c(P["steam"]), 0, 200)
    _comp_q(img, op, op[..., None] * col, 0.80, 0.78)


# ---- sparks --------------------------------------------------------------
def _sparks(img, R, G, T, P, i, mesh_pts):
    fps = R.fps
    lo = np.searchsorted(T["sp_f"], i - int(1.4 * fps))
    hi = np.searchsorted(T["sp_f"], i, side="right")
    if hi <= lo:
        return
    qh, qw = G["q"]
    glow = np.zeros((qh, qw), f32)
    th = max(1, int(round(R.H / 720)))
    k, ox, oy = _proj(R, T, i, G["L"]["depth"]["main"])
    sc = _c(P["spark"])
    drew = False
    for f0, kind, strength, seed in T["sp"][lo:hi]:
        age = (i - f0) / fps
        rng = np.random.default_rng(seed)
        if kind == "h":
            n_, lf, v0 = 7, 0.42, 0.55
            src = mesh_pts[int(rng.integers(len(mesh_pts)))]
            xs = np.full(n_, src[0])
            ys = np.full(n_, src[1])
            ang = rng.uniform(-np.pi, 0.2, n_)
        elif kind == "m":
            n_, lf, v0 = 26, 0.7, 0.8
            src = mesh_pts[int(rng.integers(len(mesh_pts)))]
            xs = np.full(n_, src[0])
            ys = np.full(n_, src[1])
            ang = rng.uniform(-np.pi, 0.0, n_)
        else:                                  # drop entry: shower from above + mesh bursts
            n_, lf, v0 = int(150 * strength), 1.25, 0.6
            xs = rng.uniform(-0.85, 0.85, n_)
            ys = rng.uniform(-0.62, -0.42, n_)
            ang = rng.uniform(0.3, np.pi - 0.3, n_)
            pick = rng.random(n_) < 0.35
            mp = np.asarray(mesh_pts)[rng.integers(len(mesh_pts), size=n_)]
            xs = np.where(pick, mp[:, 0], xs)
            ys = np.where(pick, mp[:, 1], ys)
            ang = np.where(pick, rng.uniform(-np.pi, 0, n_), ang)
        if age > lf:
            continue
        sp = v0 * rng.uniform(0.4, 1.3, len(xs)) * strength ** 0.3
        life = lf * rng.uniform(0.5, 1.0, len(xs))
        vx, vy = sp * np.cos(ang), sp * np.sin(ang)
        g = 1.7
        dt = 1.4 / fps
        for j in range(len(xs)):
            if age > life[j]:
                continue
            a0 = max(0.0, age - dt)
            x1 = xs[j] + vx[j] * age
            y1 = ys[j] + vy[j] * age + 0.5 * g * age * age
            x0 = xs[j] + vx[j] * a0
            y0 = ys[j] + vy[j] * a0 + 0.5 * g * a0 * a0
            u = age / life[j]
            heat = (1 - u) ** 1.5
            col = sc * (0.45 + 0.75 * heat) * np.array([1.0, 0.55 + 0.45 * heat, 0.25 + 0.75 * heat ** 2], f32)
            p0 = (int((k * x0 + ox) * 16), int((k * y0 + oy) * 16))
            p1 = (int((k * x1 + ox) * 16), int((k * y1 + oy) * 16))
            cv2.line(img, p0, p1, tuple(float(c) for c in col), th, cv2.LINE_AA, 4)
            cv2.line(glow, (p0[0] // 4, p0[1] // 4), (p1[0] // 4, p1[1] // 4), float(heat), 1,
                     cv2.LINE_AA, 4)
            drew = True
    if drew:
        glow = cv2.GaussianBlur(glow, (0, 0), 2.0)
        img += cv2.resize(glow, (R.W, R.H))[..., None] * (sc * 0.55)


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------
def draw(R, img, i, a, lv):
    G = _geo(R, lv)
    C = R.scene_cache
    if "sp_T" not in C:
        C["sp_T"] = _timeline(R, a, lv, G["L"])
    T = C["sp_T"]
    L, P = G["L"], G["P"]
    W, H = R.W, R.H
    fps = R.fps
    t = i / fps
    black = bool(T["black"][i])
    hot_f = float(T["hot_f"][i])
    big_f = float(T["big_f"][i])
    kenv = float(T["kenv"][i])
    E2 = float(T["E2"][i])
    heat = float(T["heat"][i])
    spark = float(T["spark"][i])
    cols = _cols(T, P, i)
    D = L["depth"]
    img[:] = 0.0
    # ---- 1. back wall + the clock window (glass lit from outside)
    wl = G["plates"]["wall"]
    gl = G["glass"]
    win_I = (0.35 if black else 1.0) * (0.80 + 0.15 * E2 + 0.12 * kenv * hot_f) \
        * (1.0 - 0.25 * big_f)

    def add_glass(rgb, cu0, cv0):
        y0, x0 = gl["y0"] - cv0, gl["x0"] - cu0
        gi = gl["img"] * win_I
        # clock hands (iron silhouettes behind the tracery): minute ticks per bar
        mnt = float(T["minute"][i])
        hr = mnt / 12.0 - 1.9
        hand = np.zeros(gi.shape[:2], f32)
        for ang, ln, wd in ((hr, 0.55, 0.035), (mnt, 0.86, 0.022)):
            c_, s_ = math.cos(ang), math.sin(ang)
            Rr = gl["R"]
            cx, cy = gl["cx"] - gl["x0"], gl["cy"] - gl["y0"]
            pts = np.array([[cx - c_ * 0.10 * Rr - s_ * wd * Rr, cy - s_ * 0.10 * Rr + c_ * wd * Rr],
                            [cx + c_ * ln * Rr - s_ * wd * 0.35 * Rr, cy + s_ * ln * Rr + c_ * wd * 0.35 * Rr],
                            [cx + c_ * (ln + 0.05) * Rr, cy + s_ * (ln + 0.05) * Rr],
                            [cx + c_ * ln * Rr + s_ * wd * 0.35 * Rr, cy + s_ * ln * Rr - c_ * wd * 0.35 * Rr],
                            [cx - c_ * 0.10 * Rr + s_ * wd * Rr, cy - s_ * 0.10 * Rr - c_ * wd * Rr]])
            cv2.fillPoly(hand, [np.round(pts * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
        gi = gi * (1.0 - 0.88 * hand[..., None])
        hh, ww = gi.shape[:2]
        ya, xa = max(0, y0), max(0, x0)
        yb, xb = min(rgb.shape[0], y0 + hh), min(rgb.shape[1], x0 + ww)
        reg = rgb[ya:yb, xa:xb]
        al = wl["a"][ya + cv0:yb + cv0, xa + cu0:xb + cu0, None]
        reg += gi[ya - y0:yb - y0, xa - x0:xb - x0] * (1.0 - al)
        return rgb
    pl = dict(wl)
    pl["a"] = np.ones_like(wl["a"])
    kw, oxw, oyw = _draw_plate(img, R, T, i, pl, cols, extra=add_glass)
    _haze(img, G, P, cols, 0.22 + 0.04 * E2)
    # ---- 2. far gear train in the haze
    root = float(T["theta"][i])
    th = _gear_angles(G["chain"], {"M": root, "F1": -0.55 * root + 0.3, "F3": 0.42 * root})
    kf, oxf, oyf = _proj(R, T, i, D["far"])
    fogc = _c(P["fog"]) * 1.4
    for name, g in G["chain"].items():
        if g["d"] != "far":
            continue
        x, y = g["c"]
        lights = _gear_lights(G, "pipes", x, y)
        _draw_gear(img, g["mc"], lights, cols, th[name], kf, oxf, oyf, x, y, g["mc"].ppu / g["sc"],
                   fade=(0.58, fogc))
    # ---- 3. god rays from the window into the steam
    if not black:
        X, Y = G["XY"]
        wx, wy = L["window"]["c"]
        k_, ox_, oy_ = _proj(R, T, i, D["wall"])
        cxw = (k_ * wx + ox_ - W / 2) / H
        cyw = (k_ * wy + oy_ - H / 2) / H
        dx, dy = X - cxw, Y - cyw
        r = np.sqrt(dx * dx + dy * dy) / (L["window"]["r"] * k_ / H)
        ang = np.arctan2(dy, dx)
        rays = (0.5 + 0.5 * np.sin(ang * 9 + t * 0.11) * np.sin(ang * 23 - t * 0.07 + 1.3)
                + 0.35 * np.sin(ang * 5 - t * 0.05))
        rays = np.clip(rays, 0, None) ** 2.2
        fall = np.clip((r - 0.55) / 0.5, 0, 1) * np.exp(-np.clip(r - 1.0, 0, None) / 0.9)
        amt = (0.10 + 0.10 * E2 + 0.10 * kenv * hot_f) * win_I
        ray = (rays * fall * amt)[..., None] * (cols["back"] * 255.0 * 0.55)
        img += cv2.resize(ray.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR)
    # ---- 3b. dust motes drifting through the window light (ambient life;
    # hi-hat sparkle makes them glint)
    if not black:
        k_, ox_, oy_ = _proj(R, T, i, 2.3)
        wx, wy = L["window"]["c"]
        nm = 70
        u = np.arange(nm, dtype=np.float64)
        hx = np.array([_h01(j, 1.0) for j in range(nm)])
        hy = np.array([_h01(j, 2.0) for j in range(nm)])
        hs = np.array([_h01(j, 3.0) for j in range(nm)])
        mx = wx + (((hx + t * (0.004 + 0.006 * hs)) % 1.0) - 0.5) * 0.95 \
            + 0.015 * np.sin(t * 0.6 + u)
        my = wy + (((hy - t * (0.010 + 0.012 * hs)) % 1.0) - 0.5) * 0.80
        dr = np.sqrt((mx - wx) ** 2 + (my - wy) ** 2) / L["window"]["r"]
        lit = np.clip(1.15 - dr, 0, 1) * (0.35 + 0.65 * hs)
        tw = 1.0 + 2.2 * spark * (np.array([_h01(i, j) for j in range(nm)]) > 0.6)
        bc = cols["back"] * 255.0 * 0.55 * win_I
        rr_ = max(1, int(round(H / 900)))
        for j in range(nm):
            if lit[j] > 0.05:
                cv2.circle(img, (int((k_ * mx[j] + ox_) * 16), int((k_ * my[j] + oy_) * 16)), rr_ * 16,
                           tuple(float(v) for v in np.clip(bc * lit[j] * tw[j], 0, 255)), -1,
                           cv2.LINE_AA, 4)
    # ---- 4. pipes + their steam
    _draw_plate(img, R, T, i, G["plates"]["pipes"], cols)
    _haze(img, G, P, cols, 0.10 + 0.02 * E2)
    _steam(img, R, G, T, P, cols, i, "back", black)
    # ---- 5. main plane: engines, flywheels, the gear train, the medallion
    km, oxm, oym = _proj(R, T, i, D["main"])
    E_ = L["engine"]
    phi = float(T["phi"][i])
    for side, pl_name, ph in ((1, "engine", phi), (-1, "engine_L", phi + np.pi)):
        _draw_plate(img, R, T, i, G["plates"][pl_name], cols)
        fx_, fy_ = side * E_["x"], E_["fly_y"]
        ang = ph if side > 0 else np.pi - ph
        pin = (fx_ + E_["crank"] * math.cos(ang), fy_ + E_["crank"] * math.sin(ang))
        dxp = pin[0] - fx_
        ych = pin[1] + math.sqrt(max(1e-6, E_["rod"] ** 2 - dxp * dxp))
        _engine_rods(img, km, oxm, oym, fx_, pin, ych, E_, P, cols, R)
        fl = _gear_lights(G, "main", fx_, fy_)
        _draw_gear(img, G["fly"], fl, cols, ang, km, oxm, oym, fx_, fy_, G["fly"].ppu)
    mesh_pts = []
    for name in ("B", "C", "D", "M"):
        g = G["chain"][name]
        x, y = g["c"]
        lights = _gear_lights(G, "main", x, y)
        _draw_gear(img, g["mc"], lights, cols, th[name], km, oxm, oym, x, y, g["mc"].ppu)
        if g["parent"] is not None:
            p = G["chain"][g["parent"]]
            rp = L["module"] * p["N"] / 2
            mesh_pts.append((p["c"][0] + rp * math.cos(g["ang"]), p["c"][1] + rp * math.sin(g["ang"])))
    # the medallion: fixed upright, kick punch, sung-bump punch, coal-glow plume
    md = G["medallion"]
    mx, my = L["main_c"]
    s_ = 1.0 + (0.025 + 0.035 * hot_f) * kenv + 0.05 * float(T["punch"][i])
    rgb = _lit_plate(md, cols)
    glow_c = _c(P["en_red"]) * (0.25 + 1.6 * heat * (0.4 + 0.6 * hot_f)) * (0.2 if black else 1.0)
    rgb = rgb + md["emis"][..., None] * glow_c
    M = _rot_M(0.0, km * s_ / md["ppu"], (md["w"] - 1) / 2, (md["h"] - 1) / 2, km * mx + oxm, km * my + oym)
    _blit(img, rgb, md["a"], M)
    # ---- 6. hi-hat sparks + entry shower (behind the near planes)
    _sparks(img, R, G, T, P, i, mesh_pts)
    # ---- 7. bulbs (the spectrum)
    _bulbs(img, R, G, T, P, cols, i)
    # ---- 8. backhead: things behind its holes first, then the plate
    _backhead(img, R, G, T, P, cols, i)
    _steam(img, R, G, T, P, cols, i, "front", black)
    # ---- 9. out-of-focus foreground gears
    _near(img, R, G, T, P, cols, i, root)
    # ---- 10. HIT A BUMP maker's plate on drop entries
    _plate(img, R, G, T, P, cols, i)
    # ---- big drop: rotating alarm wash from the warning lamps
    if big_f > 0.05 and not black:
        ph2 = ((float(T["bi"][i]) + float(T["ph"][i])) / 2.0) % 1.0
        X, Y = G["XY"]
        xc = (ph2 * 2.2 - 1.1) * (W / H) / 2
        wash = np.exp(-((X - xc) / 0.30) ** 2) * (0.25 + 0.75 * np.clip(Y + 0.5, 0, 1))
        img += cv2.resize((wash * big_f * (14 + 18 * kenv)).astype(f32), (W, H))[..., None] \
            * (_c(P["lamp"]) / 255.0)
    del spark


def _engine_rods(img, k, ox, oy, fx_, pin, ych, E_, P, cols, R):
    """Connecting rod, crosshead and piston rod (simple shaded bars)."""
    steel = _c(P["steel"]) / 255.0
    brass = _c(P["brass"]) / 255.0
    lit = cols["amb"] * 1.6 + cols["fire"] * 0.55 + cols["key"] * 0.35 + cols["back"] * 0.25
    def bar(p0, p1, w, col, hi=0.8):
        x0, y0 = k * p0[0] + ox, k * p0[1] + oy
        x1, y1 = k * p1[0] + ox, k * p1[1] + oy
        ww = max(1, int(round(w * k)))
        c0 = tuple(float(v) for v in np.clip(col * lit * 255.0, 0, 255))
        cv2.line(img, (int(x0 * 16), int(y0 * 16)), (int(x1 * 16), int(y1 * 16)), c0, ww,
                 cv2.LINE_AA, 4)
        c1 = tuple(float(v) for v in np.clip(col * lit * 255.0 * (1 + hi) + 30 * hi, 0, 255))
        off = -0.25 * w * k
        cv2.line(img, (int((x0 + off) * 16), int(y0 * 16)), (int((x1 + off) * 16), int(y1 * 16)), c1,
                 max(1, ww // 4), cv2.LINE_AA, 4)
    x = fx_
    # piston rod from crosshead into the cylinder
    bar((x, ych), (x, E_["cyl_top"] + 0.01), 0.013, steel, 1.0)
    # crosshead block
    bar((x, ych - 0.016), (x, ych + 0.016), 0.050, brass, 0.6)
    # connecting rod (crank pin -> crosshead), thicker at the big end
    bar(pin, (x, ych), 0.018, steel * 0.85, 0.7)
    cv2.circle(img, (int((k * pin[0] + ox) * 16), int((k * pin[1] + oy) * 16)), max(1, int(0.014 * k)) * 16,
               tuple(float(v) for v in np.clip(brass * lit * 300, 0, 255)), -1, cv2.LINE_AA, 4)


def _bulbs(img, R, G, T, P, cols, i):
    L = G["L"]
    B = L["bulbs"]
    d = L["depth"]["bulbs"]
    k, ox, oy = _proj(R, T, i, d)
    black = bool(T["black"][i])
    lv_ = T["bulbs"][i]
    spark = float(T["spark"][i])
    kage = int(T["kage"][i])
    hot_f = float(T["hot_f"][i])
    ea = int(T["entry_age"][i])
    surge = 0.82 ** ea if ea < 20 else 0.0
    t = i / R.fps
    qh, qw = G["q"]
    glow = np.zeros((qh, qw, 3), f32)
    th = max(1, int(round(R.H / 700)))
    for b in range(B["n"]):
        x, y = G["bulb_xy"][b]
        top = -0.62
        cord = y - top
        # pendulum: kicks in a drop swing the bulbs; they follow through and settle
        sw = 0.012 * math.sin(t * 0.7 + b * 1.3)
        if kage < 60 and hot_f > 0.2:
            at = kage / R.fps
            sw += 0.07 * hot_f * math.sin(2 * np.pi * 1.1 * at + b * 0.4) * math.exp(-at / 0.55) \
                * (0.6 + 0.4 * ((b * 7) % 5) / 4.0)
        bx, by = x + cord * math.sin(sw), top + cord * math.cos(sw)
        lev = 0.0 if black else float(lv_[b])
        if not black and spark > 0.08 and float(_h01(i, b * 3.7)) < 0.45 * spark:
            lev *= 0.25                                      # filament flicker on hats
        lev = min(1.25, lev + (0.0 if black else surge * 0.9))
        P0 = (k * x + ox, k * top + oy)
        P1 = (k * bx + ox, k * by + oy)
        cc = tuple(float(v) for v in np.clip(_c(P["iron"]) * 0.8, 0, 255))
        cv2.line(img, (int(P0[0] * 16), int(P0[1] * 16)), (int(P1[0] * 16), int(P1[1] * 16)), cc, th,
                 cv2.LINE_AA, 4)
        r = B["r"] * k
        ca, sa = math.cos(sw), math.sin(sw)
        # socket (brass), lit by the room + its own bulb
        sk = _c(P["brass"]) / 255.0 * (cols["amb"] * 1.4 + cols["fire"] * 0.3 + 0.5 * lev) * 255.0
        cxs, cys = P1[0] + sa * r * 0.35, P1[1] + ca * r * 0.35
        box = cv2.boxPoints(((cxs, cys), (r * 0.75, r * 0.9), -math.degrees(sw)))
        cv2.fillConvexPoly(img, np.round(box * 16).astype(np.int32),
                           tuple(float(v) for v in np.clip(sk, 0, 255)), cv2.LINE_AA, 4)
        # glass + filament
        gx, gy = P1[0] + sa * r * 1.55, P1[1] + ca * r * 1.55
        hot = np.clip(_c(P["bulb"]) * (0.25 + 0.9 * lev), 0, 255)
        # Edison bulb: near-clear glass (a faint tint + one specular streak),
        # the light lives in the filament and a warm core around it
        bc = _c(P["bulb"])
        sy0, sx0 = int(gy - r * 1.3), int(gx - r * 1.0)
        sub = img[sy0:int(gy + r * 1.3) + 1, sx0:int(gx + r * 1.0) + 1]
        if sub.size and sy0 >= 0 and sx0 >= 0:
            hh, ww = sub.shape[:2]
            yy, xx = np.mgrid[0:hh, 0:ww].astype(f32)
            ox2, oy2 = gx - sx0, gy - sy0
            e = ((xx - ox2) / (r * 0.74)) ** 2 + ((yy - oy2) / (r * 1.08)) ** 2
            m = np.clip((1.0 - e) * r * 0.45, 0, 1)
            core = np.clip(1.0 - e, 0, 1) ** 1.6
            sub *= (1 - 0.18 * m[..., None])
            sub += (m * (0.05 + 0.85 * lev * core))[..., None] * bc \
                + (m * 6.0)[..., None] * cols["back"]
            streak = np.clip(1 - np.abs(e - 0.62) / 0.10, 0, 1) \
                * np.clip((ox2 - xx) / (r * 0.5), 0, 1) * np.clip((oy2 - yy) / (r * 0.4) + 0.6, 0, 1)
            sub += (streak * (35 + 90 * lev))[..., None]
        # filament: a looped zigzag, dull red when cold, white-amber when hot
        zz = []
        for j in range(9):
            u = j / 8.0
            zz.append((gx + (0.22 if j % 2 else -0.22) * r, gy - 0.45 * r + u * 0.75 * r))
        zz = np.array(zz)
        fil = _c((110, 38, 14)) * (1 - min(1.0, lev)) + _c(P["bulb_hot"]) * min(1.0, lev * 1.2)
        cv2.polylines(img, [np.round(zz * 16).astype(np.int32)], False,
                      tuple(float(v) for v in fil), th, cv2.LINE_AA, 4)
        if lev > 0.02:
            _add_k3(glow, _kernel(G, max(2, r * (0.9 + 1.0 * lev) / 4 * 1.7)), gx / 4, gy / 4,
                    hot * (0.42 * lev))
    if glow.any():
        img += cv2.resize(cv2.GaussianBlur(glow, (0, 0), 1.0), (R.W, R.H))


def _add_k3(D, K, cx, cy, col):
    r = K.shape[0] // 2
    h, W = D.shape[:2]
    x0, y0 = int(cx) - r, int(cy) - r
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + K.shape[1]), min(h, y0 + K.shape[0])
    if xb <= xa or yb <= ya:
        return
    D[ya:yb, xa:xb] += K[ya - y0:yb - y0, xa - x0:xb - x0, None] * col


def _backhead(img, R, G, T, P, cols, i):
    L = G["L"]
    d = L["depth"]["backhead"]
    k, ox, oy = _proj(R, T, i, d)
    black = bool(T["black"][i])
    heat = float(T["heat"][i])
    kenv = float(T["kenv"][i])
    hot_f = float(T["hot_f"][i])
    big_f = float(T["big_f"][i])
    ea = int(T["entry_age"][i])
    flare = 0.85 ** ea if ea < 30 else 0.0
    t = i / R.fps
    W, H = R.W, R.H
    # ---- firebox (behind the door slots)
    Dr = L["door"]
    x0 = int(k * (Dr["c"][0] - Dr["w"] * 0.5) + ox)
    x1 = int(k * (Dr["c"][0] + Dr["w"] * 0.5) + ox)
    y0 = int(k * (Dr["c"][1] - Dr["h"] * 0.5) + oy)
    y1 = int(k * (Dr["c"][1] + Dr["h"] * 0.5) + oy)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if xb > xa and yb > ya:
        tex = G["tex"]
        hh, ww = (yb - ya), (xb - xa)
        qy = ((np.arange(hh) * 128 // max(1, hh) + int(t * 70)) % 128)
        qx = (np.arange(ww) * 128 // max(1, ww)) % 128
        qy2 = ((np.arange(hh) * 256 // max(1, hh) + int(t * 140)) % 128)
        n1 = tex[qy][:, qx] * 0.65 + tex[qy2][:, (qx * 2) % 128] * 0.45
        yy = (np.arange(ya, yb, dtype=f32)[:, None] - y0) / max(1, (y1 - y0))
        lvl = 0.12 if black else (0.35 + 0.85 * heat + 0.4 * kenv * hot_f + 0.8 * flare)
        f = np.clip(n1 * (0.6 + 0.7 * lvl) + yy * 0.7 * lvl - 0.35, 0, 1.2)
        ramp = np.array([[0, 0, 0], [90, 10, 2], [210, 60, 8], [255, 150, 30], [255, 225, 150]], f32)
        if big_f > 0.3:
            ramp[3:] = [[255, 110, 20], [255, 190, 110]]
        fi = np.clip(f * (len(ramp) - 1), 0, len(ramp) - 1.001)
        lo_ = np.floor(fi).astype(int)
        fr = (fi - lo_)[..., None]
        img[ya:yb, xa:xb] = ramp[lo_] * (1 - fr) + ramp[lo_ + 1] * fr
    # ---- gauge dials + needles
    gv = T["gauge"][i]
    pre = float(T["pre"][i])
    dl = G["dials"]
    lit_d = (cols["amb"] * 1.3 + cols["fire"] * 0.35 + cols["key"] * 0.45) * 0.9 + (0.08 if black else 0.42)
    glints = []
    for g_, (gx, gy) in enumerate(L["gauges"]):
        D_ = dl[g_]
        s = D_["s"]
        cxp, cyp = k * gx + ox, k * gy + oy
        sc = k * 2 * L["face_r"] / (s - 2)
        M = _rot_M(0.0, sc, (s - 1) / 2, (s - 1) / 2, cxp, cyp)
        _blit(img, D_["rgb"] * lit_d, D_["a"], M)
        v = float(gv[g_])
        if pre > 0.2 and not black:      # needles shiver under the build
            v += pre ** 1.5 * 0.028 * math.sin(i * 2.9 + g_ * 1.7) * (0.5 + float(_h01(i, g_)))
        ang = math.radians(135 + 270 * min(1.1, max(0.0, v)))
        rr = k * L["face_r"]
        c_, s_ = math.cos(ang), math.sin(ang)
        wd = rr * 0.055
        pts = np.array([[cxp - c_ * rr * 0.20 - s_ * wd, cyp - s_ * rr * 0.20 + c_ * wd],
                        [cxp + c_ * rr * 0.86, cyp + s_ * rr * 0.86],
                        [cxp - c_ * rr * 0.20 + s_ * wd, cyp - s_ * rr * 0.20 - c_ * wd]])
        sh = pts + np.array([rr * 0.05, rr * 0.07])
        _poly_mul(img, sh, 0.55)
        ncol = _c(P["needle"]) * (0.6 + float(np.mean(lit_d)))
        if v > 0.78:                     # in the red: the needle tip glows
            ncol = ncol + _c(P["red_zone"]) * 0.6 * min(1.0, (v - 0.78) / 0.2)
        cv2.fillConvexPoly(img, np.round(pts * 16).astype(np.int32), tuple(float(x) for x in ncol),
                           cv2.LINE_AA, 4)
        cv2.circle(img, (int(cxp * 16), int(cyp * 16)), max(1, int(rr * 0.11)) * 16,
                   tuple(float(x) for x in np.clip(_c(P["brass"]) * lit_d * 1.2, 0, 255)), -1,
                   cv2.LINE_AA, 4)
        glints.append((D_, M))
    # ---- Nixie pressure readout
    Nx = L["nixie"]
    NG = G["nixie"]
    val = int(T["psi"][i])
    digs = f"{val:04d}"
    tw, th_ = NG["tw"] * k, NG["th"] * k
    nx0 = k * (Nx["c"][0] - Nx["w"] * 0.92 / 2) + ox
    ny0 = k * (Nx["c"][1] - Nx["h"] * 0.78 / 2) + oy
    nglow = []
    for j in range(Nx["n"]):
        x_ = nx0 + j * tw
        # dark glass tube
        xa_, ya_ = int(x_ + tw * 0.06), int(ny0)
        xb_, yb_ = int(x_ + tw * 0.94), int(ny0 + th_)
        if xb_ > xa_ and yb_ > ya_ and xa_ >= 0 and yb_ <= H:
            img[ya_:yb_, xa_:xb_] = img[ya_:yb_, xa_:xb_] * 0.15 + np.array([14, 9, 6], f32)
        if black:
            continue
        gl = NG["g"]
        on = gl[int(digs[j])]
        ghost = gl.max(0) * 0.05
        # draw into the local box only
        bw, bh = max(1, int(round(tw))), max(1, int(round(th_)))
        mm = cv2.resize(np.maximum(on, ghost), (bw, bh), interpolation=cv2.INTER_AREA)
        xi, yi = int(round(x_)), int(round(ny0))
        if xi < 0 or yi < 0 or xi + bw > W or yi + bh > H:
            continue
        col = _c(P["nixie"]) * (0.85 + 0.3 * kenv)
        img[yi:yi + bh, xi:xi + bw] += mm[..., None] * col + (mm ** 3)[..., None] * (_c(P["nixie_core"]) * 0.5)
        nglow.append((xi + bw / 2, yi + bh / 2, bw, bh))
    # ---- warning lamps (behind their cages)
    lamp = _lamp_level(T, i)
    lr = L["lamp_r"] * k
    for (lx, ly) in L["lamps"]:
        cx_, cy_ = k * lx + ox, k * ly + oy
        cv2.circle(img, (int(cx_ * 16), int(cy_ * 16)), int(lr * 16),
                   tuple(float(v) for v in _c(P["lamp"]) * (0.10 + 0.85 * lamp)), -1, cv2.LINE_AA, 4)
        cv2.circle(img, (int((cx_ - lr * 0.3) * 16), int((cy_ - lr * 0.3) * 16)), int(lr * 0.35 * 16),
                   tuple(float(v) for v in np.clip(_c(P["lamp"]) * (0.2 + 0.9 * lamp) + 90 * lamp, 0, 255)),
                   -1, cv2.LINE_AA, 4)
    # ---- the backhead plate itself
    _draw_plate(img, R, T, i, G["plates"]["backhead"], cols)
    # ---- glass glints over the gauges, glows over the readouts / lamps
    for D_, M in glints:
        g_ = D_["glint"]
        _blit(img, g_[..., None] * (np.array([150, 140, 125], f32)
                                    * (0.4 + float(np.mean(cols["key"])))), None, M)
    qh, qw = G["q"]
    glow = np.zeros((qh, qw, 3), f32)
    if not black:
        for (gx_, gy_, bw, bh) in nglow:
            _add_k3(glow, _kernel(G, max(2, bw / 4 * 0.9)), gx_ / 4, gy_ / 4, _c(P["nixie"]) * 0.35)
    if lamp > 0.2:
        for (lx, ly) in L["lamps"]:
            _add_k3(glow, _kernel(G, max(3, lr / 4 * 3.0)), (k * lx + ox) / 4, (k * ly + oy) / 4,
                    _c(P["lamp"]) * 0.55 * lamp)
    # firebox glow spilling out of the door
    if True:
        fl_ = 0.06 if black else (0.25 + 0.45 * heat + 0.5 * flare)
        _add_k3(glow, _kernel(G, max(4, k * Dr["w"] / 4 * 0.7)), (k * Dr["c"][0] + ox) / 4,
                (k * Dr["c"][1] + oy) / 4, np.array([255, 110, 30], f32) * fl_ * 0.5)
    if glow.any():
        img += cv2.resize(cv2.GaussianBlur(glow, (0, 0), 1.5), (W, H))


def _poly_mul(img, pts, f):
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), f32)
    cv2.fillConvexPoly(m, np.round((pts - [x0, y0]) * 16).astype(np.int32), 1.0, cv2.LINE_AA, 4)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    img[y0:y1, x0:x1] *= (1.0 - f * m[..., None])


def _near(img, R, G, T, P, cols, i, root):
    """Foreground gears at d=1.0, shaded at quarter res, defocused."""
    d = G["L"]["depth"]["near"]
    k, ox, oy = _proj(R, T, i, d)
    qh, qw = G["q"]
    buf = np.zeros((qh, qw, 4), f32)
    for nr in G["near"]:
        x, y = nr["c"]
        th = nr["dir"] * root * (40.0 / nr["N"]) * 0.7 + 0.4
        lights = _gear_lights(G, "main", x, y)
        mc = nr["mc"]
        c2 = dict(cols)
        c2["spot"] = cols["spot"] * 0.0
        rgb = mc.shade(lights, c2, th) * 0.55
        M = _rot_M(th, (k / 4) / nr["ppu"], mc.cx, mc.cy, (k * x + ox) / 4, (k * y + oy) / 4)
        _blit4(buf, rgb, mc.a, M)
    if buf[..., 3].any():
        buf = cv2.GaussianBlur(buf, (0, 0), max(1.0, 2.2 * R.H / 1080 * 2))
        _comp_q(img, np.ascontiguousarray(buf[..., 3]), np.ascontiguousarray(buf[..., :3]), 1.0, 1.0)


def _blit4(buf, rgb, a, M):
    """Like _blit but into an RGBA premultiplied buffer (over)."""
    H, W = buf.shape[:2]
    al = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    co = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    buf *= (1.0 - al)[..., None]
    buf[..., :3] += co
    buf[..., 3] += al


def _plate(img, R, G, T, P, cols, i):
    """The maker's plate SLAMS in on each drop entry: falls in from big
    (accelerating INTO the hit), squashes, settles, holds, exits upward."""
    ea = int(T["entry_age"][i])
    fps = R.fps
    hold = int(1.15 * fps)
    out_ = int(0.30 * fps)
    if ea > hold + out_ or not T["hot"][i]:
        return
    pl = G["plates"]["plate"]
    big = bool(T["big"][i])
    if ea < 4:
        u = ea / 4.0
        s = 1.0 + 0.55 * (1 - u * u)         # ease-in: accelerating into the impact
        alpha = 0.85 + 0.15 * u
        dy = 0.0
    else:
        a_ = (ea - 4) / fps
        s = 1.0 - 0.06 * math.exp(-a_ / 0.07) * math.cos(a_ * 40)     # squash, settle
        alpha = 1.0
        dy = 0.0
        if ea > hold:
            u = (ea - hold) / out_
            dy = -0.65 * u * u
            alpha = 1.0 - u * u
    cols2 = {ch: np.zeros(3, f32) for ch in cols}
    cols2["amb"] = _c((0.30, 0.27, 0.24))
    cols2["key"] = _c((0.95, 0.80, 0.58))
    cols2["spot"] = _c((0.55, 0.48, 0.38))
    cols2["fire"] = _c(P["big_fire"] if big else P["fire"]) * 0.45
    rgb = _lit_plate(pl, cols2)
    W, H = R.W, R.H
    k = H * 1.0
    cx, cy = W / 2.0, H / 2.0 + (-0.07 + dy) * H
    ppu = pl["ppu"]
    h, w = pl["a"].shape
    M = _rot_M(0.0, k * s / ppu, (w - 1) / 2, (h - 1) / 2, cx, cy)
    # contact shadow so it sits ON the machine, not pasted over it
    Ms = M.copy()
    Ms[0, 2] += 0.012 * H
    Ms[1, 2] += 0.020 * H
    _blit(img, None, pl["a"] * (0.55 * alpha), Ms)
    _blit(img, rgb * alpha, pl["a"] * alpha, M)


# ---------------------------------------------------------------------------
# post-FX
# ---------------------------------------------------------------------------
def fx(R, img, i, a, lv_):
    T = R.scene_cache.get("sp_T")
    if T is None:
        return img
    P = _geo(R, lv_)["P"]
    W, H = R.W, R.H
    hot_f = float(T["hot_f"][i])
    kenv = float(T["kenv"][i])
    if T["black"][i]:
        img *= 0.80
    sa = int(T["strobe_age"][i])
    if R.style.strobe and sa < 4:
        # warm, moderate, and never on consecutive near-white frames
        img += _c(P["flash"]) * (0.30 * (0.45 ** sa))
    if hot_f > 0.05:
        off = int(round((1.0 + 2.0 * kenv) * hot_f * W / 1920))
        if off > 0:
            img[..., 0] = np.roll(img[..., 0], off, axis=1)
            img[..., 2] = np.roll(img[..., 2], -off, axis=1)
    z = 1.0
    dx = dy = 0.0
    sh_a = int(T["sh_age"][i])
    if sh_a < 14:
        s = float(T["shake"][int(T["sh_idx"][i])])
        amp = s * 0.014 * H * (0.72 ** sh_a)
        th = 6.2832 * float(lv_._hash01(i, 17.0))
        dx, dy = amp * math.cos(th), amp * math.sin(th)
        z += 0.015 * s * (0.7 ** sh_a)
    if z > 1.0005 or abs(dx) + abs(dy) > 0.3:
        M = np.array([[z, 0, (1 - z) * W / 2 + dx], [0, z, (1 - z) * H / 2 + dy]], np.float32)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # exposure guard: the drop entry stacks steam + flare + strobe; keep the
    # frame from going near-white (lesson from the megacity render)
    m = float(img[::8, ::8].mean())
    if m > 108.0:
        img *= 108.0 / m
    return img
