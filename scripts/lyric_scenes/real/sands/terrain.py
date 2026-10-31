"""Sands — the dune field as ONE pure-numpy height function h(x, y) (metres), shared by the mesh, the cactus scatter,
the camel path, the ground strikes and the camera clearance asserts. No bpy.

Transverse dunes, wind from −X (west): a long gentle WINDWARD slope (~8°) up to a sharp brink, then a SLIP FACE at
~27–30° (angle of repose 30–34°) down the lee side (+X). Crests snake (domain warp along y) and their height varies
along the crest, a second smaller dune set crosses at ~28°, a 900 m swell rolls underneath. Around the cube (the
origin) the field relaxes into a flat interdune pan so the hero shots have a floor.
References: Bagnold, The Physics of Blown Sand and Desert Dunes (1941) — windward 5–15°, slip face at the angle of
repose; USGS "Types of dunes" (pubs.usgs.gov/gip/deserts/dunes).
"""
import math

import numpy as np

PAN_R0, PAN_R1 = 32.0, 120.0      # flat pan radius → full dunes by PAN_R1
PAN_H = 0.0                       # pan floor height (the cube's ground)


def _frac(u):
    return u - np.floor(u)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def dune_profile(s, brink=0.80):
    """one dune period, s in [0,1): windward rise to 1 at s=brink (slightly convex), slip face down to 0 at s=1"""
    w = np.clip(s / brink, 0, 1)
    up = 0.45 * w + 0.55 * (1 - (1 - w) ** 2)
    t = np.clip((1 - s) / (1 - brink), 0, 1)
    down = t ** 1.15
    return np.where(s < brink, up, down)


def wave_noise(x, y, scale, seed, n=14):
    """smooth isotropic-ish noise in [-1, 1]: a sum of n plane waves with random directions/phases around 1/scale"""
    rg = np.random.default_rng(seed)
    th = rg.uniform(0, math.pi, n)
    f = rg.uniform(0.6, 1.6, n) / scale
    ph = rg.uniform(0, 2 * math.pi, n)
    acc = 0.0
    for t, k, p in zip(th, f, ph):
        acc = acc + np.sin(2 * math.pi * k * (x * math.cos(t) + y * math.sin(t)) + p)
    return acc / math.sqrt(n / 2)


def _ridges(x, y, lam, amp, ang_deg, seed, brink=0.78, warp_amp=0.42):
    a = math.radians(ang_deg)
    u = x * math.cos(a) + y * math.sin(a)                 # along the wind
    warp = warp_amp * (0.8 * wave_noise(x, y, lam * 4.0, seed) + 0.2 * wave_noise(x, y, lam * 1.6, seed + 50))
    s = _frac(u / lam + warp)
    hv = 0.68 + 0.30 * wave_noise(x, y, lam * 2.2, seed + 100) + 0.12 * wave_noise(x, y, lam * 0.8, seed + 150)
    return amp * np.clip(hv, 0.12, 1.25) * dune_profile(s, brink)


def height(x, y):
    """terrain height (m) at world x, y (arrays or scalars)"""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    big = _ridges(x, y, 150.0, 15.0, 0.0, 1)
    small = _ridges(x, y, 52.0, 2.2, 24.0, 2, brink=0.75, warp_amp=0.35)
    swell = 9.0 * np.sin(2 * math.pi * x / 1300.0 + 0.7) * np.sin(2 * math.pi * y / 1700.0 + 0.3) \
        + 5.0 * np.sin(2 * math.pi * (x + 0.6 * y) / 870.0 + 1.9)
    h = big + small + swell
    r = np.hypot(x, y)
    m = _smooth((r - PAN_R0) / (PAN_R1 - PAN_R0))
    h0 = PAN_H + 0.35 * np.sin(x / 9.0 + 0.4) * np.sin(y / 11.0)      # the pan is not a billiard table
    h = h0 * (1 - m) + (h - _ORIGIN_H) * m
    return h - (r * r) / (2 * 6.371e6)                    # earth curvature (5 m at 8 km, 31 m at 20 km)


_ORIGIN_H = 0.0
_ORIGIN_H = float(np.mean(height(np.linspace(-200, 200, 41)[:, None], np.linspace(-200, 200, 41)[None, :])))


def slope(x, y, d=0.5):
    """|grad h| (rise over run)"""
    gx = (height(x + d, y) - height(x - d, y)) / (2 * d)
    gy = (height(x, y + d) - height(x, y - d)) / (2 * d)
    return np.hypot(gx, gy)


def polar_grid(r_max=20000.0, segs=768, r_min=0.6):
    """vertices (n,3) + quad faces for a polar grid with ~square cells everywhere (geometric ring spacing),
    centre fan closed. Cell size ≈ r·2π/segs."""
    q = 1.0 + 2 * math.pi / segs
    nr = int(math.ceil(math.log(r_max / r_min) / math.log(q))) + 1
    radii = r_min * q ** np.arange(nr)
    radii[-1] = r_max
    ang = 2 * math.pi * np.arange(segs) / segs
    X = (radii[:, None] * np.cos(ang)[None, :]).ravel()
    Y = (radii[:, None] * np.sin(ang)[None, :]).ravel()
    X = np.r_[0.0, X]
    Y = np.r_[0.0, Y]
    Z = height(X, Y)
    V = np.stack([X, Y, Z], 1)
    ring = lambda j, k: 1 + j * segs + (k % segs)  # noqa: E731
    jj, kk = np.meshgrid(np.arange(nr - 1), np.arange(segs), indexing="ij")
    jj, kk = jj.ravel(), kk.ravel()
    quads = np.stack([ring(jj, kk), ring(jj, kk + 1), ring(jj + 1, kk + 1), ring(jj + 1, kk)], 1)
    k = np.arange(segs)
    tris = np.stack([np.zeros(segs, int), ring(0, k), ring(0, k + 1)], 1)
    return V, quads, tris, radii


if __name__ == "__main__":
    print("origin offset", round(_ORIGIN_H, 2))
    for p in [(0, 0), (40, 0), (150, 30), (-300, 200), (800, -500), (3000, 3000)]:
        print(p, "h=%.2f slope=%.2f" % (float(height(*p)), float(slope(*p))))
    g = np.linspace(-600, 600, 601)
    H = height(g[:, None], g[None, :])
    S = slope(g[:, None], g[None, :])
    print("h range", H.min().round(1), H.max().round(1), "slope p50/p95/max", np.percentile(S, [50, 95]).round(2), S.max().round(2))
    print("slope deg max", math.degrees(math.atan(S.max())))
