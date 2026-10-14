"""Real terrain for the Be My One city: Copernicus GLO-30 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
2014-2018 provided under COPERNICUS by the European Union and ESA) around Mong Kok, ±16 km.

The tile is a DSM (it includes buildings), so the bare earth is recovered with a grey-scale OPENING (min then max
filter, ~210 m) + a light blur. The urban plain is flattened to z=0 (scene ground) so every building sits on it; hills
(Lion Rock range, Beacon Hill, the HK Island peaks) keep their real shape; sea cells (DEM ≤ 0.3 m) sink to −8 m
under a water plane; the window's rim sinks too, so no cut edge can show on the horizon. Pure numpy.
"""
import math

import numpy as np

CORE_C = (0.0, 500.0)


def _filt1(a, k, fn, axis):
    pad = k // 2
    ap = np.pad(a, [(pad, pad) if i == axis else (0, 0) for i in range(2)], mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(ap, k, axis=axis)
    return fn(sw, axis=-1)


def _box(a, k):
    for ax in (0, 1):
        a = _filt1(a, k, np.mean, ax)
    return a


def load(path):
    d = np.load(path)
    z = d["z"].astype(np.float32)
    x0, y0, sx, sy = float(d["x0"]), float(d["y0"]), float(d["sx"]), float(d["sy"])
    water = z <= 0.3
    mn = z
    for ax in (0, 1):
        mn = _filt1(mn, 7, np.min, ax)
    op = mn
    for ax in (0, 1):
        op = _filt1(op, 7, np.max, ax)
    bare = _box(np.minimum(op, z), 3)
    H, W = z.shape
    X = x0 + np.arange(W) * sx
    Y = y0 + np.arange(H) * sy
    XX, YY = np.meshgrid(X, Y)
    dist = np.hypot(XX - CORE_C[0], YY - CORE_C[1])
    h = np.maximum(bare - 4.0, 0.0)
    lowland = np.clip((h - 10.0) / 45.0, 0.0, 1.0)
    lowland = lowland * lowland * (3 - 2 * lowland)
    zt = h * lowland                                          # plains → 0, hills keep their height
    core = np.clip((dist - 2600.0) / 1200.0, 0.0, 1.0)
    zt *= core * core * (3 - 2 * core)                        # dead flat under the detailed city
    zt = np.where(water, -8.0, zt)
    rim = np.clip((15500.0 - np.maximum(np.abs(XX - CORE_C[0]), np.abs(YY - CORE_C[1]))) / 2000.0, 0.0, 1.0)
    zt = np.where(rim < 1.0, zt * rim + (-12.0) * (1 - rim), zt)   # far rim sinks into the sea
    urban = (~water) & (h < 18.0)                             # flat built-up land
    return dict(z=zt, water=water, urban=urban, hill=h, x0=x0, y0=y0, sx=sx, sy=sy, X=X, Y=Y, dist=dist)


def height_at(T, x, y):
    """bilinear terrain height at scene (x, y) arrays"""
    fx = (np.asarray(x, float) - T["x0"]) / T["sx"]
    fy = (np.asarray(y, float) - T["y0"]) / T["sy"]
    H, W = T["z"].shape
    i0 = np.clip(np.floor(fy).astype(int), 0, H - 2)
    j0 = np.clip(np.floor(fx).astype(int), 0, W - 2)
    ty, tx = np.clip(fy - i0, 0, 1), np.clip(fx - j0, 0, 1)
    z = T["z"]
    return (z[i0, j0] * (1 - tx) * (1 - ty) + z[i0, j0 + 1] * tx * (1 - ty) + z[i0 + 1, j0] * (1 - tx) * ty +
            z[i0 + 1, j0 + 1] * tx * ty)


def mask_at(T, key, x, y):
    fx = np.clip(np.round((np.asarray(x, float) - T["x0"]) / T["sx"]).astype(int), 0, T["z"].shape[1] - 1)
    fy = np.clip(np.round((np.asarray(y, float) - T["y0"]) / T["sy"]).astype(int), 0, T["z"].shape[0] - 1)
    return T[key][fy, fx]


def grid_mesh(T, step=2):
    """vertices + quad faces of the terrain at every `step`-th DEM sample"""
    z = T["z"][::step, ::step]
    X, Y = T["X"][::step], T["Y"][::step]
    H, W = z.shape
    XX, YY = np.meshgrid(X, Y)
    V = np.stack([XX, YY, z], -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange(H * W).reshape(H, W)
    F = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    return V, F
