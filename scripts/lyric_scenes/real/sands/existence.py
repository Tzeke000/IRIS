"""Sands v2 — per-shot existence (Zeke 10-30: "only render what's going to be seen in the camera shot";
blender-render-optimization §3). Pure numpy: masks → the build keys hide_render CONSTANT (bpyx.show_mask, combined
with any keys an object already has), so a hidden object is skipped in sync and its GN tree is never evaluated.

  cacti   per SHOT: kept if, at any sampled frame of the shot, its top/mid is inside a 15 %-widened frustum, ≥ 0.8 px
          tall and not behind a dune — OR any point of its long low-sun shadow lands inside the frustum.
  FX      per FRAME (the masks only switch while the thing renders nothing, so pixels are identical): strikes /
          flashes while a strike is alive, arcs + arc light while energy > 0 — each widened by ±1 frame for the shutter.
  camel   only inside the camel shot.
Proof is a with/without pixel diff (build --no-cull), not this reasoning.
"""
import math

import numpy as np

from lib.camera import half_tangents

import terrain as TR


def _visible_pts(A, idx, P, margin=0.15, sensor=36.0, aspect=16 / 9):
    """P (k,3) world points; frames idx → bool (len(idx), k) in the widened frustum + depth (len(idx), k)"""
    R, L = A["R"][idx], A["loc"][idx]
    pc = np.einsum("nji,nkj->nki", R, P[None] - L[:, None])
    tx, ty = half_tangents(A["lens"][idx], sensor, aspect)
    d = -pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (d > 0.5) & (np.abs(pc[..., 0] / d) <= (tx * (1 + margin))[:, None]) & \
             (np.abs(pc[..., 1] / d) <= (ty * (1 + margin))[:, None])
    return ok, d


def _los(c, p, step=3.0, margin=0.2):
    c, p = np.asarray(c, float), np.asarray(p, float)
    Lh = np.linalg.norm(p - c)
    if Lh < 6.0:
        return True
    t = np.arange(step, Lh - 3.0, step) / Lh
    q = c[None] + (p - c)[None] * t[:, None]
    return bool(np.all(q[:, 2] > TR.height(q[:, 0], q[:, 1]) + margin))


def cactus_masks(A, cacti, sun_el_deg, sun_rot_deg, n, every=4, min_px=0.8, width_px=1920, sensor=36.0):
    """cacti = [(name, x, y, z_base, height)] → ({name: bool[n] visible}, per-shot kept counts)"""
    rot = math.radians(sun_rot_deg)
    away = -np.array([math.sin(rot), math.cos(rot)])                    # shadows fall away from the sun
    names = A["names"]
    starts = list(A["starts"]) + [n + 1]
    out = {c[0]: np.zeros(n, bool) for c in cacti}
    kept = {}
    for k, nm in enumerate(names):
        f0, f1 = starts[k] - 1, starts[k + 1] - 1                         # 0-based [f0, f1)
        idx = np.unique(np.r_[np.arange(f0, f1, every), f1 - 1])
        el = math.radians(float(np.min(sun_el_deg[idx])))
        cnt = 0
        for name, x, y, zb, H in cacti:
            pts = [(x, y, zb + H), (x, y, zb + 0.5 * H)]
            Ls = min(H / math.tan(max(el, math.radians(0.5))), 600.0)
            for s in np.linspace(0.15, 1.0, 7):                         # shadow line on the ground
                sx, sy = x + away[0] * Ls * s, y + away[1] * Ls * s
                pts.append((sx, sy, float(TR.height(sx, sy)) + 0.1))
            P = np.array(pts, float)
            ok, d = _visible_pts(A, idx, P)
            px = H / np.maximum(d[:, 0], 1e-3) * A["lens"][idx] / sensor * width_px
            body = ok[:, :2].any(1) & (px >= min_px)
            shadow = ok[:, 2:].any(1) & (px >= min_px * 0.5)
            vis = False
            if shadow.any():
                vis = True
            else:
                for i in np.flatnonzero(body)[::2]:
                    c = A["loc"][idx[i]]
                    if _los(c, P[0]) or _los(c, P[1]):
                        vis = True
                        break
            if vis:
                out[name][f0:f1] = True
                cnt += 1
        kept[nm] = cnt
    return out, kept


def widen(mask, k=1):
    m = np.asarray(mask, bool).copy()
    for s in range(1, k + 1):
        m[:-s] |= mask[s:]
        m[s:] |= mask[:-s]
    return m


def shot_mask(A, names, n):
    m = np.zeros(n, bool)
    for nm in names:
        m |= A["shot"] == A["names"].index(nm)
    return m
