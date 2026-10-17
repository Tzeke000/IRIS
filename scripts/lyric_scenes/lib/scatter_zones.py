"""scatter_zones — "a box where you want grass, a box where you don't" (Zeke 10-15).

Generalises the boss-fight v3c courtyard grass (build_arena_v3.py::build_grass): a density field from zones +
falloffs, points computed in numpy, instanced by ONE Geometry Nodes tree (instances, never realized unless asked),
and the keep-clear / grounding asserts run on the REAL evaluated instances.

    import sys; sys.path.insert(0, "<repo>/scripts")
    from lyric_scenes.lib.scatter_zones import scatter, box, circle, poly, path, obj_bounds, obj_footprint, mask
    from lyric_scenes.lib.scatter_zones import near, noise, any_of, draw_zones

    rep = scatter("courtyard_grass", surface=[paving, terrain], items=[tuft_a, tuft_b, weed], weights=[3, 3, 1],
                  include=[box(-50, -78, 50, 86)],
                  exclude=[box(-25.5, -28, 25.5, 56), path(path_curve, width=3.4), obj_bounds(statue, pad=0.5)],
                  mode="random", density=18.0,
                  density_falloff=[any_of(near("include_edges", 3.5, outer=0.05), near(walls, 2.0)), noise(7.0)],
                  blockers=walls, seed=1)

Every random number is a hash of (seed, crc32(name), the candidate's lattice key, a stream id) — no Python hash(),
no global RNG — so the same call gives the same points on every run AND growing one zone does not reshuffle the
points that were already there. Exclusion always wins over inclusion. See profiles/blender/scatter_zones.md.
"""
import math
import time
import zlib

import numpy as np

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import interpolate_bezier

__all__ = ["scatter", "box", "circle", "poly", "path", "obj_bounds", "obj_footprint", "mask",
           "near", "noise", "any_of", "all_of", "draw_zones", "surface_bvh", "clear_cache"]

_TAG = "scatter_zones"
_INF = 1e18


# ===================================================================================== deterministic hashing
_M1, _M2, _GOLD = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB), np.uint64(0x9E3779B97F4A7C15)


def _mix(h):
    h = (h ^ (h >> np.uint64(30))) * _M1
    h = (h ^ (h >> np.uint64(27))) * _M2
    return h ^ (h >> np.uint64(31))


def _as_u64(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
    return a.view(np.uint64)


def _u01(base, keys, stream):
    """uniform [0,1) per row of integer keys (tuple of int64 arrays), independent per stream."""
    with np.errstate(over="ignore"):
        h = np.full(len(keys[0]), np.uint64(base) ^ (np.uint64(stream + 1) * _GOLD), dtype=np.uint64)
        for k in keys:
            h = _mix(h ^ (_as_u64(k) + _GOLD))
        h = _mix(h)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _seed_base(name, seed):
    return (zlib.crc32(name.encode("utf-8")) << 20) ^ (int(seed) & 0xFFFFFFFF) ^ 0x5CA77E2


def _smoothstep(e0, e1, x):
    if e0 == e1:
        return (x >= e0).astype(np.float64) if e1 >= e0 else (x <= e0).astype(np.float64)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ===================================================================================== geometry helpers
def _world_tris(o, dg, clip=None, margin=0.0):
    """(verts Nx3 world, tris Mx3) of the evaluated object; optionally only tris touching the XY clip box."""
    oe = o.evaluated_get(dg)
    me = oe.to_mesh()
    try:
        me.calc_loop_triangles()
        n = len(me.vertices)
        co = np.empty(n * 3, np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        tr = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tr)
        tr = tr.reshape(-1, 3)
    finally:
        oe.to_mesh_clear()
    M = np.array(o.matrix_world)
    co = co @ M[:3, :3].T + M[:3, 3]
    if clip is not None and len(tr):
        x0, y0, x1, y1 = clip
        tx, ty = co[tr, 0], co[tr, 1]
        keep = ((tx.max(1) >= x0 - margin) & (tx.min(1) <= x1 + margin) &
                (ty.max(1) >= y0 - margin) & (ty.min(1) <= y1 + margin))
        tr = tr[keep]
    return co, tr


def _as_objs(x):
    if x is None:
        return []
    if isinstance(x, bpy.types.Collection):
        return sorted(x.all_objects, key=lambda o: o.name)
    if isinstance(x, bpy.types.Object):
        return [x]
    out = []
    for e in x:
        out += _as_objs(e)
    return out


_BVH_CACHE = []        # [(key, clip_box_or_None, (bvh, zmin, zmax, ntris))] — reused across scatter() calls in one build
BVH_STATS = {"built": 0, "reused": 0, "build_s": 0.0}


def clear_cache():
    """forget cached BVHs (call after moving/editing objects that were used as surface / blockers / near targets)."""
    _BVH_CACHE.clear()


def _obj_sig(o):
    d = o.data
    n = len(d.vertices) if isinstance(d, bpy.types.Mesh) else 0
    return (o.name, n, tuple(round(v, 5) for row in o.matrix_world for v in row))


def surface_bvh(objs, clip=None, margin=2.0, flatten=False):
    """BVH over the evaluated world triangles of objs (clipped to an XY box + margin). Returns (bvh, zmin, zmax, ntris).
    Cached: a later call with the same objects whose clip box lies inside an earlier one reuses that BVH."""
    objs = [o for o in _as_objs(objs) if o.type in ("MESH", "CURVE", "SURFACE", "FONT", "META")]
    key = (tuple(_obj_sig(o) for o in objs), bool(flatten))
    want = None if clip is None else (clip[0] - margin, clip[1] - margin, clip[2] + margin, clip[3] + margin)
    for k, cb, res in _BVH_CACHE:
        if k == key and (cb is None or (want is not None and cb[0] <= want[0] and cb[1] <= want[1]
                                        and cb[2] >= want[2] and cb[3] >= want[3])):
            BVH_STATS["reused"] += 1
            return res
    t0 = time.time()
    dg = bpy.context.evaluated_depsgraph_get()
    V, T, off = [], [], 0
    for o in objs:
        co, tr = _world_tris(o, dg, want, 0.0)
        if not len(tr):
            continue
        used, inv = np.unique(tr, return_inverse=True)          # keep only the vertices the clipped tris use
        V.append(co[used])
        T.append(inv.reshape(-1, 3) + off)
        off += len(used)
    if not V:
        res = (None, 0.0, 0.0, 0)
    else:
        V = np.concatenate(V)
        T = np.concatenate(T)
        zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
        if flatten:
            V = V.copy()
            V[:, 2] = 0.0
        res = (BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True), zmin, zmax, len(T))
    _BVH_CACHE.append((key, want, res))
    BVH_STATS["built"] += 1
    BVH_STATS["build_s"] = round(BVH_STATS["build_s"] + time.time() - t0, 3)
    return res


def _hull2d(P):
    P = sorted(set(map(tuple, np.round(np.asarray(P, float)[:, :2], 6))))
    if len(P) < 3:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


def _seg_dist(P, A, B):
    ab = B - A
    L2 = float(ab @ ab)
    if L2 <= 1e-18:
        return np.hypot(P[:, 0] - A[0], P[:, 1] - A[1])
    t = np.clip(((P[:, 0] - A[0]) * ab[0] + (P[:, 1] - A[1]) * ab[1]) / L2, 0.0, 1.0)
    return np.hypot(P[:, 0] - (A[0] + t * ab[0]), P[:, 1] - (A[1] + t * ab[1]))


def _curve_polylines(o, resolution=None):
    """world-space polylines (list of Nx3) of a curve object's splines (bezier interpolated)."""
    M = o.matrix_world
    out = []
    for sp in o.data.splines:
        if sp.type == "BEZIER":
            bp = sp.bezier_points
            n = len(bp)
            res = max(2, int(resolution or sp.resolution_u))
            segs = n if sp.use_cyclic_u else n - 1
            pts = []
            for i in range(segs):
                a, b = bp[i], bp[(i + 1) % n]
                seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, res + 1)
                pts += list(seg[:-1])
            pts.append(bp[0].co if sp.use_cyclic_u else bp[n - 1].co)
        else:
            pts = [p.co.xyz for p in sp.points]
            if sp.use_cyclic_u and pts:
                pts.append(pts[0])
        if len(pts) >= 2:
            out.append(np.array([tuple(M @ Vector(p)) for p in pts], float))
    return out


def _polylines(src):
    """curve object | mesh object (edge chains -> segments) | list of points | list of point lists -> list of Nx3."""
    if isinstance(src, bpy.types.Object):
        if src.type == "CURVE":
            return _curve_polylines(src)
        if src.type == "MESH":
            M = np.array(src.matrix_world)
            co = np.array([tuple(v.co) for v in src.data.vertices], float) @ M[:3, :3].T + M[:3, 3]
            return [co[[e.vertices[0], e.vertices[1]]] for e in src.data.edges]
        raise TypeError(f"path source {src.name}: need a CURVE or MESH object")
    arr = None
    try:
        arr = np.asarray(src, float)
    except (ValueError, TypeError):
        arr = None
    if arr is not None and arr.ndim == 2:
        lines = [arr]
    else:
        lines = [np.asarray(p, float) for p in src]
    out = []
    for L in lines:
        if L.shape[1] == 2:
            L = np.concatenate([L, np.zeros((len(L), 1))], 1)
        out.append(L)
    return out


# ===================================================================================== zones
class Zone:
    """A 2-D footprint (+ optional z-range). sdist(P) < 0 inside; pad grows the zone; weight = include density mult."""
    kind = "zone"
    cost = 0           # 0 = vectorized numpy, 1 = per-point BVH queries (evaluated last, on survivors only)
    exact_dist = True  # False: sdist is only a sign (mask) -> footprint padding cannot be honoured

    def __init__(self, z=None, pad=0.0, weight=1.0, label=None):
        self.z = tuple(z) if z is not None else None
        self.pad = float(pad)
        self.weight = float(weight)
        self.label = label

    def _sd(self, P, reach):            # raw signed distance (no pad)
        raise NotImplementedError

    def _bb(self):                      # raw bbox (no pad)
        raise NotImplementedError

    def bbox(self, extra=0.0):
        x0, y0, x1, y1 = self._bb()
        e = self.pad + extra
        return (x0 - e, y0 - e, x1 + e, y1 + e)

    def sdist(self, P, reach=_INF):
        """signed distance with pad; points further than reach outside may return any value > reach."""
        P = np.asarray(P, float)
        out = np.full(len(P), _INF)
        x0, y0, x1, y1 = self.bbox(min(reach, 1e6))
        m = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
        if m.any():
            out[m] = self._sd(P[m], reach + self.pad) - self.pad
        if self.cost == 0 and reach >= 1e6:      # full distance requested: compute outside the bbox too
            if (~m).any():
                out[~m] = self._sd(P[~m], _INF) - self.pad
        return out

    def z_ok(self, z):
        if self.z is None:
            return np.ones(len(z), bool)
        return (z >= self.z[0]) & (z <= self.z[1])

    def outlines(self):
        x0, y0, x1, y1 = self.bbox()
        return [np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])]

    def name(self, i):
        return self.label or f"{self.kind}{i}"

    def describe(self):
        return {"kind": self.kind, "bbox": [round(v, 2) for v in self.bbox()], "z": self.z, "pad": self.pad}


class _Box(Zone):
    kind = "box"

    def __init__(self, cx, cy, hx, hy, angle, **kw):
        super().__init__(**kw)
        self.c = np.array([cx, cy], float)
        self.h = np.array([hx, hy], float)
        self.a = math.radians(angle)

    def _local(self, P):
        d = P - self.c
        ca, sa = math.cos(self.a), math.sin(self.a)
        return d[:, 0] * ca + d[:, 1] * sa, -d[:, 0] * sa + d[:, 1] * ca

    def _sd(self, P, reach):
        lx, ly = self._local(P)
        qx, qy = np.abs(lx) - self.h[0], np.abs(ly) - self.h[1]
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)

    def _corners(self, grow=0.0):
        ca, sa = math.cos(self.a), math.sin(self.a)
        hx, hy = self.h + grow
        loc = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        return np.array([(self.c[0] + x * ca - y * sa, self.c[1] + x * sa + y * ca) for x, y in loc])

    def _bb(self):
        c = self._corners()
        return (*c.min(0), *c.max(0))

    def outlines(self):
        c = self._corners(self.pad)
        return [np.vstack([c, c[:1]])]


def box(x0=None, y0=None, x1=None, y1=None, *, center=None, size=None, angle=0.0, z=None, pad=0.0, weight=1.0, label=None):
    """axis box box(x0, y0, x1, y1) or rotated box(center=(x, y), size=(w, h), angle=deg). z=(zmin, zmax) optional."""
    if center is not None:
        w, h = (size, size) if np.isscalar(size) else size
        return _Box(center[0], center[1], w / 2, h / 2, angle, z=z, pad=pad, weight=weight, label=label)
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return _Box((xa + xb) / 2, (ya + yb) / 2, (xb - xa) / 2, (yb - ya) / 2, angle, z=z, pad=pad, weight=weight, label=label)


class _Circle(Zone):
    kind = "circle"

    def __init__(self, c, r, **kw):
        super().__init__(**kw)
        self.c = np.array(c[:2], float)
        self.r = float(r)

    def _sd(self, P, reach):
        return np.hypot(P[:, 0] - self.c[0], P[:, 1] - self.c[1]) - self.r

    def _bb(self):
        return (self.c[0] - self.r, self.c[1] - self.r, self.c[0] + self.r, self.c[1] + self.r)

    def outlines(self):
        a = np.linspace(0, 2 * math.pi, 65)
        r = self.r + self.pad
        return [np.stack([self.c[0] + r * np.cos(a), self.c[1] + r * np.sin(a)], 1)]


def circle(center, r, *, z=None, pad=0.0, weight=1.0, label=None):
    return _Circle(center, r, z=z, pad=pad, weight=weight, label=label)


class _Poly(Zone):
    kind = "poly"

    def __init__(self, pts, **kw):
        super().__init__(**kw)
        P = np.asarray(pts, float)[:, :2]
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        assert len(P) >= 3, "poly needs >= 3 points"
        self.P = P

    def _sd(self, Q, reach):
        P = self.P
        n = len(P)
        inside = np.zeros(len(Q), bool)
        d = np.full(len(Q), _INF)
        for i in range(n):
            A, B = P[i], P[(i + 1) % n]
            cond = (A[1] > Q[:, 1]) != (B[1] > Q[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = A[0] + (Q[:, 1] - A[1]) * (B[0] - A[0]) / (B[1] - A[1])
            inside ^= cond & (Q[:, 0] < xint)
            d = np.minimum(d, _seg_dist(Q, A, B))
        return np.where(inside, -d, d)

    def _bb(self):
        return (*self.P.min(0), *self.P.max(0))

    def outlines(self):
        P = self.P
        out = [np.vstack([P, P[:1]])]
        if self.pad > 0:                                   # mitred offset of the polygon by pad (drawn + the original)
            area = 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
            sgn = 1.0 if area > 0 else -1.0
            e = np.roll(P, -1, 0) - P
            e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            nrm = sgn * np.stack([e[:, 1], -e[:, 0]], 1)            # outward edge normals
            n_prev = np.roll(nrm, 1, 0)
            b = nrm + n_prev
            b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
            cosh = np.maximum(np.sum(b * nrm, axis=1), 0.2)
            Q = P + b * (self.pad / cosh)[:, None]
            out.insert(0, np.vstack([Q, Q[:1]]))
        return out


def poly(points, *, z=None, pad=0.0, weight=1.0, label=None):
    """any simple polygon [(x, y), ...] (closed implicitly; even-odd rule)."""
    return _Poly(points, z=z, pad=pad, weight=weight, label=label)


class _Path(Zone):
    kind = "path"

    def __init__(self, lines, width, **kw):
        super().__init__(**kw)
        self.lines = [L[:, :2] for L in lines]
        segs = [(L[i], L[i + 1]) for L in self.lines for i in range(len(L) - 1)]
        self.A = np.array([s[0] for s in segs])
        self.B = np.array([s[1] for s in segs])
        self.hw = width / 2.0

    def _sd(self, Q, reach):
        d = np.full(len(Q), _INF)
        r = self.hw + reach
        order = np.argsort(Q[:, 0], kind="stable")
        xs = Q[order, 0]
        for A, B in zip(self.A, self.B):
            if r < 1e6:
                lo = np.searchsorted(xs, min(A[0], B[0]) - r, "left")
                hi = np.searchsorted(xs, max(A[0], B[0]) + r, "right")
                idx = order[lo:hi]
                if not len(idx):
                    continue
                y = Q[idx, 1]
                idx = idx[(y >= min(A[1], B[1]) - r) & (y <= max(A[1], B[1]) + r)]
                if not len(idx):
                    continue
                d[idx] = np.minimum(d[idx], _seg_dist(Q[idx], A, B))
            else:
                d = np.minimum(d, _seg_dist(Q, A, B))
        return d - self.hw

    def _bb(self):
        P = np.vstack([self.A, self.B])
        return (P[:, 0].min() - self.hw, P[:, 1].min() - self.hw, P[:, 0].max() + self.hw, P[:, 1].max() + self.hw)

    def outlines(self):
        out = []
        w = self.hw + self.pad
        for L in self.lines:
            out.append(L)
            if len(L) < 2:
                continue
            t = np.gradient(L, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            n = np.stack([-t[:, 1], t[:, 0]], 1)
            out += [L + n * w, L - n * w]
        return out


def path(src, width=2.0, *, z=None, pad=0.0, weight=1.0, label=None):
    """a strip `width` wide along a curve object / mesh edges / point list(s). The 2-D distance is exact (capsules)."""
    return _Path(_polylines(src), width, z=z, pad=pad, weight=weight, label=label)


def obj_bounds(o, pad=0.0, *, use_z=False, weight=1.0, label=None):
    """the convex hull of the object's world bound-box corners in XY (rotation-aware). use_z: also its z-range."""
    if isinstance(o, (list, tuple)):
        return [obj_bounds(e, pad, use_z=use_z, weight=weight) for e in o]
    C = np.array([tuple(o.matrix_world @ Vector(c)) for c in o.bound_box])
    z = (float(C[:, 2].min()) - pad, float(C[:, 2].max()) + pad) if use_z else None
    zn = _Poly(_hull2d(C), z=z, pad=pad, weight=weight, label=label or f"bounds:{o.name}")
    zn.kind = "obj_bounds"
    return zn


class _Footprint(Zone):
    """the exact XY shadow of a mesh (all its triangles flattened to z=0): ray-cast for inside, nearest for distance."""
    kind = "obj_footprint"
    cost = 1

    def __init__(self, objs, **kw):
        super().__init__(**kw)
        self.objs = _as_objs(objs)
        self.bvh, _, _, self.ntris = surface_bvh(self.objs, flatten=True)
        assert self.bvh is not None, f"obj_footprint: no triangles in {[o.name for o in self.objs]}"
        C = np.array([tuple(o.matrix_world @ Vector(c)) for o in self.objs for c in o.bound_box])
        self._b = (C[:, 0].min(), C[:, 1].min(), C[:, 0].max(), C[:, 1].max())
        self._hull = _hull2d(C)

    def _sd(self, P, reach):
        out = np.empty(len(P))
        bvh, down = self.bvh, Vector((0, 0, -1))
        rr = min(reach, 1e6)
        for i, (x, y) in enumerate(P.tolist()):
            if bvh.ray_cast(Vector((x, y, 1.0)), down, 2.0)[0] is not None:
                out[i] = -1e-6
            else:
                hit = bvh.find_nearest(Vector((x, y, 0.0)), rr + 1e-3)
                out[i] = hit[3] if hit[0] is not None else rr + 1.0
        return out

    def _bb(self):
        return self._b

    def outlines(self):
        h = self._hull
        return [np.vstack([h, h[:1]])]


def obj_footprint(objs, pad=0.0, *, z=None, weight=1.0, label=None):
    """exact XY footprint of mesh object(s) (e.g. a road ribbon, a building) — slower than obj_bounds, but exact."""
    names = ",".join(o.name for o in _as_objs(objs))[:60]
    return _Footprint(objs, z=z, pad=pad, weight=weight, label=label or f"footprint:{names}")


class _Mask(Zone):
    kind = "mask"
    exact_dist = False

    def __init__(self, fn, bb, **kw):
        super().__init__(**kw)
        self.fn = fn
        self._b = bb if bb is not None else (-1e6, -1e6, 1e6, 1e6)

    def _sd(self, P, reach):
        return np.where(np.asarray(self.fn(P[:, 0], P[:, 1]), bool), -1e-6, _INF)

    def _bb(self):
        return self._b


def mask(fn, bbox=None, *, z=None, weight=1.0, label=None):
    """custom zone: fn(x_array, y_array) -> bool array (True = in the zone). Give a bbox when used as an include."""
    return _Mask(fn, bbox, z=z, weight=weight, label=label)


def _zone(z, i, kind):
    if isinstance(z, Zone):
        return z
    if isinstance(z, bpy.types.Object):
        return obj_bounds(z)
    raise TypeError(f"{kind}[{i}]: not a zone ({type(z).__name__}); use box/circle/poly/path/obj_bounds/obj_footprint/mask")


def _zones(lst, kind):
    out = []
    for i, z in enumerate(lst or []):
        if isinstance(z, (list, tuple)) and z and isinstance(z[0], Zone):
            out += list(z)
        else:
            out.append(_zone(z, i, kind))
    return out


# ===================================================================================== density falloffs
class _Term:
    late = False      # needs the surface z (evaluated after the ray-casts, on survivors only)

    def __call__(self, ctx):
        raise NotImplementedError


class near(_Term):
    """denser NEAR something: multiplier = inner at distance <= start, easing to outer at distance >= radius.

    targets: "include_edges" (inside the include zones, distance to their border = the v3 'along the walls' band),
             "exclude_edges" (around the keep-out zones), zone(s), or object(s) (3-D nearest distance to their
             evaluated mesh, measured from the ground point + lift — the v3 'around building / rubble bases' band).
    curve: "smooth" | "linear".
    """

    def __init__(self, targets, radius, inner=1.0, outer=0.0, start=0.0, curve="smooth", lift=0.3):
        self.targets, self.radius, self.inner, self.outer = targets, float(radius), float(inner), float(outer)
        self.start, self.curve, self.lift = float(start), curve, float(lift)
        if isinstance(targets, str):
            assert targets in ("include_edges", "exclude_edges"), targets
            self.zones, self.objs = None, []
        else:
            lst = targets if isinstance(targets, (list, tuple)) else [targets]
            self.zones = [t for t in lst if isinstance(t, Zone)]
            self.objs = _as_objs([t for t in lst if not isinstance(t, Zone)])
        self.late = bool(self.objs)
        self._bvh = None

    def _ease(self, d):
        if self.curve == "linear":
            t = np.clip((self.radius - d) / max(self.radius - self.start, 1e-9), 0.0, 1.0)
        else:
            t = _smoothstep(self.radius, self.start, d)
        return self.outer + (self.inner - self.outer) * t

    def __call__(self, ctx):
        P = ctx["xy"]
        d = np.full(len(P), _INF)
        R = self.radius + 1e-3
        if self.targets == "include_edges":
            depth = np.full(len(P), -_INF)
            for zn in ctx["include"]:
                depth = np.maximum(depth, -zn.sdist(P, R))
            d = np.maximum(depth, 0.0)
        elif self.targets == "exclude_edges":
            for zn in ctx["exclude"]:
                d = np.minimum(d, np.maximum(zn.sdist(P, R), 0.0))
        else:
            for zn in self.zones:
                d = np.minimum(d, np.maximum(zn.sdist(P, R), 0.0))
            if self.objs:
                if self._bvh is None:
                    self._bvh = surface_bvh(self.objs, clip=ctx["region"], margin=max(5.0, self.radius + 1.0))[0]
                if self._bvh is not None:
                    z = ctx["z"]
                    bvh = self._bvh
                    for i, (x, y, zz) in enumerate(zip(P[:, 0].tolist(), P[:, 1].tolist(), z.tolist())):
                        h = bvh.find_nearest(Vector((x, y, zz + self.lift)), R)
                        if h[0] is not None:
                            d[i] = min(d[i], h[3])
        return self._ease(d)


def _perlin2(x, y, seed):
    rs = np.random.default_rng(seed)
    perm = rs.permutation(256).astype(np.int64)
    perm = np.concatenate([perm, perm])
    ang = rs.uniform(0.0, 2.0 * math.pi, 256)
    gx, gy = np.cos(ang), np.sin(ang)
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    xi, yi = x0.astype(np.int64) & 255, y0.astype(np.int64) & 255

    def g(ix, iy, dx, dy):
        h = perm[perm[ix] + iy]
        return gx[h] * dx + gy[h] * dy
    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    n00, n10 = g(xi, yi, fx, fy), g((xi + 1) & 255, yi, fx - 1, fy)
    n01, n11 = g(xi, (yi + 1) & 255, fx, fy - 1), g((xi + 1) & 255, (yi + 1) & 255, fx - 1, fy - 1)
    a = n00 + u * (n10 - n00)
    b = n01 + u * (n11 - n01)
    return (a + v * (b - a)) * 1.414


class noise(_Term):
    """patchiness: Perlin noise (cell `scale` m, ~[-1, 1]) through smoothstep(lo, hi) mapped to [floor, 1].
    v3c used noise(7.0, -0.15, 0.35) for the patches and noise(1.6, -0.4, 0.4, floor=0.6) for the clumps."""

    def __init__(self, scale, lo=-0.15, hi=0.35, floor=0.0, seed=0):
        self.scale, self.lo, self.hi, self.floor, self.seed = float(scale), float(lo), float(hi), float(floor), int(seed)

    def __call__(self, ctx):
        P = ctx["xy"]
        n = _perlin2(P[:, 0] / self.scale, P[:, 1] / self.scale, (ctx["seed"] * 7919 + self.seed) & 0xFFFFFFFF)
        return self.floor + (1.0 - self.floor) * _smoothstep(self.lo, self.hi, n)


class _Combine(_Term):
    def __init__(self, terms, op):
        self.terms = [_term(t) for t in terms]
        self.op = op
        self.late = any(t.late for t in self.terms)

    def __call__(self, ctx):
        vals = [t(ctx) for t in self.terms]
        out = vals[0]
        for v in vals[1:]:
            out = np.maximum(out, v) if self.op == "max" else out * v
        return out


def any_of(*terms):
    """max of the terms (e.g. dense along the walls OR around rubble)."""
    return _Combine(terms, "max")


def all_of(*terms):
    """product of the terms (a plain list does the same)."""
    return _Combine(terms, "mul")


class _Fn(_Term):
    def __init__(self, fn):
        import inspect
        self.fn = fn
        try:
            self.late = len(inspect.signature(fn).parameters) >= 3
        except (TypeError, ValueError):
            self.late = True

    def __call__(self, ctx):
        P = ctx["xy"]
        if self.late:
            return np.asarray(self.fn(P[:, 0], P[:, 1], ctx["z"]), float)
        return np.asarray(self.fn(P[:, 0], P[:, 1]), float)


def _term(t):
    if isinstance(t, _Term):
        return t
    if callable(t):
        return _Fn(t)
    if np.isscalar(t):
        v = float(t)
        return _Fn(lambda x, y: np.full(len(x), v))
    raise TypeError(f"density_falloff term {t!r}")


# ===================================================================================== candidates
def _union_keys(bbs, cell_fn):
    keys = [cell_fn(bb) for bb in bbs]
    keys = [k for k in keys if len(k)]
    if not keys:
        return np.zeros((0, 2), np.int64)
    return np.unique(np.concatenate(keys), axis=0)


def _cells_random(bb, cell):
    i0, i1 = math.floor(bb[0] / cell), math.floor(bb[2] / cell)
    j0, j1 = math.floor(bb[1] / cell), math.floor(bb[3] / cell)
    I, J = np.meshgrid(np.arange(i0, i1 + 1, dtype=np.int64), np.arange(j0, j1 + 1, dtype=np.int64))
    return np.stack([I.ravel(), J.ravel()], 1)


def _lattice_frame(angle, origin):
    a = math.radians(angle)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)]), np.asarray(origin, float)


def _cells_lattice(bb, sx, sy, u, v, o, stagger):
    C = np.array([(bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3])]) - o
    lu, lv = C @ u, C @ v
    i0, i1 = math.floor(lu.min() / sx) - 1, math.ceil(lu.max() / sx) + 1
    j0, j1 = math.floor(lv.min() / sy), math.ceil(lv.max() / sy)
    I, J = np.meshgrid(np.arange(i0, i1 + 1, dtype=np.int64), np.arange(j0, j1 + 1, dtype=np.int64))
    return np.stack([I.ravel(), J.ravel()], 1)


def _resample(L, spacing, start, end):
    seg = np.linalg.norm(np.diff(L[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    st = np.arange(start, total - end + 1e-9, spacing)
    if not len(st):
        return np.zeros((0, 3)), np.zeros((0, 2)), st
    k = np.clip(np.searchsorted(s, st, "right") - 1, 0, len(seg) - 1)
    t = (st - s[k]) / np.maximum(seg[k], 1e-12)
    P = L[k] + (L[k + 1] - L[k]) * t[:, None]
    T = (L[k + 1] - L[k])[:, :2]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    return P, T, st


# ===================================================================================== rotation
def _euler_xyz(normals, yaw, t):
    """Euler XYZ of R_align(+Z -> n blended by t) @ Rz(yaw), vectorized."""
    n = np.asarray(normals, float)
    n = (1.0 - t) * np.array([0.0, 0.0, 1.0]) + t * n
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    kx, ky, c = -n[:, 1], n[:, 0], n[:, 2]
    f = 1.0 / np.maximum(1.0 + c, 1e-9)
    N = len(n)
    K = np.zeros((N, 3, 3))
    K[:, 0, 2], K[:, 2, 0] = ky, -ky
    K[:, 1, 2], K[:, 2, 1] = -kx, kx
    R = np.eye(3)[None] + K + (K @ K) * f[:, None, None]
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rz = np.zeros((N, 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cz, -sz, sz, cz, 1.0
    R = R @ Rz
    ey = np.arcsin(np.clip(-R[:, 2, 0], -1, 1))
    ex = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    ez = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.stack([ex, ey, ez], 1)


# ===================================================================================== geometry nodes
_ATTR_ITEM, _ATTR_ROT, _ATTR_SCALE = "scatter_item", "scatter_rot", "scatter_scale"


def _remove_old(name):
    o = bpy.data.objects.get(name)
    if o is not None and o.get(_TAG):
        me = o.data
        for md in o.modifiers:
            if md.type == "NODES" and md.node_group and md.node_group.get(_TAG):
                bpy.data.node_groups.remove(md.node_group)
        bpy.data.objects.remove(o)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    elif o is not None:
        raise RuntimeError(f"scatter: an object named {name!r} exists and is not a scatter holder — pick another name")


def _build_tree(name, items, realize):
    ng = bpy.data.node_groups.new("GN_Scatter_" + name, "GeometryNodeTree")
    ng[_TAG] = True
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    g2i = N.new("GeometryNodeGeometryToInstance")
    # MEASURED 10-15 on 5.2: a multi-input socket puts the LAST-linked geometry at instance index 0, so link in
    # reverse to make Instance Index k == items[k] (asserted on the evaluated instances by scatter()).
    for k, o in reversed(list(enumerate(items))):
        oi = N.new("GeometryNodeObjectInfo")
        oi.inputs["Object"].default_value = o
        oi.transform_space = "ORIGINAL"
        oi.location = (-600, -200 - 160 * k)
        L.new(oi.outputs["Geometry"], g2i.inputs["Geometry"])

    def attr(nm, dt, y):
        a = N.new("GeometryNodeInputNamedAttribute")
        a.data_type = dt
        a.inputs["Name"].default_value = nm
        a.location = (-400, y)
        return a.outputs["Attribute"]
    e2r = N.new("FunctionNodeEulerToRotation")
    L.new(attr(_ATTR_ROT, "FLOAT_VECTOR", 200), e2r.inputs["Euler"])
    iop = N.new("GeometryNodeInstanceOnPoints")
    L.new(gi.outputs["Geometry"], iop.inputs["Points"])
    L.new(g2i.outputs["Instances"], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    L.new(attr(_ATTR_ITEM, "INT", 100), iop.inputs["Instance Index"])
    L.new(e2r.outputs["Rotation"], iop.inputs["Rotation"])
    L.new(attr(_ATTR_SCALE, "FLOAT", 0), iop.inputs["Scale"])
    out = iop.outputs["Instances"]
    if realize:
        rl = N.new("GeometryNodeRealizeInstances")
        L.new(out, rl.inputs["Geometry"])
        out = rl.outputs["Geometry"]
    L.new(out, go.inputs["Geometry"])
    gi.location, iop.location, go.location = (-600, 200), (0, 0), (300, 0)
    return ng


def _scatter_collection(collection):
    if collection is not None:
        return collection
    c = bpy.data.collections.get("SCATTER")
    if c is None:
        c = bpy.data.collections.new("SCATTER")
        bpy.context.scene.collection.children.link(c)
    return c


def _create(name, P, item, rot, scl, items, realize, shadows, collection):
    me = bpy.data.meshes.new("Scatter_" + name)
    me.vertices.add(len(P))
    me.vertices.foreach_set("co", P.astype(np.float32).ravel())
    me.attributes.new(_ATTR_ITEM, "INT", "POINT").data.foreach_set("value", item.astype(np.int32))
    me.attributes.new(_ATTR_ROT, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", rot.astype(np.float32).ravel())
    me.attributes.new(_ATTR_SCALE, "FLOAT", "POINT").data.foreach_set("value", scl.astype(np.float32))
    me.update()
    o = bpy.data.objects.new(name, me)
    o[_TAG] = True
    _scatter_collection(collection).objects.link(o)
    o.visible_shadow = bool(shadows)
    md = o.modifiers.new("Scatter", "NODES")
    md.node_group = _build_tree(name, items, realize)
    return o


def _evaluated_instances(holder, sample=2000):
    """world origins of every evaluated instance of the holder + the vertex counts of the first `sample` ones."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    out, vc = [], []
    for inst in dg.object_instances:
        if inst.is_instance and inst.parent is not None and inst.parent.original == holder:
            out.append(tuple(inst.matrix_world.translation))
            if len(vc) < sample:
                d = inst.object.data
                vc.append(len(d.vertices) if isinstance(d, bpy.types.Mesh) else -1)
    return np.array(out, float).reshape(-1, 3), vc


# ===================================================================================== main entry
def scatter(name, surface=None, items=None, include=None, exclude=(), mode="random", density=4.0, *,
            spacing=None, row_spacing=None, angle=0.0, jitter=0.0, stagger=False, origin=(0.0, 0.0),
            path=None, offset=0.0, both_sides=False, face="path", path_start=0.0, path_end=0.0,
            density_falloff=None, weights=None, scale=(0.8, 1.2), rotate_z=None, align_to_normal=False,
            seed=1, min_spacing=None, spacing_factor=1.0, respect_footprint=True, fit_inside=False,
            blockers=None, z_fn=None, z=0.0, sink=0.0, max_slope=None,
            shadows=False, as_instances=True, hide_items=True, collection=None, create=True,
            tol=0.02, max_candidates=8_000_000, verbose=True):
    """Scatter `items` over `surface` inside the include zones and outside the exclude zones. Returns a report dict.

    mode: "random"  stratified-jittered (blue-ish) points at `density` per m^2, thinned by zone weight x falloff;
                    + min_spacing (m, or "auto" = item footprint radii x spacing_factor) for props that must not overlap.
          "rows"    crop rows: `spacing` along each row, `row_spacing` between rows, rows run at `angle` degrees.
          "grid"    lattice `spacing` (float or (sx, sy)) at `angle`; stagger=True -> hex-like offset rows.
          "along_path" stations every `spacing` m along `path` (curve obj / points / list), `offset` m to the side
                    (both_sides), facing `face` = "path" | "away" | "along" | None (item local +Y = its front).
          For rows/grid/along_path `density` doubles as `spacing` when spacing is None.
    surface: object(s) to ray-cast down onto (first hit from above wins), else z_fn(x, y) -> z, else constant z.
    blockers: object(s) — a point whose downward ray hits a blocker above the surface is dropped (under a roof/rock).
    include zones: union; empty/None is only allowed for along_path. exclude zones ALWAYS win (with
    respect_footprint the item's footprint radius x scale must also stay out; fit_inside=True also keeps the
    footprint inside the include zones). Zones with z=(lo, hi) test the ground z.
    density_falloff: list of terms multiplied (near / noise / any_of / all_of / callable(x, y[, z]) / float).
    scale: float or (lo, hi). rotate_z: True = random yaw, False = none, float = +-deg jitter; default random for
    area modes, False for along_path. align_to_normal: bool or 0..1 blend toward the surface normal.
    create=False: compute only; the report then carries "points" (Nx3), "items", "rotation", "scale" arrays.
    """
    T0 = time.time()
    tm = {}

    def tick(k, t):
        tm[k] = round(time.time() - t, 3)
        return time.time()
    t = time.time()
    base = _seed_base(name, seed)
    items = _as_objs(items)
    if create and not items:
        raise ValueError("scatter: items are required when create=True")
    nI = max(1, len(items))
    w = np.ones(nI) if weights is None else np.asarray(weights, float)
    assert len(w) == nI and (w >= 0).all() and w.sum() > 0, "weights must match items"
    wc = np.cumsum(w) / w.sum()
    rad = np.zeros(nI)
    for k, o in enumerate(items):
        rad[k] = max(math.hypot(c[0], c[1]) for c in o.bound_box)
    s_lo, s_hi = (float(scale), float(scale)) if np.isscalar(scale) else (float(scale[0]), float(scale[1]))
    if rotate_z is None:
        rotate_z = mode != "along_path"
    INC = _zones(include, "include")
    EXC = _zones(exclude, "exclude")
    for i, zn in enumerate(INC):
        zn.label = zn.name(i)
    for i, zn in enumerate(EXC):
        zn.label = zn.name(i)
    terms = [_term(x) for x in (density_falloff or [])]
    if mode not in ("random", "rows", "grid", "along_path"):
        raise ValueError(f"mode {mode!r}")
    if not INC and mode != "along_path":
        raise ValueError("scatter: include zones are required for random/rows/grid")

    # ---------------------------------------------------------------- candidates (+ integer keys for hashing)
    yaw0 = None
    if mode == "random":
        wmax = max([zn.weight for zn in INC] + [1e-9])
        cell = 1.0 / math.sqrt(density * wmax)
        n_est = sum((zn.bbox()[2] - zn.bbox()[0]) * (zn.bbox()[3] - zn.bbox()[1]) for zn in INC) / cell ** 2
        if n_est > max_candidates:
            raise ValueError(f"scatter {name}: ~{n_est:.0f} candidates > max_candidates={max_candidates}; "
                             f"lower density or shrink the include zones")
        K = _union_keys([zn.bbox() for zn in INC], lambda bb: _cells_random(bb, cell))
        keys = (K[:, 0], K[:, 1])
        jx, jy = _u01(base, keys, 1), _u01(base, keys, 2)
        P = np.stack([(K[:, 0] + jx) * cell, (K[:, 1] + jy) * cell], 1)
    elif mode in ("rows", "grid"):
        sp = spacing if spacing is not None else density
        if mode == "rows":
            sx = float(sp)
            sy = float(row_spacing if row_spacing is not None else sp)
        else:
            sx, sy = (float(sp), float(sp)) if np.isscalar(sp) else (float(sp[0]), float(sp[1]))
        u, v, o = _lattice_frame(angle, origin)
        K = _union_keys([zn.bbox() for zn in INC], lambda bb: _cells_lattice(bb, sx, sy, u, v, o, stagger))
        if len(K) > max_candidates:
            raise ValueError(f"scatter {name}: {len(K)} lattice candidates > max_candidates")
        keys = (K[:, 0], K[:, 1])
        li = K[:, 0] * sx + (np.where(K[:, 1] % 2 == 1, 0.5 * sx, 0.0) if stagger else 0.0)
        lj = K[:, 1] * sy
        jit = (jitter, jitter) if np.isscalar(jitter) else jitter
        li = li + (_u01(base, keys, 1) * 2 - 1) * jit[0]
        lj = lj + (_u01(base, keys, 2) * 2 - 1) * jit[1]
        P = o + li[:, None] * u + lj[:, None] * v
    else:  # along_path
        assert path is not None, "along_path needs path="
        sp = float(spacing if spacing is not None else density)
        Ps, Ys, Ks = [], [], []
        for li_, L in enumerate(_polylines(path)):
            S, Tn, st = _resample(L, sp, path_start, path_end)
            if not len(S):
                continue
            nrm = np.stack([-Tn[:, 1], Tn[:, 0]], 1)            # left of travel
            sides = (1, -1) if both_sides else (1,)
            for sd in sides:
                q = S[:, :2] + nrm * (offset * sd)
                if face == "path":
                    d = -nrm * sd
                elif face == "away":
                    d = nrm * sd
                elif face == "along":
                    d = Tn
                else:
                    d = None
                yaw = np.arctan2(-d[:, 0], d[:, 1]) if d is not None else np.zeros(len(q))
                Ps.append(q)
                Ys.append(yaw)
                Ks.append(np.stack([np.full(len(q), li_), np.arange(len(q)), np.full(len(q), sd)], 1))
        P = np.concatenate(Ps) if Ps else np.zeros((0, 2))
        yaw0 = np.concatenate(Ys) if Ys else np.zeros(0)
        K3 = np.concatenate(Ks).astype(np.int64) if Ks else np.zeros((0, 3), np.int64)
        keys = (K3[:, 0], K3[:, 1], K3[:, 2])
        jit = (jitter, jitter) if np.isscalar(jitter) else jitter
        if len(P) and (jit[0] or jit[1]):
            P = P + np.stack([(_u01(base, keys, 1) * 2 - 1) * jit[0], (_u01(base, keys, 2) * 2 - 1) * jit[1]], 1)
    n_cand = len(P)
    # per-candidate attributes (hash streams -> independent of filtering order)
    item = np.minimum(np.searchsorted(wc, _u01(base, keys, 3), "right"), nI - 1).astype(np.int64) if n_cand else np.zeros(0, np.int64)
    scl = s_lo + (s_hi - s_lo) * _u01(base, keys, 4) if n_cand else np.zeros(0)
    if rotate_z is True:
        yaw = _u01(base, keys, 5) * 2 * math.pi if n_cand else np.zeros(0)
    elif rotate_z is False or rotate_z == 0:
        yaw = np.zeros(n_cand)
    else:
        yaw = (_u01(base, keys, 5) * 2 - 1) * math.radians(float(rotate_z))
    if yaw0 is not None and rotate_z is not True:
        yaw = yaw + yaw0
    keep_u = _u01(base, keys, 6) if n_cand else np.zeros(0)
    late_u = _u01(base, keys, 7) if n_cand else np.zeros(0)
    rpt = rad[item] * scl if (respect_footprint or fit_inside) else np.zeros(n_cand)
    t = tick("candidates", t)

    # ---------------------------------------------------------------- 2-D zone filter (exclusion wins)
    idx = np.arange(n_cand)
    wgt = np.ones(n_cand)
    stats_exc = {zn.label: 0 for zn in EXC}
    if INC:
        wgt = np.zeros(n_cand)
        for zn in sorted(INC, key=lambda z_: z_.cost):
            sd = zn.sdist(P, 0.0)
            m = sd < (-rpt if (fit_inside and zn.exact_dist) else 0.0)
            wgt = np.where(m, np.maximum(wgt, zn.weight), wgt)
        idx = idx[wgt > 0]
    n_inc = len(idx)
    rmax = float(rpt.max()) if len(rpt) else 0.0
    for zn in sorted(EXC, key=lambda z_: z_.cost):
        if zn.z is not None or not len(idx):
            continue                                  # z-ranged keep-outs are tested after the ground is known
        sd = zn.sdist(P[idx], rmax + 1e-3)
        bad = sd < (rpt[idx] if zn.exact_dist else 0.0)
        stats_exc[zn.label] += int(bad.sum())
        idx = idx[~bad]
    t = tick("zones", t)

    # ---------------------------------------------------------------- early density falloff
    wmax_ = max([zn.weight for zn in INC] + [1.0]) if mode == "random" else 1.0
    ctx = {"include": INC, "exclude": EXC, "seed": int(seed), "region": None}   # noise ignores the name: same field across calls
    if INC:
        allb = [zn.bbox() for zn in INC]
    elif n_cand:
        allb = [(P[:, 0].min() - 1, P[:, 1].min() - 1, P[:, 0].max() + 1, P[:, 1].max() + 1)]
    else:
        allb = [(0.0, 0.0, 1.0, 1.0)]
    region = (min(b[0] for b in allb), min(b[1] for b in allb), max(b[2] for b in allb), max(b[3] for b in allb))
    ctx["region"] = region
    early = [tm_ for tm_ in terms if not tm_.late]
    late = [tm_ for tm_ in terms if tm_.late]
    p = wgt[idx] / wmax_
    if early and len(idx):
        ctx["xy"] = P[idx]
        for tm_ in early:
            p = p * tm_(ctx)
    keep = keep_u[idx] < np.clip(p, 0, 1)
    n_thin_early = int((~keep).sum())
    idx = idx[keep]
    t = tick("falloff_early", t)

    # ---------------------------------------------------------------- ground: ray-cast / z_fn / constant
    n_nohit = n_covered = n_slope = 0
    Z = np.zeros(len(idx))
    NRM = np.tile([0.0, 0.0, 1.0], (len(idx), 1))
    surf_bvh = None
    if surface is not None and len(idx):
        surf_bvh, zlo, zhi, ntr = surface_bvh(surface, clip=region, margin=5.0)
        assert surf_bvh is not None, f"scatter {name}: surface has no triangles inside the include region"
        blk = surface_bvh(blockers, clip=region, margin=5.0) if blockers else (None, zlo, zhi, 0)
        top = max(zhi, blk[2]) + 1.0
        maxd = top - min(zlo, blk[1]) + 2.0
        down = Vector((0, 0, -1))
        ok = np.zeros(len(idx), bool)
        covered = np.zeros(len(idx), bool)
        xy = P[idx].tolist()
        for i, (x, y) in enumerate(xy):
            o_ = Vector((x, y, top))
            loc, nr, _, _ = surf_bvh.ray_cast(o_, down, maxd)
            if loc is None:
                continue
            ok[i] = True
            Z[i] = loc.z
            if nr.z < 0:
                nr = -nr
            NRM[i] = nr
            if blk[0] is not None:
                bl = blk[0].ray_cast(o_, down, maxd)[0]
                if bl is not None and bl.z > loc.z + 0.02:
                    covered[i] = True
        n_nohit = int((~ok).sum())
        n_covered = int((ok & covered).sum())
        sel = ok & ~covered
    elif len(idx):
        if z_fn is not None:
            x_, y_ = P[idx, 0], P[idx, 1]
            Z = np.asarray(z_fn(x_, y_), float)
            e = 0.05
            dzx = (np.asarray(z_fn(x_ + e, y_), float) - np.asarray(z_fn(x_ - e, y_), float)) / (2 * e)
            dzy = (np.asarray(z_fn(x_, y_ + e), float) - np.asarray(z_fn(x_, y_ - e), float)) / (2 * e)
            NRM = np.stack([-dzx, -dzy, np.ones_like(dzx)], 1)
            NRM /= np.linalg.norm(NRM, axis=1, keepdims=True)
        else:
            Z = np.full(len(idx), float(z))
        sel = np.ones(len(idx), bool)
    else:
        sel = np.zeros(0, bool)
    if max_slope is not None and len(idx):
        steep = NRM[:, 2] < math.cos(math.radians(max_slope))
        n_slope = int((sel & steep).sum())
        sel &= ~steep
    idx, Z, NRM = idx[sel], Z[sel], NRM[sel]
    t = tick("ground", t)

    # ---------------------------------------------------------------- z-ranged zones
    if any(zn.z is not None for zn in INC) and len(idx):
        okz = np.zeros(len(idx), bool)
        for zn in INC:
            okz |= (zn.sdist(P[idx], 0.0) < 0) & zn.z_ok(Z)
        idx, Z, NRM = idx[okz], Z[okz], NRM[okz]
    for zn in EXC:
        if zn.z is None or not len(idx):
            continue
        sd = zn.sdist(P[idx], rmax + 1e-3)
        bad = (sd < (rpt[idx] if zn.exact_dist else 0.0)) & zn.z_ok(Z)
        stats_exc[zn.label] += int(bad.sum())
        idx, Z, NRM = idx[~bad], Z[~bad], NRM[~bad]

    # ---------------------------------------------------------------- late falloff (needs ground z)
    n_thin_late = 0
    if late and len(idx):
        ctx["xy"], ctx["z"] = P[idx], Z
        pl = np.ones(len(idx))
        for tm_ in late:
            pl = pl * tm_(ctx)
        k2 = late_u[idx] < np.clip(pl, 0, 1)
        n_thin_late = int((~k2).sum())
        idx, Z, NRM = idx[k2], Z[k2], NRM[k2]
    t = tick("falloff_late", t)

    # ---------------------------------------------------------------- min spacing (greedy, big items first)
    n_spacing = 0
    if min_spacing is not None and len(idx):
        if min_spacing == "auto":
            rr = rad[item[idx]] * scl[idx] * float(spacing_factor)
            fixed = 0.0
        else:
            rr = np.zeros(len(idx))
            fixed = float(min_spacing)
        cellm = max(2 * float(rr.max()), fixed, 1e-6)
        order = np.argsort(-rr, kind="stable")
        xy = P[idx].tolist()
        rl = rr.tolist()
        grid = {}
        kept = np.zeros(len(idx), bool)
        for i in order.tolist():
            x, y = xy[i]
            cx, cy = math.floor(x / cellm), math.floor(y / cellm)
            okp = True
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for j in grid.get((gx, gy), ()):
                        need = max(rl[i] + rl[j], fixed)
                        dx, dy = x - xy[j][0], y - xy[j][1]
                        if dx * dx + dy * dy < need * need:
                            okp = False
                            break
                    if not okp:
                        break
                if not okp:
                    break
            if okp:
                kept[i] = True
                grid.setdefault((cx, cy), []).append(i)
        n_spacing = int((~kept).sum())
        idx, Z, NRM = idx[kept], Z[kept], NRM[kept]
    t = tick("spacing", t)

    # ---------------------------------------------------------------- final arrays + python-side asserts
    XY = P[idx]
    PTS = np.concatenate([XY, (Z - sink)[:, None]], 1) if len(idx) else np.zeros((0, 3))
    ITEM, SCL = item[idx], scl[idx]
    ROT = _euler_xyz(NRM, yaw[idx], float(align_to_normal)) if len(idx) else np.zeros((0, 3))

    def violations(Q, Zq, rq):
        out = {}
        for zn in EXC:
            sd = zn.sdist(Q, rmax + 1e-3)
            bad = sd < 0
            if respect_footprint and zn.exact_dist:
                bad |= sd < rq - 1e-6
            bad &= zn.z_ok(Zq)
            out[zn.label] = int(bad.sum())
        return out
    v_py = violations(XY, Z, rpt[idx]) if len(idx) else {}
    assert sum(v_py.values()) == 0, (name, "keep-out violations (python)", v_py)
    t = tick("assert_py", t)

    rep = {"name": name, "mode": mode, "seed": seed, "count": int(len(PTS)), "candidates": int(n_cand),
           "in_include": int(n_inc), "excluded_by": stats_exc, "thinned_early": n_thin_early, "thinned_late": n_thin_late,
           "no_surface": n_nohit, "covered_by_blockers": n_covered, "too_steep": n_slope, "spacing_rejected": n_spacing,
           "per_include_zone": {zn.label: int((zn.sdist(XY, 0.0) < 0).sum()) if len(XY) else 0 for zn in INC},
           "per_item": {(items[k].name if items else str(k)): int((ITEM == k).sum()) for k in range(nI)},
           "item_radius_m": {(items[k].name if items else str(k)): round(float(rad[k]), 3) for k in range(nI)},
           "bbox": ([[round(float(v), 2) for v in PTS.min(0)], [round(float(v), 2) for v in PTS.max(0)]] if len(PTS) else None),
           "keep_out_violations": 0, "zones": {"include": [z_.describe() | {"label": z_.label} for z_ in INC],
                                                "exclude": [z_.describe() | {"label": z_.label} for z_ in EXC]}}
    if not create:
        rep.update(points=PTS, items=ITEM, rotation=ROT, scale=SCL, timing_s=tm, build_s=round(time.time() - T0, 2))
        if verbose:
            print(f"[scatter] {name}: {rep['count']} points (no geometry) in {rep['build_s']} s", flush=True)
        return rep

    # ---------------------------------------------------------------- geometry + asserts on the EVALUATED instances
    _remove_old(name)
    holder = _create(name, PTS, ITEM, ROT, SCL, items, not as_instances, shadows, collection)
    if hide_items:
        for o in items:
            o.hide_render = True
    t = tick("create", t)
    rep["object"] = holder.name
    if as_instances:
        E, VC = _evaluated_instances(holder)
        assert len(E) == len(PTS), (name, "evaluated instance count", len(E), "!= planned", len(PTS))
        if len(E):
            P32 = PTS.astype(np.float32).astype(np.float64)              # the mesh stores float32
            seq_err = float(np.abs(E - P32).max())
            if seq_err < 1e-3:                                          # instances come out in point order
                ob = np.arange(len(PTS))
                a, pos_err = E, seq_err
            else:
                oa = np.lexsort((E[:, 2], E[:, 1], E[:, 0]))
                ob = np.lexsort((P32[:, 2], P32[:, 1], P32[:, 0]))
                a = E[oa]
                pos_err = float(np.abs(a - P32[ob]).max())
            assert pos_err < 1e-3, (name, "evaluated instances moved", pos_err)
            vio = violations(a[:, :2], a[:, 2] + sink, rpt[idx][ob])     # origin AND footprint, on the real instances
            assert sum(vio.values()) == 0, (name, "evaluated instance inside a keep-out", vio)
            # instance -> item: the n-th evaluated instance must carry the mesh of the n-th point's item
            if len(VC) and len({len(o.data.vertices) for o in items if o.type == "MESH"}) == len(items):
                seq = np.abs(E[:len(VC)] - PTS[:len(VC)]).max() < 1e-3
                want = np.array([len(items[k].data.vertices) for k in ITEM[:len(VC)]])
                rep["item_pick_checked"] = int(len(VC))
                assert seq and (np.array(VC) == want).all(), (name, "instance -> item mapping is off", VC[:6], want[:6].tolist())
            rep["evaluated_instances"] = int(len(E))
            rep["evaluated_vs_planned_max_err_m"] = round(pos_err, 6)
            # grounding: every instance origin within tol (+ sink) of the surface
            if surf_bvh is not None:
                gerr = 0.0
                worst = []
                for x, y, zz in E.tolist():
                    h = surf_bvh.find_nearest(Vector((x, y, zz)), 10.0)
                    d = h[3] if h[0] is not None else 10.0
                    if d > gerr:
                        gerr = d
                        worst = [round(x, 2), round(y, 2), round(zz, 2)]
                rep["max_ground_error_m"] = round(float(gerr), 4)
                rep["worst_ground_point"] = worst
                assert gerr <= sink + tol, (name, "instance off the surface", gerr, worst)
            elif z_fn is not None:
                gerr = float(np.abs(E[:, 2] + sink - np.asarray(z_fn(E[:, 0], E[:, 1]), float)).max())
                rep["max_ground_error_m"] = round(gerr, 4)
                assert gerr <= tol, (name, "instance off z_fn", gerr)
    else:
        rep["evaluated_instances"] = "realized"
    t = tick("assert_eval", t)
    rep["timing_s"] = tm
    rep["bvh_cache"] = dict(BVH_STATS)
    rep["build_s"] = round(time.time() - T0, 2)
    if verbose:
        print(f"[scatter] {name}: {rep['count']} instances, keep-out violations 0, "
              f"ground err {rep.get('max_ground_error_m', '-')} m, {rep['build_s']} s", flush=True)
    return rep


# ===================================================================================== debug drawing
def _mat(name, rgb):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*rgb, 1.0)
        em.inputs["Strength"].default_value = 4.0
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def draw_zones(include=(), exclude=(), name="ScatterZones", z=0.05, width=0.08, z_fn=None, collection=None):
    """Draw the zones as glowing wire curves (include = green, exclude = red) so a top-down still shows them.
    z: constant height (or z_fn(x, y) to drape on terrain). Re-running replaces the previous drawing."""
    col = collection
    if col is None:
        col = bpy.data.collections.get(name)
        if col is None:
            col = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(col)
    for o in list(col.objects):
        if o.get(_TAG + "_debug"):
            cu = o.data
            bpy.data.objects.remove(o)
            if cu.users == 0:
                bpy.data.curves.remove(cu)
    made = []
    for kind, zs, rgb in (("inc", _zones(include, "include"), (0.1, 1.0, 0.2)), ("exc", _zones(exclude, "exclude"), (1.0, 0.1, 0.05))):
        mt = _mat(f"M_ScatterZone_{kind}", rgb)
        for i, zn in enumerate(zs):
            cu = bpy.data.curves.new(f"{name}_{kind}{i}", "CURVE")
            cu.dimensions = "3D"
            cu.bevel_depth = width
            cu.bevel_resolution = 0
            for L in zn.outlines():
                L = np.asarray(L, float)
                zz = (np.asarray(z_fn(L[:, 0], L[:, 1]), float) + 0.05) if z_fn is not None else np.full(len(L), float(z))
                sp = cu.splines.new("POLY")
                sp.points.add(len(L) - 1)
                sp.points.foreach_set("co", np.stack([L[:, 0], L[:, 1], zz, np.ones(len(L))], 1).ravel())
            cu.materials.append(mt)
            o = bpy.data.objects.new(f"{name}_{kind}{i}_{zn.kind}", cu)
            o[_TAG + "_debug"] = True
            o.visible_shadow = False
            col.objects.link(o)
            made.append(o)
    return made
