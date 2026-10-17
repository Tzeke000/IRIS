"""Build-time verification kit: assert, don't eyeball (each one caught a real bug once). numpy core is pure (unit-tested
without Blender); the bpy_* helpers take a depsgraph. Rules — which objects, what margin — stay in the builder.
2-D keep-out / keep-clear zones are lib/scatter_zones.py's job (Zone.sdist) — not duplicated here.

    from lib import checks as K
    worst = K.direction_check(pos, nose, min_dot=0.9)          # police cars backwards ×2, rooftop nose axes
    d, f, i = K.clearance(cam_pos, tower_boxes)                 # rooftop [flycheck]/[carcheck], police camera-in-building
    n = K.inside_count(debris_pts, boxes)                       # rocket4 "no debris vertex inside" gate
    bad = K.floating(lowest_pts, ground_fn, gap=(0.5, 15))      # rooftop verify_world floating-edge test
    K.finite("cam_pos", cam_pos)                                # NaN guard (gotcha 2: NaN → glare white-out)
    bvh = K.bpy_bvh(objs, dg);  K.bpy_camera_clear(cam_pos, bvh, 0.3)       # camera inside geometry
    K.bpy_instances(ob, dg) > 0                                 # gotcha 1: instancing from a hidden source → EMPTY
    K.bpy_bad_images()                                          # arcade: relative/missing textures → magenta
Each function returns plain data; pair with lib/report.Report.assert_ to record + fail the build.
"""
import numpy as np


# ------------------------------------------------------------------------------------------------ pure numpy

def finite(name, a):
    """raise ValueError if `a` has NaN/inf (name + first bad index in the message)"""
    a = np.asarray(a, np.float64)
    bad = ~np.isfinite(a)
    if bad.any():
        idx = np.argwhere(bad)[0].tolist()
        raise ValueError(f"{name}: {int(bad.sum())} non-finite values, first at {idx}")
    return True


def velocity(pos, fps=1.0):
    """(F,3) positions → (F,3) velocity (central differences; one-sided at the ends)"""
    return np.gradient(np.asarray(pos, np.float64), axis=0) * fps


def direction_check(pos, nose, min_dot=0.9, min_speed=1e-3):
    """Things that move must face their motion. pos (F,3), nose (F,3) world forward vectors (or (3,) constant).
    Returns dict(worst_dot, worst_frame_index, n_checked, n_bad, ok). Frames slower than min_speed are skipped."""
    v = velocity(pos)
    sp = np.linalg.norm(v, axis=1)
    nose = np.broadcast_to(np.asarray(nose, np.float64), v.shape)
    nn = np.linalg.norm(nose, axis=1)
    m = (sp > min_speed) & (nn > 0)
    if not m.any():
        return {"worst_dot": None, "worst_frame_index": None, "n_checked": 0, "n_bad": 0, "ok": True}
    dot = np.full(len(v), np.nan)
    dot[m] = np.einsum("ij,ij->i", v[m], nose[m]) / (sp[m] * nn[m])
    i = int(np.nanargmin(dot))
    nbad = int((dot[m] < min_dot).sum())
    return {"worst_dot": float(dot[i]), "worst_frame_index": i, "n_checked": int(m.sum()), "n_bad": nbad,
            "ok": nbad == 0}


def box_distance(pts, boxes):
    """(P,3) points × (N,6) AABBs [x0,y0,z0,x1,y1,z1] → (P,N) distance (0 inside)."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 3)
    b = np.asarray(boxes, np.float64).reshape(1, -1, 6)
    lo, hi = np.minimum(b[..., :3], b[..., 3:]), np.maximum(b[..., :3], b[..., 3:])
    d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    return np.linalg.norm(d, axis=2)


def clearance(pts, boxes, chunk=4096):
    """min distance from any of pts (P,3) to any box → (dmin, point_index, box_index)."""
    best = (np.inf, -1, -1)
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    for s in range(0, len(pts), chunk):
        d = box_distance(pts[s:s + chunk], boxes)
        if d.size == 0:
            continue
        k = int(np.argmin(d))
        pi, bi = divmod(k, d.shape[1])
        if d[pi, bi] < best[0]:
            best = (float(d[pi, bi]), s + pi, bi)
    return best


def inside_count(pts, boxes, pad=0.0):
    """number of points inside any box (grown by pad)"""
    return int((box_distance(pts, boxes) <= pad).any(axis=1).sum()) if len(boxes) else 0


def floating(pts, ground_fn, gap=(0.5, 15.0)):
    """Indices of 'lowest points' that hover gap[0] < z − ground(x, y) < gap[1] above the ground (sunk or far-up
    things are not this test's business). ground_fn takes x, y arrays → z array."""
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    dz = p[:, 2] - np.asarray(ground_fn(p[:, 0], p[:, 1]), np.float64)
    return np.nonzero((dz > gap[0]) & (dz < gap[1]))[0]


def min_spacing(pos):
    """(F,N,3) swarm positions → (min pairwise distance, frame, i, j) — drone-show safety."""
    pos = np.asarray(pos, np.float64)
    best = (np.inf, -1, -1, -1)
    for f in range(pos.shape[0]):
        d = np.linalg.norm(pos[f, :, None] - pos[f, None], axis=2)
        np.fill_diagonal(d, np.inf)
        k = int(np.argmin(d))
        i, j = divmod(k, d.shape[1])
        if d[i, j] < best[0]:
            best = (float(d[i, j]), f, i, j)
    return best


def check_order(**named):
    """assert named values strictly increase in the order given (frames/seconds) — rocket4's event chain"""
    items = list(named.items())
    for (a, va), (b, vb) in zip(items, items[1:]):
        if not va < vb:
            raise AssertionError(f"order: {a}={va} must be < {b}={vb}")
    return True


# ------------------------------------------------------------------------------------------------ bpy

def bpy_world_tris(objs, dg):
    """world-space (verts, tris) of the EVALUATED meshes of objs"""
    V, T, off = [], [], 0
    for o in objs:
        if o.type not in ("MESH", "CURVE", "FONT", "SURFACE", "META"):
            continue
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        try:
            me.calc_loop_triangles()
            co = np.empty(len(me.vertices) * 3, np.float64)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            M = np.array(oe.matrix_world)
            co = co @ M[:3, :3].T + M[:3, 3]
            tr = np.empty(len(me.loop_triangles) * 3, np.int64)
            me.loop_triangles.foreach_get("vertices", tr)
            V.append(co)
            T.append(tr.reshape(-1, 3) + off)
            off += len(co)
        finally:
            oe.to_mesh_clear()
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(V), np.concatenate(T)


def bpy_bvh(objs, dg):
    from mathutils.bvhtree import BVHTree
    V, T = bpy_world_tris(objs, dg)
    return BVHTree.FromPolygons(V.tolist(), T.tolist())


def bpy_inside(p, bvh, dirs=((0, 0, 1), (1, 0, 0)), eps=1e-4, max_hits=256):
    """True if point p is inside closed geometry: an ODD number of surface crossings along EVERY ray in dirs
    (two axes, so a single open/degenerate face can't fake it)."""
    from mathutils import Vector
    for d in dirs:
        d = Vector(d).normalized()
        o, n = Vector(p), 0
        while n < max_hits:
            loc, _nrm, _i, _dist = bvh.ray_cast(o, d)
            if loc is None:
                break
            n += 1
            o = loc + d * eps
        if n % 2 == 0:
            return False
    return True


def bpy_camera_clear(cam_pos, bvh, min_d=0.3):
    """frames (indices) where the camera is closer than min_d to the BVH surface OR inside closed geometry (however
    deep) → [(index, distance; 0.0 = inside)]"""
    from mathutils import Vector
    out = []
    for i, p in enumerate(np.asarray(cam_pos, np.float64).reshape(-1, 3)):
        hit = bvh.find_nearest(Vector(p), min_d)
        if hit[0] is not None:
            out.append((i, float(hit[3])))
        elif bpy_inside(p, bvh):
            out.append((i, 0.0))
    return out


def bpy_instances(ob, dg):
    """number of instances the evaluated `ob` emits (GN Instance on Points, collection instancing...)"""
    n = 0
    for inst in dg.object_instances:
        if inst.is_instance and inst.parent is not None and inst.parent.original == ob.original:
            n += 1
    return n


def bpy_bad_images():
    """[(image, reason)] for every image in the file that is relative or missing (and not packed)"""
    import bpy
    from . import importer
    return importer.bad_images(list(bpy.data.images))
