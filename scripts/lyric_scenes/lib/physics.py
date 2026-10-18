"""Rigid-body bakes and grounding for scene builders (Blender 5.2). One copy of boss v2/v3 settle() (~80 lines each),
rocket4's debris sim (§10c: own scene, temp_override, numpy bake → keys) and the drop-to-surface / floating tests
(boss v3 surf(), rooftop verify_world). Recipe + measured numbers: profiles/blender/rigid-body/rigid-body.md.

    from lib import physics as PH
    res = PH.settle(rubble, statics=[floor, walls], frames=180)          # rest poses applied; [(ob, Matrix, moved_m)]
    with PH.Sim((BOOM - 1, N), substeps=60, iters=20, gravity=False) as S:   # rocket4-style launch + collisions
        for pc in pieces: S.add(pc, "ACTIVE", density=2600); S.release(pc, REL)
        S.add(proxy, "PASSIVE", shape="BOX", kinematic=True, temp=True)
        S.bake()
        MW = S.sample(pieces)                                               # (k, frames, 4, 4) — run build gates on this
        S.to_keys(pieces, n=N)                                              # plain keys; rigid bodies removed on close
    g = PH.ground_from([terrain], dg);  PH.drop_to_ground(props, g)        # rest on the surface below
    PH.floating(props, g, tol=0.05)  → [(name, gap_m)]                       # nothing hovers

WHY KEYS, NOT CACHES: a .blend carrying a Bullet point cache re-simulates (or reads a cache path) in every render
worker; lib/render/scene.sh renders chunks in parallel processes, so the film must contain only plain keyframes.
Sim.to_keys samples matrix_world once per frame and writes bpyx.bake keys (7.7× faster than the stock
bake_to_keyframes operator, which also fails headless without a real selection + 3D-view context) and close() removes
every rigid body, the sim scene and its collection — the film scene never had a rigidbody_world.
THE SIM RUNS IN ITS OWN SCENE (rocket4): the 6,000-frame film never re-evaluates during the bake, and nothing of the
sim (world, cache, proxies) can leak into the saved file.
Tuning (the card): substeps 30+ for fast bodies (v/(fps·substeps) ≪ FRAGMENT size), split impulse on, kinematic
colliders that move need ~1 m pad, contact frames can't be aimed — bake, READ them, re-aim.
"""
import math
import time

import numpy as np

from .bpyx import bake_vec, euler_seq


def _bpy():
    import bpy
    return bpy


# ------------------------------------------------------------------------------------------------ pure numpy

def decompose(MW):
    """(n,4,4) world matrices → (loc (n,3), R (n,3,3) pure rotations, scale (n,3)). No shear assumed."""
    M = np.asarray(MW, np.float64).reshape(-1, 4, 4)
    loc = M[:, :3, 3].copy()
    A = M[:, :3, :3]
    s = np.linalg.norm(A, axis=1)                       # column norms
    R = A / np.maximum(s[:, None, :], 1e-12)
    neg = np.linalg.det(R) < 0                          # mirrored: put the flip in x scale
    s[neg, 0] *= -1
    R[neg, :, 0] *= -1
    return loc, R, s


def hold_extend(arr, start, n):
    """samples for Blender frames start..start+len-1 → length-n array (frames 1..n), holding the first/last value."""
    arr = np.asarray(arr, np.float64)
    out = np.empty((n,) + arr.shape[1:])
    a, b = start - 1, start - 1 + len(arr)
    if a < 0 or b > n:
        raise ValueError(f"samples for frames {start}..{start + len(arr) - 1} don't fit 1..{n}")
    out[:a] = arr[0]
    out[a:b] = arr
    out[b:] = arr[-1]
    return out


def lowest_columns(pts, cell=0.1):
    """Lowest point of every vertical column (x, y rounded to `cell`) — the bottom edge of buildings/prisms
    (rooftop verify_world grouping). Feed to lib/checks.floating(cols, ground_fn, gap)."""
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    if len(p) == 0:
        return p
    key = np.round(p[:, :2] / cell).astype(np.int64)
    order = np.lexsort((p[:, 2], key[:, 1], key[:, 0]))
    ks = key[order]
    first = np.ones(len(order), bool)
    first[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    return p[order][first]


def gap_of(pts, ground_fn):
    """min over points of (z − ground(x, y)) and the index of that point; points with no ground below are ignored.
    > 0 hovers that high, < 0 sinks that deep, (nan, -1) if nothing is above ground."""
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    g = np.asarray(ground_fn(p[:, 0], p[:, 1]), np.float64)
    d = p[:, 2] - g
    ok = np.isfinite(d)
    if not ok.any():
        return math.nan, -1
    i = int(np.nanargmin(np.where(ok, d, np.nan)))
    return float(d[i]), i


# ------------------------------------------------------------------------------------------------ the sim scene

def _half_box_volume(ob):
    """half the scaled bbox volume of the mesh DATA (no depsgraph: ob.dimensions is zero on a fresh object)"""
    if ob.type != "MESH" or not len(ob.data.vertices):
        return 0.005
    co = np.empty(len(ob.data.vertices) * 3, np.float32)
    ob.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    d = np.ptp(co, axis=0) * np.abs(np.array(ob.matrix_world.to_scale()))
    return max(float(np.prod(d)) * 0.5, 0.005)


class Sim:
    """Throw-away rigid-body scene. frames = (first, last) Blender frames of the sim (the point cache range).
    Use as a context manager (close() always runs — the film scene is restored even when a build gate raises)."""

    def __init__(self, frames, name="SIM", fps=30, substeps=20, iters=15, split_impulse=True, gravity=True):
        bpy = _bpy()
        D = bpy.data
        self.f0, self.f1 = int(frames[0]), int(frames[1])
        self.sc = D.scenes.new(name)
        self.sc.render.fps, self.sc.render.fps_base = fps, 1.0
        self.sc.frame_start, self.sc.frame_end = self.f0, self.f1
        wins = bpy.context.window_manager.windows
        self.win = wins[0] if len(wins) else None
        self.main = self.win.scene if self.win else bpy.context.scene
        if self.win:
            self.win.scene = self.sc
        with self._ctx():
            bpy.ops.rigidbody.world_add()
        rbw = self.sc.rigidbody_world
        rbw.substeps_per_frame, rbw.solver_iterations, rbw.use_split_impulse = substeps, iters, split_impulse
        if not gravity:
            rbw.effector_weights.gravity = 0.0
        rbw.point_cache.frame_start, rbw.point_cache.frame_end = self.f0, self.f1
        self.rbc = D.collections.new(name + "_RB")
        self.sc.collection.children.link(self.rbc)
        rbw.collection = self.rbc
        self.bodies, self.temp, self.closed, self.baked = [], [], False, False

    def _ctx(self, ob=None):
        kw = dict(scene=self.sc)
        if self.win:
            kw["window"] = self.win
        if ob is not None:
            kw.update(object=ob, active_object=ob, selected_objects=[ob])
        return _bpy().context.temp_override(**kw)

    def add(self, ob, type="ACTIVE", shape="CONVEX_HULL", margin=0.004, density=None, temp=False, **rb):
        """Make ob a rigid body of this sim. density (kg/m³) → mass from half the bbox volume (boss). temp=True: the
        object is sim-only (proxy, collider copy) and is DELETED on close(). Extra kwargs → ob.rigid_body.<attr>."""
        if ob.name not in self.rbc.objects:
            self.rbc.objects.link(ob)
        if ob.rigid_body is None:
            with self._ctx(ob):
                _bpy().ops.rigidbody.object_add(type=type)
        r = ob.rigid_body
        r.type, r.collision_shape = type, shape
        r.use_margin, r.collision_margin = True, margin
        if density is not None:
            r.mass = density * _half_box_volume(ob)
        for k, v in rb.items():
            setattr(r, k, v)
        (self.temp if temp else self.bodies).append(ob)
        return ob

    def passive_mesh(self, name, V, T, friction=0.9, restitution=0.0, **rb):
        """One static MESH collider from world-space verts (k,3) + tris (m,3) (boss v3: ONE merged collider)."""
        D = _bpy().data
        me = D.meshes.new(name)
        me.from_pydata(np.asarray(V, float).tolist(), [], np.asarray(T, int).tolist())
        ob = D.objects.new(name, me)
        return self.add(ob, "PASSIVE", "MESH", temp=True, friction=friction, restitution=restitution, **rb)

    def passive_box(self, name, lo, hi, friction=0.9, restitution=0.0, **rb):
        """Static BOX collider spanning lo..hi (the BOX shape is built around the origin — boss v3's floor note)."""
        D = _bpy().data
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        h = (hi - lo) / 2
        v = [(x, y, z) for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])]
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        me = D.meshes.new(name)
        me.from_pydata(v, [], f)
        ob = D.objects.new(name, me)
        ob.location = tuple((lo + hi) / 2)
        return self.add(ob, "PASSIVE", "BOX", temp=True, friction=friction, restitution=restitution, **rb)

    def release(self, ob, frame):
        """Animated (kinematic) until `frame`, simulated from frame+1 (rocket4's launch: key the motion, then let go)."""
        r = ob.rigid_body
        r.kinematic = True
        r.keyframe_insert("kinematic", frame=frame)
        r.kinematic = False
        r.keyframe_insert("kinematic", frame=frame + 1)

    def bake(self):
        """Bullet bake of the whole sim (ptcache.bake_all under temp_override). Returns seconds."""
        t0 = time.time()
        with self._ctx():
            res = _bpy().ops.ptcache.bake_all(bake=True)
        if "FINISHED" not in res:
            raise RuntimeError(f"rigid-body bake failed: {res}")
        self.baked = True
        return time.time() - t0

    def sample(self, objs, frames=None):
        """matrix_world of objs at every sim frame (or `frames`) → (k, nF, 4, 4) numpy."""
        frames = list(range(self.f0, self.f1 + 1)) if frames is None else list(frames)
        out = np.zeros((len(objs), len(frames), 4, 4))
        for fi, f in enumerate(frames):
            self.sc.frame_set(f)
            for j, o in enumerate(objs):
                out[j, fi] = np.array(o.matrix_world)
        return out

    def to_keys(self, objs, n=None, MW=None, scale=False):
        """Bake the sim onto objs as plain keys: location + continuous rotation_euler (+ scale), frames f0..f1; with n
        the keys cover frames 1..n holding the first/last pose (rocket4). Clears the objects' old animation (the
        rigid_body.kinematic keys). Returns MW (k, nF, 4, 4)."""
        MW = self.sample(objs) if MW is None else MW
        for j, o in enumerate(objs):
            loc, R, s = decompose(MW[j])
            eul = euler_seq(R)
            o.animation_data_clear()
            o.rotation_mode = "XYZ"
            if n is not None:
                loc, eul, s = (hold_extend(a, self.f0, n) for a in (loc, eul, s))
                kw = {}
            else:
                kw = dict(frames=np.arange(self.f0, self.f1 + 1))
            bake_vec(o, "location", loc, **kw)
            bake_vec(o, "rotation_euler", eul, **kw)
            if scale or np.ptp(s, axis=0).max() > 1e-6:
                bake_vec(o, "scale", s, **kw)
            else:
                o.scale = tuple(s[0])
        return MW

    def close(self, relink=None):
        """Free the cache, strip rigid bodies from kept bodies (they stay where the last frame_set left them unless
        keyed), delete temp objects, restore the film scene, delete the sim scene + collection. Kept bodies that would
        end up in no collection are linked into `relink` (default: the film scene's collection). Idempotent."""
        if self.closed:
            return
        bpy = _bpy()
        D = bpy.data
        try:
            if self.baked:
                with self._ctx():
                    bpy.ops.ptcache.free_bake_all()
            for o in self.bodies:
                if o.name in D.objects and o.rigid_body is not None:
                    with self._ctx(o):
                        bpy.ops.rigidbody.object_remove()
            for o in self.temp:
                if o.name in D.objects:
                    me = o.data if o.type == "MESH" else None
                    D.objects.remove(o)
                    if me is not None and me.users == 0:
                        D.meshes.remove(me)
        finally:
            if self.win:
                self.win.scene = self.main
            target = relink or self.main.collection
            for o in self.bodies:
                if o.name in D.objects:
                    if o.name in self.rbc.objects:
                        self.rbc.objects.unlink(o)
                    if not o.users_collection:
                        target.objects.link(o)
            D.collections.remove(self.rbc)
            D.scenes.remove(self.sc)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ------------------------------------------------------------------------------------------------ settle

def settle(dynamic, statics=(), frames=180, substeps=20, iters=15, density=2400.0, friction=0.85,
           static_friction=0.9, damping=(0.4, 0.85), floor_z=None, apply=True, dg=None, log=print):
    """Drop loose pieces onto the static geometry and keep their REST pose (boss v2/v3 settle). The originals are not
    simulated: sim copies share their mesh data, so modifiers and keys on the originals are untouched.
    statics: objects (merged into ONE static MESH collider from their evaluated world triangles, boss v3) and/or
    (V, T) world-space arrays; floor_z → an extra wide static box whose top is at floor_z.
    Returns [(ob, final Matrix, moved_m over the last 12 frames — > ~0.01 means not at rest)]; apply → poses set."""
    from . import checks
    bpy = _bpy()
    D = bpy.data
    dg = dg or bpy.context.evaluated_depsgraph_get()
    Vs, Ts, off = [], [], 0
    for s in statics:
        V, T = s if isinstance(s, tuple) else checks.bpy_world_tris([s], dg)
        Vs.append(np.asarray(V, float))
        Ts.append(np.asarray(T, int) + off)
        off += len(V)
    with Sim((1, frames), "SIM_settle", substeps=substeps, iters=iters) as S:
        if Vs:
            S.passive_mesh("SIMP_static", np.concatenate(Vs), np.concatenate(Ts), friction=static_friction)
        sims = []
        for o in dynamic:
            so = D.objects.new("SIMD_" + o.name, o.data)
            so.matrix_world = o.matrix_world.copy()
            S.add(so, "ACTIVE", "CONVEX_HULL", density=density, temp=True, friction=friction, restitution=0.0,
                  linear_damping=damping[0], angular_damping=damping[1])
            sims.append(so)
        if floor_z is not None:
            P = np.array([list(o.matrix_world.translation) for o in dynamic] or [[0, 0, 0]])
            lo, hi = P.min(0) - 50.0, P.max(0) + 50.0
            S.passive_box("SIMP_floor", (lo[0], lo[1], floor_z - 1.0), (hi[0], hi[1], floor_z), friction=static_friction)
        t = S.bake()
        S.sc.frame_set(max(1, frames - 12))
        before = [so.matrix_world.translation.copy() for so in sims]
        S.sc.frame_set(frames)
        out = [(o, so.matrix_world.copy(), float((so.matrix_world.translation - b).length))
               for o, so, b in zip(dynamic, sims, before)]
    if log:
        log(f"[physics] settle: {len(out)} bodies × {frames} frames, static tris {sum(len(t_) for t_ in Ts)}, "
            f"bake {t:.1f} s, still moving > 1 cm: {sum(m > 0.01 for _, _, m in out)}")
    if apply:
        for o, M, _ in out:
            o.matrix_world = M
    return out


# ------------------------------------------------------------------------------------------------ grounding

def surface_fn(bvh, top=1.0e4, depth=2.0e4):
    """ground(x, y) → z of the FIRST surface hit by a ray straight down from z=top (nan where nothing is below).
    For terrain-like ground; for a room, pass a BVH of just the floor (a ray from above hits the roof first)."""
    from mathutils import Vector
    down = Vector((0.0, 0.0, -1.0))

    def ground(x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        out = np.full(x.shape, np.nan)
        for k, (a, b) in enumerate(zip(x.ravel(), y.ravel())):
            hit = bvh.ray_cast(Vector((a, b, top)), down, depth)[0]
            if hit is not None:
                out.flat[k] = hit.z
        return out
    return ground


def ground_from(objs, dg=None, **kw):
    """surface_fn over the evaluated world triangles of objs (terrain, floor, plinths)."""
    from . import checks
    dg = dg or _bpy().context.evaluated_depsgraph_get()
    return surface_fn(checks.bpy_bvh(objs, dg), **kw)


def bottom_points(ob, dg=None, band=1.0, max_pts=4000):
    """World-space vertices of the evaluated object within `band` m of its lowest point (≤ max_pts, evenly strided)."""
    from . import checks
    dg = dg or _bpy().context.evaluated_depsgraph_get()
    V, _ = checks.bpy_world_tris([ob], dg)
    if len(V) == 0:
        return V
    V = V[V[:, 2] <= V[:, 2].min() + band]
    return V[::max(1, len(V) // max_pts)]


def drop_to_ground(objs, ground_fn, dg=None, clearance=0.0, band=1.0, only_down=False):
    """Translate each object vertically so its lowest vertex (vs the ground right under it) touches the ground
    (+ clearance). only_down=True never lifts a sunk object. Returns [(ob, shift_z)] (nan shift = no ground below)."""
    bpy = _bpy()
    out = []
    for o in objs:
        dg_ = dg or bpy.context.evaluated_depsgraph_get()
        g, _ = gap_of(bottom_points(o, dg_, band), ground_fn)
        if not math.isfinite(g):
            out.append((o, math.nan))
            continue
        dz = clearance - g
        if only_down and dz > 0:
            dz = 0.0
        M = o.matrix_world.copy()
        tr = M.translation.copy()
        tr.z += dz
        M.translation = tr
        o.matrix_world = M
        bpy.context.view_layer.update()
        out.append((o, dz))
    return out


def floating(objs, ground_fn, tol=0.05, dg=None, band=1.0, sink=None):
    """[(name, gap_m)] of objects whose lowest vertex hovers more than tol above the ground under it (and, with
    sink=d, also those sunk deeper than d). Objects with no ground below are reported with gap nan."""
    dg = dg or _bpy().context.evaluated_depsgraph_get()
    bad = []
    for o in objs:
        g, _ = gap_of(bottom_points(o, dg, band), ground_fn)
        if not math.isfinite(g) or g > tol or (sink is not None and g < -sink):
            bad.append((o.name, g))
    return bad
