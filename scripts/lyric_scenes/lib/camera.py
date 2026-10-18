"""Camera: shot table → per-frame camera bake that is CUT-SAFE, still cameras for look-dev, DOF, look_at, on-screen QA.
One copy of rocket4's @shot registry + loop, rooftop's shot() table + post-hoc CONSTANT patch, police's shot(kind, ...)
loop + "lead car on screen" QA, and boss v1/v2/v3's cam() + look_at().

    import sys; sys.path.insert(0, "/ABS/scripts/lyric_scenes")
    from lib import camera as CM
    SHOTS = [CM.Shot(1, "wide", lambda i, u, n: dict(cam=(0, -40, 3), tgt=(0, 0, 8), lens=35, fstop=5.6)),
             CM.Shot(S.snap_fr(29.0), "drop", drop_fn)]           # start = Blender frame (lib/song: S.snap_fr)
    A = CM.bake_shots(CAM, SHOTS, n=S.n, post=shake)               # → arrays for QA + A["cuts"] for other bakes
    bake(CTRL, 'nodes["laser"].outputs[0].default_value', laser, cuts=A["cuts"])   # same hold on shader controls
    frac = CM.on_screen_by_shot(A, lead_pos)                       # police QA, generalised
    CM.still_cam("CAM_hall", (0, -20, 1.7), (0, 10, 4), 24, fstop=4, exposure_offset=0.3)   # lib/render/frames.py --cams

SHOT FUNCTIONS stay in the builder (they are the creative core): fn(i, u, n) with i = 0-based frame index (song arrays:
index i ↔ Blender frame i+1), u = frames since the shot started, n = shot length − 1 (u/n runs 0 → 1). Return a dict
    cam (alias loc), tgt (alias target), lens; optional fstop, focus (default |tgt − cam|), up (default +Z), roll (rad)
    + any other numeric keys (haze, shake…) → per-frame arrays in A["extra"] (0 where a shot doesn't set them)
or rooftop's tuple (cam, tgt, lens[, fstop[, focus]]). post(i, d, shot) → d is the per-scene shake/offset hook.

CUT SAFETY (the rooftop v3.1 glitches 5–7, fixed here once):
  * every channel — location, rotation, lens, f-stop, focus — is baked with bpyx.bake(cuts=…): the key on the LAST
    frame of each shot is CONSTANT, so with render.motion_blur_position = "START" (lib/look.eevee_defaults sets it) the
    shutter of that frame never sees the next shot. CENTER/END would still smear — verify_saved checks the position.
  * cut frames are the shots' own start frames (no t→frame rounding, rooftop fix 6); Euler continuity restarts at
    every cut (rocket/police did, rooftop didn't).
  * the cut list is written to scene["cuts"] (+ scene["shots"] "start:name;…", + one timeline marker per shot) so
    lib/qa/verify_saved.py can prove from the SAVED file that nothing moves across a cut; pass A["cuts"] to every other
    bake that steps at a cut (shader controls, visibility), or hold_at_cuts() them afterwards.
One camera per shot with bound markers was considered and NOT used: lib/render/frames.py, qa/time_ab and the
per-shot culling all read scene.camera, and a held key on one camera gives the same render.
"""
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .bpyx import bake, bake_vec, euler_seq, euler_to_mat, look_at, look_quat, track_matrix  # noqa: F401

SENSOR = 36.0


@dataclass
class Shot:
    start: int                                   # Blender frame where the shot starts (the first shot starts at 1)
    name: str
    fn: Callable = field(repr=False)             # (i, u, n) -> dict | tuple


# ------------------------------------------------------------------------------------------------ pure (numpy)

def plan(shots, n):
    """Validated, sorted shots → (shots, per-frame shot index int[n], cuts = Blender frames where shots 2.. start)."""
    shots = sorted(shots, key=lambda s: s.start)
    if not shots:
        raise ValueError("no shots")
    if shots[0].start != 1:
        raise ValueError(f"the first shot must start at frame 1 (got {shots[0].start} for {shots[0].name!r})")
    st = [int(s.start) for s in shots]
    if len(set(st)) != len(st):
        raise ValueError(f"two shots start on the same frame: {st}")
    if st[-1] > n:
        raise ValueError(f"shot {shots[-1].name!r} starts at {st[-1]} > n={n}")
    names = [s.name for s in shots]
    if len(set(names)) != len(names):
        raise ValueError(f"shot names must be unique (markers, verify): {names}")
    idx = np.searchsorted(np.array(st), np.arange(1, n + 1), side="right") - 1
    return shots, idx, st[1:]


def _norm(d):
    """shot fn output → dict with cam, tgt, lens (+ optional keys)"""
    if isinstance(d, (tuple, list)):
        keys = ("cam", "tgt", "lens", "fstop", "focus")
        if not 3 <= len(d) <= 5:
            raise ValueError(f"a shot tuple is (cam, tgt, lens[, fstop[, focus]]), got {len(d)} items")
        d = dict(zip(keys, d))
    d = dict(d)
    for a, b in (("loc", "cam"), ("target", "tgt")):
        if a in d:
            d[b] = d.pop(a)
    for k in ("cam", "tgt", "lens"):
        if k not in d:
            raise KeyError(f"shot result needs {k!r} (got {sorted(d)})")
    return d


def eval_shots(shots, n, post=None):
    """Evaluate the shot table per frame (no bpy). Returns dict:
    loc (n,3), rot (n,3) XYZ Euler (continuous within a shot), R (n,3,3), lens, fstop (or None if no shot sets one),
    focus (n,), shot int[n], names [shot names], starts [Blender frames], cuts [Blender frames], extra {key: (n,) or (n,k)}."""
    shots, sidx, cuts = plan(shots, n)
    starts = [s.start for s in shots]
    loc, tgt, up = np.zeros((n, 3)), np.zeros((n, 3)), np.tile([0.0, 0.0, 1.0], (n, 1))
    lens, focus, roll = np.zeros(n), np.full(n, np.nan), np.zeros(n)
    fstop = np.full(n, np.nan)
    extra = {}
    bounds = [s - 1 for s in starts] + [n]
    for k, sh in enumerate(shots):
        s0, s1 = bounds[k], bounds[k + 1]
        for i in range(s0, s1):
            d = _norm(sh.fn(i, i - s0, max(s1 - s0 - 1, 1)))
            if post is not None:
                d = _norm(post(i, d, sh))
            loc[i], tgt[i], lens[i] = np.asarray(tuple(d.pop("cam")), float), np.asarray(tuple(d.pop("tgt")), float), \
                float(d.pop("lens"))
            if d.get("up") is not None:
                up[i] = np.asarray(tuple(d.pop("up")), float)
            roll[i] = float(d.pop("roll", 0.0) or 0.0)
            fs = d.pop("fstop", None)
            fstop[i] = np.nan if fs is None else float(fs)
            fo = d.pop("focus", None)
            if fo is not None:
                focus[i] = float(fo)
            d.pop("up", None)
            for key, v in d.items():
                v = np.asarray(v, np.float64)
                if key not in extra:
                    extra[key] = np.zeros((n,) + v.shape)
                extra[key][i] = v
    have = ~np.isnan(fstop)
    if have.any() and not have.all():
        bad = sorted({shots[j].name for j in sidx[~have]})
        raise ValueError(f"fstop must be given by every shot or by none (missing in {bad})")
    fwd = tgt - loc
    if (np.linalg.norm(fwd, axis=1) < 1e-9).any():
        i = int(np.argmax(np.linalg.norm(fwd, axis=1) < 1e-9))
        raise ValueError(f"camera and target coincide at frame {i + 1} ({shots[sidx[i]].name})")
    focus = np.where(np.isnan(focus), np.linalg.norm(fwd, axis=1), focus)
    R = track_matrix(fwd, up, roll)
    rot = euler_seq(R, resets=[c - 1 for c in cuts])
    out = dict(loc=loc, tgt=tgt, rot=rot, R=R, lens=lens, fstop=fstop if have.all() else None, focus=focus,
               shot=sidx, names=[s.name for s in shots], starts=starts, cuts=list(cuts), extra=extra)
    for key in ("loc", "rot", "lens", "focus") + (("fstop",) if have.all() else ()):
        a = out[key]
        if not np.isfinite(a).all():
            i = int(np.argwhere(~np.isfinite(np.atleast_2d(a.T).T))[0][0])
            raise ValueError(f"camera {key} is not finite at frame {i + 1} ({shots[sidx[i]].name}) — NaN glare white-out")
    return out


def half_tangents(lens, sensor=SENSOR, aspect=16 / 9):
    """sensor fit AUTO: the larger image side spans the sensor → (tan half-FOV x, tan half-FOV y)"""
    t = 0.5 * sensor / np.asarray(lens, np.float64)
    return (t, t / aspect) if aspect >= 1 else (t * aspect, t)


def to_camera(A, pts):
    """world points → camera space per frame: pts (n,3) or (3,) → (n,3) (x right, y up, −z forward)"""
    p = np.broadcast_to(np.asarray(pts, np.float64), A["loc"].shape)
    return np.einsum("nji,nj->ni", A["R"], p - A["loc"])


def on_screen(A, pts, sensor=SENSOR, aspect=16 / 9, margin=0.0):
    """bool[n]: is the point (per-frame (n,3) or fixed (3,)) inside the frame? margin = fraction trimmed off each edge."""
    pc = to_camera(A, pts)
    tx, ty = half_tangents(A["lens"], sensor, aspect)
    d = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (d > 0) & (np.abs(pc[:, 0] / d) <= tx * (1 - margin)) & (np.abs(pc[:, 1] / d) <= ty * (1 - margin))


def on_screen_by_shot(A, pts, **kw):
    """{shot name: fraction of its frames with the point on screen} — police's lead-car QA"""
    v = on_screen(A, pts, **kw)
    return {nm: float(v[A["shot"] == k].mean()) for k, nm in enumerate(A["names"])}


def dof_limits(lens, fstop, focus, coc=0.03):
    """thin-lens near/far limits of acceptable sharpness (mm lens, metres focus, coc in mm; far = inf past hyperfocal)"""
    f = lens / 1000.0
    H = f * f / (fstop * coc / 1000.0) + f
    near = focus * (H - f) / (H + focus - 2 * f)
    far = math.inf if focus >= H else focus * (H - f) / (H - focus)
    return near, far


# ------------------------------------------------------------------------------------------------ bpy

def _bpy():
    import bpy
    return bpy


def dof(cam, fstop, focus=None, target=None, blades=None):
    """Turn DOF on for a camera object (or its data): f-stop, focus distance (or |target − camera|), aperture blades."""
    cd = cam.data if hasattr(cam, "data") and hasattr(cam.data, "dof") else cam
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = fstop
    if focus is None and target is not None:
        p = cam.location if cam.parent is None else cam.matrix_world.translation   # matrix_world lags until an update
        focus = float(np.linalg.norm(np.asarray(tuple(target), float) - np.asarray(tuple(p), float)))
    if focus is not None:
        cd.dof.focus_distance = focus
    if blades:
        cd.dof.aperture_blades = blades
    return cd


def still_cam(name, loc, target, lens, coll=None, ortho=None, rotz=None, fstop=None, focus=None, sensor=SENSOR,
              clip=(0.05, 6000.0), roll=0.0, **props):
    """Look-dev still camera (boss v2/v3 cam(), v1 lacked DOF). Replaces an existing object of the same name.
    ortho = ortho_scale (top-down plan view, rotz = its Z rotation); fstop → DOF focused at target (or focus).
    props → custom props on the object, read by lib/render/frames.py --cams (exposure_offset, hide_collections, patch)."""
    bpy = _bpy()
    D = bpy.data
    old = D.objects.get(name)
    if old is not None:
        D.objects.remove(old)
    cd = D.cameras.new(name)
    cd.lens, cd.sensor_width = lens, sensor
    cd.clip_start, cd.clip_end = clip
    o = D.objects.new(name, cd)
    (coll or bpy.context.scene.collection).objects.link(o)
    o.location = loc
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho
        o.rotation_euler = (0.0, 0.0, rotz or 0.0)
    else:
        look_at(o, target, roll)
    if fstop:
        dof(o, fstop, focus=focus, target=None if (focus is not None or ortho) else target)
    for k, v in props.items():
        o[k] = v
    return o


def bake_shots(cam_ob, shots, n, post=None, cut_safe=True, markers=True, scene=None, sensor=SENSOR, clip=None,
               blades=None):
    """Evaluate + bake the shot table onto cam_ob (location, rotation_euler) and its data (lens, dof.aperture_fstop,
    dof.focus_distance). Returns eval_shots()'s arrays. cut_safe → CONSTANT hold on the last frame of every shot for
    EVERY channel; markers → one timeline marker per shot (named after it). scene["cuts"] / scene["shots"] always set."""
    bpy = _bpy()
    sc = scene or bpy.context.scene
    A = eval_shots(shots, n, post)
    cuts = A["cuts"] if cut_safe else None
    cam_ob.rotation_mode = "XYZ"
    bake_vec(cam_ob, "location", A["loc"], cuts=cuts)
    bake_vec(cam_ob, "rotation_euler", A["rot"], cuts=cuts)
    cd = cam_ob.data
    cd.sensor_width = sensor
    if clip:
        cd.clip_start, cd.clip_end = clip
    bake(cd, "lens", A["lens"], cuts=cuts)
    if A["fstop"] is not None:
        cd.dof.use_dof = True
        cd.dof.focus_object = None
        if blades:
            cd.dof.aperture_blades = blades
        bake(cd, "dof.aperture_fstop", A["fstop"], cuts=cuts)
        bake(cd, "dof.focus_distance", A["focus"], cuts=cuts)
    else:
        cd.dof.use_dof = False
    if markers:
        names = set(A["names"])
        for m in [m for m in sc.timeline_markers if m.name in names]:
            sc.timeline_markers.remove(m)
        for nm, st in zip(A["names"], A["starts"]):
            sc.timeline_markers.new(nm, frame=st)
    sc["cuts"] = [int(c) for c in A["cuts"]] if cut_safe else []
    sc["shots"] = ";".join(f"{s}:{nm}" for s, nm in zip(A["starts"], A["names"]))
    sc.camera = cam_ob
    return A


def _all_fcurves(idb):
    ad = getattr(idb, "animation_data", None)
    if not ad or not ad.action:
        return []
    return [fc for ly in ad.action.layers for st in ly.strips for cb in st.channelbags
            if ad.action_slot is None or cb.slot == ad.action_slot for fc in cb.fcurves]


def hold_at_cuts(idb, cuts, paths=None):
    """Post-hoc cut safety for curves baked WITHOUT cuts= (rooftop fix 7, without its layers[0].strips[0] path): on
    every F-curve of idb (or only `paths`), make sure keys exist on cut−1 and cut (inserted at the evaluated value) and
    set the cut−1 key CONSTANT. Returns the number of curves touched."""
    n = 0
    for fc in _all_fcurves(idb):
        if paths is not None and fc.data_path not in paths:
            continue
        for c in cuts:
            for f in (c - 1, c):
                if not any(abs(k.co[0] - f) < 1e-3 for k in fc.keyframe_points):
                    fc.keyframe_points.insert(f, fc.evaluate(f), options={"FAST"})
        fc.update()
        for c in cuts:
            for k in fc.keyframe_points:
                if abs(k.co[0] - (c - 1)) < 1e-3:
                    k.interpolation = "CONSTANT"
        n += 1
    return n


def cut_violations(idb, cuts, tol=1e-6):
    """[(data_path, index, cut, delta)] for every F-curve of idb whose value moves inside the START shutter of the last
    frame before a cut (evaluate(cut − 0.5) ≠ evaluate(cut − 1)) — i.e. the frame would smear toward the next shot."""
    out = []
    for fc in _all_fcurves(idb):
        for c in cuts:
            dlt = abs(fc.evaluate(c - 0.5) - fc.evaluate(c - 1))
            if dlt > tol * max(1.0, abs(fc.evaluate(c - 1))):
                out.append((fc.data_path, fc.array_index, int(c), float(dlt)))
    return out


def view_hit(cam_ob, scene=None, dg=None, skip=None, max_hops=8):
    """First surface along the camera's view axis → (object name, distance) or (None, None). skip(obj) → True passes
    through it (haze boxes, render-hidden proxies). Boss v1's "camera not buried in a wall" sanity."""
    from mathutils import Vector
    bpy = _bpy()
    sc = scene or bpy.context.scene
    dg = dg or bpy.context.evaluated_depsgraph_get()
    mw = cam_ob.matrix_world
    fwd = (mw.to_quaternion() @ Vector((0, 0, -1))).normalized()
    orig = mw.translation.copy()
    for _ in range(max_hops):
        hit, loc, _n, _i, hob, _m = sc.ray_cast(dg, orig, fwd)
        if not hit:
            return None, None
        if skip is not None and skip(hob):
            orig = loc + fwd * 0.01
            continue
        return hob.name, float((loc - mw.translation).length)
    return None, None


__all__ = ["Shot", "plan", "eval_shots", "half_tangents", "to_camera", "on_screen", "on_screen_by_shot", "dof_limits",
           "dof", "still_cam", "bake_shots", "hold_at_cuts", "cut_violations", "view_hit", "look_at", "look_quat",
           "track_matrix", "euler_to_mat"]
