"""Locomotion — stride-matched clips (v2; Zeke's v1 note: 'floating backward', 'sliding', 'no walk/sidestep').

v1 played in-place clips at a hand-picked constant scale while the root slid along its own spline, so the feet skated
whenever the root speed ≠ the clip's stride speed (and skated backwards when the root moved backwards under a forward
walk). v2: each locomotion segment gets its OWN action, the clip TIME-WARPED so its stride speed equals the root's
speed frame by frame (phase advances by v_root·dir_clip / v_clip), with the clip chosen from the root's travel
direction in the character frame (forward walk/jog/sprint, strafe left/right, back-step). Residual slide (turns,
crossfades, contact nudges) is removed afterwards by the foot-lock IK stage (s55_feet).

    g = gait(skel, P, feet)                 clip analysis: per-frame foot heights/down flags, stride speed + direction
    ph = warp_phases(vchar, g)             phases (float clip frames) for the segment's frames
    act = warp_action(name, owner, src, ph, n, lower=..., upper=(src2, ph2))   one keyed action (0..len-1)
"""
import math

import numpy as np

from anim import common as C


def bone_point(skel, W, name, tail=False):
    """(n,3) world/char-frame position of a bone head (or tail) from (n,B,4,4) world matrices"""
    j = skel.idx[name]
    if not tail:
        return W[:, j, :3, 3]
    ln = skel.arm.data.bones[name].length
    return (W[:, j] @ np.array([0.0, ln, 0.0, 1.0]))[:, :3]


def gait(skel, P, feet, thr_frac=0.22):
    """feet: [(ankle_bone, toe_bone, toe_at_tail)] → dict(n, h, down, v (m per clip frame), dir (2,) char travel dir)."""
    W = skel.world(P)
    n = W.shape[0]
    H, XY = [], []
    for ank, toe, tail in feet:
        a = bone_point(skel, W, ank)
        t = bone_point(skel, W, toe, tail)
        h = np.minimum(a[:, 2], t[:, 2])
        H.append(h)
        XY.append(t[:, :2])
    H = np.array(H).T                               # (n, F)
    XY = np.array(XY).transpose(1, 0, 2)            # (n, F, 2)
    lo, hi = H.min(0), H.max(0)
    down = H < lo + thr_frac * np.maximum(hi - lo, 1e-3)
    vel = []
    for f in range(H.shape[1]):
        for i in range(n - 1):
            if down[i, f] and down[i + 1, f]:
                vel.append(XY[i + 1, f] - XY[i, f])
    vel = np.array(vel) if vel else np.zeros((1, 2))
    travel = -np.median(vel, axis=0)                # planted feet move opposite to travel
    v = float(np.linalg.norm(travel))
    d = travel / v if v > 1e-6 else np.array([0.0, 1.0])
    return dict(n=n, h=H, down=down, v=v, dir=d, lo=lo)


def root_vel_char(pos, yaw_deg):
    """world root positions (n,3) + yaw (n,) → velocity per frame in the character frame (n,2)"""
    v = np.gradient(np.asarray(pos, np.float64)[:, :2], axis=0)
    a = -np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    return np.c_[c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]]


def warp_phases(vchar, g, phase0=0.0, min_rate=0.45, max_rate=2.4, settle_need=None):
    """phase per frame: rate = (v·dir)/v_clip; while the root is (nearly) stopped the phase keeps a minimum rate until
    the clip reaches a support phase (enough feet down), so the character never freezes on one leg."""
    n = g["n"]
    cyc = n - 1
    need = settle_need if settle_need is not None else g["down"].shape[1]
    need = min(need, int(g["down"].sum(1).max()))            # the best support this clip ever has
    out = np.empty(len(vchar))
    ph = phase0
    for i, v in enumerate(vchar):
        out[i] = ph
        rate = float(np.dot(v, g["dir"])) / max(g["v"], 1e-6)
        rate = max(rate, 0.0)
        k = int(round(ph)) % cyc
        if rate < min_rate:
            rate = min_rate if g["down"][k].sum() < need else rate    # finish the step, then stand
        ph += min(rate, max_rate)
    return out


def choose(vchar, sword, speeds):
    """pick the locomotion clip from the mean char-frame travel (vchar m/frame). speeds: {clip: m/s at rate 1}"""
    v = vchar.mean(0) * C.FPS
    sp = float(np.percentile(np.linalg.norm(vchar, axis=1), 80) * C.FPS)      # the clip must cope with the PEAK pace
    ang = math.degrees(math.atan2(-v[0], v[1]))          # 0 = forward, +90 = left, -90 = right, ±180 = back
    if abs(ang) <= 45:
        if sp < 2.7:
            return "Walk_Loop"
        return "Jog_Fwd_Loop" if sp < 3.6 else "Sprint_Loop"
    # side / back travel: a fencer's SHUFFLE — the stance clip stays on and the foot-placement solver (s55) steps the
    # feet one at a time in the travel direction (the warrior donor's strafes are a wide greatsword crouch that
    # read as a lunge on the knight); only a fast retreat uses the donor's back-run
    if abs(ang) >= 135:
        return "SHUFFLE" if sp < 2.2 else "W_RunBack"
    return "SHUFFLE" if sp < 2.4 else ("W_StrafeL" if ang > 0 else "W_StrafeR")


def eval_action(act, frames, bones=None, skip=()):
    """{(data_path, index): values(len(frames))} for the action's F-curves (optionally only these bones)"""
    from bpy_extras import anim_utils
    cb = anim_utils.action_get_channelbag_for_slot(act, act.slots[0])
    out = {}
    for fc in cb.fcurves:
        p = fc.data_path
        if p.startswith('pose.bones["'):
            b = p[len('pose.bones["'):p.index('"]')]
            if bones is not None and b not in bones:
                continue
            if b in skip:
                continue
        out[(p, fc.array_index, fc.group.name if fc.group else "")] = np.array([fc.evaluate(float(f)) for f in frames])
    return out


def warp_action(name, owner, act, phases, n, lower=None, upper=None, f0=0.0):
    """one action keyed 0..len-1: channels of `act` at phases (mod the loop) — only `lower` bones if given — plus
    every OTHER bone from upper=(action, phases, n_upper) (looped). Quaternion channels stay sign-continuous."""
    m = len(phases)
    fr = np.arange(m, dtype=np.float64)
    ph = np.mod(phases, n - 1) if n > 1 else np.zeros(m)
    ch = eval_action(act, ph + f0, bones=lower)
    if upper is not None:
        a2, ph2, n2 = upper
        low = set(lower or ())
        ch2 = eval_action(a2, np.mod(ph2, n2 - 1) if n2 > 1 else np.zeros(m), skip=low)
        for k, v in ch2.items():
            ch.setdefault(k, v)
    new, slot, cb = C.new_action(name, owner)
    quat = {}
    for (p, i, grp), v in ch.items():
        if p.endswith("rotation_quaternion"):
            quat.setdefault((p, grp), [None] * 4)[i] = v
            continue
        C.write_fc(cb, p, i, fr, v, group=grp)
    for (p, grp), comps in quat.items():
        Q = np.array([c if c is not None else (np.ones(m) if j == 0 else np.zeros(m)) for j, c in enumerate(comps)]).T
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        Q = C.q_continuous(Q)
        for j in range(4):
            C.write_fc(cb, p, j, fr, Q[:, j], group=grp)
    return new
