"""Foot placement + analytic 2-bone IK (v2) — planted feet do not slide, the body never glides on locked feet.

One evaluation pass (frame_set) records the evaluated pose of every leg/arm chain bone; then, per leg, a small state
machine decides where the foot IS:
  free    airborne windows (jumps, throws, flight, sitting) — the animation as it is
  plant   the clip puts the foot down → LOCKED at that world point (z snapped to the real ground under it, knight soles
          flattened), held until the clip lifts it
  aswing  the clip lifts the foot → follow the clip, with the lock offset fading out by the clip's next plant
  pswing  PROCEDURAL STEP: a planted foot drifted too far from where the clip wants it (root moved/turned without the
          clip stepping) → lift on an arc and re-plant where the clip will be at the end of the step; at most one
          knight foot / two dragon feet off the ground at a time.
Then per frame and leg: 2-bone IK (upper bone rotates about its head so the knee lands on the circle of solutions in
the animation's own knee plane, lower bone aims the ankle at the target, end bone takes the planted rotation) and the
three bones' local quaternions are written as one REPLACE action on the TOP NLA track ('*_FOOTIK').
Hand targets (hilt, RPG grips, gem) use the same IK with a weight ramp.
"""
import math

import numpy as np

from anim import common as C


def rot_between(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        ax = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1.0, 0])
        ax /= np.linalg.norm(ax)
        return C.q_to_mat(C.q_axis_angle(ax, math.pi))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def orth(M):
    return M / np.linalg.norm(M, axis=0)


class Chain:
    """upper → (…) → lower → (…) → end; bones between are carried rigidly."""

    def __init__(self, skel, name, upper, lower, end, toe=None, toe_tail=False, **kw):
        self.name = name
        self.skel = skel
        self.upper, self.lower, self.end, self.toe = upper, lower, end, toe
        self.toe_tail = toe_tail
        par = lambda b: skel.names[skel.parent[skel.idx[b]]]
        self.p_upper, self.p_lower, self.p_end = par(upper), par(lower), par(end)
        self.bones = [b for b in dict.fromkeys([self.p_upper, upper, self.p_lower, lower, self.p_end, end] +
                                               ([toe] if toe else []))]
        self.kw = kw

    def solve(self, M, target, rot=None, prev_bend=None):
        """M: {bone: 4x4 armature-space pose}; target: armature-space ankle point; rot: 3x3 armature-space end rotation
        (None = carried rigidly). Returns ({bone: basis 3x3}, new end head, bend)"""
        sk = self.skel
        Mu, Ml, Me = M[self.upper], M[self.lower], M[self.end]
        H, Kp, E = Mu[:3, 3], Ml[:3, 3], Me[:3, 3]
        l1, l2 = np.linalg.norm(Kp - H), np.linalg.norm(E - Kp)
        TH = target - H
        dist = np.linalg.norm(TH)
        u = TH / max(dist, 1e-9)
        d = float(np.clip(dist, abs(l1 - l2) * 1.02 + 1e-5, (l1 + l2) * 0.9995))
        Tc = H + u * d
        bend = (Kp - H) - u * np.dot(Kp - H, u)
        if np.linalg.norm(bend) < 1e-5 * max(l1, 1e-3):
            bend = prev_bend if prev_bend is not None else np.cross(u, [1.0, 0, 0])
        b = bend / max(np.linalg.norm(bend), 1e-12)
        a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
        h = math.sqrt(max(l1 * l1 - a * a, 0.0))
        Ks = H + u * a + b * h
        R1 = rot_between(Kp - H, Ks - H)
        E1 = H + R1 @ (E - H)
        R2 = rot_between(E1 - Ks, Tc - Ks)
        R21 = R2 @ R1

        def moved(bone, R, pivot, newpivot):
            X = M[bone].copy()
            X[:3, :3] = R @ X[:3, :3]
            X[:3, 3] = newpivot + R @ (M[bone][:3, 3] - pivot)
            return X
        N = {self.upper: moved(self.upper, R1, H, H)}
        N[self.p_lower] = moved(self.p_lower, R1, H, H) if self.p_lower != self.upper else N[self.upper]
        N[self.lower] = moved(self.lower, R21, Kp, Ks)
        N[self.p_end] = moved(self.p_end, R21, Kp, Ks) if self.p_end != self.lower else N[self.lower]
        Ne = M[self.end].copy()
        if rot is not None:
            sc = np.linalg.norm(Me[:3, :3], axis=0)
            Ne[:3, :3] = rot * sc
        else:
            Ne[:3, :3] = R21 @ Me[:3, :3]
        Ne[:3, 3] = Tc
        N[self.end] = Ne
        out = {}
        for bone, parent in ((self.upper, self.p_upper), (self.lower, self.p_lower), (self.end, self.p_end)):
            Pp = M[parent] if bone == self.upper else N[parent]
            L = sk.L[sk.idx[bone]]
            B = np.linalg.inv(L) @ np.linalg.inv(Pp) @ N[bone]
            out[bone] = orth(B[:3, :3])
        return out, Tc, bend


def smooth01(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def plan_leg(F, air, ground, kind, look=None, others=None, t_index=None):
    """F: dict(ankle (N,3) world, toe (N,3) world, rot (N,3,3) world end rotation, rootz (N,)) → dict(pos, rot, mode)
    kind: dict(thr, drift, yaw, dur, lift, flat, e0, h0)"""
    A, T, R, rz = F["ankle"], F["toe"], F["rot"], F["rootz"]
    N = len(A)
    hc = np.minimum(A[:, 2], T[:, 2]) - rz
    ok = ~air
    a0 = np.percentile(hc[ok], 8) if ok.any() else 0.0
    # down = near the LOCAL stance height (a clip with bent knees keeps its feet higher than the global minimum),
    # but never more than 30 cm up
    wl = 24
    loc = np.array([hc[max(0, t - wl):t + wl + 1][ok[max(0, t - wl):t + wl + 1]].min()
                    if ok[max(0, t - wl):t + wl + 1].any() else a0 for t in range(N)])
    down = (hc < np.maximum(loc, a0) + kind["thr"]) & (hc < a0 + 0.30)
    gA = np.array([ground(p[0], p[1]) for p in A])
    gT = np.array([ground(p[0], p[1]) for p in T])
    h_ank = A[:, 2] - gA
    h0 = kind.get("h0")
    if h0 is None:
        m = down & ok
        h0 = float(np.percentile(h_ank[m], 15)) if m.any() else 0.08
    pos = A.copy()
    rot = R.copy()
    mode = np.array(["free"] * N, dtype=object)
    st = "free"
    lock_p = lock_r = None
    sw = {}
    for t in range(N):
        if air[t]:
            st = "free"
            continue
        anim_p = A[t]
        if st == "pswing":
            s = (t - sw["t0"]) / sw["dur"]
            if s >= 1.0:
                st = "plant"
                lock_p, lock_r = sw["p1"], sw["r1"]
            else:
                e = smooth01((s - 0.12) / 0.8)                 # lift first, then travel, set down
                p = sw["p0"] + (sw["p1"] - sw["p0"]) * e
                p[2] += kind["lift"] * math.sin(math.pi * s) ** 0.6
                pos[t] = p
                rot[t] = slerp_m(sw["r0"], sw["r1"], e)
                mode[t] = "pswing"
                continue
        if down[t]:
            if st in ("free", "aswing"):
                st = "plant"
                p = pos[t - 1].copy() if t > 0 and mode[t - 1] != "free" else anim_p.copy()
                p[2] = ground(p[0], p[1]) + h0
                lock_p = p
                lock_r = flat_rot(R[t], A[t], T[t], kind) if kind["flat"] else R[t]
            elif st == "plant":
                drift = np.linalg.norm(lock_p[:2] - anim_p[:2])
                yaw_d = yaw_between(lock_r, R[t])
                busy = others(t) if others else 0
                if (drift > kind["drift"] or yaw_d > kind["yaw"]) and busy < kind.get("max_swing", 1):
                    dur = kind["dur"]
                    j = min(t + dur, N - 1)
                    p1 = A[j].copy() + (A[j] - A[t]) * kind.get("lead", 0.0) * np.array([1, 1, 0])
                    p1[2] = ground(p1[0], p1[1]) + h0
                    r1 = flat_rot(R[j], A[j], T[j], kind) if kind["flat"] else R[j]
                    sw = dict(t0=t, dur=dur, p0=lock_p.copy(), p1=p1, r0=lock_r, r1=r1)
                    st = "pswing"
                    pos[t] = lock_p
                    rot[t] = lock_r
                    mode[t] = "pswing"
                    continue
            pos[t] = lock_p
            rot[t] = lock_r
            mode[t] = "plant"
        else:
            if st == "plant":
                st = "aswing"
                t1 = t
                while t1 < N - 1 and not down[t1] and t1 - t < 45:
                    t1 += 1
                sw = dict(t0=t, t1=t + max(1, min(t1 - t, kind.get("fade", 10))), off=lock_p - anim_p, r0=lock_r)
            if st == "aswing":
                w = smooth01((t - sw["t0"]) / (sw["t1"] - sw["t0"]))
                p = anim_p + sw["off"] * (1 - w)
                g = ground(p[0], p[1]) + h0
                p[2] = max(p[2], g)
                pos[t] = p
                rot[t] = slerp_m(sw["r0"], R[t], w)
                mode[t] = "aswing"
            else:
                st = "free"
    return dict(pos=pos, rot=rot, mode=mode, h0=h0, a0=a0, down=down, hc=hc, h_ank=h_ank)


def yaw_between(Ra, Rb):
    fa, fb = Ra[:, 1], Rb[:, 1]
    a = math.atan2(fa[1], fa[0])
    b = math.atan2(fb[1], fb[0])
    return abs((math.degrees(a - b) + 180) % 360 - 180)


def slerp_m(Ra, Rb, t):
    qa, qb = C.mat_to_q(orth(Ra)), C.mat_to_q(orth(Rb))
    return C.q_to_mat(C.q_slerp(qa, qb, t))


def flat_rot(R, A, T, kind):
    """rotate the foot about its ankle so the ankle→toe vector has its rest elevation e0 (sole flat on the ground)"""
    f = T - A
    hz = np.linalg.norm(f[:2])
    if hz < 1e-6:
        return R
    L = np.linalg.norm(f)
    e0 = kind["e0"]
    f2 = np.r_[f[:2] / hz * L * math.cos(e0), L * math.sin(e0)]
    return rot_between(f, f2) @ orth(R)
