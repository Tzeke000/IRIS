"""Hand-key DSL + procedural flinch, both written as COMBINE (additive) quaternion layers.

A pose is {group: (rx, ry, rz)} in DEGREES about the CHARACTER axes (X right, Y forward, Z up; right-hand rule:
+rx tips a forward-pointing bone UP / swings a hanging arm FORWARD, +rz turns LEFT, +ry rolls the top to the RIGHT).
A group spreads its rotation over its bones by weight. Per bone the char-frame rotation is conjugated into the bone's
own frame using the rig's reference pose (dragon: Idle Stand f0, knight: bind pose):  layer = R_ref^-1 · q_char · R_ref,
which is exactly "rotate this bone about its head by q_char" when the base pose is near the reference.
"""
import math

import numpy as np

from anim import common as C


class Kit:
    def __init__(self, skel, ref_world, groups, alt_refs=None):
        """groups: {name: [(bone, weight), ...]}; alt_refs: {group: ref_world} for groups whose base pose differs a lot
        from the main reference (the dragon's tail is curled in Idle Stand but straight during strikes / in flight)."""
        self.skel = skel
        self.groups = groups

        def qs(W):
            out = {}
            for i, n in enumerate(skel.names):
                R = W[i, :3, :3]
                out[n] = C.mat_to_q(R / np.linalg.norm(R, axis=0))
            return out
        self.Rq = qs(ref_world)
        self.Rq_g = {g: qs(W) for g, W in (alt_refs or {}).items()}

    def char_to_local(self, bone, qc, group=None):
        r = self.Rq_g[group][bone] if group in self.Rq_g else self.Rq[bone]
        return C.q_mul(C.q_mul(C.q_conj(np.broadcast_to(r, qc.shape)), qc), np.broadcast_to(r, qc.shape))

    def layers(self, angles):
        """angles: {group: (n,3) degrees} → {bone: (n,4) local layer quats} (groups sharing a bone compose)."""
        out = {}
        for g, a in angles.items():
            a = np.radians(np.asarray(a, np.float64))
            for bone, w in self.groups[g]:
                qx = C.q_axis_angle(np.array([1.0, 0, 0]), a[:, 0] * w)
                qy = C.q_axis_angle(np.array([0, 1.0, 0]), a[:, 1] * w)
                qz = C.q_axis_angle(np.array([0, 0, 1.0]), a[:, 2] * w)
                qc = C.q_mul(qz, C.q_mul(qy, qx))
                ql = self.char_to_local(bone, qc, g)
                out[bone] = ql if bone not in out else C.q_mul(out[bone], ql)
        return out


def chain_group(bones, mode="uniform"):
    n = len(bones)
    if mode == "uniform":
        w = np.ones(n)
    elif mode == "ramp":            # more toward the tip
        w = np.linspace(0.5, 1.5, n)
    elif mode == "root":            # more toward the root
        w = np.linspace(1.5, 0.5, n)
    else:
        raise ValueError(mode)
    w = w / w.sum()
    return [(b, float(x)) for b, x in zip(bones, w)]


def move_angles(keys, groups_used, n):
    """keys: [(rel_frame, {group: (rx,ry,rz)}, opt)] → {group: (n,3)}; groups missing in a key are 0 there.
    Pose-to-pose: every key is a 'stop' (ease in/out) unless opt contains 'flow'."""
    out = {}
    fr = np.arange(n)
    for g in groups_used:
        ks = []
        for k in keys:
            opt = k[2] if len(k) > 2 else ""
            ks.append((k[0], k[1].get(g, (0, 0, 0)), "" if "flow" in opt else ("stop" + (" lin" if "lin" in opt else ""))))
        out[g] = C.hermite_keys(ks, fr)
    return out


def write_layer_action(name, owner, layers, n, step=1):
    act, slot, cb = C.new_action(name, owner)
    fr = np.arange(0, n, step, dtype=np.float64)
    if fr[-1] != n - 1:
        fr = np.r_[fr, n - 1]
    idx = fr.astype(int)
    for b, q in layers.items():
        C.write_bone_quats(cb, b, fr, q[idx])
    return act


# ---------------------------------------------------------------------------------------- procedural flinch
PRESETS = {
    "dragon_heavy": dict(w=0.45, z=0.35, max=18.0, lag=1.0),
    "dragon_light": dict(w=0.6, z=0.4, max=9.0, lag=1.0),
    "dragon_tail": dict(w=0.35, z=0.25, max=25.0, lag=1.5),
    "knight": dict(w=0.8, z=0.45, max=12.0, lag=0.0),
    "knight_heavy": dict(w=0.6, z=0.35, max=22.0, lag=0.5),
}


def spring(n, t0, A, w, z):
    t = np.arange(n, dtype=np.float64) - t0
    wd = w * math.sqrt(1 - z * z)
    return np.where(t >= 0, A * np.exp(-z * w * np.maximum(t, 0)) * np.sin(wd * np.maximum(t, 0)), 0.0)


def flinch_layers(kit, events, n, f0=1):
    """events: [dict(frame, chain=[bones from hit outward], dir=char-frame unit vector (push direction),
    strength, preset)]. Bone k along the chain gets axis = y_bone × dir, amplitude max·strength·0.75^k, delay k·lag.
    Returns {bone: (n,4)} local layer quats over frames f0..f0+n-1."""
    acc = {}
    for ev in events:
        p = PRESETS[ev["preset"]]
        d = np.asarray(ev["dir"], np.float64)
        d = d / max(np.linalg.norm(d), 1e-9)
        for k, b in enumerate(ev["chain"]):
            R = C.q_to_mat(kit.Rq[b])
            yb = R[:, 1]
            ax = np.cross(yb, d)
            if np.linalg.norm(ax) < 1e-6:
                continue
            ax /= np.linalg.norm(ax)
            A = math.radians(p["max"]) * ev["strength"] * (0.75 ** k)
            th = spring(n, ev["frame"] - f0 + k * p["lag"], A, p["w"], p["z"])
            acc.setdefault(b, np.zeros((n, 3)))
            acc[b] += np.outer(th, ax)
    out = {}
    for b, rv in acc.items():
        ang = np.linalg.norm(rv, axis=1)
        lim = math.radians(35)
        rv = np.where((ang > lim)[:, None], rv * (lim / np.maximum(ang, 1e-9))[:, None], rv)
        qc = C.q_from_rotvec(rv)
        out[b] = kit.char_to_local(b, qc)
    return out
