"""Analytic retargeting (no frame_set, no constraints, no operators): FK straight from an action's F-curves, then a
world-delta transfer onto the target rig, written as a fresh slotted action per clip (→ NLA strips).

Method = the one proven in scratch/medieval_prep/scripts/retarget_test.py, vectorised:
  desired_world(t) = D_src(f) @ Align @ W_tgt_ref        D_src(f) = W_src(f) @ W_src_ref^-1   (world = CHARACTER frame:
                                                           each rig re-oriented to face +Y, X right, Z up)
  Align = limb fix: rotate the target's ref joint→child vector onto the source's (UE-style X-along-bone vs Blender Y).
  local basis solved parent→child so the pose matrix reproduces desired_world; root bone also gets its head
  position (source pelvis travel scaled by the hip-height ratio; the clip's net XY drift removed → in place).
Chain mode (creatures): target chain bone at arc fraction u takes the slerped source delta at the same u
(5-bone source tail → 17-bone tail), no Align; written as a COMBINE layer = ref_basis^-1 @ basis (additive delta).
"""
import math

import numpy as np

from . import common as C


def _topo(arm):
    bones = list(arm.data.bones)
    order, seen = [], set()

    def visit(b):
        if b.name in seen:
            return
        if b.parent is not None:
            visit(b.parent)
        seen.add(b.name)
        order.append(b)
    for b in bones:
        visit(b)
    return order


class Skel:
    """Rest data of an armature in its CHARACTER frame. char_from_world: 4x4 (numpy) mapping world → character frame
    (for a rig in the scene: root.matrix_world^-1; for a donor import: a yaw that turns it to face +Y)."""

    def __init__(self, arm, char_from_world=None):
        self.arm = arm
        order = _topo(arm)
        self.names = [b.name for b in order]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.parent = np.array([self.idx[b.parent.name] if b.parent else -1 for b in order])
        self.rest = np.array([np.array(b.matrix_local) for b in order])          # armature space
        L = np.empty_like(self.rest)
        for i, p in enumerate(self.parent):
            L[i] = self.rest[i] if p < 0 else np.linalg.inv(self.rest[p]) @ self.rest[i]
        self.L = L
        cw = np.eye(4) if char_from_world is None else np.asarray(char_from_world, np.float64)
        self.A = cw @ np.array(arm.matrix_world)                                   # armature → character frame
        self.Ainv = np.linalg.inv(self.A)
        sc = np.linalg.norm(self.A[:3, :3], axis=0)
        self.Arot = self.A[:3, :3] / sc

    def short(self, name):
        """bone name without Sketchfab's _NN suffix"""
        import re
        return re.sub(r"_\d+$", "", name)

    def find(self, short):
        for n in self.names:
            if self.short(n) == short or n == short:
                return n
        return None

    # ---- FK
    def basis_from_action(self, action, frames, slot=None):
        """(n, B, 4, 4) basis matrices from the action's F-curves (missing channels = identity / rest)."""
        from bpy_extras import anim_utils
        slot = slot or (action.slots[0] if action.slots else None)
        cb = anim_utils.action_get_channelbag_for_slot(action, slot)
        n, B = len(frames), len(self.names)
        loc = np.zeros((n, B, 3))
        rot = np.zeros((n, B, 4))
        rot[..., 0] = 1
        scl = np.ones((n, B, 3))
        fr = np.asarray(frames, np.float64)
        for fc in cb.fcurves:
            p = fc.data_path
            if not p.startswith('pose.bones["'):
                continue
            bname = p[len('pose.bones["'):p.index('"]')]
            if bname not in self.idx:
                continue
            j = self.idx[bname]
            vals = np.array([fc.evaluate(f) for f in fr])
            if p.endswith("rotation_quaternion"):
                rot[:, j, fc.array_index] = vals
            elif p.endswith(".location"):
                loc[:, j, fc.array_index] = vals
            elif p.endswith(".scale"):
                scl[:, j, fc.array_index] = vals
        Bm = np.zeros((n, B, 4, 4))
        Bm[..., :3, :3] = C.q_to_mat(rot) * scl[..., None, :]
        Bm[..., :3, 3] = loc
        Bm[..., 3, 3] = 1
        return Bm

    def fk(self, Bm):
        """(n,B,4,4) basis → (n,B,4,4) armature-space pose matrices."""
        P = np.empty_like(Bm)
        for i, p in enumerate(self.parent):
            P[:, i] = (self.L[i] if p < 0 else P[:, p] @ self.L[i]) @ Bm[:, i]
        return P

    def world(self, P):
        """armature-space pose → character-frame matrices (n,B,4,4)"""
        return self.A @ P

    def ident(self, n=1):
        Bm = np.zeros((n, len(self.names), 4, 4))
        Bm[..., :, :] = np.eye(4)
        return Bm


def facing_yaw_fix(points_fwd_from, points_fwd_to):
    """4x4 rotation about Z that turns direction (to - from) (projected on XY) into +Y."""
    d = np.asarray(points_fwd_to, np.float64) - np.asarray(points_fwd_from, np.float64)
    a = math.atan2(d[0], d[1])           # angle from +Y, clockwise positive
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    M[:2, :2] = [[c, -s], [s, c]]
    return M


def _solve_local(tgt, P_parent_fn, desired, ref_basis, root_head=None, root_name=None):
    """desired: dict bone -> (n,3,3) character-frame rotations. Returns basis (n,B,4,4)."""
    n = next(iter(desired.values())).shape[0]
    Bm = np.repeat(ref_basis[None], n, 0) if ref_basis.ndim == 3 else ref_basis.copy()
    P = np.empty_like(Bm)
    for i, p in enumerate(tgt.parent):
        M0 = tgt.L[i] if p < 0 else P[:, p] @ tgt.L[i]
        M0 = np.broadcast_to(M0, (n, 4, 4))
        name = tgt.names[i]
        if name in desired:
            Ra = tgt.Arot.T @ desired[name]                       # (n,3,3) armature-space rotation
            Rb = np.swapaxes(M0[:, :3, :3], 1, 2) @ Ra
            Bm[:, i, :3, :3] = Rb
            Bm[:, i, :3, 3] = 0
            if root_head is not None and name == root_name:
                h = np.concatenate([root_head, np.ones((n, 1))], 1) @ tgt.Ainv.T     # (n,4) armature space
                Bm[:, i, :3, 3] = np.einsum("nji,nj->ni", M0[:, :3, :3], h[:, :3] - M0[:, :3, 3])
        P[:, i] = M0 @ Bm[:, i]
    return Bm, P


def retarget_human(src, src_basis, tgt, pairs, child_of, root_pair, tgt_ref_basis=None, in_place=True,
                   src_ref_basis=None, root_ref=None, ratio=None):
    """pairs: [(src_bone, tgt_bone)]; child_of: src_bone -> src_child_bone used for limb alignment;
    root_pair: (src_pelvis, tgt_hips). Returns (tgt_basis (n,B,4,4), info)."""
    n = src_basis.shape[0]
    ref_b = tgt.ident(1)[0] if tgt_ref_basis is None else tgt_ref_basis
    Ps = src.fk(src_basis)
    Ws = src.world(Ps)
    Ps0 = src.fk(src.ident(1) if src_ref_basis is None else src_ref_basis[None])
    Ws0 = src.world(Ps0)[0]
    Pt0 = tgt.fk(ref_b[None])
    Wt0 = tgt.world(Pt0)[0]
    pmap = dict(pairs)
    desired = {}
    for s, t in pairs:
        i, j = src.idx[s], tgt.idx[t]
        rs0 = Ws0[i, :3, :3] / np.linalg.norm(Ws0[i, :3, :3], axis=0)
        rs = Ws[:, i, :3, :3] / np.linalg.norm(Ws[:, i, :3, :3], axis=1)[:, None, :]
        rt0 = Wt0[j, :3, :3] / np.linalg.norm(Wt0[j, :3, :3], axis=0)
        align = np.eye(3)
        ch = child_of.get(s)
        if ch and ch in pmap:
            hs, cs = Ws0[i, :3, 3], Ws0[src.idx[ch], :3, 3]
            ht, ct = Wt0[j, :3, 3], Wt0[tgt.idx[pmap[ch]], :3, 3]
            if np.linalg.norm(cs - hs) > 1e-6 and np.linalg.norm(ct - ht) > 1e-6:
                align = _rot_between(ct - ht, cs - hs)
        D = rs @ rs0.T
        desired[t] = D @ align @ rt0
    sp, tp = root_pair
    hs0 = Ws0[src.idx[sp], :3, 3] if root_ref is None else np.asarray(root_ref, np.float64)
    ht0 = Wt0[tgt.idx[tp], :3, 3]
    if ratio is None:
        ratio = ht0[2] / max(hs0[2], 1e-6)
    hs = Ws[:, src.idx[sp], :3, 3]
    delta = (hs - hs0) * ratio
    travel = delta[-1, :2] - delta[0, :2]
    if in_place and n > 2:
        tt = np.arange(n)
        for k in (0, 1):
            a, b = np.polyfit(tt, delta[:, k], 1)
            delta[:, k] -= a * tt + b
    head = ht0 + delta
    Bm, P = _solve_local(tgt, None, desired, ref_b, root_head=head, root_name=tp)
    return Bm, dict(ratio=float(ratio), travel=travel.tolist(), P=P)


def _rot_between(a, b):
    """3x3 rotation taking direction a onto direction b (shortest arc)."""
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    b = np.asarray(b, np.float64) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        return C.q_to_mat(C.q_axis_angle(ax, math.pi))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def retarget_chains(src, src_basis, src_ref_basis, tgt, chains, tgt_ref_basis):
    """Creature chains. chains: [(src_bones[], tgt_bones[], gain)]. World-delta transfer by arc fraction, relative to
    the source's OWN ref pose (src_ref_basis, e.g. the clip's first frame) → additive COMBINE layer on the target's
    ref pose. Returns layer quats dict tgt_bone -> (n,4)."""
    n = src_basis.shape[0]
    Ws = src.world(src.fk(src_basis))
    Ws0 = src.world(src.fk(src_ref_basis[None]))[0]
    Wt0 = tgt.world(tgt.fk(tgt_ref_basis[None]))[0]
    desired = {}
    for sb, tb, gain in chains:
        Ds = []
        for s in sb:
            i = src.idx[s]
            r0 = Ws0[i, :3, :3] / np.linalg.norm(Ws0[i, :3, :3], axis=0)
            r = Ws[:, i, :3, :3] / np.linalg.norm(Ws[:, i, :3, :3], axis=1)[:, None, :]
            D = r @ r0.T
            Ds.append(C.mat_to_q(D))                      # (n,4)
        Ds = np.array(Ds)                                    # (S,n,4)
        S, T = len(sb), len(tb)
        for k, t in enumerate(tb):
            u = (k + 0.5) / T * S - 0.5
            i0 = int(np.clip(math.floor(u), 0, S - 1))
            i1 = min(i0 + 1, S - 1)
            fr = float(np.clip(u - i0, 0, 1))
            q = np.array([C.q_slerp(Ds[i0, f], Ds[i1, f], fr) for f in range(n)])
            if gain != 1.0:
                ang = 2 * np.arccos(np.clip(q[:, 0], -1, 1))
                ax = q[:, 1:] / np.maximum(np.linalg.norm(q[:, 1:], axis=1, keepdims=True), 1e-12)
                q = C.q_axis_angle(ax, ang * gain)
            j = tgt.idx[t]
            rt0 = Wt0[j, :3, :3] / np.linalg.norm(Wt0[j, :3, :3], axis=0)
            desired[t] = C.q_to_mat(q) @ rt0
    Bm, _P = _solve_local(tgt, None, desired, tgt_ref_basis)
    ref_q = C.mat_to_q(tgt_ref_basis[:, :3, :3])
    out = {}
    for t in desired:
        j = tgt.idx[t]
        bq = C.mat_to_q(Bm[:, j, :3, :3])
        out[t] = C.q_mul(C.q_conj(np.broadcast_to(ref_q[j], bq.shape)), bq)
    # parents of mapped bones that are NOT mapped keep identity layers; children inherit
    return out


def basis_to_quats(Bm, idx):
    return C.mat_to_q(Bm[:, idx, :3, :3])
