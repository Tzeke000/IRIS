"""s55 (v2) — ONE evaluation pass of both rigs (frame_set every frame) →
  * foot placement + 2-bone leg IK for the knight (2 legs) and the dragon (4 legs): planted feet are locked to the
    real ground (no slide, no float, stairs stepped), procedural steps when the body moves/turns without the clip
    stepping (anim/feet.py);
  * hand targets for the knight (2-bone arm IK with a weight ramp): left hand resting on the sheathed hilt, right
    hand gripping the sheathed sword for the draw / the sheathe, both hands ON the golden RPG grips (the RPG path is
    computed here from the body, the hands are constrained to it — no drifting, no twisting);
  * S.trk — the 'where is everything' key points (was s60; the feet are taken from the IK result).
The leg/arm bones' local quaternions go into one REPLACE action per rig on the TOP NLA track (K_FOOTIK / D_FOOTIK)."""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

from anim import common as C
from anim import feet as FT
from anim import track as TK


def kchains(K):
    f = lambda n: K.skel.find("mixamorig:" + n)
    legs = [FT.Chain(K.skel, s, f(S + "UpLeg"), f(S + "Leg"), f(S + "Foot"), toe=f(S + "ToeBase"))
            for s, S in (("L", "Left"), ("R", "Right"))]
    arms = [FT.Chain(K.skel, "h" + s, f(S + "Arm"), f(S + "ForeArm"), f(S + "Hand"))
            for s, S in (("L", "Left"), ("R", "Right"))]
    return legs, arms


def dchains(D):
    f = D.skel.find
    out = []
    for s in "LR":
        out.append(FT.Chain(D.skel, "H" + s, f(f"DEF-thigh.{s}"), f(f"DEF-shin.{s}"), f(f"DEF-foot.{s}"),
                            toe=f(f"DEF-toe.{s}"), toe_tail=True))
        out.append(FT.Chain(D.skel, "F" + s, f(f"DEF-upper_arm.{s}"), f(f"DEF-forearm.{s}"), f(f"DEF-forearm.{s}.002"),
                            toe=f(f"DEF-palm.02.{s}"), toe_tail=True))
    return out


def in_windows(ws, N):
    m = np.zeros(N, bool)
    for a, b in ws:
        m[max(int(a), 1) - 1:min(int(b), N)] = True
    return m


def run(S):
    K, D, N = S.K, S.D, S.N
    end = S.args.slice_end or N
    klegs, karms = kchains(K)
    dlegs = dchains(D)
    kb = sorted({b for c in klegs + karms for b in c.bones})
    db = sorted({b for c in dlegs for b in c.bones})
    kp, dp = TK.knight_points(S), TK.dragon_points(S)
    trk = {f"K.{k}": np.zeros((end, 3)) for k in kp}
    trk.update({f"D.{k}": np.zeros((end, 3)) for k in dp})
    for k in ("K.tip", "K.grip", "K.root", "D.root", "K.rshoulder", "K.lshoulder"):
        trk[k] = np.zeros((end, 3))
    tip_y = float(S.R.get("sword.tip_local_y"))
    MK = np.zeros((end, len(kb), 4, 4))
    MD = np.zeros((end, len(db), 4, 4))
    AWK = np.zeros((end, 4, 4))
    AWD = np.zeros((end, 4, 4))
    SHW = np.zeros((end, 4, 4))
    sc = S.sc
    import time
    t0 = time.time()
    rsh, lsh = K.skel.find("mixamorig:RightArm"), K.skel.find("mixamorig:LeftArm")
    for i, fr in enumerate(range(1, end + 1)):
        sc.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        ke, de = K.arm.evaluated_get(dg), D.arm.evaluated_get(dg)
        AWK[i], AWD[i] = np.array(ke.matrix_world), np.array(de.matrix_world)
        for j, b in enumerate(kb):
            MK[i, j] = np.array(ke.pose.bones[b].matrix)
        for j, b in enumerate(db):
            MD[i, j] = np.array(de.pose.bones[b].matrix)
        for arm_e, pts, pre in ((ke, kp, "K"), (de, dp, "D")):
            mw = arm_e.matrix_world
            for k, (b, w) in pts.items():
                trk[f"{pre}.{k}"][i] = TK._bone_pt(arm_e, mw, b, w)
        trk["K.rshoulder"][i] = TK._bone_pt(ke, ke.matrix_world, rsh, "head")
        trk["K.lshoulder"][i] = TK._bone_pt(ke, ke.matrix_world, lsh, "head")
        sw = K.sword.evaluated_get(dg).matrix_world
        trk["K.tip"][i] = np.array(sw @ Vector((0, tip_y, 0)))
        trk["K.grip"][i] = np.array(sw.translation)
        SHW[i] = np.array(K.sheathed.evaluated_get(dg).matrix_world)
        trk["K.root"][i] = np.array(K.root.evaluated_get(dg).matrix_world.translation)
        trk["D.root"][i] = np.array(D.root.evaluated_get(dg).matrix_world.translation)
    S.R.add("feet.eval_pass_s", round(time.time() - t0, 1))
    kix = {b: j for j, b in enumerate(kb)}
    dix = {b: j for j, b in enumerate(db)}

    def world_of(AW, M, ix, bone, tail=False, arm=None):
        X = AW @ M[:, ix[bone]]
        if not tail:
            return X[:, :3, 3], X[:, :3, :3]
        ln = arm.data.bones[bone].length
        return (X @ np.array([0, ln, 0, 1.0]))[:, :3], X[:, :3, :3]

    # ------------------------------------------------------------------ knight sole elevation (bind pose)
    W0 = K.W0
    a0 = W0[K.skel.idx[klegs[0].end], :3, 3]
    t0_ = W0[K.skel.idx[klegs[0].toe], :3, 3]
    e0 = math.atan2(t0_[2] - a0[2], np.linalg.norm((t0_ - a0)[:2]))
    kind_k = dict(thr=0.035, drift=0.22, yaw=35.0, dur=9, lift=0.11, flat=True, e0=e0, max_swing=1, lead=0.5)
    kind_d = dict(thr=0.16, drift=0.6, yaw=30.0, dur=14, lift=0.45, flat=False, e0=0.0, max_swing=2, lead=0.4)
    air_k = in_windows(K.plan["air"], N)[:end]
    air_d = in_windows(D.plan["air"], N)[:end]
    plans = {}
    rootz_k, rootz_d = trk["K.root"][:, 2], trk["D.root"][:, 2]
    for rig, chains, AW, M, ix, arm, kind, air, rz in (("K", klegs, AWK, MK, kix, K.arm, kind_k, air_k, rootz_k),
                                                       ("D", dlegs, AWD, MD, dix, D.arm, kind_d, air_d, rz_d_fix(rootz_d))):
        done = []
        for ch in chains:
            A, R = world_of(AW, M, ix, ch.end)
            T, _ = world_of(AW, M, ix, ch.toe, ch.toe_tail, arm)
            F = dict(ankle=A, toe=T, rot=np.array([FT.orth(r) for r in R]), rootz=rz)

            def others(t, done=done):
                return sum(1 for p in done if p["mode"][t] in ("pswing", "aswing"))
            pl = FT.plan_leg(F, air, S.ground, kind, others=others)
            pl["chain"] = ch
            done.append(pl)
            plans[(rig, ch.name)] = pl
    # ------------------------------------------------------------------ pelvis drop: a planted foot the leg can't reach
    # (stairs, long strides, uneven rubble) lowers the hips (knees bend) instead of stretching the leg off the ground
    drops = {}
    # no drop near a registered contact: the solved contact (s40) must land exactly on its frame
    calm = np.ones(end)
    frs_ = np.arange(1, end + 1)
    for c in S.contacts:
        calm *= np.clip(np.abs(frs_ - c.f) / 14.0 - 0.65, 0.0, 1.0)
    for rig, chains, AW, M, ix, cap in (("K", klegs, AWK, MK, kix, 0.12), ("D", dlegs, AWD, MD, dix, 0.30)):
        need = np.zeros(end)
        for ch in chains:
            pl = plans[(rig, ch.name)]
            Hw = np.einsum("tij,tj->ti", AW, M[:, ix[ch.upper], :, 3])[:, :3]
            Kw = np.einsum("tij,tj->ti", AW, M[:, ix[ch.lower], :, 3])[:, :3]
            Ew = np.einsum("tij,tj->ti", AW, M[:, ix[ch.end], :, 3])[:, :3]
            R = 0.998 * (np.linalg.norm(Kw - Hw, axis=1) + np.linalg.norm(Ew - Kw, axis=1))   # only when truly out of reach
            act = pl["mode"] != "free"
            h = np.linalg.norm((pl["pos"] - Hw)[:, :2], axis=1)
            v = Hw[:, 2] - pl["pos"][:, 2]
            d = v - np.sqrt(np.maximum(R * R - h * h, 0.0))
            need = np.maximum(need, np.where(act, np.clip(d, 0.0, cap), 0.0))
        k = 4
        mx = np.array([need[max(0, t - k):t + k + 1].max() for t in range(end)])
        from anim.timeline import gsmooth
        drop = np.clip(gsmooth(mx, 3.0), 0.0, cap) * gsmooth(calm, 2.0)
        drops[rig] = drop
        AW[:, 2, 3] -= drop
        X = S.K if rig == "K" else S.D
        X.off[:end, 2] -= drop
        for key in trk:
            if key.startswith(rig + "."):
                trk[key][:, 2] -= drop
    from anim.stages.s40_contacts import rebake
    rebake(S, "K")
    rebake(S, "D")
    S.R.add("feet.pelvis_drop", {r: dict(max=round(float(d.max()), 3), frames_over_2cm=int((d > 0.02).sum()))
                                 for r, d in drops.items()})
    # ------------------------------------------------------------------ hand targets (knight)
    S.rpg = rpg_path(S, trk, AWK, MK, kix, karms, end)
    hand = hand_targets(S, trk, SHW, AWK, MK, kix, karms, end)
    # ------------------------------------------------------------------ IK + write
    stats = {}
    for rig, AW, M, ix, chains, owner in (("K", AWK, MK, kix, klegs + karms, K.arm), ("D", AWD, MD, dix, dlegs, D.arm)):
        keys = {}
        for ch in chains:
            pl = plans.get((rig, ch.name))
            ht = hand.get(ch.name) if rig == "K" else None
            Q = {b: np.zeros((end, 4)) for b in (ch.upper, ch.lower, ch.end)}
            prev = None
            clamp = 0
            clamp_runs = []
            for t in range(end):
                Mt = {b: M[t, ix[b]] for b in ch.bones}
                Ai = np.linalg.inv(AW[t])
                Arot = FT.orth(AW[t][:3, :3])
                tgt, rot = None, None
                if pl is not None and pl["mode"][t] != "free":
                    tgt = pl["pos"][t]
                    rot = pl["rot"][t]
                if ht is not None and ht["w"][t] > 0:
                    w = ht["w"][t]
                    cur = (AW[t] @ Mt[ch.end])
                    tgt = cur[:3, 3] * (1 - w) + ht["pos"][t] * w
                    rot = FT.slerp_m(FT.orth(cur[:3, :3]), ht["rot"][t], w) if ht["rot"] is not None else None
                if tgt is None:
                    for b in (ch.upper, ch.lower, ch.end):
                        par = ch.p_upper if b == ch.upper else (ch.p_lower if b == ch.lower else ch.p_end)
                        L = ch.skel.L[ch.skel.idx[b]]
                        Bm = np.linalg.inv(L) @ np.linalg.inv(Mt[par]) @ Mt[b]
                        Q[b][t] = C.mat_to_q(FT.orth(Bm[:3, :3]))
                    continue
                ta = (Ai @ np.r_[tgt, 1.0])[:3]
                ra = Arot.T @ rot if rot is not None else None
                out, endp, prev = ch.solve(Mt, ta, ra, prev)
                if np.linalg.norm(endp - ta) > 0.02 * max(np.linalg.norm(Mt[ch.lower][:3, 3] - Mt[ch.upper][:3, 3]), 1e-3):
                    clamp += 1
                    if clamp_runs and t + 1 - clamp_runs[-1][1] <= 2:
                        clamp_runs[-1][1] = t + 1
                    else:
                        clamp_runs.append([t + 1, t + 1])
                for b, R3 in out.items():
                    Q[b][t] = C.mat_to_q(R3)
            for b, q in Q.items():
                keys[b] = C.q_continuous(q)
            stats[f"{rig}.{ch.name}"] = dict(clamped=clamp, clamp_runs=clamp_runs[:12], **({k: int((pl["mode"] == k).sum()) for k in
                                                                ("plant", "pswing", "aswing", "free")} if pl else {}))
        act, slot, cb = C.new_action(f"{rig}_FOOTIK", owner)
        frs = np.arange(1, end + 1, dtype=np.float64)
        for b, q in keys.items():
            C.write_bone_quats(cb, b, frs, q)
        st = C.put(owner, f"{rig}_FOOTIK", act, 1, blend="REPLACE", name=f"{rig}_FOOTIK")
        st.extrapolation = "NOTHING"
    S.R.add("feet.stats", stats)
    # ------------------------------------------------------------------ tracks: feet from the IK result
    for nm, key in (("L", "K.lfoot"), ("R", "K.rfoot")):
        pl = plans[("K", nm)]
        m = pl["mode"] != "free"
        trk[key][m] = pl["pos"][m]
    for nm, key in (("HL", "D.footL"), ("HR", "D.footR")):
        pl = plans[("D", nm)]
        m = pl["mode"] != "free"
        A0, _ = world_of(AWD, MD, dix, pl["chain"].end)
        trk[key][m] = trk[key][m] + (pl["pos"][m] - A0[m])
    np.savez_compressed(os.path.join(S.debug, "feet_debug.npz"),
                        **{f"{r}_{n}_{k}": np.asarray(v[k], dtype=(object if k == "mode" else np.float64)).astype(str if k == "mode" else np.float64)
                           for (r, n), v in plans.items() for k in ("hc", "h_ank", "down", "mode")},
                        **{f"{r}_{n}_a0": np.array([v["a0"], v["h0"]]) for (r, n), v in plans.items()})
    S.trk = trk
    S.trk_frames = np.arange(1, end + 1)
    S.feet_plans = {k: dict(mode=v["mode"], pos=v["pos"]) for k, v in plans.items()}
    np.savez_compressed(os.path.join(S.debug, "track.npz"), frames=S.trk_frames, **trk)


def rz_d_fix(rz):
    return rz


# ---------------------------------------------------------------------------------------------- hand targets
def ramp(N, lo, hi, a, b):
    w = np.zeros(N)
    fr = np.arange(1, N + 1, dtype=np.float64)
    up = np.clip((fr - lo) / max(a, 1), 0, 1)
    dn = np.clip((hi - fr) / max(b, 1), 0, 1)
    w = np.minimum(up, dn)
    w = w * w * (3 - 2 * w)
    w[(fr < lo) | (fr > hi)] = 0
    return w


def hand_targets(S, trk, SHW, AWK, MK, kix, karms, end):
    """{chain_name: dict(w (end,), pos (end,3) world, rot (end,3,3) world | None)} from K.plan['hands']"""
    K = S.K
    out = {}
    spec = K.plan.get("hands", [])
    off_inv = np.linalg.inv(K.sword_off)
    for h in spec:
        ch = h["chain"]
        w = ramp(end, h["lo"], h["hi"], h.get("rin", 8), h.get("rout", 8))
        d = out.setdefault(ch, dict(w=np.zeros(end), pos=np.zeros((end, 3)), rot=np.tile(np.eye(3), (end, 1, 1))))
        m = w > 0
        if not m.any():
            continue
        if h["kind"] == "hilt_grip":               # right hand closes on the sheathed sword's grip (draw / sheathe)
            Hw = SHW @ off_inv                     # hand world = sheathed-sword world @ (hand→sword)^-1
            d["pos"][m] = Hw[m, :3, 3]
            d["rot"][m] = np.array([FT.orth(x) for x in Hw[m, :3, :3]])
        elif h["kind"] == "hilt_rest":             # left hand resting on the pommel (rotation from the animation)
            ax = SHW[:, :3, 1] / np.linalg.norm(SHW[:, :3, 1], axis=1, keepdims=True)
            pom = SHW[:, :3, 3] - ax * 0.12
            j = kix[karms[0].end]
            cur = AWK @ MK[:, j]
            d["pos"][m] = pom[m] + np.array([0, 0, 0.03])
            d["rot"][m] = np.array([FT.orth(x) for x in cur[m, :3, :3]])
        elif h["kind"] == "rpg":
            G = S.rpg["hand_" + ch]                  # (end, 4, 4) world hand matrices on the RPG grips
            d["pos"][m] = G[m, :3, 3]
            d["rot"][m] = G[m, :3, :3]
        d["w"] = np.maximum(d["w"], w)
    return out


# ---------------------------------------------------------------------------------------------- the golden RPG
def rpg_path(S, trk, AWK, MK, kix, karms, end):
    """RPG world matrix per frame (rest on the hoard → lifted at the pickup → carried/admired in front → shouldered and
    aimed at the dragon's head → recoil on the shot) and the two hand matrices ON its grips (pistol grip = right hand,
    fore grip = left). The hands' grip rotation relative to the RPG is FIXED (taken from the animation at the pickup),
    so the hands can't twist or drift off it (Zeke's v1 note)."""
    K, E = S.K, S.E
    rpg = bpy.data.objects["PRIZE_GoldRPG7"]
    vco = np.array([v.co for v in rpg.data.vertices])
    lo, hi = vco[:, 0].min(), vco[:, 0].max()
    cm = vco.mean(0)
    rad = np.linalg.norm(vco[:, 1:] - cm[1:], axis=1)
    end_w = [rad[vco[:, 0] > hi - 0.15].max(), rad[vco[:, 0] < lo + 0.15].max()]
    msign = 1.0 if end_w[0] >= end_w[1] else -1.0
    mid = np.abs(vco[:, 0] - cm[0]) < 0.12 * (hi - lo)
    r_tube = float(np.percentile(rad[mid], 60))
    Lr = float(hi - lo)
    S.R.add("rpg.len_m", round(Lr, 3))
    S.R.add("rpg.tube_r", round(r_tube, 3))
    sc0 = np.array(rpg.matrix_world.to_scale())
    rest = np.array(rpg.matrix_world)
    fp = int(K.plan["rpg"]["pickup"])
    fl = int(K.plan["rpg"]["lifted"])
    fsh, fR = E["shoulder_rpg"].f, E["fire_rpg"].f
    fsh_done = fsh + int(K.plan["rpg"]["shoulder_frames"])
    W = np.repeat(rest[None], end, 0)
    yaw = (K.base_yaw + K.yoff)[:end]
    tgt = trk["D.head"][min(fR, end) - 1]
    rec = np.zeros(end)
    tt = np.arange(1, end + 1) - (fR - 16)          # recoil at the LAUNCH (the rocket hits on fR)
    rec[tt >= 0] = 0.13 * np.exp(-tt[tt >= 0] / 5.0) * np.cos(tt[tt >= 0] / 3.0)

    def frame_at(t):
        i = t - 1
        fw, rt = C.fwd(yaw[i]), C.right(yaw[i])
        up = np.array([0, 0, 1.0])
        ch = trk["K.chest"][i]
        carry_c = ch + fw * 0.46 - up * 0.10 + rt * 0.02
        ang = math.radians(28 + 6 * math.sin((t - fl) / 40.0))
        carry_ax = -rt * math.cos(ang) + fw * math.sin(ang) + up * 0.12     # muzzle to his LEFT: no crossed arms
        sh = trk["K.rshoulder"][i]
        aim = tgt - sh
        aim = aim / np.linalg.norm(aim)
        sh_c = sh + up * (r_tube + 0.09) + rt * 0.06 + aim * (0.06 * Lr)       # tube further forward: L arm reaches less across
        w = C.smooth01((t - fsh) / max(fsh_done - fsh, 1))
        c = carry_c * (1 - w) + sh_c * w
        ax = carry_ax * (1 - w) + aim * w
        ax = ax / np.linalg.norm(ax)
        c = c - ax * rec[i]
        return c, ax

    def mat(c, ax):
        xa = ax * msign
        za = np.array([0, 0, 1.0]) - xa * xa[2]
        za /= np.linalg.norm(za)
        ya = np.cross(za, xa)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2] = xa * sc0[0], ya * sc0[1], za * sc0[2]
        M[:3, 3] = c - M[:3, :3] @ cm
        return M
    for t in range(fp, end + 1):
        c, ax = frame_at(t)
        M = mat(c, ax)
        if t < fl:                                   # lift off the hoard: blend from rest to the carry pose
            w = C.smooth01((t - fp) / max(fl - fp, 1))
            q0 = C.mat_to_q(FT.orth(rest[:3, :3]))
            q1 = C.mat_to_q(FT.orth(M[:3, :3]))
            R = C.q_to_mat(C.q_slerp(q0, q1, w)) * sc0
            cw = rest[:3, :3] @ cm + rest[:3, 3]
            cc = M[:3, :3] @ cm + M[:3, 3]
            ccur = cw * (1 - w) + cc * w
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = ccur - R @ cm
        W[t - 1] = M
    # grips in RPG mesh space: pistol grip below the tube behind the centre, fore grip ahead of it
    gR = cm + np.array([-0.06 * Lr * msign, 0, -(r_tube + 0.06)])
    gL = cm + np.array([0.22 * Lr * msign, 0, -(r_tube + 0.04)])
    out = {"world": W, "muzzle_local": np.array([hi if msign > 0 else lo, cm[1], cm[2]]), "msign": msign,
           "rear_local": np.array([lo if msign > 0 else hi, cm[1], cm[2]])}
    for name, g, chain in (("hR", gR, karms[1]), ("hL", gL, karms[0])):
        if fp > end:                                 # slice build: the RPG is never picked up
            out["hand_" + name] = np.tile(np.eye(4), (end, 1, 1))
            continue
        j = kix[chain.end]
        Hanim = AWK @ MK[:, j]
        i0 = fp - 1
        Rr0 = FT.orth(W[i0, :3, :3])
        Rel = Rr0.T @ FT.orth(Hanim[i0, :3, :3])
        s_h = np.linalg.norm(Hanim[i0, :3, 0])
        v = K.sword_off[:3, 3].copy()
        if name == "hL":
            v[0] = -v[0]
        G = np.zeros((end, 4, 4))
        for t in range(end):
            R = FT.orth(W[t, :3, :3]) @ Rel
            gp = W[t, :3, :3] @ g + W[t, :3, 3]
            G[t, :3, :3] = R
            G[t, :3, 3] = gp - R @ (v * s_h)
            G[t, 3, 3] = 1
        out["hand_" + name] = G
    return out
