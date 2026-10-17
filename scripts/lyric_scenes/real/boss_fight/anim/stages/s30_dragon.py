"""s30 — dragon: native clips on BASE_A/B, TAIL_STRAIGHT overlay (REPLACE tail bones with the Walk tail) for the tail
strikes, T-rex donor layers (COMBINE), hand-keyed moves (COMBINE), root path (ground, spins, flight)."""
import os

import numpy as np

from anim import common as C
from anim import dragon_clips as DC
from anim import posekit as PK
from anim import timeline as TL


def groups(D):
    ch = D.chains
    G = {"spine": PK.chain_group(ch["spine"]), "neck": PK.chain_group(ch["neck"][:4]), "head": [(ch["neck"][4], 1.0)],
         "jaw": [(ch["jaw"][0], 1.0)], "tail": PK.chain_group(ch["tail"]), "tail_tip": PK.chain_group(ch["tail"][8:]),
         "tail_base": PK.chain_group(ch["tail"][:6], "root"),
         "neck_sit": PK.chain_group(ch["neck"][:4]), "head_sit": [(ch["neck"][4], 1.0)]}
    for s in "LR":
        G["arm" + s] = PK.chain_group(ch["arm" + s][1:3])
        G["forearm" + s] = PK.chain_group(ch["arm" + s][3:6])
        G["leg" + s] = PK.chain_group(ch["leg" + s][1:3])
        G["wing" + s] = [(ch["wing" + s][2], 1.0)]
        G["wingfold" + s] = [(ch["wing" + s][3], 0.5), (ch["wing" + s][4], 0.5)]
    sk = D.skel
    lids = []
    for s in "LR":
        for i in range(1, 5):
            t = sk.find(f"DEF-lid{i}.T.{s}")
            b = sk.find(f"DEF-lid{i}.B.{s}")
            if t:
                lids.append((t, -35.0))
            if b:
                lids.append((b, 20.0))
    G["lids"] = []          # v2: lids are CLOSED by translation (lids_action) — rotating them never covered the eye
    return G


def lids_action(S, windows):
    """eyelids shut: every top/bottom lid bone slides 88% of the way to its eyeball centre (pose LOCATION, in the
    bone's rest frame) → COMBINE keys. windows: [(close_from, open_at)] — closed until open_at-3, open by open_at."""
    D = S.D
    arm = D.arm
    act, slot, cb = C.new_action("D_lids", arm)
    eye = {s: np.array(arm.data.bones[D.skel.find(f"DEF-eye.{s}")].head_local) for s in "LR"}
    keys = []
    for (a, b) in windows:
        keys += ([(a - 6, 0.0)] if a > 6 else []) + [(a, 1.0), (b - 3, 1.0), (b, 0.0)]
    keys = sorted(keys) + [(S.N, 0.0)]
    fr = [k[0] for k in keys]
    w = [k[1] for k in keys]
    n = 0
    head_c = np.array(arm.data.bones[D.skel.find("DEF-neck.004")].head_local)
    lidr = {s: np.mean([np.linalg.norm(np.array(b.head_local) - eye[s]) for b in arm.data.bones
                        if D.skel.short(b.name).startswith("DEF-lid") and D.skel.short(b.name)[-1] == s]) for s in "LR"}
    for b in arm.data.bones:
        nm = D.skel.short(b.name)
        if nm.startswith(("DEF-eye.", "DEF-eye_master.")) and nm[-1] in "LR":
            # the eyeball sinks into the socket while the lids are shut (it bulged through the closed lids)
            side = nm[-1]
            back = head_c - eye[side]
            back[2] = 0.0
            d_arm = back / max(np.linalg.norm(back), 1e-9) * lidr[side] * 0.55
            if nm.startswith("DEF-eye_master"):
                continue
        elif nm.startswith("DEF-lid"):
            side = nm[-1]
            d_arm = (eye[side] - np.array(b.head_local)) * 0.95
        else:
            continue
        Rbl = np.array(b.matrix_local)[:3, :3]
        t = Rbl.T @ d_arm
        for i in range(3):
            C.write_fc(cb, f'pose.bones["{b.name}"].location', i, fr, [x * t[i] for x in w], "BEZIER", group=b.name)
        n += 1
    C.put(arm, "D_LIDS", act, 1, blend="COMBINE")
    S.R.add("dragon.lid_bones", n)


def tail_straight_action(S):
    D = S.D
    Bw = D.skel.basis_from_action(D.native["Walk"]["action"], [0])[0]
    act, slot, cb = C.new_action("D_tail_straight", D.arm)
    for b in D.chains["tail"]:
        q = C.mat_to_q(Bw[D.skel.idx[b], :3, :3])
        C.write_bone_quats(cb, b, [0.0, 1.0], np.array([q, q]), "CONSTANT")
    return act


def curl_action(S):
    D = S.D
    act, slot, cb = C.new_action("D_tail_curl", D.arm)
    for b in D.chains["tail"]:
        q = C.mat_to_q(D.ref[D.skel.idx[b], :3, :3])
        C.write_bone_quats(cb, b, [0.0, 1.0], np.array([q, q]), "CONSTANT")
    return act


def run(S):
    DC.build(S)
    DC.debug(S, os.path.join(S.debug, "dragon_sticks.json"))
    D, E, N = S.D, S.E, S.N
    plan = D.plan
    meta = {k: dict(action=v["action"], n=v["n"]) for k, v in D.native.items()}
    # ---- root FIRST (v2): yaw with whole-body strike pivots, height; then the walks are time-warped to it
    yaw, vals = TL.resolve_yaw(plan["yaw"], D.xy, S.K.xy, N)
    D.xy = D.xy.copy()
    for st in plan["strikes"]:                 # the body turns about its FRONT LEGS: root = pivot - fwd(yaw)·CHEST
        from anim.staging import CHEST
        a, b = st["a"], st["b"]
        fr = np.arange(a, b + 1)
        w = np.clip(np.minimum((fr - a) / 6.0, (b - fr) / 6.0), 0, 1)
        w = w * w * (3 - 2 * w)
        yr = np.radians(yaw[fr - 1])
        piv = np.c_[st["pivot"][0] + np.sin(yr) * CHEST, st["pivot"][1] - np.cos(yr) * CHEST]
        D.xy[fr - 1] = D.xy[fr - 1] * (1 - w[:, None]) + piv * w[:, None]
    z, g = TL.height(plan["hz"], D.xy, S.ground, N, smooth=3.0)
    D.base_pos = np.c_[D.xy, z]
    D.base_yaw = yaw
    loco_segments(S, plan, meta)
    D.placed = TL.place_base(D.arm, plan["seg"], meta, N, S.R, "D_")
    # ---- curled tail (Idle Stand f0) during every ground Walk/Run: a straight 10 m walking tail sweeps the cover
    tc = curl_action(S)
    for p in D.placed:
        if (p["clip"] in ("Walk", "Run") or "_Walk" in p["clip"] or "_Run" in p["clip"]) and p["start"] < E["launch"].f:
            st = C.put(D.arm, "D_TAIL_CURL", tc, p["start"] - 4, scale=float(p["end"] - p["start"] + 8))
            st.blend_in = st.blend_out = 10.0
    # ---- straight tail for strikes
    ts = tail_straight_action(S)
    for a, b in plan["tail_straight"]:
        st = C.put(D.arm, "D_TAIL_STRAIGHT", ts, a, scale=float(b - a))
        st.blend_in = st.blend_out = 8.0
    # ---- donors
    items = []
    for d in plan["donors"]:
        items.append((d["start"], d["start"] + d["n"], d))
    for ti, s, e, d in TL.pack_tracks(items):
        act = D.donors[d["clip"]]["action"]
        st = C.put(D.arm, f"D_DONOR_{ti}", act, s, blend="COMBINE", name=f"{d['clip']}@{s}")
        st.frame_end = float(s + min(d["n"], act.frame_range[1] - act.frame_range[0]))
        st.blend_in = st.blend_out = float(d.get("xf", 10))
        if d.get("gain", 1.0) != 1.0:
            st.use_animated_influence = True
            st.influence = d["gain"]
    # ---- hand keys
    Ww = DC.native_pose(S, "Walk", 0)
    D.kit = PK.Kit(D.skel, D.W0, groups(D), alt_refs={"tail": Ww, "tail_tip": Ww, "tail_base": Ww,
                                                    "neck_sit": DC.native_pose(S, "Idle Sit", 0),
                                                    "head_sit": DC.native_pose(S, "Idle Sit", 0)})
    items = []
    for mv in plan["hk"]:
        n = int(mv["keys"][-1][0]) + 1
        gs = sorted({g for k in mv["keys"] for g in k[1]})
        lay = D.kit.layers(PK.move_angles(mv["keys"], gs, n))
        act = PK.write_layer_action(f"D_hk_{mv['name']}", D.arm, lay, n, step=1 if n < 400 else 2)
        items.append((mv["start"], mv["start"] + n - 1, act))
    for ti, s, e, act in TL.pack_tracks(items):
        C.put(D.arm, f"D_HANDKEY_{ti}", act, s, blend="COMBINE")
    E_ = S.E
    lids_action(S, [(1, E_["eye_open"].f + 3), (E_["collapse"].f + 40, E_["eye_again"].f + 3)])
    S.R.add("dragon.strips", sum(len(t.strips) for t in D.arm.animation_data.nla_tracks))

    D.off = np.zeros((N, 3))
    D.yoff = np.zeros(N)
    D.ground_z = g
    TL.bake_root(D.root, D.base_pos, D.base_yaw, N)
    sp = np.linalg.norm(np.diff(D.base_pos, axis=0), axis=1) * C.FPS
    S.R.add("dragon.max_speed_mps", round(float(sp.max()), 1))
    S.R.add("dragon.max_yaw_rate_dps", round(float(np.abs(np.diff(yaw)).max() * C.FPS), 0))


def dragon_feet(sk):
    f = sk.find
    return [(f("DEF-foot.L"), f("DEF-toe.L"), True), (f("DEF-foot.R"), f("DEF-toe.R"), True),
            (f("DEF-forearm.L.002"), f("DEF-palm.02.L"), True), (f("DEF-forearm.R.002"), f("DEF-palm.02.R"), True)]


def loco_segments(S, plan, meta):
    """'LOCO:Walk' / 'LOCO:Run' segments → the native clip time-warped to the root speed (diagonal gait kept, the
    stride matches the ground covered; the walk keeps stepping until three feet are down when the root stops)."""
    from anim import loco as LO
    D, N = S.D, S.N
    feet = dragon_feet(D.skel)
    segs = sorted(plan["seg"], key=lambda sg: sg["start"])
    gaits, log = {}, []
    for i, sg in enumerate(segs):
        if not str(sg["clip"]).startswith("LOCO"):
            continue
        clip = sg["clip"].split(":", 1)[1]
        a = int(sg["start"])
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        b = min(N, (nxt["start"] + nxt.get("xf", 10)) if nxt else N) + 2
        b = min(b, N)
        fr = np.arange(a, b + 1)
        nat = D.native[clip]
        if clip not in gaits:
            frames = list(range(nat["f0"], nat["f1"] + 1))
            Bm = D.skel.basis_from_action(nat["action"], frames)
            gaits[clip] = LO.gait(D.skel, D.skel.fk(Bm), feet)
        gt = gaits[clip]
        vch = LO.root_vel_char(D.base_pos[fr - 1], D.base_yaw[fr - 1])
        ph = LO.warp_phases(vch, gt, settle_need=3) + nat["f0"]
        name = f"loco{i:02d}_{clip}"
        act = LO.warp_action(f"D_{name}", D.arm, nat["action"], ph - nat["f0"], nat["n"], f0=nat["f0"])
        meta[name] = dict(action=act, n=len(fr))
        sg["clip"], sg["loop"], sg["scale"] = name, False, 1.0
        log.append(dict(seg=name, frames=[a, b], v_clip_mps=round(gt["v"] * 30, 2),
                        v_root_mps=round(float(np.linalg.norm(vch, axis=1).mean() * 30), 2),
                        cycles=round(float((ph[-1] - ph[0]) / (gt["n"] - 1)), 2)))
    S.R.add("dragon.loco", log)
