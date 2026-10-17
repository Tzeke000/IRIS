"""s20 — knight: UAL clip library → NLA (BASE_A/B crossfades, PARTS overlay, HANDKEY COMBINE), root path on the
ground (stairs smoothed), sword draw/sheathe swaps. Also computes the dragon's planned XY (both yaws need both paths)."""
import os
import re

import numpy as np

from anim import common as C
from anim import knight_clips as KC
from anim import posekit as PK
from anim import staging as ST
from anim import timeline as TL

PARTS = {"arms_head": ("Shoulder", "Arm", "ForeArm", "Hand", "Neck", "Head")}


def groups(sk):
    def g(n):
        b = sk.find("mixamorig:" + n)
        assert b, n
        return b
    G = {"spine": PK.chain_group([g("Spine"), g("Spine1"), g("Spine2")]),
         "neck": [(g("Neck"), 0.5), (g("Head"), 0.5)]}
    for side, Sd in (("R", "Right"), ("L", "Left")):
        G["arm" + side] = [(g(Sd + "Arm"), 1.0)]
        G["forearm" + side] = [(g(Sd + "ForeArm"), 1.0)]
        G["hand" + side] = [(g(Sd + "Hand"), 1.0)]
        G["fingers" + side] = [(g(f"{Sd}Hand{f}{i}"), 1.0) for f in ("Index", "Middle", "Ring") for i in (1, 2, 3)]
    return G


def swing_peaks(spd, k=3, rel=0.6, sep=6):
    """first k blade-speed peaks (time order) above rel·max — the clip's own hits"""
    out = []
    for i in range(1, len(spd) - 1):
        if spd[i] >= spd[i - 1] and spd[i] >= spd[i + 1] and spd[i] >= rel * spd.max():
            if not out or i - out[-1] >= sep:
                out.append(i)
    return out[:k]


def top_peaks(spd, k=3, sep=8):
    cand = [i for i in range(1, len(spd) - 1) if spd[i] >= spd[i - 1] and spd[i] >= spd[i + 1]]
    cand.sort(key=lambda i: -spd[i])
    keep = []
    for i in cand:
        if all(abs(i - j) >= sep for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return sorted(keep)


def part_action(K, clip, part):
    from bpy_extras import anim_utils
    src = K.clips[clip]["action"]
    cb0 = anim_utils.action_get_channelbag_for_slot(src, src.slots[0])
    keys = PARTS[part]
    act, slot, cb = C.new_action(f"K_{clip}__{part}", K.arm)
    for fc in cb0.fcurves:
        bone = fc.data_path.split('"')[1]
        short = re.sub(r"_\d+$", "", bone.split(":")[-1])
        if short in ("Neck", "Head") or any(short.startswith(sd + k) for sd in ("Left", "Right")
                                            for k in ("Shoulder", "Arm", "ForeArm", "Hand")):
            n = len(fc.keyframe_points)
            co = np.empty(2 * n)
            fc.keyframe_points.foreach_get("co", co)
            C.write_fc(cb, fc.data_path, fc.array_index, co[0::2], co[1::2], group=bone)
    return act


def run(S):
    KC.build(S)
    KC.debug_dump(S, os.path.join(S.debug, "knight_sticks.json"))
    K, E, N = S.K, S.E, S.N
    plan = ST.knight_plan(E)
    K.plan = plan
    meta = {c: dict(action=d["action"], n=d["n"]) for c, d in K.clips.items()}

    def resolve(clip, what):
        spd = K.clips[clip]["speed"]
        if what in ("takeoff", "land"):          # first frame both feet leave / last frame before both are down
            from anim import loco as LO
            g = LO.gait(K.skel, K.clips[clip]["P"], knight_feet(K.skel), thr_frac=0.12)
            up = ~g["down"].any(axis=1)
            idx = np.nonzero(up)[0]
            if not len(idx):
                return 0
            return int(idx[0]) if what == "takeoff" else int(idx[-1]) + 1
        if what == "hitmax":
            return int(np.argmax(spd))
        pk = swing_peaks(spd, 3, rel=0.38)
        return pk[int(what[3:])]
    K.resolve = resolve
    # the combo ends 6 frames after its 3rd hit (the clip has a 4th swing we don't want touching the leg)
    ci = next(i for i, sg in enumerate(plan["seg"]) if sg["clip"] == "Sword_Regular_Combo")
    sg = plan["seg"][ci]
    pk = swing_peaks(K.clips["Sword_Regular_Combo"]["speed"], 3, rel=0.38)
    plan["seg"][ci + 1]["start"] = int(round(E["combo"].f + (pk[2] - pk[0]) * sg["scale"])) + 6
    S.R.add("knight.combo_hits_local", pk)
    # ---- root path FIRST (v2): locomotion clips are time-warped to it
    K.xy = TL.path_xy(plan["root"], N)
    S.D.plan = ST.dragon_plan(E)
    S.D.xy = TL.path_xy(S.D.plan["root"], N)
    yaw, ykeys = TL.resolve_yaw(plan["yaw"], K.xy, S.D.xy, N)
    z, g = TL.height(plan["dz"], K.xy, S.ground, N, smooth=5.0)
    K.base_pos = np.c_[K.xy, z]
    K.base_yaw = yaw
    loco_segments(S, plan, meta)
    K.placed = TL.place_base(K.arm, plan["seg"], meta, N, S.R, "K_", resolve)
    for p in plan["parts"]:
        act = part_action(K, p["clip"], p["part"])
        n = K.clips[p["clip"]]["n"]
        st = C.put(K.arm, "K_PARTS", act, p["start"], scale=p.get("scale", 1.0))
        st.blend_in = st.blend_out = float(p.get("xf", 6))
    # ---- hand keys
    K.kit = PK.Kit(K.skel, K.W0, groups(K.skel))
    items = []
    for mv in plan["hk"]:
        n = int(mv["keys"][-1][0]) + 1
        gs = sorted({g for k in mv["keys"] for g in k[1]})
        lay = K.kit.layers(PK.move_angles(mv["keys"], gs, n))
        act = PK.write_layer_action(f"K_hk_{mv['name']}", K.arm, lay, n)
        items.append((mv["start"], mv["start"] + n - 1, act))
    for ti, s, e, act in TL.pack_tracks(items):
        C.put(K.arm, f"K_HANDKEY_{ti}", act, s, blend="COMBINE")
    S.R.add("knight.strips", sum(len(t.strips) for t in K.arm.animation_data.nla_tracks))

    K.off = np.zeros((N, 3))
    K.yoff = np.zeros(N)
    K.ground_z = g
    K.teleports = plan["teleports"]
    TL.bake_root(K.root, K.base_pos, K.base_yaw, N, K.teleports)
    # ---- sword: sheathed until the draw, in hand until the outro walk, sheathed again
    sk = plan["sword_keys"]                  # (frame, in_hand)
    C.key_visibility(K.sword, [(fr, not inh) for fr, inh in sk])
    C.key_visibility(K.sheathed, [(fr, inh) for fr, inh in sk])
    S.R.add("knight.path_len_m", round(float(np.linalg.norm(np.diff(K.base_pos, axis=0), axis=1).sum()), 1))
    sp = np.linalg.norm(np.diff(K.base_pos[:, :2], axis=0), axis=1) * C.FPS
    tele = set(K.teleports)
    sp_ok = [v for i, v in enumerate(sp) if (i + 2) not in tele]
    S.R.add("knight.max_speed_mps", round(float(max(sp_ok)), 2))


# ------------------------------------------------------------------------------------------------ v2 locomotion
LOWER = ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot",
         "RightToeBase")


def knight_feet(sk):
    f = lambda n: sk.find("mixamorig:" + n)
    return [(f("LeftFoot"), f("LeftToeBase"), False), (f("RightFoot"), f("RightToeBase"), False)]


def loco_segments(S, plan, meta):
    """every base segment whose clip is 'LOCO' / 'LOCO:<clip>' becomes its own time-warped action: the clip (auto =
    chosen from the root's travel direction in the character frame) plays exactly as fast as the root travels;
    strafes / back-steps take the legs+hips from the warrior donor and the upper body from `upper` (Sword_Idle with
    the sword drawn, Idle_Loop without)."""
    from anim import loco as LO
    K, N = S.K, S.N
    feet = knight_feet(K.skel)
    gaits = {}
    lower = {K.skel.find("mixamorig:" + b) for b in LOWER}
    segs = sorted(plan["seg"], key=lambda sg: sg["start"])
    log = []
    for i, sg in enumerate(segs):
        if not str(sg["clip"]).startswith("LOCO"):
            continue
        a = int(sg["start"])
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        b = min(N, (nxt["start"] + nxt.get("xf", 6)) if nxt else N) + 2
        b = min(b, N)
        fr = np.arange(a, b + 1)
        vch = LO.root_vel_char(K.base_pos[fr - 1], K.base_yaw[fr - 1])
        mov = np.linalg.norm(vch, axis=1) > 0.15 / 30
        clip = sg["clip"].split(":", 1)[1] if ":" in sg["clip"] else LO.choose(vch[mov] if mov.any() else vch, True, {})
        if clip == "SHUFFLE":
            stance = sg.get("upper") or ("Sword_Idle" if plan["sword_drawn"](a) else "Idle_Loop")
            name = f"loco{i:02d}_SHUFFLE"
            nu = K.clips[stance]["n"]
            act = LO.warp_action(f"K_{name}", K.arm, K.clips[stance]["action"], np.arange(len(fr), dtype=np.float64), nu)
            meta[name] = dict(action=act, n=len(fr))
            sg["clip"], sg["loop"], sg["scale"] = name, False, 1.0
            log.append(dict(seg=name, frames=[int(a), int(b)], stance=stance,
                            v_root_mps=round(float(np.linalg.norm(vch, axis=1).mean() * 30), 2)))
            continue
        if clip not in gaits:
            gaits[clip] = LO.gait(K.skel, K.clips[clip]["P"], feet)
        gt = gaits[clip]
        ph = LO.warp_phases(vch, gt, phase0=float(sg.get("phase0", 0.0)), settle_need=2)
        up = sg.get("upper") or ("Sword_Idle" if plan["sword_drawn"](a) else ("Idle_Loop" if clip.startswith("W_") else None))
        upper = None
        if up:
            nu = K.clips[up]["n"]
            upper = (K.clips[up]["action"], np.arange(len(fr), dtype=np.float64), nu)
        name = f"loco{i:02d}_{clip}"
        act = LO.warp_action(f"K_{name}", K.arm, K.clips[clip]["action"], ph, K.clips[clip]["n"],
                             lower=lower if upper else None, upper=upper)
        meta[name] = dict(action=act, n=len(fr))
        sg["clip"] = name
        sg["loop"] = False
        sg["scale"] = 1.0
        log.append(dict(seg=name, frames=[int(a), int(b)], v_clip_mps=round(gt["v"] * 30, 2),
                        v_root_mps=round(float(np.linalg.norm(vch, axis=1).mean() * 30), 2),
                        cycles=round(float((ph[-1] - ph[0]) / (gt["n"] - 1)), 2), upper=up))
    S.R.add("knight.loco", log)
