"""Knight clip library: Quaternius UAL1/UAL2 (CC0) actions retargeted onto the rx_sculpt knight (Mixamo rig, rest =
bind pose after s10) → one action per clip named K_<clip>, plus per-clip metadata (length, hand/blade-tip paths in the
character frame, blade-speed peaks = the clip's own hit frames)."""
import math

import bpy
import numpy as np

from anim import common as C
from anim.retarget import Skel, facing_yaw_fix, retarget_human

UE = ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "Head"]
MIX = dict(pelvis="Hips", spine_01="Spine", spine_02="Spine1", spine_03="Spine2", neck_01="Neck", Head="Head")
LIMB = dict(clavicle="Shoulder", upperarm="Arm", lowerarm="ForeArm", hand="Hand", thigh="UpLeg", calf="Leg",
            foot="Foot", ball="ToeBase")
FINGERS = ("thumb", "index", "middle", "ring")

CLIPS = {
    "UAL1": ["Idle_Loop", "Walk_Loop", "Jog_Fwd_Loop", "Sprint_Loop", "Sword_Idle", "Sword_Attack",
             "Spell_Simple_Enter", "Spell_Simple_Idle_Loop", "Spell_Simple_Shoot", "Spell_Simple_Exit", "Roll",
             "Jump_Start", "Jump_Loop", "Jump_Land", "Hit_Chest", "Hit_Head", "Crouch_Idle_Loop", "Push_Loop",
             "Interact", "PickUp_Table", "Sitting_Idle_Loop", "Sitting_Exit", "Punch_Cross", "Death01",
             "Idle_Torch_Loop", "Pistol_Aim_Neutral"],
    "UAL2": ["Sword_Block", "Sword_Dash", "Sword_Regular_A", "Sword_Regular_B", "Sword_Regular_C",
             "Sword_Regular_Combo", "Sword_Heavy_Combo", "Consume", "Hit_Knockback", "OverhandThrow",
             "NinjaJump_Start", "NinjaJump_Land", "LayToIdle", "Shield_Dash", "Slide_Start", "Slide_Loop",
             "Slide_Exit", "Idle_Shield_Loop", "Chest_Open"],
}


def _pairs(src, tgt):
    m = {}
    for s, t in MIX.items():
        m[s] = "mixamorig:" + t
    for side, Side in (("l", "Left"), ("r", "Right")):
        for s, t in LIMB.items():
            m[f"{s}_{side}"] = f"mixamorig:{Side}{t}"
        for f in FINGERS:
            for i in (1, 2, 3):
                m[f"{f}_0{i}_{side}"] = f"mixamorig:{Side}Hand{f.capitalize()}{i}"
    pairs = []
    for s, t in m.items():
        tn = tgt.find(t)
        if s in src.idx and tn is not None:
            pairs.append((s, tn))
    child = {}
    for side in "lr":
        child.update({f"clavicle_{side}": f"upperarm_{side}", f"upperarm_{side}": f"lowerarm_{side}",
                      f"lowerarm_{side}": f"hand_{side}", f"hand_{side}": f"middle_01_{side}",
                      f"thigh_{side}": f"calf_{side}", f"calf_{side}": f"foot_{side}", f"foot_{side}": f"ball_{side}"})
        for f in FINGERS:
            for i in (1, 2):
                child[f"{f}_0{i}_{side}"] = f"{f}_0{i + 1}_{side}"
    return pairs, child


def _import(path):
    b_obj, b_act = set(bpy.data.objects), set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in b_obj]
    acts = {a.name: a for a in bpy.data.actions if a not in b_act}
    arm = next(o for o in new if o.type == "ARMATURE")
    if arm.animation_data:
        arm.animation_data.action = None
    return arm, new, acts


def build(S):
    K = S.K
    tgt = K.skel
    # sword offset relative to the hand bone (character frame, rest)
    W0 = tgt.world(tgt.fk(tgt.ident(1)))[0]
    hb = tgt.idx[K.hand_bone]
    Mroot_inv = np.linalg.inv(np.array(K.root.matrix_world))
    Msw = Mroot_inv @ np.array(K.sword.matrix_world)
    K.sword_off = np.linalg.inv(W0[hb]) @ Msw
    tip_y = float(S.R.get("sword.tip_local_y"))
    K.clips = {}
    for which, path in (("UAL1", C.UAL1), ("UAL2", C.UAL2)):
        arm, objs, acts = _import(path)
        bpy.context.view_layer.update()
        sk0 = Skel(arm)
        Wr = sk0.world(sk0.fk(sk0.ident(1)))[0]
        fix = facing_yaw_fix(Wr[sk0.idx["foot_l"], :3, 3], Wr[sk0.idx["ball_l"], :3, 3])
        src = Skel(arm, fix)
        pairs, child = _pairs(src, tgt)
        root_pair = ("pelvis", tgt.find("mixamorig:Hips"))
        for clip in CLIPS[which]:
            act = acts.get(clip)
            if act is None:
                S.R.log(f"WARN UAL clip {clip} missing in {which}")
                continue
            f0, f1 = int(round(act.frame_range[0])), int(round(act.frame_range[1]))
            frames = list(range(f0, f1 + 1))
            Bs = src.basis_from_action(act, frames)
            Bt, info = retarget_human(src, Bs, tgt, pairs, child, root_pair)
            out, slot, cb = C.new_action(f"K_{clip}", K.arm)
            fr = np.arange(len(frames), dtype=np.float64)
            for _s, t in pairs:
                j = tgt.idx[t]
                C.write_bone_quats(cb, t, fr, C.mat_to_q(Bt[:, j, :3, :3]))
            jr = tgt.idx[root_pair[1]]
            C.write_bone_loc(cb, root_pair[1], fr, Bt[:, jr, :3, 3])
            P = info["P"]
            Wt = tgt.world(P)
            hand_M = Wt[:, hb]
            swM = hand_M @ K.sword_off
            tip = (swM @ np.array([0, tip_y, 0, 1.0]))[:, :3]
            grip = swM[:, :3, 3]
            spd = np.r_[0, np.linalg.norm(np.diff(tip, axis=0), axis=1)]
            peaks = [int(i) for i in range(1, len(spd) - 1) if spd[i] >= spd[i - 1] and spd[i] >= spd[i + 1]
                     and spd[i] > 0.12]
            K.clips[clip] = dict(action=out, n=len(frames), travel=info["travel"], tip=tip, grip=grip, speed=spd,
                                 peaks=peaks, lhand=Wt[:, tgt.idx[tgt.find("mixamorig:LeftHand")], :3, 3],
                                 rhand=hand_M[:, :3, 3], P=P)
        for o in objs:
            if o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)
        for a in acts.values():
            if a.name in bpy.data.actions and not a.name.startswith("K_"):
                bpy.data.actions.remove(a)
    build_warrior(S, tip_y, hb)
    S.R.add("knight.combo_speed", [round(float(x), 2) for x in K.clips["Sword_Regular_Combo"]["speed"]])
    S.R.add("knight.clips", {k: dict(n=v["n"], peaks=v["peaks"], travel=[round(x, 2) for x in v["travel"]],
                                     vmax=round(float(v["speed"].max()), 3)) for k, v in K.clips.items()})


def debug_dump(S, path, clips=("Walk_Loop", "Sword_Idle", "Sword_Regular_Combo", "Spell_Simple_Shoot", "Consume",
                               "Sword_Block", "Roll", "OverhandThrow")):
    """stick-figure data (character frame) → JSON for an offline matplotlib check (no rendering)."""
    import json
    K = S.K
    tgt = K.skel
    names = [n for n in tgt.names if not any(k in n for k in ("Thumb", "Index", "Middle", "Ring", "End", "_rootJoint"))]
    par = {n: (tgt.names[tgt.parent[tgt.idx[n]]] if tgt.parent[tgt.idx[n]] >= 0 else None) for n in names}
    out = {"bones": names, "parent": par, "clips": {}}
    for c in clips:
        if c not in K.clips:
            continue
        d = K.clips[c]
        Wt = tgt.world(d["P"])
        ks = np.linspace(0, d["n"] - 1, 4).astype(int).tolist()
        out["clips"][c] = {"frames": ks, "pts": {str(k): {n: Wt[k, tgt.idx[n], :3, 3].round(3).tolist() for n in names}
                                                  for k in ks},
                           "tip": {str(k): d["tip"][k].round(3).tolist() for k in ks},
                           "grip": {str(k): d["grip"][k].round(3).tolist() for k in ks}}
    rest = tgt.world(tgt.fk(tgt.ident(1)))[0]
    out["rest"] = {n: rest[tgt.idx[n], :3, 3].round(3).tolist() for n in names}
    with open(path, "w") as fh:
        json.dump(out, fh)


# ---------------------------------------------------------------------------------------- Animated Warrior (CC-BY fikso)
# strafe / back-step / sheathe / unsheathe / jump donors. Its skin imports broken but the deform-bone rotations are
# good (research doc §3): world-delta retarget from Pelvis_06 with a custom map (no toes, no clavicles in the donor).
WARRIOR_GLB = C.MED + "r2_anim/warrior_28anim_fikso/model.glb"
WARRIOR_CLIPS = {"Strafe Left": "W_StrafeL", "Strafe Right": "W_StrafeR", "Strafe Backward": "W_StrafeB",
                 "Strafe Forward": "W_StrafeF", "Running Backwards Sheathed": "W_RunBack", "Sheath Weapon": "W_Sheath",
                 "Unsheath Weapon": "W_Unsheath", "Unsheathed Jump": "W_Jump", "Sheathed Jump": "W_JumpSheathed"}
W_MAP = {"Pelvis": "Hips", "Spine_01": "Spine1", "Spine_02": "Spine2", "Neck": "Neck", "Head": "Head",
         "Thign.L": "LeftUpLeg", "Calf.L": "LeftLeg", "Thign.R": "RightUpLeg", "Calf.R": "RightLeg",
         "UpperArm.L": "LeftArm", "LowerArm.L": "LeftForeArm", "Hand.L": "LeftHand",
         "UpperArm.R": "RightArm", "LowerArm.R": "RightForeArm", "Hand.R": "RightHand"}
W_CHILD = {"Pelvis": "Spine_01", "Spine_01": "Spine_02", "Spine_02": "Neck", "Neck": "Head",
           "Thign.L": "Calf.L", "Thign.R": "Calf.R", "UpperArm.L": "LowerArm.L", "LowerArm.L": "Hand.L",
           "UpperArm.R": "LowerArm.R", "LowerArm.R": "Hand.R"}


def build_warrior(S, tip_y, hb):
    import re
    K = S.K
    tgt = K.skel
    b_obj, b_act = set(bpy.data.objects), set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=WARRIOR_GLB)
    objs = [o for o in bpy.data.objects if o not in b_obj]
    acts = {a.name: a for a in bpy.data.actions if a not in b_act}
    arm = next(o for o in objs if o.type == "ARMATURE")
    if arm.animation_data:
        arm.animation_data.action = None
    bpy.context.view_layer.update()

    def wn(short):
        return next(b.name for b in arm.data.bones if re.sub(r"_\d+$", "", b.name) == short)
    # the donor's glTF REST is junk (bones metres away, upside down in world Z): build the character frame from a real
    # standing pose (Sheated-Idle f0) — up = feet→head, forward = heel→toe — and retarget relative to THAT pose
    s0 = Skel(arm)
    ref_act = acts["Sheated-Idle"]
    Bref = s0.basis_from_action(ref_act, [int(round(ref_act.frame_range[0]))])
    Wr = s0.world(s0.fk(Bref))[0]
    pt = lambda n: Wr[s0.idx[wn(n)], :3, 3]
    feet = 0.5 * (pt("Foot.L") + pt("Foot.R"))
    up = pt("Head") - feet
    up /= np.linalg.norm(up)
    ln = arm.data.bones[wn("Foot.L")].length
    toe = (Wr[s0.idx[wn("Foot.L")]] @ np.array([0, ln, 0, 1.0]))[:3]
    fw = toe - pt("Foot.L")
    fw -= up * np.dot(fw, up)
    fw /= np.linalg.norm(fw)
    rt = np.cross(fw, up)
    fix = np.eye(4)
    fix[0, :3], fix[1, :3], fix[2, :3] = rt, fw, up
    fix[:3, 3] = -fix[:3, :3] @ feet
    src = Skel(arm, fix)
    S.R.add("knight.warrior_frame", dict(up=up.round(3).tolist(), fwd=fw.round(3).tolist()))
    pairs = [(wn(s), tgt.find("mixamorig:" + d)) for s, d in W_MAP.items()]
    child = {wn(a): wn(b) for a, b in W_CHILD.items()}
    root_pair = (wn("Pelvis"), tgt.find("mixamorig:Hips"))
    got = []
    for clip, name in WARRIOR_CLIPS.items():
        act = acts.get(clip)
        if act is None:
            S.R.log(f"WARN warrior clip {clip} missing")
            continue
        f0, f1 = int(round(act.frame_range[0])), int(round(act.frame_range[1]))
        frames = list(range(f0, f1 + 1))
        Bs = src.basis_from_action(act, frames)
        # root translation: the donor's pelvis relative to ITS OWN median over the clip (its rest/idle translation is
        # junk), scaled by the leg-length ratio — the knight keeps his own hip height, the clip adds the bob/sway
        Ws = src.world(src.fk(Bs))
        pel = Ws[:, src.idx[wn("Pelvis")], :3, 3]
        legs_s = np.median(np.linalg.norm(Ws[:, src.idx[wn("Calf.L")], :3, 3] - Ws[:, src.idx[wn("Thign.L")], :3, 3], axis=1)
                           + np.linalg.norm(Ws[:, src.idx[wn("Foot.L")], :3, 3] - Ws[:, src.idx[wn("Calf.L")], :3, 3], axis=1))
        Wk = tgt.world(tgt.fk(tgt.ident(1)))[0]
        kp = lambda n: Wk[tgt.idx[tgt.find("mixamorig:" + n)], :3, 3]
        legs_t = np.linalg.norm(kp("LeftLeg") - kp("LeftUpLeg")) + np.linalg.norm(kp("LeftFoot") - kp("LeftLeg"))
        Bt, info = retarget_human(src, Bs, tgt, pairs, child, root_pair, src_ref_basis=Bref[0],
                                  root_ref=np.median(pel, axis=0), ratio=float(legs_t / max(legs_s, 1e-6)))
        out, slot, cb = C.new_action(f"K_{name}", K.arm)
        fr = np.arange(len(frames), dtype=np.float64)
        for _s, tb in pairs:
            j = tgt.idx[tb]
            C.write_bone_quats(cb, tb, fr, C.mat_to_q(Bt[:, j, :3, :3]))
        jr = tgt.idx[root_pair[1]]
        C.write_bone_loc(cb, root_pair[1], fr, Bt[:, jr, :3, 3])
        P = info["P"]
        Wt = tgt.world(P)
        hand_M = Wt[:, hb]
        swM = hand_M @ K.sword_off
        tip = (swM @ np.array([0, tip_y, 0, 1.0]))[:, :3]
        spd = np.r_[0, np.linalg.norm(np.diff(tip, axis=0), axis=1)]
        K.clips[name] = dict(action=out, n=len(frames), travel=info["travel"], tip=tip, grip=swM[:, :3, 3], speed=spd,
                             peaks=[], lhand=Wt[:, tgt.idx[tgt.find("mixamorig:LeftHand")], :3, 3],
                             rhand=hand_M[:, :3, 3], P=P, donor="warrior")
        got.append(name)
    for o in objs:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    for a in acts.values():
        if a.name in bpy.data.actions:
            bpy.data.actions.remove(a)
    S.R.add("knight.warrior_clips", got)
