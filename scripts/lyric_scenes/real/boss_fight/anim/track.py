"""Evaluate the rigs (frame_set) and read world-space key points — the shared 'where is everything' pass used by the
contact solver, FX, cameras and the audit."""
import numpy as np


def _bone_pt(arm_ev, mw, bone, where):
    pb = arm_ev.pose.bones[bone]
    M = mw @ pb.matrix
    if where == "head":
        return np.array(M.translation)
    ln = arm_ev.data.bones[bone].length
    t = 1.0 if where == "tail" else float(where)
    from mathutils import Vector
    return np.array(M @ Vector((0, ln * t, 0)))


def knight_points(S):
    K = S.K
    f = K.skel.find
    return {"hips": (f("mixamorig:Hips"), "head"), "chest": (f("mixamorig:Spine2"), "head"),
            "head": (f("mixamorig:Head"), "head"), "rhand": (f("mixamorig:RightHand"), "head"),
            "lhand": (f("mixamorig:LeftHand"), "head"), "lfoot": (f("mixamorig:LeftFoot"), "head"),
            "rfoot": (f("mixamorig:RightFoot"), "head")}


def dragon_points(S):
    ch = S.D.chains
    return {"hip": (ch["spine"][0], "head"), "belly": (ch["spine"][2], "head"), "chest": (ch["spine"][5], "head"),
            "neck": (ch["neck"][1], "head"), "head": (ch["neck"][4], "head"), "jaw": (ch["jaw"][2], "tail"),
            "snout": (S.D.skel.find("DEF-Teeth_Top"), "tail"), "tail_mid": (ch["tail"][8], "head"),
            "tail_tip": (ch["tail"][16], "tail"), "palmL": (ch["armL"][6], "tail"), "palmR": (ch["armR"][6], "tail"),
            "forearmL": (ch["armL"][4], "head"), "forearmR": (ch["armR"][4], "head"),
            "wingL": (ch["wingL"][6], "tail"), "wingR": (ch["wingR"][6], "tail"), "back": (ch["spine"][3], "head"),
            "footL": (ch["legL"][7], "tail"), "footR": (ch["legR"][7], "tail"),
            "eyeL": (S.D.skel.find("DEF-eye.L"), "head")}


def sample(S, frames):
    """frame_set each frame; returns {'K.<pt>': (n,3), 'D.<pt>': (n,3), 'K.tip','K.grip'}"""
    import bpy
    sc = S.sc
    kp, dp = knight_points(S), dragon_points(S)
    out = {f"K.{k}": [] for k in kp}
    out.update({f"D.{k}": [] for k in dp})
    out["K.tip"], out["K.grip"], out["K.root"], out["D.root"] = [], [], [], []
    tip_y = float(S.R.get("sword.tip_local_y"))
    from mathutils import Vector
    for fr in frames:
        sc.frame_set(int(fr))
        dg = bpy.context.evaluated_depsgraph_get()
        for arm, pts, pre in ((S.K.arm, kp, "K"), (S.D.arm, dp, "D")):
            ev = arm.evaluated_get(dg)
            mw = ev.matrix_world
            for k, (b, w) in pts.items():
                out[f"{pre}.{k}"].append(_bone_pt(ev, mw, b, w))
        sw = S.K.sword.evaluated_get(dg).matrix_world
        out["K.tip"].append(np.array(sw @ Vector((0, tip_y, 0))))
        out["K.grip"].append(np.array(sw.translation))
        out["K.root"].append(np.array(S.K.root.evaluated_get(dg).matrix_world.translation))
        out["D.root"].append(np.array(S.D.root.evaluated_get(dg).matrix_world.translation))
    return {k: np.array(v) for k, v in out.items()}


def at(S, fr):
    d = sample(S, [fr])
    return {k: v[0] for k, v in d.items()}
