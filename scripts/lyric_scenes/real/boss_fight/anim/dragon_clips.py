"""Dragon clip library.
* native clips of the European Dragon (Fly / Idle Sit / Idle Stand / Run / Walk; CC-BY ReginaCachoa) are used as-is
  on the BASE tracks (they key all 168 bones).
* T-rex donor clips (LasquetiSpice, CC-BY): roar / bite / attack_tail retargeted chain-to-chain (spine, neck+head,
  jaw, tail 8→17 by arc length) as ADDITIVE COMBINE layers (delta from the clip's own first frame), so they ride on
  whatever the base is doing.
"""
import bpy
import numpy as np

from anim import common as C
from anim import dragon_debug as DD
from anim.retarget import Skel, facing_yaw_fix, retarget_chains

NATIVE = ["Idle Stand", "Idle Sit", "Walk", "Run", "Fly"]
TREX_CLIPS = {"roar": ("spine", "neck", "jaw"), "bite": ("spine", "neck", "jaw"), "attack_tail": ("spine", "tail")}


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
    D = S.D
    sk = D.skel
    D.chains = DD.chains(sk)
    D.native = {}
    for nm in NATIVE:
        act = bpy.data.actions[nm]
        f0, f1 = act.frame_range
        D.native[nm] = dict(action=act, f0=int(round(f0)), f1=int(round(f1)), n=int(round(f1 - f0)) + 1)
    # ---- T-rex donors
    arm, objs, acts = _import(C.TREX_GLB)
    bpy.context.view_layer.update()
    s0 = Skel(arm)
    W = s0.world(s0.fk(s0.ident(1)))[0]
    def f(b):          # T-rex names look like "bn_Spine1.5_5"
        return next(n for n in s0.names if n.split(".")[0] == b)
    fix = facing_yaw_fix(W[s0.idx[f("bn_Tail01")], :3, 3], W[s0.idx[f("bn_Head")], :3, 3])
    src = Skel(arm, fix)
    tch = D.chains
    chains = {"spine": ([f("bn_Spine"), f("bn_Spine1"), f("bn_Spine2")], tch["spine"], 0.5),
              "neck": ([f("bn_Neck"), f("bn_Neck1"), f("bn_Neck2"), f("bn_Head")], tch["neck"], 1.0),
              "jaw": ([f("bn_Jaw")], tch["jaw"][:1], 1.0),
              "tail": ([f(f"bn_Tail0{i}") for i in range(1, 9)], tch["tail"], 1.0)}
    D.donors = {}
    for clip, use in TREX_CLIPS.items():
        gain = 1.0
        act = acts[clip]
        f0, f1 = int(round(act.frame_range[0])), int(round(act.frame_range[1]))
        frames = list(range(f0, f1 + 1))
        Bs = src.basis_from_action(act, frames)
        layer = retarget_chains(src, Bs, Bs[0], sk, [chains[u] for u in use], D.ref)
        out, slot, cb = C.new_action(f"D_trex_{clip}", D.arm)
        fr = np.arange(len(frames), dtype=np.float64)
        for b, q in layer.items():
            C.write_bone_quats(cb, b, fr, q)
        D.donors[clip] = dict(action=out, n=len(frames), layer=layer)
    for o in objs:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    for a in acts.values():
        if a.name in bpy.data.actions:
            bpy.data.actions.remove(a)
    S.R.add("dragon.donors", {k: v["n"] for k, v in D.donors.items()})


def native_pose(S, clip, frame):
    D = S.D
    B = D.skel.basis_from_action(D.native[clip]["action"], [frame])
    return D.skel.world(D.skel.fk(B))[0]


def layered_pose(S, base_basis, layer, k):
    """base basis (B,4,4) with COMBINE layer quats at clip frame k → char-frame matrices."""
    D = S.D
    Bm = base_basis.copy()
    for b, q in layer.items():
        j = D.skel.idx[b]
        Bm[j, :3, :3] = Bm[j, :3, :3] @ C.q_to_mat(q[k])
    return D.skel.world(D.skel.fk(Bm[None]))[0]


def debug(S, path):
    D = S.D
    poses = {"ref Idle Stand f0": D.W0}
    for nm in NATIVE:
        nat = D.native[nm]
        for k in np.linspace(nat["f0"], nat["f1"], 3).astype(int)[:2 if nm != "Fly" else 3]:
            poses[f"{nm} f{k}"] = native_pose(S, nm, int(k))
    for clip, d in D.donors.items():
        for k in np.linspace(0, d["n"] - 1, 4).astype(int)[1:]:
            poses[f"trex {clip} k{k}"] = layered_pose(S, D.ref, d["layer"], int(k))
    DD.dump(path, D.skel, poses)
