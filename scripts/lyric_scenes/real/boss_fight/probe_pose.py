"""Measure posed (evaluated) bboxes of the dragon per action and the knight. -- OUT.json"""
import bpy, sys, json
import numpy as np
out = sys.argv[sys.argv.index("--") + 1:][0]
A = "/data/iris_assets/medieval/sketchfab/"
res = {}
def pts_of(objs):
    dg = bpy.context.evaluated_depsgraph_get(); P = []
    for ob in objs:
        ev = ob.evaluated_get(dg); me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        M = np.array(ev.matrix_world); P.append(co @ M[:3, :3].T + M[:3, 3]); ev.to_mesh_clear()
    return np.concatenate(P)
# dragon
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=A + "dragon_european_regina/model.glb")
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
meshes = [o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("Icosphere")]
print("dragon active action", arm.animation_data.action.name if arm.animation_data and arm.animation_data.action else None)
sc = bpy.context.scene
for ac in bpy.data.actions:
    arm.animation_data.action = ac
    try:
        arm.animation_data.action_slot = ac.slots[0]
    except Exception as e: print("slot", e)
    f0, f1 = map(int, ac.frame_range); best = None; rows = []
    for f in range(f0, f1 + 1, max(1, (f1 - f0) // 8)):
        sc.frame_set(f); P = pts_of(meshes)
        lo, hi = P.min(0), P.max(0); rows.append(dict(f=f, lo=lo.round(3).tolist(), hi=hi.round(3).tolist(), dims=(hi - lo).round(3).tolist()))
    res["dragon:" + ac.name] = rows
    print("DRAGON", ac.name, rows[0]["dims"], max(r["dims"][0] for r in rows), flush=True)
# head/nose/tail bone positions in Idle Stand
for ac in bpy.data.actions:
    if ac.name == "Idle Stand":
        arm.animation_data.action = ac; arm.animation_data.action_slot = ac.slots[0]
sc.frame_set(1)
bones = {b.name: (arm.matrix_world @ b.head).to_tuple(3) for b in arm.pose.bones}
res["dragon_bones_idle"] = bones
# knight
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=A + "knight_rx_sculpt/model.glb")
sc = bpy.context.scene
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
meshes = [o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("Icosphere")]
for f in (1, 40, 80):
    sc.frame_set(f); per = {}
    for m in meshes:
        P = pts_of([m]); per[m.name] = [P.min(0).round(3).tolist(), P.max(0).round(3).tolist()]
    P = pts_of(meshes); res[f"knight_f{f}"] = dict(lo=P.min(0).round(3).tolist(), hi=P.max(0).round(3).tolist(), per=per)
    print("KNIGHT", f, P.min(0).round(2), P.max(0).round(2), flush=True)
res["knight_bones"] = {b.name: (arm.matrix_world @ b.head).to_tuple(3) for b in arm.pose.bones}
json.dump(res, open(out, "w"), indent=1)
