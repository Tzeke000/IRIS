"""Patch the fire_rpg shot of the saved fight_v2.blend (camera was behind the open east door leaf)."""
import sys
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0, "/home/iris/IRIS/scripts/lyric_scenes/real/boss_fight")
sys.path.insert(0, "/home/iris/IRIS/scripts/lyric_scenes")
from anim import common as C
from lib import bpyx as X
E = C.load_events()
f0, f1 = E["fire_rpg"].f, bpy.context.scene.frame_end
sc = bpy.context.scene
cam = bpy.data.objects["CAM_animatic"]
darm = bpy.data.objects["PX_Dragon"].find_armature()
bn = next(b.name for b in darm.pose.bones if "neck.004" in b.name)
from bpy_extras import anim_utils
cb = anim_utils.animdata_get_channelbag_for_assigned_slot(cam.animation_data)
loc = {fc.array_index: fc for fc in cb.fcurves if fc.data_path == "location"}
rot = {fc.array_index: fc for fc in cb.fcurves if fc.data_path == "rotation_euler"}
cbd = anim_utils.animdata_get_channelbag_for_assigned_slot(cam.data.animation_data)
lens = next(fc for fc in cbd.fcurves if fc.data_path == "lens")
p = np.array([-3.5, 14.0, 10.8])
sc.frame_set(f0)
c = np.array(darm.matrix_world @ darm.pose.bones[bn].head)
e = X.look_quat(Vector(p), Vector(c)).to_euler("XYZ")
for fr in range(f0, f1 + 1):
    for k in range(3):
        for fc, v in ((loc[k], p[k]), (rot[k], e[k])):
            for kp in fc.keyframe_points:
                if int(round(kp.co[0])) == fr:
                    kp.co[1] = v; kp.handle_left[1] = v; kp.handle_right[1] = v
    for kp in lens.keyframe_points:
        if int(round(kp.co[0])) == fr:
            kp.co[1] = 26.0; kp.handle_left[1] = 26.0; kp.handle_right[1] = 26.0
for fc in list(loc.values()) + list(rot.values()) + [lens]:
    fc.update()
bpy.ops.wm.save_mainfile()
print("PATCHED", f0, f1)
