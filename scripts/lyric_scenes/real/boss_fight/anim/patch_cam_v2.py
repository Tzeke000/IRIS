"""Patch the saved fight_v2.blend camera for one shot without a rebuild (same formula as s80): the KIT-3 orbit
(frames kit_rocks..doors-1) stays south of the closed doors.  blender -b fight_v2.blend --python patch_cam_v2.py"""
import math
import sys
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0, "/home/iris/IRIS/scripts/lyric_scenes/real/boss_fight")
sys.path.insert(0, "/home/iris/IRIS/scripts/lyric_scenes")
from anim import common as C
from lib import bpyx as X
from anim.timeline import gsmooth
E = C.load_events()
f0, f1 = E["kit_rocks"].f, E["doors"].f - 1
sc = bpy.context.scene
cam = bpy.data.objects["CAM_animatic"]
karm = bpy.data.objects["PX_Knight"].find_armature()
bn = next(b.name for b in karm.pose.bones if "Spine2" in b.name)
frs = list(range(f0 - 20, f1 + 21))
KC = []
for fr in frs:
    sc.frame_set(fr)
    KC.append(np.array(karm.matrix_world @ karm.pose.bones[bn].head))
KC = np.array(KC)
KC = np.c_[gsmooth(KC[:, 0], 6.0), gsmooth(KC[:, 1], 6.0), gsmooth(KC[:, 2], 6.0)]
from bpy_extras import anim_utils
cb = anim_utils.animdata_get_channelbag_for_assigned_slot(cam.animation_data)
loc = {fc.array_index: fc for fc in cb.fcurves if fc.data_path == "location"}
rot = {fc.array_index: fc for fc in cb.fcurves if fc.data_path == "rotation_euler"}
prev = None
for fr in range(f0, f1 + 1):
    i = frs.index(fr)
    t = np.clip((fr - f0) / max(f1 + 1 - f0, 1), 0, 1)
    a = math.radians(-70 + 55 * t)
    c = KC[i]
    p = np.array([c[0] + 3.2 * math.cos(a), c[1] + 3.2 * math.sin(a), c[2] + 0.3])
    q = X.look_quat(Vector(p), Vector(c))
    e = q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")
    prev = e
    for k in range(3):
        for kp in loc[k].keyframe_points:
            if int(round(kp.co[0])) == fr:
                kp.co[1] = p[k]; kp.handle_left[1] = p[k]; kp.handle_right[1] = p[k]
        for kp in rot[k].keyframe_points:
            if int(round(kp.co[0])) == fr:
                kp.co[1] = e[k]; kp.handle_left[1] = e[k]; kp.handle_right[1] = e[k]
for fc in list(loc.values()) + list(rot.values()):
    fc.update()
bpy.ops.wm.save_mainfile()
print("PATCHED", f0, f1)
