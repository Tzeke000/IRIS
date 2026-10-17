"""Quick look-dev/debug stills of the fight blend (WORKBENCH, no saving):
    blender -b fight.blend --python debug_shots.py -- OUT_DIR SPEC [SPEC ...]
SPEC = frame:target:dist:azimuth_deg:elev_deg[:lens]   target = K | Kfeet | Khands | D | Dhead | Dfeet | x,y,z
azimuth is measured from the target character's FACING (0 = in front, 90 = its left side)."""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def bone_w(arm, short, tail=False):
    import re
    for b in arm.pose.bones:
        if re.sub(r"_\d+$", "", b.name) in (short, "mixamorig:" + short, "DEF-" + short):
            M = arm.matrix_world @ b.matrix
            return np.array(M @ Vector((0, b.length if tail else 0, 0)))
    raise KeyError(short)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    out = argv[0]
    os.makedirs(out, exist_ok=True)
    sc = bpy.context.scene
    if sc.rigidbody_world is not None:
        sc.rigidbody_world.enabled = False
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 480, 360, 100
    sc.display.shading.show_cavity = False
    sc.display.shading.show_shadows = False
    sc.display.render_aa = "OFF"
    for cn in ("FX_haze", "V3_EXT_FOG", "SIZE_PROOF", "ANIM_PROXIES"):
        c = bpy.data.collections.get(cn)
        if c is not None:
            c.hide_render = True
    cam_d = bpy.data.cameras.new("DBG")
    cam = bpy.data.objects.new("DBG", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    karm = bpy.data.objects["PX_Knight"].find_armature()
    darm = bpy.data.objects["PX_Dragon"].find_armature()
    kroot, droot = bpy.data.objects["K_ROOT"], bpy.data.objects["D_ROOT"]
    for spec in argv[1:]:
        p = spec.split(":")
        fr, tgt, dist, az, el = int(p[0]), p[1], float(p[2]), float(p[3]), float(p[4])
        lens = float(p[5]) if len(p) > 5 else 35.0
        sc.frame_set(fr)
        root = kroot if tgt.startswith("K") else droot
        yaw = root.matrix_world.to_euler().z
        if tgt == "K":
            c = bone_w(karm, "Spine1")
        elif tgt == "Kfeet":
            c = 0.5 * (bone_w(karm, "LeftFoot") + bone_w(karm, "RightFoot")) + np.array([0, 0, 0.25])
        elif tgt == "Khands":
            c = 0.5 * (bone_w(karm, "LeftHand") + bone_w(karm, "RightHand"))
        elif tgt == "D":
            c = bone_w(darm, "Spine.003")
        elif tgt == "Deye":
            c = bone_w(darm, "eye.L")
        elif tgt == "Dhead":
            c = bone_w(darm, "neck.004")
        elif tgt == "Dfeet":
            c = np.array(droot.matrix_world.translation) + np.array([0, 0, 0.8])
        else:
            c = np.array([float(x) for x in tgt.split(",")])
            yaw = 0.0
        a = yaw + math.radians(az)
        fwd = np.array([-math.sin(a), math.cos(a), 0.0])
        e = math.radians(el)
        pos = c + dist * (fwd * math.cos(e) + np.array([0, 0, math.sin(e)]))
        cam.location = Vector(pos)
        d = Vector(c) - cam.location
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cam_d.lens = lens
        sc.render.filepath = os.path.join(out, f"dbg_{fr:04d}_{tgt}_{int(az)}.png")
        bpy.ops.render.render(write_still=True)
        print("[dbg]", sc.render.filepath)


main()
