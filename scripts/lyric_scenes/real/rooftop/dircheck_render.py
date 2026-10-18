"""Direction-check render: for every flying-car rig (…Rig with a …MarkPlusY child), park a temporary camera
beside it at a frame where it is visible and render it. The green +Y marker must sit AHEAD of the nose
(headlights / windshield side) and the car must travel toward it. Run on a --markers build:
    blender -b bemyone.blend --python dircheck_render.py -- OUTDIR"""
import os, sys
import bpy
from mathutils import Vector
out = sys.argv[sys.argv.index("--") + 1]
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 640, 360, 100
sc.eevee.taa_render_samples = 16
sc.render.use_motion_blur = False
cam = bpy.data.objects.new("DirCam", bpy.data.cameras.new("DirCam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens = 35
for rig in [o for o in bpy.data.objects if o.name.endswith("Rig") and any(c.name.endswith("MarkPlusY") for c in o.children)]:
    fc = next(f for f in rig.animation_data.action.layers[0].strips[0].channelbags[0].fcurves if f.data_path == "hide_render")
    frame = next(int(k.co[0]) for k in fc.keyframe_points if k.co[1] < 0.5) + 60
    sc.frame_set(frame)
    p = rig.matrix_world.translation.copy()
    fwd = (rig.matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()
    side = fwd.cross(Vector((0, 0, 1))).normalized()
    cam.location = p + side * 16 + Vector((0, 0, 5)) - fwd * 4
    cam.rotation_euler = (p - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.frame_set(frame)
    sc.render.filepath = os.path.join(out, f"{rig.name}_f{frame}.png")
    bpy.ops.render.render(write_still=True)
    sc.frame_set(frame + 3)
    p2 = rig.matrix_world.translation.copy()
    print(f"[dirrender] {rig.name} frame {frame}: moved {(p2 - p).length:.2f} m in 3 frames, dir·(+Y) = {(p2 - p).normalized().dot(fwd):+.3f}")

# v3: the GN sky-traffic bodies (SkyTraffic instances of the _skycar_kit). One instance per model: camera parked
# beside it, the green +Y marker (in the kit mesh on a --markers build) must lead, and the body must move toward it.


def sky_instances(frame):
    sc.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    return {di.persistent_id[0]: (di.object.original.name, di.matrix_world.copy()) for di in dg.object_instances
            if di.is_instance and di.parent and di.parent.original.name == "SkyTraffic"}


SKY_FRAME = 1021                                        # drop 1 crane: SkyTraffic is live here
if "SkyTraffic" in bpy.data.objects:
    a, b = sky_instances(SKY_FRAME), sky_instances(SKY_FRAME + 3)
    done = set()
    for pid, (onm, mw) in sorted(a.items()):
        if onm in done:
            continue
        done.add(onm)
        p = mw.translation.copy()
        fwd = (mw.to_3x3() @ Vector((0, 1, 0))).normalized()
        side = fwd.cross(Vector((0, 0, 1))).normalized()
        cam.location = p + side * 14 + Vector((0, 0, 4)) - fwd * 3
        cam.rotation_euler = (p + fwd * 1.5 - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.frame_set(SKY_FRAME)
        sc.render.filepath = os.path.join(out, f"Sky{pid}_{onm}_f{SKY_FRAME}.png")
        bpy.ops.render.render(write_still=True)
        mv = b[pid][1].translation - p
        print(f"[dirrender] SkyTraffic instance {pid} ({onm}) frame {SKY_FRAME}: moved {mv.length:.2f} m in 3 frames, "
              f"dir·(+Y) = {mv.normalized().dot(fwd):+.3f}")
