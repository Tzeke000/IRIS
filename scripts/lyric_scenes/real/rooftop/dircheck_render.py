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
