"""Tiny animated test scene for the render-runner tests (48 frames, 128x72, EEVEE 1 spp, two cameras).
    blender -b --factory-startup --python bpy_tiny_scene.py -- --out /ABS/tiny.blend [--nf 48]"""
import argparse
import sys

import bpy

a = argparse.ArgumentParser()
a.add_argument("--out", required=True)
a.add_argument("--nf", type=int, default=48)
a = a.parse_args(sys.argv[sys.argv.index("--") + 1:])
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 128, 72, 100
sc.render.fps = 30
sc.frame_start, sc.frame_end = 1, a.nf
sc.eevee.taa_render_samples = 1
sc["spp"] = [1] * a.nf                                  # per-frame prop path in frames.py
bpy.ops.mesh.primitive_cube_add(size=2)
cube = bpy.context.active_object
cube.name = "Mover"
mat = bpy.data.materials.new("emit")
nt = mat.node_tree
nt.nodes.clear()
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (1, 0.4, 0.1, 1)
em.inputs["Strength"].default_value = 4
nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
cube.data.materials.append(mat)
cube.location = (-6, 0, 0)
cube.keyframe_insert("location", frame=1)
cube.location = (6, 0, 0)
cube.keyframe_insert("location", frame=a.nf)
for name, loc, rot in (("CAM_front", (0, -16, 0), (1.5708, 0, 0)), ("CAM_top", (0, 0, 20), (0, 0, 0))):
    c = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    sc.collection.objects.link(c)
    c.location, c.rotation_euler = loc, rot
sc.camera = bpy.data.objects["CAM_front"]
bpy.data.objects["CAM_top"]["exposure_offset"] = 2.0     # camera custom prop → frames.py applies it
bpy.ops.wm.save_as_mainfile(filepath=a.out)
print("[tiny] saved", a.out)
