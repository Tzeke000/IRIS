"""World-edge verification (v2.1): haze OFF, render the highest/widest scene cameras + a top-down ortho of the whole
world (red background → any area with no ground shows RED), and report min-Z of every building-ish object.
    blender -b bemyone.blend --python verify_world.py -- OUTDIR TAG"""
import os, sys
import bpy
import numpy as np
from mathutils import Vector
out, tag = sys.argv[sys.argv.index("--") + 1:][:2]
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
sc.render.resolution_percentage = 50
sc.eevee.taa_render_samples = 8
sc.render.use_motion_blur = False
for o in bpy.data.objects:
    if o.name.startswith("HazeBox"):
        o.hide_render = True
        if o.animation_data:
            o.animation_data_clear()
dg = bpy.context.evaluated_depsgraph_get()
# --- min Z of building geometry (evaluated, world space)
rows = []
for o in bpy.data.objects:
    if o.type != "MESH" or o.hide_render or not any(k in o.name for k in ("City", "Hero", "Tallest", "HandBlocks", "HoloHost", "OSM", "Hills", "Suburb", "Island")):
        continue
    oe = o.evaluated_get(dg)
    me = oe.to_mesh()
    if len(me.vertices) == 0:
        oe.to_mesh_clear(); continue
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    M = np.array(o.matrix_world)
    w = co @ M[:3, :3].T + M[:3, 3]
    rows.append((o.name, float(w[:, 2].min()), float(w[:, 0].min()), float(w[:, 0].max()), float(w[:, 1].min()), float(w[:, 1].max())))
    oe.to_mesh_clear()
# floating test: every vertical building edge (verts grouped by x,y) — its LOWEST vertex vs the terrain under it.
# gap 0.5–15 m = floating (crown tiers sit 50+ m up on the tier below and are excluded by the 15 m cap)
import sys as _s
_s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import terrain as _ter
    TER = _ter.load("/data/iris_assets/bemyone/dem_cop30/window.npz")
except Exception as ex:
    TER = None
    print("[verify] no terrain module:", ex)
float_total, checked = 0, 0
for o in bpy.data.objects:
    if o.type != "MESH" or not any(k in o.name for k in ("CityOSM", "Suburbs", "HandBlocks", "Tallest", "HoloHost", "City")):
        continue
    co = np.empty(len(o.data.vertices) * 3, np.float32); o.data.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    M = np.array(o.matrix_world); w = co @ M[:3, :3].T + M[:3, 3]
    key = np.round(w[:, :2], 1)
    order = np.lexsort((w[:, 2], key[:, 1], key[:, 0]))
    k_sorted = key[order]
    first = np.ones(len(order), bool); first[1:] = np.any(k_sorted[1:] != k_sorted[:-1], axis=1)
    bot = w[order][first]
    ground = _ter.height_at(TER, bot[:, 0], bot[:, 1]) if TER is not None else np.zeros(len(bot))
    gap = bot[:, 2] - ground
    fl = (gap > 0.5) & (gap < 15.0)
    checked += len(bot); float_total += int(fl.sum())
    print(f"[verify]   {o.name}: {len(bot)} building edges, floating {int(fl.sum())}, worst gap {gap[gap < 15].max() if (gap < 15).any() else 0:.2f} m")
print(f"[verify] floating edges total: {float_total} / {checked}")
zs = [r[1] for r in rows]
print(f"[verify] {tag}: {len(rows)} building objects; min-Z range {min(zs):.2f} … {max(zs):.2f}")
for r in sorted(rows, key=lambda r: -abs(r[1]))[:5]:
    print(f"[verify]   worst {r[0]}: minZ {r[1]:.2f}  x {r[2]:.0f}..{r[3]:.0f}  y {r[4]:.0f}..{r[5]:.0f}")
for o in bpy.data.objects:
    if o.name in ("Ground", "Water", "Terrain", "Hills"):
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        print(f"[verify]   {o.name} extent x {min(v.x for v in bb):.0f}..{max(v.x for v in bb):.0f} y {min(v.y for v in bb):.0f}..{max(v.y for v in bb):.0f} z {min(v.z for v in bb):.0f}..{max(v.z for v in bb):.0f}")
# --- scene-camera views (the high ones)
for f in (1081, 2881, 3200, 3796):
    sc.frame_set(f)
    sc.render.filepath = os.path.join(out, f"{tag}_cam_f{f}.png")
    bpy.ops.render.render(write_still=True)
# --- top-down ortho of the whole world, red where there is no ground
w = sc.world
bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
red = w.node_tree.nodes.new("ShaderNodeBackground"); red.inputs["Color"].default_value = (1, 0, 0, 1); red.inputs["Strength"].default_value = 1.0
wo = next(n for n in w.node_tree.nodes if n.type == "OUTPUT_WORLD")
w.node_tree.links.new(red.outputs[0], wo.inputs["Surface"])
cam = bpy.data.objects.new("TopCam", bpy.data.cameras.new("TopCam")); sc.collection.objects.link(cam)
cam.data.type = "ORTHO"; cam.data.clip_end = 20000
cam.location = (0, 700, 6000); cam.rotation_euler = (0, 0, 0)
sc.camera = cam
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1200, 1200, 100
sun = bpy.data.objects.new("TopSun", bpy.data.lights.new("TopSun", "SUN")); sun.data.energy = 1.5; sc.collection.objects.link(sun)
sc.frame_set(1021)
for nm_, sz in (("topdown", 9000), ("topdown_wide", 34000)):
    cam.data.ortho_scale = sz
    sc.render.filepath = os.path.join(out, f"{tag}_{nm_}.png")
    bpy.ops.render.render(write_still=True)
print("[verify] done")
