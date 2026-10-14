"""Probe asset glbs: per mesh object world bbox, armature + actions. Usage: blender -b --factory-startup -P probe_assets.py -- OUT.json glb..."""
import bpy, sys, json
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]; out, paths = a[0], a[1:]
res = {}
def wbb(o):
    cs = [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = [min(c[i] for c in cs) for i in range(3)]; hi = [max(c[i] for c in cs) for i in range(3)]
    return [round(x, 3) for x in lo], [round(x, 3) for x in hi]
for p in paths:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=p)
    objs = []
    for o in bpy.data.objects:
        if o.type in ("MESH", "ARMATURE"):
            lo, hi = wbb(o)
            objs.append(dict(name=o.name, type=o.type, parent=o.parent.name if o.parent else None, lo=lo, hi=hi,
                             dims=[round(hi[i]-lo[i], 3) for i in range(3)], verts=len(o.data.vertices) if o.type == "MESH" else len(o.data.bones),
                             mats=[m.name for m in o.data.materials] if o.type == "MESH" else []))
    alllo = [min(o["lo"][i] for o in objs if o["type"] == "MESH") for i in range(3)]
    allhi = [max(o["hi"][i] for o in objs if o["type"] == "MESH") for i in range(3)]
    res[p] = dict(objects=objs[:80], n=len(objs), all_lo=alllo, all_hi=allhi, actions=[(ac.name, list(ac.frame_range)) for ac in bpy.data.actions])
    print("PROBE", p, len(objs), alllo, allhi, flush=True)
json.dump(res, open(out, "w"), indent=1)
