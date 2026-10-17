"""Headless test for lib/checks.py bpy helpers. Through the slice:
    scripts/server/render.sh t-libchk --wait --collect -P ~/.local/bin/blender -b --factory-startup \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/tests/bpy_test_checks.py"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import bpyx as X  # noqa: E402
from lib import checks as K  # noqa: E402

FAILS = []


def check(name, cond, info=""):
    print(f"[bpy_test_checks] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAILS.append(name)


for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
wall = X.boxes_mesh("Wall", [(-5, 2, 0, 5, 3, 4)], bottom=True)
dg = bpy.context.evaluated_depsgraph_get()
bvh = K.bpy_bvh([wall], dg)
cam = np.array([(0, -5, 1.5), (0, 1.8, 1.5), (0, 2.5, 1.5), (0, 10, 1.5)])
hits = K.bpy_camera_clear(cam, bvh, 0.3)
check("camera_clear flags the 2 frames within 0.3 m (incl. inside)", [i for i, _ in hits] == [1, 2], str(hits))

pts = X.points_obj("Pts", np.c_[np.arange(7), np.zeros(7), np.zeros(7)])
ng, nb, gi, go = X.gn_tree("Inst")
iop = nb.n("GeometryNodeInstanceOnPoints")
nb.link(gi.outputs[0], iop.inputs["Points"])
nb.link(nb.n("GeometryNodeMeshCube").outputs["Mesh"], iop.inputs["Instance"])
nb.link(iop.outputs[0], go.inputs[0])
X.add_gn(pts, ng)
dg = bpy.context.evaluated_depsgraph_get()
check("bpy_instances counts GN instances", K.bpy_instances(pts, dg) == 7, str(K.bpy_instances(pts, dg)))

im = bpy.data.images.new("rel", 4, 4)
im.source, im.filepath = "FILE", "//textures/nope.png"
bad = K.bpy_bad_images()
check("bpy_bad_images flags relative + missing", any("relative" in w for n, w in bad if n == "rel") and
      any("missing" in w for n, w in bad if n == "rel"), str(bad))
print(f"[bpy_test_checks] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
sys.exit(1 if FAILS else 0)
