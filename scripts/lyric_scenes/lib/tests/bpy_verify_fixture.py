"""Tiny scene for tests/test_verify_saved.sh: --mode good passes every verify_saved check, --mode bad plants one defect
per check. Built with the lib itself (camera.bake_shots, look.eevee_defaults, bpyx), so it doubles as an integration run.
    blender -b --factory-startup --python bpy_verify_fixture.py -- --out /ABS/x.blend --mode good|bad"""
import argparse
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import bpyx as X  # noqa: E402
from lib import camera as CM  # noqa: E402
from lib import look as L  # noqa: E402
from lib.tests.bpy_test_importer import png_bytes  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--out", required=True)
a.add_argument("--mode", choices=("good", "bad"), required=True)
a = a.parse_args(sys.argv[sys.argv.index("--") + 1:])
BAD = a.mode == "bad"
N = 48

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
sc = bpy.context.scene
sc.frame_start, sc.frame_end = 1, N
L.eevee_defaults(sc, samples=1, res=(64, 36), fps=30)
sc.world = bpy.data.worlds.new("W")


def box(name, lo, hi, coll=None, mat=None):
    o = X.boxes_mesh(name, [(*lo, *hi)], coll=coll, bottom=True)
    if mat:
        o.data.materials.append(mat)
    return o


# textured ground (absolute texture next to the .blend; bad: a relative path)
tex = os.path.join(os.path.dirname(os.path.abspath(a.out)), "tex_ground.png")
open(tex, "wb").write(png_bytes())
gm, nb, out = X.new_mat("GroundMat")
t = nb.n("ShaderNodeTexImage")
t.image = bpy.data.images.load(tex)
if BAD:
    t.image.filepath = "//textures/nope.png"
nb.link(X.principled(nb, t.outputs["Color"]).outputs[0], out.inputs["Surface"])
box("Ground", (-30, -30, -1), (30, 30, 0), mat=gm)
box("Wall", (-6, 10, 0), (6, 11, 4))
box("Prop_crate", (2, 2, 1.5 if BAD else 0.0), (3, 3, 2.5 if BAD else 1.0))           # bad: hovers 1.5 m
hz, hb, ho = X.new_mat("HazeMat")                                                          # volume-only: camera may sit inside
v = hb.n("ShaderNodeVolumePrincipled")
v.inputs["Density"].default_value = 0.001
hb.link(v.outputs[0], ho.inputs["Volume"])
box("HazeBox", (-40, -40, -2), (40, 40, 30), mat=hz)

# a car driving +Y; bad: its nose points −Y (the police scar)
car = box("Car_1", (-1, -2, 0), (1, 2, 1.4))
car.rotation_euler = (0, 0, np.pi if BAD else 0.0)
X.bake_vec(car, "location", np.c_[np.full(N, 8.0), np.linspace(-15, 5, N), np.zeros(N)])

# GN instancer of a kit collection (kit hide_render only, parked underground); bad: the kit object is hide_viewport
kitc = X.coll("Kit")
kc = box("KitCube", (-0.3, -0.3, -100.3), (0.3, 0.3, -99.7), coll=kitc)
kc.hide_render = True
if BAD:
    kc.hide_viewport = True
pts = X.points_obj("Lamps", np.c_[np.linspace(-8, 8, 5), np.full(5, 5.0), np.zeros(5)])
ng, gb, gi, go = X.gn_tree("LampInst")
ci = gb.n("GeometryNodeCollectionInfo", transform_space="RELATIVE")
ci.inputs["Collection"].default_value = kitc
ci.inputs["Reset Children"].default_value = True
iop = gb.n("GeometryNodeInstanceOnPoints")
gb.link(gi.outputs[0], iop.inputs["Points"])
gb.link(ci.outputs[0], iop.inputs["Instance"])
gb.link(iop.outputs[0], go.inputs[0])
X.add_gn(pts, ng)

# compositor: glare + lens (both modes). bad: a NaN made IN the compositor (0 × exp(1000)) — raw clean, composited NaN:
# gotcha 2's signature. (EEVEE 5.2.1 rendered shader-side NaN / 0·inf emission as black, measured 10-18, so the
# fixture can't plant it in a material.) Plus a NaN keyframe value for the finite check.
P = L.post_chain(sc, glare=[dict(type="Fog Glow", threshold=1.2, strength=0.4)], lens=(-0.006, 0.01))
if BAD:
    car["glow"] = 0.0
    X.bake(car, '["glow"]', np.r_[np.zeros(N - 1), np.nan], thin=False)
    nb_ = P["nb"]
    nan = nb_.m("MULTIPLY", nb_.m("EXPONENT", nb_.val("Big", 1000.0)), 0.0)
    nb_.link(nb_.mix(1.0, P["image"], nan, blend="ADD"), P["out"].inputs["Image"])

# camera: two shots, cut at 25. bad: shot 2 starts inside the wall, and the bake is NOT cut-safe
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
sc.collection.objects.link(cam)
y2 = 10.5 if BAD else -12.0
shots = [CM.Shot(1, "wide", lambda i, u, n: dict(cam=(0, -20 + 4 * u / n, 2), tgt=(0, 10, 2), lens=24)),
         CM.Shot(25, "close", lambda i, u, n: dict(cam=(1.0 * u / n, y2, 2), tgt=(0, 20, 2), lens=35))]
CM.bake_shots(cam, shots, N, cut_safe=not BAD)
if BAD:
    sc["cuts"] = [25]                       # the cut exists; the keys just don't hold across it
CM.still_cam("CAM_still", (15, -15, 3), (0, 0, 1), 30)
X.save_blend(a.out)
print("[fixture] saved", a.out, a.mode)
