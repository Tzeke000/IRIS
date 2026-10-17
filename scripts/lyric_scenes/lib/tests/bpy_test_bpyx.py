"""Headless Blender test for lib/bpyx.py. Through the slice:
    scripts/server/render.sh t-libbpyx --wait --collect -P ~/.local/bin/blender -b --factory-startup \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/tests/bpy_test_bpyx.py"""
import os
import sys

import bpy
import numpy as np

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(LIB))
from lib import bpyx as X  # noqa: E402

FAILS = []


def check(name, cond, info=""):
    print(f"[bpy_test_bpyx] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAILS.append(name)


def main():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    C = X.coll("TEST")
    check("coll get-or-create is idempotent", X.coll("TEST") is C and len([c for c in sc.collection.children if c.name == "TEST"]) == 1)

    # --- keys: a camera dolly with a cut at frame 31 (new shot starts) ---
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    X.link(cam, C)
    n = 60
    x = np.r_[np.linspace(0, 10, 30), np.linspace(100, 110, 30)]     # jump at the cut
    pos = np.stack([x, np.zeros(n), np.full(n, 2.0)], 1)
    fcs = X.bake_vec(cam, "location", pos, cuts=[31])
    fx = fcs[0]
    k30 = [k for k in fx.keyframe_points if int(k.co[0]) == 30][0]
    check("key on the frame before the cut is CONSTANT", k30.interpolation == "CONSTANT")
    check("no smear: x(30.5) == x(30) (shutter START sees the old shot only)", abs(fx.evaluate(30.5) - x[29]) < 1e-5,
          f"{fx.evaluate(30.5):.3f} vs {x[29]:.3f}")
    check("new shot value at frame 31", abs(fx.evaluate(31) - 100) < 1e-5)
    check("y channel thinned to its plateau ends (+ cut keys)", len(fcs[1].keyframe_points) == 4,
          str([int(k.co[0]) for k in fcs[1].keyframe_points]))
    sc.frame_set(45)
    check("evaluates through the 5.x slotted action", abs(cam.location.x - x[44]) < 1e-4, f"{cam.location.x:.3f}")
    fc = X.bake(cam, "location", x, index=0, thin=False, interp="LINEAR")
    check("rooftop mode (thin=False) keys every frame, replaces old keys", len(fc.keyframe_points) == n)

    # --- show_mask combines with existing hide_render keys ---
    ob = X.mesh_obj("Thing", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)], coll=C)
    X.bake(ob, "hide_render", (np.arange(n) < 10).astype(float), interp="CONSTANT")    # hidden frames 1..10
    X.show_mask(ob, np.arange(n) < 40)                                                 # visible only frames 1..40
    hid = [X.find_fcurves(ob, "hide_render")[0].evaluate(f) > 0.5 for f in range(1, n + 1)]
    check("show_mask AND-s with existing keys (hidden 1-10 and 41-60)",
          hid == [True] * 10 + [False] * 30 + [True] * 20)

    # --- materials + node DSL ---
    m, nb, out = X.new_mat("M")
    b = X.principled(nb, (0.8, 0.1, 0.1), rough=nb.mr(nb.val("Rough", 0.5), 0, 1, 0.2, 0.9))
    nb.link(b.outputs[0], out.inputs["Surface"])
    check("NB: principled with a linked map-range roughness", b.inputs["Roughness"].is_linked and
          np.allclose(tuple(b.inputs["Base Color"].default_value), (0.8, 0.1, 0.1, 1.0)),
          f'linked={b.inputs["Roughness"].is_linked} base={tuple(b.inputs["Base Color"].default_value)}')
    check("simple_mat / emit_mat", X.simple_mat("S", (1, 1, 1)).node_tree.nodes.get("Principled BSDF") is not None
          and X.emit_mat("E", (1, 0, 0), 5).node_tree.nodes["Emission"].inputs["Strength"].default_value == 5)
    try:
        X.img("relative/path.png")
        check("img() refuses a relative path", False)
    except FileNotFoundError:
        check("img() refuses a relative path", True)

    # --- GN: 10 points → instance a cube on each → realize; count verts in the evaluated mesh ---
    pts = X.points_obj("Pts", np.c_[np.arange(10), np.zeros(10), np.zeros(10)], attrs={"s": np.linspace(0.1, 1, 10),
                                                                                     "dir": np.ones((10, 3))}, coll=C)
    check("points_obj attributes", "s" in pts.data.attributes and pts.data.attributes["dir"].data_type == "FLOAT_VECTOR")
    ng, gb, gi, go = X.gn_tree("Inst")
    cube = gb.n("GeometryNodeMeshCube")
    iop = gb.n("GeometryNodeInstanceOnPoints")
    gb.link(gi.outputs[0], iop.inputs["Points"])
    gb.link(cube.outputs["Mesh"], iop.inputs["Instance"])
    rz = gb.n("GeometryNodeRealizeInstances")
    gb.link(iop.outputs[0], rz.inputs[0])
    gb.link(rz.outputs[0], go.inputs[0])
    X.add_gn(pts, ng)
    dg = bpy.context.evaluated_depsgraph_get()
    nv = len(pts.evaluated_get(dg).data.vertices)
    check("gn_tree + add_gn evaluate (10 cubes × 8 verts)", nv == 80, str(nv))

    # --- meshes ---
    bx = X.boxes_mesh("Boxes", [(0, 0, 0, 1, 1, 1), (2, 0, 0, 3, 2, 5)], face_attrs={"lit": [0.0, 1.0]}, coll=C)
    check("boxes_mesh 2×5 faces + per-box face attr", len(bx.data.polygons) == 10 and
          [a.value for a in bx.data.attributes["lit"].data][5:] == [1.0] * 5)
    mb = X.MB().box((0, 0, 0), (2, 2, 2)).beam((0, 0, 0), (0, 0, 4), 0.2)
    o = mb.obj("MB", coll=C)
    check("MB box + beam → 16 verts, 12 faces", len(o.data.vertices) == 16 and len(o.data.polygons) == 12)
    zs = [v.co.z for v in o.data.vertices][8:]
    check("MB beam spans a→b", abs(min(zs)) < 1e-5 and abs(max(zs) - 4) < 1e-5)
    s = X.uv_sphere("Ball", 2.0, seg=16, rings=8, coll=C)
    r = np.linalg.norm(np.array([v.co for v in s.data.vertices]), axis=1)
    check("uv_sphere radius + UVs", np.allclose(r, 2.0, atol=1e-5) and len(s.data.uv_layers) == 1)

    # --- lights (v2 semantics) ---
    sp = X.light("Spot", "SPOT", (0, 0, 10), target=(0, 0, 0), energy=500, spot_deg=40, coll=C)
    bpy.context.view_layer.update()
    d = sp.matrix_world.to_3x3() @ __import__("mathutils").Vector((0, 0, -1))
    check("light aims at its target (-Z → down)", (d - __import__("mathutils").Vector((0, 0, -1))).length < 1e-5)
    check("spot cone set", abs(sp.data.spot_size - np.radians(40)) < 1e-6)
    ar = X.light("Area", "AREA", (5, 0, 5), target=(0, 0, 0), size=3.0, coll=C)
    check("AREA size (dropped by v3's copy)", ar.data.size == 3.0)

    print(f"[bpy_test_bpyx] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


main()
