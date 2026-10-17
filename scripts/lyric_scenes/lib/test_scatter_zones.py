"""Headless test for scatter_zones on a small procedural scene (60 x 60 m bumpy ground, boxes, a roof slab, a path).

    ~/.local/bin/blender -b --factory-startup --python-exit-code 1 --python scripts/lyric_scenes/lib/test_scatter_zones.py -- \
        [--render /abs/out.png] [--blend /abs/out.blend] [--report /abs/report.json]

Renders ONLY when --render is given — launch that through scripts/server/render.sh (+ the GPU lock).
Asserts: counts, 0 keep-out violations on the evaluated instances, grounding, row structure, torch offsets/facing,
no overlapping props, determinism, "growing a zone does not reshuffle", and 100k+ points in seconds.
"""
import argparse
import json
import math
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # <repo>/scripts
from lyric_scenes.lib.scatter_zones import (scatter, box, circle, poly, path, obj_bounds,  # noqa: E402
                                            near, noise, any_of, draw_zones)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--render")
ap.add_argument("--blend")
ap.add_argument("--report")
A = ap.parse_args(argv)
D = bpy.data
SC = bpy.context.scene
for o in list(D.objects):
    D.objects.remove(o)
REP = {}


def mat(name, rgb, emit=0.0):
    m = D.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.8
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    return m


def link(o, c=None):
    (c or SC.collection).objects.link(o)
    return o


def bm_obj(name, bm, m):
    me = D.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(m)
    return link(D.objects.new(name, me))


# ------------------------------------------------------------------ scene: bumpy ground, boxes, roof slab, path
def ground_z(x, y):
    return 0.6 * np.sin(np.asarray(x) / 7.0) * np.cos(np.asarray(y) / 9.0)


N = 121
xs = np.linspace(-30, 30, N)
X, Y = np.meshgrid(xs, xs)
V = np.stack([X.ravel(), Y.ravel(), ground_z(X.ravel(), Y.ravel())], 1)
F = [(j * N + i, j * N + i + 1, (j + 1) * N + i + 1, (j + 1) * N + i) for j in range(N - 1) for i in range(N - 1)]
gme = D.meshes.new("Ground")
gme.from_pydata(V.tolist(), [], F)
gme.materials.append(mat("M_Ground", (0.09, 0.07, 0.05)))
GROUND = link(D.objects.new("Ground", gme))

M_STONE = mat("M_Stone", (0.35, 0.33, 0.3))
BOXES = []
for nm, (x, y, sx, sy, h, rz) in {"BoxA": (-12, 10, 3, 2, 2.5, 20), "BoxB": (8, -14, 2, 4, 3, -35), "BoxC": (16, 14, 2.5, 2.5, 1.5, 0)}.items():
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx * 2, sy * 2, h * 2), verts=bm.verts)
    o = bm_obj(nm, bm, M_STONE)
    o.location = (x, y, float(ground_z(x, y)))
    o.rotation_euler = (0, 0, math.radians(rz))
    BOXES.append(o)
# a roof slab on a post: grass must not grow UNDER it (blocker), but the slab is not an exclude zone
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.scale(bm, vec=(6, 6, 0.3), verts=bm.verts)
ROOF = bm_obj("RoofSlab", bm, M_STONE)
ROOF.location = (-14, -12, 2.6)
# the path: a bezier curve across the ground
cu = D.curves.new("PathCurve", "CURVE")
cu.dimensions = "3D"
sp = cu.splines.new("BEZIER")
sp.bezier_points.add(3)
for bp, co in zip(sp.bezier_points, [(-30, -24, 0), (-10, -4, 0), (6, 4, 0), (30, 26, 0)]):
    bp.co = co
    bp.handle_left_type = bp.handle_right_type = "AUTO"
cu.resolution_u = 16
PATHC = link(D.objects.new("PathCurve", cu))
bpy.context.view_layer.update()

# ------------------------------------------------------------------ item templates (distinct vertex counts)
TPL = D.collections.new("TEMPLATES")
SC.collection.children.link(TPL)
M_GR = mat("M_Grass", (0.12, 0.3, 0.06))
M_GR2 = mat("M_GrassDry", (0.3, 0.26, 0.1))


def tuft(name, n, h, m, seed):
    r = np.random.default_rng(seed)
    bm = bmesh.new()
    for _ in range(n):
        a = r.uniform(0, 2 * math.pi)
        d = Vector((math.cos(a), math.sin(a), 0))
        s = Vector((-d.y, d.x, 0)) * 0.03
        b = d * r.uniform(0, 0.08)
        hh = h * r.uniform(0.6, 1.2)
        vs = [bm.verts.new(b - s), bm.verts.new(b + s), bm.verts.new(b + d * hh * 0.4 + Vector((0, 0, hh)))]
        bm.faces.new(vs)
    o = bm_obj(name, bm, m)
    TPL.objects.link(o)
    SC.collection.objects.unlink(o)
    return o


def prim(name, kind, m, **kw):
    bm = bmesh.new()
    if kind == "rock":
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=kw.get("r", 0.5))
        bmesh.ops.scale(bm, vec=(1.0, 0.8, 0.55), verts=bm.verts)
    elif kind == "crate":
        bmesh.ops.create_cube(bm, size=0.9)
        bmesh.ops.translate(bm, vec=(0, 0, 0.45), verts=bm.verts)
    elif kind == "torch":
        bmesh.ops.create_cone(bm, segments=6, radius1=0.06, radius2=0.06, depth=1.6, cap_ends=True)
        bmesh.ops.translate(bm, vec=(0, 0, 0.8), verts=bm.verts)
        # a flame plate on the FRONT (+Y) so facing is visible from above
        bmesh.ops.create_cube(bm, size=0.25, matrix=__import__("mathutils").Matrix.Translation((0, 0.2, 1.6)))
    elif kind == "tree":
        bmesh.ops.create_cone(bm, segments=10, radius1=kw.get("r", 1.6), radius2=0.0, depth=4.0, cap_ends=True)
        bmesh.ops.translate(bm, vec=(0, 0, 3.0), verts=bm.verts)
    elif kind == "crop":
        bmesh.ops.create_cone(bm, segments=5, radius1=0.18, radius2=0.0, depth=0.6, cap_ends=True)
        bmesh.ops.translate(bm, vec=(0, 0, 0.3), verts=bm.verts)
    o = bm_obj(name, bm, m)
    TPL.objects.link(o)
    SC.collection.objects.unlink(o)
    return o


GRASS = [tuft("TuftA", 10, 0.25, M_GR, 1), tuft("TuftB", 14, 0.35, M_GR, 2), tuft("TuftDry", 6, 0.4, M_GR2, 3)]
ROCK = prim("Rock", "rock", M_STONE)
CRATE = prim("Crate", "crate", mat("M_Crate", (0.35, 0.2, 0.08)))
TORCH = prim("Torch", "torch", mat("M_Torch", (1.0, 0.45, 0.1), emit=6.0))
TREE = prim("Tree", "tree", mat("M_Tree", (0.03, 0.12, 0.04)))
CROP = prim("Crop", "crop", mat("M_Crop", (0.55, 0.45, 0.08)))

# ------------------------------------------------------------------ 1. random grass with keep-outs + falloff + blockers
FIELD_X = (-28, 28)
INC_G = [box(-28, -28, 28, 28, label="field")]
EXC_G = [obj_bounds(BOXES, pad=0.3),
         path(PATHC, width=3.0, label="path"),
         box(center=(12, -2), size=(8, 5), angle=25, label="threshing_floor"),
         circle((-2, 18), 3.0, label="well"),
         box(4, 14, 28, 28, label="crop_field"),          # rows go there (test 2)
         circle((14, -22), 7.0, label="grove")]          # trees go there (test 5)
t = time.time()
r1 = scatter("t_grass", surface=GROUND, items=GRASS, weights=[3, 3, 1], include=INC_G, exclude=EXC_G,
             mode="random", density=40.0,
             density_falloff=[any_of(near("exclude_edges", 2.5, inner=1.0, outer=0.35), near(BOXES, 1.5, outer=0.0)),
                              noise(6.0, -0.3, 0.3, floor=0.3)],
             blockers=[ROOF], scale=(0.7, 1.3), align_to_normal=0.5, seed=7)
REP["random_grass"] = {k: r1[k] for k in ("count", "candidates", "excluded_by", "covered_by_blockers", "max_ground_error_m",
                                          "per_item", "item_pick_checked", "timing_s", "build_s", "keep_out_violations")}
assert r1["count"] > 15000, r1["count"]
assert r1["covered_by_blockers"] > 100, ("roof slab did not block", r1["covered_by_blockers"])
assert all(v > 0 for v in r1["excluded_by"].values()), r1["excluded_by"]
assert r1["item_pick_checked"] > 0
print("[test] random grass", r1["count"], r1["build_s"], "s", flush=True)

# ------------------------------------------------------------------ 2. rows (crop rows in a rotated field)
r2 = scatter("t_crops", surface=GROUND, items=[CROP], include=[box(4, 14, 28, 28, label="crop_field")],
             exclude=[obj_bounds(BOXES[2], pad=0.4)], mode="rows", spacing=0.6, row_spacing=1.5, angle=30.0,
             origin=(4, 14), jitter=(0.06, 0.03), scale=(0.85, 1.15), seed=3)
P = np.array([v.co[:] for v in D.objects["t_crops"].data.vertices])
u = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
v = np.array([-u[1], u[0]])
across = (P[:, :2] - np.array([4, 14])) @ v
row_err = np.abs(across - np.round(across / 1.5) * 1.5).max()
gx, gy = np.meshgrid(np.arange(4, 28, 0.05) + 0.025, np.arange(14, 28, 0.05) + 0.025)
G = np.stack([gx.ravel(), gy.ravel()], 1)
area = float((obj_bounds(BOXES[2], pad=0.4).sdist(G) >= 0.18).sum()) * 0.0025      # field minus the padded box
expect = area / (0.6 * 1.5)
REP["rows"] = {"count": r2["count"], "expected_approx": round(expect), "max_off_row_m": round(float(row_err), 3),
               "build_s": r2["build_s"]}
assert row_err <= 0.031, ("crops off their rows", row_err)
assert abs(r2["count"] - expect) / expect < 0.1, (r2["count"], expect)
print("[test] rows", r2["count"], "expected ~", round(expect), flush=True)

# ------------------------------------------------------------------ 3. grid (rocks + crates, staggered, jittered, no overlap)
r3 = scatter("t_yard", surface=GROUND, items=[ROCK, CRATE], weights=[2, 1],
             include=[poly([(-28, 2), (-18, 2), (-18, 8), (-24, 14), (-28, 14)], label="yard")],
             exclude=[obj_bounds(BOXES[0], pad=0.2)], mode="grid", spacing=1.6, stagger=True, jitter=0.35,
             min_spacing="auto", scale=(0.7, 1.1), seed=5)
P3 = np.array([v.co[:] for v in D.objects["t_yard"].data.vertices])
rad3 = np.array([r3["item_radius_m"][("Rock", "Crate")[k]] for k in
                 np.array([a.value for a in D.objects["t_yard"].data.attributes["scatter_item"].data])])
s3 = np.array([a.value for a in D.objects["t_yard"].data.attributes["scatter_scale"].data])
rr3 = rad3 * s3
dd = np.hypot(P3[:, None, 0] - P3[None, :, 0], P3[:, None, 1] - P3[None, :, 1]) - (rr3[:, None] + rr3[None, :])
np.fill_diagonal(dd, 1e9)
REP["grid"] = {"count": r3["count"], "spacing_rejected": r3["spacing_rejected"], "min_gap_m": round(float(dd.min()), 3),
               "per_item": r3["per_item"]}
assert r3["count"] > 20 and dd.min() >= -1e-6, ("overlapping props", dd.min())
print("[test] grid", r3["count"], "min gap", round(float(dd.min()), 3), flush=True)

# ------------------------------------------------------------------ 4. along_path: torches both sides, facing the path
r4 = scatter("t_torches", surface=GROUND, items=[TORCH], mode="along_path", path=PATHC, spacing=4.0, offset=2.2,
             both_sides=True, face="path", path_start=2.0, path_end=2.0, scale=1.0,
             exclude=[obj_bounds(BOXES, pad=0.3), circle((-2, 18), 3.0)], seed=1)
me4 = D.objects["t_torches"].data
P4 = np.array([v.co[:] for v in me4.vertices])
R4 = np.array([a.vector[:] for a in me4.attributes["scatter_rot"].data])
cen = path(PATHC, width=0.0)
dist = cen.sdist(P4[:, :2])                                          # distance to the centre line
front = np.stack([-np.sin(R4[:, 2]), np.cos(R4[:, 2])], 1)           # local +Y after yaw
probe = P4[:, :2] + front * 0.5
toward = cen.sdist(probe) < dist                                     # stepping forward gets closer to the path
plen = sum(np.linalg.norm(np.diff(L[:, :2], axis=0), axis=1).sum() for L in path(PATHC, 1.0).lines)
REP["along_path"] = {"count": r4["count"], "path_len_m": round(float(plen), 1), "offset_err_m": round(float(np.abs(dist - 2.2).max()), 3),
                     "facing_path": int(toward.sum()), "excluded_by": r4["excluded_by"]}
assert np.abs(dist - 2.2).max() < 0.05, ("torch offset", np.abs(dist - 2.2).max())
assert toward.all(), ("torches not facing the path", int((~toward).sum()))
assert r4["count"] >= 2 * int((plen - 4) / 4) - 2 * 6, r4["count"]
print("[test] along_path", r4["count"], flush=True)

# ------------------------------------------------------------------ 5. big props: trees, min spacing auto
r5 = scatter("t_grove", surface=GROUND, items=[TREE, ROCK], weights=[3, 1], include=[circle((14, -22), 7.0, label="grove")],
             exclude=[obj_bounds(BOXES[1], pad=0.5)], mode="random", density=0.6, min_spacing="auto",
             spacing_factor=0.9, scale=(0.6, 1.0), shadows=True, fit_inside=True, seed=2)
me5 = D.objects["t_grove"].data
P5 = np.array([v.co[:] for v in me5.vertices])
r5r = np.array([r5["item_radius_m"][("Tree", "Rock")[a.value]] for a in me5.attributes["scatter_item"].data]) * \
    np.array([a.value for a in me5.attributes["scatter_scale"].data])
outside = (np.hypot(P5[:, 0] - 14, P5[:, 1] + 22) + r5r - 7.0).max()
REP["grove"] = {"count": r5["count"], "spacing_rejected": r5["spacing_rejected"], "per_item": r5["per_item"],
                "fit_inside_worst_overhang_m": round(float(outside), 3)}
assert r5["count"] >= 6 and r5["spacing_rejected"] > 0 and outside <= 1e-6, REP["grove"]
print("[test] grove", r5["count"], flush=True)

# ------------------------------------------------------------------ 6. determinism + locality + speed (no geometry)
cfg = dict(surface=GROUND, items=GRASS, include=[box(-28, -28, 0, 0)], exclude=[circle((-14, -14), 4)], mode="random",
           density=30.0, seed=11, create=False, verbose=False)
a = scatter("t_det", **cfg)
b = scatter("t_det", **cfg)
assert np.array_equal(a["points"], b["points"]) and np.array_equal(a["items"], b["items"]), "not deterministic"
c = scatter("t_det", **{**cfg, "seed": 12})
assert not np.array_equal(a["points"][:100], c["points"][:100]), "seed ignored"
g = scatter("t_det", **{**cfg, "include": [box(-28, -28, 10, 0)]})       # grow the zone east by 10 m
ka = {tuple(np.round(p[:2], 5)) for p in a["points"]}
kg = {tuple(np.round(p[:2], 5)) for p in g["points"] if p[0] < 0}
za = {tuple(np.round(p[:2], 5)): p[2] for p in a["points"]}
zg = {tuple(np.round(p[:2], 5)): p[2] for p in g["points"] if p[0] < 0}
print("[test] locality xy-diff", len(ka ^ kg), "max z diff", max((abs(za[k] - zg[k]) for k in ka & kg), default=0), flush=True)
REP["determinism"] = {"same_seed_identical": True, "grown_zone_keeps_old_points": len(ka & kg), "of": len(ka),
                      "grown_zone_new_points_in_old_area": len(kg - ka)}
assert ka == kg, ("growing the zone reshuffled the old area", len(ka ^ kg))
assert max(abs(za[k] - zg[k]) for k in ka) < 1e-5            # z may differ in the last float bit (BVH over other tris)
t = time.time()
big = scatter("t_big", surface=GROUND, items=GRASS, include=[box(-30, -30, 30, 30)], exclude=EXC_G[:3], mode="random",
              density=45.0, density_falloff=[noise(6.0, -0.5, 0.2, floor=0.5)], seed=4, create=False, verbose=False)
REP["speed_100k"] = {"points": big["count"], "candidates": big["candidates"], "s": round(time.time() - t, 2), "timing_s": big["timing_s"]}
assert big["count"] > 100000 and time.time() - t < 30, REP["speed_100k"]
print("[test] determinism ok; speed", REP["speed_100k"], flush=True)

# ------------------------------------------------------------------ debug zones + optional render
draw_zones(INC_G + [box(4, 14, 28, 28), circle((14, -22), 7.0)], EXC_G, z=1.2, width=0.09)
print("[test] ALL PASS", json.dumps(REP, default=str)[:2000], flush=True)
if A.report:
    json.dump(REP, open(A.report, "w"), indent=1, default=str)
if A.blend:
    bpy.ops.wm.save_as_mainfile(filepath=A.blend)
if A.render:
    cd = D.cameras.new("Top")
    cd.type = "ORTHO"
    cd.ortho_scale = 62
    cam = link(D.objects.new("Top", cd))
    cam.location = (0, 0, 60)
    SC.camera = cam
    sd = D.lights.new("Sun", "SUN")
    sd.energy = 3.0
    sun = link(D.objects.new("Sun", sd))
    sun.rotation_euler = (math.radians(35), 0, math.radians(30))
    SC.world = SC.world or D.worlds.new("W")
    SC.world.use_nodes = True
    SC.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.25, 0.27, 0.3, 1)
    SC.render.engine = "BLENDER_EEVEE"
    SC.eevee.taa_render_samples = 8
    SC.render.resolution_x = SC.render.resolution_y = 900
    SC.render.image_settings.file_format = "PNG"
    SC.render.filepath = A.render
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print("[test] rendered", A.render, round(time.time() - t, 1), "s", flush=True)
