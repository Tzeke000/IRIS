"""Proof on a real scene: re-create the boss-fight v3c courtyard grass with scatter_zones, in a COPY of arena_v3.blend.

    scripts/server/render.sh sz-arena --wait --collect -P flock /tmp/iris_gpu_render.lock \
      ~/.local/bin/blender -b --factory-startup <repo>/scratch/scenes/boss_fight/arena_v3.blend --python-exit-code 1 \
      --python <repo>/scripts/lyric_scenes/lib/prove_arena_v3.py -- --out <...>/arena_v3_scatter_test.blend --report <...>.json

Never writes arena_v3.blend. The zones are the v3c ones (build_arena_v3.py::build_grass), expressed as zones:
  courtyard  include box(courtyard) on CourtyardPaving; denser within 3.5 m of the walls (include_edges) and within 2 m of
             anything standing on the ground (near objects); patch noise; nothing under anything (blockers)
  verge      include box(REG) minus the compound footprint, on Terrain; denser within 14 m of the walls
  keep-out   hall/platform/stairs box, main path box (gate -> stairs), the outside path ribbon (exact footprint),
             a 1.2 m circle around the m_grass lens
Then the v3c asserts are re-run independently on the evaluated instances.
"""
import argparse
import json
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lyric_scenes.lib.scatter_zones import scatter, box, circle, obj_footprint, near, noise, any_of, surface_bvh  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--density", type=float, default=22.0)   # v3c: 18/m^2 x its own falloff; 22 here lands the same visible count
A = ap.parse_args(argv)
assert os.path.basename(A.out) != "arena_v3.blend", "never overwrite arena_v3.blend"
D = bpy.data
T0 = time.time()

# v3 constants (build_arena_v3.py)
F, IX, IY0, IY1 = 8.0, 20.0, 0.0, 50.0
CXc, CY0c, CY1c, CWT = 50.0, -78.0, 86.0, 3.0
RECT = (-CXc - CWT, CY0c - CWT, CXc + CWT, CY1c + CWT)
REG = (-120.0, -150.0, 120.0, 160.0)
M_GRASS_CAM = (-13.5, -74.4, 0.55)

# ---- drop the v3c grass holder (keep its 4 templates)
old = D.objects["V3C_Grass"]
ng = old.modifiers[0].node_group
me = old.data
D.objects.remove(old)
D.node_groups.remove(ng)
D.meshes.remove(me)
TPLS = [D.objects[n] for n in ("TPL_V3G_tuftShort", "TPL_V3G_tuftMid", "TPL_V3G_tuftDry", "TPL_V3G_weed")]
C_G = D.collections["V3C_GRASS"]


# ---- obstacles = the v3c selection (anything standing on the courtyard / near-wall ground)
def top_parent(o):
    while o.parent is not None:
        o = o.parent
    return o


OBST = []
skip = {"Terrain", "CourtyardPaving", "ArenaFloor"}
for o in D.objects:
    if o.type != "MESH" or o.hide_render or o.name in skip or any(c.hide_render for c in o.users_collection):
        continue
    if any(c.name.startswith(("V3_EXT", "V3C_", "FX_", "SIZE_PROOF", "V3_EXTERIOR")) for c in o.users_collection):
        continue
    if top_parent(o).name in ("DRAGON", "KNIGHT"):
        continue
    pts = np.array([list(o.matrix_world @ Vector(c)) for c in o.bound_box])
    lo_, hi_ = pts.min(0), pts.max(0)
    if hi_[0] < REG[0] or lo_[0] > REG[2] or hi_[1] < REG[1] or lo_[1] > REG[3] or lo_[2] > 6.0:
        continue
    OBST.append(o)
PATH = D.objects["V3_Path_GateToTown"]
t_obst = time.time() - T0

KEEP_CLEAR = [box(-25.5, -28.0, 25.5, 56.0, label="hall_platform_stairs"), box(-6.5, -82.0, 6.5, -24.0, label="main_path")]
ROAD = obj_footprint(PATH, label="outside_path")
LENS = circle(M_GRASS_CAM[:2], 1.2, label="m_grass_lens")
PATCH = [noise(7.0, -0.15, 0.35, floor=0.5), noise(1.6, -0.4, 0.4, floor=0.6)]
BASES = near(OBST, 2.0, inner=0.95, outer=0.0, curve="linear")
COMMON = dict(items=TPLS, blockers=OBST, scale=(0.6, 1.5), shadows=False, collection=C_G, density=A.density)

t = time.time()
r_out = scatter("V3S_GrassVerge", surface=D.objects["Terrain"], seed=12,
                include=[box(*REG, label="region")], exclude=[box(*RECT, label="compound_footprint")] + KEEP_CLEAR + [ROAD],
                density_falloff=[any_of(near(box(*RECT), 14.0, start=1.5, inner=0.93, outer=0.08), BASES)] + PATCH, **COMMON)
r_in = scatter("V3S_GrassCourtyard", surface=D.objects["CourtyardPaving"], seed=11,
               include=[box(-CXc, CY0c, CXc, CY1c, label="courtyard")], exclude=KEEP_CLEAR + [LENS, ROAD],
               density_falloff=[any_of(near("include_edges", 3.5, start=0.8, inner=1.0, outer=0.05), BASES)] + PATCH, **COMMON)
t_scatter = time.time() - t

# ---- the v3c asserts, re-run independently on the evaluated instances of BOTH holders
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
holders = {D.objects["V3S_GrassCourtyard"], D.objects["V3S_GrassVerge"]}
P = np.array([tuple(i.matrix_world.translation) for i in dg.object_instances
              if i.is_instance and i.parent is not None and i.parent.original in holders]).reshape(-1, 3)


def keep_clear(x, y):
    return ((-25.5 < x) & (x < 25.5) & (-28.0 < y) & (y < 56.0)) | ((np.abs(x) < 6.5) & (-82.0 < y) & (y < -24.0))


PROOF = []
for o in D.collections["SIZE_PROOF"].objects:
    if o.name.startswith("SizeProof_Dragon"):
        w = np.array([tuple(o.matrix_world @ v.co) for v in o.data.vertices])
        PROOF.append((*w.min(0)[:2], *w.max(0)[:2]))
in_proof = 0
for a, b, c, d in PROOF:
    in_proof += int(((P[:, 2] > F + 0.01) & (P[:, 0] > a) & (P[:, 0] < c) & (P[:, 1] > b) & (P[:, 1] < d)).sum())
hall = int(((np.abs(P[:, 0]) < IX + 3) & (P[:, 1] > IY0 - 3) & (P[:, 1] < IY1 + 3)).sum())
pbvh = surface_bvh([PATH])[0]
on_path = sum(pbvh.ray_cast(Vector((x, y, z + 5)), Vector((0, 0, -1)), 10.0)[0] is not None for x, y, z in P.tolist())
inside = (np.abs(P[:, 0]) < CXc) & (P[:, 1] > CY0c) & (P[:, 1] < CY1c)
in_rect = (P[:, 0] > RECT[0]) & (P[:, 0] < RECT[2]) & (P[:, 1] > RECT[1]) & (P[:, 1] < RECT[3])
in_wall_body = int((in_rect & ~inside).sum())
lens = int((np.hypot(P[:, 0] - M_GRASS_CAM[0], P[:, 1] - M_GRASS_CAM[1]) < 1.2).sum())
CHECK = dict(instances=int(len(P)), keep_clear_violations=int(keep_clear(P[:, 0], P[:, 1]).sum()), in_dragon_footprints=in_proof,
             in_hall_or_platform=hall, on_outside_path_ALL_checked=int(on_path), inside_wall_body=in_wall_body,
             within_1p2m_of_m_grass_lens=lens)
assert CHECK["keep_clear_violations"] == 0 and in_proof == 0 and hall == 0 and on_path == 0 and lens == 0, CHECK

v3 = json.load(open(os.path.join(os.path.dirname(A.out), "build_report_v3.json")))["added"]["grass"]
strip = ("points", "items", "rotation", "scale")
REP = {"v3c": {k: v3[k] for k in ("instances", "hidden_inside_wall_body", "keep_clear_violations", "in_dragon_footprints",
                                  "in_hall_or_platform", "on_outside_path_sampled", "max_ground_error_m", "build_s")},
       "scatter_zones": {"checks": CHECK, "courtyard": {k: v for k, v in r_in.items() if k not in strip},
                         "verge": {k: v for k, v in r_out.items() if k not in strip},
                         "density_per_m2": A.density, "obstacles": len(OBST), "obstacle_scan_s": round(t_obst, 2), "scatter_s": round(t_scatter, 2),
                         "visible_instances_vs_v3c_visible": [int(len(P)), int(v3["instances"] - v3["hidden_inside_wall_body"])]}}
json.dump(REP, open(A.report, "w"), indent=1, default=str)
print("[prove]", json.dumps(REP["scatter_zones"]["checks"]), "v3c:", json.dumps(REP["v3c"]), flush=True)
bpy.ops.wm.save_as_mainfile(filepath=A.out)
print("[prove] saved", A.out, round(time.time() - T0, 1), "s", flush=True)
