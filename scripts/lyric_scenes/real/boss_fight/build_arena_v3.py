"""BOSS FIGHT — arena v3: INTERIOR PASS (Zeke's notes 2026-10-14), layered ON TOP of arena_v2.blend.

    blender -b --factory-startup arena_v2.blend --python-exit-code 1 --python build_arena_v3.py -- --out arena_v3.blend
            [--report build_report_v3.json] [--nosim] [--no-exterior]
ALWAYS launched through:  scripts/server/render.sh <unit> scripts/lyric_scenes/real/boss_fight/render.sh build3

What v3 adds INSIDE the cathedral hall (nothing outside is touched):
  1. COVER — waist-high (0.9-1.3 m) things the knight can crouch behind, spread along his route from the gate to the hoard:
     broken screen walls (coursed ashlar, jagged broken top, stepped broken ends), fallen column drums lying on their
     sides (a toppled round column = a line of drums), collapsed arch-ring chunks (several voussoirs still bonded, fallen
     flat), and knocked-askew stone pews. All hand-built here (no new downloads), grounded by construction (lowest vertex
     on the floor) and placed so nothing interpenetrates (BVH overlap test against every visible mesh in the hall).
     Where the nave is the 12-dragon floor the cover sits only in the arcade/aisle bands and in the two ~1.25 m strips
     between the dragon rows (the pulpitum line and a fallen column) — never on a proof footprint, never in the main aisle.
     Two tipped pews (lying on their backs, 0.62 m) are dressing, not cover.
  2. RUBBLE — denser: masonry heaps against every cover piece, along the aisle walls and the pier bases, a few large
     fallen cornice blocks, all dropped and SETTLED with Bullet against the real geometry (v2 rubble included as a
     collider), then fine gravel/chips scattered analytically (each chip ray-tested onto bare floor).
  3. Two cover cameras for the stills: CAM_i_cover (knight's eye, crouched behind the fallen column) and CAM_j_cover_run
     (along the east arcade band, the cover sequence), plus CAM_d_hallplan (roof-off plan of the hall, for QA).

PHASE 2 (v3b, Zeke 10-14 notes):
  4. HOARD — 3 more chests (2 OPEN: lid on its hinge, coin dome, coins + gems spilling down the dais), the 2 v1 chests
     re-seated on the gold, Zweihaender + gold angel sword planted in the gold mounds, bow + holy greatsword + mace + war
     hammer on oak display boards on the north wall. All kept out of the e_prize frame (asserted): the RPG-7 stays the hero.
  5. EXTERIOR (`build_exterior()`, backdrop only — the fight stays in the courtyard): terrain v3 (valley from the gate
     down to the town, hills keep the skyline, sinking rim at 4.5 km), 2 houses just outside the south wall, a dirt path
     to a small town (36 procedural cottages + church spire, lit windows, street glow), village cottages along the path,
     strip fields + hedges, ~14k GN-instanced procedural trees (east woods stretch + hill woods; terrain tint beyond 820 m),
     low drifting ground fog (bounded volume boxes, `frame` drivers), horizon-mist band widened (HDRI's own trees hidden).
PHASE 3 (v3c): the KHR_materials_unlit scans (self-lit -> rendered near-white) become lit Principled materials;
  patchy courtyard grass + weeds (walls, building bases, paving cracks, outside verge), GN instances, shadows off,
  keep-clear zones asserted on the evaluated instances; new low shot CAM_m_grass; a camera-inside-object assert.

Build asserts (report: build_report_v3.json): 12-dragon footprint clear (vertex-level for v3 + the v2 bbox check over
v2+v3), main aisle clear, cover heights in range, cover grounded + stable + not interpenetrating, every crouch spot free
and its line of sight to the threat blocked, cover spacing along the route, cameras not inside geometry.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import zlib

import bmesh
import bpy
import mathutils
import numpy as np
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

T0 = time.time()
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--report", default="")
ap.add_argument("--nosim", action="store_true", help="skip the rubble settle sim (fast layout iteration)")
ap.add_argument("--no-exterior", action="store_true", help="skip the exterior (fast interior iteration)")
A = ap.parse_args(argv)

D = bpy.data
SC = bpy.context.scene
assert SC.name == "BossArena" and "V2_RUBBLE" in D.collections and "V3_COVER" not in D.collections, "run on arena_v2.blend"

# ---------------------------------------------------------------------------------------------------------
# v1 dimensions (MUST match build_arena.py)
# ---------------------------------------------------------------------------------------------------------
F = 8.0                     # hall floor (ArenaFloor top)
IX = 20.0                   # inner half-width
IY0, IY1 = 0.0, 50.0        # inner south / north faces
COL_X, PIER_W = 13.0, 4.71
COL_Y = [6.25, 18.75, 31.25, 43.75]
DAIS_Y0 = 45.5
GATE_HALF = 6.8             # gate opening between the gate piers (x = +-8 +- 1.2)
DRAGON_POS = Vector((0.0, 33.5, F))
AISLE_HALF = 2.0            # main aisle kept clear: |x| < 2.0 from outside the gate to the dais
MAIN_AISLE_Y = (-3.0, DAIS_Y0)
COVER_H = (0.9, 1.3)
PH = "/data/iris_assets/medieval/polyhaven/"

REPORT = {"version": 3, "checks": {}, "added": {}, "removed_v2": {}, "timing_s": {}}
rng = random.Random(3131)


def sd(*a):
    """stable seed (str hash() is salted per process)."""
    return zlib.crc32(repr(a).encode()) % 99991


def log(*a):
    print("[v3]", *a, flush=True)


def tick(label, t):
    REPORT["timing_s"][label] = round(time.time() - t, 1)
    log(f"{label}: {REPORT['timing_s'][label]} s")
    return time.time()


def coll(name, parent=None):
    c = D.collections.new(name)
    (parent or SC.collection).children.link(c)
    return c


C_TMP = coll("_tmp3")
C_V3 = coll("V3_INTERIOR")
C_COVER = coll("V3_COVER", C_V3)
C_RUB = coll("V3_RUBBLE", C_V3)
C_GRAV = coll("V3_GRAVEL", C_V3)
C_CAM = D.collections["CAMERAS"]


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
def mesh_co(me):
    a = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def world_co(o):
    co = mesh_co(o.data)
    M = np.array(o.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def world_bbox(o):
    w = world_co(o)
    return w.min(0), w.max(0)


def tris_world(o, dg):
    """evaluated (modifiers/GN applied) world-space triangles of an object."""
    oe = o.evaluated_get(dg)
    me = oe.to_mesh()
    try:
        me.calc_loop_triangles()
        co = mesh_co(me)
        M = np.array(oe.matrix_world)
        w = co @ M[:3, :3].T + M[:3, 3]
        t = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", t)
    finally:
        oe.to_mesh_clear()
    return w, t.reshape(-1, 3)


def bvh_of(parts):
    """parts: [(verts Nx3, tris Mx3)] -> one BVH"""
    vs, ts, off = [], [], 0
    for w, t in parts:
        if len(t) == 0:
            continue
        vs.append(w)
        ts.append(t + off)
        off += len(w)
    if not vs:
        return None
    V, T = np.concatenate(vs), np.concatenate(ts)
    return BVHTree.FromPolygons(V.tolist(), T.tolist())


def obj_bvh(o):
    me = o.data
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", t)
    return bvh_of([(world_co(o), t.reshape(-1, 3))])


def new_obj(name, me, c, loc=(0, 0, 0), rotz=0.0):
    o = D.objects.new(name, me)
    c.objects.link(o)
    o.location = loc
    o.rotation_euler = (0, 0, rotz)
    return o


def to_mesh(bm, name, mats):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = D.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    for p in me.polygons:
        p.use_smooth = False
    return me


def ground_mesh(me):
    """shift so the lowest vertex sits at z = 0 (the object is then placed at z = F: grounded by construction)."""
    co = mesh_co(me)
    me.transform(Matrix.Translation((0, 0, -float(co[:, 2].min()))))


def block_verts(bm, lo, hi, bevel=0.02, jitter=0.0, rr=None, M=None):
    """a bevelled box (one ashlar) into bm; optional vertex jitter (weathering) and a placement matrix."""
    g = bmesh.ops.create_cube(bm, size=1.0)
    vs = g["verts"]
    sx, sy, sz = (hi[i] - lo[i] for i in range(3))
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=vs)
    bmesh.ops.translate(bm, vec=((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2), verts=vs)
    if bevel > 0:
        bm.edges.index_update()
        es = sorted({e for v in vs for e in v.link_edges}, key=lambda e: e.index)          # sorted: sets iterate in
        r = bmesh.ops.bevel(bm, geom=es, offset=bevel, segments=2, affect="EDGES", clamp_overlap=True)   # address order
        bm.verts.index_update()
        vs = sorted({v for f in r.get("faces", []) for v in f.verts} | set(v for v in vs if v.is_valid), key=lambda v: v.index)
    vs = [v for v in vs if v.is_valid]
    if jitter and rr is not None:
        for v in vs:
            v.co += Vector((rr.uniform(-1, 1), rr.uniform(-1, 1), rr.uniform(-1, 1))) * jitter
    if M is not None:
        bmesh.ops.transform(bm, matrix=M, verts=vs)
    return vs


# ---------------------------------------------------------------------------------------------------------
# material — joint-free limestone for hand-built masonry (the block JOINTS are geometry, so the texture must not have
# any). Poly Haven rock_pitted_mossy (CC0) luminance, re-tinted to the measured mean of the piers' own baked albedo.
# ---------------------------------------------------------------------------------------------------------
_IMG = {}


def img(path, color=True):
    if (path, color) not in _IMG:
        im = D.images.load(path, check_existing=True)
        if not color:
            im.colorspace_settings.name = "Non-Color"
        _IMG[(path, color)] = im
    return _IMG[(path, color)]


def img_mean(im, step=37):
    px = np.empty(len(im.pixels), np.float32)
    im.pixels.foreach_get(px)
    px = px.reshape(-1, 4)[::step, :3]
    return px.mean(0)


def principled(mat):
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def make_ashlar_mat():
    m = D.materials.new("M_AshlarV3")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = principled(m)
    tex = PH + "tex_rock_pitted_mossy/rock_pitted_mossy_"
    diff = img(tex + "diff_2k.jpg")
    gc = next((n.image for n in D.materials["GothCol_BAKED"].node_tree.nodes
               if n.type == "TEX_IMAGE" and n.image and "nor" not in n.image.name.lower()
               and "rough" not in n.image.name.lower()), None)
    target = img_mean(gc) if gc else np.array([0.30, 0.27, 0.23])
    lum = float(np.dot(img_mean(diff), [0.2126, 0.7152, 0.0722]))
    tint = np.clip(0.62 * target / max(lum, 1e-3), 0, 1.0)   # x0.62: matched by eye to the piers under brazier light
    REPORT["added"]["ashlar_tint"] = dict(target_mean=[round(float(v), 3) for v in target], rock_lum=round(lum, 3),
                                         tint=[round(float(v), 3) for v in tint], target_src=gc.name if gc else None)
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / 0.9,) * 3
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    def tx(im):
        t = N.new("ShaderNodeTexImage")
        t.image = im
        t.projection = "BOX"
        t.projection_blend = 0.3
        L.new(mp.outputs["Vector"], t.inputs["Vector"])
        return t
    td, tr, tn = tx(diff), tx(img(tex + "rough_2k.jpg", False)), tx(img(tex + "nor_gl_2k.jpg", False))
    bw = N.new("ShaderNodeRGBToBW")
    L.new(td.outputs["Color"], bw.inputs["Color"])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, 2 * lum
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.35, 1.65
    L.new(bw.outputs["Val"], mr.inputs["Value"])
    nz = N.new("ShaderNodeTexNoise")                     # macro grime / soot
    nz.inputs["Scale"].default_value = 0.7
    nz.inputs["Detail"].default_value = 6
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    gm = N.new("ShaderNodeMapRange")
    gm.inputs["From Min"].default_value, gm.inputs["From Max"].default_value = 0.4, 0.7
    gm.inputs["To Min"].default_value, gm.inputs["To Max"].default_value = 1.0, 0.5
    L.new(nz.outputs["Fac"], gm.inputs["Value"])
    mul = N.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    L.new(mr.outputs["Result"], mul.inputs[0])
    L.new(gm.outputs["Result"], mul.inputs[1])
    mx = N.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    mx.blend_type = "MULTIPLY"
    mx.inputs["Factor"].default_value = 1.0
    mx.inputs[6].default_value = (*[float(v) for v in tint], 1)
    cr = N.new("ShaderNodeCombineColor")
    for i in range(3):
        L.new(mul.outputs[0], cr.inputs[i])
    L.new(cr.outputs["Color"], mx.inputs[7])
    L.new(mx.outputs[2], b.inputs["Base Color"])
    rr_ = N.new("ShaderNodeMapRange")
    rr_.inputs["To Min"].default_value, rr_.inputs["To Max"].default_value = 0.7, 0.95
    L.new(tr.outputs["Color"], rr_.inputs["Value"])
    L.new(rr_.outputs["Result"], b.inputs["Roughness"])
    nm = N.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = 1.2
    L.new(tn.outputs["Color"], nm.inputs["Color"])
    L.new(nm.outputs["Normal"], b.inputs["Normal"])
    return m


M_ASH = make_ashlar_mat()

# ---------------------------------------------------------------------------------------------------------
# the 12-dragon proof footprints + the scene's obstacles in the hall
# ---------------------------------------------------------------------------------------------------------
PROOF = [o for o in D.collections["SIZE_PROOF"].objects if o.name.startswith("SizeProof_Dragon")]
PROOF_BOX = []
for o in PROOF:
    lo_, hi_ = world_bbox(o)
    PROOF_BOX.append((float(lo_[0]), float(lo_[1]), float(hi_[0]), float(hi_[1])))
assert len(PROOF_BOX) == 12
PROOF_ARR = np.array(PROOF_BOX)


def pts_in_proof(P, pad=0.0):
    """P: Nx3 world points -> bool mask of points above the floor inside any proof footprint (+pad)."""
    m = np.zeros(len(P), bool)
    above = P[:, 2] > F + 0.01
    for a, b, c, d in PROOF_BOX:
        m |= above & (P[:, 0] > a - pad) & (P[:, 0] < c + pad) & (P[:, 1] > b - pad) & (P[:, 1] < d + pad)
    return m


def pts_in_aisle(P, pad=0.0):
    return ((np.abs(P[:, 0]) < AISLE_HALF + pad) & (P[:, 1] > MAIN_AISLE_Y[0]) & (P[:, 1] < MAIN_AISLE_Y[1])
            & (P[:, 2] > F + 0.01) & (P[:, 2] < F + 4.0))


def strip_gaps():
    """y-gaps between the dragon rows (the only nave floor that is not a footprint)."""
    ys = sorted({(round(b, 3), round(d, 3)) for a, b, c, d in PROOF_BOX})
    return [(ys[i][1], ys[i + 1][0]) for i in range(len(ys) - 1)]


STRIPS = strip_gaps()
REPORT["added"]["nave_strips_between_dragon_rows"] = [[round(a, 2), round(b, 2)] for a, b in STRIPS]
assert len(STRIPS) == 2 and all(b - a > 1.1 for a, b in STRIPS), STRIPS


def top_parent(o):
    while o.parent is not None:
        o = o.parent
    return o


SKIP_COLL = {"SIZE_PROOF", "FX_haze", "CAMERAS", "V3_COVER", "V3_RUBBLE", "V3_GRAVEL", "_tmp3"}
HALL_LO, HALL_HI = np.array([-IX - 0.6, IY0 - 0.6, F - 0.6]), np.array([IX + 0.6, IY1 + 0.6, F + 5.0])


def is_loose_rubble(o):
    return o.name.startswith(("Rubble2_", "Rubble_chunk", "Rubble0", "Rubble1"))


def hall_objects():
    """every visible mesh whose bbox touches the hall interior volume (characters + floor excluded)."""
    out = []
    for o in D.objects:
        if o.type != "MESH" or o.hide_render or o.name == "ArenaFloor" or any(c.hide_render for c in o.users_collection):
            continue
        if any(c.name in SKIP_COLL for c in o.users_collection):
            continue
        if top_parent(o).name in ("DRAGON", "KNIGHT"):
            continue
        pts = np.array([list(o.matrix_world @ Vector(c)) for c in o.bound_box])
        if (pts.max(0) < HALL_LO).any() or (pts.min(0) > HALL_HI).any():
            continue
        out.append(o)
    return out


t = time.time()
dg = bpy.context.evaluated_depsgraph_get()
HALL = hall_objects()
TRIS = {o.name: tris_world(o, dg) for o in HALL}
STATIC_NAMES = [o.name for o in HALL if not is_loose_rubble(o)]
LOOSE_NAMES = [o.name for o in HALL if is_loose_rubble(o)]
STATIC_BVH = bvh_of([TRIS[n] for n in STATIC_NAMES])
OBJ_BVH = {}


def bvh_named(n):
    if n not in OBJ_BVH:
        OBJ_BVH[n] = bvh_of([TRIS[n]])
    return OBJ_BVH[n]


AABB = {n: (TRIS[n][0].min(0), TRIS[n][0].max(0)) for n in TRIS}
log(f"hall objects {len(HALL)} (static {len(STATIC_NAMES)}, loose rubble {len(LOOSE_NAMES)})")
t = tick("scan_hall", t)


# ---------------------------------------------------------------------------------------------------------
# =========================================  INTERIOR — 1. COVER  =========================================
# ---------------------------------------------------------------------------------------------------------
def wall_mesh(name, L, t_, seed, courses=3, base_h=0.2):
    """broken screen wall, local X = length, Y = thickness, z from 0: a projecting base course + `courses` courses of
    running-bond ashlar (8 mm joints). Broken: the ends step down course by course, one notch in the top, top-course
    dropouts. Returns mesh; its height is base_h + courses * ch (ch drawn so the full height lands in 1.04-1.24 m)."""
    rr = random.Random(seed)
    ch = rr.uniform(0.28, 0.34)
    gap = 0.008
    bm = bmesh.new()
    # base course (plinth), slightly wider, in 3-4 long stones, also broken at the ends
    x = -L / 2 + rr.uniform(0.0, 0.25)
    xe = L / 2 - rr.uniform(0.0, 0.25)
    while x < xe - 0.2:
        ln = min(rr.uniform(0.9, 1.5), xe - x)
        block_verts(bm, (x + gap / 2, -t_ / 2 - 0.05, 0.0), (x + ln - gap / 2, t_ / 2 + 0.05, base_h - gap / 2), 0.02, 0.006, rr)
        x += ln
    # top profile: how many courses survive at position s along the wall
    lstep = rr.uniform(0.45, 0.7)                            # broken ends lose a course every lstep
    notch_c, notch_w = rr.uniform(-L / 4, L / 4), rr.uniform(0.5, 0.9)

    def n_at(s):
        n = courses
        n = min(n, 1 + int((s + L / 2) / lstep), 1 + int((L / 2 - s) / lstep))
        if abs(s - notch_c) < notch_w / 2:
            n = min(n, courses - 1)
        return max(n, 1)
    for c in range(courses):
        z0 = base_h + c * ch
        x = -L / 2 + (rr.uniform(0.25, 0.45) if c % 2 else 0.0)
        if c % 2:                                             # odd courses start with a short closer
            s = (-L / 2 + x) / 2
            if c < n_at(s):
                block_verts(bm, (-L / 2 + gap / 2, -t_ / 2, z0), (x - gap / 2, t_ / 2, z0 + ch - gap), 0.02, 0.006, rr)
        while x < L / 2 - 0.15:
            ln = min(rr.uniform(0.5, 0.85), L / 2 - x)
            s = x + ln / 2
            keep = c < n_at(s) and not (c == courses - 1 and rr.random() < 0.22)
            if keep:
                dy = rr.uniform(-0.012, 0.012)
                block_verts(bm, (x + gap / 2, -t_ / 2 + dy, z0), (x + ln - gap / 2, t_ / 2 + dy, z0 + ch - gap), 0.022, 0.007, rr)
            x += ln
    me = to_mesh(bm, name, [M_ASH])
    ground_mesh(me)
    return me


def drum_mesh(name, r, ln, seed):
    """a column drum lying on its side: local X = axis. Bevelled rims, chipped arrises, slightly irregular."""
    rr = random.Random(seed)
    bm = bmesh.new()
    g = bmesh.ops.create_cone(bm, cap_ends=True, segments=36, radius1=r, radius2=r, depth=ln)
    vs = g["verts"]
    rim = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-6]
    bmesh.ops.bevel(bm, geom=rim, offset=0.035, segments=2, affect="EDGES", clamp_overlap=True)
    for v in bm.verts:
        rad = math.hypot(v.co.x, v.co.y)
        if v.co.x > 0.8 * r:                                 # the bearing line (bottom after the turn): left true, so
            continue                                         # the drum rests on BOTH ends (no rocking, no floating end)
        edge = abs(abs(v.co.z) - ln / 2) < 0.06 and rad > r - 0.06
        if edge and rr.random() < 0.35:                      # chipped arris
            k = 1.0 - rr.uniform(0.0, 0.06) / max(rad, 1e-3)
            v.co.x *= k
            v.co.y *= k
            v.co.z -= math.copysign(rr.uniform(0, 0.03), v.co.z)
        elif rad > 1e-3:
            k = 1.0 - rr.uniform(0.0, 0.006) / rad
            v.co.x *= k
            v.co.y *= k
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "Y"))   # axis -> X
    me = to_mesh(bm, name, [M_ASH])
    for p in me.polygons:
        p.use_smooth = abs(p.normal.x) < 0.5                  # smooth barrel, flat beds
    ground_mesh(me)
    return me


def arch_chunk_mesh(name, seed, ri=1.5, ring=0.75, depth=1.12, span_deg=100.0, n=6):
    """a run of `n` voussoirs of an arch ring still bonded together, fallen FLAT: the ring lies in the XY plane, the
    arch's depth becomes the height (z). Convex side towards local +Y. The end voussoirs are broken short."""
    rr = random.Random(seed)
    bm = bmesh.new()
    a0 = math.radians(90 - span_deg / 2)
    da = math.radians(span_deg) / n
    gap_a = 0.008 / ri
    for k in range(n):
        lo_a, hi_a = a0 + k * da + gap_a, a0 + (k + 1) * da - gap_a
        if k == 0:
            lo_a += da * rr.uniform(0.15, 0.4)
        if k == n - 1:
            hi_a -= da * rr.uniform(0.15, 0.4)
        ro = ri + ring * (1.0 if 0 < k < n - 1 else rr.uniform(0.8, 1.0))
        segs = 3
        ring_pts = []
        for j in range(segs + 1):
            a = lo_a + (hi_a - lo_a) * j / segs
            ring_pts.append((math.cos(a), math.sin(a)))
        verts = []
        for z in (0.0, depth - (0 if 0 < k < n - 1 else rr.uniform(0.0, 0.06))):
            row_i = [bm.verts.new((ri * c, ri * s, z)) for c, s in ring_pts]
            row_o = [bm.verts.new((ro * c, ro * s, z)) for c, s in ring_pts]
            verts.append((row_i, row_o))
        (bi, bo), (ti, to) = verts
        for j in range(segs):
            bm.faces.new((bi[j], bo[j], bo[j + 1], bi[j + 1]))
            bm.faces.new((ti[j], ti[j + 1], to[j + 1], to[j]))
            bm.faces.new((bi[j], bi[j + 1], ti[j + 1], ti[j]))      # intrados
            bm.faces.new((bo[j], to[j], to[j + 1], bo[j + 1]))      # extrados
        bm.faces.new((bi[0], ti[0], to[0], bo[0]))
        bm.faces.new((bi[-1], bo[-1], to[-1], ti[-1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    hard = [e for e in bm.edges if e.calc_face_angle(0.0) > math.radians(40)]
    bmesh.ops.bevel(bm, geom=hard, offset=0.025, segments=2, affect="EDGES", clamp_overlap=True)
    for v in bm.verts:
        v.co += Vector((rr.uniform(-1, 1), rr.uniform(-1, 1), rr.uniform(-1, 1))) * 0.005
    co = np.array([v.co[:] for v in bm.verts])
    c = (co.min(0) + co.max(0)) / 2
    bmesh.ops.translate(bm, verts=bm.verts, vec=(-c[0], -c[1], 0))
    me = to_mesh(bm, name, [M_ASH])
    ground_mesh(me)
    return me


def pew_parts(bm, L, rr, M):
    """stone pew in bm (local X = length, facing +Y, back towards -Y, z from 0), transformed by M."""
    d0, d1 = -0.32, 0.30
    prof = [(d0, 0.0), (d1, 0.0), (d1, 0.56), (d1 - 0.07, 0.63), (-0.10, 0.63), (-0.13, 1.0), (-0.20, 1.04), (d0, 1.04)]
    allv = []
    for sx in (-1, 1):
        x0, x1 = sorted((sx * (L / 2 - 0.1), sx * L / 2))
        a = [bm.verts.new((x0, y, z)) for y, z in prof]
        b = [bm.verts.new((x1, y, z)) for y, z in prof]
        bm.faces.new(list(reversed(a)))
        bm.faces.new(b)
        for i in range(len(prof)):
            j = (i + 1) % len(prof)
            bm.faces.new((a[i], a[j], b[j], b[i]))
        allv += a + b
    xi = L / 2 - 0.1
    allv += block_verts(bm, (-xi, -0.22, 0.40), (xi, 0.28, 0.48), 0.012, 0.0)
    allv += block_verts(bm, (-xi, d0 + 0.01, 0.48), (xi, -0.22, 0.98), 0.012, 0.0)
    for v in allv:
        if v.is_valid:
            v.co += Vector((rr.uniform(-1, 1), rr.uniform(-1, 1), 0.0)) * 0.004
    bmesh.ops.transform(bm, matrix=M, verts=[v for v in allv if v.is_valid])


def pew_mesh(name, L, seed, tipped=False):
    rr = random.Random(seed)
    bm = bmesh.new()
    M = Matrix.Rotation(math.pi / 2, 4, "X") if tipped else Matrix()      # tipped = lying on its back
    pew_parts(bm, L, rr, M)
    me = to_mesh(bm, name, [M_ASH])
    ground_mesh(me)
    return me


def rotz_facing(d):
    """rotz that turns local +Y to point along the 2D direction d."""
    return math.atan2(-d[0], d[1])


def threat_point(kind, c):
    if kind == "dragon":
        return Vector((DRAGON_POS.x, DRAGON_POS.y, F + 1.4))
    return Vector((0.0, c[1], F + 1.4))                     # "nave": straight out of the nave, square to the spot


# the layout: (id, kind, centre, params, threat). Strip covers sit on the y-gaps between dragon rows.
S1, S2 = [(a + b) / 2 for a, b in STRIPS]
SPOTS = [
    ("E0", "wall", (12.9, 2.15), dict(L=2.8, axis="y"), "nave"),
    ("W0", "arch", (-13.0, 2.1), dict(), "nave"),
    ("WC", "pews", (-17.6, 6.6), dict(pews=((0.0, -0.2, 2.0),)), "nave"),
    ("EC", "arch", (17.6, 5.4), dict(), "nave"),
    ("W1", "wall", (-12.9, 11.8), dict(L=4.4, axis="y"), "nave"),
    ("E1", "pews", (13.0, 12.1), dict(pews=((-0.7, -1.0, 2.2), (0.6, 1.3, 2.0))), "dragon"),
    ("S1W", "drums", (-6.9, S1), dict(r=0.47, lens=(1.55, 1.3, 1.65), axis="x"), "dragon"),
    ("S1E", "wall", (5.3, S1), dict(L=4.2, t=0.55, axis="x"), "dragon"),
    ("WD", "drums", (-17.7, 20.8), dict(r=0.5, lens=(1.4, 1.2), axis="y"), "nave"),
    ("W2", "pews", (-12.9, 24.7), dict(pews=((0.3, -1.4, 2.2), (-0.4, 1.3, 2.2))), "nave"),
    ("E2", "arch", (13.0, 25.0), dict(), "nave"),
    ("S2W", "wall", (-7.35, S2), dict(L=5.9, t=0.55, axis="x"), "dragon"),
    ("S2E", "drums", (5.7, S2), dict(r=0.47, lens=(1.5, 1.4), axis="x"), "dragon"),
    ("W3", "wall", (-12.9, 37.4), dict(L=6.2, axis="y"), "nave"),
    ("E3", "drums", (13.0, 37.2), dict(r=0.56, lens=(1.45, 1.6), axis="y"), "nave"),
    ("EA", "wall", (16.6, 30.6), dict(L=3.4, axis="y"), "nave"),
    ("WA", "pews", (-17.6, 28.3), dict(pews=((0.55, -0.6, 2.0), (-1.0, 0.7, 2.0))), "nave"),
    ("WB", "wall", (-16.7, 42.6), dict(L=3.2, axis="y"), "nave"),
    ("EB", "drums", (17.4, 42.4), dict(r=0.52, lens=(1.5, 1.3), axis="y"), "nave"),
    ("E4", "wall", (15.7, 48.9), dict(L=3.8, axis="x"), "dragon"),
    ("W4", "arch", (-16.0, 48.35), dict(), "dragon"),
]
TIPPED = [("TippedPew_W2", (-14.6, 27.4), 2.1, 20.0), ("TippedPew_E1", (15.2, 15.6), 2.0, -70.0)]

# remove loose rubble (v1/v2) under a cover footprint before placing; static things are hard obstacles
COVER = []           # dicts: id, kind, objs, centre, threat, crouch, thin_axis
REMOVED = []
PLACED_BVH = []      # (name, bvh)


def candidate_ok(objs, zone_pad=0.12):
    """True/False, reason. objs already transformed (matrix_world valid after an update)."""
    bpy.context.view_layer.update()
    for o in objs:
        w = world_co(o)
        if pts_in_proof(w, pad=zone_pad).any():
            return False, "proof"
        if pts_in_aisle(w, pad=0.3).any():
            return False, "aisle"
        if (np.abs(w[:, 0]) > IX - 0.02).any() or (w[:, 1] < IY0 + 0.02).any() or (w[:, 1] > IY1 - 0.02).any():
            return False, "outside"
        b = obj_bvh(o)
        lo, hi = w.min(0), w.max(0)
        for n in STATIC_NAMES:
            alo, ahi = AABB[n]
            if (ahi < lo - 0.05).any() or (alo > hi + 0.05).any():
                continue
            if b.overlap(bvh_named(n)):
                return False, "hits " + n
        for n, pb in PLACED_BVH:
            if b.overlap(pb):
                return False, "hits cover " + n
    return True, ""


def clear_loose_under(objs, pad=0.25):
    bpy.context.view_layer.update()
    W = np.concatenate([world_co(o) for o in objs])
    lo, hi = W.min(0) - pad, W.max(0) + pad
    gone = []
    for n in list(LOOSE_NAMES):
        alo, ahi = AABB[n]
        if (ahi[:2] < lo[:2]).any() or (alo[:2] > hi[:2]).any():
            continue
        if n in D.objects:
            D.objects.remove(D.objects[n])
        LOOSE_NAMES.remove(n)
        gone.append(n)
    return gone


NUDGES = [(0, 0)] + [(r * math.cos(a), r * math.sin(a)) for r in (0.15, 0.3, 0.5, 0.75) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]


def place_group(sid, parts, zone_pad=0.12):
    """parts: [(name, mesh, (x, y), rotz)]; nudge the whole group until it is legal. Returns the objects."""
    objs = [new_obj(nm, me, C_COVER, (x, y, F), rz) for nm, me, (x, y), rz in parts]
    base = [(o.location.x, o.location.y) for o in objs]
    why = ""
    for dx, dy in NUDGES:
        for o, (x, y) in zip(objs, base):
            o.location.x, o.location.y = x + dx, y + dy
        ok, why = candidate_ok(objs, zone_pad)
        if ok:
            break
    else:
        raise AssertionError(f"cover {sid}: no legal placement ({why})")
    for o in objs:
        PLACED_BVH.append((o.name, obj_bvh(o)))
        o["cover_spot"] = sid
    return objs, (dx, dy)


rr = random.Random(77)
for sid, kind, (cx, cy), p, threat in SPOTS:
    T = threat_point(threat, (cx, cy))
    to_t = Vector((T.x - cx, T.y - cy)).normalized()
    parts, thin = [], None
    if kind == "wall":
        rz = 0.0 if p["axis"] == "x" else math.pi / 2
        rz += math.radians(rr.uniform(-1.0, 1.0))
        me = wall_mesh(f"CV_{sid}_wall", p["L"], p.get("t", 0.6), seed=sd(sid))
        parts.append((f"CV_{sid}_ScreenWall", me, (cx, cy), rz))
        thin = Vector((-math.sin(rz), math.cos(rz)))
        half_t = p.get("t", 0.6) / 2 + 0.05
    elif kind == "drums":
        lens = p["lens"]
        total = sum(lens) + 0.3 * (len(lens) - 1)
        ax = Vector((1, 0)) if p["axis"] == "x" else Vector((0, 1))
        s = -total / 2
        for k, ln in enumerate(lens):
            c = Vector((cx, cy)) + ax * (s + ln / 2)
            jit = rr.uniform(-0.015, 0.015) if p["axis"] == "x" else rr.uniform(-0.25, 0.25)
            perp = Vector((-ax.y, ax.x))
            c += perp * jit
            rz = math.atan2(ax.y, ax.x) + math.radians(rr.uniform(-0.8, 0.8) if p["axis"] == "x" else rr.uniform(-9, 9))
            parts.append((f"CV_{sid}_Drum{k}", drum_mesh(f"CV_{sid}_drum{k}", p["r"], ln, seed=sd(sid, k)), c[:], rz))
            s += ln + rr.uniform(0.2, 0.4)
        thin = Vector((-ax.y, ax.x))
        half_t = p["r"] + 0.05
    elif kind == "arch":
        rz = rotz_facing(to_t) + math.radians(rr.uniform(-8, 8))
        me = arch_chunk_mesh(f"CV_{sid}_arch", seed=sd(sid))
        parts.append((f"CV_{sid}_ArchChunk", me, (cx, cy), rz))
        thin = Vector((-math.sin(rz), math.cos(rz)))
        half_t = 0.6
    elif kind == "pews":
        base_rz = rotz_facing(-to_t)                          # backs towards the threat
        for k, (u, v, ln) in enumerate(p["pews"]):
            rz = base_rz + math.radians(rr.uniform(-22, 22))
            axis = Vector((math.cos(base_rz), math.sin(base_rz)))
            c = Vector((cx, cy)) + axis * v + Vector((-axis.y, axis.x)) * u
            parts.append((f"CV_{sid}_Pew{k}", pew_mesh(f"CV_{sid}_pew{k}", ln, seed=sd(sid, k)), c[:], rz))
        thin = Vector((-math.sin(base_rz), math.cos(base_rz)))
        half_t = 0.7
    # clear loose rubble under the group, then place (nudging if a static object is in the way)
    tmp = [new_obj("_probe", me, C_TMP, (x, y, F), rz) for nm, me, (x, y), rz in parts]
    REMOVED += clear_loose_under(tmp)
    for o in tmp:
        D.objects.remove(o)
    objs, nudge = place_group(sid, parts)
    REMOVED += clear_loose_under(objs)
    bpy.context.view_layer.update()
    W = np.concatenate([world_co(o) for o in objs])
    cen = Vector(((W[:, 0].min() + W[:, 0].max()) / 2, (W[:, 1].min() + W[:, 1].max()) / 2))
    sgn = 1.0 if (cen - T.xy).dot(thin) > 0 else -1.0
    COVER.append(dict(id=sid, kind=kind, objs=objs, centre=cen, threat=T, threat_kind=threat, crouch=None, thin=thin * sgn,
                      nudge=[round(nudge[0], 2), round(nudge[1], 2)]))

# dressing: two pews tipped onto their backs (0.62 m — below crouch height, NOT counted as cover)
TIPPED_OBJS = []
for nm, (x, y), ln, rzd in TIPPED:
    tmp = [new_obj("_probe", pew_mesh(nm + "_m", ln, seed=len(nm), tipped=True), C_TMP, (x, y, F), math.radians(rzd))]
    REMOVED += clear_loose_under(tmp)
    me = tmp[0].data
    D.objects.remove(tmp[0])
    objs, _ = place_group(nm, [(nm, me, (x, y), math.radians(rzd))])
    REMOVED += clear_loose_under(objs)
    TIPPED_OBJS += objs

# crouch spots: walk along each cover on its sheltered side and take the spot nearest its middle where a crouched knight
# (eye 0.8 m) fits (nothing within 0.3 m at knee and eye height) AND this cover is the first thing between him and the
# threat. Loose v1/v2 rubble on the chosen spot is cleared.
class TagBVH:
    def __init__(self, names):
        self.names, starts, parts, n0 = [], [], [], 0
        for n in names:
            w, t_ = TRIS[n]
            if not len(t_):
                continue
            self.names.append(n)
            starts.append(n0)
            parts.append((w, t_))
            n0 += len(t_)
        self.starts = np.array(starts)
        self.bvh = bvh_of(parts)

    def ray(self, o, d, dist):
        loc, nrm, idx, dd = self.bvh.ray_cast(o, d, dist)
        if loc is None:
            return None, None
        return self.names[int(np.searchsorted(self.starts, idx, side="right") - 1)], dd


bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
COVER_OBJS = [o for c in COVER for o in c["objs"]] + TIPPED_OBJS
for o in COVER_OBJS:
    TRIS[o.name] = tris_world(o, dg)
LOS = TagBVH(STATIC_NAMES + [o.name for o in COVER_OBJS])
FIT = bvh_of([TRIS[n] for n in STATIC_NAMES + [o.name for o in COVER_OBJS]])


def crouch_ok(c, q):
    if abs(q.x) > IX - 0.35 or q.y < IY0 + 0.35 or q.y > IY1 - 0.35:
        return False
    for z in (0.35, 0.8):
        hit = FIT.find_nearest(Vector((q.x, q.y, F + z)))
        if hit[0] is not None and hit[3] < 0.3:
            return False
    eye = Vector((q.x, q.y, F + 0.8))
    d = c["threat"] - eye
    nm, _ = LOS.ray(eye, d.normalized(), d.length)
    return nm is not None and D.objects[nm].get("cover_spot") == c["id"]


for c in COVER:
    W = np.concatenate([world_co(o) for o in c["objs"]])
    thin = c["thin"]
    lng = Vector((-thin.y, thin.x))
    s_ = W[:, :2] @ np.array(lng[:])
    u_ = W[:, :2] @ np.array(thin[:])
    s0 = float(lng.dot(c["centre"]))
    cands = sorted(np.arange(s_.min() + 0.25, s_.max() - 0.2, 0.15), key=lambda v: abs(v - s0))
    for sv in cands:
        near = np.abs(s_ - sv) < 0.35
        if not near.any():
            continue
        q = lng * float(sv) + thin * float(u_[near].max() + 0.4)
        if crouch_ok(c, q):
            c["crouch"] = q
            break
    if c["crouch"] is not None:
        cq = c["crouch"]
        for n in list(LOOSE_NAMES):
            alo, ahi = AABB[n]
            if alo[0] < cq.x + 0.45 and ahi[0] > cq.x - 0.45 and alo[1] < cq.y + 0.45 and ahi[1] > cq.y - 0.45 and alo[2] < F + 1.0:
                if n in D.objects:
                    D.objects.remove(D.objects[n])
                LOOSE_NAMES.remove(n)
                REMOVED.append(n)
    log(f"  cover {c['id']:4s} {c['kind']:5s} crouch {None if c['crouch'] is None else (round(c['crouch'].x, 2), round(c['crouch'].y, 2))}")

REPORT["removed_v2"]["loose_rubble_under_cover"] = dict(count=len(REMOVED), names=REMOVED)
log(f"cover: {len(COVER)} spots, {sum(len(c['objs']) for c in COVER)} pieces; removed {len(REMOVED)} loose v1/v2 pieces under them")
t = tick("cover", t)


# ---------------------------------------------------------------------------------------------------------
# =========================================  INTERIOR — 2. RUBBLE  ========================================
# ---------------------------------------------------------------------------------------------------------
def frag_mesh(name, seed, size):
    """irregular masonry fragment: convex hull of a jittered box (a broken corner of an ashlar)."""
    r_ = random.Random(seed)
    bm = bmesh.new()
    sx, sy, sz = size * r_.uniform(0.7, 1.0), size * r_.uniform(0.5, 0.9), size * r_.uniform(0.35, 0.7)
    for _ in range(14):
        bm.verts.new((r_.uniform(-sx, sx), r_.uniform(-sy, sy), r_.uniform(-sz, sz)))
    for cx_ in (-1, 1):                                      # keep some flat dressed faces
        for cy_ in (-1, 1):
            if r_.random() < 0.6:
                bm.verts.new((cx_ * sx, cy_ * sy, r_.choice((-1, 1)) * sz))
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:])
    bmesh.ops.delete(bm, geom=list({g for g in res["geom_interior"] + res["geom_unused"] if isinstance(g, bmesh.types.BMVert)}), context="VERTS")
    co = np.array([v.co[:] for v in bm.verts])
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(-co.mean(0)))
    return to_mesh(bm, name, [M_ASH])


def block_mesh(name, seed, dims, bevel=0.03):
    r_ = random.Random(seed)
    bm = bmesh.new()
    sx, sy, sz = dims
    block_verts(bm, (-sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, sz / 2), bevel, 0.01, r_)
    if r_.random() < 0.6:                                     # a broken corner
        c = Vector((r_.choice((-1, 1)) * sx / 2, r_.choice((-1, 1)) * sy / 2, sz / 2))
        n = (c.normalized() + Vector((r_.uniform(-.3, .3), r_.uniform(-.3, .3), 0))).normalized()
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=c - n * min(sx, sy, sz) * 0.35,
                               plane_no=n, clear_outer=True)
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
    return to_mesh(bm, name, [M_ASH])


t = time.time()
POOL_CELL = [o.data for o in D.objects if o.get("role") == "cell" and o.name.startswith(("Pier_E1_pier", "Pier_E1_arch"))]
POOL_SLAB = [o.data for o in D.objects if o.get("role") == "cell" and o.name.startswith(("Vault_cell", "VaultRib0"))]
POOL_FRAG = [frag_mesh(f"V3Frag{k}", 500 + k, rng.uniform(0.12, 0.3)) for k in range(14)]
POOL_BLOCK = [block_mesh(f"V3Block{k}", 600 + k, (rng.uniform(0.45, 0.8), rng.uniform(0.3, 0.5), rng.uniform(0.26, 0.34))) for k in range(8)]
POOL_BIG = [block_mesh(f"V3Cornice{k}", 700 + k, (rng.uniform(1.0, 1.5), rng.uniform(0.55, 0.75), rng.uniform(0.4, 0.55)), 0.04) for k in range(4)]


def mesh_radius(me):
    co = mesh_co(me)
    return float(np.linalg.norm(co - (co.min(0) + co.max(0)) / 2, axis=1).max()), (co.min(0) + co.max(0)) / 2


SITES = []        # (x, y, radius, count, kind, max_r)
for c in COVER:   # heaps at the foot of every cover piece, mostly on the threat side (rubble thrown against it)
    cen, thin = c["centre"], c["thin"]
    for o in c["objs"]:
        p = Vector(o.matrix_world.translation.xy)
        SITES.append((p.x - thin.x * 0.95, p.y - thin.y * 0.95, 0.75, 7, "heap", 0.45))
        SITES.append((p.x + thin.x * 0.9, p.y + thin.y * 0.9, 0.6, 3, "frag", 0.25))
for s in (1, -1):
    for y in np.arange(2.0, 48.5, 3.1):                       # along the aisle walls
        SITES.append((s * 19.25, y + rng.uniform(-0.6, 0.6), 0.6, 5, "heap", 0.4))
    for y in COL_Y:                                           # pier bases, aisle + band sides
        SITES.append((s * 16.0, y + rng.uniform(-1.6, 1.6), 0.7, 6, "heap", 0.45))
        SITES.append((s * 13.0, y - 3.2 + rng.uniform(-0.3, 0.3), 0.6, 4, "frag", 0.3))
        SITES.append((s * 13.0, y + 3.2 + rng.uniform(-0.3, 0.3), 0.6, 4, "frag", 0.3))
    for y in (6.0, 21.0, 34.5, 44.0):                        # big fallen cornice blocks in the aisles
        SITES.append((s * 17.6 + rng.uniform(-0.6, 0.6), y + rng.uniform(-1.5, 1.5), 0.6, 1, "big", 1.2))
    SITES.append((s * 7.5, 0.33, 2.6, 6, "frag", 0.16))       # inside the south wall, either side of the gate
    for yy in STRIPS:                                         # small debris in the strips between dragon rows
        SITES.append((s * 9.0, (yy[0] + yy[1]) / 2, 1.2, 4, "frag", 0.2))
        SITES.append((s * 3.2, (yy[0] + yy[1]) / 2, 0.8, 3, "frag", 0.18))

# collision world for spawning = static + cover + remaining loose rubble
ALL_SOLID = STATIC_NAMES + LOOSE_NAMES + [o.name for o in COVER_OBJS]
SOLID_BVH = bvh_of([TRIS[n] for n in ALL_SOLID if n in TRIS])
CROUCH = [(c["crouch"], 0.55) for c in COVER if c["crouch"] is not None]


def blocked(p, r):
    hit = SOLID_BVH.find_nearest(p)
    if hit[0] is not None and (hit[3] < r + 0.02 or (p - hit[0]).dot(hit[1]) < 0):
        return True
    return False


spawned, DYN = [], []
n_skip = 0
for (sx_, sy_, rad, cnt, kind, max_r) in SITES:
    for k in range(cnt):
        u = rng.random()
        if kind == "big":
            me, sc = rng.choice(POOL_BIG), 1.0
        elif kind == "heap":
            if u < 0.35:
                me, sc = rng.choice(POOL_BLOCK), rng.uniform(0.8, 1.15)
            elif u < 0.6 and POOL_CELL:
                me, sc = rng.choice(POOL_CELL), rng.uniform(0.14, 0.32)
            elif u < 0.7 and POOL_SLAB:
                me, sc = rng.choice(POOL_SLAB), rng.uniform(0.1, 0.2)
            else:
                me, sc = rng.choice(POOL_FRAG), rng.uniform(0.8, 1.6)
        else:
            me, sc = rng.choice(POOL_FRAG), rng.uniform(0.6, 1.2)
        r0, mid = mesh_radius(me)
        if r0 * sc > max_r:
            sc = max_r / r0
        r = r0 * sc
        placed = False
        for _t in range(30):
            a, d = rng.uniform(0, 2 * math.pi), rad * math.sqrt(rng.random())
            p = Vector((sx_ + d * math.cos(a), sy_ + d * math.sin(a), F + r + 0.02 + rng.uniform(0, 0.25)))
            P = np.array([[p.x - r, p.y - r, F + 0.1], [p.x + r, p.y + r, F + 0.1], [p.x - r, p.y + r, F + 0.1], [p.x + r, p.y - r, F + 0.1]])
            if pts_in_proof(P, pad=0.25).any() or pts_in_aisle(P, pad=0.4).any():
                continue
            if any((Vector((p.x, p.y)) - cp).length < cr + r for cp, cr in CROUCH):
                continue
            if abs(p.x) > IX - r - 0.03 or p.y < IY0 + r + 0.03 or p.y > IY1 - r - 0.03:
                continue
            for _h in range(8):
                if all((p - q).length > r + rq + 0.02 for q, rq in spawned) and not blocked(p, r):
                    placed = True
                    break
                p.z += r * 0.8
            if placed and p.z < F + 2.2:
                break
            placed = False
        if not placed:
            n_skip += 1
            continue
        o = D.objects.new(f"Rubble3_{len(spawned):03d}", me)
        C_RUB.objects.link(o)
        # the mesh origin may not be its bbox centre (fracture cells): put the bbox centre at p
        rot = Euler((rng.uniform(0, 6.3), rng.uniform(0, 6.3), rng.uniform(0, 6.3))).to_matrix()
        o.matrix_world = Matrix.Translation(p - rot @ (Vector(mid) * sc)) @ rot.to_4x4() @ Matrix.Scale(sc, 4)
        spawned.append((p, r))
        DYN.append(o)
log(f"rubble spawned {len(spawned)}, skipped {n_skip}")
t = tick("rubble_spawn", t)


def settle(dyn, frames=180):
    """drop every loose piece onto ONE merged static collider (the hall's real geometry incl. cover + v1/v2 rubble) in a
    throw-away scene; keep the final rest pose. Mirrors the v2 settle (which proved stable)."""
    SIM = D.scenes.new("SIM_settle3")
    SIM.frame_start, SIM.frame_end = 1, frames
    win = bpy.context.window_manager.windows[0]
    main = win.scene
    win.scene = SIM
    with bpy.context.temp_override(window=win, scene=SIM):
        bpy.ops.rigidbody.world_add()
    rbw = SIM.rigidbody_world
    rbw.substeps_per_frame, rbw.solver_iterations, rbw.use_split_impulse = 20, 15, True
    rbw.point_cache.frame_start, rbw.point_cache.frame_end = 1, frames
    rbc = D.collections.new("SIM_rb3")
    SIM.collection.children.link(rbc)
    rbw.collection = rbc

    def make_rb(ob, typ, shape, **kw):
        rbc.objects.link(ob)
        with bpy.context.temp_override(window=win, scene=SIM, object=ob, active_object=ob, selected_objects=[ob]):
            bpy.ops.rigidbody.object_add(type=typ)
        ob.rigid_body.collision_shape = shape
        ob.rigid_body.use_margin = True
        ob.rigid_body.collision_margin = 0.004
        for k_, v_ in kw.items():
            setattr(ob.rigid_body, k_, v_)
    # static: one mesh in world space (flames excluded) + the floor
    # only colliders a piece can reach (bbox within 1.5 m of a spawn): the big scans elsewhere just cost bake time
    SP = np.array([[p.x, p.y] for p, _r in spawned])
    names = []
    for n in ALL_SOLID:
        if n not in TRIS or "lame" in n:
            continue
        lo_, hi_ = AABB.get(n) or (TRIS[n][0].min(0), TRIS[n][0].max(0))
        if lo_[2] > F + 3.0:
            continue
        if ((SP[:, 0] > lo_[0] - 1.5) & (SP[:, 0] < hi_[0] + 1.5) & (SP[:, 1] > lo_[1] - 1.5) & (SP[:, 1] < hi_[1] + 1.5)).any():
            names.append(n)
    big = sorted(names, key=lambda n: -len(TRIS[n][1]))[:5]
    log("sim colliders", len(names), "of", len(ALL_SOLID), "largest:", [(n, len(TRIS[n][1])) for n in big])
    parts = [TRIS[n] for n in names]
    vs, ts, off = [], [], 0
    for w, t_ in parts:
        vs.append(w)
        ts.append(t_ + off)
        off += len(w)
    V, Tt = np.concatenate(vs), np.concatenate(ts)
    me = D.meshes.new("SIMP_static3")
    me.from_pydata(V.tolist(), [], Tt.tolist())
    st = D.objects.new("SIMP_static3", me)
    make_rb(st, "PASSIVE", "MESH", friction=0.9, restitution=0.0)
    fb = bmesh.new()
    block_verts(fb, (-IX - 1, -(IY1 - IY0) / 2 - 1, -0.5), (IX + 1, (IY1 - IY0) / 2 + 1, 0.5), 0.0)   # centred: a
    fme = to_mesh(fb, "SIMP_floor3", [])                                                               # BOX shape is
    fo = D.objects.new("SIMP_floor3", fme)                                                             # built around
    fo.location = (0.0, (IY0 + IY1) / 2, F - 0.5)                                                      # the ORIGIN
    make_rb(fo, "PASSIVE", "BOX", friction=0.9, restitution=0.0)
    sims = []
    for o in dyn:
        so = D.objects.new("SIMD_" + o.name, o.data)
        so.matrix_world = o.matrix_world.copy()
        dims = Vector(o.dimensions)
        vol = max(dims.x * dims.y * dims.z * 0.5, 0.005)
        make_rb(so, "ACTIVE", "CONVEX_HULL", mass=2400.0 * vol, friction=0.85, restitution=0.0,
                linear_damping=0.4, angular_damping=0.85)
        sims.append(so)
    t0 = time.time()
    with bpy.context.temp_override(window=win, scene=SIM):
        bpy.ops.ptcache.bake_all(bake=True)
    SIM.frame_set(frames - 12)
    before = [so.matrix_world.translation.copy() for so in sims]
    SIM.frame_set(frames)
    out = [(o, so.matrix_world.copy(), (so.matrix_world.translation - b).length) for o, so, b in zip(dyn, sims, before)]
    log(f"settle: {len(sims)} bodies x {frames} frames, static tris {len(Tt)}, bake {time.time() - t0:.1f} s")
    with bpy.context.temp_override(window=win, scene=SIM):
        bpy.ops.ptcache.free_bake_all()
    win.scene = main
    for o in sims + [st, fo]:
        D.objects.remove(o)
    D.meshes.remove(me)
    D.meshes.remove(fme)
    D.collections.remove(rbc)
    D.scenes.remove(SIM)
    return out


n_moving = n_fell = n_proof = n_aisle = n_crouch = n_float = 0
if not A.nosim and DYN:
    res = settle(DYN)
    for o, M, mv in res:
        o.matrix_world = M
    bpy.context.view_layer.update()
    for o, M, mv in res:
        w = world_co(o)
        why = None
        if w[:, 2].min() < F - 0.08:
            why = "fell"
        elif pts_in_proof(w).any():
            why = "proof"
        elif pts_in_aisle(w).any():
            why = "aisle"
        elif any(np.hypot(w[:, 0] - cp.x, w[:, 1] - cp.y).min() < 0.4 and w[:, 2].min() < F + 1.0 for cp, _ in CROUCH):
            why = "crouch"
        if why:
            n_fell += why == "fell"
            n_proof += why == "proof"
            n_aisle += why == "aisle"
            n_crouch += why == "crouch"
            DYN.remove(o)
            D.objects.remove(o)
            continue
        if mv > 0.03:
            n_moving += 1
else:     # no sim: drop each piece straight down onto the floor (layout preview only)
    for o in DYN:
        w = world_co(o)
        o.location.z -= float(w[:, 2].min()) - F
REPORT["checks"]["rubble_settle_v3"] = dict(spawned=len(spawned), kept=len(DYN), still_moving_gt_3cm=n_moving,
                                            fell_removed=n_fell, rolled_into_dragon_floor_removed=n_proof,
                                            rolled_into_main_aisle_removed=n_aisle, rolled_into_crouch_spot_removed=n_crouch,
                                            sim=not A.nosim)
log("settle", REPORT["checks"]["rubble_settle_v3"])
t = tick("rubble_settle", t)


# ---------------------------------------------------------------------------------------------------------
# =========================================  INTERIOR — 3. GRAVEL  ========================================
# ---------------------------------------------------------------------------------------------------------
def chip_template(seed):
    r_ = random.Random(seed)
    pts = [(r_.uniform(-1, 1), r_.uniform(-0.8, 0.8), r_.uniform(-0.5, 0.5)) for _ in range(10)]
    bm = bmesh.new()
    for p_ in pts:
        bm.verts.new(p_)
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:])
    bmesh.ops.delete(bm, geom=list({g for g in res["geom_interior"] + res["geom_unused"] if isinstance(g, bmesh.types.BMVert)}), context="VERTS")
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    co = np.array([v.co[:] for v in bm.verts])
    vi = {v: i for i, v in enumerate(bm.verts)}
    fs = np.array([[vi[v] for v in f.verts] for f in bm.faces])
    bm.free()
    return co - co.mean(0), fs


t = time.time()
CHIPS = [chip_template(800 + k) for k in range(10)]
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
GRAV_BVH = bvh_of([TRIS[n] for n in ALL_SOLID if n in TRIS and n in D.objects] + [tris_world(o, dg) for o in DYN])
ZONES = []    # (name, x0, x1, y0, y1, density /m^2)
for s, nm in ((1, "E"), (-1, "W")):
    ZONES.append((f"Band{nm}", *sorted((s * 10.7, s * 15.3)), 0.05, 49.95, 2.2))
    ZONES.append((f"Aisle{nm}", *sorted((s * 15.3, s * 19.95)), 0.05, 49.95, 2.6))
    for k, (a, b) in enumerate(STRIPS):
        ZONES.append((f"Strip{k}{nm}", *sorted((s * (AISLE_HALF + 0.3), s * 10.6)), a + 0.06, b - 0.06, 2.0))
    ZONES.append((f"South{nm}", *sorted((s * (AISLE_HALF + 0.3), s * 10.6)), 0.05, float(PROOF_ARR[:, 1].min()) - 0.06, 2.0))
heap_xy = np.array([[o.matrix_world.translation.x, o.matrix_world.translation.y] for o in DYN]) if DYN else np.zeros((0, 2))
n_chips_total = 0
for nm, x0, x1, y0, y1, dens in ZONES:
    area = (x1 - x0) * (y1 - y0)
    n_try = int(area * dens * 4.0)
    V, Fc, off, n_ok = [], [], 0, 0
    for _ in range(n_try):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if len(heap_xy):                                       # denser near the rubble heaps, sparse in the open
            dmin = float(np.min(np.hypot(heap_xy[:, 0] - x, heap_xy[:, 1] - y)))
            if rng.random() > 0.25 + 0.75 * math.exp(-dmin / 1.2):
                continue
        r = rng.uniform(0.02, 0.09) if rng.random() < 0.85 else rng.uniform(0.09, 0.14)
        P = np.array([[x - r, y - r, F + 0.05], [x + r, y + r, F + 0.05], [x - r, y + r, F + 0.05], [x + r, y - r, F + 0.05]])
        if pts_in_proof(P, pad=0.03).any() or pts_in_aisle(P, pad=0.1).any():
            continue
        ok = True
        for ox, oy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)):
            loc = GRAV_BVH.ray_cast(Vector((x + ox, y + oy, F + 2.5)), Vector((0, 0, -1)), 3.0)[0]
            if loc is not None and loc.z > F + 0.003:      # something stands here (the floor itself is not in the BVH)
                ok = False
                break
        if not ok:
            continue
        co, fs = CHIPS[rng.randrange(len(CHIPS))]
        R = np.array(Euler((rng.uniform(0, 6.3), rng.uniform(0, 6.3), rng.uniform(0, 6.3))).to_matrix())
        w = (co * np.array([r, r, r * rng.uniform(0.6, 1.0)])) @ R.T
        w = w + np.array([x, y, F - w[:, 2].min() - 0.002])     # 2 mm bedded into the dust
        V.append(w)
        Fc.append(fs + off)
        off += len(w)
        n_ok += 1
    if not n_ok:
        continue
    me = D.meshes.new(f"V3Gravel_{nm}")
    me.from_pydata(np.concatenate(V).tolist(), [], np.concatenate(Fc).tolist())
    me.materials.append(M_ASH)
    o = D.objects.new(f"V3Gravel_{nm}", me)
    C_GRAV.objects.link(o)
    o.visible_shadow = False           # 2-14 cm chips: shadow pages for nothing (render-cost A/B)
    n_chips_total += n_ok
REPORT["added"]["gravel_chips"] = n_chips_total
log("gravel chips", n_chips_total)
t = tick("gravel", t)


def light(name, typ, loc, target, energy, color, size=0.5, c=None):
    ld = D.lights.new(name, typ)
    ld.energy = energy
    ld.color = color
    ld.shadow_soft_size = size
    o = D.objects.new(name, ld)
    (c or SC.collection).objects.link(o)
    o.location = loc
    return o


def boxes(name, specs, mat, c):
    bm = bmesh.new()
    for lo, hi in specs:
        block_verts(bm, lo, hi, 0.0)
    o = D.objects.new(name, to_mesh(bm, name, [mat]))
    c.objects.link(o)
    return o


# ---------------------------------------------------------------------------------------------------------
# =========================================  INTERIOR — 4. HOARD  =========================================
# (Zeke 10-14 phase 2) 3 new chests (2 OPEN, coins + gems spilling down the dais), the 2 v1 chests re-seated ON the gold,
# a Zweihaender and the gold angel sword planted in the two gold mounds, bow + holy greatsword + mace + war hammer on
# iron wall mounts on the north wall. Everything stays out of the e_prize frame (|x| >= 3.5 on the wall, chests |x| >= 3.2)
# so the RPG-7 stays the hero.
# ---------------------------------------------------------------------------------------------------------
MED = "/data/iris_assets/medieval/"
R2P, R4P, PHM = MED + "r2_props/", MED + "r4_v3b/", MED + "polyhaven/"


def import_glb(path):
    before = set(D.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in D.objects if o not in before]
    for o in new:
        for u in list(o.users_collection):
            u.objects.unlink(o)
        C_TMP.objects.link(o)
    bpy.context.view_layer.update()
    return new


def merge_world(objs, name, keep=None):
    """mesh objects -> ONE mesh in world space; materials de-duplicated; UVs kept. Removes the source objects."""
    bm = bmesh.new()
    mats = []
    for o in [o for o in objs if o.type == "MESH" and (keep is None or keep(o))]:
        me = o.data.copy()
        me.transform(o.matrix_world)
        idx = []
        for m in me.materials:
            if m not in mats:
                mats.append(m)
            idx.append(mats.index(m))
        if idx and len(me.polygons):
            mi = np.zeros(len(me.polygons), np.int32)
            me.polygons.foreach_get("material_index", mi)
            me.polygons.foreach_set("material_index", np.array(idx, np.int32)[np.clip(mi, 0, len(idx) - 1)])
        bm.from_mesh(me)
        D.meshes.remove(me)
    out = D.meshes.new(name)
    bm.to_mesh(out)
    bm.free()
    for m in mats:
        out.materials.append(m)
    for o in objs:
        if o.name in D.objects:
            D.objects.remove(o)
    return out


def orient(me, src_long, src_norm, dst_long, dst_norm):
    """rotate mesh so its src_long axis -> dst_long and src_norm -> dst_norm (orthonormal frames)."""
    def frame(a, b):
        a = Vector(a).normalized()
        b = (Vector(b) - a * a.dot(Vector(b))).normalized()
        return Matrix((a, b, a.cross(b))).transposed()
    R = frame(dst_long, dst_norm) @ frame(src_long, src_norm).inverted()
    me.transform(R.to_4x4())


def gn_surface_coins(o):
    """reuse the v1 GN_Coins tree (Distribute on the 'pile' attribute, Set Material on the coin template)."""
    o.modifiers.new("Coins", "NODES").node_group = D.node_groups["GN_Coins"]


C_HOARD3 = coll("V3_HOARD", C_V3)
t = time.time()
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
_heap = D.objects["GoldHeap"]
_hme = _heap.data
_hme.calc_loop_triangles()
_ht = np.empty(len(_hme.loop_triangles) * 3, np.int32)
_hme.loop_triangles.foreach_get("vertices", _ht)
HOARD_BVH = bvh_of([(world_co(_heap), _ht.reshape(-1, 3)), tris_world(D.objects["Dais"], dg), tris_world(D.objects["ArenaFloor"], dg)])


def surf(x, y, bvh=None):
    loc = (bvh or HOARD_BVH).ray_cast(Vector((x, y, F + 8.0)), Vector((0, 0, -1)), 12.0)[0]
    return float(loc.z) if loc is not None else F


M_COINS = D.materials["M_GoldCoins"]
CH_S = 1.4                                                     # v1 chest scale (0.96 x 0.52 x 0.62 m scan -> 1.34 m)
ch_objs = import_glb(PHM + "model_treasure_chest/treasure_chest_2k.gltf")
CH = {o.name.split(".")[0]: o for o in ch_objs if o.type == "MESH"}
ME_CLOSED = merge_world([o.copy() for o in CH.values()], "V3Chest_closed")
ME_BODY = merge_world([CH[k].copy() for k in CH if k != "treasure_chest_lid"], "V3Chest_body")
ME_LID = merge_world([CH["treasure_chest_lid"].copy()], "V3Chest_lid_open")
drop_ = [o for o in ch_objs if o.name in D.objects]
for o in drop_:
    D.objects.remove(o)
HINGE = Vector((0.0, 0.26, 0.42))                             # lid hinge: back top edge (lock/front is -Y)
ME_LID.transform(Matrix.Translation(HINGE) @ Matrix.Rotation(math.radians(-105), 4, "X") @ Matrix.Translation(-HINGE))
for me in (ME_CLOSED, ME_BODY, ME_LID):
    me.transform(Matrix.Scale(CH_S, 4))
HW, HD, RIM = 0.48 * CH_S, 0.26 * CH_S, 0.42 * CH_S


def chest_seat(x, y, rz):
    """z for a chest resting on the gold: mean of the coin surface under its four corners + centre."""
    c, s = math.cos(rz), math.sin(rz)
    pts = [(x + c * u - s * v, y + s * u + c * v) for u, v in ((-HW, -HD), (HW, -HD), (-HW, HD), (HW, HD), (0, 0))]
    zs = [surf(px, py) for px, py in pts]
    return float(np.mean(zs)), zs


def grid_patch(name, fn, u0, u1, v0, v1, step, M, pile_fn, c=C_HOARD3, mat=M_COINS):
    """height-field patch in a local (u, v) frame -> world by M; fn(u, v) -> world z or None (hole); pile attr for GN."""
    us, vs = np.arange(u0, u1 + 1e-6, step), np.arange(v0, v1 + 1e-6, step)
    idx, V, P = {}, [], []
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            w = M @ Vector((u, v, 0))
            z = fn(u, v, w)
            if z is None:
                continue
            idx[(i, j)] = len(V)
            V.append((w.x, w.y, z))
            P.append(pile_fn(u, v))
    Fc = [(idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]) for j in range(len(vs) - 1) for i in range(len(us) - 1)
          if all(k in idx for k in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))]
    me = D.meshes.new(name)
    me.from_pydata(V, [], Fc)
    a = me.attributes.new("pile", "FLOAT", "POINT")
    a.data.foreach_set("value", np.array(P, np.float32))
    me.materials.append(mat)
    me.shade_smooth()
    o = D.objects.new(name, me)
    c.objects.link(o)
    gn_surface_coins(o)
    o.visible_shadow = False           # coin skins/domes/spills: no shadow casting (render-cost A/B)
    return o


CHESTS = []
# v1 chests: re-seat on the top tier (they were sunk 0.6 m into the dais steps), closed
for nm, (x, y, rzd) in (("Prop_treasure_chest", (-6.9, 47.9, 24.0)), ("Prop_treasure_chest.001", (6.8, 48.3, -18.0))):
    root = D.objects[nm]
    z, zs = chest_seat(x, y, math.radians(rzd))
    root.location = (x, y, z)
    root.rotation_euler = (0, 0, math.radians(rzd))
    CHESTS.append(dict(name=nm, open=False, x=x, y=y, rz=math.radians(rzd), z=z, spread=max(zs) - min(zs)))
for nm, (x, y, rzd), opened in (("V3Chest_OpenW", (-3.3, 47.65, -10.0), True), ("V3Chest_OpenE", (3.85, 47.7, 14.0), True),
                                ("V3Chest_ClosedNW", (-5.75, 48.95, 30.0), False)):
    rz = math.radians(rzd)
    z, zs = chest_seat(x, y, rz)
    if opened:
        new_obj(nm + "_body", ME_BODY, C_HOARD3, (x, y, z), rz)
        new_obj(nm + "_lid", ME_LID, C_HOARD3, (x, y, z), rz)
    else:
        new_obj(nm, ME_CLOSED, C_HOARD3, (x, y, z), rz)
    CHESTS.append(dict(name=nm, open=opened, x=x, y=y, rz=rz, z=z, spread=max(zs) - min(zs)))

# coin skirts (every chest nestles in the coins: no gap under a corner) + domes and spills for the open ones
for ch in CHESTS:
    M = Matrix.Translation((ch["x"], ch["y"], 0)) @ Matrix.Rotation(ch["rz"], 4, "Z")
    base = ch["z"]
    R_ = 0.4

    def skirt_z(u, v, w, base=base):
        du, dv = max(0.0, abs(u) - HW), max(0.0, abs(v) - HD)
        d = math.hypot(du, dv)
        if d <= 0.0 or d > R_:
            return None if d > R_ else None
        hrel = (1 - d / R_) ** 1.6
        return max(surf(w.x, w.y), base + 0.16 * hrel) + 0.01
    grid_patch(ch["name"] + "_CoinSkirt", skirt_z, -HW - R_, HW + R_, -HD - R_, HD + R_, 0.05, M,
               lambda u, v: 0.25 * max(0.2, 1 - max(0.0, math.hypot(max(0.0, abs(u) - HW), max(0.0, abs(v) - HD))) / R_))
    if not ch["open"]:
        continue
    # dome heaped above the rim (inside the walls: 3 cm wall allowance)
    iw, idp = HW - 0.04, HD - 0.04
    grid_patch(ch["name"] + "_CoinDome",
               lambda u, v, w, base=base: base + RIM + 0.02 + 0.2 * CH_S * max(0.0, 1 - (u / iw) ** 2) * max(0.0, 1 - (v / idp) ** 2),
               -iw, iw, -idp, idp, 0.04, M, lambda u, v: 0.25)
    # spill: over the front wall and down the gold towards the nave (local -Y), fanning out
    L_ = 1.9 if ch["x"] < 0 else 1.6

    def spill_z(u, v, w, base=base, L_=L_):
        s = -(v + HD)                                         # distance in front of the chest front face
        width = 0.32 + 0.55 * max(0.0, s) / L_
        if abs(u) > width or s > L_:
            return None
        g = surf(w.x, w.y)
        if s < 0.0:                                           # still over the chest: ride over the front rim
            return base + RIM + 0.04
        edge = 1 - (abs(u) / width) ** 2
        slide = max(0.0, 1 - s / 0.5)                         # the cascade down the chest front (first 0.5 m)
        return max(g + 0.02 + 0.09 * edge * (1 - s / L_), g + (RIM + 0.03) * slide * edge)
    grid_patch(ch["name"] + "_CoinSpill", spill_z, -0.9, 0.9, -HD - L_, -HD + 0.04, 0.05, M,
               lambda u, v, L_=L_: 0.25 * max(0.15, 1 - (abs(u) / (0.32 + 0.55 * max(0.0, -(v + HD)) / L_)) ** 2))

# gems: the 21-cut set (CC-BY), each normalised to 1 unit, then scattered on the domes / spills / heap
gem_objs = import_glb(R2P + "prop_gems_21cuts_blue_odym/model.glb")
GEMS = []
for k, o in enumerate([o for o in gem_objs if o.type == "MESH" and o.data.materials]):
    me = merge_world([o], f"V3Gem{k}")
    co = mesh_co(me)
    c = (co.min(0) + co.max(0)) / 2
    me.transform(Matrix.Scale(1.0 / float((co.max(0) - co.min(0)).max()), 4) @ Matrix.Translation(Vector(-c)))
    GEMS.append(me)
# gem materials ship BLENDED + ray-traced refraction: measured +4.5 s on EVERY still (even with no gem in view).
# At 3-6 cm on a coin pile a glossy faceted dielectric reads the same: dithered, no transmission, no refraction.
GEM_FIX = []
for me in GEMS:
    for m in me.materials:
        if m and m.name not in GEM_FIX:
            b = principled(m)
            if b is not None:
                b.inputs["Transmission Weight"].default_value = 0.0
                b.inputs["Roughness"].default_value = 0.04
                b.inputs["Specular IOR Level"].default_value = 1.0
                b.inputs["IOR"].default_value = 2.0
            m.surface_render_method = "DITHERED"
            try:
                m.use_raytrace_refraction = False
            except AttributeError:
                pass
            GEM_FIX.append(m.name)
REPORT["added"]["gem_materials_made_cheap"] = GEM_FIX
for o in gem_objs:
    try:
        D.objects.remove(o)
    except ReferenceError:
        pass
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
COIN_BVH = bvh_of([tris_world(o, dg) for o in C_HOARD3.objects if o.type == "MESH" and "Coin" in o.name])
n_gems = 0
rg = random.Random(4242)
for ch in [c_ for c_ in CHESTS if c_["open"]]:
    M = Matrix.Translation((ch["x"], ch["y"], 0)) @ Matrix.Rotation(ch["rz"], 4, "Z")
    pts = [(rg.uniform(-HW + 0.08, HW - 0.08), rg.uniform(-HD + 0.08, HD - 0.08)) for _ in range(14)]
    pts += [(rg.uniform(-0.5, 0.5), -HD - rg.uniform(0.2, 1.5)) for _ in range(10)]
    for u, v in pts:
        w = M @ Vector((u, v, 0))
        loc = COIN_BVH.ray_cast(Vector((w.x, w.y, F + 8)), Vector((0, 0, -1)), 12.0)[0]
        if loc is None:
            continue
        sz = rg.uniform(0.035, 0.065)
        g = new_obj(f"V3Gem_{n_gems:03d}", rg.choice(GEMS), C_HOARD3, (w.x, w.y, loc.z + sz * 0.18), 0)
        g.rotation_euler = (rg.uniform(-0.5, 0.5), rg.uniform(-0.5, 0.5), rg.uniform(0, 6.3))
        g.scale = (sz,) * 3
        g.visible_shadow = False
        n_gems += 1
for _ in range(16):                                            # loose gems on the main heap near the chests
    x, y = rg.uniform(-6.5, 6.5), rg.uniform(45.6, 47.4)
    if abs(x) < 1.6:
        continue
    sz = rg.uniform(0.035, 0.06)
    g = new_obj(f"V3Gem_{n_gems:03d}", rg.choice(GEMS), C_HOARD3, (x, y, surf(x, y) + sz * 0.15), 0)
    g.rotation_euler = (rg.uniform(-0.5, 0.5), rg.uniform(-0.5, 0.5), rg.uniform(0, 6.3))
    g.scale = (sz,) * 3
    g.visible_shadow = False
    n_gems += 1

# planted swords: Zweihaender (CC-BY leeeck) in the west mound, the gold angel sword (CC-BY Bl4ckGh0st) in the east
WEAP = {}
zw = import_glb(R2P + "prop_zweihander_leeeck/model.glb")
me = merge_world(zw, "V3_Zweihander")
co = mesh_co(me)
me.transform(Matrix.Scale(1.81 / float(co[:, 1].max() - co[:, 1].min()), 4))      # measured long axis = Y, tip at +Y
orient(me, (0, 1, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0))                             # tip down, flat facing the nave
co = mesh_co(me)
me.transform(Matrix.Translation((-(co[:, 0].min() + co[:, 0].max()) / 2, -(co[:, 1].min() + co[:, 1].max()) / 2, -co[:, 2].min())))
WEAP["V3_Zweihander"] = (me, (-4.55, 48.55), (math.radians(-7), math.radians(5), math.radians(14)))
an = import_glb(R2P + "prop_holy_sword_gold_bl4ckgh0st/model.glb")
me = merge_world(an, "V3_AngelSword")
orient(me, (0, 0, 1), (0, 1, 0), (0, 0, -1), (0, 1, 0))                             # tip (+Z, 1.94 m) down
co = mesh_co(me)
me.transform(Matrix.Translation((-(co[:, 0].min() + co[:, 0].max()) / 2, -(co[:, 1].min() + co[:, 1].max()) / 2, -co[:, 2].min())))
WEAP["V3_AngelSword"] = (me, (4.95, 48.45), (math.radians(6), math.radians(-6), math.radians(-12)))
PLANT = 0.38                                                   # blade buried this deep below the coin surface
for nm, (me, (x, y), rot) in WEAP.items():
    o = new_obj(nm, me, C_HOARD3, (x, y, 0), 0)
    o.rotation_euler = rot
    o.location.z = surf(x, y) - PLANT                         # mesh origin = blade tip
    REPORT["added"].setdefault("hoard_weapons", {})[nm] = dict(planted_depth_m=PLANT, tip_z=round(o.location.z, 2),
                                                               surface_z=round(surf(x, y), 2))


# wall displays on the north wall face (y = 50): iron pegs + weapons hung 3 cm off the stone
BOARD_T = 0.06                                                 # oak display board thickness (on the wall face)


def wall_hang(name, me, cx, cz, c=C_HOARD3):
    co = mesh_co(me)
    me.transform(Matrix.Translation((-(co[:, 0].min() + co[:, 0].max()) / 2, -co[:, 1].max(), -(co[:, 2].min() + co[:, 2].max()) / 2)))
    return new_obj(name, me, c, (cx, IY1 - BOARD_T - 0.03, cz), 0)


M_OAK = D.materials.new("M_V3DisplayOak")
M_OAK.use_nodes = True
_ob = principled(M_OAK)
_ob.inputs["Base Color"].default_value = (0.05, 0.03, 0.018, 1)
_ob.inputs["Roughness"].default_value = 0.6
for _nm, _cx, _w, _h, _cz in (("V3_DisplayBoard_W", -5.6, 2.0, 1.45, F + 3.35), ("V3_DisplayBoard_E", 5.6, 1.3, 1.3, F + 3.35)):
    _bm = bmesh.new()
    block_verts(_bm, (_cx - _w / 2, IY1 - BOARD_T, _cz - _h / 2), (_cx + _w / 2, IY1, _cz + _h / 2), 0.015)
    block_verts(_bm, (_cx - _w / 2 - 0.05, IY1 - BOARD_T - 0.03, _cz - _h / 2 - 0.05), (_cx + _w / 2 + 0.05, IY1 - BOARD_T, _cz - _h / 2 + 0.03), 0.01)
    new_obj(_nm, to_mesh(_bm, _nm, [M_OAK]), C_HOARD3)
    _l = D.lights.new(_nm + "_Key", "AREA")
    _l.energy, _l.size, _l.color = 60.0, 0.6, (1.0, 0.72, 0.42)
    _l.use_shadow = False                      # measured: the two shadowed keys cost +2.5 s on every still
    _l.use_custom_distance, _l.cutoff_distance = True, 4.5
    _lo = D.objects.new(_nm + "_Key", _l)
    C_HOARD3.objects.link(_lo)
    _lo.location = (_cx * 0.82, IY1 - 1.9, _cz + 1.2)
    _lo.rotation_euler = (Vector((_cx, IY1, _cz)) - _lo.location).to_track_quat("-Z", "Y").to_euler()


def pegs(name, pts, item=None):
    """iron wall pegs; with item= each peg's top is raised/lowered to the item's lowest vertex above it (it rests)."""
    bm = bmesh.new()
    if item is not None:
        bpy.context.view_layer.update()
        w = world_co(item)
        log(name, "item x", round(float(w[:, 0].min()), 2), round(float(w[:, 0].max()), 2), "z", round(float(w[:, 2].min()), 2),
            round(float(w[:, 2].max()), 2), "y", round(float(w[:, 1].min()), 2), round(float(w[:, 1].max()), 2))
        pts = [(x, float(w[np.argsort(np.abs(w[:, 0] - x))[:12], 2].min()) - 0.06) for x, _z in pts]
    for x, z in pts:
        block_verts(bm, (x - 0.025, IY1 - BOARD_T - 0.2, z - 0.07), (x + 0.025, IY1 - BOARD_T, z - 0.02), 0.006)
        block_verts(bm, (x - 0.025, IY1 - BOARD_T - 0.2, z - 0.02), (x + 0.025, IY1 - BOARD_T - 0.16, z + 0.06), 0.006)
    return new_obj(name, to_mesh(bm, name, [M_IRON] if "M_Iron" in D.materials else [M_ASH]), C_HOARD3)


M_IRON = D.materials.get("M_Iron")
bw = import_glb(R4P + "prop_bow_cgdude0007/model.glb")
me = merge_world(bw, "V3_Bow", keep=lambda o: bool(o.data.materials) and o.data.materials[0] is not None
               and o.data.materials[0].name.split(".")[0] in ("material", "string", "Metallic_part_of_Bow"))   # bow, string, fittings (not the arrows)
co = mesh_co(me)
me.transform(Matrix.Scale(1.62 / float(co[:, 2].max() - co[:, 2].min()), 4))    # 1.62 m bow (long axis = Z)
orient(me, (0, 0, 1), (1, 0, 0), (1, 0, 0), (0, -1, 0))                            # horizontal, its thin axis off the wall
_bo = wall_hang("V3_Bow_Wall", me, -5.6, F + 3.7)
pegs("V3_Pegs_Bow", [(-6.05, 0), (-5.15, 0)], _bo)
hg = import_glb(R2P + "prop_holy_greatsword_bl4ckgh0st/model.glb")
me = merge_world(hg, "V3_HolyGreatsword")
co = mesh_co(me)
me.transform(Matrix.Scale(1.45 / float(co[:, 1].max() - co[:, 1].min()), 4))
orient(me, (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, -1, 0))
_hs = wall_hang("V3_HolyGreatsword_Wall", me, -5.6, F + 2.95)
pegs("V3_Pegs_Sword", [(-6.0, 0), (-5.05, 0)], _hs)
for nm, path, ang, cx in (("V3_Mace_Wall", "model_ornate_medieval_mace/ornate_medieval_mace_2k.gltf", 32, 5.55),
                          ("V3_WarHammer_Wall", "model_ornate_war_hammer/ornate_war_hammer_2k.gltf", -32, 5.65)):
    ob = import_glb(PHM + path)
    me = merge_world(ob, nm)
    co = mesh_co(me)
    me.transform(Matrix.Scale(0.95 / float(co[:, 2].max() - co[:, 2].min()), 4))   # 0.95 m war mace / war hammer
    a = math.radians(ang)
    orient(me, (0, 0, 1), (0, 1, 0), (math.sin(a), 0, math.cos(a)), (0, -1, 0))
    _cr = wall_hang(nm, me, cx, F + 3.4)
pegs("V3_Pegs_Crossed", [(5.52, 0), (5.78, 0)], _cr)
for o in list(C_TMP.objects):
    D.objects.remove(o)
REPORT["added"]["hoard"] = dict(chests=[dict(name=c_["name"], open=c_["open"], seat_z=round(c_["z"], 2),
                                             coin_surface_spread_m=round(c_["spread"], 2)) for c_ in CHESTS],
                                gems=n_gems, weapons=["V3_Zweihander (planted W mound)", "V3_AngelSword (planted E mound)",
                                                      "V3_Bow_Wall", "V3_HolyGreatsword_Wall", "V3_Mace_Wall", "V3_WarHammer_Wall"])
log("hoard", REPORT["added"]["hoard"])
t = tick("hoard", t)


# ---------------------------------------------------------------------------------------------------------
# =========================================  v3c — COMPOUND FIX: unlit scan materials  ====================
# Zeke 10-14 (v3b notes): two courtyard buildings rendered almost WHITE. Cause (probed): their glTFs use
# KHR_materials_unlit, which Blender's importer builds as Emission(base colour) for camera rays + Transparent for every
# other ray -> self-lit at full texture value, ignores the moon, casts no shadow; with the exterior +0.9 exposure -> white.
# Fix: every material with that pattern becomes a lit Principled BSDF (scan albedo scaled to the neighbouring scans'
# measured mean, tiled rock detail normal, rough stone).
# ---------------------------------------------------------------------------------------------------------
def is_unlit(m):
    """the glTF-importer KHR_materials_unlit graph: an image texture feeding an Emission that is mixed with Transparent by
    'Is Camera Ray'. (Hand-made emissive materials — flames, candles — never have an image into Emission: left alone.)"""
    if not m or not m.node_tree:
        return False
    ty = {n.type for n in m.node_tree.nodes}
    if not ({"EMISSION", "LIGHT_PATH", "BSDF_TRANSPARENT", "MIX_SHADER"} <= ty) or "BSDF_PRINCIPLED" in ty:
        return False
    return any(l.from_node.type == "TEX_IMAGE" and l.to_node.type == "EMISSION" for l in m.node_tree.links)


def unlit_to_lit(m, target_mean=0.30):
    nt, N, L = m.node_tree, m.node_tree.nodes, m.node_tree.links
    tex = next(n for n in N if n.type == "TEX_IMAGE")
    mean = float(img_mean(tex.image).mean())
    for n in list(N):
        if n.type in ("EMISSION", "LIGHT_PATH", "BSDF_TRANSPARENT", "MIX_SHADER"):
            N.remove(n)
    out = next(n for n in N if n.type == "OUTPUT_MATERIAL")
    b = N.new("ShaderNodeBsdfPrincipled")
    mx = N.new("ShaderNodeMix")
    mx.data_type, mx.blend_type = "RGBA", "MULTIPLY"
    mx.inputs["Factor"].default_value = 1.0
    k = min(1.0, target_mean / max(mean, 1e-3))
    mx.inputs[7].default_value = (k, k, k, 1)
    L.new(tex.outputs["Color"], mx.inputs[6])
    L.new(mx.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.88
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / 1.2,) * 3
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    tn = N.new("ShaderNodeTexImage")
    tn.image = img(PH + "tex_rock_pitted_mossy/rock_pitted_mossy_nor_gl_2k.jpg", False)
    tn.projection, tn.projection_blend = "BOX", 0.3
    L.new(mp.outputs["Vector"], tn.inputs["Vector"])
    nm_ = N.new("ShaderNodeNormalMap")
    nm_.inputs["Strength"].default_value = 0.35
    L.new(tn.outputs["Color"], nm_.inputs["Color"])
    L.new(nm_.outputs["Normal"], b.inputs["Normal"])
    L.new(b.outputs["BSDF"], out.inputs["Surface"])
    m.surface_render_method = "DITHERED"
    return round(mean, 3), round(k, 3)


UNLIT = {}
for m in D.materials:
    if is_unlit(m):
        users = sorted({o.name for o in D.objects if o.type == "MESH" and any(s.material == m for s in o.material_slots)})
        mean, k = unlit_to_lit(m)
        UNLIT[m.name] = dict(objects=users, texture_mean=mean, albedo_scale=k)
REPORT["added"]["v3c_unlit_materials_fixed"] = UNLIT
log("v3c unlit -> lit", UNLIT)


# ---------------------------------------------------------------------------------------------------------
# =========================================  EXTERIOR  ====================================================
# Zeke 10-14: SMALL TOWN (not walled); the fight stays in the courtyard -> all of this is BACKDROP, sized for distance.
# Terrain v3 (valley from the gate down to the town, hills behind, sinking rim) · 2 buildings just outside the south wall
# · a dirt path to the town · village cottages along it · a stretch of woods (east) + woods on the hills · strip fields
# with hedges · the town (cottages + church spire, lit windows, street glow) · low drifting ground fog (bounded boxes).
# ---------------------------------------------------------------------------------------------------------
CXc, CY0c, CY1c, CWT = 50.0, -78.0, 86.0, 3.0
RECT = (-CXc - CWT, CY0c - CWT, CXc + CWT, CY1c + CWT)          # compound outer footprint
TOWN_C = Vector((85.0, -480.0))
CASTLE_C = Vector((0.0, 4.0))
PATH_CTRL = [(0.0, -81.6), (0.5, -112), (5, -150), (16, -200), (31, -255), (47, -315), (61, -375), (73, -430), (84, -468)]


def rect_dist(x, y):
    dx = np.maximum(np.maximum(RECT[0] - x, x - RECT[2]), 0)
    dy = np.maximum(np.maximum(RECT[1] - y, y - RECT[3]), 0)
    return np.hypot(dx, dy)


def seg_dist(x, y, A, B):
    ab = np.array(B) - np.array(A)
    t_ = np.clip(((x - A[0]) * ab[0] + (y - A[1]) * ab[1]) / float(ab @ ab), 0, 1)
    return np.hypot(x - (A[0] + t_ * ab[0]), y - (A[1] + t_ * ab[1])), t_


def smooth(e0, e1, x):
    t_ = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t_ * t_ * (3 - 2 * t_)


def catmull(pts, step):
    P = [Vector((p[0], p[1])) for p in pts]
    P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int((p2 - p1).length / step))
        for k in range(n):
            s = k / n
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3))
    out.append(P[-2])
    return out


def poly_dist(x, y, pl):
    """distance from points (arrays) to a polyline (list of Vector)."""
    d = np.full(np.shape(x), 1e9)
    for a, b in zip(pl[:-1], pl[1:]):
        dd, _ = seg_dist(x, y, (a.x, a.y), (b.x, b.y))
        d = np.minimum(d, dd)
    return d


def tex_mat(name, tex_dir, stem, tile, tint, rough=(0.75, 0.95), nstr=0.8, coord="Object", attr_tint=None, emis=None):
    """box-projected Poly Haven PBR (diff/rough/nor) with a tint; optional per-vertex 'hv' variation attribute."""
    m = D.materials.new(name)
    m.use_nodes = True
    nt, N, L = m.node_tree, m.node_tree.nodes, m.node_tree.links
    b = principled(m)
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / tile,) * 3
    L.new(tc.outputs[coord], mp.inputs["Vector"])

    def tx(f, col):
        t_ = N.new("ShaderNodeTexImage")
        t_.image = img(f"{tex_dir}/{stem}_{f}_2k.jpg", col)
        t_.projection = "BOX" if coord == "Object" else "FLAT"
        t_.projection_blend = 0.25
        L.new(mp.outputs["Vector"], t_.inputs["Vector"])
        return t_
    td = tx("diff", True)
    mx = N.new("ShaderNodeMix")
    mx.data_type, mx.blend_type = "RGBA", "MULTIPLY"
    mx.inputs["Factor"].default_value = 1.0
    L.new(td.outputs["Color"], mx.inputs[6])
    mx.inputs[7].default_value = (*tint, 1)
    col = mx.outputs[2]
    if attr_tint:
        at = N.new("ShaderNodeAttribute")
        at.attribute_name = "hv"
        ramp = N.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*attr_tint[0], 1)
        ramp.color_ramp.elements[1].color = (*attr_tint[1], 1)
        L.new(at.outputs["Fac"], ramp.inputs["Fac"])
        m2 = N.new("ShaderNodeMix")
        m2.data_type, m2.blend_type = "RGBA", "MULTIPLY"
        m2.inputs["Factor"].default_value = 1.0
        L.new(col, m2.inputs[6])
        L.new(ramp.outputs["Color"], m2.inputs[7])
        col = m2.outputs[2]
    L.new(col, b.inputs["Base Color"])
    if os.path.exists(f"{tex_dir}/{stem}_rough_2k.jpg"):
        tr = tx("rough", False)
        mr = N.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = rough
        L.new(tr.outputs["Color"], mr.inputs["Value"])
        L.new(mr.outputs["Result"], b.inputs["Roughness"])
    if os.path.exists(f"{tex_dir}/{stem}_nor_gl_2k.jpg"):
        tn = tx("nor_gl", False)
        nm_ = N.new("ShaderNodeNormalMap")
        nm_.inputs["Strength"].default_value = nstr
        L.new(tn.outputs["Color"], nm_.inputs["Color"])
        L.new(nm_.outputs["Normal"], b.inputs["Normal"])
    return m


def flat_mat(name, col, rough=0.8, emis=None, estr=0.0):
    m = D.materials.new(name)
    m.use_nodes = True
    b = principled(m)
    b.inputs["Base Color"].default_value = (*col, 1)
    b.inputs["Roughness"].default_value = rough
    if emis:
        b.inputs["Emission Color"].default_value = (*emis, 1)
        b.inputs["Emission Strength"].default_value = estr
    return m


EXT_HALF = [2000.0]


def build_exterior():
    T_EXT = time.time()
    t_ = time.time()
    C_EXT = coll("V3_EXTERIOR")
    C_TPL = coll("V3_EXT_TEMPLATES", C_EXT)
    C_WD = coll("V3_EXT_WOODS", C_EXT)
    C_RD = coll("V3_EXT_ROAD", C_EXT)
    C_FM = coll("V3_EXT_FARMS", C_EXT)
    C_TN = coll("V3_EXT_TOWN", C_EXT)
    C_FG = coll("V3_EXT_FOG", C_EXT)
    C_EL = coll("V3_EXT_LIGHTS", C_EXT)
    ER = REPORT["added"].setdefault("exterior", {})
    rx = random.Random(9090)

    # ---- terrain v3 (replaces the v1 Terrain mesh; same object, same M_Ground) ------------------------------------
    N_T, HALF = 481, 4500.0
    u_ = np.linspace(-1, 1, N_T)
    xs = HALF * u_ * (0.24 + 0.76 * u_ * u_)                     # ~10 m cells near the castle, ~70 m at the rim
    X, Y = np.meshgrid(xs, xs)
    xf, yf = X.ravel(), Y.ravel()
    nz1 = np.array([mathutils.noise.fractal(Vector((x / 420, y / 420, 0.3)), 0.6, 2.0, 4) for x, y in zip(xf, yf)])
    nz2 = np.array([mathutils.noise.noise(Vector((x / 130, y / 130, 1.7))) for x, y in zip(xf, yf)])
    fr = np.array([mathutils.noise.fractal(Vector((x / 300, y / 300, 4.1)), 0.55, 2.0, 3) for x, y in zip(xf, yf)])
    Dc, tt = seg_dist(xf, yf, CASTLE_C, TOWN_C)
    dr = rect_dist(xf, yf)
    dtown = np.hypot(xf - TOWN_C.x, yf - TOWN_C.y)
    floor_z = -0.35 - 9.0 * smooth(0.22, 0.8, tt)
    und = 1.1 * nz2 * smooth(25, 90, dr) * (0.25 + 0.75 * smooth(60, 160, dtown))
    hills = (80 * nz1 + 130 + 70 * smooth(900, 2200, Dc)) * np.clip((Dc - 210) / 800, 0, 1) ** 1.3
    rim = 1 - 0.85 * smooth(3600, 4450, np.maximum(np.abs(xf), np.abs(yf)))       # sinking rim far behind the skyline
    H = floor_z + und + hills * rim
    H = -0.35 + smooth(10, 55, dr) * (H + 0.35)                                     # dead flat by the walls
    terr = D.objects["Terrain"]
    old = terr.data
    tme = D.meshes.new("Terrain_v3")
    faces = [(i * N_T + j, i * N_T + j + 1, (i + 1) * N_T + j + 1, (i + 1) * N_T + j) for i in range(N_T - 1) for j in range(N_T - 1)]
    tme.from_pydata(np.stack([xf, yf, H], 1).tolist(), [], faces)
    for m in old.materials:
        tme.materials.append(m)
    tme.shade_smooth()
    FOREST = np.clip((fr - 0.0) / 0.25, 0, 1) * smooth(140, 260, Dc) * smooth(70, 120, dr)    # far woods = terrain tint
    fa = tme.attributes.new("forest", "FLOAT", "POINT")
    fa.data.foreach_set("value", FOREST.astype(np.float32))
    terr.data = tme
    D.meshes.remove(old)
    terr.visible_shadow = False        # measured: the 460k-tri terrain in the sun's shadow maps cost +3.6-5 s on EVERY still
    # M_Ground: darken + green where the (far) forest mask is -> distant woods read without instancing them
    mg = D.materials["M_Ground"]
    bg_ = principled(mg)
    src = bg_.inputs["Base Color"].links[0].from_socket
    at = mg.node_tree.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "forest"
    mxf = mg.node_tree.nodes.new("ShaderNodeMix")
    mxf.data_type = "RGBA"
    mg.node_tree.links.new(at.outputs["Fac"], mxf.inputs["Factor"])
    mg.node_tree.links.new(src, mxf.inputs[6])
    mxf.inputs[7].default_value = (0.012, 0.02, 0.012, 1)
    mg.node_tree.links.new(mxf.outputs[2], bg_.inputs["Base Color"])
    TV = np.stack([xf, yf, H], 1)
    TT = np.array([(a, b, c) for a, b, c, d in faces] + [(a, c, d) for a, b, c, d in faces])
    TER = BVHTree.FromPolygons(TV.tolist(), TT.tolist())

    def th(x, y):
        loc = TER.ray_cast(Vector((x, y, 3000.0)), Vector((0, 0, -1)), 6000.0)[0]
        assert loc is not None, ("terrain miss", x, y)
        return float(loc.z)
    ER["terrain"] = dict(grid=f"{N_T}x{N_T} over +-{HALF:.0f} m (~10 m cells near, ~70 m at the rim)", town_floor_z=round(th(*TOWN_C), 2),
                         gate_z=round(th(0, -84), 2), max_z=round(float(H.max()), 1),
                         edge_max_z=round(float(H[np.maximum(np.abs(xf), np.abs(yf)) > HALF - 10].max()), 1))
    EXT_HALF[0] = HALF
    t_ = tick("ext_terrain", t_)

    # ---- materials -------------------------------------------------------------------------------------------
    MP = MED + "polyhaven/"
    M_PATH = tex_mat("M_V3Path", MP + "tex_brown_mud_leaves_01", "brown_mud_leaves_01", 2.5, (0.2, 0.17, 0.14), rough=(0.8, 1.0))
    M_WALLT = tex_mat("M_V3TownWall", "/data/iris_assets/polyhaven/tex_painted_plaster_wall", "painted_plaster_wall", 3.0,
                      (0.4, 0.37, 0.32), attr_tint=((0.45, 0.4, 0.34), (0.9, 0.84, 0.74)))
    M_STONET = tex_mat("M_V3TownStone", MP + "tex_medieval_blocks_02", "medieval_blocks_02", 2.5, (0.55, 0.52, 0.48),
                       attr_tint=((0.6, 0.58, 0.55), (0.9, 0.85, 0.8)))
    M_ROOFT = tex_mat("M_V3TownRoof", MP + "tex_roof_slates_02", "roof_slates_02", 3.0, (0.5, 0.45, 0.42),
                      attr_tint=((0.35, 0.33, 0.33), (0.75, 0.45, 0.32)))
    M_TIMB = tex_mat("M_V3Timber", MP + "tex_medieval_wood", "medieval_wood", 1.5, (0.16, 0.12, 0.09))
    M_WLIT = flat_mat("M_V3WindowLit", (0.05, 0.03, 0.01), 0.4, emis=(1.0, 0.52, 0.17), estr=1.0)
    _b = principled(M_WLIT)                                    # per-window glow from the 'glow' attribute
    _at = M_WLIT.node_tree.nodes.new("ShaderNodeAttribute")
    _at.attribute_name = "glow"
    _mm = M_WLIT.node_tree.nodes.new("ShaderNodeMath")
    _mm.operation = "MULTIPLY"
    _mm.inputs[1].default_value = 14.0
    M_WLIT.node_tree.links.new(_at.outputs["Fac"], _mm.inputs[0])
    M_WLIT.node_tree.links.new(_mm.outputs[0], _b.inputs["Emission Strength"])
    M_WDARK = flat_mat("M_V3WindowDark", (0.015, 0.013, 0.012), 0.35)
    M_CONI = flat_mat("M_V3Conifer", (0.016, 0.03, 0.017), 0.85)
    M_DECI = flat_mat("M_V3Leaves", (0.03, 0.04, 0.018), 0.85)
    M_BARK = flat_mat("M_V3Bark", (0.035, 0.028, 0.022), 0.9)
    M_STRAW = flat_mat("M_V3Straw", (0.22, 0.17, 0.08), 0.9)

    def field_mat(name, base, stripe, contrast):
        m = D.materials.new(name)
        m.use_nodes = True
        nt, N, L = m.node_tree, m.node_tree.nodes, m.node_tree.links
        b = principled(m)
        uv = N.new("ShaderNodeUVMap")
        wv = N.new("ShaderNodeTexWave")
        wv.wave_type, wv.bands_direction = "BANDS", "X"
        wv.inputs["Scale"].default_value = stripe
        wv.inputs["Distortion"].default_value = 1.5
        wv.inputs["Detail"].default_value = 2.0
        L.new(uv.outputs["UV"], wv.inputs["Vector"])
        nz = N.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 0.08
        L.new(uv.outputs["UV"], nz.inputs["Vector"])
        mr = N.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1 - contrast, 1.0
        L.new(wv.outputs["Fac"], mr.inputs["Value"])
        mr2 = N.new("ShaderNodeMapRange")
        mr2.inputs["To Min"].default_value, mr2.inputs["To Max"].default_value = 0.7, 1.15
        L.new(nz.outputs["Fac"], mr2.inputs["Value"])
        ml = N.new("ShaderNodeMath")
        ml.operation = "MULTIPLY"
        L.new(mr.outputs["Result"], ml.inputs[0])
        L.new(mr2.outputs["Result"], ml.inputs[1])
        mx = N.new("ShaderNodeMix")
        mx.data_type, mx.blend_type = "RGBA", "MULTIPLY"
        mx.inputs["Factor"].default_value = 1.0
        mx.inputs[6].default_value = (*base, 1)
        cc = N.new("ShaderNodeCombineColor")
        for i in range(3):
            L.new(ml.outputs[0], cc.inputs[i])
        L.new(cc.outputs["Color"], mx.inputs[7])
        L.new(mx.outputs[2], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.92
        bp = N.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = 0.35
        L.new(wv.outputs["Fac"], bp.inputs["Height"])
        L.new(bp.outputs["Normal"], b.inputs["Normal"])
        return m
    FIELD_MATS = [field_mat("M_V3FieldPlough", (0.04, 0.03, 0.02), 0.9, 0.45),
                  field_mat("M_V3FieldStubble", (0.085, 0.07, 0.04), 1.4, 0.25),
                  field_mat("M_V3FieldCrop", (0.03, 0.038, 0.018), 1.1, 0.3),
                  field_mat("M_V3FieldPasture", (0.032, 0.038, 0.022), 0.2, 0.08)]

    # ---- the path + town streets: ribbons draped on the terrain ---------------------------------------------------
    PATH = catmull(PATH_CTRL, 1.5)
    STREETS = [PATH[-60:], catmull([(TOWN_C.x, TOWN_C.y), (118, -474), (150, -462), (182, -452)], 1.5),
               catmull([(TOWN_C.x, TOWN_C.y), (88, -515), (94, -555), (99, -595)], 1.5),
               catmull([(TOWN_C.x, TOWN_C.y), (60, -495), (30, -503), (2, -520)], 1.5),
               catmull([(TOWN_C.x, TOWN_C.y), (102, -505), (128, -530)], 1.5)]

    def ribbon(name, pl, width, c=C_RD):
        V, Fc, A_ = [], [], []
        offs = [-1.3, -1.0, -0.5, 0.0, 0.5, 1.0, 1.3]
        n_ok = 0
        for k, p in enumerate(pl):
            a = pl[min(k + 1, len(pl) - 1)] - pl[max(k - 1, 0)]
            nrm = Vector((-a.y, a.x)).normalized()
            w = width * (1 + 0.15 * mathutils.noise.noise(Vector((p.x / 9, p.y / 9, 2.2)))) / 2
            for o_ in offs:
                q = p + nrm * (o_ * w)
                edge = abs(o_) > 1.05
                V.append((q.x, q.y, th(q.x, q.y) + (0.015 if edge else 0.05)))
                A_.append(1.0 if edge else abs(o_) * 0.5)
            if k:
                b0, b1 = (k - 1) * len(offs), k * len(offs)
                for i in range(len(offs) - 1):
                    Fc.append((b0 + i, b0 + i + 1, b1 + i + 1, b1 + i))
            n_ok += 1
        me = D.meshes.new(name)
        me.from_pydata(V, [], Fc)
        at_ = me.attributes.new("verge", "FLOAT", "POINT")
        at_.data.foreach_set("value", np.array(A_, np.float32))
        me.materials.append(M_PATH)
        me.shade_smooth()
        o = D.objects.new(name, me)
        c.objects.link(o)
        return o
    # verge: path dirt fades into the ground colour at the ribbon edge
    _b = principled(M_PATH)
    _src = _b.inputs["Base Color"].links[0].from_socket
    _at = M_PATH.node_tree.nodes.new("ShaderNodeAttribute")
    _at.attribute_name = "verge"
    _mx = M_PATH.node_tree.nodes.new("ShaderNodeMix")
    _mx.data_type = "RGBA"
    M_PATH.node_tree.links.new(_at.outputs["Fac"], _mx.inputs["Factor"])
    M_PATH.node_tree.links.new(_src, _mx.inputs[6])
    _mx.inputs[7].default_value = (0.02, 0.022, 0.015, 1)
    M_PATH.node_tree.links.new(_mx.outputs[2], _b.inputs["Base Color"])
    ribbon("V3_Path_GateToTown", PATH, 3.4)
    for k, s in enumerate(STREETS[1:]):
        ribbon(f"V3_TownStreet{k}", s, 4.0)
    ALLROADS = [PATH] + STREETS
    t_ = tick("ext_path", t_)

    # ---- houses (procedural; merged per material) ------------------------------------------------------------
    class HouseBM:
        def __init__(self):
            self.bm = {k: bmesh.new() for k in ("wall", "stone", "roof", "timber", "lit", "dark")}
            self.attr = {k: [] for k in self.bm}             # per-face (hv) / (glow)

        def box(self, k, M, lo, hi, val=0.5):
            vs = block_verts(self.bm[k], lo, hi, 0.0, 0.0, None, M)
            self.attr[k] += [val] * 6

        def quad(self, k, pts, val):
            bm = self.bm[k]
            vs = [bm.verts.new(p) for p in pts]
            bm.faces.new(vs)
            self.attr[k].append(val)

        def prism(self, k, M, prof, x0, x1, val):
            bm = self.bm[k]
            a = [bm.verts.new(M @ Vector((x0, y, z))) for y, z in prof]
            b = [bm.verts.new(M @ Vector((x1, y, z))) for y, z in prof]
            bm.faces.new(list(reversed(a)))
            bm.faces.new(b)
            n = len(prof)
            for i in range(n):
                j = (i + 1) % n
                bm.faces.new((a[i], a[j], b[j], b[i]))
            self.attr[k] += [val] * (n + 2)

        def finish(self, prefix, c, mats):
            out = []
            for k, bm in self.bm.items():
                if not bm.faces:
                    bm.free()
                    continue
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                me = D.meshes.new(f"{prefix}_{k}")
                bm.to_mesh(me)
                bm.free()
                nm_ = "glow" if k == "lit" else "hv"
                a = me.attributes.new(nm_, "FLOAT", "FACE")
                vals = np.array(self.attr[k], np.float32)
                if len(vals) != len(me.polygons):
                    vals = np.resize(vals, len(me.polygons))
                a.data.foreach_set("value", vals)
                me.materials.append(mats[k])
                o = D.objects.new(f"{prefix}_{k}", me)
                c.objects.link(o)
                if k == "lit":
                    o.visible_shadow = False
                out.append(o)
            return out
    HMATS = dict(wall=M_WALLT, stone=M_STONET, roof=M_ROOFT, timber=M_TIMB, lit=M_WLIT, dark=M_WDARK)
    FOOT = []        # (x, y, r) house footprints for the woods/fields exclusion
    stats = dict(houses=0, windows=0, lit=0)

    def house(H_, cx, cy, rz, w, d, storeys, seed, stone=False, timber=None, lit_p=0.4):
        r_ = random.Random(seed)
        M = Matrix.Translation((cx, cy, 0)) @ Matrix.Rotation(rz, 4, "Z")
        cs = [M @ Vector((u, v, 0)) for u, v in ((-w / 2, -d / 2), (w / 2, -d / 2), (-w / 2, d / 2), (w / 2, d / 2), (0, 0))]
        gz = [th(p.x, p.y) for p in cs]
        base, g0 = min(gz) - 0.5, max(gz)
        top = g0 + 0.35 + 3.0 * storeys
        hv = r_.random()
        wk = "stone" if stone else "wall"
        H_.box(wk, M, (-w / 2, -d / 2, base), (w / 2, d / 2, top), hv)
        pitch = math.radians(r_.uniform(48, 56))
        ov, gov = 0.45, 0.35
        ridge = top + (d / 2) * math.tan(pitch)
        H_.prism(wk, M, [(-d / 2, top), (d / 2, top), (0, ridge - 0.12)], -w / 2, w / 2, hv)           # gables
        rh = r_.random()
        for s in (-1, 1):                                                                            # roof slabs
            e0 = (s * (d / 2 + ov), top - ov * math.tan(pitch))
            prof = [e0, (0.0, ridge), (0.0, ridge + 0.2), (e0[0], e0[1] + 0.2)]
            if s < 0:
                prof = list(reversed(prof))
            H_.prism("roof", M, prof, -w / 2 - gov, w / 2 + gov, rh)
        if r_.random() < 0.6:                                                                        # chimney
            cxl = r_.uniform(-w / 2 + 0.8, w / 2 - 0.8)
            H_.box("stone", M, (cxl - 0.3, 0.6, top), (cxl + 0.3, 1.2, ridge + 0.9), hv)
        tim = (not stone) and (timber if timber is not None else r_.random() < 0.55)
        for s in (-1, 1):                                                                            # long facades
            yv = s * (d / 2 + 0.03)
            for st in range(storeys):
                z0 = g0 + 0.35 + 3.0 * st
                n = max(1, int(w / 2.6))
                for i in range(n):
                    xw = -w / 2 + (i + 0.5) * w / n
                    if st == 0 and s < 0 and i == n // 2:                                            # door
                        H_.quad("dark", [M @ Vector(p) for p in ((xw - 0.5, yv, z0 - 0.3), (xw + 0.5, yv, z0 - 0.3), (xw + 0.5, yv, z0 + 1.9), (xw - 0.5, yv, z0 + 1.9))], 0.5)
                        continue
                    lit = r_.random() < lit_p
                    k = "lit" if lit else "dark"
                    pts = [(xw - 0.38, yv, z0 + 0.95), (xw + 0.38, yv, z0 + 0.95), (xw + 0.38, yv, z0 + 1.95), (xw - 0.38, yv, z0 + 1.95)]
                    if s > 0:
                        pts = list(reversed(pts))
                    H_.quad(k, [M @ Vector(p) for p in pts], r_.uniform(0.45, 1.0) if lit else 0.5)
                    stats["windows"] += 1
                    stats["lit"] += lit
                if tim:                                                                              # timber frame
                    for xp in np.linspace(-w / 2, w / 2, max(3, int(w / 1.7)) + 1):
                        H_.box("timber", M, (xp - 0.09, s * d / 2 - 0.06 if s < 0 else d / 2 - 0.0, z0),
                               (xp + 0.09, -d / 2 + 0.0 if s < 0 else d / 2 + 0.06, z0 + 3.0), 0.5)
                    H_.box("timber", M, (-w / 2, -d / 2 - 0.06 if s < 0 else d / 2, z0 + 2.9), (w / 2, -d / 2 if s < 0 else d / 2 + 0.06, z0 + 3.1), 0.5)
        for s in (-1, 1):                                                                            # gable window
            if storeys >= 1 and r_.random() < 0.7:
                xv = s * (w / 2 + 0.03)
                zc = top + 0.6
                lit = r_.random() < lit_p * 0.8
                pts = [(xv, -0.3, zc), (xv, 0.3, zc), (xv, 0.3, zc + 0.8), (xv, -0.3, zc + 0.8)]
                if s < 0:
                    pts = list(reversed(pts))
                H_.quad("lit" if lit else "dark", [M @ Vector(p) for p in pts], r_.uniform(0.4, 0.9) if lit else 0.5)
                stats["windows"] += 1
                stats["lit"] += lit
        FOOT.append((cx, cy, 0.6 * max(w, d) + 1.0))
        stats["houses"] += 1
        return base, min(gz)

    # town layout: houses along both sides of every street, facing it
    TH = HouseBM()
    placed = []
    CHURCH = Vector((114.0, -499.0))

    def free(x, y, r):
        if (Vector((x, y)) - CHURCH).length < r + 17:
            return False
        if any((Vector((x, y)) - Vector((px, py))).length < r + pr + 0.8 for px, py, pr in placed):
            return False
        P = np.array([x]), np.array([y])
        return all(poly_dist(*P, s)[0] > r + 1.6 for s in STREETS)
    n_seed = 0
    for s in STREETS:
        cum = 0.0
        for k in range(1, len(s)):
            cum += (s[k] - s[k - 1]).length
            if cum < 7.6:
                continue
            cum = rx.uniform(-1.5, 1.0)
            a = (s[min(k + 1, len(s) - 1)] - s[k - 1]).normalized()
            nrm = Vector((-a.y, a.x))
            if (s[k] - TOWN_C).length > 125:
                continue
            for side in (-1, 1):
                if rx.random() < 0.07:
                    continue
                w, d = rx.uniform(6.0, 9.0), rx.uniform(5.0, 7.0)
                off = d / 2 + rx.uniform(2.6, 4.2)
                p = s[k] + nrm * side * off
                r = 0.55 * math.hypot(w, d)
                if not free(p.x, p.y, r):
                    continue
                rz = math.atan2(a.y, a.x) + (0 if side > 0 else math.pi) + math.radians(rx.uniform(-6, 6))   # door (-Y) -> street
                house(TH, p.x, p.y, rz, w, d, 2 if rx.random() < 0.55 else 1, 100 + n_seed, stone=rx.random() < 0.25)
                placed.append((p.x, p.y, r))
                n_seed += 1
                if rx.random() < 0.6:                              # a second row behind (backyard house / barn)
                    w2, d2 = rx.uniform(5.5, 8.0), rx.uniform(4.5, 6.5)
                    q = p + nrm * side * (d / 2 + d2 / 2 + rx.uniform(3.0, 6.0)) + a * rx.uniform(-2, 2)
                    r2 = 0.55 * math.hypot(w2, d2)
                    if free(q.x, q.y, r2) and (q - TOWN_C).length < 135:
                        house(TH, q.x, q.y, rz + math.radians(rx.uniform(-10, 10)), w2, d2, 1, 700 + n_seed, timber=False, lit_p=0.3)
                        placed.append((q.x, q.y, r2))
                        n_seed += 1
    for _ in range(14):                                          # outlying farmsteads around the town
        a_, rr_ = rx.uniform(0, 2 * math.pi), rx.uniform(95, 150)
        p = TOWN_C + Vector((math.cos(a_), math.sin(a_))) * rr_
        w, d = rx.uniform(7, 10), rx.uniform(5.5, 7.5)
        r = 0.55 * math.hypot(w, d)
        if not free(p.x, p.y, r) or poly_dist(np.array([p.x]), np.array([p.y]), PATH)[0] < 8:
            continue
        house(TH, p.x, p.y, rx.uniform(0, 6.3), w, d, 1, 300 + n_seed, timber=False, lit_p=0.5)
        placed.append((p.x, p.y, r))
        n_seed += 1
    # the church: nave + chancel + west tower + octagonal spire (~40 m), candle-lit lancets
    Mc = Matrix.Translation((CHURCH.x, CHURCH.y, 0)) @ Matrix.Rotation(math.radians(8), 4, "Z")
    cg = [th(*(Mc @ Vector((u, v, 0))).xy) for u, v in ((-14, -6), (14, -6), (-14, 6), (14, 6))]
    cb, c0 = min(cg) - 0.6, max(cg) + 0.3
    TH.box("stone", Mc, (-12, -5, cb), (12, 5, c0 + 11), 0.6)                                   # nave
    TH.prism("stone", Mc, [(-5, c0 + 11), (5, c0 + 11), (0, c0 + 11 + 5 * math.tan(math.radians(55)) - 0.15)], -12, 12, 0.6)
    for s in (-1, 1):
        e0 = (s * 5.5, c0 + 11 - 0.5 * math.tan(math.radians(55)))
        rt = c0 + 11 + 5 * math.tan(math.radians(55))
        prof = [e0, (0.0, rt), (0.0, rt + 0.25), (e0[0], e0[1] + 0.25)]
        TH.prism("roof", Mc, prof if s > 0 else list(reversed(prof)), -12.4, 12.4, 0.15)
    TH.box("stone", Mc, (12, -3.5, cb), (19, 3.5, c0 + 8.5), 0.6)                               # chancel
    TH.prism("stone", Mc, [(-3.5, c0 + 8.5), (3.5, c0 + 8.5), (0, c0 + 8.5 + 3.5 * 1.3)], 12, 19, 0.6)
    for s in (-1, 1):
        e0 = (s * 3.9, c0 + 8.5 - 0.4 * 1.3)
        prof = [e0, (0.0, c0 + 8.5 + 3.5 * 1.3 + 0.05), (0.0, c0 + 8.5 + 3.5 * 1.3 + 0.25), (e0[0], e0[1] + 0.2)]
        TH.prism("roof", Mc, prof if s > 0 else list(reversed(prof)), 12, 19.3, 0.15)
    TT_ = c0 + 24.0
    TH.box("stone", Mc, (-18.5, -3.25, cb), (-12, 3.25, TT_), 0.6)                              # west tower
    bm = TH.bm["roof"]                                                                           # octagonal spire
    ring = [bm.verts.new(Mc @ Vector((-15.25 + 3.5 * math.cos(a), 3.5 * math.sin(a), TT_))) for a in np.linspace(0, 2 * math.pi, 9)[:-1] + math.pi / 8]
    apex = bm.verts.new(Mc @ Vector((-15.25, 0, TT_ + 17.0)))
    for i in range(8):
        bm.faces.new((ring[i], ring[(i + 1) % 8], apex))
    bm.faces.new(list(reversed(ring)))
    TH.attr["roof"] += [0.1] * 9
    TH.box("timber", Mc, (-15.33, -0.05, TT_ + 17.0), (-15.17, 0.05, TT_ + 18.6), 0.5)          # cross
    TH.box("timber", Mc, (-15.33, -0.5, TT_ + 17.9), (-15.17, 0.5, TT_ + 18.05), 0.5)
    for s in (-1, 1):                                                                            # lancets
        for xw in (-8.5, -4.5, -0.5, 3.5, 7.5):
            yv = s * 5.03
            pts = [(xw - 0.45, yv, c0 + 3.5), (xw + 0.45, yv, c0 + 3.5), (xw + 0.45, yv, c0 + 7.5), (xw - 0.45, yv, c0 + 7.5)]
            TH.quad("lit", [Mc @ Vector(p) for p in (pts if s < 0 else list(reversed(pts)))], 0.55)
        for zb in (TT_ - 5.0,):                                                                  # belfry openings
            pts = [(-16.0, s * 3.28, zb), (-14.5, s * 3.28, zb), (-14.5, s * 3.28, zb + 2.6), (-16.0, s * 3.28, zb + 2.6)]
            TH.quad("dark", [Mc @ Vector(p) for p in (pts if s < 0 else list(reversed(pts)))], 0.5)
    FOOT.append((CHURCH.x, CHURCH.y, 22.0))
    town_objs = TH.finish("V3_Town", C_TN, HMATS)
    ER["town"] = dict(centre=[TOWN_C.x, TOWN_C.y], houses=stats["houses"], church_spire_top_z=round(TT_ + 18.6, 1),
                      distance_from_gate_m=round((TOWN_C - Vector((0, -81))).length, 0))
    # village cottages along the path (nearish) + lanterns
    VH = HouseBM()
    VILL = []
    for k, (tpar, side, rzo) in enumerate(((0.36, 1, 0), (0.47, -1, 0.3), (0.58, 1, -0.2))):
        i = int(tpar * (len(PATH) - 1))
        a = (PATH[i + 1] - PATH[i - 1]).normalized()
        nrm = Vector((-a.y, a.x))
        p = PATH[i] + nrm * side * 13.0
        rz = math.atan2(a.y, a.x) + (math.pi if side > 0 else 0) + rzo + math.pi
        house(VH, p.x, p.y, rz, rx.uniform(7, 9), rx.uniform(5.5, 6.5), 1 + (k == 1), 500 + k, timber=True, lit_p=0.55)
        VILL.append(p)
        lp = p - nrm * side * 3.8
        light(f"V3_VillageLantern{k}", "POINT", (lp.x, lp.y, th(lp.x, lp.y) + 2.4), None, 220.0, (1.0, 0.55, 0.2), size=0.15, c=C_EL)
    VH.finish("V3_Village", C_TN, HMATS)
    # street glow in the town (a few broad warm lights: windows alone vanish at 450 m)
    for k, s in enumerate(STREETS[1:] + [PATH[-30:]]):
        p = s[len(s) // 3]
        light(f"V3_TownGlow{k}", "POINT", (p.x, p.y, th(p.x, p.y) + 4.0), None, 2600.0, (1.0, 0.5, 0.18), size=2.0, c=C_EL)
    t_ = tick("ext_town", t_)

    # two buildings just outside the south curtain wall (linked copies of the v2 compound scans) + door lanterns
    OUTS = []
    for nm, src, (x, y), rzd in (("V3_Outside_StoneHouse", "House_Stone", (-27.0, -97.0), 180.0),
                                 ("V3_Outside_HalfTimber", "House_HalfTimber1", (27.5, -100.0), 0.0)):
        so = D.objects[src]
        o = new_obj(nm, so.data, C_TN, (x, y, 0), math.radians(rzd))
        o.scale = so.scale
        bpy.context.view_layer.update()
        w = world_co(o)
        g = [th(px, py) for px, py in ((w[:, 0].min(), w[:, 1].min()), (w[:, 0].max(), w[:, 1].max()),
                                         (w[:, 0].min(), w[:, 1].max()), (w[:, 0].max(), w[:, 1].min()), (x, y))]
        o.location.z = min(g) - float(w[:, 2].min()) - 0.12          # lowest vertex 12 cm into the ground
        OUTS.append(o)
        FOOT.append((x, y, 0.6 * float(max(np.ptp(w[:, 0]), np.ptp(w[:, 1]))) + 2.0))
        light(nm + "_Lantern", "POINT", (x + (2.5 if x < 0 else -2.5), y + 5.5, min(g) + 2.6), None, 160.0, (1.0, 0.55, 0.2),
              size=0.12, c=C_EL)
    ER["outside_wall_buildings"] = [o.name for o in OUTS]

    # ---- fields (strip fields in the valley floor) + hedgerows ---------------------------------------------------
    ang = math.atan2(TOWN_C.y - CASTLE_C.y, TOWN_C.x - CASTLE_C.x) + math.radians(8)
    ca, sa = math.cos(ang), math.sin(ang)
    FIELDS = []
    HEDGE = []
    fV, fF, fUV, fMI = [], [], [], []
    for gu in np.arange(-260, 700, 70.0):
        for gv in np.arange(-420, 420, 38.0):
            L_, Wd = rx.uniform(55, 85), rx.uniform(24, 40)
            cu, cv = gu + rx.uniform(-6, 6), gv + rx.uniform(-4, 4)
            cx, cy = CASTLE_C.x + ca * cu - sa * cv, CASTLE_C.y + sa * cu + ca * cv
            corners = [(cx + ca * u - sa * v, cy + sa * u + ca * v) for u, v in ((-L_ / 2, -Wd / 2), (L_ / 2, -Wd / 2), (L_ / 2, Wd / 2), (-L_ / 2, Wd / 2))]
            X_, Y_ = np.array([c[0] for c in corners] + [cx]), np.array([c[1] for c in corners] + [cy])
            dcs, _ = seg_dist(X_, Y_, CASTLE_C, TOWN_C)
            if (dcs > 205).any() or (rect_dist(X_, Y_) < 75).any() or (np.hypot(X_ - TOWN_C.x, Y_ - TOWN_C.y) < 125).any():
                continue
            if (poly_dist(X_, Y_, PATH) < 9).any() or any((poly_dist(X_, Y_, s) < 7).any() for s in STREETS):
                continue
            if any(np.hypot(X_ - fx, Y_ - fy).min() < fr_ + 6 for fx, fy, fr_ in FOOT):
                continue
            ew = ((X_ - 175) / 150) ** 2 + ((Y_ + 165) / 95) ** 2                    # the east woods stretch stays wood
            if (ew < 1.0).any():
                continue
            mi = rx.randrange(len(FIELD_MATS))
            nu, nv = max(2, int(L_ / 5)), max(2, int(Wd / 5))
            b0 = len(fV)
            for j in range(nv + 1):
                for i in range(nu + 1):
                    u, v = -L_ / 2 + L_ * i / nu, -Wd / 2 + Wd * j / nv
                    x, y = cx + ca * u - sa * v, cy + sa * u + ca * v
                    fV.append((x, y, th(x, y) + 0.06))
                    fUV.append((u, v))
            for j in range(nv):
                for i in range(nu):
                    a0 = b0 + j * (nu + 1) + i
                    fF.append((a0, a0 + 1, a0 + nu + 2, a0 + nu + 1))
                    fMI.append(mi)
            FIELDS.append((cx, cy, L_, Wd))
            for e in range(4):                                                         # hedges on ~60% of edges
                if rx.random() < 0.6:
                    p0, p1 = Vector(corners[e]), Vector(corners[(e + 1) % 4])
                    nb = int((p1 - p0).length / 2.6)
                    for k in range(nb):
                        q = p0.lerp(p1, (k + rx.random() * 0.6) / max(nb, 1))
                        q += Vector((rx.uniform(-0.8, 0.8), rx.uniform(-0.8, 0.8)))
                        HEDGE.append(q)
    fme = D.meshes.new("V3_Fields")
    fme.from_pydata(fV, [], fF)
    uvl = fme.uv_layers.new(name="UVMap")
    loop_v = np.empty(len(fme.loops), np.int32)
    fme.loops.foreach_get("vertex_index", loop_v)
    uva = np.array(fUV, np.float32)[loop_v]
    uvl.data.foreach_set("uv", uva.ravel())
    for m in FIELD_MATS:
        fme.materials.append(m)
    fme.polygons.foreach_set("material_index", np.array(fMI, np.int32))
    fo = D.objects.new("V3_Fields", fme)
    C_FM.objects.link(fo)
    ER["fields"] = dict(count=len(FIELDS), hedge_bushes=len(HEDGE))
    # haystacks on some stubble fields
    hs = bmesh.new()
    n_hay = 0
    for (cx, cy, L_, Wd) in FIELDS:
        if rx.random() < 0.25:
            for _ in range(rx.randint(2, 5)):
                x, y = cx + rx.uniform(-L_ / 3, L_ / 3) * ca, cy + rx.uniform(-L_ / 3, L_ / 3) * sa
                z = th(x, y) - 0.1
                g = bmesh.ops.create_cone(hs, cap_ends=True, segments=12, radius1=1.5, radius2=0.25, depth=2.6)
                bmesh.ops.translate(hs, verts=g["verts"], vec=(x, y, z + 1.3))
                n_hay += 1
    o = new_obj("V3_Haystacks", to_mesh(hs, "V3_Haystacks", [M_STRAW]), C_FM)
    ER["fields"]["haystacks"] = n_hay
    t_ = tick("ext_fields", t_)

    # ---- trees: procedural templates + GN instancing (instances, never realized) --------------------------------
    def conifer_mesh(name, seed, tiers, spikes):
        r_ = random.Random(seed)
        bm = bmesh.new()
        g = bmesh.ops.create_cone(bm, cap_ends=False, segments=6, radius1=0.022, radius2=0.012, depth=0.5)
        bmesh.ops.translate(bm, verts=g["verts"], vec=(0, 0, 0.22))
        trunk = len(bm.faces)
        for i in range(tiers):
            zb = 0.16 + 0.74 * i / tiers
            R = 0.24 * (1 - i / tiers) ** 0.85 * r_.uniform(0.85, 1.12) + 0.03
            top = zb + (0.16 + 0.1 * (1 - i / tiers)) * r_.uniform(0.9, 1.1)
            ap = bm.verts.new((0, 0, top))
            ctr = bm.verts.new((0, 0, zb + 0.03))
            rg_ = []
            for k in range(2 * spikes):
                a = 2 * math.pi * k / (2 * spikes) + r_.uniform(-0.08, 0.08)
                rr__ = R * (r_.uniform(0.7, 1.15) if k % 2 == 0 else r_.uniform(0.35, 0.55))
                rg_.append(bm.verts.new((rr__ * math.cos(a), rr__ * math.sin(a),
                                         zb - (r_.uniform(0.03, 0.07) if k % 2 == 0 else -0.01) * (R / 0.24))))
            for k in range(len(rg_)):
                bm.faces.new((rg_[k], rg_[(k + 1) % len(rg_)], ap))
                bm.faces.new((rg_[(k + 1) % len(rg_)], rg_[k], ctr))
        me = D.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(M_BARK)
        me.materials.append(M_CONI)
        mi = np.ones(len(me.polygons), np.int32)
        mi[:trunk] = 0
        me.polygons.foreach_set("material_index", mi)
        return me

    def deci_mesh(name, seed, blobs, sub=2):
        r_ = random.Random(seed)
        bm = bmesh.new()
        g = bmesh.ops.create_cone(bm, cap_ends=False, segments=6, radius1=0.035, radius2=0.02, depth=0.55)
        bmesh.ops.translate(bm, verts=g["verts"], vec=(0, 0, 0.25))
        trunk = len(bm.faces)
        for k in range(blobs):
            a, el = r_.uniform(0, 2 * math.pi), r_.uniform(0.0, 1.0)
            rad = r_.uniform(0.13, 0.2)
            c = Vector((math.cos(a) * 0.17 * (1 - el), math.sin(a) * 0.17 * (1 - el), 0.6 + 0.25 * el))
            g = bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=rad)
            for v in g["verts"]:
                v.co = v.co * r_.uniform(0.82, 1.12) + c
        me = D.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        me.shade_smooth()                                  # faceted crowns read low-poly up close
        me.materials.append(M_BARK)
        me.materials.append(M_DECI)
        mi = np.ones(len(me.polygons), np.int32)
        mi[:trunk] = 0
        me.polygons.foreach_set("material_index", mi)
        return me
    TPL = {"coniferA": conifer_mesh("V3T_coniferA", 1, 12, 13), "coniferB": conifer_mesh("V3T_coniferB", 2, 10, 12),
           "coniferC": conifer_mesh("V3T_coniferC", 3, 14, 14), "conifer_lo": conifer_mesh("V3T_conifer_lo", 4, 4, 7),
           "deciA": deci_mesh("V3T_deciA", 5, 8), "deciB": deci_mesh("V3T_deciB", 6, 6), "deci_lo": deci_mesh("V3T_deci_lo", 7, 3, 1),
           "bush": deci_mesh("V3T_bush", 8, 4, 1)}
    TPL_OBJ = {}
    for k, me in TPL.items():
        o = D.objects.new("TPL_" + k, me)
        C_TPL.objects.link(o)
        o.location = (0, 0, -5000)                     # parked far underground, viewport-visible (gotcha 1),
        o.hide_render = True                           # not rendered itself
        TPL_OBJ[k] = o

    def instancer(name, tpl, P, rot, scl, c):
        me = D.meshes.new(name + "_pts")
        me.from_pydata(P, [], [])
        a = me.attributes.new("rot", "FLOAT", "POINT")
        a.data.foreach_set("value", np.array(rot, np.float32))
        b = me.attributes.new("scl", "FLOAT", "POINT")
        b.data.foreach_set("value", np.array(scl, np.float32))
        o = D.objects.new(name, me)
        c.objects.link(o)
        ng = D.node_groups.new("GN_" + name, "GeometryNodeTree")
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        Nn, Ln = ng.nodes, ng.links
        gi, go = Nn.new("NodeGroupInput"), Nn.new("NodeGroupOutput")
        oi = Nn.new("GeometryNodeObjectInfo")
        oi.inputs["Object"].default_value = tpl
        oi.transform_space = "ORIGINAL"
        ar = Nn.new("GeometryNodeInputNamedAttribute")
        ar.data_type = "FLOAT"
        ar.inputs["Name"].default_value = "rot"
        asc = Nn.new("GeometryNodeInputNamedAttribute")
        asc.data_type = "FLOAT"
        asc.inputs["Name"].default_value = "scl"
        cx_ = Nn.new("ShaderNodeCombineXYZ")
        Ln.new(ar.outputs["Attribute"], cx_.inputs["Z"])
        iop = Nn.new("GeometryNodeInstanceOnPoints")
        Ln.new(gi.outputs[0], iop.inputs["Points"])
        Ln.new(oi.outputs["Geometry"], iop.inputs["Instance"])
        Ln.new(cx_.outputs["Vector"], iop.inputs["Rotation"])
        Ln.new(asc.outputs["Attribute"], iop.inputs["Scale"])
        Ln.new(iop.outputs["Instances"], go.inputs[0])
        o.modifiers.new("Instances", "NODES").node_group = ng
        return o
    # candidate points: jittered 8.5 m grid inside 820 m of the castle and 260 m of the town, masked
    gx = np.arange(-820, 820, 8.5)
    GX, GY = np.meshgrid(gx, np.arange(-900, 820, 8.5))
    px = GX.ravel() + np.array([rx.uniform(-3.5, 3.5) for _ in range(GX.size)])
    py = GY.ravel() + np.array([rx.uniform(-3.5, 3.5) for _ in range(GX.size)])
    keep = (np.hypot(px, py - 4) < 820) | (np.hypot(px - TOWN_C.x, py - TOWN_C.y) < 300)
    px, py = px[keep], py[keep]
    fnz = np.array([mathutils.noise.fractal(Vector((x / 300, y / 300, 4.1)), 0.55, 2.0, 3) for x, y in zip(px, py)])
    dcs, _ = seg_dist(px, py, CASTLE_C, TOWN_C)
    ew = ((px - 175) / 150) ** 2 + ((py + 165) / 95) ** 2
    wood = (fnz > 0.02) & (dcs > 150) | (ew < 1.0) & (fnz > -0.45)          # valley floor stays open (fields)
    wood &= rect_dist(px, py) > 55
    wood &= poly_dist(px, py, PATH) > 18
    for s in STREETS:
        wood &= poly_dist(px, py, s) > 8
    wood &= np.hypot(px - TOWN_C.x, py - TOWN_C.y) > 135
    for fx, fy, fr_ in FOOT:
        wood &= np.hypot(px - fx, py - fy) > fr_ + 6
    for (cx, cy, L_, Wd) in FIELDS:
        u = (px - cx) * ca + (py - cy) * sa
        v = -(px - cx) * sa + (py - cy) * ca
        wood &= ~((np.abs(u) < L_ / 2 + 3) & (np.abs(v) < Wd / 2 + 3))
    px, py = px[wood], py[wood]
    GROUPS = {k: ([], [], []) for k in TPL if k != "bush"}
    n_miss = 0
    for x, y in zip(px, py):
        z = th(x, y)
        far = math.hypot(x, y - 4) > 520 and math.hypot(x - TOWN_C.x, y - TOWN_C.y) > 260
        con = rx.random() < (0.55 + 0.4 * min(1.0, max(0.0, (z + 2) / 40)))          # more conifers up the hills
        if con:
            k = "conifer_lo" if far else rx.choice(("coniferA", "coniferB", "coniferC"))
            h = rx.uniform(13, 24)
        else:
            k = "deci_lo" if far else rx.choice(("deciA", "deciB"))
            h = rx.uniform(9, 16)
        GROUPS[k][0].append((x, y, z - 0.03 * h))                                       # trunk base sunk below grade
        GROUPS[k][1].append(rx.uniform(0, 2 * math.pi))
        GROUPS[k][2].append(h)
    for k, (P, R_, S_) in GROUPS.items():
        if P:
            instancer(f"V3_Woods_{k}", TPL_OBJ[k], P, R_, S_, C_WD)
    HP = [(q.x, q.y, th(q.x, q.y) - 0.1) for q in HEDGE]
    instancer("V3_Hedges", TPL_OBJ["bush"], HP, [rx.uniform(0, 6.3) for _ in HP], [rx.uniform(2.6, 4.2) for _ in HP], C_FM)
    ER["woods"] = dict(trees={k: len(v[0]) for k, v in GROUPS.items()}, total=int(sum(len(v[0]) for v in GROUPS.values())),
                       template_tris={k: len(me.polygons) for k, me in TPL.items()},
                       far_woods="beyond ~820 m: terrain 'forest' tint only (no instances)")
    t_ = tick("ext_woods", t_)

    # ---- ground fog: bounded boxes in the valley (outside the walls only), height falloff + drifting noise -----------
    M_FOG = D.materials.new("M_V3GroundFog")
    M_FOG.use_nodes = True
    nt, N, L = M_FOG.node_tree, M_FOG.node_tree.nodes, M_FOG.node_tree.links
    for n_ in list(N):
        if n_.type == "BSDF_PRINCIPLED":
            N.remove(n_)
    out = next(n_ for n_ in N if n_.type == "OUTPUT_MATERIAL")
    vol = N.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.72, 0.78, 0.9, 1)
    vol.inputs["Anisotropy"].default_value = 0.3
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Generated"], sep.inputs["Vector"])
    hf = N.new("ShaderNodeMapRange")
    hf.inputs["From Min"].default_value, hf.inputs["From Max"].default_value = 0.1, 0.75
    hf.inputs["To Min"].default_value, hf.inputs["To Max"].default_value = 1.0, 0.0
    L.new(sep.outputs["Z"], hf.inputs["Value"])
    mpn = N.new("ShaderNodeMapping")
    L.new(tc.outputs["Object"], mpn.inputs["Vector"])
    fc = mpn.inputs["Location"].driver_add("default_value", 0)                # drift: 1.2 m/s along +X (simple expr)
    fc.driver.type = "SCRIPTED"
    fc.driver.expression = "frame * 0.04"
    fc2 = mpn.inputs["Location"].driver_add("default_value", 1)
    fc2.driver.type = "SCRIPTED"
    fc2.driver.expression = "frame * 0.015"
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.022
    nz.inputs["Detail"].default_value = 3.0
    nz.inputs["Roughness"].default_value = 0.55
    L.new(mpn.outputs["Vector"], nz.inputs["Vector"])
    nm_ = N.new("ShaderNodeMapRange")
    nm_.inputs["From Min"].default_value, nm_.inputs["From Max"].default_value = 0.46, 0.66
    L.new(nz.outputs["Fac"], nm_.inputs["Value"])
    m1 = N.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    L.new(hf.outputs["Result"], m1.inputs[0])
    L.new(nm_.outputs["Result"], m1.inputs[1])
    m2 = N.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m2.inputs[1].default_value = 0.045
    L.new(m1.outputs[0], m2.inputs[0])
    L.new(m2.outputs[0], vol.inputs["Density"])
    L.new(vol.outputs[0], out.inputs["Volume"])
    FOGB = []
    for k, (x0, x1, y0, y1, z0, z1) in enumerate(((-330, 340, -300, -95, -3.0, 6.0), (-340, 360, -440, -300, -8.6, 1.5),
                                                  (-220, 360, -640, -440, -10.8, -1.0), (60, 360, -95, 40, -2.0, 7.0))):
        o = boxes(f"V3_GroundFog{k}", [((x0, y0, z0), (x1, y1, z1))], M_FOG, C_FG)
        o.visible_shadow = False
        FOGB.append((x0, x1, y0, y1, z0, z1))
    SC.eevee.volumetric_end = 650.0                                            # was 400: fog over the far fields
    for n_ in SC.world.node_tree.nodes:
        if n_.type == "MAP_RANGE" and n_.inputs["Value"].links and n_.inputs["Value"].links[0].from_node.type == "SEPXYZ":
            ER["world_mist_band_z"] = [n_.inputs["From Min"].default_value, n_.inputs["From Max"].default_value, "->", 0.15, 0.6]
            n_.inputs["From Min"].default_value, n_.inputs["From Max"].default_value = 0.15, 0.6
    ER["fog"] = dict(boxes=[list(b) for b in FOGB], drift="driver frame*0.04 / frame*0.015 (m) on the noise mapping")
    t_ = tick("ext_fog", t_)

    # ---- exterior asserts --------------------------------------------------------------------------------------
    ck = {}
    # fog never enters the compound / hall (boxes vs the compound's outer footprint)
    ck["fog_outside_compound"] = all(not (b[1] > RECT[0] and b[0] < RECT[2] and b[3] > RECT[1] and b[2] < RECT[3]) for b in FOGB)
    assert ck["fog_outside_compound"], FOGB
    # nothing exterior inside the compound (trees, houses, fields, path start is outside the wall face)
    allpts = np.array([p for v in GROUPS.values() for p in v[0]] + HP)
    ck["trees_inside_compound"] = int((rect_dist(allpts[:, 0], allpts[:, 1]) <= 0).sum())
    ck["fields_inside_compound"] = int((rect_dist(np.array(fV)[:, 0], np.array(fV)[:, 1]) <= 0).sum()) if fV else 0
    assert ck["trees_inside_compound"] == 0 and ck["fields_inside_compound"] == 0, ck
    # grounding: every tree base is at/below its terrain point; houses go below their lowest corner; buildings sunk
    tz = np.array([th(p[0], p[1]) for p in allpts[:: max(1, len(allpts) // 2000)]])
    ck["trees_base_below_grade_m"] = round(float((allpts[:: max(1, len(allpts) // 2000), 2] - tz).max()), 3)
    assert ck["trees_base_below_grade_m"] <= 0.0, ck
    fz = np.array(fV)
    ck["field_vertices_checked"] = len(fz)
    bpy.context.view_layer.update()
    for o in OUTS:
        w = world_co(o)
        gmin = min(th(px_, py_) for px_, py_ in ((w[:, 0].min(), w[:, 1].min()), (w[:, 0].max(), w[:, 1].max()),
                                                 (w[:, 0].min(), w[:, 1].max()), (w[:, 0].max(), w[:, 1].min())))
        assert float(w[:, 2].min()) <= gmin + 0.001, (o.name, "floats")
    ck["outside_buildings_grounded"] = True
    ER["checks"] = ck
    ER["build_s"] = round(time.time() - T_EXT, 1)
    return dict(TER=TER, th=th)




# ---------------------------------------------------------------------------------------------------------
# =========================================  v3c — COURTYARD GRASS  =======================================
# Patchy scruffy grass + weeds along the walls, around building bases, sparse in the paving, and on the ground just
# outside the curtain walls. One emitter grid (2 m) carries a 'grass' density attribute computed here (masks below);
# GN Distribute Points on Faces -> Instance on Points (4 low-poly tuft/weed templates, picked by white noise), kept as
# INSTANCES, shadows off. Keep-clear (asserted on the real evaluated instances): the hall/platform + stairs, the main path
# gate -> stairs (the knight's route), the outside path.
# ---------------------------------------------------------------------------------------------------------
M_GRASS_CAM = (-13.5, -74.4, 0.55)                   # low courtyard shot: the grass is parted 1.2 m around the lens


def build_grass(th):
    T_G = time.time()
    C_G = coll("V3C_GRASS")
    C_GT = coll("V3C_GRASS_TEMPLATES", C_G)
    GR = REPORT["added"].setdefault("grass", {})
    rg_ = random.Random(5150)
    M_GR = flat_mat("M_V3Grass", (0.1, 0.12, 0.05), 0.65)
    M_GD = flat_mat("M_V3GrassDry", (0.16, 0.13, 0.07), 0.75)
    for m in (M_GR, M_GD):                                     # a little macro colour variation
        b = principled(m)
        nt = m.node_tree
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 0.4
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.7, 1.3
        nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
        mx = nt.nodes.new("ShaderNodeMix")
        mx.data_type, mx.blend_type = "RGBA", "MULTIPLY"
        mx.inputs["Factor"].default_value = 1.0
        mx.inputs[6].default_value = b.inputs["Base Color"].default_value
        cc = nt.nodes.new("ShaderNodeCombineColor")
        for i in range(3):
            nt.links.new(mr.outputs["Result"], cc.inputs[i])
        nt.links.new(cc.outputs["Color"], mx.inputs[7])
        nt.links.new(mx.outputs[2], b.inputs["Base Color"])

    def tuft(name, n, h0, h1, w, lean, mat, seed, leaves=False):
        r_ = random.Random(seed)
        bm = bmesh.new()
        for i in range(n):
            a = r_.uniform(0, 2 * math.pi)
            r0 = r_.uniform(0, 0.13) * math.sqrt(r_.random())
            bx, by = r0 * math.cos(a), r0 * math.sin(a)
            h = r_.uniform(h0, h1)
            d = Vector((math.cos(a), math.sin(a), 0))
            side = Vector((-d.y, d.x, 0)) * (w * r_.uniform(0.7, 1.3))
            base = Vector((bx, by, -0.02))
            if leaves:                                         # weed: broad leaf, a quad bent outward + flat
                tip = base + d * h * 1.2 + Vector((0, 0, h * 0.35))
                mid = base + d * h * 0.55 + Vector((0, 0, h * 0.3))
                vs = [bm.verts.new(base - side * 0.3), bm.verts.new(mid - side), bm.verts.new(tip), bm.verts.new(mid + side),
                      bm.verts.new(base + side * 0.3)]
                bm.faces.new((vs[0], vs[1], vs[2], vs[3], vs[4]))
            else:                                              # blade: 2 segments, curved lean
                l1 = r_.uniform(0.1, lean)
                m1 = base + d * l1 * h * 0.35 + Vector((0, 0, h * 0.55))
                tip = base + d * l1 * h + Vector((0, 0, h))
                vs = [bm.verts.new(base - side), bm.verts.new(base + side), bm.verts.new(m1 + side * 0.55),
                      bm.verts.new(m1 - side * 0.55), bm.verts.new(tip)]
                bm.faces.new((vs[0], vs[1], vs[2], vs[3]))
                bm.faces.new((vs[3], vs[2], vs[4]))
        me = D.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        o = D.objects.new("TPL_" + name, me)
        C_GT.objects.link(o)
        o.location = (0, 0, -5000)
        o.hide_render = True
        return o
    TPLS = [tuft("V3G_tuftShort", 28, 0.05, 0.16, 0.016, 0.8, M_GR, 1), tuft("V3G_tuftMid", 22, 0.12, 0.3, 0.016, 0.9, M_GR, 2),
            tuft("V3G_tuftDry", 16, 0.18, 0.42, 0.012, 0.8, M_GD, 3), tuft("V3G_weed", 8, 0.08, 0.16, 0.05, 0.0, M_GR, 4, leaves=True)]
    GR["template_tris"] = {o.name: sum(len(p.vertices) - 2 for p in o.data.polygons) for o in TPLS}

    # obstacles (anything standing on the courtyard / near-wall ground) for the "under something" + "near a base" masks
    dg_ = bpy.context.evaluated_depsgraph_get()
    REG = (-120.0, -150.0, 120.0, 160.0)
    skip = {"Terrain", "CourtyardPaving", "ArenaFloor"}
    parts = []
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
        parts.append(tris_world(o, dg_))
    for nm in ("V3_Path_GateToTown",):                         # the path ribbon is "something on the ground" too
        if nm in D.objects:
            parts.append(tris_world(D.objects[nm], dg_))
    OBST = bvh_of(parts)
    PATH_OB = bvh_of([tris_world(D.objects["V3_Path_GateToTown"], dg_)]) if "V3_Path_GateToTown" in D.objects else None

    def keep_clear(x, y):
        """the asserted no-grass zones: hall platform + stairs (+3 m), the main path gate->stairs foot (knight's route)."""
        if -25.5 < x < 25.5 and -28.0 < y < 56.0:
            return True
        if abs(x) < 6.5 and -82.0 < y < -24.0:
            return True
        return False
    step = 2.0
    xs_ = np.arange(REG[0], REG[2] + 1e-6, step)
    ys_ = np.arange(REG[1], REG[3] + 1e-6, step)
    V, Dn, ZF = [], [], []                                                       # ZF: zero every face touching it
    n_cov = 0
    for y in ys_:
        for x in xs_:
            d_rect = float(rect_dist(np.array([x]), np.array([y]))[0])
            inside = -CXc < x < CXc and CY0c < y < CY1c
            if inside:
                z = 0.0
                dwall = min(x + CXc, CXc - x, y - CY0c, CY1c - y)
            elif d_rect > 0:
                z = th(x, y)
                dwall = d_rect
            else:                                              # inside the curtain-wall body: no grass; the vertex takes
                din = math.hypot(max(abs(x) - CXc, 0.0), max(CY0c - y, y - CY1c, 0.0))   # distance to the courtyard; the nearer
                V.append((x, y, 0.0 if din < CWT / 2 else -0.35))                                     # side's ground
                Dn.append(0.0)
                ZF.append(keep_clear(x, y))
                continue
            dens = 0.0
            if not keep_clear(x, y) and math.hypot(x - M_GRASS_CAM[0], y - M_GRASS_CAM[1]) > 1.2:
                n1 = mathutils.noise.noise(Vector((x / 7.0, y / 7.0, 3.3)))
                n2 = mathutils.noise.noise(Vector((x / 1.6, y / 1.6, 8.1)))
                patch = float(smooth(-0.15, 0.35, n1)) * (0.6 + 0.4 * float(smooth(-0.4, 0.4, n2)))
                if inside:
                    base = 0.05 + 0.95 * float(smooth(3.5, 0.8, dwall))          # cracks everywhere (sparse) + wall band
                else:
                    base = 0.08 + 0.85 * float(smooth(14.0, 1.5, dwall))         # outside: a rough verge along the walls
                hit = OBST.find_nearest(Vector((x, y, z + 0.3)))
                if hit[0] is not None and hit[3] < 2.0:
                    base = max(base, 0.95 * (1 - hit[3] / 2.0))                  # around building/plinth/rubble bases
                loc = OBST.ray_cast(Vector((x, y, z + 40.0)), Vector((0, 0, -1)), 60.0)[0]
                if loc is not None and loc.z > z + 0.02:
                    base = 0.0                                                   # under something: nothing grows
                    n_cov += 1
                band = min(1.0, base / 0.6)                  # in the wall/base bands the patch noise only thins it
                dens = max(0.0, min(1.0, base * (band * (0.5 + 0.5 * patch) + (1 - band) * patch) * 1.4))
            V.append((x, y, z + 0.0))
            Dn.append(dens)
            onp = PATH_OB is not None and PATH_OB.ray_cast(Vector((x, y, z + 5)), Vector((0, 0, -1)), 10.0)[0] is not None
            ZF.append(keep_clear(x, y) or onp)
    nx, ny = len(xs_), len(ys_)
    Fc = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i) for j in range(ny - 1) for i in range(nx - 1)]
    # faces touching a keep-clear vertex get density 0 on all corners (no point can interpolate into the zone)
    Dn = np.array(Dn, np.float32)
    kc = np.array(ZF)
    for f in Fc:
        if kc[list(f)].any():
            Dn[list(f)] = 0.0
    me = D.meshes.new("V3C_GrassEmitter")
    me.from_pydata(V, [], Fc)
    a = me.attributes.new("grass", "FLOAT", "POINT")
    a.data.foreach_set("value", Dn)
    go_ = D.objects.new("V3C_Grass", me)
    C_G.objects.link(go_)
    go_.visible_shadow = False
    DENS_MAX = 18.0                                                              # tufts / m^2 at full mask
    ng = D.node_groups.new("GN_V3C_Grass", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    Nn, Ln = ng.nodes, ng.links
    gi, go = Nn.new("NodeGroupInput"), Nn.new("NodeGroupOutput")
    at = Nn.new("GeometryNodeInputNamedAttribute")
    at.data_type = "FLOAT"
    at.inputs["Name"].default_value = "grass"
    mul = Nn.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = DENS_MAX
    Ln.new(at.outputs["Attribute"], mul.inputs[0])
    dp = Nn.new("GeometryNodeDistributePointsOnFaces")
    dp.distribute_method = "RANDOM"
    dp.inputs["Seed"].default_value = 11
    Ln.new(gi.outputs[0], dp.inputs["Mesh"])
    Ln.new(mul.outputs[0], dp.inputs["Density"])
    g2i = Nn.new("GeometryNodeGeometryToInstance")
    for o in TPLS:
        oi = Nn.new("GeometryNodeObjectInfo")
        oi.inputs["Object"].default_value = o
        oi.transform_space = "ORIGINAL"
        Ln.new(oi.outputs["Geometry"], g2i.inputs["Geometry"])
    pos = Nn.new("GeometryNodeInputPosition")
    wn = []
    for off in ((0, 0, 0), (17.3, 5.1, 0), (3.7, 29.9, 0)):                     # three independent uniform randoms
        ad = Nn.new("ShaderNodeVectorMath")
        ad.operation = "ADD"
        ad.inputs[1].default_value = off
        Ln.new(pos.outputs["Position"], ad.inputs[0])
        w_ = Nn.new("ShaderNodeTexWhiteNoise")
        w_.noise_dimensions = "3D"
        Ln.new(ad.outputs["Vector"], w_.inputs["Vector"])
        wn.append(w_.outputs["Value"])
    idx = Nn.new("ShaderNodeMath")
    idx.operation = "MULTIPLY"
    idx.inputs[1].default_value = len(TPLS) - 0.001
    Ln.new(wn[0], idx.inputs[0])
    flo = Nn.new("ShaderNodeMath")
    flo.operation = "FLOOR"
    Ln.new(idx.outputs[0], flo.inputs[0])
    rot = Nn.new("ShaderNodeMath")
    rot.operation = "MULTIPLY"
    rot.inputs[1].default_value = 6.2832
    Ln.new(wn[1], rot.inputs[0])
    cxr = Nn.new("ShaderNodeCombineXYZ")
    Ln.new(rot.outputs[0], cxr.inputs["Z"])
    scm = Nn.new("ShaderNodeMapRange")
    scm.inputs["To Min"].default_value, scm.inputs["To Max"].default_value = 0.6, 1.5
    Ln.new(wn[2], scm.inputs["Value"])
    iop = Nn.new("GeometryNodeInstanceOnPoints")
    Ln.new(dp.outputs["Points"], iop.inputs["Points"])
    Ln.new(g2i.outputs["Instances"], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    Ln.new(flo.outputs[0], iop.inputs["Instance Index"])
    Ln.new(cxr.outputs["Vector"], iop.inputs["Rotation"])
    Ln.new(scm.outputs["Result"], iop.inputs["Scale"])
    Ln.new(iop.outputs["Instances"], go.inputs[0])
    go_.modifiers.new("Grass", "NODES").node_group = ng
    # ---- asserts on the REAL evaluated instances --------------------------------------------------------------
    bpy.context.view_layer.update()
    dg2 = bpy.context.evaluated_depsgraph_get()
    P = []
    for inst in dg2.object_instances:
        if inst.is_instance and inst.parent and inst.parent.original == go_:
            P.append(tuple(inst.matrix_world.translation))
    P = np.array(P) if P else np.zeros((0, 3))
    bad_kc = int(sum(keep_clear(p[0], p[1]) for p in P))
    in_proof = int(pts_in_proof(P + np.array([0, 0, 0.0])).sum()) if len(P) else 0
    hall = int(((np.abs(P[:, 0]) < IX + 3) & (P[:, 1] > IY0 - 3) & (P[:, 1] < IY1 + 3)).sum()) if len(P) else 0
    on_path = 0
    if PATH_OB is not None:
        for p in P[:: max(1, len(P) // 4000)]:
            loc = PATH_OB.ray_cast(Vector((p[0], p[1], p[2] + 5)), Vector((0, 0, -1)), 10.0)[0]
            on_path += loc is not None
    # grounded: every tuft sits on its ground (paving z=0 inside; the terrain outside)
    inwall = (rect_dist(P[:, 0], P[:, 1]) <= 0) & ~((np.abs(P[:, 0]) < CXc) & (P[:, 1] > CY0c) & (P[:, 1] < CY1c)) if len(P) else np.zeros(0, bool)
    GR["hidden_inside_wall_body"] = int(inwall.sum())
    samp = P[~inwall][:: max(1, int((~inwall).sum()) // 3000)]
    errs = [(abs(p[2] - (0.0 if (-CXc < p[0] < CXc and CY0c < p[1] < CY1c) else th(p[0], p[1]))), tuple(np.round(p, 2))) for p in samp]
    gerr = max((e for e, _ in errs), default=0.0)
    GR["worst_ground_points"] = [w for w in sorted(errs, reverse=True)[:5]]
    GR.update(instances=len(P), max_density_per_m2=DENS_MAX, emitter_cells=len(Fc), covered_cells_skipped=n_cov,
              keep_clear_violations=bad_kc, in_dragon_footprints=in_proof, in_hall_or_platform=hall,
              on_outside_path_sampled=on_path, max_ground_error_m=round(float(gerr), 3),
              keep_clear="hall/platform/stairs |x|<25.5, y -28..56; main path gate->stairs |x|<6.5, y -82..-24; outside path",
              build_s=round(time.time() - T_G, 1))
    log("grass", GR)
    assert bad_kc == 0 and in_proof == 0 and hall == 0 and on_path == 0, GR
    assert gerr < 0.05, ("grass floats/sinks", gerr)


EXT = None
if not A.no_exterior:
    t = time.time()
    EXT = build_exterior()
    t = tick("exterior", t)
    build_grass(EXT["th"])
    t = tick("grass", t)


# ---------------------------------------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------------------------------------
t = time.time()
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
V3_MESHES = [o for c in (C_COVER, C_RUB, C_GRAV, C_HOARD3) for o in c.all_objects if o.type == "MESH"]

# 1. 12-dragon footprints: vertex-level for every v3 mesh, plus the v2 bbox-level check over v2 + v3 objects
hits_v = [o.name for o in V3_MESHES if pts_in_proof(world_co(o)).any()]
bbox_hits = []
for cn in ("V2_STUMPS", "V2_RUBBLE", "V2_STATUES_INSIDE", "V2_COLOSSI", "V2_COMPOUND", "V3_COVER", "V3_RUBBLE", "V3_HOARD"):
    for o in D.collections[cn].all_objects:
        if o.type != "MESH":
            continue
        lo_, hi_ = world_bbox(o)
        if hi_[2] > F - 0.5 and lo_[2] < F + 30 and any(hi_[0] > a and lo_[0] < c and hi_[1] > b and lo_[1] < d for a, b, c, d in PROOF_BOX):
            bbox_hits.append(o.name)
REPORT["checks"]["twelve_dragons_fit_v3"] = dict(ok=not hits_v and not bbox_hits, v3_vertices_on_proof_footprints=hits_v,
                                                 v2v3_bboxes_on_proof_footprints=bbox_hits,
                                                 note="v1 geometry assert (3x4 dragons) unchanged; v3: no v3 vertex above the floor "
                                                      "inside any proof footprint + the v2 bbox check re-run over v2 and v3")
assert not hits_v and not bbox_hits, REPORT["checks"]["twelve_dragons_fit_v3"]

# 2. main aisle (gate -> dais) clear of v3; report anything older standing in it
aisle_v3 = [o.name for o in V3_MESHES if pts_in_aisle(world_co(o)).any()]
aisle_old = sorted({n for n in STATIC_NAMES + LOOSE_NAMES if n in D.objects and pts_in_aisle(TRIS[n][0]).any()})
REPORT["checks"]["main_aisle_clear"] = dict(ok=not aisle_v3, corridor=f"|x| < {AISLE_HALF} m, y {MAIN_AISLE_Y[0]} .. {MAIN_AISLE_Y[1]}",
                                            v3_objects_in_it=aisle_v3, older_objects_in_it=aisle_old)
assert not aisle_v3, aisle_v3

# 3-5. cover: heights, grounding + stability, interpenetration, crouch spot, line of sight
def vol_centroid(o):
    """true volume centroid (signed tetrahedra) in world space — vertex means are biased by bevel loops."""
    w, t_ = tris_world(o, bpy.context.evaluated_depsgraph_get())
    a, b, c = w[t_[:, 0]], w[t_[:, 1]], w[t_[:, 2]]
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    if abs(v.sum()) < 1e-9:
        return w.mean(0)
    return ((a + b + c) / 4.0 * v[:, None]).sum(0) / v.sum()


def hull2d(P):
    P = sorted(set(map(tuple, np.round(P, 4))))
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def in_hull(h, q, tol=0.01):
    if len(h) >= 3:          # a sliver hull (< 4 cm wide) is a LINE support (a drum lying on its side)
        H = np.array(h)
        area = 0.5 * abs(np.dot(H[:, 0], np.roll(H[:, 1], 1)) - np.dot(H[:, 1], np.roll(H[:, 0], 1)))
        span = max(np.linalg.norm(H[i] - H[j]) for i in range(len(H)) for j in range(len(H)))
        if area / max(span, 1e-6) < 0.04:
            i, j = max(((i, j) for i in range(len(H)) for j in range(len(H))), key=lambda ij: np.linalg.norm(H[ij[0]] - H[ij[1]]))
            h = [tuple(H[i]), tuple(H[j])]
    if len(h) < 3:          # line support (a drum): within tol of the segment
        if len(h) < 2:
            return False
        a, b = np.array(h[0]), np.array(h[-1])
        ab = b - a
        s = np.clip(np.dot(np.array(q) - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        return np.linalg.norm(a + s * ab - np.array(q)) < 0.05
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) < -tol:
            return False
    return True


FLOOR_BVH = bvh_of([tris_world(D.objects["ArenaFloor"], dg)])
STATIC_BVH2 = bvh_of([TRIS[n] for n in STATIC_NAMES + LOOSE_NAMES if n in D.objects])
EVERY_BVH = bvh_of([TRIS[n] for n in STATIC_NAMES + LOOSE_NAMES if n in D.objects] + [TRIS[o.name] for o in COVER_OBJS]
                   + [tris_world(o, dg) for o in DYN])
cover_rep, problems = [], []
for c in COVER + [dict(id=o.name, kind="tipped_pew", objs=[o], dressing=True) for o in TIPPED_OBJS]:
    hs, grounded, stable, pen = [], True, True, []
    for o in c["objs"]:
        w = world_co(o)
        hs.append(float(w[:, 2].max() - F))
        zmin = float(w[:, 2].min())
        low = w[w[:, 2] < zmin + 0.015]
        # floor really under the lowest vertices
        for q in low[:: max(1, len(low) // 6)]:
            loc = FLOOR_BVH.ray_cast(Vector((q[0], q[1], zmin + 0.05)), Vector((0, 0, -1)), 0.2)[0]
            if loc is None or abs(loc.z - zmin) > 0.01:
                grounded = False
        grounded &= abs(zmin - F) < 0.01
        com = vol_centroid(o)
        st_ = in_hull(hull2d(low[:, :2]), com[:2])
        if not st_:
            log("  unstable?", o.name, "com", np.round(com, 3).tolist(), "support hull", np.round(hull2d(low[:, :2]), 3).tolist())
        stable &= bool(st_)
        b = obj_bvh(o)
        if b.overlap(STATIC_BVH2):
            pen.append(o.name + " x static")
        for o2 in COVER_OBJS:
            if o2 is not o and b.overlap(obj_bvh(o2)):
                pen.append(o.name + " x " + o2.name)
    rec = dict(id=c["id"], kind=c["kind"], pieces=len(c["objs"]), height_m=round(max(hs), 2),
               heights_m=[round(h, 2) for h in hs], grounded=grounded, stable=stable, interpenetrations=pen)
    if not c.get("dressing"):
        cr = c["crouch"] if c["crouch"] is not None else Vector((0.0, 0.0))
        crouch_free = c["crouch"] is not None
        for z in (0.35, 0.8):
            hit = EVERY_BVH.find_nearest(Vector((cr.x, cr.y, F + z)))
            crouch_free &= not (hit[0] is not None and hit[3] < 0.3)
        eye = Vector((cr.x, cr.y, F + 0.8))
        d = c["threat"] - eye
        orig, left, hob, hit = eye.copy(), d.length, None, False
        for _ in range(40):
            hit, loc, _n, _i, hob, _m = SC.ray_cast(dg, orig, d.normalized(), distance=left)
            if not hit or not (hob.name.startswith(("Rubble", "V3Gravel")) or hob.users_collection[0].name in ("SIZE_PROOF", "FX_haze")):
                break
            left -= (loc - orig).length + 0.01
            orig = loc + d.normalized() * 0.01
        los_blocked = bool(hit and hob.get("cover_spot") == c["id"])
        rec.update(centre=[round(c["centre"].x, 2), round(c["centre"].y, 2)], crouch=[round(cr.x, 2), round(cr.y, 2)],
                   threat=c["threat_kind"],
                   crouch_free=crouch_free, los_to_threat_blocked_by_this_cover=los_blocked, first_hit=hob.name if hit else None,
                   nudge_m=c["nudge"])
        in_range = all(COVER_H[0] <= h <= COVER_H[1] for h in hs)
        rec["height_in_range"] = in_range
        if not (in_range and grounded and stable and not pen and crouch_free and los_blocked):
            problems.append(rec)
    elif not (grounded and stable and not pen):
        problems.append(rec)
    cover_rep.append(rec)
REPORT["added"]["cover"] = cover_rep
ys = sorted(c["centre"].y for c in COVER)
ys = [y for y in ys if y < DAIS_Y0]
gaps = [b - a for a, b in zip([IY0] + ys, ys + [DAIS_Y0])]
REPORT["checks"]["cover"] = dict(ok=not problems, spots=len(COVER), pieces=sum(len(c["objs"]) for c in COVER),
                                 heights_m=[min(r["height_m"] for r in cover_rep if r["kind"] != "tipped_pew"),
                                            max(r["height_m"] for r in cover_rep if r["kind"] != "tipped_pew")],
                                 max_gap_along_route_m=round(max(gaps), 2), problems=problems)
log("cover check", {k: v for k, v in REPORT["checks"]["cover"].items() if k != "problems"}, "problems:", len(problems))
for pr in problems:
    log("  PROBLEM", pr)
assert not problems, problems
assert max(gaps) <= 9.0, ("cover gap along the route", gaps)

# rubble interpenetration (report): v3 pieces overlapping cover / static / each other after the settle
n_pen = 0
if DYN:
    rb = {o.name: obj_bvh(o) for o in DYN}
    for o in DYN:
        if rb[o.name].overlap(STATIC_BVH2) or any(rb[o.name].overlap(obj_bvh(c_)) for c_ in COVER_OBJS
                                                  if (Vector(c_.matrix_world.translation) - o.matrix_world.translation).length < 4):
            n_pen += 1
REPORT["checks"]["rubble_v3_touching_or_overlapping"] = dict(pieces=n_pen, of=len(DYN),
                                                            note="BVH surface overlap (convex-hull sim => contact-level touches count here)")
t = tick("checks", t)


# ---------------------------------------------------------------------------------------------------------
# CAMERAS (render_stills.py picks CAM_<name>)
# ---------------------------------------------------------------------------------------------------------
def look_at(ob, target):
    ob.rotation_euler = (Vector(target) - ob.location).to_track_quat("-Z", "Y").to_euler()


def cam(name, loc, target, lens, ortho=None, rotz=None, fstop=None, focus=None):
    old = D.objects.get(name)
    if old:
        D.objects.remove(old)
    cd = D.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36
    cd.clip_start = 0.05
    cd.clip_end = 6000
    o = D.objects.new(name, cd)
    C_CAM.objects.link(o)
    o.location = loc
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho
        o.rotation_euler = (0, 0, rotz or 0)
    else:
        look_at(o, target)
    if fstop:
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        cd.dof.focus_distance = focus if focus else (Vector(target) - Vector(loc)).length
    return o


SPOT = {c["id"]: c for c in COVER}
cr = SPOT["S1W"]["crouch"]
cE = SPOT["E0"]
CAMS3 = {
    # knight's eye: crouched (0.95 m) behind the fallen column, looking past the drums at the knight and the dragon
    "i_cover": cam("CAM_i_cover", (cr.x + 0.3, cr.y - 0.3, F + 1.12), (-1.5, DRAGON_POS.y, F + 2.6), 22, fstop=4.0,
                   focus=(Vector((1.0, 25.5, F + 1.2)) - Vector((cr.x + 0.3, cr.y - 0.3, F + 1.12))).length),
    # the knight's route from just inside the gate (raised): strip cover between the dragon rows, band cover each side
    "j_cover_run": cam("CAM_j_cover_run", (3.0, 0.9, F + 4.8), (-2.0, 32.0, F), 16),
    # roof-off plan of the hall (QA): d_* = render_stills hides roof, dragon, haze and shows the 12-dragon proof
    "d_hallplan": cam("CAM_d_hallplan", (0, 25.0, F + 60), None, 50, ortho=60, rotz=math.radians(90)),
    # hoard wide: the dais from the nave's east side — chests, spills, planted swords, wall displays, the RPG on its plinth
    "e_hoard": cam("CAM_e_hoard", (0.6, 42.4, F + 4.3), (0.0, 48.9, F + 2.1), 18),
    # exterior: from the south wall top over the gate (path, woods, fields, town) and a raised wide over the valley
    "k_town": cam("CAM_k_town", (7.0, -80.2, 14.6), (78.0, -470.0, -4.0), 42),
    "l_valley": cam("CAM_l_valley", (-75.0, -55.0, 62.0), (60.0, -360.0, -6.0), 24),
    # v3c: low in the courtyard, south-west corner: wall-base grass + weeds near camera, houses beyond
    "m_grass": cam("CAM_m_grass", M_GRASS_CAM, (-42.0, -77.2, 0.5), 24),     # along the moonlit south-wall base
}
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
for k in ("c_hall", "h_aisle", "e_prize", "i_cover", "j_cover_run", "e_hoard") + (() if A.no_exterior else ("k_town", "l_valley")):
    o = D.objects["CAM_" + k]
    fwd = o.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    orig, hit, hob, dist = o.matrix_world.translation.copy(), False, None, None
    for _ in range(60):
        hit, loc, _n, _i, hob, _m = SC.ray_cast(dg, orig, fwd)
        if not hit:
            break
        if hob.users_collection[0].name in ("FX_haze", "SIZE_PROOF"):
            orig = loc + fwd * 0.05
            continue
        dist = (loc - o.matrix_world.translation).length
        break
    REPORT["checks"][f"cam_{k}_first_hit"] = dict(obj=hob.name if hit else None, dist=round(dist, 2) if dist else None)
    assert (not hit) or dist > 1.0, (k, hob.name, dist)
    # the camera itself is not inside anything
    near = SOLID_BVH.find_nearest(o.matrix_world.translation)
    assert near[0] is None or near[3] > 0.15, (k, "camera within 15 cm of geometry", near[0], near[3])
# every still camera: not INSIDE an object (6 axis rays all hitting the same mesh within 4 m = inside it)
for k in [n[4:] for n in (o.name for o in C_CAM.objects) if n.startswith("CAM_")]:
    o = D.objects["CAM_" + k]
    if o.data.type == "ORTHO":
        continue
    hits = []
    for d in (Vector((1, 0, 0)), Vector((-1, 0, 0)), Vector((0, 1, 0)), Vector((0, -1, 0)), Vector((0, 0, 1)), Vector((0, 0, -1))):
        hit, loc, _n, _i, hob, _m = SC.ray_cast(dg, o.matrix_world.translation, d, distance=4.0)
        hits.append(hob.name if hit and hob.users_collection[0].name not in ("FX_haze", "SIZE_PROOF", "V3_EXT_FOG") else None)
    inside = hits[0] is not None and all(h == hits[0] for h in hits)
    REPORT["checks"].setdefault("cameras_not_inside_objects", {})[k] = not inside
    assert not inside, (k, "camera inside", hits[0])

# the RPG-7 stays the hero of e_prize: no new hoard vertex inside that frame
from bpy_extras.object_utils import world_to_camera_view
cp = D.objects["CAM_e_prize"]
intr = []
for o in C_HOARD3.objects:
    if o.type != "MESH":
        continue
    w = world_co(o)
    for q in w[:: max(1, len(w) // 400)]:
        v = world_to_camera_view(SC, cp, Vector(q))
        if 0 < v.x < 1 and 0 < v.y < 1 and v.z > 0:
            intr.append((o.name, [round(float(c), 2) for c in q], [round(v.x, 2), round(v.y, 2), round(v.z, 2)]))
            break
REPORT["checks"]["e_prize_frame_free_of_new_hoard"] = dict(ok=not intr, intruders=intr)
assert not intr, intr

# exterior shots: no square world edge (every terrain hit from the exterior cameras is well inside the 4 km tile)
if EXT:
    edge = {}
    for k in ("a_compound", "a_compound_se", "k_town", "l_valley", "b_colossi", "f_colossus"):
        o = D.objects["CAM_" + k]
        fr_ = [o.matrix_world @ v for v in o.data.view_frame(scene=SC)]
        org = o.matrix_world.translation
        worst = 0.0
        for i in range(48):
            for j in range(27):
                a_, b_ = (i + 0.5) / 48, (j + 0.5) / 27
                top = fr_[3].lerp(fr_[0], a_)
                bot = fr_[2].lerp(fr_[1], a_)
                pt = top.lerp(bot, b_)
                loc = EXT["TER"].ray_cast(org, (pt - org).normalized(), 12000.0)[0]
                if loc is not None:
                    worst = max(worst, abs(loc.x), abs(loc.y))
        edge[k] = round(worst, 1)
    lim = 0.95 * EXT_HALF[0]
    REPORT["checks"]["no_world_edge_in_shots"] = dict(ok=all(v < lim for v in edge.values()), max_terrain_hit_xy=edge,
                                                      tile_half_m=EXT_HALF[0])
    assert all(v < lim for v in edge.values()), edge

# ---------------------------------------------------------------------------------------------------------
# cleanup + save
# ---------------------------------------------------------------------------------------------------------
for o in list(C_TMP.objects):
    D.objects.remove(o)
D.collections.remove(C_TMP)
for me in list(D.meshes):
    if me.users == 0:
        D.meshes.remove(me)
SC.frame_set(1)
REPORT["added"]["counts"] = dict(cover_spots=len(COVER), cover_pieces=sum(len(c["objs"]) for c in COVER),
                                 tipped_pews_dressing=len(TIPPED_OBJS), rubble_settled=len(DYN), gravel_chips=n_chips_total,
                                 v1v2_loose_removed_under_cover=len(REMOVED))
# v3 adds ~3.8k shadow casters: EEVEE overflowed the 512 MB shadow pool ("Shadow buffer full" on every interior still)
SC.eevee.shadow_pool_size = "1024"
REPORT["objects"] = len(D.objects)
REPORT["tris_v3"] = int(sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in V3_MESHES))
REPORT["build_s"] = round(time.time() - T0, 1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(A.out), compress=True)
log("saved", A.out, "objects", len(D.objects), "build", REPORT["build_s"], "s")
if A.report:
    json.dump(REPORT, open(A.report, "w"), indent=1, default=str)
