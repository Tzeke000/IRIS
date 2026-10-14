"""BOSS FIGHT — arena v1 (knight vs European dragon, night). Builds the whole arena from nothing -> .blend.

    blender -b --factory-startup --python-exit-code 1 --python build_arena.py -- --out arena_v1.blend [--report r.json]
ALWAYS launched through:  scripts/server/render.sh <unit> scripts/lyric_scenes/real/boss_fight/render.sh build

Layout (Z up, metres, north = +Y). The knight comes from the courtyard (south, z=0), climbs a grand two-flight staircase
to the arena: a ruined gothic hall raised F=8 m on a stone platform. Inner floor 40 m (X) x 50 m (Y).
  * nave between two arcades of 4 stout gothic piers (x=+-13, y=6.25..43.75, bay 12.5 m) — every pier + its arch is
    PRE-FRACTURED (seeded Voronoi cell_fracture); the cells sit hidden, stacked exactly where the intact pier is.
  * north half of the nave is roofed by a pointed barrel vault + transverse ribs, also PRE-FRACTURED (the dragon
    bursts out through it); the south half of the roof is already gone (moonlight + open sky over the approach).
  * the dragon's hoard + the prize (gold rocket-launcher PLACEHOLDER, swap at PRIZE_anchor) on a dais under the rose window.
Size interpretation (assert-checked below): 12 dragons standing nose-to-tail with wings folded, 3 rows x 4, fit on the
nave floor between the piers without touching each other, the piers or the dais.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import addon_utils
import bmesh
import bpy
import mathutils
import numpy as np
from mathutils import Matrix, Vector

T_START = time.time()
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--report", default="")
A = ap.parse_args(argv)

addon_utils.enable("bl_ext.blender_org.cell_fracture", default_set=True)

SF = "/data/iris_assets/medieval/sketchfab/"
PH = "/data/iris_assets/medieval/polyhaven/"
REPORT = {"fracture": {}, "checks": {}}

# ---------------------------------------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------------------------------------
F = 8.0                       # arena floor height above the courtyard
IX = 20.0                     # inner half-width (x = -20..20)
IY0, IY1 = 0.0, 50.0          # inner floor y range
WT = 2.5                      # outer wall thickness
COL_X = 13.0                  # arcade (pier) line
COL_Y = [6.25, 18.75, 31.25, 43.75]
BAY = 12.5
COL_S = (BAY / 2) / 8.745     # gothic-kit bay half-span 8.745 m at import -> 6.25 m
DRAGON_S = 5.0                # model 2.8 u nose-to-tail (Walk) -> 14 m; wingspan 3.71 u (Fly) -> 18.6 m
KNIGHT_S = 1.85 / 2.85        # helmet crest 2.85 u -> 1.85 m
ZS_VAULT = F + 22.5           # vault springing (just above the arcade arches, 21.7 m)
VAULT_H = 15.0                # rise of the pointed vault (intrados); MUST exceed the half-span (13 m) or the arcs dip into a V
VAULT_T = 1.2
DAIS_Y0 = 45.5

# stairs: 2 flights x 25 risers of 0.16 m (8.0 m), tread 0.36 m, 4 m landing, 14 m wide
ST_W = 7.0
RISE, TREAD, NR = 0.16, 0.36, 25
ST_Y_TOP = -2.5
ST_RUN = 2 * (NR - 1) * TREAD + 4.0 + 0.72
ST_Y0 = ST_Y_TOP - ST_RUN

WORLD_LIGHT, WORLD_CAM, MOON_E, FIRE_E = 0.18, 0.45, 1.4, 650.0     # night balance (tuned on stills)

random.seed(1414)
mathutils.noise.seed_set(1414)

# ---------------------------------------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("bl_ext.blender_org.cell_fracture", default_set=True)
SC = bpy.context.scene
SC.name = "BossArena"
SC.render.fps = 30
SC.frame_start, SC.frame_end = 1, 300
SC.render.resolution_x, SC.render.resolution_y = 1280, 720
SC.render.engine = "BLENDER_EEVEE"
SC.view_settings.view_transform = "AgX"
SC.view_settings.look = "AgX - Medium High Contrast"
SC.view_settings.exposure = 0.0
SC.unit_settings.system = "METRIC"
ee = SC.eevee
ee.taa_render_samples = 64
ee.use_raytracing = True
ee.ray_tracing_method = "SCREEN"
ee.ray_tracing_options.resolution_scale = "2"
ee.ray_tracing_options.trace_max_roughness = 0.4
ee.use_fast_gi = True
ee.use_shadows = True
ee.shadow_ray_count = 1
ee.shadow_step_count = 6
ee.clamp_surface_indirect = 10.0
ee.volumetric_start = 0.5
ee.volumetric_end = 400.0
ee.volumetric_tile_size = "8"
ee.volumetric_samples = 64
ee.volumetric_sample_distribution = 0.8
ee.use_volumetric_shadows = True


def coll(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or SC.collection).children.link(c)
    return c


C_ARENA = coll("ARENA_static")
C_PILLARS = coll("PILLARS_breakable")
C_ROOF = coll("ROOF_breakable")
C_RUIN = coll("ROOF_static_ruin")
C_COURT = coll("COURTYARD")
C_PROPS = coll("PROPS")
C_HOARD = coll("HOARD")
C_PRIZE = coll("PRIZE_placeholder")
C_CHARS = coll("CHARACTERS")
C_LIGHTS = coll("LIGHTS_FIRE")
C_FX = coll("FX_haze")
C_PROOF = coll("SIZE_PROOF")          # visible only in the overhead still (render_stills.py)
C_PROOF.hide_render = True
C_CAM = coll("CAMERAS")
C_TMP = coll("_tmp")


def link(ob, c):
    for u in list(ob.users_collection):
        u.objects.unlink(ob)
    c.objects.link(ob)
    return ob


# ---------------------------------------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------------------------------------
_IMG = {}


def img(path, color=True):
    key = (path, color)
    if key not in _IMG:
        im = bpy.data.images.load(path, check_existing=True)
        if not color:
            im.colorspace_settings.name = "Non-Color"
        _IMG[key] = im
    return _IMG[key]


def new_mat(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


def pbr(name, tex, tile=3.0, tint=(1, 1, 1), bump=1.0, grime=0.35, rough_add=0.0, moss=0.0):
    """Poly Haven scan, box-projected in object space (tile = metres per repeat), with macro grime variation."""
    d = f"{PH}tex_{tex}/{tex}_"
    m, nt, out = new_mat(name)
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / tile,) * 3
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    def tx(f, col):
        t = N.new("ShaderNodeTexImage")
        t.image = img(d + f + "_2k.jpg", col)
        t.projection = "BOX"
        t.projection_blend = 0.25
        L.new(mp.outputs["Vector"], t.inputs["Vector"])
        return t
    tdif, trough, tnor = tx("diff", True), tx("rough", False), tx("nor_gl", False)
    b = N.new("ShaderNodeBsdfPrincipled")
    # macro grime: large-scale noise darkens/greens the scan so repeats don't read
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.08
    nz.inputs["Detail"].default_value = 6
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.35, 0.7
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.0, 1.0 - grime
    L.new(nz.outputs["Fac"], mr.inputs["Value"])
    mul = N.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    L.new(tdif.outputs["Color"], mul.inputs[6])
    mul.inputs[7].default_value = (*tint, 1)
    mul2 = N.new("ShaderNodeMix")
    mul2.data_type = "RGBA"
    mul2.blend_type = "MULTIPLY"
    mul2.inputs["Factor"].default_value = 1.0
    L.new(mul.outputs[2], mul2.inputs[6])
    L.new(mr.outputs["Result"], mul2.inputs[7])
    base = mul2.outputs[2]
    if moss > 0:
        # moss in crevices/up-facing: noise mask * up-facing normal
        mz = N.new("ShaderNodeTexNoise")
        mz.inputs["Scale"].default_value = 0.6
        mz.inputs["Detail"].default_value = 8
        L.new(tc.outputs["Object"], mz.inputs["Vector"])
        mm = N.new("ShaderNodeMapRange")
        mm.inputs["From Min"].default_value, mm.inputs["From Max"].default_value = 0.55, 0.75
        mm.inputs["To Max"].default_value = moss
        L.new(mz.outputs["Fac"], mm.inputs["Value"])
        mx = N.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        L.new(mm.outputs["Result"], mx.inputs["Factor"])
        L.new(base, mx.inputs[6])
        mx.inputs[7].default_value = (0.045, 0.06, 0.025, 1)
        base = mx.outputs[2]
    L.new(base, b.inputs["Base Color"])
    if rough_add:
        ra = N.new("ShaderNodeMath")
        ra.operation = "ADD"
        ra.use_clamp = True
        L.new(trough.outputs["Color"], ra.inputs[0])
        ra.inputs[1].default_value = rough_add
        L.new(ra.outputs[0], b.inputs["Roughness"])
    else:
        L.new(trough.outputs["Color"], b.inputs["Roughness"])
    nm = N.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = bump
    L.new(tnor.outputs["Color"], nm.inputs["Color"])
    L.new(nm.outputs["Normal"], b.inputs["Normal"])
    L.new(b.outputs[0], out.inputs["Surface"])
    return m


def simple(name, base, rough=0.5, metal=0.0, emis=None, estr=0.0):
    m, nt, out = new_mat(name)
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emis:
        b.inputs["Emission Color"].default_value = (*emis, 1)
        b.inputs["Emission Strength"].default_value = estr
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def gold_mat(name="M_Gold", coins=False):
    m, nt, out = new_mat(name)
    N, L = nt.nodes, nt.links
    b = N.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (1.0, 0.70, 0.30, 1)
    b.inputs["Metallic"].default_value = 1.0
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.0
    rr = N.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = 0.16, 0.38
    L.new(nz.outputs["Fac"], rr.inputs["Value"])
    L.new(rr.outputs["Result"], b.inputs["Roughness"])
    if coins:   # a heap of coins: voronoi cells as bumps
        tc = N.new("ShaderNodeTexCoord")
        vo = N.new("ShaderNodeTexVoronoi")
        vo.feature = "DISTANCE_TO_EDGE" if hasattr(vo, "feature") else vo.feature
        vo.inputs["Scale"].default_value = 18.0
        L.new(tc.outputs["Object"], vo.inputs["Vector"])
        bp = N.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = 0.8
        bp.inputs["Distance"].default_value = 0.02
        L.new(vo.outputs["Distance"], bp.inputs["Height"])
        L.new(bp.outputs["Normal"], b.inputs["Normal"])
    L.new(b.outputs[0], out.inputs["Surface"])
    return m


M_WALL = pbr("M_Wall", "rough_block_wall", tile=3.5, tint=(0.78, 0.78, 0.8), moss=0.3)
M_WALL2 = pbr("M_WallOld", "castle_wall_varriation", tile=3.0, tint=(0.8, 0.8, 0.8), moss=0.3)
M_PLATFORM = pbr("M_Platform", "castle_wall_slates", tile=4.0, tint=(0.8, 0.8, 0.8), moss=0.5)
M_FLOOR = pbr("M_ArenaFloor", "stone_tiles_02", tile=3.5, tint=(0.62, 0.62, 0.64), grime=0.55, moss=0.25)
M_STAIRS = pbr("M_Stairs", "cobblestone_floor_08", tile=1.6, tint=(0.7, 0.7, 0.72), grime=0.45, moss=0.3)
M_COURT = pbr("M_Courtyard", "grassy_cobblestone", tile=3.0, tint=(0.55, 0.55, 0.55), grime=0.6, moss=0.4)
M_GROUND = pbr("M_Ground", "forest_ground_04", tile=8.0, tint=(0.09, 0.1, 0.08), grime=0.6, moss=0.8)
M_VAULT = pbr("M_Vault", "old_stone_wall", tile=3.0, tint=(0.8, 0.8, 0.8), moss=0.2)
M_ROOFSLATE = pbr("M_RoofSlate", "roof_slates_02", tile=3.0, moss=0.4)
M_CURTAIN = pbr("M_Curtain", "castle_wall_varriation", tile=4.0, tint=(0.75, 0.75, 0.75), moss=0.45)
M_INTERIOR = pbr("M_StoneInterior", "stone_tiles_02", tile=1.2, tint=(0.95, 0.93, 0.9), grime=0.2)
M_GOLD = gold_mat()
M_COINS = gold_mat("M_GoldCoins", coins=True)
M_IRON = simple("M_Iron", (0.06, 0.055, 0.05), rough=0.55, metal=1.0)


def flame_mat(name="M_Flame", strength=18.0):
    m, nt, out = new_mat(name)
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Generated"], sep.inputs["Vector"])
    bb = N.new("ShaderNodeBlackbody")
    tmr = N.new("ShaderNodeMapRange")       # hotter at the base
    tmr.inputs["To Min"].default_value, tmr.inputs["To Max"].default_value = 1750, 1150
    L.new(sep.outputs["Z"], tmr.inputs["Value"])
    L.new(tmr.outputs["Result"], bb.inputs["Temperature"])
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 4.0
    nz.inputs["Detail"].default_value = 4
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    a1 = N.new("ShaderNodeMapRange")        # alpha: fades to the tip + noisy tongues
    a1.inputs["From Min"].default_value, a1.inputs["From Max"].default_value = 0.15, 1.0
    a1.inputs["To Min"].default_value, a1.inputs["To Max"].default_value = 1.0, 0.0
    L.new(sep.outputs["Z"], a1.inputs["Value"])
    am = N.new("ShaderNodeMath")
    am.operation = "MULTIPLY"
    L.new(a1.outputs["Result"], am.inputs[0])
    nm = N.new("ShaderNodeMapRange")
    nm.inputs["From Min"].default_value, nm.inputs["From Max"].default_value = 0.35, 0.6
    L.new(nz.outputs["Fac"], nm.inputs["Value"])
    L.new(nm.outputs["Result"], am.inputs[1])
    em = N.new("ShaderNodeEmission")
    L.new(bb.outputs["Color"], em.inputs["Color"])
    em.inputs["Strength"].default_value = strength
    tr = N.new("ShaderNodeBsdfTransparent")
    mix = N.new("ShaderNodeMixShader")
    L.new(am.outputs[0], mix.inputs["Fac"])
    L.new(tr.outputs[0], mix.inputs[1])
    L.new(em.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED" if hasattr(m, "surface_render_method") else None
    return m


M_FLAME = flame_mat(strength=3.2)


def stained_glass_mat():
    m, nt, out = new_mat("M_StainedGlass")
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    vo = N.new("ShaderNodeTexVoronoi")
    vo.inputs["Scale"].default_value = 2.2
    L.new(tc.outputs["Object"], vo.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "CONSTANT"
    cols = [(0.02, 0.06, 0.5), (0.5, 0.03, 0.02), (0.05, 0.25, 0.08), (0.6, 0.4, 0.05), (0.15, 0.03, 0.4)]
    cr.elements[0].color = (*cols[0], 1)
    cr.elements[1].position = 0.2
    cr.elements[1].color = (*cols[1], 1)
    for i, c in enumerate(cols[2:]):
        e = cr.elements.new(0.4 + 0.2 * i)
        e.color = (*c, 1)
    L.new(vo.outputs["Color"], ramp.inputs["Fac"])
    ve = N.new("ShaderNodeTexVoronoi")          # lead cames between the panes
    ve.feature = "DISTANCE_TO_EDGE"
    ve.inputs["Scale"].default_value = 2.2
    L.new(tc.outputs["Object"], ve.inputs["Vector"])
    lead = N.new("ShaderNodeMapRange")
    lead.inputs["From Min"].default_value, lead.inputs["From Max"].default_value = 0.02, 0.06
    L.new(ve.outputs["Distance"], lead.inputs["Value"])
    mul = N.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    L.new(ramp.outputs["Color"], mul.inputs[6])
    L.new(lead.outputs["Result"], mul.inputs[7])
    em = N.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.1
    L.new(mul.outputs[2], em.inputs["Color"])
    L.new(em.outputs[0], out.inputs["Surface"])
    return m


M_GLASS = stained_glass_mat()


def haze_mat(name, density, color=(0.75, 0.82, 1.0), aniso=0.35):
    m, nt, out = new_mat(name)
    v = nt.nodes.new("ShaderNodeVolumePrincipled")
    v.inputs["Density"].default_value = density
    v.inputs["Color"].default_value = (*color, 1)
    v.inputs["Anisotropy"].default_value = aniso
    nt.links.new(v.outputs[0], out.inputs["Volume"])
    return m


# ---------------------------------------------------------------------------------------------------------
# geometry helpers (bmesh)
# ---------------------------------------------------------------------------------------------------------
def _p3(axis, a, u, v):
    if axis == "X":
        return (a, u, v)
    if axis == "Y":
        return (u, a, v)
    return (u, v, a)


def bm_obj(name, bm, mats, c, smooth=False):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    c.objects.link(ob)
    return ob


def add_extrusion(bm, pts, axis, a0, a1, mi=0):
    """closed 2-D polygon pts (u,v) in the plane normal to axis, extruded from a0 to a1."""
    lo = [bm.verts.new(_p3(axis, a0, u, v)) for u, v in pts]
    hi = [bm.verts.new(_p3(axis, a1, u, v)) for u, v in pts]
    n = len(pts)
    fs = [bm.faces.new(lo[::-1]), bm.faces.new(hi)]
    for i in range(n):
        j = (i + 1) % n
        fs.append(bm.faces.new((lo[i], lo[j], hi[j], hi[i])))
    for f in fs:
        f.material_index = mi
    return fs


def extrusion(name, pts, axis, a0, a1, mats, c):
    bm = bmesh.new()
    add_extrusion(bm, pts, axis, a0, a1)
    return bm_obj(name, bm, mats, c)


def add_box(bm, lo, hi, mi=0):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return add_extrusion(bm, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "Z", z0, z1, mi)


def boxes(name, specs, mats, c):
    bm = bmesh.new()
    for s in specs:
        add_box(bm, *s)
    return bm_obj(name, bm, mats, c)


def sweep_y(name, prof, y0_fn, y1_fn, nseg, mats, c, band=None):
    """closed (x,z) profile swept along +Y; per-vertex ragged start/end y.
    band=n: prof = outer[0..n-1] + inner[n-1..0] (two matching polylines) → caps are quad strips (a crescent ngon
    triangulates across its hollow and self-intersects). Otherwise caps are convex ngons."""
    bm = bmesh.new()
    rings = []
    for k in range(nseg + 1):
        t = k / nseg
        ring = []
        for (x, z) in prof:
            y0, y1 = y0_fn(x, z), y1_fn(x, z)
            ring.append(bm.verts.new((x, y0 + (y1 - y0) * t, z)))
        rings.append(ring)
    n = len(prof)
    for k in range(nseg):
        a, b = rings[k], rings[k + 1]
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
    for ring in (rings[0], rings[-1]):
        if band:
            outer, inner = ring[:band], ring[band:][::-1]
            for i in range(band - 1):
                bm.faces.new((outer[i], outer[i + 1], inner[i + 1], inner[i]))
        else:
            bm.faces.new(ring)
    return bm_obj(name, bm, mats, c)


def pointed_window(cu, w, z0, zs, n=8):
    """equilateral pointed-arch opening outline (u = along the wall, v = z)."""
    pts = [(cu - w / 2, z0), (cu + w / 2, z0)]
    for i in range(n + 1):          # right arc, centred on the left springer
        t = math.radians(60) * i / n
        pts.append((cu - w / 2 + w * math.cos(t), zs + w * math.sin(t)))
    for i in range(n - 1, -1, -1):  # left arc, centred on the right springer
        t = math.radians(60) * i / n
        pts.append((cu + w / 2 - w * math.cos(t), zs + w * math.sin(t)))
    # drop duplicated apex
    out = []
    for p in pts:
        if not out or (abs(out[-1][0] - p[0]) + abs(out[-1][1] - p[1])) > 1e-6:
            out.append(p)
    return out


def boolean_cut(ob, cutter):
    md = ob.modifiers.new("cut", "BOOLEAN")
    md.operation = "DIFFERENCE"
    md.solver = "EXACT"
    md.object = cutter
    with bpy.context.temp_override(object=ob, active_object=ob):
        bpy.ops.object.modifier_apply(modifier=md.name)
    bpy.data.objects.remove(cutter)


def jag(n_pts, u0, u1, z_fn, rnd=0.8, seed=0):
    """jagged ruin top line: list of (u, z) from u1 down to u0 (polygon goes bottom left->right, then top right->left)."""
    rr = random.Random(seed)
    us = np.linspace(u1, u0, n_pts)
    pts = []
    for i, u in enumerate(us):
        z = z_fn(u) + rr.uniform(-rnd, rnd)
        pts.append((float(u), z))
        if 0 < i < n_pts - 1 and rr.random() < 0.35:      # broken step
            pts.append((float(u) - (u1 - u0) / n_pts * 0.4, z - rr.uniform(0.4, 1.6) * rnd))
    return pts


def wall_obj(name, axis, u0, u1, a0, a1, top, mats, c, zb=F - 0.5):
    pts = [(u0, zb), (u1, zb)] + top
    return extrusion(name, pts, axis, a0, a1, mats, c)


def look_at(ob, target, roll=0.0):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    if roll:
        ob.rotation_euler.rotate_axis("Z", roll)


# ---------------------------------------------------------------------------------------------------------
# 1. platform, floor, stairs
# ---------------------------------------------------------------------------------------------------------
PX0, PX1, PY0, PY1 = -IX - WT, IX + WT, IY0 - WT, IY1 + WT
bm = bmesh.new()
fs = add_box(bm, (PX0, PY0, -0.5), (PX1, PY1, F - 0.25), 0)
add_box(bm, (PX0 - 0.35, PY0 - 0.35, F - 1.0), (PX1 + 0.35, PY1 + 0.35, F - 0.25), 0)     # string course
platform = bm_obj("Platform", bm, [M_PLATFORM], C_ARENA)
floor_ob = boxes("ArenaFloor", [((PX0, PY0, F - 0.25), (PX1, PY1, F), 0)], [M_FLOOR], C_ARENA)

prof = [(ST_Y0, 0.0)]
y, z = ST_Y0, 0.0
for k in range(NR):                       # flight 1
    z = (k + 1) * RISE
    prof.append((y, z))
    y = y + (TREAD if k < NR - 1 else 4.0)
    prof.append((y, z))
LAND_Y0, LAND_Y1 = y - 4.0, y
for k in range(NR):                       # flight 2
    z = 4.0 + (k + 1) * RISE
    prof.append((y, z))
    y = y + (TREAD if k < NR - 1 else 0.72)
    prof.append((y, z))
assert abs(y - ST_Y_TOP) < 1e-6 and abs(z - F) < 1e-6, (y, z)
prof.append((ST_Y_TOP + 0.5, F))
prof.append((ST_Y_TOP + 0.5, 0.0))
stairs = extrusion("GrandStairs", prof, "X", -ST_W, ST_W, [M_STAIRS], C_ARENA)

cheek_top = [(ST_Y_TOP, F + 1.05), (LAND_Y1, 4.0 + 1.05 + RISE), (LAND_Y0, 4.0 + 1.05), (ST_Y0, 1.05 + RISE)]
for s, nm in ((1, "E"), (-1, "W")):
    pts = [(ST_Y0 - 0.6, 0.0), (ST_Y_TOP, 0.0)] + cheek_top + [(ST_Y0 - 0.6, 1.05 + RISE)]
    a0, a1 = sorted((s * ST_W, s * (ST_W + 1.6)))
    extrusion(f"StairCheek_{nm}", pts, "X", a0, a1, [M_STAIRS], C_ARENA)
# brazier pedestals at the bottom, the landing and the top
PEDS = []
for s in (1, -1):
    for (yc, zt) in ((ST_Y0 - 1.0, 1.6), (0.5 * (LAND_Y0 + LAND_Y1), 4.0 + 1.6)):
        PEDS.append((Vector((s * (ST_W + 0.8), yc, zt))))
boxes("StairPedestals", [((p.x - 1.0, p.y - 1.0, 0.0 if p.z < 3 else 3.5), (p.x + 1.0, p.y + 1.0, p.z), 0) for p in PEDS]
      + [((p.x - 1.15, p.y - 1.15, p.z - 0.15), (p.x + 1.15, p.y + 1.15, p.z), 0) for p in PEDS], [M_STAIRS], C_ARENA)

# ---------------------------------------------------------------------------------------------------------
# 2. walls (ruined gothic hall)
# ---------------------------------------------------------------------------------------------------------
def ew_top(seed):
    def zf(yy):
        if yy > 22:
            return F + 17.0
        return F + 7.5 + (yy - PY0) / (22 - PY0) * 8.5
    pts = jag(26, PY0, PY1, zf, rnd=1.1, seed=seed)
    # keep the north part crisp (intact) — flatten jag where intact
    return [(u, F + 17.0 if u > 21.0 else z) for (u, z) in pts]


WIN_Y = [12.5, 25.0, 37.5]
for s, nm in ((1, "E"), (-1, "W")):
    a0, a1 = sorted((s * IX, s * (IX + WT)))
    w = wall_obj(f"Wall_{nm}", "X", PY0, PY1, a0, a1, ew_top(7 if s > 0 else 8), [M_WALL], C_ARENA)
    bm = bmesh.new()
    for yy in WIN_Y:
        add_extrusion(bm, pointed_window(yy, 4.2, F + 4.5, F + 10.5), "X", s * (IX - 1), s * (IX + WT + 1))
    cut = bm_obj("cut", bm, [], C_TMP)
    boolean_cut(w, cut)
    # buttresses down to the courtyard (between the windows, on the pier lines)
    bspec = []
    for yy in COL_Y:
        x0, x1 = sorted((s * (IX + WT), s * (IX + WT + 3.0)))
        x2, x3 = sorted((s * (IX + WT), s * (IX + WT + 1.8)))
        top1 = F + (12.0 if yy > 20 else 7.0)
        bspec.append(((x0, yy - 1.1, -0.3), (x1, yy + 1.1, top1), 0))
        if yy > 20:
            bspec.append(((x2, yy - 0.9, top1), (x3, yy + 0.9, F + 16.0), 0))
    boxes(f"Buttresses_{nm}", bspec, [M_WALL], C_ARENA)
    bm = bmesh.new()       # gothic pinnacles on the intact buttresses
    for yy in COL_Y:
        if yy < 20:
            continue
        cx_ = s * (IX + WT + 0.9)
        add_box(bm, (cx_ - 0.6, yy - 0.6, F + 16.0), (cx_ + 0.6, yy + 0.6, F + 18.2))
        g = bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.85, radius2=0.0, depth=3.2)
        bmesh.ops.rotate(bm, verts=g["verts"], cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 4, 3, "Z"))
        bmesh.ops.translate(bm, verts=g["verts"], vec=(cx_, yy, F + 18.2 + 1.6))
    bm_obj(f"Pinnacles_{nm}", bm, [M_WALL], C_ARENA)

# north gable wall with the rose window opening
ROSE_Z, ROSE_R = F + 24.5, 6.1
RIDGE_Z_GABLE = ZS_VAULT + (COL_X + 1.2) * math.tan(math.radians(56)) + 1.0   # gable follows the roof pitch
nw = extrusion("Wall_N_Gable", [(PX0, F - 0.5), (PX1, F - 0.5), (PX1, F + 17.0), (0.0, RIDGE_Z_GABLE), (PX0, F + 17.0)],
               "Y", IY1, PY1, [M_WALL], C_ARENA)
bm = bmesh.new()
circ = [(ROSE_R * math.cos(2 * math.pi * i / 48), ROSE_Z + ROSE_R * math.sin(2 * math.pi * i / 48)) for i in range(48)]
add_extrusion(bm, circ, "Y", IY1 - 1, PY1 + 1)
for xx in (-12.0, 12.0):     # two lancets beside the rose
    add_extrusion(bm, pointed_window(xx, 3.0, F + 6.0, F + 13.0), "Y", IY1 - 1, PY1 + 1)
boolean_cut(nw, bm_obj("cut", bm, [], C_TMP))

# south wall: ruined, with the 16 m entrance where the stairs arrive
for s, nm in ((1, "E"), (-1, "W")):
    u0, u1 = sorted((s * 8.0, s * (IX + WT)))
    def zf(xx, s=s):
        t = (abs(xx) - 8.0) / (IX + WT - 8.0)
        return F + 4.0 + 8.0 * t
    top = jag(9, u0, u1, zf, rnd=1.4, seed=21 if s > 0 else 22)
    wall_obj(f"Wall_S_{nm}", "Y", u0, u1, PY0, IY0, top, [M_WALL], C_ARENA)
boxes("GatePiers", [((s * 8.0 - 1.2, PY0 - 0.6, F - 0.5), (s * 8.0 + 1.2, IY0 + 0.6, F + h), 0)
                    for s, h in ((1, 13.5), (-1, 11.0))], [M_WALL], C_ARENA)

# ---------------------------------------------------------------------------------------------------------
# 3. fracture helper (cell_fracture add-on, seeded, seeds INSIDE the solid)
# ---------------------------------------------------------------------------------------------------------
def bm_stats(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    v = abs(bm.calc_volume())
    bm.free()
    return nm, v


def repair(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.holes_fill(bm, edges=bm.edges, sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()


def fracture(src, ncell, seed, label, island=False):
    """Seeded Voronoi pre-fracture. Cells come from the cell_fracture add-on's own Voronoi calc
    (fracture_cell_calc.points_as_bmesh_cells); the INTERSECT is done here with the MANIFOLD solver and a per-cell
    fallback (the add-on's built-in EXACT boolean silently returned empty cells on these meshes: 26/44 vault cells,
    0/7 rib cells, 59% arch volume). Returns hidden cell objects (origins at centroids, src's world space)."""
    from bl_ext.blender_org.cell_fracture import fracture_cell_calc
    work = src.copy()
    work.data = src.data.copy()
    C_TMP.objects.link(work)
    work.data.transform(src.matrix_world)
    work.matrix_world = Matrix()
    if M_INTERIOR.name not in [m.name for m in work.data.materials if m]:
        work.data.materials.append(M_INTERIOR)
    imat = len(work.data.materials) - 1
    nm, vol = bm_stats(work.data)
    random.seed(seed)
    mathutils.noise.seed_set(seed)
    vco = [v.co.copy() for v in work.data.vertices]
    lo = Vector([min(v[i] for v in vco) for i in range(3)])
    hi = Vector([max(v[i] for v in vco) for i in range(3)])
    pts, tries = [], 0
    while len(pts) < ncell and tries < ncell * 800:
        tries += 1
        q = Vector([random.uniform(lo[i], hi[i]) for i in range(3)])
        ok, loc, nrm, _ = work.closest_point_on_mesh(q)
        if ok and (loc - q).dot(nrm) > 0 and (loc - q).length > 0.08:
            pts.append(q)
    cells_calc = fracture_cell_calc.points_as_bmesh_cells(vco, pts, None, margin_cell=0.001)
    uvn = [l.name for l in work.data.uv_layers]
    cells, used = [], {}
    for k, (cen, cpts) in enumerate(cells_calc):
        bm = bmesh.new()
        for co in cpts:
            bm.verts.new(co)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.002)
        res = bmesh.ops.convex_hull(bm, input=bm.verts)
        bmesh.ops.delete(bm, geom=[v for v in res.get("geom_interior", []) if isinstance(v, bmesh.types.BMVert)], context="VERTS")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
        for f in bm.faces:
            f.material_index = imat
        me = bpy.data.meshes.new(f"{label}_cell")
        bm.to_mesh(me)
        bm.free()
        for m in work.data.materials:
            me.materials.append(m)
        for n in uvn:
            me.uv_layers.new(name=n)
        ob = bpy.data.objects.new(f"{label}_cell{k:02d}", me)
        C_TMP.objects.link(ob)
        ob.location = cen
        result = None
        for solver in ("MANIFOLD", "EXACT", "FLOAT"):
            md = ob.modifiers.new("isect", "BOOLEAN")
            md.operation = "INTERSECT"
            md.solver = solver
            md.object = work
            dg = bpy.context.evaluated_depsgraph_get()
            mnew = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), depsgraph=dg)
            ob.modifiers.remove(md)
            # expected volume of this cell ∩ solid is unknown, but a valid result is closed and non-empty
            n_, v_ = bm_stats(mnew) if len(mnew.vertices) else (1, 0.0)
            if len(mnew.vertices) and n_ == 0 and v_ > 1e-4:
                result = (mnew, solver, v_)
                break
            bpy.data.meshes.remove(mnew)
        old = ob.data
        if result is None:
            bpy.data.objects.remove(ob)
            bpy.data.meshes.remove(old)
            continue
        ob.data = result[0]
        bpy.data.meshes.remove(old)
        used[result[1]] = used.get(result[1], 0) + 1
        # recenter origin on the centroid
        c = sum((v.co for v in ob.data.vertices), Vector()) / len(ob.data.vertices)
        ob.data.transform(Matrix.Translation(-c))
        ob.location = ob.location + c
        cells.append(ob)
    cvol = sum(bm_stats(c.data)[1] for c in cells)
    REPORT["fracture"][label] = dict(nonmanifold_src=nm, seeds=len(pts), cells=len(cells), solvers=used,
                                     vol=round(vol, 2), vol_ratio=round(cvol / vol, 3) if vol else None)
    print("[fracture]", label, REPORT["fracture"][label], flush=True)
    bpy.data.objects.remove(work)
    for c in cells:
        c.hide_render = True
        c.hide_viewport = True
        c["breakable_cell"] = True
    return cells, None


# ---------------------------------------------------------------------------------------------------------
# 4. gothic piers + arcade arches (Sketchfab "Gothic Architectural Structures", CC-BY) — pre-fractured
# ---------------------------------------------------------------------------------------------------------
def import_glb(path, c):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        if o.type == "MESH" and o.name.startswith("Icosphere") and len(o.data.polygons) <= 80:
            bpy.data.objects.remove(o)
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        link(o, c)
    return new


def bake_world_mesh(o, name, M=None):
    """copy o's mesh with its world transform (and optional extra matrix) applied; object at identity."""
    me = o.data.copy()
    me.name = name
    me.transform((M or Matrix()) @ o.matrix_world)
    ob = bpy.data.objects.new(name, me)
    C_TMP.objects.link(ob)
    return ob


kit = import_glb(SF + "env_gothic_arch_structures/model.glb", C_TMP)
bpy.context.view_layer.update()
col_src = next(o for o in kit if o.type == "MESH" and o.name == "Object_7")
arch_src = next(o for o in kit if o.type == "MESH" and o.name == "Object_8")
Mmod = Matrix.Scale(COL_S, 4) @ Matrix.Translation((0, -0.291, -0.063))
COL_T = bake_world_mesh(col_src, "PierTemplate", Mmod)
ARCH_T = bake_world_mesh(arch_src, "ArchTemplate", Mmod)
for o in kit:
    bpy.data.objects.remove(o)
repair(COL_T.data)
repair(ARCH_T.data)
col_cells, _ = fracture(COL_T, 22, 101, "pier")
arch_cells, _ = fracture(ARCH_T, 10, 202, "arch")
bb = [COL_T.matrix_world @ Vector(c) for c in COL_T.bound_box]
PIER_W = max(v.x for v in bb) - min(v.x for v in bb)
PIER_H = max(v.z for v in bb)
ab = [ARCH_T.matrix_world @ Vector(c) for c in ARCH_T.bound_box]
ARCH_TOP = max(v.z for v in ab)
print(f"[piers] width {PIER_W:.2f} m, height {PIER_H:.2f} m, arch top {ARCH_TOP:.2f} m above floor", flush=True)

PIERS = []
for s, side in ((1, "E"), (-1, "W")):
    for i, yy in enumerate(COL_Y):
        nm = f"Pier_{side}{i + 1}"
        root = bpy.data.objects.new(nm, None)
        root.empty_display_type = "PLAIN_AXES"
        C_PILLARS.objects.link(root)
        root.location = (s * COL_X, yy, F)
        rot = (i + (0 if s > 0 else 2)) % 4
        root["breakable_group"] = nm
        sub = bpy.data.collections.new(nm)
        C_PILLARS.children.link(sub)
        for tmpl, cells, part, r in ((COL_T, col_cells, "pier", rot), (ARCH_T, arch_cells, "arch", 2 * (rot % 2))):
            holder = bpy.data.objects.new(f"{nm}_{part}_frame", None)
            sub.objects.link(holder)
            holder.parent = root
            holder.rotation_euler = (0, 0, r * math.pi / 2)
            it = bpy.data.objects.new(f"{nm}_{part}_INTACT", tmpl.data)
            sub.objects.link(it)
            it.parent = holder
            it["breakable_group"] = nm
            it["role"] = "intact"
            for k, c in enumerate(cells):
                co = bpy.data.objects.new(f"{nm}_{part}_cell{k:02d}", c.data)
                sub.objects.link(co)
                co.parent = holder
                co.location = c.location
                co.hide_render = co.hide_viewport = True
                co["breakable_group"] = nm
                co["role"] = "cell"
        PIERS.append(root)
CHUNK_MESHES = [c.data for c in col_cells + arch_cells]
for c in col_cells + arch_cells:
    bpy.data.objects.remove(c)
bpy.data.objects.remove(COL_T)
bpy.data.objects.remove(ARCH_T)

# ---------------------------------------------------------------------------------------------------------
# 5. clerestory, vault (north half intact + pre-fractured), ruined haunches (south), aisle roofs
# ---------------------------------------------------------------------------------------------------------
CL_X0, CL_X1 = COL_X - 1.2, COL_X + 1.2
for s, nm in ((1, "E"), (-1, "W")):
    def zf(yy):
        return ZS_VAULT if yy > 3.0 else F + 21.6
    top = [(u, ZS_VAULT if u > 3.5 else z) for (u, z) in jag(20, IY0, IY1, zf, rnd=0.6, seed=31 if s > 0 else 32)]
    a0, a1 = sorted((s * CL_X0, s * CL_X1))
    wall_obj(f"Clerestory_{nm}", "X", IY0, IY1, a0, a1, top, [M_WALL], C_ARENA, zb=F + 20.8)

# pointed (drop) vault: two arcs, centres at +-cx on the springing line
R_IN = (VAULT_H ** 2 + COL_X ** 2) / (2 * COL_X)   # each half-arc centred on the OPPOSITE side of the axis
assert R_IN > COL_X, "pointed vault needs rise > half-span"


def vault_profile(r_in, r_out, a_max_frac=1.0, n=22):
    """pointed-vault band (x,z rel. springing). Returns (prof, band): prof = outer polyline (right spring -> apex ->
    left spring) + inner polyline reversed. Half-arcs are centred on the opposite side (c = COL_X - R_IN < 0).
    a_max_frac < 1: only the right haunch up to that fraction of the arc."""
    c = COL_X - R_IN

    def half(r, side, frac):
        a_end = math.atan2(math.sqrt(max(r * r - c * c, 0.0)), -c)    # angle where x = 0
        return [(side * (c + r * math.cos(a_end * frac * i / n)), r * math.sin(a_end * frac * i / n)) for i in range(n + 1)]
    if a_max_frac >= 1.0:
        outer = half(r_out, 1, 1.0) + half(r_out, -1, 1.0)[::-1][1:]
        inner = half(r_in, 1, 1.0) + half(r_in, -1, 1.0)[::-1][1:]
    else:
        outer, inner = half(r_out, 1, a_max_frac), half(r_in, 1, a_max_frac)
    return outer + inner[::-1], len(outer)


_vp, VBAND = vault_profile(R_IN, R_IN + VAULT_T)
VPROF = [(x, z + ZS_VAULT) for (x, z) in _vp]
VAULT_APEX = max(z for _, z in VPROF)


def vault_start(x, z):
    return 25.0 + 1.8 * math.sin(0.55 * x + 1.3) + 1.6 * mathutils.noise.noise(Vector((x * 0.35, z * 0.35, 0.7)))


vault = sweep_y("Vault_INTACT", VPROF, vault_start, lambda x, z: IY1 + 0.6, 12, [M_VAULT], C_ROOF, band=VBAND)
vault["breakable_group"] = "Vault"
vault["role"] = "intact"
vcells, _ = fracture(vault, 44, 303, "vault")
CHUNK_MESHES += [c.data for c in vcells]
for k, c in enumerate(vcells):
    c.name = f"Vault_cell{k:02d}"
    link(c, C_ROOF)
    c["breakable_group"] = "Vault"
    c["role"] = "cell"
# transverse ribs under the intact vault
_rp, RBAND = vault_profile(R_IN - 0.75, R_IN + 0.05)
RIB_PROF = [(x, z + ZS_VAULT) for (x, z) in _rp]
for k, yy in enumerate((COL_Y[2], 37.5, COL_Y[3])):
    rib = sweep_y(f"VaultRib{k}_INTACT", RIB_PROF, lambda x, z, yy=yy: yy - 0.55, lambda x, z, yy=yy: yy + 0.55, 1,
                  [M_VAULT], C_ROOF, band=RBAND)
    rib["breakable_group"] = f"VaultRib{k}"
    rib["role"] = "intact"
    rcells, _ = fracture(rib, 7, 400 + k, f"rib{k}")
    for j, c in enumerate(rcells):
        c.name = f"VaultRib{k}_cell{j:02d}"
        link(c, C_ROOF)
        c["breakable_group"] = f"VaultRib{k}"
        c["role"] = "cell"
# south ruin: vault haunches survive low on both sides, broken ends
for s, nm in ((1, "E"), (-1, "W")):
    _hp, HB = vault_profile(R_IN, R_IN + VAULT_T, 0.33, 8)
    hp = [(s * x, z + ZS_VAULT) for (x, z) in _hp]
    sweep_y(f"VaultHaunch_{nm}", hp, lambda x, z: 6.0 + 1.5 * mathutils.noise.noise(Vector((x * 0.3, z * 0.4, 3.0))),
            lambda x, z: 25.5 + 1.5 * mathutils.noise.noise(Vector((x * 0.4, z * 0.3, 9.0))), 6, [M_VAULT], C_RUIN, band=HB)
# aisle lean-to roofs (north part), slate on stone
for s, nm in ((1, "E"), (-1, "W")):
    x_out, x_in = s * (IX + WT), s * CL_X1
    pr = [(x_out, F + 16.8), (x_in, ZS_VAULT - 0.6), (x_in, ZS_VAULT - 0.1), (x_out, F + 17.3)]
    sweep_y(f"AisleRoof_{nm}", pr, lambda x, z: 24.0 + 1.0 * mathutils.noise.noise(Vector((x * 0.5, z, 5.0))),
            lambda x, z: IY1 + 0.3, 6, [M_ROOFSLATE], C_RUIN)

# charred timber roof frame over the vault (the lead/slate roof burned; trusses survive) + one slate section at the
# north end — both break with the vault. Every truss is its own object (no fracture needed); the slates are pre-fractured.
M_TIMBER = pbr("M_CharredTimber", "medieval_wood", tile=2.0, tint=(0.18, 0.15, 0.13), grime=0.5)
PITCH = math.radians(56)
RIDGE_Z = ZS_VAULT + CL_X1 * math.tan(PITCH)
rt = random.Random(5)
n_tr = 0
yy = 26.0
while yy < IY1 + 1.5:
    if rt.random() < 0.18 and yy < 40:            # a few trusses gone
        yy += 1.6
        continue
    bm = bmesh.new()
    lean = rt.uniform(-0.06, 0.06) if yy < 36 else 0.0
    for side in (1, -1):                          # rafters
        x0, z0 = side * CL_X1, ZS_VAULT + 0.1
        L_ = math.hypot(CL_X1, RIDGE_Z - z0)
        g = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(0.35, 0.3, L_), verts=g["verts"])
        ang = math.atan2(-side * CL_X1, RIDGE_Z - z0)
        bmesh.ops.rotate(bm, verts=g["verts"], cent=(0, 0, 0), matrix=Matrix.Rotation(ang, 3, "Y"))
        bmesh.ops.translate(bm, verts=g["verts"], vec=(x0 / 2, 0, (z0 + RIDGE_Z) / 2))
    g = bmesh.ops.create_cube(bm, size=1.0)       # collar beam
    zc = ZS_VAULT + 0.62 * (RIDGE_Z - ZS_VAULT)
    bmesh.ops.scale(bm, vec=(2 * CL_X1 * 0.38 * 1.02, 0.28, 0.3), verts=g["verts"])
    bmesh.ops.translate(bm, verts=g["verts"], vec=(0, 0, zc))
    for v in bm.verts:
        v.co.y += yy
        v.co.y += lean * (v.co.z - ZS_VAULT)
    tr = bm_obj(f"RoofTruss{n_tr:02d}", bm, [M_TIMBER], C_ROOF)
    tr["breakable_group"] = "RoofTimber"
    tr["role"] = "piece"
    n_tr += 1
    yy += 1.6
bm = bmesh.new()                                    # ridge + purlins
for (x_, z_) in [(0.0, RIDGE_Z)] + [(s_ * CL_X1 * f_, ZS_VAULT + (1 - f_) * (RIDGE_Z - ZS_VAULT)) for s_ in (1, -1) for f_ in (0.35, 0.7)]:
    add_box(bm, (x_ - 0.17, 33.0 if x_ == 0 else 30.0 + 4 * abs(x_) / CL_X1, z_ - 0.15), (x_ + 0.17, IY1 + 1.5, z_ + 0.2))
pur = bm_obj("RoofPurlins", bm, [M_TIMBER], C_ROOF)
pur["breakable_group"] = "RoofTimber"
pur["role"] = "piece"
for side, nm in ((1, "E"), (-1, "W")):
    ang = math.atan2(RIDGE_Z - ZS_VAULT, CL_X1)
    nrm = Vector((side * math.sin(ang), 0, math.cos(ang)))
    pr = [(side * (CL_X1 + 0.5), ZS_VAULT - 0.2 + 0.4), (0.0, RIDGE_Z + 0.4)]
    p0, p1 = Vector((pr[0][0], 0, pr[0][1])), Vector((pr[1][0], 0, pr[1][1]))
    q0, q1 = p0 + nrm * 0.35, p1 + nrm * 0.35
    prof = [(p0.x, p0.z), (p1.x, p1.z), (q1.x, q1.z), (q0.x, q0.z)]
    if side < 0:
        prof = prof[::-1]
    sl = sweep_y(f"RoofSlates_{nm}_INTACT", prof,
                 lambda x, z: 41.0 + 2.5 * mathutils.noise.noise(Vector((x * 0.4, z * 0.4, 1.7))) + 0.15 * (z - ZS_VAULT),
                 lambda x, z: IY1 + 1.6, 4, [M_ROOFSLATE], C_ROOF)
    sl["breakable_group"] = f"RoofSlates_{nm}"
    sl["role"] = "intact"
    scells, _ = fracture(sl, 9, 600 + (side > 0), f"slates_{nm}")
    for j, c in enumerate(scells):
        c.name = f"RoofSlates_{nm}_cell{j:02d}"
        link(c, C_ROOF)
        c["breakable_group"] = f"RoofSlates_{nm}"
        c["role"] = "cell"
REPORT["roof_trusses"] = n_tr

# ---------------------------------------------------------------------------------------------------------
# 6. rose window (Sketchfab "Rosetón 2021", CC-BY) — stone tracery + emissive stained glass behind
# ---------------------------------------------------------------------------------------------------------
rose = import_glb(SF + "env_rose_window/model.glb", C_ARENA)
rose_root = bpy.data.objects.new("RoseWindow", None)
C_ARENA.objects.link(rose_root)
for o in rose:
    if o.parent is None:
        o.parent = rose_root
s_rose = (2 * ROSE_R + 0.3) / 13.2
rose_root.scale = (s_rose,) * 3
rose_root.rotation_euler = (0, 0, math.radians(90))
rose_root.location = (0, IY1 + 1.3, ROSE_Z - 6.6 * s_rose)
M_ROSE_STONE = M_VAULT
for o in rose:
    if o.type == "MESH":
        o.data.materials.clear()
        o.data.materials.append(M_GLASS if o.name.startswith("CRISTAL") else M_ROSE_STONE)
glass_back = extrusion("RoseGlassBack", circ, "Y", IY1 + 2.0, IY1 + 2.05, [M_GLASS], C_ARENA)
for xx in (-12.0, 12.0):
    extrusion(f"LancetGlass_{'E' if xx > 0 else 'W'}", pointed_window(xx, 3.0, F + 6.0, F + 13.0), "Y", IY1 + 1.9,
              IY1 + 1.95, [M_GLASS], C_ARENA)

# ---------------------------------------------------------------------------------------------------------
# 7. dais + hoard + PRIZE placeholder (gold rocket launcher; swap the PRIZE_placeholder collection at PRIZE_anchor)
# ---------------------------------------------------------------------------------------------------------
dsp = []
for k in range(3):
    dsp.append(((-9.0 + k * 0.6, DAIS_Y0 + k * 0.8, F - 0.1), (9.0 - k * 0.6, IY1, F + 0.3 * (k + 1)), 0))
PLINTH = (Vector((-0.75, 47.7, F + 0.9)), Vector((0.75, 49.1, F + 2.2)))
dsp.append((tuple(PLINTH[0]), tuple(PLINTH[1]), 0))
dsp.append(((PLINTH[0].x - 0.12, PLINTH[0].y - 0.12, PLINTH[1].z - 0.12), (PLINTH[1].x + 0.12, PLINTH[1].y + 0.12, PLINTH[1].z), 0))
boxes("Dais", dsp, [M_STAIRS], C_HOARD)


def dais_h(x, y):
    h = 0.0
    for k in range(3):
        if abs(x) <= 9.0 - k * 0.6 and y >= DAIS_Y0 + k * 0.8:
            h = 0.3 * (k + 1)
    return h


MOUNDS = [(-4.6, 48.6, 1.45, 2.3), (4.9, 48.5, 1.55, 2.4), (-2.1, 49.4, 0.9, 1.2), (2.2, 49.4, 0.95, 1.2),
          (0.6, 46.0, 0.45, 1.5), (-6.8, 46.4, 0.55, 1.4), (7.2, 46.2, 0.5, 1.3), (-1.8, 44.6, 0.18, 1.0)]
gx = np.arange(-9.6, 9.61, 0.15)
gy = np.arange(DAIS_Y0 - 2.4, IY1 - 0.05, 0.15)
HV, PV = [], []
for y_ in gy:
    for x_ in gx:
        m_ = sum(h * math.exp(-((x_ - mx) ** 2 + (y_ - my) ** 2) / (2 * sg * sg)) for mx, my, h, sg in MOUNDS)
        m_ += 0.06 * mathutils.noise.noise(Vector((x_ * 2.5, y_ * 2.5, 0.3))) * min(1.0, m_ * 4)
        HV.append((x_, y_, F + dais_h(x_, y_) + m_ - 0.04))
        PV.append(m_)
nx_, ny_ = len(gx), len(gy)
hf = [(j * nx_ + i, j * nx_ + i + 1, (j + 1) * nx_ + i + 1, (j + 1) * nx_ + i) for j in range(ny_ - 1) for i in range(nx_ - 1)]
hme = bpy.data.meshes.new("GoldHeap")
hme.from_pydata(HV, [], hf)
pa = hme.attributes.new("pile", "FLOAT", "POINT")
pa.data.foreach_set("value", PV)
hme.materials.append(M_COINS)
hme.shade_smooth()
heap = bpy.data.objects.new("GoldHeap", hme)
C_HOARD.objects.link(heap)
# coins: GN instances on the heap surface where the pile is deep enough (Set Material on the template, not after)
ng = bpy.data.node_groups.new("GN_Coins", "GeometryNodeTree")
ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
Nn, Ln = ng.nodes, ng.links
gi_, go_ = Nn.new("NodeGroupInput"), Nn.new("NodeGroupOutput")
na = Nn.new("GeometryNodeInputNamedAttribute")
na.data_type = "FLOAT"
na.inputs["Name"].default_value = "pile"
mrc = Nn.new("ShaderNodeMapRange")
mrc.inputs["From Min"].default_value, mrc.inputs["From Max"].default_value = 0.03, 0.25
Ln.new(na.outputs["Attribute"], mrc.inputs["Value"])
dp = Nn.new("GeometryNodeDistributePointsOnFaces")
dp.distribute_method = "RANDOM"
Ln.new(gi_.outputs[0], dp.inputs["Mesh"])
mrc.inputs["To Max"].default_value = 260.0        # coins per m^2 (Density is a field in RANDOM mode)
Ln.new(mrc.outputs["Result"], dp.inputs["Density"])
dp.inputs["Seed"].default_value = 7
cy_ = Nn.new("GeometryNodeMeshCylinder")
cy_.inputs["Vertices"].default_value = 14
cy_.inputs["Radius"].default_value = 0.032
cy_.inputs["Depth"].default_value = 0.006
sm = Nn.new("GeometryNodeSetMaterial")
sm.inputs["Material"].default_value = M_GOLD
Ln.new(cy_.outputs["Mesh"], sm.inputs["Geometry"])
iop = Nn.new("GeometryNodeInstanceOnPoints")
Ln.new(dp.outputs["Points"], iop.inputs["Points"])
Ln.new(sm.outputs["Geometry"], iop.inputs["Instance"])
Ln.new(dp.outputs["Rotation"], iop.inputs["Rotation"])
jg = Nn.new("GeometryNodeJoinGeometry")
Ln.new(gi_.outputs[0], jg.inputs[0])
Ln.new(iop.outputs["Instances"], jg.inputs[0])
Ln.new(jg.outputs[0], go_.inputs[0])
heap.modifiers.new("Coins", "NODES").node_group = ng
# prize: placeholder launcher (tube, flared breech, warhead, grip, sight) on a stand on the plinth
PRIZE_POS = Vector((0, (PLINTH[0].y + PLINTH[1].y) / 2, PLINTH[1].z))
anchor = bpy.data.objects.new("PRIZE_anchor", None)
anchor.empty_display_type = "ARROWS"
C_PRIZE.objects.link(anchor)
anchor.location = PRIZE_POS + Vector((0, 0, 0.72))
anchor.rotation_euler = (math.radians(14), 0, math.radians(90))   # launcher along X, nose up 14 deg
anchor.scale = (1.35,) * 3                                          # ~2.2 m: a trophy-sized placeholder
bm = bmesh.new()


def cyl(bm, r1, r2, z0, z1, segs=32):
    g = bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r1, radius2=r2, depth=z1 - z0)
    bmesh.ops.translate(bm, verts=g["verts"], vec=(0, 0, (z0 + z1) / 2))


cyl(bm, 0.055, 0.055, -0.55, 0.45)          # tube (along local Z, nose at +Z)
cyl(bm, 0.11, 0.06, -0.72, -0.55)           # flared breech
cyl(bm, 0.06, 0.075, 0.45, 0.52)            # muzzle ring
cyl(bm, 0.075, 0.075, 0.52, 0.66)           # warhead body
cyl(bm, 0.075, 0.0, 0.66, 0.95)             # warhead nose
cyl(bm, 0.03, 0.03, 0.40, 0.55)             # warhead tail
g = bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.scale(bm, vec=(0.04, 0.05, 0.16), verts=g["verts"])
bmesh.ops.rotate(bm, verts=g["verts"], cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(15), 3, "Y"))
bmesh.ops.translate(bm, verts=g["verts"], vec=(0.11, 0, -0.08))           # pistol grip
g = bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.scale(bm, vec=(0.06, 0.03, 0.12), verts=g["verts"])
bmesh.ops.translate(bm, verts=g["verts"], vec=(-0.09, 0, 0.12))           # optic
prize = bm_obj("PRIZE_GoldRocketLauncher_PLACEHOLDER", bm, [M_GOLD], C_PRIZE, smooth=True)
prize.parent = anchor
prize.rotation_euler = (math.radians(90), 0, 0)
prize["placeholder"] = "swap for the sourced model; keep PRIZE_anchor transform"
py_ = PRIZE_POS.y
pz_ = PRIZE_POS.z
stand = boxes("PrizeStand", [((-0.42, py_ - 0.05, pz_), (-0.34, py_ + 0.05, pz_ + 0.55), 0),
                            ((0.34, py_ - 0.05, pz_), (0.42, py_ + 0.05, pz_ + 0.78), 0)], [M_IRON], C_PRIZE)
REPORT["prize"] = dict(anchor=list(anchor.location), note="placeholder gold launcher, ~2.2 m long incl. anchor scale 1.35")

# ---------------------------------------------------------------------------------------------------------
# 8. courtyard, curtain walls, towers, terrain
# ---------------------------------------------------------------------------------------------------------
CY0, CY1, CX = -78.0, 86.0, 50.0
boxes("CourtyardPaving", [((-CX, CY0, -0.3), (CX, CY1, 0.0), 0)], [M_COURT], C_COURT)
# terrain: 4 km, flat near the castle, low hills further out
N_T = 160
xs = np.linspace(-2000, 2000, N_T)
X, Y = np.meshgrid(xs, xs)
D = np.sqrt(X ** 2 + (Y - 4) ** 2)
H = np.zeros_like(X)
for i in range(N_T):
    for j in range(N_T):
        H[i, j] = mathutils.noise.fractal(Vector((X[i, j] / 420, Y[i, j] / 420, 0.3)), 0.6, 2.0, 4)
H = (H * 80.0 + 120.0) * np.clip((D - 230) / 900, 0, 1) ** 1.2 - 0.35
verts = np.stack([X.ravel(), Y.ravel(), H.ravel()], 1)
faces = [(i * N_T + j, i * N_T + j + 1, (i + 1) * N_T + j + 1, (i + 1) * N_T + j) for i in range(N_T - 1) for j in range(N_T - 1)]
tme = bpy.data.meshes.new("Terrain")
tme.from_pydata(verts.tolist(), [], faces)
tme.materials.append(M_GROUND)
tme.shade_smooth()
terrain = bpy.data.objects.new("Terrain", tme)
C_COURT.objects.link(terrain)


def crenel_top(u0, u1, h, step=2.4):
    pts, u = [], u1
    while u - step > u0:
        pts += [(u, h + 1.7), (u - step * 0.55, h + 1.7), (u - step * 0.55, h), (u - step, h)]
        u -= step
    pts += [(u0, h)]
    return pts


CW_T, CW_H = 3.0, 12.0
for nm, axis, u0, u1, a0, a1 in (("S", "Y", -CX, CX, CY0 - CW_T, CY0), ("N", "Y", -CX, CX, CY1, CY1 + CW_T),
                                 ("E", "X", CY0 - CW_T, CY1 + CW_T, CX, CX + CW_T),
                                 ("W", "X", CY0 - CW_T, CY1 + CW_T, -CX - CW_T, -CX)):
    w = wall_obj(f"CurtainWall_{nm}", axis, u0, u1, a0, a1, crenel_top(u0, u1, CW_H), [M_CURTAIN], C_COURT, zb=-0.5)
    if nm == "S":
        bm = bmesh.new()
        add_extrusion(bm, pointed_window(0.0, 7.0, -1.0, 5.0), "Y", a0 - 1, a1 + 1)
        boolean_cut(w, bm_obj("cut", bm, [], C_TMP))
# walkway behind the merlons
boxes("WallWalks", [((-CX, CY0, CW_H - 0.4), (CX, CY0 + 2.0, CW_H), 0), ((-CX, CY1 - 2.0, CW_H - 0.4), (CX, CY1, CW_H), 0),
                    ((CX - 2.0, CY0, CW_H - 0.4), (CX, CY1, CW_H), 0), ((-CX, CY0, CW_H - 0.4), (-CX + 2.0, CY1, CW_H), 0)],
      [M_CURTAIN], C_COURT)
towers = import_glb(SF + "env_ruined_towers_pack/model.glb", C_TMP)
bpy.context.view_layer.update()
TW = {}
for o in towers:
    if o.type != "MESH":
        continue
    mat = o.data.materials[0].name if o.data.materials else o.name
    t = bake_world_mesh(o, "Tower_" + mat, None)
    co = [v.co for v in t.data.vertices]
    lo = Vector([min(c[i] for c in co) for i in range(3)])
    hi = Vector([max(c[i] for c in co) for i in range(3)])
    t.data.transform(Matrix.Scale(0.3, 4) @ Matrix.Translation((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z)))
    TW[mat] = t
for o in towers:
    if o.name in bpy.data.objects:
        bpy.data.objects.remove(o)
for (tx, ty, key, rz) in ((CX + 1.5, CY0 - 1.5, "tower_intact2", 0), (-CX - 1.5, CY0 - 1.5, "tower_intact2", 90),
                         (CX + 1.5, CY1 + 1.5, "tower_damaged2", 180), (-CX - 1.5, CY1 + 1.5, "tower_ruined3", 270)):
    src = next((v for k, v in TW.items() if key in k), None)
    if src is None:
        continue
    t = bpy.data.objects.new(f"CornerTower_{key}", src.data)
    C_COURT.objects.link(t)
    t.location = (tx, ty, -0.2)
    t.rotation_euler = (0, 0, math.radians(rz))
for t in TW.values():
    bpy.data.objects.remove(t)

# ---------------------------------------------------------------------------------------------------------
# 9. props: braziers (Sketchfab, CC-BY), wall torches (Sketchfab, CC-BY), rubble, chests (Poly Haven CC0)
# ---------------------------------------------------------------------------------------------------------
bz = import_glb(SF + "prop_braziers_burnedhrum/model.glb", C_TMP)
bpy.context.view_layer.update()
parts = [bake_world_mesh(o, "BrazierPart", None) for o in bz if o.type == "MESH"]
for o in bz:
    if o.name in bpy.data.objects:
        bpy.data.objects.remove(o)
# keep the LEFT brazier of the pair (x<0), scale to 1.3 m
for p in parts:
    me = p.data
    b2 = bmesh.new()
    b2.from_mesh(me)
    bmesh.ops.delete(b2, geom=[v for v in b2.verts if v.co.x > 0], context="VERTS")
    b2.to_mesh(me)
    b2.free()
allco = [v.co.copy() for p in parts for v in p.data.vertices]
lo = Vector([min(c[i] for c in allco) for i in range(3)])
hi = Vector([max(c[i] for c in allco) for i in range(3)])
BZ_S = 1.3 / (hi.z - lo.z)
for p in parts:
    p.data.transform(Matrix.Scale(BZ_S, 4) @ Matrix.Translation((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z)))
BZ_TOP = 1.3
REPORT["brazier_scale"] = BZ_S


def flame_obj(name, loc, h, r, c, seed=0):
    """a fire = 5 thin noise-bent tongues + a low glowing core (emissive, alpha fades to the tips)."""
    rr = random.Random(seed)
    bm = bmesh.new()
    for t_ in range(5):
        hh = h * (1.0 if t_ == 0 else rr.uniform(0.45, 0.85))
        rad = r * (0.75 if t_ == 0 else rr.uniform(0.35, 0.6))
        off = Vector((rr.uniform(-1, 1), rr.uniform(-1, 1), 0)) * r * (0 if t_ == 0 else 0.9)
        g = bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=10, radius=1.0)
        ph = rr.uniform(0, 10)
        for v in g["verts"]:
            t = (v.co.z + 1) / 2
            w = (1 - t) ** 0.8 * (0.35 + 0.65 * min(1.0, t * 4))
            bend = Vector((math.sin(ph + t * 3.2), math.cos(ph * 1.3 + t * 2.7), 0)) * rad * 0.9 * t
            v.co = Vector((v.co.x * rad * w, v.co.y * rad * w, t * hh)) + off + bend
    ob = bm_obj(name, bm, [M_FLAME], c, smooth=True)
    ob.location = loc
    ob.visible_shadow = False
    return ob


def fire_light(name, loc, energy, radius=0.25):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = energy
    ld.color = (1.0, 0.52, 0.2)
    ld.shadow_soft_size = radius
    o = bpy.data.objects.new(name, ld)
    C_LIGHTS.objects.link(o)
    o.location = loc
    return o


BRAZIERS = [p + Vector((0, 0, 0)) for p in PEDS]
BRAZIERS += [Vector((s * 9.6, IY0 + 2.5, F)) for s in (1, -1)]
BRAZIERS += [Vector((s * 8.6, BAY, F)) for s in (1, -1)] + [Vector((s * 9.4, 38.5, F)) for s in (1, -1)]
BRAZIERS += [Vector((s * 8.8, DAIS_Y0 - 1.6, F)) for s in (1, -1)]
BRAZIERS += [Vector((-4.2, 21.2, F))]          # lights the knight's face side where he stops before the dragon
for k, p in enumerate(BRAZIERS):
    for j, part in enumerate(parts):
        o = bpy.data.objects.new(f"Brazier{k:02d}_{j}", part.data)
        C_PROPS.objects.link(o)
        o.location = p
        o.rotation_euler = (0, 0, k * 1.1)
    flame_obj(f"Brazier{k:02d}_flame", p + Vector((0, 0, BZ_TOP - 0.18)), 1.05, 0.3, C_LIGHTS, seed=k)
    fire_light(f"Brazier{k:02d}_light", p + Vector((0, 0, BZ_TOP + 0.45)), FIRE_E)
for p in parts:
    bpy.data.objects.remove(p)

tq = import_glb(SF + "prop_wall_torch_kigha/model.glb", C_TMP)
bpy.context.view_layer.update()
torch_src = bake_world_mesh(next(o for o in tq if o.type == "MESH"), "WallTorch", None)
for o in tq:
    if o.name in bpy.data.objects:
        bpy.data.objects.remove(o)
co = [v.co.copy() for v in torch_src.data.vertices]
zmax = max(c.z for c in co)
xmin = min(c.x for c in co)
head = [c for c in co if c.z > zmax - 6.0]
hx = sum(c.x for c in head) / len(head)
TQ_S = 0.75 / (zmax - min(c.z for c in co))
torch_src.data.transform(Matrix.Scale(TQ_S, 4) @ Matrix.Translation((-xmin, 0, 0)))
TORCH_HEAD = Vector(((hx - xmin) * TQ_S, 0, zmax * TQ_S))


def torch_at(name, base, out_dir):
    """mount plate at base on a wall whose outward normal is out_dir (unit xy)."""
    yaw = math.atan2(out_dir.y, out_dir.x)
    o = bpy.data.objects.new(name, torch_src.data)
    C_PROPS.objects.link(o)
    o.location = base
    o.rotation_euler = (0, 0, yaw)
    hw = Matrix.Rotation(yaw, 4, "Z") @ TORCH_HEAD
    flame_obj(name + "_flame", base + hw + Vector((0, 0, -0.08)), 0.5, 0.11, C_LIGHTS, seed=hash(name) % 997)
    fire_light(name + "_light", base + hw + Vector((0, 0, 0.3)), FIRE_E * 0.3, 0.12)


TORCHES = []
for s in (1, -1):
    for yy in (BAY * 0.5, BAY * 1.5, BAY * 2.5, BAY * 3.5):
        TORCHES.append((Vector((s * IX, yy, F + 3.2)), Vector((-s, 0, 0))))
    TORCHES.append((Vector((s * 11.0, PY0, 4.6)), Vector((0, -1, 0))))                   # platform face beside the stairs
    TORCHES.append((Vector((s * 3.6, CY0, 3.2)), Vector((0, 1, 0))))                     # inside the castle gate
    for yy in (-60.0, -35.0, 20.0, 55.0):
        TORCHES.append((Vector((s * CX, yy, 3.6)), Vector((-s, 0, 0))))                  # curtain wall
for k, (b, d) in enumerate(TORCHES):
    torch_at(f"Torch{k:02d}", b, d)
bpy.data.objects.remove(torch_src)

# rubble: photoscan piles + loose fallen blocks (mostly in the ruined south half and under the broken roof edge)
rb = import_glb(SF + "env_rubble_pile_scan/model.glb", C_TMP)
bpy.context.view_layer.update()
rb_src = bake_world_mesh(next(o for o in rb if o.type == "MESH" and "1001" in o.name), "RubblePile", None)
for o in rb:
    if o.name in bpy.data.objects:
        bpy.data.objects.remove(o)
for k, (rx, ry, rz, sc_) in enumerate(((15.5, 3.0, F, 1.4), (-16.0, 6.5, F, 1.2), (4.0, 1.5, F, 0.9), (-6.5, 23.0, F, 1.1),
                                        (16.5, 21.5, F, 1.0), (-12.0, -10.0, 0.0, 1.6), (14.0, -30.0, 0.0, 1.5),
                                        (-30.0, -55.0, 0.0, 1.8), (26.0, 30.0, 0.0, 1.6), (-24.5, 12.0, 0.0, 1.4))):
    o = bpy.data.objects.new(f"Rubble{k:02d}", rb_src.data)
    C_PROPS.objects.link(o)
    o.location = (rx, ry, rz - 0.1)
    o.scale = (sc_,) * 3
    o.rotation_euler = (0, 0, k * 2.1)
bpy.data.objects.remove(rb_src)
# fallen masonry = real broken stone: copies of the pier/arch/vault fracture cells (textured outside + fresh interior)
rr = random.Random(77)
n_chunks = 0
for _ in range(140):
    zone = rr.random()
    if zone < 0.4:
        x, y = rr.uniform(-19, 19), rr.uniform(0.5, 26)          # under the fallen south roof
    elif zone < 0.65:
        x, y = rr.choice((-1, 1)) * rr.uniform(14.0, 19.5), rr.uniform(0.5, 49)   # aisles
    else:
        x, y = rr.uniform(-46, 46), rr.uniform(-74, -27) if rr.random() < 0.6 else rr.uniform(54, 84)
    if abs(x) < ST_W + 2.5 and -27 < y < 1.5:       # stairs + their foot stay clear
        continue
    if abs(x) < 10.5 and 26 < y < 50:                # the lair floor stays clear
        continue
    if abs(x) < 6.5 and 3 < y < 30:                  # the knight's walk up the nave stays clear
        continue
    if any(abs(x - s_ * COL_X) < 3.0 and abs(y - yy) < 3.0 for s_ in (1, -1) for yy in COL_Y):
        continue
    inside = -IX < x < IX and 0 < y < 50
    zb_ = F if inside else 0.0
    me = rr.choice(CHUNK_MESHES)
    sc_ = rr.uniform(0.25, 0.75) if me.name.startswith("pier") or me.name.startswith("arch") else rr.uniform(0.12, 0.3)
    rot = mathutils.Euler((rr.uniform(-0.5, 0.5), rr.uniform(-0.5, 0.5), rr.uniform(0, 6.28)))
    R = rot.to_matrix()
    minz = min((R @ v.co).z for v in me.vertices) * sc_
    o = bpy.data.objects.new(f"Rubble_chunk{n_chunks:03d}", me)
    C_PROPS.objects.link(o)
    o.location = (x, y, zb_ - minz - 0.05 * sc_)
    o.rotation_euler = rot
    o.scale = (sc_,) * 3
    n_chunks += 1
REPORT["rubble_chunks"] = n_chunks
# chests (Poly Haven, CC0) by the hoard + shield/sword of a fallen knight on the floor
for slug, loc, rz, scl in (("treasure_chest", (-7.6, 47.6, F + 0.3), 20, 1.4), ("treasure_chest", (7.4, 48.0, F + 0.3), -15, 1.4),
                           ("kite_shield", (-3.2, 24.0, F + 0.05), 0, 1.0), ("antique_estoc", (-2.0, 23.2, F + 0.04), 0, 1.0)):
    path = f"{PH}model_{slug}/{slug}_2k.gltf"
    objs = import_glb(path, C_TMP)
    bpy.context.view_layer.update()
    root = bpy.data.objects.new(f"Prop_{slug}", None)
    C_HOARD.objects.link(root)
    for o in objs:
        link(o, C_HOARD)
        if o.parent is None:
            o.parent = root
    root.location = loc
    root.rotation_euler = (math.radians(90) if slug != "treasure_chest" else 0, 0, math.radians(rz))
    if slug == "kite_shield":
        root.rotation_euler = (math.radians(-88), 0, math.radians(30))
    if slug == "antique_estoc":
        root.rotation_euler = (0, math.radians(90), math.radians(65))
    root.scale = (scl,) * 3

# courtyard dressing (Poly Haven, CC0): statues flanking the stair foot, barrels/crates by the walls, dead trees
def ph_prop(slug, loc, rz, height=None, scale=1.0, c=C_PROPS, name=None):
    objs = import_glb(f"{PH}model_{slug}/{slug}_2k.gltf", C_TMP)
    bpy.context.view_layer.update()
    root = bpy.data.objects.new(name or f"Prop_{slug}", None)
    c.objects.link(root)
    meshes = [o for o in objs if o.type == "MESH"]
    pts = [o.matrix_world @ Vector(b) for o in meshes for b in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    for o in objs:
        link(o, c)
        if o.parent is None:
            o.parent = root
            o.location -= Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    k = height / (hi.z - lo.z) if height else scale
    root.scale = (k,) * 3
    root.location = loc
    root.rotation_euler = (0, 0, math.radians(rz))
    return root, hi - lo


for s_ in (1, -1):
    px = s_ * (ST_W + 4.2)
    boxes(f"StatuePlinth_{'E' if s_ > 0 else 'W'}", [((px - 1.0, ST_Y0 + 0.5, 0.0), (px + 1.0, ST_Y0 + 2.5, 1.6), 0),
                                                     ((px - 1.15, ST_Y0 + 0.35, 1.45), (px + 1.15, ST_Y0 + 2.65, 1.6), 0)],
          [M_STAIRS], C_PROPS)
    _r, dims = ph_prop("gothic_statue", (px, ST_Y0 + 1.5, 1.6), 180, height=2.6, name=f"Statue_{'E' if s_ > 0 else 'W'}")
REPORT["statue_dims_raw"] = list(dims)
for k_, (x, y, rz) in enumerate(((-45.5, -60, 10), (45.0, -40, 80), (-45.0, 30, 140), (44.5, 70, 200), (-20, -74.5, 0))):
    ph_prop("wooden_barrels_01", (x, y, 0.0), rz, scale=1.0, name=f"Barrels{k_}")
    ph_prop("wooden_crate_01", (x + 2.2 * (1 if x < 0 else -1), y + 1.5, 0.0), rz + 25, scale=1.0, name=f"Crate{k_}")
for k_, (x, y, rz) in enumerate(((-30, -45, 30), (33, -58, 200), (-35, 65, 120))):
    ph_prop("dead_tree_trunk", (x, y, 0.0), rz, scale=1.0, name=f"DeadTree{k_}")      # a fallen log: natural size

# ---------------------------------------------------------------------------------------------------------
# 10. characters: dragon (Idle Stand, rigged, for scale) + knight (rest/idle)
# ---------------------------------------------------------------------------------------------------------
def char_import(path, name, c):
    objs = import_glb(path, c)
    root = bpy.data.objects.new(name, None)
    root.empty_display_type = "ARROWS"
    c.objects.link(root)
    for o in objs:
        if o.parent is None:
            o.parent = root
    arm = next(o for o in objs if o.type == "ARMATURE")
    return root, arm, [o for o in objs if o.type == "MESH"]


def set_action(arm, name):
    ac = bpy.data.actions[name]
    arm.animation_data.action = ac
    try:
        arm.animation_data.action_slot = ac.slots[0]
    except Exception as e:   # noqa
        print("[slot]", e)


d_root, d_arm, d_meshes = char_import(SF + "dragon_european_regina/model.glb", "DRAGON", C_CHARS)
# size-proof copy: the Walk pose (wings folded, tail straight) baked to a static mesh
set_action(d_arm, "Walk")
SC.frame_set(3)
dg = bpy.context.evaluated_depsgraph_get()
bm = bmesh.new()
proof_mats = []
for o in d_meshes:
    ev = o.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
    me.transform(o.matrix_world)
    off = len(proof_mats)
    proof_mats += list(o.data.materials)
    for p in me.polygons:
        p.material_index += off
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
WALK = bm_obj("DragonWalkStatic", bm, proof_mats, C_TMP)
wco = [v.co for v in WALK.data.vertices]
W_LO = Vector([min(c[i] for c in wco) for i in range(3)])
W_HI = Vector([max(c[i] for c in wco) for i in range(3)])
set_action(d_arm, "Idle Stand")
SC.frame_set(1)
d_root.scale = (DRAGON_S,) * 3
DRAGON_POS = Vector((0.0, 33.5, F))
d_root.location = DRAGON_POS
d_root["scale_note"] = "model x5: 14 m nose-to-tail (Walk), 6.9 m tall (Idle Stand head), 18.6 m wingspan (Fly)"
k_root, k_arm, k_meshes = char_import(SF + "knight_rx_sculpt/model.glb", "KNIGHT", C_CHARS)
k_root.scale = (KNIGHT_S,) * 3
KNIGHT_POS = Vector((1.0, 25.5, F))
k_root.location = KNIGHT_POS
k_root.rotation_euler = (0, 0, math.radians(180 - 8))      # glTF front is -Y; face the dragon (north)

# 12-dragon size proof: 3 rows x 4, nose to tail, wings folded, on the nave floor between the piers
L_D = (W_HI.y - W_LO.y) * DRAGON_S
W_D = (W_HI.x - W_LO.x) * DRAGON_S
clear_x = COL_X - PIER_W / 2
pitch_x = 2 * clear_x / 4
pitch_y = (DAIS_Y0 - IY0) / 3
proof_boxes = []
for r_ in range(3):
    for c_ in range(4):
        cx = -clear_x + pitch_x * (c_ + 0.5)
        cy = IY0 + pitch_y * (r_ + 0.5)
        o = bpy.data.objects.new(f"SizeProof_Dragon_{r_ * 4 + c_ + 1:02d}", WALK.data)
        C_PROOF.objects.link(o)
        o.scale = (DRAGON_S,) * 3
        o.location = (cx - (W_LO.x + W_HI.x) / 2 * DRAGON_S, cy - (W_LO.y + W_HI.y) / 2 * DRAGON_S, F - W_LO.z * DRAGON_S)
        proof_boxes.append((cx - W_D / 2, cx + W_D / 2, cy - L_D / 2, cy + L_D / 2))
gap_x = pitch_x - W_D
gap_y = pitch_y - L_D
ok = gap_x > 0 and gap_y > 0 and proof_boxes[0][0] >= -clear_x and proof_boxes[-1][3] <= DAIS_Y0
REPORT["checks"]["twelve_dragons_fit"] = dict(ok=bool(ok), dragon_len_m=round(L_D, 2), dragon_width_folded_m=round(W_D, 2),
                                              gap_side_m=round(gap_x, 2), gap_nose_tail_m=round(gap_y, 2),
                                              nave_clear_width_m=round(2 * clear_x, 2), floor_depth_m=DAIS_Y0 - IY0)
assert ok, REPORT["checks"]["twelve_dragons_fit"]
M_PROOF = simple("M_SizeProofDragon", (0.9, 0.25, 0.08), rough=0.6, emis=(1.0, 0.3, 0.08), estr=1.5)
WALK.data.materials.clear()
WALK.data.materials.append(M_PROOF)
bpy.data.objects.remove(WALK)
# scale bar (10 m, 1 m checks) for the overhead still
M_W = simple("M_BarWhite", (0.9, 0.9, 0.9), emis=(1, 1, 1), estr=2.0)
M_K = simple("M_BarBlack", (0.02, 0.02, 0.02))
bm = bmesh.new()
for i in range(10):
    add_box(bm, (-18.0, -8.0 - i, 0.02), (-17.0, -7.0 - i, 0.08), i % 2)
bar = bm_obj("ScaleBar_10m", bm, [M_W, M_K], C_PROOF)
bar.location = (0, 0, 0)
tcu = bpy.data.curves.new("ScaleBarLabel", "FONT")
tcu.body = "10 m"
tcu.size = 2.2
tl = bpy.data.objects.new("ScaleBarLabel", tcu)
C_PROOF.objects.link(tl)
tl.location = (-21.5, -14.0, 0.1)
tl.rotation_euler = (0, 0, math.radians(90))
tcu.materials.append(M_W)

# ---------------------------------------------------------------------------------------------------------
# 11. world (moonlit HDRI, moon found in the image), moon key, rose glow, haze
# ---------------------------------------------------------------------------------------------------------
hdr_path = PH + "hdri_moonlit_golf/moonlit_golf_4k.hdr"
hdr = bpy.data.images.load(hdr_path)
w_, h_ = hdr.size
px = np.empty(w_ * h_ * 4, dtype=np.float32)
hdr.pixels.foreach_get(px)
lum = px.reshape(h_, w_, 4)[:, :, :3].sum(2)
iy, ix = np.unravel_index(np.argmax(lum), lum.shape)
u, v = (ix + 0.5) / w_, (iy + 0.5) / h_
az_img = (u - 0.5) * 2 * math.pi            # lookup dir = (-cos a, sin a) in the image's frame
el = (v - 0.5) * math.pi
d_img = Vector((-math.cos(az_img), math.sin(az_img), 0))
MOON_AZ_WANT = math.radians(118)             # world azimuth (from +X, CCW): north-north-west, behind the cathedral
rot = math.atan2(d_img.y, d_img.x) - MOON_AZ_WANT
MOON_DIR = Vector((math.cos(MOON_AZ_WANT) * math.cos(el), math.sin(MOON_AZ_WANT) * math.cos(el), math.sin(el)))
REPORT["moon"] = dict(pixel=[int(ix), int(iy)], elevation_deg=round(math.degrees(el), 1), azimuth_world_deg=118,
                      hdri_rot_deg=round(math.degrees(rot), 1))
wd = bpy.data.worlds.new("NightHDRI")
SC.world = wd
wn = wd.node_tree
env = wn.nodes.new("ShaderNodeTexEnvironment")
env.image = hdr
tc = wn.nodes.new("ShaderNodeTexCoord")
mp = wn.nodes.new("ShaderNodeMapping")
mp.inputs["Rotation"].default_value = (0, 0, rot)
wn.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
wn.links.new(mp.outputs["Vector"], env.inputs["Vector"])
bg = wn.nodes["Background"]                       # light the scene: dim moonlit night
wn.links.new(env.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = WORLD_LIGHT
bgc = wn.nodes.new("ShaderNodeBackground")       # what the camera sees: the sky a little brighter (stars, moon)
hz = wn.nodes.new("ShaderNodeSeparateXYZ")      # camera sky: a dark misty band at the horizon hides the HDRI's
wn.links.new(tc.outputs["Generated"], hz.inputs["Vector"])   # own ground (a modern clubhouse + golf course)
hm = wn.nodes.new("ShaderNodeMapRange")
hm.inputs["From Min"].default_value, hm.inputs["From Max"].default_value = 0.03, 0.34
wn.links.new(hz.outputs["Z"], hm.inputs["Value"])
hmix = wn.nodes.new("ShaderNodeMix")
hmix.data_type = "RGBA"
wn.links.new(hm.outputs["Result"], hmix.inputs["Factor"])
hmix.inputs[6].default_value = (0.035, 0.045, 0.08, 1)
wn.links.new(env.outputs["Color"], hmix.inputs[7])
wn.links.new(hmix.outputs[2], bgc.inputs["Color"])
bgc.inputs["Strength"].default_value = WORLD_CAM
lp = wn.nodes.new("ShaderNodeLightPath")
mx = wn.nodes.new("ShaderNodeMixShader")
wn.links.new(lp.outputs["Is Camera Ray"], mx.inputs["Fac"])
wn.links.new(bg.outputs[0], mx.inputs[1])
wn.links.new(bgc.outputs[0], mx.inputs[2])
wn.links.new(mx.outputs[0], wn.nodes["World Output"].inputs["Surface"])
try:
    wd.sun_threshold = 1000.0     # we light the moon with an explicit sun lamp
except AttributeError:
    pass
ld = bpy.data.lights.new("Moon", "SUN")
ld.energy = MOON_E
ld.color = (0.55, 0.68, 1.0)
ld.angle = math.radians(0.6)
moon = bpy.data.objects.new("Moon", ld)
C_LIGHTS.objects.link(moon)
moon.rotation_euler = (-MOON_DIR).to_track_quat("-Z", "Y").to_euler()
# rose window back-glow (moonlight through stained glass) + warm hoard bounce
for nm, loc, rotx, e, col, size in (("RoseBackLight", (0, IY1 + 6.0, ROSE_Z), 90, 4000, (0.6, 0.7, 1.0), 12.0),
                                    ("HoardGlow", (0, IY1 - 7.0, F + 7.0), 0, 350, (1.0, 0.6, 0.25), 6.0)):
    l2 = bpy.data.lights.new(nm, "AREA")
    l2.energy = e
    l2.color = col
    l2.size = size
    lo2 = bpy.data.objects.new(nm, l2)
    C_LIGHTS.objects.link(lo2)
    lo2.location = loc
    lo2.rotation_euler = (math.radians(rotx), 0, 0)
    if nm == "HoardGlow":
        look_at(lo2, (0, IY1 - 2.0, F))
    else:
        look_at(lo2, (0, IY1 - 20, ROSE_Z - 8))
# haze: bounded boxes (a world volume would black out the sky)
M_HAZE = haze_mat("M_HallHaze", 0.006)
M_MIST = haze_mat("M_GroundMist", 0.012, color=(0.8, 0.85, 1.0), aniso=0.2)
boxes("HallHaze", [((-IX, IY0 - 6, F), (IX, IY1, F + 34.0), 0)], [M_HAZE], C_FX)
boxes("GroundMist", [((-CX, CY0, 0.0), (CX, PY0, 2.5), 0)], [M_MIST], C_FX)

# ---------------------------------------------------------------------------------------------------------
# 12. cameras for the stills (render_stills.py picks them by name)
# ---------------------------------------------------------------------------------------------------------
def cam(name, loc, target, lens, ortho=None, rotz=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36
    cd.clip_start = 0.1
    cd.clip_end = 5000
    o = bpy.data.objects.new(name, cd)
    C_CAM.objects.link(o)
    o.location = loc
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho
        o.rotation_euler = (0, 0, rotz or 0)
    else:
        look_at(o, target)
    return o


CAMS = {
    "a_wide": cam("CAM_a_wide", (74, -84, 34), (0, 14, 14), 32),
    "b_stairs": cam("CAM_b_stairs", (0, ST_Y0 - 5.5, 1.75), (0, 6, 11.5), 22),
    "c_scale": cam("CAM_c_scale", (-10.0, 29.0, F + 1.8), (1.0, 30.6, F + 3.8), 20),
    "d_overhead": cam("CAM_d_overhead", (0, (ST_Y0 + PY1) / 2, 160), None, 50, ortho=88, rotz=math.radians(90)),
    "e_pillar": cam("CAM_e_pillar", (3.0, 6.5, F + 3.0), (COL_X, COL_Y[1], F + 9.0), 22),
    "g_hoard": cam("CAM_g_hoard", (4.6, 41.0, F + 2.5), (0.0, 48.4, F + 2.1), 30),
    "f_roof": cam("CAM_f_roof", (-6.0, 10.0, F + 4.0), (2.0, 38.0, ZS_VAULT + 6.0), 18),
}
SC.camera = CAMS["a_wide"]
# camera sanity: first hit along the view ray must be far enough away (camera not buried in a wall)
dg = bpy.context.evaluated_depsgraph_get()
for k, o in CAMS.items():
    if o.data.type == "ORTHO":
        continue
    fwd = o.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    orig, hit, hob, dist = o.location.copy(), False, None, None
    for _ in range(8):     # skip volume boxes and the (render-hidden) size-proof dragons
        hit, loc, _n, _i, hob, _m = SC.ray_cast(dg, orig, fwd)
        if not hit:
            break
        if hob.users_collection[0].name in ("FX_haze", "SIZE_PROOF"):
            orig = loc + fwd * 0.01
            continue
        dist = (loc - o.location).length
        break
    REPORT["checks"][f"cam_{k}_first_hit"] = dict(obj=hob.name if hit else None, dist=round(dist, 2) if dist else None)
    assert (not hit) or dist > 2.0, (k, hob.name, dist)

# ---------------------------------------------------------------------------------------------------------
# 13. compositor: gentle fog glow on true highlights (fire), slight lens dispersion
# ---------------------------------------------------------------------------------------------------------
try:
    cg = bpy.data.node_groups.new("Post", "CompositorNodeTree")
    cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = cg.nodes.new("CompositorNodeRLayers")
    go = cg.nodes.new("NodeGroupOutput")
    gl = cg.nodes.new("CompositorNodeGlare")
    gl.inputs["Type"].default_value = "Fog Glow"
    for nm_, val in (("Quality", "Medium"), ("Threshold", 1.2), ("Size", 0.6), ("Strength", 0.6)):
        try:
            gl.inputs[nm_].default_value = val
        except Exception as e:    # noqa
            print("[comp]", nm_, e)
    lz = cg.nodes.new("CompositorNodeLensdist")
    try:
        lz.inputs["Dispersion"].default_value = 0.006
    except Exception as e:  # noqa
        print("[comp] lens", e)
    cg.links.new(rl.outputs["Image"], gl.inputs["Image"])
    cg.links.new(gl.outputs["Image"], lz.inputs["Image"])
    cg.links.new(lz.outputs["Image"], go.inputs["Image"])
    SC.compositing_node_group = cg
    SC.render.compositor_device = "GPU"
except Exception as e:   # noqa
    print("[comp] skipped:", e)

# ---------------------------------------------------------------------------------------------------------
# 14. cleanup + save + report
# ---------------------------------------------------------------------------------------------------------
fr = bpy.data.collections.get("Frags")
if fr:
    for o in list(fr.objects):
        if o.name not in C_ROOF.objects:
            bpy.data.objects.remove(o)
    bpy.data.collections.remove(fr)
for o in list(C_TMP.objects):
    bpy.data.objects.remove(o)
bpy.data.collections.remove(C_TMP)
SC.frame_set(1)
n_cells = sum(1 for o in bpy.data.objects if o.get("role") == "cell")
REPORT["dims"] = dict(arena_inner_x=2 * IX, arena_inner_y=IY1 - IY0, arena_floor_area_m2=2 * IX * (IY1 - IY0),
                      floor_height=F, platform=[PX1 - PX0, PY1 - PY0], pier_w=round(PIER_W, 2), pier_h=round(PIER_H, 2),
                      arch_top=round(ARCH_TOP, 2), vault_spring_above_floor=ZS_VAULT - F,
                      vault_crown_above_floor=round(VAULT_APEX - F, 2), stairs_run=round(ST_RUN, 2), stairs_width=2 * ST_W,
                      dragon_scale=DRAGON_S, knight_scale=round(KNIGHT_S, 4))
REPORT["cells_total"] = n_cells
REPORT["objects"] = len(bpy.data.objects)
REPORT["build_s"] = round(time.time() - T_START, 1)
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(A.out), compress=True)
print("[build] saved", A.out, json.dumps(REPORT["dims"]), "cells", n_cells, "objects", len(bpy.data.objects), flush=True)
if A.report:
    json.dump(REPORT, open(A.report, "w"), indent=1, default=str)
