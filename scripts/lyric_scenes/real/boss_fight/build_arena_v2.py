"""BOSS FIGHT — arena v2: Zeke's notes on v1, built ON TOP of arena_v1.blend (v1 stays reproducible on its own).

    blender -b --factory-startup arena_v1.blend --python-exit-code 1 --python build_arena_v2.py -- --out arena_v2.blend [--report r.json]
ALWAYS launched through:  scripts/server/render.sh <unit> scripts/lyric_scenes/real/boss_fight/render.sh build2

What v2 adds (Zeke's five notes, 2026-10-14):
  1. PRIZE: the placeholder launcher is replaced by the real RPG-7 scan (javadbayat, CC-BY), gilded (metallic 1, base
     (1.0, 0.77, 0.34), roughness 0.18-0.32 from the scan's own roughness, scan normal kept) on an iron cradle on the
     plinth, with its own warm key / rim / top light so it gleams. PRIZE_anchor = launcher centre, local +X = muzzle.
  2. RUBBLE inside: fallen masonry piles along the walls, around the pier bases and in the arcade gaps — fracture cells
     of the v1 piers/arches/vault + cut ashlar blocks + photoscan stone piles, dropped and SETTLED with Bullet (nothing
     floats, nothing interpenetrates). Build gate: no v2 object intrudes on the 12-dragon proof footprints.
  3. STUMPS: snapped pier stumps (the lower Voronoi cells of the v1 pier fracture = a jagged top), fallen upper cells
     and whole cut drums lying beside them — 2 in the aisles (0.6 scale), 4 as a ruined processional colonnade outside.
  4. STATUES: inside — 3 baroque saints (noe-3d scans, CC0) on plinths under the aisle lancets, a stone knight tomb
     (Southwark effigy scan, CC-BY, re-surfaced as stone), gargoyles on the gate piers and flanking the hoard dais.
     Outside — two 26 m colossi (Hl. Ludwig scan, CC0, armoured king with banner; one mirrored) on stepped plinths
     flanking the foot of the grand stairs, fire-lit from below.
  5. COMPOUND: St Martin church + bell tower (scan, CC-BY), Windleshaw chantry tower ruin (scan), St Alphage church
     ruin (scan), abbey wall ruins, three half-timbered houses + a stone house — around the cathedral inside the
     curtain walls, with braziers.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

T0 = time.time()
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--report", default="")
ap.add_argument("--nosim", action="store_true", help="skip the rubble settle sim (fast layout iteration)")
A = ap.parse_args(argv)

D = bpy.data
SC = bpy.context.scene
assert SC.name == "BossArena" and "PRIZE_anchor" in D.objects, "run on arena_v1.blend"

# ---------------------------------------------------------------------------------------------------------
# v1 dimensions (MUST match build_arena.py)
# ---------------------------------------------------------------------------------------------------------
F = 8.0
IX = 20.0
IY0, IY1 = 0.0, 50.0
WT = 2.5
COL_X = 13.0
COL_Y = [6.25, 18.75, 31.25, 43.75]
ST_W = 7.0
ST_Y_TOP = -2.5
ST_Y0 = ST_Y_TOP - (2 * (25 - 1) * 0.36 + 4.0 + 0.72)        # -24.5: foot of the stairs
PY0, PY1 = IY0 - WT, IY1 + WT
CX, CY0, CY1 = 50.0, -78.0, 86.0
DAIS_Y0 = 45.5
PLINTH_TOP = F + 2.2                                          # prize plinth top (v1 PLINTH[1].z)
PLINTH_Y = (47.7 + 49.1) / 2

MED = "/data/iris_assets/medieval/"
SF, R2, R3, PH = MED + "sketchfab/", MED + "r2_props/", MED + "r3/", MED + "polyhaven/"
REPORT = {"version": 2, "checks": {}, "added": {}, "removed_v1": []}
rng = random.Random(2424)


def log(*a):
    print("[v2]", *a, flush=True)


def coll(name, parent=None):
    c = D.collections.new(name)
    (parent or SC.collection).children.link(c)
    return c


C_TMP = coll("_tmp2")
C_STUMP = coll("V2_STUMPS")
C_RUB = coll("V2_RUBBLE")
C_STAT = coll("V2_STATUES_INSIDE")
C_COL = coll("V2_COLOSSI")
C_CMP = coll("V2_COMPOUND")
C_LT = coll("V2_LIGHTS")
C_PRIZE = D.collections["PRIZE_placeholder"]
C_PRIZE.name = "PRIZE"
C_CAM = D.collections["CAMERAS"]

M_WALL, M_STAIRS, M_PLAT = D.materials["M_Wall"], D.materials["M_Stairs"], D.materials["M_Platform"]
M_VAULT, M_INT, M_IRON = D.materials["M_Vault"], D.materials["M_StoneInterior"], D.materials["M_Iron"]

# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
_IMG = {}


def img(path, color=True):
    if (path, color) not in _IMG:
        im = D.images.load(path, check_existing=True)
        if not color:
            im.colorspace_settings.name = "Non-Color"
        _IMG[(path, color)] = im
    return _IMG[(path, color)]


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


def drop(objs):
    for o in objs:
        try:
            if o.name in D.objects:
                D.objects.remove(o)
        except ReferenceError:          # already removed
            pass


def mesh_co(me):
    a = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def merge(objs, name, M=Matrix(), keep=None):
    """all mesh objects -> ONE mesh in world space (times M); materials de-duplicated; UVs kept (bmesh merges by name)."""
    bm = bmesh.new()
    mats = []
    meshes = [o for o in objs if o.type == "MESH" and (keep is None or keep(o))]
    for o in meshes:
        me = o.data.copy()
        me.transform(M @ o.matrix_world)
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
    drop(objs)
    return out


def drop_flat_bottom(objs, frac=0.1):
    """keep-filter: drop scan display slabs (a mesh whose z-extent is < frac of the whole and sits at the bottom)."""
    meshes = [o for o in objs if o.type == "MESH"]
    pts = np.array([list(o.matrix_world @ Vector(c)) for o in meshes for c in o.bound_box])
    zlo, zhi = pts[:, 2].min(), pts[:, 2].max()

    def keep(o):
        z = [(o.matrix_world @ Vector(c)).z for c in o.bound_box]
        return not ((max(z) - min(z)) < frac * (zhi - zlo) and min(z) < zlo + 0.02 * (zhi - zlo))
    return keep


def bbox(me):
    co = mesh_co(me)
    return co.min(0), co.max(0)


def normalize(me, height=None, scale=None):
    """centre XY on the origin, bottom at z=0, scale to height (or by scale). Returns the scale used."""
    lo, hi = bbox(me)
    s = scale if scale else height / float(hi[2] - lo[2])
    me.transform(Matrix.Scale(s, 4) @ Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    return s


def crop(me, keep_fn):
    co = mesh_co(me)
    keep = keep_fn(co)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(~keep)[0]], context="VERTS")
    bm.to_mesh(me)
    bm.free()


def decimate(me, ratio):
    ob = D.objects.new("_dec", me)
    C_TMP.objects.link(ob)
    md = ob.modifiers.new("d", "DECIMATE")
    md.ratio = ratio
    dg = bpy.context.evaluated_depsgraph_get()
    new = D.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    D.objects.remove(ob)
    nm = me.name
    D.meshes.remove(me)
    new.name = nm
    return new


def place(me, name, loc, rotz_deg, c, sx=1.0):
    o = D.objects.new(name, me)
    c.objects.link(o)
    o.location = loc
    o.rotation_euler = (0, 0, math.radians(rotz_deg))
    o.scale = (sx, 1.0, 1.0)
    return o


def world_bbox(o):
    co = mesh_co(o.data)
    M = np.array(o.matrix_world)
    w = co @ M[:3, :3].T + M[:3, 3]
    return w.min(0), w.max(0)


def add_box(bm, lo, hi):
    vs = [bm.verts.new((x, y, z)) for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])]
    for f in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        bm.faces.new([vs[i] for i in f])


def boxes(name, specs, mat, c):
    bm = bmesh.new()
    for lo, hi in specs:
        add_box(bm, lo, hi)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = D.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    o = D.objects.new(name, me)
    c.objects.link(o)
    return o


def look_at(ob, target):
    ob.rotation_euler = (Vector(target) - ob.location).to_track_quat("-Z", "Y").to_euler()


def light(name, typ, loc, target, energy, color, size=0.5, spot_deg=None, c=C_LT):
    ld = D.lights.new(name, typ)
    ld.energy = energy
    ld.color = color
    if typ == "AREA":
        ld.size = size
    else:
        ld.shadow_soft_size = size
    if spot_deg:
        ld.spot_size = math.radians(spot_deg)
        ld.spot_blend = 0.5
    o = D.objects.new(name, ld)
    c.objects.link(o)
    o.location = loc
    if target is not None:
        look_at(o, target)
    return o


def node_lin(nt, a, b):
    nt.links.new(a, b)


def maprange(nt, src, a, b, c_, d, clamp=True):
    m = nt.nodes.new("ShaderNodeMapRange")
    m.clamp = clamp
    m.inputs["From Min"].default_value, m.inputs["From Max"].default_value = a, b
    m.inputs["To Min"].default_value, m.inputs["To Max"].default_value = c_, d
    nt.links.new(src, m.inputs["Value"])
    return m.outputs["Result"]


def mix_mul(nt, a, b):
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    m.blend_type = "MULTIPLY"
    m.inputs["Factor"].default_value = 1.0
    for sock, v in ((m.inputs[6], a), (m.inputs[7], b)):
        if isinstance(v, tuple):
            sock.default_value = (*v, 1)
        else:
            nt.links.new(v, sock)
    return m.outputs[2]


def principled(mat):
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mat and mat.node_tree else None


def src_of(sock):
    return sock.links[0].from_socket if sock.links else None


def unlink(nt, sock):
    for l in list(sock.links):
        nt.links.remove(l)


DETAIL = PH + "tex_rock_pitted_mossy/rock_pitted_mossy_"


def stone_surface(mat, tile=1.0, nstr=0.6, rough=(0.72, 0.95), grime=0.3, to_stone=None, moss=0.0, gain=1.0):
    """scan material -> weathered stone: keep the scan's albedo (or convert it to stone tone: to_stone=RGB), add a tiled
    Poly Haven rock normal + roughness for close-up detail (box-projected, object space = metres), macro grime."""
    b = principled(mat)
    if b is None:
        return
    nt = mat.node_tree
    N = nt.nodes
    col = src_of(b.inputs["Base Color"])
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / tile,) * 3
    node_lin(nt, tc.outputs["Object"], mp.inputs["Vector"])

    def tx(f):
        t = N.new("ShaderNodeTexImage")
        t.image = img(DETAIL + f + "_2k.jpg", False)
        t.projection = "BOX"
        t.projection_blend = 0.3
        node_lin(nt, mp.outputs["Vector"], t.inputs["Vector"])
        return t
    tn, tr = tx("nor_gl"), tx("rough")
    nm = N.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = nstr
    node_lin(nt, tn.outputs["Color"], nm.inputs["Color"])
    if not b.inputs["Normal"].links:
        node_lin(nt, nm.outputs["Normal"], b.inputs["Normal"])
    for k in ("Roughness", "Metallic"):
        unlink(nt, b.inputs[k])
    b.inputs["Metallic"].default_value = 0.0
    node_lin(nt, maprange(nt, tr.outputs["Color"], 0, 1, *rough), b.inputs["Roughness"])
    base = col if col is not None else None
    if to_stone is not None and col is not None:
        bw = N.new("ShaderNodeRGBToBW")
        node_lin(nt, col, bw.inputs["Color"])
        base = mix_mul(nt, maprange(nt, bw.outputs["Val"], 0.0, 0.6, 0.55, 1.25), to_stone)
    if base is None:
        return
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.25
    nz.inputs["Detail"].default_value = 6
    node_lin(nt, tc.outputs["Object"], nz.inputs["Vector"])
    g = maprange(nt, nz.outputs["Fac"], 0.35, 0.7, gain, gain * (1 - grime))
    base = mix_mul(nt, base, g)
    if moss > 0:
        mz = N.new("ShaderNodeTexNoise")
        mz.inputs["Scale"].default_value = 0.9
        mz.inputs["Detail"].default_value = 8
        node_lin(nt, tc.outputs["Object"], mz.inputs["Vector"])
        sep = N.new("ShaderNodeSeparateXYZ")
        gm = N.new("ShaderNodeNewGeometry")
        node_lin(nt, gm.outputs["Normal"], sep.inputs["Vector"])
        up = maprange(nt, sep.outputs["Z"], 0.45, 0.9, 0.0, 1.0)
        mm = N.new("ShaderNodeMath")
        mm.operation = "MULTIPLY"
        node_lin(nt, maprange(nt, mz.outputs["Fac"], 0.5, 0.7, 0.0, moss), mm.inputs[0])
        node_lin(nt, up, mm.inputs[1])
        mx = N.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        node_lin(nt, mm.outputs[0], mx.inputs["Factor"])
        node_lin(nt, base, mx.inputs[6])
        mx.inputs[7].default_value = (0.04, 0.055, 0.025, 1)
        base = mx.outputs[2]
    unlink(nt, b.inputs["Base Color"])
    node_lin(nt, base, b.inputs["Base Color"])


GOLD = (1.0, 0.77, 0.34)


def gild(mat):
    """Zeke's spec: metallic 1, base ~(1.0, 0.77, 0.34), roughness 0.2-0.3, keep the scan's normal map. The scan's own
    albedo luminance (compressed to 0.82..1.0) and roughness (remapped to 0.18..0.32) give wear without its colours."""
    b = principled(mat)
    if b is None:
        return
    nt = mat.node_tree
    col, rough = src_of(b.inputs["Base Color"]), src_of(b.inputs["Roughness"])
    for k in ("Base Color", "Metallic", "Roughness"):
        unlink(nt, b.inputs[k])
    if col is not None:
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        node_lin(nt, col, bw.inputs["Color"])
        node_lin(nt, mix_mul(nt, maprange(nt, bw.outputs["Val"], 0.0, 0.45, 0.82, 1.0), GOLD), b.inputs["Base Color"])
    else:
        b.inputs["Base Color"].default_value = (*GOLD, 1)
    b.inputs["Metallic"].default_value = 1.0
    if rough is not None:
        node_lin(nt, maprange(nt, rough, 0.0, 1.0, 0.18, 0.32), b.inputs["Roughness"])
    else:
        b.inputs["Roughness"].default_value = 0.24
    for k in ("Emission Strength",):
        if k in b.inputs:
            unlink(nt, b.inputs[k])
            b.inputs[k].default_value = 0.0


def world_of(o):
    if o.parent is None:
        return o.matrix_basis.copy()
    return world_of(o.parent) @ o.matrix_parent_inverse @ o.matrix_basis


# keep-out register: every v2 object's XY footprint (for clearing v1 dressing + the rubble spawner)
KEEP = []          # (x0, y0, x1, y1, label)


def keep_out(o_or_box, label, pad=0.0):
    if isinstance(o_or_box, tuple):
        x0, y0, x1, y1 = o_or_box
    else:
        lo, hi = world_bbox(o_or_box)
        x0, y0, x1, y1 = lo[0], lo[1], hi[0], hi[1]
    KEEP.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad, label))


def clear_v1_dressing(x0, y0, x1, y1, label):
    """v1 courtyard dressing (barrels, crates, dead trees, rubble piles/chunks) standing where a v2 building goes."""
    gone = []
    for o in D.objects:
        if not o.name.startswith(("Barrels", "Crate", "DeadTree", "Rubble")) or o.parent is not None:
            continue
        p = o.matrix_world.translation
        if x0 <= p.x <= x1 and y0 <= p.y <= y1:
            gone.append(o.name)
    for n in gone:
        o = D.objects[n]
        for ch in [c.name for c in o.children_recursive]:
            D.objects.remove(D.objects[ch])
        D.objects.remove(o)
    if gone:
        REPORT["removed_v1"].append({label: gone})


# ---------------------------------------------------------------------------------------------------------
# 1. PRIZE — the gold RPG-7
# ---------------------------------------------------------------------------------------------------------
for n in ("PRIZE_GoldRocketLauncher_PLACEHOLDER", "PrizeStand"):
    o = D.objects.get(n)
    if o:
        me = o.data
        D.objects.remove(o)
        if me.users == 0:
            D.meshes.remove(me)
PRIZE_S = 1.25                 # scan is real size (1.30 m incl. warhead) -> 1.63 m trophy
rp = import_glb(R2 + "prop_rpg7_javadbayat/model.glb")
Rz = Matrix.Rotation(-math.pi / 2, 4, "Z")                                       # scan muzzle +Y -> +X
rocket_objs = [o for o in rp if o.type == "MESH" and "rocket" in o.name.lower()]
me_body = merge([o for o in rp if o.type == "MESH" and o not in rocket_objs], "RPG7_body", M=Rz)
me_rock = merge(rocket_objs, "RPG7_rocket", M=Rz)
drop(rp)
# the scan ships UNLOADED: the PG-7 rocket lies parallel above the tube. Load it: rocket axis onto the tube axis,
# warhead out of the muzzle (back of the warhead = where its radius drops below 60 % of the max, just past the muzzle).
cb, cr = mesh_co(me_body), mesh_co(me_rock)
blo = cb.min(0)
rear = cb[cb[:, 0] < blo[0] + 0.03]
ax_y, ax_z = float(rear[:, 1].mean()), float(rear[:, 2].mean())
near_ax = cb[np.hypot(cb[:, 1] - ax_y, cb[:, 2] - ax_z) < 0.055]
muzzle_x = float(near_ax[:, 0].max())
r_lo, r_hi = cr.min(0), cr.max(0)
ry, rz_ = float(r_lo[1] + r_hi[1]) / 2, float(r_lo[2] + r_hi[2]) / 2
rad = np.hypot(cr[:, 1] - ry, cr[:, 2] - rz_)
xs_r = np.linspace(r_hi[0], r_lo[0], 120)
prof = np.array([rad[np.abs(cr[:, 0] - x) < 0.006].max() if np.any(np.abs(cr[:, 0] - x) < 0.006) else 0 for x in xs_r])
i_max = int(np.argmax(prof))
i_back = next((i for i in range(i_max, len(xs_r)) if prof[i] < 0.6 * prof[i_max]), len(xs_r) - 1)
x_back = float(xs_r[i_back])
me_rock.transform(Matrix.Translation((muzzle_x + 0.05 - x_back, ax_y - ry, ax_z - rz_)))
REPORT["prize_loading"] = dict(muzzle_x=round(muzzle_x, 3), warhead_r=round(float(prof[i_max]), 3),
                               protrudes_m=round(float(r_hi[0] - x_back) + 0.05, 3))
tmp = [D.objects.new("_b", me_body), D.objects.new("_r", me_rock)]
for o in tmp:
    C_TMP.objects.link(o)
me = merge(tmp, "PRIZE_RPG7_gold")
D.meshes.remove(me_body)
D.meshes.remove(me_rock)
co = mesh_co(me)
lo, hi = co.min(0), co.max(0)
me.transform(Matrix.Scale(PRIZE_S, 4) @ Matrix.Translation((-(lo[0] + hi[0]) / 2, -ax_y, -ax_z)))
for m in me.materials:
    gild(m)
TILT = math.radians(12)
anchor = D.objects["PRIZE_anchor"]
anchor.rotation_euler = (0, -TILT, math.radians(-8))           # muzzle up 12 deg, turned 8 deg toward the hall
anchor.scale = (1, 1, 1)
co = mesh_co(me)
Rw = (Matrix.Rotation(math.radians(-8), 3, "Z") @ Matrix.Rotation(-TILT, 3, "Y"))
cw = co @ np.array(Rw).T
lift = -float(cw[:, 2].min()) + 0.16
anchor.location = (0.0, PLINTH_Y, PLINTH_TOP + lift)
prize = D.objects.new("PRIZE_GoldRPG7", me)
C_PRIZE.objects.link(prize)
prize.parent = anchor
prize["note"] = "RPG-7 scan (javadbayat, CC-BY) gilded; anchor local +X = muzzle, origin on the tube axis mid-length"
bpy.context.view_layer.update()
# iron cradle: two forked posts under the tube where nothing hangs below it (found from the mesh, not guessed)
L_ = float(co[:, 0].max() - co[:, 0].min())
xs = np.linspace(co[:, 0].min() + 0.06 * L_, co[:, 0].max() - 0.06 * L_, 48)
bot = np.array([co[np.abs(co[:, 0] - x) < 0.012 * L_][:, 2].min() if np.any(np.abs(co[:, 0] - x) < 0.012 * L_) else -9
                for x in xs])
rear_i = int(np.argmax(np.where(xs < np.percentile(xs, 40), bot, -9)))
mz_l = (muzzle_x - (lo[0] + hi[0]) / 2) * PRIZE_S          # muzzle in the final local frame
front_i = int(np.argmax(np.where((xs > np.percentile(xs, 55)) & (xs < mz_l - 0.04), bot, -9)))
bm = bmesh.new()
Mw = anchor.matrix_world
for i in (rear_i, front_i):
    p = Mw @ Vector((xs[i], 0.0, float(bot[i])))
    add_box(bm, (p.x - 0.025, p.y - 0.025, PLINTH_TOP), (p.x + 0.025, p.y + 0.025, p.z - 0.012))
    add_box(bm, (p.x - 0.07, p.y - 0.11, p.z - 0.045), (p.x + 0.07, p.y + 0.11, p.z - 0.012))     # saddle
    for sy in (-1, 1):                                                                             # fork prongs
        add_box(bm, (p.x - 0.02, p.y + sy * 0.11 - 0.015, p.z - 0.045), (p.x + 0.02, p.y + sy * 0.11 + 0.015, p.z + 0.06))
add_box(bm, (-0.62, PLINTH_Y - 0.2, PLINTH_TOP), (0.62, PLINTH_Y + 0.2, PLINTH_TOP + 0.03))           # foot plate
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
cme = D.meshes.new("PrizeCradle")
bm.to_mesh(cme)
bm.free()
cme.materials.append(M_IRON)
cradle = D.objects.new("PrizeCradle", cme)
C_PRIZE.objects.link(cradle)
A_ = Vector(anchor.location)
light("Prize_Key", "AREA", A_ + Vector((-1.3, -2.2, 1.7)), A_, 90.0, (1.0, 0.86, 0.62), size=1.2, c=C_PRIZE)
light("Prize_Rim", "AREA", A_ + Vector((0.6, 1.5, 0.9)), A_ + Vector((0, -0.4, 0)), 70.0, (1.0, 0.8, 0.5), size=0.7, c=C_PRIZE)
light("Prize_Top", "SPOT", A_ + Vector((0.0, -1.2, 6.5)), A_, 900.0, (1.0, 0.85, 0.6), size=0.15, spot_deg=14, c=C_PRIZE)
REPORT["added"]["prize"] = dict(asset="RPG-7 (javadbayat, CC-BY)", scale=PRIZE_S, length_m=round(L_, 2),
                               anchor=[round(v, 3) for v in anchor.location], cradle_posts_x=[round(float(xs[rear_i]), 2), round(float(xs[front_i]), 2)])
log("prize", REPORT["added"]["prize"])

# ---------------------------------------------------------------------------------------------------------
# 2. STUMPS — lower Voronoi cells of the v1 pier fracture (jagged top) + fallen upper cells + cut drums
# ---------------------------------------------------------------------------------------------------------
CELLS = sorted([o for o in D.objects if o.name.startswith("Pier_E1_pier_cell")], key=lambda o: o.location.z)
PIER_ME = D.objects["Pier_E1_pier_INTACT"].data


def drum_mesh(z0, z1, name):
    """a whole cut drum of the pier template: bisect at z0 and z1, fill both cut faces with fresh-stone material."""
    me = PIER_ME.copy()
    me.name = name
    if M_INT.name not in [m.name for m in me.materials if m]:
        me.materials.append(M_INT)
    imat = [m.name for m in me.materials].index(M_INT.name)
    bm = bmesh.new()
    bm.from_mesh(me)
    for z, outer, inner in ((z1, True, False), (z0, False, True)):
        res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, z),
                                     plane_no=(0, 0, 1), clear_outer=outer, clear_inner=inner)
        edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
        fr = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
        for f in fr["faces"]:
            f.material_index = imat
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    c = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
    bmesh.ops.translate(bm, verts=bm.verts, vec=-c)
    bm.to_mesh(me)
    bm.free()
    return me


DRUMS = [drum_mesh(5.6, 7.9, "PierDrum_A"), drum_mesh(8.4, 10.2, "PierDrum_B"), drum_mesh(10.6, 12.6, "PierDrum_C")]
DYN = []        # (object, ground_z) — loose pieces for the settle sim
STUMPS = []


def stump(name, loc, rotz, s, cut, coll_, seed, n_fallen=3, drum=0, fall_dir=(1, 0)):
    rr = random.Random(seed)
    root = D.objects.new(name, None)
    coll_.objects.link(root)
    root.location = loc
    root.rotation_euler = (0, 0, math.radians(rotz))
    root.scale = (s, s, s)
    root["v2"] = "pier stump (lower cells of the v1 pier fracture)"
    kept = [c for c in CELLS if c.location.z < cut]
    fallen = [c for c in CELLS if c.location.z >= cut][:n_fallen]
    for k, c in enumerate(kept):
        o = D.objects.new(f"{name}_cell{k:02d}", c.data)
        coll_.objects.link(o)
        o.parent = root
        o.location = c.location
    fd = Vector((fall_dir[0], fall_dir[1], 0)).normalized()
    side = Vector((-fd.y, fd.x, 0))
    for k, c in enumerate(fallen):
        o = D.objects.new(f"{name}_fallen{k}", c.data)
        coll_.objects.link(o)
        o.scale = (s, s, s)
        o.location = Vector(loc) + fd * s * (3.2 + 1.6 * k) + side * s * rr.uniform(-1.6, 1.6) + Vector((0, 0, s * (2.4 + 1.2 * k)))
        o.rotation_euler = (rr.uniform(-1, 1), rr.uniform(-1, 1), rr.uniform(0, 6.3))
        DYN.append((o, loc[2]))
    if drum:
        dm = DRUMS[(seed + drum) % len(DRUMS)]
        r = max(abs(v.co.x) for v in dm.vertices) * s
        o = D.objects.new(f"{name}_drum", dm)
        coll_.objects.link(o)
        o.scale = (s, s, s)
        o.location = Vector(loc) + fd * s * 3.0 + side * s * 3.4 + Vector((0, 0, r + 0.25))
        o.rotation_euler = (math.radians(84), 0, math.atan2(fd.y, fd.x) + rr.uniform(-0.5, 0.5))   # lying on its side
        DYN.append((o, loc[2]))
    STUMPS.append(root)
    return root


# inside: two snapped aisle piers (0.6 scale: 2.8 m wide) under the open south roof
stump("Stump_AisleW", (-17.6, 14.0, F), 90, 0.6, 5.2, C_STUMP, 11, n_fallen=2, drum=1, fall_dir=(0, 1))
stump("Stump_AisleE", (17.6, 9.2, F), 0, 0.6, 7.7, C_STUMP, 12, n_fallen=3, drum=0, fall_dir=(0, 1))
# outside: a ruined processional colonnade either side of the approach to the stairs (full scale, 4.7 m wide)
for k, (x, y, cut, rz, fdir) in enumerate(((-16.0, -46.0, 4.5, 0, (-1, 0.3)), (16.0, -46.0, 9.8, 90, (1, -0.2)),
                                           (-16.0, -64.0, 7.2, 180, (-1, -0.4)), (16.0, -64.0, 3.2, 270, (1, 0.5)))):
    stump(f"Stump_Colonnade{k}", (x, y, 0.0), rz, 1.0, cut, C_STUMP, 20 + k, n_fallen=2 if cut > 4 else 1, drum=1, fall_dir=fdir)
bpy.context.view_layer.update()
for st in STUMPS:
    parts = [c for c in st.children]
    los = np.array([world_bbox(c)[0] for c in parts])
    his = np.array([world_bbox(c)[1] for c in parts])
    keep_out((los[:, 0].min(), los[:, 1].min(), his[:, 0].max(), his[:, 1].max()), st.name, pad=0.3)
    clear_v1_dressing(los[:, 0].min() - 1, los[:, 1].min() - 1, his[:, 0].max() + 1, his[:, 1].max() + 1, st.name)
    st["height_m"] = round(float(his[:, 2].max() - st.location.z), 2)
REPORT["added"]["stumps"] = {s.name: s["height_m"] for s in STUMPS}
log("stumps", REPORT["added"]["stumps"])

# ---------------------------------------------------------------------------------------------------------
# 3. STATUES INSIDE
# ---------------------------------------------------------------------------------------------------------
def statue_mesh(path, name, height, dec=None, front="-Y"):
    objs = import_glb(path)
    me = merge(objs, name, keep=drop_flat_bottom(objs))
    if dec:
        me = decimate(me, dec)
    if front == "+X":                       # bring every statue to the glTF convention: front = -Y
        me.transform(Matrix.Rotation(-math.pi / 2, 4, "Z"))
    normalize(me, height=height)
    return me


FACE_IN = {1: -90.0, -1: 90.0}           # rotz that turns a -Y-facing statue to face the nave from the E(+1)/W(-1) wall


def plinth(name, cx, cy, w, d, h, mat, c, rotz=0.0):
    """stepped stone plinth: base course, die, cornice."""
    specs = [((-w / 2 - 0.08, -d / 2 - 0.08, 0.0), (w / 2 + 0.08, d / 2 + 0.08, 0.18)),
             ((-w / 2, -d / 2, 0.18), (w / 2, d / 2, h - 0.14)),
             ((-w / 2 - 0.1, -d / 2 - 0.1, h - 0.14), (w / 2 + 0.1, d / 2 + 0.1, h))]
    o = boxes(name, specs, mat, c)
    o.location = (cx, cy, 0)
    o.rotation_euler = (0, 0, math.radians(rotz))
    return o


SAINTS = (("statue_hl_benedikt_noe3d", "Saint_Benedict", -1, 25.0), ("statue_hl_servatius_noe3d", "Saint_Servatius", 1, 25.0),
          ("statue_hl_nepomuk_noe3d", "Saint_Nepomuk", 1, 37.5))
for slug, nm, s, y in SAINTS:
    me = statue_mesh(R3 + slug + "/model.glb", nm, 3.0, dec=0.32)
    for m in me.materials:
        stone_surface(m, tile=0.8, nstr=0.35, grime=0.25, gain=0.95)
    lo, hi = bbox(me)
    pw = max(1.3, float(max(hi[0] - lo[0], hi[1] - lo[1])) * 0.85)
    px = s * (IX - 0.06 - pw / 2 - 0.1)
    pl = plinth(nm + "_Plinth", px, y, pw, pw, 1.25, M_VAULT, C_STAT)
    pl.location.z = F
    st = place(me, nm, (px, y, F + 1.25), FACE_IN[s], C_STAT)
    keep_out(pl, nm, pad=0.4)
M_CANDLE = D.materials.new("M_Candle")
M_CANDLE.use_nodes = True
principled(M_CANDLE).inputs["Base Color"].default_value = (0.78, 0.72, 0.6, 1)
principled(M_CANDLE).inputs["Roughness"].default_value = 0.45
principled(M_CANDLE).inputs["Subsurface Weight"].default_value = 0.4
M_FL = D.materials["M_Flame"]


def candles(cx, cy, z, n, seed, c=C_STAT):
    rr = random.Random(seed)
    bm, bf = bmesh.new(), bmesh.new()
    for k in range(n):
        x, y = cx + rr.uniform(-0.35, 0.35), cy + rr.uniform(-0.45, 0.45)
        h, r = rr.uniform(0.12, 0.42), rr.uniform(0.025, 0.045)
        g = bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=r, radius2=r, depth=h)
        bmesh.ops.translate(bm, verts=g["verts"], vec=(x, y, z + h / 2))
        g = bmesh.ops.create_uvsphere(bf, u_segments=8, v_segments=6, radius=1.0)
        for v in g["verts"]:
            t = (v.co.z + 1) / 2
            v.co = Vector((v.co.x * 0.011 * (1 - t) ** 0.6, v.co.y * 0.011 * (1 - t) ** 0.6, t * 0.05)) + Vector((x, y, z + h + 0.006))
    for b_, nm, m in ((bm, "Candles", M_CANDLE), (bf, "CandleFlames", M_FL)):
        me_ = D.meshes.new(f"{nm}_{seed}")
        b_.to_mesh(me_)
        b_.free()
        me_.materials.append(m)
        o = D.objects.new(f"{nm}_{seed}", me_)
        c.objects.link(o)
        if nm == "CandleFlames":
            o.visible_shadow = False
    light(f"CandleLight_{seed}", "POINT", (cx, cy, z + 0.55), None, 60.0, (1.0, 0.55, 0.22), size=0.08)


for k, (slug, nm, s, y) in enumerate(SAINTS):
    candles(s * (IX - 2.6), y + 0.6, F, 9, 300 + k)
# stone knight tomb (Southwark effigy scan; the oak is re-surfaced as weathered limestone)
me = statue_mesh(R3 + "statue_knight_effigy_southwark_natesid/model.glb", "Tomb_KnightEffigy", 1.0, dec=0.45)
lo, hi = bbox(me)
s_ = 3.4 / float(hi[0] - lo[0])
me.transform(Matrix.Scale(s_, 4))
lo, hi = bbox(me)
for m in me.materials:
    stone_surface(m, tile=0.7, nstr=0.4, grime=0.3, to_stone=(0.42, 0.405, 0.38))
tomb = place(me, "Tomb_KnightEffigy", (-(IX - 0.05 - float(hi[1] - lo[1]) / 2), 37.5, F), 90.0, C_STAT)
keep_out(tomb, "tomb", pad=0.4)
# gargoyles: photogrammetry gargoyle on each gate-pier top (facing out, south); gargoyle columns flanking the dais
me = statue_mesh(R3 + "statue_gargoyle_photogrammetry_pigfinite/model.glb", "Gargoyle_Perched", 2.1, front="+X")
for m in me.materials:
    stone_surface(m, tile=0.6, nstr=0.3, grime=0.3)
for s, top in ((1, F + 13.5), (-1, F + 11.0)):
    place(me, f"Gargoyle_GatePier_{'E' if s > 0 else 'W'}", (s * 8.0, -1.4, top), 0.0, C_STAT)
me = statue_mesh(R3 + "statue_gargoyle_column_alban/model.glb", "Gargoyle_Column", 3.6)
for m in me.materials:
    stone_surface(m, tile=0.6, nstr=0.3, grime=0.25)
for s in (1, -1):
    g = place(me, f"Gargoyle_Column_{'E' if s > 0 else 'W'}", (s * 11.3, 48.6, F), s * -12.0, C_STAT, sx=s)
    keep_out(g, g.name, pad=0.3)
REPORT["added"]["statues_inside"] = [o.name for o in C_STAT.objects if not o.name.endswith("Plinth")]

# ---------------------------------------------------------------------------------------------------------
# 4. COLOSSI — two 26 m guardians (Hl. Ludwig scan, CC0) flanking the foot of the stairs
# ---------------------------------------------------------------------------------------------------------
COL_H = 26.0
me = statue_mesh(R3 + "statue_hl_ludwig_noe3d/model.glb", "Colossus_Ludwig", COL_H)
for m in me.materials:
    stone_surface(m, tile=2.2, nstr=0.8, rough=(0.78, 0.97), grime=0.35, moss=0.35, gain=0.9)
co = mesh_co(me)
base = co[co[:, 2] < 0.05 * COL_H]
blo, bhi = base.min(0), base.max(0)
upper = co[co[:, 2] > 0.4 * COL_H]
banner_side = 1 if upper[:, 0].mean() > 0 else -1          # where the banner arm reaches (statue's own +X / -X)
PL_H = 5.0
COLOSSI = []
for s in (-1, 1):
    cx, cy = s * 19.5, -19.0
    w = float(bhi[0] - blo[0]) + 2.4
    d = float(bhi[1] - blo[1]) + 2.4
    pl = boxes(f"Colossus_Plinth_{'E' if s > 0 else 'W'}",
               [((-w / 2 - 1.6, -d / 2 - 3.2, 0.0), (w / 2 + 1.6, d / 2 + 1.6, 1.4)),
                ((-w / 2 - 0.6, -d / 2 - 0.6, 1.4), (w / 2 + 0.6, d / 2 + 0.6, 1.9)),
                ((-w / 2, -d / 2, 1.9), (w / 2, d / 2, PL_H - 0.5)),
                ((-w / 2 - 0.45, -d / 2 - 0.45, PL_H - 0.5), (w / 2 + 0.45, d / 2 + 0.45, PL_H))], M_PLAT, C_COL)
    yaw = -s * 7.0                                          # turned a little toward the approach axis
    pl.location = (cx, cy, 0.0)
    pl.rotation_euler = (0, 0, math.radians(yaw))
    sx = 1.0 if banner_side * s > 0 else -1.0               # banner on the OUTER side of each (mirror one)
    off = Matrix.Rotation(math.radians(yaw), 3, "Z") @ Vector((float(blo[0] + bhi[0]) / 2 * sx, float(blo[1] + bhi[1]) / 2, 0))
    col = place(me, f"Colossus_{'E' if s > 0 else 'W'}", (cx - off.x, cy - off.y, PL_H), yaw, C_COL, sx=sx)
    col["scale_note"] = f"Hl. Ludwig scan (noe-3d, CC0) at {COL_H} m on a {PL_H} m plinth; mirrored={sx < 0}"
    COLOSSI.append((col, pl))
    keep_out(pl, pl.name, pad=0.5)
    clear_v1_dressing(cx - w / 2 - 3, cy - d / 2 - 5, cx + w / 2 + 3, cy + d / 2 + 3, pl.name)
REPORT["added"]["colossi"] = dict(height_m=COL_H, plinth_m=PL_H, banner_side=banner_side)

# braziers + fire up-light for the colossi (moon is behind them: their faces need the fire)
BZ_PARTS = sorted([o for o in D.objects if o.name.startswith("Brazier00_") and o.name[len("Brazier00_"):].isdigit()],
                  key=lambda o: o.name)
BZ_FLAME = D.objects["Brazier00_flame"]
BZ_LIGHT = D.objects["Brazier00_light"]
n_bz = 0


def brazier(pos, energy_k=1.0, scale=1.0):
    global n_bz
    k = n_bz
    n_bz += 1
    for j, p in enumerate(BZ_PARTS):
        o = D.objects.new(f"V2Brazier{k:02d}_{j}", p.data)
        C_CMP.objects.link(o)
        o.location = pos
        o.scale = (scale,) * 3
        o.rotation_euler = (0, 0, k * 1.7)
    f = D.objects.new(f"V2Brazier{k:02d}_flame", BZ_FLAME.data)
    C_LT.objects.link(f)
    f.location = Vector(pos) + (BZ_FLAME.location - BZ_PARTS[0].location) * scale
    f.scale = (scale,) * 3
    f.visible_shadow = False
    ld = BZ_LIGHT.data.copy()
    ld.energy = BZ_LIGHT.data.energy * energy_k
    lo_ = D.objects.new(f"V2Brazier{k:02d}_light", ld)
    C_LT.objects.link(lo_)
    lo_.location = Vector(pos) + (BZ_LIGHT.location - BZ_PARTS[0].location) * scale
    keep_out((pos[0] - 0.8 * scale, pos[1] - 0.8 * scale, pos[0] + 0.8 * scale, pos[1] + 0.8 * scale), "brazier")
    return lo_


for col, pl in COLOSSI:
    s = 1 if col.location.x > 0 else -1
    R = Matrix.Rotation(pl.rotation_euler.z, 3, "Z")
    w = float(bhi[0] - blo[0]) + 2.4
    d = float(bhi[1] - blo[1]) + 2.4
    for sx_ in (-1, 1):
        p = pl.location + R @ Vector((sx_ * (w / 2 + 0.4), -d / 2 - 2.2, 1.4))
        brazier(p, energy_k=2.2, scale=1.5)
    tgt = pl.location + Vector((0, 0, PL_H + 0.5 * COL_H))
    light(f"Colossus_FireWash_{'E' if s > 0 else 'W'}", "SPOT", pl.location + R @ Vector((0, -d / 2 - 9.0, 1.2)), tgt,
          60000.0, (1.0, 0.55, 0.24), size=1.5, spot_deg=34)

# ---------------------------------------------------------------------------------------------------------
# 5. COMPOUND — smaller buildings around the cathedral, inside the curtain walls
# ---------------------------------------------------------------------------------------------------------
def heightmap_footprint(me, cell=1.0, rise=2.5, ring=2.0):
    """scan with its own terrain/base: ground z = median top height of the outer ring; footprint = cells standing
    more than `rise` above that ground."""
    co = mesh_co(me)
    lo, hi = co.min(0), co.max(0)
    nx, ny = int((hi[0] - lo[0]) / cell) + 1, int((hi[1] - lo[1]) / cell) + 1
    ix = ((co[:, 0] - lo[0]) / cell).astype(int)
    iy = ((co[:, 1] - lo[1]) / cell).astype(int)
    H = np.full((nx, ny), -1e9)
    np.maximum.at(H, (ix, iy), co[:, 2])
    xs = lo[0] + (np.arange(nx) + 0.5) * cell
    ys = lo[1] + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ringm = (X < lo[0] + ring) | (X > hi[0] - ring) | (Y < lo[1] + ring) | (Y > hi[1] - ring)
    ground = float(np.median(H[ringm & (H > -1e8)]))
    fp = H > ground + rise
    return ground, (float(X[fp].min()), float(Y[fp].min()), float(X[fp].max()), float(Y[fp].max()))


def strip_floor(me, g, tol=0.3):
    """delete near-horizontal faces lying at ground height g (scan display stands / ground patches)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    dead = [f for f in bm.faces if f.normal.z > 0.85 and all(abs(v.co.z - g) < tol for v in f.verts)]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    bm.to_mesh(me)
    bm.free()
    return len(dead)


def building(slug, name, loc, rotz, scale=1.0, max_w=None, max_l=None, skirt=1.5, keep=None, trim=True, flat=False, sink=0.0,
             floor=False, gain=None):
    objs = import_glb(R3 + slug + "/model.glb") if not slug.startswith("/") else import_glb(slug)
    me = merge(objs, name, keep=keep)
    if trim:
        g, (x0, y0, x1, y1) = heightmap_footprint(me)
        if floor:
            log(name, "floor faces stripped", strip_floor(me, g))
        crop(me, lambda co: (co[:, 0] > x0 - skirt) & (co[:, 0] < x1 + skirt) & (co[:, 1] > y0 - skirt)
             & (co[:, 1] < y1 + skirt) & (co[:, 2] > g - 0.4))
        lo, hi = bbox(me)
        me.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -g)))
    else:
        normalize(me, scale=1.0)
    if gain:
        for m in me.materials:
            stone_surface(m, tile=1.5, nstr=0.4, grime=0.3, gain=gain)
    lo, hi = bbox(me)
    s = scale
    if max_w:
        s = min(s, max_w / float(hi[0] - lo[0]))
    if max_l:
        s = min(s, max_l / float(hi[1] - lo[1]))
    me.transform(Matrix.Scale(s, 4))
    o = place(me, name, (loc[0], loc[1], 0.03 - sink), rotz, C_CMP)
    bpy.context.view_layer.update()
    blo, bhi = world_bbox(o)
    clear_v1_dressing(blo[0] - 1.5, blo[1] - 1.5, bhi[0] + 1.5, bhi[1] + 1.5, name)
    keep_out(o, name, pad=1.0)
    REPORT["added"].setdefault("compound", {})[name] = dict(scale=round(s, 3), size_m=[round(float(v), 1) for v in (bhi - blo)],
                                                           centre=[round(float(v), 1) for v in (blo + bhi) / 2])
    for x_, y_ in ((blo[0], blo[1]), (bhi[0], bhi[1])):
        assert -CX + 0.5 < x_ < CX - 0.5 and CY0 + 0.5 < y_ < CY1 - 0.5, (name, "outside the curtain walls", x_, y_)
    assert not (bhi[0] > -IX - WT - 3.5 and blo[0] < IX + WT + 3.5 and bhi[1] > PY0 - 26 and blo[1] < PY1 + 1), (name, "hits the cathedral/stairs")
    return o


# glTF front = -Y. long axes measured in probe_r3b (scans: real metres unless noted)
small = lambda frac: (lambda o: len(o.data.polygons) > frac)                      # noqa: E731  (drop display bases/proxies)
building("bld_st_martin_church_noe3d", "Chapel_StMartin", (-37.5, 24.0), 90.0, max_w=30.0, max_l=16.0, skirt=1.2)
building("bld_windleshaw_chantry_capptis", "BellTower_Chantry", (36.0, 15.0), 200.0, scale=1.25, trim=False,
         keep=lambda o: "Base" not in o.name)
building("bld_st_alphage_ruin_artfletch", "ChurchRuin_StAlphage", (27.0, 70.0), 0.0, max_w=38.0, max_l=25.0, skirt=0.5,
         keep=small(1000), floor=True, gain=0.5)
building(SF + "env_abbey_wall_ruins/model.glb", "AbbeyWallRuins", (-27.0, 73.0), 0.0, scale=0.5, trim=False)
house_objs = import_glb(R3 + "bld_medieval_houses3_aagawde/model.glb")
HOUSE_MESHES = [o for o in house_objs if o.type == "MESH"]
HOUSE_MESHES.sort(key=lambda o: o.name)
for k, ((hx, hy, rz), src) in enumerate(zip(((-41.5, -48.0, -90.0), (-41.0, -62.0, -90.0), (40.0, -63.0, 90.0)), HOUSE_MESHES)):
    me = merge([src], f"House_HalfTimber{k}")
    normalize(me, scale=1.0)
    o = place(me, f"House_HalfTimber{k}", (hx, hy, 0.0), rz, C_CMP)
    bpy.context.view_layer.update()
    blo, bhi = world_bbox(o)
    clear_v1_dressing(blo[0] - 1.5, blo[1] - 1.5, bhi[0] + 1.5, bhi[1] + 1.5, o.name)
    keep_out(o, o.name, pad=1.0)
drop(house_objs)
building("bld_medieval_house_rx", "House_Tall", (38.0, -46.0), 90.0, scale=1.0, trim=False)
building("bld_stone_house_1scanaday", "House_Stone", (39.5, -26.0), 90.0, scale=2.3, trim=False, sink=0.25)
# compound fire: braziers by the chapel, the houses and the chantry
for p in ((-29.5, 9.0, 0.0), (-29.5, 39.0, 0.0), (-35.0, -40.0, 0.0), (32.5, -36.0, 0.0), (30.0, 8.0, 0.0), (14.0, 60.0, 0.0),
          (-12.0, 66.0, 0.0)):
    brazier(Vector(p), energy_k=1.4, scale=1.2)

# ---------------------------------------------------------------------------------------------------------
# 6. RUBBLE — masonry piles, settled with Bullet
# ---------------------------------------------------------------------------------------------------------
POOL_BIG = [o.data for o in D.objects if o.get("role") == "cell" and o.name.startswith(("Pier_E1_pier", "Pier_E1_arch"))]
POOL_SLAB = [o.data for o in D.objects if o.get("role") == "cell" and o.name.startswith(("Vault_cell", "VaultRib0"))]


def ashlar(k):
    rr = random.Random(900 + k)
    bm = bmesh.new()
    sx, sy, sz = rr.uniform(0.45, 0.9), rr.uniform(0.3, 0.5), rr.uniform(0.25, 0.42)
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.035, segments=2, affect="EDGES")
    for v in bm.verts:                                       # weathered, chipped
        v.co += Vector((rr.uniform(-1, 1), rr.uniform(-1, 1), rr.uniform(-1, 1))) * 0.012
    me = D.meshes.new(f"Ashlar{k}")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M_WALL)
    return me


POOL_ASH = [ashlar(k) for k in range(8)]
# photoscan stone piles (decimated) as static cores of a few piles
sp = import_glb(SF + "env_stones_pile/model.glb")
STONES = decimate(merge(sp, "StonesPile"), 0.18)
normalize(STONES, scale=1.0)
lo, hi = bbox(STONES)
STONES.transform(Matrix.Scale(2.6 / float(max(hi[0] - lo[0], hi[1] - lo[1])), 4))
STATIC_PILES = []
for k, (x, y, z, rz, s) in enumerate(((-18.4, 2.3, F, 30, 1.0), (18.2, 46.6, F, 140, 0.9), (17.6, 20.5, F, 260, 0.8),
                                       (-18.3, 31.4, F, 75, 0.85), (-13.2, -48.5, 0.0, 10, 1.4), (19.5, -61.0, 0.0, 200, 1.2),
                                       (12.0, 58.5, 0.0, 120, 1.3))):
    o = place(STONES, f"StonesPile{k}", (x, y, z - 0.05), rz, C_RUB)
    o.scale = (s, s, s)
    STATIC_PILES.append(o)
bpy.context.view_layer.update()

PROOF = [o for o in D.collections["SIZE_PROOF"].objects if o.name.startswith("SizeProof_Dragon")]
PROOF_BOX = []
for o in PROOF:
    lo_, hi_ = world_bbox(o)
    PROOF_BOX.append((lo_[0], lo_[1], hi_[0], hi_[1]))
assert len(PROOF_BOX) == 12


def in_proof(x0, y0, x1, y1, pad=0.0):
    return any(x1 > a - pad and x0 < c + pad and y1 > b - pad and y0 < d + pad for a, b, c, d in PROOF_BOX)


SITES = []      # (x, y, ground_z, radius, count, kind)
for s in (1, -1):
    for y in COL_Y:                                    # pier bases, aisle side
        SITES.append((s * 16.4, y + rng.uniform(-1.2, 1.2), F, 1.1, 10, "big"))
    for y in (12.5, 25.0, 37.5):                       # arcade gaps (between piers, on the pier line, outer half)
        SITES.append((s * 13.3, y + rng.uniform(-1.5, 1.5), F, 0.9, 5, "mix"))
    for y in (2.0, 20.5, 31.0, 46.8):                  # along the aisle walls
        SITES.append((s * 18.9, y, F, 1.0, 9, "mix"))
    SITES.append((s * 13.0, 1.2, F, 1.2, 9, "mix"))     # inside the ruined south wall
    SITES.append((s * 17.0, 1.4, F, 1.0, 7, "slab"))
for st in STUMPS:                                      # around every stump
    sc = st.scale.x
    p = st.location
    for a in (0.6, 2.4, 4.2):
        SITES.append((p.x + math.cos(a) * 2.6 * sc, p.y + math.sin(a) * 2.6 * sc, p.z, 1.0 * sc, 4 if sc < 1 else 7, "big"))
for x, y in ((6.0, 62.0), (30.0, 54.5), (30.0, 22.5), (-24.0, 62.5), (-20.5, -50.0), (21.0, -56.0)):   # outside, by the ruins
    SITES.append((x, y, 0.0, 1.4, 7, "mix"))

# passive colliders = the real meshes (piers, walls, stumps, plinths, statues' plinths, static piles, floors)
PASSIVE_SRC = [o for o in D.objects if o.type == "MESH" and (
    o.name.endswith("_pier_INTACT") or o.name.endswith("_arch_INTACT") or o.name.startswith(
        ("Wall_E", "Wall_W", "Wall_S_", "GatePiers", "Buttresses", "StairCheek", "GrandStairs", "Platform", "Dais", "Colossus_Plinth"))
    or o.name.endswith("_Plinth") or (o.parent in STUMPS) or o in STATIC_PILES
    or o.name in ("Tomb_KnightEffigy", "Gargoyle_Column_E", "Gargoyle_Column_W"))]
AVOID_ONLY = [o for o in C_CMP.objects if o.type == "MESH"]      # spawn-avoid only: 1M-tri scans as Bullet colliders cost minutes


def near_passive(p, r):
    """True if a sphere (p, r) touches/penetrates any collider (closest point in local space; inside test by normal)."""
    for o in PASSIVE_SRC + AVOID_ONLY:
        lo_, hi_ = BBX[o.name]
        if not (lo_[0] - r < p.x < hi_[0] + r and lo_[1] - r < p.y < hi_[1] + r and lo_[2] - r < p.z < hi_[2] + r):
            continue
        Mi = MWI[o.name]
        lp = Mi @ p
        ok, loc, nrm, _ = o.closest_point_on_mesh(lp)
        if not ok:
            continue
        wl = MW_[o.name] @ loc
        if (wl - p).length < r or (lp - loc).dot(nrm) < 0:
            return True
    return False


bpy.context.view_layer.update()
BBX, MWI, MW_ = {}, {}, {}
for o in PASSIVE_SRC + AVOID_ONLY:
    M = world_of(o) if o.hide_viewport else o.matrix_world.copy()
    MW_[o.name] = M
    MWI[o.name] = M.inverted()
    lo_, hi_ = world_bbox(o)
    BBX[o.name] = (lo_, hi_)

spawned = []    # (pos, r)
n_skip = 0
for (sx_, sy_, gz, rad, cnt, kind) in SITES:
    for k in range(cnt):
        u = rng.random()
        if kind == "big" or (kind == "mix" and u < 0.45):
            me = rng.choice(POOL_BIG)
            sc = rng.uniform(0.16, 0.42)
        elif kind == "slab" or (kind == "mix" and u < 0.6):
            me = rng.choice(POOL_SLAB) if POOL_SLAB else rng.choice(POOL_BIG)
            sc = rng.uniform(0.12, 0.25)
        else:
            me = rng.choice(POOL_ASH)
            sc = rng.uniform(0.85, 1.25)
        co = mesh_co(me)
        r = float(np.linalg.norm(co, axis=1).max()) * sc
        placed = False
        for _t in range(40):
            a, d = rng.uniform(0, 2 * math.pi), rad * math.sqrt(rng.random())
            p = Vector((sx_ + d * math.cos(a), sy_ + d * math.sin(a), gz + r + 0.03 + rng.uniform(0, 0.6)))
            for _h in range(10):
                if all((p - q).length > r + rq + 0.02 for q, rq in spawned) and not near_passive(p, r):
                    placed = True
                    break
                p.z += r * 0.9
            if placed:
                break
        if not placed:
            n_skip += 1
            continue
        if gz > 1 and in_proof(p.x - r, p.y - r, p.x + r, p.y + r, pad=0.2):
            n_skip += 1
            continue
        o = D.objects.new(f"Rubble2_{len(spawned):03d}", me)
        C_RUB.objects.link(o)
        o.location = p
        o.scale = (sc, sc, sc)
        o.rotation_euler = (rng.uniform(0, 6.3), rng.uniform(0, 6.3), rng.uniform(0, 6.3))
        spawned.append((p, r))
        DYN.append((o, gz))
log(f"rubble spawned {len(spawned)} (+{len(DYN) - len(spawned)} stump pieces), skipped {n_skip}")


def settle(dyn, frames=150):
    """drop every loose piece onto the real collider meshes in a throw-away scene, keep the final rest pose."""
    SIM = D.scenes.new("SIM_settle")
    SIM.frame_start, SIM.frame_end = 1, frames
    win = bpy.context.window_manager.windows[0]
    main = win.scene
    win.scene = SIM
    with bpy.context.temp_override(window=win, scene=SIM):
        bpy.ops.rigidbody.world_add()
    rbw = SIM.rigidbody_world
    rbw.substeps_per_frame, rbw.solver_iterations, rbw.use_split_impulse = 20, 15, True
    rbw.point_cache.frame_start, rbw.point_cache.frame_end = 1, frames
    rbc = D.collections.new("SIM_rb")
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
    proxies = []
    for o in PASSIVE_SRC:
        px = D.objects.new("SIMP_" + o.name, o.data)
        px.matrix_world = MW_[o.name]
        make_rb(px, "PASSIVE", "MESH", friction=0.9, restitution=0.0)
        proxies.append(px)
    for nm, lo_, hi_ in (("SIMP_floorIn", (-IX - 0.1, IY0 - 0.1, F - 1.0), (IX + 0.1, IY1 + 0.1, F)),
                         ("SIMP_ground", (-CX, CY0, -1.0), (CX, CY1, 0.0))):
        fo = boxes(nm, [(lo_, hi_)], M_WALL, C_TMP)
        C_TMP.objects.unlink(fo)
        make_rb(fo, "PASSIVE", "BOX", friction=0.9, restitution=0.0)
        proxies.append(fo)
    sims = []
    for o, gz in dyn:
        so = D.objects.new("SIMD_" + o.name, o.data)
        so.matrix_world = o.matrix_world.copy() if o.parent is None else world_of(o)
        so.location, so.rotation_euler, so.scale = o.location.copy(), o.rotation_euler.copy(), o.scale.copy()
        dims = Vector(o.dimensions)
        vol = max(dims.x * dims.y * dims.z * 0.5, 0.01)
        make_rb(so, "ACTIVE", "CONVEX_HULL", mass=2400.0 * vol, friction=0.85, restitution=0.0,
                linear_damping=0.4, angular_damping=0.85)
        sims.append(so)
    t0 = time.time()
    with bpy.context.temp_override(window=win, scene=SIM):
        bpy.ops.ptcache.bake_all(bake=True)
    SIM.frame_set(frames - 12)
    before = [so.matrix_world.translation.copy() for so in sims]
    SIM.frame_set(frames)
    out = []
    for (o, gz), so, b in zip(dyn, sims, before):
        out.append((o, so.matrix_world.copy(), (so.matrix_world.translation - b).length, gz))
    log(f"settle: {len(sims)} bodies x {frames} frames, {len(proxies)} colliders, bake {time.time() - t0:.1f} s")
    with bpy.context.temp_override(window=win, scene=SIM):
        bpy.ops.ptcache.free_bake_all()
    win.scene = main
    for o in sims + proxies:
        D.objects.remove(o)
    D.collections.remove(rbc)
    D.scenes.remove(SIM)
    return out


n_moving = n_fell = n_proof = 0
if not A.nosim and DYN:
    res = settle(DYN)
    for o, M, mv, gz in res:
        o.parent = None
        o.matrix_world = M
        bpy.context.view_layer.update()
        lo_, hi_ = world_bbox(o)
        if lo_[2] < gz - 0.25:                        # fell off something / through the floor
            n_fell += 1
            D.objects.remove(o)
            continue
        if gz > 1 and in_proof(lo_[0], lo_[1], hi_[0], hi_[1]):
            n_proof += 1                              # rolled into the dragons' floor: take it out (honest count)
            D.objects.remove(o)
            continue
        if mv > 0.03:
            n_moving += 1
REPORT["checks"]["rubble_settle"] = dict(bodies=len(DYN), still_moving_gt_3cm=n_moving, fell_removed=n_fell,
                                         rolled_into_dragon_floor_removed=n_proof, sim=not A.nosim)
log("settle", REPORT["checks"]["rubble_settle"])

# ---------------------------------------------------------------------------------------------------------
# 7. CHECKS — the 12 dragons still fit (nothing v2 or v1 sits on their footprints), compound inside the walls
# ---------------------------------------------------------------------------------------------------------
bpy.context.view_layer.update()
v2_objs = [o for c in (C_STUMP, C_RUB, C_STAT, C_COL, C_CMP) for o in c.all_objects if o.type == "MESH"]
hits = []
for o in v2_objs:
    lo_, hi_ = world_bbox(o) if o.parent is None else (None, None)
    if lo_ is None:
        M = world_of(o)
        co = mesh_co(o.data)
        w = co @ np.array(M)[:3, :3].T + np.array(M)[:3, 3]
        lo_, hi_ = w.min(0), w.max(0)
    if hi_[2] > F - 0.5 and lo_[2] < F + 30 and in_proof(lo_[0], lo_[1], hi_[0], hi_[1]):
        hits.append(o.name)
v1_hits = []
for o in D.objects:
    if o.name.startswith("Rubble_chunk") or o.name.startswith("Rubble0"):
        lo_, hi_ = world_bbox(o)
        if lo_[2] > F - 1 and in_proof(lo_[0], lo_[1], hi_[0], hi_[1]):
            v1_hits.append(o.name)
for n in v1_hits:                                # v1 debris lying on the fight floor: cleared for v2 (reported)
    D.objects.remove(D.objects[n])
REPORT["removed_v1"].append({"v1 rubble on the dragon proof footprints": v1_hits})
REPORT["checks"]["twelve_dragons_fit_v2"] = dict(ok=not hits, v2_objects_on_proof_footprints=hits,
                                                 v1_rubble_cleared_from_footprints=len(v1_hits),
                                                 note="v1 geometry assert (3x4 dragons, 0.94 m side / 1.25 m nose-tail gaps) unchanged; "
                                                      "v2 adds: no v2 mesh overlaps any proof dragon's XY footprint above the floor")
assert not hits, ("v2 objects on the 12-dragon floor", hits)
log("12-dragon check", REPORT["checks"]["twelve_dragons_fit_v2"])

# ---------------------------------------------------------------------------------------------------------
# 8. CAMERAS (render_stills.py picks CAM_<name>)
# ---------------------------------------------------------------------------------------------------------
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


AP = Vector(anchor.location)
CAMS = {
    "a_compound": cam("CAM_a_compound", (-70, -112, 66), (2, 10, 2), 27),
    "a_compound_se": cam("CAM_a_compound_se", (80, -108, 54), (-4, 8, 4), 27),
    "b_colossi": cam("CAM_b_colossi", (0, -64, 1.75), (0, -14, 15.5), 18),
    "c_hall": cam("CAM_c_hall", (-11.0, 1.5, F + 3.4), (10.0, 30.0, F + 4.0), 16),
    "d_plan": cam("CAM_d_plan", (0, 4, 220), None, 50, ortho=190, rotz=math.radians(90)),
    "e_prize": cam("CAM_e_prize", (AP.x + 0.9, AP.y - 2.65, AP.z + 1.0), AP + Vector((0.12, 0, -0.1)), 40, fstop=3.5),
    "f_colossus": cam("CAM_f_colossus", (31.0, -45.0, 1.7), (17.5, -17.0, 19.0), 20),
    "h_aisle": cam("CAM_h_aisle", (-11.6, 9.3, F + 3.0), (-18.5, 30.0, F + 1.6), 18),
}
SC.camera = CAMS["a_compound"]
dg = bpy.context.evaluated_depsgraph_get()
for k, o in CAMS.items():
    if o.data.type == "ORTHO":
        continue
    fwd = o.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    orig, hit, hob, dist = o.location.copy(), False, None, None
    for _ in range(60):
        hit, loc, _n, _i, hob, _m = SC.ray_cast(dg, orig, fwd)
        if not hit:
            break
        if hob.users_collection[0].name in ("FX_haze", "SIZE_PROOF"):
            orig = loc + fwd * 0.05
            continue
        dist = (loc - o.location).length
        break
    else:
        hit = False
    REPORT["checks"][f"cam_{k}_first_hit"] = dict(obj=hob.name if hit else None, dist=round(dist, 2) if dist else None)
    assert (not hit) or dist > 1.2, (k, hob.name, dist)

# ---------------------------------------------------------------------------------------------------------
# 9. cleanup + save
# ---------------------------------------------------------------------------------------------------------
for o in list(C_TMP.objects):
    D.objects.remove(o)
D.collections.remove(C_TMP)
for me in list(D.meshes):
    if me.users == 0:
        D.meshes.remove(me)
SC.frame_set(1)
REPORT["objects"] = len(D.objects)
REPORT["tris_v2"] = int(sum(len(o.data.polygons) for c in (C_STUMP, C_RUB, C_STAT, C_COL, C_CMP, C_PRIZE)
                            for o in c.all_objects if o.type == "MESH"))
REPORT["build_s"] = round(time.time() - T0, 1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(A.out), compress=True)
log("saved", A.out, "objects", len(D.objects), "build", REPORT["build_s"], "s")
if A.report:
    json.dump(REPORT, open(A.report, "w"), indent=1, default=str)
