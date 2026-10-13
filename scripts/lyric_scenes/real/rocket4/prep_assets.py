"""ROCKET v2 asset prep: downloaded PD/CC0 models → one clean library .blend (run once).

    ~/.local/bin/blender -b --factory-startup --disable-autoexec --python-exit-code 1 \
        --python scripts/lyric_scenes/real/rocket2/prep_assets.py -- \
        [--assets /data/iris_assets] [--out /data/iris_assets/rocket2/rocket2_lib.blend]

Inputs (see ASSETS.md for URLs + licences; all are DATA — nothing downloaded is executed:
the NASA .blend is read with --disable-autoexec + use_scripts=False and only meshes are taken):
  nasa3d/sls_block1_x/Block 1 (GNC markings).blend   SLS external hardware (NASA, PD)
  nasa3d/Space Shuttle Parts/Solid Rocket Booster.glb   booster body (NASA, PD)
  nasa3d/orion_capsule/orion_no_fbc.stl                 Orion crew module (NASA, PD)
  nasa3d/Mobile Launcher/Mobile Launcher (assembled).glb  launch tower + deck (NASA, PD)
  polyhaven/mdl_shrub_0{1,2,4}, mdl_grass_bermuda_01     scrub scatter (Poly Haven, CC0)

Output objects (metres, +Z up, vehicle axis = Z, engines' exit plane ~ z 0.8):
  SLS_Engines, SLS_CoreHW, SLS_SRBatt_L/R, SLS_LVSA, SLS_ICPS, SLS_MSA, SLS_ESM0..2,
  SLS_Ogive, SLS_LAS, SRB_Body (5-segment length, nozzle exit z≈-0.6, axis at origin),
  Orion_CM (heat shield at z=0, Ø5.02 m), ML (deck top at z=0, rocket hole centred at origin),
  Shrub_*, Grass_* (scatter sources).
Every material slot is renamed to a SEMANTIC name (SEM_white, SEM_foam, SEM_metal, SEM_gray,
SEM_dark, SEM_nozzle_out, SEM_nozzle_in, SEM_gold, SEM_srb, SEM_ml, SEM_cm, SEM_plant_*), so the
scene builder assigns its own PBR materials by name.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--assets", default="/data/iris_assets")
ap.add_argument("--out", default="/data/iris_assets/rocket2/rocket2_lib.blend")
A = ap.parse_args(argv)
NASA = os.path.join(A.assets, "nasa3d")
SLS_BLEND = os.path.join(NASA, "sls_block1_x", "Block 1 (GNC markings).blend")

# ---------------------------------------------------------------------------------------------
# 1. SLS hardware: read objects from the NASA .blend (no scripts, no texts), group by subtree
# ---------------------------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
with bpy.data.libraries.load(SLS_BLEND, link=False) as (src, dst):
    assert len(src.texts) == 0, "unexpected text blocks in the NASA .blend"
    dst.objects = [n for n in src.objects]
sc = bpy.context.scene
for o in dst.objects:
    if o is not None:
        sc.collection.objects.link(o)
# MEASURED: every one of the 4,882 NASA objects carries an Action (a staging animation) and the file
# is saved at frame 0, the assembled pose. A fresh scene sits at frame 1 — part-way into it (SAJ
# panels swing out as 'fins'). Evaluate at frame 0, bake world transforms, drop the actions.
sc.frame_start = 0
sc.frame_set(0)
bpy.context.view_layer.update()
print("[prep] SLS objects", len(dst.objects), "evaluated at frame", sc.frame_current)

SEM = {}   # original material name → semantic slot


def sem_for(mname):
    m = mname or ""
    if m.startswith(("SLS_WHITE", "WH_Box", "BoatTail_WH", "ESM_Panels", "Jettison", "SLS_CoreStage")) or m in (
            "Material #99517", "Material #14", "Material #9947", "Material #9948", "Material #994976",
            "MLT_Green2", "Material #99518", "Material #9957", "Material #9959"):
        return "SEM_white"
    if m in ("Foam_LtYellow", "MainFeedLine", "Material #11"):
        return "SEM_foam"
    if m in ("Burh Metal", "Sliver", "Material #99511"):
        return "SEM_metal"
    if m == "Material #7":
        return "SEM_nozzle_out"
    if m == "Material #8":
        return "SEM_nozzle_in"
    if m in ("Material #994966", "Material #995052", "Material #99510"):
        return "SEM_gold"
    if m in ("FeedLineBrace", "Bot_Feed_Brace", "TopFeedLine_Cap", "12 - Default", "TSM_plugs", "SLS_Gray") or \
            m.startswith("Material #994") and m[-1].isdigit() and len(m) == len("Material #9941"):
        return "SEM_gray"
    return "SEM_dark"


def subtree(root_name):
    root = bpy.data.objects[root_name]
    out = [root]
    for c in root.children_recursive:
        out.append(c)
    return out


def dims(o):
    ws = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return max(max(w[i] for w in ws) - min(w[i] for w in ws) for i in range(3))


def join_group(name, roots, drop=()):
    """Join every mesh under the roots into ONE object `name` (world space, semantic slots)."""
    meshes = []
    for r in roots:
        for o in subtree(r):
            if o.type != "MESH" or o.name in drop or len(o.data.polygons) == 0:
                continue
            tris = sum(len(p.vertices) - 2 for p in o.data.polygons)
            if dims(o) < 0.35 and tris > 1500:          # dense bolts: invisible at 1080p, 1.6M tris
                continue
            meshes.append(o)
    if not meshes:
        raise SystemExit(f"[prep] empty group {name}")
    # bake world transforms + semantic material names into fresh mesh copies, then join
    parts = []
    for o in meshes:
        me = o.data.copy()
        me.transform(o.matrix_world)                       # frame-0 (assembled) world pose
        sems = [sem_for(s.material.name if s.material else "") for s in o.material_slots] or ["SEM_dark"]
        me.materials.clear()
        for s in sems:
            mat = bpy.data.materials.get(s) or bpy.data.materials.new(s)
            me.materials.append(mat)
        ob = bpy.data.objects.new(o.name + "_c", me)
        sc.collection.objects.link(ob)
        parts.append(ob)
    with bpy.context.temp_override(active_object=parts[0], selected_editable_objects=parts, object=parts[0]):
        bpy.ops.object.join()
    j = parts[0]
    j.name = name
    j.data.name = name
    # merge identical semantic slots (join keeps duplicates)
    uniq = []
    for s in j.data.materials:
        if s.name not in [u.name for u in uniq]:
            uniq.append(s)
    remap = {k: [u.name for u in uniq].index(s.name) for k, s in enumerate(j.data.materials)}
    mi = np.empty(len(j.data.polygons), np.int32)
    j.data.polygons.foreach_get("material_index", mi)
    mi = np.vectorize(lambda v: remap[int(v)])(mi).astype(np.int32) if len(mi) else mi
    j.data.materials.clear()
    for u in uniq:
        j.data.materials.append(u)
    j.data.polygons.foreach_set("material_index", mi)
    j.data.update()
    tris = sum(len(p.vertices) - 2 for p in j.data.polygons)
    print(f"[prep] {name}: {len(meshes)} parts, {tris} tris, slots {[u.name for u in uniq]}")
    return j


GROUPS = [
    ("SLS_Engines", ["Dummy_Engs"], ()),
    ("SLS_CoreHW", ["Dummy_Core"], ()),
    ("SLS_SRBatt_L", ["Dummy_SRB_"], ("Box7096",)),
    ("SLS_SRBatt_R", ["Dummy_SRB0"], ()),
    ("SLS_LVSA", ["Dummy_LVSA"], ()),
    ("SLS_ICPS", ["ICPS_2015_"], ()),
    ("SLS_MSA", ["Dummy_MPCV", "Dummy_NewS", "Dummy_Orio"], ()),
    ("SLS_ESM0", ["Dummy_ESM0"], ()),
    ("SLS_ESM1", ["Dummy_ESM1"], ()),
    ("SLS_ESM2", ["Dummy_ESM2", "Panel_camp"], ()),
    ("SLS_Ogive", ["Ogive_Orio"], ()),
    ("SLS_LAS", ["LAS_&_Abor"], ()),
]
# names in the NASA file are truncated 3ds-Max names; resolve by prefix
def resolve(prefix):
    hits = [o.name for o in dst.objects if o is not None and o.name.startswith(prefix)]
    hits.sort(key=len)
    if not hits:
        raise SystemExit("[prep] no object named " + prefix)
    return hits[0]


keep = []
for name, roots, drop in GROUPS:
    keep.append(join_group(name, [resolve(r) for r in roots], drop))
for o in list(bpy.data.objects):
    if o not in keep:
        bpy.data.objects.remove(o, do_unlink=True)
for o in keep:
    o.animation_data_clear()
for act in list(bpy.data.actions):
    bpy.data.actions.remove(act)
sc.frame_start = 1
sc.frame_set(1)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras, bpy.data.lights):
    for d in list(coll):
        if d.users == 0:
            coll.remove(d)


# ---------------------------------------------------------------------------------------------
# 2. Booster body: Shuttle SRB (inches, axis along X) → metres, axis +Z, stretched to 5 segments
# ---------------------------------------------------------------------------------------------
def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    return new


def bake_mesh(objs, name, sem):
    """Join imported meshes into one object with baked transforms and one semantic slot per
    original slot (sem may be a str or a callable(material_name) -> str)."""
    parts = []
    for o in objs:
        if o.type != "MESH":
            continue
        me = o.data.copy()
        me.transform(o.matrix_world)
        names = [s.material.name if s.material else "" for s in o.material_slots] or [""]
        me.materials.clear()
        for k, nm in enumerate(names):
            if sem is None:                                   # keep the asset's own material
                me.materials.append(o.material_slots[k].material if o.material_slots else None)
                continue
            s = sem(nm) if callable(sem) else sem
            me.materials.append(bpy.data.materials.get(s) or bpy.data.materials.new(s))
        ob = bpy.data.objects.new(name + "_p", me)
        sc.collection.objects.link(ob)
        parts.append(ob)
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)
    if len(parts) > 1:
        with bpy.context.temp_override(active_object=parts[0], selected_editable_objects=parts, object=parts[0]):
            bpy.ops.object.join()
    j = parts[0]
    j.name = name
    j.data.name = name
    return j


srb_new = import_glb(os.path.join(NASA, "Space Shuttle Parts", "Solid Rocket Booster.glb"))
for im in bpy.data.images:                      # keep the booster texture (joint bands, marks)
    if im.name.upper().startswith("SRB_ALL") and im.has_data:
        im.filepath_raw = os.path.join(os.path.dirname(A.out), "SRB_ALL.png")
        im.file_format = "PNG"
        os.makedirs(os.path.dirname(A.out), exist_ok=True)
        im.save()
        print("[prep] saved", im.filepath_raw)
srb = bake_mesh(srb_new, "SRB_Body", "SEM_srb")
me = srb.data
co = np.empty(len(me.vertices) * 3, np.float32)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3) * 0.0254                                  # inches → metres
# long axis → Z (nose = the end with the smaller radius)
ax = int(np.argmax(co.max(0) - co.min(0)))
L = co[:, ax]
rad = np.linalg.norm(np.delete(co, ax, axis=1) - np.delete(co, ax, axis=1).mean(0), axis=1)
lo_r = rad[L < L.min() + 2.0].mean()
hi_r = rad[L > L.max() - 2.0].mean()
z = (L - L.min()) if lo_r > hi_r else (L.max() - L)               # z = 0 at the nozzle end
others = [k for k in range(3) if k != ax]
x, y = co[:, others[0]], co[:, others[1]]
x = x - (x.max() + x.min()) / 2
y = y - (y.max() + y.min()) / 2
# 4-segment Shuttle SRB (45.5 m) → 5-segment SLS booster (~54 m): stretch only the cylinder
zlen = z.max()
za, zb = 0.17 * zlen, 0.80 * zlen                                # aft skirt top / forward skirt bottom
k = (zb - za + (54.2 - zlen)) / (zb - za)
z2 = np.where(z < za, z, np.where(z > zb, za + (zb - za) * k + (z - zb), za + (z - za) * k))
co2 = np.stack([x, y, z2 - 0.6], 1).astype(np.float32)
me.vertices.foreach_set("co", co2.ravel())
me.update()
srb.matrix_world = Matrix.Identity(4)
uv_ok = len(me.uv_layers) > 0
print(f"[prep] SRB_Body: length {z2.max():.1f} m (from {zlen:.1f}), max radius {np.hypot(x, y).max():.2f} m, uv {uv_ok}")

# ---------------------------------------------------------------------------------------------
# 3. Orion crew module (STL, units unknown) → Ø5.02 m, heat shield at z=0
# ---------------------------------------------------------------------------------------------
before = set(bpy.data.objects)
bpy.ops.wm.stl_import(filepath=os.path.join(NASA, "orion_capsule", "orion_no_fbc.stl"))
cm_new = [o for o in bpy.data.objects if o not in before]
cm = bake_mesh(cm_new, "Orion_CM", "SEM_cm")
me = cm.data
co = np.empty(len(me.vertices) * 3, np.float32)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
ext = co.max(0) - co.min(0)
ax = int(np.argmin(ext))                                          # the squat axis is the cone axis
others = [k for k in range(3) if k != ax]
dia = max(ext[others[0]], ext[others[1]])
s = 5.02 / dia
h = (co[:, ax] - co[:, ax].min()) * s
# heat shield end = the WIDE end
wide_low = np.ptp(co[h < 0.3 * h.max()][:, others[0]]) > np.ptp(co[h > 0.7 * h.max()][:, others[0]])
if not wide_low:
    h = h.max() - h
cx = (co[:, others[0]] - (co[:, others[0]].max() + co[:, others[0]].min()) / 2) * s
cy = (co[:, others[1]] - (co[:, others[1]].max() + co[:, others[1]].min()) / 2) * s
me.vertices.foreach_set("co", np.stack([cx, cy, h], 1).astype(np.float32).ravel())
me.update()
me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
print(f"[prep] Orion_CM: height {h.max():.2f} m, tris {sum(len(p.vertices) - 2 for p in me.polygons)}")

# ---------------------------------------------------------------------------------------------
# 4. Mobile Launcher (glb, ~1/48 scale): deck top at z=0, exhaust hole at the origin
# ---------------------------------------------------------------------------------------------
ml_new = import_glb(os.path.join(NASA, "Mobile Launcher", "Mobile Launcher (assembled).glb"))
ml = bake_mesh(ml_new, "ML", lambda n: "SEM_ml_rail" if "railing" in n.lower() else "SEM_ml")
me = ml.data
co = np.empty(len(me.vertices) * 3, np.float32)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
# scale: the ML base is 41.1 m wide across Y (real ML-1: 160 ft x 135 ft)
S = 41.1 / np.ptp(co[:, 1])
co *= S
# deck top = the height below which most of the footprint is solid: densest z band in the low 15 m
zs = co[:, 2]
hist, edges = np.histogram(zs[zs < zs.min() + 15], bins=60)
deck = edges[int(np.argmax(hist)) + 1]
# tower = the tall cluster; the rocket hole sits off the tower on the deck: find the footprint
# centre of verts in the deck band EXCLUDING the tower column
tower = co[co[:, 2] > deck + 30]
tx, ty = np.median(tower[:, 0]), np.median(tower[:, 1])
base = co[np.abs(co[:, 2] - deck) < 1.0]
bx0, bx1 = base[:, 0].min(), base[:, 0].max()
print(f"[prep] ML scale {S:.2f}, deck z {deck:.2f} (min {zs.min():.2f}), tower median ({tx:.1f},{ty:.1f}), "
      f"deck x {bx0:.1f}..{bx1:.1f}, height {zs.max() - deck:.1f} m above deck")
co[:, 2] -= deck
me.vertices.foreach_set("co", co.ravel().astype(np.float32))
me.update()
ml["deck_x"] = (float(bx0), float(bx1))
ml["tower_xy"] = (float(tx), float(ty))

# ---------------------------------------------------------------------------------------------
# 5. Poly Haven scatter sources (CC0): shrubs + grass, real scale already (metres)
# ---------------------------------------------------------------------------------------------
PH = os.path.join(A.assets, "polyhaven")
for d in ("mdl_shrub_01", "mdl_shrub_02", "mdl_shrub_04", "mdl_grass_bermuda_01"):
    gl = [f for f in os.listdir(os.path.join(PH, d)) if f.endswith(".gltf")]
    if not gl:
        print("[prep] no gltf in", d)
        continue
    new = import_glb(os.path.join(PH, d, gl[0]))
    nm = d.replace("mdl_", "Scatter_")
    ob = bake_mesh(new, nm, None)                             # Poly Haven PBR materials kept as-is
    print(f"[prep] {nm}: tris {sum(len(p.vertices) - 2 for p in ob.data.polygons)}, dims {tuple(round(v, 2) for v in ob.dimensions)}")

# ---------------------------------------------------------------------------------------------
for o in list(bpy.data.objects):
    if o.type != "MESH":
        bpy.data.objects.remove(o, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
    for d in list(coll):
        if d.users == 0:
            coll.remove(d)
for im in bpy.data.images:                                    # plant textures travel inside the library
    if im.source == "FILE" and not im.packed_file:
        im.pack()
os.makedirs(os.path.dirname(A.out), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=A.out, compress=True)
tot = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects)
print(f"[prep] saved {A.out}: {len(bpy.data.objects)} objects, {tot} tris")
