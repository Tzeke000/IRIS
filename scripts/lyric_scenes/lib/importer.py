"""Model import + mesh cleanup for scene builders (Blender 5.2, bpy). Replaces ~11 import_glb/_import copies, the 6
merge/bake-world copies, bbox/normalize/crop/decimate/islands, and boss v3's one-off KHR_materials_unlit fix.

    import sys; sys.path.insert(0, "/ABS/scripts/lyric_scenes")
    from lib import importer as imp
    objs = imp.import_model("/data/iris_assets/<scene>/<slug>/model.glb", coll=C_TMP)   # fix_unlit="report" (default)
    me = imp.bake_world(objs, "House", keep=imp.drop_flat_bottom(objs))                  # ONE mesh, world space
    imp.normalize(me, height=6.0)                                                        # centred XY, z0 = ground
    imp.write_log(OUT + "/import_log.json")      # every import (path, objects, polys, size, unlit, bad images)
                                                 # + every file loaded → lib/assets/credits.py --used

THE SCARS THIS MODULE MAKES DEFAULT
* KHR_materials_unlit (boss 10-15, white buildings): the glTF importer turns an unlit material into Emission mixed with
  Transparent by 'Is Camera Ray' — self-lit, ignores the lights, casts no shadow, glows white under exposure. EVERY
  import_model() runs find_unlit() and logs + prints the hits. fix_unlit="lit" converts them (unlit_to_lit: Principled,
  scan albedo scaled to target_mean, optional tiled detail normal — boss v3's hard-coded rock normal is now a parameter).
  Default is "report" because the fix changes the look; opt in per call.
* Missing / relative image paths (arcade: '//../../' textures → magenta): every import lists images that are not
  packed and whose file does not exist, or whose path is relative. strict_images=True raises instead.
* Scale (blender-rules validate-imported-mesh-scale): the log records each import's world bbox size in metres and any
  object with a non-unit scale, so a centimetre model shows up as "size 180.0 m" before it is placed.
* Hidden-collection instancing renders EMPTY (gotcha 1): import_model never hides the collection it links into.

Decision vs the old copies: drop=() by default (boss v1 deleted every "Icosphere*" — pass drop=("Icosphere",) to keep that).
"""
import json
import os

import numpy as np

_CACHE = {}
LOG = []          # one dict per import_model call (feeds lib/report + credits: what was USED)
USED_FILES = set()  # every model / image file actually loaded through this module (credits.py walks up to SOURCE.json)
UNLIT_NODE_TYPES = {"EMISSION", "LIGHT_PATH", "BSDF_TRANSPARENT", "MIX_SHADER"}


# ------------------------------------------------------------------------------------------------ pure helpers

def label_islands(nv, edges):
    """Connected-component label per vertex (min vertex index of its island) from an (E, 2) edge array. Pure numpy."""
    lab = np.arange(nv)
    ed = np.asarray(edges, np.int64).reshape(-1, 2)
    if len(ed) == 0:
        return lab
    while True:
        mn = np.minimum(lab[ed[:, 0]], lab[ed[:, 1]])
        new = lab.copy()
        np.minimum.at(new, ed[:, 0], mn)
        np.minimum.at(new, ed[:, 1], mn)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def normalize_matrix(lo, hi, height=None, scale=None, ground=True, center=True):
    """4x4 numpy matrix: centre XY on the origin, bottom at z=0 (ground), then scale (to height, or by scale)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    s = float(scale) if scale else (float(height) / float(hi[2] - lo[2]) if height else 1.0)
    T = np.eye(4)
    if center:
        T[0, 3], T[1, 3] = -(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2
    if ground:
        T[2, 3] = -lo[2]
    S = np.diag([s, s, s, 1.0])
    return S @ T, s


# ------------------------------------------------------------------------------------------------ bpy helpers

def _bpy():
    import bpy
    return bpy


def mesh_co(me):
    a = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def world_co(ob):
    co = mesh_co(ob.data).astype(np.float64)
    M = np.array(ob.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def bbox(x):
    """(lo, hi) of a Mesh (local coords), an Object (world, mesh data), or a list of objects (world, meshes only)."""
    bpy = _bpy()
    if isinstance(x, bpy.types.Mesh):
        co = mesh_co(x)
    elif isinstance(x, bpy.types.Object):
        co = world_co(x)
    else:
        parts = [world_co(o) for o in x if o.type == "MESH" and len(o.data.vertices)]
        if not parts:
            raise ValueError("bbox of no mesh vertices")
        co = np.concatenate(parts)
    return co.min(0), co.max(0)


def load_img(path, color=True, _cache={}):
    """Cached image load; ABSOLUTE existing path required (relative paths are the arcade magenta scar)."""
    if not os.path.isabs(path) or not os.path.isfile(path):
        raise FileNotFoundError(f"image must be an absolute existing path: {path}")
    key = (path, color)
    USED_FILES.add(os.path.abspath(path))
    im = _cache.get(key)
    if im is None or im.name not in _bpy().data.images:
        im = _bpy().data.images.load(path, check_existing=True)
        if not color:
            im.colorspace_settings.name = "Non-Color"
        _cache[key] = im
    return im


def img_mean(im, step=37):
    px = np.empty(len(im.pixels), np.float32)
    im.pixels.foreach_get(px)
    return px.reshape(-1, 4)[::step, :3].mean(0)


def bad_images(images):
    """[(image name, reason)] for images that are not packed and missing on disk, or have a relative path."""
    bpy = _bpy()
    out = []
    for im in images:
        if im is None or im.source not in ("FILE", "SEQUENCE", "TILED") or im.packed_file is not None:
            continue
        fp = im.filepath_raw or im.filepath
        if fp.startswith("//"):
            out.append((im.name, f"relative path {fp}"))
        if not os.path.isfile(bpy.path.abspath(fp, library=im.library)):
            out.append((im.name, f"missing file {bpy.path.abspath(fp)}"))
    return out


def _materials_of(objs):
    mats = []
    for o in objs:
        for s in getattr(o, "material_slots", []):
            if s.material and s.material not in mats:
                mats.append(s.material)
    return mats


def _images_of(mats):
    ims = []
    for m in mats:
        if m.node_tree:
            for n in m.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image and n.image not in ims:
                    ims.append(n.image)
    return ims


# ------------------------------------------------------------------------------------------------ unlit

def is_unlit(m):
    """The glTF importer's KHR_materials_unlit graph: an image texture → Emission, mixed with Transparent by 'Is Camera
    Ray', and no Principled BSDF. Hand-made emissive materials (flames, candles) never feed an image into Emission."""
    if not m or not m.node_tree:
        return False
    ty = {n.type for n in m.node_tree.nodes}
    if not (UNLIT_NODE_TYPES <= ty) or "BSDF_PRINCIPLED" in ty:
        return False
    return any(lk.from_node.type == "TEX_IMAGE" and lk.to_node.type == "EMISSION" for lk in m.node_tree.links)


def find_unlit(mats):
    return [m for m in mats if is_unlit(m)]


def unlit_to_lit(m, target_mean=0.30, rough=0.88, normal_tex=None, normal_scale=1.2, normal_strength=0.35):
    """Rewire an unlit material to a lit Principled BSDF (boss v3c fix, parameterised). The scan's albedo is scaled so
    its mean is ≤ target_mean (scans baked with light read too bright when lit). normal_tex = absolute path of a tiled
    detail normal map (box-projected in Object space, `normal_scale` m per tile) or None. Returns (texture_mean, scale)."""
    N, L = m.node_tree.nodes, m.node_tree.links
    tex = next(n for n in N if n.type == "TEX_IMAGE")
    mean = float(img_mean(tex.image).mean()) if tex.image else target_mean
    for n in list(N):
        if n.type in UNLIT_NODE_TYPES:
            N.remove(n)
    out = next(n for n in N if n.type == "OUTPUT_MATERIAL")
    b = N.new("ShaderNodeBsdfPrincipled")
    mx = N.new("ShaderNodeMix")
    mx.data_type, mx.blend_type = "RGBA", "MULTIPLY"
    mx.inputs["Factor"].default_value = 1.0
    k = min(1.0, target_mean / max(mean, 1e-3))
    mx.inputs[7].default_value = (k, k, k, 1)                  # RGBA B input
    L.new(tex.outputs["Color"], mx.inputs[6])                 # RGBA A input
    L.new(mx.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    if normal_tex:
        tc, mp = N.new("ShaderNodeTexCoord"), N.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1 / normal_scale,) * 3
        L.new(tc.outputs["Object"], mp.inputs["Vector"])
        tn = N.new("ShaderNodeTexImage")
        tn.image = load_img(normal_tex, color=False)
        tn.projection, tn.projection_blend = "BOX", 0.3
        L.new(mp.outputs["Vector"], tn.inputs["Vector"])
        nm = N.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        L.new(tn.outputs["Color"], nm.inputs["Color"])
        L.new(nm.outputs["Normal"], b.inputs["Normal"])
    L.new(b.outputs["BSDF"], out.inputs["Surface"])
    m.surface_render_method = "DITHERED"
    return round(mean, 3), round(k, 3)


# ------------------------------------------------------------------------------------------------ import

def import_model(path, coll=None, cache=True, fix_unlit="report", unlit_kw=None, drop=(), strict_images=False,
                 quiet=False):
    """Import .glb/.gltf/.fbx/.obj/.stl/.ply into `coll` (default: scene collection; never hidden). Returns the NEW objects.
    cache=True: a second call with the same path returns the same objects (copy them with bake_world/obj.copy()).
    fix_unlit: "report" (default) | "lit" (unlit_to_lit with unlit_kw) | None (skip the check)."""
    bpy = _bpy()
    D = bpy.data
    path = os.path.abspath(path)
    key = (path, fix_unlit)
    if cache and key in _CACHE and all(o.name in D.objects for o in _CACHE[key]):
        return _CACHE[key]
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    USED_FILES.add(path)
    ext = os.path.splitext(path)[1].lower()
    before = set(D.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise ValueError(f"unsupported model type {ext}")
    new = [o for o in D.objects if o not in before]
    if drop:
        for o in [o for o in new if o.name.startswith(tuple(drop))]:
            new.remove(o)
            D.objects.remove(o)
    target = coll or bpy.context.scene.collection
    for o in new:
        for u in list(o.users_collection):
            u.objects.unlink(o)
        target.objects.link(o)
    bpy.context.view_layer.update()
    mats = _materials_of(new)
    unlit = find_unlit(mats) if fix_unlit else []
    fixed = {}
    if fix_unlit == "lit":
        for m in unlit:
            fixed[m.name] = unlit_to_lit(m, **(unlit_kw or {}))
    bad = bad_images(_images_of(mats))
    meshes = [o for o in new if o.type == "MESH"]
    size = None
    if meshes and any(len(o.data.vertices) for o in meshes):
        lo, hi = bbox(meshes)
        size = [round(float(v), 3) for v in hi - lo]
    rec = {"path": path, "objects": [o.name for o in new], "meshes": len(meshes),
           "polys": int(sum(len(o.data.polygons) for o in meshes)), "materials": [m.name for m in mats],
           "size_m": size, "scaled_objects": [o.name for o in new if any(abs(s - 1) > 1e-4 for s in o.scale)],
           "armatures": [o.name for o in new if o.type == "ARMATURE"],
           "unlit": [m.name for m in unlit], "unlit_fixed": fixed, "bad_images": bad}
    LOG.append(rec)
    if not quiet:
        print(f"[importer] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: {len(new)} objs, "
              f"{rec['polys']} polys, size {size} m", flush=True)
        if unlit:
            what = "converted to lit" if fixed else "self-lit, ignore lights; pass fix_unlit='lit'"
            print(f"[importer] WARNING {len(unlit)} KHR_materials_unlit material(s) {rec['unlit']} — {what}", flush=True)
        for nm, why in bad:
            print(f"[importer] WARNING image {nm}: {why}", flush=True)
    if bad and strict_images:
        raise FileNotFoundError(f"{path}: bad images {bad}")
    if cache:
        _CACHE[key] = new
    return new


def write_log(path):
    """{"imports": LOG, "files": every model/image loaded} → JSON (lib/assets/credits.py --used reads it)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"imports": LOG, "files": sorted(USED_FILES)}, f, indent=1)
    return path


# ------------------------------------------------------------------------------------------------ mesh cleanup

def bake_world(objs, name, keep=None, dedup_mats=True, remove_sources=False):
    """Mesh objects → ONE new Mesh with world transforms baked in; materials de-duplicated (slot indices remapped);
    UVs/attributes kept. Sources are left alone unless remove_sources=True (then also evicted from the import cache)."""
    import bmesh
    bpy = _bpy()
    D = bpy.data
    bm = bmesh.new()
    mats = []
    try:
        for o in [o for o in objs if o.type == "MESH" and (keep is None or keep(o))]:
            me = o.data.copy()
            try:
                me.transform(o.matrix_world)
                idx = []
                for m in me.materials:
                    if not dedup_mats or m not in mats:
                        mats.append(m)
                    idx.append(len(mats) - 1 if not dedup_mats else mats.index(m))
                if idx and len(me.polygons):
                    mi = np.zeros(len(me.polygons), np.int32)
                    me.polygons.foreach_get("material_index", mi)
                    me.polygons.foreach_set("material_index", np.array(idx, np.int32)[np.clip(mi, 0, len(idx) - 1)])
                bm.from_mesh(me)
            finally:
                D.meshes.remove(me)
        out = D.meshes.new(name)
        bm.to_mesh(out)
    finally:
        bm.free()
    for m in mats:
        out.materials.append(m)
    if remove_sources:
        names = {o.name for o in objs}
        for k in [k for k, v in _CACHE.items() if any(o.name in names for o in v if o.name in D.objects)]:
            del _CACHE[k]
        for o in objs:
            if o.name in D.objects:
                D.objects.remove(o)
    return out


def normalize(me, height=None, scale=None, ground=True, center=True):
    """Centre XY on the origin, bottom at z=0, scale to `height` (or by `scale`). Returns the scale used."""
    from mathutils import Matrix
    lo, hi = bbox(me)
    M, s = normalize_matrix(lo, hi, height, scale, ground, center)
    me.transform(Matrix(M.tolist()))
    return s


def crop(me, keep_fn):
    """Delete vertices where keep_fn(co[N,3]) is False."""
    import bmesh
    keep = np.asarray(keep_fn(mesh_co(me)), bool)
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(~keep)[0]], context="VERTS")
        bm.to_mesh(me)
    finally:
        bm.free()


def decimate(me, ratio):
    """Collapse-decimate a mesh (via a temporary object + evaluated copy). Returns the NEW mesh (old one removed)."""
    bpy = _bpy()
    D = bpy.data
    ob = D.objects.new("_dec", me)
    bpy.context.scene.collection.objects.link(ob)
    try:
        md = ob.modifiers.new("d", "DECIMATE")
        md.ratio = ratio
        dg = bpy.context.evaluated_depsgraph_get()
        new = D.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    finally:
        D.objects.remove(ob)
    nm = me.name
    D.meshes.remove(me)
    new.name = nm
    return new


def islands(me):
    """Per-vertex island label (connected components over edges)."""
    ed = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ed)
    return label_islands(len(me.vertices), ed)


def sub_mesh(me, keep_mask, name):
    """New Mesh with only the vertices where keep_mask is True (materials kept). None if nothing is left."""
    import bmesh
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep_mask[v.index]], context="VERTS")
        if len(bm.verts) == 0:
            return None
        out = _bpy().data.meshes.new(name)
        bm.to_mesh(out)
    finally:
        bm.free()
    for m in me.materials:
        out.materials.append(m)
    return out


def drop_flat_bottom(objs, frac=0.1):
    """keep-filter for bake_world: drop scan display slabs (a mesh whose z-extent is < frac of the whole and that sits
    at the bottom)."""
    from mathutils import Vector
    meshes = [o for o in objs if o.type == "MESH"]
    pts = np.array([list(o.matrix_world @ Vector(c)) for o in meshes for c in o.bound_box])
    zlo, zhi = pts[:, 2].min(), pts[:, 2].max()

    def keep(o):
        z = [(o.matrix_world @ Vector(c)).z for c in o.bound_box]
        return not ((max(z) - min(z)) < frac * (zhi - zlo) and min(z) < zlo + 0.02 * (zhi - zlo))
    return keep
