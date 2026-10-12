"""Vehicle prototypes for the POLICE scene: import a glTF (single model or a pack), cut out one vehicle,
normalise it (nose +Y, centred on x/y, wheels on z=0), split the wheels per axle (connected islands), and
put it in two kinds of UNLINKED collections that scene objects instance:

    proto.body      — everything that doesn't spin + emissive head/tail lamp quads (placed by ray-casting
                      onto the actual body surface, so they sit on the bumper whatever its shape)
    proto.axles     — [(y, z, radius, collection)] wheel parts translated so the axle centre is the origin

Packs (comrade1280's passenger / civil-service packs) lay their cars out in a RING, radially; the nose
direction is resolved per vehicle by `flip` (checked by eye from a lineup render).
"""
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

CARS = "/data/iris_assets/cars"

# name: (slug, group-node name or None, flip(bool), role). Flips verified 10-12 with a MARKER CUBE at +Y in the
# lineup render (camera on +X looking -X shows +Y on the RIGHT — the first lineup read it backwards).
# Real lengths for models authored at the wrong scale:
TRUE_LENGTH = {"charger": 5.10, "crownvic": 5.39, "hwy": 5.20}
SPECS = {
    "compact":   ("sf_passenger_pack", "Compact Body", True, "traffic"),
    "coupe":     ("sf_passenger_pack", "Coupe Body", True, "traffic"),
    "hatchback": ("sf_passenger_pack", "Hatchback Body", True, "traffic"),
    "minivan":   ("sf_passenger_pack", "minivan body", True, "traffic"),
    "offroad":   ("sf_passenger_pack", "Offroad Body", True, "traffic"),
    "pickup":    ("sf_passenger_pack", "Pickup Body", True, "traffic"),
    "sedan":     ("sf_passenger_pack", "Sedan Body", True, "traffic"),
    "sport":     ("sf_passenger_pack", "Sport body", True, "traffic"),
    "suv":       ("sf_passenger_pack", "SUV Body", True, "traffic"),
    "wagon":     ("sf_passenger_pack", "Wagon Body", True, "traffic"),
    "taxi":      ("sf_civil_pack", "taxi", False, "traffic"),
    "postvan":   ("sf_civil_pack", "postvan", False, "traffic"),
    "citybus":   ("sf_civil_pack", "citybus", False, "traffic"),
    "boxtruck":  ("sf_box_truck", None, False, "traffic"),
    "pol_sedan": ("sf_civil_pack", "police_sedan", False, "police"),
    "charger":   ("sf_police_charger", None, True, "police"),
    "crownvic":  ("sf_police_crownvic", None, True, "police"),
    "hwy":       ("sf_police_hwy", None, True, "police"),
}


class Proto:
    pass


def _meshes_under(o):
    r = [o] if o.type == "MESH" else []
    for c in o.children:
        r += _meshes_under(c)
    return r


def _bbox(objs):
    pts = np.array([list(o.matrix_world @ Vector(c)) for o in objs for c in o.bound_box])
    return pts.min(0), pts.max(0)


def _islands(me):
    nv = len(me.vertices)
    ed = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    lab = np.arange(nv)
    if len(ed) == 0:
        return lab
    while True:
        a, b = lab[ed[:, 0]], lab[ed[:, 1]]
        mn = np.minimum(a, b)
        new = lab.copy()
        np.minimum.at(new, ed[:, 0], mn)
        np.minimum.at(new, ed[:, 1], mn)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def _sub_mesh(me, keep_mask, name):
    import bmesh
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        kill = [v for v in bm.verts if not keep_mask[v.index]]
        bmesh.ops.delete(bm, geom=kill, context="VERTS")
        if len(bm.verts) == 0:
            return None
        out = bpy.data.meshes.new(name)
        bm.to_mesh(out)
    finally:
        bm.free()
    for m in me.materials:
        out.materials.append(m)
    return out


def _import(slug):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=os.path.join(CARS, slug, "model.glb"))
    new = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before]
    bpy.context.view_layer.update()
    return new


_PACK_CACHE = {}


def _pack_objects(slug):
    """Import a pack once; return its new objects (kept around for every vehicle cut from it)."""
    if slug not in _PACK_CACHE:
        _PACK_CACHE[slug] = _import(slug)
    return _PACK_CACHE[slug]


def _lamp_mats():
    if "VehHead" not in bpy.data.materials:
        for name, col, s in (("VehHead", (1.0, 0.86, 0.66), 14.0), ("VehTail", (1.0, 0.02, 0.01), 7.0)):
            m = bpy.data.materials.new(name)
            nt = m.node_tree
            nt.nodes.clear()
            e = nt.nodes.new("ShaderNodeEmission")
            e.inputs["Color"].default_value = col + (1,)
            e.inputs["Strength"].default_value = s
            o = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(e.outputs[0], o.inputs["Surface"])
    return bpy.data.materials["VehHead"], bpy.data.materials["VehTail"]


def build(name):
    slug, group, flip, role = SPECS[name]
    if group:
        objs = _pack_objects(slug)
        node = next(o for o in objs if o.name.split(".")[0] == group or o.name == group)
        body_meshes = _meshes_under(node)
        a, b = _bbox(body_meshes)
        c = (a + b) / 2
        if slug == "sf_passenger_pack":                       # wheels are separate top-level objects
            ang = math.atan2(c[1], c[0])
            rad = Vector((math.cos(ang), math.sin(ang), 0))
            tan = Vector((-rad.y, rad.x, 0))
            half_l = max(b[0] - a[0], b[1] - a[1]) / 2 + 0.2
            for o in objs:
                if o.type == "MESH" and o.name.startswith("Wheel"):
                    p = o.matrix_world.translation if o.data.vertices else None
                    wa, wb = _bbox([o])
                    wc = Vector(((wa + wb) / 2).tolist())
                    d = wc - Vector(c.tolist())
                    if abs(d.dot(rad)) < half_l and abs(d.dot(tan)) < 1.3:
                        body_meshes.append(o)
            yaw = math.pi / 2 - ang                           # radial outward → +Y
        else:
            ang = math.atan2(c[1], c[0])
            yaw = math.pi / 2 - ang
        src = body_meshes
    else:
        objs = _import(slug)
        src = [o for o in objs if o.type == "MESH"]
        a, b = _bbox(src)
        yaw = 0.0 if (b[1] - a[1]) >= (b[0] - a[0]) else math.pi / 2
        c = (a + b) / 2
    if flip:
        yaw += math.pi
    R = Matrix.Rotation(yaw, 4, "Z")
    T = Matrix.Translation(Vector((-c[0], -c[1], 0.0)))
    # bake world transforms into mesh copies
    meshes = []
    for o in src:
        if o.type != "MESH" or len(o.data.vertices) == 0:
            continue
        me = o.data.copy()
        me.transform(R @ T @ o.matrix_world)
        meshes.append(me)
    if not group:
        for o in objs:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except ReferenceError:
                pass
    co_all = np.concatenate([np.array([v.co[:] for v in me.vertices]) for me in meshes])
    lo, hi = co_all.min(0), co_all.max(0)
    if name in TRUE_LENGTH:
        k = TRUE_LENGTH[name] / float(hi[1] - lo[1])
        for me in meshes:
            me.transform(Matrix.Scale(k, 4))
        lo, hi = lo * k, hi * k
    shift = Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    for me in meshes:
        me.transform(Matrix.Translation(shift))
    lo, hi = lo + np.array(shift), hi + np.array(shift)
    # wheels: islands that look like a tyre (round in y/z, thin in x, touching the ground, off-centre)
    wheels = []                                               # (mesh, mask, centre, size)
    body_parts = []
    for me in meshes:
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        lab = _islands(me)
        cls = np.full(len(co), -1)
        tyres = []
        for u in np.unique(lab):
            sel = lab == u
            mn, mx = co[sel].min(0), co[sel].max(0)
            sz = mx - mn
            if (mn[2] < 0.12 and 0.35 < sz[2] < 1.4 and abs(sz[1] - sz[2]) < 0.25 * sz[2] and sz[0] < 0.75
                    and min(abs(mn[0]), abs(mx[0])) > 0.15 and np.sign(mn[0]) == np.sign(mx[0])):
                tyres.append((u, mn, mx))
        wheels.append((me, co, lab, tyres))
    allt = [(mn, mx) for _, _, _, ts in wheels for _, mn, mx in ts]
    ys = sorted(((mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2, (mx[2] - mn[2]) / 2) for mn, mx in allt)
    axles = []
    for y, z, r in ys:
        if axles and abs(axles[-1][0][-1] - y) < 0.8:
            axles[-1][0].append(y)
            axles[-1][1].append(z)
            axles[-1][2].append(r)
        else:
            axles.append(([y], [z], [r]))
    axles = [(float(np.mean(a)), float(np.mean(b)), float(np.mean(c))) for a, b, c in axles]
    proto = Proto()
    proto.name = name
    proto.role = role
    proto.length, proto.width, proto.height = float(hi[1] - lo[1]), float(hi[0] - lo[0]), float(hi[2])
    proto.body = bpy.data.collections.new(f"V_{name}_body")
    proto.axles = []
    axle_parts = {i: [] for i in range(len(axles))}
    for me, co, lab, tyres in wheels:
        cls = np.full(len(co), -1)
        boxes = []
        for u, mn, mx in tyres:
            yc = (mn[1] + mx[1]) / 2
            ai = int(np.argmin([abs(yc - a[0]) for a in axles]))
            boxes.append((mn - 0.04, mx + 0.04, ai))
        if boxes:
            for u in np.unique(lab):
                sel = lab == u
                mn, mx = co[sel].min(0), co[sel].max(0)
                for bmn, bmx, ai in boxes:
                    if (mn >= bmn).all() and (mx <= bmx).all():
                        cls[sel] = ai
                        break
        if (cls >= 0).any():
            for ai in range(len(axles)):
                if (cls == ai).any():
                    sm = _sub_mesh(me, cls == ai, f"V_{name}_ax{ai}")
                    if sm:
                        axle_parts[ai].append(sm)
            if (cls < 0).any():
                sm = _sub_mesh(me, cls < 0, f"V_{name}_b")
                if sm:
                    body_parts.append(sm)
        else:
            body_parts.append(me)
    for me in body_parts:
        proto.body.objects.link(bpy.data.objects.new(me.name, me))
    for ai, (y, z, r) in enumerate(axles):
        col = bpy.data.collections.new(f"V_{name}_axle{ai}")
        for me in axle_parts[ai]:
            me.transform(Matrix.Translation(Vector((0, -y, -z))))
            col.objects.link(bpy.data.objects.new(me.name, me))
        proto.axles.append((y, z, r, col))
    # lamps: ray-cast onto the body from the front and the back
    head, tail = _lamp_mats()
    bvh_src = [o for o in proto.body.objects]
    verts, faces = [], []
    for o in bvh_src:
        off = len(verts)
        verts += [v.co.copy() for v in o.data.vertices]
        faces += [tuple(off + i for i in p.vertices) for p in o.data.polygons]
    bvh = BVHTree.FromPolygons(verts, faces) if faces else None
    lz = 0.18 + 0.32 * min(proto.height, 1.6)
    lx = proto.width / 2 - 0.36
    lamps = []
    for sgn, mat in ((1, head), (-1, tail)):
        for sx in (-1, 1):
            org = Vector((sx * lx, sgn * (proto.length / 2 + 1.0), lz))
            hit = bvh.ray_cast(org, Vector((0, -sgn, 0)), 3.0) if bvh else (None,)
            y = (hit[0].y + sgn * 0.012) if hit and hit[0] is not None else sgn * proto.length / 2
            me = bpy.data.meshes.new(f"V_{name}_lamp")
            w, h = (0.30, 0.11) if sgn > 0 else (0.26, 0.10)
            me.from_pydata([(sx * lx - w / 2, y, lz - h / 2), (sx * lx + w / 2, y, lz - h / 2),
                            (sx * lx + w / 2, y, lz + h / 2), (sx * lx - w / 2, y, lz + h / 2)], [], [(0, 1, 2, 3)])
            me.materials.append(mat)
            ob = bpy.data.objects.new(me.name, me)
            ob.visible_shadow = False
            proto.body.objects.link(ob)
            lamps.append(y)
    # roof (for light bars): ray down at the middle
    hit = bvh.ray_cast(Vector((0, 0, proto.height + 2)), Vector((0, 0, -1)), 10.0) if bvh else (None,)
    proto.roof_z = float(hit[0].z) if hit and hit[0] is not None else proto.height
    proto.roof_y = 0.0
    return proto


def cleanup_imports():
    """Remove the imported source objects (prototypes keep only their own mesh copies)."""
    for slug, objs in list(_PACK_CACHE.items()):
        for o in objs:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except ReferenceError:
                pass
    _PACK_CACHE.clear()
