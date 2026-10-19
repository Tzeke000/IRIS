"""Ruins: CC-BY Sketchfab arches/towers/castle (re-materialled into the stylised violet stone, scan albedo kept as
detail), the procedural ledge the big ghost sits on, GN ivy on every ruin, Quaternius (CC0) foreground foliage."""
import math

import numpy as np

import common as C
from common import NB, X
from lib import importer as IMP

KIT_Z = -5000.0          # parked kit meshes (gotcha 1: never hide an instancing source; park it underground)


# ---------------------------------------------------------------- materials

def haze_wrap(nb, shader, strength=0.7, near=80.0, far=1400.0):
    """aerial perspective inside the material: mix toward the violet haze by camera distance (clamped, no exp)"""
    cd = nb.n("ShaderNodeCameraData")
    f = nb.m("MULTIPLY", nb.m("POWER", nb.mr(cd.outputs["View Distance"], near, far), 0.8), strength)
    return C.mix_shader(nb, f, shader, C.emission(nb, C.HAZE_FAR, 0.8))


def stylise(mat, tint_dark=C.STONE_DARK, tint_lit=C.STONE_LIT, rim=0.35, haze=0.85, moon_rim=None):
    """In place: the scan's albedo → luminance → violet ramp; rough, non-metal; violet rim; distance haze.
    moon_rim = (strength, rgb, origin): an extra back-light rim on the faces turned toward the moon (silhouette edges
    against the sky glow), measured from `origin` (the object's position) so it is a constant world direction."""
    nt = mat.node_tree
    nb = NB(nt)
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) or \
        next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if bsdf is None or out is None:
        return mat
    src = bsdf.inputs["Base Color"].links[0].from_socket if bsdf.inputs["Base Color"].is_linked else None
    if src is not None:
        bw = nb.n("ShaderNodeRGBToBW")
        nb.link(src, bw.inputs[0])
        col = C.ramp(nb, bw.outputs[0], [(0.0, tint_dark), (0.75, tint_lit)])
        nb.link(col, bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*tint_dark, 1)
    for k in ("Metallic", "Roughness"):
        for l in list(bsdf.inputs[k].links):
            nt.links.remove(l)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.88
    for l in list(bsdf.inputs["Emission Color"].links):
        nt.links.remove(l)
    bsdf.inputs["Emission Strength"].default_value = 0.0
    r = nb.m("MULTIPLY", nb.m("POWER", C.facing(nb, 0.3), 3.0), rim)
    sh = C.add_shader(nb, bsdf.outputs[0], C.emission(nb, C.RIM_VIOLET, r))
    if moon_rim:
        k, rgb, origin = moon_rim
        md = tuple(float(x) for x in C.dir_to(origin, C.MOON_POS))
        nrm = nb.n("ShaderNodeNewGeometry").outputs["Normal"]
        toward = nb.mr(nb.v("DOT_PRODUCT", nrm, md), -0.15, 0.6)                # faces turned to the moon
        edge = nb.m("POWER", C.facing(nb, 0.5), 1.2)                             # grazing → silhouette edges
        sh = C.add_shader(nb, sh, C.emission(nb, rgb, nb.m("MULTIPLY", nb.m("MULTIPLY", toward, edge), k)))
    sh = haze_wrap(nb, sh, haze)
    for l in list(out.inputs["Surface"].links):
        nt.links.remove(l)
    nb.link(sh, out.inputs["Surface"])
    mat.surface_render_method = "DITHERED"
    return mat


def proc_stone(name="LedgeStone"):
    m, nb, out = X.new_mat(name)
    tc = nb.n("ShaderNodeTexCoord")
    nz = C.noise(nb, tc.outputs["Object"], scale=1.4, detail=6.0, rough=0.62)
    vo = nb.n("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
    nb.set(vo.inputs["Scale"], 2.2)
    nb.link(tc.outputs["Object"], vo.inputs["Vector"])
    cracks = nb.mr(vo.outputs["Distance"], 0.0, 0.02, 0.55, 1.0)
    lum = nb.m("MULTIPLY", nb.mr(nz.outputs["Fac"], 0.3, 0.7), cracks)
    b = X.principled(nb, base=C.ramp(nb, lum, [(0.0, (0.006, 0.004, 0.014)), (1.0, (0.075, 0.045, 0.16))]), rough=0.9)
    bump = nb.n("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    nb.link(nb.m("ADD", nz.outputs["Fac"], cracks), bump.inputs["Height"])
    nb.link(bump.outputs["Normal"], b.inputs["Normal"])
    r = nb.m("MULTIPLY", nb.m("POWER", C.facing(nb, 0.3), 3.0), 0.3)
    sh = haze_wrap(nb, C.add_shader(nb, b.outputs[0], C.emission(nb, C.RIM_VIOLET, r)))
    nb.link(sh, out.inputs["Surface"])
    return m


def silhouette_mat(name="Foliage", rim_col=(0.55, 0.18, 0.75), rim=0.08):
    """near-black violet leaves with a soft pink rim (the cover's foreground silhouettes); alpha from the kit texture"""
    m, nb, out = X.new_mat(name)
    b = X.principled(nb, base=(0.006, 0.004, 0.014), rough=0.6)
    b.inputs["Subsurface Weight"].default_value = 0.0
    r = nb.m("MULTIPLY", nb.m("POWER", C.facing(nb, 0.4), 2.5), rim)
    sh = C.add_shader(nb, b.outputs[0], C.emission(nb, rim_col, r))
    nb.link(sh, out.inputs["Surface"])
    m.use_backface_culling = False
    return m


# ---------------------------------------------------------------- import helpers

def _kit(path, name, height, coll, keep=None, stylise_kw=None):
    """import → one baked mesh (world transforms), normalised to `height` m, materials stylised → Mesh"""
    objs = IMP.import_model(path, coll=coll)
    me = IMP.bake_world(objs, name, keep=keep, remove_sources=False)
    IMP.normalize(me, height=height)
    for m in me.materials:
        if m is not None and not m.get("_stylised"):
            stylise(m, **(stylise_kw or {}))
            m["_stylised"] = 1
    return me, objs


def _place(me, name, loc, rot_z=0.0, scale=1.0, coll=None, tilt=(0.0, 0.0)):
    import bpy
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (C.deg(tilt[0]), C.deg(tilt[1]), C.deg(rot_z))
    ob.scale = (scale,) * 3
    return ob


def _remove(objs):
    import bpy
    for o in objs:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o)


# ---------------------------------------------------------------- the ruins

def build(coll, R):
    import bpy
    out = {}
    tmp = X.coll("_import_tmp")
    A = C.ASSETS
    # arches: three arches (full / partial / destroyed) — split by the importer's parent empties
    objs = IMP.import_model(A + "/sf_arch_pillars/model.glb", coll=tmp)
    parts = {}
    for o in objs:
        if o.type == "MESH":
            key = (o.parent.name if o.parent else o.name).split("_")[0]
            parts.setdefault(key, []).append(o)
    R.log("arch parts", {k: len(v) for k, v in parts.items()})
    arch_me = {}
    for k, v in parts.items():
        me = IMP.bake_world(v, f"Arch_{k}")
        IMP.normalize(me, height=7.5)
        arch_me[k] = me
    for m in {m for me in arch_me.values() for m in me.materials if m}:
        stylise(m)
    keys = sorted(arch_me)
    full = next((k for k in keys if "Full" in k), keys[0])
    part = next((k for k in keys if "Partial" in k), keys[-1])
    dest = next((k for k in keys if "Destroy" in k), keys[-1])
    # centre: a broken arcade standing in the water (the cover's mid-ground)
    out["arch_a"] = _place(arch_me[part], "Ruin_ArchA", (2.0, 46.0, -0.6), rot_z=4, coll=coll, scale=1.5)
    out["arch_b"] = _place(arch_me[dest], "Ruin_ArchB", (14.0, 58.0, -0.8), rot_z=-8, coll=coll, scale=1.6)
    out["arch_c"] = _place(arch_me[full], "Ruin_ArchC", (-12.0, 52.0, -0.6), rot_z=12, coll=coll, scale=1.7)
    out["arch_d"] = _place(arch_me[part], "Ruin_ArchD", (-1.0, 70.0, -0.8), rot_z=-10, coll=coll, scale=1.8)
    # right foreground: the brick arch (aqueduct-like pair)
    br, o2 = _kit(A + "/sf_ancient_arch/model.glb", "ArchBrick", 8.5, tmp)
    # v2: the cover's right arch sits low at the frame's right edge (ndc ≈ 0.86–1.0); the castle window stays clear
    out["brick_a"] = _place(br, "Ruin_BrickA", (23.5, 20.7, -0.8), rot_z=58, coll=coll, scale=0.78)     # side-on to the camera
    out["brick_b"] = _place(br, "Ruin_BrickB", (37.0, 31.0, -0.8), rot_z=50, coll=coll, scale=0.95)
    out["brick_c"] = _place(br, "Ruin_BrickC", (5.5, 24.0, -1.2), rot_z=6, coll=coll, scale=0.95)      # mid-ground bridge
    out["brick_d"] = _place(br, "Ruin_BrickD", (-1.5, 30.0, -2.4), rot_z=-4, coll=coll, scale=0.9, tilt=(0, 4))
    # overgrown arch remnant close to the ledge, in the shallows
    og, o3 = _kit(A + "/sf_overgrown_arch/model.glb", "ArchOvergrown", 3.6, tmp)
    out["overgrown"] = _place(og, "Ruin_Overgrown", (5.4, 8.5, -0.3), rot_z=-30, coll=coll)
    # towers: intact / damaged / ruined — left edge (tall broken) + the far ruined fortress hill
    objs = IMP.import_model(A + "/sf_ruined_towers/model.glb", coll=tmp)
    tw = {}
    for o in [o for o in objs if o.type == "MESH"]:
        mname = o.data.materials[0].name if o.data.materials else o.name
        me = IMP.bake_world([o], f"Tower_{mname}")
        IMP.normalize(me, height=1.0)
        tw[mname] = me
    for m in {m for me in tw.values() for m in me.materials if m}:
        stylise(m)
    R.log("towers", sorted(tw))
    tk = sorted(tw)
    dmg = next((k for k in tk if "damaged" in k), tk[0])
    rui = next((k for k in tk if "ruined" in k), tk[-1])
    itc = next((k for k in tk if "intact" in k), tk[0])
    out["tower_l1"] = _place(tw[dmg], "Ruin_TowerL1", (-13.5, 9.0, -0.5), rot_z=30, coll=coll, scale=15.0)
    out["tower_l2"] = _place(tw[rui], "Ruin_TowerL2", (-6.5, 21.0, -0.5), rot_z=-60, coll=coll, scale=11.0)
    out["tower_l3"] = _place(tw[dmg], "Ruin_TowerL3", (-24.0, 30.0, -0.5), rot_z=100, coll=coll, scale=17.0)
    # distant ruined fortress on a hill (left of the castle)
    hill = hill_mesh(coll)
    out["hill"] = hill
    for i, (x, y, s, rz, k) in enumerate([(-58, 205, 14, 10, rui), (-44, 214, 18, 70, dmg), (-30, 208, 11, 0, itc),
                                          (-72, 220, 12, 45, dmg)]):
        out[f"fort_{i}"] = _place(tw[k], f"Ruin_Fort{i}", (x, y, hill_z(x, y) - 1.0), rot_z=rz, coll=coll, scale=s)
    # the castle on the cliff (centimetre model → metres via normalize)
    # v2: taller + less hazed (cover). v3 (Zeke 10-19: "the castle is faint in the haze"): darker stone and about half
    # the aerial haze so the silhouette separates from the lit ridge behind it, plus a pink moon-side rim on the
    # spires; still hazier than anything in the foreground (it must stay 430 m away)
    cx, cy, _ = C.CASTLE_POS
    cm, o5 = _kit(A + "/sf_fantasy_castle/model.glb", "Castle", 70.0, tmp,
                  stylise_kw=dict(tint_dark=(0.002, 0.002, 0.010), tint_lit=(0.016, 0.015, 0.075), haze=0.22, rim=0.15,
                                  moon_rim=(3.0, (1.0, 0.50, 0.92), (cx, cy, C.CASTLE_CLIFF_H + 30.0))))
    out["castle"] = _place(cm, "Castle", (cx + 4, cy + 4, C.CASTLE_CLIFF_H - 7.0), rot_z=200, coll=coll)
    _remove(list(tmp.objects))
    bpy.data.collections.remove(tmp)
    # the ledge the big ghost sits on (procedural blocks)
    out["ledge"] = ledge(coll)
    R.add("ruins.objects", sorted(o.name for o in out.values()))
    return out


def hill_z(x, y):
    hx, hy = -50.0, 212.0
    r = math.hypot((x - hx) / 60.0, (y - hy) / 34.0)
    return 26.0 * max(0.0, 1.0 - r * r) - 2.0


def hill_mesh(coll):
    n = 50
    V, F = [], []
    xs = np.linspace(-130, 30, n)
    ys = np.linspace(160, 260, n)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            z = hill_z(x, y) + 1.5 * math.sin(x * 0.3) * math.cos(y * 0.25)
            V.append((x, y, z if z > -2 else -3.0))
    for j in range(n - 1):
        for i in range(n - 1):
            F.append((j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i))
    return X.mesh_obj("FortHill", V, F, coll=coll, mat=proc_stone("HillStone"), smooth=True)


def ledge(coll):
    """broken stone wall the big ghost sits on (x≈-2.2). v2: less regular — every block draws its own height, width,
    depth, setback, tilt and wear; courses wander (each block sits on the skyline of the blocks below it, not on a
    straight row line); some blocks are missing (a recessed core fills the hole), some slump, the top is ragged and
    stepped. A doorway gap + lintel and a broken pillar stump stay."""
    from mathutils import Euler
    rng = np.random.default_rng(7)
    mb = X.MB()
    x0, x1 = -9.0, 1.6
    y0, depth = 0.2, 1.5
    zb = -0.6
    res = 0.05
    xs = np.arange(x0 - 0.6, x1 + 0.6, res)
    sky = np.full(len(xs), zb)                                     # current top of the wall along x
    # target top per column: full height in the middle (the ghost's seat), ragged stepped ends
    target = np.full(len(xs), C.LEDGE_TOP)
    step = np.repeat(rng.uniform(0.25, 1.1, len(xs) // 12 + 1), 12)[:len(xs)]
    ends = (xs < -7.2) | (xs > 0.4)
    target[ends] -= step[ends]
    top_at = lambda x: float(target[min(len(xs) - 1, max(0, int(np.searchsorted(xs, x))))])
    door = lambda x, z: -6.4 < x < -5.0 and z < 1.9
    for _ in range(4000):
        if (sky >= target - 0.02).all():
            break
        i = int(np.argmin(sky - target + rng.uniform(0, 0.08, len(sky))))   # the column furthest below its top
        x = xs[i]
        w = float(np.clip(rng.lognormal(math.log(0.85), 0.35), 0.35, 1.7))
        a = max(x0 - 0.5, x - rng.uniform(0.05, 0.4) * w)
        b = min(x1 + (0.25 if rng.random() < 0.5 else 0.0), a + w)
        if b - a < 0.2:
            b = a + 0.2
        ia, ib = np.searchsorted(xs, a), np.searchsorted(xs, b)
        base = float(sky[ia:ib].max()) if ib > ia else float(sky[i])
        h = float(np.clip(rng.normal(0.46, 0.09), 0.26, 0.7))
        lim = top_at(0.5 * (a + b))
        h = min(h, max(lim - base, 0.0))
        if h <= 0.03:
            sky[ia:ib] = np.maximum(sky[ia:ib], target[ia:ib])        # this span is done
            continue
        sky[ia:ib] = base + h
        if h < 0.12 or door(0.5 * (a + b), base):
            continue
        if rng.random() < 0.05 and base > 0.3:                    # a missing block: the core shows through
            continue
        slump = rng.uniform(-0.04, 0.04) if rng.random() > 0.12 else rng.uniform(-0.15, 0.15)
        rot = Euler((rng.uniform(-0.05, 0.05), rng.uniform(-0.07, 0.07) + slump * 0.6, rng.uniform(-0.06, 0.06))).to_matrix()
        dz = rng.uniform(-0.03, 0.03)
        set_back = rng.uniform(-0.12, 0.10)                         # some stick out, some sit back
        dep = 0.55 + rng.uniform(-0.12, 0.2)
        gap = rng.uniform(0.03, 0.09)
        mb.box((0.5 * (a + b), y0 + set_back + dep / 2, base + h / 2 + dz),
               (b - a - gap, dep, h - rng.uniform(0.03, 0.08)), rot)
    # recessed core behind the front blocks (fills missing-block holes), its top follows the ragged skyline
    for k in range(0, len(xs) - 8, 8):
        zt = float(sky[k:k + 8].min()) - 0.08
        if zt > zb + 0.2 and not (-6.4 < xs[k] + 0.2 < -5.0):
            mb.box((xs[k] + 0.2, y0 + 0.55 + (depth - 0.55) / 2, (zb + zt) / 2), (0.42, depth - 0.6, zt - zb))
    # lintel over the doorway (cracked in two, one half dropped) + a broken pillar stump at the right end
    mb.box((-6.15, y0 + depth / 2, 2.05), (0.95, depth, 0.3), Euler((0, 0.04, 0.02)).to_matrix())
    mb.box((-5.2, y0 + depth / 2, 1.98), (0.95, depth, 0.3), Euler((0, -0.09, -0.03)).to_matrix())
    for k in range(3):
        mb.box((2.6 + rng.uniform(-0.05, 0.05), 1.0, -0.4 + 0.5 * k + 0.25),
               (0.8 - 0.07 * k, 0.8 - 0.06 * k, 0.48), Euler((0, 0, rng.uniform(-0.15, 0.15))).to_matrix())
    # fallen blocks at the foot
    for k in range(7):
        mb.box((rng.uniform(x0, x1), y0 - rng.uniform(0.3, 1.4), -0.45 + rng.uniform(0, 0.2)),
               (rng.uniform(0.35, 0.8), rng.uniform(0.3, 0.6), rng.uniform(0.25, 0.45)),
               Euler(tuple(rng.uniform(-0.4, 0.4, 3))).to_matrix())
    ob = mb.obj("Ledge", mat=proc_stone(), coll=coll)
    bev = ob.modifiers.new("Bevel", "BEVEL")
    bev.width = 0.045
    bev.segments = 2
    return ob


# ---------------------------------------------------------------- ivy (GN, static)

def ivy_tree(leaf_mat, density=22.0, seed=3, wall_bias=0.65, obj_scale=1.0):
    """leaves per m² = density × mask; obj_scale compensates a scaled object (local units)"""
    import bpy
    ng, nb, gi, go = X.gn_tree(f"Ivy_{seed}")
    pos = nb.n("GeometryNodeInputPosition").outputs[0]
    nrm = nb.n("GeometryNodeInputNormal").outputs[0]
    _, _, nz_ = nb.sep(nrm)
    tex = nb.n("ShaderNodeTexNoise")
    nb.set(tex.inputs["Scale"], 0.35)
    nb.set(tex.inputs["Detail"], 4.0)
    nb.link(pos, tex.inputs["Vector"])
    _, _, pz = nb.sep(pos)
    patch = nb.mr(tex.outputs["Fac"], 0.48, 0.62)
    # more ivy low on walls and draped from tops; less on upward faces in the middle
    mask = nb.m("MULTIPLY", patch, nb.m("ADD", wall_bias, nb.m("MULTIPLY", nb.m("ABSOLUTE", nz_), 1.0 - wall_bias)))
    dist = nb.n("GeometryNodeDistributePointsOnFaces", distribute_method="RANDOM")
    nb.link(gi.outputs[0], dist.inputs["Mesh"])
    nb.link(nb.m("MULTIPLY", mask, density * obj_scale ** 2), dist.inputs["Density"])
    nb.set(tex.inputs["Scale"], 0.35 * obj_scale)
    nb.set(dist.inputs["Seed"], seed)
    # leaf template: a 6-gon pinched into a leaf, normal +Z, lifted off the wall a bit
    circ = nb.n("GeometryNodeMeshCircle", fill_type="TRIANGLE_FAN")
    nb.set(circ.inputs["Vertices"], 6)
    nb.set(circ.inputs["Radius"], 1.0)
    tf = nb.n("GeometryNodeTransform")
    nb.link(circ.outputs[0], tf.inputs["Geometry"])
    k = 1.0 / obj_scale
    nb.set(tf.inputs["Scale"], (0.07 * k, 0.11 * k, k))
    nb.set(tf.inputs["Translation"], (0, 0.08 * k, 0.02 * k))
    smt = nb.n("GeometryNodeSetMaterial")
    nb.link(tf.outputs[0], smt.inputs["Geometry"])
    smt.inputs["Material"].default_value = leaf_mat
    rot = nb.n("FunctionNodeRotateRotation", rotation_space="LOCAL")
    nb.link(dist.outputs["Rotation"], rot.inputs["Rotation"])
    rr = nb.n("FunctionNodeRandomValue", data_type="FLOAT_VECTOR")
    nb.set(rr.inputs["Min"], (-0.5, -0.5, -3.14))
    nb.set(rr.inputs["Max"], (0.5, 0.5, 3.14))
    nb.set(rr.inputs["Seed"], seed + 1)
    er = nb.n("FunctionNodeEulerToRotation")
    nb.link(rr.outputs["Value"], er.inputs["Euler"])
    nb.link(er.outputs[0], rot.inputs["Rotate By"])
    rs = nb.n("FunctionNodeRandomValue", data_type="FLOAT")
    nb.set(rs.inputs["Min"], 0.6)
    nb.set(rs.inputs["Max"], 1.6)
    nb.set(rs.inputs["Seed"], seed + 2)
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(dist.outputs["Points"], inst.inputs["Points"])
    nb.link(smt.outputs[0], inst.inputs["Instance"])
    nb.link(rot.outputs[0], inst.inputs["Rotation"])
    nb.link(rs.outputs["Value"], inst.inputs["Scale"])
    jg = nb.n("GeometryNodeJoinGeometry")
    nb.link(inst.outputs[0], jg.inputs[0])
    nb.link(gi.outputs[0], jg.inputs[0])
    nb.link(jg.outputs[0], go.inputs[0])
    return ng


def add_ivy(objs, R, density=22.0, dense=("Ledge",), dense_k=2.6):
    leaf = silhouette_mat("IvyLeaf", rim_col=(0.6, 0.25, 0.85), rim=0.5)
    for i, o in enumerate(objs):
        if o.type != "MESH":
            continue
        k = dense_k if o.name.startswith(dense) else 1.0
        X.add_gn(o, ivy_tree(leaf, density=density * k, seed=11 + i, obj_scale=float(o.scale[0]),
                             wall_bias=0.85 if k > 1 else 0.65), name="Ivy")
    return leaf


# ---------------------------------------------------------------- foreground foliage (Quaternius CC0)

FOLIAGE = [  # (file, name, height m, [(x, y, z, rot°, scale)]) — v2: on the near bank (bottom-left) + the ledge foot;
    # nothing tall between the camera and the ledge face (v1's bushes there walled it off from the new camera)
    ("Fern_1", "Fern", 1.1, [(-4.2, -6.4, 0.2, 20, 1.2), (-6.6, -5.6, 0.3, 140, 1.5), (-2.6, -7.4, 0.1, 75, 1.0),
                             (-8.4, -7.0, 0.1, 210, 1.6), (-1.2, -6.2, 0.0, -30, 0.8), (-6.8, 0.0, 0.5, 60, 1.0),
                             (0.6, 0.0, 0.3, 200, 0.8)]),
    ("Plant_7_Big", "Plant7", 0.5, [(-5.5, -7.6, 0.2, 0, 1.3), (-8.8, -8.4, 0.1, 90, 1.5), (-3.0, -8.4, 0.0, 200, 1.0)]),
    ("Bush_Common", "Bush", 1.8, [(-11.0, -2.0, 0.6, 30, 1.6), (-12.5, 4.0, 0.6, 250, 1.8), (-9.4, -0.6, 0.6, 110, 1.0),
                                  (-7.0, -7.4, 0.2, 75, 0.85)]),
    ("Plant_1_Big", "Plant1", 1.3, [(-3.6, -6.0, 0.2, 160, 0.9), (-5.2, -0.4, 0.4, 20, 0.8)]),
    ("TwistedTree_3", "TwistTree", 9.0, [(-16.5, 1.5, 0.6, 40, 1.0)]),
]


def foliage(coll, R):
    """foreground silhouettes: kit meshes linked-duplicated, re-materialled to the violet silhouette shader (kit alpha
    textures kept via the leaf cutout: the base colour texture's alpha → Alpha)"""
    import bpy
    tmp = X.coll("_import_fol")
    mat = silhouette_mat("FoliageSil")
    obs = []
    for fn, nm, hgt, places in FOLIAGE:
        objs = IMP.import_model(f"{C.NATURE}/{fn}.gltf", coll=tmp)
        me = IMP.bake_world(objs, f"Fol_{nm}")
        IMP.normalize(me, height=hgt)
        # keep the kit's alpha cutout (leaf cards) by reusing its texture's alpha
        for i, m in enumerate(list(me.materials)):
            me.materials[i] = cutout_from(m, mat)
        for k, (x, y, z, r, s) in enumerate(places):
            obs.append(_place(me, f"Fol_{nm}_{k}", (x, y, z), rot_z=r, scale=s, coll=coll))
    _remove(list(tmp.objects))
    bpy.data.collections.remove(tmp)
    R.add("foliage.count", len(obs))
    return obs


_CUT = {}


def cutout_from(kit_mat, sil):
    """silhouette material + the kit texture's alpha (leaf cards) → new material per kit texture"""
    if kit_mat is None:
        return sil
    tex = next((n for n in kit_mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image), None)
    if tex is None:
        return sil
    key = tex.image.name
    if key in _CUT:
        return _CUT[key]
    m = sil.copy()
    m.name = f"FoliageSil_{key}"
    nt = m.node_tree
    nb = NB(nt)
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    shader = out.inputs["Surface"].links[0].from_socket
    t = nb.n("ShaderNodeTexImage")
    t.image = tex.image
    uv = nb.n("ShaderNodeUVMap")
    nb.link(uv.outputs[0], t.inputs["Vector"])
    nb.link(C.mix_shader(nb, t.outputs["Alpha"], C.transparent(nb), shader), out.inputs["Surface"])
    m.surface_render_method = "DITHERED"
    _CUT[key] = m
    return m
