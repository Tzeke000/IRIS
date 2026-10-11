"""SAMURAI v2: the downloaded characters and the retarget solver (bpy; imported by build_scene.py).

Two figures replace v1's hand-built mannequins (provenance + credits: ASSETS.md):

* RONIN = Renderpeople "Eric Rigged 001" (photoreal scan, 89-bone rig with fingers) wearing the
  "Japanese Clothes" haori + hakama + jinbei, plus v1's procedural kasa (bone-parented to the head).
* DEATH SAMURAI = "Death Samurai Character" (78 rigid armour parts, unrigged). We fit our own
  armature to it (body joints from mesh slices, finger chains from the glove), bone-parent the rigid
  plates (zero deform cost) and skin only the cloth/sleeve/glove meshes.

Everything is put in ONE canonical frame before rigging: armature object = identity, metres,
character faces +Y, +X = his right, feet on z = 0, hip joint over the origin. That is choreo's
"standing" frame, so the armature object takes the same per-frame location/yaw keys as v1's rig.

The solver (FigureSolver) is v1's Rig generalised to any skeleton: it reads each bone's REST joint
positions (head -> child head, never the glTF bone tails, which point anywhere), solves
  * spine/neck/head as world-delta rotations from the plan (twist preserved),
  * arms and legs with analytic two-bone IK on the asset's OWN lengths, oriented by hinge frames
    (upper arm + forearm share the elbow axis, so the elbow never candy-wraps),
  * the clavicle as a reach assist (shoulder protraction, <= 24 deg) before any stretch,
  * the hands from a calibrated GRIP: the fist's hole axis (fitted to the curled finger skin) is
    put on the hilt axis; the roll about the hilt is chosen per frame so the hand points away
    from its shoulder; forearm twist is shared with the twist bone,
  * fingers: a constant grip curl, blended to a relaxed curl by choreo's grip weight,
  * spring bones (skirt panels, hakama legs, sleeve bags, sode) integrated over the song,
and bakes quaternions with foreach_set (no constraints, no drivers, no handlers).
"""
import math
import os
import re

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

CHARS = "/data/iris_assets/characters/"


# =====================================================================================
# small numpy maths (vectorised over frames)
# =====================================================================================
def nrm(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def axis_angle(ax, ang):
    """Rotation matrices (N,3,3) about unit axes ax (N,3) by angles ang (N,)."""
    ax = nrm(ax)
    ang = np.asarray(ang, np.float64)
    x, y, z = ax[..., 0], ax[..., 1], ax[..., 2]
    c, s = np.cos(ang), np.sin(ang)
    t = 1 - c
    return np.stack([np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], -1),
                     np.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], -1),
                     np.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], -1)], -2)


def swing(a, b):
    """Minimal rotation(s) taking direction a to direction b (both (...,3))."""
    a, b = nrm(a), nrm(b)
    v = np.cross(a, b)
    c = np.sum(a * b, -1)
    vx = np.zeros(v.shape[:-1] + (3, 3))
    vx[..., 0, 1], vx[..., 0, 2] = -v[..., 2], v[..., 1]
    vx[..., 1, 0], vx[..., 1, 2] = v[..., 2], -v[..., 0]
    vx[..., 2, 0], vx[..., 2, 1] = -v[..., 1], v[..., 0]
    k = 1.0 / np.maximum(1 + c, 1e-6)
    R = np.eye(3) + vx + np.einsum("...ij,...jk->...ik", vx, vx) * k[..., None, None]
    flip = c < -0.9999
    if np.any(flip):
        aa = a[flip] if a.ndim > 1 else a[None]
        perp = nrm(np.cross(aa, np.where(np.abs(aa[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])))
        Rf = axis_angle(perp, np.full(len(perp), math.pi))
        if R.ndim == 2:
            R = Rf[0]
        else:
            R[flip] = Rf
    return R


def frame(y, z):
    """Orthonormal frames with columns X, Y = y, Z ~ z (N,3,3)."""
    y = nrm(y)
    z = nrm(z - np.sum(z * y, -1, keepdims=True) * y)
    return np.stack([np.cross(y, z), y, z], -1)


def mat2quat(m):
    m = np.asarray(m, np.float64)
    sh = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    w = np.sqrt(np.maximum(0, 1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])) / 2
    x = np.sqrt(np.maximum(0, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])) / 2
    y = np.sqrt(np.maximum(0, 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2])) / 2
    z = np.sqrt(np.maximum(0, 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2])) / 2
    x = np.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    y = np.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    z = np.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    q = np.stack([w, x, y, z], 1)
    return nrm(q).reshape(sh + (4,))


def quat2mat(q):
    q = nrm(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def qslerp_w(qa, qb, w):
    """Normalised lerp (sign-aligned) from qa to qb by weight w (N,)."""
    s = np.sign(np.sum(qa * qb, -1, keepdims=True))
    s[s == 0] = 1
    return nrm(qa * (1 - w)[:, None] + qb * s * w[:, None])


def qcontinuous(q):
    q = q.copy()
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    return q


def twist_angle(R, axis):
    """Signed twist (rad) of rotations R (N,3,3) about unit axes axis (N,3) (swing-twist split)."""
    q = mat2quat(R)
    p = np.sum(q[:, 1:] * axis, 1)
    return 2 * np.arctan2(p, q[:, 0])


def two_bone(S0, T, P, l1, l2, stretch_max=1.0):
    """Analytic 2-bone IK (vectorised). Returns (joint, end, stretch, miss)."""
    d = T - S0
    dist = np.linalg.norm(d, axis=1)
    L = l1 + l2
    st = np.clip(dist / (L * 0.9995), 1.0, stretch_max)
    dl = np.minimum(dist, L * 0.9995 * st)
    n = d / np.maximum(dist, 1e-9)[:, None]
    l1s, l2s = l1 * st, l2 * st
    a = (l1s ** 2 - l2s ** 2 + dl ** 2) / (2 * np.maximum(dl, 1e-9))
    h = np.sqrt(np.clip(l1s ** 2 - a ** 2, 0, None))
    p = P - S0
    p = nrm(p - np.sum(p * n, 1)[:, None] * n)
    E = S0 + n * a[:, None] + p * h[:, None]
    End = S0 + n * dl[:, None]
    return E, End, st, np.linalg.norm(End - T, axis=1)


# =====================================================================================
# import + canonical rest
# =====================================================================================
def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    for o in list(new):        # Sketchfab exports carry a stray 80-tri 'Icosphere' (light proxy)
        if o.type == "MESH" and o.name.startswith("Icosphere") and len(o.data.polygons) <= 80:
            new.remove(o)
            bpy.data.objects.remove(o)
    return new


def mesh_world_verts(o, evaluated=False):
    if evaluated:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            a = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", a)
        finally:
            ev.to_mesh_clear()
    else:
        me = o.data
        a = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", a)
    M = np.array(o.matrix_world)
    return a.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]


def bbox(objs):
    P = np.concatenate([mesh_world_verts(o) for o in objs if o.type == "MESH"])
    return P.min(0), P.max(0)


def normalise(objs, height=None, scale=None, rotz=0.0, name="ROOT"):
    """Parent import roots under an empty; turn by rotz deg; scale to height (m) or by scale; feet on
    z=0; xy centred on the bbox (exactly the research test's fit, so the garment fit carries over)."""
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    for o in objs:
        if o.parent is None:
            o.parent = root
    root.rotation_euler.z = math.radians(rotz)
    bpy.context.view_layer.update()
    lo, hi = bbox(objs)
    s = scale if scale is not None else height / (hi[2] - lo[2])
    root.scale = (s, s, s)
    root.location = (-(lo[0] + hi[0]) / 2 * s, -(lo[1] + hi[1]) / 2 * s, -lo[2] * s)
    bpy.context.view_layer.update()
    return root, s


def bake_world(objs, T=None):
    """Move every object's world transform (times T) into its data; objects end at identity with no
    parent. Empties are deleted. Returns (armatures, meshes)."""
    T = T if T is not None else Matrix.Identity(4)
    bpy.context.view_layer.update()
    mw = {o: o.matrix_world.copy() for o in objs}
    arms, meshes = [], []
    for o in objs:
        if o.type == "ARMATURE":
            o.data.transform(T @ mw[o])
            arms.append(o)
        elif o.type == "MESH":
            if o.data.users > 1:
                o.data = o.data.copy()
            o.data.transform(T @ mw[o])
            o.data.update()
            meshes.append(o)
    for o in arms + meshes:
        o.parent = None
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.rotation_mode = "XYZ"           # the glTF importer leaves QUATERNION (yaw keys would be ignored)
        o.matrix_world = Matrix.Identity(4)
        o.animation_data_clear()
    for o in objs:
        if o.type == "EMPTY":
            bpy.data.objects.remove(o)
    for a in arms:
        for pb in a.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
            pb.scale = (1, 1, 1)
            for c in list(pb.constraints):
                pb.constraints.remove(c)
    bpy.context.view_layer.update()
    return arms, meshes


def add_bones(arm, specs):
    """specs: name -> (head, tail, parent). Edit-mode add (roll 0)."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for n, (h, t, p) in specs.items():
        b = eb.new(n)
        b.head, b.tail = Vector(h), Vector(t)
        b.roll = 0.0
        if p:
            b.parent = eb[p]
    bpy.ops.object.mode_set(mode="OBJECT")
    for n in specs:
        pb = arm.pose.bones[n]
        pb.rotation_mode = "QUATERNION"


def resolve(arm, *cands):
    """Bone name by a suffix-free key (Sketchfab appends _<n> to every node)."""
    table = {}
    for b in arm.data.bones:
        table.setdefault(re.sub(r"_\d+$", "", b.name).lower(), b.name)
    for c in cands:
        if c.lower() in table:
            return table[c.lower()]
    raise KeyError(f"no bone among {cands}")


# =====================================================================================
# skinning helpers
# =====================================================================================
def seg_dist(P, a, c):
    d = c - a
    t = np.clip(((P - a) @ d) / max(float(d @ d), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)


def seg_weights(P, segs, sharp=4.0, top=3):
    names = list(segs)
    W = np.stack([1.0 / (seg_dist(P, *segs[n]) + 0.015) ** sharp for n in names], 1)
    if top < len(names):
        thr = -np.sort(-W, 1)[:, top - 1:top]
        W = np.where(W >= thr, W, 0)
    W /= W.sum(1, keepdims=True)
    return names, W


def mesh_edges(me):
    e = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def smooth_weights(me, W, iters=6, lam=0.5, edges=None):
    e = mesh_edges(me) if edges is None else edges
    n = len(W)
    for _ in range(iters):
        acc = np.zeros_like(W)
        cnt = np.zeros(n)
        np.add.at(acc, e[:, 0], W[e[:, 1]])
        np.add.at(acc, e[:, 1], W[e[:, 0]])
        np.add.at(cnt, e[:, 0], 1)
        np.add.at(cnt, e[:, 1], 1)
        nb = acc / np.maximum(cnt, 1)[:, None]
        W = np.where(cnt[:, None] > 0, W * (1 - lam) + nb * lam, W)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-9)


def write_weights(o, names, W, keep_top=4):
    """Replace o's vertex groups by names/W (rows renormalised, top-k per vertex)."""
    W = np.asarray(W, np.float64)
    if keep_top and W.shape[1] > keep_top:
        thr = -np.sort(-W, 1)[:, keep_top - 1:keep_top]
        W = np.where(W >= thr, W, 0)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    for vg in list(o.vertex_groups):
        o.vertex_groups.remove(vg)
    for k, n in enumerate(names):
        idx = np.nonzero(W[:, k] > 2e-3)[0]
        if len(idx) == 0:
            continue
        vg = o.vertex_groups.new(name=n)
        q = np.round(W[idx, k], 3)
        for w in np.unique(q):
            vg.add(idx[q == w].tolist(), float(w), "REPLACE")


def read_weights(o, names_order=None):
    """(vertex groups as dense (V, G) array, group names)."""
    names = [vg.name for vg in o.vertex_groups]
    V = len(o.data.vertices)
    W = np.zeros((V, len(names)))
    for v in o.data.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    return names, W


def armature_mod(o, arm, preserve_volume=False):
    for m in list(o.modifiers):
        if m.type == "ARMATURE":
            o.modifiers.remove(m)
    am = o.modifiers.new("Armature", "ARMATURE")
    am.object = arm
    am.use_deform_preserve_volume = preserve_volume
    if len(o.modifiers) > 1:       # armature first
        with bpy.context.temp_override(object=o):
            bpy.ops.object.modifier_move_to_index(modifier=am.name, index=0)
    return am


def bone_parent(o, arm, bone):
    """Rigid-attach o to a bone without moving it (rest pose); no deform cost."""
    bpy.context.view_layer.update()
    W = o.matrix_world.copy()
    b = arm.data.bones[bone]
    o.parent = arm
    o.parent_type = "BONE"
    o.parent_bone = bone
    o.matrix_parent_inverse = Matrix.Identity(4)
    P = arm.matrix_world @ b.matrix_local @ Matrix.Translation((0, b.length, 0))
    o.matrix_basis = P.inverted() @ W


def weld_index(P, tol=1e-5):
    """Map each vertex to a representative at the same position (split normals/UV seams)."""
    kd = KDTree(len(P))
    for i, p in enumerate(P):
        kd.insert(p, i)
    kd.balance()
    rep = np.arange(len(P))
    for i, p in enumerate(P):
        best = min(kd.find_range(p, tol), key=lambda r: r[1])[1]
        rep[i] = best
    return rep


def welded_edges(me, rep):
    e = mesh_edges(me)
    return np.stack([rep[e[:, 0]], rep[e[:, 1]]], 1)


def delete_covered(o, covers, max_dist=0.12, dirs=5, keep=None):
    """Delete o's faces whose vertices are hidden under the `covers` meshes: every probe ray from a vertex
    (its normal and `dirs`-1 tilted variants) hits a cover within max_dist. keep(P) -> bool mask of
    vertices that must stay (e.g. hands, the head). Removes poke-through and saves deform time."""
    import bmesh
    from mathutils.bvhtree import BVHTree
    trees = []
    for c in covers:
        P = mesh_world_verts(c)
        polys = [tuple(p.vertices) for p in c.data.polygons]
        trees.append(BVHTree.FromPolygons([tuple(v) for v in P], polys, all_triangles=False))
    me = o.data
    P = mesh_world_verts(o)
    Nn = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("normal", Nn)
    Nn = nrm(Nn.reshape(-1, 3).astype(np.float64))
    covered = np.ones(len(P), bool)
    rng = np.random.default_rng(1)
    tilts = [np.zeros(3)] + [rng.normal(size=3) * 0.55 for _ in range(dirs - 1)]
    for k in range(len(P)):
        for t in tilts:
            d = Vector(nrm(Nn[k] + t))
            hit = False
            for tr in trees:
                loc, n, idx, dist = tr.ray_cast(Vector(P[k] + Nn[k] * 0.002), d, max_dist)
                if loc is not None:
                    hit = True
                    break
            if not hit:
                covered[k] = False
                break
    if keep is not None:
        covered &= ~keep(P)
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        kill = [f for f in bm.faces if all(covered[v.index] for v in f.verts)]
        bmesh.ops.delete(bm, geom=kill, context="FACES_ONLY")
        loose = [v for v in bm.verts if not v.link_faces]
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()
    return int(covered.sum())


# =====================================================================================
# figure description shared by both characters
# =====================================================================================
class Figure:
    """arm: armature (identity transform, canonical frame). bmap: our joint name -> bone name.
    fingers[side] = [(b1, b2, b3, tip_rest_xyz), ...] thumb..pinky. springs: name -> spec dict.
    meshes: deforming/attached meshes. ankle_z: rest ankle-joint height."""

    def __init__(self, tag, arm, bmap, fingers, meshes):
        self.tag, self.arm, self.bmap, self.fingers, self.meshes = tag, arm, bmap, fingers, meshes
        self.springs = {}
        self.helpers = {}            # helper bone -> (our joint it follows, our joint it rests on, fraction)
        self.fore_twist = {}         # side -> (bone that takes the forearm twist, fraction)
        self.grip = {}
        self.curl = {}
        bones = arm.data.bones
        self.J = {k: np.array(bones[b].head_local) for k, b in bmap.items()}
        self.ankle_z = float(min(self.J["foot.R"][2], self.J["foot.L"][2]))
        self.hip_z = float(self.J["thigh.R"][2] + self.J["thigh.L"][2]) / 2
        self.leg_len = float(np.mean([np.linalg.norm(self.J[f"shin.{s}"] - self.J[f"thigh.{s}"])
                                      + np.linalg.norm(self.J[f"foot.{s}"] - self.J[f"shin.{s}"]) for s in "RL"]))

    def joint(self, name):
        return self.J[name]

    def summary(self):
        J = self.J
        out = [f"[fig {self.tag}] bones {len(self.arm.data.bones)}  hip z {self.hip_z:.3f}  ankle z {self.ankle_z:.3f}"
               f"  leg {self.leg_len:.3f}"]
        for s in "RL":
            l1 = np.linalg.norm(J[f"fore.{s}"] - J[f"upper.{s}"])
            l2 = np.linalg.norm(J[f"hand.{s}"] - J[f"fore.{s}"])
            out.append(f"[fig {self.tag}] arm.{s}: shoulder {np.round(J[f'upper.{s}'], 3)}  upper {l1:.3f}  fore {l2:.3f}"
                       f"  reach {l1 + l2:.3f}")
        tris = 0
        for o in self.meshes:
            tris += sum(len(p.vertices) - 2 for p in o.data.polygons)
        out.append(f"[fig {self.tag}] meshes {len(self.meshes)}  tris {tris}")
        return "\n".join(out)


# =====================================================================================
# RONIN: Renderpeople Eric + Japanese Clothes
# =====================================================================================
ERIC_HEIGHT = 1.80
JC_SCALE, JC_DZ = 0.0105, 0.06


def eric_map(arm):
    r = lambda *c: resolve(arm, *c)
    m = {"root": r("root"), "hips": r("hip"), "spine": r("spine_01"), "spine2": r("spine_02"), "chest": r("spine_03"),
         "neck": r("neck"), "head": r("head")}
    fingers = {}
    for s, sl in (("R", "r"), ("L", "l")):
        m.update({f"clav.{s}": r(f"shoulder_{sl}"), f"upper.{s}": r(f"upperarm_{sl}"), f"fore.{s}": r(f"lowerarm_{sl}"),
                  f"hand.{s}": r(f"hand_{sl}"), f"twist_fore.{s}": r(f"lowerarm_twist_{sl}"),
                  f"twist_upper.{s}": r(f"upperarm_twist_{sl}"),
                  f"thigh.{s}": r(f"upperleg_{sl}"), f"shin.{s}": r(f"lowerleg_{sl}"), f"foot.{s}": r(f"foot_{sl}"),
                  f"toe.{s}": r(f"ball_{sl}")})
        chains = []
        for fn in ("thumb", "index", "middle", "ring", "pinky"):
            chains.append([r(f"{fn}_0{k}_{sl}") for k in (1, 2, 3)])
        fingers[s] = chains
    return m, fingers


def finger_tips(arm, chains, frac=0.85):
    """Tip estimate per chain: distal joint + (distal - middle) * frac (the glTF '_end' bones are broken)."""
    out = []
    B = arm.data.bones
    for b1, b2, b3 in chains:
        p2, p3 = np.array(B[b2].head_local), np.array(B[b3].head_local)
        out.append((b1, b2, b3, p3 + (p3 - p2) * frac))
    return out


def load_ronin(log=print):
    objs = import_glb(CHARS + "sf_renderpeople_eric/model.glb")
    normalise(objs, height=ERIC_HEIGHT, rotz=180.0, name="ERIC_ROOT")
    g = import_glb(CHARS + "sf_japanese_clothes/model.glb")
    groot, _ = normalise(g, scale=JC_SCALE, rotz=180.0, name="JC_ROOT")
    groot.location.z += JC_DZ
    bpy.context.view_layer.update()
    arm0 = next(o for o in objs if o.type == "ARMATURE")
    hip = arm0.matrix_world @ arm0.data.bones[resolve(arm0, "hip")].head_local
    T = Matrix.Translation((-hip.x, -hip.y, 0.0))
    arms, meshes = bake_world(objs + g, T)
    arm = arms[0]
    arm.name = "RigA"
    arm.data.name = "RigA"
    body = next(o for o in meshes if any(m.type == "ARMATURE" for m in o.modifiers))
    body.name = "A_Body"
    gar = {}
    for o in meshes:
        if o is body:
            continue
        key = o.name.split("_")[0].lower()           # hakama / haori / jinbei
        o.name = "A_" + key.capitalize()
        gar[key] = o
    bmap, chains = eric_map(arm)
    fingers = {s: finger_tips(arm, chains[s]) for s in "RL"}
    fig = Figure("A", arm, bmap, fingers, [body] + list(gar.values()))
    fig.body, fig.gar = body, gar
    fig.fore_twist = {s: (bmap[f"twist_fore.{s}"], 0.6) for s in "RL"}
    # --- poke-through: drop what the outer layers fully cover
    # only the trunk and legs: sleeves swing (springs) and would expose cut edges on the arms
    keep_b = lambda P: ((np.abs(P[:, 0]) > 0.30) & (P[:, 2] > 0.95)) | (P[:, 2] > 1.50) | (P[:, 2] < 0.16)
    n1 = delete_covered(body, [gar["haori"], gar["hakama"], gar["jinbei"]], keep=keep_b)
    n2 = delete_covered(gar["jinbei"], [gar["haori"], gar["hakama"]], max_dist=0.08,
                        keep=lambda P: (np.abs(P[:, 0]) > 0.30) | ((P[:, 2] > 1.40) & (np.abs(P[:, 0]) < 0.12)))
    log(f"[fig A] covered verts removed: body {n1}, jinbei {n2}")
    for k in ("haori", "jinbei"):
        n3 = deflate_sleeves(fig, gar[k], keep=0.07 if k == "haori" else 0.06, factor=0.22 if k == "haori" else 0.40)
        log(f"[fig A] {k}: sleeve bag deflated toward the arm ({n3} verts)")
    inner_mask(fig, [gar["haori"], gar["jinbei"]])
    add_ronin_springs(fig)
    skin_ronin_garments(fig)
    for o in fig.meshes:
        armature_mod(o, arm, preserve_volume=True)
        o.parent = arm
    return fig


def arm_polyline_closest(P, pts):
    """Closest points on the polyline pts (k,3) for every row of P."""
    best_d = np.full(len(P), np.inf)
    best_q = np.zeros_like(P)
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        t = np.clip(((P - a) @ d) / float(d @ d), 0, 1)
        q = a + t[:, None] * d
        dd = np.linalg.norm(P - q, axis=1)
        m = dd < best_d
        best_d[m], best_q[m] = dd[m], q[m]
    return best_q, best_d


def deflate_sleeves(fig, g, keep=0.075, factor=0.40):
    """The haori's wide kimono sleeves hang as bags under the arm; skinned to a raised arm they turn into
    wings (no cloth sim here). Pull every sleeve vertex toward the arm line: radius r -> keep + (r - keep) *
    factor beyond `keep`. The sleeve stays loose, but a raised arm now carries it like a coat sleeve."""
    J = fig.J
    me = g.data
    P = mesh_world_verts(g)
    shoulder_x = float(abs(J["upper.R"][0]))
    moved = 0
    for s, sx in (("R", 1), ("L", -1)):
        sl = (P[:, 0] * sx > shoulder_x + 0.04) & (P[:, 2] > 1.0)
        if not sl.any():
            continue
        pts = np.array([J[f"upper.{s}"], J[f"fore.{s}"], J[f"hand.{s}"]])
        Q, d = arm_polyline_closest(P[sl], pts)
        f = np.clip((P[sl, 0] * sx - shoulder_x - 0.04) / 0.08, 0, 1)        # ease in from the shoulder seam
        r_new = np.where(d > keep, keep + (d - keep) * factor, d)
        r_new = d + (r_new - d) * f
        P[sl] = Q + (P[sl] - Q) * (r_new / np.maximum(d, 1e-9))[:, None]
        moved += int(sl.sum())
    me.vertices.foreach_set("co", P.astype(np.float32).ravel())
    me.update()
    return moved


def inner_mask(fig, covers, near=0.10):
    """Colour attribute 'inner' on the body: 1 where the scan's shirt sits under the haori/jinbei (seen
    through the kimono under-arm slits); the material darkens it to an inner layer."""
    from mathutils.bvhtree import BVHTree
    body = fig.body
    P = mesh_world_verts(body)
    trees = []
    for c in covers:
        Pc = mesh_world_verts(c)
        trees.append(BVHTree.FromPolygons([tuple(v) for v in Pc], [tuple(p.vertices) for p in c.data.polygons]))
    m = np.zeros(len(P))
    for k, p in enumerate(P):
        d = min((t.find_nearest(Vector(p))[3] or 9.0) for t in trees)
        m[k] = np.clip(1 - (d - near * 0.5) / (near * 0.5), 0, 1)
    m *= (P[:, 2] > 0.85) & (P[:, 2] < 1.52) & (np.abs(P[:, 0]) < 0.40)
    me = body.data
    ca = me.color_attributes.new("inner", "FLOAT_COLOR", "POINT")
    col = np.repeat(m[:, None], 4, 1).astype(np.float32)
    col[:, 3] = 1
    ca.data.foreach_set("color", col.ravel())
    return int((m > 0.5).sum())


def add_ronin_springs(fig):
    """Spring bones (heads in the canonical rest): 4 haori skirt panels on the hips, one hakama leg per
    thigh, one sleeve bag per forearm."""
    J = fig.J
    hips = fig.bmap["hips"]
    sp = {
        "spr_skirt.F": ((0, 0.12, 1.00), (0, 0.17, 0.62), hips),
        "spr_skirt.B": ((0, -0.12, 1.00), (0, -0.17, 0.62), hips),
        "spr_skirt.R": ((0.15, 0, 1.00), (0.22, 0, 0.62), hips),
        "spr_skirt.L": ((-0.15, 0, 1.00), (-0.22, 0, 0.62), hips),
    }
    for s, sx in (("R", 1), ("L", -1)):
        h = J[f"thigh.{s}"]
        sp[f"spr_hakama.{s}"] = (tuple(h), (h[0] + sx * 0.04, h[1], 0.10), hips)
        el, wr = J[f"fore.{s}"], J[f"hand.{s}"]
        a = el + (wr - el) * 0.35
        sp[f"spr_sleeve.{s}"] = (tuple(a), (a[0], a[1] - 0.02, a[2] - 0.30), fig.bmap[f"fore.{s}"])
        u0, u1 = J[f"upper.{s}"], J[f"fore.{s}"]
        sp[f"hlp_shoulder.{s}"] = (tuple(u0), tuple(u0 + (u1 - u0) * 0.5), fig.bmap[f"clav.{s}"])
        fig.helpers[f"hlp_shoulder.{s}"] = (f"upper.{s}", f"clav.{s}", 0.5)
    add_bones(fig.arm, sp)
    # stiffness, damping, max swing (rad), follow mode, gravity mix
    for s in ("F", "B", "R", "L"):
        fig.springs[f"spr_skirt.{s}"] = dict(k=0.10, damp=0.80, amax=0.9, follow=s, down=0.0)
    for s in "RL":
        fig.springs[f"spr_hakama.{s}"] = dict(k=0.24, damp=0.68, amax=0.45, follow="leg" + s, down=0.0)
        fig.springs[f"spr_sleeve.{s}"] = dict(k=0.09, damp=0.84, amax=2.6, follow=None, down=0.92)


def body_weights_for(fig, names):
    """Body vertex weights rolled up onto the listed bones (unlisted groups -> nearest listed ancestor)."""
    arm, body = fig.arm, fig.body
    col = {n: i for i, n in enumerate(names)}
    up = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}

    def listed(n):
        while n and n not in col:
            n = up.get(n)
        return n
    gnames, Wg = read_weights(body)
    BW = np.zeros((len(Wg), len(names)))
    for gi, gn in enumerate(gnames):
        t = listed(gn)
        if t is not None:
            BW[:, col[t]] += Wg[:, gi]
    BW /= np.maximum(BW.sum(1, keepdims=True), 1e-9)
    return BW


def skin_ronin_garments(fig):
    """Garment weights: nearest body vertex near the skin, bone-segment weights far from it; then
      * haori yoke: no upper-arm influence inside the shoulder line (fixes the bunching in jodan),
      * sleeves: only that side's clavicle/arm bones; the hanging bag goes to spr_sleeve,
      * haori skirt (below the waist): hips -> spr_skirt panels by azimuth,
      * hakama legs: hips -> spr_hakama.R/L by side (no cross-leg mixing: fixes the stripe swirl)."""
    m, arm = fig.bmap, fig.arm
    core = ["hips", "spine", "spine2", "chest", "neck", "head", "clav.R", "clav.L", "upper.R", "upper.L",
            "fore.R", "fore.L", "hand.R", "hand.L", "thigh.R", "thigh.L", "shin.R", "shin.L", "foot.R", "foot.L"]
    names = [m[k] for k in core]
    springs = [b for b in fig.springs] + list(fig.helpers)
    allnames = names + springs
    col = {n: i for i, n in enumerate(allnames)}
    J = fig.J
    child = {"hips": "spine", "spine": "spine2", "spine2": "chest", "chest": "neck", "neck": "head"}
    segs = {}
    for k in core:
        a = J[k]
        if k in child:
            b = J[child[k]]
        elif k == "head":
            b = a + np.array([0, 0, 0.18])
        elif k.startswith("clav"):
            b = J["upper" + k[4:]]
        elif k.startswith("upper"):
            b = J["fore" + k[5:]]
        elif k.startswith("fore"):
            b = J["hand" + k[4:]]
        elif k.startswith("hand"):
            b = a + nrm(a - J["fore" + k[4:]]) * 0.09
        elif k.startswith("thigh"):
            b = J["shin" + k[5:]]
        elif k.startswith("shin"):
            b = J["foot" + k[4:]]
        else:
            b = a + np.array([0, 0.12, -0.08])
        segs[m[k]] = (a, b)
    BP = mesh_world_verts(fig.body)
    BW = body_weights_for(fig, names)
    kd = KDTree(len(BP))
    for i, p in enumerate(BP):
        kd.insert(p, i)
    kd.balance()
    shoulder_x = float(abs(J["upper.R"][0]))
    waist_z = 1.00
    for key, g in fig.gar.items():
        P = mesh_world_verts(g)
        _, SW = seg_weights(P, segs)
        NW = np.zeros_like(SW)
        dist = np.zeros(len(P))
        for i, p in enumerate(P):
            co, idx, d = kd.find(p)
            NW[i] = BW[idx]
            dist[i] = d
        a = np.clip(1 - (dist - 0.03) / (0.12 - 0.03), 0, 1)[:, None]
        W = np.zeros((len(P), len(allnames)))
        W[:, :len(names)] = a * NW + (1 - a) * SW
        ax = np.abs(P[:, 0])
        if key in ("haori", "jinbei"):
            # sleeves: |x| beyond the shoulder joint and above the waist -> that side's arm chain only
            for s, sx in (("R", 1), ("L", -1)):
                sl = (P[:, 0] * sx > shoulder_x + 0.03) & (P[:, 2] > 1.02)
                if not sl.any():
                    continue
                arm_b = [m[f"clav.{s}"], m[f"upper.{s}"], m[f"fore.{s}"]]
                nm, SWa = seg_weights(P[sl], {b: segs[b] for b in arm_b}, top=2)
                W[sl] = 0
                for j, b in enumerate(nm):
                    W[np.nonzero(sl)[0], col[b]] = SWa[:, j]
                # near the shoulder seam the sleeve follows a half-way helper, not the full upper arm
                Su, Wu = J[f"upper.{s}"], J[f"hand.{s}"]
                tt = np.clip(((P[sl] - Su) @ (Wu - Su)) / float((Wu - Su) @ (Wu - Su)), 0, 1)
                hf = 1 - np.clip((tt - 0.12) / 0.40, 0, 1)
                rows = np.nonzero(sl)[0]
                cu, ch = col[m[f"upper.{s}"]], col[f"hlp_shoulder.{s}"]
                mv = W[rows, cu] * hf
                W[rows, cu] -= mv
                W[rows, ch] += mv
                if key == "haori":
                    # the bag: well below the arm line -> spring
                    d_arm = np.minimum(seg_dist(P[sl], *segs[m[f"upper.{s}"]]), seg_dist(P[sl], *segs[m[f"fore.{s}"]]))
                    below = np.clip((d_arm - 0.05) / 0.06, 0, 1) * 0.6
                    rows = np.nonzero(sl)[0]
                    W[rows] *= (1 - below)[:, None]
                    W[rows, col[f"spr_sleeve.{s}"]] += below
            # yoke: inside the shoulder line, upper arm -> clavicle (bunching fix)
            for s in "RL":
                u, c = col[m[f"upper.{s}"]], col[m[f"clav.{s}"]]
                fade = np.clip((shoulder_x + 0.03 - ax) / 0.08, 0, 1) * (P[:, 2] > 1.02)
                mv = W[:, u] * fade
                W[:, u] -= mv
                W[:, c] += mv
            if key == "haori":
                lo = P[:, 2] < waist_z
                t = np.clip((waist_z - P[:, 2]) / 0.22, 0, 1)[lo]
                az = np.arctan2(P[lo, 0], P[lo, 1])      # 0 = front (+Y), +pi/2 = right (+X)
                pw = {"F": np.cos(az), "R": np.sin(az), "B": -np.cos(az), "L": -np.sin(az)}
                pw = {k: np.clip(v, 0, None) ** 2 for k, v in pw.items()}
                tot = sum(pw.values())
                rows = np.nonzero(lo)[0]
                W[rows] *= (1 - t)[:, None]
                for k, v in pw.items():
                    W[rows, col[f"spr_skirt.{k}"]] += t * v / tot
        if key == "hakama":
            crotch = 0.80
            t = np.clip((crotch + 0.12 - P[:, 2]) / 0.30, 0, 1)        # 0 above the hips, 1 below the crotch
            side = np.clip(0.5 + P[:, 0] / 0.10, 0, 1)                  # 1 = right leg, 0 = left
            W *= (1 - t)[:, None]
            W[:, col["spr_hakama.R"]] += t * side
            W[:, col["spr_hakama.L"]] += t * (1 - side)
            # the waist band rides the hips/spine only
            for k in ("thigh.R", "thigh.L", "shin.R", "shin.L", "foot.R", "foot.L"):
                W[:, col[m[k]]] *= np.clip((1.05 - P[:, 2]) / 0.25, 0, 1)
        W = smooth_weights(g.data, W, iters=5)
        write_weights(g, allnames, W)




# =====================================================================================
# FINGERS: grip curl + the fist's hole axis (calibrated on the deformed skin)
# =====================================================================================
# per-joint flexion (deg) for index..pinky [MCP, PIP, DIP] and the thumb [CMC opposition, MCP, IP]
GRIP_CURL = {"fing": [(62, 82, 48), (66, 84, 46), (70, 84, 44), (74, 82, 42)], "thumb": (38, 18, 28)}
RELAX = 0.22                 # off-sword curl (fraction of the grip)


def hand_axes(fig, side):
    """Rest (armature-space) hand direction h (wrist -> middle MCP), knuckle line k (pinky -> index) and
    palm normal n (back of hand -> palm)."""
    B = fig.arm.data.bones
    ch = fig.fingers[side]
    wr = np.array(B[fig.bmap[f"hand.{side}"]].head_local)
    mid = np.array(B[ch[2][0]].head_local)
    idx, pin = np.array(B[ch[1][0]].head_local), np.array(B[ch[4][0]].head_local)
    h = nrm(mid - wr)
    k = nrm(idx - pin)
    k = nrm(k - (k @ h) * h)
    n = np.cross(h, k) * (-1.0 if side == "R" else 1.0)
    return wr, h, k, n


def curl_basis(fig, side, scale=1.0, thumb_scale=None):
    """bone -> local basis quaternion for the grip curl (scale 1) or a fraction of it."""
    thumb_scale = scale if thumb_scale is None else thumb_scale
    B = fig.arm.data.bones
    wr, h, k, n = hand_axes(fig, side)
    out = {}
    for fi, (b1, b2, b3, tip) in enumerate(fig.fingers[side]):
        pts = [np.array(B[b].head_local) for b in (b1, b2, b3)] + [tip]
        if fi == 0:
            angs = GRIP_CURL["thumb"]
        else:
            angs = GRIP_CURL["fing"][fi - 1]
        for j, b in enumerate((b1, b2, b3)):
            d = nrm(pts[j + 1] - pts[j])
            if fi == 0 and j == 0:
                ax = h if side == "R" else -h          # opposition: swing the thumb across the palm
                ax = nrm(ax - (ax @ d) * d)
                sc = thumb_scale
            elif fi == 0:
                ax = nrm(np.cross(d, n))
                sc = thumb_scale
            else:
                ax = nrm(np.cross(d, n))
                sc = scale
            Rr = np.array(B[b].matrix_local)[:3, :3]
            ax_l = Rr.T @ ax
            a = math.radians(angs[j]) * sc
            out[b] = np.array([math.cos(a / 2), *(math.sin(a / 2) * ax_l)])
    return out


def subtree(arm, root):
    out = []

    def rec(b):
        out.append(b.name)
        for c in b.children:
            rec(c)
    rec(arm.data.bones[root])
    return out


def calibrate_grip(fig, side, skin, scale=1.0, log=print):
    """Pose the grip curl on the rest rig, read the deformed hand skin and fit the largest empty
    tunnel through the fist (axis ~ the knuckle line). Returns hand-bone-LOCAL hole centre c, axis a
    (pinky -> index), hand direction h, clearance (m)."""
    arm = fig.arm
    hb = fig.bmap[f"hand.{side}"]
    q = curl_basis(fig, side, scale)
    for pb in arm.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
    for b, qq in q.items():
        arm.pose.bones[b].rotation_quaternion = tuple(qq)
    bpy.context.view_layer.update()
    P = mesh_world_verts(skin, evaluated=True)
    sub = set(subtree(arm, hb))
    gn, Wg = read_weights(skin)
    wsum = sum(Wg[:, j] for j, g in enumerate(gn) if g in sub) if any(g in sub for g in gn) else np.zeros(len(P))
    V = P[wsum > 0.5]
    for pb in arm.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
    bpy.context.view_layer.update()
    Bh = arm.data.bones[hb]
    Rr = np.array(Bh.matrix_local)[:3, :3]
    hr = np.array(Bh.head_local)
    U = (V - hr) @ Rr                      # hand-local coords
    B = arm.data.bones
    ch = fig.fingers[side]
    loc = lambda b: Rr.T @ (np.array(B[b].head_local) - hr)
    idx, pin, mid = loc(ch[1][0]), loc(ch[4][0]), loc(ch[2][0])
    wr, h, k, n = hand_axes(fig, side)
    k_l, n_l, h_l = Rr.T @ k, Rr.T @ n, Rr.T @ h
    best = None
    for t1 in np.radians(np.linspace(12, 30, 7)):          # diagonal across the palm (index end distal)
        for t2 in np.radians(np.linspace(-20, 20, 9)):
            a = nrm(k_l + math.tan(t1) * h_l + math.tan(t2) * n_l)
            e1 = nrm(np.cross(a, n_l))
            e2 = np.cross(a, e1)
            s = U @ a
            lo, hi = min(pin @ a, idx @ a) - 0.008, max(pin @ a, idx @ a) + 0.006
            sel = (s > lo) & (s < hi)
            X = np.stack([U[sel] @ e1, U[sel] @ e2], 1)
            c0 = (mid + n_l * 0.025 + h_l * 0.012)
            g0 = np.array([c0 @ e1, c0 @ e2])
            for dx in np.linspace(-0.035, 0.035, 15):
                for dy in np.linspace(-0.035, 0.035, 15):
                    g = g0 + (dx, dy)
                    r = X - g
                    dist = np.hypot(r[:, 0], r[:, 1])
                    near = dist < 0.045
                    if near.sum() < 20:
                        continue
                    ang = np.arctan2(r[near, 1], r[near, 0])
                    cover = len(np.unique(((ang + math.pi) / (2 * math.pi) * 16).astype(int) % 16))
                    if cover < 13:
                        continue
                    clr = float(dist.min())
                    if best is None or clr > best[0]:
                        cpt = a * ((lo + hi) / 2) + e1 * g[0] + e2 * g[1]
                        best = (clr, cpt, a, cover)
    if best is None:
        raise RuntimeError(f"grip calibration failed for {fig.tag}.{side}")
    clr, c, a, cover = best
    a = a if a @ (idx - pin) > 0 else -a
    log(f"[grip {fig.tag}.{side}] curl x{scale:.2f}: hole clearance {clr * 100:.2f} cm, coverage {cover}/16, "
        f"wrist->hole {np.linalg.norm(c) * 100:.1f} cm")
    return dict(c=c, a=a, h=Rr.T @ h, clear=clr, curl=q, scale=scale)


def fit_grip(fig, skins, target=0.0150, log=print):
    """Choose the curl scale whose fist hole is closest to `target` clearance (hilt ~1.25 x 1.65 cm).
    skins: side -> the mesh carrying that hand's skin."""
    for side in "RL":
        skin = skins[side]
        res = None
        for sc in (1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.60):
            try:
                g = calibrate_grip(fig, side, skin, sc, log=lambda *a: None)
            except RuntimeError:
                continue
            if res is None or abs(g["clear"] - target) < abs(res["clear"] - target):
                res = g
            if g["clear"] >= target:
                break
        if res is None:
            raise RuntimeError(f"no grip for {fig.tag}.{side}")
        log(f"[grip {fig.tag}.{side}] curl x{res['scale']:.2f}: hole clearance {res['clear'] * 100:.2f} cm, "
            f"wrist->hole {np.linalg.norm(res['c']) * 100:.1f} cm")
        fig.grip[side] = res
        fig.curl[side] = res["curl"]
        fig.relax = getattr(fig, "relax", {})
        fig.relax[side] = curl_basis(fig, side, res["scale"] * RELAX)


# =====================================================================================
# THE SOLVER
# =====================================================================================
PROTRACT_MAX = math.radians(24.0)
TWIST_FREE = math.radians(75.0)      # forearm twist beyond this (from the rest hand) is priced
TWIST_COST = 0.6


class FigureSolver:
    """Per-frame pose of one figure from choreo's plan (vectorised over all N frames)."""

    def __init__(self, fig, i, DU, C):
        self.fig, self.i, self.DU, self.C = fig, i, DU, C
        self.N = N = DU.n
        arm = fig.arm
        B = arm.data.bones
        self.Rr = {b.name: np.array(b.matrix_local)[:3, :3] for b in B}
        self.hr = {b.name: np.array(b.head_local) for b in B}
        self.par = {b.name: (b.parent.name if b.parent else None) for b in B}
        self.P, self.H, self.basis = {}, {}, {}
        m = fig.bmap
        self.m = m
        # vertical scale: this figure's hip-over-ankle height vs the plan skeleton's
        plan_h = C.BONES["thigh.R"][0][2] - C.ANKLE_Z
        self.kz = (fig.hip_z - fig.ankle_z) / plan_h
        self.loc = np.concatenate([DU.root[i], (self.kz * (DU.pelvis[i] - C.PELVIS_REST))[:, None]], 1)
        self.yaw = DU.yaw[i]
        self.Rz = self._rotz(self.yaw)
        self.report = {}

    @staticmethod
    def _rotz(a):
        c, s = np.cos(a), np.sin(a)
        z, o = np.zeros_like(a), np.ones_like(a)
        return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)

    # ---- conventions ------------------------------------------------------------------
    def to_arm(self, pw):
        return np.einsum("nji,nj->ni", self.Rz, pw - self.loc)

    def rot_to_arm(self, Rw):
        return np.einsum("nji,njk->nik", self.Rz, Rw)

    def ensure(self, b):
        if b not in self.P:
            self.P[b] = self.carried(b)
            self.H[b] = self.head_of(b)

    def carried(self, b):
        p = self.par[b]
        if p is None:
            return np.repeat(self.Rr[b][None], self.N, 0)
        self.ensure(p)
        return np.einsum("nij,jk->nik", self.P[p], self.Rr[p].T @ self.Rr[b])

    def head_of(self, b):
        p = self.par[b]
        if p is None:
            return np.repeat(self.hr[b][None], self.N, 0)
        self.ensure(p)
        return self.H[p] + np.einsum("nij,j->ni", self.P[p], self.Rr[p].T @ (self.hr[b] - self.hr[p]))

    def point(self, b, x_rest):
        """Posed armature-space position of a rest point rigidly attached to bone b."""
        self.ensure(b)
        return self.H[b] + np.einsum("nij,j->ni", self.P[b], self.Rr[b].T @ (np.asarray(x_rest) - self.hr[b]))

    def set(self, b, W):
        Cm = self.carried(b)
        self.H[b] = self.head_of(b)
        self.P[b] = W
        self.basis[b] = np.einsum("nji,njk->nik", Cm, W)

    def delta(self, b, R):
        """Absolute rest->posed armature rotation R (N,3,3) for bone b."""
        self.set(b, np.einsum("nij,jk->nik", R, self.Rr[b]))

    # ---- solve ------------------------------------------------------------------------
    @staticmethod
    def _smooth_need(x, grow=4, sigma=2.5):
        """Spread a per-frame requirement a few frames each way (min-filter of -x), then Gaussian."""
        y = x.copy()
        for k in range(1, grow + 1):
            y[:-k] = np.maximum(y[:-k], x[k:])
            y[k:] = np.maximum(y[k:], x[:-k])
        r = int(3 * sigma)
        g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        g /= g.sum()
        return np.convolve(np.pad(y, r, mode="edge"), g, mode="valid")

    def _pelvis_assist(self):
        """Drop the pelvis where a planted foot is out of this body's leg reach (fast steps, the charge)."""
        DU, C, fig, N, i = self.DU, self.C, self.fig, self.N, self.i
        need = np.zeros(N)
        for side, si in (("R", 0), ("L", 1)):
            hip = fig.J[f"thigh.{side}"]
            L = (np.linalg.norm(fig.J[f"shin.{side}"] - hip) + np.linalg.norm(fig.J[f"foot.{side}"] - fig.J[f"shin.{side}"])) * 0.995
            Aw = DU.ankle[i][si].copy()
            Aw[:, 2] = fig.ankle_z + self.kz * (Aw[:, 2] - C.ANKLE_Z)
            T = self.to_arm(Aw)
            dxy = np.linalg.norm(T[:, :2] - hip[:2], axis=1)
            zmax = T[:, 2] + np.sqrt(np.clip(L ** 2 - dxy ** 2, 0, None))     # highest hip that still reaches
            need = np.maximum(need, hip[2] - zmax)
        dz = np.clip(self._smooth_need(np.clip(need, 0, 0.18)), 0, 0.18)
        self.loc[:, 2] -= dz
        self.report["pelvis_drop"] = dz

    def _yaw_assist(self, Rsc):
        """Extra chest yaw (about the spine2/chest pivot) where a gripping hand is out of reach even with
        protraction: the iai draw and the noto reach across the body like a real draw does."""
        DU, C, m, fig, N, i = self.DU, self.C, self.m, self.fig, self.N, self.i
        piv = fig.J["spine2"] if "spine2" in m else fig.J["chest"]
        alphas = np.radians(np.linspace(-40, 40, 33))
        deficit = np.zeros((len(alphas), N))
        for side, si in (("R", 0), ("L", 1)):
            gw = DU.grip_w[i][si]
            if not (gw > 0).any():
                continue
            _, Hg = self.grip_targets(side)
            sh_rest = fig.J[f"upper.{side}"]
            l1 = np.linalg.norm(fig.J[f"fore.{side}"] - sh_rest)
            l2 = np.linalg.norm(fig.J[f"hand.{side}"] - fig.J[f"fore.{side}"])
            cap = (l1 + l2) * 0.995 + 0.045
            S_base = piv + np.einsum("nij,j->ni", Rsc, sh_rest - piv)
            for a_i, a in enumerate(alphas):
                c, s_ = math.cos(a), math.sin(a)
                R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
                Sa = piv + (S_base - piv) @ R.T
                deficit[a_i] = np.maximum(deficit[a_i], (np.linalg.norm(Hg - Sa, axis=1) - cap) * (gw > 0.5))
        best = np.argmin(deficit + 0.002 * np.abs(alphas)[:, None], 0)
        a = np.where(deficit[len(alphas) // 2] > 0, alphas[best], 0.0)
        a = np.sign(a) * self._smooth_need(np.abs(a), grow=5, sigma=3.0)
        self.report["yaw_assist"] = a
        c, s_ = np.cos(a), np.sin(a)
        z, o = np.zeros_like(a), np.ones_like(a)
        Ra = np.stack([np.stack([c, -s_, z], -1), np.stack([s_, c, z], -1), np.stack([z, z, o], -1)], -2)
        return Ra

    def solve(self):
        DU, C, m, fig, i, N = self.DU, self.C, self.m, self.fig, self.i, self.N
        Rs, Rc, ch_h, shR, shL, nk = DU.fk[i]
        self._pelvis_assist()
        self.delta(m["hips"], np.repeat(np.eye(3)[None], N, 0))
        self.delta(m["spine"], Rs)
        Rsc = np.einsum("nij,njk->nik", Rs, Rc)
        Rsc = np.einsum("nij,njk->nik", self._yaw_assist(Rsc), Rsc)
        for b in [b for b in self.P if b not in self.basis]:      # drop parents cached before the chest moved
            del self.P[b], self.H[b]
        if "spine2" in m:
            self.delta(m["spine2"], Rsc)
            self.ensure(m["chest"])
        else:
            self.delta(m["chest"], Rsc)
        # look at the opponent's head: neck 40 %, head the rest (v1)
        o = 1 - i
        opp = self.to_arm(np.concatenate([DU.root[o], (DU.pelvis[o] + 0.62)[:, None]], 1))
        hp = self.head_of(m["neck"]) + np.array([0, 0, 0.2])
        fwd_c = np.einsum("nij,j->ni", Rsc, np.array([0, 1.0, 0]))
        want = nrm(opp - hp)
        want[:, 2] = np.clip(want[:, 2], -0.35, 0.35)
        Q = swing(fwd_c, nrm(want))
        q = mat2quat(Q)
        Rn = quat2mat(nrm(q * 0.4 + np.array([1.0, 0, 0, 0]) * 0.6))
        self.delta(m["neck"], np.einsum("nij,njk->nik", Rn, Rsc))
        self.delta(m["head"], np.einsum("nij,njk->nik", Q, Rsc))
        self._arms()
        for hb, (follow, base, frac) in self.fig.helpers.items():
            Ru = np.einsum("nij,kj->nik", self.P[m[follow]], self.Rr[m[follow]])
            self.ensure(m[base])
            Rb = np.einsum("nij,kj->nik", self.P[m[base]], self.Rr[m[base]])
            Rh = quat2mat(qslerp_w(mat2quat(Rb), mat2quat(Ru), np.full(N, frac)))
            self.delta(hb, Rh)
        self._legs()
        self._fingers()
        self._springs()

    def _hinge(self, up, lo, end, S0, E, End, pole_n):
        """Upper/lower absolute rotations from rest joint vectors to posed ones sharing a hinge axis."""
        J = self.fig.J
        u_r, f_r = J[lo] - J[up], J[end] - J[lo]
        b_r = np.cross(u_r, f_r)
        if np.linalg.norm(b_r) < 1e-3 * np.linalg.norm(u_r) * np.linalg.norm(f_r):
            raise RuntimeError(f"rest {up}/{lo} is straight: no hinge axis")
        b_r = nrm(b_r)
        u_p, f_p = E - S0, End - E
        b_p = nrm(np.cross(u_p, f_p) + 1e-4 * pole_n)
        Fr_u = frame(u_r[None], b_r[None])[0]
        Fr_f = frame(f_r[None], b_r[None])[0]
        Ru = np.einsum("nij,kj->nik", frame(u_p, b_p), Fr_u)
        Rf = np.einsum("nij,kj->nik", frame(f_p, b_p), Fr_f)
        return Ru, Rf

    def grip_targets(self, side, E=None):
        """Hand-bone posed rotation W (N,3,3) and wrist (hand head) target, armature space. The roll
        about the hilt: without E the hand points away from its shoulder; with an elbow estimate E the
        hand direction is aligned with the forearm (least wrist bend)."""
        DU, C, i = self.DU, self.C, self.i
        g = self.fig.grip[side]
        Ps = self.to_arm(DU.sword_pos[i])
        Rsa = self.rot_to_arm(DU.sword_rot[i])
        gy = C.GRIP_R_Y if side == "R" else C.GRIP_L_Y
        G = Ps + Rsa[:, :, 1] * gy
        Q0 = swing(g["a"], np.array([0.0, 1.0, 0.0]))
        qh = Q0 @ g["h"]
        sh = self.head_of(self.m[f"upper.{side}"])
        d_s = np.einsum("nji,nj->ni", Rsa, nrm(G - sh))
        # roll about the hilt: the hand points away from its shoulder (forearm-like)
        psis = np.linspace(-math.pi, math.pi, 73)[:-1]
        best = np.full(self.N, -9.0)
        psi = np.zeros(self.N)
        cl = Q0 @ g["c"]
        if E is not None:
            # forearm frame of this iteration (hinge), to price the forearm twist each roll would need
            up = self.m[f"upper.{side}"]
            S0 = self.head_of(up)
            pole = nrm(np.cross(nrm(G - S0), np.array([0, 0, 1.0])) + 1e-6)
            _, Rf = self._hinge(f"upper.{side}", f"fore.{side}", f"hand.{side}", S0, E, G, pole)
            Wfree = np.einsum("nij,jk->nik", Rf, self.Rr[self.m[f"hand.{side}"]])
            fdir_ = nrm(G - E)
        for p_ in psis:
            c, s = math.cos(p_), math.sin(p_)
            hx, hz = qh[0] * c + qh[2] * s, -qh[0] * s + qh[2] * c
            if E is None:
                sc = hx * d_s[:, 0] + qh[1] * d_s[:, 1] + hz * d_s[:, 2]
            else:
                cx, cz = cl[0] * c + cl[2] * s, -cl[0] * s + cl[2] * c
                Hs = -np.array([cx, cl[1], cz])                       # wrist in sword-local, from the hole
                Hw = G + np.einsum("nij,j->ni", Rsa, Hs)
                fd = np.einsum("nji,nj->ni", Rsa, nrm(Hw - E))       # forearm dir, sword-local
                sc = hx * fd[:, 0] + qh[1] * fd[:, 1] + hz * fd[:, 2]
                Ry1 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                Wp = np.einsum("nij,jk->nik", Rsa, Ry1 @ Q0)
                phi = np.abs(twist_angle(np.einsum("nij,nkj->nik", Wp, Wfree), fdir_))
                sc = sc - TWIST_COST * np.clip((phi - TWIST_FREE) / math.radians(60), 0, None) ** 2
            better = sc > best
            best[better] = sc[better]
            psi[better] = p_
        psi = np.unwrap(psi)
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        k /= k.sum()
        psi = np.convolve(np.pad(psi, 6, mode="edge"), k, mode="valid")
        c, s = np.cos(psi), np.sin(psi)
        z, o = np.zeros_like(psi), np.ones_like(psi)
        Ry = np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)
        W = np.einsum("nij,njk,kl->nil", Rsa, Ry, Q0)
        Hh = G - np.einsum("nij,j->ni", W, g["c"])
        self.report[f"grip_align.{side}"] = best
        return W, Hh

    def _arms(self):
        DU, C, m, fig, i, N = self.DU, self.C, self.m, self.fig, self.i, self.N
        self.miss = {}
        for side, si in (("R", 0), ("L", 1)):
            up, fo, ha = m[f"upper.{side}"], m[f"fore.{side}"], m[f"hand.{side}"]
            gw = DU.grip_w[i][si]
            free = self.to_arm(DU.wrist_free[i][si])
            Pp = self.to_arm(DU.elbow_pole[i][si])
            E = None
            for it in range(3):                     # roll <-> elbow: alternate (converges in 2-3)
                Wg, Hg = self.grip_targets(side, E)
                T = Hg * gw[:, None] + free * (1 - gw)[:, None]
                l1_ = float(np.linalg.norm(fig.J[f"fore.{side}"] - fig.J[f"upper.{side}"]))
                l2_ = float(np.linalg.norm(fig.J[f"hand.{side}"] - fig.J[f"fore.{side}"]))
                E = two_bone(self.head_of(up), T, Pp, l1_, l2_, 1.0)[0]
            l1 = float(np.linalg.norm(fig.J[f"fore.{side}"] - fig.J[f"upper.{side}"]))
            l2 = float(np.linalg.norm(fig.J[f"hand.{side}"] - fig.J[f"fore.{side}"]))
            reach = (l1 + l2) * 0.995
            cl = m.get(f"clav.{side}")
            if cl:
                S0 = self.head_of(up)
                Cc = self.head_of(cl)
                Wc = self.carried(cl)
                need = np.linalg.norm(T - S0, axis=1) - reach
                Sstar = T - nrm(T - S0) * reach
                v0, v1 = S0 - Cc, Sstar - Cc
                ang = np.arccos(np.clip(np.sum(nrm(v0) * nrm(v1), 1), -1, 1))
                frac = np.where(need > 0, np.minimum(1.0, PROTRACT_MAX / np.maximum(ang, 1e-6)), 0.0)
                ax = nrm(np.cross(v0, v1) + 1e-9)
                Rcl = axis_angle(ax, ang * frac)
                self.set(cl, np.einsum("nij,njk->nik", Rcl, Wc))
                self.report[f"protract.{side}"] = np.linalg.norm(self.head_of(up) - S0, axis=1)
            S0 = self.head_of(up)
            E, End, st, miss = two_bone(S0, T, Pp, l1, l2, 1.0)
            n_ = nrm(T - S0)
            pp = nrm((Pp - S0) - np.sum((Pp - S0) * n_, 1)[:, None] * n_)
            Ru, Rf = self._hinge(f"upper.{side}", f"fore.{side}", f"hand.{side}", S0, E, End, np.cross(pp, n_))
            self.delta(up, Ru)
            Wfo = np.einsum("nij,jk->nik", Rf, self.Rr[fo])
            # hand: grip orientation, blended toward the forearm-carried one off the sword
            Wfree = np.einsum("nij,jk->nik", Rf, self.Rr[ha])
            Wh = quat2mat(qslerp_w(mat2quat(Wfree), mat2quat(Wg), gw))
            # forearm twist shared with the twist bone (or the forearm itself)
            fdir = nrm(End - E)
            Dl = np.einsum("nij,nkj->nik", Wh, Wfree)
            phi = np.clip(twist_angle(Dl, fdir), -2.6, 2.6)
            tb, frac = fig.fore_twist.get(side, (fo, 0.6))
            if tb == fo:
                Wfo = np.einsum("nij,njk->nik", axis_angle(fdir, phi * frac), Wfo)
                self.set(fo, Wfo)
            else:
                self.set(fo, Wfo)
                self.set(tb, np.einsum("nij,njk->nik", axis_angle(fdir, phi * frac), self.carried(tb)))
            self.set(ha, Wh)
            self.miss[side] = np.linalg.norm(self.head_of(ha) - T, axis=1)
            self.report[f"twist.{side}"] = phi
            wb = np.einsum("nji,njk->nik", np.einsum("nij,njk->nik", axis_angle(fdir, phi), Wfree), Wh)
            self.report[f"wrist_bend.{side}"] = np.degrees(np.arccos(np.clip((np.trace(wb, axis1=1, axis2=2) - 1) / 2, -1, 1)))
            self.report[f"hand_miss.{side}"] = self.miss[side]
            self.report[f"grip_w.{side}"] = gw

    def _legs(self):
        DU, C, m, fig, i, N = self.DU, self.C, self.m, self.fig, self.i, self.N
        for side, si in (("R", 0), ("L", 1)):
            th, sh, ft = m[f"thigh.{side}"], m[f"shin.{side}"], m[f"foot.{side}"]
            S0 = self.head_of(th)
            Aw = DU.ankle[i][si].copy()
            Aw[:, 2] = fig.ankle_z + self.kz * (Aw[:, 2] - C.ANKLE_Z)
            T = self.to_arm(Aw)
            yaw_rel = DU.foot_yaw[i][si] - DU.yaw[i]
            fwd = np.stack([-np.sin(yaw_rel), np.cos(yaw_rel), np.zeros(N)], 1)
            Pp = T + fwd * 0.8 + np.array([0.0, 0.0, 0.45]) + np.array([0.12 if side == "R" else -0.12, 0, 0])
            l1 = float(np.linalg.norm(fig.J[f"shin.{side}"] - fig.J[f"thigh.{side}"]))
            l2 = float(np.linalg.norm(fig.J[f"foot.{side}"] - fig.J[f"shin.{side}"]))
            E, End, st, miss = two_bone(S0, T, Pp, l1, l2, 1.0)
            n_ = nrm(T - S0)
            pp = nrm((Pp - S0) - np.sum((Pp - S0) * n_, 1)[:, None] * n_)
            Ru, Rf = self._hinge(f"thigh.{side}", f"shin.{side}", f"foot.{side}", S0, E, End, np.cross(pp, n_))
            self.delta(th, Ru)
            self.delta(sh, Rf)
            Rft = np.einsum("nij,njk->nik", self._rotz(yaw_rel), self._rotx(DU.foot_pitch[i][si]))
            self.delta(ft, Rft)
            self.report[f"foot_miss.{side}"] = miss

    @staticmethod
    def _rotx(a):
        c, s = np.cos(a), np.sin(a)
        z, o = np.zeros_like(a), np.ones_like(a)
        return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)

    def _fingers(self):
        DU, i, fig = self.DU, self.i, self.fig
        for side, si in (("R", 0), ("L", 1)):
            gw = DU.grip_w[i][si]
            for b, qg in fig.curl[side].items():
                qr = fig.relax[side][b]
                q = qslerp_w(np.repeat(qr[None], self.N, 0), np.repeat(qg[None], self.N, 0), gw)
                Wb = np.einsum("nij,njk->nik", self.carried(b), quat2mat(q))
                self.set(b, Wb)

    def _springs(self):
        """Secondary motion (v1): damped springs in world space; follow modes lean the rest direction
        toward the thighs/legs; 'down' mixes in gravity (sleeve bags fall toward the shoulder in jodan)."""
        DU, C, m, i, N = self.DU, self.C, self.m, self.i, self.N
        Rw = self.Rz
        loc = self.loc
        thigh = {s: nrm(self.H[m[f"shin.{s}"]] - self.H[m[f"thigh.{s}"]]) for s in "RL"}
        leg = {s: nrm(self.H[m[f"foot.{s}"]] - self.H[m[f"thigh.{s}"]]) for s in "RL"}
        down_a = np.einsum("nji,j->ni", Rw, np.array([0, 0, -1.0]))
        for b, sp in self.fig.springs.items():
            Cm = self.carried(b)
            head = self.head_of(b)
            rest_dir = Cm[:, :, 1].copy()
            f = sp.get("follow")
            if f in ("R", "L"):
                rest_dir = nrm(rest_dir * 0.35 + thigh[f] * 0.65)
            elif f in ("F", "B"):
                sg = 1 if f == "F" else -1
                pick = np.where((thigh["R"][:, 1] * sg > thigh["L"][:, 1] * sg)[:, None], thigh["R"], thigh["L"])
                rest_dir = nrm(rest_dir * 0.35 + pick * 0.65)
            elif f in ("legR", "legL"):
                rest_dir = nrm(rest_dir * 0.06 + leg[f[-1]] * 0.94)
            elif f in ("armR", "armL"):
                ua = nrm(self.H[m[f"fore.{f[-1]}"]] - self.H[m[f"upper.{f[-1]}"]])
                rest_dir = nrm(rest_dir * 0.55 + ua * 0.45)
            if sp.get("down", 0) > 0:
                rest_dir = nrm(rest_dir * (1 - sp["down"]) + down_a * sp["down"])
            L = float(self.fig.arm.data.bones[b].length)
            hw = np.einsum("nij,nj->ni", Rw, head) + loc
            tw = hw + np.einsum("nij,nj->ni", Rw, rest_dir) * L
            x = tw[0].copy()
            v = np.zeros(3)
            out = np.empty((N, 3))
            g = np.array([0, 0, -0.0035])
            k, damp = sp["k"], sp["damp"]
            for t in range(N):
                v = v * damp + (tw[t] - x) * k + g
                x = x + v
                d = x - hw[t]
                x = hw[t] + d / max(np.linalg.norm(d), 1e-9) * L
                out[t] = x
            dir_a = np.einsum("nji,nj->ni", Rw, nrm(out - hw))
            cosang = np.sum(dir_a * rest_dir, 1)
            ang = np.arccos(np.clip(cosang, -1, 1))
            over = ang > sp["amax"]
            if np.any(over):
                ax = nrm(np.cross(rest_dir[over], dir_a[over]))
                Rl = axis_angle(ax, np.full(over.sum(), sp["amax"]))
                dir_a[over] = np.einsum("nij,nj->ni", Rl, rest_dir[over])
            self.set(b, np.einsum("nij,njk->nik", swing(Cm[:, :, 1], dir_a), Cm))

    # ---- bake -------------------------------------------------------------------------
    def bake(self, bake_fn, bake_vec_fn):
        arm = self.fig.arm
        for b, Bm in self.basis.items():
            q = qcontinuous(mat2quat(Bm))
            for k in range(4):
                bake_fn(arm, f'pose.bones["{b}"].rotation_quaternion', q[:, k], index=k)
        bake_vec_fn(arm, "location", self.loc)
        bake_fn(arm, "rotation_euler", self.yaw, index=2)

    def stats(self):
        r = self.report
        out = []
        for s in "RL":
            g = r[f"grip_w.{s}"] > 0.99
            if g.any():
                hm = r[f"hand_miss.{s}"][g] * 100
                wb = r[f"wrist_bend.{s}"][g]
                out.append(f"[solve {self.fig.tag}] hand.{s} on hilt: miss mean {hm.mean():.2f} cm, p99 "
                           f"{np.percentile(hm, 99):.2f}, max {hm.max():.2f} cm | wrist bend median "
                           f"{np.median(wb):.0f} deg, p95 {np.percentile(wb, 95):.0f} | twist p95 "
                           f"{np.degrees(np.percentile(np.abs(r[f'twist.{s}'][g]), 95)):.0f} deg"
                           + (f" | protraction max {r[f'protract.{s}'].max() * 100:.1f} cm" if f"protract.{s}" in r else ""))
            fm = r[f"foot_miss.{s}"] * 100
            out.append(f"[solve {self.fig.tag}] foot.{s}: plant miss mean {fm.mean():.2f} cm, max {fm.max():.2f} cm")
        return "\n".join(out)


# =====================================================================================
# DEATH SAMURAI: our armature fitted to the armour; plates bone-parented, cloth skinned
# =====================================================================================
DS_SCALE = 1.0            # natural units: kabuto crest at 1.95 m, shoulders 1.43, hips 0.95
DS_SHIFT = (0.0, 0.02, 0.0)
# joints measured from mesh slices (dev/ds_fit.py: arm-tube centres, pants leg centres, feet, skull);
# right side (+X), mirrored. Elbow sits behind the shoulder->wrist line so the rest arm has a hinge.
DS_J = {
    "hips": ((0, 0.0, 0.95), (0, -0.01, 1.09)), "spine": ((0, -0.01, 1.09), (0, -0.01, 1.23)),
    "chest": ((0, -0.01, 1.23), (0, 0.01, 1.50)), "neck": ((0, 0.01, 1.50), (0, 0.02, 1.60)),
    "head": ((0, 0.02, 1.60), (0, 0.02, 1.82)),
    "clav.R": ((0.03, 0.0, 1.45), (0.175, -0.005, 1.425)),
    "upper.R": ((0.175, -0.005, 1.425), (0.385, -0.045, 1.31)),
    "fore.R": ((0.385, -0.045, 1.31), (0.552, 0.03, 1.195)),
    "hand.R": ((0.552, 0.03, 1.195), (0.60, 0.06, 1.13)),
    "thigh.R": ((0.10, 0.0, 0.93), (0.155, 0.02, 0.50)),
    "shin.R": ((0.155, 0.02, 0.50), (0.200, -0.03, 0.105)),
    "foot.R": ((0.200, -0.03, 0.105), (0.225, 0.13, 0.03)),
}
DS_PARENT = {"hips": None, "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
             "clav": "chest", "upper": "clav", "fore": "upper", "hand": "fore",
             "thigh": "hips", "shin": "thigh", "foot": "shin"}


def ds_joints():
    J = {}
    for n, (h, t) in DS_J.items():
        J[n] = (np.array(h, float), np.array(t, float))
        if n.endswith(".R"):
            J[n[:-2] + ".L"] = (np.array(h, float) * (-1, 1, 1), np.array(t, float) * (-1, 1, 1))
    return J


def dijkstra(nb, src, n):
    import heapq
    d = np.full(n, np.inf)
    pq = []
    for v in src:
        d[v] = 0.0
        pq.append((0.0, v))
    heapq.heapify(pq)
    while pq:
        dv, v = heapq.heappop(pq)
        if dv > d[v]:
            continue
        for w, l in nb[v]:
            nd_ = dv + l
            if nd_ < d[w]:
                d[w] = nd_
                heapq.heappush(pq, (nd_, w))
    return d


def fit_glove_fingers(P, rep, edges, wrist, side, log=print):
    """Finger chains from an open glove mesh (geodesic). Fingertips = the 5 strongest local maxima of
    geodesic distance from the cuff; each finger's joints are centroids of geodesic bands measured from
    its tip (DIP 0.26 L, PIP 0.53 L, MCP 1.0 L; L = 0.47 x tip-to-cuff, thumb: own proportions).
    Returns [(j1, j2, j3, tip)] thumb..pinky."""
    U = np.unique(rep)
    n = len(P)
    nb = [[] for _ in range(n)]
    for a, b in edges:
        if a != b:
            l = float(np.linalg.norm(P[a] - P[b]))
            nb[a].append((b, l))
            nb[b].append((a, l))
    ax_ = np.linalg.norm(P[U] - wrist, axis=1)
    cuff = U[ax_ < ax_.min() + 0.012]
    g = dijkstra(nb, cuff, n)
    ok = np.isfinite(g)
    loc_max = [v for v in U if ok[v] and all(g[v] >= g[w] for w, _ in nb[v])]
    loc_max.sort(key=lambda v: -g[v])
    tips = []
    for v in loc_max:
        if all(dijkstra(nb, [t], n)[v] > 0.028 for t in tips):
            tips.append(v)
        if len(tips) == 5:
            break
    if len(tips) < 5:
        raise RuntimeError(f"glove {side}: only {len(tips)} fingertips")
    gt = [dijkstra(nb, [t], n) for t in tips]
    owner = np.argmin(np.stack(gt, 1), 1)
    cen = P[U].mean(0)
    hand_dir = nrm(np.mean([P[t] for t in tips], 0) - wrist)
    ang = [math.acos(np.clip(nrm(P[t] - wrist) @ hand_dir, -1, 1)) for t in tips]
    th = int(np.argmax(ang))

    def band(k, dist, w=0.006):
        sel = np.nonzero((np.abs(gt[k] - dist) < w) & ((owner == k) | (dist > 0.05)))[0]
        sel = sel[np.isin(sel, U)]
        if len(sel) < 3:
            sel = U[np.argsort(np.abs(gt[k][U] - dist))[:6]]
        return P[sel].mean(0)
    chains = {}
    for k, t in enumerate(tips):
        Lt = g[t]
        if k == th:
            fr = (0.80, 0.52, 0.24)        # CMC deep toward the wrist, MCP, IP (from the tip, x tip-to-cuff)
            js = [band(k, Lt * f) for f in fr]
        else:
            L = 0.47 * Lt
            js = [band(k, L * f) for f in (1.0, 0.53, 0.26)]
        chains[k] = (js[0], js[1], js[2], P[t].copy())
    others = sorted([k for k in range(5) if k != th], key=lambda k: np.linalg.norm(chains[k][0] - chains[th][0]))
    out = [chains[th]] + [chains[k] for k in others]
    log(f"[glove {side}] tips at geodesic " + " ".join(f"{g[tips[k]] * 100:.1f}" for k in [th] + others)
        + " cm; finger lengths (cm) " + " ".join(f"{(np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[1]) + np.linalg.norm(c[3] - c[2])) * 100:.1f}" for c in out))
    return out


def build_armature(name, J, parent_of, fingers=None):
    ad = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, ad)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = ad.edit_bones
    for n, (h, t) in J.items():
        b = eb.new(n)
        b.head, b.tail = Vector(h), Vector(t)
        b.roll = 0.0
    for n in J:
        p = parent_of(n)
        if p:
            eb[n].parent = eb[p]
    names = {}
    if fingers:
        for side, chains in fingers.items():
            names[side] = []
            for fi, (j1, j2, j3, tip) in enumerate(chains):
                fn = ("thumb", "index", "middle", "ring", "pinky")[fi]
                pts = [j1, j2, j3, tip]
                ch = []
                for k in range(3):
                    bn = f"{fn}_0{k + 1}.{side}"
                    b = eb.new(bn)
                    b.head, b.tail = Vector(pts[k]), Vector(pts[k + 1])
                    b.parent = eb[ch[-1]] if ch else eb[f"hand.{side}"]
                    ch.append(bn)
                names[side].append(ch)
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    arm.data.display_type = "STICK"
    return arm, names


def load_death_samurai(log=print):
    objs = import_glb(CHARS + "sf_death_samurai/model.glb")
    normalise(objs, scale=DS_SCALE, rotz=0.0, name="DS_ROOT")
    _, meshes = bake_world(objs, Matrix.Translation(DS_SHIFT))
    J = ds_joints()
    # gloves -> finger chains
    fingers = {}
    gloves = {}
    for o in meshes:
        if o.name.startswith("hand_"):
            P = mesh_world_verts(o)
            side = "R" if P[:, 0].mean() > 0 else "L"
            gloves[side] = o
            rep = weld_index(P, 1e-5)
            E = welded_edges(o.data, rep)
            wrist = np.array(J[f"hand.{side}"][0])
            fingers[side] = fit_glove_fingers(P, rep, E, wrist, side, log)
    def parent_of(n):
        base = n.split(".")[0]
        p = DS_PARENT[base]
        if p is None:
            return None
        return p + ("." + n.split(".")[1] if "." in n and p not in ("hips", "spine", "chest", "neck", "head") else "")
    arm, fnames = build_armature("RigB", J, parent_of, fingers)
    bmap = {n: n for n in J}
    chains = {s: [(c[0], c[1], c[2], fingers[s][k][3]) for k, c in enumerate(fnames[s])] for s in "RL"}
    fig = Figure("B", arm, bmap, chains, meshes)
    fig.fore_twist = {s: (f"fore.{s}", 0.75) for s in "RL"}
    fig.gloves = gloves
    rig_death_samurai(fig, log)
    return fig


def rig_death_samurai(fig, log=print):
    """Spring bones for the kusazuri (4 tassets) and the sode (shoulder plates); every armour part is
    bone-parented RIGIDLY to its dominant bone (or to a spring); only parts that really bend (sleeves,
    pants, the knee ropes, the gloves, the do's waist) are skinned."""
    arm, J = fig.arm, fig.J
    sp = {
        "spr_kusazuri.F": ((0, 0.12, 0.97), (0, 0.16, 0.62), "hips"),
        "spr_kusazuri.B": ((0, -0.12, 0.97), (0, -0.16, 0.62), "hips"),
        "spr_kusazuri.R": ((0.16, 0, 0.97), (0.23, 0, 0.62), "hips"),
        "spr_kusazuri.L": ((-0.16, 0, 0.97), (-0.23, 0, 0.62), "hips"),
    }
    for s_, sx in (("R", 1), ("L", -1)):
        sp[f"spr_sode.{s_}"] = ((sx * 0.20, -0.01, 1.49), (sx * 0.31, -0.02, 1.28), f"clav.{s_}")
    add_bones(arm, sp)
    for k in "FBRL":
        fig.springs[f"spr_kusazuri.{k}"] = dict(k=0.13, damp=0.78, amax=0.8, follow=k, down=0.0)
    for s_ in "RL":
        fig.springs[f"spr_sode.{s_}"] = dict(k=0.12, damp=0.80, amax=0.7, follow="arm" + s_, down=0.0)
    core = ["hips", "spine", "chest", "neck", "head", "clav.R", "clav.L", "upper.R", "upper.L", "fore.R", "fore.L",
            "hand.R", "hand.L", "thigh.R", "thigh.L", "shin.R", "shin.L", "foot.R", "foot.L"]
    B = arm.data.bones
    segs = {n: (np.array(B[n].head_local), np.array(B[n].tail_local)) for n in core}
    rigid, skinned, sprung = 0, 0, 0
    for o in fig.meshes:
        n = o.name
        P = mesh_world_verts(o)
        c = P.mean(0)
        if n.startswith(("plane_068", "plane_023")):
            if n.startswith("plane_068"):
                b = "spr_kusazuri.R" if c[0] > 0 else "spr_kusazuri.L"
            else:
                b = "spr_kusazuri.F" if c[1] > 0 else "spr_kusazuri.B"
            bone_parent(o, arm, b)
            sprung += 1
            continue
        if n.startswith(("plane_083", "plane_006")):
            bone_parent(o, arm, "spr_sode.R" if c[0] > 0 else "spr_sode.L")
            sprung += 1
            continue
        if n.startswith("hand_"):
            continue                                   # gloves below
        bendy = any(t in n for t in ("Samurai_Arm", "Samurai_pants")) or n.startswith(("nurbspath_067", "nurbspath_061"))
        if not bendy:
            # rigid plate: the bone nearest to most of its vertices (torso plates: hips/spine/chest only)
            cand = core if not (abs(c[0]) < 0.2 and 0.9 < c[2] < 1.5) else ["hips", "spine", "chest"]
            dist = np.stack([seg_dist(P, *segs[b]) for b in cand], 1)
            votes = np.bincount(np.argmin(dist, 1), minlength=len(cand))
            bone_parent(o, arm, cand[int(np.argmax(votes))])
            rigid += 1
            continue
        names, W = seg_weights(P, segs, sharp=4.0, top=3)
        rep = weld_index(P, 1e-5)
        Wr = W[rep]
        Wr = smooth_weights(o.data, Wr, iters=6, edges=welded_edges(o.data, rep))
        Wr = Wr[rep]
        write_weights(o, names, Wr, keep_top=3)
        armature_mod(o, arm, preserve_volume=True)
        o.parent = arm
        skinned += 1
    # gloves: hand + finger segments
    for side, g in fig.gloves.items():
        P = mesh_world_verts(g)
        fb = [b for ch in fig.fingers[side] for b in ch[:3]]
        gs = {f"hand.{side}": segs[f"hand.{side}"], f"fore.{side}": segs[f"fore.{side}"]}
        for b in fb:
            gs[b] = (np.array(B[b].head_local), np.array(B[b].tail_local))
        names, W = seg_weights(P, gs, sharp=5.0, top=2)
        rep = weld_index(P, 1e-5)
        Wr = smooth_weights(g.data, W[rep], iters=2, edges=welded_edges(g.data, rep))[rep]
        write_weights(g, names, Wr, keep_top=3)
        armature_mod(g, arm, preserve_volume=True)
        g.parent = arm
        skinned += 1
    log(f"[fig B] parts: {rigid} rigid on bones, {sprung} on spring bones, {skinned} skinned")
