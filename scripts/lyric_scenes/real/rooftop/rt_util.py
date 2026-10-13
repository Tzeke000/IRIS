"""Shared helpers for the Be My One rooftop scene (Blender 5.2, headless).

Node building (shader + geometry + compositor share the same API), bulk keyframe baking on 5.x slotted
Actions, mesh creation from numpy, and the beat grid.
"""
import math

import bpy
import numpy as np

# ---------------------------------------------------------------------------------------------------------
# timing (fitted 2026-10-13 from kick onsets: every section boundary lands within 1 frame of this grid)
# ---------------------------------------------------------------------------------------------------------
FPS = 30
BPM = 134.095
P = 60.0 / BPM            # beat period, s
T0 = 0.09864              # beat 0
NF = 3852                 # 128.4 s


def bt(k):
    """time (s) of beat k"""
    return T0 + k * P


def fr(t):
    """scene frame for time t (frame 1 = t 0)"""
    return 1 + int(round(t * FPS))


TT = np.arange(NF) / FPS  # time of each frame (index i ↔ frame i+1)


# ---------------------------------------------------------------------------------------------------------
# node builder
# ---------------------------------------------------------------------------------------------------------
class NB:
    """Tiny node-graph DSL. Any argument may be a socket (linked) or a constant (default_value)."""

    def __init__(self, nt):
        self.nt = nt

    def n(self, typ, **props):
        nd = self.nt.nodes.new(typ)
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    def set(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        else:
            if isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == "RGBA":
                v = (*v, 1.0)
            sock.default_value = v

    def link(self, a, b):
        self.nt.links.new(a, b)

    def m(self, op, a, b=None, c=None, clamp=False):
        nd = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            self.set(nd.inputs[i], v)
        return nd.outputs[0]

    def v(self, op, a, b=None, c=None):
        nd = self.n("ShaderNodeVectorMath", operation=op)
        if op == "SCALE":
            self.set(nd.inputs[0], a)
            self.set(nd.inputs["Scale"], b)
        else:
            for i, x in enumerate((a, b, c)):
                self.set(nd.inputs[i], x)
        if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE"):
            return nd.outputs["Value"]
        return nd.outputs["Vector"]

    def xyz(self, x=0.0, y=0.0, z=0.0):
        nd = self.n("ShaderNodeCombineXYZ")
        for i, s in enumerate((x, y, z)):
            self.set(nd.inputs[i], s)
        return nd.outputs[0]

    def sep(self, vec):
        nd = self.n("ShaderNodeSeparateXYZ")
        self.set(nd.inputs[0], vec)
        return nd.outputs[0], nd.outputs[1], nd.outputs[2]

    def mr(self, val, a, b, c=0.0, d=1.0, interp="LINEAR", clamp=True):
        nd = self.n("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        for name, x in (("Value", val), ("From Min", a), ("From Max", b), ("To Min", c), ("To Max", d)):
            self.set(nd.inputs[name], x)
        return nd.outputs["Result"]

    def mix(self, fac, a, b, blend="MIX", dtype="RGBA"):
        nd = self.n("ShaderNodeMix", data_type=dtype, blend_type=blend)
        self.set(nd.inputs[0], fac)
        if dtype == "RGBA":
            self.set(nd.inputs[6], a)
            self.set(nd.inputs[7], b)
            return nd.outputs[2]
        if dtype == "VECTOR":
            self.set(nd.inputs[4], a)
            self.set(nd.inputs[5], b)
            return nd.outputs[1]
        self.set(nd.inputs[2], a)
        self.set(nd.inputs[3], b)
        return nd.outputs[0]

    def val(self, name, v=0.0):
        nd = self.n("ShaderNodeValue")
        nd.name = nd.label = name
        nd.outputs[0].default_value = v
        return nd.outputs[0]

    # -- geometry-node specific --
    def attr(self, name, dtype="FLOAT"):
        nd = self.n("GeometryNodeInputNamedAttribute", data_type=dtype)
        nd.inputs["Name"].default_value = name
        return nd.outputs["Attribute"]

    def store(self, geo, name, value, dtype="FLOAT", domain="POINT"):
        nd = self.n("GeometryNodeStoreNamedAttribute", data_type=dtype, domain=domain)
        self.set(nd.inputs["Geometry"], geo)
        nd.inputs["Name"].default_value = name
        self.set(nd.inputs["Value"], value)
        return nd.outputs["Geometry"]

    def cmp(self, op, a, b):
        nd = self.n("FunctionNodeCompare", data_type="FLOAT", operation=op)
        self.set(nd.inputs["A"], a)
        self.set(nd.inputs["B"], b)
        return nd.outputs["Result"]

    def band(self, a, b, op="AND"):
        nd = self.n("FunctionNodeBooleanMath", operation=op)
        self.set(nd.inputs[0], a)
        if b is not None:
            self.set(nd.inputs[1], b)
        return nd.outputs[0]

    def rand(self, lo, hi, seed=0, id_=None):
        nd = self.n("FunctionNodeRandomValue", data_type="FLOAT")
        self.set(nd.inputs["Min"], lo)
        self.set(nd.inputs["Max"], hi)
        self.set(nd.inputs["Seed"], seed)
        if id_ is not None:
            self.set(nd.inputs["ID"], id_)
        return nd.outputs["Value"]

    # -- shader specific --
    def sattr(self, name, kind="GEOMETRY"):
        nd = self.n("ShaderNodeAttribute", attribute_name=name, attribute_type=kind)
        return nd

    def geo(self):
        return self.n("ShaderNodeNewGeometry")


def gn_tree(name, inputs=(), with_geo_in=True):
    """Geometry node group with Geometry in/out (+ extra (name, socket_type) inputs)."""
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    if with_geo_in:
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for nm, st in inputs:
        ng.interface.new_socket(nm, in_out="INPUT", socket_type=st)
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nb = NB(ng)
    gi = nb.n("NodeGroupInput")
    go = nb.n("NodeGroupOutput")
    return ng, nb, gi, go


def add_gn(ob, ng, name="GN"):
    mod = ob.modifiers.new(name, "NODES")
    mod.node_group = ng
    return mod


# ---------------------------------------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------------------------------------
def new_mat(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    nb = NB(nt)
    out = nb.n("ShaderNodeOutputMaterial")
    return m, nb, out


def principled(nb, base=(0.5, 0.5, 0.5), rough=0.5, metal=0.0, emis=None, estr=0.0):
    b = nb.n("ShaderNodeBsdfPrincipled")
    nb.set(b.inputs["Base Color"], base)
    nb.set(b.inputs["Roughness"], rough)
    nb.set(b.inputs["Metallic"], metal)
    if emis is not None:
        nb.set(b.inputs["Emission Color"], emis)
        nb.set(b.inputs["Emission Strength"], estr)
    return b


def simple_mat(name, base, rough=0.5, metal=0.0, emis=None, estr=0.0):
    m, nb, out = new_mat(name)
    b = principled(nb, base, rough, metal, emis, estr)
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def emit_mat(name, color, strength):
    m, nb, out = new_mat(name)
    e = nb.n("ShaderNodeEmission")
    nb.set(e.inputs["Color"], color)
    nb.set(e.inputs["Strength"], strength)
    nb.link(e.outputs[0], out.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------------------------------------
# keyframes (5.x slotted actions, bulk)
# ---------------------------------------------------------------------------------------------------------
_IP = {k: bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items[k].value
       for k in ("CONSTANT", "LINEAR", "BEZIER")}


def bake(idb, path, vals, index=0, frames=None, interp="LINEAR"):
    """Write keys on idb.<path>[index]. vals: per-frame array (frames 1..len) or with explicit frames."""
    vals = np.asarray(vals, dtype=np.float64)
    if frames is None:
        frames = np.arange(1, len(vals) + 1, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idb.name}_act")
    fc = ad.action.fcurve_ensure_for_datablock(idb, path, index=index)
    kp = fc.keyframe_points
    kp.clear()
    n = len(vals)
    kp.add(n)
    co = np.empty(2 * n, np.float32)
    co[0::2] = frames
    co[1::2] = vals
    kp.foreach_set("co", co)
    kp.foreach_set("interpolation", np.full(n, _IP[interp], np.int32))
    fc.update()
    return fc


def bake_vec(idb, path, arr, interp="LINEAR", frames=None):
    arr = np.asarray(arr)
    for i in range(arr.shape[1]):
        bake(idb, path, arr[:, i], index=i, frames=frames, interp=interp)


# ---------------------------------------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------------------------------------
def mesh_obj(name, verts, faces=(), coll=None, edges=()):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in np.asarray(verts).tolist()], list(edges), [tuple(f) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def points_obj(name, pts, attrs=None, coll=None):
    """Vertex-only mesh with float / vector point attributes (bulk foreach_set)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(pts))
    me.vertices.foreach_set("co", pts.ravel())
    for k, v in (attrs or {}).items():
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 2:
            a = me.attributes.new(k, "FLOAT_VECTOR", "POINT")
            a.data.foreach_set("vector", v.ravel())
        else:
            a = me.attributes.new(k, "FLOAT", "POINT")
            a.data.foreach_set("value", v)
    me.update()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def boxes_mesh(name, boxes, face_attrs=None, coll=None):
    """boxes: (N,6) [x0,y0,z0,x1,y1,z1]. 5 faces each (no bottom). face_attrs: {name: (N,) per box}."""
    b = np.asarray(boxes, dtype=np.float64)
    n = len(b)
    x0, y0, z0, x1, y1, z1 = b.T
    V = np.stack([
        np.stack([x0, y0, z0], 1), np.stack([x1, y0, z0], 1), np.stack([x1, y1, z0], 1), np.stack([x0, y1, z0], 1),
        np.stack([x0, y0, z1], 1), np.stack([x1, y0, z1], 1), np.stack([x1, y1, z1], 1), np.stack([x0, y1, z1], 1),
    ], 1).reshape(-1, 3)
    base = (np.arange(n) * 8)[:, None]
    F = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]])
    faces = (F[None, :, :] + base[:, None, :]).reshape(-1, 4)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.ravel().astype(np.int32))
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", (np.arange(len(faces)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    for k, v in (face_attrs or {}).items():
        a = me.attributes.new(k, "FLOAT", "FACE")
        a.data.foreach_set("value", np.repeat(np.asarray(v, dtype=np.float32), 5))
    me.validate()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def look_quat(loc, target, roll=0.0):
    from mathutils import Vector
    d = Vector(target) - Vector(loc)
    q = d.to_track_quat("-Z", "Y")
    if roll:
        from mathutils import Quaternion
        q = q @ Quaternion((0, 0, 1), roll)
    return q


def coll(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def deg(a):
    return math.radians(a)
