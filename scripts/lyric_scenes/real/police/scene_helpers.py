"""Blender helpers shared by real/ scene builders (extracted from rocket3/build_scene.py, 10-12).

Call init(N, cut_set) before baking: bake() keeps a key at every cut frame and holds it (CONSTANT) so

motion blur never smears across a cut or a wrap jump."""

import math

import numpy as np

import bpy

from mathutils import Matrix, Vector



N = 0

CUT_SET = None

SC = bpy.context.scene

VIS = {}

KEY_ENUM = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items

LIN, CONST = KEY_ENUM["LINEAR"].value, KEY_ENUM["CONSTANT"].value



def init(n, cut_set):

    global N, CUT_SET

    N = n

    CUT_SET = np.asarray(cut_set, bool)



def _fcurve(idb, path, index):
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idb.name}_bake")
    return ad.action.fcurve_ensure_for_datablock(idb, path, index=index)



def bake(idb, path, vals, index=0, const=False):
    """Bake one value per timeline frame onto idb.path[index] (bulk foreach_set)."""
    v = np.asarray(vals, np.float64)
    if v.ndim == 0:
        v = np.full(N, float(v))
    assert len(v) == N, (path, len(v))
    keep = np.ones(N, bool)
    keep[1:-1] = (v[1:-1] != v[:-2]) | (v[1:-1] != v[2:])
    keep |= CUT_SET
    keep[np.minimum(np.nonzero(CUT_SET)[0] + 1, N - 1)] = True
    ix = np.nonzero(keep)[0]
    fc = _fcurve(idb, path, index)
    kp = fc.keyframe_points
    kp.clear()
    kp.add(len(ix))
    co = np.empty(2 * len(ix), np.float32)
    co[0::2] = ix + 1
    co[1::2] = v[ix]
    kp.foreach_set("co", co)
    interp = np.full(len(ix), CONST if const else LIN, np.int32)
    interp[CUT_SET[ix]] = CONST
    kp.foreach_set("interpolation", interp)
    fc.update()
    return fc



def bake_vec(idb, path, arr, const=False):
    arr = np.asarray(arr, np.float64)
    for k in range(arr.shape[1]):
        bake(idb, path, arr[:, k], index=k, const=const)



def link(ob, parent=None):
    SC.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob



def empty(name, loc=(0, 0, 0), parent=None):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    return link(e, parent)



def mesh_obj(name, verts, faces, mat=None, parent=None, loc=(0, 0, 0), smooth=False, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in verts], [], [tuple(map(int, f)) for f in faces])
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        loop_v = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loop_v)
        uvl.data.foreach_set("uv", np.asarray(uvs, np.float32)[loop_v].ravel())
    me.update()
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    if mat is not None:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    return link(ob, parent)



def lathe(name, prof, segs=64, mat=None, parent=None, loc=(0, 0, 0), a0=0.0, a1=2 * math.pi,
          cap=False, smooth=True):
    """Surface of revolution around +Z. prof = [(r, z), ...]. UV: u = angle/2pi, v = profile t."""
    prof = np.asarray(prof, np.float64)
    full = abs((a1 - a0) - 2 * math.pi) < 1e-6
    na = segs if full else segs + 1
    ang = a0 + (a1 - a0) * np.arange(na) / segs
    seglen = np.r_[0, np.cumsum(np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1])))]
    tv = seglen / max(seglen[-1], 1e-9)
    verts, uvs, faces = [], [], []
    for j, (r, z) in enumerate(prof):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
            uvs.append(((a % (2 * math.pi)) / (2 * math.pi), tv[j]))
    for j in range(len(prof) - 1):
        for k in range(segs):
            k2 = (k + 1) % na
            faces.append((j * na + k, j * na + k2, (j + 1) * na + k2, (j + 1) * na + k))
    if cap:
        c = len(verts)
        verts.append((0, 0, prof[0, 1]))
        uvs.append((0.5, 0))
        for k in range(segs):
            faces.append((c, (k + 1) % na, k))
    # per-vertex UV; seam duplicates avoided by computing per-loop below
    me_uvs = np.asarray(uvs, np.float32)
    ob = mesh_obj(name, verts, faces, mat, parent, loc, smooth=smooth, uvs=None)
    me = ob.data
    uvl = me.uv_layers.new(name="UVMap")
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    uv = me_uvs[loop_v].copy()
    # fix the wrap seam: in a face whose u spans > 0.5, lift the small ones by 1
    poly_start = np.empty(len(me.polygons), np.int32)
    poly_tot = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_start", poly_start)
    me.polygons.foreach_get("loop_total", poly_tot)
    for ps, pt in zip(poly_start, poly_tot):
        us = uv[ps:ps + pt, 0]
        if us.max() - us.min() > 0.5:
            us[us < 0.5] += 1.0
            uv[ps:ps + pt, 0] = us
    uvl.data.foreach_set("uv", uv.ravel())
    return ob



class MB:
    """Accumulate boxes / polys for ONE material, then emit one mesh object."""

    def __init__(self):
        self.v, self.f = [], []

    def box(self, c, s, rot=None):
        cx, cy, cz = c
        sx, sy, sz = (s[0] / 2, s[1] / 2, s[2] / 2)
        pts = [Vector((x * sx, y * sy, z * sz)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        if rot is not None:
            pts = [rot @ p for p in pts]
        o = len(self.v)
        self.v += [(p.x + cx, p.y + cy, p.z + cz) for p in pts]
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.f.append(tuple(o + i for i in q))

    def beam(self, a, b, t):
        a, b = Vector(a), Vector(b)
        d = b - a
        rot = d.to_track_quat("Z", "Y").to_matrix()
        self.box(tuple((a + b) / 2), (t, t, d.length), rot)

    def poly(self, pts, faces):
        o = len(self.v)
        self.v += [tuple(p) for p in pts]
        self.f += [tuple(o + i for i in f) for f in faces]

    def obj(self, name, mat=None, parent=None, loc=(0, 0, 0)):
        return mesh_obj(name, self.v, self.f, mat, parent, loc)



def uv_sphere(name, r=1.0, seg=32, rings=16, mat=None, parent=None, loc=(0, 0, 0)):
    prof = [(r * math.sin(math.pi * k / rings), -r * math.cos(math.pi * k / rings)) for k in range(rings + 1)]
    return lathe(name, prof, seg, mat, parent, loc)



def cube_obj(name, mat=None, parent=None, loc=(0, 0, 0), scale=(1, 1, 1)):
    m = MB()
    m.box((0, 0, 0), (2, 2, 2))
    ob = m.obj(name, mat, parent, loc)
    ob.scale = scale
    return ob



def nd(nt, typ, **kw):
    n = nt.nodes.new(typ)
    for k, v in kw.items():
        if k == "inp":
            for name, val in v.items():
                n.inputs[name].default_value = val
        else:
            setattr(n, k, v)
    return n



def lk(nt, a, b):
    nt.links.new(a, b)



def _sock(nt, x, vec=False):
    if isinstance(x, (int, float)):
        v = nd(nt, "ShaderNodeValue")
        v.outputs[0].default_value = float(x)
        return v.outputs[0]
    if isinstance(x, (tuple, list)):
        c = nd(nt, "ShaderNodeCombineXYZ")
        for k in range(3):
            c.inputs[k].default_value = float(x[k])
        return c.outputs[0]
    return x



def M(nt, op, a, b=None, c=None, clamp=False):
    n = nd(nt, "ShaderNodeMath", operation=op, use_clamp=clamp)
    for k, x in enumerate((a, b, c)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = float(x)
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[0]



def V(nt, op, a, b=None, scale=None):
    n = nd(nt, "ShaderNodeVectorMath", operation=op)
    for k, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (tuple, list)):
            n.inputs[k].default_value = x
        else:
            lk(nt, x, n.inputs[k])
    if scale is not None:
        if isinstance(scale, (int, float)):
            n.inputs["Scale"].default_value = scale
        else:
            lk(nt, scale, n.inputs["Scale"])
    return n.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]



def value_node(nt, name, v=0.0):
    n = nd(nt, "ShaderNodeValue", name=name, label=name)
    n.outputs[0].default_value = v
    return n



def value_path(name):
    return f'nodes["{name}"].outputs[0].default_value'



def mix_col(nt, fac, a, b):
    n = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    for k, x in ((0, fac), (6, a), (7, b)):
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = x
        elif isinstance(x, (tuple, list)):
            n.inputs[k].default_value = tuple(x) + ((1.0,) if len(x) == 3 else ())
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[2]



def new_mat(name):
    m = bpy.data.materials.new(name)
    m.node_tree.nodes.clear()
    return m, m.node_tree



def principled(name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0, bump=None):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = tuple(color) + (1,)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = tuple(emit) + (1,)
        b.inputs["Emission Strength"].default_value = emit_strength
    if bump:
        tc = nd(nt, "ShaderNodeTexCoord")
        nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": bump[0], "Detail": 8.0})
        lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
        bp = nd(nt, "ShaderNodeBump", inp={"Strength": bump[1]})
        lk(nt, nz.outputs["Fac"], bp.inputs["Height"])
        lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
        cr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.3, "From Max": 0.7,
                                               "To Min": rough * 0.7, "To Max": min(1, rough * 1.3)})
        lk(nt, nz.outputs["Fac"], cr.inputs["Value"])
        lk(nt, cr.outputs["Result"], b.inputs["Roughness"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m



def emissive(name, color, strength, objcolor_gain=False):
    """Emission; with objcolor_gain the per-object colour (keyable!) multiplies it."""
    m, nt = new_mat(name)
    e = nd(nt, "ShaderNodeEmission", inp={"Color": tuple(color) + (1,), "Strength": strength})
    if objcolor_gain:
        oi = nd(nt, "ShaderNodeObjectInfo")
        mc = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        mc.inputs[0].default_value = 1.0
        mc.inputs[6].default_value = tuple(color) + (1,)
        lk(nt, oi.outputs["Color"], mc.inputs[7])
        lk(nt, mc.outputs[2], e.inputs["Color"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, e.outputs[0], o.inputs["Surface"])
    return m



def set_color_keys(ob, rgb_arr, const=True):
    """Bake obj.color (RGBA) — read in shaders through Object Info > Color."""
    arr = np.asarray(rgb_arr, np.float64)
    if arr.ndim == 1:
        arr = np.repeat(arr[:, None], 3, 1)
    for k in range(3):
        bake(ob, "color", arr[:, k], index=k, const=const)



def no_shadow(ob):
    ob.visible_shadow = False
    return ob



def show(ob, mask):
    VIS[ob.name] = np.asarray(mask, bool)
    return ob



def blackbody(k):
    """Approx sRGB-linear colour of a blackbody at k kelvin (Tanner Helland fit)."""
    t = k / 100.0
    r = 1.0 if t <= 66 else min(1, max(0, 329.698727446 * (t - 60) ** -0.1332047592 / 255))
    g = (min(1, max(0, (99.4708025861 * math.log(t) - 161.1195681661) / 255)) if t <= 66
         else min(1, max(0, 288.1221695283 * (t - 60) ** -0.0755148492 / 255)))
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else min(1, max(0, (138.5177312231 * math.log(t - 10)
                                                                - 305.0447927307) / 255)))
    return tuple(c ** 2.2 for c in (r, g, b))



def load_img(path, data=False):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = "Non-Color" if data else "sRGB"
    return im



def tex_box(nt, path, tile_m, data=False, coord=None, blend=0.25):
    """Box-projected image texture in object space; tile_m = metres per tile. Returns the node."""
    if coord is None:
        tc = nd(nt, "ShaderNodeTexCoord")
        coord = tc.outputs["Object"]
    mp = nd(nt, "ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / tile_m,) * 3
    lk(nt, coord, mp.inputs["Vector"])
    t = nd(nt, "ShaderNodeTexImage", image=load_img(path, data), projection="BOX", projection_blend=blend)
    lk(nt, mp.outputs[0], t.inputs["Vector"])
    return t


