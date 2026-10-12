"""HIT A BUMP — photoreal late-night arcade, built deterministically from the music timeline.

    ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/real/arcade/build_scene.py -- \
        [--timeline state/tzeke_songs/hit_a_bump/timeline_30fps.npz] \
        [--assets assets/scenes/arcade] [--wav state/tzeke_songs/hit_a_bump/hit_a_bump_half.wav] \
        [--out assets/scenes/arcade/arcade.blend]

Needs (run first, venv python): game_frames.py (CRT image sequences + screen_stats.npz) and
make_art.py (printed art). No GPU is used here; rendering is render_frames.py / render.sh.

Engine: EEVEE (5.2) — volumetric haze, screen-space ray tracing + fast GI, DOF, camera motion
blur, AgX, compositor bloom. Scene frame f <-> timeline index f-1 <-> audio time (f-1)/30.

Music -> motion (everything is BAKED as F-curves; no handlers, no Python drivers):
  * one shader node group "AUD" holds per-frame Value nodes (kick, snare, hat, sub, neon, flood,
    power, glow, beat, strobe, pre, hot) that every audio-reactive material reads;
  * lights (screen spill, neon wash, beams, panel/coin-door glows) get energy/colour curves;
  * the camera is planned shot-by-shot from the sections (calm dollies, the build's push into
    the CRT, the drop entry pulling back out of the glass into a flooded room, 2-bar handheld
    shots cut on bar downbeats in the drops, a boss-fight cut pattern in the BIG drop).
"""
from __future__ import annotations

import argparse
import colorsys
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L  # noqa: E402

FPS = 30


# ====================================================================== args
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    root = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    ap.add_argument("--timeline", default=os.path.join(root, "state/tzeke_songs/hit_a_bump/timeline_30fps.npz"))
    ap.add_argument("--assets", default=os.path.join(root, "assets/scenes/arcade"))
    ap.add_argument("--wav", default=os.path.join(root, "state/tzeke_songs/hit_a_bump/hit_a_bump_half.wav"))
    ap.add_argument("--out", default=os.path.join(root, "assets/scenes/arcade/arcade.blend"))
    return ap.parse_args(argv)


# ====================================================================== numpy helpers
def gsmooth(x, s):
    if s <= 0:
        return x.copy()
    r = int(3 * s)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, "valid")


def envelope(x, attack_f, release_f):
    a = math.exp(-1 / max(attack_f, 1e-3))
    r = math.exp(-1 / max(release_f, 1e-3))
    out = np.empty_like(x)
    v = 0.0
    for i, s in enumerate(x):
        k = a if s > v else r
        v = k * v + (1 - k) * s
        out[i] = v
    return out


def decay(age, tau):
    age = np.asarray(age, float)
    return np.where(age < 10 ** 5, np.exp(-age / tau), 0.0)


def h01(*a):
    x = 0x9E3779B1
    for v in a:
        x = (x ^ (int(v) & 0xFFFFFFFF)) * 0x85EBCA6B & 0xFFFFFFFF
        x ^= x >> 13
        x = x * 0xC2B2AE35 & 0xFFFFFFFF
        x ^= x >> 16
    return x / 4294967296.0


# ====================================================================== bpy helpers
LINEAR = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value
CONSTANT = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["CONSTANT"].value


def bake(idb, path, values, index=0, const_frames=None):
    """Bulk-bake one F-curve (frames 1..n) on any ID. const_frames: keys that HOLD (cuts)."""
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idb.name}_bake")
    fc = ad.action.fcurve_ensure_for_datablock(idb, path, index=index)
    kp = fc.keyframe_points
    kp.clear()
    n = len(values)
    kp.add(n)
    co = np.empty(2 * n, np.float32)
    co[0::2] = np.arange(1, n + 1)
    co[1::2] = np.asarray(values, np.float32)
    kp.foreach_set("co", co)
    interp = np.full(n, LINEAR, np.int32)
    if const_frames is not None and len(const_frames):
        idx = np.asarray(const_frames, int) - 1
        interp[idx[(idx >= 0) & (idx < n)]] = CONSTANT
    kp.foreach_set("interpolation", interp)
    fc.update()
    return fc


class MB:
    """Mesh builder: faces with per-corner UVs + material index -> one object."""

    def __init__(self):
        self.v, self.f, self.uv, self.mi = [], [], [], []

    def face(self, pts, uvs=None, mi=0):
        o = len(self.v)
        self.v += [tuple(p) for p in pts]
        self.f.append(tuple(range(o, o + len(pts))))
        self.uv.append(uvs or [(0, 0)] * len(pts))
        self.mi.append(mi)

    def quad(self, a, b, c, d, mi=0, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
        self.face([a, b, c, d], list(uv), mi)

    def box(self, c, s, mi=0):
        x, y, z = c
        hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
        P = [(x - hx, y - hy, z - hz), (x + hx, y - hy, z - hz), (x + hx, y + hy, z - hz), (x - hx, y + hy, z - hz),
             (x - hx, y - hy, z + hz), (x + hx, y - hy, z + hz), (x + hx, y + hy, z + hz), (x - hx, y + hy, z + hz)]
        for q in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            self.quad(*[P[k] for k in q], mi=mi)

    def build(self, name, mats, coll, smooth=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        uvl = me.uv_layers.new(name="UVMap")
        flat = np.array([c for fuv in self.uv for uv in fuv for c in uv], np.float32)
        uvl.data.foreach_set("uv", flat)
        me.polygons.foreach_set("material_index", np.array(self.mi, np.int32))
        me.polygons.foreach_set("use_smooth", np.full(len(self.f), smooth, bool))
        for m in mats:
            me.materials.append(m)
        me.update()
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        return ob


def bm_mesh(name, fn):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    try:
        fn(bm)
        bm.to_mesh(me)
    finally:
        bm.free()
    return me


def cyl_mesh(name, r, h, seg=32, r2=None):
    def f(bm):
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r,
                              radius2=r if r2 is None else r2, depth=h, matrix=Matrix.Translation((0, 0, h / 2)))
    me = bm_mesh(name, f)
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    return me


def sphere_mesh(name, r, seg=24):
    def f(bm):
        bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=r)
    me = bm_mesh(name, f)
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    return me


def obj(name, me, coll, loc=(0, 0, 0), rot=(0, 0, 0), parent=None, mat=None):
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = rot
    if mat is not None:                       # shared meshes -> material linked per OBJECT
        if not me.materials:
            me.materials.append(mat)
        ob.material_slots[0].link = "OBJECT"
        ob.material_slots[0].material = mat
    if parent is not None:
        ob.parent = parent
    return ob


# ====================================================================== materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    return m, nt, out


def principled(nt, **kw):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    for k, v in kw.items():
        b.inputs[k].default_value = v
    return b


def img_node(nt, path, cs="sRGB", interp="Linear", ext="REPEAT", seq_len=None):
    # ABSOLUTE path (10-12 fix): a CWD-relative path got saved as a bogus '//../../../..' path relative
    # to the .blend, so every render from the saved file showed Blender's missing-image pink on the CRTs
    im = bpy.data.images.load(os.path.abspath(path), check_existing=True)
    im.colorspace_settings.name = cs
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = im
    t.interpolation = interp
    t.extension = ext
    if seq_len:
        im.source = "SEQUENCE"
        iu = t.image_user
        iu.frame_duration = seq_len
        iu.frame_start = 1
        iu.frame_offset = 0
        iu.use_auto_refresh = True
        iu.use_cyclic = False
    return t


def aud(nt):
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = bpy.data.node_groups["AUD"]
    return g


def math_node(nt, op, a=None, b=None, clamp=False):
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = op
    m.use_clamp = clamp
    for k, v in ((0, a), (1, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[k].default_value = v
        else:
            nt.links.new(v, m.inputs[k])
    return m.outputs[0]


AUD_KEYS = ["kick", "snare", "hat", "sub", "neon", "flood", "power", "glow", "beat", "strobe", "pre",
            "hot", "marquee", "btn_kick", "btn_snare", "coin"]


def make_aud_group():
    ng = bpy.data.node_groups.new("AUD", "ShaderNodeTree")
    for k in AUD_KEYS:
        ng.interface.new_socket(k, in_out="OUTPUT", socket_type="NodeSocketFloat")
    go = ng.nodes.new("NodeGroupOutput")
    for j, k in enumerate(AUD_KEYS):
        v = ng.nodes.new("ShaderNodeValue")
        v.name = v.label = k
        v.location = (-300, -j * 90)
        ng.links.new(v.outputs[0], go.inputs[k])
    return ng


def mat_body(name="Body", col=(0.018, 0.018, 0.02), rough=0.42):
    m, nt, out = new_mat(name)
    b = principled(nt, **{"Base Color": (*col, 1), "Roughness": rough, "Coat Weight": 0.15, "Coat Roughness": 0.25})
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 60
    noise.inputs["Detail"].default_value = 6
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = rough * 0.75
    rr.inputs["To Max"].default_value = min(1, rough * 1.35)
    nt.links.new(noise.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.04
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_printed(name, path, rough=0.35, hue_attr=None, coat=0.6, emit=0.0):
    """Printed laminate art (side art / bezel / cp). Optional per-object hue (Attribute OBJECT)."""
    m, nt, out = new_mat(name)
    t = img_node(nt, path)
    col = t.outputs["Color"]
    if hue_attr:
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type = "OBJECT"
        at.attribute_name = hue_attr
        hs = nt.nodes.new("ShaderNodeHueSaturation")
        nt.links.new(at.outputs["Fac"], hs.inputs["Hue"])
        nt.links.new(col, hs.inputs["Color"])
        col = hs.outputs["Color"]
    b = principled(nt, **{"Roughness": rough, "Coat Weight": coat, "Coat Roughness": 0.12})
    nt.links.new(col, b.inputs["Base Color"])
    if emit:
        nt.links.new(col, b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = emit
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_cpanel(name, art, wear):
    m, nt, out = new_mat(name)
    t = img_node(nt, art)
    w = img_node(nt, wear, cs="Non-Color")
    b = principled(nt, **{"Coat Weight": 0.5, "Coat Roughness": 0.1})
    # wear: lightens + roughens the print and dents the coat
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "SCREEN"
    wf = math_node(nt, "MULTIPLY", w.outputs["Color"], 0.22)
    nt.links.new(wf, mix.inputs["Factor"])
    nt.links.new(t.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (0.16, 0.155, 0.15, 1)
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = 0.25
    rr.inputs["To Max"].default_value = 0.75
    nt.links.new(w.outputs["Color"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    cr = nt.nodes.new("ShaderNodeMapRange")
    cr.inputs["To Min"].default_value = 0.05
    cr.inputs["To Max"].default_value = 0.6
    nt.links.new(w.outputs["Color"], cr.inputs["Value"])
    nt.links.new(cr.outputs["Result"], b.inputs["Coat Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.06
    bump.inputs["Distance"].default_value = 0.0003
    nt.links.new(w.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Coat Normal"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_crt(name, seq_path, n, strength=4.2, hue_attr=None, aud_power=False):
    """CRT phosphor face: the baked game image as emission under a glossy tube-glass surface."""
    m, nt, out = new_mat(name)
    t = img_node(nt, seq_path, interp="Linear", ext="EXTEND", seq_len=n)
    col = t.outputs["Color"]
    if hue_attr:
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type = "OBJECT"
        at.attribute_name = hue_attr
        hs = nt.nodes.new("ShaderNodeHueSaturation")
        nt.links.new(at.outputs["Fac"], hs.inputs["Hue"])
        nt.links.new(col, hs.inputs["Color"])
        col = hs.outputs["Color"]
    b = principled(nt, **{"Base Color": (0.018, 0.022, 0.02, 1), "Roughness": 0.05, "IOR": 1.52,
                          "Specular IOR Level": 0.9})
    nt.links.new(col, b.inputs["Emission Color"])
    if aud_power:
        g = aud(nt)
        s = math_node(nt, "MULTIPLY", g.outputs["power"], strength)
        nt.links.new(s, b.inputs["Emission Strength"])
    else:
        b.inputs["Emission Strength"].default_value = strength
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_glass(name="Glass"):
    m, nt, out = new_mat(name)
    m.surface_render_method = "DITHERED"
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.02
    gl.inputs["Color"].default_value = (1, 1, 1, 1)
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.12
    f = math_node(nt, "MULTIPLY_ADD", lw.outputs["Fresnel"], 0.75)
    f.node.inputs[2].default_value = 0.035
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(f, mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def mat_emit_aud(name, color, key, scale, base=0.0, body=(0.6, 0.6, 0.6), rough=0.3, color_attr=None, gain_attr=None):
    """Plastic/neon part whose emission = base + scale*AUD[key]."""
    m, nt, out = new_mat(name)
    b = principled(nt, **{"Base Color": (*body, 1), "Roughness": rough, "Emission Color": (*color, 1),
                          "Coat Weight": 0.4, "Coat Roughness": 0.08})
    if color_attr:
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type = "OBJECT"
        at.attribute_name = color_attr
        nt.links.new(at.outputs["Color"], b.inputs["Emission Color"])
        nt.links.new(at.outputs["Color"], b.inputs["Base Color"])
    g = aud(nt)
    s = math_node(nt, "MULTIPLY_ADD", g.outputs[key], scale)
    s.node.inputs[2].default_value = base
    if gain_attr:
        ga = nt.nodes.new("ShaderNodeAttribute")
        ga.attribute_type = "OBJECT"
        ga.attribute_name = gain_attr
        s = math_node(nt, "MULTIPLY", s, ga.outputs["Fac"])
    nt.links.new(s, b.inputs["Emission Strength"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_marquee(name, path):
    m, nt, out = new_mat(name)
    t = img_node(nt, path)
    b = principled(nt, **{"Roughness": 0.18, "Coat Weight": 1.0, "Coat Roughness": 0.03})
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
    g = aud(nt)
    s = math_node(nt, "MULTIPLY", g.outputs["marquee"], 3.2)
    nt.links.new(s, b.inputs["Emission Strength"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_simple(name, col, rough=0.5, metal=0.0, coat=0.0):
    m, nt, out = new_mat(name)
    b = principled(nt, **{"Base Color": (*col, 1), "Roughness": rough, "Metallic": metal, "Coat Weight": coat})
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_coindoor(path):
    m, nt, out = new_mat("CoinDoor")
    t = img_node(nt, path)
    b = principled(nt, **{"Metallic": 0.92, "Roughness": 0.32})
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_carpet(path, glow_path):
    m, nt, out = new_mat("Carpet")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / 2.4, 1 / 2.4, 1)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    t = img_node(nt, path)
    gw = img_node(nt, glow_path, cs="Non-Color")
    nt.links.new(mp.outputs[0], t.inputs["Vector"])
    nt.links.new(mp.outputs[0], gw.inputs["Vector"])
    b = principled(nt, **{"Roughness": 0.97, "Sheen Weight": 0.6, "Sheen Roughness": 0.4})
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
    g = aud(nt)
    e = math_node(nt, "MULTIPLY", gw.outputs["Color"], g.outputs["glow"])
    e2 = math_node(nt, "MULTIPLY", e, 0.55)
    nt.links.new(e2, b.inputs["Emission Strength"])
    # tufted pile: fine noise bump
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 900
    nz.inputs["Detail"].default_value = 3
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_brick():
    m, nt, out = new_mat("BrickWall")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Color1"].default_value = (0.035, 0.03, 0.034, 1)
    br.inputs["Color2"].default_value = (0.022, 0.02, 0.026, 1)
    br.inputs["Mortar"].default_value = (0.012, 0.012, 0.014, 1)
    br.inputs["Scale"].default_value = 4.0
    br.inputs["Mortar Size"].default_value = 0.012
    br.inputs["Brick Width"].default_value = 0.5
    br.inputs["Row Height"].default_value = 0.17
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (math.pi / 2, 0, 0)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], br.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 30
    nz.inputs["Detail"].default_value = 8
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    b = principled(nt, Roughness=0.88)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.5
    nt.links.new(br.outputs["Color"], mix.inputs[6])
    nt.links.new(nz.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    hgt = math_node(nt, "ADD", br.outputs["Fac"], math_node(nt, "MULTIPLY", nz.outputs["Fac"], 0.3))
    nt.links.new(hgt, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


def mat_neon(name, color, key="neon", scale=14.0):
    m, nt, out = new_mat(name)
    b = principled(nt, **{"Base Color": (*[0.5 + 0.5 * c for c in color], 1), "Roughness": 0.15,
                          "Emission Color": (*color, 1), "Coat Weight": 1.0, "Coat Roughness": 0.02})
    g = aud(nt)
    s = math_node(nt, "MULTIPLY", g.outputs[key], scale)
    nt.links.new(s, b.inputs["Emission Strength"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return m


# ====================================================================== geometry: cabinet
def yz_n(a, b):
    """unit direction a->b in (y,z) and the outward normal (0, -dz, dy)."""
    dy, dz = b[0] - a[0], b[1] - a[1]
    ln = math.hypot(dy, dz)
    return (dy / ln, dz / ln), (-dz / ln, dy / ln), ln


def frame_on(seg, u, v, h=0.0):
    """Point on a profile segment: u = x, v = metres along seg, h = metres along its normal."""
    (dy, dz), (ny, nz), _ = yz_n(*seg)
    a = seg[0]
    return Vector((u, a[0] + dy * v + ny * h, a[1] + dz * v + nz * h))


def seg_rot(seg):
    """Euler for an object whose local Z = the segment's outward normal, local Y = along it."""
    (dy, dz), _, _ = yz_n(*seg)
    return Euler((math.atan2(dz, dy), 0, 0))


class Mats:
    pass


def build_materials(A, n):
    M = Mats()
    art = os.path.join(A, "art")
    M.body = mat_body()
    M.side_hero = mat_printed("SideHero", os.path.join(art, "side_hero.png"))
    M.side_gen = mat_printed("SideGeneric", os.path.join(art, "side_generic.png"), hue_attr="hue")
    M.bezel_hero = mat_printed("BezelHero", os.path.join(art, "bezel_hero.png"), rough=0.3, coat=0.0)
    M.bezel_gen = mat_simple("BezelGeneric", (0.012, 0.012, 0.014), rough=0.35)
    M.cp_hero = mat_cpanel("CPHero", os.path.join(art, "cpanel_hero.png"), os.path.join(art, "cpanel_wear.png"))
    M.cp_gen = mat_cpanel("CPGeneric", os.path.join(art, "side_generic.png"), os.path.join(art, "cpanel_wear.png"))
    M.speaker = mat_simple("Speaker", (0.01, 0.01, 0.01), rough=0.7)
    M.crt_hero = mat_crt("CRTHero", os.path.join(A, "game", "g_0001.png"), n, strength=4.2)
    M.crt_att = [mat_crt(f"CRTAttract{k}", os.path.join(A, "attract", f"a{k}_0001.png"), n, strength=2.3,
                         hue_attr="hue", aud_power=True) for k in range(2)]
    M.glass = mat_glass()
    M.marquee_hero = mat_marquee("MarqueeHero", os.path.join(art, "marquee_hero.png"))
    M.marquee = [mat_marquee(f"Marquee{k}", os.path.join(art, f"marquee_{k}.png")) for k in range(len(L.OTHER_GAMES))]
    M.chrome = mat_simple("Chrome", (0.9, 0.9, 0.92), rough=0.12, metal=1.0)
    M.rubber = mat_simple("Rubber", (0.01, 0.01, 0.01), rough=0.8)
    M.joyball = mat_simple("JoyBall", (0.6, 0.02, 0.02), rough=0.12, coat=1.0)
    M.tmold = mat_emit_aud("TMolding", (1, 1, 1), "neon", 3.2, base=0.0, body=(0.3, 0.3, 0.3), rough=0.25, color_attr="tm_col",
                           gain_attr="gain")
    BC = dict(red=(1, 0.06, 0.04), yellow=(1, 0.75, 0.05), blue=(0.1, 0.35, 1), green=(0.1, 1, 0.25),
              white=(1, 1, 1), purple=(0.55, 0.15, 1))
    M.btn = {}
    for c, rgb in BC.items():
        key = "btn_kick" if c in ("red", "yellow", "blue") else "btn_snare"
        M.btn[c] = mat_emit_aud(f"Btn_{c}", rgb, key, 9.0, base=0.6, body=tuple(0.5 * v for v in rgb), rough=0.22)
    M.btn_start = mat_emit_aud("Btn_start", (1, 1, 1), "beat", 5.0, base=0.8, body=(0.8, 0.8, 0.8), rough=0.3)
    M.coindoor = mat_coindoor(os.path.join(art, "coindoor.png"))
    M.coinlight = mat_emit_aud("CoinLight", (1, 0.12, 0.04), "coin", 7.0, base=0.0, body=(0.4, 0.02, 0.02), rough=0.3)
    M.slot = mat_simple("Slot", (0.0, 0.0, 0.0), rough=0.9)
    M.carpet = mat_carpet(os.path.join(art, "carpet.png"), os.path.join(art, "carpet_glow.png"))
    M.brick = mat_brick()
    M.paint = mat_body("WallPaint", col=(0.022, 0.016, 0.03), rough=0.85)
    M.ceiling = mat_simple("Ceiling", (0.012, 0.012, 0.014), rough=0.9)
    M.sign = mat_printed("SignRules", os.path.join(art, "sign_rules.png"), rough=0.5, coat=0.3)
    M.neon_pink = mat_neon("NeonPink", (1.0, 0.08, 0.45))
    M.neon_cyan = mat_neon("NeonCyan", (0.05, 0.75, 1.0))
    M.neon_gold = mat_neon("NeonGold", (1.0, 0.55, 0.05))
    M.housing = mat_simple("Housing", (0.03, 0.03, 0.035), rough=0.4, metal=0.6)
    return M


def build_cabinet(coll, M, name, loc, yaw, hero, k=0, tm_col=(1, 0.2, 0.05), hue=0.5, variant=0, mirror=False):
    root = bpy.data.objects.new(name, None)
    coll.objects.link(root)
    root.location = loc
    root.rotation_euler = (0, 0, yaw)
    I = L.INNER
    # ---------------- body: front pieces, top, back (one mesh, material slots)
    mb = MB()
    slots = [M.body, M.cp_hero if hero else M.cp_gen, M.bezel_hero if hero else M.bezel_gen,
             M.speaker, M.marquee_hero if hero else M.marquee[k % len(M.marquee)]]
    for seg, mi in ((L.KICK, 0), (L.CP_UNDER, 0), (L.CP_LIP, 0), (L.CP_TOP, 1), (L.SPEAKER, 3),
                    (L.MARQUEE, 4), (L.TOP, 0), (L.BACK, 0)):
        a, b = seg
        mb.quad((-I, a[0], a[1]), (I, a[0], a[1]), (I, b[0], b[1]), (-I, b[0], b[1]), mi=mi)
    # bezel with a screen-sized hole: 4 strips (UVs from the strip's position on the bezel)
    (dy, dz), (ny, nz), blen = yz_n(*L.BEZEL)
    s0 = (blen - L.SCREEN_H) / 2
    s1 = s0 + L.SCREEN_H
    hw = L.SCREEN_W / 2

    def bp(x, s):
        p = frame_on(L.BEZEL, x, s)
        return tuple(p), ((x + I) / (2 * I), s / blen)

    for (xa, xb, sa, sb) in ((-I, I, 0, s0), (-I, I, s1, blen), (-I, -hw, s0, s1), (hw, I, s0, s1)):
        P = [bp(xa, sa), bp(xb, sa), bp(xb, sb), bp(xa, sb)]
        mb.face([p for p, _ in P], [uv for _, uv in P], mi=2)
    # bezel recess cheeks (the bezel sits 2 cm behind the side-panel edge) — closed by side panels
    body = mb.build(name + "_body", slots, coll)
    body.parent = root
    # ---------------- side panels (profile extruded), outer face gets the side art
    ys = [p[0] for p in L.SIDE]
    zs = [p[1] for p in L.SIDE]
    ymin, ymax, zmax = min(ys), max(ys), max(zs)
    for sgn in (-1, 1):
        sp = MB()
        xo, xi = sgn * (I + L.SIDE_T), sgn * I
        outer = [(xo, y, z) for (y, z) in L.SIDE]
        inner = [(xi, y, z) for (y, z) in L.SIDE]
        uvs = [((y - ymin) / (ymax - ymin) if sgn > 0 else 1 - (y - ymin) / (ymax - ymin), z / zmax) for (y, z) in L.SIDE]
        if sgn > 0:
            sp.face(outer, uvs, mi=0)
            sp.face(inner[::-1], uvs[::-1], mi=1)
        else:
            sp.face(outer[::-1], uvs[::-1], mi=0)
            sp.face(inner, uvs, mi=1)
        nP = len(L.SIDE)
        for j in range(nP):
            a, b = L.SIDE[j], L.SIDE[(j + 1) % nP]
            q = [(xi, a[0], a[1]), (xo, a[0], a[1]), (xo, b[0], b[1]), (xi, b[0], b[1])]
            sp.quad(*(q if sgn > 0 else q[::-1]), mi=1)
        side = sp.build(f"{name}_side{'R' if sgn > 0 else 'L'}", [M.side_hero if hero else M.side_gen, M.body], coll)
        side.parent = root
        side["hue"] = hue
        # T-molding: a lit tube running the panel's edge (everything but the floor edge)
        cu = bpy.data.curves.new(f"{name}_tm{sgn}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.003
        cu.bevel_resolution = 3
        spl = cu.splines.new("POLY")
        pts = L.SIDE[0:]
        spl.points.add(len(pts) - 1)
        for j, (y, z) in enumerate(pts):
            # nudge outward from the panel's centroid so the tube sits ON the edge
            cy, cz = 0.4, 0.95
            dyy, dzz = y - cy, z - cz
            ln = math.hypot(dyy, dzz)
            spl.points[j].co = (sgn * (I + L.SIDE_T / 2), y + 0.003 * dyy / ln, z + 0.003 * dzz / ln, 1)
        tm = bpy.data.objects.new(f"{name}_tmold{sgn}", cu)
        coll.objects.link(tm)
        cu.materials.append(M.tmold)
        tm.parent = root
        tm["tm_col"] = tm_col
        tm["gain"] = 1.0 if hero else 0.4
    # ---------------- CRT tube face (curved), glass, bezel lighting
    NX, NY = 48, 36
    W2, H2 = L.SCREEN_W * 1.025 / 2, L.SCREEN_H * 1.025 / 2
    sc = (s0 + s1) / 2
    verts, uvs, faces = [], [], []
    for j in range(NY + 1):
        for i in range(NX + 1):
            x = -W2 + 2 * W2 * i / NX
            s = -H2 + 2 * H2 * j / NY
            r2 = (x / W2) ** 2 * 0.5 + (s / H2) ** 2 * 0.5
            h = -0.006 - L.SCREEN_BULGE * r2
            p = frame_on(L.BEZEL, x, sc + s, h)
            verts.append(tuple(p))
            uvs.append((i / NX, j / NY))
    for j in range(NY):
        for i in range(NX):
            a = j * (NX + 1) + i
            faces.append((a, a + 1, a + NX + 2, a + NX + 1))
    me = bpy.data.meshes.new(name + "_crt")
    me.from_pydata(verts, [], faces)
    uvl = me.uv_layers.new(name="UVMap")
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    uva = np.array(uvs, np.float32)[loop_v]
    if mirror:
        uva[:, 0] = 1 - uva[:, 0]
    uvl.data.foreach_set("uv", uva.ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    me.materials.append(M.crt_hero if hero else M.crt_att[variant])
    crt = bpy.data.objects.new(name + "_crt", me)
    coll.objects.link(crt)
    crt.parent = root
    crt["hue"] = hue
    # tube surround (black, behind the face) so gaps never show the cabinet interior
    surround = MB()
    for (xa, xb, sa, sb) in ((-I, I, 0, blen),):
        P = [frame_on(L.BEZEL, xa, sa, -0.03), frame_on(L.BEZEL, xb, sa, -0.03),
             frame_on(L.BEZEL, xb, sb, -0.03), frame_on(L.BEZEL, xa, sb, -0.03)]
        surround.quad(*[tuple(p) for p in P])
    so = surround.build(name + "_surround", [M.speaker], coll)
    so.parent = root
    # (no separate protective-glass pane: EEVEE's dithered transparency speckled the screens at
    #  render sample counts; the tube face itself carries the glass reflection)
    # glass retainer strips (chrome) top + bottom
    for s_ in (0.0, blen):
        c = frame_on(L.BEZEL, 0, s_, 0.012)
        strip = obj(name + "_ret", bpy.data.meshes.get("RetStrip") or _box_mesh("RetStrip", (2 * I, 0.012, 0.006)),
                    coll, loc=c, rot=seg_rot(L.BEZEL), parent=root, mat=M.chrome)
    # ---------------- marquee retainers
    for z in (L.MARQUEE[0][1], L.MARQUEE[1][1]):
        obj(name + "_mret", bpy.data.meshes.get("MRet") or _box_mesh("MRet", (2 * I, 0.01, 0.014)), coll,
            loc=(0, L.MARQUEE[0][0] - 0.004, z), parent=root, mat=M.chrome)
    # ---------------- control panel
    rot = seg_rot(L.CP_TOP)
    ju, jv = L.JOY
    obj(name + "_washer", _shared_cyl("Washer", 0.032, 0.002), coll, loc=frame_on(L.CP_TOP, ju, jv, 0.0), rot=rot,
        parent=root, mat=M.rubber)
    obj(name + "_shaft", _shared_cyl("Shaft", 0.0055, 0.068), coll, loc=frame_on(L.CP_TOP, ju, jv, 0.0), rot=rot,
        parent=root, mat=M.chrome)
    obj(name + "_ball", bpy.data.meshes.get("Ball") or sphere_mesh("Ball", 0.0175), coll,
        loc=frame_on(L.CP_TOP, ju, jv, 0.083), rot=rot, parent=root, mat=M.joyball)
    for (u, v, c, _lab) in L.BUTTONS:
        obj(name + "_bring", _shared_cyl("BtnRing", L.BTN_R + 0.004, 0.005), coll, loc=frame_on(L.CP_TOP, u, v, 0), rot=rot,
            parent=root, mat=M.rubber)
        obj(name + "_btn", _shared_cyl("BtnCap", L.BTN_R, 0.011, r2=L.BTN_R * 0.93), coll,
            loc=frame_on(L.CP_TOP, u, v, 0.0), rot=rot, parent=root, mat=M.btn[c])
    for (u, v, _lab) in L.START:
        obj(name + "_start", _shared_cyl("StartCap", 0.0105, 0.008), coll, loc=frame_on(L.CP_TOP, u, v, 0), rot=rot,
            parent=root, mat=M.btn_start)
    # ---------------- coin door
    cx, cz, cw, ch = L.COIN_DOOR
    door = obj(name + "_door", bpy.data.meshes.get("Door") or _box_mesh("Door", (cw, 0.008, ch), uv=True), coll,
               loc=(cx, -0.004, cz), parent=root, mat=M.coindoor)
    for sx in (-0.06, 0.06):
        obj(name + "_coinlt", bpy.data.meshes.get("CoinLt") or _box_mesh("CoinLt", (0.03, 0.012, 0.042)), coll,
            loc=(cx + sx, -0.010, cz + 0.035), parent=root, mat=M.coinlight)
        obj(name + "_slot", bpy.data.meshes.get("Slot") or _box_mesh("Slot", (0.004, 0.004, 0.03)), coll,
            loc=(cx + sx, -0.0165, cz + 0.035), parent=root, mat=M.slot)
        obj(name + "_ret2", bpy.data.meshes.get("CoinRet") or _box_mesh("CoinRet", (0.05, 0.02, 0.03)), coll,
            loc=(cx + sx, -0.012, cz - 0.10), parent=root, mat=M.housing)
    obj(name + "_lock", _shared_cyl("Lock", 0.009, 0.01), coll, loc=(cx, -0.008, cz - 0.03),
        rot=(math.pi / 2, 0, 0), parent=root, mat=M.chrome)
    return root


def _box_mesh(name, s, uv=False):
    mb = MB()
    mb.box((0, 0, 0), s)
    if uv:  # front face full UV (for the coin-door plate decal)
        pass
    me = bpy.data.meshes.new(name)
    me.from_pydata(mb.v, [], mb.f)
    uvl = me.uv_layers.new(name="UVMap")
    # planar xz projection -> front decals map 0..1
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    V = np.array(mb.v, np.float32)[loop_v]
    u = (V[:, 0] + s[0] / 2) / s[0]
    v = (V[:, 2] + s[2] / 2) / s[2]
    uvl.data.foreach_set("uv", np.stack([u, v], 1).ravel())
    return me


def _shared_cyl(name, r, h, r2=None):
    return bpy.data.meshes.get(name) or cyl_mesh(name, r, h, r2=r2)


# ====================================================================== room
def build_room(coll, M):
    mb = MB()
    mb.quad((-12, -12, 0), (12, -12, 0), (12, 12, 0), (-12, 12, 0))
    floor = mb.build("Floor", [M.carpet], coll)
    WY = 1.06
    wall = MB()
    wall.quad((-4.3, WY, 0), (-4.3, WY, 3.0), (4.3, WY, 3.0), (4.3, WY, 0))
    wall.build("BackWall", [M.brick], coll)
    sw = MB()
    sw.quad((-4.3, -7, 0), (-4.3, -7, 3.0), (-4.3, WY, 3.0), (-4.3, WY, 0)[::1])
    sw.quad((4.3, WY, 0), (4.3, WY, 3.0), (4.3, -7, 3.0), (4.3, -7, 0))
    sw.quad((-4.3, -7, 0), (4.3, -7, 0), (4.3, -7, 3.0), (-4.3, -7, 3.0))
    sw.build("SideWalls", [M.paint], coll)
    c = MB()
    c.quad((-4.3, -7, 3.0), (-4.3, WY, 3.0), (4.3, WY, 3.0), (4.3, -7, 3.0))
    c.build("Ceiling", [M.ceiling], coll)
    # baseboard trim strip (dark) + a cyan neon line along the top of the back wall
    nl = MB()
    nl.box((0, WY - 0.03, 2.86), (8.4, 0.02, 0.018))
    nl.build("NeonLine", [M.neon_cyan], coll)
    # neon sign "ARCADE" on the back wall + "GAME ON" left wall + "PLAY" right wall
    signs = []
    for txt, loc, rot, size, mat in (("ARCADE", (0, WY - 0.04, 2.25), (math.pi / 2, 0, 0), 0.52, M.neon_pink),
                                     ("GAME ON", (-4.26, -1.2, 1.9), (math.pi / 2, 0, math.pi / 2), 0.42, M.neon_cyan),
                                     ("HIGH SCORE", (4.26, -1.6, 1.95), (math.pi / 2, 0, -math.pi / 2), 0.34, M.neon_gold)):
        cu = bpy.data.curves.new("Sign_" + txt, "FONT")
        cu.body = txt
        cu.size = size
        cu.align_x = "CENTER"
        cu.align_y = "CENTER"
        cu.bevel_depth = 0.0065
        cu.bevel_resolution = 3
        cu.fill_mode = "NONE"
        fp = os.path.join(os.path.dirname(HERE), "..", "..", "..", "assets", "fonts", "BebasNeue-Regular.ttf")
        cu.font = bpy.data.fonts.load(os.path.abspath(fp), check_existing=True)
        cu.materials.append(mat)
        ob = bpy.data.objects.new("Sign_" + txt, cu)
        coll.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = rot
        signs.append(ob)
    # rules sign on the left wall
    rs = MB()
    rs.quad((-4.28, -2.6, 1.2), (-4.28, -3.4, 1.2), (-4.28, -3.4, 1.7), (-4.28, -2.6, 1.7))
    rs.build("RulesSign", [M.sign], coll)
    # ceiling track with 4 moving-head housings
    tr = MB()
    tr.box((0, -0.45, 2.97), (6.0, 0.05, 0.05))
    tr.build("Track", [M.housing], coll)
    # change machine on the right wall
    ch = MB()
    ch.box((3.95, -2.8, 0.8), (0.6, 0.5, 1.6))
    ch.build("ChangeMachine", [M.housing], coll)
    cp = MB()
    cp.quad((3.65, -2.95, 1.0), (3.65, -2.65, 1.0), (3.65, -2.65, 1.3), (3.65, -2.95, 1.3))
    cp.build("ChangeLight", [M.neon_gold], coll)
    return floor, signs


# ====================================================================== audio -> control arrays
def control_arrays(d, stats):
    n = len(d["rms"])
    C = {}
    hot = d["T_hot"].astype(float)
    big = d["T_big"].astype(float)
    pre = d["T_pre"].astype(float)
    black = d["T_black"].astype(float)
    kage, sage = d["T_kage"], d["T_sage"]
    hot_s = envelope(hot, 0.5, 18)
    sub = envelope(d["bass"].astype(float), 1.0, 6.0)
    sub = sub / (np.percentile(sub, 98) + 1e-6)
    sub = np.clip(sub, 0, 1)
    high = d["high"].astype(float)
    hat = np.clip((high - gsmooth(high, 8)) * 5, 0, 1)
    hat = envelope(hat, 0.5, 2.0)
    beat_age = np.full(n, 10 ** 6)
    last = -10 ** 6
    bi = d["T_bi"]
    for i in range(n):
        if i > 0 and bi[i] != bi[i - 1]:
            last = i
        beat_age[i] = i - last if last > -10 ** 6 else 10 ** 6
    power = 1 - black
    C["kick"] = decay(kage, 3.5)
    C["snare"] = decay(sage, 4.0)
    C["hat"] = hat
    C["sub"] = sub
    C["hot"] = hot_s
    C["pre"] = pre
    C["black"] = black
    C["big"] = envelope(big, 0.5, 10)
    C["power"] = power
    C["flood"] = decay(d["T_entry_age"], 5.0) * hot
    sa = d["T_strobe_age"]
    C["strobe"] = np.where(hot > 0, (sa == 0) * 1.0 + (sa == 1) * 0.45, 0.0)
    C["beat"] = decay(beat_age, 5.0)
    C["neon"] = (0.32 + 0.68 * sub) * (1 - 0.7 * pre) * power * (0.85 + 0.3 * hot_s)
    C["glow"] = (0.15 + 0.85 * sub) * (0.25 + 0.75 * hot_s) * power
    flick = 0.94 + 0.06 * np.array([h01(i // 2, 77) for i in range(n)])
    C["marquee"] = power * flick * (1 + 0.35 * C["kick"] * hot_s) * (1 - 0.35 * pre)
    C["btn_kick"] = C["kick"] * (0.45 + 0.55 * hot_s) * power
    C["btn_snare"] = C["snare"] * (0.45 + 0.55 * hot_s) * power
    C["coin"] = np.clip(0.25 + 1.2 * hat, 0, 1.4) * power
    # coin inserted on the title screen (game_frames: credit at index 200) -> both coin lights flare
    C["coin"][200:212] += np.linspace(2.0, 0.0, 12)
    C["punch"] = d["punch"].astype(float)
    g = stats["game"].astype(float)
    C["game_rgb"] = g
    C["att_rgb"] = stats["attract"].astype(float)
    C["n"] = n
    return C


def rgb_norm(m):
    lum = 0.2126 * m[:, 0] + 0.7152 * m[:, 1] + 0.0722 * m[:, 2]
    col = m / (m.max(1, keepdims=True) + 1e-6)
    return np.clip(col, 0.02, 1), lum


def hue_shift_rgb(m, hue_node_value):
    """Match ShaderNodeHueSaturation(Hue=h): rotates hue by (h - 0.5) turns."""
    out = np.empty_like(m)
    dh = hue_node_value - 0.5
    for i, (r, g, b) in enumerate(m):
        h, s, v = colorsys.rgb_to_hsv(max(r, 0), max(g, 0), max(b, 0))
        out[i] = colorsys.hsv_to_rgb((h + dh) % 1.0, s, v)
    return out


# ====================================================================== camera plan
SC = None      # set in main: screen centre, bezel normal, cp centre, marquee centre


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def ease_in(u, p=2.6):
    return np.clip(u, 0, 1) ** p


def ease_out(u, p=3.0):
    return 1 - (1 - np.clip(u, 0, 1)) ** p


def V(*a):
    return np.array(a, float)


def shot_params(kind, side, idx):
    """-> dict(eye0, eye1, tgt0, tgt1, lens0, lens1, fstop, hh, ease, focus)"""
    P = SC
    s = side
    j = (h01(idx, 3) - 0.5)
    if kind == "c_wide":
        return dict(eye0=V(0.9 * s, -4.9, 1.62), eye1=V(0.35 * s, -3.1, 1.42), tgt0=V(0.15 * s, 0.2, 1.05),
                    tgt1=V(0, 0.15, 1.2), lens0=30, lens1=33, fstop=4.0, hh=0.35, ease="smooth", focus=P["sc"])
    if kind == "c_side":
        return dict(eye0=V(1.95 * s, -1.75, 0.82), eye1=V(1.35 * s, -1.25, 0.98), tgt0=V(0.2 * s, 0.3, 1.1),
                    tgt1=V(0.0, 0.25, 1.25), lens0=32, lens1=35, fstop=2.8, hh=0.35, ease="smooth", focus=P["sc"])
    if kind == "c_crane":
        return dict(eye0=V(0.42 * s, -1.0, 0.92), eye1=V(0.3 * s, -0.95, 1.5), tgt0=P["cp"], tgt1=P["sc"],
                    lens0=40, lens1=42, fstop=2.4, hh=0.3, ease="smooth", focus=None)
    if kind == "c_marq":
        return dict(eye0=V(0.55 * s, -0.95, 1.18), eye1=V(0.38 * s, -0.78, 1.26), tgt0=P["mq"] + V(0, 0, -0.05),
                    tgt1=P["mq"] + V(0, 0, -0.25), lens0=38, lens1=40, fstop=2.4, hh=0.3, ease="smooth", focus=None)
    if kind == "c_player":
        return dict(eye0=V(0.06 * s, -0.95, 1.64), eye1=V(0.03 * s, -0.72, 1.6), tgt0=P["sc"] + V(0, 0, -0.12),
                    tgt1=P["sc"] + V(0, 0, -0.08), lens0=32, lens1=34, fstop=2.8, hh=0.4, ease="smooth", focus=P["sc"])
    if kind == "h_front":
        a0, a1 = math.radians(32 * s), math.radians(14 * s)
        r0, r1 = 1.45, 1.15
        c = P["sc"] + V(0, 0, -0.18)
        e0 = c + V(math.sin(a0) * r0, -math.cos(a0) * r0, -0.25)
        e1 = c + V(math.sin(a1) * r1, -math.cos(a1) * r1, -0.2)
        return dict(eye0=e0, eye1=e1, tgt0=c, tgt1=c + V(0, 0, 0.05), lens0=28, lens1=30, fstop=2.8, hh=1.0,
                    ease="out", focus=P["sc"])
    if kind == "h_cp":
        return dict(eye0=V(0.5 * s, -0.78, 1.16), eye1=V(0.36 * s, -0.62, 1.1), tgt0=P["cp"] + V(-0.04 * s, 0.0, 0.0),
                    tgt1=P["cp"] + V(0.02 * s, 0.02, 0.0), lens0=40, lens1=44, fstop=2.0, hh=0.9, ease="out", focus=None)
    if kind == "h_row":
        return dict(eye0=V(3.35 * s, -1.25, 1.35), eye1=V(2.85 * s, -1.0, 1.25), tgt0=V(-0.6 * s, 0.25, 1.15),
                    tgt1=V(-0.2 * s, 0.25, 1.2), lens0=26, lens1=30, fstop=4.0, hh=0.9, ease="out", focus=P["sc"])
    if kind == "h_screen":
        n = P["n"]
        e0 = P["sc"] + n * 0.95 + V(0.18 * s, 0, 0.02)
        e1 = P["sc"] + n * 0.70 + V(0.06 * s, 0, 0.0)
        return dict(eye0=e0, eye1=e1, tgt0=P["sc"] + V(0.02 * s, 0, 0), tgt1=P["sc"], lens0=34, lens1=36, fstop=2.8,
                    hh=1.1, ease="out", focus=P["sc"])
    if kind == "h_low":
        return dict(eye0=V(0.4 * s, -1.25, 0.46), eye1=V(0.26 * s, -1.02, 0.5), tgt0=V(0.06 * s, 0.1, 1.3),
                    tgt1=V(0, 0.1, 1.36), lens0=24, lens1=26, fstop=3.2, hh=0.9, ease="out", focus=P["sc"])
    raise KeyError(kind)


def plan_camera(d, C):
    n = C["n"]
    segs = [tuple(int(v) for v in s) for s in d["segs"]]
    pre = d["T_pre"]
    black = d["T_black"].astype(bool)
    bars = sorted(set(int(b) for b in d["T_bar_start"]))
    entries = [int(e) for e in d["entries"]]
    big0 = next(s0 for s0, s1, h in segs if h and d["T_big"][s0])
    end_clear = next(s1 for s0, s1, h in segs if h and d["T_big"][s0])
    shots = []                         # (start, end, kind, side, extra)

    def bars_in(a, b):
        return [x for x in bars if a < x < b]

    calm_cycle = ["c_marq", "c_player", "c_side", "c_crane"]
    hot_cycle = ["h_front", "h_cp", "h_row", "h_screen", "h_low"]
    big_cycle = ["h_screen", "h_front", "h_screen", "h_low", "h_screen", "h_cp", "h_screen", "h_row"]
    ci = hi = 0
    side = 1
    for (a, b, h) in segs:
        if h:
            # entry: pull back OUT of the glass, 1 bar
            first = next((x for x in bars_in(a, b) if x - a >= 30), b)
            shots.append((a, first, "entry", side, None))
            cuts = [first]
            for x in bars_in(first, b):
                if x - cuts[-1] >= 90 and b - x >= 45:
                    cuts.append(x)
            cuts.append(b)
            cyc = big_cycle if d["T_big"][a] else hot_cycle
            for k in range(len(cuts) - 1):
                side = -side
                kind = cyc[(hi if not d["T_big"][a] else k) % len(cyc)]
                hi += 0 if d["T_big"][a] else 1
                shots.append((cuts[k], cuts[k + 1], kind, side, None))
            continue
        if a >= end_clear:                     # outro
            shots.append((a, b, "outro", side, None))
            continue
        nxt = next((e for e in entries if e >= b), None)
        pre_idx = np.nonzero(pre[a:b] > 0)[0]
        if nxt is None or len(pre_idx) == 0:
            shots.append((a, b, "c_wide", side, None))
            continue
        p0 = a + int(pre_idx[0])
        cand = [x for x in bars if a <= x <= p0 - 90]
        push0 = cand[-1] if cand else a
        if push0 - a < 120:
            push0 = a
        # calm shots in [a, push0)
        if push0 > a:
            if a == 0:
                cuts = [0] + [x for x in (248, 535) if x < push0 - 120] + [push0]
                kinds = ["c_wide", "c_side", "c_crane"]
            else:
                cuts, kinds = [a], []
                for x in bars_in(a, push0):
                    if x - cuts[-1] >= 150 and push0 - x >= 120:
                        cuts.append(x)
                cuts.append(push0)
                kinds = [calm_cycle[(ci + k) % len(calm_cycle)] for k in range(len(cuts) - 1)]
                ci += len(cuts) - 1
            for k in range(len(cuts) - 1):
                side = -side
                shots.append((cuts[k], cuts[k + 1], kinds[k], side, None))
        side = -side
        shots.append((push0, b, "push", side, dict(black0=next((i for i in range(push0, b) if black[i]), b))))
    # ---- per-frame arrays
    eye = np.zeros((n, 3))
    tgt = np.zeros((n, 3))
    lens = np.zeros(n)
    fstop = np.zeros(n)
    focus = np.zeros(n)
    hh_amp = np.zeros(n)
    cut_frames = []
    P = SC
    nrm = P["n"]
    for si, (a, b, kind, side, ex) in enumerate(shots):
        if a > 0:
            cut_frames.append(a)              # scene frame a (index a-1) holds: last frame of prev shot
        T = np.arange(a, b)
        u = (T - a) / max(b - a - 1, 1)
        if kind == "push":
            # the build: a slow dolly that ACCELERATES into the CRT (anticipation), arriving at the
            # glass on the blackout; during the half-beat of static it creeps the last centimetres.
            b0 = ex["black0"]
            dur = b - a
            start_d = 1.15 + min(dur, 300) / 300 * 0.9
            e_start = P["sc"] + nrm * start_d + V(0.22 * side, 0, 0.05)
            e_end = P["sc"] + nrm * 0.56
            uu = np.clip((T - a) / max(b0 - a, 1), 0, 1)
            k = ease_in(uu, 2.4)
            k = np.where(T >= b0, 1.0, k)
            e = e_start[None] * (1 - k[:, None]) + e_end[None] * k[:, None]
            creep = np.where(T >= b0, (T - b0 + 1) / max(b - b0, 1), 0.0)
            e = e - nrm[None] * (0.06 * creep)[:, None]
            eye[a:b] = e
            tgt[a:b] = P["sc"][None] + V(0, 0, -0.02)
            lens[a:b] = 32 + 3 * k
            fstop[a:b] = 2.8
            focus[a:b] = np.linalg.norm(e - P["sc"][None], axis=1)
            hh_amp[a:b] = 0.35 + 0.5 * k
            continue
        if kind == "entry":
            # drop entry: we start INSIDE the screen (the game fills the frame) and get blown back
            # out through the glass into the room, which is flooded with the game's light.
            e_in = P["sc"] + nrm * 0.50
            e_out = P["sc"] + nrm * 1.55 + V(0.55 * side, 0, -0.22)
            k = ease_out(u, 4.0)
            e = e_in[None] * (1 - k[:, None]) + e_out[None] * k[:, None]
            eye[a:b] = e
            tgt[a:b] = P["sc"][None] * (1 - k[:, None]) + (P["sc"] + V(0, 0, -0.25))[None] * k[:, None]
            lens[a:b] = 35 - 7 * k
            fstop[a:b] = 2.8
            focus[a:b] = np.linalg.norm(e - P["sc"][None], axis=1)
            hh_amp[a:b] = 0.6 + 0.5 * k
            continue
        if kind == "outro":
            e0, e1 = P["sc"] + nrm * 0.7, P["sc"] + nrm * 2.6 + V(0.3, 0, 0.15)
            k = smooth(u)
            e = e0[None] * (1 - k[:, None]) + e1[None] * k[:, None]
            eye[a:b] = e
            tgt[a:b] = P["sc"][None] + V(0, 0, -0.1)
            lens[a:b] = 35
            fstop[a:b] = 2.8
            focus[a:b] = np.linalg.norm(e - P["sc"][None], axis=1)
            hh_amp[a:b] = 0.3
            continue
        sp = shot_params(kind, side, si)
        k = smooth(u) if sp["ease"] == "smooth" else ease_out(u, 2.2)
        e = sp["eye0"][None] * (1 - k[:, None]) + sp["eye1"][None] * k[:, None]
        t = sp["tgt0"][None] * (1 - k[:, None]) + sp["tgt1"][None] * k[:, None]
        eye[a:b], tgt[a:b] = e, t
        lens[a:b] = sp["lens0"] + (sp["lens1"] - sp["lens0"]) * k
        fstop[a:b] = sp["fstop"]
        fp = sp["focus"]
        focus[a:b] = np.linalg.norm(e - (t if fp is None else fp[None]), axis=1)
        hh_amp[a:b] = sp["hh"]
    # ---- handheld: low-frequency drift (calm) + operator energy (drops) + kick shake
    tt = np.arange(n) / FPS
    hot = C["hot"]
    amp_pos = (0.0035 + 0.011 * hot) * hh_amp
    amp_rot = (0.12 + 0.45 * hot) * hh_amp            # degrees
    rng = np.random.default_rng(7)
    def wob(seed):
        r = np.random.default_rng(seed)
        f = r.uniform(0.15, 0.45, 3) * np.array([1, 2.3, 4.1])
        ph = r.uniform(0, 6.28, 3)
        w = np.array([1.0, 0.45, 0.2])
        return sum(w[q] * np.sin(2 * np.pi * f[q] * tt + ph[q]) for q in range(3)) / w.sum()
    jitter = np.stack([wob(1), wob(2), wob(3)], 1) * amp_pos[:, None]
    eye = eye + jitter
    kick = C["kick"] * hot * (1 + 0.6 * C["big"])
    # the shake direction changes per kick, not per frame
    kidx = d["T_kidx"]
    kdir = np.array([[h01(int(k), 11) - 0.5, h01(int(k), 12) - 0.5, h01(int(k), 13) - 0.5] for k in kidx])
    osc = np.cos(np.minimum(d["T_kage"], 50) * 1.9)
    eye = eye + kdir * (0.02 * kick * osc)[:, None]
    roll = wob(4) * amp_rot * 0.6 + (kdir[:, 0] * 2.2 * kick * osc)
    pitch_j = wob(5) * amp_rot
    yaw_j = wob(6) * amp_rot
    # punch-in on the sung "bump"
    lens = lens * (1 + 0.45 * (C["punch"] - 1))
    return dict(eye=eye, tgt=tgt, lens=lens, fstop=fstop, focus=focus, roll=roll, pitch=pitch_j, yaw=yaw_j,
                cuts=cut_frames, shots=shots)


def bake_camera(cam_ob, plan, n):
    eye, tgt = plan["eye"], plan["tgt"]
    rots = np.zeros((n, 3))
    prev = None
    for i in range(n):
        dvec = Vector(tgt[i] - eye[i])
        q = dvec.to_track_quat("-Z", "Y")
        q = q @ Quaternion((0, 0, 1), math.radians(plan["roll"][i]))
        q = q @ Quaternion((1, 0, 0), math.radians(plan["pitch"][i]))
        q = q @ Quaternion((0, 1, 0), math.radians(plan["yaw"][i]))
        e = q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")
        rots[i] = e[:]
        prev = e
    hold = np.array(plan["cuts"], int)                       # scene frame == last frame of the old shot
    for ax in range(3):
        bake(cam_ob, "location", eye[:, ax], index=ax, const_frames=hold)
        bake(cam_ob, "rotation_euler", rots[:, ax], index=ax, const_frames=hold)
    cam = cam_ob.data
    bake(cam, "lens", plan["lens"], const_frames=hold)
    bake(cam, "dof.focus_distance", plan["focus"], const_frames=hold)
    bake(cam, "dof.aperture_fstop", plan["fstop"], const_frames=hold)


# ====================================================================== lights
def area_light(coll, name, size, loc, direction, color=(1, 1, 1), energy=10, shape="RECTANGLE", shadow=True,
               spread=180, volume=0.08):
    L_ = bpy.data.lights.new(name, "AREA")
    L_.shape = shape
    if shape in ("RECTANGLE", "ELLIPSE"):
        L_.size, L_.size_y = size
    else:
        L_.size = size[0]
    L_.color = color
    L_.energy = energy
    L_.use_shadow = shadow
    L_.spread = math.radians(spread)
    L_.volume_factor = volume          # staging: the haze should catch the BEAMS, not every wash light
    ob = bpy.data.objects.new(name, L_)
    coll.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    return ob


def bake_light_color(light, rgb):
    for ax in range(3):
        bake(light, "color", rgb[:, ax], index=ax)


# ====================================================================== scene settings
def setup_render(sc, n):
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.fps, sc.render.fps_base = FPS, 1.0
    sc.frame_start, sc.frame_end = 1, n
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.resolution_percentage = 100
    e = sc.eevee
    e.taa_render_samples = 32
    e.use_raytracing = True
    e.ray_tracing_method = "SCREEN"
    o = e.ray_tracing_options
    o.resolution_scale = "2"
    o.trace_max_roughness = 0.45
    o.use_denoise = True
    e.use_fast_gi = True
    e.fast_gi_method = "GLOBAL_ILLUMINATION"
    e.fast_gi_resolution = "2"
    e.fast_gi_distance = 2.0
    e.use_shadows = True
    e.shadow_ray_count = 1
    e.shadow_step_count = 6
    e.volumetric_start = 0.1
    e.volumetric_end = 16.0
    e.volumetric_tile_size = "4"
    e.volumetric_samples = 48
    e.volumetric_sample_distribution = 0.85
    e.use_volumetric_shadows = True
    e.volumetric_shadow_samples = 12
    e.clamp_surface_indirect = 10.0
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    sc.render.motion_blur_position = "START"
    e.motion_blur_steps = 1
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    for look in ("AgX - Punchy", "Punchy", "AgX - Medium High Contrast"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            pass
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.render.image_settings.compression = 30
    sc.render.use_overwrite = False
    sc.render.use_placeholder = False
    # compositor: bloom + a breath of lens dispersion
    ng = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    gl = ng.nodes.new("CompositorNodeGlare")
    gl.inputs["Type"].default_value = "Bloom"
    gl.inputs["Quality"].default_value = "High"
    gl.inputs["Threshold"].default_value = 1.2
    gl.inputs["Strength"].default_value = 0.3
    gl.inputs["Size"].default_value = 0.45
    ld = ng.nodes.new("CompositorNodeLensdist")
    ld.inputs["Dispersion"].default_value = 0.012
    ld.inputs["Distortion"].default_value = -0.008
    go = ng.nodes.new("NodeGroupOutput")
    ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
    ng.links.new(gl.outputs["Image"], ld.inputs["Image"])
    ng.links.new(ld.outputs["Image"], go.inputs["Image"])
    sc.compositing_node_group = ng
    sc.render.use_compositing = True


def setup_world(sc):
    w = bpy.data.worlds.new("ArcadeWorld")
    sc.world = w
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.0025, 0.0022, 0.004, 1)
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    return w


def build_haze(coll):
    """Room-sized haze volume (object volume: scatters every local light, unlike a world volume);
    a slow 3-D noise makes it uneven, like real fog-machine haze."""
    mb = MB()
    mb.box((0, -2.97, 1.5), (8.58, 8.06, 2.98))
    ob = mb.build("Haze", [], coll)
    m, nt, out = new_mat("Haze")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.6
    nz.inputs["Detail"].default_value = 2.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["From Min"].default_value = 0.3
    rr.inputs["From Max"].default_value = 0.7
    rr.inputs["To Min"].default_value = 0.012
    rr.inputs["To Max"].default_value = 0.045
    nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Anisotropy"].default_value = 0.5
    vol.inputs["Color"].default_value = (0.92, 0.92, 1.0, 1)
    nt.links.new(rr.outputs["Result"], vol.inputs["Density"])
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    ob.data.materials.append(m)
    return ob


# ====================================================================== main
def main():
    global SC
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    d = np.load(args.timeline)
    stats = np.load(os.path.join(args.assets, "screen_stats.npz"))
    n = len(d["rms"])
    C = control_arrays(d, stats)
    setup_render(sc, n)
    setup_world(sc)
    coll = sc.collection
    aud_ng = make_aud_group()
    M = build_materials(args.assets, n)
    # ---------------- row of cabinets
    rng_cols = [(1.0, 0.18, 0.04), (0.05, 0.7, 1.0), (1.0, 0.05, 0.6), (0.2, 1.0, 0.3), (1.0, 0.7, 0.05), (0.6, 0.2, 1.0)]
    cabs = []
    hero = build_cabinet(coll, M, "Hero", (0, 0, 0), 0, True, tm_col=(1.0, 0.16, 0.03))
    cabs.append(("Hero", hero, 0.0, True, 0, False))
    k = 0
    for j in range(1, L.N_EACH_SIDE + 1):
        for sgn in (-1, 1):
            x = sgn * j * L.SPACING
            yaw = math.radians((h01(j, sgn) - 0.5) * 3.0)
            hue = (0.5 + 0.17 * (k + 1)) % 1.0
            name = f"Cab{k}"
            ob = build_cabinet(coll, M, name, (x, 0.02 * (h01(k, 9) - 0.5), 0), yaw, False, k=k,
                               tm_col=rng_cols[k % 6], hue=hue, variant=k % 2, mirror=False)
            cabs.append((name, ob, hue, False, k % 2, False))
            k += 1
    # a second row behind the camera (backs to the room centre, screens facing the hero row)
    for j in range(5):
        x = (j - 2) * L.SPACING
        name = f"Back{j}"
        hue = (0.1 + 0.21 * j) % 1.0
        ob = build_cabinet(coll, M, name, (x + 0.37, -6.2, 0), math.pi, False, k=j + 1, tm_col=rng_cols[(j + 2) % 6],
                           hue=hue, variant=j % 2)
        cabs.append((name, ob, hue, False, j % 2, False))
    floor, signs = build_room(coll, M)
    build_haze(coll)
    bpy.context.view_layer.update()
    # ---------------- points of interest (hero cabinet)
    (dy, dz), (ny, nz), blen = yz_n(*L.BEZEL)
    sc_pt = frame_on(L.BEZEL, 0, blen / 2, 0)
    SC = dict(sc=np.array(sc_pt[:]), n=np.array((0, ny, nz)), cp=np.array(frame_on(L.CP_TOP, -0.02, L.CP_LEN / 2, 0.02)[:]),
              mq=np.array((0, L.MARQUEE[0][0], (L.MARQUEE[0][1] + L.MARQUEE[1][1]) / 2)))
    # ---------------- AUD group values
    for key in AUD_KEYS:
        bake(aud_ng, f'nodes["{key}"].outputs[0].default_value', C[key])
    # ---------------- lights
    lights = bpy.data.collections.new("Lights")
    coll.children.link(lights)
    nrm = Vector(SC["n"])
    g_col, g_lum = rgb_norm(C["game_rgb"])
    # hero CRT spill: colour = the game's mean colour; the white flash at a drop entry FLOODS the room
    hs = area_light(lights, "L_HeroScreen", (L.SCREEN_W, L.SCREEN_H), Vector(SC["sc"]) + nrm * 0.03, nrm,
                    energy=10, shadow=True, volume=0.5)
    hs.rotation_euler = nrm.to_track_quat("-Z", "Y").to_euler()
    bake(hs.data, "energy", 160 * g_lum * (1 + 0.8 * C["flood"]))
    bake_light_color(hs.data, g_col)
    # a broad "flood" bounce from the screen toward the room on entries
    fl = area_light(lights, "L_Flood", (1.6, 1.2), Vector(SC["sc"]) + nrm * 0.6 + Vector((0, 0, 0.3)), nrm + Vector((0, 0, -0.3)),
                    energy=0, shadow=False, volume=0.2)
    bake(fl.data, "energy", 220 * C["flood"] * np.clip(g_lum * 3, 0.3, 1.0))
    bake_light_color(fl.data, g_col)
    # other screens
    for (name, ob, hue, is_hero, variant, mirror) in cabs:
        if is_hero:
            continue
        a_rgb = hue_shift_rgb(C["att_rgb"][variant], hue)
        a_col, a_lum = rgb_norm(a_rgb)
        M_ = ob.matrix_world
        p = M_ @ Vector(SC["sc"])
        nn = (M_.to_3x3() @ nrm).normalized()
        al = area_light(lights, f"L_{name}_scr", (L.SCREEN_W, L.SCREEN_H), p + nn * 0.03, nn, energy=8,
                        shadow=name.startswith("Cab") and abs(ob.location.x) < 1.6)
        bake(al.data, "energy", 110 * a_lum * C["power"])
        bake_light_color(al.data, a_col)
        mq = area_light(lights, f"L_{name}_mq", (0.55, 0.1), M_ @ Vector((0, -0.05, 1.66)), M_.to_3x3() @ Vector((0, -0.35, -1)),
                        color=(1, 0.9, 0.8), energy=2.5, shadow=False)
        mq.data.specular_factor = 0.0           # no fluorescent strip reflected across the screens
        bake(mq.data, "energy", 3.0 * C["marquee"])
    # hero marquee down-light (lights the bezel + control panel)
    mq = area_light(lights, "L_Hero_mq", (0.55, 0.08), Vector((0, -0.04, 1.655)), Vector((0, -0.45, -1)),
                    color=(1, 0.92, 0.82), energy=5, shadow=True)
    mq.data.specular_factor = 0.0               # its reflection hid the score line on the CRT
    bake(mq.data, "energy", 6.0 * C["marquee"])
    # control panel LED spill (kick/snare colours)
    cpl = bpy.data.lights.new("L_CP", "POINT")
    cpl.shadow_soft_size = 0.03
    cpl.volume_factor = 0.0
    cpl_ob = bpy.data.objects.new("L_CP", cpl)
    lights.objects.link(cpl_ob)
    cpl_ob.location = frame_on(L.CP_TOP, 0.06, 0.15, 0.05)
    bake(cpl, "energy", 2.5 * C["btn_kick"] + 1.6 * C["btn_snare"] + 0.15)
    cmix = np.stack([0.4 * C["btn_kick"] + 0.3 * C["btn_snare"] + 0.05, 0.12 * C["btn_kick"] + 0.5 * C["btn_snare"] + 0.05,
                     0.3 * C["btn_kick"] + 0.6 * C["btn_snare"] + 0.05], 1)
    bake_light_color(cpl, cmix / cmix.max(1, keepdims=True))
    # coin-door glow (hats)
    cdl = bpy.data.lights.new("L_Coin", "POINT")
    cdl.color = (1, 0.15, 0.05)
    cdl.shadow_soft_size = 0.02
    cdl.volume_factor = 0.0
    cdo = bpy.data.objects.new("L_Coin", cdl)
    lights.objects.link(cdo)
    cdo.location = (0, -0.06, L.COIN_DOOR[1] + 0.035)
    bake(cdl, "energy", 0.6 * C["coin"])
    # neon washes: ARCADE (pink) on the back wall, GAME ON (cyan) left, HIGH SCORE (gold) right
    for nm, loc, dirv, sz, col, e0 in (("L_NeonArcade", (0, 0.95, 2.25), (0, -1, -0.25), (2.4, 0.5), (1, 0.1, 0.45), 160),
                                       ("L_NeonArcadeWall", (0, 0.85, 2.25), (0, 1, 0), (2.4, 0.5), (1, 0.1, 0.45), 30),
                                       ("L_NeonLine", (0, 0.95, 2.84), (0, -1, -0.6), (8.0, 0.08), (0.05, 0.7, 1), 90),
                                       ("L_NeonGameOn", (-4.15, -1.2, 1.9), (1, 0, -0.2), (1.6, 0.4), (0.05, 0.7, 1), 70),
                                       ("L_NeonHigh", (4.15, -1.6, 1.95), (-1, 0, -0.2), (1.8, 0.35), (1, 0.55, 0.05), 55)):
        nl = area_light(lights, nm, sz, Vector(loc), Vector(dirv), color=col, energy=e0, shadow=False)
        bake(nl.data, "energy", e0 * C["neon"])
    # soft overhead house light (very dim; dies in the blackout, dips in the build)
    house = area_light(lights, "L_House", (3.0, 2.0), Vector((0, -1.3, 2.95)), Vector((0, 0.2, -1)),
                       color=(1.0, 0.82, 0.7), energy=40, shadow=True)
    bake(house.data, "energy", 26 * C["power"] * (1 - 0.6 * C["pre"]) * (1 - 0.4 * C["hot"]))
    # moving-head beams through the haze: off-ish in calm, sweeping in drops, converge red in the build
    n_ = n
    tt = np.arange(n_) / FPS
    hot, pre, big, strobe = C["hot"], C["pre"], C["big"], C["strobe"]
    beat_ph = d["T_ph"].astype(float)
    seg_id = d["T_seg_id"]
    pal = [((1.0, 0.1, 0.6), (0.1, 0.6, 1.0)), ((0.2, 1.0, 0.4), (1.0, 0.3, 0.05)), ((0.5, 0.2, 1.0), (1.0, 0.85, 0.1)),
           ((1.0, 0.05, 0.3), (0.05, 1.0, 0.9))]
    xs = [-2.2, -0.75, 0.75, 2.2]
    for bi_, x in enumerate(xs):
        spl = bpy.data.lights.new(f"L_Beam{bi_}", "SPOT")
        spl.spot_size = math.radians(12)
        spl.spot_blend = 0.25
        spl.shadow_soft_size = 0.03
        spl.volume_factor = 6.0
        spl.use_shadow = True
        so = bpy.data.objects.new(f"L_Beam{bi_}", spl)
        lights.objects.link(so)
        so.location = (x, -0.45, 2.9)
        # sweep: aim point on the floor/cabinets moving in slow figure-8s; in the build all four
        # converge on the hero cabinet (staging: one idea); on bar downbeats in drops they snap
        sweep = np.stack([x * 0.55 + 1.1 * np.sin(tt * (0.9 + 0.13 * bi_) + bi_ * 1.7),
                          -0.75 + 0.55 * np.sin(tt * (0.6 + 0.07 * bi_) + bi_),
                          0.4 + 0.6 * np.abs(np.sin(tt * 0.5 + bi_))], 1)
        calm_aim = np.tile(np.array([x * 0.6, -0.95, 0.0]), (n_, 1))
        conv = np.tile(np.array([0.0, -0.15, 0.95]), (n_, 1))
        w_hot = hot[:, None]
        aim = calm_aim * (1 - w_hot) + sweep * w_hot
        aim = aim * (1 - pre[:, None]) + conv * pre[:, None]
        rots = []
        prev = None
        for i in range(n_):
            dv = Vector(aim[i]) - so.location
            e_ = dv.to_track_quat("-Z", "Y").to_euler("XYZ", prev) if prev else dv.to_track_quat("-Z", "Y").to_euler("XYZ")
            rots.append(e_[:])
            prev = e_
        rots = np.array(rots)
        for ax in range(3):
            bake(so, "rotation_euler", rots[:, ax], index=ax)
        # colour per drop palette (alternating beams), red in the build, white on strobe hits
        pk = np.array([pal[(int(s) // 2) % len(pal)][bi_ % 2] for s in seg_id])
        red = np.tile(np.array([1.0, 0.04, 0.02]), (n_, 1))
        calm_col = np.tile(np.array([0.35, 0.45, 1.0]), (n_, 1))
        col = calm_col * (1 - w_hot) + pk * w_hot
        col = col * (1 - pre[:, None]) + red * pre[:, None]
        col = col * (1 - strobe[:, None]) + np.ones_like(col) * strobe[:, None]
        bake_light_color(spl, np.clip(col, 0, 1))
        # energy: dim in calm, alive in drops (pulse on the beat), red build ramps, blackout = 0
        beat_pulse = 0.6 + 0.4 * np.exp(-beat_ph * 4)
        en = 300 * (1 - hot) + 2600 * hot * beat_pulse * (1 + 0.4 * big) + 3200 * pre ** 1.5
        en = en * C["power"] + 7000 * strobe
        bake(spl, "energy", en)
    # back-wall shafts: two spots behind the cabinet row firing over the marquees toward the
    # camera — in haze they read as rays; they PUNCH on every kick in the drops
    kick = C["kick"]
    for bi_, x in enumerate((-1.6, 1.6)):
        spl = bpy.data.lights.new(f"L_Back{bi_}", "SPOT")
        spl.spot_size = math.radians(18)
        spl.spot_blend = 0.4
        spl.volume_factor = 14.0
        spl.use_shadow = True
        spl.shadow_soft_size = 0.05
        so = bpy.data.objects.new(f"L_Back{bi_}", spl)
        lights.objects.link(so)
        so.location = (x, 0.98, 2.75)
        aim = Vector((x * 0.35 + (0.4 if x < 0 else -0.4), -3.0, 0.6))
        so.rotation_euler = (aim - so.location).to_track_quat("-Z", "Y").to_euler()
        pk = np.array([pal[(int(s) // 2) % len(pal)][(bi_ + 1) % 2] for s in seg_id])
        red = np.tile(np.array([1.0, 0.04, 0.02]), (n_, 1))
        calm_col = np.tile(np.array([0.5, 0.35, 1.0]), (n_, 1))
        col = calm_col * (1 - hot[:, None]) + pk * hot[:, None]
        col = col * (1 - pre[:, None]) + red * pre[:, None]
        bake_light_color(spl, np.clip(col, 0, 1))
        en = 120 * (1 - hot) + hot * (500 + 2600 * kick) * (1 + 0.3 * big) + 1500 * pre ** 2 * (0.5 + 0.5 * (beat_ph < 0.5))
        bake(spl, "energy", en * C["power"])
    # ---------------- camera
    cam = bpy.data.cameras.new("Cam")
    cam.sensor_width = 36
    cam.dof.use_dof = True
    cam.dof.aperture_blades = 7
    cam.clip_start = 0.02
    cam.clip_end = 60
    cam_ob = bpy.data.objects.new("Cam", cam)
    coll.objects.link(cam_ob)
    sc.camera = cam_ob
    plan = plan_camera(d, C)
    bake_camera(cam_ob, plan, n)
    # ---------------- sound (scrubbing / convenience; render.sh muxes with ffmpeg)
    if os.path.exists(args.wav):
        se = sc.sequence_editor_create()
        se.strips.new_sound("Music", args.wav, channel=1, frame_start=1)
    # shot list for the README / review
    with open(os.path.splitext(args.out)[0] + "_shots.txt", "w") as fh:
        for (a, b, kind, side, _ex) in plan["shots"]:
            fh.write(f"{a + 1:5d}-{b:5d}  {(a) / FPS:6.2f}s  {kind:9s} side {side:+d}\n")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=args.out, compress=False)
    print(f"[build_scene] saved {args.out}: {n} frames, {len(plan['shots'])} shots, "
          f"{len(bpy.data.objects)} objects, {len(bpy.data.lights)} lights")


if __name__ == "__main__":
    main()
