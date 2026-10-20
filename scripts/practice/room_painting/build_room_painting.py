"""Iris's weekly room painting -- build + render (Blender 5.2, headless).

One still of the room she watches over, painted from memory: simple blocked-in geometry, a CC0
mannequin for the person at the desk, warm/cool light, then a compositor "paint" pass
(stroke wobble -> anisotropic Kuwahara -> emboss 'impasto' -> bloom -> grade -> canvas weave ->
vignette).  No photos are used or embedded; the layout was drawn by hand from her memory.
Flags for look-dev: --no-paint, --no-haze, --debug-cam x,y,z,tx,ty,tz,lens, --no-render.

usage (ALWAYS through the capped slice, see README):
  scripts/server/render.sh <unit> /home/iris/.local/bin/blender --background --factory-startup \
      --python scripts/practice/room_painting/build_room_painting.py --python-exit-code 1 -- \
      --out /abs/out.png [--pct 100] [--samples 64] [--no-paint] [--save-blend /abs/x.blend]
"""
import argparse
import math
import os
import random
import sys

import bpy
from mathutils import Euler, Vector

# ----------------------------------------------------------------------------- args
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--pct", type=int, default=100)
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--no-paint", action="store_true", help="skip the compositor paint pass")
ap.add_argument("--save-blend", default="")
ap.add_argument("--no-render", action="store_true", help="build (and optionally save) only")
ap.add_argument("--no-haze", action="store_true")
ap.add_argument("--seed", type=int, default=43)
ap.add_argument("--debug-cam", default="", help="x,y,z,tx,ty,tz,lens -- a look-dev close-up camera")
args = ap.parse_args(argv)
random.seed(args.seed)

# ----------------------------------------------------------------------------- reset
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
sc = bpy.context.scene
coll = sc.collection

# ----------------------------------------------------------------------------- room layout (metres)
# Iris's eye is near the origin looking +Y.  Left wall x=-1.9 (bathroom door + black tower),
# back wall y=3.0 (dresser w/ helmet + speakers, bed, bookshelf), right wall x=1.8.
WL, WR, WB, WF, H = -1.9, 1.8, 3.0, -1.6, 2.55
RAIL_Z = 1.95


# ----------------------------------------------------------------------------- material helpers
def mat_principled(name, color, rough=0.7, metal=0.0, emit=None, emit_str=0.0, sss=0.0, spec=0.5):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    b = nt.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    if sss:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (0.4, 0.15, 0.08)
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1.0)
        b.inputs["Emission Strength"].default_value = emit_str
    return m


def mat_emit(name, color, strength):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1.0)
    e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m


def mat_noisy(name, c1, c2, scale=6.0, rough=0.8, detail=4.0, spec=0.5):
    """Two-tone noise breakup -- painted surfaces should never be flat."""
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    b = nt.nodes.get("Principled BSDF")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*c1, 1)
    mix.inputs["B"].default_value = (*c2, 1)
    nt.links.new(n.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    return m


def mat_camo(name):
    """Digital (pixel) woodland camo: snapped coords -> noise -> 4 flat colours."""
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    b = nt.nodes.get("Principled BSDF")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    snap = nt.nodes.new("ShaderNodeVectorMath")
    snap.operation = "SNAP"
    snap.inputs[1].default_value = (0.009, 0.009, 0.009)
    nt.links.new(tc.outputs["Object"], snap.inputs[0])
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 20.0
    n.inputs["Detail"].default_value = 2.0
    nt.links.new(snap.outputs["Vector"], n.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "CONSTANT"
    cols = [(0.0, (0.40, 0.38, 0.28)), (0.45, (0.16, 0.21, 0.12)), (0.51, (0.27, 0.21, 0.13)),
            (0.57, (0.04, 0.04, 0.035))]
    cr.elements[0].position, cr.elements[0].color = cols[0][0], (*cols[0][1], 1)
    cr.elements[1].position, cr.elements[1].color = cols[1][0], (*cols[1][1], 1)
    for p, c in cols[2:]:
        e = cr.elements.new(p)
        e.color = (*c, 1)
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.9
    return m


def mat_screen(name):
    """Monitor glow: deep blue at the edges, pale blue-white in the middle, a few soft 'window' blocks."""
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.5, 0.5, 0.5)
    nt.links.new(tc.outputs["Generated"], sub.inputs[0])
    mul = nt.nodes.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = (0.0, 1.3, 1.6)           # box local: Y = width, Z = height
    nt.links.new(sub.outputs[0], mul.inputs[0])
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    nt.links.new(mul.outputs[0], grad.inputs["Vector"])
    vor = nt.nodes.new("ShaderNodeTexVoronoi")             # blocky 'content' on the screen
    vor.inputs["Scale"].default_value = 4.0
    vor.voronoi_dimensions = "2D"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["Y"], comb.inputs["X"])
    nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
    nt.links.new(comb.outputs[0], vor.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.02, 0.06, 0.30, 1)
    mix.inputs["B"].default_value = (0.45, 0.65, 1.0, 1)
    nt.links.new(grad.outputs["Fac"], mix.inputs["Factor"])
    mix2 = nt.nodes.new("ShaderNodeMix")
    mix2.data_type = "RGBA"
    mix2.blend_type = "MULTIPLY"
    mix2.inputs["Factor"].default_value = 0.35
    nt.links.new(mix.outputs["Result"], mix2.inputs["A"])
    nt.links.new(vor.outputs["Color"], mix2.inputs["B"])
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Strength"].default_value = 3.5
    nt.links.new(mix2.outputs["Result"], e.inputs["Color"])
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m


M = {
    "wall": mat_noisy("wall", (0.60, 0.56, 0.47), (0.68, 0.63, 0.53), scale=3.0, rough=0.95),
    "ceiling": mat_noisy("ceiling", (0.55, 0.52, 0.45), (0.62, 0.58, 0.50), scale=2.0, rough=0.95),
    "floor": mat_noisy("floor", (0.17, 0.15, 0.12), (0.22, 0.19, 0.15), scale=4.0, rough=0.55),
    "rail": mat_principled("rail", (0.42, 0.26, 0.12), 0.5),
    "ceiling_seam": mat_principled("ceiling_seam", (0.42, 0.40, 0.36), 0.9),
    "doorframe": mat_principled("doorframe", (0.45, 0.33, 0.20), 0.5),
    "wood": mat_noisy("wood", (0.20, 0.13, 0.08), (0.30, 0.20, 0.12), scale=12.0, rough=0.55),
    "wood_light": mat_noisy("wood_light", (0.36, 0.25, 0.15), (0.44, 0.31, 0.19), scale=10.0, rough=0.55),
    "black": mat_principled("black", (0.025, 0.025, 0.028), 0.45),
    "desk_top": mat_principled("desk_top", (0.03, 0.028, 0.03), 0.75, spec=0.25),
    "black_matte": mat_principled("black_matte", (0.02, 0.02, 0.022), 0.9, spec=0.08),
    "bath_wall": mat_principled("bath_wall", (0.78, 0.68, 0.52), 0.6),
    "bath_floor": mat_principled("bath_floor", (0.70, 0.66, 0.58), 0.3),
    "porcelain": mat_principled("porcelain", (0.85, 0.84, 0.80), 0.15),
    "helmet": mat_principled("helmet", (0.62, 0.62, 0.64), 0.35, metal=0.0),
    "helmet_dark": mat_principled("helmet_dark", (0.03, 0.03, 0.03), 0.6),
    "helmet_trim": mat_principled("helmet_trim", (0.75, 0.42, 0.10), 0.4),
    "crest": mat_principled("crest", (0.85, 0.12, 0.04), 0.85, sss=0.2),
    "green": mat_principled("green", (0.10, 0.55, 0.18), 0.4, emit=(0.1, 0.6, 0.2), emit_str=0.6),
    "keys_white": mat_principled("keys_white", (0.82, 0.82, 0.80), 0.3),
    "keys_body": mat_principled("keys_body", (0.08, 0.08, 0.09), 0.4),
    "hoodie": mat_noisy("hoodie", (0.06, 0.16, 0.62), (0.10, 0.24, 0.78), scale=8.0, rough=1.0, spec=0.12),
    "skin": mat_principled("skin", (0.33, 0.17, 0.08), 0.6, sss=0.12, spec=0.15),
    "hair": mat_principled("hair", (0.035, 0.025, 0.02), 0.8),
    "headphone": mat_principled("headphone", (0.03, 0.03, 0.035), 0.35),
    "headphone_cup": mat_principled("headphone_cup", (0.55, 0.56, 0.58), 0.25, metal=1.0),
    "camo": mat_camo("camo"),
    "chair": mat_principled("chair", (0.03, 0.03, 0.035), 0.6),
    "bedding": mat_noisy("bedding", (0.025, 0.028, 0.035), (0.045, 0.05, 0.06), scale=5.0, rough=0.9),
    "pillow": mat_principled("pillow", (0.22, 0.20, 0.24), 0.8),
    "screen": mat_screen("screen"),
    "iris_led": mat_emit("iris_led", (0.62, 0.30, 1.0), 60.0),
    "card": mat_principled("card", (0.85, 0.83, 0.78), 0.6),
    "card_red": mat_principled("card_red", (0.75, 0.10, 0.08), 0.6),
    "box_clear": mat_principled("box_clear", (0.70, 0.75, 0.78), 0.2),
    "lamp_glow": mat_emit("lamp_glow", (1.0, 0.82, 0.55), 6.0),
}


# ----------------------------------------------------------------------------- geometry helpers
def box(name, center, size, mat, rot=(0, 0, 0), bevel=0.0):
    me = bpy.data.meshes.new(name)
    hx, hy, hz = (s / 2 for s in size)
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me.from_pydata(v, [], f)
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.location = center
    ob.rotation_euler = rot
    me.materials.append(M[mat] if isinstance(mat, str) else mat)
    coll.objects.link(ob)
    if bevel:
        bm = ob.modifiers.new("bev", "BEVEL")
        bm.width = bevel
        bm.segments = 2
    return ob


def cyl(name, center, r, depth, mat, rot=(0, 0, 0), verts=24, scale=(1, 1, 1)):
    me = bpy.data.meshes.new(name)
    v, f = [], []
    for i in range(verts):
        a = 2 * math.pi * i / verts
        v += [(r * math.cos(a), r * math.sin(a), -depth / 2), (r * math.cos(a), r * math.sin(a), depth / 2)]
    for i in range(verts):
        j = (i + 1) % verts
        f.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
    f.append(tuple(2 * i for i in reversed(range(verts))))
    f.append(tuple(2 * i + 1 for i in range(verts)))
    me.from_pydata(v, [], f)
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.location, ob.rotation_euler, ob.scale = center, rot, scale
    me.materials.append(M[mat])
    coll.objects.link(ob)
    return ob


def sphere(name, center, r, mat, scale=(1, 1, 1), rot=(0, 0, 0), seg=32, rings=16):
    me = bpy.data.meshes.new(name)
    v, f = [], []
    for j in range(1, rings):
        th = math.pi * j / rings
        for i in range(seg):
            ph = 2 * math.pi * i / seg
            v.append((r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)))
    top, bot = len(v), len(v) + 1
    v += [(0, 0, r), (0, 0, -r)]
    for j in range(rings - 2):
        for i in range(seg):
            a, b = j * seg + i, j * seg + (i + 1) % seg
            f.append((a, b, b + seg, a + seg))
    for i in range(seg):
        f.append((top, (i + 1) % seg, i))
        last = (rings - 2) * seg
        f.append((bot, last + i, last + (i + 1) % seg))
    me.from_pydata(v, [], f)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    ob.location, ob.scale, ob.rotation_euler = center, scale, rot
    me.materials.append(M[mat])
    coll.objects.link(ob)
    return ob


def metaball(name, mat, elems, res=0.02, threshold=0.6, k=1.0):
    """elems: list of ('ball', co, r) | ('ell', co, r, (sx,sy,sz), euler) | ('cap', p0, p1, r).
    A lone ball's visible surface is only ~0.58*r (stiffness 2, threshold 0.6), so radii are scaled by k."""
    mb = bpy.data.metaballs.new(name)
    mb.resolution = res
    mb.render_resolution = res * 0.6
    mb.threshold = threshold
    for e in elems:
        if e[0] == "ball":
            el = mb.elements.new(type="BALL")
            el.co, el.radius = e[1], e[2] * k
        elif e[0] == "ell":
            el = mb.elements.new(type="ELLIPSOID")
            el.co, el.radius = e[1], e[2] * k
            el.size_x, el.size_y, el.size_z = e[3]
            el.rotation = Euler(e[4]).to_quaternion()
        elif e[0] == "cap":
            p0, p1 = Vector(e[1]), Vector(e[2])
            el = mb.elements.new(type="CAPSULE")
            el.co = (p0 + p1) / 2
            el.radius = e[3] * k
            el.size_x = (p1 - p0).length / 2
            el.rotation = Vector((1, 0, 0)).rotation_difference((p1 - p0).normalized())
    ob = bpy.data.objects.new(name, mb)
    mb.materials.append(M[mat])
    coll.objects.link(ob)
    return ob


def area_light(name, loc, rot, size, color, power, size_y=None, spread=180):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.color = power, color
    if size_y:
        ld.shape, ld.size, ld.size_y = "RECTANGLE", size, size_y
    else:
        ld.size = size
    ld.spread = math.radians(spread)
    ob = bpy.data.objects.new(name, ld)
    ob.location, ob.rotation_euler = loc, rot
    coll.objects.link(ob)
    return ob


def point_light(name, loc, color, power, radius=0.1):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy, ld.color, ld.shadow_soft_size = power, color, radius
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    coll.objects.link(ob)
    return ob


def aim(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


# ----------------------------------------------------------------------------- shell
DOOR_Y0, DOOR_Y1, DOOR_H = 0.0, 0.9, 2.05
box("floor", (0, 0.7, -0.01), (WR - WL + 2, WB - WF + 2, 0.02), "floor")
box("ceiling", (0, 0.7, H + 0.01), (WR - WL + 2, WB - WF + 2, 0.02), "ceiling")
box("wall_back", (0, WB + 0.05, H / 2), (WR - WL + 0.2, 0.1, H), "wall")
box("wall_right", (WR + 0.05, 0.7, H / 2), (0.1, WB - WF + 2, H), "wall")
# left wall with a door opening: three pieces
box("wall_left_a", (WL - 0.15, (WF - 1 + DOOR_Y0) / 2, H / 2), (0.3, DOOR_Y0 - WF + 1, H), "wall")
box("wall_left_b", (WL - 0.15, (DOOR_Y1 + WB + 0.1) / 2, H / 2), (0.3, WB + 0.1 - DOOR_Y1, H), "wall")
box("wall_left_top", (WL - 0.15, (DOOR_Y0 + DOOR_Y1) / 2, (DOOR_H + H) / 2), (0.3, DOOR_Y1 - DOOR_Y0, H - DOOR_H), "wall")
# door trim
for y in (DOOR_Y0 - 0.03, DOOR_Y1 + 0.03):
    box("doortrim", (WL + 0.01, y, DOOR_H / 2), (0.04, 0.07, DOOR_H), "doorframe")
box("doortrim_top", (WL + 0.01, (DOOR_Y0 + DOOR_Y1) / 2, DOOR_H + 0.03), (0.04, DOOR_Y1 - DOOR_Y0 + 0.13, 0.07), "doorframe")
# chair rail strip on both walls
box("rail_back", (0, WB - 0.01, RAIL_Z), (WR - WL, 0.025, 0.06), "rail")
box("rail_left_a", (WL + 0.012, (WF + DOOR_Y0) / 2 - 0.3, RAIL_Z), (0.025, DOOR_Y0 - WF + 0.6, 0.06), "rail")
box("rail_left_b", (WL + 0.012, (DOOR_Y1 + WB) / 2, RAIL_Z), (0.025, WB - DOOR_Y1, 0.06), "rail")
# ceiling tile seams (painted as faint darker lines)
for x in (-0.7, 0.5):
    box("tile_seam", (x, 0.7, H - 0.002), (0.012, WB - WF + 2, 0.004), "ceiling_seam")
for y in (0.4, 1.6):
    box("tile_seam", (0, y, H - 0.002), (WR - WL, 0.012, 0.004), "ceiling_seam")

# ----------------------------------------------------------------------------- bathroom (behind the door)
BX0 = WL - 1.7
box("bath_floor", (BX0 / 2 + WL / 2, 1.0, 0.0), (WL - BX0, 2.4, 0.02), "bath_floor")
box("bath_back", (BX0 - 0.05, 1.0, H / 2), (0.1, 2.4, H), "bath_wall")
box("bath_side_a", ((BX0 + WL) / 2, -0.2, H / 2), (WL - BX0, 0.1, H), "bath_wall")
box("bath_side_b", ((BX0 + WL) / 2, 2.2, H / 2), (WL - BX0, 0.1, H), "bath_wall")
box("bath_ceil", ((BX0 + WL) / 2, 1.0, H), (WL - BX0, 2.4, 0.05), "bath_wall")
box("bath_counter", (BX0 + 0.35, 1.55, 0.85), (0.55, 0.9, 0.06), "porcelain")
box("bath_cabinet", (BX0 + 0.35, 1.55, 0.42), (0.5, 0.85, 0.8), "bath_wall")
cyl("toilet", (BX0 + 0.35, 0.55, 0.22), 0.2, 0.42, "porcelain")
box("toilet_tank", (BX0 + 0.12, 0.55, 0.55), (0.18, 0.45, 0.35), "porcelain")
area_light("bath_light", ((BX0 + WL) / 2 - 0.2, 0.45, H - 0.08), (0, 0, 0), 0.6, (1.0, 0.76, 0.48), 120)
sphere("bath_bulb", ((BX0 + WL) / 2 - 0.2, 0.45, H - 0.04), 0.09, "lamp_glow", scale=(1.6, 1.6, 0.4))

# ----------------------------------------------------------------------------- left side: black tower cabinet next to the door
box("tower_cab", (WL + 0.27, 1.35, 0.85), (0.5, 0.6, 1.7), "black_matte", bevel=0.01)
box("tower_cab_top", (WL + 0.27, 1.35, 1.78), (0.4, 0.4, 0.16), "black_matte")  # the box on top

# ----------------------------------------------------------------------------- back wall: dresser + helmet + speakers + keyboard
DX0, DX1, DD, DH = -1.0, 0.25, 0.48, 0.80
dcx = (DX0 + DX1) / 2
box("dresser", (dcx, WB - DD / 2, DH / 2), (DX1 - DX0, DD, DH), "wood", bevel=0.01)
box("dresser_top", (dcx, WB - DD / 2, DH + 0.015), (DX1 - DX0 + 0.04, DD + 0.03, 0.03), "black")
for k, z in enumerate((0.22, 0.52)):
    box("drawer_line", (dcx, WB - DD - 0.002, z + 0.12), (DX1 - DX0 - 0.04, 0.006, 0.012), "black")
    box("drawer_pull", (dcx, WB - DD - 0.01, z), (0.18, 0.02, 0.02), "black")

# helmet (Spartan-style, sits on the dresser like a statue bust)
HX, HY, HZ = -0.80, WB - 0.24, DH + 0.03
sphere("helmet_dome", (HX, HY, HZ + 0.27), 0.16, "helmet", scale=(0.92, 1.05, 1.2))
sphere("helmet_cheek", (HX, HY - 0.02, HZ + 0.13), 0.15, "helmet", scale=(0.88, 1.0, 0.95))
box("helmet_slot_v", (HX, HY - 0.165, HZ + 0.15), (0.045, 0.03, 0.17), "helmet_dark")
box("helmet_slot_h", (HX, HY - 0.165, HZ + 0.25), (0.20, 0.03, 0.04), "helmet_dark")
box("helmet_brow", (HX, HY - 0.175, HZ + 0.30), (0.25, 0.02, 0.03), "helmet_trim")
# crest: a thick half-disc front-to-back; the lower half is cut away inside the dome
crest = cyl("helmet_crest", (HX, HY + 0.02, HZ + 0.30), 0.25, 0.085, "crest", rot=(0, math.radians(90), 0), verts=48)
cut = box("crest_cut", (HX, HY, HZ + 0.30 - 0.20), (0.3, 0.8, 0.40), "crest")
cut.hide_render = True
cut.hide_viewport = True
bo = crest.modifiers.new("cut", "BOOLEAN")
bo.object, bo.operation, bo.solver = cut, "DIFFERENCE", "EXACT"

# speakers: big monitor + small box with green triangle on top
SPX = -0.40
box("spk_big", (SPX, WB - 0.20, DH + 0.03 + 0.21), (0.27, 0.30, 0.42), "black", bevel=0.008)
cyl("spk_woofer", (SPX, WB - 0.352, DH + 0.03 + 0.15), 0.085, 0.01, "black_matte", rot=(math.radians(90), 0, 0))
cyl("spk_tweeter", (SPX, WB - 0.352, DH + 0.03 + 0.32), 0.035, 0.01, "black_matte", rot=(math.radians(90), 0, 0))
box("spk_small", (SPX, WB - 0.20, DH + 0.45 + 0.17), (0.19, 0.22, 0.34), "black_matte", bevel=0.006)
# green triangle logo on its front
tri = bpy.data.meshes.new("logo")
tri.from_pydata([(-0.05, 0, 0), (0.05, 0, 0), (0, 0, 0.075)], [], [(0, 1, 2)])
tob = bpy.data.objects.new("logo", tri)
tob.location = (SPX, WB - 0.312, DH + 0.45 + 0.03)
tri.materials.append(M["green"])
coll.objects.link(tob)
cyl("bt_speaker", (SPX + 0.02, WB - 0.42, DH + 0.08), 0.05, 0.22, "black_matte", rot=(0, math.radians(90), 0))
# MIDI keyboard
box("midi_body", (0.02, WB - 0.25, DH + 0.06), (0.42, 0.17, 0.05), "keys_body")
box("midi_keys", (0.02, WB - 0.30, DH + 0.088), (0.38, 0.07, 0.012), "keys_white")
for i in range(8):
    cyl("midi_pad", (-0.12 + i * 0.04, WB - 0.21, DH + 0.09), 0.011, 0.01, "keys_white")

# bed (dark, low, behind) between dresser and bookshelf
box("bed_frame", (0.62, WB - 0.9, 0.22), (0.95, 1.85, 0.44), "black")
box("bed_mattress", (0.62, WB - 0.9, 0.5), (0.9, 1.8, 0.14), "bedding")
sphere("pillow", (0.62, WB - 0.22, 0.60), 0.17, "pillow", scale=(1.7, 0.6, 0.45))

# bookshelf / dresser on the right
BSX0, BSX1, BSD, BSH = 1.15, 1.75, 0.38, 1.05
bcx = (BSX0 + BSX1) / 2
box("shelf_body", (bcx, WB - BSD / 2, 0.35), (BSX1 - BSX0, BSD, 0.70), "wood_light", bevel=0.008)   # drawers
box("shelf_topb", (bcx, WB - BSD / 2, BSH - 0.015), (BSX1 - BSX0, BSD, 0.03), "wood_light")         # open cubby
for sx in (BSX0 + 0.015, BSX1 - 0.015):
    box("shelf_side", (sx, WB - BSD / 2, (0.70 + BSH) / 2), (0.03, BSD, BSH - 0.70), "wood_light")
box("shelf_hole", (bcx, WB - 0.02, 0.86), (BSX1 - BSX0 - 0.06, 0.03, 0.30), "black_matte")
for z in (0.17, 0.43):
    box("shelf_pull", (bcx, WB - BSD - 0.01, z + 0.06), (0.14, 0.02, 0.02), "black")
    box("shelf_line", (bcx, WB - BSD - 0.003, z + 0.17), (BSX1 - BSX0 - 0.03, 0.006, 0.01), "black")
# colourful books
book_cols = [(0.70, 0.18, 0.08), (0.85, 0.45, 0.10), (0.12, 0.30, 0.55), (0.10, 0.45, 0.42), (0.85, 0.78, 0.55),
             (0.55, 0.10, 0.12), (0.20, 0.20, 0.22), (0.90, 0.60, 0.15), (0.25, 0.45, 0.70), (0.80, 0.80, 0.75)]
x = BSX0 + 0.05
i = 0
while x < BSX1 - 0.18:
    w = random.uniform(0.022, 0.04)
    h = random.uniform(0.20, 0.27)
    c = book_cols[i % len(book_cols)]
    bm = mat_principled(f"book{i}", c, 0.6)
    M[f"book{i}"] = bm
    tilt = math.radians(random.uniform(-6, 2)) if i < 3 else 0
    box(f"book{i}", (x + w / 2, WB - BSD + 0.14, 0.70 + h / 2), (w, 0.17, h), f"book{i}", rot=(0, tilt, 0))
    x += w + 0.003
    i += 1
for k in range(3):  # a small horizontal stack on the right
    box("book_stack", (BSX1 - 0.10, WB - BSD + 0.14, 0.70 + 0.025 + k * 0.05), (0.15, 0.18, 0.045),
        M["keys_white"] if k != 1 else M["book2"])
# on top: little white sign/photo with red, a clear box
box("sign", (BSX0 + 0.15, WB - 0.12, BSH + 0.10), (0.16, 0.02, 0.2), "card", rot=(math.radians(-8), 0, 0))
box("sign_red", (BSX0 + 0.15, WB - 0.135, BSH + 0.08), (0.10, 0.01, 0.07), "card_red", rot=(math.radians(-8), 0, 0))
box("clear_box", (BSX1 - 0.15, WB - 0.18, BSH + 0.08), (0.24, 0.18, 0.16), "box_clear")
cyl("bottle", (BSX0 + 0.04, WB - 0.15, BSH + 0.09), 0.025, 0.18, "box_clear")

# ----------------------------------------------------------------------------- his desk spot (he faces the left wall, his screen just ahead)
DESK_Z = 0.74
ZX, ZY = -0.22, -0.30                       # where he sits
ZFWD = Vector((-0.95, -0.28, 0)).normalized()  # facing his screen (toward the left wall)
MON = Vector((ZX, ZY, 0)) + ZFWD * 0.78 + Vector((0, 0, DESK_Z + 0.27))
MON_YAW = math.atan2(-ZFWD.y, -ZFWD.x)       # monitor local +X points back at him
box("desk", tuple(Vector((ZX, ZY, DESK_Z - 0.02)) + ZFWD * 0.62), (0.75, 1.5, 0.04), "desk_top", rot=(0, 0, MON_YAW))
for dy in (-0.68, 0.68):
    leg = Vector((ZX, ZY, 0)) + ZFWD * 0.85 + Vector((-ZFWD.y, ZFWD.x, 0)) * dy
    box("desk_leg", (leg.x, leg.y, DESK_Z / 2), (0.04, 0.04, DESK_Z), "black")
box("monitor", tuple(MON), (0.035, 0.66, 0.40), "black", rot=(0, 0, MON_YAW))
scr = box("screen", tuple(MON), (0.006, 0.62, 0.36), "screen", rot=(0, 0, MON_YAW))
scr.location = MON - ZFWD * 0.02             # on the side facing him
box("monitor_neck", tuple(MON + ZFWD * 0.04 - Vector((0, 0, 0.2))), (0.04, 0.04, 0.2), "black")
# Iris -- a tiny purple eye on top of the monitor (her one self-portrait in the room)
eye = MON + Vector((0, 0, 0.23)) - ZFWD * 0.01
box("iris_cam", tuple(eye - Vector((0, 0, 0.012))), (0.05, 0.10, 0.04), "black_matte", rot=(0, 0, MON_YAW))
sphere("iris_eye", tuple(eye - ZFWD * 0.026), 0.016, "iris_led")
point_light("iris_glow", tuple(eye - ZFWD * 0.06), (0.62, 0.32, 1.0), 1.2, 0.02)
# Iris's colour as a faint halo behind the screen, on the wall (how she remembers being there)
point_light("iris_halo", tuple(MON + ZFWD * 0.12 + Vector((0, 0, 0.05))), (0.55, 0.28, 1.0), 6.0, 0.15)

# screen light on him (soft cool area light just in front of the screen)
sl = area_light("screen_light", tuple(MON - ZFWD * 0.10), (0, 0, 0), 0.62, (0.30, 0.50, 1.0), 38, size_y=0.38)
aim(sl, (ZX, ZY, 1.25))

# ----------------------------------------------------------------------------- the person at the desk (sitting, chin on fist)
fwd = ZFWD
side = Vector((-fwd.y, fwd.x, 0))  # up x fwd = his LEFT (the camera side); -side = his right
up = Vector((0, 0, 1))
base = Vector((ZX, ZY, 0))
# which side faces the camera? camera is at -Y; side=(−0.15,−1)·k -> +side points to -Y = camera side.


def P(f, s, z):  # forward / camera-side / up in his body space
    return tuple(base + fwd * f + side * s + up * z)


yaw = math.atan2(fwd.y, fwd.x)

# --- the figure: Quaternius "Universal Animation Library" mannequin (CC0), Sitting_Idle_Loop pose, then IK for
#     the chin-on-fist (near/left arm) and the hand on the mouse (far/right arm).  Faceless on purpose.
UAL = ("/data/iris_assets/library/quaternius/universal-animation-library/"
       "Universal Animation Library[Standard]/Unreal-Godot/UAL1_Standard.glb")
ZSCALE = 0.95                                # mannequin ~1.83 m -> 1.74 m
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=UAL)
new_obs = [o for o in bpy.data.objects if o not in before]
arm = next(o for o in new_obs if o.type == "ARMATURE")
body = next(o for o in new_obs if o.type == "MESH" and o.name.startswith("Mannequin"))
for o in new_obs:
    if o not in (arm, body):
        bpy.data.objects.remove(o, do_unlink=True)
for o in (arm, body):                        # importer may have used its own collection
    for c in list(o.users_collection):
        c.objects.unlink(o)
    coll.objects.link(o)
act = bpy.data.actions["Sitting_Idle_Loop"]
arm.animation_data.action = act
if arm.animation_data.action_slot is None and len(act.slots):
    arm.animation_data.action_slot = act.slots[0]
sc.frame_set(12)
basis = {pb.name: pb.matrix_basis.copy() for pb in arm.pose.bones}
arm.animation_data.action = None             # freeze the pose, then layer constraints on top
for pb in arm.pose.bones:
    pb.matrix_basis = basis[pb.name]
# place: mannequin faces local -Y; its pelvis sits at local (0, 0.30, 0.54)
arm.rotation_euler = (0, 0, yaw + math.pi / 2)
arm.scale = (ZSCALE,) * 3
arm.location = Vector((ZX, ZY, 0.0)) - Euler((0, 0, yaw + math.pi / 2)).to_matrix() @ Vector((0, 0.30, 0)) * ZSCALE \
    + fwd * 0.02 - up * 0.02
bpy.context.view_layer.update()


def bone_w(name, tail=False):
    pb = arm.pose.bones[name]
    return arm.matrix_world @ (pb.tail if tail else pb.head)


def empty(name, loc):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    coll.objects.link(e)
    return e


def track(bone, target_loc, influence=1.0):
    c = arm.pose.bones[bone].constraints.new("DAMPED_TRACK")
    c.target = empty(f"trk_{bone}", target_loc)
    c.track_axis = "TRACK_Y"
    c.influence = influence


# lean the upper body toward the screen, head heavy and tipped onto the fist
for bn, f_amt in (("spine_02", 0.30), ("spine_03", 0.45), ("neck_01", 0.60)):
    track(bn, bone_w(bn) + up * 1.0 + fwd * f_amt + side * 0.05)
track("Head", bone_w("Head") + up * 1.0 + fwd * 0.40 + side * 0.22)
bpy.context.view_layer.update()

head_base = bone_w("Head")
head_top = bone_w("Head", tail=True)
head_up = (head_top - head_base).normalized()
# near arm: elbow down on the desk edge, forearm up, fist under the jaw
wrist = head_base + fwd * 0.03 + side * 0.085 - up * 0.15
pole_l = bone_w("upperarm_l") + fwd * 0.45 - up * 0.55 + side * 0.25
ik = arm.pose.bones["lowerarm_l"].constraints.new("IK")
ik.target, ik.pole_target, ik.chain_count = empty("ik_l", wrist), empty("pole_l", pole_l), 2
ik.pole_angle = math.radians(-90)
# far arm: hand on the mouse
MOUSE = Vector(P(0.66, -0.20, DESK_Z + 0.02))
ik2 = arm.pose.bones["lowerarm_r"].constraints.new("IK")
ik2.target = empty("ik_r", MOUSE - fwd * 0.07 + up * 0.005)
ik2.pole_target = empty("pole_r", bone_w("upperarm_r") - side * 0.45 - up * 0.4 - fwd * 0.1)
ik2.chain_count = 2
ik2.pole_angle = math.radians(-90)
# curl the near hand into a loose fist (fingers bend about their local X)
for fn in ("index", "middle", "ring", "pinky"):
    for seg in ("01", "02", "03"):
        pb = arm.pose.bones.get(f"{fn}_{seg}_l")
        if pb:
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = (math.radians(70), 0, 0)
        pb = arm.pose.bones.get(f"{fn}_{seg}_r")                  # far hand relaxed over the mouse
        if pb:
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = (math.radians(28), 0, 0)
bpy.context.view_layer.update()

# materials by dominant bone weight: hoodie top, dark jeans, skin on the head/neck/hands
for mname in ("jeans",):
    M[mname] = mat_noisy("jeans", (0.035, 0.04, 0.06), (0.06, 0.065, 0.09), scale=10.0, rough=0.85)
me = body.data
me.materials.clear()
for m in (M["hoodie"], M["skin"], M["jeans"]):
    me.materials.append(m)
gname = {g.index: g.name for g in body.vertex_groups}
vbone = []
for v in me.vertices:
    best = max(v.groups, key=lambda g: g.weight, default=None)
    vbone.append(gname.get(best.group, "") if best else "")
SKIN = ("Head", "neck_01", "hand_", "index_", "middle_", "ring_", "pinky_", "thumb_")
LEGS = ("thigh_", "calf_", "foot_", "ball_", "pelvis")
for poly in me.polygons:
    names = [vbone[i] for i in poly.vertices]
    nm = max(set(names), key=names.count)
    poly.material_index = 1 if nm.startswith(SKIN) else 2 if nm.startswith(LEGS) else 0
    poly.use_smooth = True

# his chair, placed under the evaluated pelvis (low back, mostly hidden)
pel = bone_w("pelvis")
seat = Vector((pel.x, pel.y, 0)) - fwd * 0.04
box("chair_seat", tuple(seat + up * 0.47), (0.48, 0.48, 0.07), "chair", rot=(0, 0, yaw), bevel=0.02)
box("chair_back", tuple(seat - fwd * 0.26 + up * 0.72), (0.06, 0.40, 0.34), "chair",
    rot=(0, math.radians(-10), yaw), bevel=0.02)
cyl("chair_post", tuple(seat + up * 0.25), 0.03, 0.42, "chair")
cyl("chair_base", tuple(seat + up * 0.05), 0.28, 0.03, "chair")
sub_ = body.modifiers.new("smooth", "SUBSURF")
sub_.levels = sub_.render_levels = 1
# hair, hood, headphones placed on the evaluated head
bpy.context.view_layer.update()
hb_m = arm.matrix_world @ arm.pose.bones["Head"].matrix
h0 = hb_m.translation
hx = (hb_m.to_3x3() @ Vector((1, 0, 0))).normalized()     # across the ears
hy = (hb_m.to_3x3() @ Vector((0, 1, 0))).normalized()     # up the skull
hz = hy.cross(hx).normalized()
if hz.dot(fwd) < 0:                                         # make hz the face direction
    hz = -hz
HEAD = h0 + hy * 0.12 * ZSCALE
hrot = hb_m.to_3x3().to_euler()
metaball("hair", "hair", [
    ("ell", tuple(HEAD + hy * 0.035 - hz * 0.03), 0.125, (1.0, 0.95, 1.0), tuple(hrot)),
    ("ell", tuple(HEAD - hz * 0.05 - hy * 0.01), 0.10, (1.0, 0.9, 1.0), tuple(hrot)),
], res=0.008, k=1.7)
# hood: soft bunched blue collar behind and around the neck
nb = bone_w("neck_01")
metaball("hood", "hoodie", [
    ("ell", tuple(nb - hz * 0.07 + hy * 0.01), 0.16, (1.15, 0.8, 0.7), tuple(hrot)),
    ("cap", tuple(nb + hx * 0.11 - hz * 0.02), tuple(nb - hz * 0.09), 0.08),
    ("cap", tuple(nb - hx * 0.11 - hz * 0.02), tuple(nb - hz * 0.09), 0.08),
], res=0.012, k=1.25)
# headphones: a bevelled band over the skull + two cups (silver disc outside) + a boom mic
cu = bpy.data.curves.new("hp_band", "CURVE")
cu.dimensions = "3D"
cu.bevel_depth = 0.015
cu.bevel_resolution = 3
spl = cu.splines.new("POLY")
pts = [HEAD + hx * (0.105 * math.cos(math.pi * k / 16)) + hy * (0.13 * math.sin(math.pi * k / 16)) - hz * 0.01
       for k in range(17)]
spl.points.add(len(pts) - 1)
for sp, p in zip(spl.points, pts):
    sp.co = (p.x, p.y, p.z, 1)
cu.materials.append(M["headphone"])
coll.objects.link(bpy.data.objects.new("hp_band", cu))
cup_q = hz.to_track_quat("Y", "Z")                         # cylinder axis (local Z) -> across the ears
for s_ in (1, -1):
    cc_ = HEAD + hx * 0.10 * s_ - hy * 0.015
    rot = (hb_m.to_3x3() @ Euler((0, math.radians(90), 0)).to_matrix()).to_euler()
    cyl("hp_cup", tuple(cc_), 0.058, 0.055, "headphone", rot=tuple(rot), verts=28, scale=(1.15, 1.0, 1.0))
    cyl("hp_disc", tuple(cc_ + hx * 0.029 * s_), 0.036, 0.008, "headphone_cup", rot=tuple(rot), verts=28)
cu2 = bpy.data.curves.new("hp_mic", "CURVE")
cu2.dimensions = "3D"
cu2.bevel_depth = 0.0045
spl2 = cu2.splines.new("BEZIER")
spl2.bezier_points.add(1)
for bp, p in zip(spl2.bezier_points, (HEAD + hx * 0.11 - hy * 0.05, HEAD + hx * 0.05 - hy * 0.11 + hz * 0.09)):
    bp.co = p
    bp.handle_left_type = bp.handle_right_type = "AUTO"
cu2.materials.append(M["headphone"])
coll.objects.link(bpy.data.objects.new("hp_mic", cu2))
# mouse with a little green glow (his mouse glows green)
sphere("mouse", tuple(MOUSE - up * 0.005), 0.04, "black", scale=(1.4, 0.9, 0.5))
point_light("mouse_glow", tuple(MOUSE + up * 0.02), (0.2, 1.0, 0.3), 0.25, 0.02)

# ----------------------------------------------------------------------------- foreground right: office chair with the digital-camo jacket
CC = Vector((0.66, -0.42, 0))
crot = math.radians(200)                     # chair faces roughly away-left
cR = Euler((0, 0, crot)).to_matrix()


def C(x, y, z):  # chair-local -> world (chair local -Y = its front)
    return tuple(CC + cR @ Vector((x, y, z)))


box("ochair_seat", C(0, 0, 0.48), (0.5, 0.5, 0.08), "chair", rot=(0, 0, crot))
box("ochair_back", C(0, 0.24, 0.88), (0.46, 0.07, 0.62), "chair", rot=(math.radians(-8), 0, crot))
cyl("ochair_post", C(0, 0, 0.25), 0.03, 0.44, "chair")
# the jacket: a cloth sheet draped over the backrest (front panel up, over the top, down the back), wrinkled
# with noise, plus two hanging sleeves.  Built in chair-local space.
def drape_jacket():
    from mathutils import noise
    path = [(0.10, 0.56), (0.13, 0.80), (0.17, 1.08), (0.24, 1.23), (0.31, 1.08), (0.35, 0.80), (0.38, 0.58)]
    # resample the (y, z) path by arc length
    seg = [((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2) ** 0.5 for i in range(len(path) - 1)]
    total = sum(seg)
    NU, NX = 48, 22
    verts, faces = [], []
    for iu in range(NU):
        t = total * iu / (NU - 1)
        k_ = 0
        while k_ < len(seg) - 1 and t > seg[k_]:
            t -= seg[k_]
            k_ += 1
        f_ = min(t / seg[k_], 1.0)
        y = path[k_][0] + (path[k_ + 1][0] - path[k_][0]) * f_
        z = path[k_][1] + (path[k_ + 1][1] - path[k_][1]) * f_
        flare = 1.0 + 0.35 * max(0.0, (1.0 - z) / 0.45)            # fabric widens as it hangs
        for ix in range(NX):
            xn = -1 + 2 * ix / (NX - 1)
            x = 0.27 * xn * flare
            q = Vector((x * 6, y * 6, z * 6))
            dy = 0.035 * noise.noise(q) + 0.015 * noise.noise(q * 3.1)
            sag = 0.03 * (abs(xn) ** 2) * (1.0 if z < 1.15 else 0.0)  # edges curl back toward the chair
            yy = y + (dy if y < 0.24 else -dy) * (1 if z < 1.15 else 0.3)
            yy += (sag if y < 0.24 else -sag)
            verts.append(C(x, yy, z + 0.02 * noise.noise(q * 1.7)))
    for iu in range(NU - 1):
        for ix in range(NX - 1):
            a0 = iu * NX + ix
            faces.append((a0, a0 + 1, a0 + NX + 1, a0 + NX))
    mesh = bpy.data.meshes.new("jacket")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for pl in mesh.polygons:
        pl.use_smooth = True
    mesh.materials.append(M["camo"])
    ob = bpy.data.objects.new("jacket", mesh)
    coll.objects.link(ob)
    so = ob.modifiers.new("thick", "SOLIDIFY")
    so.thickness = 0.022
    ob.modifiers.new("sub", "SUBSURF").levels = 1
    return ob


drape_jacket()
metaball("sleeves", "camo", [
    ("cap", C(-0.30, 0.13, 1.06), C(-0.33, 0.06, 0.58), 0.06),
    ("cap", C(0.30, 0.13, 1.06), C(0.31, 0.02, 0.60), 0.06),
    ("ell", C(0.0, 0.05, 0.55), 0.18, (1.3, 0.8, 0.25), (0, 0, crot)),                 # hem slumped on the seat
], res=0.014, k=1.35)

# lights: night. The bathroom is the warm key; a faint warm bounce keeps the back wall readable.
area_light("warm_fill", (0.45, 2.0, H - 0.45), (0, 0, 0), 1.6, (1.0, 0.66, 0.38), 7.5)
# a soft SPOT from inside the bathroom through the door.  (An area light here sat 0.3 m from the black tower
#  cabinet beside the door and lit it cream -- found by soloing every light.)
sd = bpy.data.lights.new("door_spill", "SPOT")
sd.energy, sd.color, sd.shadow_soft_size = 900, (1.0, 0.72, 0.42), 0.35
sd.spot_size, sd.spot_blend = math.radians(58), 0.85   # cone must miss the black tower (34 deg off-axis)
dl = bpy.data.objects.new("door_spill", sd)
dl.location = (WL - 0.9, 0.25, 1.85)
coll.objects.link(dl)
aim(dl, (0.45, 0.25, 0.0))
camo_l = area_light("camo_glow", (0.1, -1.1, 1.9), (0, 0, 0), 0.5, (1.0, 0.70, 0.45), 28, spread=45)
aim(camo_l, C(0, 0.2, 0.9))
# a whisper of purple high in the far corner (Iris's colour), as if remembered
point_light("purple_whisper", (1.50, 2.6, 2.3), (0.55, 0.30, 1.0), 2.0, 0.4)

# ----------------------------------------------------------------------------- volume haze (soft light shafts)
haze = box("haze", (-0.2, 0.7, H / 2), (WR - WL + 3.5, WB - WF + 0.4, H - 0.02), "wall")
hm = bpy.data.materials.new("haze")
hnt = hm.node_tree
hnt.nodes.clear()
pv = hnt.nodes.new("ShaderNodeVolumePrincipled")
pv.inputs["Density"].default_value = 0.0025
pv.inputs["Color"].default_value = (1.0, 0.9, 0.8, 1)
ho = hnt.nodes.new("ShaderNodeOutputMaterial")
hnt.links.new(pv.outputs[0], ho.inputs["Volume"])
haze.data.materials.clear()
haze.data.materials.append(hm)
haze.hide_render = args.no_haze

# ----------------------------------------------------------------------------- world, camera
w = bpy.data.worlds.new("night")
sc.world = w
bg = w.node_tree.nodes.get("Background")
bg.inputs["Color"].default_value = (0.004, 0.0035, 0.004, 1)
bg.inputs["Strength"].default_value = 1.0

cd = bpy.data.cameras.new("cam")
cd.lens = 21
cd.sensor_width = 36
cd.dof.use_dof = True
cd.dof.aperture_fstop = 2.8
cam = bpy.data.objects.new("cam", cd)
cam.location = (0.45, -2.0, 1.72)
coll.objects.link(cam)
aim(cam, (-0.50, 1.8, 0.78))
cd.dof.focus_distance = (HEAD - cam.location).length
sc.camera = cam
if args.debug_cam:
    v = [float(t) for t in args.debug_cam.split(",")]
    cam.location = v[:3]
    aim(cam, v[3:6])
    cd.lens = v[6]
    cd.dof.use_dof = False

# ----------------------------------------------------------------------------- render settings
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.render.resolution_percentage = args.pct
ee = sc.eevee
ee.taa_render_samples = args.samples
ee.use_shadows = True
ee.volumetric_tile_size = "4"
ee.volumetric_samples = 64
ee.volumetric_end = 8.0
ee.use_raytracing = True
ee.use_fast_gi = True
vs = sc.view_settings
vs.view_transform = "AgX"
vs.look = "AgX - Medium High Contrast"
vs.exposure = -0.35
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_depth = "8"
sc.render.image_settings.color_mode = "RGB"

# ----------------------------------------------------------------------------- the paint pass (compositor)
if not args.no_paint:
    scale = args.pct / 100.0
    cg = bpy.data.node_groups.new("Paint", "CompositorNodeTree")
    cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    N, L = cg.nodes, cg.links
    rl = N.new("CompositorNodeRLayers")
    out = N.new("NodeGroupOutput")

    # 1) brush wobble: displace by a stretched noise so edges go hand-drawn
    nz = N.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "2D"
    nz.inputs["Scale"].default_value = 9.0
    nz.inputs["Detail"].default_value = 3.0
    sub = N.new("ShaderNodeMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = 0.5
    L.new(nz.outputs["Fac"], sub.inputs[0])
    vec = N.new("ShaderNodeCombineXYZ")
    mx = N.new("ShaderNodeMath")
    mx.operation = "MULTIPLY"
    mx.inputs[1].default_value = 7.0 * scale
    my = N.new("ShaderNodeMath")
    my.operation = "MULTIPLY"
    my.inputs[1].default_value = 4.0 * scale
    L.new(sub.outputs[0], mx.inputs[0])
    nz2 = N.new("ShaderNodeTexNoise")
    nz2.noise_dimensions = "4D"
    nz2.inputs["Scale"].default_value = 13.0
    nz2.inputs["W"].default_value = 3.0
    sub2 = N.new("ShaderNodeMath")
    sub2.operation = "SUBTRACT"
    sub2.inputs[1].default_value = 0.5
    L.new(nz2.outputs["Fac"], sub2.inputs[0])
    L.new(sub2.outputs[0], my.inputs[0])
    L.new(mx.outputs[0], vec.inputs["X"])
    L.new(my.outputs[0], vec.inputs["Y"])
    disp = N.new("CompositorNodeDisplace")
    L.new(rl.outputs["Image"], disp.inputs["Image"])
    L.new(vec.outputs[0], disp.inputs["Displacement"])

    # 2) oil paint: anisotropic Kuwahara (big strokes), then a small classic pass to merge
    kw = N.new("CompositorNodeKuwahara")
    kw.inputs["Type"].default_value = "Anisotropic"
    kw.inputs["Size"].default_value = max(2, int(round(24 * scale)))
    kw.inputs["Uniformity"].default_value = 4
    kw.inputs["Sharpness"].default_value = 0.6
    kw.inputs["Eccentricity"].default_value = 1.4
    L.new(disp.outputs[0], kw.inputs["Image"])
    kw2 = N.new("CompositorNodeKuwahara")
    kw2.inputs["Type"].default_value = "Classic"
    kw2.inputs["Size"].default_value = max(1, int(round(4 * scale)))
    L.new(kw.outputs[0], kw2.inputs["Image"])

    # 3) bloom on the screen + the bathroom light
    gl = N.new("CompositorNodeGlare")
    gl.inputs["Type"].default_value = "Fog Glow"
    gl.inputs["Quality"].default_value = "Medium"
    gl.inputs["Threshold"].default_value = 0.55
    gl.inputs["Strength"].default_value = 0.55
    gl.inputs["Size"].default_value = 0.75
    # impasto: an emboss of the painted image, soft-lit back on top so strokes catch light like ridges
    emb = N.new("CompositorNodeFilter")
    emb.inputs["Type"].default_value = "Shadow"
    emb.inputs["Factor"].default_value = 1.0
    L.new(kw2.outputs[0], emb.inputs["Image"])
    ebw = N.new("CompositorNodeRGBToBW")                        # emboss is signed -> grey around 0.5, clamped
    L.new(emb.outputs[0], ebw.inputs[0])
    emr = N.new("ShaderNodeMapRange")
    emr.clamp = True
    emr.inputs["From Min"].default_value = -0.6
    emr.inputs["From Max"].default_value = 0.6
    emr.inputs["To Min"].default_value = 0.84                   # multiply, not soft-light: the compositor
    emr.inputs["To Max"].default_value = 1.16                   # runs on HDR scene-linear values
    L.new(ebw.outputs[0], emr.inputs["Value"])
    imp = N.new("ShaderNodeMix")
    imp.data_type = "RGBA"
    imp.blend_type = "MULTIPLY"
    imp.inputs["Factor"].default_value = 1.0
    L.new(kw2.outputs[0], imp.inputs["A"])
    L.new(emr.outputs["Result"], imp.inputs["B"])
    L.new(imp.outputs["Result"], gl.inputs["Image"])

    # 4) grade: warm lift in the shadows, slight desaturation of the highs (aged canvas)
    cbal = N.new("CompositorNodeColorBalance")
    cbal.inputs["Type"].default_value = "Lift/Gamma/Gain"
    cin = {i.identifier: i for i in cbal.inputs}
    cin["Color Lift"].default_value = (1.01, 1.0, 0.99, 1)     # warm umber shadows
    cin["Color Gamma"].default_value = (1.02, 1.0, 0.97, 1)
    cin["Color Gain"].default_value = (1.04, 1.0, 0.94, 1)
    L.new(gl.outputs["Image"], cbal.inputs["Image"])

    # 5) canvas weave + paper grain (multiply, very subtle)
    weave = N.new("ShaderNodeTexWave")
    weave.wave_type = "BANDS"
    weave.inputs["Scale"].default_value = 260.0 * scale
    weave.inputs["Distortion"].default_value = 1.5
    weave2 = N.new("ShaderNodeTexWave")
    weave2.wave_type = "BANDS"
    weave2.bands_direction = "Y"
    weave2.inputs["Scale"].default_value = 260.0 * scale
    weave2.inputs["Distortion"].default_value = 1.5
    grain = N.new("ShaderNodeTexWhiteNoise")
    grain.noise_dimensions = "2D"
    wsum = N.new("ShaderNodeMath")
    wsum.operation = "ADD"
    L.new(weave.outputs["Fac"], wsum.inputs[0])
    L.new(weave2.outputs["Fac"], wsum.inputs[1])
    wsum2 = N.new("ShaderNodeMath")
    wsum2.operation = "MULTIPLY_ADD"
    wsum2.inputs[1].default_value = 0.032
    wsum2.inputs[2].default_value = 0.92
    gsum = N.new("ShaderNodeMath")
    gsum.operation = "ADD"
    L.new(wsum.outputs[0], gsum.inputs[0])
    gscale = N.new("ShaderNodeMath")
    gscale.operation = "MULTIPLY"
    gscale.inputs[1].default_value = 0.6
    L.new(grain.outputs["Value"], gscale.inputs[0])
    L.new(gscale.outputs[0], gsum.inputs[1])
    L.new(gsum.outputs[0], wsum2.inputs[0])
    cmix = N.new("ShaderNodeMix")
    cmix.data_type = "RGBA"
    cmix.blend_type = "MULTIPLY"
    cmix.inputs["Factor"].default_value = 1.0
    L.new(cbal.outputs[0], cmix.inputs["A"])
    L.new(wsum2.outputs[0], cmix.inputs["B"])

    # 6) soft vignette (painted edges fall into dark umber)
    em = N.new("CompositorNodeEllipseMask")
    em.inputs["Size"].default_value = (0.92, 0.88)
    bl = N.new("CompositorNodeBlur")
    bl.inputs["Size"].default_value = (int(260 * scale), int(260 * scale))
    L.new(em.outputs["Mask"], bl.inputs["Image"])
    vmr = N.new("ShaderNodeMapRange")
    vmr.inputs["To Min"].default_value = 0.25
    L.new(bl.outputs[0], vmr.inputs["Value"])
    vmix = N.new("ShaderNodeMix")
    vmix.data_type = "RGBA"
    vmix.blend_type = "MULTIPLY"
    vmix.inputs["Factor"].default_value = 1.0
    L.new(cmix.outputs["Result"], vmix.inputs["A"])
    L.new(vmr.outputs["Result"], vmix.inputs["B"])
    L.new(vmix.outputs["Result"], out.inputs[0])
    sc.compositing_node_group = cg
    sc.render.use_compositing = True

# ----------------------------------------------------------------------------- go
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
if args.save_blend:
    bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
sc.render.filepath = args.out
if args.no_render:
    sys.exit(0)
import time

t0 = time.time()
bpy.ops.render.render(write_still=True)
print(f"RENDERED {args.out} in {time.time() - t0:.1f}s")
