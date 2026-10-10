"""Steampunk scene assets for scripts/lyric_scenes/steampunk.py — rendered
HEADLESS in Blender as MAPS, not pictures (Zeke 2026-10-10: use Blender for
the assets).

  ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
      --python scripts/lyric_scenes/blender/steampunk_assets.py -- \
      [--out assets/scenes/steampunk] [--only main,g24,...] [--scale 1.0]

Why maps: the gears ROTATE every frame and the fire / bulbs / window that light
them change every frame, so a baked beauty render would carry its highlights
round with the teeth. Each asset is rendered top-down through an orthographic
camera as three emission passes (a shared node group switches what every
material emits), so the per-frame numpy renderer can light it itself:
  pass 0  NORMAL   world normal * 0.5 + 0.5 (camera looks down -Z, so x/y are
                   screen right/up)
  pass 1  WEIGHTS  material mix: R = brass, G = copper, B = steel,
                   the rest = dark iron (palettes recolour these per render)
  pass 2  AO+DET   R = ambient occlusion, G = grime/patina detail noise
Saved per asset as <name>.npz (uint8, straight alpha) + manifest.json, which
also carries the LAYOUT — the scene reads its geometry from the manifest, so
the assets and the code that animates them cannot drift apart.

Deterministic: fixed geometry, fixed noise seeds, fixed Cycles seed.
Coordinates: "screen units" = frame heights, origin at the frame centre,
y DOWN (the numpy convention); flipped to Blender's y-up only when building.
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Matrix

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FONT_DIR = os.path.join(REPO, "assets", "fonts")
VERSION = 3

# ---------------------------------------------------------------------------
# LAYOUT — the whole engine room, in screen units at rest (camera z = 0).
# depth = distance from the camera; a layer's screen scale is d / (d - camz).
# ---------------------------------------------------------------------------
M_MOD = 0.0125                       # gear module of the main train (units)
LAYOUT = dict(
    depth=dict(wall=4.0, far=3.0, pipes=2.6, main=2.0, bulbs=1.7,
               backhead=1.45, near=1.0),
    module=M_MOD,
    window=dict(c=(0.0, -0.11), r=0.36, ring=0.038),
    # main gear train (all on the d=2.0 plane, all module M_MOD):
    # name, sprite, teeth, parent, mesh angle (deg, y-down screen)
    main_c=(0.0, -0.08),
    train=[("M", "main", 40, None, 0.0),
           ("B", "g24", 24, "M", 200.0),
           ("C", "g20", 20, "M", 22.0),
           ("D", "g12", 12, "C", -62.0)],
    medallion_r=0.122,
    # far train (d=3.0): sprite scaled by s => module M_MOD*s
    far=[("F1", "g24", 24, None, 0.0, (-0.66, -0.24), 1.5),
         ("F2", "g20", 20, "F1", 62.0, None, 1.5),
         ("F3", "g20", 20, None, 0.0, (0.70, -0.30), 1.6),
         ("F4", "g12", 12, "F3", 118.0, None, 1.6)],
    # near foreground (d=1.0): blurred dark silhouettes framing the shot
    near=[("N1", "g24", 24, (-0.98, 0.47), 2.3, 1),
          ("N2", "g20", 20, (1.02, -0.50), 2.0, -1)],
    # vertical steam engines either side: flywheel up top, crank, rod,
    # crosshead, cylinder below
    engine=dict(x=0.72, fly_y=-0.28, fly_r=0.105, crank=0.050, rod=0.25,
                cyl_top=0.08, cyl_bot=0.31, cyl_r=0.062),
    backhead=dict(top_c=0.255, top_k=0.075),   # top edge y(x)=top_c+top_k*(x/.889)^2
    gauges=[(-0.585, 0.372), (-0.405, 0.362), (-0.225, 0.356),
            (0.225, 0.356), (0.405, 0.362), (0.585, 0.372)],
    gauge_r=0.074, face_r=0.061,
    nixie=dict(c=(0.0, 0.318), w=0.25, h=0.072, n=4),
    lamps=[(-0.79, 0.368), (0.79, 0.368)], lamp_r=0.034,
    door=dict(c=(0.0, 0.455), w=0.30, h=0.15),
    whistles=[(-0.105, 0.262), (0.105, 0.262)],
    pipes=dict(y_low=0.14, r_low=0.030, y_top=-0.435, r_top=0.018,
               x_vert=0.575, r_vert=0.026),
    # steam vents on the pipe layer: (x, y, dir x, dir y)
    vents=[(-0.80, 0.112, -0.35, -1.0), (0.80, 0.112, 0.35, -1.0),
           (-0.30, 0.112, -0.15, -1.0), (0.52, 0.112, 0.2, -1.0),
           (-0.575, -0.36, -1.0, -0.35), (0.575, -0.36, 1.0, -0.35),
           (-0.25, -0.455, 0.0, -1.0), (0.30, -0.455, 0.0, -1.0)],
    bulbs=dict(n=14, x0=-0.80, x1=0.80, y_lo=-0.40, y_hi=-0.355, r=0.040),
)

# sprite extents (screen units) + pixels per unit for the static plates
PLATES = dict(
    wall=dict(x=(-1.06, 1.06), y=(-0.62, 0.62), ppu=900),
    pipes=dict(x=(-1.08, 1.08), y=(-0.62, 0.30), ppu=1100),
    engine=dict(x=(0.55, 0.92), y=(-0.43, 0.36), ppu=1300),   # right; mirrored
    backhead=dict(x=(-1.12, 1.12), y=(0.18, 0.66), ppu=1300),
    medallion=dict(x=(-0.135, 0.135), y=(-0.135, 0.135), ppu=1400),
    plate=dict(x=(-0.50, 0.50), y=(-0.135, 0.135), ppu=1400),
)
GEAR_PPU = 1400

# ---------------------------------------------------------------------------
# materials: every material = noise detail + bump -> the SP_PASS group
# ---------------------------------------------------------------------------
W_BRASS, W_COPPER, W_STEEL, W_IRON = (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)


def _sock(node, ident, out=False):
    for s in (node.outputs if out else node.inputs):
        if s.identifier == ident:
            return s
    raise KeyError(f"{node.bl_idname}: no socket {ident!r}")


def build_pass_group():
    g = bpy.data.node_groups.new("SP_PASS", "ShaderNodeTree")
    g.interface.new_socket("W", in_out="INPUT", socket_type="NodeSocketColor")
    g.interface.new_socket("Detail", in_out="INPUT", socket_type="NodeSocketFloat")
    g.interface.new_socket("Normal", in_out="INPUT", socket_type="NodeSocketVector")
    g.interface.new_socket("Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    N, L = g.nodes, g.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    mode = N.new("ShaderNodeValue")
    mode.name = mode.label = "MODE"
    mode.outputs[0].default_value = 0.0
    nrm = N.new("ShaderNodeVectorMath")
    nrm.operation = "MULTIPLY_ADD"
    nrm.inputs[1].default_value = (0.5, 0.5, 0.5)
    nrm.inputs[2].default_value = (0.5, 0.5, 0.5)
    L.new(gi.outputs["Normal"], nrm.inputs[0])
    ao = N.new("ShaderNodeAmbientOcclusion")
    ao.samples = 24
    ao.inputs["Distance"].default_value = 0.045
    L.new(gi.outputs["Normal"], ao.inputs["Normal"])
    cc = N.new("ShaderNodeCombineColor")
    L.new(ao.outputs["AO"], cc.inputs[0])
    L.new(gi.outputs["Detail"], cc.inputs[1])
    gt1, gt2 = N.new("ShaderNodeMath"), N.new("ShaderNodeMath")
    for gt, th in ((gt1, 0.5), (gt2, 1.5)):
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = th
        L.new(mode.outputs[0], gt.inputs[0])
    m1, m2 = N.new("ShaderNodeMix"), N.new("ShaderNodeMix")
    for m in (m1, m2):
        m.data_type = "RGBA"
    L.new(gt1.outputs[0], _sock(m1, "Factor_Float"))
    L.new(nrm.outputs[0], _sock(m1, "A_Color"))
    L.new(gi.outputs["W"], _sock(m1, "B_Color"))
    L.new(gt2.outputs[0], _sock(m2, "Factor_Float"))
    L.new(_sock(m1, "Result_Color", True), _sock(m2, "A_Color"))
    L.new(cc.outputs[0], _sock(m2, "B_Color"))
    em = N.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    L.new(_sock(m2, "Result_Color", True), em.inputs["Color"])
    L.new(em.outputs[0], go.inputs["Shader"])
    return g


_MATS = {}


def mat(name, w, bump=0.10, nscale=40.0, rough=0.0, seed=0.0):
    """w = material weights (brass, copper, steel); bump = hammered-surface
    strength; nscale = grime noise scale (object space)."""
    key = (name, w, bump, nscale, rough, seed)
    if key in _MATS:
        return _MATS[key]
    m = bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):
        m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links
    N.clear()
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Location"].default_value = (seed * 3.1, seed * 1.7, seed * 2.3)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = nscale
    nz.inputs["Detail"].default_value = 6.0
    nz.inputs["Roughness"].default_value = 0.62
    L.new(mp.outputs["Vector"], nz.inputs["Vector"])
    nz2 = N.new("ShaderNodeTexNoise")          # fine hammered bump
    nz2.inputs["Scale"].default_value = nscale * 6.0
    nz2.inputs["Detail"].default_value = 3.0
    L.new(mp.outputs["Vector"], nz2.inputs["Vector"])
    bp = N.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = bump
    bp.inputs["Distance"].default_value = 0.002
    L.new(nz2.outputs["Fac"], bp.inputs["Height"])
    rgb = N.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (w[0], w[1], w[2], 1.0)
    gn = N.new("ShaderNodeGroup")
    gn.node_tree = bpy.data.node_groups["SP_PASS"]
    L.new(rgb.outputs[0], gn.inputs["W"])
    L.new(nz.outputs["Fac"], gn.inputs["Detail"])
    L.new(bp.outputs["Normal"], gn.inputs["Normal"])
    out = N.new("ShaderNodeOutputMaterial")
    L.new(gn.outputs["Shader"], out.inputs["Surface"])
    _MATS[key] = m
    return m


# ---------------------------------------------------------------------------
# geometry helpers (screen units in, Blender y-up out)
# ---------------------------------------------------------------------------
COL = None


def link(obj):
    COL.objects.link(obj)
    return obj


def curve_obj(name, loops, z=0.0, ext=0.01, bev=0.004, bres=2, material=None,
              smooth=True):
    """Filled 2-D curve: loops = list of (n,2) arrays in SCREEN coords (y down);
    nested loops become holes. Extruded +-ext around z, edges bevelled."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = ext
    cu.bevel_depth = bev
    cu.bevel_resolution = bres
    for lp in loops:
        lp = np.asarray(lp, np.float64)
        sp = cu.splines.new("POLY")
        sp.points.add(len(lp) - 1)
        co = np.zeros((len(lp), 4))
        co[:, 0] = lp[:, 0]
        co[:, 1] = -lp[:, 1]
        co[:, 3] = 1.0
        sp.points.foreach_set("co", co.ravel())
        sp.use_cyclic_u = True
        sp.use_smooth = smooth
    ob = bpy.data.objects.new(name, cu)
    ob.location.z = z
    if material is not None:
        cu.materials.append(material)
    return link(ob)


def circle(cx, cy, r, n=96, a0=0.0):
    a = a0 + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def rrect(cx, cy, w, h, r, n=8):
    """Rounded rectangle loop."""
    pts = []
    for (qx, qy, a0) in ((cx + w / 2 - r, cy + h / 2 - r, 0.0),
                         (cx - w / 2 + r, cy + h / 2 - r, 0.5 * np.pi),
                         (cx - w / 2 + r, cy - h / 2 + r, np.pi),
                         (cx + w / 2 - r, cy - h / 2 + r, 1.5 * np.pi)):
        for k in range(n + 1):
            a = a0 + 0.5 * np.pi * k / n
            pts.append((qx + r * np.cos(a), qy + r * np.sin(a)))
    return np.asarray(pts)


def gear_profile(N, module, cx=0.0, cy=0.0, per_tooth=14):
    """Outer outline of a spur gear: rounded root, straight-ish involute-like
    flanks, slightly crowned tip. Tooth 0 points along +x."""
    rp = module * N / 2.0
    ra, rr = rp + module, rp - 1.25 * module
    wp = np.pi / (2 * N)                       # half tooth width at pitch
    pts = []
    for k in range(N):
        c = 2 * np.pi * k / N
        prof = [(-2.0 * wp, rr), (-1.45 * wp, rr), (-1.22 * wp, rr + 0.35 * (rp - rr)),
                (-1.05 * wp, rp), (-0.85 * wp, rp + 0.55 * module),
                (-0.62 * wp, ra - 0.08 * module), (-0.35 * wp, ra), (0.0, ra + 0.02 * module),
                (0.35 * wp, ra), (0.62 * wp, ra - 0.08 * module),
                (0.85 * wp, rp + 0.55 * module), (1.05 * wp, rp),
                (1.22 * wp, rr + 0.35 * (rp - rr)), (1.45 * wp, rr)]
        for da, r in prof:
            a = c + da
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.asarray(pts), rp, ra, rr


def sector_hole(r0, r1, a0, a1, corner=0.25, n=24, twist=0.0):
    """Lightening cutout between spokes: annular sector with rounded ends.
    twist bends it (curved spokes)."""
    outer = [(r1, a0 + (a1 - a0) * t) for t in np.linspace(0.12, 0.88, n)]
    inner = [(r0, a1 - (a1 - a0) * t) for t in np.linspace(0.18, 0.82, n)]
    # rounded joins
    pts = []
    rm = 0.5 * (r0 + r1)
    seq = ([(rm + (r1 - rm) * np.sin(t), a0 + (a1 - a0) * 0.12 * (1 - np.cos(t)))
            for t in np.linspace(0, np.pi / 2, 6)] + outer
           + [(rm + (r1 - rm) * np.cos(t), a1 - (a1 - a0) * 0.12 * (1 - np.sin(t)))
              for t in np.linspace(0, np.pi / 2, 6)]
           + [(rm - (rm - r0) * np.sin(t), a1 - (a1 - a0) * 0.18 * (1 - np.cos(t)))
              for t in np.linspace(0, np.pi / 2, 6)] + inner
           + [(rm - (rm - r0) * np.cos(t), a0 + (a1 - a0) * 0.18 * (1 - np.sin(t)))
              for t in np.linspace(0, np.pi / 2, 6)])
    for r, a in seq:
        a2 = a + twist * (r - r0) / max(1e-6, (r1 - r0))
        pts.append((r * np.cos(a2), r * np.sin(a2)))
    return np.asarray(pts)


def bolts(cx, cy, rad, n, size, z, material, a0=0.0, hexa=True, name="bolt"):
    for k in range(n):
        a = a0 + 2 * np.pi * k / n
        x, y = cx + rad * np.cos(a), cy + rad * np.sin(a)
        if hexa:
            lp = circle(x, y, size, 6, a0=a)
        else:
            lp = circle(x, y, size, 20)
        curve_obj(f"{name}{k}", [lp], z=z, ext=size * 0.35, bev=size * 0.35,
                  bres=2, material=material)


def mesh_from_bm(name, bm, material):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    me.update()
    sm = np.ones(len(me.polygons), bool)
    me.polygons.foreach_set("use_smooth", sm)
    ob = bpy.data.objects.new(name, me)
    me.materials.append(material)
    return link(ob)


def domes(name, pts, r, z, material, flat=0.55, u=10, v=6):
    """Many rivet heads (flattened hemispheres) in ONE mesh, tiled with numpy
    (bmesh.ops.create_uvsphere on a growing bmesh is O(n^2): 5 min for the
    wall's rivets)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    th = np.linspace(0, 2 * np.pi, u, endpoint=False)
    ph = np.linspace(0, 0.5 * np.pi, v + 1)[:-1]          # rings, equator up
    ring = np.stack([np.outer(np.cos(ph), np.cos(th)).ravel(),
                     np.outer(np.cos(ph), np.sin(th)).ravel(),
                     np.repeat(np.sin(ph), u)], 1)
    vs = np.vstack([ring, [[0.0, 0.0, 1.0]]]) * np.array([r, r, r * flat])
    faces = []
    for j in range(v - 1):
        for i in range(u):
            a, b = j * u + i, j * u + (i + 1) % u
            faces.append((a, b, b + u, a + u))
    top = len(vs) - 1
    for i in range(u):
        faces.append(((v - 1) * u + i, (v - 1) * u + (i + 1) % u, top))
    nv = len(vs)
    V = (vs[None] + np.stack([pts[:, 0], -pts[:, 1], np.full(len(pts), z)], 1)[:, None])
    F = [tuple(f_ + k * nv for f_ in f) for k in range(len(pts)) for f in faces]
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.reshape(-1, 3).tolist(), [], F)
    me.update()
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    ob = bpy.data.objects.new(name, me)
    me.materials.append(material)
    return link(ob)


def cyl(name, p0, p1, r, material, z=0.0, segs=40, cap=True):
    """Cylinder between two SCREEN points, lying in the plane (axis in x/y),
    its axis at height z — pipes, rods, boiler tubes."""
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    ang = math.atan2(-(y1 - y0), x1 - x0)
    bm = bmesh.new()
    try:
        M = (Matrix.Translation(((x0 + x1) / 2, -(y0 + y1) / 2, z))
             @ Matrix.Rotation(ang, 4, "Z") @ Matrix.Rotation(math.pi / 2, 4, "Y"))
        bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=segs,
                              radius1=r, radius2=r, depth=L, matrix=M)
        return mesh_from_bm(name, bm, material)
    finally:
        bm.free()


def ball(name, x, y, r, z, material, flat=1.0):
    return domes(name, [(x, y)], r, z, material, flat=flat)


def flange(name, x, y, r, z, material, axis="v", nb=6):
    """Pipe flange seen side-on: a short fat cylinder across the pipe."""
    if axis == "v":                # vertical pipe -> flange is horizontal bar
        cyl(name, (x, y - 0.008), (x, y + 0.008), r, material, z=z, segs=40)
    else:
        cyl(name, (x - 0.008, y), (x + 0.008, y), r, material, z=z, segs=40)


def handwheel(name, x, y, r, z, material):
    loops = [circle(x, y, r, 72), circle(x, y, r * 0.80, 72)]
    curve_obj(name + "_rim", loops, z=z, ext=0.004, bev=0.004, material=material)
    for k in range(5):
        a = 2 * np.pi * k / 5 + 0.3
        cyl(f"{name}_sp{k}", (x, y), (x + r * 0.85 * np.cos(a), y + r * 0.85 * np.sin(a)),
            r * 0.07, material, z=z)
    ball(name + "_hub", x, y, r * 0.22, z + 0.004, material, flat=0.6)


# ---------------------------------------------------------------------------
# ASSETS
# ---------------------------------------------------------------------------
def a_gear(N, style, module=M_MOD, metal="brass"):
    """Gear sprites. style: 'main' (decorative web, centre opening for the
    fixed logo medallion), 'spoked' (curved spokes), 'holes' (round
    lightening holes), 'solid' (pinion)."""
    # the HERO gear is brass; the supporting cast is iron / copper / steel so
    # the main gear + medallion read first (staging)
    body_w = dict(brass=W_BRASS, iron=W_IRON, copper=W_COPPER, steel=W_STEEL)[metal]
    brass = mat(f"gear_{metal}", body_w, bump=0.05 if metal != "iron" else 0.12,
                nscale=30.0, seed=N)
    iron = mat("gear_iron", W_IRON, bump=0.10, nscale=26.0, seed=N + 1)
    steel = mat("gear_steel", W_STEEL, bump=0.04, nscale=50.0, seed=N + 2)
    prof, rp, ra, rr = gear_profile(N, module)
    rim_in = rr - 2.2 * module
    hub_r = max(0.032, 0.20 * rp)
    # toothed rim (raised)
    curve_obj("rim", [prof, circle(0, 0, rim_in, 160)], z=0.0, ext=0.010,
              bev=0.0035, bres=3, material=brass)
    if style == "main":
        web_outer = rim_in + 0.004
        open_r = LAYOUT["medallion_r"] + 0.010
        loops = [circle(0, 0, web_outer, 160), circle(0, 0, open_r, 160)]
        n_h = 8
        for k in range(n_h):
            a = 2 * np.pi * (k + 0.5) / n_h
            rm = 0.5 * (open_r + rim_in) + 0.002
            loops.append(circle(rm * np.cos(a), rm * np.sin(a),
                                0.30 * (rim_in - open_r), 48))
        curve_obj("web", loops, z=-0.006, ext=0.004, bev=0.002, material=brass)
        # inner collar around the medallion opening (raised ring)
        curve_obj("collar", [circle(0, 0, open_r + 0.016, 160),
                             circle(0, 0, open_r, 160)], z=0.002, ext=0.008,
                  bev=0.004, bres=3, material=brass)
        # decorative bead ring + rivets between holes
        curve_obj("bead", [circle(0, 0, rim_in - 0.004, 160),
                           circle(0, 0, rim_in - 0.010, 160)], z=-0.001,
                  ext=0.004, bev=0.003, material=steel)
        rm = 0.5 * (open_r + rim_in) + 0.002
        domes("rivets", [(rm * np.cos(2 * np.pi * k / n_h), rm * np.sin(2 * np.pi * k / n_h))
                         for k in range(n_h)], 0.008, -0.002, steel)
        # the opening is EMPTY: the medallion (fixed, upright) sits behind it
    else:
        if style == "spoked":
            n_s = 5
            loops = [circle(0, 0, rim_in + 0.004, 140), circle(0, 0, hub_r * 0.45, 40)]
            for k in range(n_s):
                a0 = 2 * np.pi * k / n_s
                a1 = a0 + 2 * np.pi / n_s
                sw = 0.22 * (2 * np.pi / n_s)
                loops.append(sector_hole(hub_r + 0.012, rim_in - 0.010,
                                         a0 + sw / 2, a1 - sw / 2, twist=0.35))
            curve_obj("web", loops, z=-0.004, ext=0.006, bev=0.003, material=brass)
        elif style == "holes":
            n_s = 6
            loops = [circle(0, 0, rim_in + 0.004, 140), circle(0, 0, hub_r * 0.45, 40)]
            rm = 0.5 * (hub_r + rim_in)
            for k in range(n_s):
                a = 2 * np.pi * (k + 0.5) / n_s
                loops.append(circle(rm * np.cos(a), rm * np.sin(a),
                                    0.36 * (rim_in - hub_r), 48))
            curve_obj("web", loops, z=-0.005, ext=0.004, bev=0.003, material=brass)
        else:   # solid pinion
            curve_obj("web", [circle(0, 0, rim_in + 0.004, 120),
                              circle(0, 0, hub_r * 0.45, 40)], z=-0.003,
                      ext=0.006, bev=0.003, material=brass)
        # hub boss + steel axle cap + bolts
        curve_obj("hub", [circle(0, 0, hub_r, 72), circle(0, 0, hub_r * 0.45, 40)],
                  z=0.004, ext=0.010, bev=0.005, bres=3, material=iron)
        ball("cap", 0, 0, hub_r * 0.48, 0.010, steel, flat=0.45)
        bolts(0, 0, hub_r * 0.72, 5 if style != "solid" else 4,
              hub_r * 0.13, 0.016, steel)
    return ra * 1.03


def a_flywheel():
    E = LAYOUT["engine"]
    R = E["fly_r"]
    iron = mat("fly_iron", W_IRON, bump=0.12, nscale=24.0, seed=7)
    steel = mat("fly_steel", W_STEEL, bump=0.03, nscale=50.0, seed=8)
    brass = mat("fly_brass", W_BRASS, bump=0.04, nscale=30.0, seed=9)
    curve_obj("rim", [circle(0, 0, R, 160), circle(0, 0, R * 0.80, 160)],
              z=0.0, ext=0.012, bev=0.005, bres=3, material=iron)
    curve_obj("rimband", [circle(0, 0, R * 0.965, 160), circle(0, 0, R * 0.90, 160)],
              z=0.010, ext=0.004, bev=0.002, material=brass)
    for k in range(6):
        a = 2 * np.pi * k / 6
        cyl(f"spoke{k}", (0.0, 0.0), (R * 0.82 * np.cos(a), R * 0.82 * np.sin(a)),
            0.0085, iron, z=0.0, segs=24)
    curve_obj("hub", [circle(0, 0, 0.026, 60)], z=0.004, ext=0.010, bev=0.005,
              material=iron)
    ball("cap", 0, 0, 0.013, 0.012, steel, flat=0.5)
    # crank disc + crank pin at +x (crank radius)
    c = E["crank"]
    curve_obj("crankarm", [rrect(c / 2, 0, c + 0.030, 0.030, 0.014)], z=0.016,
              ext=0.004, bev=0.003, material=steel)
    ball("pin", c, 0, 0.011, 0.022, brass, flat=0.6)
    return R * 1.04


def a_medallion():
    r = LAYOUT["medallion_r"]
    brass = mat("med_brass", W_BRASS, bump=0.03, nscale=24.0, seed=11)
    steel = mat("med_steel", W_STEEL, bump=0.02, nscale=40.0, seed=12)
    # dished face (the logo inlay is painted on in numpy, see steampunk.py)
    curve_obj("face", [circle(0, 0, r - 0.012, 160)], z=-0.004, ext=0.002,
              bev=0.001, material=steel)
    curve_obj("bezel", [circle(0, 0, r, 160), circle(0, 0, r - 0.014, 160)],
              z=0.0, ext=0.006, bev=0.006, bres=4, material=brass)
    domes("screws", [((r - 0.007) * np.cos(a), (r - 0.007) * np.sin(a))
                     for a in np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi / 12],
          0.0042, 0.008, steel)


def a_wall():
    """Back wall: riveted iron plates, two lattice girders, the round clock
    window (an opening — the glowing glass is drawn per frame behind it) with
    its iron tracery."""
    P = PLATES["wall"]
    W_ = LAYOUT["window"]
    wx, wy, wr = W_["c"][0], W_["c"][1], W_["r"]
    iron = mat("wall_iron", W_IRON, bump=0.22, nscale=9.0, seed=21)
    iron2 = mat("wall_iron2", W_IRON, bump=0.14, nscale=14.0, seed=22)
    steel = mat("wall_steel", W_STEEL, bump=0.05, nscale=30.0, seed=23)
    copper = mat("wall_copper", W_COPPER, bump=0.08, nscale=18.0, seed=24)
    x0, x1 = P["x"]
    y0, y1 = P["y"]
    pw, ph = 0.30, 0.22
    rng = np.random.default_rng(5)
    riv = []
    hole = circle(wx, wy, wr + W_["ring"] * 0.6, 160)
    for i, xa in enumerate(np.arange(x0 - 0.1, x1, pw)):
        for j, ya in enumerate(np.arange(y0 - 0.05, y1, ph)):
            off = (0.5 * pw) if j % 2 else 0.0
            cx_, cy_ = xa + off + pw / 2, ya + ph / 2
            loops = [rrect(cx_, cy_, pw - 0.006, ph - 0.006, 0.010, 3)]
            # plate fully inside the window? skip; intersecting: cut the hole
            d = np.hypot(np.clip(wx, cx_ - pw / 2, cx_ + pw / 2) - wx,
                         np.clip(wy, cy_ - ph / 2, cy_ + ph / 2) - wy)
            far = max(np.hypot(cx_ - pw / 2 - wx, cy_ - ph / 2 - wy),
                      np.hypot(cx_ + pw / 2 - wx, cy_ + ph / 2 - wy),
                      np.hypot(cx_ - pw / 2 - wx, cy_ + ph / 2 - wy),
                      np.hypot(cx_ + pw / 2 - wx, cy_ - ph / 2 - wy))
            if far < wr + W_["ring"] * 0.6:
                continue
            if d < wr + W_["ring"]:
                # clip the plate polygon against the circle crudely: keep the
                # plate, the window ring (drawn on top) hides the seam, and
                # the hole is punched by the curve fill (nested loop)
                loops = [rrect(cx_, cy_, pw - 0.006, ph - 0.006, 0.010, 3)]
                inside = hole[(np.abs(hole[:, 0] - cx_) < pw / 2 - 0.004)
                              & (np.abs(hole[:, 1] - cy_) < ph / 2 - 0.004)]
                if len(inside) == len(hole):
                    loops.append(hole)
                else:
                    # partial overlap: build plate minus disc as a polygon
                    loops = [_rect_minus_disc(cx_, cy_, pw - 0.006, ph - 0.006,
                                              wx, wy, wr + W_["ring"] * 0.6)]
                    if loops[0] is None:
                        continue
            m_ = iron if (i + j) % 3 else iron2
            curve_obj(f"plate{i}_{j}", loops, z=-0.02 + 0.002 * rng.random(),
                      ext=0.004, bev=0.003, material=m_)
            # rivet rows along the plate edges
            for t in np.linspace(-0.45, 0.45, 9):
                for (rx, ry) in ((cx_ + t * pw, cy_ - ph / 2 + 0.012),
                                 (cx_ + t * pw, cy_ + ph / 2 - 0.012)):
                    if np.hypot(rx - wx, ry - wy) > wr + W_["ring"] + 0.01:
                        riv.append((rx, ry))
            for t in np.linspace(-0.3, 0.3, 4):
                for rx in (cx_ - pw / 2 + 0.012, cx_ + pw / 2 - 0.012):
                    ry = cy_ + t * ph
                    if np.hypot(rx - wx, ry - wy) > wr + W_["ring"] + 0.01:
                        riv.append((rx, ry))
    domes("wall_rivets", riv, 0.0055, -0.014, steel)
    # window: iron ring + brass inner bead + tracery
    curve_obj("win_ring", [circle(wx, wy, wr + W_["ring"], 200),
                           circle(wx, wy, wr, 200)], z=0.0, ext=0.012, bev=0.006,
              bres=3, material=iron)
    curve_obj("win_bead", [circle(wx, wy, wr + 0.006, 200),
                           circle(wx, wy, wr - 0.004, 200)], z=0.012, ext=0.003,
              bev=0.003, material=copper)
    domes("win_rivets", [(wx + (wr + W_["ring"] / 2) * np.cos(a),
                          wy + (wr + W_["ring"] / 2) * np.sin(a))
                         for a in np.linspace(0, 2 * np.pi, 48, endpoint=False)],
          0.0065, 0.012, steel)
    for rr_ in (0.62, 0.30):
        curve_obj(f"win_in{rr_}", [circle(wx, wy, wr * rr_ + 0.006, 160),
                                   circle(wx, wy, wr * rr_ - 0.006, 160)],
                  z=-0.004, ext=0.004, bev=0.003, material=iron)
    for k in range(12):
        a = 2 * np.pi * k / 12
        cyl(f"mull{k}", (wx + wr * 0.30 * np.cos(a), wy + wr * 0.30 * np.sin(a)),
            (wx + (wr + 0.004) * np.cos(a), wy + (wr + 0.004) * np.sin(a)),
            0.0055, iron, z=-0.004, segs=16)
    # lattice girders left and right + a crossbeam under the ceiling
    for gx in (-0.84, 0.84):
        for side in (-1, 1):
            cyl(f"gird{gx}{side}", (gx + side * 0.05, y0 - 0.1), (gx + side * 0.05, y1 + 0.1),
                0.010, iron, z=0.004, segs=16)
        for k, yy in enumerate(np.arange(y0, y1, 0.10)):
            s_ = 1 if k % 2 else -1
            cyl(f"lat{gx}{k}", (gx - 0.05 * s_, yy), (gx + 0.05 * s_, yy + 0.10),
                0.006, iron2, z=0.0, segs=12)
        domes(f"gr{gx}", [(gx + s * 0.05, yy) for yy in np.arange(y0, y1, 0.10)
                          for s in (-1, 1)], 0.007, 0.012, steel)
    cyl("beam", (x0 - 0.1, -0.545), (x1 + 0.1, -0.545), 0.022, iron2, z=0.02)
    domes("beam_r", [(x, -0.545) for x in np.arange(x0, x1, 0.06)], 0.006, 0.040, steel)


def _rect_minus_disc(cx, cy, w, h, dx, dy, dr):
    """Rectangle minus a disc, as one polygon (for plates the window bites
    into). Samples the rectangle boundary densely and pushes points that fall
    inside the disc out onto its circle."""
    n = 60
    xs = np.concatenate([np.linspace(cx - w / 2, cx + w / 2, n),
                         np.full(n, cx + w / 2), np.linspace(cx + w / 2, cx - w / 2, n),
                         np.full(n, cx - w / 2)])
    ys = np.concatenate([np.full(n, cy - h / 2), np.linspace(cy - h / 2, cy + h / 2, n),
                         np.full(n, cy + h / 2), np.linspace(cy + h / 2, cy - h / 2, n)])
    out = []
    for x, y in zip(xs, ys):
        d = math.hypot(x - dx, y - dy)
        if d < dr:
            if d < 1e-6:
                return None
            x, y = dx + (x - dx) * dr / d, dy + (y - dy) * dr / d
        out.append((x, y))
    out = np.asarray(out)
    keep = np.ones(len(out), bool)
    keep[1:] = np.hypot(np.diff(out[:, 0]), np.diff(out[:, 1])) > 1e-5
    return out[keep]


def a_pipes():
    P = LAYOUT["pipes"]
    copper = mat("pipe_copper", W_COPPER, bump=0.06, nscale=20.0, seed=31)
    brass = mat("pipe_brass", W_BRASS, bump=0.04, nscale=30.0, seed=32)
    iron = mat("pipe_iron", W_IRON, bump=0.10, nscale=20.0, seed=33)
    steel = mat("pipe_steel", W_STEEL, bump=0.03, nscale=40.0, seed=34)
    X0, X1 = PLATES["pipes"]["x"]
    yl, rl = P["y_low"], P["r_low"]
    yt, rt = P["y_top"], P["r_top"]
    xv, rv = P["x_vert"], P["r_vert"]
    # low main steam pipe
    cyl("low", (X0 - 0.05, yl), (X1 + 0.05, yl), rl, copper, z=0.0)
    for x in (-0.95, -0.80, -0.575, -0.30, 0.0, 0.30, 0.52, 0.575, 0.80, 0.95):
        flange(f"lf{x}", x, yl, rl * 1.45, 0.0, brass, axis="h")
    domes("lfb", [(x + s * 0.0, yl + o) for x in (-0.95, -0.80, -0.30, 0.0, 0.30, 0.52, 0.80, 0.95)
                  for o in (-rl * 1.25, rl * 1.25) for s in (0,)], 0.0045, 0.012, steel)
    # vertical risers + tees + upper pipe
    for sx in (-1, 1):
        x = sx * xv
        cyl(f"vert{sx}", (x, yl), (x, yt), rv, copper, z=0.0)
        ball(f"tee{sx}", x, yl, rl * 1.35, 0.0, brass, flat=1.0)
        ball(f"elb{sx}", x, yt, rv * 1.30, 0.0, brass, flat=1.0)
        for y in (-0.05, -0.25, -0.36):
            flange(f"vf{sx}{y}", x, y, rv * 1.45, 0.0, brass, axis="v")
        # relief valve + nozzle where the vent is
        cyl(f"relief{sx}", (x, -0.36), (x + sx * 0.040, -0.375), 0.010, brass, z=0.01)
        ball(f"reliefb{sx}", x + sx * 0.040, -0.375, 0.012, 0.01, steel, flat=0.8)
        handwheel(f"hw{sx}", x, 0.02, 0.032, 0.03, brass)
    cyl("top", (X0 - 0.05, yt), (X1 + 0.05, yt), rt, copper, z=-0.005)
    cyl("top2", (X0 - 0.05, yt - 0.055), (X1 + 0.05, yt - 0.055), rt * 0.8, iron, z=-0.010)
    for x in np.arange(-1.0, 1.01, 0.25):
        flange(f"tf{x:.2f}", x, yt, rt * 1.5, -0.005, brass, axis="h")
    # vent nozzles on the pipes (small stubs + caps) at LAYOUT vents
    for k, (x, y, dx, dy) in enumerate(LAYOUT["vents"]):
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        cyl(f"nz{k}", (x - ux * 0.01, y - uy * 0.01), (x + ux * 0.022, y + uy * 0.022),
            0.0075, brass, z=0.012, segs=20)
    # pipe hangers from the top pipe
    for x in (-0.9, -0.35, 0.35, 0.9):
        cyl(f"hang{x}", (x, yt - 0.20), (x, yt), 0.004, iron, z=-0.02, segs=10)


def a_engine():
    """The right-hand vertical engine's static parts (mirrored for the left):
    cylinder, steam chest, crosshead guides, A-frame columns, flywheel
    bearing pedestal."""
    E = LAYOUT["engine"]
    x, fy = E["x"], E["fly_y"]
    iron = mat("eng_iron", W_IRON, bump=0.12, nscale=22.0, seed=41)
    brass = mat("eng_brass", W_BRASS, bump=0.04, nscale=30.0, seed=42)
    steel = mat("eng_steel", W_STEEL, bump=0.03, nscale=40.0, seed=43)
    copper = mat("eng_copper", W_COPPER, bump=0.06, nscale=20.0, seed=44)
    ct, cb, cr = E["cyl_top"], E["cyl_bot"], E["cyl_r"]
    # cylinder (vertical, seen side-on) with brass bands + covers
    cyl("cyl", (x, ct), (x, cb), cr, copper, z=-0.01)
    for y in (ct + 0.012, cb - 0.014):
        cyl(f"cover{y}", (x, y - 0.012), (x, y + 0.012), cr * 1.18, brass, z=-0.01)
        domes(f"cvb{y}", [(x + t * cr, y) for t in np.linspace(-1.0, 1.0, 6)],
              0.005, 0.010 + cr * 0.15, steel)
    for y in np.linspace(ct + 0.06, cb - 0.06, 3):
        cyl(f"band{y}", (x, y - 0.004), (x, y + 0.004), cr * 1.04, brass, z=-0.01)
    # steam chest on the outside + pipe
    sx = x + 0.090
    curve_obj("chest", [rrect(sx, (ct + cb) / 2 + 0.01, 0.050, 0.15, 0.010)],
              z=-0.03, ext=0.012, bev=0.006, material=iron)
    domes("chestb", [(sx + dx, y) for dx in (-0.018, 0.018)
                     for y in np.linspace(ct + 0.03, cb - 0.01, 5)], 0.0045, -0.012, steel)
    # crosshead guides (two bars) from cylinder top up to y = ct - 0.17
    gt = ct - 0.17
    for s in (-1, 1):
        cyl(f"guide{s}", (x + s * 0.030, gt), (x + s * 0.030, ct), 0.0065, steel, z=0.0)
    curve_obj("gtie", [rrect(x, gt - 0.006, 0.085, 0.014, 0.005)], z=0.0,
              ext=0.005, bev=0.004, material=iron)
    # A-frame columns from the floor to the flywheel bearing
    for s in (-1, 1):
        cyl(f"col{s}", (x + s * 0.115, cb + 0.05), (x + s * 0.020, fy), 0.010, iron,
            z=-0.05, segs=16)
    curve_obj("pedestal", [rrect(x, fy + 0.010, 0.07, 0.035, 0.010)], z=-0.04,
              ext=0.010, bev=0.005, material=iron)
    domes("pedb", [(x - 0.024, fy + 0.010), (x + 0.024, fy + 0.010)], 0.0055, -0.025, steel)
    # drain cock under the cylinder (steam blast nozzle)
    cyl("cock", (x - cr * 0.6, cb - 0.02), (x - cr * 1.25, cb - 0.005), 0.0065, brass, z=0.02)


def a_backhead():
    """The near console = a boiler backhead: riveted dome-topped plate,
    firebox door with slots (holes: the fire is drawn behind), gauge bezels
    (face drawn per frame), Nixie window, caged warning lamps, whistles."""
    B = LAYOUT["backhead"]
    iron = mat("bh_iron", W_IRON, bump=0.16, nscale=10.0, seed=51)
    iron2 = mat("bh_iron2", W_IRON, bump=0.08, nscale=20.0, seed=52)
    brass = mat("bh_brass", W_BRASS, bump=0.04, nscale=26.0, seed=53)
    steel = mat("bh_steel", W_STEEL, bump=0.03, nscale=40.0, seed=54)
    copper = mat("bh_copper", W_COPPER, bump=0.06, nscale=20.0, seed=55)
    X0, X1 = PLATES["backhead"]["x"]
    Y1 = PLATES["backhead"]["y"][1]
    xs = np.linspace(X0, X1, 200)
    top = B["top_c"] + B["top_k"] * (xs / 0.889) ** 2
    outline = np.concatenate([np.stack([xs, top], 1),
                              [[X1, Y1 + 0.05], [X0, Y1 + 0.05]]])
    holes = []
    D = LAYOUT["door"]
    dx, dy, dw, dh = D["c"][0], D["c"][1], D["w"], D["h"]
    # the firebox opening (door + frame are separate objects on top)
    holes.append(rrect(dx, dy, dw * 0.86, dh * 0.80, 0.02))
    for (gx, gy) in LAYOUT["gauges"]:
        holes.append(circle(gx, gy, LAYOUT["face_r"] + 0.003, 96))
    Nx = LAYOUT["nixie"]
    holes.append(rrect(Nx["c"][0], Nx["c"][1], Nx["w"] * 0.92, Nx["h"] * 0.78, 0.008))
    for (lx, ly) in LAYOUT["lamps"]:
        holes.append(circle(lx, ly, LAYOUT["lamp_r"] * 0.92, 64))
    curve_obj("plate", [outline] + holes, z=0.0, ext=0.006, bev=0.004, bres=2,
              material=iron)
    # lagging bands (brass straps) following the top curve
    for off in (0.012, 0.115):
        band = np.concatenate([np.stack([xs, top + off], 1),
                               np.stack([xs[::-1], top[::-1] + off + 0.016], 1)])
        curve_obj(f"strap{off}", [band], z=0.008, ext=0.003, bev=0.002, material=brass)
    # rivet rows along the seams
    riv = [(x, t + 0.040) for x, t in zip(xs[::5], top[::5])]
    riv += [(x, t + 0.150) for x, t in zip(xs[::5], top[::5])
            if not any(abs(x - gx) < 0.085 for gx, _ in LAYOUT["gauges"]) and abs(x) > 0.15]
    domes("riv", riv, 0.0055, 0.010, steel)
    # firebox door: frame ring, two-leaf door with slots
    curve_obj("dframe", [rrect(dx, dy, dw + 0.03, dh + 0.03, 0.03),
                         rrect(dx, dy, dw * 0.86, dh * 0.80, 0.02)], z=0.012,
              ext=0.008, bev=0.005, bres=3, material=iron2)
    domes("dfr", [(dx + t * (dw + 0.012) / 2, dy - (dh + 0.012) / 2) for t in np.linspace(-0.9, 0.9, 9)]
          + [(dx + t * (dw + 0.012) / 2, dy + (dh + 0.012) / 2) for t in np.linspace(-0.9, 0.9, 9)],
          0.0055, 0.022, brass)
    for s in (-1, 1):
        lx = dx + s * dw * 0.215
        leaf = [rrect(lx, dy, dw * 0.40, dh * 0.70, 0.012)]
        for k in range(5):
            leaf.append(rrect(lx, dy - dh * 0.25 + k * dh * 0.125, dw * 0.30, dh * 0.045, 0.008))
        curve_obj(f"leaf{s}", leaf, z=0.022, ext=0.005, bev=0.004, material=iron)
        cyl(f"handle{s}", (lx - s * dw * 0.10, dy - 0.01), (lx - s * dw * 0.10, dy + 0.03),
            0.006, brass, z=0.034)
    # gauge bezels: brass ring + lugs + a pipe stub below each
    for k, (gx, gy) in enumerate(LAYOUT["gauges"]):
        fr = LAYOUT["face_r"]
        curve_obj(f"bez{k}", [circle(gx, gy, LAYOUT["gauge_r"], 120),
                              circle(gx, gy, fr, 120)], z=0.016, ext=0.007,
                  bev=0.006, bres=4, material=brass)
        domes(f"bezs{k}", [(gx + (LAYOUT["gauge_r"] - 0.0065) * np.cos(a),
                            gy + (LAYOUT["gauge_r"] - 0.0065) * np.sin(a))
                           for a in np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8],
              0.0036, 0.028, steel)
        cyl(f"gstub{k}", (gx, gy + LAYOUT["gauge_r"]), (gx, gy + LAYOUT["gauge_r"] + 0.05),
            0.008, copper, z=0.006, segs=20)
    # Nixie housing: brass frame with corner screws + a little name rail
    curve_obj("nframe", [rrect(Nx["c"][0], Nx["c"][1], Nx["w"] + 0.022, Nx["h"] + 0.022, 0.012),
                         rrect(Nx["c"][0], Nx["c"][1], Nx["w"] * 0.92, Nx["h"] * 0.78, 0.008)],
              z=0.014, ext=0.006, bev=0.005, bres=3, material=brass)
    domes("nfs", [(Nx["c"][0] + sx * (Nx["w"] / 2 + 0.002), Nx["c"][1] + sy * (Nx["h"] / 2 + 0.002))
                  for sx in (-1, 1) for sy in (-1, 1)], 0.0045, 0.024, steel)
    # caged warning lamps: ring + vertical cage bars over the opening
    for k, (lx, ly) in enumerate(LAYOUT["lamps"]):
        r = LAYOUT["lamp_r"]
        curve_obj(f"lring{k}", [circle(lx, ly, r * 1.18, 80), circle(lx, ly, r * 0.92, 80)],
                  z=0.016, ext=0.006, bev=0.005, material=brass)
        for t in np.linspace(-0.6, 0.6, 4):
            h = np.sqrt(max(0.0, 1 - t * t)) * r * 0.95
            cyl(f"cage{k}{t:.2f}", (lx + t * r, ly - h), (lx + t * r, ly + h), 0.0032,
                steel, z=0.030, segs=12)
        cyl(f"cageh{k}", (lx - r * 0.95, ly), (lx + r * 0.95, ly), 0.0032, steel, z=0.032, segs=12)
    # whistles on top: stem + bell + lever
    for k, (wx, wy) in enumerate(LAYOUT["whistles"]):
        tw = B["top_c"] + B["top_k"] * (wx / 0.889) ** 2
        cyl(f"wstem{k}", (wx, tw + 0.01), (wx, wy - 0.035), 0.0075, brass, z=0.0)
        cyl(f"wbell{k}", (wx, wy - 0.035), (wx, wy - 0.085), 0.016, brass, z=0.0)
        ball(f"wcap{k}", wx, wy - 0.088, 0.010, 0.0, brass, flat=0.8)
        cyl(f"wlev{k}", (wx, wy - 0.045), (wx + (0.04 if wx > 0 else -0.04), wy - 0.06),
            0.0035, steel, z=0.012, segs=12)


def a_plate():
    """The 'HIT A BUMP' maker's plate that slams in on drop entries: riveted
    brass plate, embossed letters (real 3-D text in Anton)."""
    P = PLATES["plate"]
    brass = mat("pl_brass", W_BRASS, bump=0.05, nscale=18.0, seed=61)
    iron = mat("pl_iron", W_IRON, bump=0.10, nscale=20.0, seed=62)
    steel = mat("pl_steel", W_STEEL, bump=0.03, nscale=30.0, seed=63)
    w = (P["x"][1] - P["x"][0]) - 0.02
    h = (P["y"][1] - P["y"][0]) - 0.02
    curve_obj("back", [rrect(0, 0, w, h, 0.03)], z=-0.01, ext=0.006, bev=0.006,
              bres=3, material=iron)
    curve_obj("rimb", [rrect(0, 0, w - 0.03, h - 0.03, 0.022),
                       rrect(0, 0, w - 0.05, h - 0.05, 0.016)], z=0.0, ext=0.004,
              bev=0.004, bres=3, material=brass)
    curve_obj("field", [rrect(0, 0, w - 0.05, h - 0.05, 0.016)], z=-0.004, ext=0.002,
              bev=0.001, material=mat("pl_enamel", W_IRON, bump=0.02, nscale=40.0, seed=64))
    curve_obj("inner", [rrect(0, 0, w - 0.06, h - 0.06, 0.014),
                        rrect(0, 0, w - 0.072, h - 0.072, 0.010)], z=0.006,
              ext=0.0015, bev=0.0015, material=brass)
    domes("pr", [(sx * (w / 2 - 0.022), sy * (h / 2 - 0.022)) for sx in (-1, 1) for sy in (-1, 1)],
          0.008, 0.008, steel)
    fnt = bpy.data.fonts.load(os.path.join(FONT_DIR, "Anton-Regular.ttf"))
    cu = bpy.data.curves.new("txt", "FONT")
    cu.body = "HIT A BUMP"
    cu.font = fnt
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.size = h * 0.66
    cu.space_character = 1.06
    cu.extrude = 0.006
    cu.bevel_depth = 0.0035
    cu.bevel_resolution = 3
    ob = bpy.data.objects.new("txt", cu)
    ob.location = (0, -0.004, 0.012)
    cu.materials.append(brass)
    link(ob)
    # fit the text width: measured after a depsgraph update
    bpy.context.view_layer.update()
    tw, th = ob.dimensions.x, ob.dimensions.y
    s = min(w * 0.84 / tw, h * 0.60 / th)
    ob.scale = (s, s, 1.0)


# ---------------------------------------------------------------------------
# render plumbing
# ---------------------------------------------------------------------------
def setup_scene(use_gpu=True):
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    sc.render.engine = "CYCLES"
    sc.cycles.seed = 7
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    sc.render.film_transparent = True
    sc.render.resolution_percentage = 100
    sc.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    sc.cycles.filter_width = 1.5
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    if use_gpu:
        try:
            pr = bpy.context.preferences.addons["cycles"].preferences
            pr.compute_device_type = "CUDA"
            pr.refresh_devices()
            for d in pr.devices:
                d.use = d.type == "CUDA"
            sc.cycles.device = "GPU"
        except Exception as e:      # noqa: BLE001
            print("GPU unavailable, CPU render:", e)
            sc.cycles.device = "CPU"
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.clip_start = 0.01
    cam.data.clip_end = 20.0
    sc.camera = cam
    return sc, cam


def render_maps(name, x0, x1, y0, y1, ppu, out_dir, sc, cam, extra=None):
    w = max(8, int(round((x1 - x0) * ppu)))
    h = max(8, int(round((y1 - y0) * ppu)))
    w += w % 2
    h += h % 2
    x1, y1 = x0 + w / ppu, y0 + h / ppu          # exact pixel grid
    sc.render.resolution_x, sc.render.resolution_y = w, h
    cam.data.ortho_scale = max(x1 - x0, y1 - y0)
    cam.location = ((x0 + x1) / 2, -(y0 + y1) / 2, 5.0)
    mode = bpy.data.node_groups["SP_PASS"].nodes["MODE"]
    res = []
    for k, spp in ((0, 16), (1, 16), (2, 96)):
        mode.outputs[0].default_value = float(k)
        sc.cycles.samples = spp
        path = os.path.join(out_dir, f"_tmp_{name}_{k}.exr")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(path)
        a = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(a)
        bpy.data.images.remove(im)
        os.remove(path)
        res.append(a.reshape(h, w, 4)[::-1].copy())
    alpha = res[0][..., 3]
    inv = 1.0 / np.maximum(alpha, 1e-4)
    nrm = res[0][..., :3] * inv[..., None] * 2.0 - 1.0
    nrm[alpha < 1e-3] = (0.0, 0.0, 1.0)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    wts = np.clip(res[1][..., :3] * inv[..., None], 0, 1)
    aod = np.clip(res[2][..., :2] * inv[..., None], 0, 1)
    q = lambda v: np.clip(np.round(v * 255.0), 0, 255).astype(np.uint8)  # noqa: E731
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"),
                        alpha=q(alpha), normal=q(nrm * 0.5 + 0.5), weights=q(wts),
                        ao=q(aod[..., 0]), detail=q(aod[..., 1]))
    meta = dict(x=(x0, x1), y=(y0, y1), ppu=ppu, size=(w, h))
    if extra:
        meta.update(extra)
    return meta


def clear_scene_objects(cam):
    for o in list(bpy.data.objects):
        if o is not cam:
            bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.curves):
        if c.users == 0:
            bpy.data.curves.remove(c)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)


def main():
    global COL
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "assets", "scenes", "steampunk"))
    ap.add_argument("--only", default="")
    ap.add_argument("--scale", type=float, default=1.0, help="ppu multiplier (test: 0.3)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    sc, cam = setup_scene(not args.cpu)
    COL = sc.collection
    build_pass_group()
    only = set(filter(None, args.only.split(",")))
    man_path = os.path.join(args.out, "manifest.json")
    manifest = dict(version=VERSION, layout=LAYOUT, assets={})
    if os.path.isfile(man_path):
        try:
            old = json.load(open(man_path))
            if old.get("version") == VERSION:
                manifest["assets"] = old.get("assets", {})
        except Exception:       # noqa: BLE001
            pass
    S = args.scale
    jobs = [
        ("main", lambda: a_gear(40, "main"), "gear", 40),
        ("g24", lambda: a_gear(24, "spoked", metal="iron"), "gear", 24),
        ("g20", lambda: a_gear(20, "holes", metal="copper"), "gear", 20),
        ("g12", lambda: a_gear(12, "solid", metal="steel"), "gear", 12),
        ("flywheel", a_flywheel, "gear", 0),
        ("medallion", a_medallion, "plate", 0),
        ("wall", a_wall, "plate", 0),
        ("pipes", a_pipes, "plate", 0),
        ("engine", a_engine, "plate", 0),
        ("backhead", a_backhead, "plate", 0),
        ("plate", a_plate, "plate", 0),
    ]
    for name, fn, kind, N in jobs:
        if only and name not in only:
            continue
        t0 = time.time()
        clear_scene_objects(cam)
        r = fn()
        if kind == "gear":
            meta = render_maps(name, -r, r, -r, r, GEAR_PPU * S, args.out, sc, cam,
                               dict(kind="gear", teeth=N, module=M_MOD))
        else:
            P = PLATES[name]
            meta = render_maps(name, P["x"][0], P["x"][1], P["y"][0], P["y"][1],
                               P["ppu"] * S, args.out, sc, cam, dict(kind="plate"))
        manifest["assets"][name] = meta
        print(f"[steampunk_assets] {name}: {meta['size']} in {time.time() - t0:.1f}s",
              flush=True)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print("[steampunk_assets] manifest ->", man_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
