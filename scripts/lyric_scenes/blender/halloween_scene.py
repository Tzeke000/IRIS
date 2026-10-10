"""Halloween scene assets for scripts/lyric_scenes/halloween.py — headless Blender.

    ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/blender/halloween_scene.py -- \
        --out assets/scenes/halloween [--scale 1.0] [--samples 64] [--preview]

ONE 3-D scene (haunted house on a hill + a graveyard at the camera's feet),
rendered as LIGHT-SEPARATED, LAYER-SEPARATED grey passes so the per-frame
compositor (numpy) can colour, relight and parallax them to the music:

  house layer (hill + house + fence):    h_key  (moonlight only, RGBA)
                                         h_fill (sky fill only)
                                         h_glass (window glass mask)
                                         h_spill (window light spilling out)
  graveyard layer (ground, stones, pumpkins): f_key, f_fill (pumpkins channel-
                                         packed: R-G = skin, B-G = stem)
                                         f_lit_<k> (pumpkin k's candle only)
  meta.json: projected window polygons (+ EQ column / floor), pumpkin face
             polygons, anchor points (door, chimneys, tower, spire).

Every material is neutral grey (value only) and every light is white: the
colour comes from the palette at render time, so one asset set serves every
palette and the moon's colour can drive the rim light per frame.

Deterministic: geometry from fixed seeds, fixed sample counts + seed, OIDN.
"""
import argparse
import json
import math
import os
import random
import sys

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

BASE_W, BASE_H = 2496, 1404           # 1.3 x 1920x1080: room for the push-in


# ---------------------------------------------------------------- helpers
class MB:
    """Accumulates boxes/polys for ONE material, then becomes one object."""

    def __init__(self):
        self.v, self.f = [], []

    def poly(self, pts, faces):
        o = len(self.v)
        self.v += [tuple(p) for p in pts]
        self.f += [tuple(o + k for k in fc) for fc in faces]

    def hexa(self, c):                 # 8 corners: bottom 0-3, top 4-7 (ccw)
        self.poly(c, [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                      (2, 3, 7, 6), (3, 0, 4, 7)])

    def box(self, x0, x1, y0, y1, z0, z1):
        self.hexa([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])

    def fbox(self, F, u0, u1, v0, v1, d0, d1, rot=0.0):
        """Box in a FACE frame: u along the face, v up, d outward."""
        cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        cs, sn = math.cos(rot), math.sin(rot)
        pts = []
        for d in (d0, d1):
            for (u, v) in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                du, dv = u - cu, v - cv
                uu, vv = cu + cs * du - sn * dv, cv + sn * du + cs * dv
                pts.append(F["o"] + F["U"] * uu + Vector((0, 0, vv)) + F["N"] * d)
        # pts: d0 ring then d1 ring; reorder to bottom/top convention
        b = pts[0:4]
        t = pts[4:8]
        self.poly(b + t, [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                          (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])

    def build(self, name, mat, coll):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.validate()
        me.update()
        ob = bpy.data.objects.new(name, me)
        ob.data.materials.append(mat)
        coll.objects.link(ob)
        return ob


def face(o, U, N):
    return dict(o=Vector(o), U=Vector(U).normalized(), N=Vector(N).normalized())


def grey(name, v, rough=0.8, bump=None, scale=1.0, strength=0.4):
    """Neutral grey (or channel-packed) material; optional procedural bump."""
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    col = v if isinstance(v, tuple) else (v, v, v)
    bs.inputs["Base Color"].default_value = (*col, 1.0)
    bs.inputs["Roughness"].default_value = rough
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])
    if bump:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        if bump == "boards":
            tx = nt.nodes.new("ShaderNodeTexWave")
            tx.wave_type = "BANDS"
            tx.bands_direction = "Z"
            tx.wave_profile = "SAW"
            tx.inputs["Scale"].default_value = scale
            tx.inputs["Distortion"].default_value = 0.6
            key = "Fac"
        elif bump == "shingle":
            tx = nt.nodes.new("ShaderNodeTexBrick")
            tx.inputs["Scale"].default_value = scale
            tx.inputs["Mortar Size"].default_value = 0.03
            key = "Fac"
        elif bump == "brick":
            tx = nt.nodes.new("ShaderNodeTexBrick")
            tx.inputs["Scale"].default_value = scale
            key = "Fac"
        else:                                   # stone / ground
            tx = nt.nodes.new("ShaderNodeTexNoise")
            tx.inputs["Scale"].default_value = scale
            tx.inputs["Detail"].default_value = 8.0
            key = "Fac"
        nt.links.new(tc.outputs["Object"], tx.inputs["Vector"])
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = strength
        nt.links.new(tx.outputs[key], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], bs.inputs["Normal"])
    return m


def emission(name, strength, col=(1.0, 1.0, 1.0)):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*col, 1.0)
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


# ---------------------------------------------------------------- the house
WIN = []          # dicts: corners (world), col, lvl, kind


def window(mbs, F, u, v, w, h, col, lvl, shutters=True, pediment=True,
           boarded=False, rng=None, arch=False):
    """Recessed glass + frame + mullions + sill + lintel (+ shutters)."""
    g, t, dk = mbs["glass"], mbs["trim"], mbs["dark"]
    u0, u1, v0, v1 = u - w / 2, u + w / 2, v - h / 2, v + h / 2
    g.fbox(F, u0, u1, v0, v1, 0.00, 0.02)
    WIN.append(dict(c=[F["o"] + F["U"] * uu + Vector((0, 0, vv)) + F["N"] * 0.02
                       for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))],
                    col=col, lvl=lvl))
    fw = 0.11
    t.fbox(F, u0 - fw, u1 + fw, v1, v1 + fw, 0.0, 0.14)
    t.fbox(F, u0 - fw, u1 + fw, v0 - fw, v0, 0.0, 0.14)
    t.fbox(F, u0 - fw, u0, v0, v1, 0.0, 0.14)
    t.fbox(F, u1, u1 + fw, v0, v1, 0.0, 0.14)
    # mullions: one vertical, two horizontal (6 panes)
    t.fbox(F, u - 0.03, u + 0.03, v0, v1, 0.0, 0.08)
    for fr in (0.36, 0.68):
        vv = v0 + h * fr
        t.fbox(F, u0, u1, vv - 0.03, vv + 0.03, 0.0, 0.08)
    t.fbox(F, u0 - 0.22, u1 + 0.22, v0 - fw - 0.10, v0 - fw, 0.0, 0.26)  # sill
    if pediment:
        t.fbox(F, u0 - 0.25, u1 + 0.25, v1 + fw, v1 + fw + 0.14, 0.0, 0.24)
        # a small triangular pediment
        P = [F["o"] + F["U"] * uu + Vector((0, 0, vv)) + F["N"] * dd
             for dd in (0.0, 0.2)
             for (uu, vv) in ((u0 - 0.2, v1 + fw + 0.14), (u1 + 0.2, v1 + fw + 0.14),
                              (u, v1 + fw + 0.14 + 0.42))]
        t.poly(P, [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)])
    if shutters:
        for side in (-1, 1):
            if rng is not None and rng.random() < 0.25:
                continue                                    # one fell off
            rot = 0.0
            if rng is not None and rng.random() < 0.3:
                rot = side * rng.uniform(0.15, 0.35)        # hanging crooked
            su = u0 - fw - 0.52 if side < 0 else u1 + fw + 0.02
            dk.fbox(F, su, su + 0.5, v0, v1, 0.02, 0.09, rot=rot)
            for k in range(1, 6):                           # louvre slats
                vv = v0 + h * k / 6
                dk.fbox(F, su + 0.04, su + 0.46, vv - 0.025, vv + 0.025,
                        0.09, 0.12, rot=rot)
    if boarded:
        for k, (dv, rr) in enumerate(((0.2, 0.18), (0.5, -0.12), (0.78, 0.08))):
            vv = v0 + h * dv
            mbs["wood"].fbox(F, u0 - 0.25, u1 + 0.25, vv - 0.11, vv + 0.11,
                             0.14, 0.2, rot=rr)


def round_window(mbs, F, u, v, r, col, lvl):
    g, t = mbs["glass"], mbs["trim"]
    n = 20
    ring = [(u + r * math.cos(2 * math.pi * k / n), v + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]
    P = [F["o"] + F["U"] * a + Vector((0, 0, b)) + F["N"] * 0.02 for a, b in ring]
    g.poly([F["o"] + F["U"] * u + Vector((0, 0, v)) + F["N"] * 0.02] + P,
           [(0, k + 1, (k + 1) % n + 1) for k in range(n)])
    WIN.append(dict(c=P, col=col, lvl=lvl))
    for k in range(n):                                      # frame ring
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        pts = []
        for d in (0.0, 0.14):
            for rr in (r, r + 0.13):
                for a in (a0, a1):
                    pts.append(F["o"] + F["U"] * (u + rr * math.cos(a))
                               + Vector((0, 0, v + rr * math.sin(a))) + F["N"] * d)
        # pts index: d0: r a0, r a1, R a0, R a1 ; d1 likewise (+4)
        t.poly(pts, [(0, 1, 5, 4), (2, 6, 7, 3), (4, 5, 7, 6), (0, 2, 3, 1)])
    t.fbox(F, u - r, u + r, v - 0.03, v + 0.03, 0.0, 0.08)
    t.fbox(F, u - 0.03, u + 0.03, v - r, v + r, 0.0, 0.08)


def gable(mb_roof, mb_wall, x0, x1, y0, y1, ze, zr, axis, o=0.45,
          ends=(True, True)):
    """Steep gable roof; ridge along `axis` ('x' or 'y'). Triangle ends = wall."""
    if axis == "x":
        ym = 0.5 * (y0 + y1)
        a = [(x0 - o, y0 - o, ze - 0.25), (x1 + o, y0 - o, ze - 0.25),
             (x1 + o, ym, zr), (x0 - o, ym, zr)]
        b = [(x0 - o, y1 + o, ze - 0.25), (x1 + o, y1 + o, ze - 0.25),
             (x1 + o, ym, zr), (x0 - o, ym, zr)]
        for s in (a, b):
            th = [(p[0], p[1], p[2] + 0.22) for p in s]
            mb_roof.poly(s + th, [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                                  (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)])
        for k, xx in enumerate((x0, x1)):
            if ends[k]:
                mb_wall.poly([(xx, y0, ze), (xx, y1, ze), (xx, ym, zr - 0.1)],
                             [(0, 1, 2)])
    else:
        xm = 0.5 * (x0 + x1)
        a = [(x0 - o, y0 - o, ze - 0.25), (x0 - o, y1 + o, ze - 0.25),
             (xm, y1 + o, zr), (xm, y0 - o, zr)]
        b = [(x1 + o, y0 - o, ze - 0.25), (x1 + o, y1 + o, ze - 0.25),
             (xm, y1 + o, zr), (xm, y0 - o, zr)]
        for s in (a, b):
            th = [(p[0], p[1], p[2] + 0.22) for p in s]
            mb_roof.poly(s + th, [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                                  (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)])
        for k, yy in enumerate((y0, y1)):
            if ends[k]:
                mb_wall.poly([(x0, yy, ze), (x1, yy, ze), (xm, yy, zr - 0.1)],
                             [(0, 1, 2)])


def cone(mb, cx, cy, z0, z1, r0, r1, n=8, rot=0.0):
    pts = []
    for z, r in ((z0, r0), (z1, r1)):
        for k in range(n):
            a = rot + 2 * math.pi * k / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
    faces = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    faces.append(tuple(range(n - 1, -1, -1)))
    if r1 > 1e-4:
        faces.append(tuple(range(n, 2 * n)))
    mb.poly(pts, faces)


def build_house(coll, M):
    rng = random.Random(1031)
    mbs = {k: MB() for k in ("wall", "roof", "trim", "dark", "glass", "stone",
                             "brick", "wood", "iron")}
    W_, R_, T_ = mbs["wall"], mbs["roof"], mbs["trim"]
    # foundation + main block + left wing + tower
    mbs["stone"].box(-7.3, 4.3, -5.8, 3.8, -1.5, 0.85)
    mbs["stone"].box(3.3, 7.3, -4.8, -0.7, -1.5, 0.85)
    W_.box(-6.0, 4.0, -3.5, 3.5, 0.8, 7.6)          # main
    W_.box(-7.0, -2.5, -5.5, 3.5, 0.8, 7.6)         # wing (front gable)
    W_.box(3.5, 7.0, -4.5, -1.0, 0.8, 11.4)         # tower
    # floor bands (trim) at storey lines
    for z in (4.2, 7.6):
        T_.box(-6.1, 4.1, -3.62, 3.62, z - 0.14, z + 0.06)
        T_.box(-7.12, -2.4, -5.62, 3.62, z - 0.14, z + 0.06)
    for z in (4.2, 7.6, 11.4):
        T_.box(3.38, 7.12, -4.62, -0.88, z - 0.14, z + 0.10)
    # roofs
    gable(R_, W_, -6.0, 4.0, -3.5, 3.5, 7.6, 12.2, "x", ends=(False, True))
    gable(R_, W_, -7.0, -2.5, -5.5, 3.5, 7.6, 11.6, "y", ends=(True, False))
    # cresting along the main ridge + finials
    for k in range(26):
        x = -2.4 + k * 0.25
        mbs["iron"].box(x - 0.025, x + 0.025, -0.03, 0.03, 12.35, 12.35 + (0.55 if k % 4 == 0 else 0.32))
    mbs["iron"].box(-2.4, 4.1, -0.03, 0.03, 12.6, 12.65)
    cone(mbs["iron"], -4.75, -5.95, 11.5, 12.7, 0.06, 0.0, n=6)
    # tower: cornice, steep octagonal spire with a flared skirt, finial
    T_.box(3.2, 7.3, -4.8, -0.7, 11.4, 11.75)
    cone(R_, 5.25, -2.75, 11.75, 12.6, 2.95, 2.25, n=8, rot=math.pi / 8)
    cone(R_, 5.25, -2.75, 12.6, 20.2, 2.25, 0.0, n=8, rot=math.pi / 8)
    cone(mbs["iron"], 5.25, -2.75, 20.0, 21.6, 0.07, 0.03, n=6)
    for z, r in ((20.55, 0.18), (21.0, 0.12)):
        cone(mbs["iron"], 5.25, -2.75, z - 0.08, z + 0.08, r, r, n=8)
    mbs["iron"].box(4.85, 5.65, -2.78, -2.72, 21.15, 21.21)      # weathervane arm
    # dormer on the main roof's front slope
    W_.box(0.1, 1.9, -2.7, -0.4, 8.2, 10.4)
    gable(R_, W_, 0.1, 1.9, -2.7, -0.4, 10.4, 11.5, "y", o=0.25, ends=(True, False))
    # chimneys
    mbs["brick"].box(-4.9, -3.9, 0.6, 1.6, 8.0, 13.8)
    mbs["brick"].box(-5.05, -3.75, 0.45, 1.75, 13.8, 14.1)
    mbs["brick"].box(1.9, 2.8, 1.4, 2.3, 9.0, 13.4)
    mbs["brick"].box(1.75, 2.95, 1.25, 2.45, 13.4, 13.7)
    # porch: floor, posts, rail, sloped roof, steps
    mbs["wood"].box(-2.5, 3.5, -6.3, -3.5, 0.6, 0.85)
    for x in (-2.3, -0.4, 1.4, 3.3):
        T_.box(x - 0.13, x + 0.13, -6.2, -5.94, 0.85, 3.7)
    for (xa, xb) in ((-2.3, -0.4), (1.4, 3.3)):
        T_.box(xa, xb, -6.12, -6.02, 1.75, 1.85)
        n = int((xb - xa) / 0.22)
        for k in range(1, n):
            x = xa + (xb - xa) * k / n
            T_.box(x - 0.03, x + 0.03, -6.1, -6.04, 0.85, 1.75)
    R_.poly([(-2.8, -6.6, 3.55), (3.8, -6.6, 3.55), (3.8, -3.5, 4.3), (-2.8, -3.5, 4.3),
             (-2.8, -6.6, 3.75), (3.8, -6.6, 3.75), (3.8, -3.5, 4.5), (-2.8, -3.5, 4.5)],
            [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3),
             (3, 7, 4, 0)])
    T_.box(-2.8, 3.8, -6.65, -6.55, 3.3, 3.6)                   # porch fascia
    for k in range(3):
        mbs["stone"].box(-0.5, 1.5, -7.3 + 0.33 * k, -6.3, 0.0, 0.6 - 0.2 * k)
    # door (main face, under the porch) with a transom window
    Fm = face((0, -3.5, 0), (1, 0, 0), (0, -1, 0))
    mbs["dark"].fbox(Fm, -0.05, 1.15, 0.85, 3.0, 0.0, 0.08)
    T_.fbox(Fm, -0.2, 1.3, 0.85, 3.2, 0.0, 0.05)
    window(mbs, Fm, 0.55, 3.45, 1.1, 0.38, "E", 0, shutters=False, pediment=False)
    # --- windows. EQ columns A..G, left to right on screen; lvl = floor.
    Fw_side = face((-7.0, 0, 0), (0, -1, 0), (-1, 0, 0))       # wing left side
    Fw_front = face((0, -5.5, 0), (1, 0, 0), (0, -1, 0))
    Ft = face((0, -4.5, 0), (1, 0, 0), (0, -1, 0))
    Fd = face((0, -2.7, 0), (1, 0, 0), (0, -1, 0))
    window(mbs, Fw_side, -0.6, 2.45, 1.1, 2.0, "A", 0, rng=rng)
    window(mbs, Fw_side, -0.6, 5.85, 1.1, 2.0, "A", 1, rng=rng, boarded=True)
    window(mbs, Fw_side, 3.4, 2.45, 1.1, 2.0, "B", 0, rng=rng)
    window(mbs, Fw_side, 3.4, 5.85, 1.1, 2.0, "B", 1, rng=rng)
    window(mbs, Fw_front, -4.75, 2.45, 1.3, 2.1, "C", 0, rng=rng)
    window(mbs, Fw_front, -4.75, 5.85, 1.3, 2.1, "C", 1, rng=rng)
    round_window(mbs, Fw_front, -4.75, 9.15, 0.62, "C", 2)
    window(mbs, Fm, -1.45, 2.45, 1.0, 1.9, "D", 0, shutters=False)
    window(mbs, Fm, -1.45, 5.85, 1.0, 1.9, "D", 1, rng=rng)
    window(mbs, Fm, 0.55, 5.85, 1.0, 1.9, "E", 1, rng=rng)
    window(mbs, Fd, 1.0, 9.25, 0.9, 1.3, "E", 2, shutters=False, pediment=False)
    window(mbs, Fm, 2.5, 2.45, 1.0, 1.9, "F", 0, shutters=False)
    window(mbs, Fm, 2.5, 5.85, 1.0, 1.9, "F", 1, rng=rng, boarded=True)
    window(mbs, Ft, 5.25, 2.45, 1.0, 2.0, "G", 0, shutters=False)
    window(mbs, Ft, 5.25, 5.85, 1.0, 2.0, "G", 1, shutters=False)
    window(mbs, Ft, 5.25, 9.35, 1.0, 2.0, "G", 2, shutters=False)
    # corner boards
    for (x, y) in ((-7.0, -5.5), (-2.5, -5.5), (-6.0, 3.5)):
        T_.box(x - 0.12, x + 0.12, y - 0.12, y + 0.12, 0.8, 7.6)
    for (x, y) in ((3.5, -4.5), (7.0, -4.5)):
        T_.box(x - 0.12, x + 0.12, y - 0.12, y + 0.12, 0.8, 11.4)
    obs = []
    for k, mb in mbs.items():
        if mb.v:
            ob = mb.build("house_" + k, M[k], coll)
            ob["role"] = "glass" if k == "glass" else "solid"
            obs.append(ob)
    return obs


# ---------------------------------------------------------------- terrain
def hill_z(x, y):
    r = math.sqrt((x / 1.7) ** 2 + (y / 1.0) ** 2)
    s = min(1.0, max(0.0, (r - 9.0) / 85.0))
    s = s * s * (3 - 2 * s)
    bump = 0.35 * math.sin(x * 0.21 + 1.3) * math.cos(y * 0.17) * min(1, s * 4)
    return -9.0 * s ** 0.8 - 0.6 + 0.6 * math.exp(-(r / 9.0) ** 4) + bump


def build_hill(coll, M):
    me = bpy.data.meshes.new("hill")
    bm = bmesh.new()
    try:
        bmesh.ops.create_grid(bm, x_segments=180, y_segments=120, size=1.0)
        for v in bm.verts:
            x = v.co.x * 160.0
            y = -95.0 + (v.co.y + 1.0) * 0.5 * 140.0
            v.co = Vector((x, y, hill_z(x, y)))
        bm.to_mesh(me)
    finally:
        bm.free()
    ob = bpy.data.objects.new("hill", me)
    ob.data.materials.append(M["grass"])
    coll.objects.link(ob)
    ob["role"] = "solid"
    # wrought-iron fence along the brow of the hill with a gate gap
    mb = MB()
    y = -13.0
    x = -26.0
    while x < 24.0:
        if -1.6 < x < 2.6:
            x += 0.24
            continue
        z0 = hill_z(x, y) - 0.2
        post = abs((x + 26.0) % 2.4) < 0.12
        h = 2.0 if post else 1.55
        w = 0.07 if post else 0.025
        mb.box(x - w, x + w, y - w, y + w, z0, z0 + h)
        cone(mb, x, y, z0 + h, z0 + h + (0.35 if post else 0.22), w * 2.2, 0.0, n=4)
        x += 0.24
    for zz in (0.35, 1.25):                                   # rails
        for xa in (-26.0, 2.6):
            xb = -1.6 if xa < 0 else 24.0
            n = 30
            for k in range(n):
                xx0 = xa + (xb - xa) * k / n
                xx1 = xa + (xb - xa) * (k + 1) / n
                z0 = hill_z(xx0, y) - 0.2 + zz
                z1 = hill_z(xx1, y) - 0.2 + zz
                mb.poly([(xx0, y - 0.03, z0), (xx1, y - 0.03, z1),
                         (xx1, y - 0.03, z1 + 0.06), (xx0, y - 0.03, z0 + 0.06)],
                        [(0, 1, 2, 3)])
    for xg in (-1.9, 2.9):                                     # gate pillars
        z0 = hill_z(xg, y) - 0.3
        mb.box(xg - 0.32, xg + 0.32, y - 0.32, y + 0.32, z0, z0 + 2.6)
        mb.box(xg - 0.42, xg + 0.42, y - 0.42, y + 0.42, z0 + 2.6, z0 + 2.8)
    # a gate leaf hanging open
    z0 = hill_z(-1.6, y) - 0.1
    for k in range(9):
        a = math.radians(55)
        px = -1.6 + 0.2 * k * math.cos(a)
        py = y - 0.2 * k * math.sin(a)
        mb.box(px - 0.025, px + 0.025, py - 0.025, py + 0.025, z0, z0 + 1.8)
    ob2 = mb.build("hill_fence", M["iron"], coll)
    ob2["role"] = "solid"
    pm = MB()
    pts = []
    for k in range(60):
        tt = k / 59.0
        y = -13.5 - 70.0 * tt
        x = 0.5 + 9.0 * math.sin(tt * 3.3 + 0.4) * tt - 14.0 * tt
        w = 1.1 + 2.2 * tt
        pts.append((x, y, w))
    for k in range(len(pts) - 1):
        (xa, ya, wa), (xb, yb, wb) = pts[k], pts[k + 1]
        pm.poly([(xa - wa, ya, hill_z(xa - wa, ya) + 0.04), (xa + wa, ya, hill_z(xa + wa, ya) + 0.04),
                 (xb + wb, yb, hill_z(xb + wb, yb) + 0.04), (xb - wb, yb, hill_z(xb - wb, yb) + 0.04)],
                [(0, 1, 2, 3)])
    ob3 = pm.build("hill_path", M["path"], coll)
    ob3["role"] = "solid"
    gm = MB()
    gr = random.Random(777)
    for _ in range(2600):
        x = gr.uniform(-70, 70)
        y = gr.uniform(-80, -10)
        z = hill_z(x, y)
        for _b in range(gr.randint(3, 6)):
            a_ = gr.uniform(0, 2 * math.pi)
            hh = gr.uniform(0.25, 0.7)
            lean = Vector((math.cos(a_), math.sin(a_), 0)) * gr.uniform(0.05, 0.3)
            side = Vector((-math.sin(a_), math.cos(a_), 0)) * 0.03
            q = Vector((x + gr.uniform(-0.2, 0.2), y + gr.uniform(-0.2, 0.2), z - 0.05))
            gm.poly([q - side, q + side, q + lean + Vector((0, 0, hh))], [(0, 1, 2)])
    ob4 = gm.build("hill_grass", M["dgrass"], coll)
    ob4["role"] = "solid"
    return [ob, ob2, ob3, ob4]


# ---------------------------------------------------------------- graveyard
PUMP = []      # dicts: obj, lamp, face polys (world), center, radius


FACES = {
    # (u, v) in pumpkin-front units (radius 1); eyes, nose, mouth
    "classic": dict(
        eyes=[[(-0.62, 0.12), (-0.18, 0.12), (-0.40, 0.48)],
              [(0.18, 0.12), (0.62, 0.12), (0.40, 0.48)]],
        nose=[[(-0.10, -0.08), (0.10, -0.08), (0.0, 0.10)]],
        mouth=[[(-0.70, -0.20), (-0.45, -0.28), (-0.38, -0.20), (-0.22, -0.32),
                (-0.06, -0.24), (0.06, -0.34), (0.22, -0.24), (0.38, -0.32),
                (0.45, -0.22), (0.70, -0.22), (0.50, -0.50), (0.25, -0.62),
                (0.15, -0.52), (0.0, -0.64), (-0.15, -0.52), (-0.25, -0.62),
                (-0.50, -0.50)]]),
    "angry": dict(
        eyes=[[(-0.66, 0.36), (-0.14, 0.12), (-0.50, 0.08)],
              [(0.66, 0.36), (0.14, 0.12), (0.50, 0.08)]],
        nose=[[(-0.08, -0.06), (0.08, -0.06), (0.0, 0.08)]],
        mouth=[[(-0.72, -0.14), (-0.50, -0.30), (-0.38, -0.18), (-0.20, -0.34),
                (0.0, -0.20), (0.20, -0.34), (0.38, -0.18), (0.50, -0.30),
                (0.72, -0.14), (0.45, -0.58), (0.22, -0.46), (0.0, -0.66),
                (-0.22, -0.46), (-0.45, -0.58)]]),
    "wail": dict(
        eyes=[[(-0.58, 0.22), (-0.42, 0.40), (-0.22, 0.24), (-0.30, 0.04), (-0.50, 0.04)],
              [(0.58, 0.22), (0.42, 0.40), (0.22, 0.24), (0.30, 0.04), (0.50, 0.04)]],
        nose=[[(-0.07, -0.04), (0.07, -0.04), (0.0, 0.08)]],
        mouth=[[(-0.24, -0.18), (0.0, -0.12), (0.24, -0.18), (0.30, -0.42),
                (0.16, -0.64), (0.0, -0.70), (-0.16, -0.64), (-0.30, -0.42)]]),
}


def build_pumpkin(coll, M, name, pos, size, yaw, style, seed):
    """Ribbed pumpkin shell (solidified), carved by an exact boolean."""
    rng = random.Random(seed)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    try:
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
        for v in bm.verts:
            x, y, z = v.co
            phi = math.atan2(y, x)
            groove = abs(math.cos(4.0 * phi)) ** 5
            r = 1.0 - 0.13 * groove + 0.025 * math.sin(7 * phi + seed)
            r *= 1.0 - 0.10 * abs(z) ** 3
            zz = z * 0.70
            if z > 0.6:                                   # dimple at the stem
                zz -= 0.25 * (z - 0.6) ** 1.5
            if z < -0.7:
                zz += 0.12 * (-z - 0.7)
            v.co = Vector((x * r, y * r, zz))
        bm.to_mesh(me)
    finally:
        bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(M["pskin"])
    ob.data.materials.append(M["pflesh"])
    for p in ob.data.polygons:
        p.use_smooth = True
    coll.objects.link(ob)
    sol = ob.modifiers.new("shell", "SOLIDIFY")
    sol.thickness = 0.09
    sol.offset = -1.0
    sol.material_offset = 1
    sol.material_offset_rim = 1
    # carve: one prism per hole, pointing along local -y (the face side)
    F = FACES[style]
    cm = MB()
    polys = {}
    for part in ("eyes", "nose", "mouth"):
        polys[part] = []
        for pg in F[part]:
            pts = [(u * 0.95, v * 0.95) for (u, v) in pg]
            n = len(pts)
            P = [(u, -2.0, v) for u, v in pts] + [(u, -0.2, v) for u, v in pts]
            fcs = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
            fcs += [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
            cm.poly(P, fcs)
            polys[part].append(pts)
    cut = cm.build(name + "_cut", M["pflesh"], coll)
    cut.hide_render = True
    cut.hide_viewport = True
    bo = ob.modifiers.new("carve", "BOOLEAN")
    bo.operation = "DIFFERENCE"
    bo.solver = "EXACT"
    bo.object = cut
    tilt = rng.uniform(-0.12, 0.12)
    ob.matrix_world = (Matrix.Translation(Vector(pos) + Vector((0, 0, size * 0.70)))
                       @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(tilt, 4, "Y")
                       @ Matrix.Scale(size, 4))
    cut.matrix_world = ob.matrix_world.copy()
    # stem
    sm = MB()
    lean = rng.uniform(-0.25, 0.25)
    cone(sm, 0.0, 0.0, 0.42, 0.95, 0.15, 0.08, n=7)
    sm.poly([(0.0, 0.0, 0.95), (lean, 0.05, 1.12), (lean + 0.03, 0.12, 1.08)], [(0, 1, 2)])
    st = sm.build(name + "_stem", M["pstem"], coll)
    st.matrix_world = ob.matrix_world.copy()
    # candle
    ld = bpy.data.lights.new(name + "_candle", "POINT")
    ld.energy = 14.0 * size ** 2
    ld.shadow_soft_size = 0.08 * size
    ld.color = (1.0, 1.0, 1.0)
    lo = bpy.data.objects.new(name + "_candle", ld)
    lo.location = ob.matrix_world @ Vector((0.0, 0.05, -0.25))
    coll.objects.link(lo)
    world_polys = {k: [[ob.matrix_world @ Vector((u, -1.0, v)) for (u, v) in pg]
                       for pg in pl] for k, pl in polys.items()}
    for o_ in (ob, st):
        o_["role"] = "solid"
    PUMP.append(dict(obj=ob, lamp=lo, polys=world_polys,
                     center=ob.matrix_world @ Vector((0, 0, 0)), size=size,
                     style=style))
    return [ob, st]


def tombstone(mb, kind, base, yaw, h, w, rng):
    x0, y0, z0 = base
    R = Matrix.Rotation(yaw, 3, "Z") @ Matrix.Rotation(rng.uniform(-0.12, 0.12), 3, "Y") \
        @ Matrix.Rotation(rng.uniform(-0.06, 0.10), 3, "X")
    org = Vector(base)

    def put(pts):
        return [tuple(org + R @ Vector(p)) for p in pts]

    d = 0.18 * w / 0.8
    if kind == "round":
        n = 12
        prof = [(-w / 2, 0.0), (w / 2, 0.0)]
        for k in range(n + 1):
            a = math.pi * k / n
            prof.append((w / 2 * math.cos(a), h - w / 2 + w / 2 * math.sin(a)))
        prof = [prof[0]] + [prof[1]] + prof[2:]
        m = len(prof)
        P = put([(u, -d / 2, v) for u, v in prof] + [(u, d / 2, v) for u, v in prof])
        fcs = [tuple(range(m)), tuple(range(2 * m - 1, m - 1, -1))]
        order = [0] + list(range(2, m)) + [1]
        for k in range(m):
            a_, b_ = order[k], order[(k + 1) % m]
            fcs.append((a_, b_, m + b_, m + a_))
        mb.poly(P, fcs)
    elif kind == "cross":
        for (u0, u1, v0, v1) in ((-w * 0.12, w * 0.12, 0.0, h),
                                 (-w * 0.45, w * 0.45, h * 0.62, h * 0.80)):
            c = [(u0, -d / 2, v0), (u1, -d / 2, v0), (u1, d / 2, v0), (u0, d / 2, v0),
                 (u0, -d / 2, v1), (u1, -d / 2, v1), (u1, d / 2, v1), (u0, d / 2, v1)]
            mb.hexa(put(c))
    elif kind == "obelisk":
        c = [(-w / 2, -w / 2, 0), (w / 2, -w / 2, 0), (w / 2, w / 2, 0), (-w / 2, w / 2, 0),
             (-w * 0.3, -w * 0.3, h * 0.85), (w * 0.3, -w * 0.3, h * 0.85),
             (w * 0.3, w * 0.3, h * 0.85), (-w * 0.3, w * 0.3, h * 0.85)]
        mb.hexa(put(c))
        mb.poly(put([(-w * 0.3, -w * 0.3, h * 0.85), (w * 0.3, -w * 0.3, h * 0.85),
                     (w * 0.3, w * 0.3, h * 0.85), (-w * 0.3, w * 0.3, h * 0.85),
                     (0, 0, h)]), [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
        c = [(-w * 0.7, -w * 0.7, -0.1), (w * 0.7, -w * 0.7, -0.1), (w * 0.7, w * 0.7, -0.1),
             (-w * 0.7, w * 0.7, -0.1), (-w * 0.7, -w * 0.7, 0.15), (w * 0.7, -w * 0.7, 0.15),
             (w * 0.7, w * 0.7, 0.15), (-w * 0.7, w * 0.7, 0.15)]
        mb.hexa(put(c))
    else:                                             # slab with a peaked top
        prof = [(-w / 2, 0.0), (w / 2, 0.0), (w / 2, h * 0.82), (0.0, h), (-w / 2, h * 0.82)]
        m = len(prof)
        P = put([(u, -d / 2, v) for u, v in prof] + [(u, d / 2, v) for u, v in prof])
        fcs = [tuple(range(m)), tuple(range(2 * m - 1, m - 1, -1))]
        fcs += [(k, (k + 1) % m, m + (k + 1) % m, m + k) for k in range(m)]
        mb.poly(P, fcs)
    return R, org, d


def ground_z(x, y):
    return (-0.06 * math.sin(x * 0.7) * math.cos(y * 0.9)
            + 0.05 * math.sin(x * 2.3 + y * 1.7))


def rise(d):
    """The graveyard climbs toward the hill: lifts the far graves + lanterns
    up the frame so the carved faces sit clear of the bottom edge."""
    return 0.032 * max(0.0, d - 4.0) + 0.9 * max(0.0, (d - 26.0) / 10.0) ** 2


def build_graveyard(coll, M, cam, scene, G0):
    """Everything near the camera. Placed by SCREEN position (u, depth)."""
    rng = random.Random(2077)
    fwd, rgt, cpos = cam["fwd"], cam["rgt"], cam["pos"]
    tanx = cam["tanx"]

    def at(u, d):
        # world point on the graveyard ground at screen-x u, depth d
        p = cpos + fwd * d + rgt * ((u - 0.5 + cam["sx"]) * 2 * tanx * d)
        return Vector((p.x, p.y, G0 + ground_z(p.x, p.y) + rise(d)))

    # ground: a big gently rolling sheet
    me = bpy.data.meshes.new("gy_ground")
    bm = bmesh.new()
    try:
        bmesh.ops.create_grid(bm, x_segments=160, y_segments=60, size=1.0)
        for v in bm.verts:
            u = (v.co.x + 1) * 0.5 * 1.6 - 0.3
            d = 3.0 + (v.co.y + 1) * 0.5 * 33.0
            v.co = at(u, d)
        bm.to_mesh(me)
    finally:
        bm.free()
    gob = bpy.data.objects.new("gy_ground", me)
    gob.data.materials.append(M["ground"])
    coll.objects.link(gob)
    gob["role"] = "solid"
    yaw_cam = math.atan2(-fwd.x, fwd.y)
    st = MB()
    plan = [  # kind, u, depth, height, width — the centre stays LOW so
        # the house on the hill reads (staging: the graveyard frames it)
        ("round", 0.13, 15.0, 1.05, 0.85), ("slab", 0.215, 21.0, 0.90, 0.75),
        ("cross", 0.345, 19.0, 1.20, 0.85), ("round", 0.45, 24.0, 0.75, 0.65),
        ("slab", 0.60, 20.0, 0.85, 0.75), ("round", 0.80, 19.0, 1.0, 0.80),
        ("cross", 0.835, 16.5, 1.55, 1.0), ("obelisk", 0.99, 17.0, 2.2, 0.55),
        ("round", 0.02, 22.0, 1.05, 0.80), ("slab", 0.30, 27.0, 0.75, 0.65),
        ("round", 0.69, 27.0, 0.8, 0.7),
    ]
    stones = []
    for kind, u, d, h, w in plan:
        b = at(u, d)
        R, org, dd = tombstone(st, kind, b, yaw_cam + rng.uniform(-0.35, 0.35), h, w, rng)
        stones.append((kind, R, org, dd, h, w))
    sob = st.build("gy_stones", M["tomb"], coll)
    sob["role"] = "solid"
    # raised "R.I.P." on two of the round stones
    for idx in (0, 5):
        kind, R, org, dd, h, w = stones[idx]
        cu = bpy.data.curves.new(f"rip{idx}", "FONT")
        cu.body = "R.I.P."
        cu.size = w * 0.30
        cu.extrude = 0.012
        cu.align_x = "CENTER"
        to = bpy.data.objects.new(f"rip{idx}", cu)
        to.data.materials.append(M["tomb"])
        Mt = Matrix.Translation(org) @ R.to_4x4() @ Matrix.Translation(
            Vector((0.0, -dd / 2 - 0.012, h * 0.55))) @ Matrix.Rotation(math.pi / 2, 4, "X")
        to.matrix_world = Mt
        coll.objects.link(to)
        to["role"] = "solid"
    # broken iron fence segment on the left, mid-depth
    fm = MB()
    for k in range(34):
        u = -0.02 + 0.0105 * k
        if k in (9, 10, 21):
            continue                                       # missing pickets
        p = at(u, 25.5)
        lean = rng.uniform(-0.05, 0.05) + (0.25 if k in (11, 12) else 0.0)
        top = p + Vector((lean, 0, 1.45))
        w_ = 0.025
        fm.poly([p + Vector((-w_, -w_, 0)), p + Vector((w_, -w_, 0)), p + Vector((w_, w_, 0)),
                 p + Vector((-w_, w_, 0)), top + Vector((-w_, -w_, 0)), top + Vector((w_, -w_, 0)),
                 top + Vector((w_, w_, 0)), top + Vector((-w_, w_, 0))],
                [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
                 (3, 0, 4, 7)])
        cone(fm, top.x, top.y, top.z, top.z + 0.2, 0.055, 0.0, n=4)
    for zz in (0.3, 1.15):
        pa, pb = at(-0.02, 25.5), at(0.33, 25.5)
        fm.poly([pa + Vector((0, 0, zz)), pb + Vector((0, 0, zz)),
                 pb + Vector((0, 0, zz + 0.06)), pa + Vector((0, 0, zz + 0.06))], [(0, 1, 2, 3)])
    fob = fm.build("gy_fence", M["iron"], coll)
    fob["role"] = "solid"
    # dead grass tufts (thin blades) scattered over the ground
    gm = MB()
    gr = random.Random(5150)
    for _ in range(700):
        u = gr.uniform(-0.25, 1.25)
        d = 6.0 + 27.0 * gr.random() ** 1.3
        p = at(u, d)
        for _b in range(gr.randint(3, 6)):
            a = gr.uniform(0, 2 * math.pi)
            hh = gr.uniform(0.10, 0.32)
            lean = Vector((math.cos(a), math.sin(a), 0)) * gr.uniform(0.05, 0.25)
            ww = 0.010
            side = Vector((-math.sin(a), math.cos(a), 0)) * ww
            q = p + Vector((gr.uniform(-0.1, 0.1), gr.uniform(-0.1, 0.1), 0))
            gm.poly([q - side, q + side, q + lean + Vector((0, 0, hh))], [(0, 1, 2)])
    grob = gm.build("gy_grass", M["dgrass"], coll)
    grob["role"] = "solid"
    out = [gob, sob, fob, grob] + [o for o in coll.objects if o.name.startswith("rip")]
    # pumpkins: (u, depth, size(radius m), style)
    pplan = [(0.075, 14.0, 0.52, "angry"), (0.27, 13.0, 0.58, "classic"),
             (0.50, 17.0, 0.44, "wail"), (0.72, 13.5, 0.56, "classic"),
             (0.925, 13.5, 0.50, "angry")]
    for k, (u, d, s, style) in enumerate(pplan):
        p = at(u, d)
        to_cam = cpos - p
        yaw = math.atan2(to_cam.x, -to_cam.y) + rng.uniform(-0.3, 0.3)
        out += build_pumpkin(coll, M, f"pump{k}", p, s, yaw, style, 300 + k)
    return out


# ---------------------------------------------------------------- render
def setup_render(scene, w, h, samples):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.seed = 7
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 4
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 16


def render(scene, path, mode="RGBA"):
    """mode: RGBA for passes whose coverage we need, BW / RGB otherwise
    (the compositor takes coverage from the key pass) — keeps assets small."""
    scene.render.image_settings.color_mode = mode
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("[halloween] wrote", path, flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--preview", action="store_true",
                    help="one quick combined beauty render only")
    ap.add_argument("--only", default="", help="comma list of pass names")
    ap.add_argument("--meta-only", action="store_true", help="write meta.json only")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    W, H = int(BASE_W * a.scale), int(BASE_H * a.scale)

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    scene = bpy.context.scene
    setup_render(scene, W, H, a.samples)
    world = scene.world or bpy.data.worlds.new("w")
    scene.world = world
    wn = world.node_tree
    bg = wn.nodes.get("Background") or wn.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    M = dict(
        wall=grey("wall", 0.30, 0.85, "boards", 4.5, 0.35),
        roof=grey("roof", 0.16, 0.9, "shingle", 3.2, 0.5),
        trim=grey("trim", 0.48, 0.7),
        dark=grey("dark", 0.13, 0.8),
        glass=grey("glass", 0.03, 0.15),
        stone=grey("stone", 0.30, 0.9, "noise", 3.0, 0.6),
        brick=grey("brick", 0.24, 0.9, "brick", 4.0, 0.4),
        wood=grey("wood", 0.22, 0.85, "boards", 9.0, 0.3),
        iron=grey("iron", 0.10, 0.45),
        grass=grey("grass", 0.075, 1.0, "noise", 1.1, 1.2),
        ground=grey("ground", 0.15, 1.0, "noise", 1.3, 0.7),
        dgrass=grey("dgrass", 0.20, 1.0),
        path=grey("path", 0.17, 1.0, "noise", 2.0, 0.8),
        tomb=grey("tomb", 0.42, 0.9, "noise", 4.0, 0.9),
        pskin=grey("pskin", (0.85, 0.0, 0.0), 0.45, "noise", 6.0, 0.15),
        pflesh=grey("pflesh", (0.95, 0.0, 0.0), 0.8),
        pstem=grey("pstem", (0.0, 0.0, 0.45), 0.9),
    )
    hc = bpy.data.collections.new("house")
    gc = bpy.data.collections.new("graveyard")
    scene.collection.children.link(hc)
    scene.collection.children.link(gc)
    # camera: low, 30 m left of the house's axis, yawed toward it
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = 50.0
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.sensor_width = 36.0
    cam = bpy.data.objects.new("cam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cpos = Vector((-24.0, -88.0, -7.0))
    yaw = math.atan2(24.0, 88.0)               # look toward x=0 at the house
    cam.location = cpos
    cam.rotation_euler = (math.radians(90.0), 0.0, -yaw)
    cam_d.shift_x = 0.105
    cam_d.shift_y = 0.199
    fwd = Vector((math.sin(yaw), math.cos(yaw), 0.0))
    rgt = Vector((math.cos(yaw), -math.sin(yaw), 0.0))
    tanx = 18.0 / 50.0
    G0 = cpos.z - 0.55                           # graveyard ground level
    camI = dict(pos=cpos, fwd=fwd, rgt=rgt, tanx=tanx, sx=cam_d.shift_x)

    house = build_house(hc, M)
    house += build_hill(hc, M)
    grave = build_graveyard(gc, M, camI, scene, G0)

    # lights: the MOON (key: high, from the right, grazing the fronts) and a
    # cold sky FILL from the front-left. Both white — colour comes later.
    def sun(name, d, strength, ang):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = strength
        ld.angle = math.radians(ang)
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = (-Vector(d)).normalized().to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)
        return lo

    key = sun("moon", (0.80, -0.15, 0.62), 2.0, 0.6)
    fill = sun("fill", (-0.55, -1.0, 0.35), 0.55, 8.0)
    candles = [p["lamp"] for p in PUMP]

    bpy.context.view_layer.update()      # camera matrix_world is stale until evaluated

    def project(p):
        c = world_to_camera_view(scene, cam, p)
        return [round(c.x * W, 2), round((1.0 - c.y) * H, 2), round(c.z, 3)]

    meta = dict(
        size=[W, H], cam=dict(pos=list(cpos), shift=[cam_d.shift_x, cam_d.shift_y]),
        windows=[dict(poly=[project(p)[:2] for p in w_["c"]], col=w_["col"],
                      lvl=w_["lvl"]) for w_ in WIN],
        pumpkins=[dict(center=project(p["center"]), size=p["size"], style=p["style"],
                       polys={k: [[project(q)[:2] for q in pg] for pg in v]
                              for k, v in p["polys"].items()})
                  for p in PUMP],
        anchors=dict(door=project(Vector((0.55, -3.6, 2.0))),
                     spire=project(Vector((5.25, -2.75, 21.6))),
                     tower_top=project(Vector((5.25, -4.6, 9.4))),
                     chimney1=project(Vector((-4.4, 1.1, 14.1))),
                     chimney2=project(Vector((2.35, 1.85, 13.7))),
                     attic=project(Vector((-4.75, -5.6, 9.15))),
                     house_mid=project(Vector((0.0, -3.5, 7.0))),
                     hill_crest=project(Vector((0.0, -13.0, hill_z(0.0, -13.0))))),
    )
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("[halloween] meta:", len(WIN), "windows,", len(PUMP), "pumpkins", flush=True)

    def vis(objs, on):
        for o in objs:
            o.hide_render = not on

    def lights(k=False, f=False, cands=(), world_s=0.0):
        key.hide_render = not k
        fill.hide_render = not f
        for j, c in enumerate(candles):
            c.hide_render = j not in cands
        bg.inputs["Strength"].default_value = world_s

    def glass_mode(mode):
        """'normal' | 'mask' (glass white, all else black) | 'emit'."""
        for o in house:
            if mode == "mask":
                o.data.materials[0] = M_mask_w if o["role"] == "glass" else M_mask_k
            elif mode == "emit" and o["role"] == "glass":
                o.data.materials[0] = M_emit
            else:
                o.data.materials[0] = M[orig[o.name]]

    M_mask_w = emission("mask_w", 1.0)
    M_mask_k = emission("mask_k", 0.0)
    M_emit = emission("emit", 2.5)
    orig = {}
    for o in house:
        nm = o.data.materials[0].name
        orig[o.name] = nm
    allp = tuple(range(len(candles)))
    only = set(x for x in a.only.split(",") if x)

    def want(nm):
        return not only or nm in only

    if a.meta_only:
        return
    if a.preview:
        vis(house, True)
        vis(grave, True)
        lights(True, True, allp, 0.02)
        render(scene, os.path.join(a.out, "preview.png"))
        return
    # ---- house layer
    vis(grave, False)
    vis(house, True)
    if want("h_key"):
        lights(k=True)
        render(scene, os.path.join(a.out, "h_key.png"))
    if want("h_fill"):
        lights(f=True, world_s=0.05)
        render(scene, os.path.join(a.out, "h_fill.png"), "BW")
    if want("h_glass"):
        glass_mode("mask")
        lights()
        s0 = scene.cycles.samples
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        render(scene, os.path.join(a.out, "h_glass.png"), "BW")
        scene.cycles.samples = s0
        scene.cycles.use_denoising = True
        glass_mode("normal")
    if want("h_spill"):
        glass_mode("emit")
        lights()
        render(scene, os.path.join(a.out, "h_spill.png"), "BW")
        glass_mode("normal")
    # ---- graveyard layer
    vis(house, False)
    vis(grave, True)
    if want("f_key"):
        lights(k=True)
        render(scene, os.path.join(a.out, "f_key.png"))
    if want("f_fill"):
        lights(f=True, world_s=0.05)
        render(scene, os.path.join(a.out, "f_fill.png"), "RGB")
    for j in range(len(candles)):
        if want(f"f_lit{j}") or want("f_lit"):
            lights(cands=(j,))
            render(scene, os.path.join(a.out, f"f_lit{j}.png"), "BW")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
