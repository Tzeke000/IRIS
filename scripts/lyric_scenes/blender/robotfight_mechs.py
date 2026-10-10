"""Robot-fight scene assets: the two mechs as per-limb RGBA sprites.

Run HEADLESS (never through the Blender MCP / GUI):

    nice -n 15 ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/blender/robotfight_mechs.py -- \
        [--out assets/scenes/robotfight/mechs] [--ppu 900] [--cpu]

Every limb of both mechs is modelled from bevelled primitives (bmesh, no
imported assets), posed bone-vertical with its JOINT at the origin, turned
YAW degrees toward the camera (a 3/4 view, so the plates read as volumes),
and rendered orthographically twice:

  <mech>_<part>_lit.png  neutral-grey LIT pass (key/fill/rim suns + a horizon
                         gradient world so the metal has reflections), RGBA
  <mech>_<part>_id.png   MATERIAL-ID pass, flat emission: R = paint (palette
                         primary), G = frame metal (palette dark), B = glow
                         strips (palette emissive, animated per frame)

meta.json records, per part, the sprite size, the pivot pixel (the joint),
the pixels-per-unit, the bone length and named anchor points (core, neck,
shoulders, hips ...) in sprite pixels. robotfight.py colourises the grey
pass per palette, so one render serves every palette.

Deterministic: fixed geometry, fixed Cycles seed, no randomness.
Coordinates: Blender X = forward (toward the opponent), Z = up, the camera
looks along +Y from -Y, so screen-right = +X and screen-up = +Z.
"""
import argparse
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

YAW = -22.0          # deg; negative turns the mech's front toward the camera

# ---------------------------------------------------------------------------
# geometry helpers (bmesh + bevel modifier; no bpy.ops in loops)
# ---------------------------------------------------------------------------
_PARTS: dict = {}     # part name -> list of objects
_CUR = None           # (part name, parent empty)


def _new_obj(name, bm, mat, bevel=0.0, seg=2):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = False
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if bevel > 0:
        m = ob.modifiers.new("bev", "BEVEL")
        m.width = bevel
        m.segments = seg
        m.limit_method = "ANGLE"
        m.harden_normals = False
    pname, parent = _CUR
    ob.parent = parent
    _PARTS[pname].append(ob)
    return ob


def box(c, s, mat, bevel=0.012, rot=(0, 0, 0), taper=(1.0, 1.0),
        taper_x=(1.0, 1.0), shear=0.0):
    """Box centred at c with size s. taper=(top_y, bottom_y) scales the +Z /
    -Z faces along Y; taper_x likewise along X; shear moves the top face +X."""
    bm = bmesh.new()
    try:
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            top = v.co.z > 0
            ty = taper[0] if top else taper[1]
            tx = taper_x[0] if top else taper_x[1]
            v.co = Vector((v.co.x * s[0] * tx + (shear * s[2] if top else 0.0),
                           v.co.y * s[1] * ty, v.co.z * s[2]))
        R = (Matrix.Rotation(math.radians(rot[2]), 4, "Z")
             @ Matrix.Rotation(math.radians(rot[1]), 4, "Y")
             @ Matrix.Rotation(math.radians(rot[0]), 4, "X"))
        bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(c)) @ R,
                            verts=bm.verts)
        return _new_obj("box", bm, mat, bevel)
    finally:
        bm.free()


def cyl(c, r, depth, mat, axis="Y", segs=28, bevel=0.006, r2=None):
    bm = bmesh.new()
    try:
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs,
                              radius1=r, radius2=r if r2 is None else r2,
                              depth=depth)
        R = {"Z": Matrix.Identity(4),
             "Y": Matrix.Rotation(math.radians(90), 4, "X"),
             "X": Matrix.Rotation(math.radians(90), 4, "Y")}[axis]
        bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(c)) @ R,
                            verts=bm.verts)
        return _new_obj("cyl", bm, mat, bevel, seg=1)
    finally:
        bm.free()


def sphere(c, r, mat):
    bm = bmesh.new()
    try:
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=r)
        bmesh.ops.translate(bm, vec=Vector(c), verts=bm.verts)
        ob = _new_obj("sph", bm, mat)
        for p in ob.data.polygons:
            p.use_smooth = True
        return ob
    finally:
        bm.free()


def strip(p0, p1, y, mat, w=0.008, d=0.006):
    """A thin glow strip from p0 to p1 (x, z) lying on the surface at depth y
    (camera side is -Y)."""
    x0, z0 = p0
    x1, z1 = p1
    L = math.hypot(x1 - x0, z1 - z0)
    ang = math.degrees(math.atan2(z1 - z0, x1 - x0))
    return box(((x0 + x1) / 2, y, (z0 + z1) / 2), (L, d, w), mat, bevel=0.0,
               rot=(0, -ang, 0))


def bolts(pts, y, mat, r=0.008):
    for x, z in pts:
        cyl((x, y, z), r, 0.008, mat, axis="Y", segs=12, bevel=0.0)


def part(name):
    global _CUR
    e = bpy.data.objects.new("P_" + name, None)
    bpy.context.scene.collection.objects.link(e)
    e.rotation_euler = (0, 0, math.radians(YAW))
    _PARTS[name] = []
    _CUR = (name, e)
    return e


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------
def _principled(name, base, metal, rough, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
    return m


def _emit(name, col):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*col, 1.0)
    e.inputs["Strength"].default_value = 1.0
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs["Emission"], o.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------
# the two mechs. Units: 1.0 = roughly a mech's standing height.
# Every part: joint at the origin. Limbs hang along -Z (bone = 0 -> -L),
# torso/head rise along +Z. Camera side = -Y.
# ---------------------------------------------------------------------------
def build_brawler(P, F, G):
    """Mech A — the BRAWLER: wide, low, slab armour, huge fists."""
    meta = {}
    Y = -1.0  # camera side helper sign
    # TORSO (hip -> neck, 0.34)
    part("A_torso")
    box((0.0, 0, 0.0), (0.17, 0.20, 0.10), F, 0.015)              # pelvis
    box((0.0, 0, -0.035), (0.13, 0.22, 0.05), P, 0.012)           # belt plate
    cyl((0.0, 0, 0.08), 0.055, 0.12, F, axis="Z")                  # waist
    for k in range(3):                                             # waist rings
        cyl((0.0, 0, 0.045 + 0.03 * k), 0.062, 0.012, F, axis="Z", bevel=0.0)
    box((0.005, 0, 0.245), (0.30, 0.30, 0.22), P, 0.035, taper_x=(1.0, 0.82))
    box((0.135, 0, 0.255), (0.07, 0.25, 0.17), P, 0.025, shear=0.15)  # chest front
    box((-0.165, 0, 0.27), (0.09, 0.22, 0.19), F, 0.02)            # reactor pack
    for k in range(4):                                             # pack vents
        strip((-0.205, 0.215 + 0.03 * k), (-0.13, 0.215 + 0.03 * k), Y * 0.111,
              G, w=0.007)
    cyl((0.0, -0.152, 0.235), 0.072, 0.03, F, axis="Y")            # core socket
    cyl((0.0, -0.163, 0.235), 0.058, 0.012, G, axis="Y", bevel=0.0)  # core lens
    strip((-0.11, 0.14), (-0.075, 0.14), -0.151, G)
    strip((0.075, 0.14), (0.125, 0.14), -0.151, G)
    strip((0.10, 0.31), (0.10, 0.355), -0.151, G)
    bolts([(-0.12, 0.33), (0.12, 0.33), (-0.12, 0.155), (0.13, 0.17)], -0.152, F)
    box((0.0, 0, 0.36), (0.13, 0.16, 0.05), F, 0.012)              # collar
    meta["A_torso"] = dict(bone=(0, 0.34), anchors=dict(
        core=(0.0, -0.17, 0.235), neck=(0.0, 0, 0.34),
        sh_n=(0.0, -0.17, 0.30), sh_f=(0.0, 0.17, 0.30),
        hip_n=(0.0, -0.075, 0.0), hip_f=(0.0, 0.075, 0.0)))
    # HEAD (neck -> top)
    part("A_head")
    box((0.025, 0, 0.065), (0.145, 0.14, 0.115), P, 0.03, taper_x=(0.85, 1.0))
    box((0.095, 0, 0.035), (0.05, 0.12, 0.06), F, 0.012)           # jaw guard
    box((0.088, 0, 0.083), (0.03, 0.125, 0.022), G, 0.004)         # visor slit
    strip((0.03, 0.083), (0.075, 0.083), -0.071, G, w=0.014)       # visor wrap
    cyl((-0.01, -0.07, 0.06), 0.03, 0.02, F, axis="Y")             # ear unit
    cyl((-0.01, -0.081, 0.06), 0.017, 0.006, G, axis="Y", bevel=0.0)
    box((-0.01, 0, 0.13), (0.10, 0.03, 0.025), F, 0.006)           # top ridge
    meta["A_head"] = dict(bone=(0, 0.12), anchors=dict(
        face=(0.10, 0, 0.075), center=(0.03, 0, 0.065)))
    # UPPER ARM (shoulder -> elbow, 0.23)
    part("A_uarm")
    box((-0.005, 0, 0.005), (0.19, 0.21, 0.14), P, 0.045, taper_x=(0.8, 1.0))
    box((0.0, 0, -0.06), (0.17, 0.19, 0.03), F, 0.008)             # pauldron rim
    strip((-0.07, 0.035), (0.07, 0.035), -0.106, G)
    cyl((0.0, 0, -0.14), 0.045, 0.19, F, axis="Z")
    box((0.01, 0, -0.14), (0.08, 0.10, 0.10), P, 0.02)
    cyl((0.0, 0, -0.23), 0.05, 0.11, F, axis="Y")                  # elbow hub
    meta["A_uarm"] = dict(bone=(0, -0.23), anchors={})
    # FOREARM (elbow -> wrist, 0.23)
    part("A_farm")
    cyl((0.0, 0, 0.0), 0.052, 0.115, F, axis="Y")
    cyl((0.0, -0.06, 0.0), 0.03, 0.01, G, axis="Y", bevel=0.0)
    box((0.0, 0, -0.13), (0.13, 0.14, 0.20), P, 0.03, taper_x=(0.8, 1.12),
        taper=(0.85, 1.1))
    strip((-0.045, -0.06), (-0.045, -0.20), -0.076, G)
    box((0.0, 0, -0.225), (0.10, 0.12, 0.03), F, 0.006)            # cuff
    meta["A_farm"] = dict(bone=(0, -0.23), anchors={})
    # FIST (wrist -> knuckles, 0.12)
    part("A_fist")
    box((0.0, 0, -0.065), (0.13, 0.13, 0.12), F, 0.022)
    box((0.015, 0, -0.07), (0.10, 0.135, 0.09), P, 0.02)          # back plate
    for k in range(4):
        box((0.0, -0.045 + 0.03 * k, -0.128), (0.12, 0.026, 0.03), F, 0.008)
    box((-0.06, -0.02, -0.06), (0.03, 0.05, 0.07), F, 0.01)        # thumb
    meta["A_fist"] = dict(bone=(0, -0.13), anchors=dict(
        knuckle=(0.0, 0, -0.14)))
    # THIGH (hip -> knee, 0.24)
    part("A_thigh")
    cyl((0.0, 0, 0.0), 0.065, 0.13, F, axis="Y")
    box((0.012, 0, -0.115), (0.14, 0.15, 0.20), P, 0.03, taper_x=(1.0, 0.8))
    strip((0.05, -0.05), (0.05, -0.17), -0.076, G)
    cyl((0.0, 0, -0.24), 0.055, 0.12, F, axis="Y")
    meta["A_thigh"] = dict(bone=(0, -0.24), anchors={})
    # SHIN (knee -> ankle, 0.24)
    part("A_shin")
    box((0.045, 0, -0.01), (0.09, 0.13, 0.09), P, 0.025)           # knee cap
    box((0.008, 0, -0.13), (0.12, 0.14, 0.21), P, 0.03, taper_x=(1.0, 1.15))
    cyl((-0.06, 0, -0.12), 0.018, 0.2, F, axis="Z")                 # piston
    strip((0.04, -0.07), (0.04, -0.2), -0.071, G)
    meta["A_shin"] = dict(bone=(0, -0.24), anchors={})
    # FOOT (ankle -> toe, along +X; sits flat)
    part("A_foot")
    cyl((0.0, 0, 0.0), 0.045, 0.12, F, axis="Y")
    box((0.035, 0, -0.04), (0.25, 0.16, 0.055), F, 0.012)
    box((0.11, 0, -0.03), (0.10, 0.165, 0.05), P, 0.015, shear=-0.4)
    box((-0.075, 0, -0.035), (0.05, 0.15, 0.05), P, 0.012)
    meta["A_foot"] = dict(bone=(0.2, 0), anchors=dict(sole=(0.0, 0, -0.068)))
    return meta


def build_striker(P, F, G):
    """Mech B — the STRIKER: tall, narrow, V-chest, crest, forearm blades,
    reverse-jointed (digitigrade) legs."""
    meta = {}
    part("B_torso")
    box((0.0, 0, 0.0), (0.12, 0.15, 0.08), F, 0.012)
    cyl((0.0, 0, 0.075), 0.035, 0.12, F, axis="Z")
    for k in range(4):
        cyl((0.0, 0, 0.04 + 0.025 * k), 0.045, 0.01, F, axis="Z", bevel=0.0)
    box((0.01, 0, 0.225), (0.27, 0.23, 0.20), P, 0.025, taper_x=(1.0, 0.45),
        taper=(1.0, 0.6))
    box((0.11, 0, 0.27), (0.05, 0.18, 0.10), P, 0.015, shear=0.3)
    # back fins — the silhouette spikes
    box((-0.12, -0.05, 0.36), (0.035, 0.02, 0.17), P, 0.006, rot=(0, -28, 0))
    box((-0.12, 0.05, 0.36), (0.035, 0.02, 0.17), P, 0.006, rot=(0, -28, 0))
    strip((-0.16, 0.30), (-0.09, 0.42), -0.062, G, w=0.006)
    box((0.02, -0.118, 0.23), (0.07, 0.02, 0.07), F, 0.008, rot=(0, 45, 0))
    box((0.02, -0.127, 0.23), (0.045, 0.01, 0.045), G, 0.0, rot=(0, 45, 0))
    strip((-0.08, 0.30), (0.0, 0.18), -0.116, G, w=0.006)
    strip((0.12, 0.30), (0.045, 0.18), -0.116, G, w=0.006)
    box((0.0, 0, 0.315), (0.10, 0.12, 0.04), F, 0.01)
    meta["B_torso"] = dict(bone=(0, 0.30), anchors=dict(
        core=(0.02, -0.13, 0.23), neck=(0.0, 0, 0.30),
        sh_n=(0.0, -0.135, 0.28), sh_f=(0.0, 0.135, 0.28),
        hip_n=(0.0, -0.06, 0.0), hip_f=(0.0, 0.06, 0.0)))
    part("B_head")
    box((0.035, 0, 0.06), (0.13, 0.095, 0.085), P, 0.02, taper_x=(0.75, 1.0),
        shear=-0.2)
    box((0.085, 0, 0.03), (0.07, 0.08, 0.035), F, 0.008)
    strip((0.06, 0.085), (0.11, 0.06), -0.049, G, w=0.012)          # V visor
    strip((0.06, 0.085), (0.02, 0.075), -0.049, G, w=0.008)
    box((-0.045, 0, 0.13), (0.03, 0.018, 0.16), P, 0.004, rot=(0, -50, 0))  # crest
    box((-0.01, 0, 0.11), (0.05, 0.02, 0.06), P, 0.004, rot=(0, -30, 0))
    strip((-0.03, 0.11), (-0.10, 0.19), -0.011, G, w=0.005)
    meta["B_head"] = dict(bone=(0, 0.11), anchors=dict(
        face=(0.10, 0, 0.065), center=(0.035, 0, 0.06)))
    part("B_uarm")
    sphere((0.0, 0, 0.0), 0.052, F)
    box((-0.01, 0, 0.03), (0.13, 0.15, 0.07), P, 0.015, shear=-0.6)
    box((-0.05, 0, 0.09), (0.025, 0.02, 0.12), P, 0.004, rot=(0, -35, 0))  # spike
    cyl((0.0, 0, -0.13), 0.034, 0.22, F, axis="Z")
    box((0.012, 0, -0.12), (0.06, 0.08, 0.13), P, 0.015)
    strip((0.03, -0.07), (0.03, -0.17), -0.041, G, w=0.006)
    cyl((0.0, 0, -0.25), 0.04, 0.09, F, axis="Y")
    meta["B_uarm"] = dict(bone=(0, -0.25), anchors={})
    part("B_farm")
    cyl((0.0, 0, 0.0), 0.042, 0.095, F, axis="Y")
    box((0.01, 0, -0.13), (0.075, 0.09, 0.21), P, 0.015, taper_x=(0.85, 1.1))
    # blade along the back of the forearm, sweeping past the elbow
    box((-0.055, 0, -0.07), (0.02, 0.03, 0.36), P, 0.003, taper_x=(0.3, 1.0),
        rot=(0, 6, 0))
    strip((-0.065, 0.08), (-0.048, -0.22), -0.016, G, w=0.005)
    box((0.0, 0, -0.24), (0.07, 0.08, 0.025), F, 0.005)
    meta["B_farm"] = dict(bone=(0, -0.25), anchors={})
    part("B_fist")
    box((0.0, 0, -0.05), (0.095, 0.10, 0.095), F, 0.016)
    box((0.012, 0, -0.055), (0.07, 0.105, 0.07), P, 0.012)
    for k in range(3):
        box((0.0, -0.03 + 0.03 * k, -0.105), (0.025, 0.018, 0.035), P, 0.003,
            taper_x=(0.3, 1.0), rot=(180, 0, 0))                    # knuckle spikes
    meta["B_fist"] = dict(bone=(0, -0.10), anchors=dict(
        knuckle=(0.0, 0, -0.115)))
    part("B_thigh")
    sphere((0.0, 0, 0.0), 0.05, F)
    box((0.015, 0, -0.10), (0.11, 0.12, 0.18), P, 0.02, taper_x=(1.0, 0.7))
    strip((0.045, -0.04), (0.035, -0.15), -0.061, G, w=0.006)
    cyl((0.0, 0, -0.22), 0.042, 0.1, F, axis="Y")
    meta["B_thigh"] = dict(bone=(0, -0.22), anchors={})
    part("B_shin")
    box((-0.045, 0, 0.0), (0.05, 0.08, 0.05), P, 0.01, rot=(0, 30, 0))  # knee spur
    box((-0.03, 0, 0.03), (0.02, 0.02, 0.09), P, 0.003, rot=(0, -40, 0))
    box((0.0, 0, -0.14), (0.075, 0.09, 0.25), P, 0.018, taper_x=(1.0, 0.65))
    cyl((0.035, 0, -0.14), 0.013, 0.24, F, axis="Z")
    strip((-0.015, -0.05), (-0.01, -0.24), -0.046, G, w=0.006)
    cyl((0.0, 0, -0.28), 0.035, 0.08, F, axis="Y")
    meta["B_shin"] = dict(bone=(0, -0.28), anchors={})
    part("B_foot")
    cyl((0.0, 0, 0.0), 0.035, 0.09, F, axis="Y")
    for k, yy in enumerate((-0.035, 0.0, 0.035)):
        box((0.075, yy, -0.035), (0.13, 0.025, 0.035), F, 0.006, taper_x=(1.0, 1.0),
            shear=0.0, rot=(0, -8, 0))
        box((0.15, yy, -0.045), (0.035, 0.022, 0.025), P, 0.004, rot=(0, 30, 0))
    box((-0.06, 0, -0.04), (0.08, 0.03, 0.03), F, 0.006, rot=(0, 25, 0))  # spur
    box((0.02, 0, -0.02), (0.08, 0.10, 0.04), P, 0.01)
    meta["B_foot"] = dict(bone=(0.2, 0), anchors=dict(sole=(0.0, 0, -0.058)))
    return meta


# ---------------------------------------------------------------------------
# scene + render
# ---------------------------------------------------------------------------
def setup_scene(use_cpu: bool):
    sc = bpy.context.scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    sc.render.engine = "CYCLES"
    sc.cycles.seed = 0
    if not use_cpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type == "CUDA"
            sc.cycles.device = "GPU"
        except Exception as e:  # noqa: BLE001
            print("[robotfight_mechs] GPU unavailable, CPU render:", e)
            sc.cycles.device = "CPU"
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"
    sc.render.resolution_percentage = 100
    # world: horizon gradient (bright above, dark below) so metal reflects
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.40
    ramp.color_ramp.elements[0].color = (0.015, 0.015, 0.02, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.9, 0.9, 0.95, 1)
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -1.0
    mr.inputs["From Max"].default_value = 1.0
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.45
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    # suns: key (top-front, camera side), fill (camera), rim (behind-top)
    for name, rot, e, ang in (("key", (38, 0, -28), 2.6, 12),
                              ("fill", (85, 0, -5), 0.35, 30),
                              ("rim", (-50, 0, 160), 3.0, 6)):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = e
        ld.angle = math.radians(ang)
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = tuple(math.radians(v) for v in rot)
        sc.collection.objects.link(lo)
    cd = bpy.data.cameras.new("cam")
    cd.type = "ORTHO"
    cam = bpy.data.objects.new("cam", cd)
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def part_bounds(objs):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    xs, zs = [], []
    for ob in objs:
        ev = ob.evaluated_get(dg)
        for c in ev.bound_box:
            v = ob.matrix_world @ Vector(c)
            xs.append(v.x)
            zs.append(v.z)
    return min(xs), max(xs), min(zs), max(zs)


def yaw_xz(p):
    """(x, y, z) in part space -> (screen x, screen z) after the YAW turn."""
    a = math.radians(YAW)
    x, y, z = p
    return (x * math.cos(a) - y * math.sin(a), z)


def render_part(cam, name, objs, ppu, out_dir, mats, idmats, samples):
    sc = bpy.context.scene
    for ob in bpy.data.objects:
        if ob.type == "MESH":
            ob.hide_render = ob not in objs
    x0, x1, z0, z1 = part_bounds(objs)
    pad = 0.02
    x0, x1, z0, z1 = x0 - pad, x1 + pad, z0 - pad, z1 + pad
    rx = int(math.ceil((x1 - x0) * ppu))
    rz = int(math.ceil((z1 - z0) * ppu))
    ww, hw = rx / ppu, rz / ppu
    sc.render.resolution_x, sc.render.resolution_y = rx, rz
    cam.data.ortho_scale = max(ww, hw)
    cam.location = (x0 + ww / 2, -5.0, z0 + hw / 2)
    # lit pass
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.filepath = os.path.join(out_dir, f"{name}_lit.png")
    bpy.ops.render.render(write_still=True)
    # material-ID pass: swap every material for its flat-emission twin
    saved = []
    for ob in objs:
        saved.append([s.material for s in ob.material_slots])
        for s in ob.material_slots:
            s.material = idmats[s.material.name]
    sc.cycles.samples = 16
    sc.cycles.use_denoising = False
    sc.render.filepath = os.path.join(out_dir, f"{name}_id.png")
    bpy.ops.render.render(write_still=True)
    for ob, ms in zip(objs, saved):
        for s, m in zip(ob.material_slots, ms):
            s.material = m
    # pivot = the joint at the part origin (0, 0) -> pixel
    piv = ((0.0 - x0) * ppu, (z1 - 0.0) * ppu)
    return dict(w=rx, h=rz, pivot=piv, x0=x0, z1=z1)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--out", default=os.path.join(
        repo, "assets", "scenes", "robotfight", "mechs"))
    ap.add_argument("--ppu", type=float, default=900.0)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--only", default="", help="comma list of part names")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    cam = setup_scene(args.cpu)
    P = _principled("paint", (0.72, 0.72, 0.74), 0.45, 0.30, coat=0.4)
    F = _principled("frame", (0.13, 0.13, 0.14), 0.92, 0.38)
    G = _principled("glow", (0.03, 0.03, 0.035), 0.0, 0.08)
    idm = {"paint": _emit("id_paint", (1, 0, 0)),
           "frame": _emit("id_frame", (0, 1, 0)),
           "glow": _emit("id_glow", (0, 0, 1))}
    meta = {"yaw": YAW, "ppu": args.ppu, "parts": {}}
    pm = {}
    pm.update(build_brawler(P, F, G))
    pm.update(build_striker(P, F, G))
    only = set(filter(None, args.only.split(",")))
    for name, objs in _PARTS.items():
        if only and name not in only:
            continue
        r = render_part(cam, name, objs, args.ppu, args.out,
                        None, idm, args.samples)
        m = pm[name]
        anc = {}
        for k, p in m["anchors"].items():
            sx, sz = yaw_xz(p)
            anc[k] = ((sx - r["x0"]) * args.ppu, (r["z1"] - sz) * args.ppu)
        bx, bz = m["bone"]
        meta["parts"][name] = dict(w=r["w"], h=r["h"], pivot=r["pivot"],
                                   bone=[bx, bz], anchors=anc)
        print(f"[robotfight_mechs] {name}: {r['w']}x{r['h']}")
    if not only:
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
    print("[robotfight_mechs] done ->", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
