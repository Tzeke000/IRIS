"""Car sprites for the `carstreet` scene (scripts/lyric_scenes/carstreet.py).

A stylised low sports car (no real make, no badge) modelled from code and
rendered from the scene's chase camera as RGBA sprites, one set per camera
height, with SEPARATE lighting passes so the 2-D scene can relight it per
frame (a streetlight sweeping over the roof, neon rims from the storefronts)
and separate masks (paint / glass / tail-light) so each part can react to
its own cause in the music.

Headless + deterministic (fixed geometry, fixed Cycles seed):

    ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/blender/carstreet_car.py -- \
        --out assets/scenes/carstreet

Writes  car_h<cm>_<pass>.png  (8-bit RGBA, straight alpha) + car_meta.json.
Passes: base (night ambience + faint rear fill), top0/top1/top2 (one
overhead light ahead / above / behind the car), sideL / sideR (a soft light
from each storefront side), ids (R = paint, G = glass, B = tail-light,
flat), wheels (tyres only, lit by base + top1).

The camera matches carstreet.py exactly: reference frame 1920x1080, focal
f = 0.75 W px, horizon at 0.40 H (vertical lens shift, no pitch), camera on
the car's centre-line at the given height, the car's REAR face Z_REAR metres
in front of it. Car: forward = +Y, centre at the origin, ground z = 0.
"""
import argparse
import json
import math
import os
import sys

import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

W_REF, H_REF = 1920, 1080
H_CANVAS = 1560                      # taller render canvas: same f + horizon
                                     # row, so a low sprite is never cut off
F_REF = 0.75 * W_REF                 # focal length in px
HY_FRAC = 0.40                       # horizon height (fraction of H)
Z_REAR = 4.2                         # camera -> rear face (m)
HEIGHTS = (1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95)  # camera heights (m)
L_HALF = 2.25                        # half length (m)
REAR_WHEEL_Y, FRONT_WHEEL_Y = -1.35, 1.38
TRACK = 0.80                         # wheel centre |x|
TYRE_R, TYRE_W = 0.34, 0.31


# ---------------------------------------------------------------- geometry
def smooth(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def half_width(t):
    w = 0.90 + 0.075 * math.exp(-((t - REAR_WHEEL_Y) / 0.55) ** 2) \
        + 0.035 * math.exp(-((t - FRONT_WHEEL_Y) / 0.45) ** 2)
    w *= 1.0 - 0.10 * smooth(1.75, 2.25, t)        # nose taper
    w *= 1.0 - 0.07 * smooth(-1.95, -2.25, t)      # tail taper
    return w


def deck_z(t):
    """Height of the shoulder / deck line along the car."""
    rear = 0.93 - 0.04 * smooth(-1.7, -2.25, t)
    front = 0.80 - 0.10 * smooth(1.2, 2.25, t)
    k = smooth(-0.6, 0.9, t)
    return rear * (1 - k) + front * k


def body_ring(t):
    w, zs, zb = half_width(t), deck_z(t), 0.15
    right = [(0.0, zb), (0.72 * w, zb), (0.95 * w, zb + 0.07),
             (1.00 * w, zb + 0.28), (0.99 * w, zs - 0.20),
             (0.94 * w, zs - 0.045), (0.80 * w, zs), (0.42 * w, zs + 0.035),
             (0.0, zs + 0.045)]
    left = [(-x, z) for (x, z) in reversed(right[1:-1])]
    return right + left


def loft(bm, rings, ys, cap_start=True, cap_end=True):
    """rings: list of [(x, z)], one per y. Returns the verts grid."""
    grid = []
    for ring, y in zip(rings, ys):
        grid.append([bm.verts.new((x, y, z)) for (x, z) in ring])
    n = len(rings[0])
    for a, b in zip(grid[:-1], grid[1:]):
        for k in range(n):
            k2 = (k + 1) % n
            bm.faces.new((a[k], a[k2], b[k2], b[k]))
    if cap_start:
        bm.faces.new(list(reversed(grid[0])))
    if cap_end:
        bm.faces.new(grid[-1])
    return grid


def new_obj(name, bm, mats, subsurf=0, smooth_shade=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    if smooth_shade:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if subsurf:
        mod = ob.modifiers.new("sub", "SUBSURF")
        mod.levels = subsurf
        mod.render_levels = subsurf
    return ob


def box(name, mat, x0, x1, y0, y1, z0, z1, bevel=0.0):
    bm = bmesh.new()
    try:
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x = x0 + (v.co.x + 0.5) * (x1 - x0)
            v.co.y = y0 + (v.co.y + 0.5) * (y1 - y0)
            v.co.z = z0 + (v.co.z + 0.5) * (z1 - z0)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if bevel > 0:
        b = ob.modifiers.new("bev", "BEVEL")
        b.width = bevel
        b.segments = 3
    return ob


def cylinder_x(name, mat, cx, cy, cz, r, w, segs=48):
    """Cylinder with its axis along X."""
    bm = bmesh.new()
    try:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r,
                              radius2=r, depth=w)
        for v in bm.verts:
            x, y, z = v.co
            v.co = Vector((z + cx, y + cy, -x + cz))
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.materials.append(mat)
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def cylinder_y(name, mat, cx, cy, cz, r, d, segs=32):
    """Cylinder with its axis along Y."""
    bm = bmesh.new()
    try:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r,
                              radius2=r, depth=d)
        for v in bm.verts:
            x, y, z = v.co
            v.co = Vector((x + cx, z + cy, y + cz))
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.materials.append(mat)
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


# ---------------------------------------------------------------- materials
def principled(name, col, metallic=0.0, rough=0.5, coat=0.0):
    m = bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):
        m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*col, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if coat > 0 and "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.03
    return m


def emissive(name, col, strength):
    m = bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):
        m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*col, 1.0)
    e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs["Emission"], o.inputs["Surface"])
    return m


def set_emission(m, col, strength):
    for n in m.node_tree.nodes:
        if n.type == "EMISSION":
            n.inputs["Color"].default_value = (*col, 1.0)
            n.inputs["Strength"].default_value = strength


# ---------------------------------------------------------------- the car
def build_car():
    M = dict(
        paint=principled("paint", (0.20, 0.21, 0.23), 0.55, 0.28, coat=1.0),
        glass=principled("glass", (0.004, 0.004, 0.005), 0.0, 0.04),
        trim=principled("trim", (0.012, 0.012, 0.013), 0.0, 0.55),
        chrome=principled("chrome", (0.85, 0.85, 0.88), 1.0, 0.18),
        plate=principled("plate", (0.55, 0.56, 0.58), 0.0, 0.4),
        ink=principled("ink", (0.01, 0.01, 0.01), 0.0, 0.5),
        tyre=principled("tyre", (0.018, 0.018, 0.02), 0.0, 0.62),
        rim=principled("rim", (0.30, 0.30, 0.32), 1.0, 0.3),
        tail=emissive("tail", (1.0, 0.03, 0.03), 3.0),
    )
    parts = {"body": [], "wheels": [], "tail": [], "glass": [], "paint": []}
    # BODY: a loft of cross-sections, flat-ish rear face via inset rings
    ys = [-L_HALF + 0.0] + [(-L_HALF + 4.5 * k / 22) for k in range(1, 23)]
    ys = sorted(set(round(y, 4) for y in ys))
    rings = [body_ring(y) for y in ys]
    # rear face: two inset rings just behind the tail -> crisp edge + flat face
    r0 = rings[0]
    cz = sum(z for _, z in r0) / len(r0)
    inset1 = [(x * 0.965, cz + (z - cz) * 0.95) for x, z in r0]
    inset2 = [(x * 0.55, cz + (z - cz) * 0.45) for x, z in r0]
    rings = [inset2, inset1] + rings
    ys = [-L_HALF - 0.045, -L_HALF - 0.035] + ys
    # nose: same treatment
    rn = rings[-1]
    czn = sum(z for _, z in rn) / len(rn)
    rings += [[(x * 0.9, czn + (z - czn) * 0.85) for x, z in rn]]
    ys += [L_HALF + 0.05]
    bm = bmesh.new()
    loft(bm, rings, ys)
    body = new_obj("body", bm, [M["paint"]], subsurf=2)
    parts["body"].append(body)
    parts["paint"].append(body)
    # GREENHOUSE: its own loft sitting into the deck; roof = paint, rest glass
    gys = [-1.05 + 1.95 * k / 16 for k in range(17)]
    grings = []
    for t in gys:
        zb = deck_z(t) - 0.03
        h = smooth(-1.05, -0.25, t) * (1 - smooth(0.25, 0.90, t))
        Hh = max(0.004, h * (1.19 - zb))
        gw = 0.80 - 0.10 * smooth(0.3, 0.9, t)
        right = [(0.0, zb), (gw, zb), (0.90 * gw, zb + 0.55 * Hh),
                 (0.66 * gw, zb + 0.93 * Hh), (0.0, zb + Hh)]
        left = [(-x, z) for (x, z) in reversed(right[1:-1])]
        grings.append(right + left)
    bm = bmesh.new()
    loft(bm, grings, gys)
    bm.normal_update()
    for f in bm.faces:
        c = f.calc_center_median()
        f.material_index = 0 if (f.normal.z > 0.80 and abs(c.x) < 0.42) else 1
    gh = new_obj("greenhouse", bm, [M["paint"], M["glass"]], subsurf=2)
    parts["body"].append(gh)
    # SPOILER: a wing on two stalks
    zw = 1.10
    wing = box("wing", M["paint"], -0.86, 0.86, -2.20, -1.92, zw, zw + 0.035,
               bevel=0.012)
    parts["body"].append(wing)
    for sx in (-0.48, 0.48):
        parts["body"].append(box(f"stalk{sx}", M["trim"], sx - 0.025,
                                 sx + 0.025, -2.05, -1.95, deck_z(-2.0) - 0.05,
                                 zw + 0.01))
    for sx in (-1, 1):                                  # wing end plates
        parts["body"].append(box(f"plate{sx}", M["trim"], sx * 0.86 - 0.012,
                                 sx * 0.86 + 0.012, -2.22, -1.90, zw - 0.07,
                                 zw + 0.08))
    # TAIL-LIGHT: a full-width LED bar + C-shaped ends
    zt, yt = 0.80, -L_HALF - 0.05
    tl = [box("tailbar", M["tail"], -0.80, 0.80, yt - 0.012, yt + 0.03,
              zt - 0.012, zt + 0.012)]
    for sx in (-1, 1):
        x_out = sx * 0.83
        tl.append(box(f"tailC{sx}_v", M["tail"], x_out - 0.016, x_out + 0.016,
                      yt - 0.012, yt + 0.04, zt - 0.15, zt + 0.02))
        tl.append(box(f"tailC{sx}_b", M["tail"], min(x_out, sx * 0.68),
                      max(x_out, sx * 0.68), yt - 0.012, yt + 0.04,
                      zt - 0.15, zt - 0.126))
    tl.append(box("fog", M["tail"], -0.05, 0.05, yt - 0.02, yt + 0.03, 0.27,
                  0.33))
    parts["tail"] += tl
    parts["body"] += tl
    # trim band under the tail-light (dark) and the plate recess
    parts["body"].append(box("band", M["trim"], -0.86, 0.86, yt, yt + 0.06,
                             0.60, 0.63))
    plate = box("plate", M["plate"], -0.27, 0.27, yt - 0.006, yt + 0.03, 0.42,
                0.55, bevel=0.008)
    parts["body"].append(plate)
    txt = bpy.data.curves.new("platetxt", "FONT")
    txt.body = "TZEKE000"
    txt.size = 0.095
    txt.align_x = "CENTER"
    txt.align_y = "CENTER"
    txt.extrude = 0.002
    tob = bpy.data.objects.new("platetxt", txt)
    tob.location = (0.0, yt - 0.012, 0.485)
    tob.rotation_euler = (math.radians(90), 0, 0)
    tob.data.materials.append(M["ink"])
    bpy.context.scene.collection.objects.link(tob)
    parts["body"].append(tob)
    # DIFFUSER + fins + exhaust tips
    parts["body"].append(box("diffuser", M["trim"], -0.84, 0.84, yt - 0.03,
                             -1.85, 0.11, 0.40, bevel=0.01))
    for sx in (-1, 1):                       # corner vents
        parts["body"].append(box(f"vent{sx}", M["trim"], min(sx * 0.62, sx * 0.93),
                                 max(sx * 0.62, sx * 0.93), yt - 0.01, -1.9,
                                 0.40, 0.52, bevel=0.01))
    for k in range(7):
        fx = -0.6 + 1.2 * k / 6
        parts["body"].append(box(f"fin{k}", M["chrome"], fx - 0.006, fx + 0.006,
                                 yt - 0.06, -1.9, 0.09, 0.36))
    for sx in (-0.30, -0.17, 0.17, 0.30):
        parts["body"].append(cylinder_y(f"exh{sx}", M["chrome"], sx, yt - 0.02,
                                        0.24, 0.048, 0.18))
        parts["body"].append(cylinder_y(f"exhin{sx}", M["ink"], sx, yt - 0.035,
                                        0.24, 0.036, 0.18))
    # side mirrors
    for sx in (-1, 1):
        parts["body"].append(box(f"mirror{sx}", M["paint"], sx * 0.86 - 0.09,
                                 sx * 0.86 + 0.09, 0.45, 0.60, 0.92, 1.0,
                                 bevel=0.02))
    # WHEELS
    for wy in (REAR_WHEEL_Y, FRONT_WHEEL_Y):
        for sx in (-1, 1):
            parts["wheels"].append(cylinder_x(f"tyre{sx}{wy}", M["tyre"],
                                              sx * TRACK, wy, TYRE_R, TYRE_R,
                                              TYRE_W))
            parts["wheels"].append(cylinder_x(f"rim{sx}{wy}", M["rim"],
                                              sx * (TRACK + 0.02), wy, TYRE_R,
                                              0.235, TYRE_W * 0.92))
    return M, parts


# ---------------------------------------------------------------- lights
def area(name, loc, size, power, rot=(0, 0, 0), size_y=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    if size_y is None:
        ld.shape = "SQUARE"
        ld.size = size
    else:
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = size, size_y
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    ob.rotation_euler = rot
    bpy.context.scene.collection.objects.link(ob)
    return ob


def aim(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def setup_world(strength, col=(0.02, 0.03, 0.06)):
    w = bpy.context.scene.world or bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    if bpy.app.version < (5, 0, 0):
        w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (*col, 1.0)
    bg.inputs["Strength"].default_value = strength


# ---------------------------------------------------------------- render
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--heights", default=",".join(str(h) for h in HEIGHTS))
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    heights = [float(h) for h in args.heights.split(",")]

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.seed = 0
    sc.cycles.use_animated_seed = False
    sc.cycles.use_denoising = True
    sc.render.threads_mode = "FIXED"
    sc.render.threads = args.threads
    sc.render.resolution_x, sc.render.resolution_y = W_REF, H_CANVAS
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"

    M, parts = build_car()
    cam_d = bpy.data.cameras.new("cam")
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.sensor_width = 36.0
    cam_d.lens = 36.0 * F_REF / W_REF
    cam_d.shift_y = -(0.5 * H_CANVAS - HY_FRAC * H_REF) / W_REF
    cam_d.clip_start = 0.05
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.rotation_euler = (math.radians(90), 0, 0)

    # light rigs (objects toggled per pass)
    rigs = {}
    soft = []
    for sx in (-1, 1):            # long city-glow strips high above the road
        o = area(f"soft{sx}", (sx * 3.5, 0.0, 6.0), 1.2, 260, size_y=16)
        aim(o, (sx * 0.4, 0.0, 0.8))
        soft.append(o)
    fill = area("fill", (0.0, -9.0, 2.6), 3.0, 260)
    aim(fill, (0, -1.5, 0.6))
    rigs["base"] = soft + [fill]
    for k, ly in enumerate((4.5, 0.0, -5.0)):     # overhead lamp sweep
        o = area(f"top{k}", (0.0, ly, 6.5), 1.5, 2600)
        aim(o, (0.0, ly * 0.55, 0.6))
        rigs[f"top{k}"] = [o]
    for nm, sx in (("sideL", -1), ("sideR", 1)):
        o = area(nm, (sx * 6.0, -2.5, 2.2), 5.0, 1500)
        aim(o, (0.0, -0.8, 0.8))
        rigs[nm] = [o]
    all_lights = [o for v in rigs.values() for o in v]

    tl_strength = 3.0
    meta = dict(W=W_REF, H=H_REF, f=F_REF, hy=HY_FRAC * H_REF, z_rear=Z_REAR,
                half_len=L_HALF, track=TRACK, tyre_r=TYRE_R,
                rear_wheel_y=REAR_WHEEL_Y, tail_z=0.80, exhaust_z=0.24,
                exhaust_x=[-0.30, -0.17, 0.17, 0.30], half_width=0.975,
                heights={})
    car_objs = parts["body"] + parts["wheels"]

    def show(objs_on):
        on = set(o.name for o in objs_on)
        for o in car_objs:
            o.hide_render = o.name not in on

    def lights(names):
        on = set(o.name for n in names for o in rigs[n])
        for o in all_lights:
            o.hide_render = o.name not in on

    for hc in heights:
        cam.location = (0.0, -L_HALF - Z_REAR, hc)
        bpy.context.view_layer.update()
        # crop box: project the car's bounds (+ a margin for the hop)
        xs, ys_ = [], []
        for o in car_objs:
            for c in o.bound_box:
                p = o.matrix_world @ Vector(c)
                v = world_to_camera_view(sc, cam, p)
                xs.append(v.x)
                ys_.append(v.y)
        px0 = max(0, int(math.floor(min(xs) * W_REF)) - 12)
        px1 = min(W_REF, int(math.ceil(max(xs) * W_REF)) + 12)
        py0 = max(0, int(math.floor(min(ys_) * H_CANVAS)) - 12)  # from bottom
        py1 = min(H_CANVAS, int(math.ceil(max(ys_) * H_CANVAS)) + 12)
        sc.render.use_border = True
        sc.render.use_crop_to_border = True
        sc.render.border_min_x, sc.render.border_max_x = px0 / W_REF, px1 / W_REF
        sc.render.border_min_y, sc.render.border_max_y = py0 / H_CANVAS, py1 / H_CANVAS
        tag = f"h{int(round(hc * 100)):03d}"
        meta["heights"][tag] = dict(hc=hc, x0=px0, y0=H_CANVAS - py1,
                                    w=px1 - px0, h=py1 - py0)

        def render(name, samples):
            sc.cycles.samples = samples
            sc.render.filepath = os.path.join(out, f"car_{tag}_{name}.png")
            bpy.ops.render.render(write_still=True)
            print(f"[carstreet_car] {tag} {name}", flush=True)

        # body passes
        show(parts["body"])
        set_emission(M["tail"], (1.0, 0.03, 0.03), tl_strength)
        setup_world(1.0)
        lights(["base"])
        render("base", args.samples)
        set_emission(M["tail"], (1.0, 0.03, 0.03), 0.0)   # light passes: no
        setup_world(0.0)                                  # self-emission
        for nm in ("top0", "top1", "top2", "sideL", "sideR"):
            lights([nm])
            render(nm, args.samples)
        # ID masks: flat emission per material class, everything else black
        saved = {}
        flat = {k: emissive(f"id_{k}", c, 1.0) for k, c in
                (("paint", (1, 0, 0)), ("glass", (0, 1, 0)), ("tail", (0, 0, 1)),
                 ("none", (0, 0, 0)))}
        for o in parts["body"]:
            saved[o.name] = [s.material for s in o.material_slots]
            for s_i, s in enumerate(o.material_slots):
                mn = s.material.name if s.material else ""
                key = ("paint" if mn == "paint" else "glass" if mn == "glass"
                       else "tail" if mn == "tail" else "none")
                o.material_slots[s_i].material = flat[key]
        lights([])
        sc.cycles.use_denoising = False
        render("ids", 8)
        sc.cycles.use_denoising = True
        for o in parts["body"]:
            for s_i, m in enumerate(saved[o.name]):
                o.material_slots[s_i].material = m
        # wheels (tyres + rims), lit by the base rig + the overhead lamp
        show(parts["wheels"])
        setup_world(1.0)
        lights(["base", "top1"])
        render("wheels", max(16, args.samples // 2))
        # where the rear tyres touch the ground, in reference px
        for sx in (-1, 1):
            p = Vector((sx * TRACK, REAR_WHEEL_Y, 0.0))
            v = world_to_camera_view(sc, cam, p)
            meta["heights"][tag][f"contact{'L' if sx < 0 else 'R'}"] = (
                v.x * W_REF, (1 - v.y) * H_CANVAS)
    with open(os.path.join(out, "car_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("[carstreet_car] done", json.dumps(meta["heights"]), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
