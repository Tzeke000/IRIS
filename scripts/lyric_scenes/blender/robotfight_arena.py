"""Robot-fight scene assets: the arena backdrop plate.

Run HEADLESS (never through the Blender MCP / GUI):

    nice -n 15 ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/blender/robotfight_arena.py -- \
        [--out assets/scenes/robotfight/arena] [--width 2496] [--cpu]

Everything BEHIND the fight floor, rendered from the scene's own camera
(the same pinhole robotfight.py uses for the floor and the mechs), with a
MARGIN x wider field of view so the 2-D camera can pan / pull back:

  structure.png   grandstands, barrier wall, roof truss, light towers, the
                  hanging jumbotron housing (neutral grey, dim, RGBA)
  crowd_0.png     the crowd standing (RGBA, composited over the structure)
  crowd_1.png     the crowd mid-JUMP, arms up (swapped in on the beat)
  anchors.json    camera constants + projected points in plate pixels:
                  seats (crowd light positions), lamps (floodlight heads),
                  the jumbotron screen quad, the LED board strips

Deterministic: every random choice comes from random.Random(constant).
World: X = right, Y = depth (away from the camera), Z = up; the fight
plane (where the mechs stand) is Y = 0, the camera sits at Y = -CAM_D.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

# ---- the camera contract shared with robotfight.py (keep in sync) ---------
CAM_D = 3.0          # camera distance to the fight plane (mech heights)
CAM_H = 0.55         # camera height = horizon height in the fight plane
VIS_H = 1.0 / 0.58   # world height visible at the fight plane (base frame)
HOR_Y = 0.60         # horizon at 60% of the frame height from the top
MARGIN = 1.30        # the plate covers MARGIN x the base field of view
ASPECT = 16.0 / 9.0

BARRIER_Y, BARRIER_H = 2.2, 0.30
LO_ROWS, LO_DY, LO_DZ = 19, 0.128, 0.079
UP_ROWS, UP_DY, UP_DZ = 22, 0.146, 0.099


def _mesh_obj(name, bm, mat):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    if isinstance(mat, (list, tuple)):
        for m in mat:
            me.materials.append(m)
    else:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


_CUBE_F = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6),
           (0, 2, 6, 4), (1, 5, 7, 3))


def _add_hex(bm, corners, mi):
    """8 corners in (x, y, z) binary order -> 6 quads. Direct vertex/face
    creation: bmesh.ops.* clear tool flags over the WHOLE mesh on every call,
    which made building ~10k crowd figures quadratic (9+ minutes)."""
    vs = [bm.verts.new(c) for c in corners]
    for f in _CUBE_F:
        bm.faces.new([vs[k] for k in f]).material_index = mi


def add_box(bm, c, s, mi=0, rot_z=0.0, rot_x=0.0):
    R = Matrix.Rotation(rot_z, 3, "Z") @ Matrix.Rotation(rot_x, 3, "X")
    cv = Vector(c)
    pts = []
    for k in range(8):
        v = Vector((((k >> 2) & 1) - 0.5, ((k >> 1) & 1) - 0.5, (k & 1) - 0.5))
        v = Vector((v.x * s[0], v.y * s[1], v.z * s[2]))
        pts.append(cv + R @ v)
    _add_hex(bm, pts, mi)


def add_beam(bm, p0, p1, t, mi=0):
    """A square-section strut from p0 to p1."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    L = d.length
    if L < 1e-6:
        return
    q = Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix()
    mid = (p0 + p1) / 2
    pts = []
    for k in range(8):
        v = Vector(((((k >> 2) & 1) - 0.5) * t, (((k >> 1) & 1) - 0.5) * t,
                    ((k & 1) - 0.5) * L))
        pts.append(mid + q @ v)
    _add_hex(bm, pts, mi)


_ICO = None


def add_ball(bm, c, r, mi=0):
    """Low-poly ball from a cached icosphere template (no bmesh.ops call)."""
    global _ICO
    if _ICO is None:
        t = bmesh.new()
        try:
            bmesh.ops.create_icosphere(t, subdivisions=1, radius=1.0)
            t.verts.index_update()
            _ICO = ([v.co.copy() for v in t.verts],
                    [[v.index for v in f.verts] for f in t.faces])
        finally:
            t.free()
    cv = Vector(c)
    vs = [bm.verts.new(cv + p * r) for p in _ICO[0]]
    for f in _ICO[1]:
        bm.faces.new([vs[k] for k in f]).material_index = mi


def principled(name, base, metal=0.0, rough=0.6, emit=None, es=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1.0)
        b.inputs["Emission Strength"].default_value = es
    return m


# ---------------------------------------------------------------------------
def build_structure(M):
    """Grandstands, barrier, roof truss, towers, jumbotron. Returns anchors."""
    anc = {}
    bm = bmesh.new()
    try:
        # floor apron behind the barrier (under the stands) + barrier wall
        add_box(bm, (0, BARRIER_Y + 0.08, BARRIER_H / 2), (40, 0.16, BARRIER_H), 1)
        add_box(bm, (0, BARRIER_Y + 0.02, BARRIER_H + 0.015), (40, 0.22, 0.03), 0)
        # LOWER TIER: stepped risers y 2.6 -> 5.0, z 0.40 -> 1.90
        for r in range(LO_ROWS):
            y = 2.55 + r * LO_DY
            z = 0.40 + r * LO_DZ
            add_box(bm, (0, y + LO_DY / 2, z / 2), (40, LO_DY, z), 2)
            add_box(bm, (0, y + 0.012, z + 0.006), (40, 0.024, 0.012), 0)  # nosing
        # balcony fascia between the tiers (carries an LED ribbon)
        add_box(bm, (0, 5.15, 2.22), (40, 0.25, 0.42), 1)
        # UPPER TIER: y 5.4 -> 8.6, z 2.45 -> 4.6
        for r in range(UP_ROWS):
            y = 5.40 + r * UP_DY
            z = 2.45 + r * UP_DZ
            add_box(bm, (0, y + UP_DY / 2, z / 2 + 1.2), (40, UP_DY, z - 2.4 + 0.1), 2)
            add_box(bm, (0, y + 0.012, z + 0.006), (40, 0.024, 0.012), 0)
        # back wall + roof
        add_box(bm, (0, 9.2, 3.5), (40, 0.3, 7.0), 1)
        # ROOF TRUSS: long chords across the arena + verticals + diagonals
        for zc, yc in ((5.2, 3.0), (5.2, 6.0), (5.9, 3.0), (5.9, 6.0)):
            add_beam(bm, (-20, yc, zc), (20, yc, zc), 0.05, 3)
        for xk in [i * 0.7 - 14.0 for i in range(41)]:
            for yc in (3.0, 6.0):
                add_beam(bm, (xk, yc, 5.2), (xk, yc, 5.9), 0.025, 3)
                add_beam(bm, (xk, yc, 5.2), (xk + 0.7, yc, 5.9), 0.02, 3)
            add_beam(bm, (xk, 3.0, 5.2), (xk, 6.0, 5.2), 0.02, 3)
        # LIGHT TOWERS (left/right): 4-post lattice + lamp-bank head
        lamps = []
        for sx in (-1, 1):
            cx, cy = sx * 2.6, 2.6
            hw = 0.11
            posts = [(cx - hw, cy - hw), (cx + hw, cy - hw), (cx + hw, cy + hw),
                     (cx - hw, cy + hw)]
            for px, py in posts:
                add_beam(bm, (px, py, 0.0), (px, py, 1.60), 0.025, 3)
            for k in range(10):
                z0, z1 = k * 0.16, (k + 1) * 0.16
                for j in range(4):
                    a, b = posts[j], posts[(j + 1) % 4]
                    add_beam(bm, (a[0], a[1], z0), (b[0], b[1], z1), 0.012, 3)
                    add_beam(bm, (a[0], a[1], z1), (b[0], b[1], z1), 0.012, 3)
            # head frame: 4 x 3 lamp housings, tilted toward the ring centre
            hz = 1.78
            add_box(bm, (cx, cy - 0.02, hz), (0.78, 0.10, 0.52), 3)
            for i in range(4):
                for j in range(3):
                    lx = cx + (i - 1.5) * 0.18
                    lz = hz + (j - 1) * 0.16
                    add_box(bm, (lx, cy - 0.10, lz), (0.15, 0.10, 0.13), 3)
                    lamps.append((lx, cy - 0.16, lz, sx))
        anc["lamps3d"] = lamps
        # JUMBOTRON: a hanging four-sided board, front face = the screen
        jx, jy, jz = 0.0, 3.6, 2.06
        jw, jh, jd = 1.50, 0.72, 1.1
        add_box(bm, (jx, jy + jd / 2, jz), (jw + 0.08, jd, jh + 0.08), 3)
        add_box(bm, (jx, jy - 0.005, jz), (jw, 0.02, jh), 4)       # screen glass
        add_box(bm, (jx, jy + jd / 2, jz - jh / 2 - 0.12), (jw * 0.9, jd * 0.9, 0.16), 1)
        for sx in (-1, 1):                                          # cables
            add_beam(bm, (sx * jw * 0.4, jy + 0.1, jz + jh / 2),
                     (sx * jw * 0.4, jy + 0.1, 5.2), 0.012, 3)
        anc["jumbo3d"] = [(jx - jw / 2, jy - 0.02, jz + jh / 2),
                          (jx + jw / 2, jy - 0.02, jz + jh / 2),
                          (jx + jw / 2, jy - 0.02, jz - jh / 2),
                          (jx - jw / 2, jy - 0.02, jz - jh / 2)]
        anc["board3d"] = [(-20, BARRIER_Y - 0.001, BARRIER_H - 0.03),
                          (20, BARRIER_Y - 0.001, BARRIER_H - 0.03),
                          (20, BARRIER_Y - 0.001, 0.05),
                          (-20, BARRIER_Y - 0.001, 0.05)]
        anc["ribbon3d"] = [(-20, 5.02, 2.36), (20, 5.02, 2.36),
                           (20, 5.02, 2.12), (-20, 5.02, 2.12)]
        _mesh_obj("structure", bm, M)
    finally:
        bm.free()
    return anc


def seat_list():
    """(x, y, z_seat) for every seat that holds a person; seeded."""
    rnd = random.Random(2026)
    seats = []
    aisles = (-6.0, -3.0, 3.0, 6.0)
    for r in range(LO_ROWS):                              # lower tier
        y = 2.55 + r * LO_DY + LO_DY * 0.55
        z = 0.40 + r * LO_DZ
        x = -9.0 + rnd.random() * 0.05
        while x < 9.0:
            if rnd.random() < 0.86 and min(abs(x - a) for a in aisles) > 0.10:
                seats.append((x + rnd.uniform(-0.012, 0.012), y, z))
            x += rnd.uniform(0.064, 0.078)
    for r in range(UP_ROWS):                              # upper tier
        y = 5.40 + r * UP_DY + UP_DY * 0.55
        z = 2.45 + r * UP_DZ
        x = -11.0 + rnd.random() * 0.05
        while x < 11.0:
            if rnd.random() < 0.82 and min(abs(x - a) for a in aisles) > 0.10:
                seats.append((x + rnd.uniform(-0.012, 0.012), y, z))
            x += rnd.uniform(0.066, 0.082)
    return seats


def build_crowd(seats, Mc, jump: bool):
    rnd = random.Random(77)
    bm = bmesh.new()
    try:
        for (x, y, z) in seats:
            hgt = rnd.uniform(0.085, 0.105)
            mi = rnd.randrange(len(Mc))
            up = rnd.uniform(0.018, 0.04) if jump else 0.0
            zb = z + up
            add_box(bm, (x, y, zb + hgt * 0.45), (0.045, 0.03, hgt * 0.9), mi)
            add_ball(bm, (x, y, zb + hgt + 0.012), 0.017, 0)
            arms_up = jump and rnd.random() < 0.7
            if arms_up:
                for sx in (-1, 1):
                    add_beam(bm, (x + sx * 0.022, y, zb + hgt * 0.8),
                             (x + sx * rnd.uniform(0.028, 0.05), y - 0.01,
                              zb + hgt + rnd.uniform(0.045, 0.07)), 0.011, mi)
        ob = _mesh_obj("crowd_1" if jump else "crowd_0", bm, Mc)
    finally:
        bm.free()
    return ob


def setup(use_cpu: bool, width: int):
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
            print("[robotfight_arena] GPU unavailable, CPU render:", e)
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"
    sc.render.resolution_x = width
    sc.render.resolution_y = int(round(width / ASPECT))
    sc.render.resolution_percentage = 100
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.025, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    # lights: the arena's own spill from the floor (front, low), a cold top
    # wash through the roof, and two warm backlights behind the upper tier
    def area(name, loc, rot, size, energy, col=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.size = size
        ld.energy = energy
        ld.color = col
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = tuple(math.radians(v) for v in rot)
        sc.collection.objects.link(lo)
    area("spill", (0, -0.5, 0.6), (-75, 0, 0), 8.0, 220)
    area("top", (0, 4.0, 7.5), (0, 0, 0), 14.0, 350)
    area("back", (0, 10.5, 6.5), (55, 0, 180), 18.0, 2600)
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 24.0
    tan_half = (VIS_H / 2.0) / CAM_D * MARGIN
    cd.lens = (cd.sensor_height / 2.0) / tan_half
    cd.clip_end = 200
    cam = bpy.data.objects.new("cam", cd)
    cam.location = (0.0, -CAM_D, CAM_H)
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.collection.objects.link(cam)
    sc.camera = cam
    # lens shift so the horizon sits at HOR_Y of the base frame; Blender's
    # shift is in units of the LARGER sensor dimension -> solve numerically
    want_v = 0.5 - (HOR_Y - 0.5) / MARGIN       # v counts from the BOTTOM
    cd.shift_y = 0.0
    for _ in range(4):
        bpy.context.view_layer.update()
        v = world_to_camera_view(sc, cam, Vector((0, 50.0, CAM_H))).y
        cd.shift_y += (v - want_v) * (1.0 / ASPECT if ASPECT > 1 else 1.0)
    bpy.context.view_layer.update()
    v = world_to_camera_view(sc, cam, Vector((0, 50.0, CAM_H))).y
    print(f"[robotfight_arena] horizon v={v:.4f} want {want_v:.4f}")
    return cam


def project(sc, cam, pts):
    Wp, Hp = sc.render.resolution_x, sc.render.resolution_y
    out = []
    for p in pts:
        c = world_to_camera_view(sc, cam, Vector(p[:3]))
        out.append([c.x * Wp, (1.0 - c.y) * Hp, c.z])
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--out", default=os.path.join(
        repo, "assets", "scenes", "robotfight", "arena"))
    ap.add_argument("--width", type=int, default=2496)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    cam = setup(args.cpu, args.width)
    sc = bpy.context.scene
    M = [principled("trim", (0.30, 0.30, 0.33), 0.7, 0.35),
         principled("wall", (0.10, 0.10, 0.11), 0.0, 0.7),
         principled("seats", (0.16, 0.16, 0.18), 0.1, 0.6),
         principled("steel", (0.35, 0.35, 0.38), 0.9, 0.35),
         principled("glass", (0.01, 0.01, 0.012), 0.0, 0.08)]
    t0 = time.time()
    anc = build_structure(M)
    print(f'[robotfight_arena] structure built {time.time()-t0:.1f}s')
    seats = seat_list()
    Mc = [principled(f"cl{k}", (v, v, v * 1.05), 0.0, 0.7)
          for k, v in enumerate((0.05, 0.09, 0.14, 0.20, 0.28))]
    c0 = build_crowd(seats, Mc, False)
    c1 = build_crowd(seats, Mc, True)
    print(f'[robotfight_arena] crowd built {time.time()-t0:.1f}s ({len(seats)} seats)')
    st = bpy.data.objects["structure"]
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    # structure (crowd hidden) ...
    for ob, vis in ((c0, False), (c1, False)):
        ob.hide_render = not vis
    sc.render.filepath = os.path.join(args.out, "structure.png")
    bpy.ops.render.render(write_still=True)
    # ... then each crowd pose alone, the structure as a HOLDOUT so the
    # crowd is correctly occluded by the fascia / barrier in front of it
    st.is_holdout = True
    for ob, other, name in ((c0, c1, "crowd_0"), (c1, c0, "crowd_1")):
        ob.hide_render, other.hide_render = False, True
        sc.render.filepath = os.path.join(args.out, f"{name}.png")
        bpy.ops.render.render(write_still=True)
    st.is_holdout = False
    out = dict(cam=dict(D=CAM_D, h=CAM_H, vis_h=VIS_H, hor_y=HOR_Y,
                        margin=MARGIN, aspect=ASPECT, barrier_y=BARRIER_Y,
                        barrier_h=BARRIER_H),
               size=[sc.render.resolution_x, sc.render.resolution_y],
               seats=project(sc, cam, seats),
               lamps=[pp + [l[3]] for pp, l in
                      zip(project(sc, cam, anc["lamps3d"]), anc["lamps3d"])],
               jumbo=project(sc, cam, anc["jumbo3d"]),
               board=project(sc, cam, anc["board3d"]),
               ribbon=project(sc, cam, anc["ribbon3d"]),
               horizon=project(sc, cam, [(0, 200.0, CAM_H)])[0])
    with open(os.path.join(args.out, "anchors.json"), "w") as f:
        json.dump(out, f)
    print(f"[robotfight_arena] {len(seats)} seats, {len(anc['lamps3d'])} lamps"
          f" -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
