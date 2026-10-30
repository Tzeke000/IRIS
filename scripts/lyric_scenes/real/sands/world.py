"""Sands — the desert (bpy): dune terrain mesh + sand material, Multiple-Scattering sky with an aligned sun lamp
(slowly setting over the song), sun disc, saguaro scatter, the walking camel. Heights come from terrain.height().
"""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

from lib import bpyx as X
from lib import importer

import terrain as TR

ASSETS = "/data/iris_assets/sands"
SUN_ROT_DEG = 248.0                       # sun toward −X/−Y (west-south-west): dune slip faces (+X side) in shadow
SUN_EL_START, SUN_EL_END = 4.2, 0.9       # degrees over the song — golden hour sliding toward sunset
SAND_ALBEDO = (0.56, 0.33, 0.155)         # linear; ≈ sRGB (196,156,110) — desert sand albedo ~0.35–0.45
CAMEL_SCALE = 2.2 / 60.79                 # model is 60.79 units tall (measured) → 2.2 m dromedary
CAMEL_SPEED = 0.80 * 30 * CAMEL_SCALE     # stance-foot travel 0.8 units/frame (measured) → m/s at 30 fps
CAMEL_NOSE = (0.0, -1.0, 0.0)             # measured: head toward −Y in the imported orientation (side renders)


def sun_dir(el_deg, rot_deg=SUN_ROT_DEG):
    el, rot = math.radians(el_deg), math.radians(rot_deg)
    return np.array([math.cos(el) * math.sin(rot), math.cos(el) * math.cos(rot), math.sin(el)])


# ------------------------------------------------------------------------------------------------ terrain

def terrain(coll, R):
    V, quads, tris, radii = TR.polar_grid(r_max=20000.0, segs=768)
    me = bpy.data.meshes.new("Dunes")
    faces = np.concatenate([quads.ravel(), tris.ravel()]).astype(np.int32)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(len(faces))
    me.loops.foreach_set("vertex_index", faces)
    npoly = len(quads) + len(tris)
    starts = np.r_[np.arange(len(quads)) * 4, len(quads) * 4 + np.arange(len(tris)) * 3].astype(np.int32)
    me.polygons.add(npoly)
    me.polygons.foreach_set("loop_start", starts)
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(npoly, bool))
    me.validate()
    ob = bpy.data.objects.new("Dunes", me)
    coll.objects.link(ob)
    me.materials.append(sand_material())
    R.log(f"[terrain] {len(V)} verts, {npoly} faces, r_max {radii[-1]:.0f} m, ring0 {radii[1]:.2f} m")
    return ob


def _tex(nb, path, vec, color=True):
    im = X.img(path, color)
    return nb.tex(im, vec)


def sand_material():
    m, nb, out = X.new_mat("Sand")
    geo = nb.geo()
    pos = geo.outputs["Position"]
    # scanned micro detail (CC0 Poly Haven sand_03) on a 1.6 m tile, world XY projection
    uv = nb.v("SCALE", pos, 1 / 1.6)
    d = os.path.join(ASSETS, "ph_sand_03")
    diff = _tex(nb, f"{d}/sand_03_diff_2k.jpg", uv)
    rough = _tex(nb, f"{d}/sand_03_rough_2k.jpg", uv, False)
    nrm = _tex(nb, f"{d}/sand_03_nor_gl_2k.jpg", uv, False)
    # macro colour: two noise scales mix the albedo between paler and redder sand
    n1 = nb.n("ShaderNodeTexNoise")
    nb.set(n1.inputs["Scale"], 1 / 180.0)
    nb.set(n1.inputs["Detail"], 3.0)
    nb.link(pos, n1.inputs["Vector"])
    n2 = nb.n("ShaderNodeTexNoise")
    nb.set(n2.inputs["Scale"], 1 / 22.0)
    nb.link(pos, n2.inputs["Vector"])
    pale = (0.64, 0.42, 0.22)
    red = (0.50, 0.27, 0.12)
    macro = nb.mix(nb.mr(n1.outputs["Fac"], 0.35, 0.65), red, pale)
    macro = nb.mix(nb.mr(n2.outputs["Fac"], 0.3, 0.7, 0, 0.25), macro, (0.58, 0.36, 0.18))
    lum = nb.n("ShaderNodeRGBToBW")
    nb.link(diff.outputs["Color"], lum.inputs[0])
    detail = nb.mr(lum.outputs[0], 0.15, 0.6, 0.82, 1.12, clamp=False)
    base = nb.mix(1.0, macro, nb.xyz(detail, detail, detail), blend="MULTIPLY")
    # slope from the shading normal (smooth): slip faces a touch darker + smoother (fresh avalanche sand)
    nz = nb.sep(geo.outputs["Normal"])[2]
    flat = nb.mr(nz, 0.86, 0.97)                              # 1 on gentle slopes, 0 on slip faces
    base = nb.mix(flat, nb.mix(1.0, base, (0.88, 0.86, 0.84), blend="MULTIPLY"), base)
    # wind ripples (~0.4 m, crests along Y — wind from −X), only on gentle slopes and near the camera
    wave = nb.n("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X", wave_profile="SIN")
    nb.link(pos, wave.inputs["Vector"])
    nb.set(wave.inputs["Scale"], 0.80)
    nb.set(wave.inputs["Distortion"], 7.0)
    nb.set(wave.inputs["Detail"], 2.0)
    nb.set(wave.inputs["Detail Scale"], 1.2)
    cam = nb.n("ShaderNodeCameraData")
    near = nb.mr(cam.outputs["View Distance"], 60.0, 160.0, 1.0, 0.0)
    rip = nb.m("MULTIPLY", wave.outputs["Fac"], nb.m("MULTIPLY", flat, near))
    nmap = nb.n("ShaderNodeNormalMap")
    nb.link(nrm.outputs["Color"], nmap.inputs["Color"])
    nb.set(nmap.inputs["Strength"], 0.8)
    bump = nb.n("ShaderNodeBump")
    nb.set(bump.inputs["Strength"], 0.35)
    nb.set(bump.inputs["Distance"], 0.03)
    nb.link(rip, bump.inputs["Height"])
    nb.link(nmap.outputs["Normal"], bump.inputs["Normal"])
    r = nb.mr(rough.outputs["Color"], 0, 1, 0.78, 0.97)
    b = X.principled(nb, base, r, 0.0)
    nb.link(bump.outputs["Normal"], b.inputs["Normal"])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


# ------------------------------------------------------------------------------------------------ sky + sun

def sky_and_sun(sc, n, coll, R):
    w = bpy.data.worlds.new("SandsSky")
    sc.world = w
    nt = w.node_tree
    nt.nodes.clear()
    nb = X.NB(nt)
    sky = nb.n("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.altitude = 300.0
    sky.air_density, sky.aerosol_density, sky.ozone_density = 1.0, 1.3, 1.0     # desert dust → warmer, hazier
    sky.sun_rotation = math.radians(SUN_ROT_DEG)
    sky.sun_disc = True
    bg = nb.n("ShaderNodeBackground")
    nb.set(bg.inputs["Strength"], 1.0)
    nb.link(sky.outputs["Color"], bg.inputs["Color"])
    wo = nb.n("ShaderNodeOutputWorld")
    nb.link(bg.outputs[0], wo.inputs["Surface"])
    el = np.linspace(SUN_EL_START, SUN_EL_END, n)
    X.bake(nt, 'nodes["%s"].sun_elevation' % sky.name, np.radians(el), thin=False)
    # sun lamp aligned to the sky's sun (blender-terrain-and-cityscapes §2 formula), warming as it drops
    sun = X.light("Sun", "SUN", (0, 0, 100), energy=4.0, color=(1.0, 0.72, 0.45), size=0.0)
    sun.data.angle = math.radians(0.6)
    sun.rotation_mode = "XYZ"
    rots, cols, ens = [], [], []
    for e in el:
        d = Vector(sun_dir(e))
        rots.append(tuple((-d).to_track_quat("-Z", "Y").to_euler()))
        k = np.clip((e - 1.5) / 8.0, 0, 1)
        cols.append((1.0, 0.56 + 0.20 * k, 0.30 + 0.22 * k))
        ens.append(2.2 + 2.6 * k)
    X.bake_vec(sun, "rotation_euler", np.array(rots))
    X.bake_vec(sun.data, "color", np.array(cols))
    X.bake(sun.data, "energy", np.array(ens))
    # sun disc (EEVEE draws no sky disc): an emissive disc 15 km out along the sun direction
    disc = X.uv_sphere("SunDisc", r=1.0, seg=32, rings=16, coll=coll, mat=X.emit_mat("SunDiscMat", (1.0, 0.62, 0.32), 60.0))
    disc.visible_shadow = False
    D = 15000.0
    rad = D * math.tan(math.radians(0.27)) * 1.6
    disc.scale = (rad, rad, rad)
    X.bake_vec(disc, "location", np.array([sun_dir(e) * D for e in el]))
    R.log(f"[sky] sun {SUN_EL_START}°→{SUN_EL_END}° az-rot {SUN_ROT_DEG}°, disc r {rad:.0f} m @ {D / 1000:.0f} km")
    return sun


# ------------------------------------------------------------------------------------------------ cactuses

CACTI = ["saguaro28", "saguaro09", "saguaro07", "saguaro27", "saguaro17"]


def cactus_kit(kit_coll):
    """each model joined into one mesh, origin at its base centre, 1 m tall (instances scale it)"""
    meshes = []
    for slug in CACTI:
        objs = importer.import_model(f"{ASSETS}/{slug}/model.glb", kit_coll)
        me = importer.bake_world([o for o in objs if o.type == "MESH"], f"Cactus_{slug}")
        importer.normalize(me, height=1.0, ground=True, center=True)
        for o in objs:
            bpy.data.objects.remove(o, do_unlink=True)
        meshes.append(me)
    return meshes


def cactus_positions(rng, n, keep_out):
    """dart throwing: interdune flats (gentle slope, low ground), density thinning with distance"""
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 400:
        tries += 1
        r = 34.0 + 2200.0 * rng.random() ** 1.5
        a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        if any(math.hypot(x - kx, y - ky) < kr for kx, ky, kr in keep_out):
            continue
        if float(TR.slope(x, y)) > 0.22:
            continue
        h = float(TR.height(x, y))
        hl = float(np.mean(TR.height(x + np.array([-40, 40, 0, 0]), y + np.array([0, 0, -40, 40]))))
        if h > hl + 2.0 and rng.random() < 0.85:          # mostly in the interdune lows
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (6 + 0.02 * r) ** 2 for px, py, _ in pts):
            continue
        pts.append((x, y, h))
    return pts


def cacti(coll, kit_coll, R, keep_out, extra=()):
    meshes = cactus_kit(kit_coll)
    rng = np.random.default_rng(11)
    pts = cactus_positions(rng, 170, keep_out) + [(x, y, float(TR.height(x, y))) for x, y in extra]
    obs = []
    for k, (x, y, h) in enumerate(pts):
        me = meshes[k % len(meshes)] if k >= len(pts) - len(extra) else meshes[int(rng.integers(len(meshes)))]
        ob = bpy.data.objects.new(f"Cactus_{k:03d}", me)
        coll.objects.link(ob)
        hgt = rng.uniform(4.5, 10.5)
        # sink to the LOWEST ground under the base (slopes) + 0.15 m
        lo = float(np.min(TR.height(x + np.array([-0.6, 0.6, 0, 0, 0]), y + np.array([0, 0, -0.6, 0.6, 0]))))
        ob.location = (x, y, lo - 0.15)
        ob.rotation_euler = (rng.normal(0, 0.03), rng.normal(0, 0.03), rng.uniform(0, 2 * math.pi))
        ob.scale = (hgt * rng.uniform(0.9, 1.1), hgt * rng.uniform(0.9, 1.1), hgt)
        obs.append(ob)
    R.log(f"[cacti] {len(obs)} saguaros ({len(extra)} hand-placed), heights 4.5–10.5 m")
    return obs


# ------------------------------------------------------------------------------------------------ camel

def camel(coll, R, path_xy, f0, f1, n):
    """walking camel: the CC-BY walk cycle looped (Cycles modifier), root keyed along path_xy (2 points) at the
    measured stride speed, nose (−Y local) along the velocity, feet on terrain.height. Hidden outside [f0, f1]."""
    objs = importer.import_model(f"{ASSETS}/camel_walk/model.glb", coll)
    widgets = [o for o in objs if o.type == "MESH" and o.name.startswith("Icosphere")]
    objs = [o for o in objs if o not in widgets]
    for o in widgets:
        bpy.data.objects.remove(o, do_unlink=True)                     # bone-display widget, not the camel
    roots = [o for o in objs if o.parent is None]
    rig = X.empty("CamelRig", coll=coll)
    for o in roots:
        o.parent = rig
    rig.scale = (CAMEL_SCALE,) * 3
    arm = [o for o in objs if o.type == "ARMATURE"][0]
    act = arm.animation_data.action
    for ly in act.layers:
        for st in ly.strips:
            for cb in st.channelbags:
                for fc in cb.fcurves:
                    if not any(m.type == "CYCLES" for m in fc.modifiers):
                        fc.modifiers.new("CYCLES")
    # feet offset (model units → m): lowest point −33.79 units under the root (probe)
    foot = 33.79 * CAMEL_SCALE
    a, b = np.asarray(path_xy[0], float), np.asarray(path_xy[1], float)
    d = (b - a) / np.linalg.norm(b - a)
    head = math.atan2(d[1], d[0]) + math.pi / 2                       # local −Y → d
    fr = np.arange(1, n + 1)
    s = np.clip((fr - f0), 0, None) / 30.0 * CAMEL_SPEED
    P = a[None] + d[None] * s[:, None]
    Z = TR.height(P[:, 0], P[:, 1]) + foot - 0.06
    loc = np.c_[P, Z]
    X.bake_vec(rig, "location", loc)
    # gentle pitch to the local ground slope along the walk
    g = (TR.height(P[:, 0] + d[0], P[:, 1] + d[1]) - TR.height(P[:, 0] - d[0], P[:, 1] - d[1])) / 2.0
    pitch = np.arctan(g)
    rig.rotation_mode = "XYZ"
    rot = np.c_[-pitch, np.zeros(n), np.full(n, head)]               # Rx(p)·(0,−1,0) dips the nose for p>0 → −pitch
    X.bake_vec(rig, "rotation_euler", rot)
    vis = (fr >= f0) & (fr <= f1)
    for o in objs:
        if o.type == "MESH":
            X.show_mask(o, vis)
    # facing check: nose (rig local −Y) · velocity
    nose = np.tile([math.sin(head), -math.cos(head), 0.0], (n, 1))   # Rz(head)·(0,−1,0)
    R.log(f"[camel] path {a.round(1)}→{(a + d * s[f1 - 1]).round(1)}, {CAMEL_SPEED:.2f} m/s, frames {f0}–{f1}, "
          f"walked {s[f1 - 1] - s[f0 - 1]:.1f} m")
    return rig, objs, loc, nose, d
