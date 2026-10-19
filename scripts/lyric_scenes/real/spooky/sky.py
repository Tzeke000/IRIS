"""Sky: painted gradient world + stars (shader), the moon (emissive sphere, procedural craters + bright crater rays,
halo card), painterly cloud cards (emission, alpha-shaped by noise, slow drift) + layered cloud BANKS that cross the
moon (wide strips whose shapes scroll at three speeds), analytic shooting stars fired on the song's big hits (GN on
Scene Time, constant point count). Music: stars ↔ CTRL.stars (hi-hats), moon + halo ↔ CTRL.moon_glow (bass)."""
import math

import numpy as np

import common as C
from common import NB, X


def world():
    import bpy
    w = bpy.data.worlds.new("SpookySky")
    bpy.context.scene.world = w
    nt = w.node_tree
    nt.nodes.clear()
    nb = NB(nt)
    out = nb.n("ShaderNodeOutputWorld")
    K = C.ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    D = nb.v("NORMALIZE", tc.outputs["Generated"])
    _, _, dz = nb.sep(D)
    md = C.dir_to(C.HERO_CAM, C.MOON_POS)
    # vertical gradient: horizon glow → violet → deep indigo
    el = nb.m("MAXIMUM", dz, -0.2)
    grad = C.ramp(nb, nb.mr(el, -0.05, 0.75), [(0.0, C.SKY_HORIZON), (0.10, (0.012, 0.014, 0.08)),
                                               (0.32, C.SKY_MID), (1.0, C.SKY_ZENITH)])
    # glow around the moon (painted bloom in the sky itself). v2: TIGHT — measured v1's cos^18 + cos^4 at ≈ 0.2 over the
    # whole frame (the lavender veil). Now: a pink core just outside the limb, a blue-violet skirt within ~30°.
    cosm = nb.v("DOT_PRODUCT", D, tuple(md))
    cr = math.cos(math.radians(10.6))
    core = nb.m("POWER", nb.mr(cosm, math.cos(math.radians(17.0)), cr), 2.2)              # 0 at 17° → 1 at the limb
    skirt = nb.m("POWER", nb.mr(cosm, math.cos(math.radians(38.0)), cr), 2.0)
    glow = nb.mix(nb.m("MULTIPLY", core, K["moon_glow"]), grad, (0.16, 0.05, 0.20), blend="ADD")
    glow = nb.mix(skirt, glow, C.SKY_MOONSIDE, blend="ADD")
    # stars: Voronoi cells on the view direction, sparse, twinkling on CTRL.T, faded near the horizon + the moon
    vo = nb.n("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1")
    nb.set(vo.inputs["Scale"], 260.0)
    nb.set(vo.inputs["Randomness"], 1.0)
    nb.link(D, vo.inputs["Vector"])
    rnd_r, rnd_g, _ = nb.sep(vo.outputs["Color"])
    core = nb.mr(vo.outputs["Distance"], 0.10, 0.0, 0.0, 1.0)                 # 1 at the cell point
    core = nb.m("POWER", core, 3.0)
    keep = nb.mr(rnd_r, 0.60, 0.70, 0.0, 1.0)                                  # ~25 % of cells carry a star
    tw = nb.m("ADD", 0.65, nb.m("MULTIPLY", 0.35, nb.m("SINE", nb.m("ADD", nb.m("MULTIPLY", K["T"], nb.m("ADD", 1.5, nb.m("MULTIPLY", rnd_g, 3.0))), nb.m("MULTIPLY", rnd_g, 40.0)))))
    fade = nb.m("MULTIPLY", nb.mr(dz, 0.04, 0.22), nb.mr(cosm, 0.93, 0.985, 1.0, 0.0))
    star = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", core, keep), nb.m("MULTIPLY", tw, fade)),
                nb.m("MULTIPLY", K["stars"], 4.0))
    col = nb.mix(star, glow, (1.0, 0.86, 1.0), blend="ADD")
    bg = nb.n("ShaderNodeBackground")
    nb.link(col, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    # lighting from the world is dimmer than what the camera sees (painted sky ≠ an HDRI): Is Camera Ray split
    bgl = nb.n("ShaderNodeBackground")
    nb.link(col, bgl.inputs["Color"])
    bgl.inputs["Strength"].default_value = 0.55
    lp = nb.n("ShaderNodeLightPath")
    mx = nb.n("ShaderNodeMixShader")
    nb.link(lp.outputs["Is Camera Ray"], mx.inputs[0])
    nb.link(bgl.outputs[0], mx.inputs[1])
    nb.link(bg.outputs[0], mx.inputs[2])
    nb.link(mx.outputs[0], out.inputs["Surface"])
    return w


# bright ray craters: (disc x, disc y in −1..1 as seen from the camera, ray reach rad, ray count scale, strength)
RAY_CRATERS = [(-0.26, 0.42, 0.50, 9.0, 0.38), (0.50, -0.22, 0.30, 7.0, 0.22)]
MOON_ROT = (20.0, -12.0, 35.0)


def _ray_frame(dx, dy):
    """object-space unit vector of the moon point seen at disc coords (dx, dy) from the camera + a tangent frame"""
    from mathutils import Euler
    f, r, u = C.cam_basis()
    to_cam = -C.dir_to(C.CAM_LOC, C.MOON_POS)
    # disc axes as seen by the camera: right ≈ camera right, up ≈ camera up (projected ⟂ to the view line)
    rr = r - to_cam * (r @ to_cam)
    rr /= np.linalg.norm(rr)
    uu = np.cross(to_cam, rr)
    zz = math.sqrt(max(0.0, 1 - dx * dx - dy * dy))
    w = to_cam * zz + rr * dx + uu * dy
    R = np.array(Euler(tuple(math.radians(a) for a in MOON_ROT)).to_matrix())
    c = R.T @ w
    e1 = np.cross(c, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    return tuple(c), tuple(e1), tuple(e2)


def moon(coll):
    import bpy
    m, nb, out = X.new_mat("Moon")
    K = C.ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    P = nb.v("SCALE", tc.outputs["Object"], 1.0 / C.MOON_R)          # unit sphere coords
    # maria: soft lilac patches
    mar = C.noise(nb, P, scale=1.2, detail=6.0, rough=0.6, distortion=0.3)
    base = C.ramp(nb, mar.outputs["Fac"], [(0.40, C.MOON_PINK), (0.50, (0.85, 0.45, 0.95)), (0.58, C.MOON_LILAC),
                                           (0.70, (0.36, 0.22, 0.78))])
    # craters at two scales as a HEIGHT field (shallow bowl + faint rim, cells warped by noise so they are not
    # circles) → Bump → shaded by a fixed painted light from the upper-left-front. v2: rims ×0.4, bump 0.55 → 0.32,
    # bowls darken less — v1's read as soap bubbles.
    warp = C.noise(nb, nb.v("SCALE", P, 2.0), scale=1.0, detail=2.0)
    Pw = nb.v("ADD", P, nb.v("SCALE", nb.v("SUBTRACT", warp.outputs["Color"], (0.5, 0.5, 0.5)), 0.09))

    def crater_h(scale, depth, keep=0.4):
        vo = nb.n("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1")
        nb.set(vo.inputs["Scale"], scale)
        nb.link(Pw, vo.inputs["Vector"])
        r, g, _ = nb.sep(vo.outputs["Color"])
        rad = nb.m("ADD", 0.14, nb.m("MULTIPLY", r, 0.24))
        d = nb.m("DIVIDE", vo.outputs["Distance"], rad)
        has = nb.mr(g, 1.0 - keep - 0.05, 1.0 - keep + 0.05)
        bowl = nb.m("POWER", nb.mr(d, 1.0, 0.0), 2.2)
        rim = nb.m("MULTIPLY", nb.mr(d, 0.75, 1.0), nb.mr(d, 1.35, 1.0))
        return nb.m("MULTIPLY", nb.m("SUBTRACT", nb.m("MULTIPLY", rim, 0.12), bowl), nb.m("MULTIPLY", has, depth)), \
            nb.m("MULTIPLY", bowl, has)
    h1, b1 = crater_h(3.0, 1.0, 0.45)
    h2, b2 = crater_h(7.5, 0.5, 0.3)
    mott = C.noise(nb, P, scale=16.0, detail=4.0, rough=0.62)               # fine mottling (the cover's busy surface)
    H = nb.m("ADD", nb.m("ADD", h1, nb.m("MULTIPLY", h2, 0.7)), nb.m("MULTIPLY", mott.outputs["Fac"], 0.25))
    bump = nb.n("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.32
    bump.inputs["Distance"].default_value = 0.02 * C.MOON_R
    nb.link(H, bump.inputs["Height"])
    Ld = tuple(C.dir_to((0, 0, 0), (-0.45, -1.0, 0.55)))
    lam = nb.m("MAXIMUM", nb.v("DOT_PRODUCT", bump.outputs["Normal"], Ld), 0.0)
    ref = nb.m("MAXIMUM", nb.v("DOT_PRODUCT", nb.geo().outputs["Normal"], Ld), 0.05)
    shade = nb.m("MINIMUM", nb.m("DIVIDE", lam, ref), 1.5)               # crater relief relative to the smooth sphere
    shade = nb.m("ADD", 0.35, nb.m("MULTIPLY", shade, 0.65))
    col = nb.mix(nb.m("ADD", nb.m("MULTIPLY", b1, 0.18), nb.m("MULTIPLY", b2, 0.10)), base, (0.50, 0.28, 0.85))
    col = nb.mix(nb.mr(mott.outputs["Fac"], 0.35, 0.7, 0.0, 0.10), col, (1.0, 0.62, 0.92))
    col = nb.mix(1.0, col, shade, blend="MULTIPLY")
    # bright crater RAYS (Tycho-style starbursts): streaks around fixed centres on the camera-facing side, ray pattern
    # = 2-D noise on (cos az, sin az) (seamless around the centre), fading with angular distance. All clamped (no exp).
    Pn = nb.v("NORMALIZE", P)
    rays = None
    for (cx_, cy_, reach, k, amp) in RAY_CRATERS:
        c, e1, e2 = _ray_frame(cx_, cy_)
        cosd = nb.m("MINIMUM", nb.m("MAXIMUM", nb.v("DOT_PRODUCT", Pn, c), -1.0), 1.0)
        dist = nb.m("ARCCOSINE", cosd)                                         # radians on the unit sphere
        az = nb.m("ARCTAN2", nb.v("DOT_PRODUCT", Pn, e2), nb.v("DOT_PRODUCT", Pn, e1))
        rn = C.noise(nb, nb.xyz(nb.m("MULTIPLY", nb.m("COSINE", az), k), nb.m("MULTIPLY", nb.m("SINE", az), k), cx_ * 7.0),
                     scale=1.0, detail=1.0, rough=0.5)
        brk = C.noise(nb, nb.v("SCALE", Pn, 9.0), scale=1.0, detail=2.0)          # break the streaks up along their length
        streak = nb.m("MULTIPLY", nb.m("POWER", nb.mr(rn.outputs["Fac"], 0.57, 0.80), 1.6), nb.mr(brk.outputs["Fac"], 0.35, 0.6))
        fall = nb.m("MULTIPLY", nb.mr(dist, 0.03, reach, 1.0, 0.0), nb.mr(dist, 0.0, 0.035, 0.3, 1.0))
        core = nb.mr(dist, 0.025, 0.0, 0.0, 1.0)
        r1 = nb.m("MULTIPLY", nb.m("ADD", nb.m("MULTIPLY", streak, fall), nb.m("MULTIPLY", core, 0.8)), amp)
        rays = r1 if rays is None else nb.m("ADD", rays, r1)
    col = nb.mix(nb.m("MINIMUM", rays, 1.0), col, (1.0, 0.86, 1.0), blend="ADD")
    # limb glow (the cover's bright outline) + overall strength on CTRL.moon_glow
    lim = nb.m("MULTIPLY", nb.m("POWER", C.facing(nb, 0.35), 3.0), 0.6)
    col = nb.mix(lim, col, (1.0, 0.72, 1.0), blend="ADD")
    st = nb.m("MULTIPLY", K["moon_glow"], 0.8)
    nb.link(C.emission(nb, col, st), out.inputs["Surface"])
    mo = X.uv_sphere("Moon", r=C.MOON_R, seg=96, rings=48, mat=m, coll=coll)
    mo.location = tuple(C.MOON_POS)
    mo.rotation_euler = tuple(C.deg(a) for a in MOON_ROT)
    C.hide_shadow(mo)
    # halo card behind the moon (additive emission, radial falloff)
    hm, hb, ho = X.new_mat("MoonHalo")
    K2 = C.ctrl(hb)
    tc2 = hb.n("ShaderNodeTexCoord")
    uv = hb.v("SUBTRACT", tc2.outputs["UV"], (0.5, 0.5, 0.0))
    r = hb.m("MULTIPLY", hb.v("LENGTH", uv), 2.0)                      # 0 centre → 1 edge
    fall = hb.m("POWER", hb.m("MAXIMUM", hb.m("SUBTRACT", 1.0, r), 0.0), 3.2)
    st2 = hb.m("MULTIPLY", hb.m("MULTIPLY", fall, K2["moon_glow"]), hb.m("MULTIPLY", K2["glow"], 0.32))
    hb.link(C.add_shader(hb, C.transparent(hb), C.emission(hb, (0.85, 0.40, 1.0), st2)), ho.inputs["Surface"])
    C.blended(hm)
    d = C.dir_to(C.HERO_CAM, C.MOON_POS)
    hp = C.MOON_POS + d * C.MOON_R * 1.6
    halo = X.mesh_obj("MoonHalo", [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [(0, 1, 2, 3)], coll=coll, mat=hm,
                      uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
    halo.location = tuple(hp)
    halo.scale = (C.MOON_R * 2.3,) * 3
    halo.rotation_euler = C.billboard_rot(hp)
    C.hide_shadow(halo)
    return mo


def cloud_material():
    m, nb, out = X.new_mat("CloudCard")
    K = C.ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    u, v, _ = nb.sep(tc.outputs["UV"])
    oi = nb.n("ShaderNodeObjectInfo")
    seed = nb.m("MULTIPLY", oi.outputs["Random"], 97.0)
    lit = nb.n("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="lit").outputs["Fac"]   # 0 dark .. 1 moon-lit
    side = nb.n("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="side").outputs["Fac"]  # moon side: -1 L .. 1 R
    # slow drift of the billows on CTRL.T
    drift = nb.m("MULTIPLY", K["T"], 0.0016)                               # slow drift of the billows (~0.4 card / song)
    p = nb.xyz(nb.m("ADD", nb.m("MULTIPLY", u, 3.2), drift), nb.m("MULTIPLY", v, 1.6), seed)
    n1 = C.noise(nb, p, scale=1.6, detail=5.0, rough=0.58)
    # flat-ish bottom, rounded top: ellipse mask, bottom edge harder
    du = nb.m("MULTIPLY", nb.m("SUBTRACT", u, 0.5), 2.0)
    dv = nb.m("MULTIPLY", nb.m("SUBTRACT", v, 0.42), 2.1)
    ell = nb.m("SUBTRACT", 1.0, nb.m("ADD", nb.m("MULTIPLY", du, du), nb.m("MULTIPLY", dv, dv)))
    dens = nb.m("ADD", nb.m("MULTIPLY", ell, 0.75), nb.m("MULTIPLY", nb.m("SUBTRACT", n1.outputs["Fac"], 0.5), 1.9))
    alpha = nb.mr(dens, 0.05, 0.22)
    # inner shade: top/edges toward the moon lit pink, belly dark violet; edge rim brighter (painterly silver lining)
    inner = nb.mr(dens, 0.1, 0.7)
    toward = nb.m("ADD", nb.m("MULTIPLY", nb.m("SUBTRACT", u, 0.5), side), nb.m("MULTIPLY", nb.m("SUBTRACT", v, 0.3), 0.9))
    shade = nb.m("ADD", nb.m("MULTIPLY", toward, 0.8), nb.m("MULTIPLY", nb.m("SUBTRACT", n1.outputs["Fac"], 0.5), 0.9))
    shade = nb.m("ADD", nb.m("MULTIPLY", shade, lit), nb.m("MULTIPLY", nb.m("SUBTRACT", 1.0, inner), nb.m("MULTIPLY", lit, 0.6)))
    col = C.ramp(nb, nb.mr(shade, -0.45, 0.75), [(0.0, (0.004, 0.005, 0.026)), (0.38, (0.020, 0.018, 0.115)),
                                                 (0.66, (0.13, 0.06, 0.33)), (0.86, (0.80, 0.30, 0.70)),
                                                 (1.0, (1.0, 0.70, 0.95))])
    st = nb.m("MULTIPLY", nb.m("ADD", 0.55, nb.m("MULTIPLY", K["sky_glow"], 0.45)), 0.62)
    sh = C.mix_shader(nb, alpha, C.transparent(nb), C.emission(nb, col, st))
    nb.link(sh, out.inputs["Surface"])
    C.blended(m)
    return m


# (bearing° right of the moon line from the hero cam, elevation°, distance, width, height, lit, z-order note)
CLOUDS = [   # moon: bearing 0, elevation 23°, radius 13° from the hero camera — banks clip its lower-left + right edges only
    # dark bank across the moon's lower-left edge (in front of it)
    (-17, 12.5, 2300, 1100, 300, 0.25), (-9, 9.5, 2100, 1000, 260, 0.35), (-24, 18.5, 2200, 900, 260, 0.30),
    # pink lit banks on the moon's lower-right edge and right
    (15, 10.5, 2400, 950, 280, 0.95), (22, 17.0, 2600, 800, 260, 0.85), (28, 24.5, 2300, 800, 260, 0.80),
    (8, 5.5, 2200, 1300, 260, 0.85),
    # left sky
    (-32, 33.0, 1900, 1200, 380, 0.45), (-24, 42.0, 2200, 1300, 360, 0.40), (-42, 21.0, 1700, 1000, 320, 0.35),
    (-48, 31.0, 1800, 900, 300, 0.30),
    # top
    (-8, 49.0, 2200, 1400, 380, 0.40), (20, 46.0, 2400, 1300, 360, 0.55), (38, 37.0, 2300, 1000, 320, 0.5),
    # low band over the mountains (horizon glow)
    (-35, 5.0, 1500, 1200, 190, 0.55), (-14, 4.0, 1600, 1300, 170, 0.6), (40, 6.5, 1500, 1100, 210, 0.6),
    (33, 11.0, 1800, 1000, 230, 0.8),
]


def clouds(coll):
    m = cloud_material()
    md = C.dir_to(C.HERO_CAM, C.MOON_POS)
    az0 = math.atan2(md[0], md[1])                         # bearing of the moon (from +Y toward +X)
    obs = []
    for i, (b, el, dist, w, h, lit) in enumerate(CLOUDS):
        az = az0 + math.radians(b)
        d = np.array([math.sin(az) * math.cos(math.radians(el)), math.cos(az) * math.cos(math.radians(el)),
                      math.sin(math.radians(el))])
        p = C.HERO_CAM + d * dist
        ob = X.mesh_obj(f"Cloud_{i:02d}", [(-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0)], [(0, 1, 2, 3)],
                        coll=coll, mat=m, uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
        ob.location = tuple(p - np.array([0, 0, h * 0.42]))
        ob.scale = (w, h, 1.0)
        ob.rotation_euler = C.billboard_rot(p)
        ob["lit"] = float(lit) * (1.0 if abs(b) < 30 and el < 26 else 0.55)    # v2: far-from-moon clouds stay dark
        ob["side"] = float(np.clip(-b / 15.0, -1, 1))       # clouds right of the moon are lit on their left
        C.hide_shadow(ob)
        obs.append(ob)
    return obs


# ---------------------------------------------------------------- cloud BANKS crossing the moon (v2)
# (distance m, elevation of the band centre deg, band height deg, drift deg/s (+ = toward +X on screen), feature m,
#  coverage 0..1, lit 0..1)
BANKS = [(1750.0, 13.0, 7.5, 0.060, 230.0, 0.85, 0.55),       # near, dark, thick: across the moon's lower part
         (2050.0, 18.5, 5.0, 0.040, 200.0, 0.60, 0.80),       # mid, pink-lit edges, sparse: crosses the centre
         (2350.0, 25.5, 3.6, 0.026, 170.0, 0.30, 0.95)]       # far, thin wisps over the upper moon


def bank_material():
    """wide strip; density = vertical band profile × noise scrolling sideways on CTRL.T. Thick cores dark violet,
    thin edges lit pink — strongest where the strip passes in front of / beside the moon (from world position)."""
    m, nb, out = X.new_mat("CloudBank")
    K = C.ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    u, v, _ = nb.sep(tc.outputs["UV"])
    A = {k: nb.n("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name=k).outputs["Fac"]
         for k in ("bw", "bh", "spd", "feat", "cover", "lit", "seed")}
    xm = nb.m("SUBTRACT", nb.m("MULTIPLY", nb.m("SUBTRACT", u, 0.5), A["bw"]), nb.m("MULTIPLY", K["T"], A["spd"]))
    ym = nb.m("MULTIPLY", nb.m("SUBTRACT", v, 0.5), A["bh"])
    q = nb.xyz(nb.m("DIVIDE", xm, A["feat"]), nb.m("DIVIDE", ym, nb.m("MULTIPLY", A["feat"], 0.55)), A["seed"])
    n1 = C.noise(nb, q, scale=1.0, detail=6.0, rough=0.6, distortion=0.25)
    n2 = C.noise(nb, nb.v("SCALE", q, 0.31), scale=1.0, detail=2.0)
    prof = nb.m("SUBTRACT", 1.0, nb.m("POWER", nb.m("ABSOLUTE", nb.m("MULTIPLY", nb.m("SUBTRACT", v, 0.5), 2.0)), 1.6))
    dens = nb.m("ADD", nb.m("MULTIPLY", prof, nb.m("ADD", 0.25, nb.m("MULTIPLY", A["cover"], 0.6))),
                nb.m("ADD", nb.m("MULTIPLY", nb.m("SUBTRACT", n1.outputs["Fac"], 0.5), 1.7),
                     nb.m("MULTIPLY", nb.m("SUBTRACT", n2.outputs["Fac"], 0.5), 1.4)))
    dens = nb.m("SUBTRACT", dens, nb.m("SUBTRACT", 0.62, nb.m("MULTIPLY", A["cover"], 0.35)))
    alpha = nb.m("MULTIPLY", nb.mr(dens, 0.0, 0.16), nb.m("MULTIPLY", nb.mr(u, 0.0, 0.08), nb.mr(u, 1.0, 0.92)))
    inner = nb.mr(dens, 0.05, 0.5)
    # moon proximity from the world position (any frame, any drift)
    P = nb.geo().outputs["Position"]
    dvec = nb.v("NORMALIZE", nb.v("SUBTRACT", P, tuple(C.CAM_LOC)))
    cm = nb.v("DOT_PRODUCT", dvec, tuple(C.dir_to(C.CAM_LOC, C.MOON_POS)))
    rad = math.cos(math.radians(10.6))
    near = nb.mr(cm, rad - 0.035, rad + 0.01)                   # on / just around the disc
    halo = nb.mr(cm, math.cos(math.radians(32.0)), rad)          # within ~30° of the moon
    edge = nb.m("SUBTRACT", 1.0, inner)
    litk = nb.m("MULTIPLY", A["lit"], nb.m("ADD", nb.m("MULTIPLY", halo, 0.55), nb.m("MULTIPLY", near, 0.6)))
    shade = nb.m("ADD", nb.m("MULTIPLY", edge, litk), nb.m("MULTIPLY", nb.m("SUBTRACT", n1.outputs["Fac"], 0.5), 0.3))
    col = C.ramp(nb, nb.mr(shade, -0.15, 0.95), [(0.0, (0.004, 0.005, 0.024)), (0.35, (0.020, 0.018, 0.11)),
                                                 (0.62, (0.15, 0.07, 0.36)), (0.84, (0.85, 0.34, 0.74)),
                                                 (1.0, (1.0, 0.72, 0.96))])
    st = nb.m("MULTIPLY", nb.m("ADD", 0.55, nb.m("MULTIPLY", K["sky_glow"], 0.45)), 0.9)
    sh = C.mix_shader(nb, nb.m("MINIMUM", alpha, 0.97), C.transparent(nb), C.emission(nb, col, st))
    nb.link(sh, out.inputs["Surface"])
    C.blended(m)
    return m


def cloud_banks(coll):
    m = bank_material()
    md = C.dir_to(C.CAM_LOC, C.MOON_POS)
    az0 = math.atan2(md[0], md[1])
    obs = []
    for i, (dist, el, hdeg, spd, feat, cover, lit) in enumerate(BANKS):
        e = math.radians(el)
        d = np.array([math.sin(az0) * math.cos(e), math.cos(az0) * math.cos(e), math.sin(e)])
        p = C.CAM_LOC + d * dist
        bw = 2 * dist * math.tan(math.radians(52.0))
        bh = 2 * dist * math.tan(math.radians(hdeg / 2))
        ob = X.mesh_obj(f"CloudBank_{i}", [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)], [(0, 1, 2, 3)],
                        coll=coll, mat=m, uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
        ob.location = tuple(p)
        ob.scale = (bw, bh, 1.0)
        ob.rotation_euler = C.billboard_rot(p, C.CAM_LOC)
        for k, val in (("bw", bw), ("bh", bh), ("spd", -dist * math.tan(math.radians(spd))), ("feat", feat),
                       ("cover", cover), ("lit", lit), ("seed", 13.0 + 7.0 * i)):
            ob[k] = float(val)
        C.hide_shadow(ob)
        obs.append(ob)
    return obs


def shooting_stars(coll, hits):
    """one streak per big hit (two on the biggest), fired AT the hit (attribute tf = seconds) and gone ~1 s later.
    Constant count (scale 0 before/after), the time used for positions is clamped (gotcha 10)."""
    import bpy
    rng = np.random.default_rng(31)
    md = C.dir_to(C.HERO_CAM, C.MOON_POS)
    az0 = math.atan2(md[0], md[1])
    fire = []
    for k, t in enumerate(hits):
        fire.append(t)
        if k in (3, 4):                                              # 2:43 and 3:12 — the biggest drop and its peak
            fire.append(t + 0.8)
    starts, dirs, tfs = [], [], []
    for i, t in enumerate(fire):
        az = az0 + math.radians(rng.uniform(-48, 28) if i % 2 else rng.uniform(-48, -18))   # mostly the left sky
        el = math.radians(rng.uniform(26, 36))
        d = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
        starts.append(C.HERO_CAM + d * 1400)              # in front of every cloud card (1500 m +)
        right = np.cross(d, [0, 0, 1])
        right /= np.linalg.norm(right)
        mv = right * rng.choice([1.0, 0.8]) - np.array([0, 0, 0.5])           # down-right, the cover's direction
        dirs.append(mv / np.linalg.norm(mv))
        tfs.append(float(t))
    pts = X.points_obj("ShootingStars", np.array(starts), attrs={"mv": np.array(dirs), "tf": np.array(tfs)}, coll=coll)
    pts["fire_times"] = [float(t) for t in tfs]
    # template: a crossed pair of quads along +X, 'tz' 0 tail → 1 head
    tm = X.new_mat("ShootingStar")
    m, nb_, out = tm
    K = C.ctrl(nb_)
    tz = nb_.sattr("tz").outputs["Fac"]
    a = nb_.m("POWER", tz, 2.0)
    wid = nb_.sattr("tw").outputs["Fac"]                                      # 0 at the strip edge, 1 centre
    a = nb_.m("MULTIPLY", a, nb_.m("POWER", wid, 1.5))
    st = nb_.m("MULTIPLY", nb_.m("MULTIPLY", a, K["sky_glow"]), 14.0)
    nb_.link(C.add_shader(nb_, C.transparent(nb_), C.emission(nb_, (1.0, 0.75, 1.0), st)), out.inputs["Surface"])
    C.blended(m)

    ng, nb, gi, go = X.gn_tree("ShootingStarsGN")
    T = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    tfA, mvA = nb.attr("tf"), nb.attr("mv", "FLOAT_VECTOR")
    age = nb.m("SUBTRACT", T, tfA)
    life = 1.1
    act = nb.m("MULTIPLY", nb.cmp("GREATER_EQUAL", age, 0.0), nb.cmp("LESS_THAN", age, life))
    ag = nb.m("MINIMUM", nb.m("MAXIMUM", age, 0.0), life)                       # clamped (gotcha 10)
    k = nb.m("DIVIDE", ag, life)
    env = nb.m("MULTIPLY", nb.m("POWER", nb.m("SINE", nb.m("MULTIPLY", k, math.pi)), 0.7), act)   # grow then shrink
    speed = 300.0
    pos = nb.v("ADD", nb.n("GeometryNodeInputPosition").outputs[0], nb.v("SCALE", mvA, nb.m("MULTIPLY", ag, speed)))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(gi.outputs[0], sp.inputs["Geometry"])
    nb.link(pos, sp.inputs["Position"])
    # template mesh: two quads crossing along X (x from -1 tail to 0 head)
    grid = nb.n("GeometryNodeMeshGrid")
    nb.set(grid.inputs["Size X"], 1.0)
    nb.set(grid.inputs["Size Y"], 1.0)
    nb.set(grid.inputs["Vertices X"], 8)
    nb.set(grid.inputs["Vertices Y"], 3)
    px, py, _ = nb.sep(nb.n("GeometryNodeInputPosition").outputs[0])
    g1 = nb.store(grid.outputs["Mesh"], "tz", nb.m("ADD", px, 0.5))
    g1 = nb.store(g1, "tw", nb.m("SUBTRACT", 1.0, nb.m("ABSOLUTE", nb.m("MULTIPLY", py, 2.0))))
    tf = nb.n("GeometryNodeTransform")
    nb.link(g1, tf.inputs["Geometry"])
    nb.set(tf.inputs["Translation"], (-0.5, 0, 0))
    tf2 = nb.n("GeometryNodeTransform")
    nb.link(tf.outputs[0], tf2.inputs["Geometry"])
    nb.set(tf2.inputs["Rotation"], (math.pi / 2, 0, 0))
    jg = nb.n("GeometryNodeJoinGeometry")
    nb.link(tf2.outputs[0], jg.inputs[0])
    nb.link(tf.outputs[0], jg.inputs[0])
    smt = nb.n("GeometryNodeSetMaterial")
    nb.link(jg.outputs[0], smt.inputs["Geometry"])
    smt.inputs["Material"].default_value = m
    al = nb.n("FunctionNodeAlignRotationToVector", axis="X")
    nb.link(mvA, al.inputs["Vector"])
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(sp.outputs[0], inst.inputs["Points"])
    nb.link(smt.outputs[0], inst.inputs["Instance"])
    nb.link(al.outputs[0], inst.inputs["Rotation"])
    L = nb.m("MULTIPLY", env, 190.0)
    W = nb.m("MULTIPLY", env, 3.5)
    nb.link(nb.xyz(L, W, W), inst.inputs["Scale"])
    nb.link(inst.outputs[0], go.inputs[0])
    X.add_gn(pts, ng)
    C.hide_shadow(pts)
    return pts
