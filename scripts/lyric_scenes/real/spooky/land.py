"""Land + water: glowing stylised water (SSR gloss + an analytic moon glitter path), the foreground rubble island, the
castle cliff island + waterfall, painted mountain ridge layers (aerial perspective baked into the shader), mist cards,
and ONE bounded haze box (gotcha 3: never a world volume)."""
import math

import numpy as np

import common as C
from common import NB, X


def water(coll):
    m, nb, out = X.new_mat("Water")
    K = C.ctrl(nb)
    geo = nb.geo()
    P = geo.outputs["Position"]
    px, py, _ = nb.sep(P)
    # horizontal anime ripples: noise stretched along X, two scrolling layers (4D, W = T)
    q1 = nb.xyz(nb.m("MULTIPLY", px, 0.18), nb.m("MULTIPLY", py, 1.1), 0.0)
    n1 = C.noise(nb, q1, scale=1.0, detail=3.0, rough=0.6, dims="4D", w=nb.m("MULTIPLY", K["T"], 0.35))
    q2 = nb.xyz(nb.m("MULTIPLY", px, 0.7), nb.m("MULTIPLY", py, 3.6), 7.0)
    n2 = C.noise(nb, q2, scale=1.0, detail=2.0, rough=0.5, dims="4D", w=nb.m("MULTIPLY", K["T"], 0.9))
    h = nb.m("ADD", n1.outputs["Fac"], nb.m("MULTIPLY", n2.outputs["Fac"], 0.5))
    bump = nb.n("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.08
    nb.link(h, bump.inputs["Height"])
    N = bump.outputs["Normal"]
    bsdf = X.principled(nb, base=C.WATER_DEEP, rough=0.10)
    bsdf.inputs["Specular IOR Level"].default_value = 0.14
    nb.link(N, bsdf.inputs["Normal"])
    bsdf.inputs["IOR"].default_value = 1.33
    # glitter path: reflected view ray vs the direction to the moon, per shading point (works for every camera)
    I = geo.outputs["Incoming"]
    R = nb.v("REFLECT", nb.v("SCALE", I, -1.0), N)
    M = nb.v("NORMALIZE", nb.v("SUBTRACT", tuple(C.MOON_POS), P))
    c = nb.m("MAXIMUM", nb.v("DOT_PRODUCT", R, M), 0.0)
    # stylised glitter path (painterly licence, the cover's stripe): compare only the HORIZONTAL bearing of the
    # wave-perturbed reflection with the bearing to the moon → a glittering band from the horizon down, any camera
    rx_, ry_, _ = nb.sep(R)
    mx_, my_, _ = nb.sep(M)
    hb = nb.m("MAXIMUM", nb.v("DOT_PRODUCT", nb.v("NORMALIZE", nb.xyz(rx_, ry_, 0.0)), nb.v("NORMALIZE", nb.xyz(mx_, my_, 0.0))), 0.0)
    _, _, iz = nb.sep(I)
    horizon = nb.mr(iz, 0.35, 0.02, 0.35, 1.0)                     # brighter toward the horizon
    band = nb.m("ADD", nb.m("MULTIPLY", nb.m("POWER", hb, 300.0), 2.6), nb.m("MULTIPLY", nb.m("POWER", hb, 25.0), 0.45))
    path = nb.m("ADD", nb.m("MULTIPLY", band, horizon), nb.m("MULTIPLY", nb.m("POWER", c, 140.0), 6.0))
    sp = C.noise(nb, nb.xyz(nb.m("MULTIPLY", px, 2.5), nb.m("MULTIPLY", py, 9.0), 0.0), scale=1.0, detail=1.0,
                 dims="4D", w=nb.m("MULTIPLY", K["T"], 2.2))
    spark = nb.mr(sp.outputs["Fac"], 0.5, 0.7, 0.05, 2.2)
    glit = nb.m("MULTIPLY", nb.m("MULTIPLY", path, spark), nb.m("MULTIPLY", K["water_shimmer"], K["glow"]))
    # faint violet self-glow, stronger toward the horizon (the cover's glowing water)
    fres = nb.m("POWER", C.facing(nb, 0.2), 3.0)
    base_glow = nb.m("ADD", 0.03, nb.m("MULTIPLY", fres, 0.18))
    ecol = nb.mix(nb.m("MINIMUM", glit, 1.0), C.HAZE, (1.0, 0.55, 0.92))
    est = nb.m("ADD", nb.m("MULTIPLY", base_glow, K["glow"]), nb.m("MULTIPLY", glit, 2.5))
    nb.link(ecol, bsdf.inputs["Emission Color"])
    nb.link(est, bsdf.inputs["Emission Strength"])
    nb.link(bsdf.outputs[0], out.inputs["Surface"])
    w = X.mesh_obj("Water", [(-4000, -800, 0), (4000, -800, 0), (4000, 4500, 0), (-4000, 4500, 0)], [(0, 1, 2, 3)],
                   coll=coll, mat=m)
    return w


def _grid(nx, ny, x0, x1, y0, y1, hfn):
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    gx, gy = np.meshgrid(xs, ys)
    gz = hfn(gx, gy)
    V = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    F = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i) for j in range(ny - 1) for i in range(nx - 1)]
    return V, F


def _fbm(x, y, seed=0, oct=4):
    rng = np.random.default_rng(seed)
    v = np.zeros_like(x)
    a, f = 1.0, 1.0
    for _ in range(oct):
        ph = rng.uniform(0, 6.28, 4)
        kx, ky = rng.normal(0, 1, 2)
        v += a * (np.sin(f * (0.11 * x + 0.03 * kx * y) + ph[0]) * np.cos(f * (0.13 * y + 0.03 * ky * x) + ph[1]))
        a *= 0.5
        f *= 2.1
    return v


def stone_mat(name="Stone", tex_from=None):
    """Stylised dark violet stone: real moon/fill lights + a soft moon-side rim; optional imported albedo → detail."""
    m, nb, out = X.new_mat(name)
    K = C.ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    nz = C.noise(nb, tc.outputs["Object"], scale=0.9, detail=6.0, rough=0.6)
    det = nb.mr(nz.outputs["Fac"], 0.3, 0.7)
    col = nb.mix(det, C.STONE_DARK, (0.07, 0.04, 0.13))
    b = X.principled(nb, base=col, rough=0.85)
    bump = nb.n("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    nb.link(nz.outputs["Fac"], bump.inputs["Height"])
    nb.link(bump.outputs["Normal"], b.inputs["Normal"])
    # violet rim (cover: every ruin edge glows faintly purple)
    rim = nb.m("MULTIPLY", nb.m("POWER", C.facing(nb, 0.3), 3.0), 0.35)
    nb.link(C.add_shader(nb, b.outputs[0], C.emission(nb, (0.5, 0.25, 0.9), rim)), out.inputs["Surface"])
    return m


def foreground_ground(coll):
    def hfn(x, y):
        r = np.hypot((x + 9) / 17.0, (y + 3) / 11.0)
        base = 1.2 * np.clip(1.25 - r, -0.6, 1.0) - 0.15
        return base + 0.25 * _fbm(x * 3, y * 3, 3)
    V, F = _grid(60, 44, -30, 6, -16, 9, hfn)
    ob = X.mesh_obj("Ground_FG", V, F, coll=coll, mat=stone_mat("GroundStone"), smooth=True)
    return ob


def cliff(coll):
    """castle island: noisy tapered cylinder with a plateau, displaced; foot below the water."""
    import bpy
    cx, cy, _ = C.CASTLE_POS
    H = C.CASTLE_CLIFF_H
    nt, nz = 96, 34
    V, F = [], []
    for j in range(nz):
        z = -6 + (H + 6) * j / (nz - 1)
        zf = j / (nz - 1)
        for i in range(nt):
            t = 2 * math.pi * i / nt
            r = 34 - 10 * zf + 4.0 * math.sin(3 * t + 1.1) + 3.0 * math.sin(7 * t + zf * 4) + 2.0 * math.sin(13 * t + 2 * zf)
            r *= 1.0 + 0.10 * math.sin(5 * zf * 6.28 + 2 * t)
            V.append((cx + r * math.cos(t), cy + r * math.sin(t) * 0.85, z))
    for j in range(nz - 1):
        for i in range(nt):
            a, b = j * nt + i, j * nt + (i + 1) % nt
            F.append((a, b, b + nt, a + nt))
    top = len(V)
    V.append((cx, cy, H + 1.5))
    for i in range(nt):
        F.append(((nz - 1) * nt + i, (nz - 1) * nt + (i + 1) % nt, top))
    ob = X.mesh_obj("CastleCliff", V, F, coll=coll, mat=stone_mat("CliffStone"), smooth=True)
    tx = bpy.data.textures.new("CliffNoise", "CLOUDS")
    tx.noise_scale = 6.0
    tx.noise_depth = 3
    md = ob.modifiers.new("Disp", "DISPLACE")
    md.texture = tx
    md.strength = 4.0
    md.texture_coords = "GLOBAL"
    ob.modifiers.new("Sub", "SUBSURF").levels = 1
    ob.modifiers.move(1, 0)
    return ob


def waterfall(coll):
    m, nb, out = X.new_mat("Waterfall")
    K = C.ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    u, v, _ = nb.sep(tc.outputs["UV"])
    flow = nb.m("ADD", nb.m("MULTIPLY", v, 6.0), nb.m("MULTIPLY", K["T"], 1.4))   # streaks scroll downward
    st = C.noise(nb, nb.xyz(nb.m("MULTIPLY", u, 26.0), flow, 0.0), scale=1.0, detail=3.0, rough=0.55)
    streak = nb.mr(st.outputs["Fac"], 0.35, 0.7)
    en = C.noise(nb, nb.xyz(nb.m("MULTIPLY", v, 4.0), 0.0, 0.0), scale=1.0, detail=2.0)
    rag = nb.m("MULTIPLY", nb.m("SUBTRACT", en.outputs["Fac"], 0.5), 0.5)
    edge = nb.m("MULTIPLY", nb.mr(nb.m("ADD", u, rag), 0.05, 0.3), nb.mr(nb.m("SUBTRACT", u, rag), 0.95, 0.7))
    a = nb.m("MULTIPLY", nb.m("ADD", 0.35, nb.m("MULTIPLY", streak, 0.65)), edge)
    a = nb.m("MULTIPLY", a, nb.mr(v, 1.0, 0.92))
    col = nb.mix(streak, (0.55, 0.38, 0.95), (1.0, 0.85, 1.0))
    sh = C.mix_shader(nb, nb.m("MULTIPLY", a, 0.8), C.transparent(nb), C.emission(nb, col, nb.m("MULTIPLY", K["glow"], 1.3)))
    nb.link(sh, out.inputs["Surface"])
    C.blended(m)
    cx, cy, _ = C.CASTLE_POS
    H = C.CASTLE_CLIFF_H
    # a slightly bent sheet: from the cliff lip out and down to the water
    nu, nv = 6, 20
    V, F, UV = [], [], []
    for j in range(nv):
        t = j / (nv - 1)
        z = H * 0.82 * (1 - t) - 1.0
        out_ = 2.0 + 5.0 * t ** 0.6
        for i in range(nu):
            s = i / (nu - 1)
            V.append((C.FALL_X - 7 + 14 * s, cy - 27 - out_, z))
            UV.append((s, 1 - t))
    for j in range(nv - 1):
        for i in range(nu - 1):
            F.append((j * nu + i, j * nu + i + 1, (j + 1) * nu + i + 1, (j + 1) * nu + i))
    ob = X.mesh_obj("Waterfall", V, F, coll=coll, mat=m, uvs=UV)
    C.hide_shadow(ob)
    return ob


def mist_material():
    m, nb, out = X.new_mat("MistCard")
    K = C.ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    u, v, _ = nb.sep(tc.outputs["UV"])
    oi = nb.n("ShaderNodeObjectInfo")
    n = C.noise(nb, nb.xyz(nb.m("ADD", nb.m("MULTIPLY", u, 4.0), nb.m("MULTIPLY", K["T"], 0.01)),
                           nb.m("MULTIPLY", v, 1.5), nb.m("MULTIPLY", oi.outputs["Random"], 50.0)), scale=1.5, detail=3.0)
    a = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.mr(v, 1.0, 0.1), nb.mr(v, 0.0, 0.15)),
             nb.m("MULTIPLY", nb.mr(u, 0.0, 0.2), nb.mr(u, 1.0, 0.8)))
    a = nb.m("MULTIPLY", a, nb.mr(n.outputs["Fac"], 0.3, 0.75))
    stren = nb.n("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="mist").outputs["Fac"]
    a = nb.m("MULTIPLY", a, stren)
    sh = C.mix_shader(nb, a, C.transparent(nb), C.emission(nb, (0.5, 0.26, 0.85), nb.m("MULTIPLY", K["glow"], 0.45)))
    nb.link(sh, out.inputs["Surface"])
    C.blended(m)
    return m


def mist_cards(coll):
    m = mist_material()
    cx, cy, _ = C.CASTLE_POS
    spots = [  # (x, y, z, w, h, strength)
        (C.FALL_X, cy - 58, -2, 70, 22, 0.9),          # waterfall foot
        (cx + 10, cy - 70, -2, 220, 26, 0.55),          # around the island foot
        (-120, 360, -3, 420, 40, 0.6), (200, 520, -3, 600, 60, 0.6),   # mid-distance bands
        (-500, 900, 0, 1200, 120, 0.7), (500, 1100, 0, 1400, 140, 0.7),   # mountain feet
    ]
    obs = []
    for i, (x, y, z, w, h, s) in enumerate(spots):
        ob = X.mesh_obj(f"Mist_{i:02d}", [(-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0)], [(0, 1, 2, 3)],
                        coll=coll, mat=m, uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
        ob.location = (x, y, z)
        ob.scale = (w, h, 1)
        r = C.billboard_rot((x, y, z + h / 2))
        ob.rotation_euler = (math.pi / 2, 0, r.z)
        ob["mist"] = s
        C.hide_shadow(ob)
        obs.append(ob)
    return obs


def ridge_material():
    m, nb, out = X.new_mat("Ridge")
    K = C.ctrl(nb)
    hf = nb.sattr("hf").outputs["Fac"]
    tc = nb.n("ShaderNodeTexCoord")
    u, _, _ = nb.sep(tc.outputs["UV"])
    haze = nb.n("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="haze").outputs["Fac"]
    g = C.noise(nb, nb.xyz(nb.m("MULTIPLY", u, 60.0), nb.m("MULTIPLY", hf, 3.0), 0.0), scale=1.0, detail=4.0)
    gully = nb.mr(g.outputs["Fac"], 0.35, 0.7)
    body = nb.mix(nb.m("MULTIPLY", gully, hf), (0.012, 0.006, 0.035), (0.08, 0.04, 0.17))
    rim = nb.m("POWER", hf, 10.0)
    body = nb.mix(nb.m("MULTIPLY", rim, 0.5), body, (0.55, 0.30, 0.85), blend="ADD")
    bottom = nb.mr(hf, 0.35, 0.0)                                       # mist at the feet
    col = nb.mix(nb.m("MAXIMUM", nb.m("MULTIPLY", haze, 0.75), nb.m("MULTIPLY", bottom, 0.7)), body, (0.22, 0.11, 0.42))
    nb.link(C.emission(nb, col, nb.m("ADD", 0.6, nb.m("MULTIPLY", K["sky_glow"], 0.4))), out.inputs["Surface"])
    return m


RIDGES = [  # (distance y, x0, x1, base height, amplitude, haze, seed)
    (520, -900, -40, 40, 55, 0.35, 1),
    (650, 120, 1100, 90, 120, 0.30, 2),
    (900, -1400, 200, 70, 90, 0.55, 3),
    (1100, 0, 1700, 160, 210, 0.50, 4),
    (1500, -2000, 2000, 110, 150, 0.72, 5),
]


def mountains(coll):
    m = ridge_material()
    obs = []
    for k, (y, x0, x1, base, amp, haze, seed) in enumerate(RIDGES):
        n = 220
        xs = np.linspace(x0, x1, n)
        t = (xs - x0) / (x1 - x0)
        prof = base + amp * (0.55 * np.sin(np.pi * t) ** 0.8 + 0.45 * (0.5 + 0.5 * _fbm(xs * 0.08, xs * 0.02, seed, 5)))
        prof += amp * 0.14 * np.abs(np.sin(xs * 0.021 + seed)) * (0.5 + 0.5 * np.sin(xs * 0.007 + 2 * seed))   # a few peaks
        prof *= np.clip(np.sin(np.pi * t) * 3, 0, 1)                    # taper the ends into the water
        rows = 4
        V, F, UV, HF = [], [], [], []
        for j in range(rows):
            f = j / (rows - 1)
            for i in range(n):
                V.append((xs[i], y + (1 - f) * 25 * math.sin(i * 0.3), -4 + (prof[i] + 4) * f))
                UV.append((t[i], f))
                HF.append(f)
        for j in range(rows - 1):
            for i in range(n - 1):
                F.append((j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i))
        ob = X.mesh_obj(f"Ridge_{k}", V, F, coll=coll, mat=m, uvs=UV)
        ob.data.attributes.new("hf", "FLOAT", "POINT").data.foreach_set("value", np.array(HF, np.float32))
        ob["haze"] = haze
        C.hide_shadow(ob)
        obs.append(ob)
    return obs


def haze_box(coll):
    """bounded low haze over the water between the ruins and the island (Principled Volume, forward scattering)"""
    m, nb, out = X.new_mat("HazeVol")
    K = C.ctrl(nb)
    tc = nb.n("ShaderNodeTexCoord")
    _, _, gz = nb.sep(tc.outputs["Generated"])
    dens = nb.m("MULTIPLY", nb.m("POWER", nb.mr(gz, 0.0, 1.0, 1.0, 0.0), 2.5), 0.0016)      # thick at the water, thin at the top
    pv = nb.n("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (*C.HAZE, 1)
    nb.link(dens, pv.inputs["Density"])
    pv.inputs["Anisotropy"].default_value = 0.45
    pv.inputs["Emission Color"].default_value = (0.55, 0.30, 0.90, 1)
    nb.link(nb.m("MULTIPLY", nb.m("MULTIPLY", dens, 2.0), K["glow"]), pv.inputs["Emission Strength"])
    nb.link(pv.outputs[0], out.inputs["Volume"])
    ob = X.boxes_mesh("HazeBox", [(-700, 60, -2, 900, 1000, 55)], coll=coll, bottom=True)
    ob.data.materials.append(m)
    C.hide_shadow(ob)
    return ob
