"""Sands — the floating LED cube + its energy (bpy). All motion/LED/energy values come from timeline.build() and are
baked as keys (no drivers, no handlers); the arcs / ground strikes / sand spray are analytic Geometry Nodes on Scene
Time + keyed modifier inputs, constant point counts (gotcha 10), so every frame renders on its own.

Hierarchy: CubeRoot (hover + kick dip + punch scale) → CubeYaw (Z, kick slams) → CubeFlipX (X, snare flips) →
CubeFlipY (Y, snare flips) → CubeFrame / CubePanels / CubeEdges / Arcs (arcs ride the cube, bulging out of it).
"""
import math
import os

import bpy
import numpy as np

from lib import bpyx as X
from lib.bpyx import bake, bake_vec

import faces as FC

EDGE = 6.0                 # cube edge (m)
HS = EDGE / 2
LIFT = 15.0                # bottom of the cube above the sand (Zeke: "floating in the air like 15 meters")
ZC = LIFT + HS + 0.8       # centre height above the pan (+ margin: hover + kick dip never take the resting bottom below 15 m)
N_CELLS = 40               # LED cells across a face (lyric_viz STAGE_FACE_CELLS)
LOGO_TEX = "/home/iris/IRIS/scratch/scenes/sands/tex/logo_led40.png"
LED_K = 9.0                # LED emission scale (tuned on stills)
N_ARCS, ARC_PTS = 40, 28
STRIKE_PTS = 56
GRAINS = 700
DUST = 18

CTRL_KEYS = ["kenv", "senv", "I", "hot", "big", "gain", "swap", "bi", "ph", "strobe", "energy", "glitch", "fr",
             "Ar", "Ag", "Ab", "Br", "Bg", "Bb"]
NEON_A = (0.0, 0.85, 1.0)  # cyberpunk trims: edges lean cyan, face trims lean magenta (mixed with the section colours)
NEON_B = (1.0, 0.05, 0.6)


# ------------------------------------------------------------------------------------------------ CTRL group

def ctrl_group(T, cuts=None):
    """ONE shader node group with keyed Value nodes; every cube material reads it (a section change = one curve)."""
    ng = bpy.data.node_groups.new("CTRL", "ShaderNodeTree")
    nb = X.NB(ng)
    go = nb.n("NodeGroupOutput")
    vals = dict(kenv=T["kenv"], senv=T["senv"], I=T["I"], hot=T["hot"].astype(float), big=T["big"].astype(float),
                gain=T["gain"], swap=T["swap"], bi=T["bi"], ph=T["ph"], strobe=T["strobe"], energy=T["energy"],
                glitch=T["glitch"], fr=np.arange(1, T["n"] + 1, dtype=float),
                Ar=T["A"][:, 0], Ag=T["A"][:, 1], Ab=T["A"][:, 2], Br=T["B"][:, 0], Bg=T["B"][:, 1], Bb=T["B"][:, 2])
    for k in CTRL_KEYS:
        ng.interface.new_socket(k, in_out="OUTPUT", socket_type="NodeSocketFloat")
        s = nb.val(k, float(vals[k][0]))
        nb.link(s, go.inputs[k])
        interp = "CONSTANT" if k in ("bi", "swap", "hot", "big") else "LINEAR"
        bake(ng, f'nodes["{k}"].outputs[0].default_value', vals[k], interp=interp)
    return ng


def ctrl_node(nb, ng):
    g = nb.n("ShaderNodeGroup")
    g.node_tree = ng
    return g.outputs


def _rgb(nb, r, g, b):
    c = nb.n("ShaderNodeCombineColor")
    nb.set(c.inputs[0], r)
    nb.set(c.inputs[1], g)
    nb.set(c.inputs[2], b)
    return c.outputs[0]


def _scale_col(nb, col, s):
    return nb.mix(1.0, col, nb.xyz(s, s, s), blend="MULTIPLY") if not isinstance(s, (int, float)) else \
        nb.mix(1.0, col, (s, s, s), blend="MULTIPLY")


def _add(nb, a, b):
    return nb.mix(1.0, a, b, blend="ADD")


# ------------------------------------------------------------------------------------------------ materials

def led_material(ng, tex):
    """SIX DIFFERENT FACES (faces.py header): emblem ×3 variants, album cover, spectrum visualizer, circuit core.
    LED cells (40 / 150 album / 48 visualizer) under a glossy cover, PCB circuitry with travelling pulses glowing
    between the cells, panel seams, row-shift glitch on snares/strobes. tex = dict(logo, album, album_crop, spec,
    spec_rows, circuit)."""
    m, nb, out = X.new_mat("LED")
    C = ctrl_node(nb, ng)
    A = _rgb(nb, C["Ar"], C["Ag"], C["Ab"])
    B = _rgb(nb, C["Br"], C["Bg"], C["Bb"])
    tc = nb.n("ShaderNodeTexCoord")
    uvx, uvy, _ = nb.sep(tc.outputs["UV"])
    face = nb.sattr("face").outputs["Fac"]
    isf = [nb.m("COMPARE", face, float(k), 0.1) for k in range(6)]
    kenv, hot, I, swap = C["kenv"], C["hot"], C["I"], C["swap"]
    # glitch: whole rows slide sideways for a frame or two (not on the visualizer — its bars must stay true)
    band = nb.m("FLOOR", nb.m("MULTIPLY", uvy, 14.0))
    wn = nb.n("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    nb.link(nb.xyz(band, nb.m("FLOOR", nb.m("MULTIPLY", C["fr"], 0.5)), face), wn.inputs["Vector"])
    h = wn.outputs["Value"]
    shift = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("GREATER_THAN", h, 0.72), nb.m("SUBTRACT", h, 0.72)),
                 nb.m("MULTIPLY", C["glitch"], nb.m("SUBTRACT", 1.0, isf[3])))
    uvx = nb.m("FRACT", nb.m("ADD", uvx, nb.m("MULTIPLY", shift, 0.6)))
    # cell grid, per-face pitch
    N = nb.m("ADD", 40.0, nb.m("ADD", nb.m("MULTIPLY", isf[1], 110.0), nb.m("MULTIPLY", isf[3], 8.0)))
    cx = nb.m("FLOOR", nb.m("MULTIPLY", uvx, N))
    cy = nb.m("FLOOR", nb.m("MULTIPLY", uvy, N))
    cux = nb.m("DIVIDE", nb.m("ADD", cx, 0.5), N)
    cuy = nb.m("DIVIDE", nb.m("ADD", cy, 0.5), N)
    lx = nb.m("SUBTRACT", nb.m("MULTIPLY", nb.m("FRACT", nb.m("MULTIPLY", uvx, N)), 2.0), 1.0)
    ly = nb.m("SUBTRACT", nb.m("MULTIPLY", nb.m("FRACT", nb.m("MULTIPLY", uvy, N)), 2.0), 1.0)
    r4 = nb.m("POWER", nb.m("ADD", nb.m("POWER", nb.m("ABSOLUTE", lx), 4.0), nb.m("POWER", nb.m("ABSOLUTE", ly), 4.0)), 0.25)
    dot = nb.m("MULTIPLY", nb.mr(r4, 0.80, 0.60), nb.m("ADD", 0.82, nb.m("MULTIPLY", 0.18, nb.m(
        "EXPONENT", nb.m("MULTIPLY", -1.5, nb.m("ADD", nb.m("MULTIPLY", lx, lx), nb.m("MULTIPLY", ly, ly)))))))
    dot = nb.m("ADD", dot, nb.m("MULTIPLY", isf[1], nb.m("MULTIPLY", nb.m("SUBTRACT", 1.0, dot), 0.55)))  # album: finer, softer
    cheb = nb.m("MAXIMUM", nb.m("ABSOLUTE", nb.m("SUBTRACT", cux, 0.5)), nb.m("ABSOLUTE", nb.m("SUBTRACT", cuy, 0.5)))

    # --- EMBLEM layers (R outline, G fill, B plume) sampled at the cell centre
    ltex = nb.tex(tex["logo"], nb.xyz(cux, cuy, 0.0), interp="Closest")
    ltex.extension = "CLIP"
    sepc = nb.n("ShaderNodeSeparateColor")
    nb.link(ltex.outputs["Color"], sepc.inputs[0])
    dark, fill, red = sepc.outputs[0], sepc.outputs[1], sepc.outputs[2]
    out_col = _scale_col(nb, nb.mix(swap, A, B), 0.85)
    tint = nb.mix(swap, B, A)
    fill_col = nb.mix(0.70, tint, (1.0, 1.0, 1.0))
    lay = nb.m("ADD", nb.m("ADD", dark, fill), red, clamp=True)
    nlay = nb.m("SUBTRACT", 1.0, lay)
    bgk = nb.m("ADD", 0.07, nb.m("MULTIPLY", 0.20, nb.m("MULTIPLY", kenv, hot)))
    c_fg = _scale_col(nb, out_col, dark)
    c_fg = _add(nb, c_fg, _scale_col(nb, fill_col, nb.m("MULTIPLY", fill, nb.m("ADD", 0.8, nb.m("MULTIPLY", 0.2, I)))))
    c_fg = _add(nb, c_fg, _scale_col(nb, (1.0, 0.0097, 0.0168), red))
    c_logo = _add(nb, c_fg, _scale_col(nb, B, nb.m("MULTIPLY", bgk, nlay)))
    # face 2: chase rings behind the emblem (bright in drops, a slow dim drift when calm)
    w = nb.m("SUBTRACT", nb.m("MULTIPLY", cheb, 2.5), C["ph"])
    mm = nb.m("MULTIPLY", nb.m("LESS_THAN", nb.m("FRACT", w), 0.35), nb.m("ADD", 0.6, nb.m("MULTIPLY", 0.4, kenv)))
    mx = nb.m("FLOORED_MODULO", nb.m("FLOOR", w), 2.0)
    c_chase = _scale_col(nb, nb.mix(mx, A, B), nb.m("MULTIPLY", mm, nb.m("MULTIPLY", nlay, nb.m("ADD", 0.25, nb.m("MULTIPLY", 0.75, hot)))))
    c_rings = _add(nb, c_fg, c_chase)
    # face 4 (top): checker blocks behind the emblem, beat parity
    par = nb.m("FLOORED_MODULO", nb.m("FLOOR", C["bi"]), 2.0)
    blk = nb.m("FLOORED_MODULO", nb.m("ADD", nb.m("ADD", nb.m("FLOOR", nb.m("MULTIPLY", cux, 4.0)),
                                                   nb.m("FLOOR", nb.m("MULTIPLY", cuy, 4.0))), C["bi"]), 2.0)
    blk = nb.m("LESS_THAN", blk, 0.5)
    bk = nb.m("MULTIPLY", blk, nb.m("MULTIPLY", nb.m("ADD", 0.12, nb.m("MULTIPLY", 0.40, hot)),
                                    nb.m("ADD", 0.6, nb.m("MULTIPLY", 0.4, kenv))))
    c_top = _add(nb, c_fg, _scale_col(nb, nb.mix(par, B, A), nb.m("MULTIPLY", bk, nlay)))

    # --- ALBUM COVER (path-driven image, centre-cropped square)
    kx, ky = tex["album_crop"]
    au = nb.m("ADD", 0.5, nb.m("MULTIPLY", nb.m("SUBTRACT", cux, 0.5), kx))
    av = nb.m("ADD", 0.5, nb.m("MULTIPLY", nb.m("SUBTRACT", cuy, 0.5), ky))
    atex = nb.tex(tex["album"], nb.xyz(au, av, 0.0), interp="Linear")
    atex.extension = "EXTEND"
    abri = nb.m("ADD", 0.30, nb.m("MULTIPLY", 0.42, nb.m("ADD", C["gain"], nb.m("MULTIPLY", 0.25, nb.m("MULTIPLY", kenv, hot)))))
    c_album = _scale_col(nb, atex.outputs["Color"], abri)

    # --- VISUALIZER: 16 bars × (2 lit + 1 gap) cells, level + falling peak caps from the baked atlas
    NB_, COLS = float(FC.N_BARS), float(FC.ATLAS_COLS)
    bar = nb.m("FLOOR", nb.m("DIVIDE", cx, 3.0))
    inbar = nb.m("LESS_THAN", nb.m("SUBTRACT", cx, nb.m("MULTIPLY", bar, 3.0)), 1.5)
    f0 = nb.m("SUBTRACT", C["fr"], 1.0)
    row = nb.m("FLOOR", nb.m("DIVIDE", f0, COLS))
    col = nb.m("ADD", nb.m("MULTIPLY", nb.m("SUBTRACT", f0, nb.m("MULTIPLY", row, COLS)), NB_), bar)
    su = nb.m("DIVIDE", nb.m("ADD", col, 0.5), COLS * NB_)
    sv = nb.m("SUBTRACT", 1.0, nb.m("DIVIDE", nb.m("ADD", row, 0.5), float(tex["spec_rows"])))
    stx = nb.tex(tex["spec"], nb.xyz(su, sv, 0.0), interp="Closest")
    stx.extension = "EXTEND"
    lvl, pkv, _b = nb.sep(stx.outputs["Color"])
    top = 0.90
    hcell = nb.m("DIVIDE", nb.m("ADD", cy, 0.5), N)
    lit = nb.m("MULTIPLY", inbar, nb.m("LESS_THAN", hcell, nb.m("MULTIPLY", lvl, top)))
    capy = nb.m("FLOOR", nb.m("MULTIPLY", nb.m("MULTIPLY", pkv, top), N))
    cap = nb.m("MULTIPLY", nb.m("MULTIPLY", inbar, nb.m("COMPARE", cy, capy, 0.5)), nb.m("GREATER_THAN", pkv, 0.03))
    grad = nb.mix(nb.mr(hcell, 0.0, 0.62), A, B)
    grad = nb.mix(nb.mr(hcell, 0.62, 0.85), grad, (1.0, 0.18, 0.05))
    c_vis = _scale_col(nb, grad, nb.m("MULTIPLY", lit, nb.m("ADD", 0.45, nb.m("MULTIPLY", 0.2, kenv))))   # < 1: keeps hue
    c_vis = _add(nb, c_vis, _scale_col(nb, (1.0, 1.0, 1.0), nb.m("MULTIPLY", cap, 0.85)))
    c_vis = _add(nb, c_vis, _scale_col(nb, A, nb.m("MULTIPLY", inbar, 0.035)))           # unlit bar ghosts
    vis_on = nb.m("MAXIMUM", lit, cap)

    # --- CIRCUIT CORE (bottom): square rings drifting outward, kick-lit centre
    ring = nb.m("LESS_THAN", nb.m("FRACT", nb.m("SUBTRACT", nb.m("MULTIPLY", cheb, 6.0), nb.m("MULTIPLY", C["fr"], 0.02))), 0.16)
    c_core = _scale_col(nb, A, nb.m("MULTIPLY", ring, nb.m("ADD", 0.12, nb.m("MULTIPLY", 0.6, kenv))))
    c_core = _add(nb, c_core, _scale_col(nb, B, nb.m("MULTIPLY", nb.m("LESS_THAN", cheb, 0.11), nb.m("ADD", 0.3, nb.m("MULTIPLY", 0.7, kenv)))))

    # select by face
    c = c_logo
    for k, ck in ((1, c_album), (2, c_rings), (3, c_vis), (4, c_top), (5, c_core)):
        c = nb.mix(isf[k], c, ck)
    border = nb.m("GREATER_THAN", cheb, nb.m("SUBTRACT", 0.5, nb.m("DIVIDE", 1.0, N)))
    c = _add(nb, c, _scale_col(nb, nb.mix(0.5, A, NEON_A), nb.m("MULTIPLY", border, 0.9)))
    not_album = nb.m("SUBTRACT", 1.0, isf[1])
    c = _scale_col(nb, c, nb.m("ADD", isf[1], nb.m("MULTIPLY", not_album, C["gain"])))     # album has its own level
    c = _add(nb, c, (0.0033, 0.0033, 0.0056))                          # an unlit LED still shows (11,11,17)
    c = _scale_col(nb, c, dot)
    # --- circuitry between the cells (all faces but the album): dim traces + travelling pulses
    ctex = nb.tex(tex["circuit"], nb.xyz(nb.m("FRACT", nb.m("ADD", uvx, nb.m("MULTIPLY", face, 0.31))),
                                         nb.m("FRACT", nb.m("ADD", uvy, nb.m("MULTIPLY", face, 0.17))), 0.0), interp="Linear")
    trace, along, via = nb.sep(ctex.outputs["Color"])
    pulse = nb.m("LESS_THAN", nb.m("FRACT", nb.m("SUBTRACT", nb.m("MULTIPLY", along, 2.0),
                                                    nb.m("ADD", nb.m("DIVIDE", C["fr"], 75.0), nb.m("MULTIPLY", face, 0.13)))), 0.07)
    lit_any = nb.m("ADD", nb.m("MULTIPLY", nb.m("ADD", isf[0], nb.m("ADD", isf[2], isf[4])), lay), nb.m("MULTIPLY", isf[3], vis_on))
    cpow = nb.m("MULTIPLY", nb.m("MULTIPLY", trace, nb.m("SUBTRACT", 1.0, nb.m("MINIMUM", lit_any, 1.0))),
                nb.m("ADD", nb.m("ADD", 0.05, nb.m("MULTIPLY", 0.06, via)), nb.m("MULTIPLY", pulse, nb.m("ADD", 0.25, nb.m("MULTIPLY", 0.55, I)))))
    cpow = nb.m("MULTIPLY", cpow, nb.m("MULTIPLY", not_album, nb.m("ADD", 1.0, isf[5])))
    cpow = nb.m("MULTIPLY", cpow, nb.m("ADD", 0.3, nb.m("MULTIPLY", 0.7, C["gain"])))
    c = _add(nb, c, _scale_col(nb, nb.mix(0.45, B, NEON_B), cpow))
    # panel seams: the face is four tiles (album excepted) — dark grooves, no emission
    seam = nb.m("MULTIPLY", not_album, nb.m("MAXIMUM", nb.m("LESS_THAN", nb.m("ABSOLUTE", nb.m("SUBTRACT", uvx, 0.5)), 0.006),
                                              nb.m("LESS_THAN", nb.m("ABSOLUTE", nb.m("SUBTRACT", uvy, 0.5)), 0.006)))
    c = _scale_col(nb, c, nb.m("SUBTRACT", 1.0, seam))
    b = X.principled(nb, (0.012, 0.012, 0.014), 0.16, 0.0)
    nb.link(c, b.inputs["Emission Color"])
    nb.set(b.inputs["Emission Strength"], LED_K)
    nb.set(b.inputs["Coat Weight"], 0.6)
    nb.set(b.inputs["Coat Roughness"], 0.05)
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def edge_material(ng):
    """edge light strips: colour A (white on strobes / snares), blackout aware"""
    m, nb, out = X.new_mat("CubeEdgeLight")
    C = ctrl_node(nb, ng)
    A = _rgb(nb, C["Ar"], C["Ag"], C["Ab"])
    wh = nb.m("ADD", nb.m("MULTIPLY", C["strobe"], 1.0), nb.m("MULTIPLY", C["senv"], nb.m("MULTIPLY", C["hot"], 0.6)), clamp=True)
    col = nb.mix(wh, nb.mix(0.45, A, NEON_A), (1, 1, 1))
    st = nb.m("MULTIPLY", nb.m("ADD", nb.m("MULTIPLY", C["gain"], 18.0), nb.m("MULTIPLY", C["strobe"], 60.0)), 1.0)
    e = nb.n("ShaderNodeEmission")
    nb.link(col, e.inputs["Color"])
    nb.link(st, e.inputs["Strength"])
    nb.link(e.outputs[0], out.inputs["Surface"])
    return m


def trim_material(ng):
    """neon face trim (a lit rectangle around every panel): magenta-leaning colour B, pumps on kicks"""
    m, nb, out = X.new_mat("CubeTrimNeon")
    C = ctrl_node(nb, ng)
    B = _rgb(nb, C["Br"], C["Bg"], C["Bb"])
    col = nb.mix(nb.m("MULTIPLY", C["strobe"], 0.8), nb.mix(0.5, B, NEON_B), (1, 1, 1))
    st = nb.m("ADD", nb.m("MULTIPLY", C["gain"], 12.0), nb.m("MULTIPLY", C["kenv"], nb.m("MULTIPLY", C["hot"], 10.0)))
    e = nb.n("ShaderNodeEmission")
    nb.link(col, e.inputs["Color"])
    nb.link(st, e.inputs["Strength"])
    nb.link(e.outputs[0], out.inputs["Surface"])
    return m


def frame_material():
    """dark anodised gunmetal with roughness variation (the realism recipe)"""
    m, nb, out = X.new_mat("CubeFrameMetal")
    tc = nb.n("ShaderNodeTexCoord")
    nz = nb.n("ShaderNodeTexNoise")
    nb.set(nz.inputs["Scale"], 6.0)
    nb.set(nz.inputs["Detail"], 6.0)
    nb.link(tc.outputs["Object"], nz.inputs["Vector"])
    r = nb.mr(nz.outputs["Fac"], 0.3, 0.7, 0.22, 0.42)
    b = X.principled(nb, (0.09, 0.095, 0.105), r, 1.0)
    bump = nb.n("ShaderNodeBump")
    nb.set(bump.inputs["Strength"], 0.08)
    nb.link(nz.outputs["Fac"], bump.inputs["Height"])
    nb.link(bump.outputs["Normal"], b.inputs["Normal"])
    nb.link(b.outputs[0], out.inputs["Surface"])
    return m


def arc_materials():
    core = X.emit_mat("ArcCore", (0.70, 0.78, 1.0), 80.0)
    m, nb, out = X.new_mat("ArcHalo")
    e = nb.n("ShaderNodeEmission")
    nb.set(e.inputs["Color"], (0.30, 0.38, 1.0))
    nb.set(e.inputs["Strength"], 6.0)
    tr = nb.n("ShaderNodeBsdfTransparent")
    lw = nb.n("ShaderNodeLayerWeight")
    nb.set(lw.inputs["Blend"], 0.35)
    fac = nb.m("MULTIPLY", nb.m("POWER", lw.outputs["Facing"], 3.0), 0.55)     # bright core-on, soft edges
    mix = nb.n("ShaderNodeMixShader")
    nb.link(fac, mix.inputs[0])
    nb.link(tr.outputs[0], mix.inputs[1])
    nb.link(e.outputs[0], mix.inputs[2])
    add = nb.n("ShaderNodeAddShader")
    nb.link(tr.outputs[0], add.inputs[0])
    nb.link(mix.outputs[0], add.inputs[1])
    nb.link(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    m.use_transparency_overlap = False
    halo = m
    return core, halo


# ------------------------------------------------------------------------------------------------ geometry

def _panel(name, n, coll, parent, mat):
    """six inset LED quads with face ids + upright UVs (sides: v up = +Z)"""
    s = HS * 0.985
    a = HS * 0.90
    faces = [  # (normal, u axis, v axis) — sides first: 0 −Y (front), 1 +X, 2 +Y, 3 −X, 4 +Z, 5 −Z
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0))]
    V, F, UV, FID = [], [], [], []
    for k, (nrm, u, v) in enumerate(faces):
        nrm, u, v = map(np.array, (nrm, u, v))
        o = len(V)
        for (su, sv) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            V.append(nrm * s + u * su * a + v * sv * a)
            UV.append(((su + 1) / 2, (sv + 1) / 2))
        F.append((o, o + 1, o + 2, o + 3))
        FID.append(k)
    ob = X.mesh_obj(name, V, F, coll=coll, mat=mat, parent=parent, uvs=UV)
    ob.data.attributes.new("face", "FLOAT", "FACE").data.foreach_set("value", np.array(FID, np.float32))
    return ob


def _edges_mb(t, off, shorten=0.0, mb=None):
    mb = mb or X.MB()
    for ax in range(3):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                c = [0.0, 0.0, 0.0]
                o = [i for i in range(3) if i != ax]
                c[o[0]] = s1 * (HS + off)
                c[o[1]] = s2 * (HS + off)
                sz = [t, t, t]
                sz[ax] = EDGE + 2 * off - shorten
                mb.box(c, sz)
    return mb


SIDE_AXES = [((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
             ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0))]


def _face_box(mb, k, cu, cv, du, dv, depth, out):
    """box on face k: centre (cu, cv) in face units (m), extent du × dv, `depth` thick, centred `out` m from the centre"""
    nrm, u, v = (np.array(a, float) for a in SIDE_AXES[k])
    c = nrm * out + u * cu + v * cv
    size = np.abs(u) * du + np.abs(v) * dv + np.abs(nrm) * depth
    mb.box(tuple(c), tuple(size))


def prepare_textures(song_wav, n, tex_dir, R):
    """logo (v1), album cover (path-driven), baked spectrum atlas, circuitry → dict for led_material"""
    lev, pk = FC.spectrum(song_wav, n)
    spec = FC.write_png(FC.atlas(lev, pk), f"{tex_dir}/spectrum_atlas.png", "SpectrumAtlas")
    circ = FC.write_png(FC.circuit(walkers=260), f"{tex_dir}/circuit.png", "Circuit")
    apath, placeholder = FC.album_path()
    # Zeke's own art: copied next to the generated textures (outside /data/iris_assets, so the credits tool counts it
    # as OWN, not third-party). Swap = replace FC.ALBUM + rebuild — the copy is refreshed on every build.
    import shutil
    used = f"{tex_dir}/album_cover_used{os.path.splitext(apath)[1].lower()}"
    shutil.copyfile(apath, used)
    album = X.img(used, True)
    w, h = album.size
    sq = float(min(w, h))
    R.log(f"[faces] album cover {'PLACEHOLDER — real cover missing at ' + FC.ALBUM if placeholder else apath} "
          f"({w}×{h}, centre-cropped square); spectrum {FC.N_BARS} bars × {n} frames baked")
    R.add("album_cover", dict(path=apath, placeholder=placeholder, size=[w, h]))
    return dict(logo=X.img(LOGO_TEX, False), album=album, album_crop=(sq / w, sq / h), spec=spec,
                spec_rows=spec.size[1], circuit=circ, levels=lev)


def cube(coll, T, R, ng, tex, cuts=None):
    root = X.empty("CubeRoot", (0, 0, ZC), coll=coll)
    yaw = X.empty("CubeYaw", parent=root, coll=coll)
    fx = X.empty("CubeFlipX", parent=yaw, coll=coll)
    fy = X.empty("CubeFlipY", parent=fx, coll=coll)
    n = T["n"]
    bake_vec(root, "location", np.c_[np.zeros(n), np.zeros(n), ZC + T["z_off"]])
    bake_vec(root, "scale", np.repeat(T["scale"][:, None], 3, 1))
    bake(yaw, "rotation_euler", T["yaw"], index=2)
    bake(fx, "rotation_euler", T["flip_x"], index=0)
    bake(fy, "rotation_euler", T["flip_y"], index=1)
    fm = frame_material()
    # hard-edged frame (v2 cyberpunk): heavy chamfered edge bars, oversized corner blocks, edge clamps, panel-seam
    # ribs on every face but the album, all one gunmetal object with a 1-segment (hard) chamfer
    mb = _edges_mb(0.36, 0.0)
    for c in [(sx * HS, sy * HS, sz * HS) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]:
        mb.box(c, (0.95, 0.95, 0.95))
    for ax in range(3):                                        # two clamps per edge at ±HS/2
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                o = [i for i in range(3) if i != ax]
                for t in (-0.5, 0.5):
                    c = [0.0, 0.0, 0.0]
                    c[o[0]], c[o[1]], c[ax] = s1 * HS, s2 * HS, t * HS
                    sz = [0.56, 0.56, 0.56]
                    sz[ax] = 0.34
                    mb.box(c, sz)
    for k in (0, 2, 3, 4, 5):                                  # panel seams (the shader's grooves line up)
        _face_box(mb, k, 0.0, 0.0, 0.07, 2 * HS * 0.90, 0.06, HS * 0.985 + 0.02)
        _face_box(mb, k, 0.0, 0.0, 2 * HS * 0.90, 0.07, 0.06, HS * 0.985 + 0.02)
    frame = mb.obj("CubeFrame", fm, parent=fy, coll=coll)
    bev = frame.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments, bev.limit_method = 0.05, 1, "ANGLE"
    # dark backing box inside (no see-through between panel and frame)
    back = X.MB().box((0, 0, 0), (EDGE * 0.96, EDGE * 0.96, EDGE * 0.96)).obj("CubeCore", X.simple_mat("CubeCore", (0.01, 0.01, 0.012), 0.5), parent=fy, coll=coll)
    panels = _panel("CubePanels", 6, coll, fy, led_material(ng, tex))
    strips = _edges_mb(0.08, 0.20, shorten=1.2).obj("CubeEdges", edge_material(ng), parent=fy, coll=coll)
    tmb = X.MB()                                               # neon trim rectangle around each panel
    a_, tw, out = HS * 0.90 + 0.06, 0.05, HS * 0.985 + 0.015
    for k in range(6):
        for sgn in (-1, 1):
            _face_box(tmb, k, 0.0, sgn * a_, 2 * a_ + tw, tw, 0.04, out)
            _face_box(tmb, k, sgn * a_, 0.0, tw, 2 * a_ + tw, 0.04, out)
    trim = tmb.obj("CubeTrim", trim_material(ng), parent=fy, coll=coll)
    for o in (strips, trim):
        o.visible_shadow = False
    R.log(f"[cube] edge {EDGE} m, bottom ≥ {ZC - HS + T['z_off'].min() * 1:.2f} m over the pan, "
          f"{len(T['slams'])} kick slams, {len(T['flips'])} snare flips; faces {FC.FACE_NAMES}")
    return dict(root=root, yaw=yaw, fx=fx, fy=fy, frame=frame, panels=panels, strips=strips, core=back, trim=trim)


# ------------------------------------------------------------------------------------------------ GN helpers

def _rand(nb, kind, lo, hi, id_, seed):
    nd = nb.n("FunctionNodeRandomValue", data_type=kind)
    typ = {"FLOAT": "VALUE", "FLOAT_VECTOR": "VECTOR", "INT": "INT", "BOOLEAN": "BOOLEAN"}[kind]
    ins = [s for s in nd.inputs if s.enabled] if hasattr(nd.inputs[0], "enabled") else list(nd.inputs)
    mn = [s for s in nd.inputs if s.name == "Min" and s.type == typ][0]
    mxs = [s for s in nd.inputs if s.name == "Max" and s.type == typ][0]
    nb.set(mn, lo)
    nb.set(mxs, hi)
    nb.set(nd.inputs["ID"], id_)
    nb.set(nd.inputs["Seed"], seed)
    del ins
    return [s for s in nd.outputs if s.type == typ][0]


def _noise4(nb, vec, w, scale, detail=2.0):
    nz = nb.n("ShaderNodeTexNoise", noise_dimensions="4D")
    nb.set(nz.inputs["Vector"], vec)
    nb.set(nz.inputs["W"], w)
    nb.set(nz.inputs["Scale"], scale)
    nb.set(nz.inputs["Detail"], detail)
    return nb.v("SUBTRACT", nz.outputs["Color"], (0.5, 0.5, 0.5))


def _cube_proj(nb, v, s):
    x, y, z = nb.sep(v)
    mxa = nb.m("MAXIMUM", nb.m("MAXIMUM", nb.m("ABSOLUTE", x), nb.m("ABSOLUTE", y)), nb.m("ABSOLUTE", z))
    return nb.v("SCALE", v, nb.m("DIVIDE", s, nb.m("MAXIMUM", mxa, 1e-4)))


def _tube(nb, curve, radius_field, mat, scale):
    """curve → mesh (4-sided tube scaled by radius × scale) with material"""
    sr = nb.n("GeometryNodeSetCurveRadius")
    nb.set(sr.inputs["Curve"], curve)
    nb.set(sr.inputs["Radius"], radius_field)
    prof = nb.n("GeometryNodeCurvePrimitiveCircle")
    nb.set(prof.inputs["Resolution"], 5)
    nb.set(prof.inputs["Radius"], float(scale))
    c2m = nb.n("GeometryNodeCurveToMesh")
    nb.link(sr.outputs[0], c2m.inputs["Curve"])
    nb.link(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
    if "Scale" in c2m.inputs:                                     # 5.x: profile scale is a field input
        rad = nb.n("GeometryNodeInputRadius")
        nb.link(rad.outputs[0], c2m.inputs["Scale"])
    sm = nb.n("GeometryNodeSetMaterial")
    nb.link(c2m.outputs["Mesh"], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = mat
    return sm.outputs[0]


def _socket_id(ng, name):
    return [it for it in ng.interface.items_tree if getattr(it, "name", None) == name and it.in_out == "INPUT"][0].identifier


def key_mod_input(ob, mod, ng, name, vals, interp="LINEAR", index=0):
    """5.2 removed mod["Socket_N"] (geometry-nodes-python skill): the input is RNA at .properties.inputs.<id>.value"""
    ident = _socket_id(ng, name)
    path = f'modifiers["{mod.name}"].properties.inputs.{ident}.value'
    ob.path_resolve(path)                                     # raises on a wrong path instead of keying nothing
    return bake(ob, path, vals, index=index, interp=interp)


def _frame_int(nb):
    st = nb.n("GeometryNodeInputSceneTime")
    return nb.m("FLOOR", st.outputs["Frame"])


# ------------------------------------------------------------------------------------------------ arcs (ride the cube)

def arcs(coll, parent, T, core, halo):
    """N_ARCS polylines crawling over / streaming off the cube surface; each slot re-rolls every 2–4 whole frames
    (constant inside the START shutter → no smear), visible when its random draw < Energy (keyed)."""
    ng, nb, gi, go = X.gn_tree("ArcsGN", inputs=[("Energy", "NodeSocketFloat")], with_geo_in=False)
    M = ARC_PTS
    idx = nb.n("GeometryNodeInputIndex").outputs[0]
    k = nb.m("FLOOR", nb.m("DIVIDE", idx, float(M)))
    j = nb.m("SUBTRACT", idx, nb.m("MULTIPLY", k, float(M)))
    t = nb.m("DIVIDE", j, float(M - 1))
    fr = _frame_int(nb)
    hold = nb.m("ADD", 2.0, nb.m("FLOORED_MODULO", k, 3.0))
    e = nb.m("FLOOR", nb.m("DIVIDE", nb.m("ADD", fr, nb.m("MULTIPLY", k, 7.0)), hold))
    rid = nb.m("ADD", nb.m("MULTIPLY", k, 7919.0), e)
    va = _rand(nb, "FLOAT_VECTOR", (-1, -1, -1), (1, 1, 1), rid, 1)
    vb = _rand(nb, "FLOAT_VECTOR", (-1, -1, -1), (1, 1, 1), rid, 2)
    pv = _rand(nb, "FLOAT", 0.0, 1.0, rid, 3)
    bul = _rand(nb, "FLOAT", 0.35, 1.9, rid, 4)
    typ = _rand(nb, "FLOAT", 0.0, 1.0, rid, 5)
    slen = _rand(nb, "FLOAT", 2.0, 6.5, rid, 6)
    A = _cube_proj(nb, va, HS * 1.005)
    Bs = _cube_proj(nb, nb.v("ADD", va, nb.v("SCALE", vb, 0.75)), HS * 1.005)
    outA = nb.v("NORMALIZE", A)
    Bst = nb.v("ADD", nb.v("ADD", A, nb.v("SCALE", outA, slen)), nb.v("SCALE", vb, 1.6))
    stream = nb.m("LESS_THAN", typ, 0.32)
    B = nb.mix(stream, Bs, Bst, dtype="VECTOR")
    sinpt = nb.m("SINE", nb.m("MULTIPLY", t, math.pi))
    outAB = nb.v("NORMALIZE", nb.v("ADD", A, B))
    bulge = nb.m("MULTIPLY", bul, nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", stream, 0.8)))
    base = nb.v("ADD", nb.mix(t, A, B, dtype="VECTOR"), nb.v("SCALE", outAB, nb.m("MULTIPLY", bulge, sinpt)))
    w = nb.m("ADD", nb.m("MULTIPLY", e, 0.731), nb.m("MULTIPLY", k, 3.17))
    # jitter envelope: 0 at the surface end, open toward the free end of a streamer
    env = nb.m("ADD", nb.m("MULTIPLY", nb.m("POWER", sinpt, 0.6), nb.m("SUBTRACT", 1.0, stream)),
               nb.m("MULTIPLY", nb.m("POWER", t, 0.7), stream))
    jit = nb.v("ADD", nb.v("SCALE", _noise4(nb, base, w, 0.9, 2.0), 1.1), nb.v("SCALE", _noise4(nb, base, nb.m("ADD", w, 11.0), 4.0, 1.0), 0.35))
    pos = nb.v("ADD", base, nb.v("SCALE", jit, env))
    pts = nb.n("GeometryNodePoints")
    nb.set(pts.inputs["Count"], N_ARCS * M)
    nb.link(pos, pts.inputs["Position"])
    p2c = nb.n("GeometryNodePointsToCurves")
    nb.link(pts.outputs[0], p2c.inputs["Points"])
    nb.link(k, p2c.inputs["Curve Group ID"])
    nb.link(j, p2c.inputs["Weight"])
    # radius: visible when the slot's draw < Energy; streamers taper
    # (fields inside the curve domain: recompute index-based values there — curve points keep the point order)
    idx2 = nb.n("GeometryNodeInputIndex").outputs[0]
    k2 = nb.m("FLOOR", nb.m("DIVIDE", idx2, float(M)))
    t2 = nb.m("DIVIDE", nb.m("SUBTRACT", idx2, nb.m("MULTIPLY", k2, float(M))), float(M - 1))
    hold2 = nb.m("ADD", 2.0, nb.m("FLOORED_MODULO", k2, 3.0))
    e2 = nb.m("FLOOR", nb.m("DIVIDE", nb.m("ADD", fr, nb.m("MULTIPLY", k2, 7.0)), hold2))
    rid2 = nb.m("ADD", nb.m("MULTIPLY", k2, 7919.0), e2)
    pv2 = _rand(nb, "FLOAT", 0.0, 1.0, rid2, 3)
    typ2 = _rand(nb, "FLOAT", 0.0, 1.0, rid2, 5)
    vis = nb.m("LESS_THAN", pv2, nb.m("MULTIPLY", gi.outputs["Energy"], 0.9))
    taper = nb.m("SUBTRACT", 1.0, nb.m("MULTIPLY", nb.m("LESS_THAN", typ2, 0.32), nb.m("MULTIPLY", t2, 0.75)))
    rad = nb.m("MULTIPLY", nb.m("MULTIPLY", vis, taper), 1.0)
    del pv, typ
    c_core = _tube(nb, p2c.outputs[0], rad, core, 0.028)
    c_halo = _tube(nb, p2c.outputs[0], rad, halo, 0.22)
    j_ = nb.n("GeometryNodeJoinGeometry")
    nb.link(c_core, j_.inputs[0])
    nb.link(c_halo, j_.inputs[0])
    nb.link(j_.outputs[0], go.inputs["Geometry"])
    me = bpy.data.meshes.new("Arcs")
    ob = bpy.data.objects.new("Arcs", me)
    coll.objects.link(ob)
    ob.parent = parent
    mod = X.add_gn(ob, ng)
    key_mod_input(ob, mod, ng, "Energy", T["energy"])
    ob.visible_shadow = False
    return ob


# ------------------------------------------------------------------------------------------------ ground strikes + spray

def strikes(coll, root, tracks, core, halo, sand_grain_mat, dust_mat, n):
    """world-space bolts cube-bottom → ground point, sand grains thrown out of the impact (analytic ballistics with
    drag, clamped age — gotcha 10/13), and a dust puff. One GN object; per-slot keyed inputs End_k/Age_k/Seed_k."""
    S = len(tracks)
    ins = []
    for s in range(S):
        ins += [(f"End{s}", "NodeSocketVector"), (f"Age{s}", "NodeSocketFloat"), (f"Seed{s}", "NodeSocketFloat")]
    ng, nb, gi, go = X.gn_tree("StrikesGN", inputs=ins, with_geo_in=False)
    oi = nb.n("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    oi.inputs["Object"].default_value = root
    start = nb.v("ADD", oi.outputs["Location"], (0.0, 0.0, -HS * 0.9))
    parts = []
    # --- bolts: STRIKE_PTS per slot
    M = STRIKE_PTS
    idx = nb.n("GeometryNodeInputIndex").outputs[0]
    k = nb.m("FLOOR", nb.m("DIVIDE", idx, float(M)))
    t = nb.m("DIVIDE", nb.m("SUBTRACT", idx, nb.m("MULTIPLY", k, float(M))), float(M - 1))

    def per_slot(field_k, name, kind="VALUE"):
        """select gi[name+s] by slot index field_k"""
        acc = None
        for s in range(S):
            v = gi.outputs[f"{name}{s}"]
            sel = nb.m("COMPARE", field_k, float(s), 0.1)
            if kind == "VECTOR":
                term = nb.v("SCALE", v, sel)
                acc = term if acc is None else nb.v("ADD", acc, term)
            else:
                term = nb.m("MULTIPLY", v, sel)
                acc = term if acc is None else nb.m("ADD", acc, term)
        return acc

    end = per_slot(k, "End", "VECTOR")
    age = per_slot(k, "Age")
    seed = per_slot(k, "Seed")
    afr = nb.m("FLOOR", nb.m("ADD", nb.m("MULTIPLY", age, 30.0), 0.01))
    on = nb.m("MULTIPLY", nb.m("GREATER_THAN", age, -0.001), nb.m("LESS_THAN", afr, 5.5))
    on = nb.m("MULTIPLY", on, nb.m("SUBTRACT", 1.0, nb.m("COMPARE", afr, 2.0, 0.1)))
    on = nb.m("MULTIPLY", on, nb.m("SUBTRACT", 1.0, nb.m("COMPARE", afr, 4.0, 0.1)))
    sinpt = nb.m("SINE", nb.m("MULTIPLY", t, math.pi))
    base = nb.mix(t, start, end, dtype="VECTOR")
    w = nb.m("ADD", nb.m("MULTIPLY", seed, 0.013), nb.m("MULTIPLY", afr, 0.37))
    jit = nb.v("ADD", nb.v("SCALE", _noise4(nb, base, w, 0.22, 2.0), 7.0), nb.v("SCALE", _noise4(nb, base, nb.m("ADD", w, 5.0), 1.3, 1.0), 1.2))
    jit = nb.v("MULTIPLY", jit, (1.0, 1.0, 0.25))
    pos = nb.v("ADD", base, nb.v("SCALE", jit, nb.m("POWER", sinpt, 0.8)))
    pts = nb.n("GeometryNodePoints")
    nb.set(pts.inputs["Count"], S * M)
    nb.link(pos, pts.inputs["Position"])
    p2c = nb.n("GeometryNodePointsToCurves")
    nb.link(pts.outputs[0], p2c.inputs["Points"])
    nb.link(k, p2c.inputs["Curve Group ID"])
    idx2 = nb.n("GeometryNodeInputIndex").outputs[0]
    k2 = nb.m("FLOOR", nb.m("DIVIDE", idx2, float(M)))
    age2 = per_slot(k2, "Age")
    afr2 = nb.m("FLOOR", nb.m("ADD", nb.m("MULTIPLY", age2, 30.0), 0.01))
    on2 = nb.m("MULTIPLY", nb.m("GREATER_THAN", age2, -0.001), nb.m("LESS_THAN", afr2, 5.5))
    on2 = nb.m("MULTIPLY", on2, nb.m("SUBTRACT", 1.0, nb.m("COMPARE", afr2, 2.0, 0.1)))
    on2 = nb.m("MULTIPLY", on2, nb.m("SUBTRACT", 1.0, nb.m("COMPARE", afr2, 4.0, 0.1)))
    del on
    parts.append(_tube(nb, p2c.outputs[0], on2, core, 0.07))
    parts.append(_tube(nb, p2c.outputs[0], on2, halo, 0.45))
    # --- sand grains: GRAINS per slot
    G = GRAINS
    gidx = nb.n("GeometryNodeInputIndex").outputs[0]
    gk = nb.m("FLOOR", nb.m("DIVIDE", gidx, float(G)))
    gend = per_slot(gk, "End", "VECTOR")
    gage = per_slot(gk, "Age")
    gseed = per_slot(gk, "Seed")
    gid = nb.m("ADD", gidx, nb.m("MULTIPLY", gseed, 13.0))
    az = _rand(nb, "FLOAT", 0.0, 2 * math.pi, gid, 21)
    el = _rand(nb, "FLOAT", math.radians(30), math.radians(82), gid, 22)
    sp = nb.m("POWER", _rand(nb, "FLOAT", 0.0, 1.0, gid, 23), 1.8)
    sp = nb.m("ADD", 2.0, nb.m("MULTIPLY", sp, 11.0))
    life = _rand(nb, "FLOAT", 0.9, 2.2, gid, 24)
    size = _rand(nb, "FLOAT", 0.012, 0.045, gid, 25)
    a = nb.m("MINIMUM", nb.m("MAXIMUM", gage, 0.0), 2.4)                 # clamped age (gotcha 10)
    kd = 1.4
    ex = nb.m("DIVIDE", nb.m("SUBTRACT", 1.0, nb.m("EXPONENT", nb.m("MULTIPLY", a, -kd))), kd)
    vh = nb.m("MULTIPLY", sp, nb.m("COSINE", el))
    vz = nb.m("MULTIPLY", sp, nb.m("SINE", el))
    dx = nb.m("MULTIPLY", nb.m("MULTIPLY", vh, nb.m("COSINE", az)), ex)
    dy = nb.m("MULTIPLY", nb.m("MULTIPLY", vh, nb.m("SINE", az)), ex)
    g = 9.81
    dz = nb.m("SUBTRACT", nb.m("MULTIPLY", vz, ex), nb.m("MULTIPLY", g / kd, nb.m("SUBTRACT", a, ex)))
    dz = nb.m("MAXIMUM", dz, 0.0)
    gpos = nb.v("ADD", gend, nb.xyz(dx, dy, dz))
    alive = nb.m("MULTIPLY", nb.m("GREATER_THAN", gage, 0.0), nb.m("LESS_THAN", gage, life))
    fade = nb.m("SUBTRACT", 1.0, nb.m("POWER", nb.m("DIVIDE", a, life), 3.0), clamp=True)
    gscale = nb.m("MULTIPLY", nb.m("MULTIPLY", alive, fade), size)
    gpts = nb.n("GeometryNodePoints")
    nb.set(gpts.inputs["Count"], S * G)
    nb.link(gpos, gpts.inputs["Position"])
    ico = nb.n("GeometryNodeMeshIcoSphere")
    nb.set(ico.inputs["Radius"], 1.0)
    nb.set(ico.inputs["Subdivisions"], 1)
    smg = nb.n("GeometryNodeSetMaterial")
    nb.link(ico.outputs["Mesh"], smg.inputs["Geometry"])
    smg.inputs["Material"].default_value = sand_grain_mat                 # on the TEMPLATE (gotcha 4)
    iop = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(gpts.outputs[0], iop.inputs["Points"])
    nb.link(smg.outputs[0], iop.inputs["Instance"])
    nb.link(gscale, iop.inputs["Scale"])
    parts.append(iop.outputs[0])
    # --- dust puffs: DUST per slot, expanding + drifting downwind (+X), volume density fades
    Dn = DUST
    didx = nb.n("GeometryNodeInputIndex").outputs[0]
    dk = nb.m("FLOOR", nb.m("DIVIDE", didx, float(Dn)))
    dend = per_slot(dk, "End", "VECTOR")
    dage = per_slot(dk, "Age")
    dseed = per_slot(dk, "Seed")
    did = nb.m("ADD", didx, nb.m("MULTIPLY", dseed, 17.0))
    da = nb.m("MINIMUM", nb.m("MAXIMUM", dage, 0.0), 2.4)
    off = _rand(nb, "FLOAT_VECTOR", (-1.0, -1.0, 0.0), (1.0, 1.0, 1.0), did, 31)
    grow = nb.m("ADD", 0.25, nb.m("MULTIPLY", 2.2, nb.m("POWER", nb.m("DIVIDE", da, 2.4), 0.5)))
    dpos = nb.v("ADD", dend, nb.v("ADD", nb.v("SCALE", off, nb.m("MULTIPLY", grow, 1.3)), nb.xyz(nb.m("MULTIPLY", da, 0.8), 0.0, nb.m("MULTIPLY", da, 0.6))))
    dalive = nb.m("MULTIPLY", nb.m("GREATER_THAN", dage, 0.0), nb.m("LESS_THAN", dage, 2.35))
    dsz = nb.m("MULTIPLY", dalive, nb.m("MULTIPLY", grow, _rand(nb, "FLOAT", 0.7, 1.3, did, 32)))
    dpts = nb.n("GeometryNodePoints")
    nb.set(dpts.inputs["Count"], S * Dn)
    nb.link(dpos, dpts.inputs["Position"])
    dfade = nb.m("POWER", nb.m("SUBTRACT", 1.0, nb.m("DIVIDE", da, 2.4), clamp=True), 1.6)
    dst = nb.store(dpts.outputs[0], "dust", dfade, "FLOAT", "POINT")
    dico = nb.n("GeometryNodeMeshIcoSphere")
    nb.set(dico.inputs["Radius"], 1.0)
    nb.set(dico.inputs["Subdivisions"], 2)
    smd = nb.n("GeometryNodeSetMaterial")
    nb.link(dico.outputs["Mesh"], smd.inputs["Geometry"])
    smd.inputs["Material"].default_value = dust_mat
    diop = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(dst, diop.inputs["Points"])
    nb.link(smd.outputs[0], diop.inputs["Instance"])
    nb.link(dsz, diop.inputs["Scale"])
    parts.append(diop.outputs[0])
    jn = nb.n("GeometryNodeJoinGeometry")
    for p in reversed(parts):
        nb.link(p, jn.inputs[0])
    nb.link(jn.outputs[0], go.inputs["Geometry"])
    me = bpy.data.meshes.new("Strikes")
    ob = bpy.data.objects.new("Strikes", me)
    coll.objects.link(ob)
    mod = X.add_gn(ob, ng)
    for s, tr in enumerate(tracks):
        for i in range(3):
            key_mod_input(ob, mod, ng, f"End{s}", tr["end"][:, i], interp="CONSTANT", index=i)
        key_mod_input(ob, mod, ng, f"Seed{s}", tr["seed"], interp="CONSTANT")
        key_mod_input(ob, mod, ng, f"Age{s}", tr["age"], interp="LINEAR")
    return ob


def grain_material():
    return X.simple_mat("SandGrain", (0.55, 0.34, 0.17), 0.85)


def dust_material():
    m, nb, out = X.new_mat("StrikeDust")
    at = nb.sattr("dust", "INSTANCER").outputs["Fac"]
    # soft edge: density falls off toward the sphere surface (object coords of the unit ico)
    tc = nb.n("ShaderNodeTexCoord")
    r = nb.v("LENGTH", tc.outputs["Object"])
    soft = nb.mr(r, 0.4, 1.0, 1.0, 0.0)
    nz = nb.n("ShaderNodeTexNoise")
    nb.set(nz.inputs["Scale"], 2.5)
    nb.set(nz.inputs["Detail"], 4.0)
    nb.link(tc.outputs["Object"], nz.inputs["Vector"])
    dens = nb.m("MULTIPLY", nb.m("MULTIPLY", at, soft), nb.m("MULTIPLY", nb.mr(nz.outputs["Fac"], 0.35, 0.7), 1.6))
    v = nb.n("ShaderNodeVolumePrincipled")
    nb.set(v.inputs["Color"], (0.62, 0.42, 0.24))
    nb.link(dens, v.inputs["Density"])
    nb.set(v.inputs["Anisotropy"], 0.35)
    nb.link(v.outputs[0], out.inputs["Volume"])
    return m


# ------------------------------------------------------------------------------------------------ lights

def lights(coll, parts, T, tracks):
    n = T["n"]
    lum = (T["A"] + T["B"]) / 2
    col = lum / np.maximum(lum.max(1, keepdims=True), 1e-4)
    under = X.light("CubeUnderGlow", "AREA", (0, 0, ZC - HS - 0.3), energy=0.0, size=EDGE * 0.9, coll=coll)
    under.rotation_euler = (0, 0, 0)                    # area lights point down −Z
    under.parent = parts["root"]
    under.location = (0, 0, -HS - 0.3)
    bake_vec(under.data, "color", col)
    bake(under.data, "energy", 9000.0 * T["gain"] * (0.4 + 0.6 * T["hot"]))
    arc = X.light("ArcLight", "POINT", (0, 0, 0), energy=0.0, color=(0.55, 0.66, 1.0), size=2.5, coll=coll,
                  shadow=False)                       # it sits INSIDE the cube: a shadow would black it out
    arc.parent = parts["root"]
    rng = np.random.default_rng(3)
    flick = 0.45 + 0.55 * rng.random(n)
    bake(arc.data, "energy", 120000.0 * T["energy"] * flick)
    out = [under, arc]
    for s, tr in enumerate(tracks):
        L = X.light(f"StrikeFlash{s}", "POINT", (0, 0, -50), energy=0.0, color=(0.6, 0.7, 1.0), size=1.0, coll=coll,
                    shadow=False)
        bake_vec(L, "location", tr["end"] + np.array([0, 0, 1.2]), interp="CONSTANT")
        bake(L.data, "energy", 400000.0 * tr["flash"])
        out.append(L)
    return out
