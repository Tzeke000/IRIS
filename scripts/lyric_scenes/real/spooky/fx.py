"""Petals + fireflies: analytic GN on Scene Time. Constant point counts (gotcha 10: EEVEE motion blur pairs instances
by index), positions clamped/wrapped with a fade at the wrap so a teleport is invisible and never smears.
v2: positions run on the music CLOCKS in CTRL_GN (wind_t for petals, ff_t for fireflies — faster in the drops, a gust
on every big hit) instead of Scene Time. Density is a SOFT shader gate per instance (smoothstep of CTRL.petal_density /
CTRL.ff_density − the instance's hash), so a rising density fades petals in instead of popping them, and the instance
count never changes."""
import math

import numpy as np

import common as C
from common import NB, X


def _wrap_axis(nb, v, lo, size):
    """lo + floored_mod(v − lo, size) and the 0..1 coordinate inside the box"""
    f = nb.m("FRACT", nb.m("DIVIDE", nb.m("SUBTRACT", v, lo), size))
    return nb.m("ADD", lo, nb.m("MULTIPLY", f, size)), f


def _edge_fade(nb, f, w=0.06):
    return nb.m("MULTIPLY", nb.mr(f, 0.0, w), nb.mr(f, 1.0, 1.0 - w))


def petal_material():
    m, nb, out = X.new_mat("Petal")
    K = C.ctrl(nb)
    pid = nb.sattr("pid", "INSTANCER").outputs["Fac"]
    pg = nb.sattr("pglow", "INSTANCER").outputs["Fac"]
    on = nb.mr(nb.m("SUBTRACT", K["petal_density"], pid), 0.0, 0.08, 0.0, 1.0, interp="SMOOTHSTEP")    # soft gate
    u, v, _ = nb.sep(nb.sattr("puv").outputs["Vector"])
    du, dv = nb.m("SUBTRACT", u, 0.5), nb.m("SUBTRACT", v, 0.5)
    ov = nb.mr(nb.m("ADD", nb.m("MULTIPLY", du, du), nb.m("MULTIPLY", dv, dv)), 0.25, 0.12)   # soft oval
    col = nb.mix(ov, (1.0, 0.45, 0.85), C.PETAL_PINK)
    st = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("ADD", 1.2, nb.m("MULTIPLY", pg, 2.4)), nb.m("MULTIPLY", K["glow"], K["petal_glow"])), ov)
    a = nb.m("MULTIPLY", ov, on)
    sh = C.mix_shader(nb, a, C.transparent(nb), C.emission(nb, col, st))
    nb.link(sh, out.inputs["Surface"])
    C.blended(m)
    return m


def firefly_material():
    m, nb, out = X.new_mat("Firefly")
    K = C.ctrl(nb)
    fph = nb.sattr("fph", "INSTANCER").outputs["Fac"]
    frt = nb.sattr("frt", "INSTANCER").outputs["Fac"]
    fl = nb.m("ADD", 0.25, nb.m("MULTIPLY", 0.75, nb.m("POWER", nb.m("ADD", 0.5, nb.m("MULTIPLY", 0.5, nb.m("SINE", nb.m("ADD", nb.m("MULTIPLY", K["T"], frt), fph)))), 2.0)))
    fid = nb.sattr("fid", "INSTANCER").outputs["Fac"]
    on = nb.mr(nb.m("SUBTRACT", K["ff_density"], fid), 0.0, 0.1, 0.0, 1.0, interp="SMOOTHSTEP")
    st = nb.m("MULTIPLY", nb.m("MULTIPLY", fl, K["firefly"]), nb.m("MULTIPLY", K["glow"], 7.0))
    sh = C.mix_shader(nb, on, C.transparent(nb), C.emission(nb, (1.0, 0.42, 0.85), st))
    nb.link(sh, out.inputs["Surface"])
    C.blended(m)
    return m


def petals(coll, n=500, seed=5):
    rng = np.random.default_rng(seed)
    lo = np.array([-13.0, -10.5, 0.0])                                # v2: near the camera (soft, big), fewer far dots
    size = np.array([30.0, 34.0, 9.0])
    y = lo[1] + size[1] * rng.random(n) ** 1.6                       # denser near the camera
    base = np.stack([lo[0] + size[0] * rng.random(n), y, lo[2] + size[2] * rng.random(n)], 1)
    attrs = {"pid": rng.random(n), "pglow": rng.random(n),
             "rot0": rng.uniform(0, 6.28, (n, 3)), "rrate": rng.uniform(-2.2, 2.2, (n, 3)),
             "wob": rng.uniform(0, 6.28, n), "psz": rng.uniform(0.6, 1.4, n)}
    pts = X.points_obj("Petals", base, attrs=attrs, coll=coll)
    m = petal_material()
    ng, nb, gi, go = X.gn_tree("PetalsGN")
    T = C.ctrl_gn(nb)["wind_t"]                                           # music clock (gusts on the drops)
    P = nb.n("GeometryNodeInputPosition").outputs[0]
    wind = (0.55, 0.18, -0.28)
    px, py, pz = nb.sep(nb.v("ADD", P, nb.v("SCALE", wind, T)))
    wob = nb.attr("wob")
    px = nb.m("ADD", px, nb.m("MULTIPLY", 0.35, nb.m("SINE", nb.m("ADD", nb.m("MULTIPLY", T, 1.1), wob))))
    pz = nb.m("ADD", pz, nb.m("MULTIPLY", 0.25, nb.m("SINE", nb.m("ADD", nb.m("MULTIPLY", T, 1.7), nb.m("MULTIPLY", wob, 2.0)))))
    (wx, fx), (wy, fy), (wz, fz) = [_wrap_axis(nb, a, float(lo[i]), float(size[i])) for i, a in enumerate((px, py, pz))]
    fade = nb.m("MULTIPLY", nb.m("MULTIPLY", _edge_fade(nb, fx), _edge_fade(nb, fy)), _edge_fade(nb, fz))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(gi.outputs[0], sp.inputs["Geometry"])
    nb.link(nb.xyz(wx, wy, wz), sp.inputs["Position"])
    # template: a cupped 1 × 0.6 grid
    grid = nb.n("GeometryNodeMeshGrid")
    for k, v in (("Size X", 1.0), ("Size Y", 0.6), ("Vertices X", 5), ("Vertices Y", 3)):
        nb.set(grid.inputs[k], v)
    gx, gy, _ = nb.sep(nb.n("GeometryNodeInputPosition").outputs[0])
    cup = nb.n("GeometryNodeSetPosition")
    nb.link(nb.store(grid.outputs["Mesh"], "puv", grid.outputs["UV Map"], dtype="FLOAT_VECTOR", domain="CORNER"),
            cup.inputs["Geometry"])
    nb.link(nb.xyz(0, 0, nb.m("MULTIPLY", nb.m("MULTIPLY", gx, gx), 0.3)), cup.inputs["Offset"])
    smt = nb.n("GeometryNodeSetMaterial")
    nb.link(cup.outputs[0], smt.inputs["Geometry"])
    smt.inputs["Material"].default_value = m
    eul = nb.v("ADD", nb.attr("rot0", "FLOAT_VECTOR"), nb.v("SCALE", nb.attr("rrate", "FLOAT_VECTOR"), T))
    er = nb.n("FunctionNodeEulerToRotation")
    nb.link(eul, er.inputs["Euler"])
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(sp.outputs[0], inst.inputs["Points"])
    nb.link(smt.outputs[0], inst.inputs["Instance"])
    nb.link(er.outputs[0], inst.inputs["Rotation"])
    nb.link(nb.m("MULTIPLY", nb.m("MULTIPLY", nb.attr("psz"), 0.17), fade), inst.inputs["Scale"])
    nb.link(inst.outputs[0], go.inputs[0])
    X.add_gn(pts, ng)
    C.hide_shadow(pts)
    return pts


def fireflies(coll, n=200, seed=9):
    rng = np.random.default_rng(seed)
    lo = np.array([-16.0, -6.0, 0.25])
    size = np.array([40.0, 40.0, 6.0])
    base = lo + size * rng.random((n, 3))
    base[:, 2] = lo[2] + size[2] * rng.random(n) ** 1.8               # most hover low over the water
    attrs = {"fph": rng.uniform(0, 6.28, n), "frt": rng.uniform(1.5, 5.0, n), "fid": rng.random(n),
             "wf": rng.uniform(0.15, 0.45, (n, 3)), "wp": rng.uniform(0, 6.28, (n, 3)), "fsz": rng.uniform(0.6, 1.3, n)}
    pts = X.points_obj("Fireflies", base, attrs=attrs, coll=coll)
    m = firefly_material()
    ng, nb, gi, go = X.gn_tree("FirefliesGN")
    T = C.ctrl_gn(nb)["ff_t"]
    P = nb.n("GeometryNodeInputPosition").outputs[0]
    wf, wp = nb.attr("wf", "FLOAT_VECTOR"), nb.attr("wp", "FLOAT_VECTOR")
    ang = nb.v("ADD", nb.v("SCALE", wf, T), wp)
    sx, sy, sz = nb.sep(ang)
    wander = nb.xyz(nb.m("MULTIPLY", 0.9, nb.m("SINE", sx)), nb.m("MULTIPLY", 0.9, nb.m("SINE", sy)),
                    nb.m("MULTIPLY", 0.35, nb.m("SINE", sz)))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(gi.outputs[0], sp.inputs["Geometry"])
    nb.link(nb.v("ADD", P, wander), sp.inputs["Position"])
    ico = nb.n("GeometryNodeMeshIcoSphere")
    nb.set(ico.inputs["Radius"], 1.0)
    nb.set(ico.inputs["Subdivisions"], 1)
    smt = nb.n("GeometryNodeSetMaterial")
    nb.link(ico.outputs[0], smt.inputs["Geometry"])
    smt.inputs["Material"].default_value = m
    inst = nb.n("GeometryNodeInstanceOnPoints")
    nb.link(sp.outputs[0], inst.inputs["Points"])
    nb.link(smt.outputs[0], inst.inputs["Instance"])
    nb.link(nb.m("MULTIPLY", nb.attr("fsz"), 0.04), inst.inputs["Scale"])
    nb.link(inst.outputs[0], go.inputs[0])
    X.add_gn(pts, ng)
    C.hide_shadow(pts)
    return pts
