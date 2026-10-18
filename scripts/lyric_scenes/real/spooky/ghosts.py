"""Cute sheet ghosts — procedural mesh + ANALYTIC Geometry-Nodes motion (no cloth, no caches, no handlers).

Why analytic GN (see profiles/blender/spooky_v2_research.md §2): every frame is a pure function of Scene Time, so 60-frame
render chunks in separate processes agree exactly; cloth needs a per-chunk point cache and drifts between chunks.

Mesh (local units, height 1, face toward −Y, built per ghost so its trail direction is baked into the rest shape):
  body  : lathe of a dome→skirt profile with a scalloped hem. attrs h (0 hem → 1 crown), part 0, rest (eyes in the shader)
  wisps : K ribbons from the back of the hem, streaming along `trail`, tapering. attrs u (0 root → 1 tip), part 1,
          side (−1..1 across), wk (ribbon index), perp (wave direction)
Motion (GhostMotion tree, per-ghost modifier inputs Phase / Bob / Sway / Speed / Trail / Drag / Wisp):
  bob        dz = Bob·sin(ωt + φ − 1.2·lag)             lag = 1 − h → the crown leads, the hem follows (follow-through,
                                                         and squash/stretch falls out of the phase difference)
  sway       lateral, slower, amplitude grows toward the hem
  hem ripple radial flutter ∝ lag³·sin(6θ − 2.2ωt)
  drag       back of the skirt stretches along Trail ∝ lag²·(back-facing)
  wisps      travelling waves along u (tip lags root) + they inherit the hem's bob delayed by 2.4·u rad: the tail follows
             where the body WAS (p(t − τ·u)) — the "flowing trail" without any history buffer
"""
import math

import numpy as np

import common as C
from common import NB, X

PROFILE = [(0.52, 0.0), (0.47, 0.08), (0.42, 0.2), (0.385, 0.35), (0.37, 0.5), (0.365, 0.6), (0.355, 0.68),
           (0.33, 0.77), (0.29, 0.85), (0.22, 0.92), (0.12, 0.975), (0.0, 1.0)]


def _resample(prof, n):
    p = np.asarray(prof, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, seg[-1], n)
    return np.stack([np.interp(t, seg, p[:, 0]), np.interp(t, seg, p[:, 1])], 1)


def ghost_mesh(name, trail=(0.0, 1.0, 0.1), n_wisps=5, seed=0, nt=64, nz=34, wisp_len=1.0, wisp_w=1.0):
    """→ (verts, faces, face_material_index, point attrs dict)"""
    rng = np.random.default_rng(seed)
    prof = _resample(PROFILE, nz)
    V, F, MI = [], [], []
    A = {k: [] for k in ("h", "u", "part", "side", "wk")}
    rest, perp = [], []
    tr = np.asarray(trail, float)
    tr = tr / np.linalg.norm(tr)
    trxy = np.array([tr[0], tr[1], 0.0])
    trxy = trxy / max(np.linalg.norm(trxy), 1e-6)
    for j, (r, z) in enumerate(prof[:-1]):
        for i in range(nt):
            th = 2 * math.pi * i / nt
            zz = z
            if z < 0.14:                                                   # scalloped hem
                zz = z + 0.055 * (1 - z / 0.14) * math.cos(5 * th + 0.4)
            rr = r * (1 + 0.04 * (1 - z) ** 3 * math.cos(5 * th + 0.4))
            rr *= 1 + 0.05 * max(0.0, 0.7 - z) ** 1.3 * math.sin(9 * th + 0.8 * math.sin(3 * th))   # sheet folds
            x, y = rr * math.sin(th), -rr * math.cos(th)                   # th = 0 → front (−Y)
            V.append((x, y, zz))
            for k, v in (("h", z), ("u", 0.0), ("part", 0.0), ("side", 0.0), ("wk", 0.0)):
                A[k].append(v)
            rest.append((x, y, zz))
            perp.append((0, 0, 1))
    top = len(V)
    V.append((0, 0, 1.0))
    for k, v in (("h", 1.0), ("u", 0.0), ("part", 0.0), ("side", 0.0), ("wk", 0.0)):
        A[k].append(v)
    rest.append((0, 0, 1.0))
    perp.append((0, 0, 1))
    for j in range(nz - 2):
        for i in range(nt):
            a, b = j * nt + i, j * nt + (i + 1) % nt
            F.append((a, b, b + nt, a + nt))
            MI.append(0)
    for i in range(nt):
        F.append(((nz - 2) * nt + i, (nz - 2) * nt + (i + 1) % nt, top))
        MI.append(0)
    # wisps: from the back half of the hem (where the trail points), streaming along the trail
    back_th = math.atan2(trxy[0], -trxy[1])           # angle on the body whose outward normal = trail
    nu = 28
    for k in range(n_wisps):
        th0 = back_th + rng.uniform(-1.3, 1.3)
        z0 = rng.uniform(0.04, 0.32)
        r0 = np.interp(z0, prof[:, 1], prof[:, 0]) * 0.92
        b0 = np.array([r0 * math.sin(th0), -r0 * math.cos(th0), z0])
        out = np.array([math.sin(th0), -math.cos(th0), 0.0])
        d = 0.55 * tr + 0.45 * out
        d[2] += rng.uniform(-0.05, 0.25)
        d /= np.linalg.norm(d)
        L = rng.uniform(0.9, 1.6) * wisp_len
        w0 = rng.uniform(0.07, 0.13) * wisp_w
        side_dir = np.cross(d, [0, 0, 1.0])
        side_dir /= np.linalg.norm(side_dir)
        ang = rng.uniform(0.2, 1.2)
        wdir = side_dir * math.cos(ang) + np.array([0, 0, 1.0]) * math.sin(ang)
        curl_ph = rng.uniform(0, 6.28)
        base = len(V)
        for j in range(nu):
            u = j / (nu - 1)
            c = b0 + d * L * u
            c = c + side_dir * 0.18 * u * math.sin(2.4 * math.pi * u + curl_ph) + np.array([0, 0, 0.10 * math.sin(math.pi * u) * (1 if k % 2 else -0.6)])
            w = w0 * (1 - u) ** 0.8 + 0.008
            for s in (-1.0, 1.0):
                p = c + wdir * w * s
                V.append(tuple(p))
                A["h"].append(max(0.0, z0 * (1 - u)))
                A["u"].append(u)
                A["part"].append(1.0)
                A["side"].append(s)
                A["wk"].append(float(k))
                rest.append(tuple(p))
                perp.append(tuple(side_dir))
        for j in range(nu - 1):
            a = base + 2 * j
            F.append((a, a + 1, a + 3, a + 2))
            MI.append(1)
    A["rest"] = np.asarray(rest, np.float32)
    A["perp"] = np.asarray(perp, np.float32)
    return np.asarray(V), F, MI, A


def motion_tree():
    """GhostMotion: one tree shared by every ghost; per-ghost values are modifier inputs."""
    ng, nb, gi, go = X.gn_tree("GhostMotion", inputs=[("Phase", "NodeSocketFloat"), ("Bob", "NodeSocketFloat"),
                                                      ("Sway", "NodeSocketFloat"), ("Speed", "NodeSocketFloat"),
                                                      ("Trail", "NodeSocketVector"), ("Drag", "NodeSocketFloat"),
                                                      ("Wisp", "NodeSocketFloat")])
    T = nb.n("GeometryNodeInputSceneTime").outputs["Seconds"]
    ph, bob, sway, spd = gi.outputs["Phase"], gi.outputs["Bob"], gi.outputs["Sway"], gi.outputs["Speed"]
    trail, drag, wisp = gi.outputs["Trail"], gi.outputs["Drag"], gi.outputs["Wisp"]
    wt = nb.m("MULTIPLY", nb.m("MULTIPLY", T, spd), 2 * math.pi)            # ω t
    h, u, part = nb.attr("h"), nb.attr("u"), nb.attr("part")
    wk = nb.attr("wk")
    perp = nb.attr("perp", "FLOAT_VECTOR")
    rest = nb.attr("rest", "FLOAT_VECTOR")
    rx, ry, _ = nb.sep(rest)
    lag = nb.m("SUBTRACT", 1.0, h)
    body = nb.m("SUBTRACT", 1.0, part)
    # -- shared: bob with lag (wisps add a further delay along u)
    delay = nb.m("ADD", nb.m("MULTIPLY", lag, 1.2), nb.m("MULTIPLY", u, 2.4))
    dz = nb.m("MULTIPLY", bob, nb.m("SINE", nb.m("SUBTRACT", nb.m("ADD", wt, ph), delay)))
    # -- sway (half speed), grows toward the hem / along the wisp
    grow = nb.m("ADD", 0.3, nb.m("MULTIPLY", nb.m("ADD", lag, u), nb.m("ADD", lag, u)))
    sarg = nb.m("SUBTRACT", nb.m("ADD", nb.m("MULTIPLY", wt, 0.5), nb.m("MULTIPLY", ph, 1.3)), nb.m("MULTIPLY", delay, 1.3))
    dx = nb.m("MULTIPLY", nb.m("MULTIPLY", sway, nb.m("SINE", sarg)), grow)
    dy = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", sway, 0.4), nb.m("COSINE", sarg)), grow)
    # -- body: hem ripple (radial) + drag along the trail on the back of the skirt
    th = nb.m("ARCTAN2", rx, nb.m("MULTIPLY", ry, -1.0))
    radial = nb.v("NORMALIZE", nb.xyz(rx, ry, 0.0))
    lag3 = nb.m("POWER", lag, 3.0)
    rip = nb.m("MULTIPLY", nb.m("MULTIPLY", lag3, 0.035),
               nb.m("SINE", nb.m("ADD", nb.m("SUBTRACT", nb.m("MULTIPLY", th, 6.0), nb.m("MULTIPLY", wt, 2.2)), ph)))
    back = nb.m("MAXIMUM", nb.v("DOT_PRODUCT", radial, nb.v("NORMALIZE", trail)), 0.0)
    flutter = nb.m("ADD", 1.0, nb.m("MULTIPLY", 0.2, nb.m("SINE", nb.m("SUBTRACT", nb.m("MULTIPLY", wt, 0.7), nb.m("MULTIPLY", lag, 3.0)))))
    dr = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", lag, lag), back), nb.m("MULTIPLY", drag, flutter))
    off_body = nb.v("ADD", nb.v("SCALE", radial, rip), nb.v("SCALE", nb.v("NORMALIZE", trail), dr))
    # -- wisps: travelling waves (vertical + sideways), amplitude grows toward the tip
    u13 = nb.m("POWER", u, 1.3)
    a1 = nb.m("SUBTRACT", nb.m("MULTIPLY", u, 2 * math.pi * 1.2), nb.m("MULTIPLY", wt, 1.6))
    w1 = nb.m("MULTIPLY", nb.m("MULTIPLY", wisp, u13), nb.m("SINE", nb.m("ADD", a1, wk)))
    a2 = nb.m("SUBTRACT", nb.m("MULTIPLY", u, 2 * math.pi * 0.9), nb.m("MULTIPLY", wt, 1.3))
    w2 = nb.m("MULTIPLY", nb.m("MULTIPLY", nb.m("MULTIPLY", wisp, 0.6), u13), nb.m("SINE", nb.m("ADD", a2, nb.m("MULTIPLY", wk, 2.0))))
    off_wisp = nb.v("ADD", nb.xyz(0, 0, w1), nb.v("SCALE", perp, w2))
    off = nb.v("ADD", nb.xyz(dx, dy, dz), nb.v("ADD", nb.v("SCALE", off_body, body), nb.v("SCALE", off_wisp, part)))
    sp = nb.n("GeometryNodeSetPosition")
    nb.link(gi.outputs["Geometry"], sp.inputs["Geometry"])
    nb.link(off, sp.inputs["Offset"])
    nb.link(sp.outputs[0], go.inputs[0])
    return ng


def materials():
    # body: soft white sheet, lavender self-glow, purple rim, black oval eyes from the REST position
    m, nb, out = X.new_mat("GhostBody")
    K = C.ctrl(nb)
    rest = nb.sattr("rest").outputs["Vector"]
    rx, ry, rz = nb.sep(rest)
    th = nb.m("ARCTAN2", rx, nb.m("MULTIPLY", ry, -1.0))
    eye = None
    for s in (-1.0, 1.0):
        ex = nb.m("DIVIDE", nb.m("SUBTRACT", th, 0.30 * s), 0.17)
        ey = nb.m("DIVIDE", nb.m("SUBTRACT", rz, 0.665), 0.092)
        d = nb.m("ADD", nb.m("MULTIPLY", ex, ex), nb.m("MULTIPLY", ey, ey))
        e = nb.mr(d, 1.0, 0.8)                                               # soft-edged oval
        eye = e if eye is None else nb.m("MAXIMUM", eye, e)
    front = nb.mr(nb.m("ABSOLUTE", th), 1.2, 0.9)                            # never on the back
    eye = nb.m("MULTIPLY", eye, front)
    h = nb.sattr("h").outputs["Fac"]
    hem = nb.mr(h, 0.55, 0.0, interp="SMOOTHSTEP")                          # 0 above mid-body → 1 at the hem, no kink
    base = nb.mix(nb.m("MULTIPLY", hem, 0.8), C.GHOST_WHITE, (0.80, 0.70, 1.0))    # faint lilac toward the hem
    base = nb.mix(eye, base, (0.004, 0.002, 0.008))
    b = X.principled(nb, base=base, rough=nb.mix(eye, 0.55, 0.15, dtype="FLOAT"))
    b.inputs["Subsurface Weight"].default_value = 0.0
    rim = nb.m("POWER", C.facing(nb, 0.45), 2.0)
    glow = nb.m("ADD", 0.10, nb.m("MULTIPLY", rim, 2.8))
    glow = nb.m("ADD", glow, nb.m("MULTIPLY", nb.m("POWER", hem, 1.5), 0.9))     # hem glows (cover)
    glow = nb.m("MULTIPLY", nb.m("MULTIPLY", glow, K["ghost_glow"]), nb.m("SUBTRACT", 1.0, eye))
    ecol = nb.mix(nb.m("MINIMUM", nb.m("ADD", rim, hem), 1.0), (0.88, 0.78, 1.0), (0.68, 0.24, 1.0))
    nb.link(ecol, b.inputs["Emission Color"])
    nb.link(glow, b.inputs["Emission Strength"])
    nb.link(b.outputs[0], out.inputs["Surface"])
    m.use_backface_culling = False
    # wisps: additive glowing ribbons, fading to the tip and to the ribbon edges
    w, wb, wo = X.new_mat("GhostWisp")
    K2 = C.ctrl(wb)
    u = wb.sattr("u").outputs["Fac"]
    side = wb.sattr("side").outputs["Fac"]
    wk = wb.sattr("wk").outputs["Fac"]
    a = wb.m("MULTIPLY", wb.m("POWER", wb.m("SUBTRACT", 1.0, u), 1.1),
             wb.m("POWER", wb.m("SUBTRACT", 1.0, wb.m("ABSOLUTE", side)), 0.8))
    # strands: noise stretched along the ribbon, flowing toward the tip on CTRL.T
    sn = C.noise(wb, wb.xyz(wb.m("SUBTRACT", wb.m("MULTIPLY", u, 3.0), wb.m("MULTIPLY", K2["T"], 0.5)),
                            wb.m("ADD", wb.m("MULTIPLY", side, 2.2), wb.m("MULTIPLY", wk, 3.7)), 0.0),
                 scale=2.2, detail=3.0, rough=0.55)
    a = wb.m("MULTIPLY", a, wb.mr(sn.outputs["Fac"], 0.3, 0.6, 0.15, 1.0))
    col = wb.mix(u, (0.90, 0.62, 1.0), (0.60, 0.20, 1.0))
    st = wb.m("MULTIPLY", wb.m("MULTIPLY", a, K2["ghost_glow"]), 3.4)
    sh = C.mix_shader(wb, wb.m("MINIMUM", wb.m("MULTIPLY", a, 0.9), 0.85), C.transparent(wb), C.emission(wb, col, st))
    wb.link(sh, wo.inputs["Surface"])
    C.blended(w)
    return m, w


# name, location, height (m), yaw (deg: 0 = face toward −Y), world trail direction, phase, n_wisps, light W
GHOSTS = [
    dict(C.BIG_GHOST, wisps=9, wlen=1.6, wwid=2.4, light=450.0, bob=0.035, sway=0.03, speed=0.42, drag=0.35, wisp=0.10),
    dict(name="Ghost_S1", loc=(-9.5, 15.0, 9.8), height=1.25, yaw=55.0, trail=(-1, 0.3, 0.2), phase=1.7, wisps=4,
         light=0.0, bob=0.06, sway=0.05, speed=0.55, drag=0.4, wisp=0.14),
    dict(name="Ghost_S2", loc=(15.0, 46.0, 16.5), height=1.6, yaw=-20.0, trail=(1, 0.4, 0.3), phase=3.1, wisps=4,
         light=0.0, bob=0.06, sway=0.05, speed=0.5, drag=0.4, wisp=0.14),
    dict(name="Ghost_S3", loc=(44.0, 150.0, 22.0), height=3.2, yaw=-30.0, trail=(1, 0.5, 0.1), phase=4.4, wisps=3,
         light=0.0, bob=0.05, sway=0.04, speed=0.47, drag=0.35, wisp=0.12),
    dict(name="Ghost_S4", loc=(9.0, 23.0, 4.6), height=1.0, yaw=-10.0, trail=(-0.6, 1, 0.3), phase=5.2, wisps=4,
         light=0.0, bob=0.07, sway=0.06, speed=0.6, drag=0.45, wisp=0.15),
]


def build(coll, R):
    import bpy
    tree = motion_tree()
    mb, mw = materials()
    obs = []
    for k, g in enumerate(GHOSTS):
        yaw = math.radians(g["yaw"])
        # world trail → local (inverse yaw about Z)
        tw = np.asarray(g["trail"], float)
        c, s = math.cos(-yaw), math.sin(-yaw)
        tl = np.array([c * tw[0] - s * tw[1], s * tw[0] + c * tw[1], tw[2]])
        V, F, MI, A = ghost_mesh(g["name"], trail=tl, n_wisps=g["wisps"], seed=10 + k, wisp_len=g.get("wlen", 1.0),
                                 wisp_w=g.get("wwid", 1.0))
        ob = X.mesh_obj(g["name"], V, F, coll=coll, smooth=True)
        me = ob.data
        me.materials.append(mb)
        me.materials.append(mw)
        me.polygons.foreach_set("material_index", np.asarray(MI, np.int32))
        for nm in ("h", "u", "part", "side", "wk"):
            me.attributes.new(nm, "FLOAT", "POINT").data.foreach_set("value", np.asarray(A[nm], np.float32))
        for nm in ("rest", "perp"):
            me.attributes.new(nm, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", A[nm].ravel())
        me.update()
        ob.location = g["loc"]
        ob.rotation_euler = (0, 0, yaw)
        ob.scale = (g["height"],) * 3
        mod = X.add_gn(ob, tree, name="GhostMotion")
        for nm, val in (("Phase", g["phase"]), ("Bob", g["bob"]), ("Sway", g["sway"]), ("Speed", g["speed"]),
                        ("Trail", tuple(float(x) for x in tl)), ("Drag", g["drag"]), ("Wisp", g["wisp"])):
            C.set_mod_input(mod, nm, val)
        sub = ob.modifiers.new("Sub", "SUBSURF")
        sub.levels = sub.render_levels = 1
        if g["light"]:
            L = X.light(f"{g['name']}_Light", "POINT", (0, 0, 0), energy=g["light"], color=(0.75, 0.40, 1.0), size=0.6,
                        coll=coll)
            L.parent = ob
            L.location = (0.0, -0.25, 0.45)            # local units (height 1): just in front of the belly
            L.data.use_soft_falloff = True
        obs.append(ob)
    R.add("ghosts", [g["name"] for g in GHOSTS])
    return obs
