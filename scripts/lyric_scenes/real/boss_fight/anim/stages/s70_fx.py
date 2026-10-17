"""s70 — event-driven FX, all placeholders that read in silhouette (emissive, shadowless):
shield bubble (4 windows) · palm fire · 3 orbiting rocks → launched · smolder · fireballs (knight ×2, big, dragon spit
×2, volley ×3) that STOP on contact · earth-spike wave · potion + gem + aura · fire breath truncated by the shield ·
holy sword (forms, falls, pins) · golden RPG carried + rocket · impact flashes + shockwave rings · gate doors.
Projectile endpoints are solved against the EVALUATED dragon proxy / shield sphere / colossus on the contact frame."""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from anim import common as C
from anim import fx as FX
from lib import bpyx as X
from lib import checks as CK
from lib import importer as imp

FIRE = (1.0, 0.45, 0.12)
MAGIC = (0.35, 0.65, 1.0)
HOLY = (1.0, 0.85, 0.5)


def bvh_of(objs, f=None, S=None):
    if f is not None:
        S.sc.frame_set(int(f))
    dg = bpy.context.evaluated_depsgraph_get()
    V, T = CK.bpy_world_tris(objs, dg)
    return BVHTree.FromPolygons(V.tolist(), T.tolist())


def reg(S, ob, body, owner, kind):
    S.fx_bodies[ob.name] = dict(body=body, owner=owner, kind=kind)


def run(S):
    E, T, N, col = S.E, S.trk, S.N, S.col
    f = {k: e.f for k, e in E.items()}
    S.fx_bodies = {}
    S.fx_hits = {}

    def P(name, fr):
        return T[name][int(fr) - 1]

    def kfwd(fr):
        yaw = (S.K.base_yaw + S.K.yoff)[int(fr) - 1]
        return C.fwd(yaw)

    # ------------------------------------------------------------------ shield bubble
    shield = FX.ico("K_Shield", 1.0, col.fx, FX.mat_glass("M_Shield", MAGIC, 6.0, 0.25), subdiv=3)
    S.shield_windows = [(f["ceiling"] - 16, f["ceiling"] + 22, 1.4), (f["dfireballs"] - 14, f["dfireballs"] + 24, 1.4),
                        (f["roof_rain"] - 14, f["roof_rain"] + 34, 2.2), (f["shield_breath"] - 14, f["shield_breath"] + 62, 1.6)]
    fr = np.arange(1, N + 1)
    sc = np.full(N, 0.02)
    for a, b, R in S.shield_windows:
        for i in range(a, b + 1):
            t_in = min(1.0, (i - a) / 4.0)
            t_out = min(1.0, (b - i) / 4.0)
            sc[i - 1] = max(0.02, R * min(t_in, t_out))
    cen = T["K.chest"].copy()
    FX.key_path(shield, fr, cen, scale=sc)
    C.key_visibility(shield, [(1, True)] + sum([[(a, False), (b + 1, True)] for a, b, _ in S.shield_windows], []))
    reg(S, shield, "K.shield", "K", "shield")
    S.shield_center = cen
    S.shield_radius = sc

    def shield_hit(p0, fr_hit):
        c = cen[fr_hit - 1]
        R = sc[fr_hit - 1]
        d = c - p0
        d /= np.linalg.norm(d)
        # ray-sphere: first intersection from p0
        oc = p0 - c
        b = np.dot(oc, d)
        disc = b * b - (np.dot(oc, oc) - R * R)
        t = -b - math.sqrt(max(disc, 0.0))
        return p0 + d * t

    # ------------------------------------------------------------------ projectile factory
    def projectile(name, f0, f1, p0, kind, target, radius, color, owner, arc=0.6, strength=40.0, after="burst",
                   static_bvh=None):
        p0 = np.asarray(p0, np.float64)
        if kind == "dragon":
            bvh = bvh_of([S.D.proxy], f1, S)
            hit = FX.surface_hit(bvh, p0, target)
            if hit is None:
                S.R.log(f"WARN projectile {name}: no surface on the way to the dragon target; using the anchor")
                hit = np.asarray(target)
            d = (np.asarray(target) - p0)
            d /= np.linalg.norm(d)
            p1 = hit + d * radius * 0.6
        elif kind == "shield":
            p1 = shield_hit(p0, f1) + (cen[f1 - 1] - p0) / np.linalg.norm(cen[f1 - 1] - p0) * radius * 0.6
        elif kind == "static":
            hit = FX.surface_hit(static_bvh, p0, target)
            d = (np.asarray(target) - p0)
            d /= np.linalg.norm(d)
            p1 = (hit if hit is not None else np.asarray(target)) + d * radius * 1.0
        else:
            p1 = np.asarray(target, np.float64)
        n = f1 - f0 + 1
        path = FX.path_arc(p0, p1, n, arc)
        ob = FX.ico(name, radius, col.proj, FX.mat_emit(f"M_proj_{color}", color, strength), subdiv=2)
        trail = FX.cone(f"{name}_trail", radius * 0.9, 0.02, radius * 6, col.proj,
                        FX.mat_emit(f"M_trail_{color}", color, strength * 0.4), segs=10)
        trail.parent = ob
        vel = np.gradient(path, axis=0)
        rot = np.array([list(FX.look_euler(-v if np.linalg.norm(v) > 1e-6 else (0, 0, -1))) for v in vel])
        FX.key_path(ob, np.arange(f0, f1 + 1), path, rot_euler=rot)
        FX.window(ob, f0, f1)
        FX.window(trail, f0, f1)
        reg(S, ob, name, owner, "projectile")
        FX.flash(S, name, p1, f1, size=radius * 2.5, color=color)
        S.fx_hits[name] = dict(f=f1, p=p1.round(3).tolist(), v=(vel[-1] * C.FPS).round(2).tolist())
        return ob, p1, vel[-1]

    # ------------------------------------------------------------------ KIT 2: ONE fireball spinning in the RIGHT palm
    # → KIT 3: it collapses into 3 rocks orbiting THAT hand; they crumble away before the doors (Zeke v1 note)
    kf, kr = f["kit_fire"], f["kit_rocks"]
    palm = lambda fr: P("K.rhand", fr) + np.array([0, 0, 0.13]) + kfwd(fr) * 0.05
    fb = FX.ico("FX_palmfire_Right", 0.11, col.fx, FX.mat_emit("M_palmfire", FIRE, 35.0), subdiv=2)
    frs = np.arange(kf, kr + 6)
    grow = np.clip((frs - kf) / 8.0, 0.05, 1.0) * np.clip((kr + 6 - frs) / 6.0, 0.05, 1.0)
    pul = grow * (1.0 + 0.12 * np.sin(frs * 1.3))
    FX.key_path(fb, frs, np.array([palm(x) for x in frs]),
                rot_euler=np.c_[frs * 0.31, frs * 0.17, frs * 0.53], scale=np.c_[pul, pul, pul * 1.25])
    FX.window(fb, kf, kr + 6)
    crumble = f["doors"] - 22
    for i in range(3):
        rk = FX.rock(f"kit_rock_{i + 1}", 0.075, col.fx, FX.mat_emit("M_rock_glow", (0.55, 0.22, 0.08), 2.0), seed=10 + i)
        frs = np.arange(kr, crumble + 8)
        ph = np.radians(120 * i) + 2 * math.pi * (frs - kr) / 26.0
        c = np.array([palm(x) for x in frs]) + np.array([0, 0, 0.05])
        pos = c + np.c_[0.2 * np.cos(ph), 0.2 * np.sin(ph), 0.05 * np.sin(2 * ph)]
        sc_ = np.clip((frs - kr) / 6.0, 0.05, 1.0) * np.clip((crumble + 8 - frs) / 8.0, 0.02, 1.0)
        FX.key_path(rk, frs, pos, rot_euler=np.c_[ph * 0.7, ph * 1.3, ph], scale=np.c_[sc_, sc_, sc_])
        FX.window(rk, kr, crumble + 8)
    # drop B: rocks TORN FROM THE CRACKED FLOOR rise, orbit his right hand, then launched at the head (rock_1..3)
    hits = [f["rocks"]] + S.extra_hits["rocks"]
    r0 = f["rocks"] - 60
    for i, fh in enumerate(hits):
        name = f"rock_{i + 1}"
        fl0 = fh - 10
        orb = np.arange(r0, fl0 + 1)
        ph = np.radians(120 * i) + 2 * math.pi * (orb - r0) / 30.0
        hand = np.array([palm(x) for x in orb]) + np.array([0, 0, 0.1])
        ring_ = hand + np.c_[0.42 * np.cos(ph), 0.42 * np.sin(ph), 0.08 * np.sin(2 * ph)]
        kroot = P("K.root", r0)
        src = kroot + np.array([math.cos(ph[0]) * 1.4, math.sin(ph[0]) * 1.4, 0.0])
        src[2] = S.ground(src[0], src[1]) + 0.05
        rise = np.clip((orb - r0) / 14.0, 0.0, 1.0)
        rise = rise * rise * (3 - 2 * rise)
        pos = src[None] * (1 - rise[:, None]) + ring_ * rise[:, None]
        rk = FX.rock(name, 0.14, col.proj, FX.mat_emit("M_rock_glow", (0.55, 0.22, 0.08), 2.0), seed=i)
        bvh = bvh_of([S.D.proxy], fh, S)
        tgt = P("D.head", fh) + np.array([0, 0, 0.1 * i])
        hit = FX.surface_hit(bvh, pos[-1], tgt)
        d = tgt - pos[-1]
        d /= np.linalg.norm(d)
        p1 = (hit if hit is not None else tgt) + d * 0.1
        fl = FX.path_arc(pos[-1], p1, fh - fl0 + 1, 0.4)
        v = (fl[-1] - fl[-2])
        away = (p1 - P("D.chest", fh)) * np.array([1, 1, 0])
        away = away / max(np.linalg.norm(away), 1e-6)
        vb = -0.2 * v * C.FPS + np.array([0, 0, 3.5]) + away * 6.0       # bounces OFF the head, falls clear of the body
        fall = [p1]
        g0 = S.ground(p1[0], p1[1]) + 0.12
        for k in range(1, 90):
            t = k / C.FPS
            q = p1 + vb * t + np.array([0, 0, -0.5 * C.G * t * t])
            if q[2] <= g0:
                q[2] = g0
                fall.append(q)
                break
            fall.append(q)
        fall = np.array(fall)
        allp = np.concatenate([pos, fl[1:], fall[1:]])
        allf = np.arange(r0, r0 + len(allp))
        FX.key_path(rk, allf, allp, rot_euler=np.c_[allf * 0.09, allf * 0.13, allf * 0.1])
        FX.window(rk, r0, f["gem_out"])
        reg(S, rk, name, "K", "projectile")
        S.fx_hits[name] = dict(f=fh, p=p1.round(3).tolist())
        FX.flash(S, name, p1, fh, size=0.45, color=(1.0, 0.6, 0.2))
        S.rock_rest = getattr(S, "rock_rest", []) + [(allf[-1], fall[-1])]

    # ------------------------------------------------------------------ smolder at the nostrils
    tb = S.D.skel.find("DEF-Teeth_Top")
    for i, dx in enumerate((-0.12, 0.12)):
        sm = FX.ico(f"FX_smolder_{i}", 0.07, col.fx, FX.mat_emit("M_smolder", FIRE, 25.0), subdiv=1)
        sm.parent = S.D.arm
        sm.parent_type = "BONE"
        sm.parent_bone = tb
        sm.location = (dx / C.DRAGON_S / 0.1339, 0, 0)
        frs = np.arange(f["smolder"], 1600)
        s = 1.0 + 0.6 * np.sin(frs * 0.9 + i) * np.sin(frs * 0.37)
        FX.key_path(sm, frs, np.repeat(np.array([[dx / (C.DRAGON_S * 0.1339), 0.0, 0.0]]), len(frs), 0),
                    scale=np.c_[s, s * 1.8, s] / (C.DRAGON_S * 0.1339))
        FX.window(sm, f["smolder"], 1600)

    # ------------------------------------------------------------------ knight fireballs ×2 → chest
    for i, fh in enumerate([f["fireballs"]] + S.extra_hits["fireballs"]):
        f0 = fh - 8
        tgt = P("D.chest", fh)
        hand = P("K.rhand", f0)
        projectile(f"fireball_{i + 1}", f0, fh, hand + kfwd(f0) * 0.3, "dragon", tgt, 0.22, FIRE, "K", arc=0.3)

    # ------------------------------------------------------------------ dragon spit ×2 → shield
    for i, fh in enumerate([f["dfireballs"]] + S.extra_hits["dfireballs"]):
        f0 = fh - 9
        projectile(f"dragon_fireball_{i + 1}", f0, fh, P("D.snout", f0), "shield", None, 0.35, FIRE, "D", arc=0.5)

    # ------------------------------------------------------------------ big fireball → dragon mid-air
    fh = f["big_fireball"]
    f0 = fh - 28
    projectile("fireball_big_1", f0, fh, P("K.rhand", f0) + np.array([0, 0, 0.4]), "dragon", P("D.chest", fh), 0.6,
               (1.0, 0.55, 0.2), "K", arc=2.0, strength=60.0)

    # ------------------------------------------------------------------ volley: #1 → colossus E arm, #2/#3 → ground
    col_e = bpy.data.objects["Colossus_E"]
    cb = bvh_of([col_e], None, S)
    arm_pt = np.array(cb.find_nearest(Vector((13.5, -18.6, 19.0)))[0])
    S.colossus_hit = arm_pt
    fh = f["volley"]
    projectile("dragon_fireball_v", fh - 26, fh, P("D.snout", fh - 26), "static", arm_pt, 0.45, FIRE, "D", arc=1.0,
               static_bvh=cb)
    kp = P("K.root", fh)
    for j, (df, off) in enumerate(((-6, (3.5, -2.5)), (6, (-3.0, 2.0)))):
        g = kp + np.array([off[0], off[1], 0.0])
        g[2] = S.ground(g[0], g[1]) + 0.3
        projectile(f"dragon_fireball_v{j + 2}", fh + df - 26, fh + df, P("D.snout", fh + df - 26), "ground", g, 0.45,
                   FIRE, "D", arc=1.0)
        FX.ring(S, f"volley{j}", g - np.array([0, 0, 0.25]), fh + df, 3.0, dur=10)

    # ------------------------------------------------------------------ earth spikes → belly
    fs = f["spikes"]
    belly = P("D.belly", fs)
    gz = S.ground(belly[0], belly[1])
    bvh = bvh_of([S.D.proxy], fs, S)
    up = bvh.ray_cast(Vector((belly[0], belly[1], gz + 0.05)), Vector((0, 0, 1)), 8.0)
    under = up[0].z if up[0] is not None else belly[2] - 0.6
    yawd = (S.D.base_yaw + S.D.yoff)[fs - 1]
    L0 = P("K.root", fs - 14) + kfwd(fs - 14) * 1.6
    Lend = np.array([belly[0], belly[1], gz])
    Lfront = Lend + C.fwd(yawd) * 4.6          # the wave stops 4.6 m in FRONT of the belly (clear of the forefeet)
    spikes = []
    mspike = FX.mat_emit("M_spike_glow", (0.9, 0.55, 0.25), 2.5)
    for i in range(9):
        last = i == 8
        if last:
            base = Lend
            h = (under - gz) + 0.35
        else:
            t = i / 7.0
            base = L0 + (Lfront - L0) * t
            base[2] = S.ground(base[0], base[1])
            h = 0.55 + 0.11 * i
        sp = FX.cone(f"earth_spike_{i:02d}", 0.32 if not last else 0.4, 0.0, 1.0, col.fx, mspike, segs=7)
        sp.visible_shadow = True
        fi = fs - (8 - i) * 2
        frs = np.arange(fi - 3, 2930)
        z = np.interp(frs, [fi - 3, fi - 1, fi, fi + 3], [0.02, 0.45, 1.15, 1.0]) * h
        if last:
            z = np.interp(frs, [fi - 3, fi - 2, fi, fi + 4, fi + 20], [0.02, 0.3, 1.0, 1.0, 0.02]) * h
        z = np.where(frs > 2905, z * np.clip((2925 - frs) / 20.0, 0.02, 1.0), z)
        tilt = (base - Lend)[:2] * 0.0
        FX.key_path(sp, frs, np.repeat(base[None], len(frs), 0), scale=np.c_[np.ones(len(frs)), np.ones(len(frs)), np.maximum(z, 0.02)])
        FX.window(sp, fi - 3, 2925)
        reg(S, sp, "earth_spikes", "K", "spike")
        spikes.append(sp)
    S.fx_hits["earth_spikes"] = dict(f=fs, p=[float(Lend[0]), float(Lend[1]), float(under)])
    FX.flash(S, "spikes", [Lend[0], Lend[1], under], fs, size=0.8, color=(1.0, 0.7, 0.3))

    # ------------------------------------------------------------------ potion, gem, aura
    hb = S.K.hand_bone
    pot = FX.cone("FX_potion", 0.05, 0.035, 0.16, col.props, FX.mat_emit("M_potion", (1.0, 0.1, 0.12), 6.0), segs=10)
    FX.attach(pot, S.K.arm, hb)
    FX.window(pot, f["potion"] - 30, f["potion"] + 36)
    pr = FX.torus("FX_potion_ring", 0.55, 0.03, col.fx, FX.mat_emit("M_heal", (0.3, 1.0, 0.4), 15.0))
    frs = np.arange(f["potion"] + 10, f["potion"] + 50)
    base = T["K.root"][frs - 1]
    FX.key_path(pr, frs, base + np.c_[np.zeros(len(frs)), np.zeros(len(frs)), np.linspace(0.05, 1.9, len(frs))])
    FX.window(pr, frs[0], frs[-1])
    gem = FX.ico("FX_gem", 0.07, col.props, FX.mat_emit("M_gem", (0.3, 0.9, 1.0), 20.0), subdiv=1)
    FX.attach(gem, S.K.arm, hb)
    FX.window(gem, f["gem_out"] + 30, f["gem_crush"])
    gc = P("K.rhand", f["gem_crush"])
    rng = np.random.default_rng(7)
    for i in range(8):
        sh = FX.ico(f"FX_gem_shard_{i}", 0.03, col.fx, FX.mat_emit("M_gem", (0.3, 0.9, 1.0), 20.0), subdiv=0)
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * 4.0 + np.array([0, 0, 2.0])
        frs = np.arange(f["gem_crush"], f["gem_crush"] + 18)
        t = (frs - frs[0]) / C.FPS
        FX.key_path(sh, frs, gc + np.outer(t, v) + np.outer(-0.5 * C.G * t * t, [0, 0, 1]))
        FX.window(sh, frs[0], frs[-1])
    FX.flash(S, "gem", gc, f["gem_crush"], size=0.5, color=(0.4, 0.9, 1.0), light=900.0)
    aura = S.K.proxy.copy()
    aura.data = S.K.proxy.data.copy()
    aura.name = "FX_aura"
    col.fx.objects.link(aura)
    aura.hide_render = False
    aura.display_type = "TEXTURED"
    aura.data.materials.clear()
    aura.data.materials.append(FX.mat_glass("M_aura", (1.0, 0.8, 0.35), 8.0, 0.0))
    dm = aura.modifiers.new("puff", "DISPLACE")
    dm.strength = 0.05
    aura.visible_shadow = False
    FX.window(aura, f["gem_crush"], f["sword_falls"] + 40)
    # aura STRAIN while he summons the holy sword: the shell swells and shudders harder as the sword forms
    frs = np.arange(f["gem_crush"], f["sword_falls"] + 41)
    rh, sfall = f["raise_hand"], f["sword_falls"]
    prog = np.clip((frs - rh) / max(sfall - rh, 1), 0.0, 1.0) * (frs < sfall + 6)
    jit = np.sin(frs * 2.3) * np.sin(frs * 0.71 + 1.0)
    strength = 0.05 + prog * (0.10 + 0.07 * jit)
    fc = aura.modifiers["puff"]
    for fr_, v in zip(frs[::2], strength[::2]):
        fc.strength = float(v)
        fc.keyframe_insert("strength", frame=int(fr_))

    # ------------------------------------------------------------------ fire breath (dive), stopped by the shield
    fd, fsb = f["dive"], f["shield_breath"]
    br = FX.cone("breath", 0.22, 1.0, 1.0, col.proj, FX.mat_emit("M_breath", (1.0, 0.5, 0.15), 18.0), segs=16)
    frs = np.arange(fd, fd + 71)
    pos, rot, scl = [], [], []
    for fr_ in frs:
        sn = P("D.snout", fr_)
        kc = P("K.chest", fr_)
        ahead = P("K.root", fr_) + (sn - P("K.root", fr_)) * np.array([1, 1, 0]) / max(np.linalg.norm((sn - P("K.root", fr_))[:2]), 1e-6) * 9.0
        s = C.smooth01((fr_ - fd) / max(fsb - fd, 1))
        aim = ahead * (1 - s) + kc * s
        d = aim - sn
        d /= np.linalg.norm(d)
        L = 30.0
        R = sc[fr_ - 1]
        if R > 0.3:
            oc = sn - cen[fr_ - 1]
            b = np.dot(oc, d)
            disc = b * b - (np.dot(oc, oc) - R * R)
            if disc > 0 and -b - math.sqrt(disc) > 0:
                L = min(L, -b - math.sqrt(disc) + 0.12)
        if d[2] < -1e-3:
            gz_ = S.ground(sn[0] + d[0] * 10, sn[1] + d[1] * 10)
            L = min(L, (sn[2] - gz_) / -d[2])
        pos.append(sn)
        rot.append(list(FX.look_euler(d)))
        w = min(1.0, L / 12.0) * 0.7 + 0.2
        scl.append([w, w, L])
    FX.key_path(br, frs, np.array(pos), rot_euler=np.array(rot), scale=np.array(scl))
    FX.window(br, frs[0], frs[-1])
    reg(S, br, "breath", "D", "breath")

    # ------------------------------------------------------------------ holy sword: forms, falls, pins
    hs_objs = imp.import_model(C.MED + "r2_props/prop_holy_sword_gold_bl4ckgh0st/model.glb", coll=col.props, cache=False)
    names = [o.name for o in hs_objs]
    me = imp.bake_world([o for o in hs_objs if o.type == "MESH"], "holy_sword", remove_sources=True)
    for n in names:
        o = bpy.data.objects.get(n)
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)
    co = np.array([v.co for v in me.vertices])
    cen0 = co.mean(0)
    _u, _s, vt = np.linalg.svd(co - cen0, full_matrices=False)
    ax = vt[0]
    t = (co - cen0) @ ax
    wdt = np.linalg.norm((co - cen0) - np.outer(t, ax), axis=1)
    bins = np.linspace(t.min(), t.max(), 41)
    wid = np.array([wdt[(t >= bins[i]) & (t < bins[i + 1])].max(initial=0) for i in range(40)])
    gi = int(np.argmax(wid))
    tg = 0.5 * (bins[gi] + bins[gi + 1])
    if tg < 0.5 * (t.min() + t.max()):
        ax, t = -ax, -t          # +ax: tip → hilt
    tip = cen0 + ax * t.min()
    Lsw = float(t.max() - t.min())
    s_ = 16.0 / Lsw
    bx = vt[1] - ax * np.dot(vt[1], ax)
    bx /= np.linalg.norm(bx)
    bz = np.cross(bx, ax)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = bx, bz, ax, tip     # local +Z = tip→hilt, origin at the tip
    me.transform(Matrix(np.linalg.inv(M).tolist()))
    me.transform(Matrix.Scale(s_, 4))
    hs = bpy.data.objects.new("holy_sword", me)
    col.proj.objects.link(hs)
    ff, fF, fh = f["sword_forming"], f["sword_formed"], f["sword_falls"]
    back = P("D.back", fh)
    bvh = bvh_of([S.D.proxy], fh, S)
    zs = []
    for dx in np.linspace(-0.9, 0.9, 7):
        for dy in np.linspace(-0.9, 0.9, 7):
            hit = bvh.ray_cast(Vector((back[0] + dx, back[1] + dy, back[2] + 14)), Vector((0, 0, -1)), 30)
            if hit[0] is not None:
                zs.append(hit[0].z)
    zb = max(zs) if zs else back[2] + 0.6
    gzb = S.ground(back[0], back[1])
    z0 = gzb + 42.0
    frs = np.arange(ff, 6836)
    zt = np.where(frs <= fF, z0, 0.0)
    tt = np.clip((frs - fF) / (fh - fF), 0, 1)
    zt = np.where((frs > fF) & (frs <= fh), z0 - (z0 - (zb - 0.35)) * tt ** 2, zt)
    zt = np.where(frs > fh, np.interp(frs, [fh, fh + 4], [zb - 0.35, gzb - 1.2]), zt)
    scl = np.where(frs <= fF, 0.12 + 0.88 * C.smooth01((frs - ff) / (fF - ff)), 1.0)
    pos = np.c_[np.full(len(frs), back[0]), np.full(len(frs), back[1]), zt]
    FX.key_path(hs, frs, pos, scale=scl)
    FX.window(hs, ff, 6835)
    reg(S, hs, "holy_sword", None, "holy")
    glow = FX.ico("FX_holy_glow", 2.5, col.fx, FX.mat_glass("M_holyglow", HOLY, 10.0, 0.1), subdiv=2)
    FX.key_path(glow, frs, pos + np.array([0, 0, 9.0]), scale=np.where(frs <= fF, scl * 2.0, 1.0))
    FX.window(glow, ff, fh)
    S.fx_hits["holy_sword"] = dict(f=fh, p=[float(back[0]), float(back[1]), float(zb)])
    FX.flash(S, "holy", [back[0], back[1], zb], fh, size=3.0, color=HOLY, strength=80.0, light=8000.0)
    FX.ring(S, "holy", [back[0], back[1], gzb + 0.1], fh, 14.0, dur=16, color=HOLY)

    # ------------------------------------------------------------------ golden RPG: path from s55 (hands are IK'd onto it)
    anchor = bpy.data.objects["PRIZE_anchor"]
    rpg = bpy.data.objects["PRIZE_GoldRPG7"]
    Lm = np.array(rpg.matrix_local)
    W = S.rpg["world"]
    fp, fR = int(S.K.plan["rpg"]["pickup"]), f["fire_rpg"]
    frs = np.arange(fp, N + 1)
    locs, rots = [], []
    for fr_ in frs:
        Ma = W[fr_ - 1] @ np.linalg.inv(Lm)
        Mm = Matrix(Ma.tolist())
        locs.append(np.array(Mm.translation))
        rots.append(list(Mm.to_euler()))
    pre_l = np.array(anchor.location)
    pre_r = np.array(anchor.rotation_euler)
    FX.key_path(anchor, np.r_[1, fp - 1, frs], np.r_[[pre_l, pre_l], np.array(locs)],
                rot_euler=np.r_[[pre_r, pre_r], np.unwrap(np.array(rots), axis=0)])
    # rocket: launched at L (recoil + BACK-BLAST out of the rear of the tube), flies on its trail, explodes at fR
    L = fR - 16
    tgt_head = P("D.head", fR)
    Mw = W[L - 1]
    muzzle = (Mw @ np.r_[S.rpg["muzzle_local"], 1.0])[:3]
    rear = (Mw @ np.r_[S.rpg["rear_local"], 1.0])[:3]
    rk, p1, v = projectile("rpg_rocket", L, fR, muzzle, "dragon", tgt_head, 0.12, (1.0, 0.7, 0.3), "K", arc=0.0,
                           strength=30.0)
    back = rear - muzzle
    back /= np.linalg.norm(back)
    bb = FX.cone("FX_rpg_backblast", 0.10, 0.9, 1.0, col.fx, FX.mat_emit("M_backblast", (1.0, 0.62, 0.25), 25.0), segs=14)
    frs = np.arange(L, L + 9)
    ln = np.interp(frs, [L, L + 2, L + 8], [0.3, 3.2, 3.6])
    wd = np.interp(frs, [L, L + 2, L + 8], [0.3, 1.0, 1.6])
    FX.key_path(bb, frs, np.repeat(rear[None], len(frs), 0), rot_euler=np.repeat(np.array([list(FX.look_euler(back))]), len(frs), 0),
                scale=np.c_[wd, wd, ln])
    FX.window(bb, L, L + 8)
    for j in range(5):                                   # back-blast smoke puffs rolling out behind him
        sm = FX.ico(f"FX_rpg_smoke_{j}", 0.5, col.fx, FX.mat_solid("M_rpg_smoke", (0.35, 0.33, 0.31), rough=1.0), subdiv=1)
        frs = np.arange(L + 1, L + 46)
        t = (frs - frs[0]) / C.FPS
        dirj = back + np.array([0.25 * math.cos(j * 1.3), 0.25 * math.sin(j * 1.3), 0.12 * (j - 2)])
        pos = rear[None] + np.outer(1.0 - np.exp(-t * 2.2), dirj * (2.0 + 0.6 * j))
        scl = (0.4 + 1.6 * (1.0 - np.exp(-t * 1.8))) * np.clip((L + 46 - frs) / 10.0, 0.05, 1.0)
        FX.key_path(sm, frs, pos, scale=np.c_[scl, scl, scl])
        FX.window(sm, L + 1, L + 45)
    FX.flash(S, "rpg_launch", muzzle, L, size=0.6, color=(1.0, 0.75, 0.4), light=2500.0)
    # explosion: fireball core → expanding fire shell → dark smoke column, debris sparks, a big light flash
    boom = FX.ico("FX_rpg_boom", 1.0, col.fx, FX.mat_emit("M_boom", (1.0, 0.55, 0.2), 40.0), subdiv=3)
    frs = np.arange(fR, fR + 7)
    FX.key_path(boom, frs, np.repeat(p1[None], len(frs), 0), scale=np.repeat(np.linspace(0.6, 6.5, len(frs))[:, None], 3, 1))
    FX.window(boom, fR, fR + 6)
    core = FX.ico("FX_rpg_core", 1.0, col.fx, FX.mat_emit("M_boom_core", (1.0, 0.92, 0.7), 90.0), subdiv=2)
    frs = np.arange(fR, fR + 5)
    FX.key_path(core, frs, np.repeat(p1[None], len(frs), 0), scale=np.repeat(np.interp(frs, [fR, fR + 2, fR + 4], [0.3, 2.8, 1.2])[:, None], 3, 1))
    FX.window(core, fR, fR + 4)
    smk = FX.ico("FX_rpg_boom_smoke", 1.0, col.fx, FX.mat_solid("M_boom_smoke", (0.12, 0.1, 0.09), rough=1.0), subdiv=2)
    frs = np.arange(fR + 2, fR + 7)
    FX.key_path(smk, frs, np.repeat(p1[None], len(frs), 0) + np.outer(np.linspace(0, 1.5, len(frs)), [0, 0, 1]),
                scale=np.repeat(np.linspace(2.0, 7.5, len(frs))[:, None], 3, 1))
    FX.window(smk, fR + 2, fR + 6)
    rng = np.random.default_rng(11)
    for j in range(10):
        sp = FX.ico(f"FX_rpg_spark_{j}", 0.12, col.fx, FX.mat_emit("M_spark", (1.0, 0.8, 0.4), 40.0), subdiv=0)
        vj = rng.normal(size=3)
        vj = vj / np.linalg.norm(vj) * 14.0 + np.array([0, 0, 4.0])
        frs = np.arange(fR, fR + 7)
        t = (frs - fR) / C.FPS
        FX.key_path(sp, frs, p1[None] + np.outer(t, vj) + np.outer(-0.5 * C.G * t * t, [0, 0, 1]))
        FX.window(sp, fR, fR + 6)
    FX.flash(S, "rpg_boom", p1, fR, size=4.0, color=(1.0, 0.7, 0.35), strength=90.0, light=20000.0)
    S.blackout_from = fR + 6

    # ------------------------------------------------------------------ gate doors (pushed open at `doors`)
    fdoor = f["doors"]
    for sx in (-1, 1):
        dr = FX.box(f"FX_Door_{'W' if sx < 0 else 'E'}", (6.75, 0.2, 9.0), col.props,
                    FX.mat_solid("M_DoorOak", (0.16, 0.09, 0.05), rough=0.75), origin=(-sx * 6.75 / 2, 0, 4.5))
        dr.location = (sx * 6.8, -1.0, 8.0)
        frs = np.arange(1, fdoor + 80)
        ang = np.interp(frs, [fdoor, fdoor + 62], [0.0, math.radians(82)]) * -sx          # swing INWARD (+Y), away from him
        FX.key_path(dr, frs, np.repeat(np.array([[sx * 6.8, -1.0, 8.0]]), len(frs), 0),
                    rot_euler=np.c_[np.zeros(len(frs)), np.zeros(len(frs)), ang])
        dr.visible_shadow = True

    # ------------------------------------------------------------------ impact flashes + rings for body contacts
    for c in S.contacts:
        base = c.key.split("#")[0]
        if c.kind in ("sword", "talon_parry", "tail_pillar", "slam", "body_cover", "roof"):
            if c.kind == "sword":
                p = P("K.tip", c.f) * 0.5 + P("K.grip", c.f) * 0.5
            elif c.kind == "talon_parry":
                p = P("D.palmR", c.f)
            elif c.kind == "tail_pillar":
                p = P("D.tail_tip", c.f)
            elif c.kind == "slam":
                p = P("K.head", c.f)
            elif c.kind == "roof":
                p = P("D.head", c.f) + np.array([0, 0, 0.6])
            else:
                p = P("K.hips", c.f)
            FX.flash(S, c.key.replace("#", "_"), p, c.f, size=0.6 if c.kind != "roof" else 2.0,
                     color=(1.0, 0.95, 0.8) if c.kind in ("sword", "talon_parry") else (1.0, 0.7, 0.4))
    sl = f["slam"]
    FX.ring(S, "slam", [P("D.chest", sl)[0], P("D.chest", sl)[1], S.ground(*P("D.chest", sl)[:2]) + 0.1], sl + 2, 9.0)
    ld = f["lands"]
    FX.ring(S, "lands", [P("D.root", ld + 8)[0], P("D.root", ld + 8)[1], S.ground(*P("D.root", ld + 8)[:2]) + 0.1], ld + 8, 11.0)
    S.R.add("fx.hits", S.fx_hits)
    S.R.add("fx.objects", len(S.fx_bodies))
