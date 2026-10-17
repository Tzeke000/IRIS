"""s40 — contact pins: make every planned body contact actually happen ON its frame by nudging a root (smooth bump
before → hold → ease back). Projectile / shield / spikes / holy-sword / rocket contacts are exact by construction in
s70 and are only registered here.
  tail→pillar : dragon yaw about its root + radial translation so the tail tip lands on the pier face
  sword→dragon: knight translation so the blade's closest point passes through the target bone (combo 1-3, neck)
  talon parry : knight translation (xy) so the blocking blade meets the dragon's right talon
  body slam   : dragon xy over the knight + height so its underside meets the knight's head
  thrown      : knight lands against the fallen-column drums
  roof        : dragon height so the head meets the vault intrados
"""
import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from anim import common as C
from anim import contacts as CT
from anim import timeline as TL
from anim import track as TK
from anim.stages.s20_knight import swing_peaks
from lib import checks as CK


def bump(arr, fc, val, pre, hold, post):
    N = len(arr)
    fr = np.arange(1, N + 1, dtype=np.float64)
    up = C.smooth01((fr - (fc - pre)) / max(pre, 1))
    down = 1.0 - C.smooth01((fr - (fc + hold)) / max(post, 1))
    w = np.where(fr <= fc, up, np.where(fr <= fc + hold, 1.0, down))
    if arr.ndim == 2:
        arr += np.outer(w, np.asarray(val, np.float64))
    else:
        arr += w * float(val)


def rebake(S, who):
    X = S.K if who == "K" else S.D
    TL.bake_root(X.root, X.base_pos + X.off, X.base_yaw + X.yoff, S.N, getattr(X, "teleports", ()))


def bearing(v):
    return math.degrees(math.atan2(-v[0], v[1]))


def seg_closest(a, b, p):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    return a + ab * t, t


def obj_bvh(names):
    dg = bpy.context.evaluated_depsgraph_get()
    V, T = CK.bpy_world_tris([bpy.data.objects[n] for n in names], dg)
    return BVHTree.FromPolygons(V.tolist(), T.tolist())


def ev_bvh(S, f, objs):
    S.sc.frame_set(int(f))
    dg = bpy.context.evaluated_depsgraph_get()
    V, T = CK.bpy_world_tris(objs, dg)
    return BVHTree.FromPolygons(V.tolist(), T.tolist()), V


def first_overlap(S, f, objs_a, bvh_b_fn, back=8):
    """earliest frame in [f-back, f] where objs_a overlap bvh_b_fn(frame); None if none"""
    first = None
    for fr in range(f, f - back - 1, -1):
        a, _ = ev_bvh(S, fr, objs_a)
        if a.overlap(bvh_b_fn(fr)):
            first = fr
        else:
            break
    return first


def min_gap(VA, bvh_b):
    return min(bvh_b.find_nearest(Vector(p))[3] for p in VA[:: max(1, len(VA) // 2500)])


def first_contact(S, f, a_objs, b_fn, span=10):
    for fr in range(f - span, f + span + 1):
        a_, _ = ev_bvh(S, fr, a_objs)
        if a_.overlap(b_fn(fr)):
            return fr
    return None


def time_contact(S, f, a_objs, b_fn, set_delta, lo, hi, iters=9):
    """bisect a scalar delta in [lo, hi] (larger delta → LATER contact) so the first overlap lands on f or f-1.
    set_delta(d) must set the absolute extra offset (and re-bake). Returns (delta, first_frame)."""
    best = None
    for it in range(iters):
        mid = 0.5 * (lo + hi)
        set_delta(mid)
        fo = first_contact(S, f, a_objs, b_fn)
        S.R.log(f"  time_contact f{f} mid {mid:.2f} -> first {fo}")
        if fo is not None and f - 1 <= fo <= f:
            return mid, fo
        if fo is not None and fo < f - 1:
            lo = mid                    # too early → push later
        else:
            hi = mid                    # missing or late → pull earlier
        best = (mid, fo)
    return best


def proxy_tip(S, f):
    """the dragon proxy's vertex farthest (XY) from the root = the real mesh tail tip"""
    _b, V = ev_bvh(S, f, [S.D.proxy])
    R = np.array(S.D.root.matrix_world.translation)
    d = np.linalg.norm(V[:, :2] - R[:2], axis=1)
    return V[int(np.argmax(d))], R


def derived_hits(S):
    E, K = S.E, S.K
    ex = {}
    st = next(p for p in K.placed if p["clip"] == "Sword_Regular_Combo")
    pk = swing_peaks(K.clips["Sword_Regular_Combo"]["speed"], 3, rel=0.38)
    ex["combo"] = [int(round(st["start"] + p * st["scale"])) for p in pk[1:]]
    ex["fireballs"] = [E["fireballs"].f + 6]
    ex["rocks"] = [E["rocks"].f + 6, E["rocks"].f + 12]
    ex["dfireballs"] = [E["dfireballs"].f + 12]
    ex["roof_rain"] = [E["roof_rain"].f + 6, E["roof_rain"].f + 11]
    return ex


def run(S):
    S.extra_hits = derived_hits(S)
    S.contacts = CT.build(S.E, S.extra_hits)
    S.R.add("contacts.planned", [dict(key=c.key, f=c.f, a=c.a, b=c.b, kind=c.kind, derived=c.derived) for c in S.contacts])
    log = {}
    anchors = {"combo": "D.forearmL", "neck_slash": "D.neck"}
    for c in S.contacts:
        base = c.key.split("#")[0]
        f = c.f
        if c.kind == "tail_pillar":
            pier = c.b
            bvh = obj_bvh([f"{pier}_pier_INTACT"])
            pob = bpy.data.objects[f"{pier}_pier_INTACT"]
            bb = np.array([pob.matrix_world @ Vector(cn) for cn in pob.bound_box])
            lo, hi = bb.min(0), bb.max(0)
            spin = next(sp["spin"] for sp in S.D.plan["strikes"] if sp["key"] == base)
            sgn = 1.0 if spin > 0 else -1.0
            dpsi = dr = 0.0
            for it in range(5):
                T, R = proxy_tip(S, f)
                r = float(np.linalg.norm((T - R)[:2]))
                cp = np.array(bvh.find_nearest(Vector((R[0], R[1], T[2])))[0])
                v = (cp - R)[:2]
                dn = float(np.linalg.norm(v))
                if dn > r - 0.5:                       # the sweep circle must cut ≥0.5 m into the pier
                    dr = min(dn - (r - 0.6), 0.7)      # v2: the strike spot is planned; never slide the body > 0.7 m
                    u = v / dn
                    bump(S.D.off, f, [u[0] * dr, u[1] * dr, 0.0], 18, 8, 45)
                    rebake(S, "D")
                    S.R.log(f"tail {c.key} it{it}: reach step {dr:.2f} m")
                    continue
                th = bearing((T - R)[:2])
                entry = None
                for k in range(0, 481):             # nearest entry angle either way round the root
                    for sd in (1, -1):
                        phi = th + sd * k * 0.5
                        x, y = R[0] - r * math.sin(math.radians(phi)), R[1] + r * math.cos(math.radians(phi))
                        if bvh.find_nearest(Vector((x, y, T[2])), 0.12)[0] is not None:
                            entry = phi
                            break
                    if entry is not None:
                        break
                if entry is None:
                    S.R.log(f"tail {c.key} it{it}: NO ENTRY (r {r:.2f} dn {dn:.2f})")
                    break
                dpsi = (entry - th) + (2.0 if entry >= th else -2.0)
                S.R.log(f"tail {c.key} it{it}: th {th:.1f} entry {entry:.1f} dpsi {dpsi:.1f}")
                if abs(dpsi) < 2.6:
                    break
                bump(S.D.yoff, f, dpsi, 18, 8, 45)
                rebake(S, "D")
            cur = [0.0]

            def set_yaw(d, sgn=sgn, f=f, cur=cur):          # d > 0 = less spin = later contact
                bump(S.D.yoff, f, -sgn * (d - cur[0]), 18, 8, 45)
                cur[0] = d
                rebake(S, "D")
            res_t = time_contact(S, f, [S.D.proxy], lambda fr: bvh, set_yaw, -14.0, 14.0)
            log.setdefault("tail_timing", {})[c.key] = dict(extra_yaw=round(res_t[0], 2), first=res_t[1])
            P = TK.at(S, f)
            res = bvh.find_nearest(Vector(P["D.tail_tip"]))[3]
            log[c.key] = dict(dyaw=round(float(dpsi), 1), dr=round(float(dr), 2), residual_m=round(float(res), 3),
                              tip=P["D.tail_tip"].round(2).tolist())
        elif c.kind == "sword":
            anc = anchors[base]
            for it in range(4):
                P = TK.at(S, f)
                A = P[anc]
                u = (P["K.chest"] - A) * np.array([1, 1, 0])
                u = u / max(np.linalg.norm(u), 1e-6)
                A = A + u * (0.08 if base == "combo" else 0.06)
                B = P["K.grip"] + 0.95 * (P["K.tip"] - P["K.grip"])
                t = 0.95
                d = A - B
                if base != "neck_slash":
                    d[2] = 0.0
                n = np.linalg.norm(d)
                if n > 1.6:
                    S.R.log(f"WARN contact {c.key}: blade {n:.2f} m from {anc}; clamped to 1.6")
                    d *= 1.6 / n
                if n < 0.03:
                    break
                bump(S.K.off, f, d, 12, 6, 36)
                rebake(S, "K")
            P = TK.at(S, f)
            B, t = seg_closest(P["K.grip"], P["K.tip"], P[anc])
            log[c.key] = dict(delta=d.round(2).tolist(), residual_m=round(float(np.linalg.norm(P[anc] - B)), 3), blade_t=round(t, 2))
        elif c.kind == "talon_parry":
            for it in range(3):
                P = TK.at(S, f)
                T = P["D.palmR"]
                B = P["K.grip"] + 0.92 * (P["K.tip"] - P["K.grip"])
                d = B - T
                if np.linalg.norm(d) < 0.03:
                    break
                bump(S.D.off, f, d, 14, 4, 30)
                rebake(S, "D")
            curz = [0.0]

            def set_z(dz, f=f, curz=curz):
                bump(S.D.off, f, [0, 0, dz - curz[0]], 14, 4, 30)
                curz[0] = dz
                rebake(S, "D")
            res_t = time_contact(S, f, [S.D.proxy], lambda fr: ev_bvh(S, fr, [S.K.sword])[0], set_z, -0.8, 0.8)
            P = TK.at(S, f)
            B, t = seg_closest(P["K.grip"], P["K.tip"], P["D.palmR"])
            log[c.key] = dict(delta=d.round(2).tolist(), residual_m=round(float(np.linalg.norm(P["D.palmR"] - B)), 3),
                              dz=round(float(res_t[0]), 2), first=res_t[1])
        elif c.kind == "slam":
            P = TK.at(S, f)
            H = P["K.head"] + np.array([0, 0, 0.15])
            dxy = (H - P["D.chest"])[:2]
            bump(S.D.off, f, [dxy[0], dxy[1], 0.0], 30, 6, 40)
            rebake(S, "D")
            S.sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            V, _T = CK.bpy_world_tris([S.D.proxy], dg)
            P = TK.at(S, f)
            H = P["K.head"] + np.array([0, 0, 0.15])
            near = V[np.linalg.norm(V[:, :2] - H[:2], axis=1) < 0.9]
            under = float(near[:, 2].min()) if len(near) else float(P["D.chest"][2] - 0.9)
            dz = (H[2] - 0.06) - under
            bump(S.D.off, f, [0, 0, dz], 10, 3, 30)
            rebake(S, "D")
            for it in range(4):                       # close the real proxy-to-proxy gap (contact ON the frame)
                kb, _ = ev_bvh(S, f, [S.K.proxy])
                db, VD = ev_bvh(S, f, [S.D.proxy])
                if kb.overlap(db):
                    break
                g = min_gap(VD, kb)
                bump(S.D.off, f, [0, 0, -(g + 0.06)], 10, 3, 30)
                rebake(S, "D")
                dz -= g + 0.06
            log[c.key] = dict(dxy=dxy.round(2).tolist(), dz=round(dz, 2), underside_before=round(under, 2))
        elif c.kind == "body_cover":
            P = TK.at(S, f)
            from anim.staging import SPOT
            tgt = np.array([SPOT["cover_top"][0], SPOT["cover_top"][1], 8.93 + 0.25])
            d = tgt - P["K.hips"]
            bump(S.K.off, f, d, 6, 1, 8)
            rebake(S, "K")
            log[c.key] = dict(delta=d.round(2).tolist())
        elif c.kind == "roof":
            rc = bpy.data.collections["ROOF_breakable"].all_objects        # v2: ANY roof part counts (ribs, slates, timber)
            bvh = obj_bvh([o.name for o in rc if o.type == "MESH" and ("_INTACT" in o.name or "Truss" in o.name
                                                                      or "Purlin" in o.name)])
            P = TK.at(S, f)
            Hd = P["D.head"]
            hit = bvh.ray_cast(Vector(Hd), Vector((0, 0, 1)), 60)
            if hit[0] is None:
                S.R.log(f"WARN roof: no vault above the head at {Hd.round(2)}")
                continue
            dz = hit[0].z - (Hd[2] + 0.45)
            bump(S.D.off, f, [0, 0, dz], 40, 2, 50)
            rebake(S, "D")
            for it in range(4):
                db, VD = ev_bvh(S, f, [S.D.proxy])
                if db.overlap(bvh):
                    _db1, VD1 = ev_bvh(S, f - 1, [S.D.proxy])
                    if not _db1.overlap(bvh):
                        break
                    d = -0.25                         # already touching one frame early: back off
                else:
                    gaps = [bvh.ray_cast(Vector(p), Vector((0, 0, 1)), 20)[3] or 20 for p in VD[::40]]
                    d = min(gaps) + 0.05
                bump(S.D.off, f, [0, 0, d], 40, 2, 50)
                rebake(S, "D")
                dz += d
            log[c.key] = dict(dz=round(dz, 2), vault_z=round(hit[0].z, 2))
    # separation pass: in close-combat windows, any knight-body/dragon-body overlap that is NOT a planned contact gets
    # a short local push of the knight away from the dragon (sword-hit frames are left alone)
    hit_frames = {c.f for c in S.contacts if c.kind in ("sword", "talon_parry")}
    E = S.E
    pushes = []
    for a, b in ((E["combo"].f - 20, E["combo"].f + 50), (E["parry"].f - 20, E["parry"].f + 20),
                 (E["neck_slash"].f - 20, E["neck_slash"].f + 20)):
        for it in range(6):
            bad = None
            for fr in range(a, b + 1):
                kb, _ = ev_bvh(S, fr, [S.K.proxy])
                db, _ = ev_bvh(S, fr, [S.D.proxy])
                if kb.overlap(db):
                    bad = fr
                    break
            if bad is None:
                break
            P = TK.at(S, bad)
            u = (P["K.chest"] - P["D.chest"]) * np.array([1, 1, 0])
            u /= max(np.linalg.norm(u), 1e-6)
            pre = max(1, min(4, min([abs(bad - h) for h in hit_frames] + [4]) - 1))
            bump(S.K.off, bad, u * 0.18, pre, 1, pre)
            rebake(S, "K")
            pushes.append((bad, pre))
    log["separation_pushes"] = pushes
    S.R.add("contacts.solve", log)
