"""s90 — the NO-PASS-THROUGH CONTRACT, audited per frame with physics OFF (no rigid-body world exists any more; debris
is plain keys). Bodies: knight proxy, blade, shield sphere, dragon proxy, every FX projectile/spike/breath/holy sword,
every released debris cell, the intact piers (until they break) and the static solids (walls, gate, cover, plinths,
vault, colossi). Every overlap must sit inside a registered contact window for that pair, else it is a VIOLATION.
Every registered contact must really start within ±1 frame of its planned frame, else MISSED/EARLY/LATE.
Debris↔character overlaps are the simulation's own contacts: auto-registered as kind 'rubble' (listed, not failed).
Also: intended near-misses (bites) keep ≥0.25 m clearance; feet grounding; camera clearance."""
import json
import math
import os
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from anim import common as C
from lib import checks as CK


class StaticSet:
    """one BVH over many static objects, triangle → object name"""

    def __init__(self, objs, dg):
        V, T, owner = [], [], []
        off = 0
        for o in objs:
            v, t = CK.bpy_world_tris([o], dg)
            if not len(t):
                continue
            V.append(v)
            T.append(t + off)
            owner += [o.name] * len(t)
            off += len(v)
        self.owner = owner
        self.V = np.concatenate(V) if V else None
        self.T = np.concatenate(T) if T else None
        self.bvh = BVHTree.FromPolygons(self.V.tolist(), self.T.tolist()) if V else None
        self.last_loc = {}

    def hits(self, other):
        if self.bvh is None or other is None:
            return set()
        out = set()
        for i, _ in self.bvh.overlap(other):
            n = COVER_A.get(self.owner[i], self.owner[i])
            if n not in out:
                self.last_loc[n] = self.V[self.T[i]].mean(0)
            out.add(n)
        return out


COVER_A = {f"CV_S1W_Drum{i}": "cover_A" for i in range(3)}


def part_near(S, fr, p, who="D"):
    best, bd = None, 1e9
    for k, v in S.trk.items():
        if k.startswith(who + ".") and k not in ("D.root", "K.root"):
            d = float(np.linalg.norm(v[fr - 1] - p))
            if d < bd:
                best, bd = k, d
    return f"{best}({bd:.1f}m)"


def mesh_bvh(objs, dg):
    V, T = CK.bpy_world_tris(objs, dg)
    if not len(T):
        return None
    return BVHTree.FromPolygons(V.tolist(), T.tolist())


def sphere_hits(bvh, c, r):
    if bvh is None or r < 0.05:
        return False
    h = bvh.find_nearest(Vector(c), r)
    return h[0] is not None


def run(S):
    sc, N = S.sc, S.N
    end = S.args.slice_end or N
    step = max(1, S.args.audit_step)
    assert sc.rigidbody_world is None, "physics must be OFF for the audit"
    dg = bpy.context.evaluated_depsgraph_get()
    # ---- static sets
    solid_objs = list(S.static_solids) + [bpy.data.objects[n] for n in ("Colossus_W",)]
    solid = StaticSet(solid_objs, dg)
    piers = {}
    for side in "EW":
        for i in range(1, 5):
            nm = f"Pier_{side}{i}"
            piers[nm] = StaticSet([bpy.data.objects[f"{nm}_pier_INTACT"]], dg)
    vault = StaticSet([bpy.data.objects["Vault_INTACT"]], dg)
    roof_f = S.E["roof"].f
    # ---- windows by unordered pair
    win = defaultdict(list)
    for c in S.contacts:
        lo, hi = c.window
        win[frozenset((c.a, c.b))].append((lo, hi, c.key))

    def allowed(a, b, fr):
        return any(lo <= fr <= hi for lo, hi, _ in win.get(frozenset((a, b)), ()))

    fx = S.fx_bodies                                   # obj name -> body/owner/kind
    fx_objs = {n: bpy.data.objects[n] for n in fx}
    deb_objs = {n: bpy.data.objects[n] for n in S.debris}
    over = defaultdict(list)                           # (a,b) -> [frames]
    viol = []
    rubble = {}
    t_start = __import__("time").time()

    def vis(ob):
        return not ob.evaluated_get(dg).hide_render

    where = {}

    def note(a, b, fr, loc=None):
        over[(a, b)].append(fr)
        if not allowed(a, b, fr):
            viol.append((fr, a, b))
            if (a, b) not in where and loc is not None:
                where[(a, b)] = (fr, np.round(loc, 2).tolist(), part_near(S, fr, loc, "D") if "D." in a + b else part_near(S, fr, loc, "K"))

    for fr in range(1, end + 1, step):
        sc.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        kb = mesh_bvh([S.K.proxy], dg)
        db = mesh_bvh([S.D.proxy], dg)
        sw = mesh_bvh([S.K.sword], dg) if vis(S.K.sword) else None
        shr = float(S.shield_radius[fr - 1])
        shc = S.shield_center[fr - 1]
        sh_on = vis(bpy.data.objects["K_Shield"]) and shr > 0.3
        # characters
        ov = kb.overlap(db)
        if ov:
            note("K.body", "D.body", fr, np.array(S.trk["K.chest"][fr - 1]))
        if sw is not None:
            ov = sw.overlap(db)
            if ov:
                note("K.sword", "D.body", fr, np.array(S.trk["K.tip"][fr - 1]))
        if sh_on and sphere_hits(db, shc, shr):
            note("K.shield", "D.body", fr)
        for who, b in (("K.body", kb), ("D.body", db)):
            for o in solid.hits(b):
                note(who, o if o == "cover_A" else f"static:{o}", fr, solid.last_loc.get(o))
            for pn, ps in piers.items():
                if fr <= S.pier_swap.get(pn, 10 ** 9) and ps.hits(b):     # the contact frame IS the swap frame
                    note(who, pn, fr, ps.last_loc.get(f"{pn}_pier_INTACT"))
            if fr < roof_f + 1 and vault.hits(b):
                note(who, "vault", fr, vault.last_loc.get("Vault_INTACT"))
        # FX bodies
        for n, ob in fx_objs.items():
            meta = fx[n]
            if meta["kind"] == "shield" or not vis(ob):
                continue
            bb = mesh_bvh([ob], dg)
            if bb is None:
                continue
            body = meta["body"]
            if meta["owner"] != "D" and bb.overlap(db):
                note(body, "D.body", fr)
            if meta["owner"] != "K" and bb.overlap(kb):
                note(body, "K.body", fr)
            if meta["owner"] != "K" and sh_on and sphere_hits(bb, shc, shr):
                note(body, "K.shield", fr)
            for pn, ps in piers.items():
                if fr <= S.pier_swap.get(pn, 10 ** 9) and ps.hits(bb):
                    note(body, pn, fr)
            for o in solid.hits(bb):
                note(body, o if o == "cover_A" else f"static:{o}", fr, solid.last_loc.get(o))
            arm = deb_objs.get("colossus_arm")
            if arm is not None and vis(arm):
                ab = mesh_bvh([arm], dg)
                if ab is not None and bb.overlap(ab):
                    note(body, "colossus_arm", fr)
        # debris
        for n, ob in deb_objs.items():
            if not vis(ob):
                continue
            meta = S.debris[n]
            if fr < meta["release"]:
                continue
            bb = mesh_bvh([ob], dg)
            if bb is None:
                continue
            body = n if meta["group"] == "roof_chunks" else (
                "ceiling_chunk_1" if meta["group"] == "ceiling_chunk" else ("colossus_arm" if meta["group"] == "colossus_arm" else n))
            for who, b in (("K.body", kb), ("D.body", db)):
                if bb.overlap(b):
                    key = (n, who)
                    if body.startswith(("roof_chunk_", "ceiling_chunk_")) and allowed(body, who, fr):
                        continue
                    if key not in rubble:
                        rubble[key] = dict(first=fr, last=fr, n=0)
                    rubble[key]["last"] = fr
                    rubble[key]["n"] += 1
            if sh_on and sphere_hits(bb, shc, shr):
                note(body, "K.shield", fr)
        if fr % 500 == 0:
            S.R.log(f"audit frame {fr}/{end}  overlaps {sum(len(v) for v in over.values())}  violations {len(viol)}")
    S.R.add("audit.seconds", round(__import__("time").time() - t_start, 1))
    def gap(c):
        """closest distance between the two bodies on the planned frame (diagnostic for MISSED contacts)"""
        sc.frame_set(c.f)
        dg2 = bpy.context.evaluated_depsgraph_get()

        def verts(name):
            if name == "K.body":
                return CK.bpy_world_tris([S.K.proxy], dg2)[0]
            if name == "D.body":
                return CK.bpy_world_tris([S.D.proxy], dg2)[0]
            if name == "K.sword":
                return CK.bpy_world_tris([S.K.sword], dg2)[0]
            if name == "K.shield":
                c0, r = S.shield_center[c.f - 1], float(S.shield_radius[c.f - 1])
                u = np.random.default_rng(0).normal(size=(400, 3))
                return c0 + r * u / np.linalg.norm(u, axis=1, keepdims=True)
            if name == "cover_A":
                return CK.bpy_world_tris([bpy.data.objects[f"CV_S1W_Drum{i}"] for i in range(3)], dg2)[0]
            if name == "vault":
                return CK.bpy_world_tris([bpy.data.objects["Vault_INTACT"]], dg2)[0]
            ob = bpy.data.objects.get(f"{name}_pier_INTACT") or bpy.data.objects.get(name)
            return CK.bpy_world_tris([ob], dg2)[0] if ob is not None else None
        A, B = verts(c.a), verts(c.b)
        if A is None or B is None or not len(A) or not len(B):
            return None
        from mathutils.kdtree import KDTree
        kd = KDTree(len(B))
        for i, p in enumerate(B):
            kd.insert(Vector(p), i)
        kd.balance()
        return round(min(kd.find(Vector(p))[2] for p in A[:: max(1, len(A) // 3000)]), 3)

    # ---- planned contacts: first contact within ±1 frame
    timing = []
    for c in S.contacts:
        frs = sorted(set(over.get((c.a, c.b), []) + over.get((c.b, c.a), [])))
        near = [x for x in frs if c.f - 6 <= x <= c.f + 6]
        if c.kind == "breath":
            near = [x for x in frs if c.f - 12 <= x <= c.f + 6]
        first = near[0] if near else None
        ok = (first is not None and abs(first - c.f) <= (1 if c.kind != "breath" else 10)) or (c.optional and first is None)
        timing.append(dict(key=c.key, planned=c.f, first=first, ok=bool(ok), pair=f"{c.a}<->{c.b}", kind=c.kind,
                           derived=c.derived, gap_m=None if first is not None else gap(c)))
    # ---- intended near-misses
    nm = []
    for key in ("bite_dodge", "bite_miss"):
        fe = S.E[key].f
        dmin = 1e9
        for fr in range(fe - 8, fe + 9):
            sc.frame_set(fr)
            dg = bpy.context.evaluated_depsgraph_get()
            kb = mesh_bvh([S.K.proxy], dg)
            for p in (S.trk["D.snout"][fr - 1], S.trk["D.jaw"][fr - 1], S.trk["D.head"][fr - 1]):
                h = kb.find_nearest(Vector(p))
                if h[0] is not None:
                    dmin = min(dmin, h[3])
        nm.append(dict(key=key, frame=fe, min_clearance_m=round(float(dmin), 3), ok=bool(dmin >= 0.25)))
    # ---- grounding (feet vs ground) outside airborne windows
    E = S.E
    air_k = [(E["neck_slash"].f - 14, E["neck_slash"].f + 14), (E["slam"].f, E["thrown"].f + 10),
             (E["thrown"].f + 8, E["gem_out"].f + 30)]
    air_d = [(1, E["rise_roar"].f + 30), (E["slam"].f - 44, E["slam"].f + 4), (E["launch"].f - 14, E["lands"].f + 10),
             (E["eye_again"].f + 20, 7412)]
    gk, gd = [], []
    for fr in range(1, end + 1, 3):
        if not any(a <= fr <= b for a, b in air_k) and fr != 7000:
            lf, rf = S.trk["K.lfoot"][fr - 1], S.trk["K.rfoot"][fr - 1]
            low = lf if lf[2] < rf[2] else rf
            gk.append(low[2] - S.ground(low[0], low[1]))
        if not any(a <= fr <= b for a, b in air_d):
            fl, fr_ = S.trk["D.footL"][fr - 1], S.trk["D.footR"][fr - 1]
            low = fl if fl[2] < fr_[2] else fr_
            gd.append(low[2] - S.ground(low[0], low[1]))
    gk, gd = np.array(gk), np.array(gd)
    grounding = dict(knight_ankle_above_ground_m=dict(p5=round(float(np.percentile(gk, 5)), 3), median=round(float(np.median(gk)), 3),
                                                      p95=round(float(np.percentile(gk, 95)), 3), below_minus5cm=int((gk < -0.05).sum())),
                     dragon_foot_above_ground_m=dict(p5=round(float(np.percentile(gd, 5)), 3), median=round(float(np.median(gd)), 3),
                                                     p95=round(float(np.percentile(gd, 95)), 3), below_minus10cm=int((gd < -0.10).sum())))
    # ---- camera clearance
    cam_bad = []
    cam = getattr(S, "cam", None)
    if cam is not None:
        bvh_static = solid.bvh
        pos = S.cam_pos
        bad = CK.bpy_camera_clear(pos[:end], bvh_static, 0.3)
        cam_bad = [(int(i + 1), round(d, 2)) for i, d in bad]
    # ---- summarise
    vgroups = defaultdict(list)
    for fr, a, b in viol:
        vgroups[(a, b)].append(fr)
    vlist = []
    for (a, b), frs in sorted(vgroups.items(), key=lambda kv: kv[1][0]):
        runs = []
        for x in frs:
            if runs and x - runs[-1][1] <= step:
                runs[-1][1] = x
            else:
                runs.append([x, x])
        vlist.append(dict(a=a, b=b, frames=len(frs), runs=runs[:12], first_where=where.get((a, b))))
    rub = [dict(debris=k[0], body=k[1], **v) for k, v in sorted(rubble.items(), key=lambda kv: kv[1]["first"])]
    for r in rub:
        S.contacts.append(C.Contact(f"rubble:{r['debris']}", r["first"], r["debris"], r["body"], "rubble", derived=True,
                                    note="auto-registered from the rigid-body sim"))
    audit = dict(frames=end, step=step, violations=vlist, n_violation_frames=len(viol), timing=timing,
                 near_misses=nm, rubble_auto=rub, grounding=grounding, camera_too_close=cam_bad[:40],
                 n_camera_too_close=len(cam_bad))
    S.R.add("audit", audit)
    with open(os.path.join(C.OUT_DIR, "fight_v2_audit.json"), "w") as fh:
        json.dump(audit, fh, indent=1)
    miss = [t for t in timing if not t["ok"]]
    S.R.check(not vlist, "audit_no_unregistered_overlap", f"{len(vlist)} pairs / {len(viol)} frames")
    S.R.check(not miss, "audit_contacts_on_frame", f"{len(miss)} of {len(timing)} off-frame/missed: "
              + ", ".join(f"{t['key']}@{t['planned']}->{t['first']}" for t in miss[:12]))
    S.R.check(all(x["ok"] for x in nm), "audit_bite_near_misses_clear", json.dumps(nm))
    S.R.check(not cam_bad, "audit_camera_clear", f"{len(cam_bad)} frames")
    if not S.args.no_save and not getattr(S.args, "audit_soft", False):
        S.R.assert_(not vlist and not miss, "NO_PASS_THROUGH_CONTRACT",
                    f"{len(vlist)} unregistered overlap pairs, {len(miss)} contacts off-frame — see fight_v2_audit.json")
