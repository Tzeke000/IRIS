"""STRICT physical audit of a saved boss-fight blend (v1 or v2) — the checker that v1's contact audit should have been.

    scene.sh /ABS/anim/audit_v1.env build      (blender -b fight_vN.blend --python audit_strict.py -- --meta M --out OUT)

v1's audit only tested BETWEEN bodies and auto-registered sim debris contacts, so it was blind to what Zeke saw.
This one adds, per frame, physics OFF, on the saved file (nothing from the build context is needed):
 1 SELF-INTERSECTION per rig, on the EVALUATED deformed audit proxies (PX_Knight / PX_Dragon carry the skin weights):
   every vertex gets the part of its dominant bone; for each tested pair (A → B) the signed depth of A's vertices
   behind B's surface (nearest triangle + its normal) must stay under a threshold — knight hands/forearms vs
   torso/head/legs, legs vs legs; dragon head/jaw vs neck/body/limbs, limbs vs body/limbs; wings L × R = any
   triangle crossing (membranes have no inside); the drawn blade vs the knight's own body.
 2 PHYSICAL PLAUSIBILITY: mass of every moving debris cell from its closed-mesh volume × 2300 kg/m³. Heavy (≥ 60 kg)
   cells may NEVER overlap the knight's body (only the shield may stop them) and may never be moved while touching
   him (he can't scoot 100 kg of stone); light cells may only graze him while flying (≤ 3 frames). Debris resting
   inside the dragon is flagged. NOTHING is auto-registered.
   Static obstacles = EVERY render-visible mesh near a character (not only a curated list): characters vs any
   obstacle ≥ 40 kg (rubble pieces, pillar stumps, cover, props, buildings) is a violation unless a registered
   contact covers that pair/frame. Gate doors / swinging props vs characters: always a violation.
 3 FEET: planted feet may not slide (pivot point horizontal speed while in contact), characters may not glide (root
   moving while both feet stay planted), no floating outside airborne windows, facing vs travel (side/back travel
   only with a strafe/back-step clip on the base tracks), dragon ground spins (> 120° yaw inside 20 frames).
 4 HELD PROPS: a carried prop must stay in the hand (hand-to-prop distance) and must not rotate against the hand.
Meta (--meta JSON): contacts [{a,b,lo,hi,key}], air_k / air_d / glide_ok_k [[lo,hi]], held [{prop,hands,lo,hi}],
side_clips (substrings), audit range. For v1 the meta is derived from fight_v1_build_report.json (--v1-report).
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
DENS = 2300.0

# ------------------------------------------------------------------------------------------------ rig parts
def knight_part(bone):
    b = re.sub(r"_\d+$", "", bone.split(":")[-1])
    for side, S in (("Left", "L"), ("Right", "R")):
        if b.startswith(side + "Hand"):
            return "hand" + S
        if b == side + "ForeArm":
            return "farm" + S
        if b == side + "Arm":
            return "uarm" + S
        if b == side + "UpLeg":
            return "thigh" + S
        if b == side + "Leg":
            return "shin" + S
        if b.startswith(side + "Foot") or b.startswith(side + "Toe"):
            return "foot" + S
    if b in ("Neck", "Head", "HeadTop_End"):
        return "head"
    return "torso"


def dragon_part(bone):
    b = re.sub(r"_\d+$", "", bone).replace("DEF-", "")
    if b.startswith(("neck.004", "Teeth", "Bone", "eye", "lid")):
        return "head"
    if b.startswith("neck"):
        return "neck"
    if b.startswith("tail"):
        n = int(b.split(".")[1]) if "." in b else 0
        return "tailA" if n <= 5 else "tailB"
    if b.startswith("Wing_") and not b.startswith("Wing_Back"):
        return "wing" + b[-1]
    m = re.match(r".*\.([LR])(\.\d+)?$", b)
    side = m.group(1) if m else None
    # limbs: only the DISTAL segments are tested — the upper arm / thigh skin merges into the body by design
    # (a standing dragon's shoulders flagged as 'limb in body' over thousands of frames, visibly fine)
    if side and b.startswith(("thigh", "Hip")):
        return "legU" + side
    if side and b.startswith(("shin", "foot", "toe")):
        return "leg" + side
    if side and (b.startswith(("Finger", "palm")) and m.group(2)):       # hind-foot digits carry .L.001/.L.002
        return "leg" + side
    if side and b.startswith(("upper_arm", "collar")):
        return "armU" + side
    if side and b.startswith(("forearm", "Finger", "palm")):
        return "arm" + side
    return "body"


# KNIGHT: bone CAPSULES (his armour is many layered, partly single-sided shells — surface normals lie, so a
# signed-distance test fired on idle arms). Each part = a polyline through its bone heads; radius = median distance
# of the part's own vertices to it (measured on the first audited frame). A vertex of A is INSIDE B when it is closer
# to B's axis than k × r_B (k 0.6 = well inside the body, not skin touching skin).
KNIGHT_CAPS = {"torso": ["Hips", "Spine", "Spine1", "Spine2", "Neck"], "head": ["Neck", "Head", "HeadTop_End"]}
for _s in ("Left", "Right"):
    _S = _s[0]
    KNIGHT_CAPS.update({"uarm" + _S: [_s + "Arm", _s + "ForeArm"], "farm" + _S: [_s + "ForeArm", _s + "Hand"],
                        "hand" + _S: [_s + "Hand", _s + "HandMiddle2"], "thigh" + _S: [_s + "UpLeg", _s + "Leg"],
                        "shin" + _S: [_s + "Leg", _s + "Foot"], "foot" + _S: [_s + "Foot", _s + "ToeBase", _s + "Toe_End"]})
# (A, [B...], k, min_verts, label)
KNIGHT_TESTS = [
    ("handL", ["torso", "head", "thighL", "thighR"], 0.6, 4, "hand_in_body"),
    ("handR", ["torso", "head", "thighL", "thighR"], 0.6, 4, "hand_in_body"),
    ("farmL", ["torso", "head", "thighL", "thighR"], 0.6, 4, "forearm_elbow_in_body"),
    ("farmR", ["torso", "head", "thighL", "thighR"], 0.6, 4, "forearm_elbow_in_body"),
    ("farmL", ["farmR", "uarmR"], 0.5, 4, "arms_cross"), ("handL", ["farmR"], 0.5, 4, "arms_cross"),
    ("handR", ["farmL"], 0.5, 4, "arms_cross"),
    ("shinL", ["shinR", "thighR"], 0.55, 4, "legs_cross"), ("thighL", ["thighR"], 0.5, 4, "legs_cross"),
    ("footL", ["footR", "shinR"], 0.55, 4, "legs_cross"),
]
DRAGON_TESTS = [
    ("head", ["body", "armL", "armR", "armUL", "armUR", "legL", "legR", "tailA"], 0.12, 6, "head_in_body"),
    # head → neck is NOT tested on the surface (the open junction ring gives false hits at REST); a head folded back
    # into its own neck shows up as head → body/limbs/tailA or in the knight-style capsule test below
    ("armL", ["armR"], 0.10, 6, "forelegs_cross"), ("legL", ["legR"], 0.10, 6, "hindlegs_cross"),
    ("armL", ["body", "legL", "legUL"], 0.18, 8, "limb_in_body"), ("armR", ["body", "legR", "legUR"], 0.18, 8, "limb_in_body"),
    ("legL", ["body"], 0.18, 8, "limb_in_body"), ("legR", ["body"], 0.18, 8, "limb_in_body"),
    ("tailB", ["body", "legL", "legR", "armL", "armR", "head", "neck"], 0.15, 8, "tail_in_body"),
]
WING_CROSS_MIN = 3          # triangle pairs (L × R wings) — membranes may never cross

SUPPORT = ["ArenaFloor", "GrandStairs", "CourtyardPaving", "Dais", "GoldHeap", "Platform"]
SKIP_COLL = ("ANIM_CHARACTERS", "ANIM_PROXIES", "ANIM_FX", "ANIM_PROJECTILES", "ANIM_CAMERAS", "FX_haze", "V3_EXT_FOG",
             "V3_EXT_TEMPLATES", "V3C_GRASS", "V3C_GRASS_TEMPLATES", "V3_EXT_WOODS", "V3_EXT_FARMS", "V3_GRAVEL",
             "SIZE_PROOF", "RB_World", "V2_LIGHTS", "V3_EXT_LIGHTS", "V3_EXT_ROAD")
SOFT = ("Coin", "Gem", "Candle", "Flame", "Haze", "Fog", "Mist", "Grass", "Gravel", "Moss", "Ivy", "Banner", "Cloth",
        "Glow", "Light", "Terrain", "Ground", "Sky", "Moon", "Cobweb", "Web", "Puddle")
HEAVY_KG = 60.0
OBST_KG = 40.0


# ------------------------------------------------------------------------------------------------ helpers
def world_verts(ob, dg):
    oe = ob.evaluated_get(dg)
    me = oe.to_mesh()
    try:
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        M = np.array(oe.matrix_world)
        return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    finally:
        oe.to_mesh_clear()


def tris_of(me):
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", t)
    return t.reshape(-1, 3)


def mesh_volume(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    try:
        v = abs(bm.calc_volume(signed=False))
    finally:
        bm.free()
    s = ob.matrix_world.to_scale()
    v *= abs(s[0] * s[1] * s[2])
    d = np.array(ob.dimensions)
    box = float(np.prod(np.maximum(d, 1e-3)))
    if not np.isfinite(v) or v <= 0 or v > box * 1.05:          # open / odd mesh: half its box
        v = 0.5 * box
    return v


def runs(frames, gap):
    out = []
    for x in sorted(frames):
        if out and x - out[-1][1] <= gap:
            out[-1][1] = x
        else:
            out.append([x, x])
    return out


def poly_dist(P, pts):
    """distance of points P (n,3) to the polyline through pts"""
    best = np.full(len(P), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        L2 = max(float(ab @ ab), 1e-12)
        t = np.clip((P - a) @ ab / L2, 0.0, 1.0)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


class Rig:
    def __init__(self, px, arm, partfn, tests, caps=None):
        self.px, self.arm = px, arm
        self.caps = caps
        me = px.data
        names = {g.index: g.name for g in px.vertex_groups}
        n = len(me.vertices)
        part = np.empty(n, dtype=object)
        for v in me.vertices:
            best, bw = None, -1
            for g in v.groups:
                if g.weight > bw and g.group in names:
                    best, bw = names[g.group], g.weight
            part[v.index] = partfn(best) if best else "torso"
        self.part = part
        self.T = tris_of(me)
        maj = [max(set(part[t]), key=list(part[t]).count) for t in self.T]
        self.tri_part = np.array(maj, dtype=object)
        self.tests = tests
        self.vidx = {p: np.nonzero(part == p)[0] for p in set(part)}
        self.tidx = {p: np.nonzero(self.tri_part == p)[0] for p in set(self.tri_part)}

    def bvh(self, V, parts):
        ids = np.concatenate([self.tidx.get(p, np.zeros(0, int)) for p in parts])
        if not len(ids):
            return None
        return BVHTree.FromPolygons(V.tolist(), self.T[ids].tolist())

    def cap_init(self, V, pts):
        self.cap_r = {}
        for p, chain in self.caps.items():
            ia = self.vidx.get(p)
            if ia is None or not len(ia):
                continue
            self.cap_r[p] = float(np.median(poly_dist(V[ia], [pts[b] for b in chain])))

    def cap_hits(self, V, pts):
        out = []
        for A, Bs, k, nmin, label in self.tests:
            ia = self.vidx.get(A)
            if ia is None or not len(ia):
                continue
            P = V[ia]
            deep = np.zeros(len(P), bool)
            dmax = 0.0
            for B in Bs:
                if B not in self.cap_r:
                    continue
                d = poly_dist(P, [pts[b] for b in self.caps[B]])
                lim = k * self.cap_r[B]
                m = d < lim
                deep |= m
                if m.any():
                    dmax = max(dmax, float((self.cap_r[B] - d[m]).max()))
            if deep.sum() >= nmin:
                out.append((label, f"{A}->{'+'.join(Bs)}", int(deep.sum()), round(dmax, 3)))
        return out

    def self_hits(self, V):
        out = []
        cache = {}
        for A, Bs, dthr, nmin, label in self.tests:
            key = tuple(Bs)
            if key not in cache:
                cache[key] = self.bvh(V, Bs)
            bv = cache[key]
            ia = self.vidx.get(A)
            if bv is None or ia is None or not len(ia):
                continue
            ia = ia[:: max(1, len(ia) // 250)]
            deep, dmax = 0, 0.0
            for p in V[ia]:
                loc, nrm, _i, dist = bv.find_nearest(Vector(p), dthr * 6)
                if loc is None:
                    continue
                s = (Vector(p) - loc).dot(nrm)
                if s < -dthr:
                    deep += 1
                    dmax = max(dmax, -s)
            if deep >= nmin:
                out.append((label, f"{A}->{'+'.join(Bs)}", deep, round(dmax, 3)))
        return out


# ------------------------------------------------------------------------------------------------ meta for v1
def meta_from_v1(report_path, choreo_path):
    sys.path.insert(0, os.path.dirname(HERE))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    from anim import common as C
    E = C.load_events(choreo_path)
    rep = json.load(open(report_path))["values"]
    cons = []
    for t in rep["audit"]["timing"]:
        a, b = t["pair"].split("<->")
        lo, hi = C.WINDOWS[t["kind"]]
        cons.append(dict(a=a, b=b, lo=t["planned"] - lo, hi=t["planned"] + hi, key=t["key"]))
    f = {k: e.f for k, e in E.items()}
    air_k = [(f["neck_slash"] - 14, f["neck_slash"] + 14), (f["slam"], f["thrown"] + 10), (6999, 7001)]
    sit_k = [(f["thrown"] + 8, f["gem_out"] + 30)]                              # sitting: feet not tested
    air_d = [(1, f["rise_roar"] + 30), (f["slam"] - 44, f["slam"] + 4), (f["launch"] - 14, f["lands"] + 10),
             (f["eye_again"] + 20, 7412)]
    held = [dict(prop="PRIZE_GoldRPG7", hands=["L", "R"], lo=7176, hi=7628),
            dict(prop="FX_gem", hands=["R"], lo=f["gem_out"] + 32, hi=f["gem_crush"] - 2),
            dict(prop="K_Sword", hands=["R"], lo=f["draw"] + 2, hi=f["outro_walk"] - 2)]
    return dict(version="v1", contacts=cons, air_k=air_k, sit_k=sit_k, air_d=air_d, held=held,
                thrown_k=[(f["slam"], f["thrown"] + 10)], side_clips=["Strafe", "Backward", "Back_", "Side"],
                events={k: v for k, v in f.items()})


# ------------------------------------------------------------------------------------------------ main audit
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="")
    ap.add_argument("--v1-report", default="")
    ap.add_argument("--choreo", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--from", dest="f0", type=int, default=1)
    ap.add_argument("--to", dest="f1", type=int, default=0)
    ap.add_argument("--rest", action="store_true", help="calibration: both rigs in REST position (must be clean)")
    ap.add_argument("--dump", default="", help="calibration: write flagged vertices per test for these frames (csv)")
    a = ap.parse_args(argv)
    meta = json.load(open(a.meta)) if a.meta else meta_from_v1(a.v1_report, a.choreo)
    sc = bpy.context.scene
    if sc.rigidbody_world is not None:
        sc.rigidbody_world.enabled = False
    N = a.f1 or sc.frame_end
    t0 = time.time()
    K = Rig(bpy.data.objects["PX_Knight"], bpy.data.objects["PX_Knight"].find_armature(), knight_part, KNIGHT_TESTS, KNIGHT_CAPS)
    D = Rig(bpy.data.objects["PX_Dragon"], bpy.data.objects["PX_Dragon"].find_armature(), dragon_part, DRAGON_TESTS)
    k_root, d_root = bpy.data.objects["K_ROOT"], bpy.data.objects["D_ROOT"]
    if a.rest:
        K.arm.data.pose_position = "REST"
        D.arm.data.pose_position = "REST"
    sword = bpy.data.objects.get("K_Sword")
    print(f"[strict] parts knight {sorted((p, len(v)) for p, v in K.vidx.items())}")
    print(f"[strict] parts dragon {sorted((p, len(v)) for p, v in D.vidx.items())}")

    # ---- registered contacts (NO rubble kinds accepted)
    win = defaultdict(list)
    for c in meta["contacts"]:
        if c.get("kind") == "rubble" or str(c["key"]).startswith("rubble"):
            continue
        win[frozenset((c["a"], c["b"]))].append((c["lo"], c["hi"]))

    def allowed(x, y, fr):
        return any(lo <= fr <= hi for lo, hi in win.get(frozenset((x, y)), ()))

    def in_win(ws, fr):
        return any(lo <= fr <= hi for lo, hi in ws)

    rc = bpy.data.collections.get("ROOF_breakable")
    ROOF = {o.name for o in rc.all_objects} if rc else set()
    # ---- objects: debris (animated cells) / obstacles (static) / doors
    dg = bpy.context.evaluated_depsgraph_get()
    debris, obst = {}, {}
    alias = {}
    for i in range(3):
        alias[f"CV_S1W_Drum{i}"] = "cover_A"
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not ob.data.polygons:
            continue
        colls = {c.name for c in ob.users_collection}
        if colls & set(SKIP_COLL) and ob.name not in ("FX_Door_W", "FX_Door_E", "colossus_arm"):
            continue
        if ob.name in SUPPORT or ob.name.startswith(("PX_", "RB_", "TPL_")):
            continue
        if any(m.type == "ARMATURE" for m in ob.modifiers) or (ob.parent is not None and ob.parent.type == "ARMATURE"
                                                                and ob.parent_type != "BONE"):
            continue                   # a character's own skin (v1 audit: the dragon's 2nd mesh read as a static statue)
        top = ob
        while top.parent is not None:
            top = top.parent
        if top.name in ("D_ROOT", "K_ROOT"):
            continue
        ad = ob.animation_data
        moving = False
        if ad and ad.action:
            from bpy_extras import anim_utils
            cb = anim_utils.animdata_get_channelbag_for_assigned_slot(ad)
            if cb is not None:
                moving = any(fc.data_path in ("location", "rotation_euler") and len(fc.keyframe_points) > 2
                             for fc in cb.fcurves)
        if ob.name.startswith("FX_Door"):
            debris[ob.name] = dict(ob=ob, kind="door", mass=1e9)
            continue
        if any(s.lower() in ob.name.lower() for s in SOFT):
            continue
        if moving:
            debris[ob.name] = dict(ob=ob, kind="debris", mass=mesh_volume(ob) * DENS)
        else:
            m = mesh_volume(ob) * DENS
            if m < OBST_KG and "INTACT" not in ob.name:
                continue
            obst[ob.name] = dict(ob=ob, mass=m)
    # static obstacle geometry: world tris once (they don't move), AABB for the per-frame filter
    for n, d in obst.items():
        V, T = world_tris_static(d["ob"], dg)
        d["V"], d["T"] = V, T
        d["lo"], d["hi"] = V.min(0), V.max(0)
        d["bvh"] = None
    for n, d in debris.items():
        me = d["ob"].data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        d["co"] = co.reshape(-1, 3)
        d["T"] = tris_of(me)
    print(f"[strict] debris {len(debris)} (heavy {sum(1 for d in debris.values() if d['mass'] >= HEAVY_KG)}), "
          f"obstacles {len(obst)}, setup {time.time() - t0:.0f} s")
    sup = [bpy.data.objects[n] for n in SUPPORT if n in bpy.data.objects]
    Vs, Ts = [], []
    off = 0
    for o in sup:
        V, T = world_tris_static(o, dg)
        Vs.append(V)
        Ts.append(T + off)
        off += len(V)
    ground_bvh = BVHTree.FromPolygons(np.concatenate(Vs).tolist(), np.concatenate(Ts).tolist())

    def ground(p):
        h = ground_bvh.ray_cast(Vector((p[0], p[1], p[2] + 2.0)), Vector((0, 0, -1)), 40.0)
        return h[0].z if h[0] is not None else 0.0

    # ---- bones for feet / hands
    def bone(arm, short):
        for b in arm.pose.bones:
            if re.sub(r"_\d+$", "", b.name) in (short, "mixamorig:" + short, "DEF-" + short):
                return b.name
        raise KeyError(short)
    kb = {k: bone(K.arm, k) for k in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase", "LeftHand", "RightHand",
                                       "LeftForeArm", "RightForeArm", "LeftHandMiddle1", "RightHandMiddle1")}
    cb = {c: bone(K.arm, c) for c in sorted({b for ch in KNIGHT_CAPS.values() for b in ch})}
    db = {k: bone(D.arm, k) for k in ("toe.L", "toe.R", "foot.L.001", "foot.R.001", "palm.02.L", "palm.02.R",
                                       "forearm.L.002", "forearm.R.002")}

    # ---- per-frame loop
    self_k, self_d, wing = defaultdict(list), defaultdict(list), []
    blade_self = []
    obst_hits = defaultdict(list)
    deb_rec = defaultdict(list)          # (debris, who) -> [(fr, moving)]
    door_hits = defaultdict(list)
    feet = defaultdict(list)
    roots = []
    held_rec = defaultdict(list)
    step = max(1, a.step)
    frames = list(range(a.f0, N + 1, step))
    prev_deb = {}
    for fi, fr in enumerate(frames):
        sc.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        VK = world_verts(K.px, dg)
        VD = world_verts(D.px, dg)
        # 1 self-intersection
        ka0 = K.arm.evaluated_get(dg)
        Mk0 = np.array(ka0.matrix_world)
        kpts = {c: (Mk0 @ np.r_[np.array(ka0.pose.bones[cb[c]].head), 1.0])[:3] for c in cb}
        if fi == 0:
            K.cap_init(VK, kpts)
            print(f"[strict] knight capsule radii {dict((k, round(v, 3)) for k, v in K.cap_r.items())}")
        for h in K.cap_hits(VK, kpts):
            self_k[(h[0], h[1])].append((fr, h[2], h[3]))
        for h in D.self_hits(VD):
            self_d[(h[0], h[1])].append((fr, h[2], h[3]))
        bl, br = D.bvh(VD, ["wingL"]), D.bvh(VD, ["wingR"])
        if bl is not None and br is not None:
            n = len(bl.overlap(br))
            if n >= WING_CROSS_MIN:
                wing.append((fr, n))
        se = sword.evaluated_get(dg) if sword else None
        if se is not None and not se.hide_render and not sword.hide_render:
            SV = world_verts(sword, dg)
            sb = BVHTree.FromPolygons(SV.tolist(), tris_of(sword.data).tolist())
            kb_body = K.bvh(VK, ["torso", "head", "thighL", "thighR", "shinL", "shinR", "footL", "footR", "farmL",
                                 "handL", "uarmL"])
            if kb_body is not None:
                n = len(sb.overlap(kb_body))
                if n >= 2:
                    blade_self.append((fr, n))
        kfull = BVHTree.FromPolygons(VK.tolist(), K.T.tolist())
        dfull = BVHTree.FromPolygons(VD.tolist(), D.T.tolist())
        klo, khi = VK.min(0) - 0.05, VK.max(0) + 0.05
        dlo, dhi = VD.min(0) - 0.05, VD.max(0) + 0.05
        # 2a static obstacles
        for n, d in obst.items():
            ob = d["ob"]
            if ob.evaluated_get(dg).hide_render:
                continue
            for who, lo, hi, bv in (("K.body", klo, khi, kfull), ("D.body", dlo, dhi, dfull)):
                if (d["lo"] > hi).any() or (d["hi"] < lo).any():
                    continue
                if d["bvh"] is None:
                    d["bvh"] = BVHTree.FromPolygons(d["V"].tolist(), d["T"].tolist())
                ov = d["bvh"].overlap(bv)
                if ov:
                    nm = alias.get(n, n)
                    pier = re.match(r"(Pier_[EW]\d)_(pier_INTACT|pier_cell|arch_cell)", n)
                    roof = n in ROOF or n.startswith(("RoofTruss", "RoofPurlins"))
                    reg = nm if nm == "cover_A" else (pier.group(1) if pier else ("vault" if roof else f"static:{n}"))
                    if not allowed(who, reg, fr):
                        obst_hits[(who, n)].append((fr, len(ov)))
        # 2b moving debris + doors
        for n, d in debris.items():
            ob = d["ob"]
            oe = ob.evaluated_get(dg)
            if oe.hide_render:
                continue
            M = np.array(oe.matrix_world)
            V = d["co"] @ M[:3, :3].T + M[:3, 3]
            lo, hi = V.min(0), V.max(0)
            pos = V.mean(0)
            mv = 0.0
            if n in prev_deb and prev_deb[n][0] == fi - 1:
                mv = float(np.linalg.norm(pos - prev_deb[n][1])) * FPS / step
            prev_deb[n] = (fi, pos)
            for who, klo_, khi_, bv in (("K.body", klo, khi, kfull), ("D.body", dlo, dhi, dfull)):
                if (lo > khi_ + 0.06).any() or (hi < klo_ - 0.06).any():
                    continue
                b2 = BVHTree.FromPolygons(V.tolist(), d["T"].tolist())
                ov = b2.overlap(bv)
                if ov:
                    if d["kind"] == "door":
                        door_hits[(n, who)].append(fr)
                    else:
                        deb_rec[(n, who)].append((fr, round(mv, 2), len(ov)))
                elif d["kind"] != "door" and who == "K.body" and d["mass"] >= HEAVY_KG:
                    # Bullet keeps a collision margin: a cell PUSHED by the knight (or lying on him) never overlaps
                    # him — so heavy stone within 6 cm of his body counts as touching
                    sub = V[:: max(1, len(V) // 300)]
                    if any(bv.find_nearest(Vector(p), 0.06)[0] is not None for p in sub):
                        deb_rec[(n, who)].append((fr, round(mv, 2), 0))
        # 3 feet + roots (bone world positions)
        ka, da = K.arm.evaluated_get(dg), D.arm.evaluated_get(dg)
        Mk, Md = np.array(ka.matrix_world), np.array(da.matrix_world)

        def bp(arm_e, M, name, tail=False):
            b = arm_e.pose.bones[name]
            v = b.tail if tail else b.head
            return (M @ np.r_[np.array(v), 1.0])[:3]
        for lab, an, tn in (("K.L", "LeftFoot", "LeftToeBase"), ("K.R", "RightFoot", "RightToeBase")):
            pa, pt = bp(ka, Mk, kb[an]), bp(ka, Mk, kb[tn])
            feet[lab].append((fr, pa, pt, ground(pa), ground(pt)))
        for lab, an, tn in (("D.HL", "foot.L.001", "toe.L"), ("D.HR", "foot.R.001", "toe.R"),
                            ("D.FL", "forearm.L.002", "palm.02.L"), ("D.FR", "forearm.R.002", "palm.02.R")):
            pa, pt = bp(da, Md, db[an]), bp(da, Md, db[tn], tail=True)
            feet[lab].append((fr, pa, pt, ground(pa), ground(pt)))
        roots.append((fr, np.array(k_root.matrix_world.translation), k_root.matrix_world.to_euler().z,
                      np.array(d_root.matrix_world.translation), d_root.matrix_world.to_euler().z,
                      active_clips(K.arm, fr), active_clips(D.arm, fr)))
        # 4 held props
        for h in meta.get("held", []):
            if h["lo"] <= fr <= h["hi"]:
                po = bpy.data.objects.get(h["prop"])
                if po is None:
                    continue
                pe = po.evaluated_get(dg)
                PV = world_verts(po, dg)
                pb = BVHTree.FromPolygons(PV.tolist(), tris_of(po.data).tolist())
                for side in h["hands"]:
                    hn = kb["LeftHandMiddle1" if side == "L" else "RightHandMiddle1"]
                    hp = bp(ka, Mk, hn)
                    loc, _n, _i, dist = pb.find_nearest(Vector(hp), 5.0)
                    Mh = Mk @ np.array(ka.pose.bones[kb["LeftHand" if side == "L" else "RightHand"]].matrix)
                    rel = np.linalg.inv(Mh[:3, :3] / np.linalg.norm(Mh[:3, :3], axis=0)) @ (
                        np.array(pe.matrix_world)[:3, :3] / np.linalg.norm(np.array(pe.matrix_world)[:3, :3], axis=0))
                    held_rec[(h["prop"], side)].append((fr, float(dist if dist is not None else 9.9), rel))
        if fi % 250 == 0:
            print(f"[strict] frame {fr}/{N}  {time.time() - t0:.0f} s  selfK {sum(len(v) for v in self_k.values())} "
                  f"selfD {sum(len(v) for v in self_d.values())} wing {len(wing)} obst {len(obst_hits)} deb {len(deb_rec)}",
                  flush=True)

    # ================================================================ summarise
    out = dict(version=meta.get("version"), frames=[a.f0, N], step=step, seconds=None)
    gap = step * 2

    def grp(dct, extra=None):
        res = []
        for k, v in sorted(dct.items(), key=lambda kv: kv[1][0][0]):
            frs = [x[0] for x in v]
            rr = runs(frs, gap)
            item = dict(what=list(k) if isinstance(k, tuple) else k, frames=len(frs) * step,
                        runs=[[r0, r1] for r0, r1 in rr][:15])
            if extra:
                item.update(extra(v))
            res.append(item)
        return res
    out["self_knight"] = grp(self_k, lambda v: dict(max_depth_m=max(x[2] for x in v), max_verts=max(x[1] for x in v)))
    out["self_dragon"] = grp(self_d, lambda v: dict(max_depth_m=max(x[2] for x in v), max_verts=max(x[1] for x in v)))
    out["wings_cross"] = dict(frames=len(wing) * step, runs=runs([x[0] for x in wing], gap)[:20],
                              max_tri_pairs=max([x[1] for x in wing], default=0))
    out["blade_through_own_body"] = dict(frames=len(blade_self) * step, runs=runs([x[0] for x in blade_self], gap)[:20])
    out["obstacles"] = grp(obst_hits)
    for it in out["obstacles"]:
        it["mass_kg"] = round(obst[it["what"][1]]["mass"], 0)
    out["doors"] = [dict(door=k[0], body=k[1], frames=len(v) * step, runs=runs(v, gap)[:10]) for k, v in door_hits.items()]
    # debris rules
    deb_v = []
    for (n, who), v in sorted(deb_rec.items(), key=lambda kv: kv[1][0][0]):
        m = debris[n]["mass"]
        frs = [x[0] for x in v]
        rr = runs(frs, gap)
        longest = max(r1 - r0 + step for r0, r1 in rr)
        moving_while_touch = sum(1 for x in v if x[1] > 0.15)
        rule = None
        if who == "K.body":
            if m >= HEAVY_KG:          # any overlap is a violation: only the shield may stop heavy stone
                rule = "HEAVY_DEBRIS_MOVED_WHILE_TOUCHING_KNIGHT" if moving_while_touch else "HEAVY_DEBRIS_ON_OR_IN_KNIGHT"
            elif longest > 3:
                rule = "LIGHT_DEBRIS_RESTS_IN_KNIGHT"
        else:
            if longest > 6 and moving_while_touch < len(v) // 2:
                rule = "DEBRIS_RESTS_INSIDE_DRAGON"
        if rule:
            deb_v.append(dict(debris=n, body=who, mass_kg=round(m), rule=rule, frames=len(frs) * step,
                              runs=[[r0, r1] for r0, r1 in rr][:8], moving_frames=moving_while_touch * step,
                              max_speed=max(x[1] for x in v)))
    out["debris"] = deb_v
    out["debris_touch_all"] = [dict(debris=n, body=w, mass_kg=round(debris[n]["mass"]), first=v[0][0], last=v[-1][0])
                               for (n, w), v in sorted(deb_rec.items(), key=lambda kv: kv[1][0][0])]
    # feet
    air_k, air_d = meta["air_k"] + meta.get("sit_k", []), meta["air_d"]
    out["feet"] = feet_audit(feet, roots, air_k, air_d, meta, step)
    # held props
    hp = []
    for (prop, side), v in held_rec.items():
        dist = np.array([x[1] for x in v])
        R = [x[2] for x in v]
        ref = R[len(R) // 2]
        ang = []
        for r in R:
            c = np.clip((np.trace(ref.T @ r) - 1) / 2, -1, 1)
            ang.append(math.degrees(math.acos(c)))
        ang = np.array(ang)
        bad = [x[0] for x, d_, g in zip(v, dist, ang) if d_ > 0.08 or g > 25]
        hp.append(dict(prop=prop, hand=side, frames=len(v) * step, max_dist_m=round(float(dist.max()), 3),
                       p95_dist_m=round(float(np.percentile(dist, 95)), 3), max_rot_drift_deg=round(float(ang.max()), 1),
                       bad_frames=len(bad) * step, bad_runs=runs(bad, gap)[:8]))
    out["held_props"] = hp
    out["seconds"] = round(time.time() - t0, 1)
    # verdict
    nv = dict(self_knight=len(out["self_knight"]), self_dragon=len(out["self_dragon"]),
              wings_cross=len(out["wings_cross"]["runs"]), blade_self=len(out["blade_through_own_body"]["runs"]),
              obstacles=len(out["obstacles"]), doors=len(out["doors"]), debris=len(deb_v),
              foot_slide=sum(len(x["slide_runs"]) for x in out["feet"]["per_foot"].values()),
              glide=len(out["feet"]["glide_K"]) + len(out["feet"]["glide_D"]),
              float_k=len(out["feet"]["float_K"]), facing=len(out["feet"]["facing_K"]) + len(out["feet"]["facing_D"]),
              spins=len(out["feet"]["dragon_spins"]),
              held=sum(1 for x in hp if x["bad_frames"]))
    out["violation_counts"] = nv
    out["clean"] = not any(nv.values())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("[strict] DONE", json.dumps(nv), f"{out['seconds']} s → {a.out}")


def world_tris_static(ob, dg):
    oe = ob.evaluated_get(dg)
    me = oe.to_mesh()
    try:
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        M = np.array(oe.matrix_world)
        V = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        return V, tris_of(me)
    finally:
        oe.to_mesh_clear()


def active_clips(arm, fr):
    ad = arm.animation_data
    names = []
    if not ad:
        return names
    for t in ad.nla_tracks:
        if "BASE" not in t.name or t.mute:
            continue
        for s in t.strips:
            if s.frame_start <= fr <= s.frame_end and not s.mute:
                names.append(s.name.split("@")[0])
    return names


def feet_audit(feet, roots, air_k, air_d, meta, step):
    def inw(ws, fr):
        return any(lo <= fr <= hi for lo, hi in ws)
    res = dict(per_foot={}, glide_K=[], glide_D=[], float_K=[], facing_K=[], facing_D=[], dragon_spins=[])
    contact = {}
    for lab, rec in feet.items():
        isK = lab.startswith("K")
        frs = np.array([r[0] for r in rec])
        ha = np.array([r[1][2] - r[3] for r in rec])
        ht = np.array([r[2][2] - r[4] for r in rec])
        a0, t0 = np.percentile(ha, 5), np.percentile(ht, 5)
        h = np.minimum(ha - a0, ht - t0)
        thr_c = 0.045 if isK else 0.18
        PA = np.array([r[1] for r in rec])
        PT = np.array([r[2] for r in rec])
        va = np.r_[np.inf, np.linalg.norm(np.diff(PA[:, :2], axis=0), axis=1)] * FPS / step
        vt = np.r_[np.inf, np.linalg.norm(np.diff(PT[:, :2], axis=0), axis=1)] * FPS / step
        vz = np.r_[0, np.abs(np.diff(np.minimum(PA[:, 2], PT[:, 2])))] * FPS / step
        air = np.array([inw(air_k if isK else air_d, f) for f in frs])
        c = (h < thr_c) & (vz < (0.5 if isK else 1.2)) & ~air
        contact[lab] = (frs, c, h, air)
        sp = np.minimum(va, vt)
        thr_s = 0.30 if isK else 0.60
        slide = c & np.r_[False, c[:-1]] & (sp > thr_s)
        sl_frames = frs[slide].tolist()
        rr = [r for r in runs(sl_frames, step) if r[1] - r[0] >= 2 * step]
        dist = []
        for r0, r1 in rr:
            m = (frs >= r0) & (frs <= r1)
            dist.append(round(float(sp[m].sum() * step / FPS), 2))
        res["per_foot"][lab] = dict(rest_ankle_h=round(float(a0), 3), contact_frames=int(c.sum()) * step,
                                    slide_frames=int(slide.sum()) * step,
                                    slide_runs=[[int(r[0]), int(r[1]), d] for r, d in zip(rr, dist)][:40],
                                    total_slide_m=round(float(sum(dist)), 2))
    # roots
    frs = np.array([r[0] for r in roots])
    for who, pi, yi, ci, labs in (("K", 1, 2, 5, ("K.L", "K.R")), ("D", 3, 4, 6, ("D.HL", "D.HR", "D.FL", "D.FR"))):
        P = np.array([r[pi] for r in roots])
        Y = np.unwrap(np.array([r[yi] for r in roots]))
        k = max(1, 6 // step)
        V = np.zeros_like(P)
        V[k:-k] = (P[2 * k:] - P[:-2 * k]) / (2 * k * step / FPS)
        sp = np.linalg.norm(V[:, :2], axis=1)
        fw = np.c_[-np.sin(Y), np.cos(Y)]
        air = np.array([inw(air_k if who == "K" else air_d, f) for f in frs])
        thrown = np.array([inw(meta.get("thrown_k", []), f) for f in frs])
        allc = np.all([contact[l][1] for l in labs], axis=0)
        # GLIDE = the root travels further than a leg can reach while EVERY foot stays planted (no step taken)
        lim = 0.45 if who == "K" else 1.8
        okc = allc & ~air & ~thrown
        gl, s0 = [], None
        for i in range(len(frs)):
            if not okc[i]:
                s0 = None
                continue
            if s0 is None:
                s0 = i
            if np.linalg.norm(P[i, :2] - P[s0, :2]) > lim:
                if gl and gl[-1][1] >= frs[s0]:
                    gl[-1][1] = int(frs[i])
                else:
                    gl.append([int(frs[s0]), int(frs[i])])
        res["glide_" + who] = gl[:40]
        cosang = np.einsum("ij,ij->i", V[:, :2], fw) / np.maximum(sp, 1e-6)
        side = (sp > (0.5 if who == "K" else 1.0)) & (cosang < math.cos(math.radians(50))) & ~air & ~thrown
        ok_clip = np.array([any(s in c for c in r[ci] for s in meta.get("side_clips", [])) for r in roots])
        fwd_loco = np.array([any(k in c for c in r[ci] for k in ("Walk", "Jog", "Sprint", "Run")) for r in roots])
        stepping = ~np.all([contact[l][1] for l in labs], axis=0)       # some foot is in the air this frame
        bad = side & ~ok_clip
        fc = []
        for r in runs(frs[bad].tolist(), step):
            if r[1] - r[0] < 6 * step:
                continue
            m = (frs >= r[0]) & (frs <= r[1])
            # side/back travel is fine while the feet STEP it (procedural shuffle / pivot steps) under a stance clip;
            # it is a violation when a forward walk plays (moonwalk) or no foot ever leaves the ground (floating)
            if fwd_loco[m].mean() > 0.3 or stepping[m].mean() < 0.08:
                fc.append(r)
        res["facing_" + who] = [[int(r[0]), int(r[1]), "+".join(sorted(set(sum((roots[i][ci] for i in range(len(frs))
                                                                              if r[0] <= frs[i] <= r[1]), []))))[:80]]
                                for r in fc][:40]
        if who == "K":
            low = np.minimum(contact["K.L"][2], contact["K.R"][2])
            running = np.array([any(k in c for c in r[ci] for k in ("Jog", "Sprint", "Run")) for r in roots])
            fl = (low > 0.15) & ~air & ~thrown & ~running          # a run's flight phase is not floating
            res["float_K"] = [r for r in runs(frs[fl].tolist(), step) if r[1] - r[0] >= 6 * step][:40]
        else:
            w = max(1, 20 // step)
            dy = np.abs(Y[w:] - Y[:-w])
            spin = (np.r_[dy > math.radians(120), np.zeros(w, bool)] if len(Y) > w else np.zeros(len(Y), bool)) & ~air
            res["dragon_spins"] = [[int(r[0]), int(r[1]), round(float(np.degrees(dy[(frs[:-w] >= r[0]) & (frs[:-w] <= r[1])].max())), 0)]
                                   for r in runs(frs[spin].tolist(), step)][:20]
    return res


if __name__ == "__main__":
    main()
