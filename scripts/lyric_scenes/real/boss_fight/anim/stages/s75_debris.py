"""s75 — destruction: pre-fractured cells released kinematic→dynamic on their event frames, Bullet-simulated with the
rigs as kinematic deform-mesh colliders (and the shield as fixed-radius kinematic spheres parked 1 km away when off),
baked once (CLI, deterministic), converted to plain keys, rigid-body world removed (physics OFF for the audit/render).
  pillar_1 (W2) cracks: only the cells around the tail contact chip out
  pillar_2 (E2) shatters + its arch collapses; one arch cell is held and falls onto the shield (`ceiling`)
  pillar_3 (W3) smashed
  vault: cells released on first proximity of the dragon (per-cell trigger, physics-off scan) + a spreading collapse;
         3 cells thrown onto the shield dome (`roof_rain`)
  colossus E: the arm chunk around the volley hit is cut out and drops (`volley`)"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from anim import common as C
from anim import fx as FX
from lib import checks as CK

SIM0, SIM1 = 1840, 5420
DENS = 2300.0


def _ovr(ob, sel=None):
    return bpy.context.temp_override(object=ob, active_object=ob, selected_objects=sel or [ob])


def unparent_keep(ob):
    mw = ob.matrix_world.copy()
    ob.parent = None
    ob.matrix_world = mw


def cell_center(ob):
    co = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
    return co.mean(0), co


HEAVY_KG = 60.0     # the strict audit's threshold: stone this heavy is never moved by the knight


def volume(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    try:
        v = abs(bm.calc_volume(signed=False))
    finally:
        bm.free()
    s = ob.matrix_world.to_scale()
    v *= abs(s[0] * s[1] * s[2])
    box = float(np.prod(np.maximum(np.array(ob.dimensions), 1e-3)))
    return v if (np.isfinite(v) and 0 < v <= box * 1.05) else 0.5 * box


def set_groups(ob, groups):
    """Bullet collision collections: 0 = world + heavy stone + dragon + shield, 1 = knight + light pieces + shield.
    Bodies collide only if they share a group ⇒ the knight can't shove heavy stone (v1: he scooted 1-kg 'boulders')."""
    cc = ob.rigid_body.collision_collections
    for i in range(len(cc)):
        cc[i] = i in groups


def rb_add(ob, typ="ACTIVE", shape="CONVEX_HULL", mass=None, kin=False, deform=False, groups=None):
    with _ovr(ob):
        bpy.ops.rigidbody.object_add(type=typ)
    rb = ob.rigid_body
    rb.collision_shape = shape
    rb.friction = 0.7
    rb.restitution = 0.08
    rb.collision_margin = 0.04
    rb.use_margin = True
    if typ == "ACTIVE":
        rb.mass = float(mass) if mass is not None else max(volume(ob) * DENS, 1.0)
        set_groups(ob, groups if groups is not None else ((0,) if rb.mass >= HEAVY_KG else (0, 1)))
    elif mass is not None:
        rb.mass = mass
    if groups is not None and typ != "ACTIVE":
        set_groups(ob, groups)
    rb.kinematic = kin
    if deform:
        rb.use_deform = True
        rb.mesh_source = "DEFORM"
    return rb


def clear_fc(ob, path):
    from bpy_extras import anim_utils
    ad = ob.animation_data
    if not ad or not ad.action:
        return
    cb = anim_utils.animdata_get_channelbag_for_assigned_slot(ad)
    for fc in [fc for fc in cb.fcurves if fc.data_path == path]:
        cb.fcurves.remove(fc)


def key_release(ob, f):
    rb = ob.rigid_body
    rb.kinematic = True
    rb.keyframe_insert("kinematic", frame=f - 1)
    rb.kinematic = False
    rb.keyframe_insert("kinematic", frame=f)


def scripted_throw(S, ob, f_rel, f_hit, target_pt, approach_up=True, bounce_from=None):
    """kinematic ballistic path from rest (f_rel) so the cell's lowest point lands on target_pt at f_hit; dynamic after."""
    c, co = cell_center(ob)
    low = co[np.argmin(co[:, 2])] if approach_up else co[np.argmin(np.linalg.norm(co - target_pt, axis=1))]
    off = c - low
    p0, p1 = c, np.asarray(target_pt) + off
    t = (f_hit - f_rel) / C.FPS
    g = np.array([0, 0, -C.G])
    v0 = (p1 - p0 - 0.5 * g * t * t) / t
    frs = np.arange(f_rel, f_hit + 1)
    tt = (frs - f_rel) / C.FPS
    path = p0 + np.outer(tt, v0) + np.outer(0.5 * tt * tt, g)
    base = np.array(ob.location)
    f_dyn = f_hit + 1
    if bounce_from is not None:            # deflected OFF the dome, away from the knight (never lands on him)
        out = (p1 - np.asarray(bounce_from))[:2]
        out = out / max(np.linalg.norm(out), 1e-6)
        extra = [p1 + np.r_[out * 0.16 * k, 0.07 * k] for k in (1, 2, 3)]
        frs = np.r_[frs, [f_hit + 1, f_hit + 2, f_hit + 3]]
        path = np.r_[path, extra]
        f_dyn = f_hit + 4
    FX.key_path(ob, np.r_[SIM0 - 1, frs], np.r_[[base], base + (path - p0)])
    rb = ob.rigid_body
    rb.kinematic = True
    rb.keyframe_insert("kinematic", frame=SIM0 - 1)
    rb.keyframe_insert("kinematic", frame=f_dyn - 1)
    rb.kinematic = False
    rb.keyframe_insert("kinematic", frame=f_dyn)
    return v0


def cut_chunk(ob, cxy, r, name, coll):
    """cut the part of `ob` inside the vertical square |x-cx|,|y-cy| < r (world) out into a new object (bisected, so
    long beams split cleanly); None if nothing is inside"""
    ob.data = ob.data.copy()
    M = np.array(ob.matrix_world)
    Mi = np.linalg.inv(M)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    for axis, val in ((0, cxy[0] - r), (0, cxy[0] + r), (1, cxy[1] - r), (1, cxy[1] + r)):
        n_w = np.zeros(3)
        n_w[axis] = 1.0
        p_w = np.array([cxy[0], cxy[1], 0.0])
        p_w[axis] = val
        p_l = (Mi @ np.r_[p_w, 1.0])[:3]
        n_l = M[:3, :3].T @ n_w
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=Vector(p_l), plane_no=Vector(n_l))
    inside = []
    for fc in bm.faces:
        c = M @ np.r_[np.array(fc.calc_center_median()), 1.0]
        if abs(c[0] - cxy[0]) < r and abs(c[1] - cxy[1]) < r:
            inside.append(fc)
    if not inside:
        bm.free()
        return None
    nb = bmesh.new()
    vmap = {}
    for fc in inside:
        vs = []
        for v in fc.verts:
            if v not in vmap:
                vmap[v] = nb.verts.new(v.co)
            vs.append(vmap[v])
        try:
            nb.faces.new(vs)
        except ValueError:
            pass
    me = bpy.data.meshes.new(name)
    nb.to_mesh(me)
    nb.free()
    for m in ob.data.materials:
        me.materials.append(m)
    bmesh.ops.delete(bm, geom=inside, context="FACES")
    bm.to_mesh(ob.data)
    bm.free()
    ch = bpy.data.objects.new(name, me)
    coll.objects.link(ch)
    ch.matrix_world = ob.matrix_world.copy()
    return ch


def run(S):
    sc, E, N, col = S.sc, S.E, S.N, S.col
    if S.args.no_debris:                      # fast iteration on animation/audit only
        S.debris, S.pier_swap = {}, {}
        S.R.log("WARN debris stage SKIPPED (--no-debris)")
        return
    f = {k: e.f for k, e in E.items()}
    S.debris = {}                      # object name -> dict(group, release)
    S.pier_swap = {}                   # pier -> swap frame (INTACT hidden, cells shown)
    if sc.rigidbody_world is None:
        with bpy.context.temp_override(scene=sc):
            bpy.ops.rigidbody.world_add()
    rbw = sc.rigidbody_world
    rbw.enabled = True
    rbw.substeps_per_frame = 10
    rbw.solver_iterations = 20
    rbw.point_cache.frame_start, rbw.point_cache.frame_end = SIM0, SIM1
    if rbw.collection is None:
        rbw.collection = bpy.data.collections.new("RB_World")

    # ---------------------------------------------------------- static colliders (passive)
    statics = ["ArenaFloor", "CourtyardPaving", "GrandStairs", "Wall_E", "Wall_W", "Wall_N_Gable", "Wall_S_E", "Wall_S_W",
               "GatePiers", "Dais", "Colossus_Plinth_E", "Colossus_Plinth_W", "StairPedestals"]
    statics += [o.name for o in bpy.data.collections["V3_COVER"].all_objects if o.type == "MESH" and not o.hide_render]
    for p in ("E1", "W1", "E3", "E4", "W4"):
        statics.append(f"Pier_{p}_pier_INTACT")
    for n in statics:
        o = bpy.data.objects[n]
        rb_add(o, "PASSIVE", "MESH", groups=(0, 1))
    # ---------------------------------------------------------- characters: kinematic deform colliders
    rb_add(S.K.proxy, "PASSIVE", "MESH", kin=True, deform=True, groups=(1,))
    rb_add(S.D.proxy, "PASSIVE", "MESH", kin=True, deform=True, groups=(0,))
    # shield spheres (fixed radius each, parked far away when the bubble is off)
    for i, (a, b, R) in enumerate(S.shield_windows):
        sp = FX.ico(f"RB_ShieldSphere_{i}", R, col.proxy, None, subdiv=2)
        sp.hide_render = True
        frs = np.arange(SIM0 - 1, SIM1 + 1)
        pos = np.where(((frs >= a + 3) & (frs <= b - 3))[:, None], S.shield_center[np.clip(frs - 1, 0, N - 1)],
                       np.array([0.0, 0.0, -1000.0 - 50 * i]))
        FX.key_path(sp, frs, pos)
        rb_add(sp, "PASSIVE", "SPHERE", kin=True, groups=(0, 1))

    # ---------------------------------------------------------- pillars
    def pier_cells(pid):
        cells = [o for o in bpy.data.collections[f"Pier_{pid}"].all_objects if o.type == "MESH" and "_cell" in o.name]
        for o in cells:                       # hidden objects are NOT evaluated: unhide first, then read matrix_world
            o.hide_viewport = False
        bpy.context.view_layer.update()
        for o in cells:
            unparent_keep(o)
        return cells

    def swap(pid, fr, cells):
        intact = bpy.data.objects[f"Pier_{pid}_pier_INTACT"]
        C.key_visibility(intact, [(1, False), (fr, True)], viewport=False)
        for o in cells:
            C.key_visibility(o, [(1, True), (fr, False)], viewport=False)
        S.pier_swap[f"Pier_{pid}"] = fr
        arch = bpy.data.objects.get(f"Pier_{pid}_arch_INTACT")
        if arch is not None:
            C.key_visibility(arch, [(1, False), (fr, True)], viewport=False)

    # pillar_1 = W2: crack — chips near the contact only
    tip = np.array(S.trk["D.tail_tip"][f["dodge_p1"] - 1])
    cells = pier_cells("W2")
    swap("W2", f["dodge_p1"], cells)
    for o in cells:
        c, _ = cell_center(o)
        rb_add(o)
        if "_pier_" in o.name and np.linalg.norm(c - tip) < 1.9:
            key_release(o, f["dodge_p1"])
            S.debris[o.name] = dict(group="pillar_1", release=f["dodge_p1"])
        else:
            o.rigid_body.kinematic = True
    # pillar_2 = E2: shatter + arch collapse; the highest arch cell is held for the ceiling chunk
    cells = pier_cells("E2")
    swap("E2", f["tail_p2"], cells)
    arch = [o for o in cells if "_arch_" in o.name]
    small = [o for o in arch if max(o.dimensions) < 2.6] or arch           # v2: a chunk, not a 7.7 m arch slab
    chunk = max(small, key=lambda o: cell_center(o)[0][2] - 0.2 * abs(cell_center(o)[0][1] - 18.75))
    for o in cells:
        rb_add(o)
        if o is chunk:
            continue
        rel = f["tail_p2"] if "_pier_" in o.name else f["tail_p2"] + 5 + int(abs(cell_center(o)[0][1] - 18.75))
        key_release(o, rel)
        S.debris[o.name] = dict(group="pillar_2", release=rel)
    chunk.name = "ceiling_chunk_1"
    sh = S.shield_center[f["ceiling"] - 1]
    R = [w for w in S.shield_windows if w[0] <= f["ceiling"] <= w[1]][0][2]
    scripted_throw(S, chunk, f["ceiling"] - 46, f["ceiling"], sh + np.array([0, 0, R]), bounce_from=sh - np.array([1.5, 0, 0]))
    S.debris[chunk.name] = dict(group="ceiling_chunk", release=f["ceiling"] - 46, scripted=True)
    # pillar_3 = W3: smashed
    cells = pier_cells("W3")
    swap("W3", f["tail_p3"], cells)
    for o in cells:
        rb_add(o)
        rel = f["tail_p3"] if "_pier_" in o.name else f["tail_p3"] + 6 + int(abs(cell_center(o)[0][1] - 31.25))
        key_release(o, rel)
        S.debris[o.name] = dict(group="pillar_3", release=rel)

    # ---------------------------------------------------------- vault burst
    vcells = [o for o in bpy.data.collections["ROOF_breakable"].all_objects if o.type == "MESH" and "_cell" in o.name]
    for o in vcells:
        o.hide_viewport = False
    bpy.context.view_layer.update()
    for o in vcells:
        unparent_keep(o)
    fr_ = f["roof"]
    head = np.array(S.trk["D.head"][fr_ - 1])
    # v1 hid only Vault_INTACT: the ribs / slates stayed whole and the dragon passed through them, and through the
    # trusses + purlins (no cells at all). v2: every *_INTACT swaps to its cells, timber is cut round the burst
    for o in [o for o in bpy.data.collections["ROOF_breakable"].all_objects if o.type == "MESH" and "_INTACT" in o.name]:
        C.key_visibility(o, [(1, False), (fr_, True)], viewport=False)
    for o in vcells:
        C.key_visibility(o, [(1, True), (fr_, False)], viewport=False)
    timber = []
    for o in [o for o in bpy.data.collections["ROOF_breakable"].all_objects
              if o.type == "MESH" and ("Truss" in o.name or "Purlin" in o.name)]:
        ch = cut_chunk(o, head[:2], 6.5, f"{o.name}_burst", col.debris)
        if ch is not None:                 # the timber round the burst is SMASHED: gone on the burst frame (no sim)
            timber.append(ch)
            C.key_visibility(ch, [(1, False), (fr_ - 4, True)], viewport=False)
    S.R.add("debris.roof_timber_cut", [o.name for o in timber])
    vcells = vcells
    # per-cell trigger: first frame the dragon proxy is within 0.25 m (physics OFF scan), else spreading collapse
    first = {}
    rbw.enabled = False
    for fr in range(fr_ - 4, fr_ + 40):
        sc.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        V, T = CK.bpy_world_tris([S.D.proxy], dg)
        bv = BVHTree.FromPolygons(V.tolist(), T.tolist())
        for o in vcells:
            if o.name in first:
                continue
            c, co = cell_center(o)
            if np.linalg.norm(c[:2] - head[:2]) > 12:
                continue
            for p in co[:: max(1, len(co) // 40)]:
                h = bv.find_nearest(Vector(p), 0.25)
                if h[0] is not None:
                    first[o.name] = fr
                    break
    rbw.enabled = True
    roof_throw = []
    for o in vcells:
        c, _ = cell_center(o)
        dist = float(np.linalg.norm(c[:2] - head[:2]))
        rb_add(o)
        rel = first.get(o.name)
        if rel is None and dist < 9.0:
            rel = fr_ + 2 + int(dist * 1.2)
        if rel is None:
            o.rigid_body.kinematic = True
            continue
        S.debris[o.name] = dict(group="roof", release=rel, touched=o.name in first)
        roof_throw.append((dist, o, rel))
        key_release(o, rel)
    S.R.add("debris.roof_first_touch", {k: v for k, v in sorted(first.items(), key=lambda x: x[1])[:12]})
    S.R.add("debris.roof_first_touch_frame", min(first.values()) if first else None)
    # three cells onto the shield dome
    kc = S.shield_center
    hits = [f["roof_rain"]] + S.extra_hits["roof_rain"]
    cands = sorted([t for t in roof_throw if 6.0 < t[0] < 12.0 and "_cell" in t[1].name and max(t[1].dimensions) < 2.6],
                   key=lambda t: t[0])
    for i, fh in enumerate(hits):
        if i >= len(cands):
            break
        _d, o, rel = cands[i]
        R = [w for w in S.shield_windows if w[0] <= fh <= w[1]][0][2]
        ang = math.radians(35 + 50 * i)
        tgt = kc[fh - 1] + R * np.array([math.cos(ang) * 0.5, math.sin(ang) * 0.5, 0.86])
        # the scripted throw replaces its release keys
        clear_fc(o, "rigid_body.kinematic")
        scripted_throw(S, o, fr_ + 4, fh, tgt, bounce_from=kc[fh - 1])
        S.debris.pop(o.name, None)
        o.name = f"roof_chunk_{i + 1}"
        S.debris[o.name] = dict(group="roof_chunks", release=fr_ + 4, scripted=True)

    # ---------------------------------------------------------- colossus E arm
    cole = bpy.data.objects["Colossus_E"]
    cole.data = cole.data.copy()
    hp = np.asarray(S.fx_hits["dragon_fireball_v"]["p"])     # cut the chunk round the fireball's REAL hit point
    Mi = cole.matrix_world.inverted()
    bm = bmesh.new()
    bm.from_mesh(cole.data)
    lp = Mi @ Vector(hp)
    sel = [v for v in bm.verts if (v.co - lp).length < 2.2 / cole.matrix_world.to_scale()[0]]
    faces = {fc for v in sel for fc in v.link_faces}
    arm_bm = bmesh.new()
    vmap = {}
    for fc in faces:
        vs = []
        for v in fc.verts:
            if v not in vmap:
                vmap[v] = arm_bm.verts.new(v.co)
            vs.append(vmap[v])
        try:
            arm_bm.faces.new(vs)
        except ValueError:
            pass
    me_arm = bpy.data.meshes.new("colossus_arm")
    arm_bm.to_mesh(me_arm)
    arm_bm.free()
    for m in cole.data.materials:
        me_arm.materials.append(m)
    bmesh.ops.delete(bm, geom=list(faces), context="FACES")
    bm.to_mesh(cole.data)
    bm.free()
    arm = bpy.data.objects.new("colossus_arm", me_arm)
    col.debris.objects.link(arm)
    arm.matrix_world = cole.matrix_world.copy()
    rb_add(arm, mass=4000.0)
    fv = f["volley"]
    v = np.asarray(S.fx_hits["dragon_fireball_v"]["v"]) * 0.25 + np.array([0, 0, 1.5])
    base = np.array(arm.location)
    FX.key_path(arm, [SIM0 - 1, fv, fv + 1, fv + 2], [base, base, base + v / C.FPS, base + 2 * v / C.FPS])
    rb = arm.rigid_body
    rb.kinematic = True
    rb.keyframe_insert("kinematic", frame=SIM0 - 1)
    rb.keyframe_insert("kinematic", frame=fv + 2)
    rb.kinematic = False
    rb.keyframe_insert("kinematic", frame=fv + 3)
    S.debris["colossus_arm"] = dict(group="colossus_arm", release=fv)
    S.R.add("debris.colossus_arm_faces", len(faces))

    # ---------------------------------------------------------- bake → keys → physics off
    import time
    t0 = time.time()
    with bpy.context.temp_override(scene=sc, point_cache=rbw.point_cache):
        bpy.ops.ptcache.free_bake()
        r = bpy.ops.ptcache.bake(bake=True)
    S.R.add("debris.bake_s", round(time.time() - t0, 1))
    S.R.log(f"rigid-body bake {r} in {time.time() - t0:.0f} s")
    dyn = [bpy.data.objects[n] for n in S.debris]
    t0 = time.time()
    tr = {o.name: [] for o in dyn}
    frs = np.arange(SIM0, SIM1 + 1)
    for fr in frs:
        sc.frame_set(int(fr))
        for o in dyn:
            m = o.matrix_world
            tr[o.name].append((tuple(m.translation), tuple(m.to_euler())))
    # remove rigid bodies + world, then write keys (location + rotation over the sim range, static before)
    for o in list(rbw.collection.objects):
        with _ovr(o):
            bpy.ops.rigidbody.object_remove()
    with bpy.context.temp_override(scene=sc):
        bpy.ops.rigidbody.world_remove()
    moved = 0
    for o in dyn:
        L = np.array([a for a, _ in tr[o.name]])
        Rr = np.unwrap(np.array([b for _, b in tr[o.name]]), axis=0)
        if np.abs(L - L[0]).max() < 1e-4 and np.abs(Rr - Rr[0]).max() < 1e-4:
            continue
        moved += 1
        keep = np.r_[True, (np.abs(np.diff(L, axis=0)).max(1) > 1e-5) | (np.abs(np.diff(Rr, axis=0)).max(1) > 1e-5)]
        keep = keep | np.r_[keep[1:], True] | np.r_[True, keep[:-1]]
        FX.key_path(o, frs[keep], L[keep], rot_euler=Rr[keep])
    # ---- heavy stone that would end up IN the knight (resting where he walks, or flying through him) is taken out of
    # the fracture (never shown — one fewer fragment), because he can neither shove it nor have it lie on him
    kr, kh = S.trk["K.root"], S.trk["K.head"]
    removed = []
    for o in dyn:
        meta = S.debris.get(o.name, {})
        if meta.get("scripted"):
            continue
        mass = volume(o) * DENS
        if mass < HEAVY_KG:
            continue
        L = np.array([a for a, _ in tr[o.name]])
        rad = 0.5 * float(max(o.dimensions))
        hit = None
        for t in range(int(meta.get("release", SIM0)), N + 1, 2):
            c = L[min(max(t, SIM0), SIM1) - SIM0]
            a, b = kr[t - 1] + np.array([0, 0, 0.15]), kh[t - 1]
            ab = b - a
            u = float(np.clip(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
            if np.linalg.norm(c - (a + ab * u)) < rad + 0.45:
                hit = t
                break
        if hit is not None:
            clear_fc(o, "hide_render")
            o.hide_render = True
            S.debris.pop(o.name, None)
            removed.append((o.name, int(hit), round(mass)))
    S.R.add("debris.removed_from_knight_path", removed)
    S.R.add("debris.dynamic", len(dyn))
    S.R.add("debris.moved", moved)
    S.R.add("debris.convert_s", round(time.time() - t0, 1))
    # hashed positions (rebuild-identical proof)
    h = {}
    for fr in (2300, 2900, 4500, 5300):
        sc.frame_set(fr)
        h[fr] = round(float(sum(sum(bpy.data.objects[n].matrix_world.translation) for n in sorted(S.debris))), 4)
    S.R.add("debris.position_hash", h)
