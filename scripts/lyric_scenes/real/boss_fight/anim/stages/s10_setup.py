"""s10 — scene setup: frame range, collections, characters re-rooted, knight re-imported with rest = bind pose, sword +
scabbard, audit proxies, static BVHs (ground heights + solids)."""
import math
import types

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from lib import bpyx as X
from lib import importer as imp
from lib import checks as CK
from anim import common as C
from anim.retarget import Skel


def _descendants(ob):
    out = [ob]
    for ch in ob.children:
        out += _descendants(ch)
    return out


def _override(ob, sel=None):
    return bpy.context.temp_override(object=ob, active_object=ob, selected_objects=sel or [ob],
                                     selected_editable_objects=sel or [ob])


def normalise_rest(arm, meshes):
    """Sketchfab glTF: armature rest != bind. Bake the skin, apply the pose as rest, re-skin (retarget_test.py)."""
    if arm.animation_data:
        arm.animation_data.action = None
    bpy.context.view_layer.update()
    for ob in meshes:
        for md in [m for m in ob.modifiers if m.type == "ARMATURE" and m.object == arm]:
            if ob.data.users > 1:
                ob.data = ob.data.copy()
            nm = md.name
            with _override(ob):
                bpy.ops.object.modifier_apply(modifier=nm)
            m2 = ob.modifiers.new(nm, "ARMATURE")
            m2.object = arm
    bpy.context.view_layer.objects.active = arm
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def mesh_bvh(objs, dg=None):
    dg = dg or bpy.context.evaluated_depsgraph_get()
    V, T = CK.bpy_world_tris(objs, dg)
    return BVHTree.FromPolygons(V.tolist(), T.tolist()), V, T


def make_proxy(name, meshes, arm, ratio, coll):
    """joined, decimated copy keeping the Armature modifier (deforms with the rig); render-hidden."""
    copies = []
    for m in meshes:
        c = m.copy()
        c.data = m.data.copy()
        c.modifiers.clear()
        coll.objects.link(c)
        copies.append(c)
    bpy.context.view_layer.update()
    if len(copies) > 1:
        with bpy.context.temp_override(active_object=copies[0], object=copies[0], selected_objects=copies,
                                       selected_editable_objects=copies):
            bpy.ops.object.join()
    px = copies[0]
    px.name = name
    for u in list(px.users_collection):
        u.objects.unlink(px)
    coll.objects.link(px)
    if ratio < 1.0:
        dm = px.modifiers.new("dec", "DECIMATE")
        dm.ratio = ratio
        dm.use_collapse_triangulate = True
        with _override(px):
            bpy.ops.object.modifier_apply(modifier="dec")
    am = px.modifiers.new("Armature", "ARMATURE")
    am.object = arm
    px.hide_render = True
    px.display_type = "WIRE"
    for mi in range(len(px.material_slots)):
        px.material_slots[mi].material = None
    return px


def run(S):
    sc = S.sc
    sc.render.fps = C.FPS
    sc.frame_start, sc.frame_end = 1, S.N
    sc.frame_set(1)
    col = types.SimpleNamespace()
    col.root = X.coll("ANIM")
    col.chars = X.coll("ANIM_CHARACTERS", col.root)
    col.props = X.coll("ANIM_PROPS", col.root)
    col.fx = X.coll("ANIM_FX", col.root)
    col.proj = X.coll("ANIM_PROJECTILES", col.root)
    col.debris = X.coll("ANIM_DEBRIS", col.root)
    col.cams = X.coll("ANIM_CAMERAS", col.root)
    col.proxy = X.coll("ANIM_PROXIES", col.root)
    col.tmp = X.coll("ANIM_TMP", col.root)
    S.col = col

    # ---------------------------------------------------------------- remove the arena's static knight
    kn = bpy.data.objects.get("KNIGHT")
    if kn is not None:
        doomed = _descendants(kn)
        for o in doomed:
            bpy.data.objects.remove(o, do_unlink=True)
        S.R.log(f"removed arena knight ({len(doomed)} objects)")

    # ---------------------------------------------------------------- v2: the brazier standing in the fight ring is
    # gone (the dragon walked through it in v1) — removed for the whole film, flame + light included
    gone = [o for o in bpy.data.objects if o.name.startswith(("Brazier12", "Brazier07"))]
    # three cover pieces the fight geometry can't avoid (tail arcs / strike-3 footing) are not part of v2's set
    gone += [bpy.data.objects[n] for n in ("CV_S1E_ScreenWall", "CV_S2W_ScreenWall", "CV_W2_Pew0", "CV_W2_Pew1",
                                           "CV_S2E_Drum0", "CV_S2E_Drum1") if n in bpy.data.objects]
    gone += [bpy.data.objects[f"Rubble3_{i:03d}"] for i in range(85, 100) if f"Rubble3_{i:03d}" in bpy.data.objects]
    for o in gone:
        o.hide_render = True
        o.hide_viewport = True
    S.R.add("removed_props", [o.name for o in gone])

    # ---------------------------------------------------------------- dragon: re-root so it faces +Y under D_ROOT
    dr = bpy.data.objects["DRAGON"]
    d_root = X.empty("D_ROOT", (0, 0, 0), coll=col.chars, kind="ARROWS")
    d_root.empty_display_size = 3.0
    dr.parent = d_root
    dr.location = (0, 0, 0)
    dr.rotation_euler = (0, 0, math.pi)
    dr.scale = (C.DRAGON_S,) * 3
    d_arm = bpy.data.objects["Object_6"]
    d_meshes = [o for o in _descendants(dr) if o.type == "MESH"]
    d_body = max(d_meshes, key=lambda o: len(o.data.vertices))
    ad = d_arm.animation_data or d_arm.animation_data_create()
    ad.action = None
    for t in list(ad.nla_tracks):            # the glTF importer stashed every clip as an NLA track: drop them
        ad.nla_tracks.remove(t)
    for a in bpy.data.actions:
        a.use_fake_user = True
    for pb in d_arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    dsk = Skel(d_arm)                                  # D_ROOT is identity now → world == character frame
    ref = dsk.basis_from_action(bpy.data.actions["Idle Stand"], [0])[0]
    W0 = dsk.world(dsk.fk(ref[None]))[0]
    head = W0[dsk.idx[dsk.find("DEF-neck.004")], :3, 3]
    tail = W0[dsk.idx[dsk.find("DEF-tail.016")], :3, 3]
    S.R.add("dragon.ref_head", head.round(2).tolist())
    S.R.add("dragon.ref_tail", tail.round(2).tolist())
    S.R.assert_(head[1] > tail[1] + 2, "dragon_faces_plus_y", f"head {head.round(2)} tail {tail.round(2)}")
    S.D = types.SimpleNamespace(root=d_root, top=dr, arm=d_arm, body=d_body, meshes=d_meshes, skel=dsk, ref=ref, W0=W0)

    # ---------------------------------------------------------------- knight: fresh import, rest = bind pose
    new = imp.import_model(C.KNIGHT_GLB, coll=col.chars, cache=False, drop=("Icosphere",))
    k_arm = next(o for o in new if o.type == "ARMATURE")
    k_meshes = [o for o in new if o.type == "MESH"]
    normalise_rest(k_arm, k_meshes)
    if k_arm.animation_data:
        for t in list(k_arm.animation_data.nla_tracks):
            k_arm.animation_data.nla_tracks.remove(t)
    k_root = X.empty("K_ROOT", (0, 0, 0), coll=col.chars, kind="ARROWS")
    k_inner = X.empty("K_INNER", (0, 0, 0), coll=col.chars, parent=k_root)
    k_inner.rotation_euler = (0, 0, math.pi)
    ks = C.KNIGHT_H / 2.85
    k_inner.scale = (ks,) * 3
    for o in new:
        if o.parent is None:
            o.parent = k_inner
    bpy.context.view_layer.update()
    ksk = Skel(k_arm)
    Wk = ksk.world(ksk.fk(ksk.ident(1)))[0]
    lf = Wk[ksk.idx[ksk.find("mixamorig:LeftFoot")], :3, 3]
    lt = Wk[ksk.idx[ksk.find("mixamorig:LeftToeBase")], :3, 3]
    rh = Wk[ksk.idx[ksk.find("mixamorig:RightHand")], :3, 3]
    S.R.assert_(lt[1] > lf[1] and rh[0] > 0.3, "knight_faces_plus_y_right_is_plus_x", f"toe {lt.round(2)} rhand {rh.round(2)}")
    hz = max((o.matrix_world @ Vector(c)).z for o in k_meshes for c in o.bound_box)
    S.R.add("knight.height_m", round(hz, 3))
    S.K = types.SimpleNamespace(root=k_root, inner=k_inner, arm=k_arm, meshes=k_meshes, skel=ksk, W0=Wk)

    # ---------------------------------------------------------------- sword (Poly Haven antique_estoc, CC0) + scabbard
    sw_objs = imp.import_model(C.ESTOC_DIR + "model_antique_estoc/antique_estoc_2k.gltf", coll=col.props, cache=False)
    sw_names = [o.name for o in sw_objs]
    sw_mesh = [o for o in sw_objs if o.type == "MESH"]
    sword = bpy.data.objects.new("K_Sword", imp.bake_world(sw_mesh, "K_Sword", remove_sources=True))
    col.props.objects.link(sword)
    for n in sw_names:
        o = bpy.data.objects.get(n)
        if o is not None and o is not sword:
            bpy.data.objects.remove(o, do_unlink=True)
    co = np.array([v.co for v in sword.data.vertices])
    cen = co.mean(0)
    u, s_, vt = np.linalg.svd(co - cen, full_matrices=False)
    ax = vt[0]
    t = (co - cen) @ ax
    width = np.linalg.norm((co - cen) - np.outer(t, ax), axis=1)
    bins = np.linspace(t.min(), t.max(), 61)
    wid = np.array([width[(t >= bins[i]) & (t < bins[i + 1])].max(initial=0) for i in range(60)])
    gi = int(np.argmax(wid))                        # cross-guard = widest slice
    tg = 0.5 * (bins[gi] + bins[gi + 1])
    if tg > 0.5 * (t.min() + t.max()):              # make +ax point from guard toward the blade tip
        ax, t, tg = -ax, -t, -tg
    L = float(t.max() - t.min())
    grip_t = tg - 0.09
    grip = cen + ax * grip_t
    tip = cen + ax * t.max()
    S.R.add("sword.length_m", round(L, 3))
    # sword local frame: origin at the grip, +Y = blade (guard→tip), +X = guard bar direction (2nd PCA axis)
    bx = vt[1] - ax * np.dot(vt[1], ax)
    bx /= np.linalg.norm(bx)
    bz = np.cross(bx, ax)
    Msw = np.eye(4)
    Msw[:3, 0], Msw[:3, 1], Msw[:3, 2], Msw[:3, 3] = bx, ax, bz, grip
    sword.data.transform(Matrix(np.linalg.inv(Msw).tolist()))
    sword.matrix_world = Matrix.Identity(4)
    S.R.add("sword.tip_local_y", round(float(t.max() - grip_t), 3))
    # attach to the right hand: grip between index & ring knuckles, blade along ring→index (the fist's grip axis)
    ksk = S.K.skel
    root_inv = np.linalg.inv(np.array(k_root.matrix_world))
    W = ksk.world(ksk.fk(ksk.ident(1)))[0]
    hand = ksk.find("mixamorig:RightHand")
    p_hand = W[ksk.idx[hand], :3, 3]
    p_ix = W[ksk.idx[ksk.find("mixamorig:RightHandIndex1")], :3, 3]
    p_rg = W[ksk.idx[ksk.find("mixamorig:RightHandRing1")], :3, 3]
    p_md = W[ksk.idx[ksk.find("mixamorig:RightHandMiddle1")], :3, 3]
    gy = (p_ix - p_rg) / np.linalg.norm(p_ix - p_rg)
    gpt = 0.55 * (p_ix + p_rg) / 2 + 0.45 * p_hand + 0.25 * (p_md - p_hand)
    fing = p_md - p_hand
    gx = fing - gy * np.dot(fing, gy)
    gx /= np.linalg.norm(gx)
    gz = np.cross(gx, gy)
    Mg = np.eye(4)
    Mg[:3, 0], Mg[:3, 1], Mg[:3, 2], Mg[:3, 3] = gx, gy, gz, gpt
    sword.parent = k_arm
    sword.parent_type = "BONE"
    sword.parent_bone = hand
    bpy.context.view_layer.update()
    sword.matrix_world = Matrix((np.array(k_root.matrix_world) @ Mg).tolist())
    # scabbard on the LEFT hip (character +X is right → left = -X), blade down-back
    hips = ksk.find("mixamorig:Hips")
    p_hip = W[ksk.idx[hips], :3, 3]
    sc_top = p_hip + np.array([-0.30, 0.10, -0.02])        # v2: further out + forward (the cross-draw hand stays outside the belly)
    down = np.array([0.0, -0.45, -1.0])
    down /= np.linalg.norm(down)
    sx = np.array([1.0, 0, 0])
    sy = down
    sz = np.cross(sx, sy)
    Msc = np.eye(4)
    Msc[:3, 0], Msc[:3, 1], Msc[:3, 2], Msc[:3, 3] = sx, sy, sz, sc_top
    blade_len = float(S.R.get("sword.tip_local_y"))
    scab = X.mesh_obj("K_Scabbard", *_box_mesh((0.035, 0.0, -0.02), (0.035, blade_len - 0.1, 0.02)), coll=col.props,
                      mat=X.simple_mat("M_Scabbard", (0.06, 0.035, 0.02), rough=0.6))
    scab.parent = k_arm
    scab.parent_type = "BONE"
    scab.parent_bone = hips
    sheathed = bpy.data.objects.new("K_SwordSheathed", sword.data)
    col.props.objects.link(sheathed)
    sheathed.parent = k_arm
    sheathed.parent_type = "BONE"
    sheathed.parent_bone = hips
    bpy.context.view_layer.update()
    Mroot = np.array(k_root.matrix_world)
    scab.matrix_world = Matrix((Mroot @ Msc).tolist())
    Msh = Msc.copy()
    Msh[:3, 3] = sc_top - sy * 0.09                    # blade (+Y) down into the scabbard, guard at the mouth
    sheathed.matrix_world = Matrix((Mroot @ Msh).tolist())
    S.K.sword, S.K.sheathed, S.K.scabbard = sword, sheathed, scab
    S.K.hand_bone, S.K.hips_bone = hand, hips

    # ---------------------------------------------------------------- audit proxies (deform with the rigs)
    S.K.proxy = make_proxy("PX_Knight", k_meshes, k_arm, 0.12, col.proxy)     # copies keep the armature parent
    S.D.proxy = make_proxy("PX_Dragon", [d_body], d_arm, 0.32, col.proxy)
    bpy.context.view_layer.update()
    S.R.add("proxy.knight_tris", len(S.K.proxy.data.polygons))
    S.R.add("proxy.dragon_tris", len(S.D.proxy.data.polygons))

    # ---------------------------------------------------------------- static BVHs
    sup_names = ["ArenaFloor", "GrandStairs", "CourtyardPaving", "Dais", "GoldHeap"]
    S.ground_bvh, _, _ = mesh_bvh([bpy.data.objects[n] for n in sup_names])
    solid = [bpy.data.objects[n] for n in ("Wall_E", "Wall_W", "Wall_N_Gable", "Wall_S_E", "Wall_S_W", "GatePiers",
                                            "StairCheek_E", "StairCheek_W", "StairPedestals", "Colossus_Plinth_E",
                                            "Colossus_Plinth_W", "VaultHaunch_E", "VaultHaunch_W", "AisleRoof_E",
                                            "AisleRoof_W", "Clerestory_E", "Clerestory_W")]
    solid += [o for o in bpy.data.collections["V3_COVER"].all_objects if o.type == "MESH" and not o.hide_render]
    solid += [o for o in bpy.data.collections["V2_STATUES_INSIDE"].all_objects if o.type == "MESH" and "Plinth" in o.name]
    S.static_solids = solid
    S.solid_bvh, _, _ = mesh_bvh(solid)
    S.R.add("static_solids", len(solid))

    def ground(x, y, z_from=60.0):
        h = S.ground_bvh.ray_cast(Vector((x, y, z_from)), Vector((0, 0, -1)))
        return h[0].z if h[0] is not None else 0.0
    S.ground = ground
    S.R.add("ground.stairs_mid", round(ground(0, -13.0), 2))


def _box_mesh(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    x0, x1 = -x1, x1
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return v, f
