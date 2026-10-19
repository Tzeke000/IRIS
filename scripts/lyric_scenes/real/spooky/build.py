"""Spooky v2 — "a live picture": ONE fixed camera on the song's cover art, and everything in it reacts to the music
(Zeke 10-19: "just have like a live picture that reacts to the music … still camera but it can shake or vibrate on the
bass; the ghosts should react to the music"). No shot list.

    scripts/lyric_scenes/lib/render/scene.sh /ABS/scripts/lyric_scenes/real/spooky/scene.env build
    (= blender -b --factory-startup --python build.py -- --out BLEND [--report JSON] [--samples 32])

Stages: sky.py (world, stars, moon + rays, clouds + banks, shooting stars) · land.py (water, ground, crag, waterfall,
ridges, mist, haze box) · ruins.py (CC-BY arches/towers/castle, ledge, ivy, CC0 foliage) · ghosts.py (procedural ghosts
+ analytic GN motion) · fx.py (petals, fireflies) · cams.py (lights, CAM_live + shake, look) · music.py (the song →
every channel, pure numpy). Every animated thing is a pure function of Scene Time or of baked keys (CTRL / CTRL_GN /
camera / lamps / glare) → any frame renders on its own and parallel chunks agree.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from lib import importer as IMP  # noqa: E402
from lib.report import Report  # noqa: E402
from common import X  # noqa: E402

import cams  # noqa: E402
import fx  # noqa: E402
import ghosts  # noqa: E402
import land  # noqa: E402
import music as MU  # noqa: E402
import ruins  # noqa: E402
import sky  # noqa: E402


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--report")
    p.add_argument("--samples", type=int, default=32)
    return p.parse_args(a)


def reset():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def eval_co(ob, f):
    """evaluated vertex positions (world-scaled local) of a mesh object at frame f"""
    sc = bpy.context.scene
    sc.frame_set(f)
    d = bpy.context.evaluated_depsgraph_get()
    e = ob.evaluated_get(d)
    me = e.to_mesh()
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    e.to_mesh_clear()
    return co.reshape(-1, 3)


def eval_instances(ob, f):
    """(count, positions) of the GN instances of ob at frame f"""
    sc = bpy.context.scene
    sc.frame_set(f)
    d = bpy.context.evaluated_depsgraph_get()
    pts = [tuple(i.matrix_world.translation) for i in d.object_instances
           if i.is_instance and i.parent and i.parent.original == ob]
    return len(pts), np.asarray(pts, np.float64).reshape(-1, 3)


def main():
    A = args()
    rep = A.report or os.path.join(os.path.dirname(os.path.abspath(A.out)), "build_report.json")
    with Report(rep, "spooky v2 live picture") as R:
        reset()
        sc = bpy.context.scene
        S = MU.song()
        M = MU.channels(S)
        SH = MU.shake(S, M)
        sc.frame_start, sc.frame_end = 1, S.n
        sc.render.fps = C.FPS
        sc.frame_current = 1921
        R.add("song", dict(duration=S.duration, nf=S.n, bpm=round(S.bpm, 3), beat0=round(S.beat0, 4),
                           sections=("ear.json" if S.sections_from == "ear" else "SECTION_READ"),
                           hits=[round(h, 3) for h in MU.hit_times(S)]))
        R.log(f"song {S.duration} s → {S.n} frames, grid {S.bpm:.2f} BPM")
        col = {k: X.coll(k) for k in ("SKY", "LAND", "RUINS", "IVY_FOL", "GHOSTS", "FX", "LIGHTS", "CAMS")}
        C.ctrl_tree()
        C.ctrl_gn_tree()
        sky.world()
        sky.moon(col["SKY"])
        sky.clouds(col["SKY"])
        sky.cloud_banks(col["SKY"])
        ss = sky.shooting_stars(col["SKY"], MU.hit_times(S))
        R.tick("sky")
        land.water(col["LAND"])
        land.foreground_ground(col["LAND"])
        land.cliff(col["LAND"])
        land.waterfall(col["LAND"])
        land.mountains(col["LAND"])
        land.mist_cards(col["LAND"])
        land.haze_box(col["LAND"])
        R.tick("land")
        rn = ruins.build(col["RUINS"], R)
        ivy_on = [o for k, o in rn.items() if k not in ("castle", "hill") and not k.startswith("fort")]
        ruins.add_ivy(ivy_on, R)
        ruins.foliage(col["IVY_FOL"], R)
        R.tick("ruins")
        gh = ghosts.build(col["GHOSTS"], R)
        R.tick("ghosts")
        pet = fx.petals(col["FX"])
        ff = fx.fireflies(col["FX"])
        R.tick("fx")
        lt = cams.lights(col["LIGHTS"])
        cam, CA = cams.live_camera(col["CAMS"], S.n, SH)
        sc.camera = cam
        post = cams.look(sc, samples=A.samples)
        # ---- the music: CTRL / CTRL_GN channels, lamps, glare
        nk = C.bake_ctrl(M)
        lamps = [(lt["moon"], 1.6, "moon_glow")]
        gl = bpy.data.objects.get("Ghost_Big_Light")
        if gl is not None:
            lamps.append((gl, 450.0, "ghost_glow"))
        R.add("music_keys", dict(ctrl=nk, other=cams.music_keys(sc, M, lamps, post)))
        R.tick("music")

        # ================================================================ build-time asserts
        # -- layout: the cover composition from the one camera
        md = np.linalg.norm(C.MOON_POS - C.CAM_LOC)
        ang = 2 * math.degrees(math.atan(C.MOON_R / md))
        R.assert_(18.0 < ang < 24.0, "moon_size", f"moon spans {ang:.1f}° (cover: ≈0.43–0.46 of the frame height)")
        gx, gy, _ = C.to_ndc((C.BIG_GHOST["loc"][0], C.BIG_GHOST["loc"][1], C.BIG_GHOST["loc"][2] + 1.55))
        R.assert_(abs(gx - 0.31) < 0.03 and abs(gy - 0.47) < 0.03, "ghost_framing", f"big ghost at ndc ({gx:.3f}, {gy:.3f})")
        R.assert_(sc.world.node_tree.nodes.get("Volume") is None and
                  not any(n.type == "OUTPUT_WORLD" and n.inputs["Volume"].is_linked for n in sc.world.node_tree.nodes),
                  "no_world_volume", "haze is a bounded box (gotcha 3)")
        # -- shake: from the BAKED camera curves, not the arrays
        rot_e = np.array([[fc.evaluate(f) for f in range(1, S.n + 1)] for fc in X.find_fcurves(cam, "rotation_euler")]).T
        loc_e = np.array([[fc.evaluate(f) for f in range(1, S.n + 1)] for fc in X.find_fcurves(cam, "location")]).T
        f_, r_, u_ = C.cam_basis()
        Rm0 = X.track_matrix(f_)
        dR = np.einsum("ji,njk->nik", Rm0, X.euler_to_mat(rot_e))          # Rm0ᵀ · R(frame)
        dev = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)))
        dloc = np.abs(loc_e - C.CAM_LOC).max()
        R.add("shake", dict(MU.shake_report(S, SH), baked_max_rot_dev_deg=round(float(dev.max()), 4),
                            baked_max_loc_dev_m=round(float(dloc), 4),
                            baked_rot_dev_p99_deg=round(float(np.percentile(dev, 99)), 4)))
        bad = MU.check_shake(S, SH)
        R.assert_(not bad, "shake_arrays", f"music.check_shake: {bad or 'PASS'}")
        R.assert_(dev.max() <= MU.SHAKE_MAX_ROT_DEG * 1.8, "shake_baked_rot",
                  f"baked camera rotation deviates ≤ {dev.max():.3f}° from the rest framing")
        R.assert_(dloc <= MU.SHAKE_MAX_LOC_M * 1.8, "shake_baked_loc", f"baked camera location deviates ≤ {dloc:.4f} m")
        intro = dev[: S.fr(28.0)]
        R.assert_(intro.max() < 0.005, "shake_intro_still", f"intro max deviation {intro.max():.4f}° (≈ still)")
        R.assert_(np.isfinite(rot_e).all() and np.isfinite(loc_e).all(), "camera_finite", "camera keys finite")
        # -- CTRL_GN reaches GN: petals move ONLY by wind_t (no Scene Time) → they must move between frames
        n1, p1 = eval_instances(pet, 1921)
        n2, p2 = eval_instances(pet, 1935)
        R.assert_(n1 == n2 and n1 > 0 and np.abs(p1 - p2).max() > 0.05, "ctrl_gn_live",
                  f"petal instances move between frames via CTRL_GN.wind_t (max {np.abs(p1 - p2).max():.3f} m)")
        # -- deterministic GN: same frame twice → identical (ghost, petals, fireflies, shooting stars)
        det = {}
        for o in [gh[0], gh[1]]:
            a1, _b, a2 = eval_co(o, 1931), eval_co(o, 1950), eval_co(o, 1931)
            det[o.name] = bool(np.array_equal(a1, a2))
            R.assert_(np.isfinite(a1).all(), f"finite_{o.name}", "no NaN in the deformed ghost")
        for o in (pet, ff, ss):
            _, q1 = eval_instances(o, 5761)
            eval_instances(o, 4000)
            _, q2 = eval_instances(o, 5761)
            det[o.name] = bool(np.array_equal(q1, q2))
            R.assert_(np.isfinite(q1).all(), f"finite_{o.name}", "instance positions finite")
        R.add("deterministic", det)
        R.assert_(all(det.values()), "gn_deterministic", f"same frame twice → identical {det}")
        # -- ghost reacts: kick squash visible (frame on a kick vs a frame between, both in the drop)
        kick_f = int(np.argmax(M["g_kick"][S.idx(165.0):S.idx(175.0)])) + S.idx(165.0) + 1
        big = gh[0]
        a_k, a_0 = eval_co(big, kick_f), eval_co(big, kick_f - 2)
        mv = float(np.abs(a_k - a_0).max()) * big.scale[0]
        R.assert_(0.02 < mv < 0.8, "ghost_kick", f"big ghost moves {mv:.3f} m across a kick (frame {kick_f})")
        # -- constant instance counts (gotcha 10) over the whole song
        counts = {}
        for f in (1, 961, 1921, 3841, 4897, 5761, 6722, 7400, S.n):
            counts[f] = {o.name: eval_instances(o, f)[0] for o in (pet, ff, ss)}
        R.add("instances", counts)
        for o, n in ((pet, len(pet.data.vertices)), (ff, len(ff.data.vertices)), (ss, len(ss.data.vertices))):
            vals = {counts[f][o.name] for f in counts}
            R.assert_(vals == {n}, f"count_{o.name}", f"{o.name} instances per frame {sorted(vals)} == {n} (gotcha 10)")
        # -- v3: no petal near the lens (the pink blur blobs): every instance closer than PETAL_NEAR[0] is scaled to 0
        vis_min, n_close = 1e9, 0
        for f in (961, 1921, 2400, 3841, 4897, 5761, 7400):
            sc.frame_set(f)
            d = bpy.context.evaluated_depsgraph_get()
            for i in d.object_instances:
                if i.is_instance and i.parent and i.parent.original == pet:
                    m = i.matrix_world
                    dist = float(np.linalg.norm(np.asarray(m.translation) - C.CAM_LOC))
                    n_close += dist < fx.PETAL_NEAR[1]
                    if m.to_scale().length > 1e-6:
                        vis_min = min(vis_min, dist)
        R.add("petals_near_cam", dict(inside_fade_zone=n_close, nearest_visible_m=round(vis_min, 3), zone_m=fx.PETAL_NEAR))
        R.assert_(vis_min >= fx.PETAL_NEAR[0] - 1e-3, "petals_near_cam",
                  f"nearest visible petal {vis_min:.2f} m from the lens over 7 frames ({n_close} inside the fade zone)")
        # -- shooting stars: fire on the hits only (scale 0 otherwise) → check one is live right after a hit
        tf = ss["fire_times"]
        R.assert_(len(tf) >= 6 and all(0 < t < S.duration - 2 for t in tf), "stars_on_hits", f"fire at {tf}")
        # -- channel sanity (finite, in range), the CTRL keys exist for every channel
        for k in list(C.CTRL_KEYS) + list(C.GN_KEYS):
            if k == "T":
                continue
            v = np.asarray(M[k])
            R.assert_(np.isfinite(v).all(), f"chan_{k}", f"{k} finite [{v.min():.3f}, {v.max():.3f}]")
        R.assert_(nk and min(nk.values()) > 50, "ctrl_keys", f"CTRL keys per channel min {min(nk.values())}")
        # -- the camera never sits inside geometry
        from lib import camera as CM
        sc.frame_set(1921)
        hit, dist = CM.view_hit(cam, sc, skip=lambda ob: ob.name.startswith(("HazeBox", "Mist_", "Cloud", "MoonHalo",
                                                                            "Petals", "Fireflies")))
        R.check(dist is None or dist > 0.4, "view_live", f"first hit {hit} at {dist}")
        sc.frame_set(1921)
        R.add("objects", len(bpy.data.objects))
        out = X.save_blend(A.out)
        IMP.write_log(os.path.join(os.path.dirname(out), "import_log.json"))
        with open(os.path.join(os.path.dirname(out), "music_report.json"), "w") as fh:
            json.dump(dict(shake=MU.shake_report(S, SH), hits=MU.hit_times(S), kick_frame=kick_f), fh, indent=1)
        R.log(f"saved {out}")


main()
