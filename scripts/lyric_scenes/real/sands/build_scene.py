"""Sands — build the music-video scene (Blender 5.2, headless, through lib/render/scene.sh ENV build).

    scene.sh /ABS/real/sands/scene.env build      → BLEND (+ build_report.json)

Zeke's brief (memory music_scene_plan_2026-10-30_sands_desert_cube): realistic dunes to the horizon, saguaros, golden
light, a futuristic LED cube floating ~15 m up carrying Tzeke000's logo that FLIPS and ROTATES on the kick and snare
(differently), crackling energy on the drops, a camel walking by far off in a calm part; cuts on downbeats, camera
never reacts. Modules: terrain (height fn), timeline (music → per-frame values), world, cube, shots.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # …/scripts/lyric_scenes → `lib`

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from lib import bpyx as X  # noqa: E402
from lib import camera as CM  # noqa: E402
from lib import checks, look  # noqa: E402
from lib.report import Report  # noqa: E402
from lib.song import Song  # noqa: E402

import cube as CB  # noqa: E402
import shots as SH  # noqa: E402
import terrain as TR  # noqa: E402
import timeline as TL  # noqa: E402
import world as W  # noqa: E402


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--no-cacti", action="store_true")
    return p.parse_args(a)


def main():
    A = args()
    with Report(A.report, "sands") as R:
        S = Song.load(A.map, fps=30)
        R.log(S.report())
        n = S.n
        T = TL.build(S)
        tracks = TL.strike_tracks(T, TR.height)
        R.add("music", dict(slams=len(T["slams"]), flips=len(T["flips"]), strikes=len(T["strikes"]),
                            entries=[round(e / 30, 3) for e in T["entries"]]))
        # strike-slot reuse safety (gotcha 14): grains/dust of the previous strike gone before a slot jumps
        for tr in tracks:
            ev = [e[0] for e in tr["events"]]
            R.assert_(all(b - a >= TL.STRIKE_LIFE_S * 30 - 1 for a, b in zip(ev, ev[1:])), "strike_slot_reuse")
        for k in ("yaw", "flip_x", "flip_y", "z_off", "scale", "gain", "energy"):
            checks.finite(k, T[k])

        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        sc.frame_start, sc.frame_end = 1, n
        look.eevee_defaults(sc, samples=A.samples, look="Punchy", exposure=-1.1)
        sc.render.film_transparent = False
        sc.eevee.use_shadows = True
        sc.eevee.shadow_resolution_scale = 1.0
        sc.eevee.shadow_pool_size = "1024"          # 512 overflowed (3090/2048 pages: 20 km terrain + sun + cacti)

        world_c = X.coll("World")
        cube_c = X.coll("Cube")
        fx_c = X.coll("FX")
        kit_c = X.coll("Kit")
        life_c = X.coll("Life")

        W.terrain(world_c, R)
        R.tick("terrain")
        W.sky_and_sun(sc, n, world_c, R)

        ng = CB.ctrl_group(T)
        parts = CB.cube(cube_c, T, R, ng)
        core, halo = CB.arc_materials()
        CB.arcs(fx_c, parts["fy"], T, core, halo)
        CB.strikes(fx_c, parts["root"], tracks, core, halo, CB.grain_material(), CB.dust_material(), n)
        CB.lights(fx_c, parts, T, tracks)
        low = CB.ZC + T["z_off"].min() - CB.HS * T["scale"].max()
        corner = CB.ZC + T["z_off"].min() - CB.HS * math.sqrt(3) * T["scale"].max()
        R.assert_(low >= 14.8 and abs(SH.CUBE[2] - CB.ZC) < 1e-6, "cube_lift",
                  f"resting bottom ≥ {low:.2f} m (brief ~15 m); lowest corner mid-flip {corner:.2f} m")
        R.tick("cube")

        # camel: the camel-shot camera + a visible walk line beyond the cube; walk inside the breakdown
        sec = {s["label"]: s for s in S.sections}
        bd0, bd1 = sec["breakdown"]["start"], sec["breakdown"]["end"]
        walk_s = bd1 - bd0 + 2.0
        ccam, (pa, pb), dist = SH.camel_setup(walk_s=walk_s, speed=W.CAMEL_SPEED)
        R.log(f"[camel] camera {ccam.round(1)}, walk line {np.round(pa, 1)} → {np.round(pb, 1)} ({dist:.0f} m beyond the cube)")
        f0, f1 = S.fr(bd0) - 30, S.fr(bd1) + 30
        rig, cobjs, cloc, nose, d = W.camel(life_c, R, (pa, pb), f0, f1, n)
        vel = checks.velocity(cloc, 30)
        ok = (np.einsum("ij,ij->i", nose[f0:f1], vel[f0:f1] / np.maximum(np.linalg.norm(vel[f0:f1], axis=1, keepdims=True), 1e-9)))
        R.assert_(float(ok.min()) > 0.9, "camel_direction", f"nose·velocity min {ok.min():.3f}")

        if not A.no_cacti:
            keep = [(0, 0, 24), (float(pa[0]), float(pa[1]), 30), (float(pb[0]), float(pb[1]), 30),
                    (float(ccam[0]), float(ccam[1]), 12)]
            extra = [(-27, -42), (-41, -20), (34, -27), (-17, 33), (22, 30), (55, -70), (-60, 45)]
            W.cacti(world_c, kit_c, R, keep, extra)
        R.tick("cacti")

        # camera
        cam = bpy.data.objects.new("CAM", bpy.data.cameras.new("CAM"))
        sc.collection.objects.link(cam)
        cam.data.clip_start, cam.data.clip_end = 0.1, 30000.0
        shots = SH.table(S, (ccam, np.r_[(pa + pb) / 2, TR.height(*((pa + pb) / 2)) + 1.0]))
        CA = CM.bake_shots(cam, shots, n)
        # camera never under the sand / never inside the cube
        gl = CA["loc"][:, 2] - TR.height(CA["loc"][:, 0], CA["loc"][:, 1])
        R.assert_(float(gl.min()) > 0.7, "cam_above_sand", f"min {gl.min():.2f} m")
        cubec = np.c_[np.zeros(n), np.zeros(n), CB.ZC + T["z_off"]]
        dc = np.linalg.norm(CA["loc"] - cubec, axis=1)
        R.assert_(float(dc.min()) > CB.HS * 1.8 + 1.0, "cam_off_cube", f"min {dc.min():.2f} m")
        vis = CM.on_screen_by_shot(CA, cubec)
        R.add("cube_on_screen", {k: round(v, 2) for k, v in vis.items()})
        hump = cloc + np.array([0, 0, 1.0])
        cv = CM.on_screen(CA, hump)
        lsh = CA["names"].index("l_camel_far")
        msk = CA["shot"] == lsh
        R.assert_(float(cv[msk].mean()) > 0.95, "camel_in_frame", f"camel on screen {cv[msk].mean():.2f} of the camel shot")
        los = [SH.los_clear(CA["loc"][i], hump[i]) for i in np.flatnonzero(msk)[::15]]
        R.assert_(all(los), "camel_line_of_sight", f"{sum(los)}/{len(los)} sampled frames clear")
        R.add("shots", [(nm, st, round((st - 1) / 30, 2)) for nm, st in zip(CA["names"], CA["starts"])])
        R.add("cuts_on_downbeats", [round(1000 * ((c - 1) / 30 - S.snap((c - 1) / 30)), 1) for c in CA["cuts"]])
        R.tick("camera")

        # look: haze matched to the horizon, glare for LEDs + arcs, gentle lens + vignette
        look.post_chain(sc, mist=dict(color=(0.42, 0.26, 0.15), k=0.5, cap=0.22, start=200.0, depth=7000.0),
                        glare=[dict(type="Fog Glow", threshold=1.0, strength=0.75, size=8),
                               dict(type="Streaks", threshold=4.0, strength=0.12, streaks=4, angle=45.0)],
                        lens=(-0.004, 0.008), vignette=0.80, grain=None)
        sc["audio"] = "/data/iris_assets/sands/song/Sands.wav"
        X.save_blend(A.out)
        R.log(f"[build] saved {A.out}")
        import lib.importer as IM
        IM.write_log(os.path.splitext(A.report)[0] + "_imports.json")


if __name__ == "__main__":
    main()
