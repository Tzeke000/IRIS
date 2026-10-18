"""Spooky v2 — the world INSIDE the song's cover art. Look-dev round: world + look + ghost motion,
named still cameras only (no shot plan until the drop times are confirmed by ear).

    scripts/lyric_scenes/lib/render/scene.sh /ABS/scripts/lyric_scenes/real/spooky/scene.env build
    (= blender -b --factory-startup --python build.py -- --out BLEND [--report JSON] [--samples 32])

Stages: sky.py (world, stars, moon, clouds, shooting stars) · land.py (water, ground, cliff, waterfall, ridges, mist,
haze box) · ruins.py (CC-BY arches/towers/castle, ledge, ivy, CC0 foliage) · ghosts.py (procedural ghosts + analytic
GN motion) · fx.py (petals, fireflies) · cams.py (lights, CAM_* stills, look). Every animated thing is a pure function
of Scene Time or a key → parallel chunks are safe. CTRL (common.py) holds the music-reactive channels.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from lib import checks as K  # noqa: E402
from lib import importer as IMP  # noqa: E402
from lib import song as SONG  # noqa: E402
from lib.report import Report  # noqa: E402
from common import X  # noqa: E402

import cams  # noqa: E402
import fx  # noqa: E402
import ghosts  # noqa: E402
import land  # noqa: E402
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


def main():
    A = args()
    rep = A.report or os.path.join(os.path.dirname(os.path.abspath(A.out)), "build_report.json")
    with Report(rep, "spooky v1") as R:
        reset()
        sc = bpy.context.scene
        S = SONG.Song.load(C.SONG_MAP, fps=C.FPS)
        sc.frame_start, sc.frame_end = 1, S.n
        sc.render.fps = C.FPS
        sc.frame_current = 301
        R.add("song", dict(duration=S.duration, nf=S.n, bpm=round(S.bpm, 3), beat0=round(S.beat0, 4)))
        R.log(f"song {S.duration} s → {S.n} frames, grid {S.bpm:.2f} BPM")
        col = {k: X.coll(k) for k in ("SKY", "LAND", "RUINS", "IVY_FOL", "GHOSTS", "FX", "LIGHTS", "CAMS")}
        C.ctrl_tree()
        sky.world()
        moon = sky.moon(col["SKY"])
        cl = sky.clouds(col["SKY"])
        ss = sky.shooting_stars(col["SKY"])
        R.tick("sky")
        land.water(col["LAND"])
        land.foreground_ground(col["LAND"])
        land.cliff(col["LAND"])
        land.waterfall(col["LAND"])
        land.mountains(col["LAND"])
        land.mist_cards(col["LAND"])
        haze = land.haze_box(col["LAND"])
        R.tick("land")
        rn = ruins.build(col["RUINS"], R)
        ivy_on = [o for k, o in rn.items() if k not in ("castle", "hill") and not k.startswith("fort")]
        ruins.add_ivy(ivy_on, R)
        fol = ruins.foliage(col["IVY_FOL"], R)
        R.tick("ruins")
        gh = ghosts.build(col["GHOSTS"], R)
        R.tick("ghosts")
        pet = fx.petals(col["FX"])
        ff = fx.fireflies(col["FX"])
        R.tick("fx")
        cams.lights(col["LIGHTS"])
        cm = cams.cameras(col["CAMS"], gh)
        sc.camera = cm["a_hero"]
        cams.look(sc, samples=A.samples)
        R.tick("cams+look")

        # ---------------- build-time asserts
        dg = bpy.context.evaluated_depsgraph_get()
        md = np.linalg.norm(C.MOON_POS - C.HERO_CAM)
        ang = 2 * math.degrees(math.atan(C.MOON_R / md))
        R.assert_(20.0 < ang < 30.0, "moon_size", f"moon spans {ang:.1f}° from the hero camera (cover: huge)")
        R.assert_(sc.world.node_tree.nodes.get("Volume") is None and
                  not any(n.type == "OUTPUT_WORLD" and n.inputs["Volume"].is_linked for n in sc.world.node_tree.nodes),
                  "no_world_volume", "haze is a bounded box (gotcha 3)")
        counts = {}
        for f in (1, 2400, 7000):
            sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            counts[f] = {o.name: K.bpy_instances(o, dg) for o in (pet, ff, ss)}
        R.add("instances", counts)
        for o, n in ((pet, 900), (ff, 200), (ss, 6)):
            vals = {counts[f][o.name] for f in counts}
            R.assert_(vals == {n}, f"count_{o.name}", f"{o.name} instances per frame {sorted(vals)} == {n} (gotcha 10)")
        # ghost motion is deterministic + bounded: evaluate the big ghost at two frames twice
        big = gh[0]

        def ghost_co(f):
            sc.frame_set(f)
            d = bpy.context.evaluated_depsgraph_get()
            me = big.evaluated_get(d).to_mesh()
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            big.evaluated_get(d).to_mesh_clear()
            return co.reshape(-1, 3)
        a1, b1, a2 = ghost_co(300), ghost_co(317), ghost_co(300)
        R.assert_(np.array_equal(a1, a2), "ghost_deterministic", "same frame → identical vertices")
        mv = float(np.abs(b1 - a1).max()) * big.scale[0]
        R.assert_(0.005 < mv < 0.6, "ghost_moves", f"big ghost max vertex motion over 17 frames {mv:.3f} m")
        R.assert_(np.isfinite(a1).all(), "ghost_finite", "no NaN in the deformed ghost")
        sc.frame_set(301)
        # stills never sit inside geometry (cheap ray test along the view axis)
        from lib import camera as CM
        for nm, o in cm.items():
            hit, dist = CM.view_hit(o, sc, skip=lambda ob: ob.name.startswith(("HazeBox", "Mist_", "Cloud_", "MoonHalo",
                                                                               "Petals", "Fireflies")))
            R.check(dist is None or dist > 0.4, f"view_{nm}", f"first hit {hit} at {dist}")
        R.add("objects", len(bpy.data.objects))
        out = X.save_blend(A.out)
        IMP.write_log(os.path.join(os.path.dirname(out), "import_log.json"))
        R.log(f"saved {out}")


main()
