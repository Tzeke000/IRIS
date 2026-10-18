"""Headless Blender test for lib/physics.py. Through the slice:
    scripts/server/render.sh t-libphys --wait --collect -P ~/.local/bin/blender -b --factory-startup \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/tests/bpy_test_physics.py"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import bpyx as X  # noqa: E402
from lib import checks as K  # noqa: E402
from lib import physics as PH  # noqa: E402

FAILS = []


def check(name, cond, info=""):
    print(f"[bpy_test_physics] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAILS.append(name)


def cube(name, size, loc, coll=None):
    h = size / 2
    o = X.mesh_obj(name, [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)],
                   [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)], coll=coll)
    o.location = loc
    return o


def clean_scene_state(tag, main):
    leftovers = [o.name for o in bpy.data.objects if o.name.startswith(("SIMD_", "SIMP_"))]
    check(f"{tag}: sim scene gone, film scene restored, no rigid bodies left",
          len(bpy.data.scenes) == 1 and bpy.context.scene == main and main.rigidbody_world is None and not leftovers
          and all(o.rigid_body is None for o in bpy.data.objects) and not any(c.name.endswith("_RB") for c in bpy.data.collections),
          f"scenes={len(bpy.data.scenes)} leftovers={leftovers}")


def main():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, 60

    # --- settle: a tilted 1 m cube dropped onto a floor slab comes to rest on a face ---
    floor = X.boxes_mesh("Floor", [(-10, -10, -1, 10, 10, 0)], bottom=True)
    rock = cube("Rock", 1.0, (0.3, -0.2, 2.5))
    rock.rotation_euler = (0.4, 0.3, 0.2)
    keep_data = rock.data
    bpy.context.view_layer.update()
    res = PH.settle([rock], statics=[floor], frames=120)
    o, M, moved = res[0]
    z = M.translation.z
    R = np.array(M.to_3x3())
    face_up = np.max(np.abs(R[2]))                              # some local axis points straight up when resting flat
    check("settle: resting on a face at z ≈ 0.5", abs(z - 0.5) < 0.03 and face_up > 0.99, f"z={z:.3f} |up|={face_up:.3f}")
    check("settle: at rest (moved < 1 cm in the last 12 frames)", moved < 0.01, f"{moved:.4f} m")
    check("settle: pose applied to the original, mesh untouched", (rock.matrix_world.translation - M.translation).length < 1e-6
          and rock.data == keep_data and len(rock.data.vertices) == 8)
    clean_scene_state("settle", sc)
    floor.hide_render = False

    # --- Sim → keys: a ball falls onto the floor; the film gets plain keys, nothing of Bullet ---
    ball = cube("Ball", 0.5, (0.0, 3.0, 4.0))
    ball2 = cube("Glider", 0.5, (-5.0, -3.0, 0.5))
    X.bake_vec(ball2, "location", np.c_[np.linspace(-5, 4, 10), np.full(10, -3.0), np.full(10, 2.0)],
               frames=np.arange(1, 11), thin=False)
    with PH.Sim((1, 40), substeps=20, iters=15) as S:
        S.add(ball, "ACTIVE", "BOX", density=500)
        S.add(ball2, "ACTIVE", "BOX", density=500, kinematic=True)
        S.release(ball2, 10)
        S.passive_box("FloorCol", (-10, -10, -1), (10, 10, 0))
        t = S.bake()
        MW = S.sample([ball, ball2])
        S.to_keys([ball, ball2], n=60, MW=MW)
    check("sim baked (seconds)", t < 30, f"{t:.2f} s")
    zs = MW[0, :, 2, 3]
    check("ball fell and landed on the floor (z → 0.25)", zs[0] > 3.9 and abs(zs[-1] - 0.25) < 0.03, f"{zs[0]:.2f} → {zs[-1]:.3f}")
    worst = 0.0
    for f in range(1, 41):
        sc.frame_set(f)
        worst = max(worst, float(np.abs(np.array(ball.matrix_world) - MW[0, f - 1]).max()))
    check("keys reproduce the sampled sim at every frame", worst < 1e-4, f"{worst:.2e}")
    sc.frame_set(60)
    check("after the sim: last pose held", abs(ball.matrix_world.translation.z - zs[-1]) < 1e-5)
    sc.frame_set(30)
    a = ball.matrix_world.translation.copy()
    sc.frame_set(5)
    sc.frame_set(30)
    check("deterministic out of order (keys, not a cache)", (ball.matrix_world.translation - a).length < 1e-9)
    gx = MW[1, :, 0, 3]
    v_key = (gx[9] - gx[8])                                      # 1 m/frame on the keys
    v_free = (gx[14] - gx[13])
    check("release(): keyed speed carries into the sim (rocket4 launch)", abs(v_free - v_key) / v_key < 0.02,
          f"keyed {v_key:.3f} m/f, free {v_free:.3f} m/f")
    clean_scene_state("Sim", sc)
    check("bodies kept in the film scene", ball.name in sc.collection.objects and ball2.name in sc.collection.objects)

    # --- grounding on a sloped surface ---
    slope = X.mesh_obj("Slope", [(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)], [(0, 1, 2, 3)])
    slope.location = (40, 0, 0)
    slope.rotation_euler = (0, math.radians(10), 0)
    props = [cube(f"Prop{i}", 1.0, (40 + 3 * i, 2 * i, 6.0 + i)) for i in range(3)]
    sunk = cube("Sunk", 1.0, (36, -5, -2.0))
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    g = PH.ground_from([slope], dg)
    fl = PH.floating(props + [sunk], g, tol=0.01, sink=0.2)
    check("floating() reports hovering AND sunk props", sorted(n for n, _ in fl) == ["Prop0", "Prop1", "Prop2", "Sunk"], str(fl))
    sh = PH.drop_to_ground(props, g)
    PH.drop_to_ground([sunk], g, only_down=True)
    dg = bpy.context.evaluated_depsgraph_get()
    gaps = [PH.gap_of(PH.bottom_points(p, dg), g)[0] for p in props]
    check("drop_to_ground: every prop touches the slope", max(abs(x) for x in gaps) < 1e-3, f"{[round(x, 5) for x in gaps]} shifts {[round(s, 2) for _, s in sh]}")
    check("only_down never lifts a sunk prop", abs(sunk.location.z + 2.0) < 1e-9)
    check("floating() clean after the drop", not PH.floating(props, g, tol=0.01))
    cols = PH.lowest_columns(K.bpy_world_tris(props, dg)[0])
    nfl = len(K.floating(cols, g, gap=(0.05, 15.0)))
    check("edges mode: a flat-bottomed box on a 10° slope floats on its downhill edge (2 of 4 columns each)",
          len(cols) == 12 and nfl == 6, f"{nfl} of {len(cols)} columns float (gap 1 m·tan 10° = 0.18 m)")

    print(f"[bpy_test_physics] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


main()
