"""Headless Blender test for lib/camera.py. Through the slice:
    scripts/server/render.sh t-libcam --wait --collect -P ~/.local/bin/blender -b --factory-startup \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/tests/bpy_test_camera.py"""
import math
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import bpyx as X  # noqa: E402
from lib import camera as CM  # noqa: E402

FAILS = []


def check(name, cond, info=""):
    print(f"[bpy_test_camera] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAILS.append(name)


def dolly(i, u, n):
    return dict(cam=(0.0, -20.0 + 10.0 * u / n, 2.0), tgt=(0.0, 0.0, 2.0), lens=35.0, fstop=4.0)


def orbit(i, u, n):                                    # two full turns inside the shot
    a = 4 * math.pi * u / n
    return dict(cam=(10 * math.cos(a), 10 * math.sin(a), 3.0), tgt=(0.0, 0.0, 1.0), lens=50.0, fstop=2.8, focus=9.0)


def tilted(i, u, n):                                   # explicit up vector + roll
    return dict(cam=(5.0, -5.0, 8.0), tgt=(0.0, 0.0, 0.0), lens=24.0 + u, fstop=8.0, up=(0.3, 0.0, 1.0), roll=0.2)


def main():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    n = 90
    sc.frame_start, sc.frame_end = 1, n
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    shots = [CM.Shot(1, "dolly", dolly), CM.Shot(31, "orbit", orbit), CM.Shot(61, "tilted", tilted)]
    A = CM.bake_shots(cam, shots, n)

    # --- cut safety, read back from the BAKED curves (every channel, object + data) ---
    check("scene['cuts'] + scene['shots'] written", list(sc["cuts"]) == [31, 61] and sc["shots"] == "1:dolly;31:orbit;61:tilted",
          f"{list(sc['cuts'])} {sc['shots']}")
    check("no channel moves across a cut (object + data)", not CM.cut_violations(cam, A["cuts"]) and
          not CM.cut_violations(cam.data, A["cuts"]))
    paths = sorted({fc.data_path for fc in CM._all_fcurves(cam.data)})
    check("lens / f-stop / focus all baked on the data", paths == ["dof.aperture_fstop", "dof.focus_distance", "lens"], str(paths))
    check("markers: one per shot at its start", sorted((m.name, m.frame) for m in sc.timeline_markers) ==
          [("dolly", 1), ("orbit", 31), ("tilted", 61)])
    CM.bake_shots(cam, shots, n)
    check("re-bake is idempotent (markers not duplicated)", len(sc.timeline_markers) == 3)

    # --- the evaluated camera equals the planned arrays (location, rotation through Blender's Euler) ---
    worst_r, worst_l = 0.0, 0.0
    for f in range(1, n + 1):
        sc.frame_set(f)
        M = np.array(cam.matrix_world)
        worst_l = max(worst_l, float(np.abs(M[:3, 3] - A["loc"][f - 1]).max()))
        worst_r = max(worst_r, float(np.abs(M[:3, :3] - A["R"][f - 1]).max()))
    check("evaluated camera matches the plan at every frame", worst_l < 1e-4 and worst_r < 1e-5, f"loc {worst_l:.2e} rot {worst_r:.2e}")
    sc.frame_set(45)
    check("evaluated lens / focus", abs(cam.data.lens - 50) < 1e-5 and abs(cam.data.dof.focus_distance - 9) < 1e-5)

    # --- numpy rotation helpers vs mathutils ---
    dm = 0.0
    for i in range(0, 60):
        fwd = Vector(A["tgt"][i]) - Vector(A["loc"][i])
        Rm = np.array(fwd.to_track_quat("-Z", "Y").to_matrix())
        dm = max(dm, float(np.abs(Rm - A["R"][i]).max()))
    check("track_matrix == to_track_quat('-Z','Y') (up +Z, no roll)", dm < 1e-6, f"{dm:.2e}")
    look = np.array(X.look_quat((5, -5, 8), (0, 0, 0), 0.2).to_matrix())
    check("track_matrix roll == bpyx.look_quat roll", np.allclose(look, X.track_matrix((-5, 5, -8), roll=0.2), atol=1e-6))
    from mathutils import Euler, Matrix
    agree, prev = 0, None
    for i in range(30, 60):                                                 # the orbit
        Rm = A["R"][i]
        mm = Matrix(Rm.tolist()).to_euler("XYZ", Euler(prev)) if prev is not None else Matrix(Rm.tolist()).to_euler("XYZ")
        agree += np.allclose(np.array(mm), A["rot"][i], atol=1e-5)
        prev = tuple(A["rot"][i])
    check("euler_seq == to_euler('XYZ', prev) along a 2-turn orbit", agree == 30, f"{agree}/30")

    # --- on_screen vs bpy_extras.world_to_camera_view ---
    rng = np.random.default_rng(4)
    mism, tot = 0, 0
    for f in (5, 40, 70):
        sc.frame_set(f)
        for p in rng.uniform(-15, 15, (200, 3)):
            v = world_to_camera_view(sc, cam, Vector(p))
            ref = v.z > 0 and 0 <= v.x <= 1 and 0 <= v.y <= 1
            near_edge = v.z > 0 and (min(abs(v.x), abs(v.x - 1), abs(v.y), abs(v.y - 1)) < 1e-3)
            mine = bool(CM.on_screen({k: A[k][f - 1:f] for k in ("loc", "R", "lens")}, p[None])[0])
            tot += 1
            mism += (mine != ref) and not near_edge
    check("on_screen agrees with world_to_camera_view", mism == 0, f"{mism} of {tot} differ")

    # --- still cameras + DOF ---
    CM.still_cam("CAM_hall", (0, -20, 1.7), (0, 10, 4), 24, fstop=4, exposure_offset=0.3)
    s2 = CM.still_cam("CAM_hall", (0, -20, 1.7), (0, 10, 4), 24, fstop=4, exposure_offset=0.3)
    check("still_cam replaces its namesake", "CAM_hall.001" not in bpy.data.objects and s2.name == "CAM_hall")
    check("still_cam DOF focus = |target − cam|", s2.data.dof.use_dof and abs(s2.data.dof.focus_distance -
          math.dist((0, -20, 1.7), (0, 10, 4))) < 1e-4)
    check("still_cam custom prop for frames.py --cams", abs(s2["exposure_offset"] - 0.3) < 1e-9)
    plan = CM.still_cam("CAM_plan", (0, 0, 100), None, 50, ortho=60, rotz=math.radians(90))
    check("ortho plan camera", plan.data.type == "ORTHO" and plan.data.ortho_scale == 60)

    # --- hold_at_cuts: a curve baked WITHOUT cuts gets its smear removed ---
    box = X.boxes_mesh("Wall", [(-5, 5, 0, 5, 6, 4)], bottom=True)
    X.bake(box, "location", np.linspace(0, 9, n), index=2, thin=False)
    before = CM.cut_violations(box, [31, 61])
    CM.hold_at_cuts(box, [31, 61])
    after = CM.cut_violations(box, [31, 61])
    check("hold_at_cuts removes cut smear", len(before) == 2 and not after, f"{before} → {after}")
    X.bake(box, "location", np.zeros(n), index=2)
    bpy.context.view_layer.update()

    # --- view_hit: the wall 5 m in front of a still ---
    v = CM.still_cam("CAM_wall", (0, 0, 2), (0, 10, 2), 35)
    bpy.context.view_layer.update()
    name, dist = CM.view_hit(v)
    check("view_hit finds the wall at 5 m", name == "Wall" and abs(dist - 5.0) < 1e-3, f"{name} {dist}")
    name, dist = CM.view_hit(v, skip=lambda ob: ob.name == "Wall")
    check("view_hit skip passes through", name is None)

    print(f"[bpy_test_camera] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


main()
