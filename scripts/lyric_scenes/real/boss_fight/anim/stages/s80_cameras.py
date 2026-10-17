"""s80 — rough shot list → ONE baked camera (CAM_animatic). Every cut sits on an event frame (the move's anticipation
is inside the previous shot, the cut lands on the contact/next beat); the camera never reacts to the music — the only
kicks are decaying impulses on heavy IMPACTS. Targets track smoothed key points (no jitter). Keys before each cut are
CONSTANT so nothing interpolates across a cut. Black card from the RPG explosion to the end."""
import math

import bpy
import numpy as np
from mathutils import Euler, Vector

from anim import common as C
from anim import fx as FX
from anim.timeline import gsmooth
from lib import bpyx as X


def run(S):
    E, T, N = S.E, S.trk, S.N
    f = {k: e.f for k, e in E.items()}

    def sm(name, sig=6.0):
        a = T[name]
        return np.c_[gsmooth(a[:, 0], sig), gsmooth(a[:, 1], sig), gsmooth(a[:, 2], sig)]
    KC, KH, KR = sm("K.chest"), sm("K.head"), sm("K.root")
    DH, DC, DR, DB = sm("D.head"), sm("D.chest"), sm("D.root"), sm("D.belly")
    DH12 = sm("D.head", 12.0)
    kyaw = gsmooth(S.K.base_yaw + S.K.yoff, 8.0)

    def mid(a, b, w=0.5):
        return lambda i: a[i] * (1 - w) + b[i] * w

    def const(p):
        p = np.asarray(p, np.float64)
        return lambda i: p

    def off_from(track, off):
        off = np.asarray(off, np.float64)
        return lambda i: track[i] + off

    def ots(dist=2.6, side=0.9, up=0.35, tgt=DH):
        def loc(i):
            fw, rt = C.fwd(kyaw[i]), C.right(kyaw[i])
            return KH[i] - fw * dist + rt * side + np.array([0, 0, up])
        return loc, (lambda i: tgt[i])

    def rel(track, yawarr, fwd_, right_, up_):
        def loc(i):
            return track[i] + C.fwd(yawarr[i]) * fwd_ + C.right(yawarr[i]) * right_ + np.array([0, 0, up_])
        return loc

    def dolly(a, b, f0, f1):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return lambda i: a + (b - a) * C.smooth01((i + 1 - f0) / max(f1 - f0, 1))

    def orbit(center_fn, r, h, a0, a1, f0, f1):
        def loc(i):
            t = np.clip((i + 1 - f0) / max(f1 - f0, 1), 0, 1)
            a = math.radians(a0 + (a1 - a0) * t)
            c = center_fn(i)
            return np.array([c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2] + h])
        return loc

    dyaw = gsmooth(S.D.base_yaw + S.D.yoff, 8.0)
    dyaw6 = gsmooth(np.unwrap(np.radians(S.D.base_yaw + S.D.yoff)), 20.0)
    dyaw6 = np.degrees(dyaw6)
    DC6 = sm("D.chest", 10.0)
    KC3 = sm("K.chest", 2.5)
    EYE = sm("D.eyeL", 2.0)

    def eye_cam(i):            # the dragon's LEFT eye from slightly in front of its left side, close
        return EYE[i] - C.right(dyaw[i]) * 1.3 + C.fwd(dyaw[i]) * 0.55 + np.array([0, 0, 0.12])

    def chase(back, side, up):  # flight follow cam: behind / beside the dragon, heavily smoothed
        def loc(i):
            return DC6[i] - C.fwd(dyaw6[i]) * back + C.right(dyaw6[i]) * side + np.array([0, 0, up])
        return loc
    moon = np.array([-0.3538, 0.6655, 0.6572])
    shots = [
        (1, dolly((-190, -430, 95), (-55, -150, 38), 1, 209), const((0, 25, 22)), 35),
        (f["cam_colossi"], dolly((0, -78, 3.2), (0, -50, 4.5), f["cam_colossi"], f["dragon_asleep"]), const((0, -6, 14)), 24),
        (f["dragon_asleep"], dolly((10.5, 33.5, 10.6), (8.5, 35.5, 10.2), f["dragon_asleep"], f["knight_foot"]), lambda i: DH[i], 35),
        (f["knight_foot"], const((5.5, -35.0, 1.4)), lambda i: KC[i] + np.array([0, 0, 0.4]), 32),
        (f["climb"], lambda i: np.array([-15.5, KR[i][1] - 4.0, KR[i][2] + 2.5]), lambda i: KC[i], 28),
        (f["kit_sword"], rel(KC, kyaw, 1.4, 0.9, 0.0), lambda i: KC[i] + np.array([0, 0, -0.35]), 50),
        (f["kit_fire"], rel(KC, kyaw, 1.8, -0.3, -0.3), lambda i: KC[i], 40),
        (f["kit_rocks"], orbit(lambda i: KC[i], 3.2, 0.3, -70, -15, f["kit_rocks"], f["doors"]), lambda i: KC[i], 35),  # v2: stays south of the doors
        (f["doors"], dolly((4.0, -17.0, 10.8), (3.0, -14.0, 10.6), f["doors"], f["eye_open"]), const((0, -1, 11.0)), 24),
        (f["eye_open"], eye_cam, lambda i: EYE[i], 55),                     # v2: ~1 s on the EYE itself
        (f["rise_roar"], const((1.5, 9.0, 9.4)), lambda i: DH12[i], 24),
        (f["smolder"], rel(DH, dyaw, 5.2, 3.6, -0.6), lambda i: DH[i] + np.array([0, 0, -0.3]), 30),   # v2: wider
        (f["knight_in"], *ots(3.2, 1.0, 0.5), 32),
        (f["circling"], orbit(lambda i: np.array([-1.0, 22.0, 8.0]), 15.0, 3.5, -100, -40, f["circling"], f["draw"]),
         lambda i: (KC[i] + DH[i]) / 2, 28),
        (f["draw"], rel(KC, kyaw, 2.2, 1.2, 0.2), lambda i: KC[i], 45),
        (f["tail_test"], const((5.0, 7.5, 10.4)), const((-3.0, 18.5, 9.0)), 22),
        (f["combo"], rel(KR, kyaw, -1.5, 2.6, 0.7), lambda i: DB[i], 26),
        (f["talon"], const((7.0, 17.0, 11.0)), lambda i: (KC[i] + DH[i]) / 2, 30),
        (f["tail_p2"], const((-9.0, 12.5, 12.0)), const((7.0, 21.0, 10.0)), 24),
        (f["fireballs"], *ots(3.0, 0.8, 0.4, DC), 30),
        (f["ceiling"], lambda i: KC[i] + np.array([-3.6, -2.8, -0.6]), lambda i: KC[i] + np.array([0, 0, 2.0]), 22),
        (f["neck_slash"], const((-4.0, 15.0, 9.0)), lambda i: (KH[i] + DH[i]) / 2, 26),
        (f["bite_dodge"], const((-9.5, 16.0, 10.5)), lambda i: (KC[i] + DH[i]) / 2, 28),
        (f["tail_p3"], const((6.0, 13.5, 13.0)), const((-6.0, 27.0, 9.5)), 22),
        (f["bridge"], orbit(lambda i: (KC[i] + DC[i]) / 2, 13.0, 4.0, -30, 20, f["bridge"], f["spikes"]),
         lambda i: (KC[i] + DC[i]) / 2, 28),
        (f["spikes"], rel(KR, kyaw, -2.8, -0.8, 1.0), lambda i: DB[i], 24),
        (f["stumble"], const((6.0, 13.5, 10.6)), lambda i: DH12[i], 24),
        (f["rocks"], *ots(3.0, -0.8, 0.5), 30),
        (f["dfireballs"], rel(DH, dyaw, -2.5, 3.5, 2.5), lambda i: KC[i], 30),
        (f["bite_miss"], const((5.5, 19.0, 10.0)), lambda i: (KC[i] + DH[i]) / 2, 26),
        (f["slam"], const((5.5, 9.0, 13.5)), lambda i: KC3[i] * 0.7 + DC[i] * 0.3, 22),   # v2: follows the throw
        (f["potion"], rel(KC, kyaw, 1.6, 0.6, 0.1), lambda i: KC[i], 40),
        (f["gem_out"], rel(KC, kyaw, 2.4, -0.9, 0.3), lambda i: KC[i], 40),
        (f["gem_crush"], const(rel(KR, kyaw, 2.0, 0.9, 0.5)(f["gem_crush"] - 1)), lambda i: KC[i] + np.array([0, 0, 0.3]), 28),
        (f["launch"], const((-8.0, 14.0, 9.0)), lambda i: DC[i], 22),
        (f["roof"], const((-6.0, 18.5, 9.6)), const((0, 37, 44)), 18),
        (f["roof_rain"], lambda i: KC[i] + np.array([3.5, -3.0, -0.4]), lambda i: KC[i] + np.array([0, 0, 1.5]), 22),
        (f["moon"], chase(16.0, 5.0, 4.0), lambda i: DC6[i] + C.fwd(dyaw6[i]) * 8.0, 30),          # v2: FOLLOW CAM
        (f["run_out"], lambda i: np.array([12.0, KR[i][1] - 2.0, KR[i][2] + 3.0]), lambda i: KC[i], 28),
        (f["dive"], chase(14.0, 3.0, -3.5), lambda i: DH12[i] + C.fwd(dyaw6[i]) * 6.0, 26),          # v2: FOLLOW CAM
        (f["shield_breath"], *ots(4.0, 1.4, 0.6, DH12), 26),
        (f["big_fireball"], lambda i: KR[i] + np.array([6.0, -6.0, 2.0]), lambda i: DC[i], 24),
        (f["volley"], const((-14.0, -48.0, 3.0)), const((14.0, -22.0, 14.0)), 24),
        (f["dash_dodges"], lambda i: KR[i] + np.array([0.0, -12.0, 2.5]), lambda i: KC[i], 30),
        (f["lands"], const((-13.0, -46.0, 2.4)), lambda i: DC[i], 26),
        (f["standoff"], const((-30.0, -24.0, 9.0)), lambda i: (KC[i] + DC[i]) / 2, 28),
        (f["raise_hand"], rel(KR, kyaw, 3.4, 1.6, 0.4), lambda i: KC[i] + np.array([0, 0, 0.45]), 26),   # v2: full body (effort)
        (f["sword_forming"], rel(KR, kyaw, -3.0, 1.0, 1.2), lambda i: DR[i] + np.array([0, 0, 38.0]), 24),
        (f["inhale"], lambda i: DR[i] + np.array([7.0, 8.0, 1.0]), lambda i: DH[i], 26),
        (f["sword_formed"], const((26.0, -30.0, 4.0)), lambda i: DR[i] + np.array([0, 0, 22.0]), 22),
        (f["sword_falls"], const((22.0, -38.0, 6.0)), lambda i: DR[i] + np.array([0, 0, 4.0]), 26),
        (f["thrash"], lambda i: DR[i] + np.array([-9.0, 9.0, 3.0]), lambda i: DC[i], 28),
        (f["collapse"], const((14.0, -34.0, 8.0)), lambda i: DC[i], 26),
        (f["outro_walk"], lambda i: KR[i] + np.array([2.5, -6.0, 2.2]), lambda i: KC[i] + np.array([0, 6, 0]), 30),
        (7000, const((3.5, 40.0, 10.2)), lambda i: KC[i], 30),
        (f["eye_again"], rel(DH, dyaw, 7.0, 6.0, 1.2), lambda i: DH[i] * 0.6 + DC[i] * 0.4, 32),     # v2: wider twitch
        (f["shoulder_rpg"], const((2.8, 44.0, 10.6)), lambda i: KH[i], 35),
        (f["fire_rpg"], const((-3.5, 14.0, 10.8)), lambda i: DH[i], 26),   # v2: from the hall (v1 spot is behind the open door)
    ]
    shots.sort(key=lambda s: s[0])
    cuts = [s[0] for s in shots[1:]]
    ev_frames = {e.f for e in E.values()} | {7000}
    S.R.check(all(c in ev_frames for c in cuts), "cuts_on_event_frames", str([c for c in cuts if c not in ev_frames]))
    pos = np.zeros((N, 3))
    tgt = np.zeros((N, 3))
    lens = np.zeros(N)
    for k, (f0, loc_fn, tgt_fn, ln) in enumerate(shots):
        f1 = shots[k + 1][0] - 1 if k + 1 < len(shots) else N
        for fr in range(f0, f1 + 1):
            i = fr - 1
            pos[i] = loc_fn(i)
            tgt[i] = tgt_fn(i)
            lens[i] = ln
    # impact kicks (decaying, on heavy contacts only)
    heavy = [(c.f, 1.0) for c in S.contacts if c.kind in ("slam", "tail_pillar", "roof", "holy", "rocket", "spikes")]
    heavy += [(f["lands"] + 8, 0.8), (f["volley"], 0.6)]
    fr = np.arange(1, N + 1)
    kick = np.zeros((N, 3))
    for fc, s in heavy:
        t = fr - fc
        env = np.where(t >= 0, np.exp(-np.maximum(t, 0) / 3.0) * np.cos(np.maximum(t, 0) * 1.9), 0.0)
        kick[:, 2] += 0.07 * s * env
        kick[:, 0] += 0.04 * s * env
    pos = pos + kick
    cam_d = bpy.data.cameras.new("CAM_animatic")
    cam_d.sensor_width = 36.0
    cam_d.clip_start, cam_d.clip_end = 0.05, 6000.0
    cam = bpy.data.objects.new("CAM_animatic", cam_d)
    S.col.cams.objects.link(cam)
    S.sc.camera = cam
    rot = []
    prev = None
    for i in range(N):
        q = X.look_quat(Vector(pos[i]), Vector(tgt[i]))
        e = q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")
        if (i + 1) in cuts:
            e = q.to_euler("XYZ")
        rot.append(tuple(e))
        prev = e
    rot = np.array(rot)
    C.bake_obj(cam, "location", fr.astype(float), pos)
    C.bake_obj(cam, "rotation_euler", fr.astype(float), rot)
    C.bake_obj(cam_d, "lens", fr.astype(float), lens)
    from bpy_extras import anim_utils
    for idb in (cam, cam_d):
        cb = anim_utils.animdata_get_channelbag_for_assigned_slot(idb.animation_data)
        for fc in cb.fcurves:
            for kp in fc.keyframe_points:
                if int(round(kp.co[0])) + 1 in cuts:
                    kp.interpolation = "CONSTANT"
    # black card after the explosion
    card = FX.box("FX_blackout", (4.0, 4.0, 0.01), S.col.cams, FX.mat_emit("M_black", (0, 0, 0), 0.0))
    card.parent = cam
    card.location = (0, 0, -0.12)
    FX.window(card, getattr(S, "blackout_from", N - 60), N)
    S.cam, S.cam_pos = cam, pos
    import os
    np.savez_compressed(os.path.join(S.debug, "camera.npz"), pos=pos, tgt=tgt, lens=lens, cuts=np.array(cuts))
    S.cuts = cuts
    sh = S.sc.display.shading
    sh.light, sh.color_type = "STUDIO", "MATERIAL"
    sh.show_shadows, sh.show_cavity = True, True
    for ob, col_ in ((S.K, (0.62, 0.66, 0.75, 1)), (S.D, (0.45, 0.16, 0.12, 1))):
        for m in ob.meshes:
            for sl in m.material_slots:
                if sl.material:
                    sl.material.diffuse_color = col_
    S.R.add("camera.shots", len(shots))
    S.R.add("camera.cuts", cuts)
    # shot table for the notes
    S.shot_table = [dict(start=s[0], end=(shots[k + 1][0] - 1 if k + 1 < len(shots) else N), lens=s[3]) for k, s in enumerate(shots)]
