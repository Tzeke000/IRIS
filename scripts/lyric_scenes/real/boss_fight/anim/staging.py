"""THE BLOCKING PLAN v2 — where both characters are, what clip plays, every hand-keyed move, keyed to the event table
(choreo_v2.json via common.load_events; same frames as v1). Pure data + small generators; no bpy here.

v2 (Zeke's animatic-v1 notes): locomotion segments are 'LOCO' (time-warped to the root, s20/loco.py; the clip is
chosen from the travel direction: walk/jog/sprint, strafe L/R, back-step) and every planted foot is locked by IK (s55);
`air` windows switch the foot lock off (jumps, rolls, the throw, sitting). Tail strikes are WHOLE-BODY pivots on the
front legs (≤ 75° of yaw, hips lead, tail lags then whips through) — no more 220–250° spins; the knight JUMPS the first
one. Talon swipe = sideways sweep on three planted legs, body leaning away. Hands are IK'd where they must touch
something (sheathed hilt, RPG grips).

World: hall floor z=8 (nave x∈[-10.6,10.6] between pier faces, piers at y 6.25/18.75/31.25/43.75), gate y≈-1,
stairs y -24.5..-2.5, courtyard z=0 south of the stairs. Yaw: 0 = facing +Y (north), 90 = west, 180 = south.
Pillars: pillar_1 = W2 (crack), pillar_2 = E2 (shatter), pillar_3 = W3 (smash). cover_column_A = the fallen-column
drums CV_S1W_Drum0..2 (x -9.45..-4.41, y 14.7..15.65).
"""
import math

from anim.common import yaw_to

SPOT = dict(
    k_court=(0.0, -33.0), k_foot=(0.0, -27.6), k_top=(0.0, -2.9), k_door=(0.0, -2.0), k_in=(0.0, 4.5),
    k_mid=(0.0, 16.2), d_sleep=(0.0, 41.0), d_mid=(0.0, 30.5),
    ring_c=(-1.0, 22.0),
    k_circ_end=(-4.4, 16.4), k_jump=(-5.6, 17.0),
    k_combo=(1.2, 18.1), k_dash=(1.8, 16.3), k_parry=(-0.1, 16.9),
    k_safe2=(0.0, 13.9),
    k_chunk=(5.6, 18.4),
    k_slam=(-5.9, 17.8), cover_top=(-6.9, 15.2), k_behind=(-7.0, 13.2),
    k_vault=(-2.0, 23.5), d_vault=(0.0, 37.0),
    k_court2=(4.0, -33.0), k_court3=(-6.0, -36.5), d_land=(-2.0, -61.0),
    k_hoard=(0.0, 47.3), d_gate=(0.0, -5.5),
)
CHEST = 1.9
D_COMBO = (-0.6, 21.2)   # dragon root for the combo + parry (facing the knight, ~SSW)          # dragon chest (front-leg pivot) is 1.9 m in front of its root (character frame, measured)

# tail strikes: chest pivot, start yaw → contact yaw (solver fine-tunes ±), knight jump for the first
STRIKES = {   # lash +1 = clockwise body turn (yaw decreasing): the tail lags to +rz, then whips through to -rz
    "dodge_p1": dict(pivot=(1.46, 19.5), y0=-10.0, yc=-86.0, wind=6.0, follow=-12.0, lash=1, lift=-22.0),
    "tail_p2": dict(pivot=(-0.11, 19.36), y0=180.0, yc=87.0, wind=4.0, follow=-4.0, lash=1, whip=0.25),
    "tail_p3": dict(pivot=(-1.2, 26.8), y0=-65.0, yc=-117.0, wind=8.0, follow=-4.0, lash=1, whip=0.35, lift=-5.0),
}


def strike_root(key, yaw):
    s = STRIKES[key]
    a = math.radians(yaw)
    return (s["pivot"][0] + math.sin(a) * CHEST, s["pivot"][1] - math.cos(a) * CHEST)


def strike_settle_yaw(key):
    st = STRIKES[key]
    sgn = 1.0 if st["yc"] < st["y0"] else -1.0
    return st["yc"] + st["follow"] * 0.3 * sgn


def resolve_strike_end(k):
    """root key (frame, None, 'strike_end:<key>') → the root where the strike pivot leaves it"""
    if k[1] is None:
        key = k[2].split(":", 1)[1]
        return (k[0], strike_root(key, strike_settle_yaw(key)), "stop")
    return k


def arc(c, p0, p1, n=5):
    a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
    a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
    if a1 - a0 > math.pi:
        a1 -= 2 * math.pi
    if a0 - a1 > math.pi:
        a1 += 2 * math.pi
    r0, r1 = math.dist(p0, c), math.dist(p1, c)
    out = []
    for i in range(n):
        t = i / (n - 1)
        a = a0 + (a1 - a0) * t
        r = r0 + (r1 - r0) * t
        out.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    return out


# ================================================================================================ KNIGHT
def knight_plan(E):
    f = {k: e.f for k, e in E.items()}
    S = SPOT
    P = {}
    seg = []

    def add(start, clip, end=None, xf=6, scale=1.0, align=None, loop=False, upper=None):
        seg.append(dict(start=start, clip=clip, end=end, xf=xf, scale=scale, align=align, loop=loop, upper=upper))
    nz, sl, th = f["neck_slash"], f["slam"], f["thrown"]
    J0 = f["dodge_p1"] - 10                     # take-off
    J1 = f["dodge_p1"] + 3                      # touch-down
    SH0 = th + 44                               # sheathe (sitting behind cover, before the potion)
    DR2 = f["gem_crush"] + 44                   # draw again after the crush
    add(1, "Idle_Loop", 440, loop=True)
    add(440, "LOCO", xf=8)
    add(f["knight_foot"] + 6, "Idle_Loop", loop=True, xf=10)
    add(f["climb"], "LOCO", xf=10)
    add(978, "LOCO:Push_Loop", xf=8)
    add(1052, "LOCO", xf=8)
    add(1132, "Idle_Loop", loop=True, xf=10)
    add(1330, "LOCO", xf=10)
    add(1562, "Idle_Loop", loop=True)
    add(f["circling"], "LOCO", xf=10)                                   # circling: strafe facing the dragon
    add(f["draw"] - 16, "Idle_Loop", loop=True, xf=6)                   # right hand to the hilt (IK), draws
    add(f["draw"] + 4, "Sword_Idle", loop=True, xf=8)
    add(1772, "LOCO", xf=8)                                             # side-steps into the tail's path
    add(1806, "Sword_Idle", loop=True, xf=8)
    add(J0 - 14, "NinjaJump_Start", align=("takeoff", J0), xf=5)        # JUMPS the tail swipe
    add(J1 - 4, "NinjaJump_Land", align=("land", J1), xf=4)
    add(J1 + 18, "Sword_Idle", loop=True, xf=8)
    add(1900, "LOCO", xf=8)                                             # back to the combo spot
    add(f["combo"] - 30, "Sword_Idle", loop=True, xf=6)
    add(f["combo"] - 18, "Sword_Dash", align=(10, f["combo"] - 6), xf=4)
    add(f["combo"] - 4, "Sword_Regular_Combo", align=("hit0", f["combo"]), xf=4, scale=0.82)
    add(f["combo"] + 34, "Sword_Idle", loop=True, xf=8)
    add(f["parry"] - 14, "Sword_Block", align=(12, f["parry"]), xf=5)
    add(f["parry"] + 22, "Sword_Idle", loop=True, xf=8)
    add(f["tail_p2"] - 60, "LOCO", xf=8)                                # backs off out of the tail's reach
    add(f["tail_p2"] + 4, "Sword_Idle", loop=True, xf=8)
    add(f["fireballs"] - 16, "Spell_Simple_Enter", xf=4)
    add(f["fireballs"] - 2 - 8, "Spell_Simple_Shoot", xf=3)
    add(f["fireballs"] + 4 - 8, "Spell_Simple_Shoot", xf=3)
    add(f["fireballs"] + 12, "Spell_Simple_Exit", xf=4)
    add(f["fireballs"] + 26, "Sword_Idle", loop=True, xf=8)
    add(f["ceiling"] - 52, "LOCO", xf=6)                                # sprints under the arch
    add(f["ceiling"] - 22, "Spell_Simple_Enter", xf=4)
    add(f["ceiling"] - 10, "Spell_Simple_Idle_Loop", loop=True, xf=4)
    add(f["ceiling"] + 24, "LOCO", xf=8)                                # out from under the falling arch
    add(nz - 44, "Sword_Idle", loop=True, xf=6)
    add(nz - 30, "Sword_Attack", align=("hitmax", nz), xf=6)
    add(nz + 20, "Sword_Idle", loop=True, xf=8)
    add(f["bite_dodge"] - 14, "Roll", scale=0.7, xf=4)
    add(f["bite_dodge"] + 20, "Sword_Idle", loop=True, xf=8)
    add(f["tail_p3"] + 16, "LOCO", xf=8)                                # bridge: backs off
    add(f["bridge"] + 44, "Sword_Idle", loop=True, xf=8)
    add(f["spikes"] - 24, "Spell_Simple_Enter", xf=4)
    add(f["spikes"] - 14, "Spell_Simple_Shoot", xf=3)
    add(f["spikes"] + 14, "LOCO", xf=8)                                 # keeps the pressure: circles right
    add(f["stumble"] + 20, "Sword_Idle", loop=True, xf=8)
    add(f["rocks"] - 64, "Spell_Simple_Enter", xf=4)                    # pulls rocks out of the cracked floor
    add(f["rocks"] - 52, "Spell_Simple_Idle_Loop", loop=True, xf=4)
    add(f["rocks"] - 26, "OverhandThrow", align=(11, f["rocks"] - 8), xf=5)
    add(f["rocks"] + 14, "Sword_Idle", loop=True, xf=8)
    add(f["dfireballs"] - 22, "Spell_Simple_Enter", xf=4)
    add(f["dfireballs"] - 10, "Spell_Simple_Idle_Loop", loop=True, xf=4)
    add(f["dfireballs"] + 24, "LOCO", xf=8)
    add(f["bite_miss"] - 16, "Crouch_Idle_Loop", loop=True, xf=5)
    add(f["bite_miss"] + 14, "LOCO", xf=6)
    add(sl - 30, "Sword_Idle", loop=True, xf=6)
    add(sl - 2, "Hit_Knockback", scale=0.6, xf=2)                       # launched by the slam
    add(th + 4, "Roll", scale=0.9, xf=4)                                # tumbles on landing
    add(th + 34, "Crouch_Idle_Loop", loop=True, xf=8)                   # hurt, crouched behind the drums
    add(f["gem_out"] - 4, "Idle_Loop", loop=True, xf=20)                # stands up
    add(f["gem_out"] + 26, "Interact", xf=6)
    add(f["gem_out"] + 86, "Idle_Loop", loop=True, xf=8)
    add(DR2 + 10, "Sword_Idle", loop=True, xf=8)
    add(f["launch"], "LOCO", xf=8)
    add(f["launch"] + 110, "Sword_Idle", loop=True, xf=8)
    add(f["roof_rain"] - 22, "Spell_Simple_Enter", xf=4)
    add(f["roof_rain"] - 10, "Spell_Simple_Idle_Loop", loop=True, xf=4)
    add(f["roof_rain"] + 40, "Sword_Idle", loop=True, xf=8)
    add(4505, "LOCO", xf=8)                                             # runs out to the courtyard
    add(4852, "Sword_Idle", loop=True, xf=8)
    add(f["shield_breath"] - 16, "Spell_Simple_Enter", xf=4)
    add(f["shield_breath"] - 4, "Spell_Simple_Idle_Loop", loop=True, xf=4)
    add(f["shield_breath"] + 64, "Sword_Idle", loop=True, xf=8)
    add(f["big_fireball"] - 40, "OverhandThrow", align=(11, f["big_fireball"] - 28), xf=5)
    add(f["big_fireball"] + 4, "Sword_Idle", loop=True, xf=8)
    add(f["volley"] - 24, "Roll", scale=0.8, xf=4)
    add(f["volley"] + 12, "Sword_Idle", loop=True, xf=8)
    add(f["dash_dodges"] - 16, "LOCO", xf=6)
    add(f["dash_dodges"] + 24, "Sword_Idle", loop=True, xf=10)
    add(f["standoff"] + 30, "LOCO", xf=10)                              # slow standoff: circles
    add(f["raise_hand"] - 30, "Sword_Idle", loop=True, xf=10)           # braced stance for the summoning
    add(f["sword_falls"] + 70, "Sword_Idle", loop=True, xf=12)
    add(f["outro_walk"], "LOCO", xf=10)
    add(7000, "LOCO", xf=4)                                             # (after the cut) walks to the hoard
    add(7140, "PickUp_Table", align=(13, 7158), xf=6, scale=1.3)
    add(7176, "Idle_Loop", loop=True, xf=10)
    add(f["shoulder_rpg"], "LOCO", xf=10)                               # turns round (steps) with the RPG
    add(f["shoulder_rpg"] + 40, "Idle_Loop", loop=True, xf=10)
    P["seg"] = seg
    P["parts"] = [dict(start=f["potion"] - 6, clip="Consume", part="arms_head", xf=6, scale=1.2)]
    P["sword_keys"] = [(1, False), (f["draw"], True), (SH0, False), (DR2, True), (f["outro_walk"], False)]
    P["sword_drawn"] = lambda fr: any(a <= fr < b for a, b in ((f["draw"], SH0), (DR2, f["outro_walk"])))

    # ---- hand-keyed COMBINE moves {group: (rx, ry, rz)} in character axes
    hk = []
    # KIT 2/3: ONE fireball spinning in the RIGHT palm, then the rocks orbit that hand (palm up, in front)
    hk.append(dict(name="kit_palm", start=f["kit_fire"] - 10, keys=[
        (0, {}), (12, {"armR": (50, 0, 18), "forearmR": (70, 0, 10), "handR": (0, -80, 0)}),
        (150, {"armR": (52, 0, 18), "forearmR": (72, 0, 10), "handR": (0, -80, 0)}),
        (172, {"armR": (30, 0, 10), "forearmR": (30, 0, 0), "handR": (0, -30, 0)}), (180, {})]))
    hk.append(dict(name="draw_follow", start=f["draw"] - 2, keys=[
        (0, {}), (6, {"armR": (25, 0, 10), "forearmR": (25, 0, 0)}), (16, {})]))
    hk.append(dict(name="hurt_slump", start=th + 26, keys=[
        (0, {}), (14, {"spine": (-14, 0, 0), "neck": (-18, 0, 0)}),
        (f["potion"] - th - 32, {"spine": (-12, 0, 0), "neck": (-14, 0, 0)}),
        (f["potion"] - th, {})]))
    # gem: held up in the empty right hand, fingers CRUSH it
    hk.append(dict(name="gem_crush", start=f["gem_crush"] - 34, keys=[
        (0, {}), (14, {"armR": (75, 0, 10), "forearmR": (40, 0, 0), "fingersR": (0, 0, 0), "spine": (4, 0, 0)}),
        (31, {"armR": (82, 0, 10), "forearmR": (48, 0, 0), "fingersR": (0, 0, 0)}),
        (34, {"armR": (85, 0, 10), "forearmR": (52, 0, 0), "fingersR": (-75, 0, 0), "spine": (-4, 0, 0)}),
        (60, {"armR": (80, 0, 10), "forearmR": (45, 0, 0), "fingersR": (-75, 0, 0)}), (78, {})]))
    # holy sword summoning: sword raised to the sky with visible EFFORT (tremor grows, spine braced back, head up,
    # free arm clenched out for balance); knees bend through the root drop below (feet stay planted by IK)
    rh0 = f["raise_hand"] - 10
    hold_end = f["sword_falls"] - rh0
    keys = [(0, {}), (24, {"armR": (158, 0, 4), "forearmR": (10, 0, 0), "spine": (-6, 0, 0), "neck": (22, 0, 0),
                           "armL": (25, 0, -35), "forearmL": (40, 0, 0), "fingersL": (-70, 0, 0)})]
    import random
    rnd = random.Random(7)
    k = 30
    while k < hold_end - 6:
        prog = k / hold_end
        amp = 1.0 + 5.0 * prog
        keys.append((k, {"armR": (160 + rnd.uniform(-amp, amp), rnd.uniform(-amp, amp) * 0.6, 4 + rnd.uniform(-amp, amp)),
                         "forearmR": (10 + rnd.uniform(-amp, amp) * 0.7, 0, 0),
                         "spine": (-6 - 6 * prog + rnd.uniform(-1, 1) * prog, rnd.uniform(-1.5, 1.5) * prog, 0),
                         "neck": (24 + rnd.uniform(-2, 2) * prog, 0, rnd.uniform(-2, 2) * prog),
                         "armL": (25 + 8 * prog, 0, -35 - 10 * prog), "forearmL": (40 + 15 * prog, 0, 0),
                         "fingersL": (-70, 0, 0)}, "flow"))
        k += 3 if prog > 0.5 else 4
    keys += [(hold_end - 2, {"armR": (165, 0, 4), "forearmR": (5, 0, 0), "spine": (-12, 0, 0), "neck": (30, 0, 0),
                             "armL": (35, 0, -45), "forearmL": (55, 0, 0), "fingersL": (-70, 0, 0)}),
             (hold_end + 16, {"armR": (40, 0, 0), "neck": (5, 0, 0)}), (hold_end + 60, {})]
    hk.append(dict(name="raise_hand", start=rh0, keys=keys))
    # RPG fire: recoil through the shoulders/spine (the root shove + bent knees come from the root keys)
    fR = f["fire_rpg"] - 16                    # rocket LAUNCH (it hits the dragon on the fire_rpg impact frame)
    hk.append(dict(name="rpg_recoil", start=fR - 2, keys=[
        (0, {}), (2, {}), (5, {"spine": (-12, 0, -4), "neck": (-8, 0, 0)}), (12, {"spine": (-5, 0, -2), "neck": (-3, 0, 0)}),
        (24, {})]))
    def blade_out(name, a, b):
        hk.append(dict(name=f"blade_out_{name}", start=a, keys=[
            (0, {}), (5, {"armR": (10, 0, -45), "forearmR": (5, 0, -20)}),
            (b - a - 5, {"armR": (10, 0, -45), "forearmR": (5, 0, -20)}), (b - a, {})]))
    for nm, a, b in (("jump", J0 - 8, J1 + 8), ("dash", f["combo"] - 18, f["combo"] - 6),
                     ("fireballs", f["fireballs"] - 18, f["fireballs"] + 14), ("bite", f["bite_dodge"] - 16, f["bite_dodge"] + 16),
                     ("spikes", f["spikes"] - 24, f["spikes"] + 18), ("rocks", f["rocks"] + 2, f["rocks"] + 16),
                     ("thrown", sl - 4, SH0 - 12), ("shield", f["shield_breath"] - 18, f["shield_breath"] + 62),
                     ("bigfb", f["big_fireball"] - 42, f["big_fireball"] + 6), ("volley", f["volley"] - 26, f["volley"] + 12)):
        blade_out(nm, a, b)
    P["hk"] = hk

    # ---- hand IK targets (s55)
    P["hands"] = [dict(chain="hL", kind="hilt_rest", lo=f["kit_sword"] - 8, hi=f["kit_fire"] - 14, rin=10, rout=12),
                  dict(chain="hR", kind="hilt_grip", lo=f["draw"] - 14, hi=f["draw"], rin=10, rout=1),
                  dict(chain="hR", kind="hilt_grip", lo=SH0 - 10, hi=SH0 + 2, rin=10, rout=8),
                  dict(chain="hR", kind="hilt_grip", lo=DR2 - 12, hi=DR2, rin=10, rout=1),
                  dict(chain="hR", kind="rpg", lo=7148, hi=7628, rin=10, rout=1),
                  dict(chain="hL", kind="rpg", lo=7150, hi=7628, rin=10, rout=1)]
    P["rpg"] = dict(pickup=7158, lifted=7178, shoulder_frames=26)

    # ---- root path: (frame, (x, y), opt)
    root = [(1, S["k_court"], "stop"), (440, S["k_court"], "stop"), (f["knight_foot"] + 4, S["k_foot"], "stop"),
            (f["climb"], S["k_foot"], "stop"), (968, S["k_top"]), (982, S["k_door"], "stop"), (992, S["k_door"], "stop"),
            (1052, (0.0, 0.4)), (1132, S["k_in"], "stop"), (1330, S["k_in"], "stop"), (1562, S["k_mid"], "stop"),
            (f["circling"], S["k_mid"], "stop")]
    for i, p in enumerate(arc(S["ring_c"], S["k_mid"], S["k_circ_end"], 4)[1:]):
        root.append((f["circling"] + (f["draw"] - 18 - f["circling"]) * (i + 1) // 3, p, "stop" if i == 2 else ""))
    root += [(1772, S["k_circ_end"], "stop"), (1806, S["k_jump"], "stop"), (J0 - 2, S["k_jump"], "stop"),
             (J1, (S["k_jump"][0] - 0.25, S["k_jump"][1] - 0.15), "stop"), (1900, (S["k_jump"][0] - 0.25, S["k_jump"][1] - 0.15), "stop"),
             (f["combo"] - 30, S["k_dash"], "stop"), (f["combo"] - 18, S["k_dash"], "stop"),
             (f["combo"] - 2, S["k_combo"]), (f["combo"] + 30, (1.3, 17.9), "stop"),
             (f["combo"] + 52, (0.6, 17.3), "stop"), (f["parry"] - 14, S["k_parry"], "stop"), (f["parry"] + 3, S["k_parry"], "stop"),
             (f["parry"] + 12, (-0.2, 16.4), "stop"), (f["tail_p2"] - 60, (-0.2, 16.2), "stop"),
             (f["tail_p2"] - 6, S["k_safe2"], "stop"), (f["tail_p2"] + 30, (-0.5, 15.4), "stop"),
             (f["ceiling"] - 52, (-0.5, 15.4), "stop"), (f["ceiling"] - 22, S["k_chunk"], "stop"),
             (f["ceiling"] + 24, S["k_chunk"], "stop"), (nz - 46, (0.8, 19.6), "stop"),
             (nz - 28, (0.6, 20.0), "stop"), (nz + 4, (0.4, 21.0), "stop"),
             (nz + 30, (-0.6, 20.3), "stop"), (f["bite_dodge"] - 12, (-0.6, 20.3), "stop"),
             (f["bite_dodge"] + 8, (3.6, 18.4), "stop"), (f["tail_p3"] + 16, (3.2, 18.2), "stop"),
             (f["bridge"] + 40, (1.6, 16.8), "stop"), (f["spikes"] + 14, (1.6, 16.8), "stop"),
             (f["stumble"] - 30, (2.0, 18.0)), (f["stumble"] + 20, (1.8, 19.6), "stop"),
             (f["dfireballs"] + 24, (1.8, 19.6), "stop"),
             (f["bite_miss"] - 16, (-2.6, 17.1), "stop"), (f["bite_miss"] + 14, (-2.6, 17.1), "stop"),
             (sl - 30, S["k_slam"], "stop"), (sl, S["k_slam"], "stop")]
    # THROWN (continuous ballistic arc over the drums, clipping the drum top at `thrown`), lands, tumbles, sits
    root += [(sl + 2, (S["k_slam"][0] - 0.25, S["k_slam"][1] - 0.8)), (th, S["cover_top"], "lin"),
             (th + 4, (-7.0, 14.0)), (th + 12, (-7.05, 13.0)), (th + 34, S["k_behind"], "stop"),
             (f["gem_out"] + 20, S["k_behind"], "stop"), (f["gem_out"] + 40, (-6.6, 12.4), "stop"),
             (f["launch"] - 40, (-6.6, 12.4), "stop"), (f["launch"] + 30, (-3.4, 13.6)), (f["launch"] + 60, (-3.2, 17.0)),
             (f["launch"] + 110, S["k_vault"], "stop"), (4505, S["k_vault"], "stop"),
             (4560, (0.0, 12.0)), (4600, (0.0, 1.0)), (4640, (0.0, -10.0)), (4700, (0.0, -24.8)),
             (4790, (2.5, -30.5)), (4852, S["k_court2"], "stop"), (f["big_fireball"] + 4, S["k_court2"], "stop"),
             (f["volley"] - 24, S["k_court2"], "stop"), (f["volley"] + 8, (1.8, -34.0), "stop"),
             (f["dash_dodges"] - 16, (1.8, -34.0), "stop"), (f["dash_dodges"] + 24, S["k_court3"], "stop"),
             (f["standoff"] + 30, S["k_court3"], "stop"), (f["raise_hand"] - 34, (-3.0, -38.0), "stop"),
             (f["outro_walk"], (-3.0, -38.0), "stop"),
             (6990, (-0.5, -30.0), "stop"),
             # CUT at 7000: an ellipsis (hidden by the cut) into the hall; the walk to the hoard avoids the rubble (s75)
             (6999, (-0.5, -30.0), "stop lin"), (7000, (0.0, 37.8), "stop"), (7140, S["k_hoard"], "stop"),
             (fR - 1, S["k_hoard"], "stop"), (fR + 3, (S["k_hoard"][0], S["k_hoard"][1] + 0.12)),
             (fR + 18, (S["k_hoard"][0], S["k_hoard"][1] + 0.05), "stop")]
    P["root"] = root
    P["teleports"] = [7000]
    yaw = [(1, 0), (f["knight_foot"] + 4, 0), (968, 0), (1330, 0), (1562, 0)]
    for fr in range(f["circling"], f["draw"] + 1, 20):
        yaw.append((fr, None))
    yaw += [(1772, None), (1806, None), (J0 - 2, None), (J1 + 6, None), (1900, None), (f["combo"] - 30, None),
            (f["combo"] - 18, None), (f["combo"], None), (f["parry"], None), (f["tail_p2"] - 60, None),
            (f["tail_p2"] - 6, None), (f["tail_p2"] + 30, None),
            (f["fireballs"], None), (f["ceiling"] - 52, None), ("motion", f["ceiling"] - 46, f["ceiling"] - 22),
            (f["ceiling"] - 14, None), (f["ceiling"] + 24, None), ("motion", f["ceiling"] + 30, nz - 46),
            (nz - 38, None), (nz, None),
            (f["bite_dodge"] - 16, None), (f["bite_dodge"] - 12, -80.0), (f["bite_dodge"] + 6, -80.0),
            (f["bite_dodge"] + 22, None), (f["tail_p3"], None), (f["spikes"], None),
            (f["stumble"], None), (f["rocks"], None), (f["dfireballs"], None), (f["bite_miss"], None),
            (sl - 30, None), (sl, None), (th, None),
            (th + 10, ("rel", 0.0)), (th + 34, 180.0), (f["gem_out"] + 20, 180.0), (f["gem_out"] + 60, 0.0),
            (f["launch"] - 4, None), ("motion", f["launch"] + 8, f["launch"] + 100), (f["roof_rain"] - 24, None), (f["roof_rain"], None),
            (4500, None), ("motion", 4512, 4840), (4852, None),
            (f["big_fireball"], None), (f["volley"] - 26, None), (f["volley"] - 20, 120.0), (f["volley"] + 8, 120.0),
            (f["volley"] + 20, None), ("motion", f["dash_dodges"] - 10, f["dash_dodges"] + 20),
            (f["dash_dodges"] + 30, None), (f["standoff"], None), (f["raise_hand"] - 30, None), (f["outro_walk"] - 4, None),
            ("motion", f["outro_walk"] + 12, 6990), (6999, ("rel", 0.0)), (7000, 0.0), (f["shoulder_rpg"], 0.0),
            (f["shoulder_rpg"] + 34, 180.0)]
    P["yaw"] = yaw
    # heights above ground: jump over the tail, the neck-slash jump, the slam launch arc, braced crouch, recoil
    P["dz"] = [(1, 0.0), (J0 - 2, 0.0, "stop"), (J0, 0.08), (f["dodge_p1"] - 4, 1.7), (f["dodge_p1"], 1.35),
               (J1, 0.0, "stop"),
               (nz - 14, 0.0, "stop"), (nz - 2, 1.25), (nz + 3, 1.3), (nz + 14, 0.0, "stop"),
               (sl + 1, 0.0, "stop"), (sl + 3, 0.7), (th, 1.15), (th + 4, 0.55), (th + 8, 0.12), (th + 12, 0.0, "stop"),
               (f["raise_hand"] - 30, 0.0, "stop"), (f["raise_hand"] + 20, -0.16), (f["sword_falls"] - 4, -0.2),
               (f["sword_falls"] + 30, 0.0, "stop"),
               (fR - 1, 0.0, "stop"), (fR + 3, -0.07), (fR + 20, 0.0, "stop")]
    P["air"] = [(J0 - 1, J1 + 1), (nz - 15, nz + 15), (f["bite_dodge"] - 14, f["bite_dodge"] + 10),
                (sl, th + 30), (f["volley"] - 24, f["volley"] + 10), (6997, 7003)]
    P["dragon_face"] = dragon_spots(E)
    return P


# ================================================================================================ DRAGON
def dragon_spots(E):
    f = {k: e.f for k, e in E.items()}
    S = SPOT
    s1, s2, s3 = (strike_root(k, STRIKES[k]["y0"]) for k in ("dodge_p1", "tail_p2", "tail_p3"))
    pts = [(1, S["d_sleep"]), (1300, S["d_sleep"]), (1440, S["d_mid"]), (f["circling"], S["d_mid"])]
    pts += [(f["draw"] - 10, (3.6, 24.0)), (f["tail_test"] - 30, s1), (f["combo"] - 20, (-3.4, 24.6)),
            (f["parry"] + 10, (-3.2, 25.4)), (f["tail_p2"] - 28, s2), (f["tail_p2"] + 60, s2),
            (f["neck_slash"], (2.2, 24.8)), (f["bite_dodge"] - 20, (0.0, 24.6)), (f["bite_dodge"], (0.4, 22.6)),
            (f["tail_p3"] - 30, s3), (f["tail_p3"] + 20, s3), (f["bridge"] + 40, (-1.2, 27.4)),
            (f["spikes"], (-1.2, 27.4)), (f["stumble"], (-1.0, 28.2)), (f["bite_miss"] - 30, (-1.6, 26.0)),
            (f["bite_miss"], (-2.2, 23.8)), (f["slam"] - 40, (-4.6, 22.6)), (f["slam"], (-5.9, 20.1)),
            (f["slam"] + 60, (-5.0, 21.0)), (f["gem_out"], (-1.5, 23.5)), (f["launch"] - 60, S["d_vault"]),
            (f["launch"], S["d_vault"])]
    return pts


def dragon_plan(E):
    f = {k: e.f for k, e in E.items()}
    S = SPOT
    P = {}
    seg = []

    def add(start, clip, end=None, xf=10, scale=1.0, loop=False, align=None):
        seg.append(dict(start=start, clip=clip, end=end, xf=xf, scale=scale, loop=loop, align=align))
    add(1, "Idle Sit", f["rise_roar"] + 6, loop=True, scale=3.0)
    add(f["rise_roar"], "Idle Stand", 1300, loop=True, xf=24)
    add(1300, "LOCO:Walk", xf=12)
    add(1440, "Idle Stand", loop=True, xf=12)
    add(f["circling"], "LOCO:Walk", xf=12)                              # circling: walks round, head on the knight
    add(f["tail_test"] - 64, "Idle Stand", loop=True, xf=12)            # turns away, stepping (s55)
    add(f["tail_test"] - 34, "Idle Stand", loop=True, xf=10)
    add(f["dodge_p1"] + 30, "LOCO:Walk", xf=12)                         # turns round (steps) to face the knight
    add(f["combo"] - 16, "Idle Stand", loop=True, xf=12)
    add(f["tail_p2"] + 40, "Idle Stand", loop=True, xf=12)
    add(f["neck_slash"] - 40, "Idle Stand", loop=True, xf=10)
    add(f["bite_dodge"] - 22, "LOCO:Run", xf=6)
    add(f["bite_dodge"] + 10, "Idle Stand", loop=True, xf=10)
    add(f["bite_dodge"] + 40, "Idle Stand", loop=True, xf=10)        # side-steps into the strike spot (stepped by s55)
    add(f["tail_p3"] + 30, "LOCO:Walk", xf=10)
    add(f["bridge"] + 44, "Idle Stand", loop=True, xf=10)
    add(f["stumble"] + 36, "LOCO:Walk", xf=10)                          # recovers and ADVANCES
    add(f["rocks"] - 10, "Idle Stand", loop=True, xf=10)
    add(f["bite_miss"] - 30, "LOCO:Walk", xf=10)
    add(f["slam"] - 30, "Idle Stand", loop=True, xf=8)
    add(f["slam"] + 70, "LOCO:Walk", xf=12)
    add(f["gem_out"], "Idle Stand", loop=True, xf=12)
    add(f["launch"] - 100, "LOCO:Walk", xf=12)
    add(f["launch"] - 40, "Idle Stand", loop=True, xf=10)
    add(f["launch"] - 12, "Fly", loop=True, scale=1.15, xf=14)
    add(f["lands"], "Idle Stand", loop=True, xf=8)
    add(f["eye_again"] + 20, "Fly", loop=True, scale=0.8, xf=10)
    add(7366, "Idle Stand", loop=True, xf=10)           # leaps the last stretch wings FOLDED (gate piers / stair cheeks)
    P["seg"] = seg

    P["donors"] = [
        dict(clip="roar", start=f["rise_roar"] - 10, n=150, gain=1.0, xf=12),
        dict(clip="bite", start=f["bite_dodge"] - 26, n=70, gain=0.75, xf=6),
        dict(clip="roar", start=f["stumble"] - 12, n=140, gain=0.9, xf=12),
        dict(clip="bite", start=f["bite_miss"] - 26, n=70, gain=0.75, xf=6),
        dict(clip="roar", start=f["lands"] + 30, n=130, gain=0.8, xf=12),
        dict(clip="roar", start=f["eye_again"] + 40, n=110, gain=0.7, xf=12),
    ]
    P["tail_straight"] = [(f[k] - 40, f[k] + 30) for k in STRIKES]

    hk = []
    # asleep on the hoard: neck lowered FORWARD so the head rests in front of the forepaws (v1 folded it into the
    # chest: 'dragon head inside itself'), wings folded along the back (no crossing)
    sleep = {"neck_sit": (-50, 0, 0), "head_sit": (18, 0, 0), "lids": (1, 0, 0), "wingL": (0, -12, 0), "wingR": (0, 12, 0)}
    hk.append(dict(name="sleep", start=1, keys=[
        (0, sleep), (f["eye_open"] - 1, sleep),
        (f["eye_open"] + 3, dict(sleep, lids=(0, 0, 0))),
        (f["rise_roar"] + 4, {"neck_sit": (-18, 0, 0), "head_sit": (-4, 0, 0)}),
        (f["rise_roar"] + 30, {})]))
    hk.append(dict(name="wings_half", start=f["rise_roar"] - 4, keys=[
        (0, {}), (20, {"wingL": (0, -30, 0), "wingR": (0, 30, 0), "wingfoldL": (0, 0, -22), "wingfoldR": (0, 0, 22)}),
        (70, {"wingL": (0, -26, 0), "wingR": (0, 26, 0), "wingfoldL": (0, 0, -18), "wingfoldR": (0, 0, 18)}),
        (120, {})]))
    hk.append(dict(name="smolder", start=f["smolder"] - 6, keys=[
        (0, {}), (8, {"jaw": (-10, 0, 0), "head": (6, 0, 0)}), (110, {"jaw": (-8, 0, 0), "head": (4, 0, 0)}), (130, {})]))
    # TAIL STRIKES — whole body: wind-up (hips swing the other way), the hips LEAD the yaw (spine rz), the tail LAGS
    # (+rz toward the trailing side) then WHIPS THROUGH past the contact (follow-through) and settles. The body yaw
    # itself is a pivot about the front legs (root keys from STRIKES, s30).
    for key, st in STRIKES.items():
        c = f[key]
        L = st["lash"]
        W = st.get("whip", 1.0)
        U = st.get("lift", 0.0)
        hk.append(dict(name=f"tail_{key}", start=c - 34, keys=[
            (0, {}),
            (12, {"spine": (0, 0, -6 * L), "tail_base": (-6 + U, 0, -10 * L), "tail": (-4 + U, 0, -6 * L)}),  # wind-up
            (24, {"spine": (0, 0, 10 * L), "tail_base": (-8 + U, 0, 18 * L), "tail": (-6 + U, 0, 28 * L)}),   # hips lead, tail lags
            (34, {"spine": (0, 0, 6 * L), "tail_base": (-6 + U, 0, -6 * L * W), "tail": (-4 + U, 0, -22 * L * W)}),   # whip (contact)
            (42, {"spine": (0, 0, -2 * L), "tail_base": (-4, 0, -14 * L * W), "tail": (-2, 0, -34 * L * W)}, "flow"),   # follow-through
            (58, {"tail_base": (0, 0, -4 * L * W), "tail": (0, 0, -8 * L * W)}),
            (76, {})]))
    # TALON SWIPE — sideways: the right foreleg lifts OUT to the side (3 legs planted, IK), the body leans away (left),
    # the talons sweep across horizontally into the parry, bounce back
    hk.append(dict(name="tail_up_s1", start=f["tail_test"] - 56, keys=[        # tail carried high over the drums
        (0, {}), (10, {"tail_base": (-10, 0, 0)}), (24, {"tail_base": (-10, 0, 0)}), (34, {})]))
    hk.append(dict(name="talon_swipe", start=f["parry"] - 26, keys=[
        (0, {}),
        (12, {"armR": (60, -30, -50), "forearmR": (-25, 0, 0), "spine": (0, -9, 8), "neck": (0, -4, 14), "head": (0, 0, 8)}),
        (20, {"armR": (66, -34, -58), "forearmR": (-30, 0, 0), "spine": (0, -11, 10), "neck": (0, -5, 16)}),
        (26, {"armR": (62, -10, 22), "forearmR": (-8, 0, 0), "spine": (0, -6, -6), "neck": (0, -2, -6)}),             # contact
        (31, {"armR": (58, -16, -4), "forearmR": (-14, 0, 0), "spine": (0, -5, 0)}),                                 # bounce
        (48, {"armR": (20, -6, 0), "spine": (0, -2, 0)}), (60, {})]))
    hk.append(dict(name="neck_low", start=f["neck_slash"] - 40, keys=[
        (0, {}), (24, {"neck": (-22, 0, 0), "head": (8, 0, 0), "spine": (-6, 0, 0)}),
        (40, {"neck": (-24, 0, 0), "head": (6, 0, 0), "spine": (-6, 0, 0)}), (70, {})]))
    hk.append(dict(name="stagger", start=f["spikes"] - 2, keys=[
        (0, {}), (6, {"spine": (12, 6, 0), "neck": (18, 0, 6), "head": (6, 0, 0)}),
        (20, {"spine": (6, -4, 0), "neck": (8, 0, -4)}), (40, {"spine": (2, 0, 0)}), (70, {})]))
    hk.append(dict(name="spit1", start=f["dfireballs"] - 16, keys=[
        (0, {}), (10, {"neck": (12, 0, 0), "head": (10, 0, 0), "jaw": (-5, 0, 0)}),
        (16, {"neck": (-8, 0, 0), "head": (-6, 0, 0), "jaw": (-28, 0, 0)}), (22, {"neck": (6, 0, 0), "jaw": (-6, 0, 0)}),
        (28, {"neck": (-8, 0, 0), "head": (-6, 0, 0), "jaw": (-28, 0, 0)}), (44, {})]))
    hk.append(dict(name="slam", start=f["slam"] - 44, keys=[
        (0, {}), (26, {"spine": (28, 0, 0), "neck": (10, 0, 0), "armL": (60, 0, 0), "armR": (60, 0, 0),
                       "wingL": (0, -30, 0), "wingR": (0, 30, 0)}),
        (40, {"spine": (-6, 0, 0), "neck": (-10, 0, 0), "armL": (-20, 0, 0), "armR": (-20, 0, 0)}),
        (44, {"spine": (-10, 0, 0), "neck": (-14, 0, 0), "armL": (-25, 0, 0), "armR": (-25, 0, 0)}),
        (70, {"spine": (-4, 0, 0), "neck": (-4, 0, 0)}), (100, {})]))
    hk.append(dict(name="launch_crouch", start=f["launch"] - 40, keys=[
        (0, {}), (30, {"spine": (-6, 0, 0), "legL": (-15, 0, 0), "legR": (-15, 0, 0), "wingL": (0, -40, 0),
                       "wingR": (0, 40, 0)}), (40, {})]))
    # roof burst: head + neck thrust up, the wings in ONE big flap that sweeps them UP and BACK (v1 folded them
    # forward over the head → they crossed)
    hk.append(dict(name="roof_flap", start=f["roof"] - 50, keys=[
        (0, {}), (18, {"wingL": (0, -32, 0), "wingR": (0, 32, 0), "neck": (20, 0, 0), "head": (15, 0, 0), "spine": (22, 0, 0)}),
        (34, {"wingL": (-10, -40, 0), "wingR": (-10, 40, 0), "neck": (28, 0, 0), "head": (22, 0, 0), "spine": (32, 0, 0)}),
        (46, {"wingL": (10, 26, 0), "wingR": (10, -26, 0), "neck": (30, 0, 0), "head": (25, 0, 0), "spine": (35, 0, 0)}),
        (54, {"wingL": (4, 8, 0), "wingR": (4, -8, 0), "neck": (18, 0, 0), "spine": (20, 0, 0)}),
        (64, {"wingL": (0, -24, 0), "wingR": (0, 24, 0), "neck": (10, 0, 0)}), (84, {})]))
    hk.append(dict(name="breath_dive", start=f["dive"] - 12, keys=[
        (0, {}), (12, {"jaw": (-30, 0, 0), "neck": (-7, 0, 0), "head": (-5, 0, 0)}),
        (70, {"jaw": (-30, 0, 0), "neck": (-6, 0, 0), "head": (-5, 0, 0)}), (84, {})]))
    hk.append(dict(name="hit_spin", start=f["big_fireball"] - 2, keys=[
        (0, {}), (6, {"spine": (0, 35, 0), "neck": (-15, 20, 0), "tail": (0, 0, 40)}), (24, {"spine": (0, -15, 0)}),
        (44, {})]))
    hk.append(dict(name="volley", start=f["volley"] - 24, keys=[
        (0, {}), (12, {"neck": (8, 0, 0), "jaw": (-8, 0, 0)}), (18, {"neck": (-10, 0, 0), "jaw": (-30, 0, 0)}),
        (22, {"jaw": (-10, 0, 0)}), (26, {"neck": (-10, 0, 0), "jaw": (-30, 0, 0)}), (30, {"jaw": (-10, 0, 0)}),
        (34, {"neck": (-10, 0, 0), "jaw": (-30, 0, 0)}), (50, {})]))
    hk.append(dict(name="land_squash", start=f["lands"] - 8, keys=[
        (0, {}), (8, {"spine": (-8, 0, 0), "neck": (-12, 0, 0), "legL": (-18, 0, 0), "legR": (-18, 0, 0),
                      "wingL": (0, -20, 0), "wingR": (0, 20, 0)}), (30, {})]))
    # inhale → pinned → thrash → collapse: wing flails stay OPEN-SIDED (no L/R crossing over the back)
    hk.append(dict(name="inhale", start=f["inhale"] - 20, keys=[
        (0, {}), (40, {"spine": (14, 0, 0), "neck": (22, 0, 0), "head": (10, 0, 0), "jaw": (-12, 0, 0),
                       "wingL": (0, -25, 0), "wingR": (0, 25, 0)}),
        (f["sword_falls"] - f["inhale"] - 12, {"spine": (16, 0, 0), "neck": (25, 0, 0), "head": (12, 0, 0),
                                               "jaw": (-20, 0, 0), "wingL": (0, -30, 0), "wingR": (0, 30, 0)}),
        (f["sword_falls"] - f["inhale"] - 2, {"spine": (12, 0, 0), "neck": (20, 0, 0), "head": (10, 0, 0),
                                              "jaw": (-20, 0, 0), "wingL": (0, -34, 4), "wingR": (0, 34, -4)}),
        (f["sword_falls"] - f["inhale"] + 22, {"spine": (-18, 0, 0), "neck": (-25, 0, 0), "head": (-10, 0, 0)}),
        (f["thrash"] - f["inhale"] + 20, {"spine": (-14, 0, 0), "neck": (-10, 0, 0), "wingL": (0, -32, 0),
                                          "wingR": (0, 32, 0)}),
        (f["thrash"] - f["inhale"] + 40, {"spine": (-16, 0, 0), "neck": (-30, 0, 0), "wingL": (0, 12, 0),
                                          "wingR": (0, -12, 0), "tail": (0, 0, 40)}),
        (f["thrash"] - f["inhale"] + 60, {"spine": (-14, 0, 0), "neck": (5, 0, 0), "wingL": (0, -32, 0),
                                          "wingR": (0, 32, 0), "tail": (0, 0, -40)}),
        (f["thrash"] - f["inhale"] + 90, {"spine": (-16, 0, 0), "neck": (-20, 0, 0), "wingL": (0, -20, 0),
                                          "wingR": (0, 20, 0)}),
        (f["collapse"] - f["inhale"], {"spine": (-16, 0, 0), "neck": (-20, 0, 0)}),
        (f["collapse"] - f["inhale"] + 40, {"spine": (-14, 0, 0), "neck": (-26, 0, 0), "head": (6, 0, 0),
                                            "armL": (-20, 0, 0), "armR": (-20, 0, 0), "lids": (1, 0, 0)}),
        (f["eye_again"] - f["inhale"], {"spine": (-14, 0, 0), "neck": (-26, 0, 0), "head": (6, 0, 0),
                                        "armL": (-20, 0, 0), "armR": (-20, 0, 0), "lids": (1, 0, 0)}),
        (f["eye_again"] - f["inhale"] + 3, {"spine": (-14, 0, 0), "neck": (-24, 0, 0), "head": (6, 0, 0),
                                            "armL": (-20, 0, 0), "armR": (-20, 0, 0)}),
        (f["eye_again"] - f["inhale"] + 50, {})]))
    P["hk"] = hk

    # ---- root path (x, y) + yaw + height
    s1a, s2a, s3a = (strike_root(k, STRIKES[k]["y0"]) for k in ("dodge_p1", "tail_p2", "tail_p3"))
    root = [(1, S["d_sleep"], "stop"), (1300, S["d_sleep"], "stop"), (1440, S["d_mid"], "stop"),
            (f["circling"], S["d_mid"], "stop")]
    for i, p in enumerate([(1.4, 27.0), (2.6, 23.8), (2.2, 20.8)]):          # inside the cover drums (v1 clipped them)
        root.append((f["circling"] + (f["tail_test"] - 70 - f["circling"]) * (i + 1) // 3, p, ""))
    root += [(f["tail_test"] - 34, s1a, "stop")]
    root += [(f["dodge_p1"] + 30, None, "strike_end:dodge_p1"),
             (f["combo"] - 20, D_COMBO, "stop"), (f["parry"] + 16, D_COMBO, "stop"),
             (f["tail_p2"] - 34, s2a, "stop"), (f["tail_p2"] + 40, None, "strike_end:tail_p2"),
             (f["neck_slash"] - 40, (2.2, 24.8), "stop"), (f["bite_dodge"] - 22, (1.6, 24.8), "stop"),
             (f["bite_dodge"], (0.4, 23.4)), (f["bite_dodge"] + 16, (0.4, 23.6), "stop"),
             (f["bite_dodge"] + 40, (0.4, 23.0), "stop"),
             (f["tail_p3"] - 30, s3a, "stop"), (f["tail_p3"] + 30, None, "strike_end:tail_p3"),
             (f["bridge"] + 44, (-1.2, 26.0), "stop"), (f["spikes"] - 2, (-1.2, 26.0), "stop"),
             (f["spikes"] + 12, (-1.0, 27.2), "stop"), (f["stumble"] + 36, (-1.0, 26.8), "stop"),
             (f["rocks"] - 10, (0.6, 23.8), "stop"),
             (f["bite_miss"] - 30, (-1.4, 25.4), "stop"), (f["bite_miss"], (-2.0, 24.2)),
             (f["bite_miss"] + 14, (-2.6, 24.0), "stop"), (f["slam"] - 44, (-4.4, 22.8), "stop"),
             (f["slam"], (-5.9, 20.0), "stop"), (f["slam"] + 70, (-5.0, 21.2), "stop"),
             (f["potion"], (-1.0, 23.4), "stop"), (f["gem_out"], (-1.5, 24.2), "stop"),
             (f["launch"] - 40, S["d_vault"], "stop"), (f["roof"], S["d_vault"], "stop")]
    root += flight_path(E)
    root += [(f["lands"], S["d_land"], "stop"), (f["eye_again"] + 20, S["d_land"], "stop"),
             (7340, (0.0, -36.0)), (7372, (0.0, -16.0)), (7392, S["d_gate"], "stop"), (7628, S["d_gate"], "stop")]
    P["root"] = [resolve_strike_end(k) for k in root]
    yaw = [(1, 180.0), (1300, 180.0), (1440, 180.0), (f["circling"], None),
           ("motion", f["circling"] + 8, f["tail_test"] - 74),
           (f["tail_test"] - 50, 60.0)]          # turns AWAY clockwise (tail curled, sweeping N→E, clear of him)
    strikes = []
    for key, st in STRIKES.items():
        c = f[key]
        a = c - 34
        strikes.append(dict(key=key, a=a - 6, b=c + 40, pivot=st["pivot"], spin=st["yc"] - st["y0"]))
        yaw += [(a, st["y0"], "stop"), (a + 12, st["y0"] + st["wind"] * (1 if st["yc"] < st["y0"] else -1), "stop"),
                (c, st["yc"]), (c + 8, st["yc"] + st["follow"] * (1 if st["yc"] < st["y0"] else -1), "stop"),
                (c + 30, st["yc"] + st["follow"] * 0.3 * (1 if st["yc"] < st["y0"] else -1), "stop")]
    P["strikes"] = strikes
    yaw += [(f["dodge_p1"] + 30, strike_settle_yaw("dodge_p1"), "stop"), (f["combo"] - 50, -160.0),
            (f["combo"] - 20, None), (f["combo"], None), (f["parry"], None), (f["parry"] + 30, None)]
    yaw += [(f["tail_p2"] + 50, None), (f["neck_slash"] - 40, None), (f["neck_slash"], None),
            (f["bite_dodge"] - 22, None), (f["bite_dodge"], None), (f["bite_dodge"] + 16, None)]
    yaw += [("motion", f["tail_p3"] + 36, f["bridge"] + 30), (f["bridge"] + 44, None), (f["spikes"], None),
            (f["stumble"], None), (f["rocks"], None), (f["dfireballs"], None),
            (f["bite_miss"], None), (f["slam"] - 44, None), (f["slam"], None), (f["slam"] + 70, None), ("motion", f["slam"] + 110, f["potion"] - 20),
            (f["potion"], 200.0), (f["gem_out"], 150.0), (f["launch"] - 60, 180.0), (f["roof"], 180.0)]
    yaw += flight_yaw(E)
    yaw += [(f["lands"], 0.0), (f["eye_again"] + 20, 0.0), (7392, 0.0), (7628, 0.0)]
    P["yaw"] = yaw
    hz = [(1, 2.05, "rel"), (f["rise_roar"], 2.05, "rel"), (f["rise_roar"] + 26, 0.0, "rel"),
          (f["slam"] - 40, 0.0, "rel"), (f["slam"] - 18, 1.6, "rel"), (f["slam"] - 6, 1.7, "rel"),
          (f["slam"], 0.0, "rel")]
    hz += flight_height(E)
    hz += [(f["eye_again"] + 20, 0.0, "rel")]
    hz += [(f["eye_again"] + 30, 2.5 - FLY_BODY_Z, "abs"), (7340, 15.0 - FLY_BODY_Z, "abs"),
           (7372, 15.0 - FLY_BODY_Z, "abs"), (7384, 10.5 - FLY_BODY_Z, "abs"), (7392, 0.0, "rel"), (7628, 0.0, "rel")]
    P["hz"] = hz
    P["air"] = [(1, f["rise_roar"] + 24), (f["slam"] - 44, f["slam"] + 6), (f["launch"] - 14, f["lands"] + 2),
                (f["eye_again"] + 20, 7394)]
    return P


# ------------------------------------------------------------------------------------------------ flight
FLY_BODY_Z = 7.6


def flight_path(E):
    """roof burst → climb out → loop round the moon → long dive over the cathedral → fire run over the courtyard (kept
    INSIDE the curtain walls' clearance, v1 clipped CurtainWall_S / WallWalks / a corner tower) → strafing → landing
    on clear paving (v1 landed on a rubble chunk)."""
    f = {k: e.f for k, e in E.items()}
    r, m, dv = f["roof"], f["moon"], f["dive"]
    return [(r + 20, (0.0, 36.5)), (r + 60, (-4.0, 42.0)), (r + 110, (-25.0, 70.0)), (m - 30, (-50.0, 88.0)),
            (m, (-64.0, 85.0)), (m + 39, (-89.0, 110.0)), (m + 79, (-64.0, 135.0)), (m + 119, (-39.0, 110.0)),
            (m + 159, (-20.0, 75.0)), (dv - 74, (0.0, 40.0)), (dv - 52, (0.0, 22.0)), (dv - 30, (0.0, 2.0)), (dv, (0.0, -20.0)),
            (dv + 40, (0.0, -50.0)), (dv + 80, (-22.0, -62.0)), (f["big_fireball"] - 10, (-20.0, -56.0)),
            (f["big_fireball"] + 30, (-14.0, -52.0)), (f["volley"] - 30, (-30.0, -60.0)), (f["volley"], (-24.0, -56.0)),
            (f["volley"] + 40, (-12.0, -66.0)), (f["dash_dodges"], (8.0, -68.0)), (f["lands"] - 50, (6.0, -66.0))]


def flight_yaw(E):
    return [(E["launch"].f + 70, 180.0), (E["roof"].f + 20, 180.0), ("motion", E["roof"].f + 40, E["lands"].f - 40),
            (E["lands"].f - 30, -10.0)]


def flight_height(E):
    f = {k: e.f for k, e in E.items()}
    L, r, m, dv = f["launch"], f["roof"], f["moon"], f["dive"]
    return [(L - 12, 0.0, "rel"), (L + 2, 8.0 + 2.5 - FLY_BODY_Z, "abs"), (L + 70, 8.0 + 9.0 - FLY_BODY_Z, "abs"),
            (r - 30, 8.0 + 25.0 - FLY_BODY_Z, "abs"), (r - 6, 8.0 + 28.0 - FLY_BODY_Z, "abs"),
            (r, 8.0 + 34.0 - FLY_BODY_Z, "abs"),
            (r + 20, 8.0 + 44.0 - FLY_BODY_Z, "abs"), (r + 60, 52.0, "abs"), (r + 110, 72.0, "abs"),
            (m - 30, 100.0, "abs"), (m, 108.0, "abs"), (m + 39, 112.0, "abs"), (m + 79, 115.0, "abs"),
            (m + 119, 112.0, "abs"), (m + 159, 95.0, "abs"), (dv - 74, 70.0, "abs"), (dv - 52, 64.0, "abs"), (dv - 30, 50.0, "abs"),
            (dv, 9.0 - FLY_BODY_Z, "abs"), (dv + 40, 9.0 - FLY_BODY_Z, "abs"), (dv + 80, 12.0, "abs"),
            (f["big_fireball"] - 10, 18.0, "abs"), (f["big_fireball"] + 30, 20.0, "abs"), (f["volley"] - 30, 16.0, "abs"),
            (f["volley"], 15.0, "abs"), (f["volley"] + 40, 16.0, "abs"), (f["dash_dodges"], 16.0, "abs"),
            (f["lands"] - 50, 8.0, "abs"), (f["lands"] - 12, 6.0 - FLY_BODY_Z, "abs"),
            (f["lands"], 1.0 - FLY_BODY_Z, "abs"), (f["lands"] + 8, 0.0, "rel")]
