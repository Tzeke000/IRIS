"""Blast Off v4 — a real flight model for the rocket's LOCAL motion (Zeke 10-13, option A).

Pure numpy, deterministic, no Blender. build_scene.py imports it and bakes the outputs to keys.

What is physical here (SI units, one integration step = 1/(FPS*SUB) s):
  * mass: liftoff 2,610 t; three 5-segment SRBs (730 t each loaded) + a 4-RS-25 core, propellant burned off
    at the engines' real-ish mass flow (thrust / (Isp * g0)); booster 3 leaves with whatever it has left.
  * thrust: SRB 10.0 MN each (sea level), RS-25 2.28 MN each; T/W at liftoff ~1.53 (SLS: ~1.5).
  * gravity 9.81 m/s^2, drag 0.5*rho(h)*v^2*Cd(M)*A with an exponential atmosphere (pad climb only; the
    big-picture altitude after the pad is STORY-PACED, as agreed with Zeke).
  * attitude: 2-axis rigid-body rotation I*w' = tau_dist + tau_aero + tau_ctrl, I = m*L^2/12.
    tau_ctrl comes from thrust-vector control: every engine and booster nozzle gimbals, CLAMPED to +/-8 deg,
    so the control torque saturates at T * arm * sin(8 deg). A cascaded PID (angle -> rate -> torque) with
    an integrator (it learns the booster-3 thrust imbalance and holds a trim gimbal, as a real TVC loop does).
  * aerodynamics: a rocket is statically UNSTABLE (centre of pressure ahead of the centre of mass), so the
    tilt itself adds a destabilising torque q*A*CNa*(x_cg - x_cp)*phi near max-Q.
  * disturbances: booster 3's thrust sputters after the alarm (snare-locked dips), then it separates
    (its thrust leaves an off-axis imbalance the TVC must trim), then it explodes ~60 m away: an
    overpressure shove applied BELOW the centre of mass (= a translation kick + a big tipping torque).
  * translation: the thrust vector tilts with the body (+ the gimbal), so the climb SAGS (cos) and DRIFTS
    sideways (sin); guidance (an outer loop) then commands a small counter-tilt to reel the drift back.
  * roll: the blast's roll kick is damped by RCS thrusters (on/off jets) -> puff flags for the build.

Honest calibration note: with a stock SLS mass/inertia and a ~40 m gimbal-to-CoM arm, +/-8 deg of TVC gives only
~0.15 rad/s^2 of authority and a 30 deg excursion would take well over 10 s to recover. This model uses the same
formulas with a 70 m arm (1.75x), so the near-loss settles inside the ~10 s Zeke's song gives it (0:56 -> 1:06). The
SHAPE is the real one: saturation -> ~28 deg overshoot -> a damped swing back -> trim; numbers in the build log.
"""
import math

import numpy as np

G0 = 9.81
DEG = math.pi / 180.0


class Vehicle:
    m0 = 2.61e6               # kg at liftoff
    srb_T = 10.0e6            # N each, 3 boosters
    srb_isp = 268.0
    srb_m0 = 730e3            # kg each, loaded
    rs25_T = 2.28e6           # N each, 4 engines
    rs25_isp = 366.0
    L = 98.0                  # m, stack length (inertia)
    arm = 70.0                # m, TVC moment arm (gimbal -> CoM ~40 m on SLS; 1.75x so it recovers in ~10 s)
    gimbal_max = 8.0 * DEG
    area = 87.0               # m^2, frontal (core + 3 SRBs)
    srb_r = 6.2               # m, booster axes from the core axis
    srb_az = (math.pi, math.pi / 3.0, -math.pi / 3.0)   # BoosterL 180 deg, BoosterR +60, Booster3 -60 (camera side)


V = Vehicle()


def mdot_srb():
    return V.srb_T / (V.srb_isp * G0)


def mdot_core():
    return 4 * V.rs25_T / (V.rs25_isp * G0)


def rho(h):
    return 1.225 * np.exp(-np.maximum(h, 0.0) / 8500.0)


def cd(mach):
    # subsonic 0.45, transonic bump to ~0.9, supersonic 0.6
    return 0.45 + 0.45 * np.exp(-((mach - 1.1) / 0.35) ** 2) + 0.15 * np.clip(mach - 1.1, 0, 1)


def pad_climb(n, lift, fps, sub=10):
    """Vertical climb from the pad: altitude (m) and speed (m/s) per frame index (0 before `lift`)."""
    dt = 1.0 / (fps * sub)
    alt = np.zeros(n)
    vel = np.zeros(n)
    h = v = 0.0
    m = V.m0
    T = 3 * V.srb_T + 4 * V.rs25_T
    md = 3 * mdot_srb() + mdot_core()
    for i in range(lift, n):
        for _ in range(sub):
            mach = v / 330.0
            D = 0.5 * rho(h) * v * v * cd(mach) * V.area
            a = T / m - G0 - D / m
            v = max(v + a * dt, 0.0)
            h += v * dt
            m -= md * dt
        alt[i] = h
        vel[i] = v
    return alt, vel


def b3_thrust_frac(n, alarm, sep, snare):
    """Booster 3 thrust fraction per frame: 1 → snare-locked sputter dips after the alarm → 0 at sep."""
    f = np.ones(n)
    idx = np.arange(n)
    after = (idx >= alarm) & (idx < sep)
    f[after] = 1.0 - 0.35 * snare[after] - 0.08 * np.sin(idx[after] * 1.7) ** 2
    f[idx >= sep] = 0.0
    return f


def attitude(n, fps, t0, t1, lift, sep, boom, b3frac, blast_dir_xy, blast_below=34.0,
             blast_impulse=1.4e7, roll_kick=0.06, sub=10, q_dyn=26e3,
             k_ang=1.6, w_max=20.0 * DEG, k_rate=1.05, k_int=0.25, guid_on=None, k_gv=0.35, k_gx=0.03,
             guid_max=2.0 * DEG, aero_k=10.0):
    """Integrate the near-loss-of-control tumble over frame indices [t0, t1).

    blast_dir_xy: unit 2-vector (vehicle frame) pointing FROM the blast TOWARD the stack (the shove direction).
    Controller (cascaded, like a real TVC autopilot): angle error -> rate command (limited) -> torque
    I*k_rate*(w_cmd - w) + an integrator (learns the booster-3 thrust imbalance), then CLAMPED to the gimbal
    authority T*arm*sin(8 deg). Guidance (outer loop, from frame guid_on) asks for a small counter-tilt that
    nulls the sideways drift (<= guid_max).
    Returns a dict of per-frame arrays (zeros outside [t0, t1)):
      phi (n,2) tilt angles about body x / y (rad) — WOB rotation_euler x, y
      roll (n,)  roll angle (rad)
      pos (n,3)  translation of the vehicle off its nominal path (m): x, y lateral, z = sag (<0)
      gimbal (n,2) nozzle rotation about x / y (rad), clamped to +/-8 deg
      rcs (n,)   +1/-1 when roll RCS jets fire, else 0
      sat (n,)   1 while the TVC is saturated
    """
    dt = 1.0 / (fps * sub)
    out = dict(phi=np.zeros((n, 2)), w=np.zeros((n, 2)), roll=np.zeros(n), pos=np.zeros((n, 3)),
               gimbal=np.zeros((n, 2)), rcs=np.zeros(n), sat=np.zeros(n), tw=np.zeros(n), vel=np.zeros((n, 3)))
    guid_on = boom + int(8.0 * fps) if guid_on is None else guid_on
    phi = np.zeros(2)
    w = np.zeros(2)
    integ = np.zeros(2)
    psi = dpsi = 0.0
    x = np.zeros(2)
    vx = np.zeros(2)
    z = vz = 0.0
    bd = np.asarray(blast_dir_xy, float)
    bd = bd / (np.linalg.norm(bd) + 1e-12)
    pos_b = [np.array([V.srb_r * math.cos(a), V.srb_r * math.sin(a)]) for a in V.srb_az]
    tsep = (sep - lift) / fps
    m3 = V.srb_m0 - mdot_srb() * tsep                       # what booster 3 still weighs when it leaves
    n_bl = max(int(round(0.12 * fps * sub)), 1)              # blast pulse length in substeps
    rcs = 0.0
    tau_c = np.zeros(2)
    T = 1.0
    satur = False
    for i in range(t0, t1):
        tt = (i - lift) / fps                                  # seconds since liftoff
        for s in range(sub):
            ts = tt + s * dt
            on3 = b3frac[i]
            m = V.m0 - (3 * mdot_srb() + mdot_core()) * ts
            if i >= sep:
                m = m - m3 + mdot_srb() * (ts - tsep)          # booster 3 gone (stop counting its burn)
            I = m * V.L ** 2 / 12.0
            T_b = [V.srb_T, V.srb_T, V.srb_T * on3]
            T = 4 * V.rs25_T + sum(T_b)
            tau_d = np.zeros(2)                                # booster thrust imbalance about the core axis
            for pb, tb in zip(pos_b, T_b):
                tau_d += np.array([pb[1] * tb, -pb[0] * tb])
            tau_a = q_dyn * V.area * aero_k * phi           # q*A*CNa*(x_cg-x_cp)*phi: statically unstable
            k = (i - boom) * sub + s                           # blast: a 0.12 s overpressure pulse,
            F_bl = bd * blast_impulse / 0.12 if 0 <= k < n_bl else np.zeros(2)   # applied below the CoM
            tau_bl = np.array([blast_below * F_bl[1], -blast_below * F_bl[0]])
            # guidance: counter-tilt to null the sideways drift (a = T/m * (sin phi_y, -sin phi_x))
            ref = np.zeros(2)
            if i >= guid_on:
                g = min((i - guid_on) / (2.0 * fps), 1.0)
                a_cmd = -(k_gv * vx + k_gx * x) * g
                tw = T / m
                sy = np.clip(a_cmd[0] / tw, -math.sin(guid_max), math.sin(guid_max))
                sx = np.clip(-a_cmd[1] / tw, -math.sin(guid_max), math.sin(guid_max))
                ref = np.array([math.asin(sx), math.asin(sy)])
            err = phi - ref
            w_cmd = np.clip(-k_ang * err, -w_max, w_max)
            tau_c = I * (k_rate * (w_cmd - w) - k_int * integ)
            tau_max = T * V.arm * math.sin(V.gimbal_max)
            mag = float(np.linalg.norm(tau_c))
            satur = mag > tau_max
            if satur:
                tau_c = tau_c * (tau_max / mag)
            else:
                integ = integ + err * dt                       # anti-windup: only integrate when unsaturated
            w = w + (tau_d + tau_a + tau_bl + tau_c) / I * dt
            phi = phi + w * dt
            if i == boom and s == 0:                           # roll kick, damped by on/off RCS jets
                dpsi += roll_kick
            rcs = 0.0
            if abs(psi) > 0.4 * DEG or abs(dpsi) > 0.3 * DEG:
                rcs = -math.copysign(1.0, psi + 0.9 * dpsi)
            dpsi += rcs * 0.06 * dt
            psi += dpsi * dt
            a_lat = T / m * np.array([math.sin(phi[1]), -math.sin(phi[0])]) + F_bl / m - 0.05 * vx
            vx = vx + a_lat * dt
            x = x + vx * dt
            vz += T / m * (math.cos(math.hypot(phi[0], phi[1])) - 1.0) * dt
            z += vz * dt
        out["phi"][i] = phi
        out["w"][i] = w
        out["roll"][i] = psi
        out["pos"][i] = (x[0], x[1], z)
        out["vel"][i] = (vx[0], vx[1], vz)
        dyg = tau_c[0] / (T * V.arm)                           # nozzle rotation that produces tau_c
        dxg = -tau_c[1] / (T * V.arm)
        out["gimbal"][i] = (-math.asin(np.clip(dyg, -1, 1)), math.asin(np.clip(dxg, -1, 1)))
        out["rcs"][i] = rcs
        out["sat"][i] = 1.0 if satur else 0.0
        out["tw"][i] = T / m
    for k in ("phi", "w", "roll", "pos", "vel", "gimbal"):     # hold the last state after t1
        out[k][t1:] = out[k][t1 - 1]
    return out


def booster3_path(n, fps, lift, sep, boom, a_core, sub=10, pivot_s=0.5, kick=4.0, spin_max=75.0):
    """Booster 3 after an early, uncommanded separation, in the vehicle (rig) frame — it keeps burning.

    Sequence: the aft strut lets go first, so for `pivot_s` it pivots about its FORWARD attach (thrust swings
    the nose in and the tail out ~12 deg, the Challenger-style pivot); then the forward attach rips, the
    airflow over the nose cone flips the rotation and it CARTWHEELS nose-outward (rate builds to spin_max
    deg/s), STILL THRUSTING along its own tumbling axis, kicked outward at `kick` m/s. In the vehicle frame
    it also falls behind, because the stack keeps accelerating (a_core per frame, m/s^2) while the booster's
    thrust turns sideways. Returns loc (n,3) of the nozzle exit (the booster object's origin) in the
    vehicle frame, rot (n,) its pitch-out angle about the slot tangent (rad, + = nose outward),
    and the unit axis (n,3)."""
    az = V.srb_az[2]
    radial = np.array([math.cos(az), math.sin(az), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    base = radial * V.srb_r
    loc = np.tile(base, (n, 1))
    ang_out = np.zeros(n)
    axis_out = np.tile(up, (n, 1))
    m = V.srb_m0 - mdot_srb() * (sep - lift) / fps
    T = V.srb_T
    fwd = 42.0                                                # m, forward attach above the nozzle exit
    mid = 22.0                                                # m, the booster's centre of mass above the exit
    dt = 1.0 / (fps * sub)
    c = base + up * mid                                       # integrate the CENTRE OF MASS
    v = np.zeros(3)
    ang = 0.0
    dang = 0.0
    released = False
    for i in range(sep, n):
        for s in range(sub):
            t = (i - sep) / fps + s * dt
            if t < pivot_s:                                   # pinned at the forward attach
                ang = -math.radians(8.0) * (t / pivot_s) ** 2
                axis = math.cos(ang) * up + math.sin(ang) * radial
                c_new = base + up * fwd - axis * (fwd - mid)
                v = (c_new - c) / dt
                c = c_new
                continue
            if not released:
                released = True
                v = v + radial * kick
            dang = min(dang + math.radians(spin_max) / 1.2 * dt, math.radians(spin_max))
            ang += dang * dt
            axis = math.cos(ang) * up + math.sin(ang) * radial
            a = axis * T / m - up * a_core[i]
            v = v + a * dt
            c = c + v * dt
            m -= mdot_srb() * dt
        axis = math.cos(ang) * up + math.sin(ang) * radial
        loc[i] = c - axis * mid
        ang_out[i] = ang
        axis_out[i] = axis
    return loc, ang_out, axis_out


if __name__ == "__main__":   # quick self-test with the v4 song timings (python3 + numpy)
    FPS, n = 30, 5976
    lift = int(5.47 * FPS) + 4
    alarm, sep, boom = int(43.77 * FPS), int(53.33 * FPS), int(56.07 * FPS)
    alt, vel = pad_climb(n, lift, FPS)
    print("pad: alt @28.1 s %.0f m, v %.0f m/s" % (alt[int(28.1 * FPS)], vel[int(28.1 * FPS)]))
    b3 = b3_thrust_frac(n, alarm, sep, np.zeros(n))
    az = V.srb_az[2]
    out = attitude(n, FPS, int(28.1 * FPS), int(88.17 * FPS), lift, sep, boom, b3, (-math.cos(az), -math.sin(az)))
    tilt = np.degrees(np.linalg.norm(out["phi"], axis=1))
    print("peak tilt %.1f deg at %.2f s; tilt @65.6 s %.1f deg" % (tilt.max(), tilt.argmax() / FPS, tilt[int(65.6 * FPS)]))
    loc, ang, ax = booster3_path(n, FPS, lift, sep, boom, out["tw"] + 2.0)
    mid = loc[boom] + ax[boom] * 22.0
    print("booster 3 at the blast: %.0f m off the core axis" % math.hypot(mid[0], mid[1]))
