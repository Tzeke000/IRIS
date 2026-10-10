"""NEON SAMURAI DUEL: the pure-numpy choreography for "Pull Me Close" (no bpy here).

build_scene.py imports this module. Every per-frame decision lives here, so it can be tested with any
Python that has numpy:

    ~/venvs/iris-v100/bin/python scripts/lyric_scenes/real/samurai/choreo.py   # prints a report

World: metres, +Z up. The plaza is centred on the origin. The north lane (+Y) runs to the gate.
Fighter-local frame: +Y forward, +X right, z measured in STANDING coordinates (pelvis rest 0.98).
local -> world: xy = root + Rz(yaw) @ (x, y); z = z + (pelvis_z - 0.98).
Index i = timeline frame (0-based); Blender frame = i + 1.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SONG_DIR = os.path.join(REPO, "state", "tzeke_songs", "pull_me_close")

# =====================================================================================
# skeleton (rest, standing, character faces +Y). Shared with build_scene.py.
# =====================================================================================
PELVIS_REST = 0.98
BONES = {  # name: (head, tail, parent, connected)
    "hips": ((0, 0, 0.90), (0, 0, 1.00), None, False),
    "spine": ((0, 0, 1.00), (0, 0, 1.22), "hips", True),
    "chest": ((0, 0, 1.22), (0, 0, 1.44), "spine", True),
    "neck": ((0, 0.0, 1.44), (0, 0.01, 1.56), "chest", True),
    "head": ((0, 0.01, 1.56), (0, 0.01, 1.78), "neck", True),
    "upper.R": ((0.19, 0, 1.42), (0.375, -0.02, 1.20), "chest", False),
    "fore.R": ((0.375, -0.02, 1.20), (0.535, 0.03, 0.995), "upper.R", True),
    "hand.R": ((0.535, 0.03, 0.995), (0.585, 0.05, 0.93), "fore.R", True),
    "thigh.R": ((0.10, 0, 0.93), (0.11, 0.02, 0.51), "hips", False),
    "shin.R": ((0.11, 0.02, 0.51), (0.12, -0.01, 0.09), "thigh.R", True),
    "foot.R": ((0.12, -0.01, 0.09), (0.12, 0.15, 0.02), "shin.R", True),
    # secondary-motion (spring) bones: hang from the waist / forearms / head
    "skirt.F": ((0, 0.13, 1.00), (0, 0.20, 0.40), "hips", False),
    "skirt.B": ((0, -0.13, 1.00), (0, -0.20, 0.40), "hips", False),
    "skirt.R": ((0.15, 0, 1.00), (0.24, 0, 0.40), "hips", False),
    "sleeve.R": ((0.45, 0.0, 1.10), (0.47, 0.0, 0.86), "fore.R", False),
    "hood": ((0, -0.08, 1.70), (0, -0.20, 1.45), "head", False),
}
for _n in list(BONES):
    if _n.endswith(".R"):
        h, t, p, c = BONES[_n]
        BONES[_n[:-2] + ".L"] = ((-h[0], h[1], h[2]), (-t[0], t[1], t[2]),
                                 (p[:-2] + ".L") if p and p.endswith(".R") else p, c)
ARM_L1 = float(np.linalg.norm(np.subtract(BONES["upper.R"][1], BONES["upper.R"][0])))
ARM_L2 = float(np.linalg.norm(np.subtract(BONES["fore.R"][1], BONES["fore.R"][0])))
LEG_L1 = float(np.linalg.norm(np.subtract(BONES["thigh.R"][1], BONES["thigh.R"][0])))
LEG_L2 = float(np.linalg.norm(np.subtract(BONES["shin.R"][1], BONES["shin.R"][0])))
ARM_REACH = 0.985 * (ARM_L1 + ARM_L2)
SPRING_BONES = ["skirt.F", "skirt.B", "skirt.R", "skirt.L", "sleeve.R", "sleeve.L", "hood"]

# sword (local: +Y blade from the tsuba to the tip, +Z = cutting edge, hilt along -Y)
BLADE_LEN = 0.72
HILT_LEN = 0.27
GRIP_R_Y = -0.055          # right hand just behind the tsuba
GRIP_L_Y = -0.215
HAND_DIR_S = np.array([0.0, 0.55, -0.835])       # wrist->knuckles in sword space (over the spine)
HAND_DIR_S /= np.linalg.norm(HAND_DIR_S)
WRIST_BACK = 0.065                                 # wrist sits this far behind the grip point
SAYA_LOCAL = np.array([-0.17, 0.10, 0.99])         # saya mouth (left hip), standing coords
SAYA_DIR = np.array([0.20, -0.86, -0.47])          # blade direction when sheathed (tip back-down)
SAYA_DIR /= np.linalg.norm(SAYA_DIR)
SAYA_PULL = 0.25                                   # saya-biki: the saya slides back on draw / noto


# =====================================================================================
# small maths (vectorised over frames where it matters)
# =====================================================================================
def nrm(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def rotz(a):
    a = np.asarray(a, np.float64)
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def rotx(a):
    a = np.asarray(a, np.float64)
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def roty(a):
    a = np.asarray(a, np.float64)
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def frame_yz(d, e):
    """Rotation whose columns are X, Y=d, Z=e (orthonormalised)."""
    y = nrm(d)
    z = nrm(np.asarray(e, np.float64) - np.sum(np.asarray(e) * y, -1, keepdims=True) * y)
    x = np.cross(y, z)
    return np.stack([x, y, z], -1)


def mat2quat(m):
    m = np.asarray(m, np.float64)
    sh = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for k in range(len(m)):
        a = m[k]
        t = tr[k]
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            q[k] = (0.25 * s, (a[2, 1] - a[1, 2]) / s, (a[0, 2] - a[2, 0]) / s, (a[1, 0] - a[0, 1]) / s)
        elif a[0, 0] > a[1, 1] and a[0, 0] > a[2, 2]:
            s = math.sqrt(1.0 + a[0, 0] - a[1, 1] - a[2, 2]) * 2
            q[k] = ((a[2, 1] - a[1, 2]) / s, 0.25 * s, (a[0, 1] + a[1, 0]) / s, (a[0, 2] + a[2, 0]) / s)
        elif a[1, 1] > a[2, 2]:
            s = math.sqrt(1.0 + a[1, 1] - a[0, 0] - a[2, 2]) * 2
            q[k] = ((a[0, 2] - a[2, 0]) / s, (a[0, 1] + a[1, 0]) / s, 0.25 * s, (a[1, 2] + a[2, 1]) / s)
        else:
            s = math.sqrt(1.0 + a[2, 2] - a[0, 0] - a[1, 1]) * 2
            q[k] = ((a[1, 0] - a[0, 1]) / s, (a[0, 2] + a[2, 0]) / s, (a[1, 2] + a[2, 1]) / s, 0.25 * s)
    q = nrm(q)
    return q.reshape(sh + (4,))


def quat2mat(q):
    q = nrm(np.asarray(q, np.float64))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def qmul(a, b):
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q):
    q = np.array(q, np.float64)
    q[..., 1:] *= -1
    return q


def slerp(q0, q1, t):
    q0, q1 = nrm(q0), nrm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    t = np.asarray(t, np.float64)[..., None]
    if d > 0.9995:
        return nrm(q0 + t * (q1 - q0))
    th = math.acos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / math.sin(th)


def swing(a, b):
    """Minimal rotation matrices taking unit vectors a -> b (vectorised)."""
    a, b = nrm(a), nrm(b)
    c = np.sum(a * b, -1)
    ax = np.cross(a, b)
    q = np.concatenate([(1 + c)[..., None], ax], -1)
    bad = c < -0.9999
    if np.any(bad):
        perp = np.cross(a[bad], np.array([1.0, 0, 0]))
        perp[np.linalg.norm(perp, axis=-1) < 1e-6] = (0, 1, 0)
        q[bad] = np.concatenate([np.zeros((bad.sum(), 1)), nrm(perp)], -1)
    return quat2mat(q)


def qcontinuous(q):
    q = np.array(q, np.float64)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def ease(t, kind):
    t = np.clip(np.asarray(t, np.float64), 0, 1)
    if kind == "lin":
        return t
    if kind == "in":         # accelerate (strike into contact)
        return t ** 2.3
    if kind == "out":        # decelerate (land, recoil)
        return 1 - (1 - t) ** 2.6
    if kind == "hold":
        return np.zeros_like(t)
    if kind == "step":
        return (t >= 1).astype(np.float64)
    return t * t * (3 - 2 * t)  # "io"


def curve(n, keys, default_kind="io"):
    """keys: [(frame, value, kind)] ; kind eases the segment ARRIVING at that key."""
    keys = sorted(keys, key=lambda k: k[0])
    out = np.empty(n)
    fs = [k[0] for k in keys]
    vs = [np.asarray(k[1], np.float64) for k in keys]
    out = np.zeros((n,) + vs[0].shape)
    idx = np.arange(n)
    out[idx <= fs[0]] = vs[0]
    for (f0, v0), (f1, v1, kd) in zip(zip(fs, vs), [(k[0], np.asarray(k[1], np.float64),
                                                     k[2] if len(k) > 2 else default_kind) for k in keys[1:]]):
        m = (idx > f0) & (idx <= f1)
        if not np.any(m):
            continue
        t = (idx[m] - f0) / max(f1 - f0, 1e-9)
        e = ease(t, kd).reshape((-1,) + (1,) * v0.ndim)
        out[m] = v0 + (v1 - v0) * e
    out[idx > fs[-1]] = vs[-1]
    return out


def hash01(*k):
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    return (h ^ (h >> 15)) / 4294967295.0


def gsmooth(x, sigma):
    x = np.asarray(x, np.float64)
    if sigma <= 0:
        return x
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if x.ndim == 1:
        return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")
    return np.stack([gsmooth(x[:, j], sigma) for j in range(x.shape[1])], 1)


def age_env(age, tau):
    a = np.asarray(age, np.float64)
    return np.where(a < 1e5, np.exp(-a / tau), 0.0)


# =====================================================================================
# sword poses (fighter-local, standing coords): (tsuba position, blade dir, edge dir)
# =====================================================================================
POSES = {
    "chudan": ((0.02, 0.40, 1.12), (0.0, 0.86, 0.50), (0.0, 0.50, -0.86)),
    "chudan_ext": ((0.02, 0.50, 1.20), (0.0, 0.93, 0.36), (0.0, 0.36, -0.93)),
    "jodan": ((0.02, 0.14, 1.80), (0.0, -0.62, 0.78), (0.0, 0.78, 0.62)),
    "hasso": ((0.21, 0.14, 1.46), (0.08, -0.22, 0.97), (0.0, 0.97, 0.22)),
    "hasso_L": ((-0.21, 0.14, 1.46), (-0.08, -0.22, 0.97), (0.0, 0.97, 0.22)),
    "waki": ((0.24, 0.06, 1.02), (0.32, -0.84, -0.44), (0.0, -0.45, 0.89)),
    "waki_L": ((-0.24, 0.06, 1.02), (-0.32, -0.84, -0.44), (0.0, -0.45, 0.89)),
    "low_L": ((-0.18, 0.30, 0.90), (-0.45, 0.45, -0.77), (0.45, 0.62, 0.64)),
    "tsuki_back": ((0.06, 0.18, 1.14), (0.0, 0.99, 0.12), (0.0, 0.12, -0.99)),
    "zanshin": ((0.30, 0.20, 1.06), (0.80, 0.30, -0.52), (0.0, -0.86, -0.51)),
    "rest": ((0.05, 0.34, 1.00), (0.10, 0.70, -0.70), (0.0, 0.70, 0.71)),
}


# attack templates (fighter-local frames). A = attacker, D = defender. d*/e* = blade direction /
# edge at contact, a* = contact distance from the tsuba. tD = nominal defender tsuba. The solver
# (solve_template) moves tD and picks the root distance D so both blades cross at one point AND
# all four wrists stay inside arm reach.
TEMPLATES = {
    "shomen": dict(dA=(0.25, 0.75, 0.62), eA=(0, 0.62, -0.78), aA=0.46,
                   dD=(-0.75, 0.25, 0.60), eD=(0.3, 0.3, 1.0), aD=0.40, tD=(0.06, 0.34, 1.52),
                   wind="jodan", push=0.20),
    "kesa_R": dict(dA=(0.40, 0.75, 0.52), eA=(-0.5, 0.3, -0.8), aA=0.44,
                   dD=(-0.30, 0.30, 0.90), eD=(-1.0, 0.3, 0.0), aD=0.34, tD=(-0.10, 0.36, 1.18),
                   wind="hasso", push=0.16),
    "kesa_L": dict(dA=(-0.40, 0.75, 0.52), eA=(0.5, 0.3, -0.8), aA=0.44,
                   dD=(0.30, 0.30, 0.90), eD=(1.0, 0.3, 0.0), aD=0.34, tD=(0.14, 0.36, 1.18),
                   wind="hasso_L", push=0.16),
    "do": dict(dA=(0.50, 0.86, 0.06), eA=(-1.0, 0.4, 0.0), aA=0.42,
               dD=(-0.30, 0.25, -0.92), eD=(-1.0, 0.3, 0.0), aD=0.30, tD=(-0.12, 0.34, 1.30),
               wind="waki", push=0.14),
    "tsuki": dict(dA=(0.0, 1.0, 0.06), eA=(0, 0.06, -1.0), aA=0.50,
                  dD=(-0.62, 0.40, 0.68), eD=(0.3, 1.0, 0.0), aD=0.36, tD=(0.10, 0.36, 1.12),
                  wind="tsuki_back", push=0.24),
    "kiriage": dict(dA=(-0.40, 0.80, 0.45), eA=(0.4, 0.3, 0.86), aA=0.42,
                    dD=(0.32, 0.30, -0.90), eD=(1.0, 0.2, 0.0), aD=0.30, tD=(0.10, 0.34, 1.55),
                    wind="low_L", push=0.14),
    # symmetric: both swing at once and meet high in the middle (drop entries)
    "entry": dict(dA=(0.28, 0.62, 0.73), eA=(0, 0.7, -0.7), aA=0.44,
                  dD=(0.28, 0.62, 0.73), eD=(0, 0.7, -0.7), aD=0.44, tD=(0.02, 0.42, 1.38),
                  wind="jodan", wind_D="hasso", push=0.0, sym=True),
    # blades locked near the guards (binds in the calm gaps)
    "bind": dict(dA=(0.22, 0.35, 0.91), eA=(0, 1, -0.2), aA=0.20,
                 dD=(0.22, 0.35, 0.91), eD=(0, 1, -0.2), aD=0.20, tD=(0.02, 0.40, 1.30),
                 wind="hasso", push=0.0, sym=True, Dmin=1.05),
}
# the final pass-through cut is placed by hand (they run past each other, offset sideways)
PASS_D, PASS_LAT, PASS_H = 0.70, 0.95, 1.10


def _shoulders(pitch):
    Rs = rotz(np.zeros(1)) @ rotx(np.array([0.4 * pitch]))
    Rc = rotz(np.zeros(1)) @ rotx(np.array([0.6 * pitch]))
    sp_h = np.array(BONES["spine"][0], float)
    ch_rest = np.array(BONES["chest"][0], float)
    ch_h = sp_h + Rs[0] @ (ch_rest - sp_h)
    Rsc = Rs[0] @ Rc[0]
    return (ch_h + Rsc @ (np.array(BONES["upper.R"][0]) - ch_rest),
            ch_h + Rsc @ (np.array(BONES["upper.L"][0]) - ch_rest))


def _wrists(t, R):
    hd = R @ HAND_DIR_S
    return t + R[:, 1] * GRIP_R_Y - hd * WRIST_BACK, t + R[:, 1] * GRIP_L_Y - hd * WRIST_BACK


def solve_template(T, reach=0.50):
    """Grid-search the defender tsuba (near tD) and the attacker tsuba forward distance so that
    the blades meet at one point and every wrist is within reach; returns the contact geometry."""
    RA = frame_yz(np.array(T["dA"], float), np.array(T["eA"], float))
    RD = frame_yz(np.array(T["dD"], float), np.array(T["eD"], float))
    shA = _shoulders(-0.21)
    shD = _shoulders(-0.21 if T.get("sym") else -0.04)
    t0 = np.array(T["tD"], float)
    best = None
    for dx in np.linspace(-0.15, 0.15, 7):
        for dy in np.linspace(-0.12, 0.12, 7):
            for dz in np.linspace(-0.15, 0.15, 7):
                tD = t0 + (dx, dy, dz)
                mcD = tD + RD[:, 1] * T["aD"]
                for fA in np.linspace(0.28, 0.62, 9):
                    mcA_y = fA + RA[1, 1] * T["aA"]
                    mcA = np.array([-mcD[0], mcA_y, mcD[2]])
                    tA = mcA - RA[:, 1] * T["aA"]
                    D = mcA_y + mcD[1]
                    cost = 0.0
                    for t, R, sh in ((tA, RA, shA), (tD, RD, shD)):
                        for w, s in zip(_wrists(t, R), sh):
                            cost += max(0.0, np.linalg.norm(w - s) - reach) ** 2 * 100
                    cost += max(0.0, T.get("Dmin", 1.30) - D) ** 2 * 40 + 0.05 * float(np.sum((tD - t0) ** 2))
                    cost += 0.4 * max(0.0, abs(tA[0]) - 0.18) ** 2
                    if best is None or cost < best[0]:
                        best = (cost, tA, RA, tD, RD, D, mcA)
    cost, tA, RA, tD, RD, D, mcA = best
    return dict(tA=tA, RA=RA, tD=tD, RD=RD, D=float(D), mc=mcA, cost=cost)


SOLVED = {k: solve_template(T) for k, T in TEMPLATES.items()}
for _k, _T in TEMPLATES.items():
    _T["D"] = SOLVED[_k]["D"]


# =====================================================================================
# the song
# =====================================================================================
class Song:
    def __init__(self, timeline, words):
        TL = np.load(timeline)
        self.TL = TL
        self.fps = int(TL["fps"])
        self.n = len(TL["rms"])
        self.bpm = float(TL["bpm"])
        self.beat = 60.0 * self.fps / self.bpm
        self.bar = 4 * self.beat
        self.entries = sorted(int(e) for e in TL["entries"])
        self.segs = [tuple(int(v) for v in s) for s in TL["segs"]]
        self.kick_f = np.nonzero(TL["T_kage"] == 0)[0]
        self.snare_f = np.nonzero(TL["T_sage"] == 0)[0]
        self.beats = np.nonzero(TL["beat"])[0]
        self.downbeats = np.unique(TL["T_bar_start"])
        self.one_f = np.nonzero(TL["T_one_age"] == 0)[0]
        self.strobe_f = np.nonzero(TL["T_strobe_age"] == 0)[0]
        self.hot = TL["T_hot"].astype(bool)
        self.big = TL["T_big"].astype(bool)
        self.pre = TL["T_pre"].astype(np.float64)
        self.black = TL["T_black"].astype(bool)
        self.KICK = age_env(TL["T_kage"], 3.0)
        self.SNARE = age_env(TL["T_sage"], 4.0)
        self.BASS = gsmooth(TL["bass"], 1.5)
        b = self.BASS
        self.BASSN = np.clip(b / (np.percentile(b, 98) + 1e-9), 0, 1.2)
        hh = np.clip(TL["high"] - gsmooth(TL["high"], 12), 0, None)
        self.HAT = np.clip(hh / (np.percentile(hh, 98) + 1e-6), 0, 1)
        self.RMS = np.clip(TL["rms"] / (np.percentile(TL["rms"], 98) + 1e-9), 0, 1.2)
        self.words = []
        if words and os.path.exists(words):
            for s, e, w in json.load(open(words))["heard"]:
                if s > 128.0:          # the trailing "we'll be right back" is a whisper hallucination
                    continue
                if 69.0 < s < 73.5:   # (same hallucination mid-song)
                    continue
                self.words.append((int(round(s * self.fps)), int(round(e * self.fps)),
                                   w.strip(",.!?").lower()))

    def word_frames(self, word):
        return [s for s, e, w in self.words if w == word]

    def snap_down(self, f):
        d = self.downbeats
        return int(d[np.argmin(np.abs(d - f))])

    def seg_of(self, f):
        for k, (s0, s1, h) in enumerate(self.segs):
            if s0 <= f < s1:
                return k
        return len(self.segs) - 1


# =====================================================================================
# the duel
# =====================================================================================
class Duel:
    """Plans the whole fight and produces every per-frame array build_scene.py bakes."""

    def __init__(self, song: Song):
        self.S = song
        self.n = song.n
        self.log = []
        self._plan_events()
        self._plan_global()
        self._plan_swords()
        self._solve_roots()
        self._feet()
        self._pelvis_fit()
        self._plan_torso()
        self._sword_world()
        self._clamp_reach()
        self._hands()
        self._clash_fx()
        self._springs_input()
        self._cameras()
        self._lights()

    # -------------------------------------------------------------------------------
    def _plan_events(self):
        S, n = self.S, self.n
        E = S.entries
        self.E = E
        w = S.word_frames
        closes = w("close")
        self.closes = closes
        self.w_dangerous = (w("dangerous") or [647])[0]
        self.w_pose = (w("pose") or [664])[0]
        self.w_skin = w("skin") or [892, 925]
        self.w_glows = (w("glows") or [1288])[0]
        self.w_till = (w("till") or [1237])[0]
        # clashes: snares inside the hot segments (min gap), always the entry frame
        clashes = []   # (frame, kind, attacker 0/1)
        big0, big1 = E[5], [s1 for s0, s1, h in S.segs if s0 == E[5]][0]
        for si, (s0, s1, h) in enumerate(S.segs):
            if not h:
                continue
            if s0 == E[6]:                         # final mini drop: the pass-through cut
                clashes.append((s0, "pass", 0))
                continue
            if s1 - s0 < 60:                       # the 1-second hit
                clashes.append((s0, "entry", 0))
                continue
            gap = 9 if s0 == big0 else 11
            cand = sorted(set(int(f) for f in S.snare_f if s0 + gap <= f < s1 - 6))
            sel = [s0]
            for f in cand:
                if f - sel[-1] >= gap:
                    sel.append(f)
            # thin out a little: drop a clash when it would leave < gap before the next
            for k, f in enumerate(sel):
                if k == 0:
                    clashes.append((f, "entry", 0))
                    continue
                ph = int((f - s0) // (2 * S.bar))
                att = (si + ph) % 2
                clashes.append((f, None, att))
        # assign attack kinds
        kinds_cycle = ["kesa_R", "kesa_L", "shomen", "do", "tsuki", "kiriage"]
        prev_k, prev_att = None, None
        out = []
        for k, (f, kind, att) in enumerate(clashes):
            if kind is None:
                is_down = np.min(np.abs(S.downbeats - f)) <= 2
                in_big = S.big[f]
                if prev_att is not None and att != prev_att:
                    kind = "tsuki" if hash01(f, 3) < 0.4 else ("kesa_R" if hash01(f, 5) < 0.5 else "do")
                elif is_down and hash01(f, 7) < 0.7:
                    kind = "shomen"
                elif in_big:
                    kind = "kesa_L" if prev_k == "kesa_R" else "kesa_R"
                else:
                    kind = kinds_cycle[int(hash01(f, 11) * len(kinds_cycle))]
                    if kind == prev_k:
                        kind = kinds_cycle[(kinds_cycle.index(kind) + 1) % len(kinds_cycle)]
            out.append((f, kind, att))
            prev_k, prev_att = kind, att
        self.clashes = out
        # binds: after the last clash of a drop that leads (through a short calm) into another entry
        self.binds = []        # (start, end) frames where the blades stay locked
        for (s0, s1, h), (t0, t1, h2) in zip(S.segs[:-1], S.segs[1:]):
            if h and not h2 and t1 - t0 < 100 and t1 < S.n - 1 and t1 in E:
                last = max(f for f, kk, a in out if s0 <= f < s1)
                self.binds.append((last + 8, t1 - 7))
        self.log.append(f"clashes {len(out)}; binds {self.binds}")

    # -------------------------------------------------------------------------------
    def _plan_global(self):
        """D (root distance), TH (axis angle), M (midpoint), LAT (lateral pass offset)."""
        S, n, E = self.S, self.n, self.E
        cl = self.closes
        Dk, THk, Mk, Lk = [], [], [], []

        def D(f, v, k="io"):
            Dk.append((int(f), v, k))

        def TH(f, v, k="io"):
            THk.append((int(f), v, k))

        # intro: far apart, stalking
        D(0, 9.0), D(150, 9.0)
        TH(0, -0.30), TH(140, -0.30), TH(500, 0.18)
        steps = [8.1, 7.2, 6.2, 5.0, 4.3, 3.7]
        for f, v in zip(cl[:6], steps):
            D(f - 9, Dk[-1][1], "io")
            D(f + 1, v, "out")                       # the step LANDS on the word
        TH(cl[2] + 20, 0.24), TH(self.w_pose, 0.32), TH(cl[5] + 30, 0.30)
        # verse: circle, then the tips touch on "skin on skin"
        sk0, sk1 = self.w_skin[0], self.w_skin[-1]
        D(sk0 - 40, 3.4), D(sk1, 2.30, "io"), D(sk1 + 50, 3.3), D(cl[6] - 40, 3.2)
        TH(sk0 - 40, -0.10), TH(sk1 + 40, -0.25), TH(cl[6] - 30, 0.05)
        for f, v in zip(cl[6:9], [2.95, 2.70, 2.50]):
            D(f - 8, Dk[-1][1], "io")
            D(f + 1, v, "out")
        D(E[0] - 30, 2.45)                            # anticipation: lean back before the lunge
        D(E[0] - 8, 2.45, "io")
        # clashes and gaps
        cl_frames = [f for f, k, a in self.clashes]
        th = 0.0
        m = np.array([0.0, 0.25])
        Mk.append((0, m.copy(), "io"))
        self.clash_M = {}
        bind_set = {b[0]: b for b in self.binds}
        hot_end = {s0: s1 for s0, s1, h in S.segs if h}
        prev_f = None
        for idx, (f, kind, att) in enumerate(self.clashes):
            if kind == "pass":
                continue
            T = TEMPLATES[kind]
            seg_s0 = max(s0 for s0, s1, h in S.segs if s0 <= f)
            nxt = cl_frames[idx + 1] if idx + 1 < len(cl_frames) else None
            # axis rotation: circle a little per clash, alternating direction per phrase
            ph = int((f - seg_s0) // (2 * S.bar))
            dirn = 1 if (ph + E.index(seg_s0) if seg_s0 in E else ph) % 2 == 0 else -1
            th = float(np.clip(th + dirn * (0.07 + 0.05 * hash01(f, 1)), -0.55, 0.55))
            if kind == "entry":
                th = float(np.clip(th * 0.5, -0.3, 0.3))
            # midpoint: attacker presses toward the defender
            u = np.array([math.cos(th), math.sin(th)])
            sgn = 1.0 if att == 0 else -1.0
            m = m + sgn * u * T["push"]
            r = np.linalg.norm(m - (0, 0.25))
            if r > 1.3:
                m = (0, 0.25) + (m - (0, 0.25)) * 1.3 / r
            TH(f, th, "io")
            D(f, T["D"], "in" if prev_f is not None and f - prev_f < 30 else "io")
            Mk.append((f, m.copy(), "io"))
            self.clash_M[f] = (m.copy(), th)
            # opening between clashes (recoil, then the lunge)
            if nxt is not None and nxt - f > 6 and f not in bind_set:
                gap = nxt - f
                if gap > 9:
                    op = 0.14 + 0.16 * min(1.0, (gap - 10) / 20)
                    D(f + max(3, gap // 3), T["D"] + op, "out")
                    D(nxt - max(4, min(7, gap // 3)), T["D"] + op * 0.8, "io")
            if f in bind_set:
                b0, b1 = bind_set[f]
                D(b0, TEMPLATES["bind"]["D"], "out")
                D(b1, TEMPLATES["bind"]["D"] + 0.05, "io")
                D(b1 + 5, 1.95, "out")             # the shove apart on the blackout
                Mk.append((b1, m.copy(), "io"))
            # calm gaps that are not binds: separate and circle
            end = hot_end.get(seg_s0)
            if end is not None and (nxt is None or nxt >= end) and f not in bind_set:
                # last clash of a drop
                calm = [(s0, s1) for s0, s1, h in S.segs if s0 == end]
                if calm:
                    c0, c1 = calm[0]
                    if c1 - c0 >= 100:
                        far = 3.6 if c1 < E[5] else (4.3 if c1 == E[5] else 6.6)
                        if c1 == E[6]:
                            far = 6.6
                        D(c0 + 30, far, "out")
                        TH(c0 + 30, th + 0.15, "io")
                        if c1 == E[6]:
                            # the final stand-off; the charge starts ~1.4 s before the entry
                            D(c1 - 44, far, "io")
                        else:
                            D(c1 - 70, far - 0.4, "io")
                            D(c1 - 20, 2.6, "io")
                            D(c1 - 8, 2.35, "io")       # pull back (anticipation)
                            TH(c1 - 20, th * 0.5, "io")
                        Mk.append((c0 + 40, np.array([0.0, 0.25]), "io"))
                        Mk.append((c1 - 10, np.array([0.0, 0.25]), "io"))
                        m = np.array([0.0, 0.25])
                    else:
                        D(c1 - 8, 2.35, "out")
            prev_f = f
        # the pass-through: D goes 6.6 -> 0 at E6 -> -1.0 (back to back)
        fP = E[6]
        D(fP, PASS_D, "in")
        D(fP + 16, -1.25, "out")
        D(fP + 40, -1.05, "io")
        Lk += [(0, 0.0, "io"), (fP - 44, 0.0, "io"), (fP - 10, 0.95, "io"), (fP, 0.95, "lin"),
               (fP + 16, 0.55, "out"), (fP + 60, 0.0, "io")]
        TH(fP - 40, 0.0), TH(n - 1, 0.0)
        Mk.append((fP - 60, np.array([0.0, 0.25]), "io"))
        self.D = curve(n, Dk)
        self.TH = curve(n, THk)
        self.M = curve(n, Mk)
        self.LAT = curve(n, Lk)

    # -------------------------------------------------------------------------------
    def _sword_keys(self):
        """Per fighter, a list of keys (frame, pose, kind). pose = ('L', pos, R) local or
        ('SHEATH',) ; kind = ease of the segment arriving at this key."""
        S, E = self.S, self.E
        keys = [[], []]
        role = {}

        def P(name, mirror=False):
            p, d, e = POSES[name]
            return ("L", np.array(p, float), frame_yz(np.array(d, float), np.array(e, float)))

        sheath = ("SHEATH",)
        f_draw0 = self.w_dangerous
        f_draw = self.w_pose
        for i in (0, 1):
            keys[i] += [(0, sheath, "io"), (f_draw0, sheath, "io"), (f_draw - 4, sheath, "io")]
            keys[i] += [(f_draw + 2, ("DRAW",), "lin")]          # blade clears the saya
            keys[i] += [(f_draw + 14, P("chudan" if i == 0 else "hasso"), "out")]
        sk1 = self.w_skin[-1]
        for i in (0, 1):
            keys[i] += [(self.w_skin[0] - 30, P("chudan"), "io"), (sk1 - 2, P("chudan_ext"), "io"),
                        (sk1 + 10, P("chudan_ext"), "hold"), (sk1 + 40, P("chudan"), "io")]
        # build into the first drop: A raises to jodan, B to hasso (anticipation over 2 s)
        e0 = E[0]
        keys[0] += [(e0 - 100, P("chudan"), "io"), (e0 - 40, P("jodan"), "io"), (e0 - 7, P("jodan"), "io")]
        keys[1] += [(e0 - 100, P("chudan"), "io"), (e0 - 40, P("hasso"), "io"), (e0 - 7, P("hasso"), "io")]
        # clashes
        cl = self.clashes
        bind_set = {b[0]: b for b in self.binds}
        self.contact = {}
        for k, (f, kind, att) in enumerate(cl):
            dfd = 1 - att
            if kind == "pass":
                T = dict(wind="waki_L", push=0.0, sym=True)
                y = PASS_D / 2
                RA = frame_yz(np.array([1.0, 0.25, -0.30]), np.array([0.0, 1.0, 0.0]))
                RD = frame_yz(np.array([1.0, 0.25, 0.30]), np.array([0.0, 1.0, 0.0]))
                mc = np.array([PASS_LAT / 2, y, PASS_H])
                pA, pD = mc - RA[:, 1] * 0.42, mc - RD[:, 1] * 0.42
            else:
                T = TEMPLATES[kind]
                sv = SOLVED[kind]
                RA, RD, pA, pD = sv["RA"], sv["RD"], sv["tA"], sv["tD"]
            prevf = cl[k - 1][0] if k else None
            gap = f - prevf if prevf is not None else 99
            S_ = int(np.clip(round(gap * 0.38), 4, 8))
            wind = P(T["wind"])
            wind_D = P(T.get("wind_D", T["wind"]))
            con_A = ("L", pA, RA)
            con_D = ("L", pD, RD)
            self.contact[f] = (att, con_A, con_D, kind)
            sym = kind in ("entry", "bind", "pass")
            if prevf is not None and gap < 99 and prevf not in bind_set:
                # rebound off the previous contact for both
                for i in (0, 1):
                    pa, ca, cd, _ = self.contact[prevf]
                    mine = ca if i == pa else cd
                    rb = ("L", mine[1] - mine[2][:, 1] * 0.07 + np.array([0, -0.03, 0.05]), mine[2])
                    keys[i].append((prevf + min(3, gap // 4), rb, "out"))
            if kind in ("entry", "pass") and (prevf is None or gap > 40):
                pass                                 # the build already put them in their windup
            if sym:
                if gap > 14 and prevf is not None:
                    keys[att].append((f - S_ - 2, wind, "io"))
                    keys[dfd].append((f - S_ - 2, wind_D, "io"))
                keys[att].append((f, con_A, "in"))
                keys[dfd].append((f, con_D, "in"))
            else:
                if gap > 2 * S_ + 4:
                    keys[att].append((f - S_ - min(3, (gap - 2 * S_) // 4), wind, "io"))
                keys[att].append((f - S_, wind, "io"))
                keys[att].append((f, con_A, "in"))
                pre = ("L", pD - RA[:, 2] * 0.0 - np.array([0, 0.05, -0.02]), RD)
                keys[dfd].append((f - max(2, S_ // 2), pre, "io"))
                keys[dfd].append((f, con_D, "lin"))
                pushed = ("L", pD + np.array([0, -0.05, -0.02]), RD)
                keys[dfd].append((f + 2, pushed, "out"))
            if f in bind_set:
                b0, b1 = bind_set[f]
                TB = TEMPLATES["bind"]
                RB = frame_yz(np.array(TB["dA"], float), np.array(TB["eA"], float))
                mcB = np.array(TB["mc"], float)
                pB = mcB - RB[:, 1] * TB["aA"]
                bind = ("L", pB, RB)
                self.contact[b0] = (0, bind, bind, "bind")
                for i in (0, 1):
                    keys[i] += [(b0, bind, "out"), (b1, bind, "hold"),
                                (b1 + 5, P("chudan"), "out")]
                # wind up again for the entry that follows
                nxt = [c for c in cl if c[0] > b1]
                if nxt:
                    keys[0].append((nxt[0][0] - 6, P("jodan"), "io"))
                    keys[1].append((nxt[0][0] - 6, P("hasso"), "io"))
            # last clash of a long drop: separate to guard, then wind up for the next entry
            nxt = cl[k + 1][0] if k + 1 < len(cl) else None
            if nxt is not None and nxt - f > 90 and f not in bind_set:
                for i in (0, 1):
                    keys[i].append((f + 24, P("chudan"), "io"))
                if cl[k + 1][1] == "pass":
                    for i in (0, 1):
                        keys[i] += [(nxt - 60, P("chudan"), "io"), (nxt - 30, P("waki_L"), "io"),
                                    (nxt - 8, P("waki_L"), "io")]
                else:
                    keys[0] += [(nxt - 70, P("chudan"), "io"), (nxt - 22, P("jodan"), "io"),
                                (nxt - 7, P("jodan"), "io")]
                    keys[1] += [(nxt - 70, P("chudan"), "io"), (nxt - 22, P("hasso"), "io"),
                                (nxt - 7, P("hasso"), "io")]
        # finale: zanshin, chiburi (flick), noto (sheathe), stand
        fP = E[6]
        for i in (0, 1):
            keys[i] += [(fP + 10, P("zanshin"), "out"), (fP + 60, P("zanshin"), "hold"),
                        (fP + 70, ("L", np.array([0.32, 0.30, 1.30]),
                                   frame_yz(np.array([0.6, 0.5, 0.62]), np.array([0, 0.8, -0.6]))), "io"),
                        (fP + 76, P("zanshin"), "in"),                       # chiburi flick
                        (fP + 110, ("SHEATH_MOUTH",), "io"),
                        (fP + 150, sheath, "io")]
        self.f_sheathed = fP + 150
        for i in (0, 1):
            keys[i].sort(key=lambda k: k[0])
            # drop keys that collide on the same frame (keep the last written: contacts win)
            ded = {}
            for kk in keys[i]:
                if kk[0] in ded and ded[kk[0]][1][0] == "L" and kk[1][0] == "L" and ded[kk[0]][2] == "in":
                    continue
                ded[kk[0]] = kk
            keys[i] = [ded[f] for f in sorted(ded)]
        return keys

    def _plan_swords(self):
        self.skeys = self._sword_keys()

    # -------------------------------------------------------------------------------
    def _solve_roots(self):
        n = self.n
        D, TH, M, LAT = self.D, self.TH, self.M, self.LAT
        u = np.stack([np.cos(TH), np.sin(TH)], 1)
        v = np.stack([-np.sin(TH), np.cos(TH)], 1)
        A = M - u * (D / 2)[:, None] + v * (LAT / 2)[:, None]
        B = M + u * (D / 2)[:, None] - v * (LAT / 2)[:, None]
        # facing: forward = +u for A, -u for B ; yaw such that Rz(yaw) @ (0,1) = forward
        yawA = TH - math.pi / 2
        yawB = TH + math.pi / 2
        self.root = [A, B]
        self.yaw = [np.unwrap(yawA), np.unwrap(yawB)]
        # pelvis height: lower in the fight, dips on contact, bob while walking
        S = self.S
        fight = np.clip(curve(n, [(0, 0.0), (self.w_pose - 5, 0.0), (self.w_pose + 15, 0.6),
                                  (self.E[0] - 60, 0.7), (self.E[0], 1.0),
                                  (self.E[6] + 30, 1.0), (self.E[6] + 90, 0.2), (n - 1, 0.0)]), 0, 1)
        dip = np.zeros(n)
        for f, kind, att in self.clashes:
            w = 1.0 if kind in ("entry", "pass") else 0.6
            seg = np.arange(f - 4, min(n, f + 14))
            seg = seg[seg >= 0]
            t = seg - f
            dip[seg] = np.maximum(dip[seg], w * np.where(t < 0, (1 + t / 4.0) ** 2,
                                                         np.exp(-t / 5.0)))
        self.pelvis = []
        for i in (0, 1):
            speed = np.r_[0, np.linalg.norm(np.diff(self.root[i], axis=0), axis=1)] * S.fps
            run = np.clip(gsmooth(speed, 3) / 2.5, 0, 1)
            z = PELVIS_REST - 0.015 - 0.055 * fight - 0.04 * dip - 0.035 * run
            self.pelvis.append(z)
        self.fight = fight

    # -------------------------------------------------------------------------------
    def local_to_world(self, i, p_local):
        """p_local (N,3) standing coords -> world (N,3)."""
        R = rotz(self.yaw[i])
        xy = np.einsum("nij,nj->ni", R[:, :2, :2], p_local[:, :2]) + self.root[i]
        z = p_local[:, 2] + (self.pelvis[i] - PELVIS_REST)
        return np.concatenate([xy, z[:, None]], 1)

    def _eval_sword_local(self, i):
        """Interpolate the key list into per-frame local tsuba pos + rotation (orbit-slerp)."""
        n = self.n
        keys = self.skeys[i]
        pos = np.zeros((n, 3))
        rot = np.zeros((n, 3, 3))
        sheathed = np.zeros(n)              # 1 = in the saya, (0,1) = sliding out/in
        piv = np.array([0.0, 0.05, 1.36])
        RS = frame_yz(SAYA_DIR, np.array([0.0, 0.0, -1.0]) - SAYA_DIR * SAYA_DIR[2])
        sheath_pose = (SAYA_LOCAL.copy(), RS)

        def resolve(k):
            tag = k[1][0]
            if tag in ("SHEATH",):
                return sheath_pose
            if tag in ("DRAW", "SHEATH_MOUTH"):   # tip at the mouth of the pulled-back saya
                return (SAYA_LOCAL + SAYA_DIR * (SAYA_PULL - BLADE_LEN - 0.02), RS)
            return (k[1][1], k[1][2])

        res = [resolve(k) for k in keys]
        fr = [k[0] for k in keys]
        tags = [k[1][0] for k in keys]
        for j in range(len(keys)):
            f0 = fr[j]
            f1 = fr[j + 1] if j + 1 < len(keys) else n
            p0, R0 = res[j]
            if j + 1 < len(keys):
                p1, R1 = res[j + 1]
                kind = keys[j + 1][2]
            else:
                p1, R1, kind = p0, R0, "hold"
            a, b = max(0, f0), min(n, f1)
            if b <= a:
                continue
            t = (np.arange(a, b) - f0) / max(f1 - f0, 1)
            s = ease(t, kind)
            q0, q1 = mat2quat(R0), mat2quat(R1)
            qs = slerp(q0, q1, s)
            Rs = quat2mat(qs)
            linear = tags[j] in ("SHEATH", "DRAW", "SHEATH_MOUTH") and tags[j + 1 if j + 1 < len(keys) else j] in (
                "SHEATH", "DRAW", "SHEATH_MOUTH")
            if linear:
                pp = p0 + (p1 - p0) * s[:, None]
            else:
                # orbit the chest pivot with the blade's rotation, then correct to the end key
                Rrel = R1 @ R0.T
                qrel = mat2quat(Rrel)
                qr = slerp(np.array([1.0, 0, 0, 0]), qrel, s)
                Rr = quat2mat(qr)
                r0 = p0 - piv
                orb = piv + np.einsum("nij,j->ni", Rr, r0)
                endp = piv + Rrel @ r0
                pp = orb + (p1 - endp) * s[:, None]
            pos[a:b] = pp
            rot[a:b] = Rs
            st = 1.0 if tags[j] == "SHEATH" else 0.0
            if tags[j] == "SHEATH" and j + 1 < len(keys) and tags[j + 1] == "DRAW":
                st = 1.0
            if tags[j] == "DRAW" or (j + 1 < len(keys) and tags[j + 1] == "SHEATH" and tags[j] == "SHEATH_MOUTH"):
                st = 0.5
            sheathed[a:b] = st
        return pos, rot, sheathed

    def _plan_torso(self):
        n = self.n
        fd, fP = self.w_pose, self.E[6]
        self.saya_off = curve(n, [(0, 0.0), (fd - 6, 0.0), (fd + 2, SAYA_PULL, "out"), (fd + 24, 0.0),
                                  (fP + 96, 0.0), (fP + 110, SAYA_PULL), (fP + 150, 0.0, "io")])
        self.sw_local = []
        self.sheathed = []
        self.pitch, self.twist = [], []
        for i in (0, 1):
            p, R, sh = self._eval_sword_local(i)
            self.sw_local.append([p, R])
            self.sheathed.append(sh)
        # strike intensity per fighter (attacker leans in, defender braces)
        for i in (0, 1):
            I = np.zeros(n)
            for f, kind, att in self.clashes:
                w = 1.0 if att == i or kind in ("entry", "pass", "bind") else -0.35
                seg = np.arange(max(0, f - 7), min(n, f + 16))
                t = seg - f
                e = np.where(t < 0, (1 + t / 7.0) ** 2, np.exp(-t / 6.0))
                I[seg] = np.where(np.abs(w * e) > np.abs(I[seg]), w * e, I[seg])
            p = self.sw_local[i][0]
            twist = np.clip(-1.1 * (p[:, 0] - 0.02), -0.5, 0.5) * np.clip(1 - self.sheathed[i], 0, 1)
            twist = gsmooth(twist, 1.2)
            breathe = 0.012 * np.sin(np.arange(n) / self.S.fps * 2 * math.pi / (4.2 + 0.4 * i))
            pitch = -0.20 * np.clip(I, 0, 1) + 0.07 * np.clip(-I, 0, 1) - 0.06 * self.fight + breathe
            self.pitch.append(gsmooth(pitch, 1.0))
            self.twist.append(twist)

    def chest_fk(self, i):
        """Armature-space spine/chest rotations and shoulder positions (standing coords)."""
        Rs = rotz(0.4 * self.twist[i]) @ rotx(0.4 * self.pitch[i])
        Rc = rotz(0.6 * self.twist[i]) @ rotx(0.6 * self.pitch[i])
        sp_h = np.array(BONES["spine"][0], float)
        ch_rest = np.array(BONES["chest"][0], float)
        ch_h = sp_h + np.einsum("nij,j->ni", Rs, ch_rest - sp_h)
        Rsc = Rs @ Rc
        shR = ch_h + np.einsum("nij,j->ni", Rsc, np.array(BONES["upper.R"][0]) - ch_rest)
        shL = ch_h + np.einsum("nij,j->ni", Rsc, np.array(BONES["upper.L"][0]) - ch_rest)
        nk = ch_h + np.einsum("nij,j->ni", Rsc, np.array(BONES["neck"][0]) - ch_rest)
        return Rs, Rc, ch_h, shR, shL, nk

    # -------------------------------------------------------------------------------
    def _sword_world(self):
        pass

    def _grip_points_local(self, p, R):
        """wrist targets (local) for both hands given tsuba pos/rot (local)."""
        hd = np.einsum("nij,j->ni", R, HAND_DIR_S)
        gR = p + R[:, :, 1] * GRIP_R_Y
        gL = p + R[:, :, 1] * GRIP_L_Y
        return gR - hd * WRIST_BACK, gL - hd * WRIST_BACK

    def _clamp_reach(self):
        """Pull the sword toward the body wherever a wrist would be out of reach."""
        n = self.n
        self.fk = []
        self.reach_fix = []
        for i in (0, 1):
            p, R = self.sw_local[i]
            Rs, Rc, ch_h, shR, shL, nk = self.chest_fk(i)
            self.fk.append((Rs, Rc, ch_h, shR, shL, nk))
            grip = (self.sheathed[i] <= 0).astype(np.float64)      # no clamp while in / sliding
            fix = np.zeros((n, 3))
            for _ in range(3):
                wR, wL = self._grip_points_local(p + fix, R)
                dR = np.linalg.norm(wR - shR, axis=1)
                dL = np.linalg.norm(wL - shL, axis=1)
                exR = np.clip(dR - ARM_REACH, 0, None) * grip
                exL = np.clip(dL - ARM_REACH, 0, None) * grip
                fix += nrm(shR - wR) * exR[:, None] + nrm(shL - wL) * exL[:, None]
            fix = gsmooth(fix, 1.0)
            self.reach_fix.append(np.linalg.norm(fix, axis=1))
            self.sw_local[i][0] = p + fix
        cf = [f for f, k, a in self.clashes]
        rf = np.array([max(self.reach_fix[0][f], self.reach_fix[1][f]) for f in cf])
        self.log.append(f"reach fix at clashes: mean {rf.mean() * 100:.1f} cm, max {rf.max() * 100:.1f} cm, "
                        f"overall max {max(r.max() for r in self.reach_fix) * 100:.1f} cm")

    # -------------------------------------------------------------------------------
    def _hands(self):
        """World sword transforms + wrist targets + grip weights + elbow poles."""
        n = self.n
        self.sword_pos, self.sword_rot = [], []
        self.wrist, self.hand_rot, self.grip_w, self.elbow_pole = [], [], [], []
        f_reach = self.w_dangerous                        # right hand goes to the hilt here
        for i in (0, 1):
            p, R = self.sw_local[i]
            Ryaw = rotz(self.yaw[i])
            pw = self.local_to_world(i, p)
            Rw = Ryaw @ R
            self.sword_pos.append(pw)
            self.sword_rot.append(Rw)
            wR, wL = self._grip_points_local(p, R)
            # hands off the sword: right hand hangs, left hand rests on the saya mouth
            Rs, Rc, ch_h, shR, shL, nk = self.fk[i]
            hang = np.array([0.25, 0.08, 0.90])
            on_saya = (SAYA_LOCAL + np.array([-0.02, 0.06, 0.04]))[None] + SAYA_DIR[None] * self.saya_off[:, None]
            # grip weights
            gR = curve(n, [(0, 0.0), (f_reach - 8, 0.0), (f_reach, 1.0, "io"), (self.f_sheathed - 30, 1.0),
                           (self.f_sheathed, 1.0), (self.f_sheathed + 14, 0.0)])
            gL = curve(n, [(0, 0.0), (self.w_pose + 6, 0.0), (self.w_pose + 16, 1.0),
                           (self.f_sheathed - 44, 1.0), (self.f_sheathed - 36, 0.0)])
            # left hand stays on the saya while the blade slides in, then rests there
            wR_l = wR * gR[:, None] + hang * (1 - gR[:, None])
            wL_l = wL * gL[:, None] + on_saya * (1 - gL[:, None])
            # the sheathed sword rides the saya (offset along its axis)
            slide = (self.sheathed[i] >= 1.0)[:, None] * SAYA_DIR[None] * self.saya_off[:, None]
            p = p + slide
            self.sw_local[i][0] = p
            pw = self.local_to_world(i, p)
            self.sword_pos[i] = pw
            wR, wL = self._grip_points_local(p, R)
            wR_l = wR * gR[:, None] + hang * (1 - gR[:, None])
            wL_l = wL * gL[:, None] + on_saya * (1 - gL[:, None])
            hd = np.einsum("nij,j->ni", R, HAND_DIR_S)
            G = frame_yz(HAND_DIR_S, np.array([1.0, 0, 0]))
            Hrot = R @ G                                    # hand frame (local)
            self.wrist.append([self.local_to_world(i, wR_l), self.local_to_world(i, wL_l)])
            self.hand_rot.append(Ryaw @ Hrot)
            self.grip_w.append([gR, gL])
            # elbow poles (local, standing): out + down, forward when the hands are high
            hi = np.clip((p[:, 2] - 1.35) / 0.4, 0, 1)
            poles = []
            for side, sh in ((1, shR), (-1, shL)):
                pl = sh + np.stack([side * (0.55 - 0.15 * hi), -0.25 + 0.35 * hi,
                                    -0.55 + 0.45 * hi], 1)
                poles.append(self.local_to_world(i, pl))
            self.elbow_pole.append(poles)

    # -------------------------------------------------------------------------------
    def _feet(self):
        """Foot plants: steps land on kicks in the drops, on the words in the intro;
        the attacker's front foot stamps (fumikomi) on the clash."""
        S, n = self.S, self.n
        self.ankle, self.foot_yaw, self.foot_pitch = [], [], []
        self.landings = []                     # (frame, pos(3), strength)
        kicks = set(int(k) for k in S.kick_f if S.hot[k])
        land_ok = set(kicks)
        for w in self.closes:
            land_ok.add(w + 1)
        land_ok |= set(int(b) for b in S.beats)
        land_ok |= set(int(f) for f, k, a in self.clashes)
        stomp = {}
        for f, kind, att in self.clashes:
            if kind in ("bind",):
                continue
            stomp[f] = att if kind not in ("entry", "pass") else -1
        for i in (0, 1):
            R = rotz(self.yaw[i])
            # stance offsets (local x, y) per frame: calm vs fight
            fw = self.fight
            offR = np.stack([0.14 + 0.0 * fw, 0.0 + 0.28 * fw], 1)
            offL = np.stack([-0.14 + 0.01 * fw, 0.0 - 0.30 * fw], 1)
            yawoff = [np.zeros(n) - 0.08 * fw, np.zeros(n) + 0.45 * fw]
            ideal = []
            for off in (offR, offL):
                xy = np.einsum("nij,nj->ni", R[:, :2, :2], off) + self.root[i]
                ideal.append(xy)
            # predicted-ahead ideal
            pos = [ideal[0][0].copy(), ideal[1][0].copy()]
            fyaw = [self.yaw[i][0] + yawoff[0][0], self.yaw[i][0] + yawoff[1][0]]
            out_p = np.zeros((2, n, 3))
            out_y = np.zeros((2, n))
            out_pt = np.zeros((2, n))
            busy_until = [-1, -1]
            step = [None, None]          # (t0, t1, p0, p1, y0, y1, h)
            for t in range(n):
                for ft in (0, 1):
                    if step[ft] is not None and t > step[ft][1]:
                        step[ft] = None
                # forced stomps for this fighter's front foot (right)
                if step[0] is None:
                    for f, a in stomp.items():
                        if (a == i or a == -1) and f - 5 == t:
                            L = 5
                            tgt = ideal[0][min(n - 1, f + 1)]
                            step[0] = (t, f, pos[0].copy(), tgt.copy(), fyaw[0],
                                       self.yaw[i][min(n - 1, f)] + yawoff[0][min(n - 1, f)], 0.13)
                            break
                for ft in (0, 1):
                    other = 1 - ft
                    if step[ft] is not None:
                        continue
                    hot = S.hot[t]
                    run = np.linalg.norm(ideal[ft][min(n - 1, t + 2)] - ideal[ft][t]) * S.fps / 2
                    L = 5 if run > 1.2 else (6 if hot else 9)
                    fut = min(n - 1, t + L)
                    err = np.linalg.norm(ideal[ft][fut] - pos[ft])
                    if step[other] is not None and step[other][1] - t > 2 and err < 0.28:
                        continue
                    yerr = abs((self.yaw[i][fut] + yawoff[ft][fut] - fyaw[ft] + math.pi) % (2 * math.pi) - math.pi)
                    thr = 0.10 if hot or run > 0.6 else 0.14
                    if err > thr or yerr > 0.4:
                        # prefer landing on an allowed frame (kick / word / beat) within a window
                        land = t + L
                        if err < 0.22 and run < 1.2:
                            cands = [c for c in range(t + L - 1, t + L + 4) if c in land_ok]
                            if not cands:
                                continue
                            land = cands[0]
                        land = min(land, n - 1)
                        h = float(np.clip(0.04 + 0.12 * err, 0.04, 0.16))
                        step[ft] = (t, land, pos[ft].copy(), ideal[ft][land].copy(), fyaw[ft],
                                    self.yaw[i][land] + yawoff[ft][land], h)
                        break
                for ft in (0, 1):
                    if step[ft] is not None:
                        t0, t1, p0, p1, y0, y1, h = step[ft]
                        s = (t - t0) / max(t1 - t0, 1)
                        e = s * s * (3 - 2 * s)
                        pxy = p0 + (p1 - p0) * e
                        z = 0.09 + h * math.sin(math.pi * min(1, s * 1.08)) ** 1.2
                        out_p[ft, t] = (pxy[0], pxy[1], z)
                        out_y[ft, t] = y0 + ((y1 - y0 + math.pi) % (2 * math.pi) - math.pi) * e
                        out_pt[ft, t] = -0.55 * math.sin(math.pi * min(1, s * 1.15)) * (1 - s) ** 0.5
                        if t == t1:
                            pos[ft] = p1.copy()
                            fyaw[ft] = y1
                            sp = np.linalg.norm(p1 - p0) / max(t1 - t0, 1) * S.fps
                            self.landings.append((t, (p1[0], p1[1], 0.0), float(np.clip(0.25 + 0.3 * sp + (0.5 if h > 0.12 else 0), 0, 1.4)), i))
                    else:
                        out_p[ft, t] = (pos[ft][0], pos[ft][1], 0.09)
                        out_y[ft, t] = fyaw[ft]
                        out_pt[ft, t] = 0.0
            self.ankle.append([out_p[0], out_p[1]])
            self.foot_yaw.append([out_y[0], out_y[1]])
            self.foot_pitch.append([out_pt[0], out_pt[1]])
        self.log.append(f"foot landings {len(self.landings)}")

    def _pelvis_fit(self):
        """Sink the pelvis wherever a planted foot would be out of leg reach (wide stances,
        lunges, running), and keep both pelvises level at the clashes (the contact templates
        assume equal heights)."""
        n = self.n
        L = (LEG_L1 + LEG_L2) * 0.965
        hip_dz = BONES["thigh.R"][0][2] - PELVIS_REST
        lims = []
        for i in (0, 1):
            R = rotz(self.yaw[i])
            lim = np.full(n, 9.0)
            for si, b in ((0, "thigh.R"), (1, "thigh.L")):
                hip = np.array(BONES[b][0])
                hw = np.einsum("nij,j->ni", R[:, :2, :2], hip[:2]) + self.root[i]
                a = self.ankle[i][si]
                h = np.linalg.norm(a[:, :2] - hw, axis=1)
                v = np.sqrt(np.clip(L ** 2 - h ** 2, 0.05, None))
                lim = np.minimum(lim, a[:, 2] + v - hip_dz)
            # min-filter then smooth so the body sinks a little BEFORE it needs to
            lim2 = lim.copy()
            for k in range(1, 5):
                lim2[:-k] = np.minimum(lim2[:-k], lim[k:])
                lim2[k:] = np.minimum(lim2[k:], lim[:-k])
            lims.append(np.maximum(gsmooth(lim2, 2.0) - 0.01, 0.80))
        for i in (0, 1):
            self.pelvis[i] = np.minimum(self.pelvis[i], lims[i])
        both = np.minimum(self.pelvis[0], self.pelvis[1])
        w = np.zeros(n)
        for f, kind, att in self.clashes:
            seg = np.arange(max(0, f - 8), min(n, f + 8))
            w[seg] = np.maximum(w[seg], 1 - np.abs(seg - f) / 8.0)
        for i in (0, 1):
            self.pelvis[i] = self.pelvis[i] * (1 - w) + both * w
        self.log.append(f"pelvis range {min(p.min() for p in self.pelvis):.3f}..{max(p.max() for p in self.pelvis):.3f}")

    # -------------------------------------------------------------------------------
    def _clash_fx(self):
        """Actual blade-crossing points (after the reach clamp) for sparks / lights."""
        n = self.n
        ev = []
        for f, kind, att in self.clashes + [(b0, "bind", 0) for b0, b1 in self.binds]:
            if f >= n:
                continue
            segs = []
            for i in (0, 1):
                p0 = self.sword_pos[i][f]
                d = self.sword_rot[i][f][:, 1]
                segs.append((p0 + d * 0.1, p0 + d * BLADE_LEN))
            (a0, a1), (b0, b1) = segs
            c = _closest_seg_seg(a0, a1, b0, b1)
            dA = nrm(a1 - a0)
            dB = nrm(b1 - b0)
            nn = np.cross(dA, dB)
            if np.linalg.norm(nn) < 1e-3:
                nn = np.array([0, 0, 1.0])
            strength = {"entry": 1.0, "pass": 1.3, "bind": 0.35}.get(kind, 0.55)
            if self.S.big[f]:
                strength *= 1.25
            ev.append((f, c[0], nrm(nn), strength, kind, c[1]))
        # the "skin on skin" tip touch
        sk = self.w_skin[-1]
        tips = [self.sword_pos[i][sk] + self.sword_rot[i][sk][:, 1] * BLADE_LEN for i in (0, 1)]
        ev.append((sk, (tips[0] + tips[1]) / 2, np.array([0, 0, 1.0]), 0.12, "tip", float(np.linalg.norm(tips[0] - tips[1]))))
        ev.sort(key=lambda e: e[0])
        self.sparks = ev
        gaps = [e[5] for e in ev if e[4] not in ("tip",)]
        self.log.append(f"blade gap at clashes: median {np.median(gaps) * 100:.1f} cm, "
                        f"90% {np.percentile(gaps, 90) * 100:.1f} cm, max {max(gaps) * 100:.1f} cm")

    def _springs_input(self):
        pass

    # -------------------------------------------------------------------------------
    def _cameras(self):
        """Shot list -> per-frame camera position, target, lens, focus distance, f-stop."""
        S, n, E = self.S, self.n, self.E
        A, B = self.root
        mid = (A + B) / 2
        axis = nrm(np.concatenate([B - A, np.zeros((n, 1))], 1))
        ax_safe = np.where(np.abs(self.D)[:, None] < 0.3,
                           np.stack([np.cos(self.TH), np.sin(self.TH), np.zeros(n)], 1), axis)
        ax_safe = nrm(np.stack([np.cos(self.TH), np.sin(self.TH), np.zeros(n)], 1))
        perp = np.stack([-ax_safe[:, 1], ax_safe[:, 0], np.zeros(n)], 1)    # +90 deg (toward +Y at TH=0)
        mid3 = np.concatenate([mid, np.zeros((n, 1))], 1)
        A3 = np.concatenate([A, np.zeros((n, 1))], 1)
        B3 = np.concatenate([B, np.zeros((n, 1))], 1)
        Z = np.array([0, 0, 1.0])
        hz = np.array([0, 0, 1.0])
        headA = A3 + Z * (self.pelvis[0][:, None] + 0.70)
        headB = B3 + Z * (self.pelvis[1][:, None] + 0.70)
        t_s = np.arange(n) / S.fps

        shots = []          # (start, name)

        def add(f, name):
            shots.append((int(f), name))
        cl = self.closes
        # intro (0 - E0)
        add(0, "establish")                 # crane down the gate lane onto the plaza
        add(S.snap_down(200), "sign")       # the crest lantern in the rain
        add(S.snap_down(316), "hatA")       # rain dripping off A's kasa
        add(S.snap_down(419), "wide_low")   # puddle-level wide: the two of them, far apart
        add(cl[0] - 12, "profile_wide")    # "pull me close": the steps
        add(S.snap_down(600), "hand")       # "dangerous": the hand finds the hilt
        add(self.w_pose - 4, "profile")     # "pose": the draw
        add(cl[3] - 14, "feet")             # the steps splash
        add(cl[5] + 16, "orbit")
        add(self.w_skin[0] - 20, "tips")    # skin on skin: the tips touch
        add(self.w_skin[-1] + 50, "otsB")
        add(S.snap_down(1070), "lane_tele")
        add(cl[6] - 12, "profile")
        add(S.snap_down(1218), "push_in")   # the build: slow push, wind-up
        add(E[0], "profile")                # the clash, side on
        # drops: alternate shots on 2-bar boundaries, cut ON the downbeat
        cyc = ["profile", "otsA", "low", "otsB", "orbit", "high", "profile_wide", "lane_tele"]
        ci = 0
        for si, (s0, s1, h) in enumerate(S.segs):
            if s0 < E[0]:
                continue
            if h and s1 - s0 > 60 and s0 != E[6]:
                big = S.big[s0]
                step = S.bar * (1 if big else 2)
                if s0 != E[0]:
                    add(s0, "low" if si % 2 else "profile")
                f = s0 + step
                while f < s1 - S.bar * 0.75:
                    sd = S.snap_down(f)
                    if sd > shots[-1][0] + 20:
                        add(sd, cyc[ci % len(cyc)])
                        ci += 1
                    f += step
            elif h and s0 == E[6]:
                add(s0 - 46, "charge")
                add(s0 + 18, "back2back")
                add(s0 + 120, "outro_wide")
            elif h:
                add(s0, "high")
            else:
                if s1 - s0 > 100 and s0 < E[6]:
                    add(s0, "profile_wide")
                    if s1 - s0 > 150:
                        add(s0 + (s1 - s0) // 2, "sign" if s0 > E[4] else "orbit")
                else:
                    add(s0, "profile")
        shots.sort()
        ded = []
        for s in shots:
            if ded and s[0] - ded[-1][0] < 14:
                ded[-1] = s
            else:
                ded.append(s)
        self.shots = ded
        shot_of = np.zeros(n, np.int32)
        for k, (f, nm) in enumerate(ded):
            shot_of[f:] = k
        self.shot_of = shot_of
        cam = np.zeros((n, 3))
        tgt = np.zeros((n, 3))
        lens = np.full(n, 35.0)
        fst = np.full(n, 2.8)
        roll = np.zeros(n)
        sm_mid = gsmooth(mid3, 6)
        sm_perp = nrm(gsmooth(perp, 10))
        sm_ax = nrm(gsmooth(ax_safe, 10))
        sm_hA, sm_hB = gsmooth(headA, 3), gsmooth(headB, 3)
        for k, (f0, nm) in enumerate(ded):
            f1 = ded[k + 1][0] if k + 1 < len(ded) else n
            ix = np.arange(f0, f1)
            u = (ix - f0) / max(f1 - f0 - 1, 1)
            sgn = 1.0 if hash01(f0, 2) < 0.75 else -1.0      # mostly from the south (gate behind)
            m = sm_mid[f0:f1]
            P = sm_perp[f0:f1] * sgn
            X = sm_ax[f0:f1]
            if nm == "establish":
                cam[ix] = np.array([0.4, 17.0, 6.5]) + np.outer(u, [0.0, -7.0, -4.6])
                tgt[ix] = np.array([0.0, 0.0, 1.0]) + np.outer(u, [0, 0, 0.2])
                lens[ix], fst[ix] = 30, 5.6
            elif nm == "sign":
                cam[ix] = np.array([3.55, 0.35, 1.95]) + np.outer(u, [0.25, 0.12, 0.05])
                tgt[ix] = np.array([5.88, 1.35, 2.25])
                lens[ix], fst[ix] = 50, 2.0
            elif nm == "hatA":
                hA = sm_hA[f0:f1]
                fwd = np.stack([np.cos(self.TH[ix]), np.sin(self.TH[ix]), 0 * ix], 1)
                cam[ix] = hA + fwd * 1.9 - P * 0.55 + np.outer(np.ones_like(u), [0, 0, 0.12]) + np.outer(u, [0, 0, 0.05])
                tgt[ix] = hA + np.array([0, 0, 0.08])
                lens[ix], fst[ix] = 60, 2.0
            elif nm == "wide_low":
                cam[ix] = m - P * 5.2 + np.array([0, 0, 0.32]) + X * (0.6 - 1.2 * u[:, None])
                tgt[ix] = m + np.array([0, 0, 1.2])
                lens[ix], fst[ix] = 26, 4.0
            elif nm in ("profile", "profile_wide"):
                dist = 3.6 if nm == "profile" else 5.0
                dd = np.maximum(dist, 0.8 * np.abs(self.D[ix]) + 1.6)
                cam[ix] = m - P * dd[:, None] + np.array([0, 0, 1.30]) + X * (0.3 * (u[:, None] - 0.5))
                tgt[ix] = m + np.array([0, 0, 1.22])
                lens[ix], fst[ix] = (32 if nm == "profile" else 28), 2.8
            elif nm == "hand":
                hA = sm_hA[f0:f1]
                saya = self.local_to_world(0, np.tile(SAYA_LOCAL, (n, 1)))[ix]
                fwd = np.stack([np.cos(self.TH[ix]), np.sin(self.TH[ix]), 0 * ix], 1)
                cam[ix] = saya + fwd * 0.75 - P * 0.55 + np.array([0, 0, 0.18])
                tgt[ix] = saya + np.array([0, 0, 0.05]) + fwd * 0.05
                lens[ix], fst[ix] = 60, 2.0
            elif nm == "feet":                 # A's feet stepping in (the splash on "close")
                a3 = gsmooth(A3, 4)[f0:f1]
                cam[ix] = a3 - P * 1.7 + X * (0.9 - 0.3 * u[:, None]) + np.array([0, 0, 0.14])
                tgt[ix] = a3 + X * 0.25 + np.array([0, 0, 0.22])
                lens[ix], fst[ix] = 35, 2.4
            elif nm == "orbit":
                ang = (u * 0.9 - 0.45)[:, None] * sgn
                R2 = np.cos(ang) * (-P) + np.sin(ang) * X
                dd = np.maximum(3.8, 0.7 * np.abs(self.D[ix]) + 2.0)[:, None]
                cam[ix] = m + R2 * dd + np.array([0, 0, 1.45])
                tgt[ix] = m + np.array([0, 0, 1.25])
                lens[ix], fst[ix] = 30, 2.8
            elif nm == "tips":
                tipm = gsmooth((self.sword_pos[0] + self.sword_pos[1]) / 2 + 0.5 * (
                    self.sword_rot[0][:, :, 1] + self.sword_rot[1][:, :, 1]) * BLADE_LEN, 4)[ix]
                cam[ix] = tipm - P * 1.15 + np.array([0, 0, 0.12]) + X * 0.1
                tgt[ix] = tipm
                lens[ix], fst[ix] = 70, 2.0
            elif nm in ("otsA", "otsB"):
                me, op = (sm_hA, sm_hB) if nm == "otsA" else (sm_hB, sm_hA)
                me, op = me[f0:f1], op[f0:f1]
                back = nrm(me - op) * np.array([1, 1, 0])
                side = np.cross(np.array([0, 0, 1.0]), back) * sgn
                cam[ix] = me + back * 1.05 + side * 0.6 + np.array([0, 0, 0.06])
                tgt[ix] = op + np.array([0, 0, -0.35])
                lens[ix], fst[ix] = 32, 2.4
            elif nm == "low":
                cam[ix] = m - P * 2.9 + np.array([0, 0, 0.22]) + X * (0.5 * (0.5 - u[:, None]))
                tgt[ix] = m + np.array([0, 0, 1.15])
                lens[ix], fst[ix] = 24, 2.8
            elif nm == "high":
                cam[ix] = m - P * 3.3 + np.array([0, 0, 4.1]) - P * (0.4 * u[:, None])
                tgt[ix] = m + np.array([0, 0, 1.0])
                lens[ix], fst[ix] = 28, 4.0
            elif nm == "lane_tele":
                base = np.array([-0.5, 15.0, 1.7])                 # from the gate side: pink lane behind
                cam[ix] = base + np.outer(u, [0.0, -0.6, 0.0])
                tgt[ix] = m + np.array([0, 0, 1.25])
                lens[ix], fst[ix] = 85, 2.8
            elif nm == "push_in":
                dd = (4.2 - 1.6 * u)[:, None]
                cam[ix] = m - P * dd + np.array([0, 0, 1.25])
                tgt[ix] = m + np.array([0, 0, 1.35])
                lens[ix], fst[ix] = 35, 2.4
            elif nm == "charge":
                cam[ix] = np.array([0.0, 0.0, 0.0]) + m - P * 3.4 + np.array([0, 0, 0.55])
                tgt[ix] = m + np.array([0, 0, 1.2])
                lens[ix], fst[ix] = 24, 3.2
            elif nm == "back2back":
                ang = (u * 0.5)[:, None]
                R2 = np.cos(ang) * (-P) + np.sin(ang) * X
                cam[ix] = m + R2 * 3.2 + np.array([0, 0, 1.0])
                tgt[ix] = m + np.array([0, 0, 1.3])
                lens[ix], fst[ix] = 35, 2.8
            elif nm == "outro_wide":
                cam[ix] = m - P * 5.5 + np.array([0, 0, 1.6]) + np.outer(u, [0, 0, 2.4]) - P * (2.5 * u[:, None])
                tgt[ix] = m + np.array([0, 0, 1.2]) + np.outer(u, [0, 0, 0.6])
                lens[ix], fst[ix] = 30, 4.0
            else:
                cam[ix] = m - P * 4.0 + np.array([0, 0, 1.3])
                tgt[ix] = m + np.array([0, 0, 1.2])
        # keep the camera inside the set (walls at |x| 6.5 in the plaza; lanes |x| 2.6)
        inlane = np.abs(cam[:, 1]) > 6.2
        lim = np.where(inlane, 2.3, 6.1)
        cam[:, 0] = np.clip(cam[:, 0], -lim, lim)
        cam[:, 1] = np.clip(cam[:, 1], -26, 26)
        cam[:, 2] = np.maximum(cam[:, 2], 0.12)
        # impact shake on clashes in the drops (tiny, decays fast) — target jitter only
        shake = np.zeros((n, 3))
        for f, c, nn, st, kind, g in self.sparks:
            if kind == "tip":
                continue
            seg = np.arange(f, min(n, f + 10))
            a = st * 0.035 * np.exp(-(seg - f) / 2.5)
            shake[seg, 0] += a * (hash01(f, 1) - 0.5) * 2
            shake[seg, 2] += a * (hash01(f, 2) - 0.5) * 2
        for f in S.kick_f:
            if S.hot[f]:
                seg = np.arange(f, min(n, f + 6))
                shake[seg, 2] -= 0.008 * np.exp(-(seg - f) / 2.0)
        tgt = tgt + shake
        self.cam, self.tgt, self.lens, self.fstop = cam, tgt, lens, fst
        self.focus = np.linalg.norm(gsmooth(tgt, 1) - cam, axis=1)
        cuts = np.zeros(n, bool)
        for f, nm in ded[1:]:
            cuts[f - 1] = True
        self.cut_before = cuts           # True on the last frame BEFORE a cut (hold keys)
        self.log.append(f"shots {len(ded)}")

    # -------------------------------------------------------------------------------
    def _lights(self):
        """Neon / lanterns / blackout / lightning curves (0..~3)."""
        S, n, E = self.S, self.n, self.E
        idx = np.arange(n)
        neon = np.full(n, 0.55)
        # warm up through the intro, swells on every "close", blaze on "glows"
        neon = neon + 0.25 * np.clip((idx - 120) / 600, 0, 1)
        swell = np.zeros(n)
        for f in self.closes:
            seg = np.arange(max(0, f - 3), min(n, f + 40))
            t = seg - f
            swell[seg] = np.maximum(swell[seg], np.where(t < 0, 1 + t / 3, np.exp(-t / 14)))
        neon = neon + 0.45 * swell
        neon = neon + 0.9 * S.pre                            # builds glow up
        g = self.w_glows
        gl = np.zeros(n)
        seg = np.arange(g - 6, min(n, E[0]))
        gl[seg] = np.clip((seg - (g - 6)) / 6, 0, 1)
        neon = neon + 1.4 * gl
        hot = S.hot.astype(float)
        neon = np.where(S.hot, 1.0 + 0.45 * S.BASSN + 0.25 * S.KICK, neon)
        big = S.big.astype(float)
        neon = neon + big * (1.1 + 0.4 * S.BASSN)             # "the whole room glows"
        # calm gaps after drops: dim and moody
        for s0, s1, h in S.segs:
            if not h and s0 > E[0]:
                seg = np.arange(s0, s1)
                base = 0.5 if s1 - s0 > 100 else 0.8
                neon[seg] = base + 0.9 * S.pre[seg]
        # outro: fade down slowly, lanterns stay
        fo = [s0 for s0, s1, h in S.segs if s0 > E[6] and not h][0]
        neon[fo:] = np.interp(idx[fo:], [fo, fo + 60, n - 1], [1.0, 0.7, 0.18])
        lantern = np.clip(0.75 + 0.25 * swell + 0.35 * S.pre + 0.1 * big, 0, 2)
        lantern[fo:] = np.interp(idx[fo:], [fo, n - 1], [1.0, 0.55])
        # blackout half-beats
        blk = S.black.astype(float)
        neon = neon * (1 - blk)
        lantern = lantern * (1 - 0.92 * blk)
        # neon flicker: the bass in the drops, plus a dying tube that stutters
        rng = np.random.default_rng(7)
        stutter = (rng.random(n) < 0.05).astype(float) * (rng.random(n) * 0.8)
        self.neon = np.clip(neon, 0, 4)
        self.neon_bad = np.clip(neon * (1 - stutter), 0, 4)
        self.lantern = lantern
        self.blackout = blk
        # lightning: drop entries + budgeted strobe hits + a few in the calm, glint at the end of black
        strikes = []
        for e in E:
            strikes.append((e, 1.0))
        for f in S.strobe_f:
            if f not in E:
                strikes.append((int(f), 0.7))
        for f in (95, 360, 2900, 4700, 5300, 5560):
            if f < n:
                strikes.append((f, 0.45))
        for s0, s1 in [(s0, s1) for s0, s1, h in S.segs]:
            pass
        flash = np.zeros(n)
        for f, a in strikes:
            # two-stroke flicker: main + re-strike
            for d, w in ((0, 1.0), (1, 0.35), (2, 0.8), (3, 0.25), (5, 0.45), (6, 0.15)):
                if f + d < n:
                    flash[f + d] = max(flash[f + d], a * w)
        # a faint glint just before the entries, inside the blackout
        for e in E:
            if e - 2 >= 0:
                flash[e - 2] = max(flash[e - 2], 0.18)
        self.flash = flash
        self.strikes = sorted(strikes)
        self.bolt_vis = (flash > 0.3).astype(float)
        # rain time: real time, with hit-stop slow motion after each drop entry
        rate = np.ones(n)
        for e in E:
            seg = np.arange(e, min(n, e + 14))
            rate[seg] = np.minimum(rate[seg], 0.08 + 0.92 * ((seg - e) / 14) ** 2)
        self.rain_time = np.cumsum(rate) / S.fps
        # glints on the blades: hamon edge glow
        edge = 0.08 + 0.25 * hot + 0.8 * big
        snare_glow = np.zeros(n)
        for f, c, nn, st, kind, gap in self.sparks:
            seg = np.arange(f, min(n, f + 12))
            snare_glow[seg] = np.maximum(snare_glow[seg], st * np.exp(-(seg - f) / 3.0))
        self.blade_glow = np.clip(edge + 2.5 * snare_glow, 0, 4) * (1 - 0.7 * blk)
        self.log.append(f"lightning strikes {len(strikes)}")


def _closest_seg_seg(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-9 else 0.0
    t = np.clip((b * s + f) / e, 0, 1)
    s = np.clip((b * t - c) / a, 0, 1)
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return (c1 + c2) / 2, float(np.linalg.norm(c1 - c2))


def plan(timeline=None, words=None):
    timeline = timeline or os.path.join(SONG_DIR, "timeline_30fps.npz")
    words = words or os.path.join(SONG_DIR, "words.json")
    return Duel(Song(timeline, words))


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = plan()
    print("\n".join(d.log))
    print(f"planned in {time.time() - t0:.1f}s; entries {d.E}")
    print("clashes:", [(f, k, a) for f, k, a in d.clashes[:14]], "...")
    print("shots:", d.shots[:20], "...")
    for i in (0, 1):
        dd = [np.linalg.norm(d.wrist[i][0][f] - d.wrist[i][1][f]) for f in (700, 1319)]
        print("fighter", i, "pelvis range", d.pelvis[i].min().round(3), d.pelvis[i].max().round(3),
              "wrist sep", np.round(dd, 3))
    print("min root distance during fight", np.abs(d.D[d.E[0]:d.E[6] - 60]).min().round(2))
