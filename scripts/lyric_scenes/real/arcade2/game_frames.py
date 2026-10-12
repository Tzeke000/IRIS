"""HIT A BUMP — the original 8-bit game that plays on the hero cabinet's CRT.

    ~/venvs/iris-v100/bin/python scripts/lyric_scenes/real/arcade/game_frames.py \
        --timeline state/tzeke_songs/hit_a_bump/timeline_30fps.npz \
        --logo state/tzeke_songs/_assets/tzeke000_symbol.png \
        --out assets/scenes/arcade [--frames 1-3570 | 300,987,3200] [--workers 16]

Writes (scene frame f = timeline index f-1, files numbered by SCENE frame):
  <out>/game/g_%04d.png      1024x896  hero CRT (256x224 native, 4x, baked scanlines + beam glow)
  <out>/attract/a_%04d.png    512x448  attract-mode screen for the other cabinets
  <out>/screen_stats.npz      per-frame mean LINEAR rgb of both (drives the screen lights in 3-D)
  (screen_stats is only written by a full run; a --frames subset is for tests)

Everything is a pure function of the frame index + schedules precomputed once from
the timeline, so chunks can be generated in any order by any worker (deterministic).

The game: a little Spartan-helmeted runner (the Tzeke000 emblem's helmet) runs through
an 8-bit night. Jumps peak ON the beat; on every SNARE he head-bumps a BUMP block that
shatters; the SCORE ticks on every KICK. Build (T_pre) = WARNING / BOSS APPROACHING.
Half-beat blackout = TV static. Drop entries = white flash + FEVER palette, a drone wave
whose ships explode on kicks (his bolt is fired 6 frames BEFORE the kick so the hit lands
ON it) and a mini-boss. BIG drop = the boss fight: jaw opens and spits a fireball that
reaches him exactly when he is at the top of a snare-jump; his bolts hit on kicks; the
health bar drains to zero on the last kick; STAGE CLEAR.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H = 256, 224
UP = 4                      # hero CRT texture = 1024x896
HERO_X = 56
GROUND = 184                # y of the ground top
GAME_START = 296            # title -> stage card -> gameplay (timeline index)
TITLE_END = 248

# ------------------------------------------------------------------ font (5x7)
_F = {
 "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
 "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
 "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
 "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
 "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
 "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
 "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"],
 "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
 "I": [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
 "J": ["..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##.."],
 "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
 "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
 "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
 "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
 "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
 "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
 "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
 "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
 "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
 "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
 "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
 "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
 "W": ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."],
 "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
 "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
 "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
 "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
 "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
 "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
 "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
 "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
 "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
 "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
 "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
 "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
 "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
 "!": ["..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."],
 "?": [".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."],
 "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
 "+": [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."],
 ":": [".....", "..#..", ".....", ".....", ".....", "..#..", "....."],
 ".": [".....", ".....", ".....", ".....", ".....", ".....", "..#.."],
 "/": ["....#", "...#.", "...#.", "..#..", ".#...", ".#...", "#...."],
 "'": ["..#..", "..#..", ".#...", ".....", ".....", ".....", "....."],
 "(": ["...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."],
 ")": [".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."],
 " ": ["....."] * 7,
}
FONT = {k: np.array([[c == "#" for c in r] for r in v], bool) for k, v in _F.items()}


def text_w(s, sc=1):
    return len(s) * 6 * sc - sc


def draw_text(img, x, y, s, col, sc=1, shadow=(0, 0, 0), bold=False):
    """Pixel text. x may be 'c' (centred)."""
    if x == "c":
        x = (W - text_w(s, sc)) // 2
    for pass_ in ((1, shadow), (0, col)) if shadow is not None else ((0, col),):
        off, c = pass_
        cx = x + off * sc
        for ch in s.upper():
            g = FONT.get(ch, FONT["?"])
            if sc > 1:
                g = np.kron(g, np.ones((sc, sc), bool))
            if bold:
                g = g | np.pad(g, ((0, 0), (1, 0)))[:, :-1]
            _stamp(img, cx, y + off * sc, g, c)
            cx += 6 * sc


def _stamp(img, x, y, mask, col):
    h, w = mask.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    img[y0:y1, x0:x1][m] = col


def blit(img, x, y, spr, flash=None):
    """spr: (h,w,4) uint8 RGBA (alpha 0/255)."""
    x, y = int(round(x)), int(round(y))
    h, w = spr.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[..., 3] > 0
    if flash is not None:
        img[y0:y1, x0:x1][m] = flash
    else:
        img[y0:y1, x0:x1][m] = s[..., :3][m]


def rect(img, x0, y0, x1, y1, col):
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), W), min(int(y1), H)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = col


# ------------------------------------------------------------------ sprites
PAL = {
    ".": None, "k": (12, 10, 22), "r": (228, 36, 40), "R": (150, 16, 28), "S": (236, 240, 248),
    "s": (170, 190, 214), "b": (96, 118, 150), "d": (30, 26, 52), "f": (250, 188, 140),
    "c": (200, 40, 56), "g": (255, 206, 64), "l": (120, 72, 40), "w": (255, 255, 255),
    "y": (255, 236, 120), "o": (240, 140, 30), "n": (130, 64, 16), "p": (150, 70, 220),
    "P": (90, 30, 150), "e": (255, 60, 60), "t": (60, 230, 220),
}
_HERO_TOP = [
    "....rrrrrr......",
    "..rrrrrrrrrr....",
    ".rrR.....rrrr...",
    ".r....kkkkkk....",
    ".....kSSSSSSk...",
    "....kSSSSSSSSk..",
    "....kSSSSSkddk..",
    "....kbSSSkdfdk..",
    ".....kbSSkddk...",
    "......kkkkkk....",
    ".....kccccck....",
    "....kcccccclk...",
    "....kcgggggk....",
]
_LEGS = {
    "a": [".....kcccck.....", "....kll..llk....", "...kkk....kkk..."],
    "b": [".....kcccck.....", "......kllk......", ".....kkkkk......"],
    "c": [".....kcccck.....", "...kll...kll....", "...kk.....kkk..."],
    "j": ["....kcccccck....", "...kll...kll....", "................"],
    "s": ["....kcccccck....", "...kllllllllk...", "..kkkk....kkkk.."],   # landing squash
}


def _spr(rows):
    h, w = len(rows), max(len(r) for r in rows)
    a = np.zeros((h, w, 4), np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            c = PAL.get(ch)
            if c is not None:
                a[y, x, :3] = c
                a[y, x, 3] = 255
    return a


HERO = {k: _spr(_HERO_TOP + v) for k, v in _LEGS.items()}
HERO["crouch"] = _spr(_HERO_TOP[:9] + _HERO_TOP[10:12] + _LEGS["s"][:2] + [_LEGS["s"][2]])
HERO = {k: v.repeat(2, 0).repeat(2, 1) for k, v in HERO.items()}      # 2x: reads inside a 3-D shot


def _outline(rgba):
    a = rgba[..., 3] > 0
    p = np.pad(a, 1)
    ring = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~a
    out = rgba.copy()
    out[ring] = (12, 10, 22, 255)
    return out


def _disc(n, cx, cy, r):
    yy, xx = np.mgrid[0:n[0], 0:n[1]]
    return (xx - cx + .5) ** 2 + (yy - cy + .5) ** 2 <= r * r


def make_block(hue=0):
    b = np.zeros((16, 16, 4), np.uint8)
    base = np.array([(240, 140, 30), (40, 200, 230), (230, 60, 200)][hue], np.uint8)
    b[..., :3] = base
    b[..., 3] = 255
    b[0, :, :3] = b[:, 0, :3] = np.minimum(base.astype(int) + 90, 255)
    b[1, 1:15, :3] = b[1:15, 1, :3] = np.minimum(base.astype(int) + 50, 255)
    b[15, :, :3] = b[:, 15, :3] = base // 3
    b[14, 1:15, :3] = b[1:15, 14, :3] = base // 2
    for (x, y) in ((3, 3), (12, 3), (3, 12), (12, 12)):
        b[y, x, :3] = (30, 20, 20)
    g = FONT["B"]
    m = np.zeros((16, 16), bool)
    m[4:11, 6:11] = g
    sh = np.zeros_like(m)
    sh[5:12, 7:12] = g
    b[sh & ~m, :3] = base // 3
    b[m, :3] = (255, 255, 240)
    return b


BLOCKS = [make_block(i).repeat(2, 0).repeat(2, 1) for i in range(3)]   # 32x32


def make_drone(flap, col=(150, 70, 220), dark=(80, 30, 140), scale=1):
    s = np.zeros((14, 18, 4), np.uint8)
    body = _disc((14, 18), 9, 7, 5.2)
    s[body] = (*col, 255)
    s[_disc((14, 18), 8, 5, 2.2) & body] = (*np.minimum(np.array(col) + 70, 255), 255)
    eye = _disc((14, 18), 7, 7, 2.6)
    s[eye] = (255, 255, 255, 255)
    s[_disc((14, 18), 6, 7, 1.3)] = (230, 30, 40, 255)
    yy, xx = np.mgrid[0:14, 0:18]
    if flap:
        wing = ((xx < 5) & (yy < 7) & (yy > xx * 0.6)) | ((xx > 12) & (yy < 7) & (yy > (17 - xx) * 0.6))
    else:
        wing = ((xx < 5) & (yy > 6) & (yy < 7 + (5 - xx))) | ((xx > 12) & (yy > 6) & (yy < 7 + (xx - 12)))
    s[wing & ~body] = (*dark, 255)
    s = _outline(np.pad(s, ((1, 1), (1, 1), (0, 0))))
    if scale > 1:
        s = s.repeat(scale, 0).repeat(scale, 1)
    return s


DRONE = [make_drone(0, scale=2), make_drone(1, scale=2)]
MINIBOSS = [make_drone(0, (230, 120, 30), (140, 60, 10), 3), make_drone(1, (230, 120, 30), (140, 60, 10), 3)]


def make_boss(jaw_open, glow):
    """~76x68 mech skull. jaw_open 0..1, glow 0..1 (eye brightness)."""
    h, w = 70, 78
    s = np.zeros((h, w, 4), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    steel, steel_hi, steel_dk = (88, 96, 120), (150, 160, 190), (44, 46, 66)
    head = ((xx - 39) / 30) ** 2 + ((yy - 28) / 25) ** 2 <= 1
    s[head] = (*steel, 255)
    s[head & (((xx - 32) / 22) ** 2 + ((yy - 20) / 14) ** 2 <= 1)] = (*steel_hi, 255)
    # horns
    for sx in (1, -1):
        cx = 39 + sx * 26
        horn = (np.abs(xx - cx - sx * (yy - 10) * -0.25) < (12 - yy) * 0.45) & (yy < 12) & (yy > -1)
        s[horn] = (220, 200, 160, 255)
    # brow ridge + plates
    s[(yy == 26) & head] = (*steel_dk, 255)
    s[(np.abs(xx - 39) < 1) & (yy < 24) & head] = (*steel_dk, 255)
    # eyes
    ec = np.array((255, 40 + 120 * glow, 30 + 40 * glow)).astype(np.uint8)
    for ex in (26, 52):
        sock = ((xx - ex) / 9) ** 2 + ((yy - 33) / 6) ** 2 <= 1
        s[sock] = (20, 10, 20, 255)
        s[((xx - ex) / (5 + 2 * glow)) ** 2 + ((yy - 33) / 3.2) ** 2 <= 1] = (*ec, 255)
        s[((xx - ex) / 2) ** 2 + ((yy - 33) / 1.5) ** 2 <= 1] = (255, 250, 220, 255)
    # nose
    s[(np.abs(xx - 39) < (yy - 38) * 0.6 + 1) & (yy >= 38) & (yy < 44)] = (20, 10, 20, 255)
    # jaw
    jy = int(48 + 12 * jaw_open)
    mouth_top = 47
    s[(yy >= mouth_top) & (yy < jy) & (np.abs(xx - 39) < 20)] = (60, 8, 16, 255)
    if jaw_open > 0.2:
        s[(yy >= mouth_top + 1) & (yy < jy - 1) & (np.abs(xx - 39) < 16)] = (255, 120, 40, 255)
    for tx in range(22, 58, 6):                              # upper teeth
        s[(yy >= mouth_top) & (yy < mouth_top + 4) & (np.abs(xx - tx) < 2)] = (240, 236, 220, 255)
    jaw = (yy >= jy) & (yy < jy + 10) & (np.abs(xx - 39) < 22 - (yy - jy) * 0.6)
    s[jaw] = (*steel, 255)
    s[jaw & (yy == jy + 9)] = (*steel_dk, 255)
    for tx in range(25, 56, 6):                              # lower teeth
        s[(yy >= jy - 3) & (yy < jy) & (np.abs(xx - tx) < 2)] = (240, 236, 220, 255)
    return _outline(s)


# ------------------------------------------------------------------ helpers
def bayer4():
    m = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
    return np.tile(m, (H // 4, W // 4))


BAYER = bayer4()


def h01(*a):
    """Deterministic hash -> [0,1)."""
    x = 0x9E3779B1
    for v in a:
        x = (x ^ (int(v) & 0xFFFFFFFF)) * 0x85EBCA6B & 0xFFFFFFFF
        x ^= x >> 13
        x = x * 0xC2B2AE35 & 0xFFFFFFFF
        x ^= x >> 16
    return x / 4294967296.0


def gsmooth(x, s):
    k = np.exp(-0.5 * (np.arange(-3 * s, 3 * s + 1) / s) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, 3 * s, mode="edge"), k, "valid")


def age_since(mask):
    out = np.full(len(mask), 10 ** 6, np.int64)
    last = -10 ** 6
    for i, m in enumerate(mask):
        if m:
            last = i
        out[i] = i - last if last > -10 ** 6 else 10 ** 6
    return out


def sky_grad(top, bot, y0=0, y1=GROUND, bands=8):
    """8-bit banded gradient with Bayer dither between bands -> (H,W,3)."""
    t = np.clip((np.arange(H)[:, None] - y0) / max(y1 - y0, 1), 0, 1) * (bands - 1)
    t = np.repeat(t, W, 1)
    q = np.floor(t + BAYER).clip(0, bands - 1) / (bands - 1)
    top, bot = np.array(top, float), np.array(bot, float)
    return (top + (bot - top) * q[..., None]).astype(np.uint8)


# ------------------------------------------------------------------ the schedule (global, built once)
class Sched:
    pass


S = Sched()


def build_schedule(tl_path, logo_path):
    d = np.load(tl_path)
    n = len(d["rms"])
    S.n = n
    S.bass, S.high, S.rms = d["bass"].astype(float), d["high"].astype(float), d["rms"].astype(float)
    S.kick, S.snare = d["kick"].astype(bool), d["snare"].astype(bool)
    S.hot, S.big = d["T_hot"].astype(bool), d["T_big"].astype(bool)
    S.pre, S.black = d["T_pre"].astype(float), d["T_black"].astype(bool)
    S.kage, S.sage = d["T_kage"], d["T_sage"]
    S.ph, S.bi = d["T_ph"].astype(float), d["T_bi"]
    S.bars = d["bars"].astype(float)
    S.entries = [int(e) for e in d["entries"]]
    S.segs = [tuple(int(v) for v in s) for s in d["segs"]]
    S.seg_id = d["T_seg_id"]
    S.entry_age = d["T_entry_age"]
    S.strobe_age = d["T_strobe_age"]
    S.sub = gsmooth(S.bass, 3)
    hat = np.clip((S.high - gsmooth(S.high, 8)) * 5, 0, 1)
    S.hat = hat
    S.big_rng = next((s0, s1) for s0, s1, h in S.segs if h and S.big[s0])
    S.end_clear = S.big_rng[1]
    play = np.zeros(n, bool)
    play[GAME_START:S.end_clear] = True
    S.play = play
    # --- scroll speed + distance
    sp = np.where(S.hot, 3.0, 1.4)
    sp[S.big] = 0.0                                    # boss arena is static
    sp = sp * (1 - 0.6 * S.pre)
    sp[~play] = 0
    S.D = np.cumsum(gsmooth(sp, 4))
    # --- beats (gap-filled index changes)
    beat_frames = [i for i in range(1, n) if S.bi[i] != S.bi[i - 1]]
    S.beat_frames = beat_frames
    # --- jump apexes: snares (smash) first, spacing >= 8; then beats where room
    sn = [i for i in np.nonzero(S.snare)[0] if play[i] and not S.black[i] and S.pre[i] < 0.75]
    apex, smash = [], set()
    for s in sn:
        if not apex or s - apex[-1] >= 8:
            apex.append(int(s))
            smash.add(int(s))
    S.coin_pops = [int(s) for s in sn if int(s) not in smash]     # rolls -> coins
    calm_every = 2
    for j, b in enumerate(beat_frames):
        if not play[b] or S.black[b] or S.big[b]:
            continue
        if not S.hot[b] and j % calm_every:
            continue
        i = np.searchsorted(apex, b)
        lo = apex[i - 1] if i > 0 else -99
        hi = apex[i] if i < len(apex) else 10 ** 9
        if b - lo >= 12 and hi - b >= 12:
            apex.insert(i, b)
    S.apex = np.array(apex, np.int64)
    S.smash = smash
    # half widths + heights
    hw, hh = [], []
    for i, a in enumerate(apex):
        lo = apex[i - 1] if i > 0 else -99
        hi = apex[i + 1] if i + 1 < len(apex) else 10 ** 9
        w = int(min(7, (a - lo) // 2, (hi - a) // 2))
        hw.append(w)
        hh.append(10 + w * 4.4)
    S.hw, S.hh = np.array(hw), np.array(hh)
    # hero y offset per frame + pose
    yoff = np.zeros(n)
    pose = np.array(["run"] * n, object)
    for a, w, h in zip(S.apex, S.hw, S.hh):
        for f in range(a - w, a + w + 1):
            if 0 <= f < n:
                u = (f - a) / max(w, 1)
                yoff[f] = max(yoff[f], h * (1 - u * u))
                pose[f] = "jump"
        if 0 <= a - w - 1 < n and pose[a - w - 1] == "run":
            pose[a - w - 1] = "crouch"             # anticipation
        if 0 <= a + w + 1 < n and pose[a + w + 1] == "run":
            pose[a + w + 1] = "land"               # squash on landing
    S.yoff, S.pose = yoff, pose
    # --- blocks: one per smash apex, world x under the hero at the apex
    S.blocks = []
    for a in sorted(smash):
        i = int(np.searchsorted(S.apex, a))
        h = S.hh[i]
        wx = S.D[a] + HERO_X
        if S.big[a]:
            continue
        if S.blocks and wx - S.blocks[-1]["wx"] < 36:          # too tight: a coin pops instead
            S.coin_pops.append(a)
            continue
        S.blocks.append(dict(f=a, wx=wx, y=GROUND - 32 - h - 32, hue=int(h01(a, 7) * 3)))
    # --- score: kicks +50, smashes +100, drone kills +200, boss hits +500, clear +10000
    pts = np.zeros(n)
    for k in np.nonzero(S.kick)[0]:
        if play[k]:
            pts[k] += 50
    for a in smash:
        pts[a] += 100
    S.kills = [int(k) for k in np.nonzero(S.kick)[0] if S.hot[k] and not S.big[k] and play[k]]
    S.boss_hits = [int(k) for k in np.nonzero(S.kick)[0] if S.big[k] and k < S.end_clear - 2]
    for k in S.kills:
        pts[k] += 200
    for k in S.boss_hits:
        pts[k] += 500
    S.boss_die = S.boss_hits[-1] if S.boss_hits else S.end_clear - 12
    pts[min(S.boss_die + 10, n - 1)] += 10000
    S.score = np.cumsum(pts)
    S.score_ev = np.nonzero(pts)[0]
    S.kick_play = [int(k) for k in np.nonzero(S.kick)[0] if play[k]]
    # boss fireballs: one per smash-apex snare inside the big drop (reaches hero at apex)
    S.fireballs = [a for a in sorted(smash) if S.big[a] and a - 20 > S.big_rng[0] + 16]
    # big-drop jumps: in the arena we still jump on snares (dodging the fireballs)
    # --- emblem as pixel art
    S.emblem = pixel_emblem(logo_path, 44)
    S.emblem_small = pixel_emblem(logo_path, 22)
    return S


def pixel_emblem(path, size):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    white = (a.min(-1) > 225)
    # flood-fill background white from the border (the helmet's light panels are enclosed)
    from collections import deque
    hh, ww = white.shape
    bg = np.zeros_like(white)
    q = deque([(0, 0), (0, ww - 1), (hh - 1, 0), (hh - 1, ww - 1)])
    while q:
        y, x = q.popleft()
        if 0 <= y < hh and 0 <= x < ww and white[y, x] and not bg[y, x]:
            bg[y, x] = True
            q.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
    rgba = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    small = Image.fromarray(rgba, "RGBA").resize((size, size), Image.BOX)
    s = np.asarray(small).astype(int)
    pal = np.array([(12, 10, 22), (228, 36, 40), (150, 16, 28), (214, 232, 242), (150, 180, 205), (255, 255, 255)])
    rgb = s[..., :3]
    idx = np.argmin(((rgb[..., None, :] - pal[None, None]) ** 2).sum(-1), -1)
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = pal[idx]
    out[..., 3] = np.where(s[..., 3] > 110, 255, 0)
    return out


# ------------------------------------------------------------------ themes
THEMES = {
    "night": dict(top=(10, 12, 48), bot=(70, 40, 120), city=(26, 22, 70), win=(255, 210, 90),
                  hill=(20, 50, 60), g1=(150, 80, 40), g2=(100, 50, 26), edge=(80, 200, 90)),
    "fever": dict(top=(40, 4, 60), bot=(255, 70, 120), city=(60, 10, 70), win=(80, 255, 240),
                  hill=(110, 20, 90), g1=(60, 30, 110), g2=(36, 16, 70), edge=(255, 60, 200)),
    "boss":  dict(top=(20, 0, 0), bot=(150, 20, 30), city=(40, 6, 10), win=(255, 80, 40),
                  hill=(60, 10, 16), g1=(70, 70, 86), g2=(44, 44, 58), edge=(255, 190, 40)),
}
_STAGE_HUE = [0, 25, -30, 50, -55]


def hue_rot(c, deg):
    c = np.array(c, float) / 255
    a = np.deg2rad(deg)
    k = 1 / np.sqrt(3)
    cos, sin = np.cos(a), np.sin(a)
    m = cos * np.eye(3) + sin * np.array([[0, -k, k], [k, 0, -k], [-k, k, 0]]) + (1 - cos) * k * k * np.ones((3, 3))
    return tuple(int(v) for v in np.clip(m @ c, 0, 1) * 255)


def theme_at(i):
    if S.big[i]:
        return THEMES["boss"]
    seg = int(S.seg_id[i])
    base = THEMES["fever" if S.hot[i] else "night"]
    deg = _STAGE_HUE[(seg // 2) % len(_STAGE_HUE)]
    return {k: hue_rot(v, deg) if k not in ("win",) else v for k, v in base.items()}


def stage_label(i):
    seg = int(S.seg_id[i])
    return f"STAGE {seg // 2 + 1}-{1 + (1 if S.hot[i] else 0)}"


# ------------------------------------------------------------------ layers
def draw_sky(img, i, th):
    img[:] = sky_grad(th["top"], th["bot"])
    if S.pre[i] > 0:                                   # build: the sky bleeds red
        p = S.pre[i]
        flash = 0.5 + 0.5 * (S.ph[i] < 0.5 if p > 0.5 else 0.5)
        img[:] = (img * (1 - 0.55 * p) + np.array((200, 10, 20)) * 0.55 * p * flash).astype(np.uint8)
    # stars (twinkle on hats)
    rng = np.random.default_rng(11)
    sx, sy = rng.integers(0, W, 70), rng.integers(18, 120, 70)
    tw = rng.random(70)
    par = (S.D[i] * 0.08) % W
    for x, y, t in zip(sx, sy, tw):
        b = 140 + 115 * (S.hat[i] if t > 0.6 else 0.3)
        xx = int((x - par) % W)
        img[y, xx] = (b, b, min(255, b + 20))
        if t > 0.85 and S.hat[i] > 0.4:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= xx + dx < W:
                    img[y + dy, xx + dx] = (b * .6, b * .6, b * .7)
    # moon / sun
    if S.hot[i] and not S.big[i]:                      # synth sun with stripes, pulses with sub
        r = 30 + 4 * S.sub[i]
        yy, xx = np.mgrid[0:H, 0:W]
        m = ((xx - 180) ** 2 + (yy - 112) ** 2 <= r * r) & (yy < GROUND - 30)
        stripe = ((yy - 80) % 8 < (yy - 80) / 12) & (yy > 100)
        sun = np.clip(np.array((255, 230, 90)) * (1 - (yy - 82) / 90)[..., None] + np.array((0, 0, 0)), 0, 255)
        img[m & ~stripe] = np.array((255, 220, 80)) * 0.5 + sun[m & ~stripe] * 0.5
    elif not S.big[i]:
        yy, xx = np.mgrid[0:56, 0:56]
        m = (xx - 28) ** 2 + (yy - 28) ** 2 <= 13 ** 2
        cut = (xx - 33) ** 2 + (yy - 24) ** 2 <= 11 ** 2
        sub = img[22:78, 172:228]
        sub[m & ~cut] = (250, 244, 210)


def draw_city(img, i, th):
    par = S.D[i] * 0.3
    rng = np.random.default_rng(5)
    widths = rng.integers(14, 34, 40)
    heights = rng.integers(30, 80, 40)
    x = -(par % widths.sum())
    k = 0
    base = GROUND - 22
    while x < W + 40:
        j = k % 40
        w_, h_ = widths[j], heights[j]
        rect(img, x, base - h_, x + w_ - 2, base + 1, th["city"])
        # windows (some flash on kicks)
        for wy in range(int(base - h_ + 4), base - 4, 6):
            for wx in range(int(x + 3), int(x + w_ - 5), 5):
                v = h01(j, wy, wx - int(x))
                on = v > 0.55 or (v > 0.35 and S.kage[i] < 4 and S.play[i])
                if on and 0 <= wx < W - 1:
                    c = th["win"] if v > 0.4 else tuple(int(c * .6) for c in th["win"])
                    img[wy:wy + 2, wx:wx + 2] = c
        x += w_
        k += 1


def draw_hills(img, i, th):
    par = S.D[i] * 0.55
    xs = np.arange(W) + par
    y = GROUND - 10 - (6 * np.sin(xs * 0.03) + 4 * np.sin(xs * 0.071 + 1)).astype(int)
    yy = np.arange(H)[:, None]
    m = (yy >= y[None, :]) & (yy < GROUND)
    img[m] = np.broadcast_to(np.array(th["hill"], np.uint8), img.shape)[m]


def draw_ground(img, i, th):
    off = int(S.D[i]) % 16
    rect(img, 0, GROUND, W, H, th["g2"])
    for ty in range(GROUND, H, 8):
        row = (ty - GROUND) // 8
        sh = (8 if row % 2 else 0) - off
        for tx in range(sh - 16, W + 16, 16):
            rect(img, tx + 1, ty + 1, tx + 15, ty + 7, th["g1"])
    rect(img, 0, GROUND, W, GROUND + 2, th["edge"])
    if S.hot[i] and S.kage[i] < 5:                       # kick: the ground line flashes
        rect(img, 0, GROUND, W, GROUND + 2, (255, 255, 255))


def draw_arena(img, i, th):
    """Static boss arena (no scroll): red alarm sky, hazard floor, pillars."""
    img[:] = sky_grad(th["top"], th["bot"])
    # sweeping searchlights
    yy, xx = np.mgrid[0:GROUND, 0:W]
    for k, base in enumerate((40, 216)):
        ang = 0.5 * np.sin(i * 0.05 + k * 2.1)
        dx = (xx - base) - (GROUND - yy) * np.tan(ang)
        beam = np.abs(dx) < (GROUND - yy) * 0.12
        img[:GROUND][beam] = np.clip(img[:GROUND][beam].astype(int) + 40, 0, 255)
    # pillars
    for px in (8, 232):
        rect(img, px, 40, px + 16, GROUND, (50, 40, 50))
        for py in range(48, GROUND, 24):
            rect(img, px + 2, py, px + 14, py + 3, (255, 60, 40) if (S.kage[i] < 4) else (120, 30, 30))
    rect(img, 0, GROUND, W, H, th["g2"])
    for tx in range(0, W, 16):
        rect(img, tx + 1, GROUND + 3, tx + 15, H, th["g1"])
    hz = ((np.arange(W) + i * 2) // 6) % 2 == 0
    img[GROUND:GROUND + 3][:, hz] = (255, 200, 30)
    img[GROUND:GROUND + 3][:, ~hz] = (20, 20, 20)


def hero_sprite(i):
    p = S.pose[i]
    if p == "jump":
        return HERO["j"]
    if p == "crouch" or p == "land":
        return HERO["crouch"]
    if not S.play[i]:
        return HERO["b"]
    step = int(S.D[i] / 6) % 4
    return HERO[["a", "b", "c", "b"][step]]


def hero_pos(i):
    spr = hero_sprite(i)
    y = GROUND - spr.shape[0] - S.yoff[i]
    bob = 1 if (S.pose[i] == "run" and int(S.D[i] / 6) % 2) else 0
    return HERO_X, y - bob, spr


def draw_blocks(img, i):
    for b in S.blocks:
        sx = b["wx"] - S.D[i]
        if sx < -40 or sx > W + 20:
            continue
        age = i - b["f"]
        if age < 0:
            bump = 0
            blit(img, sx, b["y"], BLOCKS[b["hue"]])
        elif age < 26:
            # shatter: 4 pieces, gravity, + coin + score popup
            spr = BLOCKS[b["hue"]]
            for q, (vx, vy) in enumerate(((-1.8, -4.6), (1.8, -4.6), (-1.2, -2.8), (1.2, -2.8))):
                px = sx + (q % 2) * 16 + vx * age
                py = b["y"] + (q // 2) * 16 + vy * age + 0.42 * age * age
                piece = spr[(q // 2) * 16:(q // 2) * 16 + 16, (q % 2) * 16:(q % 2) * 16 + 16]
                blit(img, px, py, piece)
            if age < 3:                                             # impact flash
                rect(img, sx - 3, b["y"] - 3, sx + 35, b["y"] + 35, (255, 255, 230))
            if age < 18:
                draw_text(img, int(sx + 4), int(b["y"] - 12 - age * 1.2), "100", (255, 255, 120))
            # coin spinning up
            cy = b["y"] - 10 - 3.4 * age + 0.12 * age * age
            if age < 18:
                cw = 1 + int(4 * abs(np.cos(age * 0.6)))
                rect(img, sx + 16 - cw, cy, sx + 16 + cw, cy + 12, (255, 210, 40))
                rect(img, sx + 16 - max(cw - 2, 0), cy + 2, sx + 16 + max(cw - 2, 0), cy + 10, (255, 250, 160))


def draw_coin_pops(img, i):
    for c in S.coin_pops:
        age = i - c
        if 0 <= age < 14:
            cy = GROUND - 80 - 3 * age + 0.15 * age * age
            cw = 1 + int(4 * abs(np.cos(age * 0.7)))
            rect(img, HERO_X + 16 - cw, cy, HERO_X + 16 + cw, cy + 12, (255, 210, 40))


def drone_path(k, f):
    ex = 140 + 80 * h01(k, 1)
    ey = 50 + 70 * h01(k, 2)
    x = ex + (k - f) * 1.9
    y = ey + 8 * np.sin(f * 0.14 + 6 * h01(k, 3))
    return x, y


def draw_bolt(img, x0, y0, x1, y1, u):
    x = x0 + (x1 - x0) * u
    y = y0 + (y1 - y0) * u
    for t in range(6):
        tx = x - (x1 - x0) * 0.025 * t
        ty = y - (y1 - y0) * 0.025 * t
        c = (120, 255, 250) if t < 2 else (40, 160, 220)
        rect(img, tx, ty, tx + 2, ty + 2, c)


def explosion(img, x, y, age, big=1.0, seed=0):
    if age < 0 or age > 16 * big:
        return
    r = (2 + age * 1.6) * big
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    ring = (d2 <= r * r) & (d2 >= (r * 0.65) ** 2)
    core = d2 <= (r * 0.65) ** 2
    t = age / (16 * big)
    img[core & (h01(seed, age) < 2)] = (255, 255, 200) if t < 0.3 else (255, 160, 40)
    img[ring] = (255, 90, 30) if t < 0.6 else (120, 40, 40)
    rng = np.random.default_rng(seed)
    for q in range(10):
        a = rng.random() * 6.283
        v = (1.5 + rng.random() * 2) * big
        px, py = x + np.cos(a) * v * age, y + np.sin(a) * v * age + 0.1 * age * age
        rect(img, px, py, px + 2, py + 2, (255, 230, 120) if q % 2 else (255, 120, 40))


def draw_drones(img, i, hx, hy):
    if not S.hot[i] or S.big[i]:
        return
    seg = next(s for s in S.segs if s[0] <= i < s[1])
    for k in S.kills:
        if not (seg[0] <= k < seg[1]):
            continue
        age = i - k
        if age < -80 or age > 18:
            continue
        x, y = drone_path(k, min(i, k))
        if age < 0:
            blit(img, x - 20, y - 16, DRONE[(i // 4) % 2])
            if -6 <= age < 0:                                      # bolt launched 6 frames before the kick
                bx0, by0 = HERO_X + 28, GROUND - 40 - S.yoff[k - 6]
                draw_bolt(img, bx0, by0, x, y, (age + 6) / 6)
        else:
            explosion(img, x, y, age, 1.0, seed=k)
            if age < 16:
                draw_text(img, int(x - 8), int(y - 16 - age), "200", (120, 255, 250))
    # mini-boss: descends at entry, bobs on sub, explodes at segment end
    s0, s1 = seg[0], seg[1]
    die = s1 - 16
    age = i - s0
    if i < die:
        y = -40 + min(age, 24) / 24 * 70 + 6 * S.sub[i]
        x = 186 + 10 * np.sin(i * 0.05)
        blit(img, x - 30, y - 24, MINIBOSS[(i // 5) % 2], flash=(255, 255, 255) if S.kage[i] < 2 else None)
    else:
        x, y = 186 + 10 * np.sin(die * 0.05), 30 + 6 * S.sub[die]
        explosion(img, x, y, i - die, 1.8, seed=s1)
        if i - die < 16:
            draw_text(img, "c", 64, "WAVE CLEAR", (255, 255, 120), 2)


def draw_boss(img, i):
    b0, _ = S.big_rng
    age = i - b0
    # entry: slams down from above in 10 frames
    y = -80 + min(age, 10) / 10 * 104 if age < 10 else 24 + 3 * np.sin(i * 0.11) + 4 * S.sub[i]
    x = 156
    dead = i >= S.boss_die
    if dead:
        da = i - S.boss_die
        for q in range(6):
            explosion(img, x + 10 + 60 * h01(q, 1), y + 10 + 50 * h01(q, 2), da - q * 3, 1.4, seed=q + 99)
        return x, y
    # jaw opens ahead of each fireball (anticipation), stays open briefly after
    jaw = 0.0
    for a in S.fireballs:
        t = i - (a - 20)
        if -6 <= t < 0:
            jaw = max(jaw, (t + 6) / 6)
        elif 0 <= t < 5:
            jaw = max(jaw, 1 - t / 5)
    hit = any(0 <= i - k < 3 for k in S.boss_hits)
    spr = make_boss(jaw, min(1, 0.4 + S.sub[i]))
    blit(img, x, y, spr, flash=(255, 255, 255) if hit else None)
    return x, y


def draw_fireballs(img, i, bx, by):
    for a in S.fireballs:
        t = i - (a - 20)
        if 0 <= t < 40:
            x0, y0 = bx + 39, by + 54
            x1, y1 = HERO_X + 16, GROUND - 8                        # passes UNDER him at the apex
            u = t / 20
            x = x0 + (x1 - x0) * u
            y = y0 + (y1 - y0) * min(u, 1)
            r = 5 + (t % 4 < 2)
            yy, xx = np.mgrid[0:H, 0:W]
            m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
            img[m] = (255, 140, 40)
            img[(xx - x) ** 2 + (yy - y) ** 2 <= (r - 2) ** 2] = (255, 250, 180)
            for q in range(4):                                       # trail
                rect(img, x + 6 + q * 5, y - 1 + (q % 2), x + 9 + q * 5, y + 2, (255, 90 - q * 15, 30))


def draw_boss_bolts(img, i, bx, by):
    for k in S.boss_hits:
        age = i - k
        if -6 <= age < 0:
            bx0, by0 = HERO_X + 28, GROUND - 40 - S.yoff[k - 6]
            draw_bolt(img, bx0, by0, bx + 26, by + 33, (age + 6) / 6)
        elif 0 <= age < 12:
            explosion(img, bx + 26 + 26 * h01(k, 5), by + 30 + 10 * h01(k, 6), age, 0.6, seed=k)
            draw_text(img, int(bx + 10), int(by - 4 - age), "500", (255, 255, 255))


def draw_hud(img, i):
    rect(img, 0, 0, W, 18, (0, 0, 0))
    # rolling score: ticks up over 4 frames after each event
    j = np.searchsorted(S.score_ev, i, side="right") - 1
    sc = S.score[i]
    if j >= 0 and i - S.score_ev[j] < 4:
        prev = S.score[S.score_ev[j] - 1] if S.score_ev[j] > 0 else 0
        sc = prev + (sc - prev) * (i - S.score_ev[j] + 1) / 4
    hi = max(50000, sc)
    fl = S.kage[i] < 3 and S.play[i]
    draw_text(img, 6, 2, "1UP", (255, 60, 60))
    draw_text(img, 6, 10, f"{int(sc):06d}", (255, 255, 255) if not fl else (255, 255, 120))
    draw_text(img, 96, 2, "HI SCORE", (255, 60, 60))
    draw_text(img, 102, 10, f"{int(hi):06d}", (255, 255, 255))
    draw_text(img, 190, 2, stage_label(i), (120, 220, 255))
    for q in range(3):                                         # lives = little helmets
        blit(img, 192 + q * 12, 9, S.emblem_small[::2, ::2])
    if S.big[i] and i < S.boss_die + 8:
        b0, _ = S.big_rng
        hp = 1 - sum(1 for k in S.boss_hits if k <= i) / max(len(S.boss_hits), 1)
        draw_text(img, 6, 20, "BOSS", (255, 200, 40))
        rect(img, 32, 21, 32 + 120, 27, (60, 0, 0))
        rect(img, 32, 21, 32 + int(120 * hp), 27, (255, 40, 40) if S.kage[i] > 2 else (255, 255, 255))


def draw_warning(img, i):
    p = S.pre[i]
    if p <= 0:
        return
    # hazard bands slide in
    band = int(min(1, p * 3) * 20)
    for y0 in (40, 140):
        yy, xx = np.mgrid[0:band, 0:W]
        hz = ((xx + yy + i * 2) // 8) % 2 == 0
        sub = img[y0:y0 + band]
        sub[hz] = (230, 30, 30)
        sub[~hz] = (20, 0, 0)
    rate = 0.5 if p < 0.5 else 0.25                             # blink faster as it builds
    on = (S.ph[i] % rate) < rate * 0.6
    if on:
        draw_text(img, "c", 72, "WARNING!", (255, 50, 40), 4, bold=True)
    draw_text(img, "c", 116, "BOSS APPROACHING", (255, 220, 200) if on else (180, 60, 60), 1)


def draw_static(img, i):
    rng = np.random.default_rng(1000 + i)
    v = rng.integers(0, 256, (H, W // 2), dtype=np.int32)
    v = np.repeat(v, 2, 1)
    roll = (i * 37) % H
    v[max(0, roll - 14):roll] //= 3
    img[:] = np.stack([v, v, v], -1).astype(np.uint8)
    for _ in range(3):
        y = int(rng.integers(0, H))
        img[y] = (rng.integers(0, 255, 3)).astype(np.uint8)


def draw_title(img, i):
    img[:] = 0
    rng = np.random.default_rng(3)
    for q in range(90):
        x = int((rng.integers(0, W) - i * (0.3 + rng.random())) % W)
        y = int(rng.integers(0, H))
        b = int(80 + 100 * rng.random())
        img[y, x] = (b, b, b + 20)
    em = S.emblem
    blit(img, (W - em.shape[1]) // 2, 22, em)
    # title, colour cycling top->bottom, bob on the beat
    bob = int(2 * max(0, 1 - S.ph[i] * 4))
    cols = [(255, 220, 60), (255, 140, 40), (255, 60, 60)]
    t = "HIT A BUMP"
    x0 = (W - text_w(t, 3)) // 2
    for k, ch in enumerate(t):
        c = cols[(k + i // 6) % 3]
        draw_text(img, x0 + k * 18, 76 - bob * (k % 2), ch, c, 3, shadow=(80, 0, 0), bold=True)
    blink = S.ph[i] < 0.55
    if i < 200:
        if blink:
            draw_text(img, "c", 124, "INSERT COIN", (255, 255, 255))
        draw_text(img, "c", 144, "CREDIT 00", (140, 140, 160))
    else:
        if (i // 4) % 2:
            draw_text(img, "c", 124, "PUSH START", (255, 255, 120))
        draw_text(img, "c", 144, "CREDIT 01", (255, 255, 255))
    draw_text(img, "c", 196, "(C) TZEKE000", (120, 200, 255))
    draw_text(img, "c", 208, "1 PLAYER  2 PLAYERS", (100, 100, 120))


def draw_stage_card(img, i):
    img[:] = 0
    draw_text(img, "c", 80, "STAGE 1", (255, 255, 255), 2)
    if (i // 6) % 2:
        draw_text(img, "c", 110, "READY!", (255, 220, 60), 2)
    blit(img, (W - 32) // 2, 140, HERO["b"])


def draw_outro(img, i):
    age = i - S.end_clear
    img[:] = 0
    draw_text(img, "c", 40, "STAGE CLEAR!", (255, 220, 60), 2, bold=True)
    if age > 18:
        draw_text(img, "c", 76, "THANK YOU", (255, 255, 255), 2)
        draw_text(img, "c", 96, "FOR PLAYING", (255, 255, 255), 2)
    em = S.emblem
    if age > 30:
        blit(img, (W - em.shape[1]) // 2, 124, em)
        draw_text(img, "c", 176, "TZEKE000", (120, 200, 255), 2)
    draw_text(img, "c", 210, f"SCORE {int(S.score[i]):06d}", (255, 120, 120))


def render_native(i):
    img = np.zeros((H, W, 3), np.uint8)
    if S.black[i]:
        draw_static(img, i)
        return img
    if i < TITLE_END:
        draw_title(img, i)
        return img
    if i < GAME_START:
        draw_stage_card(img, i)
        return img
    if i >= S.end_clear:
        draw_outro(img, i)
        return img
    th = theme_at(i)
    if S.big[i]:
        draw_arena(img, i, th)
        bx, by = draw_boss(img, i)
        draw_fireballs(img, i, bx, by)
        draw_boss_bolts(img, i, bx, by)
    else:
        draw_sky(img, i, th)
        draw_city(img, i, th)
        draw_hills(img, i, th)
        draw_ground(img, i, th)
        draw_blocks(img, i)
        draw_coin_pops(img, i)
    hx, hy, spr = hero_pos(i)
    draw_drones(img, i, hx, hy)
    blit(img, hx, hy, spr)
    draw_warning(img, i)
    # drop entry: white flash + FEVER! / BOSS FIGHT!
    ea = S.entry_age[i]
    if S.hot[i] and ea < 26:
        if ea == 0:
            img[:] = (255, 255, 255)
        elif ea < 5:                                  # fast decay: never two near-white frames in a row
            img[:] = (img.astype(int) + (255 - img.astype(int)) * (0.45 ** ea)).astype(np.uint8)
        sc = 4 if ea < 4 else 3 if ea < 8 else 2
        txt = "BOSS FIGHT!" if S.big[i] else "FEVER!"
        draw_text(img, "c", 64 - sc * 2, txt, (255, 255, 80) if ea % 4 < 2 else (255, 80, 200), sc, bold=True)
    if S.big[i] and S.boss_die <= i < S.boss_die + 2:
        img[:] = (255, 255, 255)
    draw_hud(img, i)
    # kick shake in drops
    if S.hot[i] and S.kage[i] < 4:
        a = (4 - S.kage[i]) * (1 if S.big[i] else 0.6)
        dx = int(round(a * (h01(i, 1) - 0.5) * 2))
        dy = int(round(a * (h01(i, 2) - 0.5) * 2))
        img = np.roll(img, (dy, dx), (0, 1))
    return img


# ------------------------------------------------------------------ attract screens (other cabinets)
def render_attract(i, variant):
    img = np.zeros((H, W, 3), np.uint8)
    bars = S.bars[i]
    if variant == 0:                                      # "STAR BEAT": equaliser + title
        img[:] = sky_grad((0, 0, 20), (30, 0, 50), 0, H, 6)
        n = 20
        bw = W // n
        for k in range(n):
            v = 0.5 * (bars[2 * k] + bars[2 * k + 1])
            h = int(8 + v * 120)
            for yb in range(0, h, 6):
                t = yb / 130
                c = (int(60 + 195 * t), int(255 - 160 * t), 120)
                rect(img, k * bw + 1, 200 - yb - 4, k * bw + bw - 1, 200 - yb, c)
        draw_text(img, "c", 20, "STAR BEAT", (255, 220, 60), 3, bold=True)
        if (i // 15) % 2:
            draw_text(img, "c", 210, "INSERT COIN", (255, 255, 255))
    else:                                                 # "NEON DRIFT": road + spectrum horizon
        img[:] = sky_grad((10, 0, 30), (200, 40, 120), 0, 120, 6)
        rect(img, 0, 120, W, H, (10, 0, 20))
        for k in range(40):
            v = bars[k]
            x = int(k * W / 40)
            rect(img, x, 120 - int(v * 50), x + 5, 120, (255, 120 + int(100 * v), 200))
        for z in range(12):                               # perspective grid
            y = 120 + int((((z + i * 0.15) % 12) / 12) ** 2 * 104)
            img[min(y, H - 1)] = (255, 40, 200)
        for gx in range(-8, 9):
            x0 = W // 2 + gx * 8
            x1 = W // 2 + gx * 60
            for t in np.linspace(0, 1, 52):
                yy = int(120 + t * 103)
                xx = int(x0 + (x1 - x0) * t)
                if 0 <= xx < W:
                    img[yy, xx] = (255, 40, 200)
        draw_text(img, "c", 30, "NEON DRIFT", (120, 255, 250), 3, bold=True)
        if (i // 15) % 2:
            draw_text(img, "c", 140, "INSERT COIN", (255, 255, 255))
    return img


# ------------------------------------------------------------------ CRT bake
def _blur1d(a, s, axis):
    r = int(3 * s)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ap = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for j, kv in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(j, j + a.shape[axis])
        out += kv * ap[tuple(sl)]
    return out


_MASKS = {}


def _crt_mask(up):
    """Rounded tube corners + vignette. The 3-D CRT mesh is slightly larger than the bezel opening,
    (2.5% oversize) so the rounded corners sit just inside the visible area."""
    if up not in _MASKS:
        y = (np.arange(H * up) + 0.5) / (H * up) * 2 - 1
        x = (np.arange(W * up) + 0.5) / (W * up) * 2 - 1
        qx, qy = np.abs(x)[None, :], np.abs(y)[:, None]
        half, rad = 0.978, 0.12
        dx, dy = np.maximum(qx - (half - rad), 0), np.maximum(qy - (half - rad), 0)
        r = np.sqrt(dx * dx + dy * dy)
        m = np.clip((rad - r) / 0.012, 0, 1)
        vig = (1 - 0.35 * qx ** 4) * (1 - 0.35 * qy ** 4)
        _MASKS[up] = (m * vig).astype(np.float32)[..., None]
    return _MASKS[up]


def crt_bake(img, up):
    """Native frame -> CRT texture: beam-softened, brightness-dependent scanlines, halation.
    Returns (uint8 sRGB image, mean LINEAR rgb of the native frame)."""
    lin = (img.astype(np.float32) / 255) ** 2.2
    big = cv2.resize(lin, (W * up, H * up), interpolation=cv2.INTER_NEAREST)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=0.45 * up, sigmaY=0.01)     # horizontal beam softness
    lum = big.max(-1, keepdims=True)
    sub = ((np.arange(H * up) % up + 0.5) / up - 0.5).astype(np.float32)[:, None, None]
    sigma = 0.20 + 0.16 * np.sqrt(lum)                                    # beam widens with brightness
    out = big * np.exp(-0.5 * (sub / sigma) ** 2) * 1.25
    glow = cv2.GaussianBlur(lin, (0, 0), 2.2)                              # halation, native res
    out += cv2.resize(glow, (W * up, H * up), interpolation=cv2.INTER_LINEAR) * 0.22
    out *= _crt_mask(up)
    return (np.clip(out, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8), lin.reshape(-1, 3).mean(0)


# ------------------------------------------------------------------ driver
_ARGS = None


def _init(args):
    global _ARGS
    _ARGS = args
    build_schedule(args.timeline, args.logo)


def _work(f):
    i = f - 1
    a = _ARGS
    g = render_native(i)
    gb, gm = crt_bake(g, UP)
    Image.fromarray(gb).save(a.out / "game" / f"g_{f:04d}.png", compress_level=3)
    at0 = render_attract(i, 0)
    at1 = render_attract(i, 1)
    ab0, am0 = crt_bake(at0, 2)
    ab1, am1 = crt_bake(at1, 2)
    Image.fromarray(ab0).save(a.out / "attract" / f"a0_{f:04d}.png", compress_level=3)
    Image.fromarray(ab1).save(a.out / "attract" / f"a1_{f:04d}.png", compress_level=3)
    return f, gm, am0, am1


def parse_frames(spec, n):
    if not spec:
        return list(range(1, n + 1))
    out = []
    for ch in spec.split(","):
        if "-" in ch:
            lo, hi = ch.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(ch))
    return [f for f in out if 1 <= f <= n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", type=Path, default=Path("state/tzeke_songs/hit_a_bump/timeline_30fps.npz"))
    ap.add_argument("--logo", type=Path, default=Path("state/tzeke_songs/_assets/tzeke000_symbol.png"))
    ap.add_argument("--out", type=Path, default=Path("assets/scenes/arcade"))
    ap.add_argument("--frames", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    (args.out / "game").mkdir(parents=True, exist_ok=True)
    (args.out / "attract").mkdir(parents=True, exist_ok=True)
    n = len(np.load(args.timeline)["rms"])
    frames = parse_frames(args.frames, n)
    full = len(frames) == n
    gm = np.zeros((n, 3), np.float32)
    am = np.zeros((2, n, 3), np.float32)
    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        for k, (f, g, a0, a1) in enumerate(pool.imap_unordered(_work, frames, chunksize=8)):
            gm[f - 1], am[0, f - 1], am[1, f - 1] = g, a0, a1
            if k % 250 == 0:
                print(f"[game_frames] {k}/{len(frames)}", flush=True)
    if full:
        np.savez(args.out / "screen_stats.npz", game=gm, attract=am)
        print(f"[game_frames] wrote {args.out / 'screen_stats.npz'}")
    print(f"[game_frames] done: {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
