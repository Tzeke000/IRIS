"""Sands v2 — data for the six DIFFERENT cube faces (Zeke 10-30 v2 note 3). Pure numpy + one bpy image writer.

Face ids (cube._panel order; the snare flips are full 360°, so at rest the top stays the top):
  0 −Y  EMBLEM (classic LED logo)            1 +X  ALBUM COVER (fine-pitch display)
  2 +Y  EMBLEM over chase rings              3 −X  AUDIO VISUALIZER (LED spectrum bars + peak caps)
  4 +Z  EMBLEM over beat checker blocks      5 −Z  CIRCUIT CORE (kick rings + circuitry)
Emblem on 3 faces max (Zeke). The album image is PATH-driven: ALBUM if it exists, else the obvious placeholder —
swap the file + rebuild. The spectrum is baked per frame into a small atlas image (stateless, chunk-safe).
"""
import math
import os
import wave

import numpy as np

ALBUM = "/data/iris_assets/sands/album_cover.png"
ALBUM_PLACEHOLDER = "/data/iris_assets/sands/album_placeholder/album_cover_PLACEHOLDER.png"
FACE_NAMES = ["emblem", "album", "emblem_rings", "visualizer", "emblem_top", "circuit_core"]
FACE_NORMALS = np.array([(0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)], float)
N_BARS = 16
ATLAS_COLS = 64                    # frames per atlas row → image (N_BARS·64) × ceil(NF/64)


def album_path():
    return (ALBUM, False) if os.path.isfile(ALBUM) else (ALBUM_PLACEHOLDER, True)


# ------------------------------------------------------------------------------------------------ spectrum

def read_wav_mono(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, nfr = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(nfr)
    assert sw == 2, f"16-bit WAV expected, got {sw * 8}-bit"
    x = np.frombuffer(raw, np.int16).astype(np.float32).reshape(-1, ch).mean(1) / 32768.0
    return x, sr


def spectrum(path, n, fps=30, bars=N_BARS, f_lo=35.0, f_hi=14000.0, win=4096):
    """→ (level (n, bars), peak (n, bars)) in 0..1: log-spaced bands, dB-mapped, fast attack / slow release, peak-hold
    caps that fall. Computed for the whole song up front (stateful smoothing is fine — it is BAKED, not live)."""
    x, sr = read_wav_mono(path)
    hop = sr / fps
    edges = np.geomspace(f_lo, f_hi, bars + 1)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    band = np.digitize(freqs, edges) - 1                      # bin → bar (−1 / bars = outside)
    w = np.hanning(win).astype(np.float32)
    pad = np.r_[np.zeros(win // 2, np.float32), x, np.zeros(win, np.float32)]
    E = np.zeros((n, bars))
    for c0 in range(0, n, 512):
        idx = np.arange(c0, min(n, c0 + 512))
        st = (idx * hop).astype(np.int64)
        seg = pad[st[:, None] + np.arange(win)[None, :]] * w
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        for b in range(bars):
            m = band == b
            E[idx, b] = P[:, m].mean(1) if m.any() else 0.0
    db = 10 * np.log10(E + 1e-12)
    # per-bar normalisation against the song's own range (98th pct = full), with a spectral tilt so highs move too
    lo = np.percentile(db, 30, axis=0)
    hi = np.percentile(db, 99.2, axis=0)
    L = np.clip((db - lo) / np.maximum(hi - lo, 1e-6), 0, 1) ** 1.3
    lev = np.zeros_like(L)
    pk = np.zeros_like(L)
    cur = np.zeros(bars)
    peak = np.zeros(bars)
    hold = np.zeros(bars)
    for i in range(n):
        cur = np.where(L[i] > cur, L[i], cur * 0.80 + L[i] * 0.20 - 0.012)   # attack instant, release ~0.25 s
        cur = np.clip(cur, 0, 1)
        up = cur >= peak
        peak = np.where(up, cur, peak)
        hold = np.where(up, 12, hold - 1)                                     # cap hangs 0.4 s, then falls
        peak = np.where(hold < 0, np.maximum(cur, peak - 0.018), peak)
        lev[i], pk[i] = cur, peak
    return lev, pk


def atlas(lev, pk):
    """(n, bars) pair → RGBA float atlas (H, W, 4): pixel (row f // COLS, col (f % COLS)·bars + b) = (level, peak)"""
    n, bars = lev.shape
    rows = math.ceil(n / ATLAS_COLS)
    img = np.zeros((rows, ATLAS_COLS * bars, 4), np.float32)
    f = np.arange(n)
    r, c = f // ATLAS_COLS, (f % ATLAS_COLS) * bars
    for b in range(bars):
        img[r, c + b, 0] = lev[:, b]
        img[r, c + b, 1] = pk[:, b]
    img[..., 3] = 1.0
    return img


# ------------------------------------------------------------------------------------------------ circuitry

def circuit(size=512, cell=8, walkers=150, seed=5):
    """PCB-style traces: Manhattan/45° random walkers on a coarse grid with vias at the ends.
    → (H, W, 4): R = trace/via mask, G = distance along the trace (0..1, for travelling pulses), B = via mask."""
    rng = np.random.default_rng(seed)
    g = size // cell
    occ = np.zeros((g, g), bool)
    R = np.zeros((size, size), np.float32)
    G = np.zeros((size, size), np.float32)
    B = np.zeros((size, size), np.float32)
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    def stamp(px, py, val, w=1):
        R[max(0, py - w):py + w + 1, max(0, px - w):px + w + 1] = 1.0
        G[max(0, py - w):py + w + 1, max(0, px - w):px + w + 1] = val

    for _ in range(walkers):
        x, y = rng.integers(1, g - 1, 2)
        if occ[y, x]:
            continue
        d = int(rng.integers(0, 4))
        L = int(rng.integers(6, 28))
        path = [(x, y)]
        for _s in range(L):
            if rng.random() < 0.18:
                d = (d + int(rng.choice([1, 3]))) % 4 if d < 4 else int(rng.integers(0, 4))
                if rng.random() < 0.25:
                    d = 4 + int(rng.integers(0, 4))
            dx, dy = dirs[d]
            nx, ny = x + dx, y + dy
            if not (1 <= nx < g - 1 and 1 <= ny < g - 1) or occ[ny, nx]:
                break
            x, y = nx, ny
            path.append((x, y))
        if len(path) < 4:
            continue
        ph = rng.random()
        tot = len(path) - 1
        for k in range(tot):
            (x0, y0), (x1, y1) = path[k], path[k + 1]
            occ[y0, x0] = True
            for s in np.linspace(0, 1, cell + 1):
                px = int(round((x0 + (x1 - x0) * s + 0.5) * cell))
                py = int(round((y0 + (y1 - y0) * s + 0.5) * cell))
                stamp(min(px, size - 1), min(py, size - 1), (ph + (k + s) / max(tot, 1) * 0.6) % 1.0)
        occ[path[-1][1], path[-1][0]] = True
        for (vx, vy) in (path[0], path[-1]):                    # vias: little pads at both ends
            cx, cy = (vx + 0.5) * cell, (vy + 0.5) * cell
            yy, xx = np.ogrid[:size, :size]
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (cell * 0.42) ** 2
            R[m] = 1.0
            B[m] = 1.0
    out = np.zeros((size, size, 4), np.float32)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = R, G, B, 1.0
    return out


# ------------------------------------------------------------------------------------------------ bpy writer

def write_png(arr, path, name):
    """float RGBA (H, W, 4) 0..1, top row first → 8-bit RGB PNG written with zlib (no colour management can touch
    the data values) → bpy image loaded Non-Color, path-referenced (not packed)."""
    import struct
    import zlib

    import bpy
    a8 = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
    h, w = a8.shape[:2]
    raw = b"".join(b"\x00" + a8[r].tobytes() for r in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    os.replace(tmp, path)
    out = bpy.data.images.load(path, check_existing=False)
    out.name = name
    out.colorspace_settings.name = "Non-Color"
    return out


# ------------------------------------------------------------------------------------------------ which face the camera sees

def rot_mats(yaw, fx, fy):
    """per-frame cube rotation Rz(yaw)·Rx(fx)·Ry(fy) (the CubeYaw → CubeFlipX → CubeFlipY parenting) → (n, 3, 3)"""
    cz, sz, cx, sx, cy, sy = np.cos(yaw), np.sin(yaw), np.cos(fx), np.sin(fx), np.cos(fy), np.sin(fy)
    n = len(yaw)
    Rz = np.zeros((n, 3, 3)); Rx = np.zeros((n, 3, 3)); Ry = np.zeros((n, 3, 3))  # noqa: E702
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cz, -sz, sz, cz, 1
    Rx[:, 0, 0], Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = 1, cx, -sx, sx, cx
    Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 1, 1], Ry[:, 2, 0], Ry[:, 2, 2] = cy, sy, 1, -sy, cy
    return Rz @ Rx @ Ry


def facing(yaw, fx, fy, cube_c, cam_loc):
    """(n, 6) cosine between each face normal and the direction cube → camera (> 0.5 ≈ clearly seen)"""
    M = rot_mats(yaw, fx, fy)
    nw = np.einsum("nij,kj->nki", M, FACE_NORMALS)
    v = cam_loc - cube_c
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.einsum("nki,ni->nk", nw, v)
