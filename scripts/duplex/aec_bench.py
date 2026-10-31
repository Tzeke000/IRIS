"""scripts/duplex/aec_bench.py - offline AEC bench: how much of my own voice survives into the mic.

Builds a mic signal = near-end (a human talking) + echo of far-end (me) through a synthetic room
(exponential-decay impulse response, configurable delay / gain / clock drift) + noise, runs it
through voice/iris_aec.py backends, and reports:

  * ERLE (dB) over far-end-only time (after a 1 s convergence allowance) - how much echo is removed
  * near-end preservation in double-talk: STOI of output vs clean near-end and the same for the
    unprocessed mic (STOI, not sample SNR: AEC3 shifts phase inaudibly), so the cost of the canceller on HIS words is visible
  * residual echo level (dBFS) in far-only time - what the barge-in rms floor / VAD would see

Optionally writes out/<scenario>_<backend>.wav (+ near/far references) for the STT pass
(scripts/duplex/aec_stt_eval.py, iris-v100 venv). Input audio is read from paths you pass; the
bench never commits or uploads audio.

usage: python scripts/duplex/aec_bench.py --far IRIS.wav [--far2 ...] --near HUMAN.wav [--out DIR]
       (needs numpy scipy soundfile + livekit for the webrtc backend)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "voice"))
import iris_aec  # noqa: E402

SR = 16000


def load16(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x / max(1e-6, float(np.abs(x).max())) * 0.5


def compact(x: np.ndarray, max_gap_s: float = 0.25) -> np.ndarray:
    """Squeeze pauses longer than max_gap_s down to max_gap_s (20 ms energy frames, -40 dB
    relative gate), so a double-talk window is dense speech instead of mostly silence."""
    f = 320
    n = x.size // f
    e = np.array([np.sqrt(np.mean(x[i * f:(i + 1) * f] ** 2)) for i in range(n)])
    on = e > (e.max() * 10 ** (-40 / 20))
    keep, run = [], 0
    for i in range(n):
        run = 0 if on[i] else run + 1
        if run * f / SR <= max_gap_s:
            keep.append(x[i * f:(i + 1) * f])
    return np.concatenate(keep) if keep else x


def room_ir(rt60: float, delay_ms: float, gain: float, seed: int, length_s: float = 0.5) -> np.ndarray:
    """Direct path at delay + exponentially decaying diffuse tail (RT60), normalised so the
    direct tap = gain."""
    rng = np.random.default_rng(seed)
    n = int(length_s * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / max(rt60, 1e-3)) * 0.3
    h = np.zeros(int(delay_ms * SR / 1000) + n)
    d = int(delay_ms * SR / 1000)
    h[d] = 1.0
    h[d + 1:d + 1 + n - 1] += tail[1:]
    # a loudspeaker isn't flat: mild low-pass on the whole path
    k = np.array([0.25, 0.5, 0.25])
    h = np.convolve(h, k)[:h.size]
    return (h * gain).astype(np.float64)


def build(far: np.ndarray, near: np.ndarray, *, delay_ms: float, gain: float, rt60: float,
          drift_ppm: float, noise_db: float, seed: int):
    """Timeline: [0, A) far only, [A, B) double-talk, [B, C) near only."""
    A = min(far.size, 8 * SR)
    far = far[:A + 6 * SR] if far.size >= A + 6 * SR else np.concatenate([far, far])[:A + 6 * SR]
    dt_len = far.size - A
    nn = near[: dt_len + 4 * SR]
    total = A + nn.size
    far_full = np.zeros(total, np.float32)
    far_full[:far.size] = far
    near_full = np.zeros(total, np.float32)
    near_full[A:A + nn.size] = nn
    echo_src = far_full.astype(np.float64)
    if drift_ppm:
        # speaker clock != mic clock: the played signal is time-stretched relative to the reference
        tt = np.arange(total) * (1.0 + drift_ppm * 1e-6)
        echo_src = np.interp(tt, np.arange(total), echo_src, right=0.0)
    echo = fftconvolve(echo_src, room_ir(rt60, delay_ms, gain, seed))[:total]
    rng = np.random.default_rng(seed + 1)
    noise = rng.standard_normal(total) * (10 ** (noise_db / 20.0))
    mic = (near_full + echo + noise).astype(np.float32)
    regions = {"far_only": (SR, A), "double": (A, A + dt_len), "near_only": (A + dt_len, total)}
    return mic, far_full, near_full, echo.astype(np.float32), regions


def run_backend(name: str, mic: np.ndarray, far: np.ndarray, chunk: int = 512, **kw):
    aec = iris_aec.EchoCanceller(backend=name, **kw)
    out = []
    t0 = time.perf_counter()
    for i in range(0, mic.size, chunk):
        aec.push_render(far[i:i + chunk])
        out.append(aec.process(mic[i:i + chunk]))
    dt = time.perf_counter() - t0
    y = np.concatenate(out)[:mic.size]
    # undo the canceller's constant frame-buffer lag (< 10 ms) before sample-wise metrics
    lag = iris_aec.LATENCY
    if lag:
        y = np.concatenate([y[lag:], np.zeros(lag, np.float32)])
    return y, dt, aec.stats()


def align(y: np.ndarray, ref: np.ndarray, region: slice, max_lag: int = 800) -> "tuple[np.ndarray, int]":
    """Shift y by the lag that best matches ref inside region (a backend's own algorithmic delay,
    e.g. AEC3's ~120-sample block delay, must not count as damage to his words)."""
    a = ref[region].astype(np.float64)
    b = y[region].astype(np.float64)
    m = max_lag
    if a.size <= 4 * m:
        return y, 0
    core = a[m:-m]
    best = max(range(-m, m + 1, 2), key=lambda L: float(np.dot(core, b[m + L:b.size - m + L])))
    best = max(range(best - 2, best + 3), key=lambda L: float(np.dot(core, b[m + L:b.size - m + L])))
    if best > 0:
        y = np.concatenate([y[best:], np.zeros(best, np.float32)])
    elif best < 0:
        y = np.concatenate([np.zeros(-best, np.float32), y[:best]])
    return y, best


def snr(ref: np.ndarray, est: np.ndarray) -> float:
    e = est - ref
    return 10 * float(np.log10((ref.astype(np.float64) ** 2).sum() / max((e.astype(np.float64) ** 2).sum(), 1e-20)))


def stoi(ref: np.ndarray, est: np.ndarray) -> float:
    """Short-Time Objective Intelligibility (0..1) of est vs clean ref. Phase-insensitive, unlike a
    sample SNR - AEC3's frequency-domain suppressor shifts phase without changing what you hear."""
    from pystoi import stoi as _stoi
    return float(_stoi(ref.astype(np.float64), est.astype(np.float64), SR, extended=False))


def dbfs(x: np.ndarray) -> float:
    return 20 * float(np.log10(max(np.sqrt(np.mean(x.astype(np.float64) ** 2)), 1e-9)))


SCENARIOS = [
    # name, delay_ms, gain, rt60, drift_ppm, noise_db
    ("speakers_near_60ms", 60, 0.5, 0.30, 0, -60),
    ("speakers_far_200ms", 200, 0.35, 0.45, 0, -55),
    ("speakers_hot_120ms", 120, 0.9, 0.35, 0, -55),
    ("speakers_drift_80ppm", 120, 0.5, 0.35, 80, -55),
    ("headset_bleed_30ms", 30, 0.05, 0.05, 0, -60),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--far", required=True, nargs="+", help="my voice (far end) wav(s), concatenated")
    ap.add_argument("--near", required=True, nargs="+", help="a human (near end) wav(s), concatenated")
    ap.add_argument("--backends", default="none,webrtc,speex")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    far = np.concatenate([load16(p) for p in a.far])
    near = compact(np.concatenate([load16(p) for p in a.near]))
    results = []
    for (name, d, g, rt, drift, nz) in SCENARIOS:
        mic, farf, nearf, echo, R = build(far, near, delay_ms=d, gain=g, rt60=rt, drift_ppm=drift,
                                          noise_db=nz, seed=7)
        fo, dbl = slice(*R["far_only"]), slice(*R["double"])
        base_dt = stoi(nearf[dbl], mic[dbl])
        for b in a.backends.split(","):
            kw = {"tail_ms": 600} if b == "speex" else {}
            try:
                y, dt, st = run_backend(b, mic, farf, **kw)
                y, lag = align(y, nearf, slice(*R["near_only"]))
            except Exception as e:  # noqa: BLE001
                print(f"{name:24s} {b:7s} unavailable: {e!r}")
                continue
            r = {"scenario": name, "backend": b, "delay_ms": d, "gain": g, "rt60": rt, "drift_ppm": drift,
                 "erle_far_only_db": round(iris_aec.erle_db(mic[fo], y[fo]), 1),
                 "residual_echo_dbfs": round(dbfs(y[fo]), 1),
                 "echo_in_dbfs": round(dbfs(mic[fo]), 1),
                 "dt_stoi_out": round(stoi(nearf[dbl], y[dbl]), 3),
                 "dt_stoi_mic": round(base_dt, 3),
                 "near_only_stoi": round(stoi(nearf[slice(*R['near_only'])], y[slice(*R['near_only'])]), 3),
                 "rtf": round(dt / (mic.size / SR), 4), "algo_lag_samples": lag}
            results.append(r)
            print(f"{name:24s} {b:7s} ERLE {r['erle_far_only_db']:5.1f} dB  residual {r['residual_echo_dbfs']:6.1f} dBFS "
                  f"(echo {r['echo_in_dbfs']:6.1f})  DT-STOI {r['dt_stoi_mic']:.3f} -> {r['dt_stoi_out']:.3f}  "
                  f"near-only STOI {r['near_only_stoi']:.3f}  RTF {r['rtf']} lag {lag}", flush=True)
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                sf.write(os.path.join(a.out, f"{name}_{b}.wav"), y, SR)
        if a.out:
            sf.write(os.path.join(a.out, f"{name}__mic.wav"), mic, SR)
            sf.write(os.path.join(a.out, f"{name}__near.wav"), nearf, SR)
            json.dump({k: [int(v[0]), int(v[1])] for k, v in R.items()},
                      open(os.path.join(a.out, f"{name}__regions.json"), "w"))
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
