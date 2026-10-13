"""Be My One — audio analysis → JSON the Blender builder reads (run with the librosa venv):

    ~/venvs/iris-v100/bin/python scripts/lyric_scenes/real/rooftop/analyse.py \
        /data/iris_assets/bemyone/be_my_one_master.wav scratch/scenes/bemyone_rooftop/audio.json

Per video frame (30 fps, index i ↔ scene frame 1+i ↔ audio time i/30): kick/bass/mid/high/rms envelopes
(percentile-normalised, fast-attack/slow-release), onsets, and a beat grid. The beat grid is a constant-tempo
grid fitted to librosa's tracked beats (least squares), so cuts never drift; downbeat phase is chosen as the
grid beat nearest Zeke's DROP 1 at 0:29 (his ear outranks the analysis).
"""
import json
import sys

import librosa
import numpy as np

wav, out = sys.argv[1], sys.argv[2]
FPS = 30
y, sr = librosa.load(wav, sr=None, mono=True)
dur = len(y) / sr
hop = sr / FPS
n = int(np.ceil(dur * FPS))
hop_i = int(round(hop))
S = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop_i))[:, :n]
f = librosa.fft_frequencies(sr=sr, n_fft=4096)


def norm(x, pct=99.0):
    return np.clip(x / (np.percentile(x, pct) + 1e-9), 0.0, 1.0)


def envelope(x, attack_s=0.01, release_s=0.15):
    a, r = np.exp(-1 / (attack_s * FPS)), np.exp(-1 / (release_s * FPS))
    o, v = np.empty_like(x), 0.0
    for i, s in enumerate(x):
        k = a if s > v else r
        v = k * v + (1 - k) * s
        o[i] = v
    return o


def pad(x):
    x = np.asarray(x, dtype=np.float64)
    return np.pad(x, (0, max(0, n - len(x))))[:n]


bands = {"kick": (30, 120), "bass": (20, 250), "mid": (250, 2000), "high": (4000, 12000)}
data = {"fps": FPS, "n": n, "dur": dur}
for name, (lo, hi) in bands.items():
    data[name] = pad(envelope(norm(S[(f >= lo) & (f < hi)].mean(axis=0)))).round(4).tolist()
data["rms"] = pad(norm(librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_i)[0])).round(4).tolist()
# slow loudness (1 s window) for section sanity checks
rms = np.array(data["rms"])
data["rms_slow"] = np.convolve(rms, np.ones(FPS) / FPS, mode="same").round(4).tolist()
env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_i)
data["onsets"] = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop_i, units="frames").tolist()
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_i, units="time")
bt = np.asarray(beats)
# fit constant tempo: t_k = t0 + k*P
P0 = 60.0 / float(np.atleast_1d(tempo)[0])
k = np.round((bt - bt[0]) / P0)
A = np.vstack([np.ones_like(k), k]).T
t0, P = np.linalg.lstsq(A, bt, rcond=None)[0]
# phase: grid beat nearest 0:29 is a downbeat
DROP1 = 29.0
kd = np.round((DROP1 - t0) / P)
tdrop = t0 + kd * P
grid = tdrop + P * np.arange(-int(tdrop / P) - 1, int((dur - tdrop) / P) + 2)
grid = grid[(grid >= 0) & (grid < dur)]
data["bpm"] = 60.0 / P
data["beat_period_s"] = P
data["beat_times"] = grid.round(4).tolist()
data["beats"] = np.round(grid * FPS).astype(int).tolist()          # 0-based frame index
data["downbeat_ref_time"] = float(tdrop)
data["tracked_beat_times"] = bt.round(4).tolist()
resid = bt - (t0 + k * P)
json.dump(data, open(out, "w"))
print(f"dur {dur:.2f}s n={n} bpm(tracked)={60/P0:.2f} bpm(fit)={60/P:.3f} resid_rms={np.sqrt(np.mean(resid**2))*1000:.1f}ms "
      f"drop-beat={tdrop:.3f}s beats={len(grid)}")
# coarse loudness per second for a sanity table
for s in range(0, int(dur), 1):
    seg = rms[s * FPS:(s + 1) * FPS]
    kk = np.array(data["kick"])[s * FPS:(s + 1) * FPS]
    print(f"{s//60}:{s%60:02d} rms={seg.mean():.2f} kick={kk.mean():.2f}", end=" | " if s % 4 != 3 else "\n")
