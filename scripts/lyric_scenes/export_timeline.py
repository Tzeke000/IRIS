"""Export a song's per-frame music timeline for Blender-rendered ("real") scenes.

    scripts/server/lyric_viz.sh is the venv runner; run this with the same venv:
    ~/venvs/iris-v100/bin/python scripts/lyric_scenes/export_timeline.py \
        --audio state/tzeke_songs/hit_a_bump/hit_a_bump_half.wav \
        --out state/tzeke_songs/hit_a_bump/timeline_30fps.npz [--fps 30] \
        [--punch-words bump,bomb] [--words-json words.json]

Same analysis + section logic the 2-D scenes use (lyric_viz.analyze +
stage_timeline), so a Blender scene drops on exactly the frames the others do.
Output .npz (all arrays length n_frames unless noted):
  rms bass high build (0..1) · kick snare drop beat (bool) · beat_i beat_ph ·
  bars (n x 40) · bpm (scalar) · fps (scalar) ·
  T_<key> for every per-frame array stage_timeline returns (I hot big pre black
  kage sage strobe_age one_age bi ph gbeat bar tier rot ...) ·
  entries (drop-entry frames) · segs (k x 3: start, end, hot) ·
  punch (per-frame zoom >=1 on sung punch words; all 1.0 if none) · punch_frames.
Blender's bundled Python has numpy, so a scene's bpy script can np.load() it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # scripts/ -> import lyric_viz
import lyric_viz as lv                         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--punch-words", default="bump,bomb")
    ap.add_argument("--words-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    a, _pcm, _sr = lv._analyze_cached(args.audio, args.fps)
    n = len(a.rms)
    T = lv.stage_timeline(a, args.fps, "acid", label="export")
    out = dict(fps=np.int32(args.fps), bpm=np.float32(a.bpm))
    for k in ("rms", "bass", "high", "build", "kick", "snare", "drop", "beat",
              "beat_i", "beat_ph", "bars"):
        out[k] = np.asarray(getattr(a, k))
    for k, v in T.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out["T_" + k] = v
    out["entries"] = np.asarray(T.get("entries", []), np.int64)
    out["segs"] = np.asarray([(s0, s1, int(h)) for s0, s1, h in T["segs"]],
                             np.int64).reshape(-1, 3)
    punch = np.ones(n, np.float32)
    pframes = np.zeros(0, np.int64)
    words = [w.strip() for w in args.punch_words.split(",") if w.strip()]
    if words:
        heard, _segs = lv._words_cached(args.audio, args.device, args.words_json)
        pz, ptimes = lv.lyric_punch_schedule(heard, words, args.fps, a.beat_i, n)
        if pz is not None:
            punch = np.asarray(pz, np.float32)
            pframes = np.asarray([int(round(t * args.fps)) for t in ptimes],
                                 np.int64)
    out["punch"], out["punch_frames"] = punch, pframes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[export_timeline] {n} frames @ {args.fps} fps, bpm {a.bpm:.1f}, "
          f"{len(out['entries'])} drop entries, {len(pframes)} punches -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
