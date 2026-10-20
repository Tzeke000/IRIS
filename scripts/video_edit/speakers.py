"""Who said what: tag every transcript segment with a speaker, so captions can give each
person their own colour (Zeke 10-20: friends' clips, "subtitles in different colors").

Uses the WeSpeaker ResNet34 voiceprint model the ears already run (voice/wren_voiceid.py,
CPU ONNX). Segments are embedded, clustered by cosine similarity, and a cluster that matches
an ENROLLED profile (state/voiceid/profiles/*.npy, e.g. zeke) takes that name; others are
S2, S3, ... in order of first appearance.

usage: /home/iris/venvs/iris-v100/bin/python scripts/video_edit/speakers.py <video> [--thr 0.55]
Rewrites <video>.transcript.json in place (adds "speaker" per segment).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO))
from voice import wren_voiceid as vid  # noqa: E402

PROFILE_MATCH = 0.50   # cluster centroid vs enrolled profile (max over profile rows)


def load_audio(video: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".f32") as f:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ac", "1", "-ar", "16000",
                        "-f", "f32le", f.name], check=True)
        return np.fromfile(f.name, dtype=np.float32)


def tag(tr: dict, audio: np.ndarray, thr: float = 0.30) -> dict:
    if not vid.warm_sync():
        raise RuntimeError(f"voiceid model unavailable: {vid.warm_error()}")
    segs = tr.get("segments", [])
    embs: list = []
    for s in segs:
        a, b = int(s["start"] * 16000), int(s["end"] * 16000)
        if b - a < int(0.6 * 16000):   # too short for a reliable voiceprint
            embs.append(None)
            continue
        embs.append(vid.embed(audio[a:b]))
    clusters: list[dict] = []          # {"sum": vec, "n": int}
    label = []
    for e in embs:
        if e is None:
            label.append(None)
            continue
        best, bs = None, -1.0
        for i, c in enumerate(clusters):
            cen = c["sum"] / np.linalg.norm(c["sum"])
            sim = float(cen @ e)
            if sim > bs:
                best, bs = i, sim
        if best is not None and bs >= thr:
            clusters[best]["sum"] += e
            clusters[best]["n"] += 1
            label.append(best)
        else:
            clusters.append({"sum": e.copy(), "n": 1})
            label.append(len(clusters) - 1)
    # short segments inherit the nearest labelled neighbour
    for i, l in enumerate(label):
        if l is None:
            near = [(abs(j - i), label[j]) for j in range(len(label)) if label[j] is not None]
            label[i] = min(near)[1] if near else 0
    profs = vid._load_profiles()
    names, used, nxt = {}, set(), 2
    for ci, c in enumerate(clusters):
        cen = c["sum"] / np.linalg.norm(c["sum"])
        scores = {n: float(np.max((a / np.linalg.norm(a, axis=1, keepdims=True)) @ cen)) for n, a in profs.items()}
        best = max(scores, key=scores.get) if scores else None
        if best and scores[best] >= PROFILE_MATCH and best not in used and best != "iris":
            names[ci] = best
            used.add(best)
    for ci in sorted(set(label), key=label.index):
        if ci not in names:
            names[ci] = f"S{nxt}"
            nxt += 1
    for s, l in zip(segs, label):
        s["speaker"] = names[l]
    tr["speakers"] = sorted(set(names.values()))
    return tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--thr", type=float, default=0.30)  # 10-20 test: same-speaker pairs ≥0.42, different ≤0.14
    a = ap.parse_args()
    v = Path(a.video)
    tp = v.with_suffix(v.suffix + ".transcript.json")
    tr = tag(json.loads(tp.read_text()), load_audio(v), a.thr)
    tp.write_text(json.dumps(tr, indent=1), encoding="utf-8")
    print(f"[speakers] {v.name}: {tr['speakers']} | " +
          " ".join(f"{s['start']:.1f}:{s['speaker']}" for s in tr["segments"]))


if __name__ == "__main__":
    main()
