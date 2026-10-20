"""Find the moments worth cutting in a long recording: spoken edit CUES, audio peaks, hype
words / laughter, and scene cuts. Produces CANDIDATES — Iris reads the transcript around
each one and makes the final pick (no paid AI API anywhere in this pipeline).

usage: /home/iris/venvs/iris-v100/bin/python scripts/video_edit/analyze.py <video>
         [--transcript X.json] [--clip-len 30] [--top 6] [--no-scenes]
Writes <video>.analysis.json and prints a readable summary.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Things Zeke might SAY while recording to direct the edit (matched on the transcript).
CUE_PATTERNS = [
    (r"\b(edit|clip|cut|save|keep)\s+(this|that)\b", "clip_here"),
    (r"\bmake\s+(this|that)\s+(a\s+)?(clip|tiktok|short)\b", "clip_here"),
    (r"\bput\s+(this|that)\s+in\b", "clip_here"),
    (r"\bzoom\s+(in\s+)?(here|on\s+(this|that|me|him))\b", "zoom"),
    (r"\b(slow[\s-]?mo|slow\s+(this|that)\s+down)\b", "slowmo"),
    (r"\b(cut|delete|remove)\s+(this|that)\s+(part|out)\b", "cut_out"),
    (r"\bstart\s+(the\s+)?clip\s+(here|now)\b", "clip_start"),
    (r"\bend\s+(the\s+)?clip\s+(here|now)\b", "clip_end"),
]
# Hype / reaction words -> excitement (also drive emoji in captions.py).
HYPE = {"let's go": 3, "lets go": 3, "no way": 3, "oh my god": 3, "omg": 2, "holy": 2, "clutch": 3,
        "what": 1, "bro": 1, "dude": 1, "yes": 1, "insane": 2, "crazy": 2, "dead": 2, "dang": 1,
        "nice": 1, "wow": 2, "run": 1, "help": 1, "oh no": 2, "haha": 2, "hahaha": 3, "lmao": 2,
        "lol": 1, "gg": 2, "gotcha": 2, "boom": 2, "sheesh": 2}


def _wav(video: Path, sr: int = 16000) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".f32") as f:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", f.name], check=True)
        return np.fromfile(f.name, dtype=np.float32)


def audio_peaks(video: Path, hop_s: float = 0.25) -> dict:
    sr = 16000
    y = _wav(video, sr)
    hop = int(sr * hop_s)
    n = max(1, len(y) // hop)
    rms = np.sqrt(np.array([np.mean(y[i * hop:(i + 1) * hop] ** 2) for i in range(n)]) + 1e-12)
    db = 20 * np.log10(rms)
    med, mad = np.median(db), np.median(np.abs(db - np.median(db))) + 1e-6
    z = (db - med) / (1.4826 * mad)
    return {"hop_s": hop_s, "z": z.round(2).tolist()}


def scene_cuts(video: Path) -> list[float]:
    from scenedetect import ContentDetector, SceneManager, open_video
    v = open_video(str(video))
    sm = SceneManager()
    sm.auto_downscale = True
    sm.add_detector(ContentDetector(threshold=27.0))
    sm.detect_scenes(v, show_progress=False)
    return [round(s[0].get_seconds(), 3) for s in sm.get_scene_list()][1:]


def words_of(tr: dict) -> list[dict]:
    return [w for s in tr.get("segments", []) for w in s.get("words", [])]


def find_cues(tr: dict) -> list[dict]:
    words = words_of(tr)
    text, idx = "", []
    for i, w in enumerate(words):   # char offset -> word index map, so a regex hit gets a time
        tok = re.sub(r"[^\w' ]", "", w["word"].lower())
        start = len(text)
        text += tok + " "
        idx.append((start, i))
    out = []
    for pat, kind in CUE_PATTERNS:
        for m in re.finditer(pat, text):
            wi = max(i for s, i in idx if s <= m.start())
            out.append({"kind": kind, "phrase": m.group(0), "t": words[wi]["start"]})
    return sorted(out, key=lambda c: c["t"])


def hype_hits(tr: dict) -> list[dict]:
    hits = []
    for s in tr.get("segments", []):
        low = " " + re.sub(r"[^\w' ]", " ", s["text"].lower()) + " "
        for k, wgt in HYPE.items():
            if f" {k} " in low:
                t = next((w["start"] for w in s["words"] if k.split()[0] in w["word"].lower()), s["start"])
                hits.append({"t": t, "word": k, "w": wgt})
    return sorted(hits, key=lambda h: h["t"])


def score_windows(dur: float, ap: dict, hype: list, cues: list, clip_len: float, step: float = 2.0) -> list:
    z, hop = np.array(ap["z"]), ap["hop_s"]
    wins = []
    t = 0.0
    while t + min(clip_len, dur) <= dur + 1e-6:
        a, b = int(t / hop), max(int(t / hop) + 1, int((t + clip_len) / hop))
        seg = z[a:b]
        loud = float(np.mean(np.sort(seg)[-max(1, len(seg) // 5):])) if len(seg) else 0.0
        hy = sum(h["w"] for h in hype if t <= h["t"] < t + clip_len)
        cu = sum(1 for c in cues if c["kind"] in ("clip_here", "clip_start") and t <= c["t"] < t + clip_len)
        wins.append({"start": round(t, 2), "end": round(min(dur, t + clip_len), 2),
                     "score": round(loud + 0.8 * hy + 6.0 * cu, 2), "loud": round(loud, 2), "hype": hy, "cues": cu})
        t += step
        if dur <= clip_len:
            break
    picked = []
    for w in sorted(wins, key=lambda w: -w["score"]):
        if all(w["end"] <= p["start"] or w["start"] >= p["end"] for p in picked):
            picked.append(w)
    return picked


def snap(t: float, cuts: list[float], tr: dict, tol: float = 1.5) -> float:
    """Move a boundary to the nearest scene cut, else the nearest gap between spoken sentences."""
    near = [c for c in cuts if abs(c - t) <= tol]
    if near:
        return min(near, key=lambda c: abs(c - t))
    gaps = [s["end"] for s in tr.get("segments", [])] + [s["start"] for s in tr.get("segments", [])]
    near = [g for g in gaps if abs(g - t) <= tol]
    return min(near, key=lambda g: abs(g - t)) if near else t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--transcript", default="")
    ap.add_argument("--clip-len", type=float, default=30.0)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--no-scenes", action="store_true")
    a = ap.parse_args()
    video = Path(a.video)
    tp = Path(a.transcript) if a.transcript else video.with_suffix(video.suffix + ".transcript.json")
    tr = json.loads(tp.read_text()) if tp.exists() else {"segments": [], "duration": None}
    aud = audio_peaks(video)
    dur = tr.get("duration") or len(aud["z"]) * aud["hop_s"]
    cuts = [] if a.no_scenes else scene_cuts(video)
    cues, hype = find_cues(tr), hype_hits(tr)
    wins = score_windows(dur, aud, hype, cues, a.clip_len)[: a.top]
    for w in wins:
        w["start_snapped"] = round(snap(w["start"], cuts, tr), 2)
        w["end_snapped"] = round(snap(w["end"], cuts, tr), 2)
        w["said"] = " ".join(s["text"] for s in tr.get("segments", [])
                             if s["end"] > w["start"] and s["start"] < w["end"])[:400]
    res = {"video": str(video), "duration": dur, "scene_cuts": cuts, "cues": cues, "hype": hype,
           "candidates": wins}
    out = video.with_suffix(video.suffix + ".analysis.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[analyze] {video.name}: {dur:.1f}s, {len(cuts)} scene cuts, {len(cues)} spoken cues, "
          f"{len(hype)} hype hits -> {out}")
    for c in cues:
        print(f"  CUE {c['t']:8.2f}s  {c['kind']:<10} \"{c['phrase']}\"")
    for i, w in enumerate(wins, 1):
        print(f"  #{i} {w['start_snapped']:7.2f}-{w['end_snapped']:7.2f}s score {w['score']:5.2f} "
              f"(loud {w['loud']}, hype {w['hype']}, cues {w['cues']}) | {w['said'][:110]}")


if __name__ == "__main__":
    main()
