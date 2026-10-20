"""Word-level transcript of a video/audio file with faster-whisper (local, free).

usage: /home/iris/venvs/iris-v100/bin/python scripts/video_edit/transcribe.py <media> [--out X.json]
       [--model large-v3-turbo] [--cpu] [--lang en]

Writes <media>.transcript.json:
  {"source", "duration", "language", "segments": [{"start","end","text",
     "words": [{"start","end","word","prob"}]}]}
Every word carries its own timestamp, so "edit this" said at 7:42 is findable to the
tenth of a second (scripts/video_edit/analyze.py --cues).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ctranslate2 needs the CUDA 12 libs that ship inside the venv's nvidia/* wheels (same fix as
# scripts/server/lyric_viz.sh). Re-exec once with them on LD_LIBRARY_PATH.
import os  # noqa: E402
_NV = Path(sys.prefix) / "lib" / "python3.11" / "site-packages" / "nvidia"
if _NV.exists() and "IRIS_EDIT_LDPATH" not in os.environ and "--cpu" not in sys.argv:
    os.environ["LD_LIBRARY_PATH"] = ":".join(str(_NV / d / "lib") for d in ("cublas", "cudnn", "cuda_runtime")) \
        + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["IRIS_EDIT_LDPATH"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

MODELS = {"large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
          "small.en": "Systran/faster-whisper-small.en"}


def transcribe(media: Path, model_name: str = "large-v3-turbo", cpu: bool = False,
               lang: str | None = "en") -> dict:
    from faster_whisper import WhisperModel
    mid = MODELS.get(model_name, model_name)
    if cpu:
        model = WhisperModel(mid, device="cpu", compute_type="int8", cpu_threads=8)
    else:
        try:
            model = WhisperModel(mid, device="cuda", compute_type="float16")
        except Exception as e:  # GPU busy/unavailable -> CPU, slower but works
            print(f"[transcribe] GPU load failed ({e!r}); using CPU", file=sys.stderr)
            model = WhisperModel(mid, device="cpu", compute_type="int8", cpu_threads=8)
    try:
        return _run(model, media, lang)
    except RuntimeError as e:   # CUDA libs/VRAM trouble shows up at decode time, not load time
        if cpu:
            raise
        print(f"[transcribe] GPU decode failed ({e!r}); retrying on CPU", file=sys.stderr)
        return _run(WhisperModel(mid, device="cpu", compute_type="int8", cpu_threads=8), media, lang)


def _run(model, media: Path, lang: str | None) -> dict:
    segs, info = model.transcribe(str(media), language=lang, word_timestamps=True,
                                  vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
                                  condition_on_previous_text=False, beam_size=5)
    out = {"source": str(media), "duration": round(float(info.duration), 3),
           "language": info.language, "segments": []}
    for s in segs:
        out["segments"].append({
            "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
            "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip(),
                       "prob": round(float(w.probability), 3)} for w in (s.words or [])]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()
    media = Path(a.media)
    t0 = time.time()
    res = transcribe(media, a.model, a.cpu, a.lang or None)
    out = Path(a.out) if a.out else media.with_suffix(media.suffix + ".transcript.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    n = sum(len(s["words"]) for s in res["segments"])
    print(f"[transcribe] {media.name}: {res['duration']:.1f}s audio, {len(res['segments'])} segments, "
          f"{n} words in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
