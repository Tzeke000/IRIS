"""scripts/duplex/aec_stt_eval.py - the downstream proof for AEC: what WHISPER hears.

Reads the wavs aec_bench.py --out wrote and transcribes, per scenario x backend:
  * far-only time (only MY voice is in the mic): words whisper produces there = echo that would
    become a phantom turn / false barge. Reported as echo_words.
  * double-talk time (he talks over me): WER of his words vs a transcript of his clean near-end
    signal (same segment). This is the question that matters: can my ears still understand him
    while I am talking, and do my own words leak into his transcript?

CPU int8 by default (never touches the V100 the live ears share). Same model family as the ears
(faster-whisper large-v3-turbo); --model small.en for a quick pass.

usage: iris-v100 python scripts/duplex/aec_stt_eval.py DIR [--model M] [--device cpu]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

SR = 16000


def words(t: str) -> list:
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()


def wer(ref: list, hyp: list) -> float:
    if not ref:
        return float(len(hyp) > 0)
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--model", default="mobiuslabsgmbh/faster-whisper-large-v3-turbo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    m = WhisperModel(a.model, device=a.device, compute_type="int8" if a.device == "cpu" else "float16",
                     cpu_threads=8)

    def stt(x: np.ndarray) -> str:
        segs, _ = m.transcribe(x.astype(np.float32), language="en", beam_size=1, vad_filter=False,
                               condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()

    out = []
    for reg_path in sorted(glob.glob(os.path.join(a.dir, "*__regions.json"))):
        scen = os.path.basename(reg_path).split("__")[0]
        R = json.load(open(reg_path))
        near, _ = sf.read(os.path.join(a.dir, scen + "__near.wav"), dtype="float32")
        dbl = slice(*R["double"])
        fo = slice(*R["far_only"])
        ref = words(stt(near[dbl]))
        for path in sorted(glob.glob(os.path.join(a.dir, scen + "_*.wav"))):
            b = os.path.basename(path)[len(scen) + 1:-4]
            if b.startswith("_"):
                continue
            y, _ = sf.read(path, dtype="float32")
            echo_txt = stt(y[fo])
            dt_txt = stt(y[dbl])
            r = {"scenario": scen, "backend": b, "echo_words": len(words(echo_txt)),
                 "echo_text": echo_txt[:120], "dt_wer": round(wer(ref, words(dt_txt)), 3),
                 "dt_extra_words": max(0, len(words(dt_txt)) - len(ref)), "ref_words": len(ref)}
            out.append(r)
            print(f"{scen:24s} {b:7s} echo_words {r['echo_words']:3d}  DT WER {r['dt_wer']:.3f} "
                  f"(ref {len(ref)} w, +{r['dt_extra_words']} extra)  echo: {echo_txt[:60]!r}", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
