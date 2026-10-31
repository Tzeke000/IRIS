"""scripts/duplex/live_words_bench.py - how live are the "live words"? (full-duplex step b)

Feeds wav speech into the REAL voice/iris_live_words.LiveWords in real time (512-sample chunks at
16 kHz, wall-clock paced, exactly like the barge path's on_capture tap) with a faster-whisper
transcribe fn shaped like the daemon's _live_whisper_text (greedy, text-only), and measures:

  * first_partial_s   - audio seconds from speech start until the first partial appears
  * decode_ms         - per partial pass, against the buffer length it decoded
  * word lag          - for every partial: how far (s) the newest correctly-shown word ends behind
                        "now" (word end times from a word-timestamped final pass of the same audio)
  * partial accuracy  - fraction of words spoken so far that the partial already shows

usage: iris-v100 python scripts/duplex/live_words_bench.py WAV [WAV...] [--device cuda|cpu]
       [--cadence 0.6] [--seconds 12] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "voice"))
import iris_live_words as lw  # noqa: E402

SR = 16000


def words(t):
    return re.sub(r"[^a-z0-9' ]+", " ", (t or "").lower()).split()


def load(p, seconds):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    e = np.abs(x)
    on = np.argmax(e > 0.05 * e.max())
    x = x[max(0, on - SR // 10):][: int(seconds * SR)]
    return x / max(1e-6, float(np.abs(x).max())) * 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="mobiuslabsgmbh/faster-whisper-large-v3-turbo")
    ap.add_argument("--cadence", type=float, default=0.6)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.device == "cuda":
        import torch  # noqa: F401  (loads the bundled libcublas/cudnn ctranslate2 needs, like the ears)
    from faster_whisper import WhisperModel
    m = WhisperModel(a.model, device=a.device, compute_type="float16" if a.device == "cuda" else "int8")
    passes = []

    def transcribe(buf):
        t0 = time.perf_counter()
        segs, _ = m.transcribe(np.ascontiguousarray(buf, dtype=np.float32), beam_size=1, language="en",
                               condition_on_previous_text=False, without_timestamps=True)
        txt = " ".join(s.text.strip() for s in segs).strip()
        passes.append({"buf_s": round(buf.size / SR, 2), "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return txt

    transcribe(np.zeros(SR, np.float32))         # warm (the live daemon's whisper is always warm)
    passes.clear()
    report = []
    for p in a.wavs:
        x = load(p, a.seconds)
        segs, _ = m.transcribe(x, beam_size=5, language="en", word_timestamps=True,
                               condition_on_previous_text=False)
        ref = [(w.word, w.end) for s in segs for w in (s.words or [])]
        ref_w = [(words(w)[0] if words(w) else "", e) for w, e in ref]
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        live = lw.LiveWords("bench", path=tmp.name, transcribe_fn=transcribe, cadence_s=a.cadence,
                            min_audio_s=0.5)
        shown = []                                   # (audio_time_s, text)
        orig_pub = live.publish

        def pub(text, final=False, dropped=None, _t0=[None]):
            shown.append((time.perf_counter() - t_start, text, final))
            return orig_pub(text, final=final, dropped=dropped)
        live.publish = pub
        passes.clear()
        t_start = time.perf_counter()
        for i in range(0, x.size, 512):
            target = t_start + (i + 512) / SR
            while time.perf_counter() < target:
                time.sleep(0.002)
            live.feed(x[i:i + 512])
        time.sleep(a.cadence + 0.3)
        live.stop()
        os.unlink(tmp.name)
        # metrics
        lags, accs = [], []
        for (t, txt, fin) in shown:
            spoken = [w for w, e in ref_w if e <= t]
            if not spoken:
                continue
            got = words(txt)
            # newest spoken word that the partial contains (in order, greedy)
            j, last_end = 0, None
            for w, e in ref_w:
                if e > t:
                    break
                if w in got[j:]:
                    j = got.index(w, j) + 1
                    last_end = e
            accs.append(min(1.0, j / len(spoken)))
            if last_end is not None:
                lags.append(t - last_end)
        r = {"wav": os.path.basename(p), "audio_s": round(x.size / SR, 1), "partials": len(shown),
             "first_partial_s": round(shown[0][0], 2) if shown else None,
             "word_lag_s_median": round(float(np.median(lags)), 2) if lags else None,
             "word_lag_s_p90": round(float(np.percentile(lags, 90)), 2) if lags else None,
             "shown_fraction_median": round(float(np.median(accs)), 2) if accs else None,
             "decode_ms_by_buf": passes[:], "final_ref": " ".join(w for w, _ in ref)[:200]}
        report.append(r)
        dm = [q["ms"] for q in passes]
        print(f"{r['wav']}: {r['partials']} partials, first at {r['first_partial_s']} s, word lag median "
              f"{r['word_lag_s_median']} s p90 {r['word_lag_s_p90']} s, shown {r['shown_fraction_median']}, "
              f"decode ms median {np.median(dm) if dm else None} max {max(dm) if dm else None} "
              f"(buffers up to {passes[-1]['buf_s'] if passes else 0} s)", flush=True)
        for (t, txt, fin) in shown[:6]:
            print(f"   t={t:5.2f}s  {txt[:90]!r}")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
