"""scripts/duplex/tier_echo_eval.py - does my OWN echo trip the barge-in, with and without AEC?

On the aec_bench.py --out wavs: run the real Silero VAD (512-sample chunks, threshold 0.5, the
daemon's settings) + voice/iris_duplex_watch.tiered_accumulate over
  * far-only time (only my echo in the mic): episodes / ducks / pauses = false interruptions
  * the start of double-talk (he starts talking over me): time from his onset to the first DUCK and
    the first PAUSE (no transcript involved: the duration ladder alone)
and the binary watch's rule (8 voiced chunks above rms 0.02 = cut) for comparison.

usage: iris-v100 python scripts/duplex/tier_echo_eval.py DIR [DIR...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "voice"))
import iris_duplex_watch as W  # noqa: E402

SR = 16000


def vad_probs(model, x):
    import torch
    model.reset_states()
    out = []
    for i in range(0, x.size - 511, 512):
        f = x[i:i + 512].astype(np.float32)
        out.append((f, float(model(torch.from_numpy(f), SR).item())))
    return out


def binary_cuts(fp, need=8, min_rms=0.02):
    cuts, run = 0, 0
    for f, p in fp:
        v = p >= 0.5 and float(np.sqrt(np.mean(f * f))) >= min_rms
        run = run + 1 if v else 0
        if run >= need:
            cuts += 1
            run = -40          # refractory ~1.3 s (the window would be re-opened)
    return cuts


def tier_run(fp):
    acts = W.MouthActions()
    events = []
    i = [0]
    acts._call = lambda fn, name, *a: events.append((i[0], name))  # noqa: E731

    def gen():
        for k, c in enumerate(fp):
            i[0] = k
            yield c
    n_ep, n_pause, n_duck = 0, 0, 0
    it = gen()
    first = {}
    while True:
        st = {}
        audio, clf = W.tiered_accumulate(it, lambda: True, acts, min_rms=0.02, stats=st)
        n_ep += st.get("episodes", 0)
        for k, name in events:
            if name == "pause":
                n_pause += 1
                first.setdefault("pause", k)
            if name == "duck":
                n_duck += 1
                first.setdefault("duck", k)
        events.clear()
        acts.state = "normal"
        if audio is None:
            break
    return n_ep, n_duck, n_pause, first


def main():
    from silero_vad import load_silero_vad
    model = load_silero_vad()
    rows = []
    for d in sys.argv[1:]:
        for reg in sorted(glob.glob(os.path.join(d, "*__regions.json"))):
            scen = os.path.basename(reg).split("__")[0]
            R = json.load(open(reg))
            for path in sorted(glob.glob(os.path.join(d, scen + "_*.wav"))):
                b = os.path.basename(path)[len(scen) + 1:-4]
                if b.startswith("_"):
                    continue
                y, _ = sf.read(path, dtype="float32")
                fo = y[slice(*R["far_only"])]
                fp = vad_probs(model, fo)
                mins = fo.size / SR / 60
                ep, duck, pause, ff = tier_run(fp)
                bc = binary_cuts(fp)
                # double talk onset: from the region start (he starts there), first 3 s
                a0 = R["double"][0]
                dt = vad_probs(model, y[a0:a0 + 3 * SR])
                _, _, _, first = tier_run(dt)
                r = {"scenario": scen, "backend": b, "far_only_min": round(mins, 2),
                     "binary_false_cuts": bc, "tier_episodes": ep, "tier_false_ducks": duck,
                     "tier_false_pauses": pause,
                     "first_false_pause_ms": ff.get("pause", None) and int(ff["pause"] * 32),
                     "dt_first_duck_ms": first.get("duck", None) and int(first["duck"] * 32),
                     "dt_first_pause_ms": first.get("pause", None) and int(first["pause"] * 32)}
                rows.append(r)
                print(f"{scen:24s} {b:7s} far-only {mins:.2f} min: binary cuts {bc:2d} | tiers ducks {duck:2d} "
                      f"pauses {pause:2d} (first @ {r['first_false_pause_ms']} ms) || he starts: duck @ {r['dt_first_duck_ms']} ms, pause @ {r['dt_first_pause_ms']} ms",
                      flush=True)
    json.dump(rows, open("/data/sandbox/duplex/out/tier_echo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
