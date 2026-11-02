"""Word-level streaming latency from saved partials, timed against large-v3 word timestamps.
first_word_lag = first non-empty partial ready - end of first word
final_lag      = moment the partial already equals the final text - end of last word  (the 'words arrive after he finishes' number)
eou_lag        = first partial containing <EOU> - end of last word (Parakeet EOU only)"""
import json, glob, sys, numpy as np
from whisper_normalizer.english import EnglishTextNormalizer
N = EnglishTextNormalizer()
T = json.load(open("/data/asr_bench/results/whisper_large-v3_b5.json"))["clips"]
for f in sorted(glob.glob("/data/asr_bench/results/*.json")):
    d = json.load(open(f))
    if not isinstance(d, dict) or "chunk_ms" not in d: continue
    fw, fl, eou, miss = [], [], [], 0
    for k, c in d["clips"].items():
        w = T[k].get("words") or []
        if not w: continue
        first_end, last_end = w[0][2], w[-1][2]
        P = c["partials"]
        fin = N(c["text"].replace("<EOU>", ""))
        if not fin: miss += 1; continue
        t1 = next((t for t, x in P if x.strip()), None)
        if t1 is not None: fw.append(t1 - first_end)
        tf = next((t for t, x in P if N(x.replace("<EOU>", "")) == fin), None)
        if tf is not None: fl.append(tf - last_end)
        te = next((t for t, x in P if "<EOU>" in x or "<eou>" in x.lower()), None)
        if te is not None: eou.append(te - last_end)
    q = lambda v, p: f"{np.percentile(v, p)*1000:5.0f}" if v else "    -"
    print(f"{d['model']:52s} firstword p50 {q(fw,50)} | final-after-last-word p50 {q(fl,50)} p90 {q(fl,90)} | eou p50 {q(eou,50)} p90 {q(eou,90)} n={len(eou)} | empty={miss}")
