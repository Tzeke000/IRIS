"""Score every results/*.json against ref/refs.json. Refs may hold {a|b|} alternatives (best-matching combo per hyp).
Normalised with the Whisper English normaliser (case, punctuation, numbers, fillers)."""
import json, glob, itertools, re, os, sys, numpy as np, jiwer
from whisper_normalizer.english import EnglishTextNormalizer
N = EnglishTextNormalizer()
R = json.load(open("/data/asr_bench/ref/refs.json"))
DUR = {k: v["dur"] for k, v in json.load(open("/data/asr_bench/results/whisper_large-v3_b5.json"))["clips"].items()}
def combos(ref):
    parts = re.split(r"(\{[^}]*\})", ref)
    opts = [p[1:-1].split("|") if p.startswith("{") else [p] for p in parts]
    return ["".join(c) for c in itertools.product(*opts)]
def best(ref, hyp):
    h = N(hyp.replace("<EOU>", " "))
    best = None
    for r in combos(ref):
        r = N(r)
        o = jiwer.process_words(r, h if h else "")
        e = o.substitutions + o.deletions + o.insertions
        if best is None or e < best[0]: best = (e, len(r.split()))
    return best
rows = []
for f in sorted(glob.glob("/data/asr_bench/results/*.json")):
    d = json.load(open(f))
    if not isinstance(d, dict) or "clips" not in d: continue
    C = d["clips"]
    E = W = 0; per = {}
    for k, ref in R.items():
        if k not in C: continue
        e, w = best(ref, C[k]["text"]); E += e; W += w; per[k] = e / w
    def pct(key, q):
        v = [c[key] for c in C.values() if c.get(key) is not None]
        return float(np.percentile(v, q)) * 1000 if v else None
    barge = [k for k in R if "_barge" in k]
    Eb = sum(best(R[k], C[k]["text"])[0] for k in barge); Wb = sum(best(R[k], C[k]["text"])[1] for k in barge)
    short = [k for k in R if DUR[k] < 4.0]
    Es = sum(best(R[k], C[k]["text"])[0] for k in short); Ws = sum(best(R[k], C[k]["text"])[1] for k in short)
    rows.append(dict(model=d["model"], wer=100 * E / W, wer_barge=100 * Eb / Wb, wer_short=100 * Es / Ws, n=len(per),
                     final_p50=pct("final_after_end_s", 50) or pct("final_s", 50),
                     final_p95=pct("final_after_end_s", 95) or pct("final_s", 95),
                     step_p50=d.get("step_ms_p50"), vram=d.get("vram_mib"), cpu_s=d.get("cpu_s")))
print(f"{'model':48s} {'WER':>6s} {'barge':>6s} {'<4s':>6s} {'fin50':>6s} {'fin95':>6s} {'step50':>6s} {'VRAM':>6s} {'cpu_s':>6s}")
for r in rows:
    f = lambda x, p=0: "-" if x is None else f"{x:.{p}f}"
    print(f"{r['model']:48s} {f(r['wer'],1):>6s} {f(r['wer_barge'],1):>6s} {f(r['wer_short'],1):>6s} {f(r['final_p50']):>6s} {f(r['final_p95']):>6s} {f(r['step_p50']):>6s} {f(r['vram']):>6s} {f(r['cpu_s'],1):>6s}")
json.dump(rows, open("/data/asr_bench/results/summary.json", "w"), indent=1)
