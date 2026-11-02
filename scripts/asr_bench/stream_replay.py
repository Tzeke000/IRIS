"""Replay clips through the LIVE incremental path (voice/iris_stream_asr_engine.py), 100 ms frames,
and compare with the offline bench's whole-clip text. usage (NeMo venv):
  stream_replay.py BENCH_RESULT_JSON [boost_alpha] [phrases,comma,sep]
Prints per-clip match / text, and the step-time and flush-time distribution. Audio stays on disk."""
import sys, os, glob, json, time, numpy as np, soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "voice"))
from iris_stream_asr_engine import NemoStreamer

bench = json.load(open(sys.argv[1]))["clips"]
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
phr = sys.argv[3].split(",") if len(sys.argv) > 3 else []
st = NemoStreamer(right_ctx=int(os.environ.get("RC", "1")), boost_phrases=phr, boost_alpha=alpha)
print("warm ms", round(st.warm(), 1), flush=True)
clips = sorted(glob.glob(os.environ.get("CLIPS_GLOB", "/data/asr_bench/clips/*.wav")))
same, flush_ms, steps, out = 0, [], [], {}
for c in clips:
    a, sr = sf.read(c, dtype="float32"); assert sr == 16000
    s = st.session()
    for i in range(0, len(a), 1600):
        s.push(a[i:i + 1600])
    t = time.perf_counter(); txt = s.finish(); flush_ms.append((time.perf_counter() - t) * 1000)
    steps += s.step_ms
    b = os.path.basename(c); ref = (bench.get(b) or {}).get("text", "")
    same += (txt == ref)
    out[b] = {"text": txt, "words": s.words()}
    print(("SAME " if txt == ref else "DIFF ") + f"{b[4:20]} live={txt[:80]!r}" + ("" if txt == ref else f"\n      bench={ref[:80]!r}"), flush=True)
print(f"identical {same}/{len(clips)}  flush ms p50 {np.percentile(flush_ms,50):.0f} p95 {np.percentile(flush_ms,95):.0f}  step ms p50 {np.percentile(steps,50):.0f} p95 {np.percentile(steps,95):.0f}")
tag = os.environ.get("TAG", f"live_a{alpha:g}")
json.dump(out, open(f"/data/asr_bench/results/stream_replay_{tag}.json", "w"), indent=1)
