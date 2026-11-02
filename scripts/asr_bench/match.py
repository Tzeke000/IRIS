"""Pair each saved utterance wav with the nearest user voice transcript entry (noisy reference)."""
import json, re, sys, wave, glob, os
from datetime import datetime
TR = "/home/iris/IRIS/state/transcript.jsonl"
clips = sorted(glob.glob("/data/asr_bench/clips_raw/utt_*.wav"))
t0 = min(datetime.strptime(os.path.basename(c)[4:23], "%Y%m%d_%H%M%S_%f").timestamp() for c in clips) - 600
ents = []
with open(TR, "rb") as f:
    f.seek(max(0, os.path.getsize(TR) - 3_000_000))
    for l in f:
        try: d = json.loads(l)
        except Exception: continue
        if d.get("role") == "user" and d.get("modality") == "voice" and d.get("ts", 0) > t0:
            ents.append(d)
def clean(s):
    s = re.sub(r"\n\[pace:.*", "", s, flags=re.S)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = s.replace("*", "").replace("↗", "").replace("↘", "").replace("…", " ")
    return re.sub(r"\s+", " ", s).strip()
out = []
for c in clips:
    b = os.path.basename(c)
    ts = datetime.strptime(b[4:23], "%Y%m%d_%H%M%S_%f").timestamp()
    with wave.open(c) as w: dur = w.getnframes() / w.getframerate()
    best = min(ents, key=lambda e: abs(e["ts"] - ts)) if ents else None
    dt = best["ts"] - ts if best else None
    out.append({"wav": b, "dur": round(dur, 2), "dt": round(dt, 1) if dt is not None else None,
                "live_text": clean(best["content"]) if best and abs(dt) < 15 else ""})
json.dump(out, open("/data/asr_bench/matched.json", "w"), indent=1)
for o in out: print(o["wav"][4:], o["dur"], o["dt"], "|", o["live_text"][:110])
