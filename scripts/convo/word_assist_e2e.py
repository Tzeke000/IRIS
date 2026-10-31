"""End-to-end timing of word assist WITHOUT the live stack: synthetic speech (Piper) -> the REAL
LiveWords partial loop (whisper turbo on the V100, paced in real time) -> the REAL WordAssist policy
-> the REAL fast LLM (Ollama) -> a fake mouth that timestamps the offer.

Measures offer latency from the END of his last word (the moment he starts searching) and checks the
negative cases (he finds the word himself / no cue). Writes only to a temp dir; unloads the Ollama
model at the end. The mouth's own first-audio time (Fish ~0.5-1.3 s) is NOT included.

usage: iris-v100 python scripts/convo/word_assist_e2e.py [--model qwen2.5:7b]
"""
import argparse, json, sys, tempfile, threading, time, urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "voice"))
import iris_live_words as lw  # noqa: E402
import iris_word_search as ws  # noqa: E402
from brain import iris_convo_layer as cl  # noqa: E402

SR = 16000
CASES = [  # (spoken text, trailing silence s, expect offer?, accepted)
    ("So for all the trees in the city, I want to use the node system in Blender. What's it called?", 2.5, True, ["geometry nodes"]),
    ("I want to make a green sauce tonight with basil and pine nuts. Oh, what's it called?", 2.5, True, ["pesto"]),
    ("We should use the node system in Blender, what's it called, geometry nodes, for the trees.", 2.5, False, []),
    ("Tomorrow I want to go to the park and fly the drone for a little while.", 2.5, False, []),
]


def _gpu_util():
    import subprocess
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return None


def synth(voice, text):
    chunks = list(voice.synthesize(text))
    x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    sr = chunks[0].sample_rate
    return resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=ws.DEFAULT_MODEL)
    ap.add_argument("--llm-timeout", type=float, default=2.5, help="production uses 2.5 s; raise to MEASURE")
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()
    from piper import PiperVoice
    from faster_whisper import WhisperModel
    pdir = ROOT / "models" / "piper"
    if not (pdir / "en_US-lessac-high.onnx.json").exists():      # worktree: models are git-ignored
        pdir = Path("/home/iris/IRIS/models/piper")
    voice = PiperVoice.load(str(pdir / "en_US-lessac-high.onnx"))
    wm = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")

    def tx(buf):
        segs, _ = wm.transcribe(np.ascontiguousarray(buf), beam_size=1, language="en",
                                condition_on_previous_text=False, without_timestamps=True, vad_filter=False)
        return " ".join(s.text.strip() for s in segs)
    tx(np.zeros(SR, np.float32))
    gloss = ws.load_glossary()
    ws.suggest("warm, what's it called", [], model=a.model, timeout=120)   # warm the LLM
    tmp = Path(tempfile.mkdtemp(prefix="wa_e2e_"))
    results = []
    for text, tail_s, expect, accept in CASES * a.rounds:
        speech = synth(voice, text)
        audio = np.concatenate([speech, np.zeros(int(tail_s * SR), np.float32)])
        live_path = tmp / "live.json"
        live_path.unlink(missing_ok=True)
        L = lw.LiveWords("e2e", path=live_path, transcribe_fn=tx, cadence_s=0.6)
        spoken = []
        wa = cl.WordAssist(read_fn=lambda: lw.read_current(path=live_path, max_age_s=10.0),
                           suggest_fn=lambda p, c: ws.suggest(p, c, model=a.model, glossary=gloss, timeout=a.llm_timeout),
                           speak_fn=lambda t: spoken.append((time.time(), t)), gate_fn=lambda: (True, ""),
                           offers=cl.Offers(), log=lambda m: print("   ", m))
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                wa.step(); time.sleep(0.15)
        th = threading.Thread(target=watch, daemon=True); th.start()
        t0 = time.time()
        for i in range(0, len(audio), 512):           # real-time paced, like the capture tap
            L.feed(audio[i:i + 512])
            dt = t0 + (i + 512) / SR - time.time()
            if dt > 0:
                time.sleep(dt)
        speech_end = t0 + len(speech) / SR
        time.sleep(0.3); stop.set(); th.join(2); L.stop()
        off = spoken[0] if spoken else None
        r = {"text": text, "partial": L.last_text, "offer": off[1] if off else None,
             "latency_from_speech_end_s": round(off[0] - speech_end, 2) if off else None,
             "expect_offer": expect,
             "ok": (bool(off) == expect) and (not expect or any(x in off[1].lower() for x in accept))}
        r["gpu_util"] = _gpu_util()
        print(json.dumps(r)); results.append(r)
    urllib.request.urlopen(urllib.request.Request(ws.OLLAMA_URL, data=json.dumps({"model": a.model, "keep_alive": 0}).encode(),
                                                  headers={"Content-Type": "application/json"}), timeout=30).read()
    lat = sorted(r["latency_from_speech_end_s"] for r in results if r["latency_from_speech_end_s"] is not None)
    print(json.dumps({"passed": sum(r["ok"] for r in results), "of": len(results), "offer_latency_s": lat}))


if __name__ == "__main__":
    main()
