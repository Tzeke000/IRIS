"""Bench the fast layer's word-finding on the V100 (Ollama): latency + accuracy + VRAM per model.

usage: python scripts/convo/word_search_bench.py [model ...]   (default: qwen2.5:0.5b llama3.2:3b qwen2.5:7b)
Each model is unloaded afterwards (keep_alive 0). Writes nothing under state/.
"""
import json, statistics, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))
import iris_word_search as ws
import unicodedata
fold = lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode().lower()
GLOSS = ws.load_glossary() if '--glossary' in sys.argv else None

CASES = [  # (context lines, partial, accepted answers)
    (["Zeke: I want the city to have more trees", "Iris: I can scatter them procedurally"],
     "yeah do it with the uh, the node thing in blender, what's it called", ["geometry nodes"]),
    (["Zeke: my head hurts from all the screens"],
     "you know that guy who painted the melting clocks, what's his name", ["dali", "salvador dali"]),
    (["Iris: the mouth is the Fish model now"],
     "the old one we used before Fish, the British one, I can't remember the name", ["styletts2", "styletts", "style tts", "kokoro"]),
    (["Zeke: we're putting the server in the closet"],
     "the thing that keeps the power on when the power goes out, what do you call it", ["ups", "uninterruptible power supply", "battery backup"]),
    ([], "it's the capital of australia, it's on the tip of my tongue", ["canberra"]),
    (["Zeke: I'm cooking tonight"],
     "I need that green sauce with basil and pine nuts, what's it called", ["pesto"]),
    (["Iris: whisper hears you"],
     "the thing that turns my voice into text, what's the word", ["speech to text", "stt", "speech recognition", "transcription", "whisper"]),
    (["Zeke: I'm building a VR world for you"],
     "the game engine we're using, the one by epic, what's it called", ["unreal", "unreal engine"]),
    (["Zeke: the robot is charging"],
     "the little robot with the forklift arms, what's his name", ["vector"]),
    ([], "you know when you can't sleep at night at all, what's the word", ["insomnia"]),
    (["Zeke: I fly the drone at the park"],
     "and the thing that films it, the camera holder that keeps it steady, what's it called", ["gimbal"]),
    ([], "the opposite of a predator, what's the word", ["prey"]),
]

def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])

def _ps_vram(model):
    ps = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5).read())
    return sum(m.get("size_vram", 0) for m in ps.get("models", []) if m.get("name") == model) // 2**20

def unload(model):
    try:
        req = urllib.request.Request(ws.OLLAMA_URL, data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print("unload failed", e)

def run(model):
    before = gpu_used()
    t0 = time.time()
    _, cold = ws.suggest("warm up, what's it called", [], model=model, timeout=120, keep_alive="2m")
    vram = before and _ps_vram(model)
    lat, ok, rows = [], 0, []
    for ctx, partial, accept in CASES:
        word, dt = ws.suggest(partial, ctx, model=model, timeout=30, keep_alive="2m", glossary=GLOSS)
        hit = bool(word) and any(a in fold(word) for a in accept)
        ok += hit; lat.append(dt); rows.append((word, round(dt * 1000), hit))
    unload(model)
    lat.sort()
    res = {"model": model, "glossary": bool(GLOSS), "cold_s": round(cold, 2), "vram_mib": vram,
           "p50_ms": round(statistics.median(lat) * 1000), "p90_ms": round(lat[int(0.9 * (len(lat) - 1))] * 1000),
           "max_ms": round(lat[-1] * 1000), "acc": f"{ok}/{len(CASES)}", "rows": rows}
    print(json.dumps(res))
    return res

if __name__ == "__main__":
    for m in ([a for a in sys.argv[1:] if not a.startswith('--')] or ["qwen2.5:0.5b", "llama3.2:3b", "qwen2.5:7b"]):
        run(m)
