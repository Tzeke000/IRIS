"""faster-whisper on the clip set, same params as the live final pass (fp16, beam 1, en, word_timestamps).
Measures decode wall time after the full clip is available (= 'final after end of audio' for whisper)."""
import sys, json, time, glob, os, numpy as np, soundfile as sf
from faster_whisper import WhisperModel
import pynvml
name, beam = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
def proc_mem():
    pid = os.getpid()
    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
        if p.pid == pid: return p.usedGpuMemory / 2**20
    return None
m = WhisperModel(name, device="cuda", compute_type="float16")
m.transcribe(np.zeros(16000, np.float32), beam_size=1, language="en")
clips = sorted(glob.glob("/data/asr_bench/clips/*.wav"))
out = {}
cpu0 = time.process_time()
for c in clips:
    a, sr = sf.read(c, dtype="float32"); assert sr == 16000
    t = time.perf_counter()
    segs, _ = m.transcribe(a, beam_size=beam, language="en", word_timestamps=True)
    segs = list(segs)
    text = " ".join(s.text.strip() for s in segs).strip()
    ws = [w for s in segs for w in (s.words or [])]
    dt = time.perf_counter() - t
    out[os.path.basename(c)] = {"text": text, "final_s": dt, "dur": len(a) / sr,
                                  "onset_s": ws[0].start if ws else None, "last_word_end_s": ws[-1].end if ws else None,
                                  "words": [[w.word, round(w.start, 3), round(w.end, 3)] for w in ws]}
    print(f"{dt*1000:6.0f}ms {os.path.basename(c)[4:20]} {text[:90]}", flush=True)
tag = f"whisper_{name.split('/')[-1]}_b{beam}"
json.dump({"model": tag, "vram_mib": proc_mem(), "cpu_s": time.process_time() - cpu0, "clips": out},
          open(f"/data/asr_bench/results/{tag}.json", "w"), indent=1)
print("VRAM MiB", proc_mem())
