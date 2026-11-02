"""Cache-aware streaming bench (Nemotron streaming / Parakeet EOU) on the clip set, simulated real-time feed.
Each step's audio becomes 'available' when wall time reaches the end of that chunk's audio (capped at clip end);
a step starts at max(available, previous step done). Latencies are on that simulated clock:
  first_partial_s  = clip-time when the first non-empty partial is ready
  final_after_end_s = ready-time of the final transcript minus clip duration (end of audio)
  step_ms          = GPU compute per chunk
usage: bench_nemo.py MODEL_ID RIGHT_CTX AMP(0|1) [cuda_graphs 0|1]"""
import sys, os, json, time, glob, numpy as np, soundfile as sf, torch
import pynvml
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from omegaconf import open_dict

mid, rc, amp = sys.argv[1], int(sys.argv[2]), bool(int(sys.argv[3]))
graphs = len(sys.argv) > 4 and bool(int(sys.argv[4]))
PRE = float(os.environ.get("PREPAD_S", "0"))  # leading zeros = stream already running before onset
PAD = float(os.environ.get("PAD_S", "0"))  # extra trailing zeros = the mic keeps running after he stops
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def proc_mem():
    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(H):
        if p.pid == os.getpid(): return p.usedGpuMemory / 2**20

m = nemo_asr.models.ASRModel.from_pretrained(mid, map_location="cuda")
if hasattr(m.encoder, "set_default_att_context_size") and rc >= 0:
    m.encoder.set_default_att_context_size([70, rc])
dc = m.cfg.decoding
with open_dict(dc):
    dc.strategy = "greedy_batch"; dc.fused_batch_size = -1
    dc.greedy.max_symbols = 10
    dc.compute_timestamps = False; dc.preserve_alignments = False  # NeMo 3.0 merge_ bug with dict timestamps (EOU cfg)
    if "loop_labels" in dc.greedy: dc.greedy.loop_labels = True
m.change_decoding_strategy(dc)
m = m.to("cuda", dtype=torch.float32).eval()
if graphs and hasattr(m.encoder, "set_streaming_cuda_graphs"):
    m.encoder.set_streaming_cuda_graphs(enabled=True)
scfg = m.encoder.streaming_cfg
print("streaming_cfg", scfg, flush=True)

def texts(t):
    t = t[0] if isinstance(t, (list, tuple)) and t and isinstance(t[0], (list, tuple)) else t
    return [h.text if isinstance(h, Hypothesis) else h for h in t]

def run(audio):
    buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False, pad_and_drop_preencoded=False)
    buf.append_audio(audio, stream_id=-1)
    cc, ct, ccl = m.encoder.get_initial_cache_state(batch_size=1)
    prev, pred = None, None
    dur = len(audio) / 16000.0
    clock, first, steps, partials = 0.0, None, [], []
    for i, (chunk, lens) in enumerate(iter(buf)):
        # mel frames consumed so far -> audio time this step needs (10 ms hop)
        need = min(buf.buffer_idx * 0.01, dur)
        start = max(need, clock)
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred, tx, cc, ct, ccl, prev = m.conformer_stream_step(
                processed_signal=chunk, processed_signal_length=lens,
                cache_last_channel=cc, cache_last_time=ct, cache_last_channel_len=ccl,
                keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=prev,
                previous_pred_out=pred, drop_extra_pre_encoded=(0 if i == 0 else scfg.drop_extra_pre_encoded),
                return_transcription=True)
        torch.cuda.synchronize(); c = time.perf_counter() - t
        clock = start + c; steps.append(c)
        txt = texts(tx)[0] or ""
        partials.append((round(clock, 3), txt))
        if first is None and txt.strip(): first = clock
    return txt.strip(), first, clock - dur, steps, partials

clips = sorted(glob.glob(os.environ.get("CLIPS_GLOB", "/data/asr_bench/clips/*.wav")))
w, _ = sf.read(clips[0], dtype="float32"); run(w); run(w)  # warm
torch.cuda.reset_peak_memory_stats()
out, allsteps = {}, []
cpu0 = time.process_time()
for c in clips:
    a, sr = sf.read(c, dtype="float32"); assert sr == 16000
    if PAD: a = np.concatenate([a, np.zeros(int(PAD * 16000), np.float32)])
    if PRE: a = np.concatenate([np.zeros(int(PRE * 16000), np.float32), a])
    text, first, fin, steps, parts = run(a)
    allsteps += steps
    out[os.path.basename(c)] = {"text": text, "first_partial_s": first, "final_after_end_s": fin,
                                "dur": len(a) / 16000, "partials": parts}
    print(f"fin {fin*1000:5.0f}ms first {first if first is None else round(first,2)} {os.path.basename(c)[4:20]} {text[:90]}", flush=True)
tag = f"{mid.split('/')[-1]}_rc{rc}_{'amp16' if amp else 'fp32'}{'_graphs' if graphs else ''}{f'_pad{PAD:g}' if PAD else ''}{f'_pre{PRE:g}' if PRE else ''}{'_adhoc' if 'CLIPS_GLOB' in os.environ else ''}"
chunk_ms = (rc + 1) * 80
res = {"model": tag, "chunk_ms": chunk_ms, "vram_mib": proc_mem(),
       "torch_infer_peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
       "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
       "step_ms_p50": float(np.percentile(allsteps, 50) * 1000), "step_ms_p95": float(np.percentile(allsteps, 95) * 1000),
       "cpu_s": time.process_time() - cpu0, "clips": out}
json.dump(res, open(f"/data/asr_bench/results/{tag}.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "clips"})
