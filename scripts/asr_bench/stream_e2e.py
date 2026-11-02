"""Offline end-to-end check of the ears' streaming path against a RUNNING sidecar (never the live ears).

Run in the EARS venv (iris-v100) with IRIS_STREAM_ASR_SOCK pointing at a test sidecar:
  IRIS_STREAM_ASR_SOCK=/path/test.sock python scripts/asr_bench/stream_e2e.py session [N]
      every clip through iris_stream_asr.StreamSession in REAL time (100 ms frames, sleep between),
      then flush: final latency (client-measured), first-partial time, text vs the offline bench.
  IRIS_STREAM_ASR_SOCK=... python scripts/asr_bench/stream_e2e.py core [N]
      N clips through wren_voice_core._capture_utterance (fake mic playing the clip in real time,
      0.5 s of room tone before, 1.2 s after) + _transcribe_stream_turn, exactly as an in-call listen:
      sidecar partials, the once-per-pause whisper prosody pass, final within stream_asr_final_ms,
      prosody alignment + enrich. Loads whisper-small (~1 GB VRAM) in this process.
Prints numbers + text; audio and transcripts stay out of git (results go to /data/asr_bench/results)."""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voice"))
import iris_stream_asr as sasr  # noqa: E402

CLIPS = sorted(glob.glob(os.environ.get("CLIPS_GLOB", "/data/asr_bench/clips/*.wav")))
BENCH = json.load(open("/data/asr_bench/results/nemotron-speech-streaming-en-0.6b_rc1_fp32.json"))["clips"]
WHISPER = json.load(open("/data/asr_bench/results/whisper_small_b1.json"))["clips"]


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def session_mode(n):
    fin_ms, first, same, out = [], [], 0, {}
    for c in CLIPS[:n]:
        a, sr = sf.read(c, dtype="float32")
        assert sr == 16000
        parts = []
        t0 = time.time()
        s = sasr.StreamSession(on_partial=lambda t: parts.append((time.time() - t0, t)), source="e2e")
        for i in range(0, len(a), 1600):
            s.feed(a[i:i + 1600])
            time.sleep(max(0.0, t0 + (i + 1600) / 16000 - time.time()))
        f = s.flush(1.0)
        s.close()
        b = os.path.basename(c)
        txt = (f or {}).get("raw_text", "")
        same += txt == BENCH.get(b, {}).get("text")
        fin_ms.append((f or {}).get("client_ms", 1e9))
        if parts:
            first.append(parts[0][0])
        out[b] = {"text": (f or {}).get("text"), "client_ms": (f or {}).get("client_ms"),
                  "server_flush_ms": (f or {}).get("flush_ms"), "n_partials": len(parts),
                  "first_partial_s": parts[0][0] if parts else None}
        print(f"{(f or {}).get('client_ms', 'NONE'):>6}ms parts={len(parts):2d} {b[4:20]} {str((f or {}).get('text'))[:70]}",
              flush=True)
    print(f"session: {len(fin_ms)} clips, raw text identical to offline bench {same}/{len(fin_ms)}; "
          f"final after flush p50 {pct(fin_ms, 50):.0f} / p95 {pct(fin_ms, 95):.0f} / max {max(fin_ms):.0f} ms; "
          f"within 150 ms: {sum(x <= 150 for x in fin_ms)}/{len(fin_ms)}; first partial p50 {pct(first, 50):.2f}s")
    json.dump(out, open("/data/asr_bench/results/stream_e2e_session.json", "w"), indent=1)


class _Mic:
    def __init__(self, audio):
        self.a = audio
        self.i = 0
        self.t0 = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def read(self, n):
        due = self.t0 + (self.i + n) / 16000
        time.sleep(max(0.0, due - time.time()))
        f = self.a[self.i:self.i + n]
        self.i += n
        if f.size < n:
            f = np.concatenate([f, np.zeros(n - f.size, np.float32)])
        return f.reshape(-1, 1), False


def core_mode(n):
    import wren_voice_core as core
    import wren_voice_status
    wren_voice_status.read_control = lambda: {"stream_asr": "on"}
    core.set_state = lambda *a, **k: None
    core.is_muted = lambda: False
    core._trusted_mic_or_none = lambda ctx: "net"
    core.wl.get_whisper()

    class Ctx:
        mic_dev_idx = "net"
        speaking = False
    rows, out = [], {}
    rng = np.random.default_rng(0)
    for c in CLIPS[:n]:
        a, _ = sf.read(c, dtype="float32")
        tone = lambda s: (0.002 * rng.standard_normal(int(16000 * s))).astype(np.float32)  # noqa: E731
        audio_in = np.concatenate([tone(0.5), a, tone(1.2)])
        mic = _Mic(audio_in)
        core._net_audio.open_mic = lambda sd, sr, idx, bs, m=mic: m
        sctl = core._stream_ctl()
        stream_text = core._StreamingPartials(live=None)
        st = core._StreamFeed("listen", on_partial=lambda t: stream_text.update(t, None, 0, 0.0),
                              preroll_s=sctl["preroll_s"])
        p = core._StreamingPartials(live=None)
        wout, timing = {}, {}
        audio = core._capture_utterance(Ctx(), 5.0, 30.0, 1.1, partials=p,
                                        transcribe_fn=core._prosody.transcribe, asr_stream=st,
                                        prosody_on_pause=True, text_partials=stream_text,
                                        worker_out=wout, timing=timing)
        st.t_end = time.time()
        text, words = core._transcribe_stream_turn(audio, st, p, wout.get("thread"), sctl["final_s"])
        turn_ms = (time.time() - st.t_end) * 1000
        marked = core._prosody.enrich(audio, words) if words else ""
        b = os.path.basename(c)
        rec = json.loads(Path(core.STREAM_ASR_LOG).read_text().splitlines()[-1])
        rows.append((turn_ms, rec.get("used"), rec.get("prosody")))
        out[b] = {"text": text, "turn_ms": round(turn_ms, 1), "rec": rec, "marked": marked,
                  "whisper_small_offline": WHISPER.get(b, {}).get("text")}
        print(f"turn {turn_ms:5.0f}ms used={rec.get('used')} prosody={rec.get('prosody')} {b[4:20]}\n"
              f"    text   {text[:90]!r}\n    marked {marked.splitlines()[0][:90] if marked else ''!r}", flush=True)
    t = [r[0] for r in rows]
    print(f"core: {len(rows)} turns; text ready after capture end p50 {pct(t, 50):.0f} / p95 {pct(t, 95):.0f} / "
          f"max {max(t):.0f} ms; used stream {sum(r[1] == 'stream' for r in rows)}/{len(rows)}; prosody aligned "
          f"{sum(r[2] == 'aligned' for r in rows)}/{len(rows)}")
    json.dump(out, open("/data/asr_bench/results/stream_e2e_core.json", "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "session"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(CLIPS)
    print("ping", sasr.ping(timeout=2.0), flush=True)
    (core_mode if mode == "core" else session_mode)(n)
