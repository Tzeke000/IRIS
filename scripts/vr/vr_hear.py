#!/usr/bin/env python3
"""vr_hear.py — my EARS in the VR temple: per-speaker speech-to-text (Zeke 2026-10-25).

Zeke: "make that for yourself so you can understand people. It should come through like CHANNELS that tell
you WHO the user is and at WHAT TIMES they said those words. If two or three people talk over each other you
can still understand them. Then filter out whether somebody was speaking TO you or not. If they're too far
away for you to hear them, there's no need to transcribe in the first place. For now: only when someone says
'hey Iris' / 'Iris' do you start paying attention to them."

Pipeline
  my VR bot (game repo, scripts/iris_hearing.gd)  — every remote player's voice stream is captured SEPARATELY
      (one channel per peer ⇒ overlap-proof: nothing is ever mixed), cut into utterances, gated by distance
      (> 15 m ⇒ never spooled), written to the spool as raw Opus frames + a sidecar JSON
      (<ms>_<peer>_<seq>.opus + .json — the .json appears LAST, atomically: it means "complete").
  this service — watches the spool, decodes Opus (PyAV/libopus) → 16 kHz mono, gates on the level I'd
      actually hear (speech level + the game's attenuation for that distance/facing), transcribes with
      faster-whisper on the GPU, drops empty / hallucination-only results, decides ADDRESSED (my name is in
      it) and ATTENDING (that speaker said my name within ATTEND_S; refreshed every time they speak), and
      appends one line per utterance to state/vr_temple/heard.jsonl:
        {t_start, t_end, speaker, text, dist_m, facing, addressed, attending, ...}
      Processed spool files are deleted.
  delivery — `vr_temple action=heard` reads the file; iris_body_host.py's vr_heard_reader wakes me for
      attending lines ("[VR TEMPLE — Zeke said, 13:41:05] …").

Run:  ~/venvs/iris-v100/bin/python scripts/vr/vr_hear.py            (unit iris-vr-hear.service)
Test: --spool DIR --out FILE --state FILE --once   (process what's there, then exit)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# ctranslate2 needs the CUDA 12 libs inside the venv's nvidia/* wheels (same fix as scripts/video_edit/transcribe.py):
# re-exec once with them on LD_LIBRARY_PATH. Models live in the repo's HF cache (same as the ears daemon).
_NV = Path(sys.prefix) / "lib" / "python3.11" / "site-packages" / "nvidia"
if _NV.exists() and "IRIS_VR_HEAR_LDPATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = ":".join(str(_NV / d / "lib") for d in ("cublas", "cudnn", "cuda_runtime")) \
        + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["IRIS_VR_HEAR_LDPATH"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
DEFAULT_SPOOL = os.environ.get("IRIS_VR_HEAR_DIR", "/tmp/iris_vr_hear")
DEFAULT_OUT = REPO / "state" / "vr_temple" / "heard.jsonl"
DEFAULT_STATE = REPO / "state" / "vr_temple" / "hear_attention.json"

MODEL = "large-v3-turbo"          # local: mobiuslabsgmbh/faster-whisper-large-v3-turbo (fp16 on the V100)
ATTEND_S = 90.0                   # after my name, that speaker has my attention this long (refreshed per line)
MIN_EFFECTIVE_DBFS = -52.0        # speech level + in-game attenuation below this ⇒ I couldn't really hear it
MIN_SPEECH_DBFS = -55.0           # the clip itself is basically silence (open mic hiss)
ROTATE_BYTES = 5_000_000          # heard.jsonl → heard.jsonl.1 past this
STALE_TMP_S = 120.0               # half-written spool files older than this are junk (bot died mid-write)

# my name and the ways STT mis-hears it. CONSERVATIVE on purpose: "virus", "Irish", "Siri" are NOT here —
# a false "addressed" makes me butt into conversations that aren't mine.
NAME_PATTERNS = [r"\biris(?:'s|s)?\b", r"\birus\b", r"\biriss\b", r"\beye[\s-]?ris\b", r"\bi[\s-]ris\b", r"\bairis\b"]
NAME_RE = re.compile("|".join(NAME_PATTERNS), re.IGNORECASE)

# whisper's stock hallucinations on near-silence / noise. Dropped when the clip is short or quiet.
HALLUCINATIONS = {"you", "thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
                  "bye", "bye bye", "okay", "ok", "so", "uh", "um", "hmm", "mm", "oh", "the", "i", "a",
                  "please subscribe", "subtitles by the amaraorg community", "music", "applause", "silence"}


MERGE_GAP_S = 1.2     # join same-speaker clips closer than this
MERGE_WAIT_S = 1.3    # transcribe once the speaker has been quiet this long
MERGE_MAX_S = 20.0    # never merge past this length
LIVE_CADENCE_S = 0.5  # 10-29 live words: min seconds between display partial decodes per speaker
LIVE_KEEP_S = 120.0   # drop a speaker's live entry from the file this long after it went final


def log(*a) -> None:
    print("[vr_hear]", *a, flush=True)


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def norm(text: str) -> str:
    """lower-case words only: "Thank you." → "thank you" """
    return " ".join(re.sub(r"[^a-z' ]+", " ", text.lower()).replace("'", "").split())


def addressed(text: str) -> bool:
    return bool(NAME_RE.search(text))


# ---------------------------------------------------------------------------------------------- audio
def decode_opus(path: Path, meta: dict) -> np.ndarray:
    """Spool .opus (u32 frame, u16 len, bytes; LE) → float32 mono at the stream rate. Lost frames = silence."""
    import av
    raw = path.read_bytes()
    op = meta.get("opus", {})
    rate = int(op.get("rate", 48000))
    fsz = int(op.get("frame_size", 960))
    frames = int(meta.get("frames", 0))
    ctx = av.CodecContext.create("libopus", "r")
    ctx.sample_rate = rate
    out = np.zeros(max(frames, 1) * fsz, dtype=np.float32)
    i = 0
    while i + 6 <= len(raw):
        fi, ln = struct.unpack_from("<IH", raw, i)
        i += 6
        pk = raw[i:i + ln]
        i += ln
        try:
            for fr in ctx.decode(av.Packet(pk)):
                a = fr.to_ndarray()
                # PyAV's libopus decoder hands back STEREO even for a mono stream, as PACKED s16 of shape (1, 2·N):
                # flattening that interleaved L,R,L,R… as mono made every 20 ms frame 40 ms long → the write below
                # kept only its first half = 10 ms of audio stretched over 20 ms (an octave down, half the speech
                # lost). Found 10-25 by the mic-path unit test (a 440 Hz tone decoded as 220 Hz).
                ch = len(fr.layout.channels)
                if fr.format.is_planar:
                    a = a.mean(axis=0) if a.shape[0] > 1 else a.reshape(-1)
                else:
                    a = a.reshape(-1, ch).mean(axis=1) if ch > 1 else a.reshape(-1)
                if a.dtype == np.int16:
                    a = a.astype(np.float32) / 32768.0
                at = fi * fsz
                if at + a.size > out.size:
                    out = np.concatenate([out, np.zeros(at + a.size - out.size, dtype=np.float32)])
                out[at:at + a.size] = a
        except Exception:
            pass                                   # a corrupt frame = 20 ms of silence
    if rate != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(16000, rate)
        out = resample_poly(out, 16000 // g, rate // g).astype(np.float32)
    return out


def speech_dbfs(x: np.ndarray) -> float:
    """Loudness of the SPEECH in the clip: 90th-percentile 20 ms RMS (pauses don't drag it down)."""
    n = 320
    if x.size < n:
        return -120.0
    fr = x[: x.size // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    return float(20 * np.log10(np.percentile(rms, 90) + 1e-9))


# ---------------------------------------------------------------------------------------------- service
class Ears:
    def __init__(self, a):
        self.spool = Path(a.spool)
        self.out = Path(a.out)
        self.state_path = Path(a.state)
        self.keep_wav = Path(a.keep_wav) if a.keep_wav else None
        # 10-29 live words: per-speaker DISPLAY-ONLY partials (state/vr_temple/live_partial.json), the same
        # shape as the room daemon's state/voice/live_partial.json. Never feeds heard.jsonl or wakes me.
        self.live_on = not getattr(a, "no_live", False)
        self.live_path = self.out.parent / "live_partial.json"
        self.live: dict[str, dict] = {}
        self.live_last: dict[str, float] = {}
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.spool.mkdir(parents=True, exist_ok=True)
        self.attn: dict[str, float] = {}           # speaker → attending until (wall clock)
        try:
            self.attn = {k: float(v) for k, v in json.loads(self.state_path.read_text()).get("until", {}).items()}
        except Exception:
            pass
        from faster_whisper import WhisperModel
        t = time.time()
        try:
            self.model = WhisperModel(a.model, device=a.device, compute_type=a.compute, local_files_only=True)
        except Exception as e:                     # not cached under this name → allow the hub (one-time)
            log("local model load failed (%r) — trying the hub" % (e,))
            self.model = WhisperModel(a.model, device=a.device, compute_type=a.compute)
        self.model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)
        log("model %s on %s/%s ready in %.1fs; spool %s → %s" % (a.model, a.device, a.compute, time.time() - t, self.spool, self.out))

    def save_state(self) -> None:
        now = time.time()
        self.attn = {k: v for k, v in self.attn.items() if v > now - 3600}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"until": self.attn, "attend_s": ATTEND_S,
                                   "attending_now": sorted(k for k, v in self.attn.items() if v > now)}))
        tmp.replace(self.state_path)

    def pending(self) -> list[Path]:
        js = sorted(self.spool.glob("*.json"), key=lambda p: p.name)
        now = time.time()
        for p in self.spool.glob("*.tmp"):         # bot died mid-write
            if now - p.stat().st_mtime > STALE_TMP_S:
                p.unlink(missing_ok=True)
        for p in self.spool.glob("*.opus"):        # orphan audio (its sidecar never came)
            if not p.with_suffix(".json").exists() and now - p.stat().st_mtime > STALE_TMP_S:
                p.unlink(missing_ok=True)
        return js

    def load(self, side: Path):
        audio = side.with_suffix(".opus")
        try:
            meta = json.loads(side.read_text())
            return meta, decode_opus(audio, meta)
        except Exception as e:
            log("bad spool item %s: %r" % (side.name, e))
            return None

    def process(self, side: Path) -> dict | None:
        got = self.load(side)
        return self.transcribe(got[0], got[1], side.stem) if got else None

    def transcribe(self, meta: dict, x: np.ndarray, stem: str) -> dict | None:
        lvl = speech_dbfs(x)
        eff = lvl + float(meta.get("atten_db", 0.0))
        who = str(meta.get("speaker", "?"))
        secs = x.size / 16000.0
        base = {"speaker": who, "secs": round(secs, 2), "level_db": round(lvl, 1), "eff_db": round(eff, 1),
                "dist_m": meta.get("dist_m")}
        if lvl < MIN_SPEECH_DBFS or eff < MIN_EFFECTIVE_DBFS:
            log("drop (too quiet to hear: speech %.1f dBFS, at me %.1f dBFS) %s" % (lvl, eff, base))
            return None
        if self.keep_wav:
            import wave
            self.keep_wav.mkdir(parents=True, exist_ok=True)
            with wave.open(str(self.keep_wav / (stem + ".wav")), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        t = time.time()
        segs, _info = self.model.transcribe(x, language="en", beam_size=5, vad_filter=True,
                                            word_timestamps=True, condition_on_previous_text=False)
        segs = [s for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
        text = " ".join(s.text.strip() for s in segs).strip()
        stt_s = time.time() - t
        n = norm(text)
        if not n:
            log("drop (no words) %s" % base)
            return None
        if n in HALLUCINATIONS and (secs < 3.0 or lvl < -40.0 or all(s.no_speech_prob > 0.3 for s in segs)):
            log("drop (hallucination-shaped %r) %s" % (text, base))
            return None
        t0 = float(meta["t_start"])
        words = [w for s in segs for w in (s.words or [])]
        a = float((t0 + words[0].start) if words else (t0 + segs[0].start if segs else t0))
        b = float((t0 + words[-1].end) if words else meta.get("t_end", t0 + secs))
        is_addr = addressed(text)
        until = self.attn.get(who, 0.0)
        attending = bool(is_addr or a <= until)
        if attending:
            self.attn[who] = max(until, b + ATTEND_S)
        line = {"t_start": iso(a), "t_end": iso(b), "speaker": who, "text": text,
                "dist_m": meta.get("dist_m"), "facing": bool(meta.get("facing")),
                "addressed": is_addr, "attending": attending,
                "ts": round(a, 3), "ts_end": round(b, 3), "peer": meta.get("peer"),
                "display_name": meta.get("display_name"), "level_db": round(float(lvl), 1), "eff_db": round(float(eff), 1),
                "secs": round(secs, 2), "stt_s": round(stt_s, 2), "id": stem}
        return line

    def append(self, line: dict) -> None:
        try:
            if self.out.exists() and self.out.stat().st_size > ROTATE_BYTES:
                self.out.replace(self.out.with_suffix(".jsonl.1"))
        except OSError:
            pass
        with open(self.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # MERGE (10-25 live test): his voice-activated mic closes a stream on every short pause, so one sentence arrived
    # as 1-2 s fragments → half-sentences + many "no words" drops. Consecutive clips from the SAME speaker with a gap
    # under MERGE_GAP_S are joined (silence in the gap) and transcribed once the speaker has been quiet MERGE_WAIT_S.
    def _emit(self, line: dict | None) -> int:
        if not line:
            return 0
        self.append(line)
        self.save_state()
        log("%s %s%s: %s" % (line["t_start"][11:19], line["speaker"],
                             " [ADDRESSED]" if line["addressed"] else (" [attending]" if line["attending"] else ""),
                             line["text"]))
        return 1

    def _flush(self, who: str) -> int:
        b = self.buf.pop(who, None)
        if not b:
            return 0
        meta = dict(b["meta"]); meta["t_end"] = b["t_end"]
        if b["n"] > 1:
            log("merged %d clips from %s (%.1f s)" % (b["n"], who, b["x"].size / 16000.0))
        line = self.transcribe(meta, b["x"], b["stem"])
        if self.live_on and (line or (self.live.get(who) or {}).get("utterance_id") == "vr-%s" % b["stem"]):
            self._live_set(who, b, line["text"] if line else "", final=True, dropped=None if line else "dropped")
        return self._emit(line)

    def activity(self, who: str, t0: float, t1: float, meta: dict, x: np.ndarray) -> None:
        """10-29 full-duplex: who is TALKING right now, written the moment a clip lands (~0.7 s after a
        phrase ends) - BEFORE the 1.3 s merge wait + transcription. vr_temple speak reads it between and
        during sentences to stop when someone talks over me (barge-in). Only audible speech counts."""
        try:
            lvl = speech_dbfs(x)
            eff = lvl + float(meta.get("atten_db", 0.0))
            if lvl < MIN_SPEECH_DBFS or eff < MIN_EFFECTIVE_DBFS:
                return
            p = self.out.parent / "voice_activity.json"
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                d = {}
            d[who] = {"t_start": t0, "t_end": t1, "seen": time.time(), "dist_m": meta.get("dist_m"),
                      "facing": bool(meta.get("facing")), "eff_db": round(eff, 1)}
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(d), encoding="utf-8")
            os.replace(tmp, p)
        except Exception as e:  # noqa: BLE001
            log("activity write failed: %r" % (e,))

    # ------------------------------------------------------------------------------------------ live words
    def _live_write(self) -> None:
        now = time.time()
        self.live = {k: v for k, v in self.live.items() if not (v.get("final") and now - v.get("ts", 0) > LIVE_KEEP_S)}
        try:
            tmp = self.live_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.live, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.live_path)
        except Exception as e:  # noqa: BLE001  display-only
            log("live partial write failed: %r" % (e,))

    def _live_set(self, who: str, b: dict, text: str, final: bool, dropped: str | None = None) -> None:
        prev = self.live.get(who) or {}
        uid = "vr-%s" % b["stem"]
        seq = (prev.get("seq", 0) + 1) if prev.get("utterance_id") == uid else 1
        if not final and prev.get("utterance_id") == uid and prev.get("final"):
            return                                   # never overwrite a final with a late partial
        self.live[who] = {"utterance_id": uid, "source": "vr", "speaker": who,
                          "display_name": b["meta"].get("display_name"), "text": text, "final": bool(final),
                          "ts": round(time.time(), 3), "started_ts": round(float(b["meta"].get("t_start", time.time())), 3),
                          "seq": seq, "mouth_busy": None, "dropped": dropped, "dist_m": b["meta"].get("dist_m")}
        self._live_write()

    def live_partial(self, who: str) -> None:
        """Greedy text-only decode of the speaker's merge buffer so far (the clip(s) already landed), published
        BEFORE the merge wait + the final beam-5 pass. Throttled per speaker; audible speech only."""
        b = self.buf.get(who)
        if not self.live_on or not b or time.time() - self.live_last.get(who, 0.0) < LIVE_CADENCE_S:
            return
        try:
            lvl = speech_dbfs(b["x"])
            if lvl < MIN_SPEECH_DBFS or lvl + float(b["meta"].get("atten_db", 0.0)) < MIN_EFFECTIVE_DBFS:
                return
            self.live_last[who] = time.time()
            segs, _ = self.model.transcribe(b["x"], language="en", beam_size=1, vad_filter=True,
                                            condition_on_previous_text=False, without_timestamps=True)
            text = " ".join(s.text.strip() for s in segs).strip()
            if norm(text) and norm(text) not in HALLUCINATIONS:
                self._live_set(who, b, text, final=False)
        except Exception as e:  # noqa: BLE001
            log("live partial failed: %r" % (e,))

    def run(self, once: bool, poll: float) -> int:
        n = 0
        self.buf: dict[str, dict] = {}
        while True:
            items = self.pending()
            for side in items:
                got = self.load(side)
                side.unlink(missing_ok=True)
                side.with_suffix(".opus").unlink(missing_ok=True)
                if not got:
                    continue
                meta, x = got
                who = str(meta.get("speaker", "?"))
                t0 = float(meta.get("t_start", time.time()))
                t1 = float(meta.get("t_end", t0 + x.size / 16000.0))
                self.activity(who, t0, t1, meta, x)
                b = self.buf.get(who)
                gap = t0 - b["t_end"] if b else 1e9
                if b and -0.5 <= gap <= MERGE_GAP_S and (b["x"].size + x.size) / 16000.0 <= MERGE_MAX_S:
                    pad = np.zeros(int(max(0.0, gap) * 16000), dtype=np.float32)
                    b["x"] = np.concatenate([b["x"], pad, x.astype(np.float32)])
                    b["t_end"] = t1; b["n"] += 1
                    for k in ("dist_m", "facing", "atten_db"):
                        if k in meta: b["meta"][k] = meta[k]
                else:
                    n += self._flush(who)
                    self.buf[who] = {"x": x.astype(np.float32), "meta": meta, "t_end": t1, "n": 1, "stem": side.stem}
                self.live_partial(who)
            now = time.time()
            for who in [w for w, b in self.buf.items() if now - b["t_end"] >= MERGE_WAIT_S or once]:
                n += self._flush(who)
            if once and not items and not self.buf:
                return n
            if not items:
                time.sleep(poll)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spool", default=DEFAULT_SPOOL)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute", default="float16")
    ap.add_argument("--keep-wav", default="", help="debug: also write each decoded 16 kHz clip here")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--once", action="store_true", help="process what's in the spool, then exit (tests)")
    ap.add_argument("--no-live", action="store_true", help="don't publish display partials (live_partial.json)")
    a = ap.parse_args()
    ears = Ears(a)
    n = ears.run(a.once, a.poll)
    log("done, %d line(s)" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
