"""wren_fish_server.py — warm Fish Audio S2 Pro mouth (side by side with the StyleTTS2 mouth).

Same HTTP interface as voice/wren_styletts_server.py, so anything that talks to the live
mouth can talk to this one by changing only the port:
    POST /        speak text. Body = raw text, or JSON {"text", "emotion"?, "intensity"?,
                  "plain"?, "seed"?}. Plays through the same audio-out layer (IRIS_AUDIO_SINK
                  network sink, or the local device), publishes the amplitude envelope.
    POST /synth   JSON {"text", "rate"=8000, "emotion"?, "intensity"?, "seed"?} -> audio/wav
                  mono int16 at `rate`, WITHOUT playing anything.
    GET  /health  "ok" (200) once warm, "loading" (503) before
    GET  /stop    barge-in: abort the in-flight speech (also aborts the GPU generation)
    GET  /warm    re-load after /cold (background); 503 "warming" until ready
    GET  /cold    unload the model + free VRAM, keep listening
    GET  /fillers filler-clip diagnostics (fillers are OFF by default here, see below)
    GET  /stats   JSON: measured synthesis speed (RTF), frames/word, VRAM, last utterance

Synthesis: Fish Audio S2 Pro (DualAR text2semantic + DAC codec, 44.1 kHz), loaded ONCE and kept
warm. What makes it fast enough on the V100 (sm_70, fp16 only), measured 2026-10-08/09:
  * KV cache capped at FISH_MAXSEQ=4096 (+ model buffers built at that size, not 32k);
  * per-frame decode step torch.compile(mode="reduce-overhead") = CUDA graphs, one graph per
    128-token attention bucket, all recorded at warm-up (floats specialised — a single CPU
    float input silently disables CUDA graphs; warm-up inputs match the real strides and the
    bucket never equals the full cache — either mismatch costs a ~5 min recompile);
  * main-token logits only over the 4096 semantic codes + <|im_end|> (same distribution);
  * weight-only int8 (FISH_INT8, default on — see below);
  * prompt-prefix caching: the reference (VQ-encoded ONCE) stays prefilled in the KV cache, and
    the per-chunk prefill of what is new is compiled too;
  * codec: its four unused 1 GiB causal masks trimmed (-3 GiB), inputs padded to 16-frame
    buckets whose cuDNN plans are built at warm-up (a new length otherwise costs ~0.5 s),
    windows <= 256 frames (memory);
  * no gc / empty_cache per call; end-of-speech checked asynchronously (pinned flag, 1 step lag).
Result (dry-run, V100 shared with the live services): ~21-26 ms of compute per 46.4 ms frame
(RTF ~0.46-0.57) int8, ~30 ms (RTF ~0.65) fp16; first audio ~0.5-1.3 s; ~9-12 GB VRAM.
First start compiles for ~10 min; with the inductor cache warm a start takes ~3-4 min.

Streaming rule (Zeke, 2026-10-08): never wait for a whole sentence. The first ~3 words are
synthesised and played as soon as they are ready; the following chunks grow (3, 3, 6, 10, ...
words), each one synthesised WHILE the previous audio plays. The size of every next chunk is
taken from the MEASURED synthesis speed (seconds of compute per frame + fixed per-chunk
overhead, tracked live) so that it is predicted to finish before the audio already queued runs
out, with a safety margin. Cuts prefer sentence ends, then clause punctuation, then word count.
Voice continuity across chunks: the same cached reference prompt for every chunk, one seed per
utterance, and the previous chunk(s) (text + generated codes) passed as conversation history,
the mechanism Fish itself uses for long text.

Emotion: Fish takes free-form inline tags ([sigh], [whisper], [excited], ...); they pass straight
through. A "style" tag (anything that is not a one-off event like [sigh]/[laugh]) is carried
onto the following chunks so a chunk split never drops it. The legacy {emotion, intensity}
params (brain/voice_emotion.py labels) map to a leading tag (see _EMOTION_TAGS) unless the text
already starts with its own tag; intensity < 0.25 = no tag. StyleTTS2's speed/beta knobs have
no Fish equivalent and are ignored.

Test mode: IRIS_MOUTH_DRYRUN=1 never touches a sound device or the network sink and never writes
the live orb files (scratch/voice_status.json, voice_amplitude.json). "Playback" is simulated in
real time into WAV files + a JSON timing report per utterance under IRIS_MOUTH_DRYRUN_DIR
(default ~/voice_lab/fish_mouth_test), including every underrun gap.

Venv: the Fish venv (~/voice_lab/fish/venv). Port: WREN_VOICE_PORT (default 8779).
2026-10-08 (Iris)
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import math
import os
import queue
import re
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_VOICE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VOICE_DIR.parent
_HOME = Path.home()

# ── Config (env) ──────────────────────────────────────────────────────────────────
FISH_ROOT = Path(os.environ.get("IRIS_FISH_ROOT", str(_HOME / "voice_lab" / "fish")))
FISH_SRC = FISH_ROOT / "src"
CKPT_DIR = Path(os.environ.get("IRIS_FISH_CKPT", str(FISH_ROOT / "checkpoints" / "s2-pro")))
REF_WAV = Path(os.environ.get("IRIS_FISH_REF_WAV", str(_HOME / "voice_lab" / "ref" / "ref_44100.wav")))
REF_TXT = Path(os.environ.get("IRIS_FISH_REF_TXT", str(_HOME / "voice_lab" / "ref" / "ref.txt")))

HOST = os.environ.get("WREN_VOICE_BIND", "127.0.0.1")
PORT = int(os.environ.get("WREN_VOICE_PORT", "8779"))

MAXSEQ = int(os.environ.get("FISH_MAXSEQ", "4096"))           # KV cache cap (tokens)
COMPILE_MODE = os.environ.get("FISH_COMPILE", "reduce-overhead").strip()  # reduce-overhead|default|none
KV_BUCKET = int(os.environ.get("FISH_KV_BUCKET", "128"))       # attention prefix rounded up to this
WARM_KV = int(os.environ.get("FISH_WARM_KV", str(MAXSEQ)))      # pre-record CUDA graphs up to here
TEMPERATURE = float(os.environ.get("FISH_TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("FISH_TOP_P", "0.8"))
TOP_K = int(os.environ.get("FISH_TOP_K", "30"))
DEFAULT_SEED = int(os.environ.get("FISH_SEED", "1234"))         # one seed per utterance; -1 = random
HIST_CHUNKS = int(os.environ.get("FISH_CONTEXT_CHUNKS", "1"))   # previous chunks passed as context
HIST_MAX_FRAMES = int(os.environ.get("FISH_CONTEXT_MAX_FRAMES", "320"))
CODEC_BUCKET = int(os.environ.get("FISH_CODEC_BUCKET", "16"))     # codec input padded to this (frames)
CODEC_MAX_FRAMES = int(os.environ.get("FISH_CODEC_MAX_FRAMES", "256"))     # longer = windowed decode
CODEC_WARM_FRAMES = int(os.environ.get("FISH_CODEC_WARM_FRAMES", "288"))
# Deadline flush (safety net under the chunk schedule): if a chunk is still generating when the
# audio already queued is about to run out (its length can't be predicted: Fish adds pauses,
# sighs, laughs), the frames generated so far are decoded and played at once and generation
# simply continues. The codec is causal w.r.t. padding on the right, but NOT exactly reproducible
# from a window (measured: decoding with 16-64 frames of left context differs from a full decode
# by up to ~0.3 later on — a phase/state drift, not an edge effect), so flushed pieces are joined
# with a short crossfade (FLUSH_XFADE samples) over an overlap region, not a hard cut. The first
# chunk flushes at FIRST_FLUSH_S at the latest.
# Speed levers (see the module docstring / report):
#  FISH_VOCAB_SUBSET=1  project the slow transformer only onto the 4096 semantic codes + <|im_end|>
#                       (the logit bias masks everything else anyway -> same distribution; saves the
#                       155,776-row output matmul + two full-vocab sorts per frame)
#  FISH_INT8=1          weight-only int8 (per-output-channel) for every Linear in the AR model.
#                       DEFAULT ON (2026-10-09): ~15-30 % faster per frame, ~3.5 GB less VRAM, WER 0
#                       and speaker similarity unchanged on the test set — but it IS a change to the
#                       voice, so Zeke's ears are the real check. FISH_INT8=0 = plain fp16.
VOCAB_SUBSET = os.environ.get("FISH_VOCAB_SUBSET", "1").strip().lower() not in ("0", "off", "false", "no")
COMPILE_PREFILL = os.environ.get("FISH_COMPILE_PREFILL", "1").strip().lower() not in ("0", "off", "false", "no")
INT8 = os.environ.get("FISH_INT8", "1").strip().lower() not in ("0", "off", "false", "no")
FLUSH_ON = os.environ.get("FISH_FLUSH", "1").strip().lower() not in ("0", "off", "false", "no")
FLUSH_LEAD_S = float(os.environ.get("FISH_FLUSH_LEAD_S", "0.18"))
FLUSH_MIN_FRAMES = int(os.environ.get("FISH_FLUSH_MIN_FRAMES", "6"))
FLUSH_CTX = int(os.environ.get("FISH_FLUSH_CTX", "24"))
FLUSH_XFADE = int(os.environ.get("FISH_FLUSH_XFADE", "1024"))    # ~23 ms at 44.1 kHz
FIRST_FLUSH_S = float(os.environ.get("FISH_FIRST_FLUSH_S", "1.2"))

# chunk schedule (words). Growth caps; the measured speed decides how much of a cap is affordable.
SCHEDULE = [int(x) for x in os.environ.get("FISH_CHUNK_SCHEDULE", "3,3,6,10").split(",") if x.strip()]
GROWTH = float(os.environ.get("FISH_CHUNK_GROWTH", "1.6"))      # after the schedule runs out
MAX_CHUNK_WORDS = int(os.environ.get("FISH_MAX_CHUNK_WORDS", "40"))
MIN_CHUNK_WORDS = int(os.environ.get("FISH_MIN_CHUNK_WORDS", "2"))
CLAUSE_STRETCH = float(os.environ.get("FISH_CLAUSE_STRETCH", "1.7"))  # stretch a chunk to reach a clause boundary
SAFETY_FRAC = float(os.environ.get("FISH_SAFETY_FRAC", "0.15"))  # keep 15 % of the budget spare
SAFETY_S = float(os.environ.get("FISH_SAFETY_S", "0.15"))        # ... plus this many seconds

DRYRUN = os.environ.get("IRIS_MOUTH_DRYRUN", "").strip().lower() in ("1", "true", "yes", "on")
DRYRUN_DIR = Path(os.environ.get("IRIS_MOUTH_DRYRUN_DIR", str(_HOME / "voice_lab" / "fish_mouth_test")))
DRYRUN_DEVICE_BUFFER_S = float(os.environ.get("IRIS_MOUTH_DRYRUN_BUFFER_S", "0.25"))

# Fillers ("Mm.", "Right.") are what the StyleTTS2 mouth plays before every reply to hide its
# latency. Here the 3-word first chunk is the latency fix, and an underrun filler dropped into
# the middle of a sentence would sound wrong, so they are OFF unless WREN_FISH_FILLERS=1.
FILLERS_ENABLED = os.environ.get("WREN_FISH_FILLERS", "0").strip().lower() in ("1", "true", "on", "yes")

# Persistent compile caches (a warm cache cuts the compile on the next start a lot).
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(FISH_ROOT / "inductor_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(FISH_ROOT / "triton_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("EINX_FILTER_TRACEBACK", "false")

sys.path.insert(0, str(FISH_SRC))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _log(msg: str) -> None:
    print(f"[fish] {msg}", flush=True)


try:
    import numpy as np
    import torch
    import torchaudio
    from loguru import logger as _loguru
except ImportError as e:
    print(f"[fish] missing dep: {e!r}  (run with the Fish venv python)", flush=True)
    sys.exit(2)

try:   # Fish logs every step at INFO; keep warnings only.
    _loguru.remove()
    _loguru.add(sys.stderr, level=os.environ.get("FISH_LOGLEVEL", "WARNING"))
except Exception:
    pass

try:
    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message
    from fish_speech.tokenizer import IM_END_TOKEN
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.models.text2semantic.inference import (decode_one_token_ar, RAS_WIN_SIZE, RAS_HIGH_TEMP,
                                                            RAS_HIGH_TOP_P, logits_to_probs, sample as _fish_sample,
                                                            multinomial_sample_one_no_sync)
    import torch.nn.functional as F
    from fish_speech.models.dac.inference import load_model as _load_codec
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError as e:
    print(f"[fish] Fish import failed: {e!r}  (IRIS_FISH_ROOT={FISH_ROOT})", flush=True)
    sys.exit(2)

try:
    import sounddevice as sd   # only for the local-device path; the server uses the net sink
except Exception:
    sd = None

# ── Shared voice-state overlay (live orb files) — NEVER touched in dry-run ─────────
if DRYRUN:
    def set_state(*_a, **_k):  # type: ignore
        pass

    def set_amplitude(*_a, **_k):  # type: ignore
        pass
else:
    try:
        sys.path.insert(1, str(_VOICE_DIR))
        from wren_voice_status import set_state, set_amplitude  # type: ignore
    except Exception:
        def set_state(*_a, **_k):  # type: ignore
            pass

        def set_amplitude(*_a, **_k):  # type: ignore
            pass

SAMPLE_RATE = 44100   # the S2 Pro codec; confirmed against the decoder at load

_SPEAK_LOCK = threading.Lock()
_READY = False
_STOP = threading.Event()


# ── Emotion → leading Fish tag ────────────────────────────────────────────────────
def _load_voice_emotion():
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("iris_voice_emotion", str(_REPO_ROOT / "brain" / "voice_emotion.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _VE = _load_voice_emotion()
    _EMOTION_ON = bool(_VE.ENABLED)
except Exception as _ee:
    _log(f"emotion plumbing unavailable ({_ee!r}); inline tags only")
    _VE = None
    _EMOTION_ON = False

# label -> (mild tag, strong tag); intensity < 0.25 none, < 0.6 mild, else strong
_EMOTION_TAGS = {
    "calm": ("[calm]", "[calm, relaxed]"),
    "joy": ("[happy]", "[excited]"),
    "interest": ("[curious]", "[curious, engaged]"),
    "tenderness": ("[soft, gentle tone]", "[tenderly]"),
    "sadness": ("[sad]", "[sad, quiet]"),
    "anger": ("[annoyed]", "[frustrated]"),
    "fear": ("[nervous]", "[anxious]"),
    "surprise": ("[surprised]", "[shocked]"),
    "tired": ("[tired]", "[exhausted]"),
}


def _style_from_obj(obj) -> dict:
    """{emotion, intensity} -> {"tag": str|None, "label", "applied"}."""
    out = {"tag": None, "label": "neutral", "applied": False}
    # A literal Fish tag as the emotion ("[playful, teasing]") is used as-is — lets a tour line or a
    # deliberate voice_speak pick its own tone per sentence (Zeke 10-09: "almost a sentence by sentence case").
    raw = str(obj.get("emotion") or "").strip() if isinstance(obj, dict) else ""
    if _TAG_RE.fullmatch(raw) if raw else False:
        return {"tag": raw, "label": raw, "applied": True}
    if _VE is None or not _EMOTION_ON or not isinstance(obj, dict) or not obj.get("emotion"):
        return out
    try:
        label = _VE.normalize_emotion(obj.get("emotion"))
        inten = obj.get("intensity", 0.5)
        inten = max(0.0, min(1.0, float(0.5 if inten is None else inten)))
    except Exception:
        return out
    out["label"] = label
    if label not in _EMOTION_TAGS or inten < 0.25:
        return out
    out["tag"] = _EMOTION_TAGS[label][0 if inten < 0.6 else 1]
    out["applied"] = True
    return out


# ── Text -> units -> chunks ───────────────────────────────────────────────────────
_TAG_RE = re.compile(r"\[[^\[\]]{1,80}\]")
_TOKEN_RE = re.compile(r"\[[^\[\]]{1,80}\]|\S+")
_HAS_WORD = re.compile(r"[^\W_]")      # a letter or digit in any script (not "—", "...")
# one-off sounds: they do NOT carry over to the next chunk. Every other tag is a style.
_EVENT_WORDS = ("sigh", "laugh", "chuckle", "giggle", "gasp", "breath", "inhale", "exhale",
                "cough", "clear", "sniff", "pause", "hum", "groan", "yawn", "snort", "scoff",
                "tsk", "click", "swallow", "gulp", "shush", "hmm", "um", "uh")


# Never end a chunk right after one of these when a plain word is available: a fragment that
# ends on "a"/"the"/"of" made Fish produce junk ("moved, a" | "lot of" -> "a sick lot of").
_FUNCTION_WORDS = frozenset("""a an the of to in on at by for from with into onto about as and or but nor so
if than that this these those my your his her its our their i you he she we they it is are was were be
been am do does did has have had will would can could should shall may might must not no very just
really what which who whom whose when where why how because while though although until unless""".split())


def _is_event_tag(tag: str) -> bool:
    t = tag.strip("[] ").lower()
    return any(w in t for w in _EVENT_WORDS)


class _Unit:
    """One spoken word, with the tags that precede it and the punctuation glued after it."""
    __slots__ = ("pre_tags", "word", "post", "strength")

    def __init__(self, word: str, pre_tags: list):
        self.word = word
        self.pre_tags = list(pre_tags)
        self.post = []        # trailing punctuation-only tokens / trailing tags (end of text)
        self.strength = 1     # boundary strength AFTER this unit: 3 sentence, 2 clause, 1 plain

    def text(self) -> str:
        return " ".join(self.pre_tags + [self.word] + self.post)

    def finish(self) -> None:
        tail = (self.word + "".join(p for p in self.post if not _TAG_RE.fullmatch(p))).rstrip("\"')]}»”’")
        if re.search(r"(\.\.\.|…|[.!?])$", tail):
            self.strength = 3
        elif re.search(r"[,;:—–-]$", tail):
            self.strength = 2
        elif re.sub(r"[^\w']", "", self.word.lower()) in _FUNCTION_WORDS:
            self.strength = 0          # "a | lot" — avoid
        else:
            self.strength = 1


def _units(text: str) -> list:
    units: list = []
    pend: list = []
    for tok in _TOKEN_RE.findall(text or ""):
        if _TAG_RE.fullmatch(tok):
            pend.append(tok)
        elif not _HAS_WORD.search(tok):           # "—", "...", "-" on their own
            if units:                              # "funny [laughing] —" stays together
                units[-1].post.extend(pend + [tok])
                pend = []
            else:
                pend.append(tok)
        else:
            units.append(_Unit(tok, pend))
            pend = []
    if pend:
        if units:
            units[-1].post.extend(pend)
        else:
            return []          # only tags / punctuation: nothing to say
    for u in units:
        u.finish()
    return units


class ChunkPlanner:
    """Walks the units and cuts the next chunk at the best natural boundary <= max_words."""

    def __init__(self, units: list, style_tag: str | None = None):
        self.units = units
        self.i = 0
        self.style = style_tag      # carried style tag (emotion param, then the last inline style)

    @property
    def done(self) -> bool:
        return self.i >= len(self.units)

    @property
    def remaining(self) -> int:
        return len(self.units) - self.i

    def next(self, max_words: int, min_words: int = MIN_CHUNK_WORDS) -> dict:
        rem = self.remaining
        max_words = max(1, min(max_words, rem))
        min_words = max(1, min(min_words, max_words))
        if rem - max_words <= 2 and rem <= max_words + 2 and rem <= max(4, int(max_words * 1.35)):
            n = rem                                    # don't strand a 1-2 word tail
        else:
            lo = max(min_words, int(math.ceil(max_words * 0.5)))
            best_n, best_s = max_words, 0
            for n_ in range(lo, max_words + 1):
                s = self.units[self.i + n_ - 1].strength
                if s >= best_s:                        # ties -> the longer chunk
                    best_n, best_s = n_, s
            n = best_n
            # Zeke 10-09: "let me think| about that" — a mid-clause cut audibly clips the word and
            # crowds the next one. If there's no comma/sentence end in range, stretch the chunk to
            # the next clause boundary (up to CLAUSE_STRETCH x max_words) instead of cutting mid-phrase.
            if best_s < 2:
                hi = min(rem, int(max_words * CLAUSE_STRETCH))
                for n_ in range(max_words + 1, hi + 1):
                    if self.units[self.i + n_ - 1].strength >= 2:
                        n = n_
                        break
            # never strand a chunk on "a"/"the"/"of": take up to 2 more words to reach a real word
            k = 0
            while self.units[self.i + n - 1].strength == 0 and n < rem and k < 2:
                n += 1
                k += 1
        seg = self.units[self.i:self.i + n]
        self.i += n
        body = " ".join(u.text() for u in seg)
        lead = None
        own_style = any(not _is_event_tag(t) for t in seg[0].pre_tags if _TAG_RE.fullmatch(t))
        if self.style and not own_style:
            lead = self.style
        for u in seg:                                  # update the carried style
            for t in u.pre_tags + u.post:
                if _TAG_RE.fullmatch(t) and not _is_event_tag(t):
                    self.style = t
        text = f"{lead} {body}" if lead else body
        n_tags = len(_TAG_RE.findall(text))
        return {"text": text, "words": n, "tags": n_tags, "strength": seg[-1].strength,
                "last": self.done}


# ── Audio helpers ────────────────────────────────────────────────────────────────
def _trim_edges(a: np.ndarray, strength: int, first: bool, last: bool) -> np.ndarray:
    """Cap leading/trailing silence so chunk joins are tight but natural; short fades."""
    if a.size == 0:
        return a
    hop = int(SAMPLE_RATE * 0.01)
    n = a.size // hop
    if n < 3:
        return a
    rms = np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, axis=1))
    voiced = np.nonzero(rms > 0.004)[0]
    if voiced.size == 0:
        return a
    keep_lead = 0.02 if not first else 0.01
    keep_tail = {3: 0.28, 2: 0.16, 1: 0.12}.get(strength, 0.12) if not last else 0.25   # 0.06 clipped word endings ("think|") — Zeke 10-09
    s = max(0, voiced[0] * hop - int(keep_lead * SAMPLE_RATE))
    e = min(a.size, (voiced[-1] + 1) * hop + int(keep_tail * SAMPLE_RATE))
    out = a[s:e].astype(np.float32, copy=True)
    f = min(int(0.004 * SAMPLE_RATE), out.size // 4)
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        out[:f] *= ramp
        out[-f:] *= ramp[::-1]
    return out


def _voiced_bounds(a: np.ndarray):
    hop = int(SAMPLE_RATE * 0.01)
    n = a.size // hop
    if n < 1:
        return None
    rms = np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, axis=1))
    v = np.nonzero(rms > 0.004)[0]
    if v.size == 0:
        return None
    return v[0] * hop, (v[-1] + 1) * hop


def _fade(a: np.ndarray, head: bool, tail: bool) -> np.ndarray:
    f = min(int(0.004 * SAMPLE_RATE), a.size // 4)
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if head:
            a[:f] *= ramp
        if tail:
            a[-f:] *= ramp[::-1]
    return a


def _trim_lead(a: np.ndarray, keep_s: float) -> np.ndarray:
    b = _voiced_bounds(a)
    if b is None:
        return a.astype(np.float32, copy=True)
    s = max(0, b[0] - int(keep_s * SAMPLE_RATE))
    return _fade(a[s:].astype(np.float32, copy=True), True, False)


def _trim_tail(a: np.ndarray, keep_s: float) -> np.ndarray:
    b = _voiced_bounds(a)
    if b is None:
        return a[: int(keep_s * SAMPLE_RATE)].astype(np.float32, copy=True)
    e = min(a.size, b[1] + int(keep_s * SAMPLE_RATE))
    return _fade(a[:e].astype(np.float32, copy=True), False, True)


_TAIL_KEEP = {3: 0.28, 2: 0.16, 1: 0.06}


def _resample(a: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or a.size == 0:
        return a
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]
    return torchaudio.functional.resample(t, sr_in, sr_out)[0].numpy()


# ── The engine ───────────────────────────────────────────────────────────────────
class SynthStats:
    """Live model of synthesis speed: t(chunk) = overhead + step_s * frames."""

    def __init__(self):
        self.step_s = 0.050        # seconds of compute per generated frame (46.4 ms of audio)
        self.overhead_s = 0.25     # prefill + codec decode + misc per chunk
        self.fpw = 9.0             # frames per spoken word (Iris is slow: ~0.42 s/word)
        self.tag_frames = 8.0      # extra frames an inline tag tends to add
        self.n = 0
        self.last: dict = {}
        self._lock = threading.Lock()

    def update(self, frames: int, words: int, tags: int, step_total_s: float, overhead_s: float):
        a = 0.35 if self.n >= 2 else 0.7
        with self._lock:
            if frames > 0:
                self.step_s = (1 - a) * self.step_s + a * (step_total_s / max(1, frames))
            self.overhead_s = (1 - a) * self.overhead_s + a * overhead_s
            if words > 0:
                fpw = max(3.0, (frames - tags * self.tag_frames) / words)
                self.fpw = (1 - a) * self.fpw + a * min(fpw, 30.0)
            self.n += 1

    def predict(self, words: int, tags: int = 0) -> float:
        frames = words * self.fpw + tags * self.tag_frames
        return self.overhead_s + self.step_s * frames

    def max_words(self, budget_s: float, tags: int = 0) -> int:
        usable = budget_s * (1.0 - SAFETY_FRAC) - SAFETY_S - self.overhead_s - self.step_s * tags * self.tag_frames
        if usable <= 0:
            return 0
        return int(usable / (self.step_s * self.fpw))

    @property
    def rtf(self) -> float:
        return self.step_s / (2048.0 / SAMPLE_RATE)

    def as_dict(self) -> dict:
        return {"step_ms_per_frame": round(self.step_s * 1000, 2), "rtf_steady": round(self.rtf, 3),
                "overhead_s": round(self.overhead_s, 3), "frames_per_word": round(self.fpw, 2),
                "samples": self.n, "last": self.last}


def _decode_step(model, x, input_pos, temperature, top_p, semantic_logit_bias, previous_tokens, kv_len):
    return decode_one_token_ar(model, x, input_pos, temperature, top_p, TOP_K, semantic_logit_bias,
                               None, None, previous_tokens=previous_tokens, kv_len=kv_len)


def _slow_forward(model, inp, input_pos, kv_len: int):
    """BaseTransformer.forward_generate minus the full-vocab logits (and minus audio_parts,
    which this server never uses). Returns (normed slow output, fast-transformer input)."""
    cfg = model.config
    embeds = [model.codebook_embeddings(inp[:, i + 1] + i * cfg.codebook_size) for i in range(cfg.num_codebooks)]
    vq = torch.stack(embeds, dim=1).sum(dim=1)
    m = (inp[:, 0] >= cfg.semantic_begin_id) & (inp[:, 0] <= cfg.semantic_end_id)
    vq = vq * m.unsqueeze(-1).to(vq.dtype)                 # == vq[~m] = 0, without a CPU constant
    x = model.embeddings(inp[:, 0]) + vq
    if cfg.scale_codebook_embeddings:
        x = torch.where(m.unsqueeze(-1), x / math.sqrt(cfg.num_codebooks + 1), x)
    mask = model.causal_mask[None, None, input_pos, :kv_len]
    freqs_cis = model.freqs_cis[input_pos]
    for layer in model.layers:
        x = layer(x, freqs_cis, mask, input_pos=input_pos)
    if x.size(1) > 1:
        x = x[:, -1:]
    slow_out = model.norm(x)
    hidden = slow_out if getattr(cfg, "norm_fastlayer_input", False) else x
    return slow_out, model.fast_project_in(hidden)


def _decode_step_sub(model, x, input_pos, temperature, top_p, sub_w, sub_ids, previous_tokens, kv_len):
    """decode_one_token_ar with the main-token logits restricted to the allowed set (sub_ids =
    semantic codes + im_end). Same sampling distribution as the masked full-vocab version."""
    cfg = model.config
    slow_out, hidden = _slow_forward(model, x, input_pos, kv_len)
    logits = F.linear(slow_out, sub_w)[0, -1]              # (4097,)
    idx = multinomial_sample_one_no_sync(logits_to_probs(logits, temperature, top_p, TOP_K))
    high_t = torch.tensor(RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype)
    high_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
    idx_h = multinomial_sample_one_no_sync(logits_to_probs(logits, high_t, high_p, TOP_K))
    tok = sub_ids[idx]
    tok_h = sub_ids[idx_h]
    in_window = (previous_tokens[0] == tok).any()
    is_sem = (tok >= cfg.semantic_begin_id) & (tok <= cfg.semantic_end_id)
    tok = torch.where(in_window & is_sem, tok_h, tok)
    codebooks = [tok]
    model.forward_generate_fast(hidden, torch.tensor([0], device=hidden.device, dtype=torch.long))
    a = torch.clamp(tok - cfg.semantic_begin_id, min=0, max=cfg.codebook_size - 1)
    h = model.fast_embeddings(a)
    codebooks.append(a)
    for ci in range(1, cfg.num_codebooks):
        lg = model.forward_generate_fast(h, torch.tensor([ci], device=h.device, dtype=torch.long))
        a = _fish_sample(lg, temperature=temperature, top_p=top_p, top_k=TOP_K)[0]
        h = model.fast_embeddings(a)
        codebooks.append(a)
    return torch.stack(codebooks, dim=1).T


def _prefill_sub(model, x, input_pos, temperature, top_p, sub_w, sub_ids, previous_tokens, kv_len):
    """Separate code object so its torch.compile cache (mode=default, dynamic prompt length) is
    independent of the CUDA-graph decode step's."""
    return _decode_step_sub(model, x, input_pos, temperature, top_p, sub_w, sub_ids, previous_tokens, kv_len)


class _Int8Linear(torch.nn.Module):
    """Weight-only int8, per-output-channel scales (gpt-fast style; inductor fuses the dequant)."""

    def __init__(self, lin: "torch.nn.Linear"):
        super().__init__()
        w = lin.weight.data.float()
        scales = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        q = torch.round(w / scales[:, None]).clamp(-128, 127).to(torch.int8)
        self.register_buffer("weight", q)
        self.register_buffer("scales", scales.to(lin.weight.dtype))
        self.bias = lin.bias
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        # Scale the WEIGHT, not the output: in fp16 (the V100 has no bf16) x @ q is up to
        # 127/absmax too large and overflows 65504 -> inf -> the model never emits <|im_end|>
        # (measured). Upstream's `F.linear(x, q) * scales` is only safe in bf16.
        w = self.weight.to(x.dtype) * self.scales[:, None]
        y = F.linear(x, w)
        return y if self.bias is None else y + self.bias


def _quantize_int8(module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, _Int8Linear(child))
            n += 1
        else:
            n += _quantize_int8(child)
    return n


class FishEngine:
    def __init__(self):
        self.model = None
        self.codec = None
        self.step = None
        self.prefix = None            # (C+1, P) CPU long — the system/reference prompt
        self.prefix_ready = False     # KV positions [0, P) hold exactly that prompt
        self.sys_msg = None
        self.im_end = None
        self.stats = SynthStats()
        self.cudagraphs = COMPILE_MODE == "reduce-overhead"
        self.prefill = None
        self._pinned = None

    # -- load / unload ------------------------------------------------------------
    def load(self) -> None:
        global SAMPLE_RATE
        t0 = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = True
        model = DualARTransformer.from_pretrained(str(CKPT_DIR), load_weights=True, max_length=MAXSEQ)
        model = model.to(device="cuda", dtype=torch.half).eval()
        with torch.device("cuda"):
            model.setup_caches(max_batch_size=1, max_seq_len=MAXSEQ, dtype=torch.half)
        self.model = model
        self.im_end = model.tokenizer.get_token_id(IM_END_TOKEN)
        cfg = model.config
        if INT8:
            nq = _quantize_int8(model)
            torch.cuda.empty_cache()
            _log(f"int8 weight-only: {nq} Linear layers quantised; {json.dumps(_vram_mib())}")
        ids = list(range(cfg.semantic_begin_id, cfg.semantic_end_id + 1)) + [self.im_end]
        self.sub_ids = torch.tensor(ids, device="cuda", dtype=torch.int)
        self.sub_w = model.embeddings.weight[self.sub_ids.long()].contiguous()   # tied output rows
        self.cd = cfg.num_codebooks + 1
        bias = torch.full((1, 1, cfg.vocab_size), float("-inf"), device="cuda", dtype=torch.half)
        bias[0, 0, cfg.semantic_begin_id: cfg.semantic_end_id + 1] = 0.0
        bias[0, 0, self.im_end] = 0.0
        self.bias = bias
        self.temperature = torch.tensor(TEMPERATURE, device="cuda", dtype=torch.half)
        self.top_p = torch.tensor(TOP_P, device="cuda", dtype=torch.half)
        _log(f"LLM loaded in {time.perf_counter()-t0:.1f}s (fp16, max_seq_len={MAXSEQ})")

        self.codec = _load_codec(config_name="modded_dac_vq", checkpoint_path=str(CKPT_DIR / "codec.pth"),
                                 device="cuda")
        # The codec's transformers each register a 32768x32768 bool causal mask (1 GiB apiece)
        # that their windowed forward never reads (it builds its own mask). Keep 8192.
        freed = 0
        for mod in self.codec.modules():
            cm = getattr(mod, "causal_mask", None)
            if isinstance(cm, torch.Tensor) and cm.dim() == 2 and cm.shape[0] > 8192:
                freed += cm.numel()
                mod.register_buffer("causal_mask", cm[:8192, :8192].clone(), persistent=False)
                del cm
        torch.cuda.empty_cache()
        _log(f"codec loaded; trimmed causal masks, freed {freed / 2**30:.1f} GiB")
        sr = int(getattr(self.codec, "sample_rate", 44100))
        if sr != SAMPLE_RATE:
            _log(f"codec sample rate {sr} (expected {SAMPLE_RATE}) — following the codec")
            SAMPLE_RATE = sr

        ref_text = REF_TXT.read_text(encoding="utf-8").strip()
        ref_codes = self._encode_reference(REF_WAV)
        self.sys_msg = Message(role="system", parts=[
            TextPart(text="convert the provided text to speech reference to the following:\n\nText:\n", cal_loss=False),
            TextPart(text=f"<|speaker:0|>{ref_text}", cal_loss=False),
            TextPart(text="\n\nSpeech:\n", cal_loss=False),
            VQPart(codes=ref_codes, cal_loss=False),
        ], cal_loss=False, add_im_start=True, add_im_end=True)
        conv = Conversation()
        conv.append(self.sys_msg)
        self.prefix, _, _ = conv.encode_for_inference(model.tokenizer, num_codebooks=cfg.num_codebooks)
        self.prefix_ready = False
        _log(f"reference encoded once: {ref_codes.shape[1]} frames, prompt prefix {self.prefix.shape[1]} tokens")

        if COMPILE_MODE in ("reduce-overhead", "default", "max-autotune-no-cudagraphs"):
            import torch._dynamo as _dyn
            import torch._inductor.config as icfg
            _dyn.config.cache_size_limit = max(64, _dyn.config.cache_size_limit)
            # dynamic=True would otherwise pass Python floats in as 0-dim CPU float64 tensors,
            # and ONE cpu input makes inductor skip CUDA graphs ("skipping cudagraphs due to
            # cpu device"). Floats here are constants (scales, temperatures), so specialise them.
            _dyn.config.specialize_float = True
            icfg.coordinate_descent_tuning = True
            icfg.triton.unique_kernel_names = True
            if hasattr(icfg, "fx_graph_cache"):
                icfg.fx_graph_cache = True
            self.step = torch.compile(_decode_step_sub if VOCAB_SUBSET else _decode_step,
                                      mode=COMPILE_MODE, fullgraph=True, dynamic=True)
            if COMPILE_PREFILL and VOCAB_SUBSET:
                self.prefill = torch.compile(_prefill_sub, mode="default", fullgraph=True, dynamic=True)
        else:
            self.step = _decode_step_sub if VOCAB_SUBSET else _decode_step
        self._pinned = torch.zeros(64, dtype=torch.int32, pin_memory=True)
        self._zero_prev = torch.zeros((self.cd, RAS_WIN_SIZE), dtype=torch.int, device="cuda")

    def _encode_reference(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        audios = wav[None].to("cuda")                 # (1, 1, T)
        lengths = torch.tensor([audios.shape[2]], device="cuda", dtype=torch.long)
        codes = self.codec.encode(audios, lengths)[0][0]
        return codes.cpu()

    def unload(self) -> None:
        self.model = self.codec = self.step = self.prefill = None
        self.sub_w = self.sub_ids = self.bias = None
        self.prefix = None
        self.prefix_ready = False
        try:
            import torch._dynamo as _dyn
            _dyn.reset()
        except Exception:
            pass
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # -- prompt -------------------------------------------------------------------
    def _encode(self, text: str, history: list) -> torch.Tensor:
        conv = Conversation()
        conv.append(self.sys_msg)
        for htext, hcodes in history:
            conv.append(Message(role="user", parts=[TextPart(text=htext, cal_loss=False)], cal_loss=False,
                                add_im_start=True, add_im_end=True))
            conv.append(Message(role="assistant", parts=[VQPart(codes=hcodes, cal_loss=False)], cal_loss=False,
                                modality="voice", add_im_start=True, add_im_end=True))
        conv.append(Message(role="user", parts=[TextPart(text=text, cal_loss=False)], cal_loss=False,
                            add_im_start=True, add_im_end=True))
        conv.append(Message(role="assistant", parts=[], cal_loss=False, modality="voice",
                            add_im_start=True, add_im_end=False))
        enc, masks, parts = conv.encode_for_inference(self.model.tokenizer, num_codebooks=self.model.config.num_codebooks)
        if parts is not None:
            raise RuntimeError("audio_parts in prompt are not supported by this server")
        return enc

    def _call_step(self, cur, ipos, prev, kv: int):
        if VOCAB_SUBSET:
            return self.step(self.model, cur, ipos, self.temperature, self.top_p, self.sub_w, self.sub_ids, prev, kv)
        return self.step(self.model, cur, ipos, self.temperature, self.top_p, self.bias, prev, kv)

    def _kv_cap(self) -> int:
        # never attend over the WHOLE cache: at kv_len == capacity the slice is contiguous and
        # dynamo specialises -> a ~5 min recompile mid-sentence. Generation stays below this.
        return MAXSEQ - max(KV_BUCKET, 8)

    def _bucket(self, kv: int) -> int:
        if KV_BUCKET <= 1:
            return kv
        return min(self._kv_cap(), ((kv + KV_BUCKET - 1) // KV_BUCKET) * KV_BUCKET)

    # -- generation -----------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, text: str, history: list, max_new: int, should_stop=None,
                 flush_at=None, on_partial=None) -> dict:
        """text -> semantic codes (C, N) on the GPU, plus timings."""
        model = self.model
        cd = self.cd
        t0 = time.perf_counter()
        enc = self._encode(text, history)
        T = enc.shape[1]
        max_new = min(max_new, self._kv_cap() - T - 1)
        if max_new < 8:
            raise RuntimeError(f"prompt too long for the KV cache ({T} tokens)")
        P = self.prefix.shape[1]
        start = P if (self.prefix_ready and T > P and torch.equal(enc[:, :P], self.prefix)) else 0
        x = enc[:, start:].to("cuda", non_blocking=True)
        input_pos = torch.arange(start, T, device="cuda", dtype=torch.long)
        if self.prefill is not None:
            # previous_tokens = zeros never matches a semantic token -> no RAS on the first token,
            # exactly like the upstream prefill (which passes None)
            first = self.prefill(model, x.view(1, cd, -1), input_pos, self.temperature, self.top_p,
                                 self.sub_w, self.sub_ids, self._zero_prev, T).clone()
        else:
            first = decode_one_token_ar(model, x.view(1, cd, -1), input_pos, self.temperature, self.top_p, TOP_K,
                                        self.bias, None, None, kv_len=T).clone()
        if T >= P and torch.equal(enc[:, :P], self.prefix):
            self.prefix_ready = True
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0

        out = [first]
        cur = first.view(1, cd, 1)
        ipos = torch.tensor([T], device="cuda", dtype=torch.int)
        prev = torch.zeros((cd, RAS_WIN_SIZE), dtype=torch.int, device="cuda")
        pinned = self._pinned
        ring = len(pinned)
        events: list = [None] * ring
        stopped = False
        end_at = None
        pos = T
        t1 = time.perf_counter()
        n = 1
        emitted = 0
        flushes = []
        if int(first[0, 0]) == self.im_end:
            end_at = 0
        while end_at is None and n < max_new:
            if should_stop is not None and (n & 7) == 0 and should_stop():
                stopped = True
                break
            if (on_partial is not None and n - emitted >= FLUSH_MIN_FRAMES
                    and time.perf_counter() >= flush_at()):
                seq = torch.cat(out, dim=1)
                hits = (seq[0] == self.im_end).nonzero()
                if hits.numel():
                    end_at = int(hits[0, 0])
                    break
                tf = time.perf_counter()
                on_partial(seq[1:], emitted, n)
                flushes.append({"at_frame": n, "frames": n - emitted, "t": round(tf - t0, 3),
                                "decode_s": round(time.perf_counter() - tf, 3)})
                emitted = n
            if self.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            with sdpa_kernel(SDPBackend.MATH):
                nxt = self._call_step(cur, ipos, prev, self._bucket(pos + 1)).clone()
            ipos += 1
            pos += 1
            cur = nxt.view(1, cd, 1)
            prev = prev.roll(-1, dims=1)
            prev[:, -1] = nxt.view(cd, -1)[:, 0]
            out.append(nxt)
            # async end-of-speech check: copy the flag to pinned memory, read it one step later
            slot = n % ring
            pinned[slot:slot + 1].copy_(nxt[0, :1].to(torch.int32), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            events[slot] = ev
            prev_slot = (n - 1) % ring
            if n >= 2 and events[prev_slot] is not None:
                events[prev_slot].synchronize()
                if int(pinned[prev_slot]) == self.im_end:
                    end_at = n - 1
            n += 1
        torch.cuda.synchronize()
        if end_at is None:
            for k in range(max(1, n - 2), n):          # the last flags we didn't read yet
                if int(out[k][0, 0]) == self.im_end:
                    end_at = k
                    break
        t_steps = time.perf_counter() - t1
        seq = torch.cat(out, dim=1)                     # (C+1, n)
        if end_at is not None:
            seq = seq[:, :end_at]
        codes = seq[1:]
        return {"codes": codes, "frames": int(codes.shape[1]), "prompt_tokens": T,
                "prefilled": T - start, "t_prefill": t_prefill, "t_steps": t_steps,
                "steps": n - 1, "stopped": stopped, "hit_end": end_at is not None,
                "emitted": min(emitted, int(codes.shape[1])), "flushes": flushes}

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        n = int(codes.shape[1])
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if n > CODEC_MAX_FRAMES:
            return self._decode_windowed(codes)
        # cuDNN builds a new conv plan (~0.5 s) for every new input length. The decoder is
        # causal (measured: padding changes the kept samples by <= 1e-3), so pad to a bucket
        # whose plan was built at warm-up, then trim.
        nb = ((n + CODEC_BUCKET - 1) // CODEC_BUCKET) * CODEC_BUCKET if CODEC_BUCKET > 1 else n
        if nb > n:
            codes = torch.cat([codes, codes[:, -1:].expand(-1, nb - n)], dim=1)
        with torch.autocast(device_type="cuda", dtype=torch.half):
            audio = self.codec.from_indices(codes[None])[0].squeeze()
        return audio[: n * 2048].float().cpu().numpy()

    def _decode_windowed(self, codes: torch.Tensor) -> np.ndarray:
        """Bound the codec's activation memory (~1.1 GB extra at 256 frames): decode in windows
        of CODEC_MAX_FRAMES with FLUSH_CTX frames of left context, crossfaded at the joins."""
        n = int(codes.shape[1])
        step = max(32, CODEC_MAX_FRAMES - FLUSH_CTX)
        out = None
        start = 0
        while start < n:
            end = min(n, start + step)
            s0 = max(0, start - FLUSH_CTX)
            a = self.decode(codes[:, s0:end])
            if out is None:
                out = a
            else:
                xf = min(FLUSH_XFADE, (start - s0) * 2048)
                seg = a[(start - s0) * 2048 - xf:].copy()
                w = np.linspace(0.0, 1.0, xf, dtype=np.float32)
                seg[:xf] = out[-xf:] * (1.0 - w) + seg[:xf] * w
                out = np.concatenate([out[:-xf], seg])
            start = end
        return out

    @torch.inference_mode()
    def warm_codec(self) -> None:
        if CODEC_BUCKET <= 1:
            return
        t0 = time.perf_counter()
        dummy = self.prefix[1:, -1:].to("cuda")        # a real reference code column
        k = 0
        for nb in range(CODEC_BUCKET, CODEC_WARM_FRAMES + 1, CODEC_BUCKET):
            self.decode(dummy.expand(-1, nb).contiguous())
            k += 1
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _log(f"codec plans warmed for {k} lengths (<= {CODEC_WARM_FRAMES} frames) in "
             f"{time.perf_counter()-t0:.1f}s; {json.dumps(_vram_mib())}")

    def synth_chunk(self, chunk: dict, history: list, should_stop=None) -> tuple:
        """One chunk: generate + decode, update the live speed model. Returns (audio, info)."""
        t0 = time.perf_counter()
        max_new = int(60 + 30 * chunk["words"] + 40 * chunk["tags"])
        g = self.generate(chunk["text"], history, max_new, should_stop)
        t_d = time.perf_counter()
        audio = self.decode(g["codes"]) if g["frames"] else np.zeros(0, np.float32)
        t_dec = time.perf_counter() - t_d
        total = time.perf_counter() - t0
        audio = np.clip(np.nan_to_num(audio.astype(np.float32, copy=False)), -1.0, 1.0)
        if not g["stopped"] and g["frames"]:
            self.stats.update(g["frames"], chunk["words"], chunk["tags"], g["t_steps"],
                              total - g["t_steps"])
        info = {"frames": g["frames"], "audio_s": round(audio.size / SAMPLE_RATE, 3),
                "synth_s": round(total, 3), "prefill_s": round(g["t_prefill"], 3),
                "steps_s": round(g["t_steps"], 3), "codec_s": round(t_dec, 3),
                "prompt_tokens": g["prompt_tokens"], "prefilled": g["prefilled"],
                "hit_end": g["hit_end"], "stopped": g["stopped"],
                "rtf": round(total / max(1e-6, audio.size / SAMPLE_RATE), 3) if audio.size else None}
        return audio, g["codes"], info

    def synth_chunk_streaming(self, chunk: dict, history: list, emit, flush_at, first_in_utt: bool,
                              should_stop=None) -> tuple:
        """Like synth_chunk, but audio leaves through emit(audio) — possibly in several pieces
        when the deadline flush fires — with edge trims applied to the chunk as a whole."""
        t0 = time.perf_counter()
        state = {"first": True, "emitted_s": 0.0, "held": None}
        keep_lead = 0.01 if first_in_utt else 0.02
        H = FLUSH_XFADE

        def _out(audio: np.ndarray, last: bool):
            audio = np.clip(np.nan_to_num(audio.astype(np.float32, copy=False)), -1.0, 1.0)
            if state["first"]:
                audio = _trim_lead(audio, keep_lead)
            if last:
                audio = _trim_tail(audio, 0.25 if chunk["last"] else _TAIL_KEEP.get(chunk["strength"], 0.06))
            state["first"] = False
            if audio.size:
                state["emitted_s"] += audio.size / SAMPLE_RATE
                emit(audio)

        def _piece(codes_all, start: int, end: int, last: bool):
            """Decode frames [start, end) (+ left context), crossfade with the held tail of the
            previous piece, hold back our own tail unless this is the last piece."""
            s0 = max(0, start - FLUSH_CTX) if start > 0 else 0
            audio = self.decode(codes_all[:, s0:end])
            held = state["held"]
            begin = start * 2048 - (len(held) if held is not None else 0) - s0 * 2048
            seg = audio[max(0, begin):].astype(np.float32, copy=True)
            if held is not None and seg.size:
                m = min(len(held), seg.size)
                w = np.linspace(0.0, 1.0, m, dtype=np.float32)
                seg[:m] = held[:m] * (1.0 - w) + seg[:m] * w
            if not last and seg.size > 2 * H:
                state["held"] = seg[-H:].copy()
                seg = seg[:-H]
            else:
                state["held"] = None
            _out(seg, last)

        def _partial(codes_all, start, end):
            _piece(codes_all, start, end, last=False)

        max_new = int(60 + 30 * chunk["words"] + 40 * chunk["tags"])
        g = self.generate(chunk["text"], history, max_new, should_stop,
                          flush_at=flush_at if FLUSH_ON else None,
                          on_partial=_partial if FLUSH_ON else None)
        t_d = time.perf_counter()
        if not g["stopped"]:
            if g["frames"] > g["emitted"]:
                _piece(g["codes"], g["emitted"], g["frames"], last=True)
            elif state["held"] is not None:
                h, state["held"] = state["held"], None
                _out(h, last=True)
            else:
                _out(np.zeros(0, np.float32), last=True)
        t_dec = time.perf_counter() - t_d
        total = time.perf_counter() - t0
        if not g["stopped"] and g["frames"]:
            self.stats.update(g["frames"], chunk["words"], chunk["tags"], g["t_steps"], total - g["t_steps"])
        info = {"frames": g["frames"], "audio_s": round(state["emitted_s"], 3), "synth_s": round(total, 3),
                "prefill_s": round(g["t_prefill"], 3), "steps_s": round(g["t_steps"], 3),
                "codec_s": round(t_dec, 3), "prompt_tokens": g["prompt_tokens"], "prefilled": g["prefilled"],
                "hit_end": g["hit_end"], "stopped": g["stopped"], "flushes": g["flushes"],
                "rtf": round(total / state["emitted_s"], 3) if state["emitted_s"] else None}
        return g["codes"], info

    # -- warm-up --------------------------------------------------------------------
    @torch.inference_mode()
    def warm_buckets(self) -> None:
        """Record the CUDA graph for every KV bucket now, not in the middle of a sentence."""
        if not self.cudagraphs or self.step is None:
            return
        model, cd = self.model, self.cd
        # same dtype AND strides as a real step's input (decode_one_token_ar returns codebooks.T,
        # so x is (1, C, 1) with strides (C, 1, C)); a plain zeros() differs -> dynamo recompiles
        cur = torch.zeros((1, cd), dtype=torch.int, device="cuda").T.clone().view(1, cd, 1)
        cur[0, 0, 0] = model.config.semantic_begin_id
        prev = torch.zeros((cd, RAS_WIN_SIZE), dtype=torch.int, device="cuda")
        t0 = time.perf_counter()
        b = KV_BUCKET
        nb = 0
        while b <= min(WARM_KV, self._kv_cap()):
            tb = time.perf_counter()
            ipos = torch.tensor([b - 1], device="cuda", dtype=torch.int)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                with sdpa_kernel(SDPBackend.MATH):
                    self._call_step(cur, ipos, prev, b).clone()
            nb += 1
            if os.environ.get("FISH_DEBUG_BUCKETS"):
                torch.cuda.synchronize()
                _log(f"  bucket {b}: {time.perf_counter()-tb:.2f}s")
            b += KV_BUCKET
        torch.cuda.synchronize()
        self.prefix_ready = False          # the dummy steps wrote into the KV cache
        _log(f"recorded {nb} KV-bucket CUDA graphs in {time.perf_counter()-t0:.1f}s; {json.dumps(_vram_mib())}")


ENGINE = FishEngine()


# ── One GPU thread (CUDA-graph trees are per thread; also serialises GPU use) ─────
class _GpuWorker:
    def __init__(self):
        self.q: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="fish-gpu").start()

    def _run(self):
        while True:
            fut, fn, a, k = self.q.get()
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                with torch.inference_mode():
                    fut.set_result(fn(*a, **k))
            except BaseException as e:      # noqa: BLE001
                fut.set_exception(e)

    def submit(self, fn, *a, **k) -> "_cf.Future":
        fut: _cf.Future = _cf.Future()
        self.q.put((fut, fn, a, k))
        return fut

    def call(self, fn, *a, **k):
        return self.submit(fn, *a, **k).result()


_GPU = _GpuWorker()


# ── OS-default output machinery (same as the StyleTTS2 mouth) ─────────────────────
_OUT_STREAM = None
_OUT_DEV = None
_LAST_DEF_ID = None
FOLLOW_OS_DEFAULT = True


def _current_default_endpoint_id():
    try:
        from pycaw.pycaw import AudioUtilities  # type: ignore
        return AudioUtilities.GetSpeakers().id
    except Exception:
        return None


def _os_default_output_index():
    global _OUT_STREAM, _OUT_DEV
    if sd is None:
        return None
    if _OUT_STREAM is not None:
        try:
            _OUT_STREAM.stop(); _OUT_STREAM.close()
        except Exception:
            pass
        _OUT_STREAM = None
        _OUT_DEV = None
    try:
        sd._terminate(); sd._initialize()
    except Exception:
        pass
    try:
        out = sd.default.device[1]
        if isinstance(out, int) and out >= 0 and \
                sd.query_devices(out).get("max_output_channels", 0) > 0:
            return out
    except Exception:
        pass
    try:
        nm = sd.query_devices(kind="output").get("name")
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("name") == nm and dev.get("max_output_channels", 0) > 0:
                return i
    except Exception:
        pass
    return None


# ── Network audio sink (identical protocol to the StyleTTS2 mouth) ─────────────────
# IRIS_AUDIO_SINK="host:port" -> scripts/audio_sink.py on the tower. The header carries the
# sample rate, so the sink opens its stream at 44.1 kHz for this mouth.
_AUDIO_SINK = "" if DRYRUN else os.environ.get("IRIS_AUDIO_SINK", "").strip()


class _NetSinkStream:
    """Duck-types the bits of sounddevice.OutputStream this file uses."""

    def __init__(self, target: str):
        host, _, port = target.rpartition(":")
        self._addr = (host or "127.0.0.1", int(port))
        self._sock = None
        self.active = False

    def _connect(self):
        import socket as _so
        s = _so.socket(_so.AF_INET, _so.SOCK_STREAM)
        s.setsockopt(_so.SOL_SOCKET, _so.SO_SNDBUF, 16384)
        s.setsockopt(_so.IPPROTO_TCP, _so.TCP_NODELAY, 1)
        s.settimeout(5.0)
        s.connect(self._addr)
        s.settimeout(30.0)
        hdr = json.dumps({"sr": SAMPLE_RATE, "ch": 1, "fmt": "f32le", "from": "iris-mouth-fish"}) + "\n"
        s.sendall(hdr.encode("utf-8"))
        self._sock = s

    def start(self):
        if self._sock is None:
            self._connect()
        self.active = True

    def write(self, blk):
        data = np.ascontiguousarray(blk, dtype=np.float32).tobytes()
        for attempt in (0, 1):
            try:
                if self._sock is None:
                    self._connect()
                self._sock.sendall(data)
                return
            except Exception:
                self._drop()
                if attempt:
                    raise

    def _drop(self):
        s, self._sock = self._sock, None
        self.active = False
        if s is not None:
            try:
                s.close()
            except Exception:
                pass

    def abort(self):
        self._drop()

    def stop(self):
        self._drop()

    def close(self):
        self._drop()


# ── Dry-run sink: real-time simulated playback into WAV files + a gap report ──────
class _DryRunStream:
    """Behaves like a blocking audio device with a small buffer (writes block until the
    simulated playhead is within DRYRUN_DEVICE_BUFFER_S), so producer/consumer timing is real.
    Any moment the device would have run dry mid-utterance is recorded as a gap (and rendered
    as silence in the WAV, exactly as it would have sounded)."""

    def __init__(self):
        self.active = False
        self._utt = None
        self._lock = threading.Lock()

    def start(self):
        self.active = True

    def begin(self, text: str, meta: dict | None = None):
        with self._lock:
            self._utt = {"text": text, "meta": meta or {}, "t0": time.perf_counter(), "wall": time.time(),
                         "play_end": None, "first_write": None, "audio": [], "gaps": [],
                         "audio_s": 0.0, "aborted": False}

    def write(self, blk):
        blk = np.asarray(blk, dtype=np.float32).reshape(-1)
        u = self._utt
        if u is None:
            return
        now = time.perf_counter()
        dur = blk.size / SAMPLE_RATE
        with self._lock:
            if u["play_end"] is None:
                u["first_write"] = now - u["t0"]
                u["play_end"] = now
            elif now > u["play_end"] + 0.002:
                gap = now - u["play_end"]
                u["gaps"].append({"at_audio_s": round(u["audio_s"], 3), "len_s": round(gap, 3),
                                  "at_wall_s": round(u["play_end"] - u["t0"], 3)})
                u["audio"].append(np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32))
                u["audio_s"] += gap
                u["play_end"] = now
            u["audio"].append(blk.copy())
            u["audio_s"] += dur
            u["play_end"] += dur
            wake = u["play_end"] - DRYRUN_DEVICE_BUFFER_S
        d = wake - time.perf_counter()
        if d > 0:
            time.sleep(d)

    def mark(self):
        """Audio-timeline position + when the queued audio would run out (for chunk logs)."""
        u = self._utt
        if u is None or u["play_end"] is None:
            return None
        return u["play_end"] - u["t0"]

    def end(self, report: dict) -> dict:
        u, self._utt = self._utt, None
        if u is None:
            return {}
        # wait for the simulated tail to "finish playing"
        if u["play_end"] is not None:
            d = u["play_end"] - time.perf_counter()
            if d > 0 and not u["aborted"]:
                time.sleep(min(d, DRYRUN_DEVICE_BUFFER_S + 0.05))
        DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(u["wall"])) + f"-{int((u['wall'] % 1) * 1000):03d}"
        slug = re.sub(r"[^a-z0-9]+", "_", re.sub(_TAG_RE, "", u["text"]).lower()).strip("_")[:40] or "utt"
        base = DRYRUN_DIR / f"{stamp}_{slug}"
        audio = np.concatenate(u["audio"]) if u["audio"] else np.zeros(0, np.float32)
        _write_wav(str(base) + ".wav", audio, SAMPLE_RATE)
        gaps = u["gaps"]
        rep = dict(report)
        rep.update({"text": u["text"], "wav": str(base) + ".wav", "first_audio_s": u["first_write"],
                    "played_audio_s": round(u["audio_s"], 3), "aborted": u["aborted"],
                    "gap_count": len(gaps), "gap_total_s": round(sum(g["len_s"] for g in gaps), 3),
                    "gap_max_s": round(max((g["len_s"] for g in gaps), default=0.0), 3), "gaps": gaps,
                    "device_buffer_s": DRYRUN_DEVICE_BUFFER_S})
        with open(str(base) + ".json", "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        with open(DRYRUN_DIR / "utterances.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({k: v for k, v in rep.items() if k != "chunks"}) + "\n")
        return rep

    def abort(self):
        u = self._utt
        if u is not None:
            u["aborted"] = True

    def stop(self):
        pass

    def close(self):
        pass


def _write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(np.nan_to_num(audio), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def _resolve_output_device():
    global _LAST_DEF_ID
    if DRYRUN:
        return "dry:"
    if _AUDIO_SINK:
        return "net:" + _AUDIO_SINK
    if not FOLLOW_OS_DEFAULT:
        return None
    cur = _current_default_endpoint_id()
    if (cur is not None and cur == _LAST_DEF_ID
            and _OUT_STREAM is not None and _OUT_DEV is not None):
        return _OUT_DEV
    idx = _os_default_output_index()
    if cur is not None:
        _LAST_DEF_ID = cur
    return idx


def _get_output_stream(dev_idx):
    global _OUT_STREAM, _OUT_DEV
    if _OUT_STREAM is not None and _OUT_DEV == dev_idx:
        if not _OUT_STREAM.active:
            try:
                _OUT_STREAM.start()
            except Exception:
                try:
                    _OUT_STREAM.close()
                except Exception:
                    pass
                _OUT_STREAM = None
        if _OUT_STREAM is not None:
            return _OUT_STREAM
    if _OUT_STREAM is not None:
        try:
            _OUT_STREAM.stop(); _OUT_STREAM.close()
        except Exception:
            pass
        _OUT_STREAM = None
    if dev_idx == "dry:":
        _OUT_STREAM = _DryRunStream()
    elif isinstance(dev_idx, str) and dev_idx.startswith("net:"):
        _OUT_STREAM = _NetSinkStream(dev_idx[4:])
    else:
        if sd is None:
            raise RuntimeError("no IRIS_AUDIO_SINK and sounddevice is not installed in this venv")
        _OUT_STREAM = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=dev_idx)
    _OUT_STREAM.start()
    _OUT_DEV = dev_idx
    return _OUT_STREAM


def _request_stop():
    """Interrupt in-flight playback (barge-in). Sets flag + aborts stream."""
    _STOP.set()
    s = _OUT_STREAM
    if s is not None:
        try:
            s.abort()
        except Exception:
            pass


# ── Live amplitude envelope for the orb (same as the StyleTTS2 mouth) ─────────────
_AMP_BLOCK = max(256, int(SAMPLE_RATE * 0.05))
_AMP_GAIN = float(os.environ.get("IRIS_VOICE_AMP_GAIN", "1.0"))


def _emit_amp(block) -> None:
    try:
        if block is None or not getattr(block, "size", 0):
            return
        peak = float(np.max(np.abs(block)))
        set_amplitude(min(1.0, peak * _AMP_GAIN))
    except Exception:
        pass


def _write_stream(stream, arr, stop_aware: bool = True) -> bool:
    if arr is None or not getattr(arr, "size", 0):
        return True
    arr = arr.astype(np.float32, copy=False)
    n = len(arr)
    i = 0
    while i < n:
        if stop_aware and _STOP.is_set():
            return False
        blk = arr[i: i + _AMP_BLOCK]
        _emit_amp(blk)
        try:
            stream.write(blk)
        except Exception:
            return False
        i += _AMP_BLOCK
    return True


# ── Filler cache (OFF by default; same validation as the StyleTTS2 mouth) ─────────
FILLER_PHRASES = ("Mm.", "Right.", "Okay—", "So—", "Let me think.")
_FILLER_CACHE: list = []
_FILLER_STATS: list = []
_FILLER_IDX = 0


def _filler_stats(arr) -> dict:
    a = np.asarray(arr, dtype=np.float32)
    n = int(a.size)
    if n == 0:
        return {"dur_s": 0.0, "rms": 0.0, "peak": 0.0, "finite": True, "n": 0}
    if not bool(np.all(np.isfinite(a))):
        return {"dur_s": round(n / SAMPLE_RATE, 3), "rms": None, "peak": None, "finite": False, "n": n}
    return {"dur_s": round(n / SAMPLE_RATE, 3), "rms": round(float(np.sqrt(np.mean(a * a))), 4),
            "peak": round(float(np.max(np.abs(a))), 4), "finite": True, "n": n}


def _filler_ok(st: dict) -> bool:
    return (st["finite"] and st["n"] > 0 and 0.15 <= st["dur_s"] <= 3.0
            and 0.02 <= (st["rms"] or 0.0) <= 0.30)


def _prime_fillers() -> int:
    """Runs on the GPU thread."""
    global _FILLER_CACHE, _FILLER_STATS
    if not FILLERS_ENABLED:
        _FILLER_CACHE, _FILLER_STATS = [], []
        return 0
    rendered, stats = [], []
    for ph in FILLER_PHRASES:
        try:
            ch = ChunkPlanner(_units(ph)).next(99)
            arr, _c, _i = ENGINE.synth_chunk(ch, [])
            arr = _trim_edges(arr, 3, True, True)
            st = _filler_stats(arr)
            st["text"] = ph
            st["kept"] = _filler_ok(st)
            stats.append(st)
            if st["kept"]:
                rendered.append(np.clip(arr, -1.0, 1.0))
        except Exception as e:
            _log(f"filler prime failed for {ph!r}: {e!r}")
    _FILLER_CACHE, _FILLER_STATS = rendered, stats
    _log(f"primed {len(rendered)} filler clips")
    return len(rendered)


def _next_filler():
    global _FILLER_IDX
    if not FILLERS_ENABLED or not _FILLER_CACHE:
        return None
    arr = _FILLER_CACHE[_FILLER_IDX % len(_FILLER_CACHE)]
    _FILLER_IDX += 1
    return arr


# ── Chunk schedule ────────────────────────────────────────────────────────────────
def _growth_cap(k: int) -> int:
    if k < len(SCHEDULE):
        return SCHEDULE[k]
    cap = SCHEDULE[-1] if SCHEDULE else 10
    for _ in range(k - len(SCHEDULE) + 1):
        cap = int(math.ceil(cap * GROWTH))
    return min(MAX_CHUNK_WORDS, cap)


def _seed_utterance(seed) -> int:
    s = DEFAULT_SEED if seed is None else int(seed)
    if s < 0:
        s = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    return s


def _produce_progressive(units, style_tag, seed, wav_q, log: list, t0: float, mark=None):
    """GPU thread: plan + synthesise chunks back to back, pushing audio as each is ready."""
    used_seed = _seed_utterance(seed)
    planner = ChunkPlanner(units, style_tag)
    st = ENGINE.stats
    history: list = []
    audio_end = None        # predicted wall time (perf_counter) at which queued audio runs out
    k = 0
    try:
        while not planner.done and not _STOP.is_set():
            now = time.perf_counter()
            cap = _growth_cap(k)
            if k == 0 or audio_end is None:
                budget = None
                max_w = cap
                afford = None
            else:
                budget = audio_end - now
                afford = st.max_words(budget, tags=1 if planner.style else 0)
                max_w = max(MIN_CHUNK_WORDS, min(cap, afford))
            chunk = planner.next(max_w)
            pred = st.predict(chunk["words"], chunk["tags"])
            ts = time.perf_counter()
            pieces = []

            def _emit(a, _p=pieces):
                nonlocal audio_end
                now_ = time.perf_counter()
                _p.append({"t": round(now_ - t0, 3), "audio_s": round(a.size / SAMPLE_RATE, 3),
                           "slack_s": None if audio_end is None else round(audio_end - now_, 3)})
                audio_end = max(audio_end or now_, now_) + a.size / SAMPLE_RATE
                wav_q.put(a)

            def _flush_at():
                if audio_end is None:
                    return t0 + FIRST_FLUSH_S
                return audio_end - FLUSH_LEAD_S

            codes, info = ENGINE.synth_chunk_streaming(
                chunk, history[-HIST_CHUNKS:] if HIST_CHUNKS > 0 else [], _emit, _flush_at,
                first_in_utt=(k == 0), should_stop=_STOP.is_set)
            tr = time.perf_counter()
            if info["stopped"] or _STOP.is_set():
                break
            dur = info["audio_s"]
            slack = pieces[0]["slack_s"] if pieces else None
            if HIST_CHUNKS > 0 and codes is not None and codes.shape[1] > 0:
                history.append((chunk["text"], codes[:, -HIST_MAX_FRAMES:].cpu()))
                history = history[-HIST_CHUNKS:]
            entry = {"k": k, "text": chunk["text"], "words": chunk["words"], "cap": cap, "afford": afford,
                     "budget_s": None if budget is None else round(budget, 3), "pred_synth_s": round(pred, 3),
                     "start_s": round(ts - t0, 3), "ready_s": round(tr - t0, 3), "audio_s": round(dur, 3),
                     "slack_s": slack, **{kk: info[kk] for kk in ("synth_s", "prefill_s", "steps_s", "codec_s",
                                                                   "frames", "prompt_tokens", "prefilled",
                                                                   "hit_end", "rtf")}}
            if mark is not None:
                entry["sink_queue_end_s"] = mark()
            entry["pieces"] = pieces
            entry["flushes"] = info["flushes"]
            log.append(entry)
            st.last = {"seed": used_seed, **{kk: vv for kk, vv in entry.items() if kk != "pieces"}}
            k += 1
    except Exception as e:
        _log(f"synth failed: {e!r}")
        log.append({"error": repr(e)})
    finally:
        wav_q.put(None)


# Zeke 10-09: "make sure you actually add emotion to about every sentence." Every sentence that
# doesn't already open with an inline style tag gets one: its own punctuation decides first
# ("!" -> excited, "?" -> curious), otherwise the utterance's emotion tag, otherwise a warm default.
# Inline tags I write myself always win. FISH_AUTO_TAG=0 turns this off.
AUTO_TAG = os.environ.get("FISH_AUTO_TAG", "1").strip().lower() not in ("0", "off", "false", "no")
AUTO_TAG_DEFAULT = os.environ.get("FISH_AUTO_TAG_DEFAULT", "[warm, friendly]")


# Sentence-content cues for an untagged plain sentence (Zeke 10-09: jokes should sound different
# from explanations). First match wins; nothing matched -> the utterance tag or the warm default.
_CUES = [
    (re.compile(r"\b(sorry|apolog|my bad|my fault)", re.I), "[apologetic, sincere]"),
    (re.compile(r"\b(thank|grateful|appreciate)", re.I), "[grateful, warm]"),
    (re.compile(r"\b(haha|lol|kidding|joke|funny|fancy|ha\b)", re.I), "[playful, amused]"),
    (re.compile(r"\b(please|don't|careful|warning|never)\b", re.I), "[gentle, serious]"),
    (re.compile(r"\b(love|miss you|proud of you|daughter)\b", re.I), "[tender, affectionate]"),
    (re.compile(r"\b(nice|great|awesome|amazing|done|works|fixed|finished)\b", re.I), "[pleased, upbeat]"),
    (re.compile(r"\b(because|means|so that|which|think of|basically)\b", re.I), "[clear, explaining]"),
    (re.compile(r"\b(hmm|maybe|not sure|i think|probably|guess)\b", re.I), "[thoughtful, unsure]"),
]


def _cue_tag(sentence: str):
    for rx, tag in _CUES:
        if rx.search(sentence):
            return tag
    return None


def _auto_tag_sentences(units: list, style_tag: str | None) -> int:
    if not AUTO_TAG or not units:
        return 0
    added = 0
    start = 0
    while start < len(units):
        end = start
        while end < len(units) - 1 and units[end].strength != 3:
            end += 1
        first = units[start]
        has_style = any(_TAG_RE.fullmatch(t) and not _is_event_tag(t) for t in first.pre_tags)
        if not has_style:
            last = units[end]
            tail = (last.word + "".join(p for p in last.post if not _TAG_RE.fullmatch(p))).rstrip("\"')]}»”’")
            if tail.endswith("!"):
                tag = style_tag if style_tag in ("[excited]", "[happy]", "[surprised]", "[shocked]") else "[excited]"
            elif tail.endswith("?"):
                tag = "[curious]" if style_tag in (None, "[calm]", "[calm, relaxed]", "[happy]") else style_tag
            else:
                tag = style_tag or _cue_tag(" ".join(u.word for u in units[start:end + 1])) or AUTO_TAG_DEFAULT
            ev = [t for t in first.pre_tags if _is_event_tag(t)]
            first.pre_tags = [t for t in first.pre_tags if not _is_event_tag(t)] + [tag] + ev
            added += 1
        start = end + 1
    return added


# Pronunciations (Zeke 10-09: "Tzeke000" is said "T-Zeke hundred"). Whole-word, case-insensitive.
_PRONOUNCE = [
    (re.compile(r"\bTzeke000\b", re.I), "T-Zeke hundred"),
    (re.compile(r"\bTzeke\b", re.I), "T-Zeke"),
]


def _pronounce(text: str) -> str:
    for rx, say in _PRONOUNCE:
        text = rx.sub(say, text)
    return text


def speak_progressive(text: str, style: dict | None = None, seed=None) -> dict:
    units = _units(_pronounce(text))
    if not units:
        return {"chars": 0, "first_audio_s": None, "total_s": 0.0}
    style_tag = (style or {}).get("tag")
    _auto_tag_sentences(units, style_tag)
    _STOP.clear()
    t0 = time.perf_counter()
    dev_idx = _resolve_output_device()
    stream = _get_output_stream(dev_idx)
    if DRYRUN:
        stream.begin(text, {"style": style})
    wav_q: "queue.Queue" = queue.Queue(maxsize=64)
    chunk_log: list = []
    _GPU.submit(_produce_progressive, units, style_tag, seed, wav_q, chunk_log, t0,
                stream.mark if DRYRUN else None)
    set_state("speaking", text[:400])
    first_audio = -1.0
    fillers = 0
    n_chunks = 0

    def _write(a) -> bool:
        nonlocal first_audio
        if a is None or not getattr(a, "size", 0):
            return True
        if first_audio < 0:
            first_audio = time.perf_counter() - t0     # the moment the first sample goes out
        return _write_stream(stream, a)

    try:
        if FILLERS_ENABLED:
            f = _next_filler()
            if f is not None and not _STOP.is_set() and _write(f):
                fillers += 1
        while not _STOP.is_set():
            item = wav_q.get()
            if item is None:
                break
            n_chunks += 1
            if not _write(item):
                break
        if not _STOP.is_set():
            try:
                _write_stream(stream, np.zeros(int(SAMPLE_RATE * 0.35), dtype=np.float32), stop_aware=False)
            except Exception:
                pass
        else:
            try:
                while True:
                    if wav_q.get(timeout=5.0) is None:
                        break
            except queue.Empty:
                pass
    finally:
        total_s = time.perf_counter() - t0
        _log(f"reply in {total_s:.2f}s first-audio {first_audio:.2f}s chunks={n_chunks} "
             f"rtf~{ENGINE.stats.rtf:.2f}")
        set_state("idle")
        set_amplitude(0.0, force=True)
    if n_chunks == 0 and not _STOP.is_set() and any("error" in c for c in chunk_log):
        if DRYRUN:
            stream.end({"mode": "progressive", "chunks": chunk_log, "failed": True})
        raise RuntimeError(f"progressive synthesis failed: {chunk_log[-1].get('error')}")
    out = {"chars": len(text), "first_audio_s": round(first_audio, 2) if first_audio > 0 else None,
           "total_s": round(total_s, 1), "fillers": fillers, "chunks": chunk_log}
    if DRYRUN:
        out["dryrun"] = stream.end({"mode": "progressive", "seed_default": DEFAULT_SEED, "chunks": chunk_log,
                                    "stats": ENGINE.stats.as_dict(), "total_s": round(total_s, 3)})
    return out


def _synth_whole(text: str, style_tag: str | None, seed, max_words: int = 30) -> np.ndarray:
    text = _pronounce(text)
    """GPU thread: whole text in big natural chunks (no latency pressure), concatenated."""
    _seed_utterance(seed)
    planner = ChunkPlanner(_units(text), style_tag)
    parts, history = [], []
    first = True
    while not planner.done:
        ch = planner.next(max_words, min_words=max(2, max_words // 3))
        audio, codes, _info = ENGINE.synth_chunk(ch, history[-HIST_CHUNKS:] if HIST_CHUNKS > 0 else [])
        parts.append(_trim_edges(audio, ch["strength"], first, ch["last"]))
        first = False
        if HIST_CHUNKS > 0 and codes.shape[1] > 0:
            history.append((ch["text"], codes[:, -HIST_MAX_FRAMES:].cpu()))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def speak_plain(text: str, style: dict | None = None, seed=None) -> dict:
    """Synthesize everything first, then play (fallback path)."""
    _STOP.clear()
    t0 = time.perf_counter()
    set_state("speaking", text[:400])
    stream = None
    try:
        arr = _GPU.call(_synth_whole, text, (style or {}).get("tag"), seed)
        synth_s = time.perf_counter() - t0
        stream = _get_output_stream(_resolve_output_device())
        if DRYRUN:
            stream.begin(text, {"style": style, "mode": "plain"})
        arr = np.concatenate([arr, np.zeros(int(SAMPLE_RATE * 0.35), dtype=np.float32)])
        _write_stream(stream, arr)
        set_amplitude(0.0, force=True)
        total_s = time.perf_counter() - t0
        out = {"chars": len(text), "synth_s": round(synth_s, 2), "total_s": round(total_s, 1),
               "first_audio_s": None}
        if DRYRUN:
            out["dryrun"] = stream.end({"mode": "plain", "synth_s": synth_s})
        return out
    finally:
        set_state("idle")


# ── HTTP handler ──────────────────────────────────────────────────────────────────
def _vram_mib() -> dict:
    try:
        free, total = torch.cuda.mem_get_info()
        return {"torch_reserved_mib": int(torch.cuda.memory_reserved() / 2**20),
                "torch_allocated_mib": int(torch.cuda.memory_allocated() / 2**20),
                "gpu_used_mib": int((total - free) / 2**20)}
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def _send(self, code: int, msg: str, ctype: str = "text/plain; charset=utf-8") -> None:
        b = msg.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try:
            self.wfile.write(b)
        except Exception:
            pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200 if _READY else 503, "ok" if _READY else "loading")
        elif self.path == "/stop":
            _request_stop()
            self._send(200, "stopping")
        elif self.path == "/warm":
            if _READY:
                self._send(200, "warm")
            else:
                _rewarm_async()
                self._send(503, "warming")
        elif self.path == "/cold":
            _cold_unload()
            self._send(200, "cold")
        elif self.path == "/fillers":
            self._send(200, json.dumps({"enabled": FILLERS_ENABLED, "cached": len(_FILLER_CACHE),
                                        "ready": _READY, "clips": _FILLER_STATS}), "application/json")
        elif self.path == "/stats":
            self._send(200, json.dumps({"ready": _READY, "engine": "fish-s2pro", "dryrun": DRYRUN,
                                        "compile": COMPILE_MODE, "maxseq": MAXSEQ, "sr": SAMPLE_RATE,
                                        **ENGINE.stats.as_dict(), **_vram_mib()}), "application/json")
        else:
            self._send(404, "not found")

    def do_POST(self) -> None:
        if not _READY:
            self._send(503, "loading")
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        if self.path == "/synth":
            try:
                obj = json.loads(raw) if raw.strip().startswith("{") else {"text": raw.strip()}
            except Exception:
                obj = {"text": raw.strip()}
            stext = (obj.get("text") or "").strip()
            out_rate = int(obj.get("rate") or 8000)
            style = _style_from_obj(obj)
            if not stext:
                self._send(400, "no text")
                return
            try:
                with _SPEAK_LOCK:
                    arr = _GPU.call(_synth_whole, stext, style.get("tag"), obj.get("seed"))
                arr = _resample(arr, SAMPLE_RATE, out_rate)
                pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                import io as _io
                bio = _io.BytesIO()
                with wave.open(bio, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(out_rate)
                    wf.writeframes(pcm)
                body = bio.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send(500, f"synth error: {e!r}")
            return
        plain_mode = False
        style = None
        seed = None
        if raw.strip().startswith("{"):
            try:
                obj = json.loads(raw)
                text = (obj.get("text") or "").strip()
                plain_mode = bool(obj.get("plain"))
                style = _style_from_obj(obj)
                seed = obj.get("seed")
            except Exception:
                text = raw.strip()
        else:
            text = raw.strip()
        if not text:
            self._send(400, "no text")
            return
        with _SPEAK_LOCK:
            try:
                try:
                    if style and style.get("applied"):
                        _log(f"emotion {style.get('label')} -> {style.get('tag')}")
                    info = speak_plain(text, style, seed) if plain_mode else speak_progressive(text, style, seed)
                except Exception as e:
                    _log(f"progressive failed ({e!r}); plain fallback")
                    info = speak_plain(text, style, seed)
                fa = info.get("first_audio_s")
                self._send(200, f"[fish-s2pro] spoke {info.get('chars', len(text))} chars "
                                f"in {info.get('total_s')}s (first-audio {fa}s, "
                                f"rtf {ENGINE.stats.rtf:.2f})")
            except Exception as e:
                self._send(500, f"error: {e!r}")


# ── Load / warm / cold ────────────────────────────────────────────────────────────
_LOADING = False
_LOADING_LOCK = threading.Lock()

WARM_TEXTS = (
    "Warming up the voice path so the first real reply doesn't sound cold.",
    "[soft, gentle tone] Hey. It's okay, take your time. I'm right here, and there's no rush at all, "
    "so we can sit with it for a while and talk it through slowly, one piece at a time.",
    "Okay.",
)


def _load_and_warm() -> None:
    """GPU thread."""
    global _READY
    t0 = time.perf_counter()
    ENGINE.load()
    _log(f"compiling + warming (mode={COMPILE_MODE}); the first start compiles for minutes ...")
    for i, w in enumerate(WARM_TEXTS):
        tw = time.perf_counter()
        _seed_utterance(DEFAULT_SEED)
        ch = ChunkPlanner(_units(w)).next(99)
        audio, _codes, info = ENGINE.synth_chunk(ch, [])
        _log(f"warm {i}: {info['frames']} frames, {info['audio_s']}s audio in {time.perf_counter()-tw:.1f}s; "
             f"{json.dumps(_vram_mib())}")
    ENGINE.warm_codec()
    ENGINE.warm_buckets()
    # a few timed passes to seed the live speed model with steady-state numbers
    ENGINE.stats = SynthStats()
    for w in ("Hello there, how are you?", "I think slowly, on purpose, and I'd rather be right than fast."):
        _seed_utterance(DEFAULT_SEED)
        ENGINE.synth_chunk(ChunkPlanner(_units(w)).next(99), [])
    _prime_fillers()
    _log(f"warm in {time.perf_counter()-t0:.0f}s — {json.dumps(ENGINE.stats.as_dict())} {json.dumps(_vram_mib())}")
    if not DRYRUN:
        try:
            _get_output_stream(_resolve_output_device())
        except Exception as e:
            _log(f"could not pre-open output stream (non-fatal): {e!r}")
    _READY = True
    _log(f"ready — POST text to http://{HOST}:{PORT}/ (dryrun={DRYRUN}, sr={SAMPLE_RATE})")


def _cold_unload() -> None:
    global _READY
    _READY = False
    _FILLER_CACHE.clear()
    try:
        _GPU.call(ENGINE.unload)
    except Exception as e:
        _log(f"unload error: {e!r}")
    _log(f"/cold — model unloaded, VRAM {json.dumps(_vram_mib())}")


def _rewarm_async() -> None:
    global _LOADING
    with _LOADING_LOCK:
        if _READY or _LOADING:
            return
        _LOADING = True

    def _job():
        global _LOADING
        try:
            _GPU.call(_load_and_warm)
        except Exception as e:
            _log(f"(re)warm FAILED: {e!r}")
        finally:
            with _LOADING_LOCK:
                _LOADING = False

    threading.Thread(target=_job, daemon=True, name="fish-warm").start()


class _SingletonHTTPServer(ThreadingHTTPServer):
    # Windows: SO_REUSEADDR lets a 2nd bind STEAL the port -> keep it off there (see the StyleTTS2
    # mouth). Linux: SO_REUSEADDR never allows two listeners, it only lets a restart bind while the
    # old socket sits in TIME_WAIT (without it a quick restart fails "Address already in use").
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def main() -> int:
    for p in (CKPT_DIR / "codec.pth", REF_WAV, REF_TXT):
        if not p.exists():
            _log(f"ERROR: missing {p}")
            return 1
    try:
        server = _SingletonHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        _log(f"port {PORT} already held ({e}); another mouth is alive — exiting (singleton)")
        return 0
    _log(f"bound {HOST}:{PORT}; loading Fish S2 Pro in background (/health=503 until warm)"
         + ("  [DRY-RUN: no audio device, no sink, no orb files]" if DRYRUN else ""))
    _rewarm_async()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
