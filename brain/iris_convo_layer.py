"""brain/iris_convo_layer.py - the FAST conversation layer beside my slow cognition (10-31).

My cognition (one Claude session) takes seconds per turn and only sees his words once he has
finished. Some human moves have to happen inside his turn, in well under a second or two:
offering the word he is searching for, a "checking, give me a sec", a backchannel. This module is
the host-side coordinator for those moves; it watches the LIVE partial transcript
(voice/iris_live_words.read_current) and acts through injected functions, so every decision is
unit-testable without a mic, a GPU or a mouth.

Built so far (flags default OFF, see the design doc memory/v3_phase0/conversation_layer_design_2026-10-31.md):
  * WordAssist  - IRIS_WORD_ASSIST=1: he says "what's it called" and pauses -> a local LLM
                  (Ollama, ~0.1 s warm on the V100) guesses the word -> I say it as a question.
  * Offers      - what the fast layer said during his turn, handed to cognition with his final
                  words so I know what he heard from me ("I offered 'geometry nodes?'").
The async lookup path ("checking, give me a second") lives in brain/iris_lookups.py.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "voice") not in sys.path:
    sys.path.insert(1, str(_REPO / "voice"))

import iris_word_search as _ws  # noqa: E402


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")


WORD_ASSIST_ON = _flag("IRIS_WORD_ASSIST")
LOOKUPS_ON = _flag("IRIS_CONVO_LOOKUPS")
WORD_MODEL = os.environ.get("IRIS_WORD_ASSIST_MODEL", _ws.DEFAULT_MODEL)
CONVO_WINDOW_S = float(os.environ.get("IRIS_WORD_ASSIST_CONVO_S", "180"))   # he spoke to me this recently
LITTLE_BRAIN_PARK = _REPO / "state" / "little_brain" / "pilot_deliberately_off.json"
CONTROL_FILE = _REPO / "scratch" / "voice_control.json"


# ── what the fast layer said during his turn (handed to cognition) ──────────────
class Offers:
    def __init__(self, keep_s: float = 90.0):
        self.keep_s = keep_s
        self._q = collections.deque(maxlen=8)
        self._lock = threading.Lock()

    def add(self, kind: str, said: str, heard: str, now: "float | None" = None) -> None:
        with self._lock:
            self._q.append({"ts": time.time() if now is None else now, "kind": kind,
                            "said": said, "heard": (heard or "")[-160:]})

    def take_note(self, now: "float | None" = None) -> str:
        """One bracketed line for the next voice turn (and forget them), '' if nothing recent."""
        now = time.time() if now is None else now
        with self._lock:
            items = [o for o in self._q if now - o["ts"] <= self.keep_s]
            self._q.clear()
        if not items:
            return ""
        bits = [f"{o['kind']}: I said {o['said']!r} when he was at \"...{o['heard'][-80:]}\"" for o in items]
        return ("[While he was talking, my fast layer already spoke for me - " + "; ".join(bits)
                + ". He heard that; don't repeat it, and if the guess was wrong just follow his lead.]")


OFFERS = Offers()


# ── gates ────────────────────────────────────────────────────────────────────────
def _read_json(p: Path) -> dict:
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def little_brain_parked(path: "Path | None" = None) -> bool:
    """The V100's local-LLM park (Zeke's call to lift). Every path that can load an Ollama model
    reads it - 'a park is only as good as its LAST reader' (09-06). Missing file = not parked."""
    return bool(_read_json(path or LITTLE_BRAIN_PARK).get("off", False))


def may_speak_now(ctl: "dict | None" = None, now=None, flag_path=None) -> "tuple[bool, str]":
    """Same sound gates as my backchannels: quiet hours, mouth_muted, the voice off-flag, plus the
    hot kill switch voice_control.json {"word_assist": false}."""
    try:
        import iris_backchannel as _bc
    except Exception:
        return False, "backchannel module unavailable"
    ctl = _read_json(CONTROL_FILE) if ctl is None else ctl
    if not bool(ctl.get("word_assist", True)):
        return False, "voice_control word_assist=false"
    if bool(ctl.get("mouth_muted", False)):
        return False, "mouth_muted"
    if _bc.in_quiet_hours(now) or _bc._voice_flag_off(flag_path):
        return False, "quiet hours / voice off"
    return True, ""


# ── word assist ──────────────────────────────────────────────────────────────────
class WordAssist:
    """Drive with step(now) every ~150 ms. Injected:
        read_fn()          -> live partial dict (iris_live_words.read_current) or None
        suggest_fn(p, ctx) -> (word | None, seconds)      [blocking; run off the event loop]
        speak_fn(text)     -> None                         [the mouth]
        gate_fn()          -> (ok, why)                    [sound + conversation gates]
        context_fn()       -> list[str]                    [recent conversation lines]
    One offer per utterance; the suggestion is re-checked against the newest partial before it is
    spoken (he may have found the word himself while the model thought)."""

    def __init__(self, read_fn, suggest_fn, speak_fn, gate_fn, context_fn=lambda: [],
                 offers: "Offers | None" = None, cfg: "dict | None" = None, log=None):
        self.read_fn, self.suggest_fn, self.speak_fn = read_fn, suggest_fn, speak_fn
        self.gate_fn, self.context_fn = gate_fn, context_fn
        self.offers = offers if offers is not None else OFFERS
        self.policy = _ws.WordSearchPolicy(cfg)
        self.log = log or (lambda msg: print("[word-assist] " + msg, file=sys.stderr, flush=True))
        self.stats = collections.Counter()

    def observe(self, now: float) -> "dict | None":
        cur = self.read_fn()
        if cur:
            self.policy.observe(str(cur.get("utterance_id")), str(cur.get("text") or ""), now,
                                cur.get("tail_silence_s"))
        return cur

    def step(self, now: "float | None" = None) -> "str | None":
        """Returns the spoken offer, or None. Blocking (LLM) - call from an executor."""
        now = time.time() if now is None else now
        cur = self.observe(now)
        if not cur:
            return None
        cue = self.policy.due(now)
        if cue is None:
            return None
        self.stats["cues"] += 1
        ok, why = self.gate_fn()
        if not ok:
            self.stats["gated"] += 1
            self.log(f"cue {cue['cue']!r} but gated: {why}")
            return None
        partial = self.policy.text
        word, dt = self.suggest_fn(partial, self.context_fn())
        if not word:
            self.stats["no_guess"] += 1
            self.log(f"cue {cue['cue']!r}: no confident guess ({dt * 1000:.0f} ms)")
            return None
        fresh = self.read_fn()
        if not fresh or str(fresh.get("utterance_id")) != self.policy.utterance_id:
            self.stats["ended"] += 1        # his turn ended meanwhile: cognition owns the reply now
            self.log(f"guess {word!r} dropped: his turn already ended")
            return None
        if not self.policy.still_relevant(str(fresh.get("text") or ""), cue, word, fresh.get("tail_silence_s")):
            self.stats["stale"] += 1
            self.log(f"guess {word!r} dropped: he found it / moved on")
            return None
        ok, why = self.gate_fn()
        if not ok:
            self.stats["gated"] += 1
            return None
        line = _ws.offer_line(word)
        self.speak_fn(line)
        self.offers.add("word offer", line, partial)
        self.stats["offered"] += 1
        self.log(f"offered {line!r} ({dt * 1000:.0f} ms LLM) for {cue['cue']!r}")
        return line


def transcript_context(n: int = 6) -> "list[str]":
    """The last n lines of the shared transcript as 'Zeke: ...' / 'Iris: ...'."""
    try:
        lines = (_REPO / "state" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            d = json.loads(ln)
        except Exception:
            continue
        out.append(("Iris" if d.get("role") == "assistant" else "Zeke") + ": " + str(d.get("content") or "")[:200])
    return out
