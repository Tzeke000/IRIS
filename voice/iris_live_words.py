"""Live words (full-duplex chunk 3, 2026-10-29): what he is saying RIGHT NOW, while he says it.

One LiveWords object = one utterance's rolling, DISPLAY-ONLY transcript. The daemon feeds it
audio as it is captured; a throttled background pass re-transcribes the growing buffer with the
ALREADY-LOADED whisper (injected transcribe_fn, never a new model) and publishes

    {"utterance_id", "source", "text", "final", "ts", "started_ts", "seq", "mouth_busy", "dropped"}

to state/voice/live_partial.json (atomic: tempfile + os.replace) for the app / VR to show.

DISPLAY-ONLY CONTRACT: nothing here decides a turn. The final text that becomes a turn still
comes from cmd_listen / cmd_bargein_watch with all their gates (phantom, self-voice drop,
voice-id). A dropped utterance publishes final=True with text "" + the drop marker so the
display clears instead of showing words that never became a turn.

Pure python + numpy, no audio I/O, so it is unit-testable without the daemon.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
LIVE_FILE = Path(os.environ.get("IRIS_LIVE_PARTIAL_FILE")
                 or (_REPO / "state" / "voice" / "live_partial.json"))
MIN_CADENCE_S = 0.5            # hard floor between partial decodes (GPU/CPU cost bound)

_SEQ = itertools.count(1)
_LAST: dict = {}               # last payload published by this process (daemon `live_partial` cmd)
_LAST_LOCK = threading.Lock()


def write_atomic(path, payload: dict) -> bool:
    """Write payload as JSON via a same-dir tempfile + os.replace (readers never see half a file)."""
    path = Path(path)
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except Exception as e:  # display-only: never break the ears over a status file
        print(f"[live_words] write failed (non-fatal): {e!r}", file=sys.stderr)
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def last() -> dict:
    with _LAST_LOCK:
        return dict(_LAST)


def read_current(path=None, max_age_s: float = 3.0, now=None) -> "dict | None":
    """For OTHER processes (host / cognition, 10-31): what he is saying right now, from the file.

    Returns the payload ({"text", "source", "utterance_id", "final", "ts", ...}) only while an
    utterance is IN PROGRESS - not final and published within max_age_s - else None. Same
    display-only contract: a partial is a guess that later passes rewrite; never act on it as a
    turn. In-process callers (the daemon) use last() / the daemon's `live_partial` command."""
    try:
        d = json.loads(Path(path or LIVE_FILE).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(d, dict) or d.get("final") or not d.get("text"):
        return None
    now = time.time() if now is None else now
    try:
        if now - float(d.get("ts") or 0) > max_age_s:
            return None
    except (TypeError, ValueError):
        return None
    return d


class LiveWords:
    def __init__(self, source: str, *, path=None, transcribe_fn=None,
                 cadence_s: float = 0.6, min_audio_s: float = 0.5, sample_rate: int = 16000,
                 lock=None, mouth_busy_fn=None, enabled: bool = True) -> None:
        self.source = source
        self.started_ts = time.time()
        self.utterance_id = f"{source}-{int(self.started_ts * 1000)}-{next(_SEQ)}"
        self.path = Path(path) if path else LIVE_FILE
        self.enabled = bool(enabled)
        self._transcribe = transcribe_fn
        self.cadence_s = max(float(cadence_s), MIN_CADENCE_S)
        self._min_samples = int(max(0.0, min_audio_s) * sample_rate)
        self._wlock = lock                  # e.g. the daemon's _WHISPER_LOCK (model isn't reentrant)
        self._mouth_busy = mouth_busy_fn
        self._buf: list = []
        self._buf_lock = threading.Lock()
        self._pub_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._seq = 0
        self._finished = False
        self.published = False              # any payload written for this utterance
        self.passes = 0
        self.last_text = ""
        self.final_text = None              # set by the caller ONLY once the turn passed its gates

    # -- publishing ---------------------------------------------------------------------------
    def publish(self, text: str, final: bool = False, dropped=None) -> bool:
        if not self.enabled:
            return False
        with self._pub_lock:
            if self._finished:              # a late partial must never overwrite the final
                return False
            if final:
                self._finished = True
            self._seq += 1
            try:
                busy = bool(self._mouth_busy()) if self._mouth_busy else False
            except Exception:
                busy = False
            payload = {"utterance_id": self.utterance_id, "source": self.source,
                       "text": (text or "").strip(), "final": bool(final),
                       "ts": round(time.time(), 3), "started_ts": round(self.started_ts, 3),
                       "seq": self._seq, "mouth_busy": busy,
                       "dropped": (str(dropped)[:200] if dropped else None)}
            ok = write_atomic(self.path, payload)
            if ok:
                self.published = True
                if not final:
                    self.last_text = payload["text"]
                with _LAST_LOCK:
                    _LAST.clear()
                    _LAST.update(payload)
            return ok

    # -- rolling partials ---------------------------------------------------------------------
    def feed(self, frame) -> None:
        """Append captured audio (float32 16 kHz). Starts the partial worker on first audio."""
        if not self.enabled or self._transcribe is None or self._finished:
            return
        with self._buf_lock:
            self._buf.append(frame)
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, daemon=True,
                                                name=f"live-words-{self.source}")
                self._thread.start()

    def _loop(self) -> None:
        import numpy as np
        last_n = 0
        while not self._stop.wait(self.cadence_s):
            with self._buf_lock:
                snap = self._buf[:]
            n = sum(len(x) for x in snap)
            if n < self._min_samples or n <= last_n:
                continue                    # too little / nothing new since the last pass
            last_n = n
            try:
                buf = np.concatenate(snap)
                if self._wlock is not None:
                    with self._wlock:
                        text = self._transcribe(buf)
                else:
                    text = self._transcribe(buf)
                self.passes += 1
                text = (text or "").strip()
                if text and text != self.last_text and not self._stop.is_set():
                    self.publish(text)
            except Exception as e:          # best-effort: a failed partial never touches capture
                print(f"[live_words] partial pass failed (non-fatal): {e!r}", file=sys.stderr)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker (call BEFORE the final transcribe so the two never overlap)."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)

    def finish(self, dropped=None) -> None:
        """Close the utterance: final text if the turn passed its gates, else clear the display
        (only if something was shown). Silent windows (nothing published, no text) write nothing."""
        self.stop()
        if self.final_text:
            self.publish(self.final_text, final=True)
        elif self.published:
            self.publish("", final=True, dropped=dropped or "no turn")
