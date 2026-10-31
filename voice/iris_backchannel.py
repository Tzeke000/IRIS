"""voice/iris_backchannel.py - my "mm-hm / yeah / right" while Zeke is mid-thought (10-31).

Zeke, live duplex test 10-31: *"you should be able to say things like mm-hmm, yeah, right, you
know, like when you're following along."* A listener's backchannel lands in the short pauses INSIDE
a long turn - not at its end (that is my reply's job). So, during a normal listen capture:

  * he has been talking for a while   (>= MIN_SPEECH_S of voiced speech, default 4 s),
  * he has just paused, briefly       (silence in [MIN_PAUSE_S, end_silence - END_MARGIN_S]),
  * smart-turn says he is NOT done    (prob < INCOMPLETE_BAR, the same 0.6 hold-open bar),
  * rate limits hold                  (>= MIN_GAP_S since the last one, and >= MIN_NEW_SPEECH_S
                                       of new speech since it - never two in a row),
  * and I am allowed to make a sound  (not quiet hours, not mouth_muted, mouth not busy),

then the mouth plays ONE pre-synthesised clip in my own voice (wren_fish_server POST /backchannel).
Never through the LLM - the pause is gone long before a reply could be written.

OFF unless the ears run with IRIS_BACKCHANNEL=1. Hot kill switch without a restart:
scratch/voice_control.json {"backchannel": false}. All tunables are hot in the same file under
"backchannel_cfg" (keys = the lower-case names in DEFAULTS).

Self-voice: the clip's own audio must never become his words. While a clip plays (+ a short tail)
the capture loop treats mic frames below RESUME_PEAK as silence (zeroed, and they don't reset the
end-of-turn clock); a frame at/above it is him talking again and is kept. On the headset there is
no acoustic path at all, so this is belt-and-braces.

The policy is pure (no I/O) so it is unit-tested without a mic; Gate/Player do the I/O.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import threading
import time
import urllib.request as _req
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "min_speech_s": 4.0,       # voiced seconds of his speech before the first backchannel
    "min_pause_s": 0.45,       # a pause shorter than this is just breath between words
    "end_margin_s": 0.3,       # stay this far below end-of-turn silence (never race my reply)
    "min_gap_s": 6.0,          # at most one backchannel per this many seconds
    "min_new_speech_s": 1.5,   # voiced seconds of NEW speech required since the last one
    "incomplete_bar": 0.6,     # smart-turn prob below this = mid-thought (same as hold-open)
    "resume_peak": 0.05,       # during a clip, a mic frame this loud is him, not my clip
    "self_tail_s": 0.2,        # self-voice gate stays up this long after the clip ends
    "max_clip_s": 1.2,         # assumed clip length until the mouth reports the real one
}


def env_enabled() -> bool:
    return os.environ.get("IRIS_BACKCHANNEL", "0").strip().lower() in ("1", "true", "on", "yes")


def _read_control() -> dict:
    try:
        from wren_voice_status import read_control
        return read_control() or {}
    except Exception:
        return {}


def config(ctl: "dict | None" = None) -> dict:
    """DEFAULTS overlaid with voice_control.json backchannel_cfg (bad values ignored)."""
    ctl = _read_control() if ctl is None else ctl
    out = dict(DEFAULTS)
    over = ctl.get("backchannel_cfg") if isinstance(ctl, dict) else None
    if isinstance(over, dict):
        for k, v in over.items():
            if k in out:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


# ── Policy (pure) ────────────────────────────────────────────────────────────────
class BackchannelPolicy:
    """Per-utterance state. Feed every frame with on_frame(); ask want_check() during silence;
    if smart-turn agrees, call fired(). One smart-turn check per pause, at most."""

    def __init__(self, cfg: "dict | None" = None):
        self.cfg = dict(DEFAULTS if cfg is None else cfg)
        self.onset_t = None
        self.voiced_s = 0.0             # total voiced seconds this utterance
        self.voiced_at_fire = 0.0       # voiced_s when the last backchannel fired
        self.last_fire_t = None
        self.checked_pause = None       # last_voice_t of the pause already checked
        self.fires = 0

    def on_frame(self, now: float, voiced: bool, frame_s: float) -> None:
        if self.onset_t is None:
            self.onset_t = now
        if voiced:
            self.voiced_s += frame_s

    def want_check(self, now: float, silence: float, last_voice_t: float,
                   end_silence_s: float) -> bool:
        c = self.cfg
        if self.onset_t is None:
            return False
        if self.checked_pause == last_voice_t:
            return False                             # this pause was already decided
        if self.voiced_s < c["min_speech_s"]:
            return False                             # voiced time, not wall time: pauses don't count
        if not (c["min_pause_s"] <= silence <= end_silence_s - c["end_margin_s"]):
            return False
        if self.last_fire_t is not None:
            if now - self.last_fire_t < c["min_gap_s"]:
                return False
            if self.voiced_s - self.voiced_at_fire < c["min_new_speech_s"]:
                return False                         # never two in a row without new speech
        self.checked_pause = last_voice_t
        return True

    def fired(self, now: float) -> None:
        self.last_fire_t = now
        self.voiced_at_fire = self.voiced_s
        self.fires += 1


def in_quiet_hours(now: "_dt.datetime | None" = None) -> bool:
    """22:30-05:10 local (scripts quiet-hours window; the flag file is checked separately)."""
    now = now or _dt.datetime.now()
    m = now.hour * 60 + now.minute
    return m >= 22 * 60 + 30 or m < 5 * 60 + 10


def _voice_flag_off(path: "Path | None" = None) -> bool:
    p = path or (_REPO / "state" / "voice_deliberately_off.json")
    try:
        if not p.is_file():
            return False
        return bool(json.loads(p.read_text(encoding="utf-8")).get("off"))
    except Exception:
        return False


def allowed(ctl: "dict | None" = None, now: "_dt.datetime | None" = None,
            flag_path: "Path | None" = None) -> "tuple[bool, str]":
    """May I make a sound at all right now? (ok, reason-if-not)."""
    if not env_enabled():
        return False, "IRIS_BACKCHANNEL off"
    ctl = _read_control() if ctl is None else ctl
    if not bool(ctl.get("backchannel", True)):
        return False, "voice_control backchannel=false"
    if bool(ctl.get("mouth_muted", False)):
        return False, "mouth_muted"
    if in_quiet_hours(now) or _voice_flag_off(flag_path):
        return False, "quiet hours"
    return True, ""


# ── Player (I/O) ─────────────────────────────────────────────────────────────────
class Player:
    """Fires POST <mouth>/backchannel on a thread (the capture loop never blocks on the mouth) and
    exposes the self-voice window. A mouth without the endpoint (404/501, e.g. the StyleTTS2
    fallback) disables the lane for this process, logged once."""

    _disabled_reason = None     # process-wide: one dead mouth endpoint = stop asking

    def __init__(self, server_url: str, cfg: "dict | None" = None, opener=None):
        self.url = server_url.rstrip("/") + "/backchannel"
        self.cfg = dict(DEFAULTS if cfg is None else cfg)
        self._open = opener or (lambda req, timeout: _req.urlopen(req, timeout=timeout))
        self.busy = False
        self.win_start = 0.0
        self.win_end = 0.0
        self.last_info = None

    @classmethod
    def available(cls) -> bool:
        return cls._disabled_reason is None

    def in_self_window(self, now: float) -> bool:
        return self.win_start <= now <= self.win_end + self.cfg["self_tail_s"]

    def fire(self, now: "float | None" = None) -> bool:
        if self.busy or not self.available():
            return False
        now = time.time() if now is None else now
        self.busy = True
        self.win_start = now
        self.win_end = now + self.cfg["max_clip_s"]
        threading.Thread(target=self._post, daemon=True, name="backchannel").start()
        return True

    def _post(self) -> None:
        try:
            req = _req.Request(self.url, data=b"{}", method="POST",
                               headers={"Content-Type": "application/json"})
            with self._open(req, 5.0) as r:
                info = json.loads(r.read().decode("utf-8", "replace") or "{}")
            self.last_info = info
            print(f"[backchannel] played {info.get('text')!r} ({info.get('dur_s')}s)", file=sys.stderr)
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (404, 501):
                Player._disabled_reason = f"mouth has no /backchannel (HTTP {code})"
                print(f"[backchannel] disabled: {Player._disabled_reason}", file=sys.stderr)
            else:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:80]   # type: ignore[attr-defined]
                except Exception:
                    pass
                print(f"[backchannel] not played ({code or repr(e)[:60]} {body})", file=sys.stderr)
        finally:
            self.win_end = time.time()   # the clip is over (played, skipped or failed)
            self.busy = False


def prime(server_url: str, opener=None) -> None:
    """Ask the mouth to render the clip set now (call_start), so the first one isn't skipped."""
    if not env_enabled():
        return
    try:
        op = opener or (lambda req, timeout: _req.urlopen(req, timeout=timeout))
        with op(_req.Request(server_url.rstrip("/") + "/backchannel/prime"), 5.0) as r:
            r.read()
    except Exception as e:
        print(f"[backchannel] prime request failed (non-fatal): {e!r}", file=sys.stderr)
