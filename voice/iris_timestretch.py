"""iris_timestretch.py — pitch-preserving slow-down for the Fish mouth (slow mode, 2026-10-31).

Fish S2 Pro has no speed knob ("[speaking slowly and clearly]" did nothing by Zeke's ear), so
slow mode stretches the decoded audio instead. WSOLA (waveform-similarity overlap-add,
Verhelst & Roelands 1993): the output is built from overlapping windows of the input, each one
taken from near its ideal position at the place where it lines up best with the natural
continuation of the previous window — so pitch and timbre are untouched and there is none of the
"phasey"/metallic smear of a phase vocoder (librosa.effects.time_stretch). numpy only, so it runs
in any venv and in tests; ~25-40 ms of CPU per second of 44.1 kHz audio (measured, see
tests/test_timestretch.py and the 10-31 report).

    stretch(audio, speed) -> audio      speed 0.8 = 80 % speed (1.25x longer); 1.0 = untouched

Also here: the per-utterance / global speed resolution used by the mouth
(leading "[slow]" / "[slower]" / "[slow 0.8]" tag, JSON "speed", voice_control.json
"speech_speed") and the spelled-letters normaliser ("C, A, T" keeps a pause per letter).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

SPEED_MIN = 0.5
SPEED_MAX = 1.5
SLOW_SPEED = float(os.environ.get("FISH_SLOW_SPEED", "0.85"))
SLOWER_SPEED = float(os.environ.get("FISH_SLOWER_SPEED", "0.72"))


def clamp_speed(v, default: float = 1.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f) or f <= 0:
        return default
    return float(min(SPEED_MAX, max(SPEED_MIN, f)))


def stretch(x: np.ndarray, speed: float, sr: int = 44100, win_s: float = 0.030,
            tol_s: float = 0.012) -> np.ndarray:
    """WSOLA time-stretch of mono float audio. speed < 1 = slower/longer, pitch unchanged.
    The first and last samples map to the first and last samples (pieces streamed one after
    another still join where they joined before)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    speed = clamp_speed(speed)
    if abs(speed - 1.0) < 1e-3 or x.size == 0:
        return x
    n = max(64, int(round(win_s * sr)) // 2 * 2)       # window (even)
    hs = n // 2                                         # synthesis hop (50 % overlap, hann sums to 1)
    ha = hs * speed                                     # analysis hop
    tol = max(8, int(round(tol_s * sr)))
    out_len = int(round(x.size / speed))
    if x.size < 2 * n or out_len < 2 * n:               # too short to window (a held crossfade tail):
        return x                                        # leave it — padding would put a gap mid-word
    win = np.hanning(n + 2)[1:-1].astype(np.float32)
    # pad so every candidate window exists
    pad = tol + n
    xp = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad + n, np.float32)])
    frames = int(np.ceil(out_len / hs)) + 1
    y = np.zeros(frames * hs + n, np.float32)
    wsum = np.zeros_like(y)
    # coarse search on a decimated signal, then refine at full rate (speed)
    dec = 4
    prev = 0                                            # chosen analysis start (input coords) of the last frame
    for k in range(frames):
        ideal = int(round(k * ha))
        if k == 0:
            pos = 0
        else:
            nat = prev + hs                             # natural continuation of the previous window
            tmpl = xp[pad + nat: pad + nat + n]
            lo = ideal - tol
            region = xp[pad + lo: pad + lo + n + 2 * tol]
            c = np.correlate(region[::dec], tmpl[::dec], mode="valid")
            best = int(np.argmax(c)) * dec
            a, b = max(0, best - dec), min(2 * tol, best + dec)
            fine = np.correlate(region[a: b + n], tmpl, mode="valid")
            pos = lo + a + int(np.argmax(fine))
        seg = xp[pad + pos: pad + pos + n]
        y[k * hs: k * hs + n] += seg * win
        wsum[k * hs: k * hs + n] += win
        prev = pos
    nz = wsum > 1e-3
    y[nz] /= wsum[nz]
    y = y[:out_len]
    # pin the edges: a short crossfade into/out of the untouched input ends
    e = min(n // 2, out_len // 4, x.size // 4)
    if e > 8:
        r = np.linspace(0.0, 1.0, e, dtype=np.float32)
        y[:e] = x[:e] * (1 - r) + y[:e] * r
        y[-e:] = y[-e:] * (1 - r) + x[-e:] * r
    return y.astype(np.float32)


# ── speed resolution ──────────────────────────────────────────────────────────────
_SLOW_TAG_RE = re.compile(r"^\s*\[(slow|slower|slowly)(?:[\s,:=]+(\d*\.?\d+))?\]\s*", re.I)


def pop_speed_tag(text: str):
    """Leading "[slow]" / "[slower]" / "[slow 0.8]" -> (text without it, speed | None).
    Only a LEADING tag counts, and it is removed so Fish never reads it."""
    m = _SLOW_TAG_RE.match(text or "")
    if not m:
        return text, None
    if m.group(2):
        sp = clamp_speed(m.group(2), SLOW_SPEED)
    else:
        sp = SLOWER_SPEED if m.group(1).lower() == "slower" else SLOW_SPEED
    return text[m.end():], sp


def global_speed(control_file) -> float:
    """voice_control.json "speech_speed" (hot, read per utterance). Missing/bad = 1.0."""
    try:
        with open(control_file, encoding="utf-8") as f:
            return clamp_speed(json.load(f).get("speech_speed", 1.0))
    except Exception:
        return 1.0


def resolve_speed(text: str, obj_speed=None, control_file=None):
    """-> (text, speed). Precedence: JSON speed > leading tag > global voice_control > 1.0.
    The tag is always stripped."""
    text, tag_speed = pop_speed_tag(text)
    if obj_speed is not None:
        sp = clamp_speed(obj_speed)
    elif tag_speed is not None:
        sp = tag_speed
    elif control_file is not None:
        sp = global_speed(control_file)
    else:
        sp = 1.0
    return text, sp


# ── spelled letters ───────────────────────────────────────────────────────────────
# "C, A, T" / "C-A-T" / "C A T" / "C. A. T." (3+ single letters/digits) -> "C. A. T." with each
# letter its own short sentence, which Fish reads letter by letter with a pause between (a
# comma list ran them together; "A" alone was read as the article). Returns the text unchanged
# when there is no run. 2-letter runs ("I, a") are left alone — too often real words.
_SPELL_RE = re.compile(r"(?<![\w'])([A-Za-z0-9])(?:(?:\s*[,.\-]\s*|\s+)([A-Za-z0-9])(?![\w'])){2,}\.?")
_SPELL_SEP = re.compile(r"\s*[,.\-]\s*|\s+")
SPELL_STYLE = os.environ.get("FISH_SPELL_STYLE", "period").strip().lower()   # period|comma|off


def _spell_join(letters: list, end: str) -> str:
    return (", " if SPELL_STYLE == "comma" else ". ").join(letters) + end


def normalize_spelling(text: str) -> str:
    if SPELL_STYLE == "off" or not text:
        return text

    def _sub(m):
        span = m.group(0)
        letters = [t for t in _SPELL_SEP.split(span.strip(" .")) if t]
        if len(letters) < 3 or any(len(t) != 1 for t in letters):
            return span
        if re.search(r"\d\.\d", span):                 # "2.1.3" is a version, not spelling
            return span
        # plain lowercase words separated by spaces ("a b c" is rare; "i a m" never) — only
        # spaced runs that are UPPERCASE count, any punctuation-separated run counts
        if re.fullmatch(r"[a-z0-9](\s+[a-z0-9])+", span.strip()):
            return span
        nxt = m.string[m.end():m.end() + 1]
        end = "" if (nxt and nxt in ",.;:!?") else "."     # "C, A, T, spells" -> "C. A. T, spells"
        return _spell_join([t.upper() for t in letters], end)

    return _SPELL_RE.sub(_sub, text)


def is_spelled_unit(word: str) -> bool:
    """A planner unit that is one letter of a normalized spelled run ("C." / "A.")."""
    return bool(re.fullmatch(r"[A-Z0-9][.,]", word or ""))
