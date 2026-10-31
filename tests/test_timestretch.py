"""Slow mode (WSOLA time-stretch) + spelled-letter normaliser for the Fish mouth (2026-10-31)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("iris_timestretch", ROOT / "voice" / "iris_timestretch.py")
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)

SR = 44100


def _tone(f0=200.0, dur=1.0, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    # a voice-like harmonic tone (fundamental + 3 harmonics)
    return (0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)).astype(np.float32)


def _peak_hz(a, sr=SR):
    spec = np.abs(np.fft.rfft(a * np.hanning(a.size)))
    return np.fft.rfftfreq(a.size, 1 / sr)[int(np.argmax(spec))]


def test_speed_one_is_identity():
    x = _tone()
    assert ts.stretch(x, 1.0) is x or np.array_equal(ts.stretch(x, 1.0), x)


@pytest.mark.parametrize("speed", [0.72, 0.85, 1.2])
def test_length_and_pitch_preserved(speed):
    x = _tone(200.0, 1.5)
    y = ts.stretch(x, speed)
    assert abs(y.size - x.size / speed) <= 1
    assert abs(_peak_hz(y[SR // 10: -SR // 10]) - 200.0) < 3.0          # pitch NOT shifted
    assert y.dtype == np.float32 and np.all(np.isfinite(y))


def test_no_clicks_on_a_clean_tone():
    x = _tone(220.0, 1.0)
    y = ts.stretch(x, 0.8)
    # a WSOLA splice that misaligns phase shows up as a sample jump far above the tone's own slope
    assert np.max(np.abs(np.diff(y))) < 1.5 * np.max(np.abs(np.diff(x)))
    # energy stays flat (no dips at window joins)
    rms = [np.sqrt(np.mean(y[i:i + 2205] ** 2)) for i in range(2205, y.size - 4410, 2205)]
    assert max(rms) / min(rms) < 1.15


def test_edges_pinned_for_streamed_pieces():
    x = _tone(180.0, 0.6)
    y = ts.stretch(x, 0.8)
    assert abs(y[0] - x[0]) < 1e-6 and abs(y[-1] - x[-1]) < 1e-6


def test_tiny_piece_left_alone():
    x = _tone(200.0, 0.02)                 # a held 1024-sample crossfade tail
    assert np.array_equal(ts.stretch(x, 0.8), x)


def test_fast_enough_for_streaming():
    x = (np.random.default_rng(0).standard_normal(SR * 3) * 0.1).astype(np.float32)
    t = time.perf_counter()
    ts.stretch(x, 0.8)
    per_s = (time.perf_counter() - t) / 3
    assert per_s < 0.15, per_s             # measured ~5-20 ms per audio second on the server


def test_speed_tag_and_precedence(tmp_path):
    assert ts.pop_speed_tag("[slow] Hello there") == ("Hello there", ts.SLOW_SPEED)
    assert ts.pop_speed_tag("[slower] Hi")[1] == ts.SLOWER_SPEED
    assert ts.pop_speed_tag("[slow 0.8] Hi") == ("Hi", 0.8)
    assert ts.pop_speed_tag("[slow, 0.1] Hi")[1] == ts.SPEED_MIN       # clamped
    assert ts.pop_speed_tag("Hi [slow] there") == ("Hi [slow] there", None)   # leading only
    assert ts.pop_speed_tag("[speaking slowly] Hi") == ("[speaking slowly] Hi", None)
    ctl = tmp_path / "voice_control.json"
    assert ts.resolve_speed("Hi", None, ctl) == ("Hi", 1.0)              # missing file = off
    ctl.write_text(json.dumps({"speech_speed": 0.9}))
    assert ts.resolve_speed("Hi", None, ctl) == ("Hi", 0.9)              # global
    assert ts.resolve_speed("[slow] Hi", None, ctl) == ("Hi", ts.SLOW_SPEED)   # tag beats global
    assert ts.resolve_speed("[slow] Hi", 0.95, ctl) == ("Hi", 0.95)      # JSON beats tag (tag still stripped)
    ctl.write_text(json.dumps({"speech_speed": "fast"}))
    assert ts.resolve_speed("Hi", None, ctl)[1] == 1.0                   # bad value = off
    ctl.write_text("{not json")
    assert ts.resolve_speed("Hi", None, ctl)[1] == 1.0


@pytest.mark.parametrize("src,want", [
    ("It's spelled C, A, T.", "It's spelled C. A. T."),
    ("C-A-T, like the animal", "C. A. T, like the animal"),
    ("say U S A now", "say U. S. A. now"),
    ("the code is 4, 2, 4, 2", "the code is 4. 2. 4. 2."),
    ("c, a, t", "C. A. T."),
])
def test_spelling_normalised(src, want):
    assert ts.normalize_spelling(src) == want


@pytest.mark.parametrize("src", [
    "version 2.1.3 is out", "I, a person", "a b c", "Plan A or B", "e.g. this", "I am OK",
    "Hello there, how are you?", "[curious] Wait, really?",
])
def test_spelling_leaves_normal_text_alone(src):
    assert ts.normalize_spelling(src) == src


FISH_PY = Path.home() / "voice_lab" / "fish" / "venv" / "bin" / "python"


@pytest.mark.skipif(not FISH_PY.exists(), reason="Fish venv not on this machine")
def test_mouth_planner_keeps_spelled_run_together():
    """Import the real mouth module (no model load) and check the spelled run reaches Fish as one
    sentence: no auto tag between the letters, and the slow tag never reaches Fish."""
    code = r"""
import sys, json; sys.path.insert(0, 'voice'); import wren_fish_server as m
u = m._units(m._pronounce('Spell it C, A, T, like the animal. Then U-S-A.'))
m._auto_tag_sentences(u, None)
txt, sp = m._TS.resolve_speed('[slow] Okay.', None, None)
print(json.dumps({"units": [x.text() for x in u], "tags": sum('[' in x.text() for x in u), "slow": [txt, sp]}))
"""
    env = dict(os.environ, IRIS_MOUTH_DRYRUN="1")
    r = subprocess.run([str(FISH_PY), "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 and "Fish import failed" in r.stdout + r.stderr:
        pytest.skip("Fish source not importable here")
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["tags"] == 2, out          # one auto tag per real sentence, none between C/A/T
    assert out["units"][2:5] == ["C.", "A.", "T,"], out
    assert out["slow"] == ["Okay.", ts.SLOW_SPEED]
