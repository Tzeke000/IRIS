"""VR barge tiers (IRIS_VR_TIERS): a backchannel / blip clip doesn't hush me; real talk still does."""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "tools.tool_registry" not in sys.modules:          # the tool module only needs register_tool
    reg = types.ModuleType("tools.tool_registry")
    reg.register_tool = lambda *a, **k: (lambda f: f)
    sys.modules.setdefault("tools", types.ModuleType("tools"))
    sys.modules["tools.tool_registry"] = reg
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("vr_temple_tool_t", str(ROOT / "tools" / "system" / "vr_temple_tool.py"))
vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)


def clip(dur, t0=100.0):
    return {"t_start": t0, "t_end": t0 + dur, "dist_m": 2.0, "eff_db": -20}


def test_backchannel_clip_keeps_me_talking():
    assert not vt._vr_tier_says_stop("zeke", clip(0.5), {"text": "yeah", "ts": 100.6})


def test_short_blip_without_words_keeps_me_talking():
    assert not vt._vr_tier_says_stop("zeke", clip(0.4), {})


def test_real_talk_stops_me():
    assert vt._vr_tier_says_stop("zeke", clip(1.0), {})
    assert vt._vr_tier_says_stop("guest1", clip(0.5), {"text": "wait what is that", "ts": 100.5})


def test_explicit_stop_even_short():
    assert vt._vr_tier_says_stop("zeke", clip(0.3), {"text": "stop", "ts": 100.4})


def test_stale_partial_from_an_older_phrase_is_ignored():
    # 'yeah' belonged to a phrase 10 s earlier: this 1.2 s clip is real speech
    assert vt._vr_tier_says_stop("zeke", clip(1.2), {"text": "yeah", "ts": 90.0})


def test_error_fails_toward_stopping(monkeypatch):
    monkeypatch.setattr(vt, "_tiers", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert vt._vr_tier_says_stop("zeke", clip(0.3), {})
