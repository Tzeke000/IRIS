"""FOLLOW DISTANCE IS AN INPUT (DK-61, Zeke 11-09: "follow <name> at 1 m / 5 m / 3.2 m away", any value).
vr_temple action=follow takes dist= or distance_m= (number or "3.2 m"), clamps to 1–30 m and SAYS so, refuses junk
before anything reaches my body, and always sends the body `dist` (old and new bodies both read it)."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import tools.tool_registry  # noqa: F401
except Exception:  # noqa: BLE001
    reg = types.ModuleType("tools.tool_registry")
    reg.register_tool = lambda *a, **k: (lambda f: f)
    pkg = sys.modules.get("tools") or types.ModuleType("tools")
    pkg.__path__ = [str(ROOT / "tools")]
    sys.modules["tools"] = pkg
    sys.modules["tools.tool_registry"] = reg
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("vr_temple_tool_fd", str(ROOT / "tools" / "system" / "vr_temple_tool.py"))
vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(vt, "_send", lambda req, timeout=4.0: calls.append(req) or {"ok": True, "dist_m": req.get("dist")})
    return calls


@pytest.mark.parametrize("key", ["dist", "distance_m"])
@pytest.mark.parametrize("val,want", [(1, 1.0), (3.2, 3.2), (5, 5.0), ("3.2", 3.2), ("3.2 m", 3.2), ("5m", 5.0),
                                      ("1 metre", 1.0), (" 12.5 meters ", 12.5)])
def test_any_gap_in_range_is_sent_as_dist(sent, key, val, want):
    r = vt._vr_temple({"action": "follow", "name": "Zeke", key: val}, {})
    assert r["ok"] is True and "clamped" not in r
    assert sent == [{"cmd": "follow", "name": "Zeke", "dist": want}]


@pytest.mark.parametrize("val,want,why", [(0.3, 1.0, "in their face"), (0, 1.0, "in their face"), (-2, 1.0, "in their face"),
                                          (99, 30.0, "cap"), ("100 m", 30.0, "cap")])
def test_out_of_range_is_clamped_and_said(sent, val, want, why):
    r = vt._vr_temple({"action": "follow", "name": "Zeke", "distance_m": val}, {})
    assert r["ok"] is True and sent[0]["dist"] == want
    assert "clamped" in r and why in r["clamped"] and f"{want:g} m" in r["clamped"]


@pytest.mark.parametrize("val", ["close", "", "3.2 km", "1e", None, True, [3], {"m": 3}, float("nan"), float("inf"), "nan"])
def test_junk_is_refused_and_nothing_is_sent(sent, val):
    r = vt._vr_temple({"action": "follow", "name": "Zeke", "dist": val}, {})
    assert r["ok"] is False and "metres" in r["error"]
    assert sent == []


def test_both_keys_disagreeing_is_refused(sent):
    r = vt._vr_temple({"action": "follow", "name": "Zeke", "dist": 2, "distance_m": 5}, {})
    assert r["ok"] is False and sent == []


def test_both_keys_agreeing_is_fine(sent):
    r = vt._vr_temple({"action": "follow", "name": "Zeke", "dist": 2, "distance_m": "2 m"}, {})
    assert r["ok"] is True and sent == [{"cmd": "follow", "name": "Zeke", "dist": 2.0}]


def test_default_unchanged_no_dist_sent(sent):
    """no distance given ⇒ nothing added: the body's own default (3 m) applies, as before"""
    vt._vr_temple({"action": "follow", "name": "Zeke"}, {})
    vt._vr_temple({"action": "follow", "on": False}, {})
    vt._vr_temple({"action": "follow"}, {})
    assert sent == [{"cmd": "follow", "name": "Zeke"}, {"cmd": "follow", "on": False}, {"cmd": "follow"}]


def test_range_matches_the_body():
    """the tool's clamp = the body's (iris_follow.gd DIST_MIN_M / DIST_MAX_M) when that body is on disk"""
    import os
    import re
    g = Path(os.environ.get("VR_PROJECT", "/data/projects/vr_iris/project")) / "scripts" / "iris_follow.gd"
    if not g.exists() or "DIST_MIN_M" not in g.read_text():
        pytest.skip("a game project with follow-1110's DIST_MIN_M isn't on disk here")
    src = g.read_text()
    lo = float(re.search(r"const DIST_MIN_M := ([\d.]+)", src).group(1))
    hi = float(re.search(r"const DIST_MAX_M := ([\d.]+)", src).group(1))
    assert (lo, hi) == (vt.FOLLOW_DIST_MIN_M, vt.FOLLOW_DIST_MAX_M)
