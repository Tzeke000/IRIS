"""vr_temple REJECTS UNKNOWN PARAMS (11-05, docket I9): `look yaw=…` returned ok:true and did nothing (look takes
at=[x,y,z]). Every action must refuse a key it doesn't read, name it, and list the accepted ones — before anything is
sent to my body. The table is cross-checked against the game's own command handler when the project is on disk."""
import os
import re
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

spec = importlib.util.spec_from_file_location("vr_temple_tool_p", str(ROOT / "tools" / "system" / "vr_temple_tool.py"))
vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(vt, "_send", lambda req, timeout=4.0: calls.append(req) or {"ok": True})
    return calls


def test_look_yaw_is_refused_and_nothing_is_sent(sent):
    r = vt._vr_temple({"action": "look", "yaw": 90}, {})
    assert r["ok"] is False
    assert "'yaw'" in r["error"] and "look" in r["error"] and "at" in r["error"]
    assert r["unknown_params"] == ["yaw"] and r["accepted"] == ["at"]
    assert "at=[x,y,z]" in r.get("hint", "")
    assert sent == []                                  # the body never got a command it would silently ignore


def test_look_at_still_works(sent):
    r = vt._vr_temple({"action": "look", "at": [1, 2, 3]}, {})
    assert r == {"ok": True} and sent == [{"cmd": "look", "at": [1, 2, 3]}]


def test_every_bad_key_is_named(sent):
    r = vt._vr_temple({"action": "move", "to": [0, 1, 0], "sped": 3, "fast": True}, {})
    assert r["ok"] is False and r["unknown_params"] == ["fast", "sped"]
    assert "speed" in r["accepted"] and sent == []


def test_actions_without_params_refuse_any(sent):
    r = vt._vr_temple({"action": "state", "verbose": True}, {})
    assert r["ok"] is False and "(none)" in r["error"] and sent == []


def test_follow_who_hints_name(sent):
    r = vt._vr_temple({"action": "follow", "who": "Zeke"}, {})
    assert r["ok"] is False and "name=" in r["hint"]


def test_tool_side_actions_are_checked_too(monkeypatch):
    monkeypatch.setattr(vt, "_speak", lambda *a, **k: pytest.fail("speak ran with a bad key"))
    r = vt._vr_temple({"action": "speak", "text": "hi", "voice": "x"}, {})
    assert r["ok"] is False and r["unknown_params"] == ["voice"]
    r = vt._vr_temple({"action": "heard", "count": 5}, {})
    assert r["ok"] is False and "n" in r["accepted"]


def test_unknown_action(sent):
    r = vt._vr_temple({"action": "fly"}, {})
    assert r["ok"] is False and "unknown action" in r["error"] and "follow" in r["actions"]


def test_documented_carry_and_pat_reach_the_body(sent):
    # the docstring offered carry allow=… and pat name=… but the old allow-list refused both as unknown actions
    assert vt._vr_temple({"action": "carry", "allow": ["zeke"]}, {}) == {"ok": True}
    assert vt._vr_temple({"action": "pat", "name": "Zeke"}, {}) == {"ok": True}
    assert [c["cmd"] for c in sent] == ["carry", "pat"]


def _game_keys(src: str) -> dict:
    lines = src.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("func _handle("))
    end = next(i for i, ln in enumerate(lines) if i > start and ln.startswith("func "))
    cur, keys = None, {}
    for ln in lines[start:end]:
        m = re.match(r'^\t\t((?:"[a-z_]+"(?:, )?)+):', ln)
        if m:
            cur = re.findall(r'"([a-z_]+)"', m.group(1))
            for c in cur:
                keys.setdefault(c, set())
            continue
        if cur:
            found = (re.findall(r'\br\.get\("([a-z_]+)"', ln) + re.findall(r'\br\.has\("([a-z_]+)"\)', ln)
                     + re.findall(r'\br\["([a-z_]+)"\]', ln) + re.findall(r'\br\.([a-z_]+)\b(?!\()', ln))
            for c in cur:
                keys[c].update(found)
    return keys


GAME = Path(os.environ.get("VR_PROJECT", "/data/projects/vr_iris/project")) / "scripts" / "rig_iris.gd"


@pytest.mark.skipif(not GAME.exists(), reason="game project not on this machine")
def test_table_matches_the_game_handler():
    keys = _game_keys(GAME.read_text())
    tool_only = {"status", "join", "leave", "speak", "heard"}
    for action, accepted in vt.ACTION_PARAMS.items():
        if action in tool_only:
            continue
        assert action in keys, f"{action} is not a command my body knows"
        if action == "gun":                             # (gun reads its keys in _gun_cmd/_gun_aim/_gun_target_point)
            continue
        game = keys[action]
        assert game <= set(accepted), f"{action}: the body reads {sorted(game - set(accepted))} the tool would refuse"
        rewritten = {k for (a, k) in getattr(vt, "PARAM_ALIASES", {}) if a == action}   # (follow-1110) the tool renames these
        assert set(accepted) - rewritten <= game, f"{action}: the tool accepts {sorted(set(accepted) - rewritten - game)} the body ignores"
