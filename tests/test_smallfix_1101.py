"""2026-11-01 docaudit small fixes: chat-side deadline + abandoned status, voice latency stage stamps,
delegated tasks -> act verification ledger.

    /home/iris/venvs/iris-v100/bin/python -m pytest -q tests/test_smallfix_1101.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import iris_chat  # noqa: E402
from brain.iris_paths import paths as _paths  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_paths():
    """iris_chat.configure() re-points the GLOBAL brain.iris_paths root; put it back after each test
    so other test modules' bridges stay in their own temp dirs."""
    prev_root, prev_base = _paths._root, iris_chat._BASE
    yield
    iris_chat._BASE = prev_base
    if prev_root is not None:
        _paths.configure(prev_root)


def _chat_tmp() -> Path:
    d = Path(tempfile.mkdtemp(prefix="iris_smallfix_chat_"))
    iris_chat.configure(d)
    return iris_chat._chat_dir()


# ---------------------------------------------------------------- 1. chat deadline + abandoned
def test_chat_submit_writes_deadline_default_is_flat_ttl() -> None:
    _chat_tmp()
    rid = iris_chat.submit("hi")
    d = iris_chat.get(rid)
    assert abs((d["deadline_ts"] - d["ts"]) - iris_chat._REQUEST_TTL_S) < 1e-6
    rid2 = iris_chat.submit("hi", deadline_s=30)
    d2 = iris_chat.get(rid2)
    assert abs((d2["deadline_ts"] - d2["ts"]) - 30.0) < 1e-6


def test_chat_past_deadline_expires_and_clears_flag() -> None:
    _chat_tmp()
    rid = iris_chat.submit("short", deadline_s=0.05)
    assert iris_chat.has_pending()
    time.sleep(0.1)
    assert iris_chat.list_pending() == []
    d = iris_chat.get(rid)
    assert d["status"] == "expired" and d["expired_reason"] == "deadline"
    assert not iris_chat.has_pending()


def test_chat_legacy_file_without_deadline_uses_flat_ttl() -> None:
    cdir = _chat_tmp()
    now = time.time()
    legacy = {"id": "legacy1", "ts": now - 30, "user_text": "old format", "status": "pending",
              "reply": None, "answered_ts": None}
    (cdir / "legacy1.json").write_text(json.dumps(legacy))
    assert [r["id"] for r in iris_chat.list_pending()] == ["legacy1"]
    legacy["ts"] = now - iris_chat._REQUEST_TTL_S - 5
    (cdir / "legacy1.json").write_text(json.dumps(legacy))
    assert iris_chat.list_pending() == []
    assert iris_chat.get("legacy1")["status"] == "expired"


def test_chat_deadline_never_extends_past_flat_ttl() -> None:
    _chat_tmp()
    rid = iris_chat.submit("long", deadline_s=10 * iris_chat._REQUEST_TTL_S)
    p = iris_chat._request_path(rid)
    d = json.loads(p.read_text())
    d["ts"] = time.time() - iris_chat._REQUEST_TTL_S - 5
    p.write_text(json.dumps(d))
    assert iris_chat.list_pending() == []


def test_chat_wait_abandons_only_when_asked_and_answer_wins() -> None:
    _chat_tmp()
    keep = iris_chat.submit("orb style")
    assert iris_chat.wait_for_reply(keep, timeout_s=0.3) is None
    assert iris_chat.is_pending(keep)                     # default: stays deliverable (orb)
    gone = iris_chat.submit("only waiter")
    assert iris_chat.wait_for_reply(gone, timeout_s=0.3, abandon_on_timeout=True) is None
    d = iris_chat.get(gone)
    assert d["status"] == "abandoned" and d["abandoned_reason"] == "waiter timed out"
    assert [r["id"] for r in iris_chat.list_pending()] == [keep]
    assert not iris_chat.is_pending(gone)                 # the host drops it at fire time
    assert iris_chat.mark_answered(keep, "late but answered")
    assert not iris_chat.abandon(keep)                    # an answer is never overwritten
    assert iris_chat.get(keep)["status"] == "answered"
    assert not iris_chat.has_pending()


def test_chat_wait_returns_early_on_abandoned() -> None:
    _chat_tmp()
    rid = iris_chat.submit("x")
    assert iris_chat.abandon(rid, "test")
    t0 = time.monotonic()
    assert iris_chat.wait_for_reply(rid, timeout_s=5.0) is None
    assert time.monotonic() - t0 < 1.0


def test_stop_hook_skips_chat_past_its_deadline() -> None:
    spec = importlib.util.spec_from_file_location("vsh_smallfix", ROOT / "scripts" / "voice_stop_hook.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cdir = _chat_tmp()
    m.CHAT_DIR = cdir
    live = iris_chat.submit("live")
    dead = iris_chat.submit("dead")
    p = iris_chat._request_path(dead)
    d = json.loads(p.read_text())
    d["ts"] -= 1.0                      # older => would be picked first (FIFO) if not skipped
    d["deadline_ts"] = time.time() - 1.0
    p.write_text(json.dumps(d))
    assert m._next_pending_chat()["id"] == live
