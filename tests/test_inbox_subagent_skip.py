"""The PostToolUse inbox checkpoint must never hand Zeke's messages to a SUB-AGENT (10-30 bug:
two DMs were drained into a Workflow helper's context). agent_id is set only inside sub-agents."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_fake_av = types.ModuleType("avaagent")
_fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
sys.modules.setdefault("avaagent", _fake_av)

import iris_body_host as h  # noqa: E402

h._blog = lambda *a, **k: None  # type: ignore[assignment]


def test_subagent_tool_call_does_not_drain_inbox():
    calls = []
    h._INBOX["turn"] = types.SimpleNamespace(active=True, guest=False)
    orig = h._inbox_drain
    h._inbox_drain = lambda: calls.append(1) or []  # type: ignore[assignment]
    try:
        out = asyncio.run(h._inbox_post_tool_hook({"agent_id": "a123", "tool_name": "Bash"}, "t1", None))
        assert out == {} and not calls, "sub-agent hook must not touch the inbox"
        asyncio.run(h._inbox_post_tool_hook({"tool_name": "Bash"}, "t2", None))
        assert calls, "main-thread hook must still drain"
    finally:
        h._inbox_drain = orig  # type: ignore[assignment]
        h._INBOX["turn"] = None
