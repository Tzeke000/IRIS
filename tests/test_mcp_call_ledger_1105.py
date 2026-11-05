"""MCP call ledger (11-05, docket I6): every tool call leaves a start + end row; a call past SLOW_DUMP_S dumps the
stacks + whether the loop still answers — so a 180 s hang says if it never reached a handler, blocked in one, or
blocked the loop."""
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain import mcp_call_ledger as L  # noqa: E402


class FakeTM:
    async def call_tool(self, name, arguments, context=None, convert_result=False):
        if arguments.get("boom"):
            raise ValueError("x")
        await asyncio.sleep(arguments.get("sleep", 0))
        return {"echo": name}


def _rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_start_end_rows_and_label(tmp_path):
    tm = FakeTM()
    assert L.install(tm, tmp_path) is True and L.install(tm, tmp_path) is False   # idempotent
    r = asyncio.run(tm.call_tool("iris_tool_call", {"name": "vr_temple", "params": {"action": "follow", "name": "Z"}}))
    assert r == {"echo": "iris_tool_call"}
    rows = _rows(tmp_path / "mcp_calls.jsonl")
    assert [x["ev"] for x in rows] == ["start", "end"]
    assert rows[0]["label"] == "iris_tool_call:vr_temple.follow" and rows[1]["result"] == "ok"
    try:
        asyncio.run(tm.call_tool("time_check", {"boom": True}))
    except ValueError:
        pass
    assert _rows(tmp_path / "mcp_calls.jsonl")[-1]["result"] == "ValueError"


def test_slow_dump_says_if_the_loop_is_blocked(tmp_path):
    L._paths["slow"] = tmp_path / "slow.log"

    async def main():
        L.note_loop(asyncio.get_running_loop())
        rec = {"seq": 1, "label": "time_check", "t0": time.time() - 61}
        # loop free: the probe answers
        txt = await asyncio.to_thread(L.dump_slow, rec, time.time())
        assert "answers in" in txt
        # loop blocked by a sync call (what an inline sync tool does): the probe says BLOCKED
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("t", L.dump_slow(rec, time.time())))
        th.start()
        time.sleep(2.5)                                  # block the loop
        th.join()
        assert "BLOCKED" in out["t"]
    asyncio.run(main())
    assert "SLOW MCP CALL #1 time_check" in (tmp_path / "slow.log").read_text()
