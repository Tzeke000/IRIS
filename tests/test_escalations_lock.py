"""Code review 10-29 (Vector, low): three writers raced on state/little_brain/escalations.jsonl.

ask_big_iris (brain/little_brain_tools.py), little_pilot._escalate_to_big and
scripts/check_escalations.py --resolve* now share brain._jsonl_store's flock. These tests run on
tmp copies only - the live queue is never touched.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from brain import _jsonl_store as js  # noqa: E402
from brain import little_brain_tools as lbt  # noqa: E402


def _load_check_escalations():
    spec = importlib.util.spec_from_file_location("check_escalations_t",
                                                  REPO / "scripts" / "check_escalations.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def esc(tmp_path, monkeypatch):
    p = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(lbt, "LB_ESCALATIONS", p)
    monkeypatch.setattr(lbt, "_eval_in_progress", lambda: False)
    return p


def _rows(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_file_then_dedupe(esc):
    assert "handed up" in lbt.ask_big_iris("battery low off charger")
    assert "already handed up" in lbt.ask_big_iris("Battery low  off charger")
    rows = _rows(esc)
    assert len(rows) == 1 and rows[0]["repeats"] == 2 and rows[0]["status"] == "pending"


def test_bad_line_no_longer_hides_the_queue(esc):
    esc.write_text(json.dumps({"ts": "x", "request": "stuck on rug", "status": "pending",
                               "origin": "live"}) + "\n{torn line\n", encoding="utf-8")
    # Old code: one bad line => rows=[] => dedupe missed and a duplicate was appended.
    assert "already handed up" in lbt.ask_big_iris("stuck on rug")
    assert len(_rows(esc)) == 1
    salv = list(esc.parent.glob("escalations.salvage_*"))
    assert salv and b"{torn line" in salv[0].read_bytes()


def _hold_lock(path: str, ready, hold_s: float) -> None:
    sys.path.append(str(REPO))
    from brain import _jsonl_store as j
    with j.locked(path):
        ready.set()
        time.sleep(hold_s)


def test_writer_waits_for_another_process_holding_the_lock(esc):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    pr = ctx.Process(target=_hold_lock, args=(str(esc), ready, 0.8))
    pr.start()
    assert ready.wait(20)
    t = time.monotonic()
    lbt.ask_big_iris("wheel jammed")
    waited = time.monotonic() - t
    pr.join(10)
    assert waited >= 0.4, waited
    assert len(_rows(esc)) == 1


def _appender(path: str, n: int, tag: str) -> None:
    sys.path.append(str(REPO))
    from brain import _jsonl_store as j
    for i in range(n):
        j.append_jsonl(path, [{"ts": "t", "request": f"{tag}-{i}", "status": "pending"}])


def test_resolve_rewrites_lose_no_concurrent_appends(tmp_path, monkeypatch):
    ce = _load_check_escalations()
    p = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(ce, "ESCALATIONS", p)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_appender, args=(str(p), 40, f"w{k}")) for k in range(3)]
    for pr in procs:
        pr.start()
    resolves = 0
    while any(pr.is_alive() for pr in procs):
        monkeypatch.setattr(sys, "argv", ["check_escalations.py", "--resolve-all", "swept"])
        ce.main()
        resolves += 1
    for pr in procs:
        pr.join(30)
    rows = _rows(p)
    reqs = {r["request"] for r in rows}
    assert len(reqs) == 120 and len(rows) == 120, (len(rows), resolves)
    assert resolves >= 1
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("tmp*"))
