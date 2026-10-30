"""10-29 code review, section "Message bridges" + body-host rows: cross-process flock, compare-and-set
status writes, chat claim/lease, list_pending, LLM priority, transcript O_APPEND + tail read, arbiter
ledger cap, iris_channel owner-loop send, host turn idle bound + VR cursor.

    /home/iris/venvs/iris-v100/bin/python -m pytest -q tests/test_bridges_review_1029.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="iris_bridges_test_"))

from brain import iris_chat, iris_llm, iris_transcript  # noqa: E402

iris_chat.configure(_TMP)
iris_llm.configure(_TMP)
iris_transcript.configure(_TMP)


def _clear(d: Path) -> None:
    for p in d.glob("*.json"):
        p.unlink()


# ---------------------------------------------------------------- iris_llm
def test_llm_flock_is_cross_process() -> None:
    """A child process holding state/iris_llm/.lock must make submit() wait (the flag race fix)."""
    code = (
        "import sys, time; sys.path.insert(0, %r)\n"
        "from brain import iris_llm\n"
        "iris_llm.configure(%r)\n"
        "with iris_llm._xlock():\n"
        "    print('held', flush=True)\n"
        "    time.sleep(0.8)\n"
    ) % (str(ROOT), str(_TMP))
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "held"
    t0 = time.monotonic()
    rid = iris_llm.submit("x", kind="general", deadline_s=60)
    waited = time.monotonic() - t0
    child.wait(timeout=10)
    assert waited >= 0.5, waited
    assert iris_llm.has_pending()
    assert iris_llm.mark_answered(rid, "ok")


def test_llm_cas_never_clobbers_answer_and_flag_tracks() -> None:
    _clear(iris_llm._llm_dir())
    rid = iris_llm.submit("q", kind="summarize", deadline_s=60)
    assert iris_llm.has_pending()
    assert iris_llm.mark_answered(rid, "the answer")
    assert not iris_llm.has_pending()
    p = iris_llm._request_path(rid)
    assert iris_llm._set_status_if_pending(p, "expired") is False
    assert iris_llm.get(rid)["status"] == "answered"
    assert iris_llm.get(rid)["reply"] == "the answer"
    rid2 = iris_llm.submit("q2", kind="summarize", deadline_s=60)
    assert iris_llm._set_status_if_pending(str(iris_llm._request_path(rid2)), "expired", "test")
    d = iris_llm.get(rid2)
    assert d["status"] == "expired" and d["expired_reason"] == "test"
    assert not iris_llm.has_pending()           # CAS refreshed the flag
    assert not list(iris_llm._llm_dir().glob("*.tmp"))


def test_llm_priority_beats_fifo() -> None:
    _clear(iris_llm._llm_dir())
    slow = iris_llm.submit("reflect", kind="reflect", deadline_s=300)
    time.sleep(0.01)
    fast = iris_llm.submit("classify", kind="classify_intent", deadline_s=30)
    nxt = iris_llm.next_pending()
    assert nxt["id"] == fast, nxt
    assert iris_llm.get(slow)["priority"] > iris_llm.get(fast)["priority"]
    explicit = iris_llm.submit("urgent", kind="reflect", deadline_s=60, priority=0)
    assert iris_llm.next_pending()["id"] == explicit
    _clear(iris_llm._llm_dir())
    iris_llm._refresh_flag()


# ---------------------------------------------------------------- iris_chat
def test_chat_claim_lease_and_list_pending() -> None:
    _clear(iris_chat._chat_dir())
    a = iris_chat.submit("one")
    time.sleep(0.01)
    b = iris_chat.submit("two")
    ids = [r["id"] for r in iris_chat.list_pending()]
    assert ids == [a, b]
    assert iris_chat.next_pending()["id"] == a
    assert iris_chat.claim(a, "host")
    assert not iris_chat.claim(a, "stop_hook")           # live lease held by someone else
    assert iris_chat.claim(a, "host")                     # renewal by the same owner
    assert iris_chat.is_pending(a)                        # status unchanged by a claim
    assert iris_chat.claimed_by_other(iris_chat.get(a), "stop_hook")
    assert iris_chat.claim(b, "stop_hook", lease_s=0.0)
    time.sleep(0.01)
    assert iris_chat.claim(b, "host")                     # expired lease can be taken over
    assert iris_chat.mark_answered(a, "hi")
    assert not iris_chat.claim(a, "host")                 # answered = nothing to deliver
    assert iris_chat.mark_answered(b, "hi")
    assert not iris_chat.has_pending()


def test_chat_ttl_expiry_is_cas() -> None:
    _clear(iris_chat._chat_dir())
    rid = iris_chat.submit("old")
    p = iris_chat._request_path(rid)
    d = json.loads(p.read_text())
    d["ts"] = time.time() - 10_000
    p.write_text(json.dumps(d))
    assert iris_chat.list_pending() == []
    assert iris_chat.get(rid)["status"] == "expired"
    assert not iris_chat.has_pending()
    assert not iris_chat.mark_answered("nope", "x")


# ---------------------------------------------------------------- transcript
def test_transcript_append_aware_iso_and_tail() -> None:
    p = iris_transcript._path()
    if p.exists():
        p.unlink()
    for i in range(400):
        iris_transcript.append("user", ("line %d " % i) + "x" * 300, "zeke", "chat")
    e = iris_transcript.recent(1)[0]
    assert e["content"].startswith("line 399 ")
    assert e["iso"][-6] in "+-" and e["iso"][-3] == ":", e["iso"]     # RFC 3339 offset
    last5 = iris_transcript.recent(5)
    assert [x["content"].split()[1] for x in last5] == ["395", "396", "397", "398", "399"]
    assert len(iris_transcript.recent(1000)) == 400
    # concurrent writers never interleave lines
    def w(tag):
        for _ in range(50):
            iris_transcript.append("user", tag * 20000, "zeke", "chat")
    ts = [threading.Thread(target=w, args=(c,)) for c in "ab"]
    [t.start() for t in ts]
    [t.join() for t in ts]
    for ln in p.read_text().splitlines():
        json.loads(ln)


# ---------------------------------------------------------------- attention arbiter
def test_arbiter_ledger_is_capped() -> None:
    from brain import attention_arbiter as arb
    led = _TMP / "situations.jsonl"
    old = (arb.ledger_path, arb.LEDGER_MAX_BYTES, arb.LEDGER_KEEP_LINES)
    arb.ledger_path = lambda: str(led)
    arb.LEDGER_MAX_BYTES, arb.LEDGER_KEEP_LINES = 20_000, 50
    try:
        for i in range(400):
            arb._append({"ts": time.time(), "situation": "s", "source": "x", "i": i})
        rows = [json.loads(x) for x in led.read_text().splitlines()]
        assert led.stat().st_size <= 20_000 + 200
        assert rows[-1]["i"] == 399
    finally:
        arb.ledger_path, arb.LEDGER_MAX_BYTES, arb.LEDGER_KEEP_LINES = old


# ---------------------------------------------------------------- iris_channel
def test_channel_emit_hands_write_to_owner_loop() -> None:
    from brain import iris_channel as ch

    sent_on = []
    owner_ready = threading.Event()
    holder = {}

    class _Stream:
        async def send(self, m):
            sent_on.append(threading.current_thread().name)

    class _Sess:
        _write_stream = _Stream()

    def _owner():
        loop = asyncio.new_event_loop()
        holder["loop"] = loop
        asyncio.set_event_loop(loop)

        async def _rec():
            ch.record_session(_Sess())
        loop.run_until_complete(_rec())
        owner_ready.set()
        loop.run_forever()

    t = threading.Thread(target=_owner, name="owner-loop", daemon=True)
    t.start()
    owner_ready.wait(5)
    try:
        ok = asyncio.run(ch.emit("hello", source="test-src", priority="attend", bypass_gates=True))
        assert ok is True
        assert sent_on == ["owner-loop"], sent_on
    finally:
        holder["loop"].call_soon_threadsafe(holder["loop"].stop)
        with ch._session_lock:
            ch._session = None
            ch._session_loop = None


# ---------------------------------------------------------------- body host
def _host():
    _fake_av = types.ModuleType("avaagent")
    _fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
    sys.modules.setdefault("avaagent", _fake_av)
    import iris_body_host as h
    h._blog = lambda *a, **k: None  # type: ignore[assignment]
    return h


def test_host_turn_idle_bound_interrupts_then_ends() -> None:
    h = _host()

    class _Turn:
        source = "orb"

        def __init__(self):
            self.done = asyncio.Event()

    class _Client:
        def __init__(self, turn, end_on_interrupt):
            self.turn, self.end, self.n = turn, end_on_interrupt, 0

        async def interrupt(self):
            self.n += 1
            if self.end:
                self.turn.done.set()

    async def ok_case():
        t = _Turn()
        c = _Client(t, True)
        await asyncio.wait_for(h._await_turn(t, c, idle_max_s=0.2, grace_s=5, poll_s=0.05), 5)
        return c.n

    assert asyncio.run(ok_case()) == 1

    async def wedged_case():
        t = _Turn()
        c = _Client(t, False)
        try:
            await asyncio.wait_for(h._await_turn(t, c, idle_max_s=0.1, grace_s=0.2, poll_s=0.05), 5)
        except RuntimeError:
            return c.n
        return -1

    assert asyncio.run(wedged_case()) == 1

    async def active_case():       # a streaming turn is never interrupted
        t = _Turn()
        c = _Client(t, True)

        async def stream():
            for _ in range(10):
                await asyncio.sleep(0.05)
                h._STREAM["last_msg_ts"] = time.time()
            t.done.set()
        asyncio.get_running_loop().create_task(stream())
        await asyncio.wait_for(h._await_turn(t, c, idle_max_s=0.2, grace_s=5, poll_s=0.05), 5)
        return c.n

    assert asyncio.run(active_case()) == 0


def test_host_vr_cursor_resume_and_rotation() -> None:
    h = _host()
    d = _TMP / "vr"
    d.mkdir(exist_ok=True)
    h.VR_HEARD_PATH = str(d / "heard.jsonl")
    h.VR_CURSOR_PATH = str(d / "heard_cursor.json")
    h._VR_CURSOR.update(pos=None, ino=None)
    Path(h.VR_HEARD_PATH).write_text(json.dumps({"id": 1, "attending": True}) + "\n")
    assert h._vr_cursor_start() == Path(h.VR_HEARD_PATH).stat().st_size     # cold start: tail only
    h._VR_CURSOR["ino"] = h._vr_file_ino()
    pos = h._vr_cursor_start()
    with open(h.VR_HEARD_PATH, "a") as f:
        f.write(json.dumps({"id": 2, "attending": True}) + "\n")
    pos, lines = h._vr_heard_read(pos)
    assert [x["id"] for x in lines] == [2]
    h._vr_cursor_save(pos)
    # a respawned reader (in-process cursor) resumes from the saved position: lines written during
    # the backoff are not lost
    with open(h.VR_HEARD_PATH, "a") as f:
        f.write(json.dumps({"id": 3, "attending": True}) + "\n")
    p2 = h._vr_cursor_start()
    assert p2 == pos
    _, lines = h._vr_heard_read(p2)
    assert [x["id"] for x in lines] == [3]
    # a fresh process honours a recent on-disk cursor
    h._VR_CURSOR.update(pos=None, ino=None)
    assert h._vr_cursor_start() == pos
    # rotation to a BIGGER file (new inode) is noticed
    h._VR_CURSOR["ino"] = h._vr_file_ino()
    new = d / "heard.new"
    new.write_text("".join(json.dumps({"id": 10 + i, "attending": True}) + "\n" for i in range(5)))
    os.replace(new, h.VR_HEARD_PATH)
    _, lines = h._vr_heard_read(pos)
    assert [x["id"] for x in lines] == [10, 11, 12, 13, 14]


def test_host_orb_claim_fail_open_and_dedupe() -> None:
    h = _host()
    _clear(iris_chat._chat_dir())
    rid = iris_chat.submit("hey")
    assert h._orb_claim(rid) is True
    assert iris_chat.claim(rid, "stop_hook") is False
    assert h._orb_claim(None) is True
    iris_chat.mark_answered(rid, "x")
    assert h._orb_claim(rid) is False
