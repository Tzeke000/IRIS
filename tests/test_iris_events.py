"""V3 Phase 1 steps 1.1 + 1.3: brain/iris_events.py — durable event store, CAS
transitions, idempotent inserts, crash/corruption recovery, fenced task claims,
file leases, checkpoints, expired-lease sweep. Every test uses a tmp DB."""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from brain import iris_events as ie  # noqa: E402
from brain.iris_events import EventStore  # noqa: E402

PY = sys.executable


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _store(tmp_path: Path, **kw) -> EventStore:
    return EventStore(tmp_path / "events.sqlite", lease_root=tmp_path / "repo", **kw)


def _env(db: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["IRIS_EVENTS_DB"] = str(db)
    return env


def _spawn(code: str, db: Path, *args: str) -> subprocess.Popen:
    return subprocess.Popen([PY, "-c", code, *args], env=_env(db), cwd=str(REPO),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _raw(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    return c


# ══ 1.1 event store ══════════════════════════════════════════════════════════

def test_default_path_and_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv("IRIS_EVENTS_DB", raising=False)
    p = ie.default_db_path()
    assert p.parts[-3:] == ("state", "v3", "events.sqlite")
    monkeypatch.setenv("IRIS_EVENTS_DB", str(tmp_path / "x.sqlite"))
    assert ie.default_db_path() == tmp_path / "x.sqlite"
    s = ie.get_store()
    assert s.path == tmp_path / "x.sqlite" and (tmp_path / "x.sqlite").is_file()
    assert ie.get_store() is s


def test_wal_mode_and_schema(tmp_path):
    s = _store(tmp_path)
    c = _raw(s.path)
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"events", "transitions", "tasks", "leases"} <= names


def test_insert_lifecycle_and_audit(tmp_path):
    s = _store(tmp_path)
    r = s.insert("discord", "hi", idempotency_key="discord:1", sender="zeke", trusted=True,
                 priority=3, meta={"chat_id": "c"})
    assert r.created and r.status == "received"
    ev = s.get(r.id)
    assert ev["trace_id"] == r.id and ev["fence"] == 0 and ev["meta"] == {"chat_id": "c"}
    assert s.transition(r.id, "received", 0, "admitted", actor="host") is not None
    a = s.transition(r.id, "admitted", 0, "assigned", owner="host-turn:1", lease_s=60)
    assert a["fence"] == 1 and a["owner"] == "host-turn:1" and a["lease_until"] > 0
    for frm, to in (("assigned", "running"), ("running", "output_pending"),
                    ("output_pending", "verified"), ("verified", "committed")):
        assert s.transition(r.id, frm, 1, to) is not None
    # terminal: nothing moves it again, not even with expect_status=None
    assert s.transition(r.id, None, None, "admitted") is None
    assert s.transition(r.id, "committed", 1, "admitted") is None
    hist = [(h["from_status"], h["to_status"]) for h in s.history(r.id)]
    assert hist[0] == (None, "received") and hist[-1] == ("verified", "committed")
    assert len(hist) == 7


def test_cas_rejects_wrong_status_or_fence_and_bad_input_fails_open(tmp_path):
    s = _store(tmp_path)
    r = s.insert("orb", "x", status="admitted")
    assert s.transition(r.id, "received", None, "assigned") is None       # wrong status
    assert s.transition(r.id, "admitted", 7, "assigned") is None          # wrong fence
    assert s.transition("nope", None, None, "assigned") is None           # missing row
    # programming errors never raise to the caller — they return the falsy default
    assert s.transition(r.id, "admitted", 0, "bogus") is None
    assert s.transition(r.id, "admitted", 0, "received") is None
    assert s.transition(r.id, "admitted", 0, "assigned", fence=9) is None  # not a field
    assert s.last_error
    assert s.get(r.id)["status"] == "admitted"


def test_transitions_are_append_only(tmp_path):
    s = _store(tmp_path)
    s.insert("orb", "x")
    c = _raw(s.path)
    with pytest.raises(sqlite3.DatabaseError):
        c.execute("UPDATE transitions SET to_status='x'")
    with pytest.raises(sqlite3.DatabaseError):
        c.execute("DELETE FROM transitions")


def test_trace_inherits_from_parent(tmp_path):
    s = _store(tmp_path)
    root = s.insert("discord", "root")
    child = s.insert("llm", "child", parent_id=root.id)
    assert s.get(child.id)["trace_id"] == root.id
    t = s.create_task("do it", parent_event_id=child.id)
    assert s.get_task(t.id)["trace_id"] == root.id
    assert {x["entity_id"] for x in s.trace(root.id)} == {root.id, child.id, t.id}


def test_duplicate_idempotency_key_one_row(tmp_path):
    s = _store(tmp_path)
    a = s.insert("discord", "hello", idempotency_key="discord:42")
    b = s.insert("discord", "hello again", idempotency_key="discord:42")
    assert a.created and not b.created and a.id == b.id
    assert s.find_by_key("discord:42")["text"] == "hello"
    assert _raw(s.path).execute("SELECT count(*) FROM events").fetchone()[0] == 1


def test_duplicate_idempotency_key_across_processes(tmp_path):
    db = tmp_path / "events.sqlite"
    EventStore(db)
    code = ("import sys, json\nfrom brain.iris_events import get_store\ns = get_store()\n"
            "out = [s.insert('discord', 't', idempotency_key=f'discord:{i}').created"
            " for i in range(40)]\nprint(json.dumps(out))")
    procs = [_spawn(code, db) for _ in range(4)]
    created = 0
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        created += sum(json.loads(out))
    assert created == 40
    assert _raw(db).execute("SELECT count(*) FROM events").fetchone()[0] == 40


_CAS_CHILD = r"""
import json, os, sys, time
from brain.iris_events import EventStore
s = EventStore(os.environ["IRIS_EVENTS_DB"])
ids = json.loads(sys.argv[1]); n = int(sys.argv[2]); me = sys.argv[3]; go = sys.argv[4]
while not os.path.exists(go):
    time.sleep(0.002)
wins = []
for i in range(n):
    eid = ids[i % len(ids)]
    row = s.get(eid)
    st, f = row["status"], row["fence"]
    if st == "admitted":
        r = s.transition(eid, "admitted", f, "assigned", owner=me, actor=me)
    else:
        r = s.transition(eid, "assigned", f, "admitted", owner=None, actor=me)
    if r is not None:
        wins.append([eid, st, f])
print(json.dumps({"wins": wins, "err": s.last_error, "degraded": s.degraded}))
"""


def test_cas_8_processes_x_500_exactly_one_winner(tmp_path):
    db = tmp_path / "events.sqlite"
    s = EventStore(db)
    ids = [s.insert("voice", f"row{i}", status="admitted").id for i in range(3)]
    go = tmp_path / "go"
    procs = [_spawn(_CAS_CHILD, db, json.dumps(ids), "500", f"p{k}", str(go)) for k in range(8)]
    time.sleep(0.5)
    go.write_text("go")
    wins = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        res = json.loads(out)
        assert res["err"] is None and not res["degraded"], res
        wins += [tuple(w) for w in res["wins"]]
    # each (row, from_status, fence) version was won by exactly one process
    assert len(wins) == len(set(wins))
    assert len(wins) > 50
    c = _raw(db)
    n_trans = c.execute("SELECT count(*) FROM transitions WHERE from_status IS NOT NULL"
                        ).fetchone()[0]
    assert n_trans == len(wins)
    for eid in ids:
        mine = [w for w in wins if w[0] == eid]
        assigns = sum(1 for w in mine if w[1] == "admitted")
        row = s.get(eid)
        assert row["fence"] == assigns
        assert row["status"] == ("admitted" if len(mine) % 2 == 0 else "assigned")
    assert s.integrity_check() == "ok"


_KILL_LOOP_CHILD = r"""
import os, sys
from brain.iris_events import EventStore
s = EventStore(os.environ["IRIS_EVENTS_DB"])
i = 0
while True:
    r = s.insert("voice", "x" * 2000, idempotency_key=f"loop:{i}")
    s.transition(r.id, "received", 0, "admitted")
    i += 1
    if i == 30:
        print("ready", flush=True)
"""

_KILL_HALF_TX_CHILD = r"""
import os, sqlite3, sys, time
c = sqlite3.connect(os.environ["IRIS_EVENTS_DB"], isolation_level=None)
c.execute("PRAGMA journal_mode=WAL")
c.execute("BEGIN IMMEDIATE")
for i in range(3000):
    c.execute("INSERT INTO events(event_id, source, text, received_ts, status, updated_ts,"
              " idempotency_key) VALUES(?, 'voice', ?, 0, 'received', 0, ?)",
              (f"half{i}", "y" * 500, f"half:{i}"))
print("mid", flush=True)
time.sleep(60)
"""


def test_kill_9_writer_mid_transaction_leaves_db_intact(tmp_path):
    db = tmp_path / "events.sqlite"
    EventStore(db)
    # (a) a busy writer killed at an arbitrary point
    p = _spawn(_KILL_LOOP_CHILD, db)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(0.15)
    os.kill(p.pid, signal.SIGKILL)
    p.wait(timeout=10)
    # (b) a writer killed holding an open, uncommitted transaction
    p = _spawn(_KILL_HALF_TX_CHILD, db)
    assert p.stdout.readline().strip() == "mid"
    os.kill(p.pid, signal.SIGKILL)
    p.wait(timeout=10)

    s = EventStore(db)
    assert s.integrity_check() == "ok"
    c = _raw(db)
    assert c.execute("SELECT count(*) FROM events WHERE event_id LIKE 'half%'").fetchone()[0] == 0
    # every event row has its insert audit row (insert + audit are one transaction)
    orphans = c.execute("SELECT count(*) FROM events e WHERE NOT EXISTS (SELECT 1 FROM"
                        " transitions t WHERE t.entity_id = e.event_id)").fetchone()[0]
    assert orphans == 0
    assert c.execute("SELECT count(*) FROM events").fetchone()[0] >= 30
    # the write lock died with the writer
    assert s.insert("discord", "after", idempotency_key="after").created


def test_corrupt_db_file_moved_aside_recreated_with_incident(tmp_path):
    db = tmp_path / "events.sqlite"
    garbage = b"this is not a sqlite database at all " * 200
    db.write_bytes(garbage)
    s = EventStore(db)
    assert not s.degraded
    inc = s.incidents()
    assert len(inc) == 1 and inc[0]["status"] == "needs_review" and inc[0]["source"] == "store"
    assert "moved aside" in inc[0]["text"]
    moved = list(tmp_path.glob("events.sqlite.corrupt-*"))
    assert [m for m in moved if m.read_bytes() == garbage]
    assert s.integrity_check() == "ok"
    # the incident is not mistaken for an input to answer
    assert s.unanswered() == []
    assert len(s.unanswered(include_incidents=True)) == 1
    assert s.insert("discord", "works", idempotency_key="discord:1").created
    assert s.stats()["open_incidents"] == 1


def test_corruption_discovered_mid_life_and_by_another_process(tmp_path):
    db = tmp_path / "events.sqlite"
    s = EventStore(db)
    for i in range(200):
        s.insert("voice", "z" * 500, idempotency_key=f"v:{i}")
    s.close()
    other = EventStore(db)   # a second process-like handle, opened while the file was good
    assert other.get("nothing") is None
    c = sqlite3.connect(str(db))
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()
    raw = bytearray(db.read_bytes())
    raw[4096:] = b"\xde\xad\xbe\xef" * ((len(raw) - 4096) // 4)   # trash every page after page 1
    db.write_bytes(bytes(raw))
    s2 = EventStore(db)
    # whatever it hits first, it must not raise, and must end up on a healthy store
    s2.unanswered()
    r = s2.insert("discord", "after", idempotency_key="after:1")
    assert r is not None and r.created
    assert s2.integrity_check() == "ok"
    assert len(s2.incidents()) == 1
    # the older handle notices the file was replaced and follows it (no second move-aside)
    assert other.find_by_key("after:1") is not None
    moved = [m for m in tmp_path.glob("events.sqlite.corrupt-*")
             if not m.name.endswith(("-wal", "-shm"))]
    assert len(moved) == 1


def _filled(db: Path) -> None:
    s = EventStore(db)
    for i in range(300):
        s.insert("voice", "z" * 500, idempotency_key=f"v:{i}")
    s.close()
    c = sqlite3.connect(str(db))
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()


def test_truncated_db_recovered(tmp_path):
    db = tmp_path / "events.sqlite"
    _filled(db)
    data = db.read_bytes()
    db.write_bytes(data[: len(data) // 3])
    s = EventStore(db)
    s.unanswered()
    assert s.insert("discord", "x", idempotency_key="k").created
    assert s.integrity_check() == "ok" and len(s.incidents()) == 1


def test_corruption_under_an_open_handle_recovers_inside_the_call(tmp_path):
    db = tmp_path / "events.sqlite"
    _filled(db)
    s = EventStore(db)
    with open(db, "r+b") as f:          # same inode: only the failing statement can tell
        f.seek(8192)
        f.write(b"\xde\xad" * 20000)
    r = s.insert("discord", "x", idempotency_key="k")
    assert r is not None and r.created   # recovered + retried within the same call
    assert s.integrity_check() == "ok" and len(s.incidents()) == 1


def test_unopenable_path_fails_open(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    s = EventStore(blocker / "sub" / "events.sqlite")
    assert s.degraded
    assert s.insert("discord", "x") is None
    assert s.unanswered() == []
    assert s.transition("e", None, None, "admitted") is None
    assert s.claim("t", "me") is None
    assert s.mark_replied("e", "r") is False
    assert s.acquire_leases("t", ["a/**"])["ok"] is True       # advisory ⇒ fail open
    assert s.check_write("t", "a/b")["allowed"] is True
    assert s.stats()["ok"] is False


def test_path_that_is_a_directory_is_moved_aside(tmp_path):
    db = tmp_path / "events.sqlite"
    db.mkdir()
    (db / "junk").write_text("x")
    s = EventStore(db)
    assert not s.degraded and db.is_file()
    assert len(s.incidents()) == 1


def test_unanswered_mark_replied_find_by_key(tmp_path):
    s = _store(tmp_path, clock=Clock())
    d1 = s.insert("discord", "a", idempotency_key="discord:1", priority=3)
    s.insert("voice", "b", idempotency_key="voice:1", priority=1)
    d3 = s.insert("discord", "c", idempotency_key="discord:3", priority=3)
    assert [e["text"] for e in s.unanswered()] == ["b", "a", "c"]   # priority ASC, then age
    assert [e["text"] for e in s.unanswered("discord")] == ["a", "c"]
    assert s.mark_replied(d1.id, "msg:900")
    assert s.mark_replied(d1.id, "msg:900")                          # idempotent
    assert s.mark_replied(d1.id, "msg:901")
    e = s.find_by_key("discord:1")
    assert e["status"] == "committed" and e["reply_ref"] == "msg:900,msg:901"
    assert [x["text"] for x in s.unanswered("discord")] == ["c"]
    assert len([h for h in s.history(d1.id) if h["note"] and "reply" in h["note"]]) == 2
    # commit=False: a reply that went out mid-turn ⇒ output_pending, still "unanswered"
    assert s.mark_replied(d3.id, "msg:902", commit=False)
    assert s.get(d3.id)["status"] == "output_pending" and s.get(d3.id)["reply_ref"] == "msg:902"
    assert s.mark_replied("missing", "x") is False
    assert s.find_by_key("discord:404") is None


def test_event_lease_expiry_sweeps_to_needs_review(tmp_path):
    clk = Clock()
    s = _store(tmp_path, clock=clk)
    r = s.insert("voice", "x", status="admitted")
    s.transition(r.id, "admitted", 0, "assigned", owner="host-turn:7", lease_s=30)
    clk.t += 10
    assert s.sweep_expired()["events_needs_review"] == 0
    clk.t += 30
    assert s.sweep_expired()["events_needs_review"] == 1
    e = s.get(r.id)
    assert e["status"] == "needs_review" and "host-turn:7" in e["note"]
    assert e in s.unanswered()


# ══ 1.3 task claims, fencing, leases, checkpoints ═══════════════════════════

def test_two_competing_claims_one_winner(tmp_path):
    s = _store(tmp_path)
    t = s.create_task("render", kind="worker")
    f1 = s.claim(t.id, "worker-a", 60)
    f2 = s.claim(t.id, "worker-b", 60)
    assert f1 == 1 and f2 is None
    assert s.get_task(t.id)["owner"] == "worker-a"


def test_competing_claims_across_processes_one_winner(tmp_path):
    db = tmp_path / "events.sqlite"
    s = EventStore(db)
    tids = [s.create_task(f"job{i}").id for i in range(10)]
    go = tmp_path / "go"
    code = ("import json, os, sys, time\nfrom brain.iris_events import get_store\n"
            "s = get_store(); tids = json.loads(sys.argv[1])\n"
            "while not os.path.exists(sys.argv[3]): time.sleep(0.002)\n"
            "print(json.dumps({t: s.claim(t, sys.argv[2], 60) for t in tids}))")
    procs = [_spawn(code, db, json.dumps(tids), f"w{k}", str(go)) for k in range(6)]
    time.sleep(0.3)
    go.write_text("go")
    winners = {t: [] for t in tids}
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        for t, f in json.loads(out).items():
            if f is not None:
                winners[t].append(f)
    assert all(v == [1] for v in winners.values()), winners


def test_late_complete_with_old_fence_rejected(tmp_path):
    clk = Clock()
    s = _store(tmp_path, clock=clk)
    t = s.create_task("encode", side_effects="reversible")
    fa = s.claim(t.id, "worker-a", lease_s=10)
    clk.t += 60                                   # A stalls past its lease
    assert s.sweep_expired()["tasks_needs_review"] == 1
    assert s.get_task(t.id)["status"] == "needs_review"
    assert s.heartbeat(t.id, fa) is False         # A learns it lost the task
    assert s.requeue_task(t.id)
    fb = s.claim(t.id, "worker-b", lease_s=10)
    assert fb == fa + 1
    assert s.complete(t.id, fa, "result-A") is False      # stale writer refused
    assert s.complete(t.id, fb, "result-B") is True
    assert s.complete(t.id, fa, "result-A") is False      # still refused afterwards
    got = s.get_task(t.id)
    assert got["result_ref"] == "result-B" and got["status"] == "output_pending"
    assert got["attempts"] == 2


def test_out_of_order_completions_each_against_own_fence(tmp_path):
    s = _store(tmp_path)
    a, b = s.create_task("A"), s.create_task("B")
    s.claim(a.id, "wa")
    s.claim(b.id, "wb")
    # bump B's fence so the two fences differ
    s.transition_task(b.id, "running", 1, "needs_review")
    s.requeue_task(b.id)
    fb = s.claim(b.id, "wb2")
    assert fb == 2
    assert s.complete(b.id, fb, "rB")              # B finishes first
    assert s.complete(a.id, 1, "rA")               # then A
    ha = [h for h in s.history(a.id) if h["to_status"] == "output_pending"]
    hb = [h for h in s.history(b.id) if h["to_status"] == "output_pending"]
    assert ha[0]["fence"] == 1 and hb[0]["fence"] == 2
    assert hb[0]["seq"] < ha[0]["seq"]
    assert s.get_task(a.id)["result_ref"] == "rA" and s.get_task(b.id)["result_ref"] == "rB"


def test_duplicate_completion_is_idempotent(tmp_path):
    s = _store(tmp_path)
    t = s.create_task("x")
    f = s.claim(t.id, "w")
    assert s.complete(t.id, f, "r1")
    assert s.complete(t.id, f, "r1")
    assert s.complete(t.id, f)                         # no result_ref ⇒ still a duplicate
    assert s.complete(t.id, f, "r2") is False          # a DIFFERENT result is not a duplicate
    assert len([h for h in s.history(t.id) if h["to_status"] == "output_pending"]) == 1
    # advancing with the same fence works; going backwards is a no-op duplicate
    assert s.complete(t.id, f, status="verified") is False          # needs a verification ref
    assert s.complete(t.id, f, status="verified", verification_ref="act:12")
    assert s.complete(t.id, f, "r1", status="output_pending")       # already past it ⇒ True
    assert s.complete(t.id, f, status="committed")
    assert s.get_task(t.id)["status"] == "committed"
    assert s.complete(t.id, f, status="bogus") is False


def test_external_side_effects_never_auto_retried(tmp_path):
    clk = Clock()
    s = _store(tmp_path, clock=clk)
    ext = s.create_task("discord send", side_effects="external")
    safe = s.create_task("lint", side_effects="none")
    s.claim(ext.id, "w", 5)
    s.claim(safe.id, "w", 5)
    clk.t += 10
    out = s.sweep_expired(auto_retry=True)
    assert out["tasks_needs_review"] == 1 and out["tasks_requeued"] == 1
    assert s.get_task(ext.id)["status"] == "needs_review"
    assert s.get_task(safe.id)["status"] == "admitted"
    assert s.requeue_task(ext.id) is False                    # must be reconciled first
    assert s.requeue_task(ext.id, force=True, note="checked: not sent")
    f = s.claim(ext.id, "w2", 5)
    assert s.complete(ext.id, f, "sent:1", status="committed") is False   # needs verification
    assert s.complete(ext.id, f, "sent:1", status="committed", verification_ref="act:9")
    assert s.create_task("x", side_effects="maybe") is None


def test_expired_task_lease_to_needs_review(tmp_path):
    clk = Clock()
    s = _store(tmp_path, clock=clk)
    t = s.create_task("long job", kind="worker")
    f = s.claim(t.id, "launcher", lease_s=30)
    clk.t += 20
    assert s.heartbeat(t.id, f)                 # extends to now+30
    clk.t += 25
    assert s.sweep_expired()["tasks_needs_review"] == 0
    assert s.stats()["expired_task_leases"] == 0
    clk.t += 10
    assert s.stats()["expired_task_leases"] == 1
    assert s.sweep_expired()["tasks_needs_review"] == 1
    got = s.get_task(t.id)
    assert got["status"] == "needs_review" and "launcher" in got["note"]
    assert s.complete(t.id, f, "late") is False     # not running any more
    assert s.claim(t.id, "other") is None           # not claimable until requeued


def test_fail_and_cancel(tmp_path):
    s = _store(tmp_path)
    t = s.create_task("x")
    f = s.claim(t.id, "w")
    assert s.fail(t.id, f + 1, "wrong fence") is False
    assert s.fail(t.id, f, "boom")
    assert s.get_task(t.id)["status"] == "failed"
    assert s.requeue_task(t.id)
    assert s.cancel_task(t.id, "no longer wanted")
    assert s.cancel_task(t.id) is False
    assert s.get_task(t.id)["status"] == "cancelled"


def test_task_idempotency_key(tmp_path):
    s = _store(tmp_path)
    a = s.create_task("x", idempotency_key="job:1")
    b = s.create_task("x", idempotency_key="job:1")
    assert a.created and not b.created and a.id == b.id
    assert s.find_task_by_key("job:1")["task_id"] == a.id
    assert [x["task_id"] for x in s.open_tasks()] == [a.id]


def test_checkpoint_read_write(tmp_path):
    s = _store(tmp_path)
    t = s.create_task("x", checkpoint={"goal": "x", "steps_done": []})
    assert s.read_checkpoint(t.id) == {"goal": "x", "steps_done": []}
    f = s.claim(t.id, "w")
    cp = {"goal": "x", "steps_done": ["a"], "outstanding": ["b"], "files": ["brain/x.py"],
          "running_jobs": [], "locks": [], "next_safe_action": "run b"}
    assert s.write_checkpoint(t.id, cp, fence=f)
    got = s.read_checkpoint(t.id)
    assert {k: v for k, v in got.items() if k != "_ts"} == cp and "_ts" in got
    assert s.write_checkpoint(t.id, {"stale": True}, fence=f - 1) is False
    assert s.get_task(t.id)["checkpoint"]["next_safe_action"] == "run b"
    assert s.write_checkpoint("missing", {}) is False
    assert s.read_checkpoint("missing") is None


def test_glob_overlap_rules():
    ov = ie.globs_overlap
    assert ov("brain/**", "brain/iris_events.py")
    assert ov("brain/**", "brain/sub/deep/x.py")
    assert ov("**", "anything/at/all")
    assert ov("tests/*.py", "tests/test_x.py")
    assert not ov("tests/*.py", "tests/sub/test_x.py")
    assert ov("tests/**/*.py", "tests/sub/test_x.py")
    assert not ov("scripts/*.sh", "scripts/*.py")
    assert not ov("brain/a_*.py", "brain/b_*.py")
    assert ov("brain/iris_*.py", "brain/*_events.py")
    assert not ov("brain/x.py", "brain/y.py")
    assert ov("brain/", "brain/x.py")                   # trailing slash = the whole dir
    assert ov("./brain/../brain/x.py", "brain/x.py")
    assert not ov("brain/x.py", "tests/x.py")


def test_overlapping_lease_refused(tmp_path):
    root = tmp_path / "repo"
    s = _store(tmp_path)
    t1, t2, t3 = (s.create_task(n) for n in ("t1", "t2", "t3"))
    f1 = s.claim(t1.id, "h1", 600)
    s.claim(t2.id, "h2", 600)
    assert s.acquire_leases(t1.id, ["brain/**"], fence=f1)["ok"]
    r = s.acquire_leases(t2.id, ["tests/test_a.py", "brain/iris_events.py"])
    assert not r["ok"] and r["conflicts"][0]["held_by"] == t1.id
    assert s.lease_holders("brain/x.py")[0]["task_id"] == t1.id
    assert not [l for l in s.lease_holders("tests/test_a.py")]    # all-or-nothing: none taken
    assert s.acquire_leases(t2.id, ["tests/*.py"])["ok"]
    assert not s.acquire_leases(t3.id, [str(root / "tests" / "test_z.py")])["ok"]   # abs path
    assert s.acquire_leases(t3.id, ["scripts/*.sh"])["ok"]
    assert s.acquire_leases(t1.id, ["brain/iris_events.py"])["ok"]   # own overlap is fine
    assert s.acquire_leases(t2.id, ["docs/**"], fence=99)["ok"] is False   # stale fence
    # the PreToolUse question
    assert s.check_write(t1.id, str(root / "brain" / "iris_events.py"))["allowed"]
    assert not s.check_write(t2.id, "brain/iris_events.py")["allowed"]
    assert not s.check_write(t1.id, "README.md")["allowed"]          # outside declared globs
    # completion releases the leases ⇒ t2 can now take the brain file
    assert s.complete(t1.id, f1, "done")
    assert s.acquire_leases(t2.id, ["brain/iris_events.py"])["ok"]


def test_leases_expire_follow_heartbeat_and_stale_fence(tmp_path):
    clk = Clock()
    s = _store(tmp_path, clock=clk)
    a, b = s.create_task("a"), s.create_task("b")
    fa = s.claim(a.id, "wa", lease_s=30)
    s.claim(b.id, "wb", lease_s=300)
    assert s.acquire_leases(a.id, ["state/**"])["ok"]
    clk.t += 20
    assert s.heartbeat(a.id, fa)                         # lease follows the task to now+30
    clk.t += 20
    assert not s.acquire_leases(b.id, ["state/x.json"])["ok"]
    clk.t += 20                                          # a's lease ran out
    assert s.acquire_leases(b.id, ["state/x.json"])["ok"]
    # a lease held under an old fence is dead once the task is re-claimed
    c = s.create_task("c")
    fc = s.claim(c.id, "wc", lease_s=1000)
    assert s.acquire_leases(c.id, ["logs/**"], ttl_s=1000)["ok"]
    s.transition_task(c.id, "running", fc, "needs_review")
    s.requeue_task(c.id)
    s.claim(c.id, "wc2", lease_s=1000)
    assert s.acquire_leases(b.id, ["logs/a.log"])["ok"]


def test_sweep_drops_expired_leases(tmp_path):
    clk = Clock()
    s = _store(tmp_path, clock=clk)
    t = s.create_task("x")
    s.claim(t.id, "w", lease_s=10_000)
    s.acquire_leases(t.id, ["a/**"], ttl_s=5)
    clk.t += 10
    assert s.sweep_expired()["leases_dropped"] == 1


def test_stats_shape(tmp_path):
    clk = Clock()
    s = _store(tmp_path, clock=clk)
    s.insert("discord", "a")
    clk.t += 12
    st = s.stats()
    assert st["ok"] and st["events"] == {"received": 1}
    assert st["oldest_open_age_s"] == {"discord": 12.0}
