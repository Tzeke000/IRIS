"""V3 Phase 3 - task dependencies (`after: [task_ids]`) in brain/iris_events.py + scripts/server/worker.py.

All offline: tmp event store + tmp jobs DB, systemd-run is never called (admission is stubbed). Covers:
  * a dependant can't be claimed until EVERY parent is verified/committed (output_pending is not enough)
  * unknown parents / self-edges refused at creation; add_dependencies refuses an edge that closes a cycle
  * a cycle that exists anyway (hand-edited DB) is reported blocked and settled to failed, never loops
  * a failed / cancelled parent fails the dependant with a clear reason; requeue re-evaluates
  * the scheduler (worker.py run --wait-deps) killed mid-wait: a FRESH process + store sees the same
    state and resumes; --key makes a re-run reuse the task instead of duplicating it
  * worker.py sweep settles blocked dependants; dep-less tasks behave exactly as before
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "server" / "worker.py"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from brain import iris_events  # noqa: E402


@pytest.fixture(autouse=True)
def _real_print(monkeypatch):
    """iris_runtime (imported by other tests) swaps builtins.print for a stderr writer for the whole
    session (MCP stdio safety); these tests read the CLI's stdout, so restore the real print here."""
    import builtins
    rt = sys.modules.get("iris_runtime")
    if rt is not None and getattr(rt, "_real_print", None):
        monkeypatch.setattr(builtins, "print", rt._real_print)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture()
def store(tmp_path):
    s = iris_events.EventStore(tmp_path / "events.sqlite", synchronous="NORMAL")
    yield s
    s.close()


def _done(store, tid, *, verify=True):
    f = store.claim(tid, "t", 60)
    assert f is not None
    assert store.complete(tid, f, result_ref="r:" + tid)
    if verify:
        assert store.complete(tid, f, status="verified", verification_ref="act_" + tid)
    return f


# ── store ──────────────────────────────────────────────────────────────────────
def test_no_deps_is_unchanged(store):
    t = store.create_task("plain")
    assert store.dependency_state(t.id)["state"] == "ready"
    assert store.claim(t.id, "w", 30) == 1


def test_waits_until_every_parent_verified(store):
    a, b = store.create_task("a"), store.create_task("b")
    c = store.create_task("c", after=[a.id, b.id])
    st = store.dependency_state(c.id)
    assert st["state"] == "waiting" and {w["task_id"] for w in st["waiting_on"]} == {a.id, b.id}
    assert store.claim(c.id, "w", 30) is None
    _done(store, a.id)
    fb = store.claim(b.id, "t", 60)
    store.complete(b.id, fb, result_ref="x")            # output_pending: a result, not checked
    assert store.dependency_state(c.id)["state"] == "waiting"
    assert store.claim(c.id, "w", 30) is None
    store.complete(b.id, fb, status="verified", verification_ref="act_b")
    assert store.dependency_state(c.id)["state"] == "ready"
    assert store.claim(c.id, "w", 30) == 1


def test_committed_parent_counts_as_done(store):
    a = store.create_task("a")
    f = store.claim(a.id, "t", 60)
    assert store.complete(a.id, f, status="committed", result_ref="r")
    c = store.create_task("c", after=[a.id])
    assert store.claim(c.id, "w", 30) == 1


def test_unknown_parent_and_self_edge_refused(store):
    assert store.create_task("c", after=["task_nope"]) is None
    assert "unknown dependency" in store.last_error
    assert store.create_task("c", task_id="task_x", after=["task_x"]) is None
    assert store.get_task("task_x") is None              # the whole create rolled back


def test_add_dependencies_refuses_cycles(store):
    a = store.create_task("a")
    b = store.create_task("b", after=[a.id])
    c = store.create_task("c", after=[b.id])
    assert store.add_dependencies(a.id, [c.id]) is False   # a -> c -> b -> a
    assert "cycle" in store.last_error
    assert store.add_dependencies(a.id, [a.id]) is False
    d = store.create_task("d")
    assert store.add_dependencies(c.id, [d.id]) is True
    assert {x["task_id"] for x in store.dependency_state(c.id)["waiting_on"]} == {b.id, d.id}


def test_cycle_present_anyway_is_blocked_and_settled(store):
    a = store.create_task("a")
    b = store.create_task("b", after=[a.id])
    store._conn.execute("INSERT INTO task_deps VALUES(?,?,0)", (a.id, b.id))   # hand-made cycle
    st = store.dependency_state(a.id)
    assert st["state"] == "blocked" and "cycle" in st["reason"]
    assert store.claim(a.id, "w", 30) is None
    out = store.settle_dependencies()
    assert {f["task_id"] for f in out["failed"]} == {a.id, b.id}
    assert "cycle" in store.get_task(a.id)["note"]


@pytest.mark.parametrize("how", ["fail", "cancel"])
def test_failed_or_cancelled_parent_fails_dependant(store, how):
    a = store.create_task("a")
    c = store.create_task("c", after=[a.id])
    g = store.create_task("g", after=[c.id])                 # grandchild
    if how == "fail":
        f = store.claim(a.id, "t", 60)
        store.fail(a.id, f, note="exit 2")
    else:
        store.cancel_task(a.id, note="Zeke said stop")
    st = store.dependency_state(c.id)
    assert st["state"] == "blocked" and a.id in st["reason"]
    out = store.settle_dependencies()
    assert [x["task_id"] for x in out["failed"]] == [c.id, g.id]   # whole cascade, one pass
    row = store.get_task(c.id)
    assert row["status"] == "failed" and row["note"].startswith("not started: dependency " + a.id)
    assert c.id in store.get_task(g.id)["note"]
    assert store.settle_dependencies()["failed"] == []       # idempotent
    hist = [h["to_status"] for h in store.history(c.id)]
    assert hist[-1] == "failed"


def test_requeue_after_parent_recovers(store):
    a = store.create_task("a")
    c = store.create_task("c", after=[a.id])
    f = store.claim(a.id, "t", 60)
    store.fail(a.id, f, note="flaky")
    store.settle_dependencies()
    assert store.get_task(c.id)["status"] == "failed"
    assert store.requeue_task(a.id) and store.requeue_task(c.id)
    assert store.claim(c.id, "w", 30) is None                 # parent requeued, not done yet
    _done(store, a.id)
    assert store.claim(c.id, "w", 30) is not None


def test_state_survives_reopen(tmp_path):
    p = tmp_path / "events.sqlite"
    s1 = iris_events.EventStore(p)
    a = s1.create_task("a")
    c = s1.create_task("c", after=[a.id])
    s1.close()
    s2 = iris_events.EventStore(p)
    assert s2.dependency_state(c.id)["state"] == "waiting"
    _done(s2, a.id)
    assert s2.claim(c.id, "w", 30) == 1
    s2.close()


def test_v1_db_migrates_to_v2(tmp_path):
    """A live v1 store (no task_deps table) gains it on open; old rows untouched."""
    import sqlite3
    p = tmp_path / "events.sqlite"
    s = iris_events.EventStore(p)
    old = s.create_task("old")
    s.close()
    c = sqlite3.connect(p)
    c.execute("DROP TABLE task_deps")
    c.execute("PRAGMA user_version=1")
    c.commit()
    c.close()
    s = iris_events.EventStore(p)
    assert s.get_task(old.id)["status"] == "admitted"
    n = s.create_task("new", after=[old.id])
    assert n is not None and s.dependency_state(n.id)["state"] == "waiting"
    s.close()


# ── worker.py ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def worker(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_WORKER_DB", str(tmp_path / "jobs.sqlite"))
    w = _load(WORKER, "iris_worker_deps_test")
    st = iris_events.EventStore(tmp_path / "events.sqlite", synchronous="NORMAL")
    w._STORE = st
    launched = []
    monkeypatch.setattr(w, "admit_with_cancel", lambda a, mem: (launched.append(a.task_id) or True,
                                                                {"estimate": 1 << 30, "history_n": 0}))
    # stop right after admission: we're testing the gate, not systemd-run
    monkeypatch.setattr(w.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("no systemd")))
    w._launched = launched
    yield w
    st.close()


def _args(**kw):
    base = dict(kind="test", name="dep", mem="", swap="1G", timeout=60, nice=10, gpu_mib=0, cwd="",
                force=False, detach=False, lockdown="off", cmd=["true"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_worker_refuses_until_parent_verified(worker, capsys):
    st = worker._STORE
    a = st.create_task("a")
    rc = worker.run(_args(after=a.id))
    out = json.loads(capsys.readouterr().out)
    assert rc == worker.EX_TEMPFAIL and out["deps"] == "waiting" and worker._launched == []
    tid = out["task_id"]
    assert st.get_task(tid)["status"] == "admitted"           # the wait lives in the store
    _done(st, a.id)
    rc = worker.run(_args(task_id=tid))
    assert worker._launched == [tid]                           # admitted + claimed now
    assert st.get_task(tid)["status"] == "failed"              # our fake systemd-run "failed to launch"


def test_worker_dep_failed_exit_and_reason(worker, capsys):
    st = worker._STORE
    a = st.create_task("a")
    st.cancel_task(a.id, note="abandoned")
    rc = worker.run(_args(after=a.id))
    out = json.loads(capsys.readouterr().out)
    assert rc == worker.EX_DEPFAILED and out["deps"] == "blocked"
    assert "cancelled" in out["why"][0] and worker._launched == []
    assert st.get_task(out["task_id"])["status"] == "failed"


def test_worker_wait_deps_resolves_while_waiting(worker, capsys, monkeypatch):
    st = worker._STORE
    a = st.create_task("a")
    t = st.create_task("c", after=[a.id])
    ticks = {"n": 0}

    def fake_sleep(_s):
        ticks["n"] += 1
        if ticks["n"] == 2:
            _done(st, a.id)
    out = worker.wait_for_deps(st, t.id, 60, poll=0.01, sleep=fake_sleep)
    assert out["state"] == "ready" and ticks["n"] == 2


def test_worker_key_reuses_task_no_duplicate(worker, capsys):
    st = worker._STORE
    a = st.create_task("a")
    worker.run(_args(after=a.id, key="build:city"))
    worker.run(_args(after=a.id, key="build:city"))
    capsys.readouterr()
    rows = [t for t in st.open_tasks() if t["kind"] == "worker:test"]
    assert len(rows) == 1


def test_sweep_settles_blocked_dependants(worker):
    st = worker._STORE
    a = st.create_task("a")
    c = st.create_task("c", after=[a.id])
    f = st.claim(a.id, "t", 60)
    st.fail(a.id, f, note="boom")
    out = worker.sweep()
    assert [x["task_id"] for x in out["deps_failed"]] == [c.id]


def test_scheduler_killed_mid_wait_resumes(tmp_path):
    """A real `worker.py run --after A --wait-deps 60 --key K` subprocess is SIGKILLed while waiting.
    A second, fresh scheduler (same --key) finds the SAME task still admitted and waiting, and once A
    is verified it gets past the dependency gate (admission is then refused by the empty slice, exit 75
    with deps=ready - proving the gate opened without launching anything)."""
    db = tmp_path / "events.sqlite"
    env = {"IRIS_EVENTS_DB": str(db), "IRIS_WORKER_DB": str(tmp_path / "jobs.sqlite"),
           "IRIS_WORKER_DEP_POLL_S": "0.1", "IRIS_WORKER_CAP": "0", "PATH": "/usr/bin:/bin",
           "HOME": str(tmp_path)}
    s = iris_events.EventStore(db)
    a = s.create_task("parent")
    cmd = [sys.executable, str(WORKER), "run", "--kind", "t", "--name", "kid", "--after", a.id,
           "--key", "kid:1", "--wait-deps", "60", "--", "true"]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import time as _t
    deadline = _t.time() + 20
    kid = None
    while _t.time() < deadline and kid is None:
        kid = s.find_task_by_key("kid:1")
        _t.sleep(0.1)
    assert kid is not None, p.stderr.read() if p.poll() is not None else "task never created"
    _t.sleep(0.5)
    assert p.poll() is None                                    # still waiting
    p.kill()
    p.wait(10)
    assert s.get_task(kid["task_id"])["status"] == "admitted"
    assert s.dependency_state(kid["task_id"])["state"] == "waiting"
    _done(s, a.id)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
    out = json.loads(r.stdout)
    assert r.returncode == 75 and out["admitted"] is False and "deps" not in out, r.stdout + r.stderr
    assert [t["task_id"] for t in s.open_tasks() if t["kind"] == "worker:t"] == [kid["task_id"]]
    s.close()


def test_cli_task_new_deps_verify(tmp_path):
    db = tmp_path / "events.sqlite"
    env = {"IRIS_EVENTS_DB": str(db), "IRIS_WORKER_DB": str(tmp_path / "jobs.sqlite"),
           "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}

    def cli(*a):
        r = subprocess.run([sys.executable, str(WORKER), *a], env=env, capture_output=True, text=True,
                           timeout=60)
        return r.returncode, json.loads(r.stdout)
    rc, a = cli("task", "new", "--goal", "parent")
    assert rc == 0
    rc, c = cli("task", "new", "--goal", "child", "--after", a["task_id"])
    assert rc == 0
    rc, st = cli("deps", c["task_id"])
    assert rc == 75 and st["state"] == "waiting"
    rc, v = cli("verify", a["task_id"], "--ref", "act_1")
    assert rc == 1 and v["ok"] is False                        # admitted: no result to verify yet
    s = iris_events.EventStore(db)
    f = s.claim(a["task_id"], "t", 60)
    s.complete(a["task_id"], f, result_ref="r")
    s.close()
    rc, v = cli("verify", a["task_id"], "--ref", "act_1")
    assert rc == 0 and v["status"] == "verified"
    rc, st = cli("deps", c["task_id"])
    assert rc == 0 and st["state"] == "ready"
