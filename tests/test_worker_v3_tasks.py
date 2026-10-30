"""V3 Phase 1 wiring of scripts/server/worker.py (steps 1.3/1.4) + latency_report.py + task_checkpoint.

All offline: systemd-run is a fake Popen, the event store and the jobs DB are tmp files, nothing in
state/ is touched. Covers:
  * fence rejection through worker.py's own run()/watch() path (a stale fence can't write a result)
  * an explicit --task-id that isn't claimable is refused before systemd-run
  * a killed launcher -> lease expires -> sweep -> needs_review (P2-T4d)
  * priority cancel ordering, external never cancelled, nothing cancelled unless it fits (P2-D3)
  * GPU attribution parsing + gpu_mib recorded from samples (P2-R1c)
  * per-job CPUQuota/CPUWeight on the systemd-run line (P2-R4b)
  * task_checkpoint round trip (P1-H1)
  * latency report on a synthetic store, and on a missing/empty DB
"""
from __future__ import annotations

import importlib.util
import io
import itertools
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "server" / "worker.py"
LATENCY = ROOT / "scripts" / "server" / "latency_report.py"
GiB = 1024 ** 3
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from brain import iris_events  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class FakeProc:
    """Stands in for `systemd-run --wait`: polls None `n` times (calling on_poll each time), then exits."""

    def __init__(self, n=3, result="success", status=0, on_poll=None):
        self.n, self.on_poll, self.returncode = n, on_poll, None
        self.stderr = io.StringIO(f"Running as unit: x\nFinished with result: {result}\n"
                                  f"Main processes terminated with: code=exited/status={status}\n"
                                  "CPU time consumed: 12ms\n")
        self.polls = 0

    def poll(self):
        if self.polls >= self.n:
            self.returncode = 0
            return 0
        self.polls += 1
        if self.on_poll:
            self.on_poll(self.polls)
        return None


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("IRIS_WORKER_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("IRIS_EVENTS_DB", str(tmp_path / "events.sqlite"))
    w = _load(WORKER, "iris_worker_v3_under_test")
    store = iris_events.EventStore(tmp_path / "events.sqlite")
    monkeypatch.setattr(w, "_STORE", store)
    monkeypatch.setattr(w, "running_workers", lambda: [])
    monkeypatch.setattr(w, "meminfo_available", lambda: 30 * GiB)
    monkeypatch.setattr(w, "slice_dir", lambda: tmp_path / "slice")
    monkeypatch.setattr(w, "unit_cgroup", lambda unit: None)
    monkeypatch.setattr(w, "gpu_by_unit", lambda: None)
    monkeypatch.setattr(w.time, "sleep", lambda s: None)
    (tmp_path / "slice").mkdir()
    (tmp_path / "slice" / "memory.max").write_text(str(20 * GiB))
    (tmp_path / "slice" / "memory.current").write_text(str(1 * GiB))
    calls = {"popen": [], "stop": []}
    monkeypatch.setattr(w, "stop_unit", lambda u: calls["stop"].append(u))

    class Ctx:
        pass
    c = Ctx()
    c.w, c.store, c.calls, c.tmp = w, store, calls, tmp_path

    def set_proc(factory):
        def popen(cmd, **kw):
            calls["popen"].append(cmd)
            return factory()
        monkeypatch.setattr(w.subprocess, "Popen", popen)
    c.set_proc = set_proc

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["worker.py", *argv])
        return w.main()
    c.main = main
    yield c
    store.close()


def _job(w, jid=None):
    with w.db() as c:
        c.row_factory = sqlite3.Row
        q = "SELECT * FROM jobs ORDER BY id DESC LIMIT 1" if jid is None else "SELECT * FROM jobs WHERE id=?"
        return dict(c.execute(q, () if jid is None else (jid,)).fetchone())


# ── fencing through worker.py ────────────────────────────────────────────────
def test_auto_task_claimed_before_systemd_run_and_completed_with_fence(env):
    seen = {}

    def factory():
        # at systemd-run time the task must ALREADY be claimed by this launcher
        t = [x for x in env.store.open_tasks() if x["kind"] == "worker:build"][0]
        seen.update(status=t["status"], fence=t["fence"], owner=t["owner"])
        return FakeProc()
    env.set_proc(factory)
    rc = env.main("run", "--kind", "build", "--name", "b1", "--mem", "1G", "--", "true")
    assert rc == 0
    assert seen["status"] == "running" and seen["fence"] == 1 and seen["owner"].startswith("worker:iris-worker-b1")
    j = _job(env.w)
    t = env.store.get_task(j["task_id"])
    assert j["fence"] == 1 and j["priority"] == 5 and j["side_effects"] == "none"
    assert t["status"] == "output_pending" and t["result_ref"] == f"job:{j['id']}"
    cmd = env.calls["popen"][0]
    assert "CPUQuota=1200%" in cmd and "CPUWeight=100" in cmd, "per-job CPU cap on the systemd-run line"
    assert f"--setenv=IRIS_TASK_ID={j['task_id']}" in cmd


def test_stale_fence_result_is_rejected_through_watch(env):
    tid = env.store.create_task("render chunk 3", kind="worker:render").id

    def steal(n):
        if n == 1:  # the launcher's lease runs out; someone sweeps, requeues and re-claims (fence 2)
            assert env.store.sweep_expired(now=time.time() + 10_000)["tasks_needs_review"] == 1
            assert env.store.requeue_task(tid)
            assert env.store.claim(tid, "other-worker") == 2
    env.set_proc(lambda: FakeProc(n=2, on_poll=steal))
    rc = env.main("run", "--kind", "render", "--name", "r3", "--task-id", tid, "--", "true")
    assert rc == env.w.EX_REJECTED
    t = env.store.get_task(tid)
    assert t["status"] == "running" and t["fence"] == 2 and t["owner"] == "other-worker"
    assert t["result_ref"] is None, "the stale worker must not have written its result"
    j = _job(env.w)
    assert j["fence"] == 1 and j["outcome"] == "ok", "the job's own measurements are still recorded"
    assert "REJECTED" in j["note"] and "fence 1 is stale" in j["note"]


def test_explicit_task_that_is_not_claimable_is_refused_before_systemd_run(env):
    tid = env.store.create_task("x", kind="worker:build").id
    assert env.store.claim(tid, "someone-else") == 1
    env.set_proc(lambda: FakeProc())
    assert env.main("run", "--kind", "build", "--name", "b", "--task-id", tid, "--", "true") == env.w.EX_NOPERM
    assert env.main("run", "--kind", "build", "--name", "b", "--task-id", "task_nope", "--", "true") \
        == env.w.EX_NOPERM
    assert env.calls["popen"] == []


def test_failed_job_fails_the_task_with_its_fence(env):
    env.set_proc(lambda: FakeProc(result="exit-code", status=2))
    assert env.main("run", "--kind", "build", "--name", "bad", "--", "false") == 1
    j = _job(env.w)
    t = env.store.get_task(j["task_id"])
    assert j["outcome"] == "exit-code" and t["status"] == "failed" and "exit 2" in t["note"]


def test_lost_claim_stops_the_unit_but_never_an_external_one(env, monkeypatch):
    monkeypatch.setattr(env.w.time, "monotonic", itertools.count(0, 100).__next__)  # every poll = 100 s

    def cancel_midway(tid):
        def on_poll(n):
            if n == 1:
                env.store.cancel_task(tid, note="test")
        return on_poll
    for fx, expect_stop in (("none", True), ("external", False)):
        env.calls["stop"].clear()
        tid = env.store.create_task("t", kind="worker:x", side_effects=fx).id
        env.set_proc(lambda: FakeProc(n=3, on_poll=cancel_midway(tid)))
        env.main("run", "--kind", "x", "--name", f"lc_{fx}", "--task-id", tid, "--lease", "8", "--", "true")
        j = _job(env.w)
        assert "lost the claim" in j["note"]
        assert bool(env.calls["stop"]) is expect_stop, fx
        if not expect_stop:
            assert "reconcile by hand" in j["note"]


def test_killed_launcher_lease_expires_to_needs_review_via_sweep(env, monkeypatch):
    """P2-T4d: the launcher dies after claiming (simulated: claim + jobs row, then nothing). The lease
    expires, `worker.py sweep` turns the task into needs_review and the orphaned row into launcher_lost."""
    # SIGKILL can't be caught, so build exactly the state a launcher leaves when it dies between
    # claim and outcome: a running claim + a `running` jobs row, then silence (no heartbeat, no result).
    tid = env.store.create_task("orphan", kind="worker:render").id
    fence = env.store.claim(tid, "worker:iris-worker-orphan-1.service", lease_s=30)
    with env.w.db() as c:
        c.execute("INSERT INTO jobs(unit,name,kind,start_ts,outcome,task_id,fence) VALUES(?,?,?,?,?,?,?)",
                  ("iris-worker-orphan-1.service", "orphan", "render", time.time() - 120, "running", tid, fence))
    # the unit outlives its launcher (it keeps its own limits) ...
    monkeypatch.setattr(env.w, "running_workers", lambda: ["iris-worker-orphan-1.service"])
    out = env.w.sweep()
    assert out["store"]["tasks_needs_review"] == 0 and out["launcher_lost"] == [], "lease live, unit live"
    # ... the lease runs out (no heartbeats) -> needs_review, never silently running
    real_now = time.time
    monkeypatch.setattr(env.store, "_clock", lambda: real_now() + 60)
    out = env.w.sweep()
    assert out["store"]["tasks_needs_review"] == 1 and out["launcher_lost"] == []
    assert env.store.get_task(tid)["status"] == "needs_review"
    # ... and once the unit itself is gone, the orphaned jobs row is closed as launcher_lost
    monkeypatch.setattr(env.w, "running_workers", lambda: [])
    out = env.w.sweep()
    assert out["launcher_lost"] == ["iris-worker-orphan-1.service"]
    assert _job(env.w)["outcome"] == "launcher_lost"


# ── priority cancel (P2-D3) ──────────────────────────────────────────────────
def _cand(unit, prio, fx="none", mem=4 * GiB, start=0.0, gpu=0, job=None):
    return {"unit": unit, "priority": prio, "side_effects": fx, "mem": mem, "start_ts": start,
            "gpu_mib": gpu, "job": job, "task_id": None, "name": unit}


def _info(running=3, avail=30 * GiB, est=2 * GiB, sl_cur=1 * GiB, gfree=None, gneed=0):
    return {"estimate": est, "running": ["u"] * running, "mem_available": avail, "slice_max": 20 * GiB,
            "slice_current": sl_cur, "gpu_free_mib": gfree, "gpu_need_mib": gneed}


def test_cancel_order_least_urgent_first_then_newest(env):
    cands = [_cand("p6_old", 6, start=100), _cand("p9", 9, start=50), _cand("p6_new", 6, start=200),
             _cand("p3", 3, start=10)]
    # cap reached (3 of 3): exactly ONE must go -> the least urgent (p9)
    plan = env.w.plan_cancellations(_info(running=3), 5, cands)
    assert plan["fits"] and [c["unit"] for c in plan["cancel"]] == ["p9"]
    # memory short by ~2 jobs' worth: p9 first, then the NEWER p6
    plan = env.w.plan_cancellations(_info(running=1, avail=6 * GiB + 2 * GiB - 7 * GiB), 5, cands)
    assert [c["unit"] for c in plan["cancel"]] == ["p9", "p6_new"]
    assert {"unit": "p3", "why": "priority 3 is not lower than 5"} in plan["skipped"]


def test_external_and_unknown_jobs_are_never_cancelled(env):
    cands = [_cand("ext", 9, fx="external"), _cand("legacy", None), _cand("same", 5)]
    plan = env.w.plan_cancellations(_info(running=3), 5, cands)
    assert not plan["fits"] and plan["cancel"] == [], "nothing eligible -> nothing is touched"
    why = {s["unit"]: s["why"] for s in plan["skipped"]}
    assert "external" in why["ext"] and "unknown" in why["legacy"] and "not lower" in why["same"]
    # a partial plan that still wouldn't fit cancels NOTHING
    cands = [_cand("small", 9, mem=1 * GiB), _cand("ext", 9, fx="external", mem=20 * GiB)]
    plan = env.w.plan_cancellations(_info(running=1, avail=2 * GiB), 5, cands)
    assert not plan["fits"] and plan["cancel"] == []


def test_cancel_lower_end_to_end_logs_and_cancels_tasks(env, monkeypatch):
    t_low = env.store.create_task("low", kind="worker:render").id
    f_low = env.store.claim(t_low, "worker:low")
    t_ext = env.store.create_task("push", kind="worker:git", side_effects="external").id
    env.store.claim(t_ext, "worker:ext")
    with env.w.db() as c:
        j_low = c.execute("INSERT INTO jobs(unit,name,kind,start_ts,outcome,task_id,fence,priority,side_effects)"
                          " VALUES(?,?,?,?,?,?,?,?,?)", ("iris-worker-low.service", "low", "render", 1.0, "running",
                                                         t_low, f_low, 8, "none")).lastrowid
        c.execute("INSERT INTO jobs(unit,name,kind,start_ts,outcome,task_id,fence,priority,side_effects)"
                  " VALUES(?,?,?,?,?,?,?,?,?)", ("iris-worker-ext.service", "ext", "git", 2.0, "running",
                                                 t_ext, 1, 9, "external"))
        c.execute("INSERT INTO jobs(unit,name,kind,start_ts,outcome) VALUES(?,?,?,?,?)",
                  ("iris-worker-old.service", "old", "x", 3.0, "running"))
    live = ["iris-worker-low.service", "iris-worker-ext.service", "iris-worker-old.service"]
    monkeypatch.setattr(env.w, "running_workers", lambda: list(live))
    monkeypatch.setattr(env.w, "stop_unit", lambda u: (env.calls["stop"].append(u), live.remove(u)))
    env.set_proc(lambda: FakeProc())
    # without --cancel-lower: refused, nothing stopped
    assert env.main("run", "--kind", "build", "--name", "urgent", "--priority", "1", "--", "true") == 75
    assert env.calls["stop"] == []
    rc = env.main("run", "--kind", "build", "--name", "urgent", "--priority", "1", "--cancel-lower", "--", "true")
    assert rc == 0
    assert env.calls["stop"] == ["iris-worker-low.service"], "only the cancellable lower-priority job"
    assert env.store.get_task(t_low)["status"] == "cancelled"
    assert env.store.get_task(t_ext)["status"] == "running"
    assert _job(env.w, j_low)["outcome"] == "cancelled"
    log = [json.loads(x) for x in (env.tmp / "cancel_log.jsonl").read_text().splitlines()]
    assert len(log) == 1 and log[0]["cancelled_unit"] == "iris-worker-low.service"
    assert log[0]["by"]["name"] == "urgent" and log[0]["by"]["priority"] == 1


def test_watch_keeps_cancelled_outcome(env):
    with env.w.db() as c:
        jid = c.execute("INSERT INTO jobs(unit,name,kind,start_ts,outcome) VALUES('u','n','k',1,'cancelled')"
                        ).lastrowid
    env.w.watch(jid, "u", FakeProc(n=0, result="signal"))
    assert _job(env.w, jid)["outcome"] == "cancelled"


# ── GPU attribution (P2-R1c) ─────────────────────────────────────────────────
def test_parse_compute_apps_maps_pids_to_units(env):
    units = {11: "iris-worker-a-1.service", 12: "iris-worker-a-1.service", 13: "iris-mouth.service", 14: ""}
    text = "11, 500\n12, 250\n13, 4096\n14, 100\n15, [N/A]\ngarbage\n"
    assert env.w.parse_compute_apps(text, lambda p: units.get(p, "")) == {
        "iris-worker-a-1.service": 750, "iris-mouth.service": 4096}


def test_gpu_attribution_reuses_health_snapshot_unit_of_pid(env):
    hs = env.w._health_snapshot()
    assert hs is not None and callable(hs.unit_of_pid)
    assert not hasattr(env.w, "unit_of_pid"), "no duplicated pid->unit code in worker.py"


def test_watch_records_peak_gpu_mib(env, monkeypatch):
    samples = iter([{"u1": 700}, {"u1": 900, "other": 5000}, {}])
    monkeypatch.setattr(env.w, "gpu_by_unit", lambda: next(samples, {}))
    monkeypatch.setattr(env.w, "GPU_SAMPLE_S", 0.0)
    with env.w.db() as c:
        jid = c.execute("INSERT INTO jobs(unit,name,kind,start_ts,outcome) VALUES('u1','n','k',1,'running')"
                        ).lastrowid
    env.w.watch(jid, "u1", FakeProc(n=3))
    assert _job(env.w, jid)["gpu_mib"] == 900


def test_slice_has_a_cpu_ceiling():
    txt = (ROOT / "scripts" / "server" / "systemd" / "iris-workers.slice").read_text()
    assert "CPUQuota=2400%" in txt and "CPUWeight=20" in txt
    drop = ROOT / "scripts" / "server" / "systemd" / "system" / "user@.service.d" / "iris-delegate.conf"
    assert "Delegate=cpu" in drop.read_text()


# ── task_checkpoint (P1-H1) ──────────────────────────────────────────────────
def test_task_checkpoint_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("IRIS_EVENTS_DB", str(tmp_path / "cp.sqlite"))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import iris_runtime
    w = iris_runtime.task_checkpoint(goal="ship V3 wiring", steps_done=["worker.py"], outstanding=["tests"],
                                     files=["scripts/server/worker.py"], running_jobs=["iris-worker-x"],
                                     locks=["scripts/server/*"], next_safe_action="run pytest")
    assert w["ok"] and w["created"], w
    tid = w["task_id"]
    r = iris_runtime.task_checkpoint(mode="read", task_id=tid)
    cp = r["checkpoint"]
    assert cp["goal"] == "ship V3 wiring" and cp["locks"] == ["scripts/server/*"]
    assert cp["next_safe_action"] == "run pytest" and "_ts" in cp
    w2 = iris_runtime.task_checkpoint(goal="ship V3 wiring", steps_done=["worker.py", "tests"],
                                      next_safe_action="commit", task_id=tid)
    assert w2["ok"] and not w2["created"]
    latest = iris_runtime.task_checkpoint(mode="read")["checkpoints"]
    assert latest[0]["task_id"] == tid and latest[0]["checkpoint"]["steps_done"] == ["worker.py", "tests"]
    assert not iris_runtime.task_checkpoint(goal="x", task_id="task_missing")["ok"]
    names = {t.name for t in iris_runtime.mcp._tool_manager.list_tools()}
    assert "task_checkpoint" in names


# ── latency report (step 1.4) ────────────────────────────────────────────────
@pytest.fixture()
def utc(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_latency_report_on_synthetic_store(tmp_path, utc):
    lr = _load(LATENCY, "iris_latency_report_under_test")
    now = [1_800_014_400.0]  # 12:00 UTC: the whole synthetic hour sits inside one day
    st = iris_events.EventStore(tmp_path / "ev.sqlite", clock=lambda: now[0])
    base = now[0]
    # 10 discord events: admitted at +0.1, assigned at +(1..10), committed 2 s later
    for i in range(10):
        now[0] = base + i * 100
        r = st.insert("discord", f"m{i}", received_ts=now[0] - 0.5, status="received",
                      meta={"tool_blocked_s": float(i)} if i < 4 else None)
        now[0] += 0.1
        st.transition(r.id, "received", None, "admitted")
        now[0] += i + 1
        st.transition(r.id, "admitted", None, "assigned")
        now[0] += 2
        st.mark_replied(r.id, f"msg{i}")
    # one voice event still open; one store incident (excluded)
    now[0] = base + 2000
    st.insert("voice", "hello", status="admitted")
    st.insert("store", "incident")
    # worker task: queued 3 s, ran 10 s
    t = st.create_task("job", kind="worker:render").id
    now[0] += 3
    f = st.claim(t, "w")
    now[0] += 10
    st.complete(t, f, "job:1")
    rep = lr.build(st._conn, days=1, now=base + 3000)
    days = rep["events"]
    (day,) = days.keys()
    d = days[day]["discord"]
    assert d["events"] == 10 and d["open"] == 0
    s = d["stages"]
    assert s["received->admitted"]["p50"] == pytest.approx(0.6)
    assert s["admitted->assigned"]["n"] == 10
    assert s["admitted->assigned"]["p50"] in (pytest.approx(5.0), pytest.approx(6.0))
    assert s["admitted->assigned"]["p90"] == pytest.approx(9.0)
    assert s["assigned->committed"]["p50"] == pytest.approx(2.0)
    assert s["reply"]["p90"] == pytest.approx(0.6 + 9 + 2)
    assert s["tool_blocked"]["n"] == 4 and rep["tool_blocked_recorded"]
    assert "store" not in days[day]
    v = days[day]["voice"]
    assert v["open"] == 1 and v["stages"]["reply"]["n"] == 0
    task = rep["tasks"][day]["worker:render"]
    assert task["queue"]["p50"] == pytest.approx(3.0) and task["run"]["p50"] == pytest.approx(10.0)
    assert "discord" in lr.render_text(dict(rep, db="x"))
    st.close()


def test_latency_report_no_data(tmp_path, capsys):
    lr = _load(LATENCY, "iris_latency_report_under_test2")

    def said():  # importing iris_runtime (another test) reroutes print() to stderr process-wide
        r = capsys.readouterr()
        return r.out + r.err
    assert lr.main(["--db", str(tmp_path / "missing.sqlite")]) == 0
    assert "no data" in said()
    (tmp_path / "empty.sqlite").write_bytes(b"")
    assert lr.main(["--db", str(tmp_path / "empty.sqlite"), "--json"]) == 0
    assert json.loads(said())["no_data"] is True
    st = iris_events.EventStore(tmp_path / "fresh.sqlite")  # schema, zero rows
    st.close()
    assert lr.main(["--db", str(tmp_path / "fresh.sqlite"), "--days", "3"]) == 0
    assert "no data" in said()
