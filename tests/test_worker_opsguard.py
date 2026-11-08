"""opsguard 11-08: the worker.py mechanisms for the LUMEN ship night's recurring operational fails.

  1. completion notify is the DEFAULT (job_done with the output tail; helpers excepted; --no-notify opts out)
  2. --after waits for the parent to FINISH OK (not 'verified'); --detach queues via a supervisor unit
     instead of check-once-and-exit; a failed parent => the dependant never runs, loudly
  3. sweep/status reconcile rows whose unit is gone from the journal ('detached' rows included)
 11. eta(): history-based p50/p90 per kind + name pattern

All offline: systemd-run is faked, DBs are tmp files, the LLM bridge is a list.
"""
from __future__ import annotations

import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_worker_v3_tasks import FakeProc, _job, env  # noqa: E402,F401  (fixture re-export)


@pytest.fixture()
def sent(env, monkeypatch):
    box = []
    monkeypatch.setattr(env.w, "_submit_llm", lambda prompt, **kw: box.append((prompt, kw)))
    return box


# ── 1. notify by default ─────────────────────────────────────────────────────
def test_job_end_notifies_by_default_with_output_tail(env, sent):
    def factory():
        cmd = env.calls["popen"][-1]
        out = Path(cmd[cmd.index("worker-tee") + 1])
        out.write_text("line1\nRESULT: 3 red suites\n")
        return FakeProc(result="exit-code", status=1)
    env.set_proc(factory)
    rc = env.main("run", "--kind", "job", "--name", "land_x", "--", "bash", "vr.sh", "land", "b")
    assert rc == 1
    assert len(sent) == 1
    prompt, kw = sent[0]
    assert kw["kind"] == "job_done" and kw["requester"] == "job:land_x"
    assert "outcome=exit-code" in prompt and "RESULT: 3 red suites" in prompt
    cmd = env.calls["popen"][0]
    assert cmd[cmd.index("--") + 1:cmd.index("--") + 3] == ["bash", "-c"], "output is tee'd to a file"
    assert "pipefail" in cmd[cmd.index("--") + 3]


def test_ok_job_notifies_too_and_caller_redirect_file_is_tailed_first(env, sent, tmp_path):
    caller = tmp_path / "land.txt"
    caller.write_text("caller's own log\nLAND GREEN\n")
    env.set_proc(lambda: FakeProc())
    env.main("run", "--kind", "job", "--name", "j", "--", "bash", "-c", f"vr.sh land > {caller} 2>&1")
    assert len(sent) == 1 and "LAND GREEN" in sent[0][0] and str(caller) in sent[0][0]


def test_no_notify_and_helpers_are_silent(env, sent):
    env.set_proc(lambda: FakeProc())
    env.main("run", "--no-notify", "--kind", "job", "--name", "q", "--", "true")
    env.main("run", "--kind", "helper", "--name", "helper-x", "--", "true")
    env.main("run", "--kind", "job", "--name", "w", "--", "bash", "scripts/server/job_notify.sh", "w", "o", "--", "true")
    assert sent == [], "helpers + job_notify.sh-wrapped commands wake me themselves"
    assert "worker-tee" not in env.calls["popen"][0] and "worker-tee" not in env.calls["popen"][1]


def test_caller_out_files_parsing(env):
    f = env.w.caller_out_files
    assert f(["bash", "-c", "x > /a/b.txt 2>&1"]) == ["/a/b.txt"]
    assert f(["bash", "-c", "x 2>&1 | tee -a out.log"]) == ["out.log"]
    assert f(["bash", "-c", "x >/dev/null"]) == []


# ── 2. --after ────────────────────────────────────────────────────────────────
def _parent_done(env, status="output_pending"):
    tid = env.store.create_task("parent", kind="worker:job").id
    fence = env.store.claim(tid, "w")
    if status == "failed":
        env.store.fail(tid, fence, note="red")
    elif status != "running":
        env.store.complete(tid, fence, result_ref="job:1")
    return tid


def test_after_starts_when_parent_merely_finished_ok_not_verified(env, sent):
    """The 11-07 bug: a worker job's own ok only reaches output_pending, and --after demanded verified."""
    parent = _parent_done(env)
    env.set_proc(lambda: FakeProc())
    rc = env.main("run", "--kind", "job", "--name", "child", "--after", parent, "--", "true")
    assert rc == 0 and len(env.calls["popen"]) == 1


def test_after_verified_keeps_the_strict_semantics(env, sent, capsys):
    parent = _parent_done(env)
    env.set_proc(lambda: FakeProc())
    rc = env.main("run", "--kind", "job", "--name", "child", "--after-verified", "--after", parent, "--", "true")
    assert rc == env.w.EX_TEMPFAIL and env.calls["popen"] == []
    assert "NOT STARTED" in capsys.readouterr().err


def test_after_parent_running_without_detach_refuses_loudly(env, sent, capsys):
    parent = _parent_done(env, "running")
    env.set_proc(lambda: FakeProc())
    rc = env.main("run", "--kind", "job", "--name", "child", "--after", parent, "--", "true")
    err = capsys.readouterr().err
    assert rc == env.w.EX_TEMPFAIL and "nothing will start it later" in err and "--detach" in err


def test_after_failed_parent_cancels_the_child_task(env, sent):
    parent = _parent_done(env, "failed")
    env.set_proc(lambda: FakeProc())
    rc = env.main("run", "--kind", "job", "--name", "child", "--after", parent, "--wait-deps", "5", "--", "true")
    assert rc == env.w.EX_DEPFAILED and env.calls["popen"] == []
    st = env.store._conn.execute("SELECT status FROM tasks WHERE goal='job: child'").fetchall()
    assert [r[0] for r in st] == ["cancelled"], "the child's task says it will never run"
    assert sent and "dep_failed" in sent[0][0], "a queued job that will never run wakes me"


def test_after_waits_then_starts_when_parent_finishes(env, sent, monkeypatch):
    parent = _parent_done(env, "running")
    polls = {"n": 0}
    orig = env.w.parents_done_state

    def finishing(store, parents):
        polls["n"] += 1
        if polls["n"] == 3:
            t = store.get_task(parent)
            store.complete(parent, t["fence"], result_ref="job:9")
        return orig(store, parents)
    monkeypatch.setattr(env.w, "parents_done_state", finishing)
    env.set_proc(lambda: FakeProc())
    rc = env.main("run", "--kind", "job", "--name", "child", "--after", parent, "--wait-deps", "600", "--", "true")
    assert rc == 0 and polls["n"] >= 3 and len(env.calls["popen"]) == 1


def test_after_accepts_job_ids_and_names(env, sent):
    parent = _parent_done(env)
    with env.w.db() as c:
        jid = c.execute("INSERT INTO jobs(unit,name,kind,start_ts,end_ts,outcome,task_id) VALUES(?,?,?,?,?,?,?)",
                        ("u", "integ4_final", "job", 1, 2, "ok", parent)).lastrowid
    env.set_proc(lambda: FakeProc())
    assert env.main("run", "--kind", "job", "--name", "c1", "--after", f"job:{jid}", "--", "true") == 0
    assert env.main("run", "--kind", "job", "--name", "c2", "--after", "name:integ4_final", "--", "true") == 0
    assert env.main("run", "--kind", "job", "--name", "c3", "--after", "job:99999", "--", "true") == env.w.EX_NOPERM


def test_detach_with_unfinished_parent_queues_a_supervisor(env, sent, monkeypatch):
    parent = _parent_done(env, "running")
    runs = []
    monkeypatch.setattr(env.w.subprocess, "run",
                        lambda cmd, **kw: runs.append(cmd) or type("R", (), {"returncode": 0, "stderr": ""})())
    env.set_proc(lambda: pytest.fail("the job itself must not start yet"))
    rc = env.main("run", "--kind", "job", "--name", "rerun", "--after", parent, "--detach", "--timeout", "100",
                  "--", "bash", "vr.sh", "rerun")
    assert rc == 0 and len(runs) == 1
    cmd = runs[0]
    assert any(x.startswith("--unit=iris-wsup-rerun-") for x in cmd)
    assert not any("--slice=" in x for x in cmd), "supervisors are not workers (no admission slot)"
    inner = cmd[cmd.index("run") + 1:]
    assert "--detach" not in inner and "--after" in inner and "--task-id" in inner and "--wait-deps" in inner
    assert "--wait-admit" in inner
    assert inner[inner.index("--") + 1:] == ["bash", "vr.sh", "rerun"]
    tid = inner[inner.index("--task-id") + 1]
    assert env.store.get_task(tid)["status"] == "admitted"


def test_detach_without_deps_also_goes_through_a_supervisor(env, sent, monkeypatch):
    runs = []
    monkeypatch.setattr(env.w.subprocess, "run",
                        lambda cmd, **kw: runs.append(cmd) or type("R", (), {"returncode": 0, "stderr": ""})())
    env.set_proc(lambda: pytest.fail("no direct launch"))
    assert env.main("run", "--kind", "job", "--name", "deploy", "--detach", "--", "true") == 0
    assert len(runs) == 1 and "--task-id" not in runs[0]


def test_supervisor_argv_strips_and_adds(env):
    out = env.w.supervisor_argv(["--kind", "j", "--detach", "--wait-deps=5", "--task-id", "old", "--name", "n",
                                 "--", "echo", "--detach"], "T1", 99)
    assert out[-3:] == ["--", "echo", "--detach"], "the command's own args are untouched"
    head = out[:-3]
    assert "--detach" not in head and "old" not in head and head[head.index("--task-id") + 1] == "T1"


def test_supervisor_retries_admission(env, sent, monkeypatch):
    seq = iter([(False, {"why": ["busy"], "estimate": 1, "history_n": 0})] * 2 +
               [(True, {"estimate": 1 << 30, "history_n": 0})])
    monkeypatch.setattr(env.w, "admit_with_cancel", lambda a, mem: next(seq))
    monkeypatch.setattr(env.w, "ADMIT_POLL_S", 0)
    env.set_proc(lambda: FakeProc())
    assert env.main("run", "--kind", "job", "--name", "q", "--wait-admit", "60", "--", "true") == 0


# ── 3. reconcile ──────────────────────────────────────────────────────────────
JOURNAL_FAIL = ("Started x.\nu.service: Main process exited, code=exited, status=1/FAILURE\n"
                "u.service: Failed with result 'exit-code'.\nu.service: Consumed 17.0s CPU time.\n")
JOURNAL_OK = "Started x.\nsome output\nu.service: Consumed 28min 12.018s CPU time.\n"


def test_parse_journal_exit(env):
    p = env.w.parse_journal_exit
    assert p(JOURNAL_FAIL) == ("exit-code", 1)
    assert p(JOURNAL_OK) == ("ok", 0), "systemd 249 logs only 'Consumed' for a clean exit"
    assert p("u.service: Failed with result 'timeout'.\n")[0] == "timeout"
    assert p("") is None


def test_sweep_reconciles_detached_rows_from_the_journal(env, monkeypatch):
    old = time.time() - 3600
    with env.w.db() as c:
        for unit, out in (("iris-worker-a-1.service", "detached"), ("iris-worker-b-1.service", "detached"),
                          ("iris-worker-c-1.service", "detached"), ("iris-worker-live-1.service", "detached")):
            c.execute("INSERT INTO jobs(unit,name,kind,start_ts,outcome) VALUES(?,?,?,?,?)", (unit, unit, "job", old, out))
    j = {"iris-worker-a-1.service": JOURNAL_FAIL, "iris-worker-b-1.service": JOURNAL_OK}
    monkeypatch.setattr(env.w, "journal_exit", lambda u: env.w.parse_journal_exit(j.get(u, "")))
    monkeypatch.setattr(env.w, "running_workers", lambda: ["iris-worker-live-1.service"])
    out = env.w.sweep()
    with env.w.db() as c:
        got = dict(c.execute("SELECT unit, outcome FROM jobs").fetchall())
    assert got["iris-worker-a-1.service"] == "exit-code" and got["iris-worker-b-1.service"] == "ok"
    assert got["iris-worker-c-1.service"] == "lost" and got["iris-worker-live-1.service"] == "detached"
    assert len(out["reconciled"]) == 2 and out["launcher_lost"] == ["iris-worker-c-1.service"]
    with env.w.db() as c:
        assert c.execute("SELECT count(*) FROM jobs WHERE end_ts IS NULL").fetchone()[0] == 1


# ── 11. eta ───────────────────────────────────────────────────────────────────
def test_eta_from_history(env):
    with env.w.db() as c:
        for i, mins in enumerate((40, 50, 60, 70, 200)):
            c.execute("INSERT INTO jobs(unit,name,kind,start_ts,end_ts,outcome) VALUES(?,?,?,?,?,?)",
                      (f"u{i}", f"lumen_deploy{i}", "job", 0, mins * 60, "ok" if i % 2 else "exit-code"))
        c.execute("INSERT INTO jobs(unit,name,kind,start_ts,end_ts,outcome) VALUES(?,?,?,?,?,?)",
                  ("ux", "land_integ4", "job", 0, 9000, "ok"))
    e = env.w.eta("job", "lumen_deploy6")
    assert e["n"] == 2 and e["like"] == "lumen_deploy%" and e["p50_min"] == 70, "OK runs only (2 of 5)"
    assert "min" in e["say"]
    assert env.w.eta("render", "x")["n"] == 0
    assert env.w._stem("land_integ4b") == "land_integ"


def test_hello_line_carries_eta_notify_and_out(env, sent, capsys):
    env.set_proc(lambda: FakeProc())
    env.main("run", "--kind", "job", "--name", "z", "--", "true")
    hello = json.loads(capsys.readouterr().out.splitlines()[0])
    assert hello["notify"] is True and hello["out"].endswith(".log") and "eta" in hello
