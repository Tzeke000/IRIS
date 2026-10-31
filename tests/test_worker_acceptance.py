"""Acceptance tests - Adaptive Worker Management brief (Project 2), scripts/server/worker.py.

Unit part (always runs, no systemd, tmp DB): admission is measurement-based and refuses launches
that exceed the cap, the core RAM reserve, the slice headroom or the GPU voice reserve; a
4th worker is admitted only when the cap is raised AND the forecast fits.

Live part (opt-in: IRIS_LIVE_WORKER_TESTS=1): real transient units in iris-workers.slice -
three concurrent workers whose recorded peaks are checked against the kernel's own VmHWM, and
one worker pushed past its MemoryMax that must die ALONE (oom outcome) while the core services
keep the same MainPID. Small (~100-300 MB) on purpose. Uses a tmp job DB, never state/.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "server" / "worker.py"
GiB = 1024 ** 3


@pytest.fixture()
def w(monkeypatch, tmp_path):
    monkeypatch.setenv("IRIS_WORKER_DB", str(tmp_path / "jobs.sqlite"))
    spec = importlib.util.spec_from_file_location("iris_worker_under_test", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    # a roomy, quiet machine unless a test says otherwise
    monkeypatch.setattr(mod, "running_workers", lambda: [])
    monkeypatch.setattr(mod, "meminfo_available", lambda: 30 * GiB)
    monkeypatch.setattr(mod, "slice_dir", lambda: tmp_path / "slice")
    (tmp_path / "slice").mkdir()
    (tmp_path / "slice" / "memory.max").write_text(str(20 * GiB))
    (tmp_path / "slice" / "memory.current").write_text(str(1 * GiB))
    return mod


def _record(mod, kind, peaks):
    with mod.db() as c:
        for p in peaks:
            c.execute("INSERT INTO jobs(kind,name,outcome,peak_mem,start_ts,end_ts) VALUES(?,?,?,?,?,?)",
                      (kind, "hist", "ok", p, time.time(), time.time()))


def test_estimate_comes_from_measured_peaks_not_a_snapshot(w):
    _record(w, "build", [2 * GiB] * 9 + [6 * GiB])
    ok, info = w.admit("build", 0)
    assert ok and info["history_n"] == 10
    assert info["estimate"] == int(1.25 * w.p90([2 * GiB] * 9 + [6 * GiB]))
    assert info["estimate"] >= 1.25 * 2 * GiB


def test_cap_of_three_refuses_the_fourth(w, monkeypatch):
    monkeypatch.setattr(w, "CAP", 3)
    monkeypatch.setattr(w, "running_workers", lambda: ["a", "b", "c"])
    ok, info = w.admit("build", 1 * GiB)
    assert not ok and any("cap 3" in r for r in info["why"])


def test_gradual_fourth_worker_needs_a_raised_cap_and_a_fitting_forecast(w, monkeypatch):
    monkeypatch.setattr(w, "CAP", 4)
    monkeypatch.setattr(w, "running_workers", lambda: ["a", "b", "c"])
    assert w.admit("build", 2 * GiB)[0], "cap 4 + room => the 4th is admitted"
    monkeypatch.setattr(w, "meminfo_available", lambda: 7 * GiB)   # 6 GiB core reserve leaves 1
    ok, info = w.admit("build", 2 * GiB)
    assert not ok and any("core reserve" in r for r in info["why"]), "forecast over budget => refused"


def test_core_reserve_is_kept_even_when_the_slice_has_room(w, monkeypatch):
    monkeypatch.setattr(w, "meminfo_available", lambda: 8 * GiB)
    ok, info = w.admit("render", 4 * GiB)
    assert not ok and any("core reserve" in r for r in info["why"])


def test_slice_headroom_is_a_separate_budget(w, tmp_path):
    (tmp_path / "slice" / "memory.current").write_text(str(19 * GiB))
    ok, info = w.admit("render", 2 * GiB)
    assert not ok and any("slice headroom" in r for r in info["why"])


def test_gpu_budget_keeps_the_voice_reserve(w, monkeypatch):
    monkeypatch.setattr(w, "gpu_free_mib", lambda: 10 * 1024)
    assert w.admit("asr", 1 * GiB, gpu_mib=3000)[0]
    ok, info = w.admit("asr", 1 * GiB, gpu_mib=6000)        # 10 GiB free - 6 GiB reserve < 6000 MiB
    assert not ok and any("voice reserve" in r for r in info["why"])
    monkeypatch.setattr(w, "gpu_free_mib", lambda: None)    # can't measure the GPU => don't guess
    assert not w.admit("asr", 1 * GiB, gpu_mib=100)[0]


def test_refused_run_launches_nothing(w, monkeypatch):
    printed = []
    monkeypatch.setattr(w, "print", lambda *a, **k: printed.append(a[0] if a else ""), raising=False)
    monkeypatch.setattr(w, "running_workers", lambda: ["a", "b", "c"])
    monkeypatch.setattr(w.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    args = type("A", (), dict(mem="1G", gpu_mib=0, force=False, kind="build", name="x", timeout=60,
                              swap="1G", nice=10, cwd="", detach=False, cmd=["true"]))()
    assert w.run(args) == 75
    assert json.loads(printed[0])["admitted"] is False


# ── live (opt-in) ─────────────────────────────────────────────────────────────────────────────
LIVE = os.environ.get("IRIS_LIVE_WORKER_TESTS") == "1" and shutil.which("systemd-run")
live = pytest.mark.skipif(not LIVE, reason="set IRIS_LIVE_WORKER_TESTS=1 (spawns small transient units)")
CORE_UNITS = ("iris-mouth.service", "iris-ears.service", "iris-postoffice.service")

_ALLOC = ("import sys,time;b=bytearray({n}*1024*1024);[b.__setitem__(i,1) for i in range(0,len(b),4096)];"
          "time.sleep({hold});open(sys.argv[1],'w').write([l for l in open('/proc/self/status') "
          "if l.startswith('VmHWM')][0].split()[1])")


def _main_pids():
    out = {}
    for u in CORE_UNITS:
        r = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", u],
                           capture_output=True, text=True, timeout=10)
        out[u] = r.stdout.strip()
    return out


def _launch(db, name, mem, alloc_mib, hold, out_file):
    env = dict(os.environ, IRIS_WORKER_DB=str(db))
    cmd = [sys.executable, str(WORKER), "run", "--kind", "acceptance", "--name", name, "--mem", mem,
           "--swap", "0", "--timeout", "120", "--force", "--",
           sys.executable, "-c", _ALLOC.format(n=alloc_mib, hold=hold), str(out_file)]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _rows(db):
    import sqlite3
    c = sqlite3.connect(str(db))
    return c.execute("SELECT name,outcome,peak_mem,mem_limit FROM jobs ORDER BY id").fetchall()


@live
def test_live_three_workers_record_peaks_that_match_the_kernel(tmp_path):
    db = tmp_path / "jobs.sqlite"
    before = _main_pids()
    procs = [_launch(db, f"acc3_{i}", "400M", 120, 4, tmp_path / f"hwm{i}") for i in range(3)]
    for p in procs:
        p.communicate(timeout=180)
    rows = {r[0]: r for r in _rows(db)}
    for i in range(3):
        name, outcome, peak, _lim = rows[f"acc3_{i}"]
        hwm = int((tmp_path / f"hwm{i}").read_text()) * 1024          # kernel's own peak RSS
        assert outcome == "ok"
        assert peak is not None and 0.8 * hwm <= peak <= hwm + 128 * 1024 ** 2, (name, peak, hwm)
    assert _main_pids() == before, "core services untouched"


@live
def test_live_worker_over_its_limit_dies_alone(tmp_path):
    db = tmp_path / "jobs.sqlite"
    before = _main_pids()
    p = _launch(db, "acc_oom", "96M", 400, 2, tmp_path / "hwm_oom")
    p.communicate(timeout=180)
    (name, outcome, peak, lim), = _rows(db)
    assert outcome == "oom", outcome
    assert lim == 96 * 1024 ** 2 and (peak or 0) <= lim + 8 * 1024 ** 2
    assert _main_pids() == before, "the OOM kill stayed inside the worker's cgroup"
    assert all(v not in ("", "0") for v in before.values())


@live
def test_live_detached_child_does_not_outlive_its_worker(tmp_path):
    db = tmp_path / "jobs.sqlite"
    pidf = tmp_path / "child.pid"
    env = dict(os.environ, IRIS_WORKER_DB=str(db))
    script = f"setsid sleep 300 </dev/null >/dev/null 2>&1 & echo $! > {pidf}; sleep 1; exit 0"
    subprocess.run([sys.executable, str(WORKER), "run", "--kind", "acceptance", "--name", "acc_leak",
                    "--mem", "200M", "--timeout", "60", "--force", "--", "/bin/bash", "-c", script],
                   env=env, capture_output=True, text=True, timeout=120)
    child = int(pidf.read_text())
    time.sleep(1.0)
    try:
        os.kill(child, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:
        os.kill(child, 9)
    assert not alive, "KillMode=control-group must reap a setsid'd grandchild when the worker exits"


def _live_env(tmp_path, **extra):
    # a scratch job DB AND a scratch event store: a live test must never add tasks to state/v3
    return dict(os.environ, IRIS_WORKER_DB=str(tmp_path / "jobs.sqlite"),
                IRIS_EVENTS_DB=str(tmp_path / "v3" / "events.sqlite"), **extra)


def _unit_active(unit):
    r = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10)
    return r.stdout.strip() in ("active", "activating")


def _wait_row_unit(db, name, timeout=30):
    import sqlite3
    t0 = time.time()
    while time.time() - t0 < timeout:
        if db.exists():
            c = sqlite3.connect(str(db))
            try:
                r = c.execute("SELECT unit FROM jobs WHERE name=?", (name,)).fetchone()
            except sqlite3.Error:
                r = None
            finally:
                c.close()
            if r and _unit_active(r[0]):
                return r[0]
        time.sleep(0.3)
    raise AssertionError(f"{name}: unit never became active")


@live
def test_live_cap_of_four_admits_up_to_four_then_refuses_under_real_load(tmp_path):
    """P2-T3: gradual admission with IRIS_WORKER_CAP=4 while the REAL workers (helpers) keep running.
    Whatever the live count is, admission must stop exactly when 4 units run - and the refused one
    launches nothing."""
    env = _live_env(tmp_path, IRIS_WORKER_CAP="4")
    before = _main_pids()
    procs, refused = [], None
    for i in range(5):
        live_before = len([u for u in subprocess.run(
            ["systemctl", "--user", "list-units", "--type=service", "--state=active,activating", "--no-legend",
             "--plain", "iris-worker-*"], capture_output=True, text=True, timeout=10).stdout.splitlines() if u.strip()])
        cmd = [sys.executable, str(WORKER), "run", "--kind", "acceptance", "--name", f"cap4_{i}", "--mem", "200M",
               "--timeout", "120", "--", "/bin/sleep", "25"]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            p.wait(timeout=4)                       # an admitted run is still waiting on its unit
        except subprocess.TimeoutExpired:
            _wait_row_unit(tmp_path / "jobs.sqlite", f"cap4_{i}")
            procs.append(p)
            continue
        out = p.stdout.read()
        refused = (i, live_before, p.returncode, out)
        break
    try:
        assert refused is not None, "a 5th launch with cap 4 must be refused"
        i, live_before, rc, out = refused
        assert rc == 75, (rc, out)
        assert live_before >= 4, f"refused while only {live_before} workers ran (cap 4)"
        assert "cap 4" in out
        print(f"\ncap4: admitted {len(procs)} beside {live_before - len(procs)} live worker(s); refused #{i + 1}")
        assert not any(r[0] == f"cap4_{i}" for r in _rows(tmp_path / "jobs.sqlite")), "refused = nothing recorded"
    finally:
        for p in procs:
            p.communicate(timeout=120)
    assert all(r[1] == "ok" for r in _rows(tmp_path / "jobs.sqlite")), _rows(tmp_path / "jobs.sqlite")
    assert _main_pids() == before, "core services untouched"


@live
def test_live_killed_launcher_leaves_launcher_lost_and_needs_review_not_running(tmp_path):
    """P2-T4d: worker.py SIGKILLed mid-job. The unit keeps its limits and finishes; `sweep` then marks
    the job launcher_lost and the task's expired lease needs_review - never 'running' forever."""
    import signal
    import sqlite3
    env = _live_env(tmp_path)
    p = subprocess.Popen([sys.executable, str(WORKER), "run", "--kind", "acceptance", "--name", "t4d", "--mem", "150M",
                          "--timeout", "120", "--lease", "6", "--force", "--", "/bin/sleep", "8"],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    unit = _wait_row_unit(tmp_path / "jobs.sqlite", "t4d")
    mem = subprocess.run(["systemctl", "--user", "show", "-p", "MemoryMax", "--value", unit],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    assert mem == str(150 * 1024 ** 2), mem
    p.send_signal(signal.SIGKILL)
    p.wait(timeout=10)
    t0 = time.time()
    while _unit_active(unit) and time.time() - t0 < 60:
        time.sleep(0.5)
    assert not _unit_active(unit)
    time.sleep(7)                                   # past the 6 s lease
    c = sqlite3.connect(str(tmp_path / "jobs.sqlite"))
    c.execute("UPDATE jobs SET start_ts = start_ts - 60 WHERE name='t4d'")   # sweep's 30 s grace
    c.commit()
    c.close()
    r = subprocess.run([sys.executable, str(WORKER), "sweep"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    c = sqlite3.connect(str(tmp_path / "jobs.sqlite"))
    outcome, tid = c.execute("SELECT outcome, task_id FROM jobs WHERE name='t4d'").fetchone()
    c.close()
    assert outcome == "launcher_lost", outcome
    e = sqlite3.connect(str(tmp_path / "v3" / "events.sqlite"))
    status, = e.execute("SELECT status FROM tasks WHERE task_id=?", (tid,)).fetchone()
    e.close()
    assert status == "needs_review", status
