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
