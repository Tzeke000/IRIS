"""iris_watchdog: backup-staleness alert (also without ~/LIVE), PID-reuse guard, DM cap (2026-10-29 review).

No DMs are sent: dm() and the state file are redirected to a tmp dir.
    ~/venvs/iris-v100/bin/python -m pytest -q tests/test_watchdog_backup_alerts.py
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "scripts" / "server" / "iris_watchdog.py"


@pytest.fixture()
def wd(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("iris_watchdog_t", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "STATE", tmp_path / "wd.json")
    monkeypatch.setattr(m, "LOG", tmp_path / "wd.log")
    monkeypatch.setattr(m, "BACKUP_LOG", tmp_path / "backup.log")
    monkeypatch.setattr(m, "HOME", tmp_path)                 # no tmp/LIVE => the not-running path
    monkeypatch.setattr(m, "HEALTH_OUT", tmp_path / "health" / "now.json")   # never the real state/health
    monkeypatch.setattr(m, "write_health", lambda now, **kw: None)          # live probes: tests/test_health_snapshot.py
    sent = []
    monkeypatch.setattr(m, "dm", lambda text: sent.append(text) or True)
    monkeypatch.setattr(m, "unit_state", lambda unit, verb: "enabled" if verb == "is-enabled" else "active")
    m._sent = sent
    return m


def _log_ok(path: Path, ts: float):
    path.write_text(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) +
                    " backup ok: iris-home_x.tar.zst 560 MB -> tower\n")


def test_fresh_backup_is_quiet(wd):
    _log_ok(wd.BACKUP_LOG, time.time() - 3600)
    assert wd.backup_problems(time.time()) == {}


def test_stale_and_missing_backup_alert(wd):
    assert "backup" in wd.backup_problems(time.time())            # no log at all
    _log_ok(wd.BACKUP_LOG, time.time() - 40 * 3600)
    p = wd.backup_problems(time.time())
    assert "backup" in p and "40 h" in p["backup"], p


def test_backup_alert_fires_without_live_and_recovers(wd, monkeypatch):
    now = time.time()
    monkeypatch.setattr(wd, "last_backup_ok_ts", lambda path=None: now - 40 * 3600)
    assert wd.main() == 0
    assert len(wd._sent) == 1 and "backup" in wd._sent[0]
    assert wd.main() == 0                                    # rate-limited: no second DM
    assert len(wd._sent) == 1
    monkeypatch.setattr(wd, "last_backup_ok_ts", lambda path=None: time.time() - 60)
    assert wd.main() == 0
    assert len(wd._sent) == 2 and "recovered" in wd._sent[1]


def test_timer_not_enabled_alerts(wd, monkeypatch):
    monkeypatch.setattr(wd, "last_backup_ok_ts", lambda path=None: time.time() - 60)
    monkeypatch.setattr(wd, "unit_state", lambda unit, verb: "disabled")
    assert "backup-timer" in wd.backup_problems(time.time())


def test_no_live_does_not_send_recovered_for_runtime_keys(wd, monkeypatch):
    monkeypatch.setattr(wd, "last_backup_ok_ts", lambda path=None: time.time() - 60)
    wd.save_state({"problems": {"heartbeat": "x"}, "alerted": {"heartbeat": time.time() - 10}})
    wd.main()
    assert wd._sent == []                                     # a deliberate stop is not a 'recovery'


def test_dm_cap_per_run(wd):
    st = {}
    wd.send_alerts(st, {f"k{i}": f"p{i}" for i in range(6)}, time.time())
    assert len(wd._sent) == wd.MAX_DMS_PER_RUN


def test_pid_reuse_guard(wd):
    assert wd.pid_alive(os.getpid())
    assert wd.pid_alive(os.getpid(), "python")
    assert not wd.pid_alive(os.getpid(), "iris_runtime.py-not-me")
    assert not wd.pid_alive(2 ** 22 + 12345, "iris_runtime")
