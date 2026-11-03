"""11-03: out-of-process helpers' file-lease hook (scripts/hooks/lease_guard.py) - log-only, never denies."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(ROOT))
from brain import iris_events as ev  # noqa: E402

HOOK = ROOT / "scripts" / "hooks" / "lease_guard.py"


def _run(env, payload):
    e = dict(os.environ, **env)
    return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload), capture_output=True,
                          text=True, env=e, timeout=30)


def test_logs_conflict_never_denies(tmp_path, monkeypatch):
    db = tmp_path / "ev.sqlite"
    s = ev.EventStore(db)
    a = s.create_task("A", kind="worker:helper").id
    b = s.create_task("B", kind="worker:helper").id
    fa = s.claim(a, "worker:a", 600)
    s.claim(b, "worker:b", 600)
    target = str(ROOT / "scratch" / "lease_test_zone" / "x.py")
    assert s.acquire_leases(a, [str(ROOT / "scratch" / "lease_test_zone" / "**")], fence=fa)["ok"]
    s.close()
    pay = {"tool_name": "Edit", "tool_input": {"file_path": target}}
    base = {"IRIS_EVENTS_DB": str(db), "IRIS_LEASE_LOG": str(tmp_path / "conf.jsonl")}
    r = _run({**base, "IRIS_TASK_ID": b}, pay)                     # B writes into A's lease
    assert r.returncode == 0 and r.stdout == "" and "LOG-ONLY" in r.stderr
    r = _run({**base, "IRIS_TASK_ID": a, "IRIS_LEASES": "x"}, pay)  # A inside its own lease
    assert r.returncode == 0 and r.stderr == ""
    r = _run({**base, "IRIS_TASK_ID": b}, {"tool_name": "Read", "tool_input": {"file_path": target}})
    assert r.returncode == 0 and r.stderr == ""                     # reads ignored
    env_no = {k: v for k, v in os.environ.items() if k != "IRIS_TASK_ID"}
    r = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(pay), capture_output=True, text=True,
                       env={**env_no, **base}, timeout=30)
    assert r.returncode == 0 and r.stderr == ""                     # not a worker: untouched
