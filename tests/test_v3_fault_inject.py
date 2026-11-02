"""V3 Phase 4 - sandbox fault injection (scripts/server/fault_inject.py).

Each scenario runs the real harness end to end in a throw-away /tmp/iris-fi-* staging root (CUDA hidden,
HF offline, no live ports) and must PASS: degraded, not crashed, incident recorded. Plus the safety rails:
a child refuses without IRIS_ROLE=staging, and refuses a staging root that isn't a fresh iris-fi-* temp dir.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FI = ROOT / "scripts" / "server" / "fault_inject.py"


def _load():
    spec = importlib.util.spec_from_file_location("iris_fault_inject_test", FI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _userns_ok() -> bool:
    if not shutil.which("unshare"):
        return False
    r = subprocess.run(["unshare", "-rm", "true"], capture_output=True)
    return r.returncode == 0


@pytest.mark.parametrize("name", ["model_load", "gpu_pressure", "mic_gone"])
def test_scenario_degrades_and_records(name):
    if name != "gpu_pressure":
        pytest.importorskip("sounddevice")
    r = _load().run_scenario(name)
    assert r["passed"], r
    assert not r["crashed"] and r["incident"]
    assert "(removed)" in r["staging"]


@pytest.mark.skipif(not _userns_ok(), reason="unprivileged user+mount namespaces unavailable")
def test_disk_full_degrades_recovers_and_falls_back():
    r = _load().run_scenario("disk_full")
    assert r["passed"], r
    assert r["filled"] == "ENOSPC" and r["insert_while_full_ok"] == 0 and r["insert_after_recovery"]
    assert "No space left" in r["incident_primary_failed"] and r["half_bundles_left"] == []


def test_child_refuses_without_staging_role(tmp_path):
    staging = Path(tempfile.mkdtemp(prefix="iris-fi-"))
    try:
        env = {k: v for k, v in os.environ.items() if k != "IRIS_ROLE"}
        r = subprocess.run([sys.executable, str(FI), "_child", "gpu_pressure", str(staging)], env=env,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode != 0 and "IRIS_ROLE" in (r.stderr + r.stdout)
        assert not (staging / "bin").exists()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@pytest.mark.parametrize("bad", ["repo", "named"])
def test_child_refuses_non_temp_staging(tmp_path, bad):
    target = ROOT / "state" if bad == "repo" else tmp_path / "not-fi"
    env = dict(os.environ, IRIS_ROLE="staging")
    r = subprocess.run([sys.executable, str(FI), "_child", "gpu_pressure", str(target)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout)
