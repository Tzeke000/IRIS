"""tests/test_iris_runtime_singleton.py — tests for the iris_runtime
single-instance guard (`_check_existing_iris_instance` + `_release_iris_pid_lock`).

Imports the functions from iris_runtime.py without booting the full MCP server.
Uses tmp_path fixture to isolate PID lockfile location per test.

Run:
    pytest tests/test_iris_runtime_singleton.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add repo root so we can import iris_runtime's module-level symbols.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def isolated_pid_file(tmp_path, monkeypatch):
    """Redirect the singleton's PID file to a tmp location for the test."""
    import iris_runtime
    fake_pid_file = tmp_path / "iris.pid"
    monkeypatch.setattr(iris_runtime, "_IRIS_PID_FILE", fake_pid_file)
    return fake_pid_file


def test_first_instance_writes_pidfile(isolated_pid_file):
    """First call (no existing lockfile) should write our PID without erroring."""
    import iris_runtime
    assert not isolated_pid_file.exists()
    iris_runtime._check_existing_iris_instance()
    assert isolated_pid_file.exists()
    written_pid = int(isolated_pid_file.read_text().strip())
    assert written_pid == os.getpid()


def test_stale_lockfile_is_overwritten(isolated_pid_file):
    """If lockfile has a PID that's dead, overwrite silently."""
    import iris_runtime
    # Write a dead PID (1 is init/systemd on Linux, but on Windows PID 1 is reserved).
    # Use a guaranteed-dead PID by reading our own then forcing one that doesn't
    # match iris_runtime in cmdline.
    dead_pid = 999999  # very unlikely to be alive
    isolated_pid_file.write_text(str(dead_pid))

    # Mock psutil so we control the "is process alive" answer.
    with patch.dict(sys.modules, {"psutil": MagicMock()}) as _:
        import psutil
        psutil.Process.side_effect = Exception("no such process")
        # Should NOT sys.exit
        iris_runtime._check_existing_iris_instance()

    # Lockfile should now contain OUR pid.
    assert int(isolated_pid_file.read_text().strip()) == os.getpid()


def test_live_lockfile_other_iris_exits(isolated_pid_file):
    """If lockfile has a PID for an actually-live iris_runtime process, exit."""
    import iris_runtime
    other_pid = 12345
    isolated_pid_file.write_text(str(other_pid))

    fake_psutil = MagicMock()
    fake_process = MagicMock()
    fake_process.cmdline.return_value = ["python", "iris_runtime.py"]
    fake_psutil.Process.return_value = fake_process

    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        with pytest.raises(SystemExit) as excinfo:
            iris_runtime._check_existing_iris_instance()
        assert excinfo.value.code == 1

    # Lockfile should NOT have been overwritten.
    assert int(isolated_pid_file.read_text().strip()) == other_pid


def test_live_lockfile_unrelated_process_overwrites(isolated_pid_file):
    """If lockfile PID is alive but the cmdline doesn't match iris_runtime
    (PID reuse case), treat as stale and overwrite."""
    import iris_runtime
    other_pid = 12345
    isolated_pid_file.write_text(str(other_pid))

    fake_psutil = MagicMock()
    fake_process = MagicMock()
    # Cmdline doesn't reference iris_runtime — PID reuse case.
    fake_process.cmdline.return_value = ["chrome.exe", "--some-flag"]
    fake_psutil.Process.return_value = fake_process

    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        # Should NOT sys.exit
        iris_runtime._check_existing_iris_instance()

    # Lockfile should now have OUR pid.
    assert int(isolated_pid_file.read_text().strip()) == os.getpid()


def test_corrupted_lockfile_is_overwritten(isolated_pid_file):
    """Non-integer content in lockfile shouldn't crash the check."""
    import iris_runtime
    isolated_pid_file.write_text("not_a_pid\n")
    # Should NOT raise
    iris_runtime._check_existing_iris_instance()
    # Lockfile should now have our PID.
    assert int(isolated_pid_file.read_text().strip()) == os.getpid()


def test_env_bypass_skips_check(isolated_pid_file, monkeypatch):
    """IRIS_SKIP_INSTANCE_CHECK=1 should skip the check entirely."""
    import iris_runtime
    monkeypatch.setenv("IRIS_SKIP_INSTANCE_CHECK", "1")
    # Write a "live" lockfile that would otherwise cause exit.
    isolated_pid_file.write_text("12345")

    fake_psutil = MagicMock()
    fake_process = MagicMock()
    fake_process.cmdline.return_value = ["python", "iris_runtime.py"]
    fake_psutil.Process.return_value = fake_process

    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        # Should NOT sys.exit (bypassed)
        iris_runtime._check_existing_iris_instance()

    # Lockfile content unchanged — bypass means we don't touch it either.
    # (Implementation detail; matches current code behavior.)


def test_release_pidlock_removes_only_our_pid(isolated_pid_file):
    """release_iris_pid_lock should only unlink if the PID matches ours."""
    import iris_runtime
    # Write our PID.
    isolated_pid_file.write_text(str(os.getpid()))
    iris_runtime._release_iris_pid_lock()
    assert not isolated_pid_file.exists()


def test_release_pidlock_leaves_others_alone(isolated_pid_file):
    """If the lockfile has someone else's PID, release shouldn't unlink."""
    import iris_runtime
    other_pid = 99999
    isolated_pid_file.write_text(str(other_pid))
    iris_runtime._release_iris_pid_lock()
    # Lockfile still present, content unchanged.
    assert isolated_pid_file.exists()
    assert int(isolated_pid_file.read_text().strip()) == other_pid


def test_release_pidlock_idempotent_missing_file(isolated_pid_file):
    """release should be safe to call when lockfile doesn't exist."""
    import iris_runtime
    assert not isolated_pid_file.exists()
    # Should NOT raise.
    iris_runtime._release_iris_pid_lock()


# ── code review 10-29: flock instead of check-then-write ─────────────────────
import subprocess  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402


def _hold_flock(lock_path, pid_path=None, argv_tag="holder", hold_s=30.0):
    """Child process that takes the instance flock (and optionally writes its pid)."""
    code = textwrap.dedent(f"""
        import fcntl, os, sys, time
        fd = os.open({str(lock_path)!r}, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        if {str(pid_path) if pid_path else ''!r}:
            open({str(pid_path)!r}, 'w').write(str(os.getpid()))
        print('locked', flush=True)
        time.sleep({hold_s})
    """)
    p = subprocess.Popen([sys.executable, "-c", code, argv_tag], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "locked"
    return p


@pytest.fixture
def fresh_lock(monkeypatch):
    import iris_runtime
    iris_runtime._release_instance_flock()
    yield iris_runtime
    iris_runtime._release_instance_flock()


def test_flock_is_held_after_a_clean_start(isolated_pid_file, fresh_lock):
    import fcntl
    m = fresh_lock
    m._check_existing_iris_instance()
    assert m._IRIS_PID_LOCK_FD is not None
    fd = os.open(str(isolated_pid_file) + ".lock", os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # nobody else can take it now
    finally:
        os.close(fd)
    m._release_iris_pid_lock()
    assert m._IRIS_PID_LOCK_FD is None and not isolated_pid_file.exists()


def test_racing_second_runtime_exits(isolated_pid_file, fresh_lock, monkeypatch):
    """The other runtime holds the flock and has written its PID: we exit, PID file untouched."""
    m = fresh_lock
    monkeypatch.setenv("IRIS_INSTANCE_LOCK_WAIT_S", "0.5")
    holder = _hold_flock(str(isolated_pid_file) + ".lock", isolated_pid_file, "iris_runtime.py")
    try:
        with patch.dict(sys.modules, {"psutil": None}):   # no psutil: /proc fallback must still see it
            with pytest.raises(SystemExit) as ei:
                m._check_existing_iris_instance()
        assert ei.value.code == 1
        assert int(isolated_pid_file.read_text()) == holder.pid
        assert m._IRIS_PID_LOCK_FD is None
    finally:
        holder.kill()
        holder.wait()


def test_waits_for_a_dying_runtime_then_starts(isolated_pid_file, fresh_lock, monkeypatch):
    """A restart: the old runtime still holds the lock for its last ~second - we wait, then win."""
    m = fresh_lock
    monkeypatch.setenv("IRIS_INSTANCE_LOCK_WAIT_S", "8")
    holder = _hold_flock(str(isolated_pid_file) + ".lock", None, "iris_runtime.py", hold_s=1.0)
    t = time.monotonic()
    m._check_existing_iris_instance()
    assert time.monotonic() - t >= 0.5
    holder.wait()
    assert m._IRIS_PID_LOCK_FD is not None
    assert int(isolated_pid_file.read_text()) == os.getpid()


def test_lock_held_by_non_runtime_fails_open(isolated_pid_file, fresh_lock, monkeypatch):
    """Lock busy but the PID file names no live runtime (orphaned child): start anyway, loudly."""
    m = fresh_lock
    monkeypatch.setenv("IRIS_INSTANCE_LOCK_WAIT_S", "0.3")
    isolated_pid_file.write_text("999999")
    holder = _hold_flock(str(isolated_pid_file) + ".lock", None, "orphan")
    try:
        m._check_existing_iris_instance()
        assert int(isolated_pid_file.read_text()) == os.getpid()
        assert m._IRIS_PID_LOCK_FD is None
    finally:
        holder.kill()
        holder.wait()


def test_proc_fallback_without_psutil(monkeypatch):
    import iris_runtime as m
    with patch.dict(sys.modules, {"psutil": None}):
        assert m._pid_is_iris_runtime(os.getpid()) is False or "iris_runtime" in open(
            f"/proc/{os.getpid()}/cmdline").read()
        assert m._pid_is_iris_runtime(999999) is False
