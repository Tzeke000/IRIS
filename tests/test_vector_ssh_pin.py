"""Code review 10-29 (Vector Low): robot ssh used StrictHostKeyChecking=no with the root key. Now
trust-on-first-use in state/vector/known_hosts; a changed key is refused with a readable reason.
The real-sshd check runs against this machine's own sshd (loopback, no login), skipped if none."""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import vector_ssh as v  # noqa: E402


@pytest.fixture
def kh(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "KNOWN_HOSTS", tmp_path / "vector" / "known_hosts")
    monkeypatch.delenv("IRIS_VECTOR_SSH_PIN", raising=False)
    return v.KNOWN_HOSTS


def test_opts_pin_by_default_and_opt_out(kh, monkeypatch):
    o = v.host_key_opts()
    assert "StrictHostKeyChecking=accept-new" in o and f"UserKnownHostsFile={kh}" in o
    assert "StrictHostKeyChecking=no" not in o and kh.parent.is_dir()
    monkeypatch.setenv("IRIS_VECTOR_SSH_PIN", "0")
    assert "StrictHostKeyChecking=no" in v.host_key_opts()
    assert v.host_key_problem("Host key verification failed.") is None, "opt-out never blames the pin"


def test_problem_text(kh):
    assert v.host_key_problem("Permission denied (publickey).") is None
    msg = v.host_key_problem("@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @\nHost key verification failed.", "ROBOT")
    assert "CHANGED" in msg and "ssh-keygen -R ROBOT" in msg


def _sshd_up() -> bool:
    if not shutil.which("ssh") or not shutil.which("ssh-keygen"):
        return False
    try:
        socket.create_connection(("127.0.0.1", 22), timeout=1).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _sshd_up(), reason="no local sshd")
def test_real_ssh_pins_then_refuses_a_changed_key(kh, tmp_path):
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR",
            "-o", "PreferredAuthentications=none"]
    subprocess.run(base + v.host_key_opts() + ["nobody@127.0.0.1", "true"], capture_output=True, timeout=20)
    assert kh.read_text().strip(), "first contact pinned a key with no manual capture"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "fk")], check=True)
    alg, key = (tmp_path / "fk.pub").read_text().split()[:2]
    kh.write_text(f"127.0.0.1 {alg} {key}\n")
    r = subprocess.run(base + v.host_key_opts() + ["-o", "HostKeyAlgorithms=ssh-ed25519", "nobody@127.0.0.1", "true"],
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 255
    assert v.host_key_problem(r.stderr, "127.0.0.1"), r.stderr
