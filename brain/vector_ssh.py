"""SSH host-key options for the Vector's root shell (code review 10-29, Vector Low: "the SSH health check
uses StrictHostKeyChecking=no with the root key").

Before: every robot ssh ran with StrictHostKeyChecking=no + UserKnownHostsFile=/dev/null, so anything
answering on the robot's IP got my root-key login attempt and was believed. Now: TRUST ON FIRST USE in a
dedicated known_hosts (state/vector/known_hosts) - the first connection pins the key with no manual capture,
and a later different key is REFUSED with a message that says what happened and how to re-pin.

A reflash or a new robot changes the key, which by design stops the root shell + the interoception probe
until the pin is cleared:  ssh-keygen -R <robot ip> -f state/vector/known_hosts
Opt-out (the old behaviour): IRIS_VECTOR_SSH_PIN=0.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KNOWN_HOSTS = Path(os.environ.get("IRIS_VECTOR_KNOWN_HOSTS") or (REPO / "state" / "vector" / "known_hosts"))

_CHANGED_MARKERS = ("host key verification failed", "remote host identification has changed",
                    "host key for", "offending")


def pinning() -> bool:
    return (os.environ.get("IRIS_VECTOR_SSH_PIN") or "1").strip() != "0"


def host_key_opts() -> list[str]:
    """The -o pairs that replace StrictHostKeyChecking=no / UserKnownHostsFile=/dev/null."""
    if not pinning():
        return ["-o", "StrictHostKeyChecking=no", "-o", f"UserKnownHostsFile={os.devnull}"]
    try:
        KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return ["-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
            "-o", f"GlobalKnownHostsFile={os.devnull}", "-o", "CheckHostIP=no"]


def host_key_problem(stderr: str | None, ip: str = "<robot ip>") -> str | None:
    """A readable reason when ssh refused because the robot's key no longer matches the pin, else None."""
    s = (stderr or "").lower()
    if pinning() and any(m in s for m in _CHANGED_MARKERS):
        return (f"Vector SSH host key CHANGED vs the pin in {KNOWN_HOSTS} (reflash / new robot - or something "
                f"else is answering on {ip}). Refused. If expected: ssh-keygen -R {ip} -f {KNOWN_HOSTS}")
    return None
