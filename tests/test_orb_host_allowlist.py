"""Code review 10-29 (security, High): :5876's Host allow-list failed OPEN when
IRIS_ORB_ALLOWED_HOSTS was unset. Now unset = loopback only; '*' is the explicit opt-out."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from brain import orb_http as oh  # noqa: E402


def _ok(monkeypatch, raw, host):
    check, hosts = oh._parse_allowed_hosts(raw)
    monkeypatch.setattr(oh, "_HOST_CHECK", check)
    monkeypatch.setattr(oh, "_ALLOWED_HOSTS", hosts)
    return oh._host_ok(host)


def test_unset_fails_closed_to_loopback(monkeypatch):
    for raw in (None, "", "  , "):
        assert _ok(monkeypatch, raw, "127.0.0.1:5876")
        assert _ok(monkeypatch, raw, "localhost")
        assert _ok(monkeypatch, raw, "[::1]:5876")
        assert not _ok(monkeypatch, raw, "evil.example:5876")
        assert not _ok(monkeypatch, raw, "iris-home:5876")
        assert not _ok(monkeypatch, raw, None)


def test_configured_list(monkeypatch):
    raw = "iris-home, Example-Host"
    assert _ok(monkeypatch, raw, "iris-home:5876")
    assert _ok(monkeypatch, raw, "example-host")
    assert _ok(monkeypatch, raw, "127.0.0.1")
    assert not _ok(monkeypatch, raw, "rebind.attacker.test")


def test_star_is_the_explicit_opt_out(monkeypatch):
    assert _ok(monkeypatch, "*", "anything.test")
