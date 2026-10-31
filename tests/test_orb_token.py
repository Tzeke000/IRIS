"""Code review 10-29 (security, High): :5876 had no authentication. Bearer token, staged off|log|enforce.

Middleware tests go to an UNROUTED path: a request the token gate lets through gets 404 from the router,
one it refuses gets 401 first - so no real handler (camera, chat, memory) ever runs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from starlette.testclient import TestClient  # noqa: E402

from brain import orb_http as oh  # noqa: E402

TOK = "t0k-" + "x" * 40
LAN = ("192.0.2.10", 50000)          # TEST-NET-1, never a real host
PATH = "/api/v1/__token_probe__"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(oh, "_HOST_CHECK", False)        # TestClient's Host is "testserver"
    monkeypatch.setattr(oh, "_TOKEN_LOGGED", {})

    def set_(mode, token=TOK):
        monkeypatch.setattr(oh, "_ORB_TOKEN_MODE", mode)
        monkeypatch.setattr(oh, "_ORB_TOKEN", token)
        return TestClient(oh.app, client=LAN)
    return set_


def test_mode_defaults():
    assert oh._token_mode(None, "") == "off"
    assert oh._token_mode("", TOK) == "log"            # a configured token never silently means off
    assert oh._token_mode("ENFORCE", TOK) == "enforce"
    assert oh._token_mode("bogus", TOK) == "log"


def test_load_token_env_then_file(tmp_path, monkeypatch):
    monkeypatch.delenv("IRIS_ORB_TOKEN", raising=False)
    f = tmp_path / "orb_token"
    monkeypatch.setenv("IRIS_ORB_TOKEN_FILE", str(f))
    assert oh._load_orb_token() == ""
    f.write_text(TOK + "\n")
    os.chmod(f, 0o600)
    assert oh._load_orb_token() == TOK
    monkeypatch.setenv("IRIS_ORB_TOKEN", "fromenv")
    assert oh._load_orb_token() == "fromenv"


def test_enforce_refuses_lan_without_token_and_accepts_every_form(gate):
    c = gate("enforce")
    assert c.post(PATH, json={}).status_code == 401
    assert c.get(PATH).status_code == 401
    assert c.post(PATH, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post(PATH, headers={"Authorization": "Bearer " + TOK}).status_code == 404
    assert c.post(PATH, headers={"X-Iris-Token": TOK}).status_code == 404
    assert c.get(PATH + "?token=" + TOK).status_code == 404            # <img>/<iframe> URLs
    assert c.get("/api/v1/health").status_code != 401                   # liveness stays open
    assert c.options(PATH).status_code != 401                           # CORS preflight


def test_enforce_exempts_loopback(gate, monkeypatch):
    gate("enforce")
    c = TestClient(oh.app, client=("127.0.0.1", 40000))
    assert c.post(PATH, json={}).status_code == 404
    assert oh._is_loopback("::1") and oh._is_loopback("::ffff:127.0.0.1") and not oh._is_loopback("192.0.2.1")


def test_enforce_without_a_token_fails_closed(gate):
    c = gate("enforce", token="")
    assert c.post(PATH, headers={"Authorization": "Bearer "}).status_code == 401
    assert c.get(PATH + "?token=").status_code == 401


def test_log_mode_never_denies_and_notes_each_call_site_once(gate, capsys):
    c = gate("log")
    for _ in range(3):
        assert c.post(PATH, json={}).status_code == 404
    assert c.post(PATH, headers={"Authorization": "Bearer " + TOK}).status_code == 404
    err = capsys.readouterr().err
    assert err.count("WOULD refuse POST " + PATH) == 1
    assert TOK not in err, "the token is never printed"


def test_off_is_the_old_behaviour(gate):
    c = gate("off", token="")
    assert c.post(PATH, json={}).status_code == 404


def test_websocket_gated_too(gate):
    c = gate("enforce")
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws") as ws:
            ws.receive_json()
    with c.websocket_connect("/ws?token=" + TOK) as ws:   # accepted (heartbeat every 30 s; just close)
        pass
