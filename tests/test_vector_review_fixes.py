"""Regression tests for the 10-29 code review, section "Vector body and
little-brain". FAKES ONLY — nothing here connects to (or moves) the robot.

Run:  /home/iris/venvs/iris-v100/bin/python -m pytest tests/test_vector_review_fixes.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from brain import vector_pilot, vector_session  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_t_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ fakes
class _Status:
    def __init__(self, cliff=False, picked=False, falling=False):
        self.is_cliff_detected = cliff
        self.is_picked_up = picked
        self.is_falling = falling


class _Motors:
    def __init__(self):
        self.calls = []

    def set_wheel_motors(self, l, r, a1=0.0, a2=0.0):
        self.calls.append((float(l), float(r)))

    def stop_all_motors(self):
        self.calls.append(("stop_all",))


class _Conn:
    control_lost_event = None


class _Robot:
    def __init__(self, **status):
        self.status = _Status(**status)
        self.motors = _Motors()
        self.conn = _Conn()

    def disconnect(self):
        pass


def _guarded_session(monkeypatch, wheels, drive_until_offset, **status):
    monkeypatch.setattr(vector_session, "_read_nerves", lambda *a, **k: {})
    s = vector_session.BodySession()
    s.robot = _Robot(**status)
    s.connected = True
    s.last_activity = time.time()
    s._wheels = wheels
    s._drive_until = time.time() + drive_until_offset
    return s


def _run_guard(s, secs=0.4):
    t = threading.Thread(target=s._guard_loop, daemon=True)
    t.start()
    time.sleep(secs)
    s._stop.set()
    t.join(timeout=2.0)


# ------------------------------------------------- HIGH: spin-in-place guard
def test_spin_setpoint_expires(monkeypatch):
    s = _guarded_session(monkeypatch, (100.0, -100.0), -1.0)   # already expired
    _run_guard(s)
    assert s._wheels == (0.0, 0.0), "a spin setpoint must expire like any drive"
    assert (0.0, 0.0) in s.robot.motors.calls


def test_spin_held_while_fresh(monkeypatch):
    s = _guarded_session(monkeypatch, (100.0, -100.0), 30.0)
    _run_guard(s, 0.25)
    assert s._wheels == (100.0, -100.0)
    assert (100.0, -100.0) in s.robot.motors.calls     # guard re-streams it


def test_spin_stopped_when_picked_up(monkeypatch):
    s = _guarded_session(monkeypatch, (100.0, -100.0), 30.0, picked=True)
    _run_guard(s, 0.25)
    assert s._wheels == (0.0, 0.0)
    assert "EDGE-GUARD" in (s._reflex or "")


def test_spin_allowed_on_cliff_escape(monkeypatch):
    # drive() documents "spin (lw=-rw)" as the way off an edge — keep it.
    s = _guarded_session(monkeypatch, (100.0, -100.0), 30.0, cliff=True)
    _run_guard(s, 0.25)
    assert s._wheels == (100.0, -100.0)


def test_forward_still_stopped_on_cliff(monkeypatch):
    s = _guarded_session(monkeypatch, (100.0, 100.0), 30.0, cliff=True)
    _run_guard(s, 0.25)
    assert s._wheels == (0.0, 0.0)


# ------------------------------------- HIGH: per-mission abort + preemption
@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.setattr(vector_pilot, "_battery_low_off_charger", lambda: False)
    monkeypatch.setattr(vector_pilot, "_return_margin", lambda *a: {"ok": True})
    monkeypatch.setattr(vector_pilot, "_session_priority", lambda: None)
    p = vector_pilot.Pilot()
    monkeypatch.setattr(p, "_event", lambda *a, **k: {})
    return p


def test_preempt_uses_fresh_event_and_old_mission_sees_its_own(pilot, monkeypatch):
    seen = {}

    def cooperative(m):
        my = pilot.abort_evt
        seen.setdefault("evts", []).append(my)
        while not my.is_set():
            time.sleep(0.02)
        seen["old_saw_abort"] = True
    monkeypatch.setattr(pilot, "_m_scan", cooperative)
    assert pilot.start({"kind": "scan"})["ok"]
    time.sleep(0.1)
    r = pilot.start({"kind": "scan"})
    assert r["ok"], r
    time.sleep(0.1)
    assert seen.get("old_saw_abort")
    old_evt, new_evt = seen["evts"]
    assert old_evt is not new_evt
    assert old_evt.is_set() and not new_evt.is_set()
    assert pilot.state == "running"           # old's finally didn't reset it
    pilot.abort()
    pilot.thread.join(timeout=2)
    assert pilot.state == "idle"


def test_preempt_refused_while_old_mission_stuck(pilot, monkeypatch):
    release = threading.Event()

    def stuck(m):                 # e.g. inside a 90s blocking SDK dock call
        release.wait(10)
    monkeypatch.setattr(pilot, "_m_scan", stuck)
    assert pilot.start({"kind": "scan"})["ok"]
    old = pilot.thread
    t0 = time.time()
    r = pilot.start({"kind": "scan"})
    assert not r["ok"] and "still unwinding" in r["refused"]
    assert time.time() - t0 >= 2.0
    assert pilot.thread is old               # no second driver spawned
    release.set()
    old.join(timeout=2)
    assert pilot.state == "idle"


def test_missions_refused_in_override(pilot, monkeypatch):
    monkeypatch.setattr(vector_pilot, "_session_priority", lambda: "override")
    r = pilot.start({"kind": "servo", "x": 1, "y": 1})
    assert not r["ok"] and "OVERRIDE" in r["refused"]
    assert pilot.thread is None


def test_session_priority_reads_session(monkeypatch):
    class S:
        connected = True
        _priority = "override"
    monkeypatch.setattr(vector_session, "get_session", lambda create=False: S())
    assert vector_pilot._session_priority() == "override"
    monkeypatch.setattr(vector_session, "get_session", lambda create=False: None)
    assert vector_pilot._session_priority() is None


# ------------------------------- MED: back-offs go through drive() (guarded)
class _DriveSession:
    def __init__(self, refuse=False):
        self.calls = []
        self.refuse = refuse

    def drive(self, lw, rw, hold=1.0, accel=0.0):
        self.calls.append((lw, rw, hold))
        if self.refuse and (lw or rw):
            return {"ok": False, "refused": "cliff (SDK) right now"}
        return {"ok": True}

    def _raw_wheels(self, *a, **k):
        raise AssertionError("back-off must not bypass the guard")


def test_backoff_routes_through_drive():
    s = _DriveSession()
    assert vector_pilot._guarded_backoff(s, -70.0, 0.3)
    assert s.calls[0] == (-70.0, -70.0, 0.3)
    assert s.calls[-1][:2] == (0.0, 0.0)


def test_backoff_refused_by_guard_does_not_sleep():
    s = _DriveSession(refuse=True)
    t0 = time.time()
    assert not vector_pilot._guarded_backoff(s, -70.0, 2.0)
    assert time.time() - t0 < 0.5
    assert s.calls[-1][:2] == (0.0, 0.0)


def test_detour_refused_in_override(pilot, monkeypatch):
    monkeypatch.setattr(vector_pilot, "_session_priority", lambda: "override")
    assert pilot._detour(_DriveSession(), "test") is False


# ------------------------- MED: possession.json atomic + keep-last-good read
def test_write_possession_atomic(tmp_path, monkeypatch):
    p = tmp_path / "possession.json"
    monkeypatch.setattr(vector_pilot, "POSSESSION_JSON", p)
    vector_pilot._write_possession(False, "test")
    d = json.loads(p.read_text())
    assert d["hold"] is False and d["by"] == "test"
    assert not list(tmp_path.glob("*.tmp"))


def test_daemon_possession_keeps_last_good(tmp_path, monkeypatch):
    dm = _load_script("vector_inhabit_daemon")
    p = tmp_path / "possession.json"
    monkeypatch.setattr(dm, "POSSESSION_CTL", p)
    assert dm._possession_wanted() is True               # missing = default hold
    p.write_text(json.dumps({"hold": False}))
    assert dm._possession_wanted() is False
    p.write_text('{"hold": fa')                           # torn mid-write
    assert dm._possession_wanted() is False, "torn read must NOT re-possess"
    p.write_text("")
    assert dm._possession_wanted() is False
    p.write_text(json.dumps({"hold": True}))
    assert dm._possession_wanted() is True


# ------------------------------ MED: connection-thread liveness + bounded exit
def test_conn_thread_dead_detection():
    dm = _load_script("vector_inhabit_daemon")

    class C:
        pass

    class R:
        conn = C()
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    R.conn._thread = t
    assert dm._conn_thread_dead(R) is True
    live = threading.Thread(target=time.sleep, args=(0.5,), daemon=True)
    live.start()
    R.conn._thread = live
    assert dm._conn_thread_dead(R) is False
    R.conn._thread = None
    assert dm._conn_thread_dead(R) is False               # not connected yet


def test_poll_loop_raises_on_dead_connection(monkeypatch):
    dm = _load_script("vector_inhabit_daemon")
    monkeypatch.setattr(dm, "POLL_S", 0.01)
    monkeypatch.setattr(dm, "_conn_thread_dead", lambda robot: True)
    with pytest.raises(RuntimeError, match="connection thread"):
        dm._poll_loop(object())


def test_bounded_disconnect_never_wedges(monkeypatch):
    dm = _load_script("vector_inhabit_daemon")
    monkeypatch.setattr(dm, "log", lambda *a: None)

    class Hung:
        def disconnect(self):
            time.sleep(30)
    t0 = time.time()
    dm._bounded_disconnect(Hung(), timeout_s=0.3)
    assert time.time() - t0 < 2.0


# ---------------------------------------------------- LOW: Linux ping argv
def test_ping_cmd_linux(monkeypatch):
    dm = _load_script("vector_inhabit_daemon")
    monkeypatch.setattr(dm.os, "name", "posix")
    assert dm._ping_cmd("host") == ["ping", "-c", "1", "-W", "2", "host"]
    monkeypatch.setattr(dm.os, "name", "nt")
    assert dm._ping_cmd("host")[:3] == ["ping", "-n", "1"]


# ------------------------------------------------------ LOW: log growth caps
def test_trim_tail_keeps_newest(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(5000)))
    vector_pilot._trim_tail(p, 400, max_bytes=1000)
    lines = p.read_text().splitlines()
    assert len(lines) == 400 and json.loads(lines[-1])["i"] == 4999
    small = tmp_path / "y.jsonl"
    small.write_text('{"i": 1}\n')
    vector_pilot._trim_tail(small, 400, max_bytes=1000)
    assert small.read_text() == '{"i": 1}\n'


def test_session_action_log_trim(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(3000)))
    vector_session._trim_jsonl_tail(p, 100, 50)
    assert len(p.read_text().splitlines()) == 50


# ------------------------------- MED: brain server stop retry + label + unit
@pytest.fixture
def brain_server():
    pytest.importorskip("fastapi")
    return _load_script("vector_brain_server")


def test_stop_retries_then_stop_all(brain_server, monkeypatch):
    import requests
    posted = []

    class Resp:
        status_code = 200

    def fake_post(url, params=None, timeout=None):
        path = url.rsplit("/", 1)[-1]
        posted.append((path, dict(params)))
        stop = all(v == 0 for k, v in params.items() if k != "serial")
        if stop and path == "move_wheels" and len(
                [p for p in posted if p[0] == "move_wheels"]) <= 4:
            raise requests.ConnectionError("wire-pod blip")
        return Resp()
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(brain_server, "MOVE_OK", True)
    monkeypatch.setattr(brain_server.time, "sleep", lambda s: None)
    brain_server._perform_body_actions("{{turn||left}}", "esn")
    paths = [p for p, _ in posted]
    # 1 start + 3 failed stop retries, then the stop-all sweep
    assert paths[:4] == ["move_wheels"] * 4
    assert "move_head" in paths and "move_lift" in paths
    assert posted[-1][1].get("speed") == 0 or posted[-1][1].get("lw") == 0


def test_stop_first_try_no_fallback(brain_server, monkeypatch):
    import requests
    posted = []

    class Resp:
        status_code = 200
    monkeypatch.setattr(requests, "post",
                        lambda url, params=None, timeout=None:
                        posted.append(url.rsplit("/", 1)[-1]) or Resp())
    monkeypatch.setattr(brain_server, "MOVE_OK", True)
    monkeypatch.setattr(brain_server.time, "sleep", lambda s: None)
    brain_server._perform_body_actions("{{look||up}}", "esn")
    assert posted == ["move_head", "move_head"]


def test_room_speech_labelled_unverified(brain_server):
    out = brain_server._flatten([{"role": "user", "content": "hi robot"}])
    assert "Zeke asked" not in out
    assert "UNVERIFIED" in out and "hi robot" in out


def test_ensure_ollama_never_spawns_on_linux(brain_server, monkeypatch):
    import requests
    import subprocess
    monkeypatch.setattr(brain_server.os, "name", "posix")
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: pytest.fail("must not spawn on Linux"))
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: pytest.fail("must not taskkill on Linux"))
    assert brain_server._ensure_ollama() is False


def test_unit_pins_model_env():
    txt = (REPO / "scripts" / "server" / "iris-vector-brain.service").read_text()
    assert "Environment=IRIS_LOCAL_MODEL=" in txt
    assert "Environment=IRIS_LB_TOOLS=1" in txt


# ------------------------- body_possess tool writes possession.json atomically too (10-30)
def test_body_possess_tool_writes_atomically(tmp_path, monkeypatch):
    from tools.system import vector_session_tool as vst
    p = tmp_path / "possession.json"
    monkeypatch.setattr(vst, "POSSESSION_JSON", p)
    monkeypatch.setattr(vst, "POSSESSION_STATUS_JSON", tmp_path / "status.json")
    out = vst._body_possess({"hold": False}, {})
    assert out["set_hold"] is False and out["wanted"] is False
    d = json.loads(p.read_text())
    assert d["hold"] is False and d["by"] == "body_possess"
    assert not [x for x in tmp_path.iterdir() if x.name.endswith(".tmp")]
    p.chmod(0o640)
    vst._body_possess({"hold": True}, {})
    assert json.loads(p.read_text())["hold"] is True
    assert (p.stat().st_mode & 0o777) == 0o640       # replace keeps the file's mode
