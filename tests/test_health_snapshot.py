"""health_snapshot (V3 Phase 1 step 1.5): state/health/now.json writer, probes mocked.

Covers: one failing/hanging probe can't take the others down, atomic write, last_success/last_error carry-over,
deliberate-off (voice flag, night eye park, no ~/LIVE) reads as off/parked - never red, the sink is a warning,
the v3 event store is a no-op until it exists. Nothing touches the live system or the real state/.
    ~/venvs/iris-v100/bin/python -m pytest -q tests/test_health_snapshot.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

SRV = Path(__file__).resolve().parents[1] / "scripts" / "server"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SRV / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def hs(tmp_path, monkeypatch):
    m = _load("health_snapshot_t", "health_snapshot.py")
    monkeypatch.setattr(m, "HOME", tmp_path / "home")
    (tmp_path / "home").mkdir()
    (tmp_path / "root" / "state").mkdir(parents=True)
    monkeypatch.setattr(m, "ROOT", tmp_path / "root")
    monkeypatch.setattr(m, "unit_active_seconds", lambda unit: None)
    return m


def ctx(hs, **kw):
    c = {"root": hs.ROOT, "now": time.time(), "live": True, "quiet": False, "voice_off": False, "in_grace": False}
    c.update(kw)
    return c


def _boom(_ctx):
    raise RuntimeError("probe exploded")


def _hang(_ctx):
    time.sleep(5)
    return {"status": "ok", "probe": "never"}


def _good(_ctx):
    return {"status": "ok", "probe": "fine", "detail": {"x": 1}}


# ── isolation + atomic write ─────────────────────────────────────────────────
def test_failing_and_hanging_probes_only_redden_themselves(hs, tmp_path):
    out = tmp_path / "health" / "now.json"
    probes = {"a_good": _good, "b_boom": _boom, "c_hang": _hang, "d_good": _good}
    t0 = time.monotonic()
    hs.PROBE_TIMEOUT_S = 0.3
    doc = hs.write_snapshot(out, hs.ROOT, probes=probes, in_grace=False)
    assert time.monotonic() - t0 < 3, "a hung probe must not hold the run"
    on_disk = json.loads(out.read_text())
    assert on_disk["ts"] == doc["ts"]
    subs = on_disk["subsystems"]
    assert list(subs) == list(probes), "entries keep the probe order"
    assert subs["a_good"]["ok"] is True and subs["d_good"]["ok"] is True
    assert subs["b_boom"]["status"] == "red" and "probe exploded" in subs["b_boom"]["last_error"]
    assert subs["c_hang"]["status"] == "red" and "timed out" in subs["c_hang"]["last_error"]
    assert on_disk["summary"]["red"] == ["b_boom", "c_hang"] and on_disk["summary"]["ok"] is False
    for v in subs.values():
        assert set(v) >= {"ok", "last_success_ts", "lag_s", "last_error", "probe"}


def test_atomic_write_replaces_whole_file_and_leaves_no_tmp(hs, tmp_path):
    out = tmp_path / "h" / "now.json"
    hs.write_atomic(out, {"v": 1, "pad": "x" * 10000})
    hs.write_atomic(out, {"v": 2})
    assert json.loads(out.read_text()) == {"v": 2}
    assert [p.name for p in out.parent.iterdir()] == ["now.json"]


def test_failed_replace_keeps_the_old_file_intact(hs, tmp_path, monkeypatch):
    out = tmp_path / "h" / "now.json"
    hs.write_atomic(out, {"v": 1})

    def bad_replace(a, b):
        raise OSError("disk went away")
    monkeypatch.setattr(hs.os, "replace", bad_replace)
    with pytest.raises(OSError):
        hs.write_atomic(out, {"v": 2})
    assert json.loads(out.read_text()) == {"v": 1}
    assert [p.name for p in out.parent.iterdir()] == ["now.json"], "tmp file cleaned up"


def test_last_success_and_last_error_carry_over(hs, tmp_path):
    out = tmp_path / "now.json"
    t1 = 1_000_000.0
    hs.write_snapshot(out, hs.ROOT, now=t1, probes={"x": _good}, in_grace=False)
    d2 = hs.write_snapshot(out, hs.ROOT, now=t1 + 120, probes={"x": _boom}, in_grace=False)["subsystems"]["x"]
    assert d2["ok"] is False and d2["last_success_ts"] == t1 and d2["lag_s"] == 120
    assert d2["error_now"] and "exploded" in d2["last_error"]
    d3 = hs.write_snapshot(out, hs.ROOT, now=t1 + 180, probes={"x": _good}, in_grace=False)["subsystems"]["x"]
    assert d3["ok"] is True and d3["lag_s"] == 0 and d3["last_success_ts"] == t1 + 180
    assert not d3["error_now"] and "exploded" in d3["last_error"] and d3["last_error_ts"] == t1 + 120


def test_age_probe_lag_is_the_age_of_the_thing(hs):
    subs = hs.finalize({"b": {"status": "ok", "probe": "p", "age_s": 300.0}}, {}, 10_000.0)
    assert subs["b"]["lag_s"] == 300.0 and subs["b"]["last_success_ts"] == 9_700.0


# ── deliberate-off handling ──────────────────────────────────────────────────
def _dead(*a, **k):
    raise ConnectionRefusedError(111, "Connection refused")


def test_voice_flag_parse_matches_the_host(hs):
    f = hs.ROOT / "state" / "voice_deliberately_off.json"
    assert hs.voice_off(hs.ROOT) is False
    f.write_text('{"off": true, "why": "quiet hours"}')
    assert hs.voice_off(hs.ROOT) is True
    f.write_text('{"off": false}')
    assert hs.voice_off(hs.ROOT) is False, "an existing off:false file is ON (09-05)"
    f.write_text("{not json")
    assert hs.voice_off(hs.ROOT) is False, "fail-open like the host"


def test_dead_ears_and_mouth_are_off_under_the_voice_flag_else_red(hs, monkeypatch):
    monkeypatch.setattr(hs, "ears_ping", _dead)
    monkeypatch.setattr(hs, "http_get", _dead)
    assert hs.probe_ears(ctx(hs, voice_off=True))["status"] == "off"
    assert hs.probe_mouth(ctx(hs, voice_off=True))["status"] == "off"
    assert hs.probe_ears(ctx(hs))["status"] == "red"
    assert hs.probe_mouth(ctx(hs))["status"] == "red"
    monkeypatch.setattr(hs, "unit_active_seconds", lambda unit: 60.0)       # just (re)started: warming
    assert hs.probe_mouth(ctx(hs))["status"] == "warn"


def test_live_ears_report_ok_even_when_muted(hs, monkeypatch):
    monkeypatch.setattr(hs, "ears_ping", lambda: {"ok": True, "result": {"pong": True}})
    r = hs.probe_ears(ctx(hs, voice_off=True))
    assert r["status"] == "ok" and r["detail"]["voice_flag_off"] is True


def test_night_eye_park_is_parked_not_red(hs, monkeypatch):
    monkeypatch.setattr(hs, "http_get", lambda url, t=3.0: (200, b'{"ok": false, "b64": null, "age_sec": 0}'))
    (hs.ROOT / ".tmp").mkdir()
    (hs.ROOT / ".tmp" / "body_pause.flag").write_text("park")
    assert hs.probe_perception(ctx(hs, quiet=True))["status"] == "parked"
    # same flag in daylight with no auto-park on record: a deliberate park OR the stale-morning-flag bug -> warn
    r = hs.probe_perception(ctx(hs, quiet=False))
    assert r["status"] == "warn" and "gpu_park" in r["error"]
    (hs.ROOT / "state" / "eyes_quiet_hours.json").write_text('{"auto_parked_active": true}')
    assert hs.probe_perception(ctx(hs, quiet=False))["status"] == "parked"


def test_perception_red_off_and_grace(hs, monkeypatch):
    monkeypatch.setattr(hs, "http_get", _dead)
    assert hs.probe_perception(ctx(hs))["status"] == "red"
    assert hs.probe_perception(ctx(hs, live=False))["status"] == "off", "no ~/LIVE = deliberate stop"
    assert hs.probe_perception(ctx(hs, in_grace=True))["status"] == "warn", "booting"
    monkeypatch.setattr(hs, "http_get", lambda url, t=3.0: (200, b'{"ok": true, "b64": "AAAA", "age_sec": 0.4}'))
    r = hs.probe_perception(ctx(hs))
    assert r["status"] == "ok" and r["age_s"] == 0.4


def test_unreachable_sink_is_a_warning_not_a_failure(hs, monkeypatch):
    cfg = hs.HOME / ".config" / "iris"
    cfg.mkdir(parents=True)
    (cfg / "voice.env").write_text("IRIS_AUDIO_SINK=sink.invalid:8775\n")
    monkeypatch.setattr(hs, "tcp_connect", _dead)
    r = hs.probe_audio_sink(ctx(hs))
    assert r["status"] == "warn" and "NOWHERE" in r["error"]
    monkeypatch.setattr(hs, "tcp_connect", lambda h, p, t=3.0: None)
    assert hs.probe_audio_sink(ctx(hs))["status"] == "ok"


def test_off_and_parked_publish_ok_none_and_stay_out_of_red(hs, tmp_path):
    probes = {"v": lambda c: {"status": "off", "probe": "p"}, "e": lambda c: {"status": "parked", "probe": "p"}}
    doc = hs.write_snapshot(tmp_path / "n.json", hs.ROOT, probes=probes, in_grace=False)
    assert doc["subsystems"]["v"]["ok"] is None and doc["subsystems"]["e"]["ok"] is None
    assert doc["summary"] == {"ok": True, "red": [], "warn": [], "off": ["v"], "parked": ["e"]}


def test_quiet_window_wraps_midnight(hs):
    def at(hhmm):
        return time.mktime(time.strptime("2026-10-30 " + hhmm, "%Y-%m-%d %H:%M"))
    assert hs.quiet_window(hs.ROOT, at("23:00")) and hs.quiet_window(hs.ROOT, at("03:00"))
    assert not hs.quiet_window(hs.ROOT, at("05:10")) and not hs.quiet_window(hs.ROOT, at("12:00"))
    (hs.ROOT / "state" / "voice_quiet_hours.json").write_text('{"enabled": false}')
    assert not hs.quiet_window(hs.ROOT, at("23:00"))


def test_sdk_stream_missing_host_is_red_live_off_without_live(hs, monkeypatch):
    monkeypatch.setattr(hs, "find_pids", lambda needle: [])
    assert hs.probe_sdk_stream(ctx(hs))["status"] == "red"
    assert hs.probe_sdk_stream(ctx(hs, live=False))["status"] == "off"
    assert hs.probe_sdk_stream(ctx(hs, in_grace=True))["status"] == "warn"


def test_runtime_heartbeat_age(hs, monkeypatch):
    now = time.time()
    hb = hs.ROOT / "state" / "runtime_loop_heartbeat.json"
    monkeypatch.setattr(hs, "_proc_cmdline", lambda pid: "python iris_runtime.py")
    hb.write_text(json.dumps({"ts": now - 4, "pid": 4242}))
    r = hs.probe_runtime(ctx(hs, now=now))
    assert r["status"] == "ok" and 3.5 < r["age_s"] < 4.5
    hb.write_text(json.dumps({"ts": now - 600, "pid": 4242}))
    assert hs.probe_runtime(ctx(hs, now=now))["status"] == "red"
    monkeypatch.setattr(hs, "_proc_cmdline", lambda pid: "something else")   # pid reused
    hb.write_text(json.dumps({"ts": now - 4, "pid": 4242}))
    assert hs.probe_runtime(ctx(hs, now=now))["status"] == "red"


# ── stores ───────────────────────────────────────────────────────────────────
def test_event_store_absent_is_a_noop(hs):
    r = hs.probe_events(ctx(hs))
    assert r["status"] == "n/a"
    assert not (hs.ROOT / "state" / "v3").exists(), "the probe must never create the store"


def test_event_store_oldest_uncommitted_and_parity(hs):
    now = time.time()
    db = hs.ROOT / "state" / "v3" / "events.sqlite"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE events(event_id TEXT PRIMARY KEY, source TEXT, channel_msg_id TEXT, "
                  "status TEXT, received_ts REAL)")
        c.executemany("INSERT INTO events VALUES(?,?,?,?,?)", [
            ("e1", "discord", "111", "admitted", now - 900), ("e2", "discord", "222", "committed", now - 5000),
            ("e3", "voice", None, "running", now - 30)])
    (hs.ROOT / "state" / "discord_inbox.json").write_text(json.dumps({"last_id": "1", "unacked": [{"id": "111"}]}))
    r = hs.probe_events(ctx(hs, now=now))
    unc = r["detail"]["uncommitted"]
    assert unc["discord"]["n"] == 1 and 899 <= unc["discord"]["oldest_age_s"] <= 901
    assert unc["voice"]["n"] == 1
    assert r["detail"]["parity"] == {"json_only": [], "db_only": []}
    assert r["status"] == "warn" and "discord: oldest" in r["error"]       # 900 s > EVENT_WARN_S
    (hs.ROOT / "state" / "discord_inbox.json").write_text(json.dumps({"unacked": [{"id": "111"}, {"id": "333"}]}))
    r = hs.probe_events(ctx(hs, now=now))
    assert r["detail"]["parity"]["json_only"] == ["333"] and "parity mismatch" in r["error"]


def test_event_store_without_tables_is_red_not_a_crash(hs):
    db = hs.ROOT / "state" / "v3" / "events.sqlite"
    db.parent.mkdir(parents=True)
    sqlite3.connect(db).close()
    assert hs.probe_events(ctx(hs))["status"] == "red"


def _fake_chroma(root: Path, n: int):
    db = root / "memory" / "chroma" / "chroma.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE collections(id TEXT, name TEXT)")
        c.execute("CREATE TABLE segments(id TEXT, scope TEXT, collection TEXT)")
        c.execute("CREATE TABLE embeddings(id INTEGER PRIMARY KEY, segment_id TEXT)")
        c.execute("INSERT INTO collections VALUES('c1','iris_memories')")
        c.executemany("INSERT INTO segments VALUES(?,?,?)", [("sv", "VECTOR", "c1"), ("sm", "METADATA", "c1")])
        c.executemany("INSERT INTO embeddings(segment_id) VALUES(?)", [("sm",)] * n)


def test_memory_index_counts(hs):
    jl = hs.ROOT / "state" / "iris_memory.jsonl"
    jl.write_text("".join(json.dumps({"id": str(i)}) + "\n" for i in range(10)) + "\n")
    (hs.ROOT / "state" / "iris_memory.chroma_orphans.jsonl").write_text('{"id": "o"}\n')
    _fake_chroma(hs.ROOT, 11)
    r = hs.probe_memory_index(ctx(hs))
    assert r["status"] == "ok" and r["detail"]["jsonl"] == 10 and r["detail"]["chroma"] == 11
    old = time.time() - 3600
    os.utime(jl, (old, old))
    jl.write_text(jl.read_text() + "".join(json.dumps({"id": f"n{i}"}) + "\n" for i in range(8)))
    os.utime(jl, (old, old))
    assert hs.probe_memory_index(ctx(hs))["status"] == "red", "8 memories never indexed for an hour"


def test_memory_index_without_chroma_is_warn(hs):
    (hs.ROOT / "state" / "iris_memory.jsonl").write_text('{"id": "1"}\n')
    assert hs.probe_memory_index(ctx(hs))["status"] == "warn"


# ── the watchdog calls the writer on both paths, and survives it failing ─────
def test_watchdog_writes_health_and_survives_a_broken_writer(tmp_path, monkeypatch):
    wd = _load("iris_watchdog_hs_t", "iris_watchdog.py")
    monkeypatch.setattr(wd, "STATE", tmp_path / "wd.json")
    monkeypatch.setattr(wd, "LOG", tmp_path / "wd.log")
    monkeypatch.setattr(wd, "BACKUP_LOG", tmp_path / "backup.log")
    monkeypatch.setattr(wd, "HOME", tmp_path)                      # no LIVE => the not-running path
    monkeypatch.setattr(wd, "dm", lambda text: True)
    monkeypatch.setattr(wd, "unit_state", lambda unit, verb: "enabled" if verb == "is-enabled" else "active")
    calls = []
    monkeypatch.setattr(wd, "write_health", lambda now, **kw: calls.append(kw))
    assert wd.main() == 0 and calls == [{}]
    # a writer that raises inside the real write_health is contained + logged
    monkeypatch.undo()
    monkeypatch.setattr(wd, "LOG", tmp_path / "wd.log")
    monkeypatch.setattr(wd, "HEALTH_OUT", tmp_path / "x" / "now.json")
    real = Path(wd.__file__).with_name("health_snapshot.py")
    assert real.is_file()
    monkeypatch.setattr(wd, "__file__", str(tmp_path / "iris_watchdog.py"))   # sibling module now missing
    assert wd.write_health(time.time()) is None
    assert "health snapshot failed" in (tmp_path / "wd.log").read_text()
