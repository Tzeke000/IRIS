"""V3 Phase 4 - incident bundles (scripts/server/incident.py) + the watchdog's capture-on-RED wiring.

Offline: a fake repo root in tmp_path (fake now.json / integrity.json / events.sqlite / logs / private
files), journalctl/nvidia-smi collectors stubbed where they'd be slow; nothing under the real state/.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INC = ROOT / "scripts" / "server" / "incident.py"
WD = ROOT / "scripts" / "server" / "iris_watchdog.py"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from brain import iris_events  # noqa: E402


@pytest.fixture(autouse=True)
def _real_print(monkeypatch):
    """iris_runtime (imported by other tests) swaps builtins.print for a stderr writer for the whole
    session (MCP stdio safety); these tests read the CLI's stdout, so restore the real print here."""
    import builtins
    rt = sys.modules.get("iris_runtime")
    if rt is not None and getattr(rt, "_real_print", None):
        monkeypatch.setattr(builtins, "print", rt._real_print)

SECRET_ENV = "s3cr3t-Value-XYZ123"
SECRET_JSON = "9876543210123"
FAKE_LAN_IP = ".".join(["192", "168", "250", "9"])   # built at runtime: the guard blocks literals
TOKEN = ".".join(["MTIzNDU2Nzg5MDEyMzQ1Njc4OTAx", "GaBcDe", "abcdefghijklmnopqrstuvwxyz0123456"])   # built at runtime: GitHub push protection flags literal token shapes


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "state" / "health").mkdir(parents=True)
    (r / "config").mkdir()
    (r / "scripts").mkdir()
    (r / "scripts" / "privacy_guard.py").write_text((ROOT / "scripts" / "privacy_guard.py").read_text())
    (r / "config" / "privacy_guard.local.txt").write_text("# test list\nZekeSecretName\n")
    (r / "state" / "health" / "now.json").write_text(json.dumps(
        {"subsystems": {"ears": {"status": "red", "last_error": f"daemon down, owner ZekeSecretName "
                                                                 f"at {FAKE_LAN_IP} token={SECRET_ENV}"}}}))
    (r / "state" / "health" / "integrity.json").write_text(json.dumps({"ok": True, "damaged": []}))
    (r / "state" / "supervise.log").write_text("x\n" * 10 + f"Authorization: Bearer {TOKEN}\n")
    (r / "state" / "server_watchdog.log").write_text(f"discord {TOKEN} sent to {SECRET_JSON}\n")
    priv_env = tmp_path / "private.env"
    priv_env.write_text(f"# comment\nDISCORD_TOKEN={SECRET_ENV}\nSHORT=ab\n")
    priv_json = r / "config" / "private.local.json"
    priv_json.write_text(json.dumps({"zeke_discord_user_id": SECRET_JSON, "flag": True}))
    monkeypatch.setenv("IRIS_INCIDENT_PRIVATE_FILES", os.pathsep.join([str(priv_env), str(priv_json)]))
    db = r / "state" / "v3" / "events.sqlite"
    monkeypatch.setenv("IRIS_EVENTS_DB", str(db))
    s = iris_events.EventStore(db)
    s.insert("discord", f"hello from {SECRET_JSON}", idempotency_key="discord:1", status="admitted")
    s.create_task("render chunk")
    s.close()
    return r


@pytest.fixture()
def inc():
    return _load(INC, "iris_incident_test")


FAST = {"system.txt": lambda: "## gpu\nfake", "procs.txt": lambda: "procs fake"}


def _blob(path: Path) -> str:
    return "".join(p.read_text(errors="replace") for p in path.iterdir())


def test_capture_writes_bundle_masked(inc, fake_root, tmp_path):
    out = tmp_path / "incidents"
    r = inc.capture("ears went red", subsystem="ears", root=fake_root, out_dir=out,
                    collectors=dict(FAST, **{f"journal_{u}.log": (lambda u=u: f"{u} log {SECRET_ENV}")
                                             for u in inc.CORE_UNITS}))
    assert r["captured"], r
    b = Path(r["path"])
    names = {p.name for p in b.iterdir()}
    for n in ("manifest.json", "health.json", "integrity.json", "events.json", "procs.txt", "system.txt",
              "git.txt", "logs_supervise.log", "logs_server_watchdog.log",
              *(f"journal_{u}.log" for u in inc.CORE_UNITS)):
        assert n in names, n
    blob = _blob(b)
    for leak in (SECRET_ENV, SECRET_JSON, TOKEN, "ZekeSecretName", FAKE_LAN_IP):
        assert leak not in blob, leak
    assert "private.env" not in names and "private.local.json" not in names
    m = json.loads((b / "manifest.json").read_text())
    assert m["subsystem"] == "ears" and m["masking"]["privacy_list"] == "1 patterns"
    assert m["masking"]["hits"] > 0 and m["collector_errors"] == {}
    ev = json.loads((b / "events.json").read_text())
    assert ev["integrity"] == "ok" and ev["open_events"] and ev["open_tasks"]
    assert oct(b.stat().st_mode & 0o777) == "0o700"
    assert oct((b / "health.json").stat().st_mode & 0o777) == "0o600"
    assert not [p for p in out.iterdir() if p.name.startswith(".tmp_")]


def test_events_db_opened_read_only(inc, fake_root, tmp_path):
    """A corrupt store must be REPORTED, never moved aside/recreated (EventStore's recovery would)."""
    db = fake_root / "state" / "v3" / "events.sqlite"
    for p in db.parent.iterdir():
        p.unlink()
    db.write_bytes(b"this is not a database" * 100)
    r = inc.capture("corrupt store", root=fake_root, out_dir=tmp_path / "i", collectors=FAST)
    assert r["captured"] and "events.json" in r["errors"]
    assert db.read_bytes().startswith(b"this is not a database")
    assert sorted(p.name for p in db.parent.iterdir()) == ["events.sqlite"]


def test_rate_limit_per_subsystem(inc, fake_root, tmp_path):
    out = tmp_path / "i"
    kw = dict(root=fake_root, out_dir=out, collectors=FAST)
    assert inc.capture("a", subsystem="ears", now=1000.0, **kw)["captured"]
    r2 = inc.capture("b", subsystem="ears", now=1000.0 + 1800, **kw)
    assert not r2["captured"] and "rate-limited" in r2["why"]
    assert inc.capture("c", subsystem="mouth", now=1000.0 + 1800, **kw)["captured"]   # other subsystem
    assert inc.capture("d", subsystem="ears", now=1000.0 + 1800, force=True, **kw)["captured"]
    assert inc.capture("e", subsystem="ears", now=1000.0 + 1800 + 3601, **kw)["captured"]
    assert inc.capture("manual", now=1000.0 + 1801, **kw)["captured"]                 # no subsystem: no limit


def test_size_caps_and_prune(inc, fake_root, tmp_path):
    out = tmp_path / "i"
    big = {"procs.txt": lambda: "P" * 100_000, "system.txt": lambda: "S" * 100_000}
    r = inc.capture("big", root=fake_root, out_dir=out, file_cap=10_000, bundle_cap=25_000, collectors=big)
    m = json.loads((Path(r["path"]) / "manifest.json").read_text())
    assert m["total_bytes"] <= 25_000 + 200
    assert all(f["bytes"] <= 10_000 + 100 for f in m["files"].values())
    assert m["skipped_over_cap"]                        # later collectors skipped once the bundle is full
    for i in range(5):
        inc.capture(f"n{i}", root=fake_root, out_dir=out, keep=3, now=2000.0 + i, collectors=FAST)
    assert len([p for p in out.iterdir() if p.is_dir()]) == 3


def test_collector_failure_is_recorded_not_fatal(inc, fake_root, tmp_path):
    def boom():
        raise RuntimeError("nvidia-smi exploded")
    r = inc.capture("x", root=fake_root, out_dir=tmp_path / "i", collectors={"system.txt": boom})
    assert r["captured"] and "system.txt" in r["errors"]
    assert "collector failed" in (Path(r["path"]) / "system.txt").read_text()


def test_missing_privacy_list_still_masks_generic(inc, fake_root, tmp_path, monkeypatch):
    (fake_root / "config" / "privacy_guard.local.txt").unlink()
    monkeypatch.setenv("IRIS_PRIVACY_LIST", str(tmp_path / "no-such-list.txt"))
    r = inc.capture("x", root=fake_root, out_dir=tmp_path / "i", collectors=FAST)
    b = Path(r["path"])
    blob = _blob(b)
    assert FAKE_LAN_IP not in blob and SECRET_ENV not in blob and TOKEN not in blob
    assert json.loads((b / "manifest.json").read_text())["masking"]["privacy_list"] == "missing"


def test_cli_list_show(inc, fake_root, tmp_path, monkeypatch, capsys):
    out = tmp_path / "i"
    monkeypatch.setenv("IRIS_INCIDENTS_DIR", str(out))
    r = inc.capture("cli check", subsystem="mouth", root=fake_root, out_dir=out, collectors=FAST)
    capsys.readouterr()
    assert inc.main(["list"]) == 0
    assert r["id"] in capsys.readouterr().out
    assert inc.main(["show", "cli-check"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == r["id"]
    assert inc.main(["show", r["id"], "--file", "health.json"]) == 0
    assert "ears" in capsys.readouterr().out
    assert inc.main(["show", r["id"], "--file", "../../x"]) == 1


# ── watchdog wiring ────────────────────────────────────────────────────────────
@pytest.fixture()
def wd(tmp_path, monkeypatch):
    m = _load(WD, "iris_watchdog_inc_test")
    monkeypatch.setattr(m, "LOG", tmp_path / "wd.log")
    monkeypatch.setattr(m, "INCIDENT_FLAG", tmp_path / "capture_on_red")
    monkeypatch.delenv("IRIS_INCIDENT_CAPTURE", raising=False)
    return m


DOC = {"subsystems": {"ears": {"status": "red", "last_error": "no daemon"}, "mouth": {"status": "warn"},
                      "gpu": {"status": "ok"}, "perception": {"status": "parked"}}}


def test_watchdog_off_by_default(wd):
    calls = []
    assert wd.capture_red_incidents(DOC, {}, 100.0, spawn=lambda k, r: calls.append(k) or True) == []
    assert calls == []


def test_watchdog_red_only_rate_limited(wd, monkeypatch):
    wd.INCIDENT_FLAG.write_text("")
    calls = []
    spawn = lambda k, r: calls.append((k, r)) or True  # noqa: E731
    st: dict = {}
    assert wd.capture_red_incidents(DOC, st, 100.0, spawn=spawn) == ["ears"]
    assert calls == [("ears", "ears red: no daemon")]
    assert wd.capture_red_incidents(DOC, st, 100.0 + 3599, spawn=spawn) == []       # once an hour
    assert wd.capture_red_incidents(DOC, st, 100.0 + 3601, spawn=spawn) == ["ears"]
    assert wd.capture_red_incidents(DOC, {}, 100.0, in_grace=True, spawn=spawn) == []  # boot grace
    assert wd.capture_red_incidents(None, {}, 100.0, spawn=spawn) == []              # snapshot failed


def test_watchdog_failed_spawn_retries_next_minute(wd, monkeypatch):
    monkeypatch.setenv("IRIS_INCIDENT_CAPTURE", "1")
    st: dict = {}
    assert wd.capture_red_incidents(DOC, st, 100.0, spawn=lambda k, r: False) == []
    assert wd.capture_red_incidents(DOC, st, 160.0, spawn=lambda k, r: True) == ["ears"]


def test_watchdog_spawn_never_raises(wd, monkeypatch):
    wd.INCIDENT_FLAG.write_text("")

    def boom(k, r):
        raise OSError("no systemd")
    assert wd.capture_red_incidents(DOC, {}, 1.0, spawn=boom) == []
