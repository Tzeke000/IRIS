"""11-03: watchdog temple liveness (VR full-stack audit #9)."""
import importlib.util
import types
from pathlib import Path

SRV = Path(__file__).resolve().parents[1] / "scripts" / "server"


def _w(tmp_path, monkeypatch, out):
    spec = importlib.util.spec_from_file_location("iris_watchdog_t", SRV / "iris_watchdog.py")
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    monkeypatch.setattr(w, "ROOT", tmp_path)
    home = tmp_path / "home"; (home / ".ssh").mkdir(parents=True)
    (home / ".iris_private.env").write_text("TEMPLE_HOST=templehost\n")
    monkeypatch.setattr(w, "HOME", home)
    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=out, stderr="", returncode=0))
    return w


def test_two_misses_then_alert_and_clears(tmp_path, monkeypatch):
    w = _w(tmp_path, monkeypatch, "inactive\n")
    st = {}
    assert w.check_temple(st, 1000.0) is None                      # first miss: no alert yet
    assert w.check_temple(st, 1100.0) is None                      # not due (5 min cadence)
    assert "isn't running" in w.check_temple(st, 1400.0)            # second miss: alert
    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="active\n", stderr="", returncode=0))
    assert w.check_temple(st, 1800.0) is None and st["temple_miss"] == 0


def test_deliberately_off_flag_silences(tmp_path, monkeypatch):
    w = _w(tmp_path, monkeypatch, "inactive\n")
    (tmp_path / "state" / "vr_temple").mkdir(parents=True)
    (tmp_path / "state" / "vr_temple" / "server_deliberately_off.json").write_text("{}")
    st = {}
    for t in (1000.0, 1400.0, 1800.0):
        assert w.check_temple(st, t) is None
