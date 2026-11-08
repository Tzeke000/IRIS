"""opsguard 11-08: vr_players.py online never invents "0 online" from a failed ssh; pve_vm.py state prints one word."""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path, monkeypatch=None):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vr_players_failed_ssh_is_unknown_not_zero():
    v = _load("vrp", ROOT / "scripts/server/vr_players.py")
    assert "count" not in v.parse_online("") and "UNKNOWN" in v.parse_online("")["error"]
    on = v.parse_online("[net] welcome 3 = Zeke (human)\n[net] welcome 4 = Iris (bot)\n__VRP_OK__")
    assert on["count"] == 1 and on["bots"] == ["Iris"]
    assert v.parse_online("[net] welcome 3 = Zeke (human)\n[net] Zeke left\n__VRP_OK__")["count"] == 0


def test_vr_players_online_subcommand_exit_codes(monkeypatch):
    v = _load("vrp2", ROOT / "scripts/server/vr_players.py")
    monkeypatch.setattr(v, "temple_host", lambda: "x")
    monkeypatch.setattr(v, "_ssh", lambda *a, **k: "")
    assert v.main(["online", "--json"]) == 3
    monkeypatch.setattr(v, "_ssh", lambda *a, **k: "__VRP_OK__\n")
    buf = io.StringIO(); monkeypatch.setattr(sys, "stdout", buf)   # other suites rebind stdout: own the buffer
    assert v.main(["online", "--json"]) == 0 and '"count": 0' in buf.getvalue()


def test_pve_vm_state_prints_one_word(tmp_path, monkeypatch):
    (tmp_path / ".iris_private.env").write_text("PVE_HOST=h\nPVE_TOKEN_ID=i\nPVE_TOKEN_SECRET=s\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    src = (ROOT / "scripts/server/pve_vm.py").read_text()
    assert 'a[0] == "state"' in src
    m = _load("pve", ROOT / "scripts/server/pve_vm.py")
    monkeypatch.setattr(m, "call", lambda path, *a, **k: {"status": "stopped"} if path == "/101/status/current" else None)
    monkeypatch.setattr(sys, "argv", ["pve_vm.py", "state", "101"])
    code = compile(src.split('if __name__ == "__main__":', 1)[1].replace("\n    ", "\n"), "pve_main", "exec")
    buf = io.StringIO(); monkeypatch.setattr(sys, "stdout", buf)
    exec(code, {**m.__dict__, "sys": sys})
    assert buf.getvalue().strip() == "stopped"
