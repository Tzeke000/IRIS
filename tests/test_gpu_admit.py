"""11-03: load-time GPU admission (scripts/server/gpu_admit.py)."""
import importlib.util
from pathlib import Path

SRV = Path(__file__).resolve().parents[1] / "scripts" / "server"


def _mod(tmp_path):
    spec = importlib.util.spec_from_file_location("gpu_admit_t", SRV / "gpu_admit.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.LOG = tmp_path / "adm.jsonl"
    return m


def test_admit_needs_room_plus_reserve(tmp_path):
    m = _mod(tmp_path)
    m.free_mib = lambda: (5000, 32768)
    assert m.admit(3000, "t")[0] is True            # 3000 + 1024 <= 5000
    assert m.admit(4500, "t")[0] is False           # 4500 + 1024 >  5000
    assert m.LOG.read_text().count("\n") == 2


def test_fail_open_without_nvidia_smi(tmp_path):
    m = _mod(tmp_path)
    m.free_mib = lambda: None
    ok, info = m.admit(99999, "t")
    assert ok and "fail-open" in info["reason"]
    assert m.main(["99999"]) == 2


# ---- 11-04 V3 3.8: CORE mode (mouth + ears) - wait + retry, never refuse for good ----

def _seq(m, frees):
    """free_mib() returns each value in turn, then repeats the last."""
    it = iter(frees)
    last = [None]

    def f():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return None if last[0] is None else (last[0], 32768)
    m.free_mib = f
    m.gpu_holders = lambda limit=5: [{"pid": 1, "mib": 20000, "name": "hog.py"}]


class _Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def sleep(self, s):
        self.sleeps.append(s); self.t += s

    def now(self):
        return self.t


def _recs(m):
    import json
    return [json.loads(x) for x in m.LOG.read_text().splitlines()]


def test_core_refuse_logs_and_retries_then_admits(tmp_path, capsys):
    m = _mod(tmp_path); c = _Clock()
    _seq(m, [2000, 2000, 2000, 12000])                     # short x3, then room
    info = m.admit_core(9300, "mouth_fish", sleep=c.sleep, clock=c.now)
    assert info["ok"] and not info.get("forced") and info["attempt"] == 4
    assert c.sleeps == [5.0, 10.0, 20.0]                   # backoff doubles
    recs = _recs(m)
    assert [r["ok"] for r in recs] == [False, False, False, True]
    assert recs[0]["retry_in_s"] == 5.0 and recs[0]["holders"][0]["name"] == "hog.py"
    err = capsys.readouterr().err
    assert err.count("REFUSED") == 3 and "hog.py" in err and "admitted" in err


def test_core_backoff_is_capped(tmp_path):
    m = _mod(tmp_path); c = _Clock()
    _seq(m, [0] * 8 + [99999])
    m.admit_core(100, "t", max_wait_s=10_000, sleep=c.sleep, clock=c.now)
    assert c.sleeps == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0, 60.0]


def test_core_admits_at_once(tmp_path, capsys):
    m = _mod(tmp_path); c = _Clock()
    _seq(m, [20000])
    info = m.admit_core(9300, "mouth_fish", sleep=c.sleep, clock=c.now)
    assert info["ok"] and info["attempt"] == 1 and c.sleeps == []
    assert len(_recs(m)) == 1
    assert "REFUSED" not in capsys.readouterr().err


def test_core_forces_start_after_max_wait(tmp_path, capsys):
    m = _mod(tmp_path); c = _Clock()
    _seq(m, [0])
    info = m.admit_core(9300, "ears", max_wait_s=100, sleep=c.sleep, clock=c.now)
    assert info["ok"] and info["forced"] and "may OOM" in info["reason"]
    assert sum(c.sleeps) == 100                            # last sleep trimmed to the deadline, never past it
    assert "FORCED START" in capsys.readouterr().err


def test_core_fails_open_without_nvidia_smi(tmp_path, capsys):
    m = _mod(tmp_path); c = _Clock()
    _seq(m, [])
    info = m.admit_core(9300, "mouth_fish", sleep=c.sleep, clock=c.now)
    assert info["ok"] and "fail-open" in info["reason"] and c.sleeps == []
    assert "WARNING" in capsys.readouterr().err


def test_core_gate_can_be_disabled(tmp_path, monkeypatch):
    m = _mod(tmp_path)
    m.free_mib = lambda: (0, 32768)
    monkeypatch.setenv("IRIS_GPU_ADMIT", "0")
    assert m.admit_core(9300, "t", sleep=lambda s: (_ for _ in ()).throw(AssertionError("slept")))["ok"]


# ---- end to end: the real CLI against a FAKE nvidia-smi on PATH ----

import json as _json
import os as _os
import subprocess as _sp
import sys as _sys

FAKE = """#!{py}
import sys, pathlib
d = pathlib.Path({d!r})
if any("compute-apps" in a for a in sys.argv):
    print("4242, 20000"); sys.exit(0)
seq = (d / "seq").read_text().split()
n = int((d / "n").read_text()) if (d / "n").exists() else 0
(d / "n").write_text(str(n + 1))
print(f"{{32768 - int(seq[min(n, len(seq) - 1)])}}, 32768")
"""


def _cli(tmp_path, args, frees=None):
    bindir = tmp_path / "bin"; bindir.mkdir(exist_ok=True)
    if frees is not None:
        (tmp_path / "seq").write_text(" ".join(map(str, frees)))
        fake = bindir / "nvidia-smi"
        fake.write_text(FAKE.format(py=_sys.executable, d=str(tmp_path)))
        fake.chmod(0o755)
    env = {"PATH": str(bindir), "IRIS_GPU_ADMIT_LOG": str(tmp_path / "adm.jsonl")}
    r = _sp.run([_sys.executable, str(SRV / "gpu_admit.py"), *args], capture_output=True, text=True, env=env, timeout=30)
    recs = [_json.loads(x) for x in (tmp_path / "adm.jsonl").read_text().splitlines()]
    return r, recs


def test_cli_core_refuse_retry_admit_then_exec(tmp_path):
    r, recs = _cli(tmp_path, ["9300", "--core", "--who", "mouth_fish", "--backoff", "0.01", "0.02",
                              "--exec", "--", _sys.executable, "-c", "print('DAEMON STARTED')"],
                   frees=[3000, 3000, 15000])
    assert r.returncode == 0, r.stderr
    assert "DAEMON STARTED" in r.stdout
    assert r.stderr.count("REFUSED") == 2 and "admitted" in r.stderr
    assert [x["ok"] for x in recs] == [False, False, True]


def test_cli_core_admit_at_once(tmp_path):
    r, recs = _cli(tmp_path, ["1100", "--core", "--who", "ears"], frees=[20000])
    assert r.returncode == 0 and len(recs) == 1 and recs[0]["ok"] and "REFUSED" not in r.stderr


def test_cli_core_missing_nvidia_smi_fails_open_and_execs(tmp_path):
    r, recs = _cli(tmp_path, ["9300", "--core", "--who", "mouth_fish",
                              "--exec", _sys.executable, "-c", "print('MOUTH STARTED')"])
    assert r.returncode == 0 and "MOUTH STARTED" in r.stdout
    assert "WARNING" in r.stderr and "fail-open" in recs[0]["reason"]


def test_cli_exec_without_core_is_an_error(tmp_path):
    r = _sp.run([_sys.executable, str(SRV / "gpu_admit.py"), "100", "--exec", "true"], capture_output=True, text=True)
    assert r.returncode == 2


def test_units_carry_the_gate():
    root = SRV.parents[1]
    ears = (root / "scripts/server/systemd/iris-ears.service").read_text()
    start = [ln for ln in ears.splitlines() if ln.startswith("ExecStart=")]
    assert len(start) == 1 and "gpu_admit.py 1100 --core" in start[0] and "--exec -- " in start[0]
    assert start[0].rstrip().endswith("wren_voice_daemon.py")
    ml = (root / "scripts/server/mouth_launch.sh").read_text()
    assert "set -e" not in ml                               # a gate error must never stop the mouth
    # every engine launch has its own gate since the previous launch
    lines, prev = ml.splitlines(), 0
    launches = [i for i, ln in enumerate(lines)
                if '"$ROOT/voice/wren_fish_server.py"' in ln or 'exec "$HOME/venvs/iris-style/bin/python"' in ln]
    assert len(launches) == 3
    for i in launches:
        assert any(ln.strip().startswith("gpu_gate ") for ln in lines[prev:i]), lines[i]
        prev = i + 1
