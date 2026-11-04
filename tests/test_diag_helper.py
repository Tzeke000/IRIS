"""11-04: V3 4.2 - the diagnostic helper's wake gate (scripts/server/diag_helper.py).

One test per refusal reason, all with fakes: a tmp root for the flag files, a fake gpu_admit, a fake MemAvailable.
No nvidia-smi, no Ollama, no real state/ is touched.
"""
import datetime as dt
import importlib.util
import json
from pathlib import Path

SRV = Path(__file__).resolve().parents[1] / "scripts" / "server"
NOON = dt.datetime(2026, 11, 4, 12, 0, 0)


def _mod():
    spec = importlib.util.spec_from_file_location("diag_helper_t", SRV / "diag_helper.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _admit(free=20000, reserve=1024, reason=None):
    def f(need, who="?"):
        if reason:
            return True, {"ok": True, "reason": reason}
        return free >= need + reserve, {"ok": free >= need + reserve, "free_mib": free, "reserve_mib": reserve}
    return f


def _wake(tmp_path, *, now=NOON, admit=None, mem=64000, **kw):
    m = _mod()
    return m, m.wake(root=tmp_path, now=now, admit_fn=admit or _admit(), mem_fn=lambda: mem, **kw)


def _put(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def _refused(rec):
    return {r["check"]: r["action"] for r in rec["refusals"]}


def test_all_clear_is_gpu_and_logged(tmp_path):
    m, rec = _wake(tmp_path)
    assert rec["verdict"] == "gpu" and rec["refusals"] == []
    assert rec["need"]["gpu_mib"] == m.MODEL_MIB_DEFAULT + m.KV_MIB_DEFAULT
    log = (tmp_path / "state" / "health" / "diag_wake.jsonl").read_text().splitlines()
    assert len(log) == 1 and json.loads(log[0])["verdict"] == "gpu"
    assert m.EXIT[rec["verdict"]] == 0


def test_gpu_admit_refusal_is_cpu_only(tmp_path):
    seen = []

    def admit(need, who="?"):
        seen.append(need)
        return False, {"ok": False, "free_mib": 3000, "reserve_mib": 1024}
    m, rec = _wake(tmp_path, admit=admit, model_mib=2000, kv_mib=500)
    assert seen == [2500]                                   # model + KV, as ONE ask
    assert rec["verdict"] == "cpu_only" and _refused(rec) == {"gpu": "cpu_only"}
    assert m.EXIT["cpu_only"] == 3


def test_no_nvidia_smi_fails_closed(tmp_path):
    _, rec = _wake(tmp_path, admit=_admit(reason="no nvidia-smi (fail-open)"))
    assert rec["verdict"] == "cpu_only" and "gpu" in _refused(rec)


def test_low_ram_defers(tmp_path):
    m, rec = _wake(tmp_path, mem=5000, ram_mib=2048)          # 5000 < 2048 + 4096
    assert rec["verdict"] == "defer" and _refused(rec) == {"ram": "defer"}
    assert rec["retry_after_s"] == m.RETRY["ram"]


def test_unreadable_meminfo_defers(tmp_path):
    _, rec = _wake(tmp_path, mem=None)
    assert _refused(rec) == {"ram": "defer"}


def test_park_flag_is_cpu_only(tmp_path):
    _put(tmp_path, "state/little_brain/pilot_deliberately_off.json", {"off": True, "reason": "Zeke's call"})
    _, rec = _wake(tmp_path)
    assert rec["verdict"] == "cpu_only" and _refused(rec) == {"park": "cpu_only"}


def test_park_flag_off_false_is_not_parked(tmp_path):
    _put(tmp_path, "state/little_brain/pilot_deliberately_off.json", {"off": False})
    _, rec = _wake(tmp_path)
    assert rec["verdict"] == "gpu"


def test_park_flag_unreadable_fails_closed(tmp_path):
    _put(tmp_path, "state/little_brain/pilot_deliberately_off.json", "{not json")
    _, rec = _wake(tmp_path)
    assert _refused(rec) == {"park": "cpu_only"}


def test_quiet_hours_window_defers_both_sides_of_midnight(tmp_path):
    for t in (dt.datetime(2026, 11, 4, 23, 0), dt.datetime(2026, 11, 5, 3, 0), dt.datetime(2026, 11, 4, 22, 30)):
        _, rec = _wake(tmp_path, now=t)
        assert _refused(rec) == {"quiet_hours": "defer"}, t
    for t in (dt.datetime(2026, 11, 4, 5, 10), dt.datetime(2026, 11, 4, 22, 29)):
        _, rec = _wake(tmp_path, now=t)
        assert rec["verdict"] == "gpu", t


def test_quiet_hours_reads_config_and_disable(tmp_path):
    _put(tmp_path, "state/voice_quiet_hours.json", {"enabled": True, "start": "11:00", "end": "13:00"})
    _, rec = _wake(tmp_path)                                   # noon is inside 11-13
    assert _refused(rec) == {"quiet_hours": "defer"}
    _put(tmp_path, "state/voice_quiet_hours.json", {"enabled": False, "start": "11:00", "end": "13:00"})
    _, rec = _wake(tmp_path)
    assert rec["verdict"] == "gpu"


def test_voice_off_flag_defers_by_value_not_existence(tmp_path):
    _put(tmp_path, "state/voice_deliberately_off.json", {"off": True})
    _, rec = _wake(tmp_path)
    assert _refused(rec) == {"quiet_hours": "defer"}
    _put(tmp_path, "state/voice_deliberately_off.json", {"off": False})   # the 09-05 trap: exists but OFF:false
    _, rec = _wake(tmp_path)
    assert rec["verdict"] == "gpu"


def test_speaking_status_defers_unless_stale(tmp_path):
    m = _mod()
    _put(tmp_path, "scratch/voice_status.json", {"state": "speaking", "ts": NOON.timestamp() - 5})
    _, rec = _wake(tmp_path)
    assert _refused(rec) == {"speaking": "defer"} and rec["retry_after_s"] == m.RETRY["speaking"]
    _put(tmp_path, "scratch/voice_status.json", {"state": "speaking", "ts": NOON.timestamp() - m.SPEAK_STALE_S - 1})
    _, rec = _wake(tmp_path)
    assert rec["verdict"] == "gpu"                              # a crashed mouth's leftover status


def test_live_amplitude_defers(tmp_path):
    _put(tmp_path, "scratch/voice_amplitude.json", {"amp": 0.4, "ts": NOON.timestamp() - 0.5})
    _, rec = _wake(tmp_path)
    assert _refused(rec) == {"speaking": "defer"}
    _put(tmp_path, "scratch/voice_amplitude.json", {"amp": 0.0, "ts": NOON.timestamp() - 0.5})
    _, rec = _wake(tmp_path)
    assert rec["verdict"] == "gpu"


def test_defer_outranks_cpu_only_and_every_check_is_logged(tmp_path):
    _put(tmp_path, "state/little_brain/pilot_deliberately_off.json", {"off": True})
    _put(tmp_path, "scratch/voice_status.json", {"state": "speaking", "ts": NOON.timestamp()})
    m, rec = _wake(tmp_path, admit=_admit(free=100), mem=1000)
    assert rec["verdict"] == "defer"
    assert _refused(rec) == {"park": "cpu_only", "speaking": "defer", "ram": "defer", "gpu": "cpu_only"}
    assert set(rec["checks"]) == {"park", "quiet_hours", "speaking", "ram", "gpu"}
    assert rec["retry_after_s"] == max(m.RETRY["speaking"], m.RETRY["ram"])


def test_cli_exit_codes(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "_default_admit", _admit(free=100))
    monkeypatch.setattr(m, "read_mem_available_mib", lambda: 64000)
    real_wake = m.wake
    monkeypatch.setattr(m, "wake", lambda *a, **k: real_wake(*a, now=NOON, **k))
    assert m.main(["wake", "--json"]) == 3
    # asserted on the decision LOG, not stdout: other suite tests swap sys.stdout and capsys then sees nothing
    log = (tmp_path / "state" / "health" / "diag_wake.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["verdict"] == "cpu_only"
