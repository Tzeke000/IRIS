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
