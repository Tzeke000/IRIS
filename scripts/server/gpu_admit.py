#!/usr/bin/env python3
"""scripts/server/gpu_admit.py — load-time GPU admission for the one V100 (11-03, V3 spec / doc audit "GPU checks").

Before a model loads, ask: is there NEED MiB free plus a reserve, so this load can't squeeze the mouth/ears
that are already resident? Usable two ways:
  CLI   gpu_admit.py NEED_MIB [--who NAME] [--wait S]   exit 0 = admitted, 1 = refused, 2 = no GPU/nvidia-smi
        (as a unit's ExecStartPre: a refused load fails the START loudly instead of OOM-ing a neighbour)
  py    from gpu_admit import admit; ok, info = admit(1200, who="whisper")
Every decision is appended to state/health/gpu_admission.jsonl. Fail-OPEN when nvidia-smi itself is missing
(exit 2 / ok=True with reason) so a broken probe never blocks the ears from starting.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "state" / "health" / "gpu_admission.jsonl"
RESERVE_MIB = 1024          # always leave this much free for activations / spikes of what's already resident


def free_mib() -> tuple[int, int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=6).stdout.strip().splitlines()[0]
        used, total = (int(float(x)) for x in out.split(","))
        return total - used, total
    except Exception:
        return None


def _log(rec: dict) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def admit(need_mib: int, who: str = "?", wait_s: float = 0.0, reserve_mib: int = RESERVE_MIB) -> tuple[bool, dict]:
    deadline = time.time() + max(0.0, wait_s)
    while True:
        fm = free_mib()
        if fm is None:
            info = {"ts": time.time(), "who": who, "need_mib": need_mib, "ok": True, "reason": "no nvidia-smi (fail-open)"}
            _log(info)
            return True, info
        free, total = fm
        ok = free >= need_mib + reserve_mib
        if ok or time.time() >= deadline:
            info = {"ts": time.time(), "who": who, "need_mib": need_mib, "reserve_mib": reserve_mib,
                    "free_mib": free, "total_mib": total, "ok": ok}
            _log(info)
            return ok, info
        time.sleep(2.0)


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("need_mib", type=int)
    ap.add_argument("--who", default="cli")
    ap.add_argument("--wait", type=float, default=0.0, help="keep re-checking up to S seconds")
    ap.add_argument("--reserve", type=int, default=RESERVE_MIB)
    a = ap.parse_args(argv)
    ok, info = admit(a.need_mib, a.who, a.wait, a.reserve)
    print(json.dumps(info))
    if info.get("reason", "").startswith("no nvidia-smi"):
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
