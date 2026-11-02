#!/usr/bin/env python3
"""update_watch.py - tell Iris when HELD packages (kernel / NVIDIA) have updates waiting.

Why (Zeke 2026-11-02): unattended-upgrades silently moved the NVIDIA driver (09-12) and the kernel
(10-31). Those are now blacklisted from AUTO updates (/etc/apt/apt.conf.d/51iris-hold-kernel-nvidia)
so we install them by hand on a chosen day - but "you never know when the update is about
security", so Iris must HEAR about every one. This runs daily (iris-update-watch.timer):

  * reads `apt list --upgradable` (apt-daily.timer keeps the lists fresh; no root needed)
  * keeps the packages matching the hold patterns
  * marks each one security if any candidate origin is a *-security pocket
  * writes state/health/updates.json
  * if the set of (package, version) is NEW since the last alert, submits an LLM-bridge request
    (kind update_alert) so Iris is woken with the list.

No installs, no root. `--print` = show only, never alert.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "state" / "health" / "updates.json"
HOLD = re.compile(r"^(linux-|nvidia-|libnvidia-|xserver-xorg-video-nvidia)")


def upgradable() -> list[dict]:
    r = subprocess.run(["apt", "list", "--upgradable"], capture_output=True, text=True, timeout=120,
                       env={"LANG": "C", "PATH": "/usr/bin:/bin"})
    rows = []
    # e.g. "linux-image-generic/jammy-updates,jammy-security 5.15.0.205.177 amd64 [upgradable from: 5.15.0.202.174]"
    for line in r.stdout.splitlines():
        m = re.match(r"^([^/\s]+)/(\S+)\s+(\S+)\s+\S+\s+\[upgradable from: ([^\]]+)\]", line)
        if not m:
            continue
        name, pockets, new, old = m.groups()
        rows.append({"package": name, "from": old, "to": new, "pockets": pockets,
                     "security": "-security" in pockets})
    return rows


def main() -> int:
    show_only = "--print" in sys.argv
    held = [r for r in upgradable() if HOLD.match(r["package"])]
    prev = {}
    try:
        prev = json.loads(OUT.read_text())
    except Exception:
        pass
    key = sorted(f"{r['package']}={r['to']}" for r in held)
    doc = {"ts": time.time(), "held_upgradable": held, "n": len(held),
           "security": sum(r["security"] for r in held), "key": key,
           "alerted_key": prev.get("alerted_key", [])}
    if show_only:
        print(json.dumps(doc, indent=1))
        return 0
    new = bool(key) and key != prev.get("alerted_key")
    if new:
        lines = [f"- {r['package']}: {r['from']} -> {r['to']}{'  [SECURITY]' if r['security'] else ''}"
                 for r in held[:40]]
        try:
            sys.path.insert(1, str(ROOT))
            from brain import iris_llm
            iris_llm.submit(
                "[UPDATE WATCH - automated, not Zeke] Held packages (kernel/NVIDIA - blacklisted from auto "
                f"updates, Zeke 11-02) have updates waiting: {len(held)} package(s), "
                f"{doc['security']} from a SECURITY pocket.\n" + "\n".join(lines) +
                "\n\nTell Zeke on Discord (say whether any are security fixes) and propose a manual update "
                "window: save memory first, `sudo apt-get install --only-upgrade <pkgs>`, check DKMS built "
                "nvidia for any new kernel (/lib/modules/<ver>/updates/dkms/nvidia.ko), reboot only with his "
                "OK. Reply here with one line via llm_reply.",
                kind="update_alert", requester="update_watch", deadline_s=7 * 24 * 3600, priority=3)
            doc["alerted_key"] = key
            doc["alerted_ts"] = time.time()
        except Exception as e:  # noqa: BLE001 - keep the snapshot even if the bridge is down
            doc["alert_error"] = repr(e)[:200]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(OUT)
    print(f"held upgradable: {len(held)} (security {doc['security']}); alerted={'yes' if new else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
