#!/usr/bin/env python3
"""scripts/hooks/lease_guard.py — PreToolUse (Edit|Write|MultiEdit|NotebookEdit) file-lease check for
OUT-OF-PROCESS helpers (11-03, V3 spec step 1.3 / doc audit "the hook never sees the out-of-process helpers").

The body host's SDK hook only sees in-session sub-agents. Helpers run as their own `claude -p` inside a worker
unit, which sets IRIS_TASK_ID (+ IRIS_LEASES when the job declared globs via `worker.py run --leases`). This hook:
  * does NOTHING unless IRIS_TASK_ID is set (my own session, and anything else, is untouched)
  * LOG-ONLY (Zeke's strict-vs-log-only question is still open): never denies, never prints to stdout,
    always exits 0. A conflict -> one line on stderr + a row in state/v3/lease_conflicts.jsonl.
  * "leased by another task" is always logged; "outside this task's declared globs" only when the task
    declared leases (IRIS_LEASES), so undeclared helpers don't spam.
Fails open on every error (a broken store must never block an edit).
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def main() -> int:
    tid = os.environ.get("IRIS_TASK_ID", "").strip()
    if not tid:
        return 0
    try:
        d = json.load(sys.stdin)
    except Exception:
        return 0
    if str(d.get("tool_name") or "") not in WRITE_TOOLS:
        return 0
    ti = d.get("tool_input") or {}
    path = str(ti.get("file_path") or ti.get("notebook_path") or "")
    if not path:
        return 0
    try:
        sys.path.insert(1, str(ROOT))
        from brain import iris_events
        res = iris_events.EventStore(busy_timeout_s=2.0, lease_root=ROOT).check_write(tid, path) or {}
    except Exception:
        return 0
    if res.get("allowed", True) or res.get("degraded"):
        return 0
    reason = str(res.get("reason") or "")
    if reason.startswith("outside") and not os.environ.get("IRIS_LEASES"):
        return 0
    rec = {"ts": time.time(), "task_id": tid, "path": path, "reason": reason, "held_by": res.get("held_by"),
           "tool": d.get("tool_name"), "mode": "log-only"}
    try:
        p = Path(os.environ.get("IRIS_LEASE_LOG") or (ROOT / "state" / "v3" / "lease_conflicts.jsonl"))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass
    print(f"[lease_guard] LOG-ONLY: task {tid} writes {path} - {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
