"""scripts/memory_decay.py — periodic decay + archive pass over iris_memory.jsonl.

Walks state/iris_memory.jsonl, computes effective_importance for each entry
via brain.iris_human_memory.effective_importance (Ebbinghaus decay + retrieval
boost), and ARCHIVES entries that meet ALL of:
  - importance_level != "permanent"  (permanent never decays)
  - effective_importance < threshold (default 0.10)
  - last access / creation more than N days ago (default 30 days)

Archived entries are moved to state/iris_memory_archive.jsonl (append-only).
Surviving entries stay in state/iris_memory.jsonl. The split happens via
brain/_jsonl_store: the whole --commit pass (re-read, decide, archive append,
rewrite) holds the SAME flock'd lock the runtime's iris_memory.add() and
record_revisit() take, so a memory written while decay runs is never lost,
and the rewrite is mkstemp + fsync + os.replace (no shared .tmp name).

Default behavior: DRY-RUN. Reports what WOULD be archived without writing
anything. Pass --commit to actually move entries.

Usage:
    python scripts/memory_decay.py                 # dry-run
    python scripts/memory_decay.py --commit        # actually archive
    python scripts/memory_decay.py --commit --threshold 0.05 --days 45

Status: PROTOTYPE 2026-05-19. Not yet wired to any cron. To wire weekly,
add a Windows Task Scheduler entry that runs this with --commit on Sundays.

Design notes:
  - We never DELETE memories. Archive is append-only; if you need a memory
    back, grep state/iris_memory_archive.jsonl.
  - "Rewrite shorter on archive" mentioned in the design memo is deferred:
    initial implementation just archives the full entry. Shortening is a
    later step that needs LLM access for summarization.
  - The semantic index (ChromaDB via brain/iris_semantic_memory.py) is not
    touched from here (a second process must not open the runtime's Chroma
    dir). Archived IDs drop out of search at the runtime's next boot, when
    iris_semantic_memory.reconcile_from_jsonl() deletes Chroma IDs that are
    no longer in the JSONL.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from brain import _jsonl_store  # noqa: E402


def _read(p: Path) -> "_jsonl_store.ReadResult":
    res = _jsonl_store.read_jsonl(p)
    if res.error:
        print(f"ERROR reading {p}: {res.error}", file=sys.stderr)
    elif res.bad:
        print(f"WARNING {p.name}: {res.bad} bad line(s) "
              f"({res.bad_decode} undecodable, {res.bad_json} unparseable) — "
              f"kept in a salvage file on rewrite", file=sys.stderr)
    return res


def _load_jsonl(p: Path) -> list[dict[str, Any]]:
    return _read(p).entries


def _write_jsonl(p: Path, entries: list[dict[str, Any]],
                 salvage: list[bytes] | None = None) -> None:
    _jsonl_store.rewrite_jsonl(p, entries, salvage=salvage)


def _append_jsonl(p: Path, entries: list[dict[str, Any]]) -> None:
    _jsonl_store.append_jsonl(p, entries)


def _split(entries: list[dict[str, Any]], args: argparse.Namespace, now_ts: float
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    to_archive: list[dict[str, Any]] = []
    surviving: list[dict[str, Any]] = []
    for e in entries:
        eligible, reason = _eligible_for_archive(e, args.threshold, args.days, now_ts)
        if args.verbose:
            print(f"  [{e.get('id','?')}] {reason}")
        if eligible:
            to_archive.append(e)
        else:
            surviving.append(e)
    return to_archive, surviving


def _eligible_for_archive(
    entry: dict[str, Any],
    threshold: float,
    min_age_days: float,
    now_ts: float,
) -> tuple[bool, dict[str, Any]]:
    """Return (eligible, reasoning) for an entry. Eligible = passes all three
    archival criteria (not permanent, low effective importance, old).
    """
    reasoning: dict[str, Any] = {}

    # 1. Permanent check.
    level = str(entry.get("importance_level") or "default").lower()
    reasoning["importance_level"] = level
    if level == "permanent":
        reasoning["reason"] = "permanent"
        return (False, reasoning)

    # 2. Age check.
    created_ts = float(entry.get("ts") or 0.0)
    if created_ts <= 0:
        reasoning["reason"] = "missing ts"
        return (False, reasoning)
    age_days = (now_ts - created_ts) / 86400.0
    reasoning["age_days"] = round(age_days, 1)
    if age_days < min_age_days:
        reasoning["reason"] = "too recent"
        return (False, reasoning)

    # 3. Effective importance check.
    try:
        from brain import iris_human_memory
        iris_human_memory.configure(REPO_ROOT)
        base_imp = float(entry.get("base_importance") or entry.get("importance") or 0.5)
        eff = iris_human_memory.effective_importance(
            memory_id=str(entry.get("id", "")),
            base_importance=base_imp,
            created_ts=created_ts,
        )
        reasoning["effective_importance"] = round(eff, 3)
    except Exception as e:
        reasoning["reason"] = f"importance compute failed: {e!r}"
        return (False, reasoning)

    # Low-level decays faster — adjust threshold up for "low" entries.
    eff_threshold = threshold
    if level == "high":
        eff_threshold = threshold * 0.5  # high is more conservative
    elif level == "low":
        eff_threshold = threshold * 2.0  # low decays faster

    if eff >= eff_threshold:
        reasoning["reason"] = f"effective {eff:.3f} >= threshold {eff_threshold:.3f}"
        return (False, reasoning)

    reasoning["reason"] = f"eligible: effective {eff:.3f} < threshold {eff_threshold:.3f}"
    return (True, reasoning)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Decay + archive iris_memory.jsonl entries")
    parser.add_argument("--commit", action="store_true",
                        help="Actually archive (default is dry-run)")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Effective importance below this is archive-eligible (default 0.10)")
    parser.add_argument("--days", type=float, default=30.0,
                        help="Minimum age in days before archive eligible (default 30)")
    parser.add_argument("--memory-path", type=str, default="state/iris_memory.jsonl",
                        help="Relative to repo root (default state/iris_memory.jsonl)")
    parser.add_argument("--archive-path", type=str, default="state/iris_memory_archive.jsonl",
                        help="Relative to repo root (default state/iris_memory_archive.jsonl)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-entry reasoning")
    args = parser.parse_args(argv[1:])

    mem_p = REPO_ROOT / args.memory_path
    archive_p = REPO_ROOT / args.archive_path

    if args.commit:
        # Hold the shared lock for the WHOLE read-decide-write pass so no
        # concurrent add()/record_revisit() append can land between our read
        # and our rewrite (fix-first #3: that append used to be lost).
        try:
            with _jsonl_store.locked(mem_p, timeout=60.0):
                return _run(mem_p, archive_p, args)
        except TimeoutError as e:
            print(f"ERROR: {e} — another writer holds the memory lock; nothing changed",
                  file=sys.stderr)
            return 1
    return _run(mem_p, archive_p, args)


def _run(mem_p: Path, archive_p: Path, args: argparse.Namespace) -> int:
    res = _read(mem_p)
    if res.error:
        return 1
    entries = res.entries
    if not entries:
        print(f"no entries in {mem_p}")
        return 0

    now_ts = time.time()
    to_archive, surviving = _split(entries, args, now_ts)

    print(f"")
    print(f"=== memory_decay summary ===")
    print(f"  total entries:    {len(entries)}")
    print(f"  archive-eligible: {len(to_archive)}")
    print(f"  surviving:        {len(surviving)}")
    print(f"  threshold:        {args.threshold}")
    print(f"  min age days:     {args.days}")
    print(f"")

    if not args.commit:
        print("DRY-RUN (no files changed). Pass --commit to actually archive.")
        return 0

    if not to_archive:
        print("Nothing to archive.")
        return 0

    # Atomic-ish: write surviving to memory_path, append archived to archive_path.
    # Order matters: if archive write fails, memory is unchanged.
    try:
        _append_jsonl(archive_p, to_archive)
    except Exception as e:
        print(f"ERROR appending to {archive_p}: {e!r}", file=sys.stderr)
        return 1

    try:
        _write_jsonl(mem_p, surviving, salvage=res.bad_raw)
    except Exception as e:
        # Archive was already written but memory write failed. Print clear warning
        # so the operator can recover (memory still has all entries; archive has
        # duplicates of the would-be-archived ones; rerun after fix).
        print(
            f"ERROR writing {mem_p}: {e!r}\n"
            f"  Archive write succeeded ({len(to_archive)} entries appended to {archive_p}).\n"
            f"  Live memory unchanged. Safe to re-run after fix.",
            file=sys.stderr,
        )
        return 1

    print(f"DONE: archived {len(to_archive)} entries to {archive_p}")
    print(f"      live memory now has {len(surviving)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
