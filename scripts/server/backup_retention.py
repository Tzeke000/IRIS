#!/usr/bin/env python3
"""scripts/server/backup_retention.py - which nightly backups to keep (used by backup_to_tower.sh).

Grandfather-father-son, 2026-10-29 code review ("7 days is too short to roll back silent memory damage"):
keep the newest archive of each of the 7 most recent DAYS, of each of the 4 most recent ISO WEEKS and of
each of the 6 most recent MONTHS that have a backup. Everything else is deleted.

    names on stdin (one per line, "NAME" or "NAME SIZE") -> prints "DELETE NAME" / "KEEP NAME reason" lines.
    --delete-only: print just the names to delete.

Names must look like iris-home_YYYYmmdd_HHMM.tar.zst[.age]; anything else is ignored (never deleted).
"""
from __future__ import annotations

import datetime as dt
import re
import sys

NAME_RE = re.compile(r"^iris-home_(\d{8})_(\d{4})\.tar\.zst(\.age)?$")
DAILY, WEEKLY, MONTHLY = 7, 4, 6


def parse(name: str):
    m = NAME_RE.match(name)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
    except ValueError:
        return None


def plan(names, daily=DAILY, weekly=WEEKLY, monthly=MONTHLY):
    """-> (keep: dict name->reason, delete: list name). Only well-formed names are considered."""
    stamped = sorted(((parse(n), n) for n in set(names) if parse(n)), reverse=True)
    keep: dict[str, str] = {}

    def bucket(keyf, limit, label):
        seen: list = []
        for ts, n in stamped:            # newest first => the first per bucket is the newest of it
            k = keyf(ts)
            if k in seen:
                continue
            if len(seen) >= limit:
                break
            seen.append(k)
            keep.setdefault(n, f"{label} {k}")

    bucket(lambda t: t.date().isoformat(), daily, "daily")
    bucket(lambda t: "%d-W%02d" % t.isocalendar()[:2], weekly, "weekly")
    bucket(lambda t: t.strftime("%Y-%m"), monthly, "monthly")
    delete = [n for _ts, n in stamped if n not in keep]
    return keep, delete


def main(argv: list[str]) -> int:
    names = [ln.split()[0] for ln in sys.stdin.read().splitlines() if ln.strip()]
    keep, delete = plan(names)
    if "--delete-only" in argv:
        for n in delete:
            print(n)
        return 0
    for n, why in sorted(keep.items()):
        print(f"KEEP {n} {why}")
    for n in delete:
        print(f"DELETE {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
