"""backup_retention: 7 daily / 4 weekly / 6 monthly (2026-10-29 code review).

    ~/venvs/iris-v100/bin/python -m pytest -q tests/test_backup_retention.py
"""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "server" / "backup_retention.py"
spec = importlib.util.spec_from_file_location("backup_retention_t", SRC)
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def _names(days: int, start=dt.datetime(2026, 10, 29, 3, 41)):
    return [f"iris-home_{(start - dt.timedelta(days=i)).strftime('%Y%m%d_%H%M')}.tar.zst" for i in range(days)]


def test_week_of_nightlies_keeps_all():
    keep, delete = br.plan(_names(7))
    assert delete == [] and len(keep) == 7


def test_a_year_of_nightlies():
    names = _names(365)
    keep, delete = br.plan(names)
    assert len(keep) + len(delete) == 365
    # the 7 newest days are all there
    assert all(n in keep for n in names[:7])
    months = {n[10:16] for n in keep}
    assert len(months) == 6                    # 6 months reach back
    assert len(keep) <= 7 + 4 + 6
    # the oldest kept is the newest backup of the 6th month back (May 2026)
    oldest = min(keep)
    assert oldest.startswith("iris-home_202605"), oldest


def test_newest_of_each_bucket_and_junk_ignored():
    names = ["iris-home_20261029_0341.tar.zst", "iris-home_20261029_1200.tar.zst.age",
             "notes.txt", "iris-home_2026_bad.tar.zst", "iris-home_20261029_0341.tar.zst.sha256"]
    keep, delete = br.plan(names)
    assert "iris-home_20261029_1200.tar.zst.age" in keep          # newest of the day wins the daily slot
    assert delete == ["iris-home_20261029_0341.tar.zst"]          # same day, older, not a week/month head
    assert all("bad" not in n and "notes" not in n and not n.endswith(".sha256") for n in keep)


def test_duplicates_and_gaps():
    # sparse: one backup every 10 days -> each is its own day; 7 newest kept + month heads
    names = [f"iris-home_{(dt.datetime(2026, 10, 29) - dt.timedelta(days=10 * i)).strftime('%Y%m%d')}_0340.tar.zst"
             for i in range(30)]
    keep, delete = br.plan(names + names[:3])
    assert len(set(keep) | set(delete)) == 30
    assert all(n in keep for n in names[:7])
