"""Code review 2026-10-29 §Memory + fix-first #3 — the shared memory store.

Covers brain/_jsonl_store.py (flock'd lock shared across processes, newline-safe
fsync'd append, tolerant per-line decode, mkstemp+replace rewrite with salvage),
and the writers that now use it: iris_memory, iris_human_memory (record_revisit,
record_accesses, meta pruning), scripts/memory_decay.py, and the Chroma
reconcile in iris_semantic_memory (paged IDs, stale-ID deletion, orphan export).

Everything runs on tmp copies — the live state/ and memory/chroma are never touched.

    /home/iris/venvs/iris-v100/bin/python -m pytest tests/test_memory_store.py -q
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import _jsonl_store as js  # noqa: E402


def _row(i: int, **kw) -> dict:
    d = {"id": f"id{i:04d}", "ts": 1000.0 + i, "text": f"memory {i}"}
    d.update(kw)
    return d


# ── _jsonl_store ────────────────────────────────────────────────────────────

def test_append_after_torn_tail_does_not_swallow_next_row(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_bytes(json.dumps(_row(1)).encode() + b"\n" + b'{"id": "torn", "te')  # no newline
    js.append_jsonl(p, [_row(2)])
    res = js.read_jsonl(p)
    assert [e["id"] for e in res.entries] == ["id0001", "id0002"]
    assert res.bad_json == 1 and res.bad_raw == [b'{"id": "torn", "te']
    assert p.read_bytes().endswith(b"\n")


def test_append_to_new_and_clean_file(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "m.jsonl"
    js.append_jsonl(p, [_row(1)])
    js.append_jsonl(p, [_row(2), _row(3)])
    assert p.read_bytes().count(b"\n") == 3
    assert not p.read_bytes().startswith(b"\n")


def test_bad_utf8_line_does_not_empty_the_store(tmp_path: Path, capsys) -> None:
    p = tmp_path / "m.jsonl"
    bad = json.dumps(_row(2)).encode().replace(b"memory 2", b"memory \xff2")
    p.write_bytes(json.dumps(_row(1)).encode() + b"\n" + bad + b"\n"
                  + json.dumps(_row(3)).encode() + b"\n")
    res = js.read_jsonl(p)
    assert [e["id"] for e in res.entries] == ["id0001", "id0002", "id0003"]
    assert res.bad_decode == 1 and res.bad_raw == [bad]
    assert "�" in res.entries[1]["text"]
    assert "1 undecodable" in capsys.readouterr().err   # counted AND logged


def test_nul_run_from_power_cut_is_recovered(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_bytes(json.dumps(_row(1)).encode() + b"\n" + b"\x00" * 300
                  + json.dumps(_row(2)).encode() + b"\n")
    res = js.read_jsonl(p)
    assert [e["id"] for e in res.entries] == ["id0001", "id0002"]
    assert res.nul_repaired == 1 and res.bad == 0


def test_rewrite_is_atomic_keeps_mode_and_salvages(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_bytes(json.dumps(_row(1)).encode() + b"\n" + b"garbage\n")
    os.chmod(p, 0o664)
    res = js.read_jsonl(p)
    js.rewrite_jsonl(p, res.entries + [_row(2)], salvage=res.bad_raw)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o664
    assert [e["id"] for e in js.read_jsonl(p).entries] == ["id0001", "id0002"]
    sal = list(tmp_path.glob("m.salvage_*_unparseable.jsonl"))
    assert len(sal) == 1 and sal[0].read_bytes() == b"garbage\n"
    assert not [x for x in tmp_path.iterdir() if x.name.endswith(".tmp")]


def test_lock_is_reentrant_and_excludes_threads(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    order: list[str] = []
    with js.locked(p):
        with js.locked(p):  # reentrant: no deadlock
            t = threading.Thread(target=lambda: (js.append_jsonl(p, [_row(9)]), order.append("t")))
            t.start()
            time.sleep(0.2)
            order.append("main")
    t.join(5)
    assert order == ["main", "t"]


def _hold_lock_then_rewrite(path: str, ready, hold_s: float) -> None:
    sys.path.insert(0, str(ROOT))
    from brain import _jsonl_store as j
    with j.locked(path):
        res = j.read_jsonl(path)
        ready.set()
        time.sleep(hold_s)
        j.rewrite_jsonl(path, res.entries[1:])  # "decay": drop the first row


def test_append_during_cross_process_rewrite_is_not_lost(tmp_path: Path) -> None:
    """fix-first #3: a memory written while decay runs used to be lost."""
    p = tmp_path / "m.jsonl"
    js.append_jsonl(p, [_row(1), _row(2)])
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_hold_lock_then_rewrite, args=(str(p), ready, 0.6))
    proc.start()
    assert ready.wait(20)
    t0 = time.monotonic()
    js.append_jsonl(p, [_row(3)])           # must wait for the other process
    waited = time.monotonic() - t0
    proc.join(20)
    assert proc.exitcode == 0
    assert waited > 0.3
    assert [e["id"] for e in js.read_jsonl(p).entries] == ["id0002", "id0003"]


def _hammer(path: str, start: int, n: int) -> None:
    sys.path.insert(0, str(ROOT))
    from brain import _jsonl_store as j
    for i in range(start, start + n):
        j.append_jsonl(path, [{"id": f"p{i}", "ts": float(i), "text": "x" * 200}])


def _rewriter(path: str, rounds: int) -> None:
    sys.path.insert(0, str(ROOT))
    from brain import _jsonl_store as j
    for _ in range(rounds):
        with j.locked(path):
            j.rewrite_jsonl(path, j.read_jsonl(path).entries)


def test_many_appenders_and_a_rewriter_lose_nothing(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.touch()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_hammer, args=(str(p), k * 1000, 60)) for k in range(4)]
    procs.append(ctx.Process(target=_rewriter, args=(str(p), 40)))
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(60)
        assert pr.exitcode == 0
    res = js.read_jsonl(p)
    assert res.bad == 0
    assert len({e["id"] for e in res.entries}) == len(res.entries) == 240


def test_lock_timeout_raises_and_fail_open_reads(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_hold_lock_then_rewrite, args=(str(p), ready, 1.5))
    proc.start()
    assert ready.wait(20)
    with pytest.raises(TimeoutError):
        js.append_jsonl(p, [_row(1)], timeout=0.2)
    t0 = time.monotonic()
    with js.locked(p, timeout=0.2, fail_open=True):
        pass
    assert time.monotonic() - t0 < 1.0
    proc.join(20)


# ── iris_memory ─────────────────────────────────────────────────────────────

def test_iris_memory_add_search_delete(tmp_path: Path, monkeypatch) -> None:
    from brain import iris_memory, iris_semantic_memory
    monkeypatch.setattr(iris_semantic_memory, "upsert", lambda e: False)
    monkeypatch.setattr(iris_semantic_memory, "delete", lambda i: False)
    mem = iris_memory.IrisMemory(tmp_path)
    a = mem.add("Zeke likes green tea")
    b = mem.add("Zeke likes oolong tea")
    # a reconsolidated version supersedes b
    js.append_jsonl(mem._path, [dict(b, id="newb", text="Zeke likes oolong tea a lot",
                                     supersedes=b["id"])])
    ids = [e["id"] for e in mem.search("oolong")]
    assert ids == ["newb"]                      # superseded entry no longer ranks
    assert mem.count() == 3                     # but stays in the log
    # torn tail then delete: the bad bytes go to salvage, not into the void
    with open(mem._path, "ab") as f:
        f.write(b'{"id": "torn')
    assert mem.delete(a["id"]) is True
    assert {e["id"] for e in mem.list()} == {b["id"], "newb"}
    tomb = js.read_jsonl(tmp_path / "state" / "iris_memory_deleted.jsonl").entries
    assert [t["id"] for t in tomb] == [a["id"]]
    assert list((tmp_path / "state").glob("iris_memory.salvage_*_unparseable.jsonl"))


# ── iris_human_memory ───────────────────────────────────────────────────────

def test_record_revisit_upserts_and_drops_superseded(tmp_path: Path, monkeypatch) -> None:
    from brain import iris_human_memory as hm, iris_semantic_memory as sm
    calls: list[tuple] = []
    monkeypatch.setattr(sm, "upsert", lambda e: calls.append(("up", e["id"])) or True)
    monkeypatch.setattr(sm, "delete", lambda i: calls.append(("del", i)) or True)
    hm.configure(tmp_path)
    mp_ = tmp_path / "state" / "iris_memory.jsonl"
    js.append_jsonl(mp_, [{"id": "orig", "ts": 1.0, "text": "the cat sat on the red mat today"}])
    out = hm.record_revisit("orig", "the cat sat on the blue mat today", mode="reconsolidate")
    assert out["ok"] and out["mode"] == "moderate_pe"
    assert calls == [("up", out["new_id"]), ("del", "orig")]
    calls.clear()
    out2 = hm.record_revisit(out["new_id"], "completely unrelated words here", mode="new_trace")
    assert calls == [("up", out2["new_id"])]     # new trace: original stays searchable
    assert len(js.read_jsonl(mp_).entries) == 3


def test_record_accesses_saves_once_per_search(tmp_path: Path, monkeypatch) -> None:
    from brain import iris_human_memory as hm
    hm.configure(tmp_path)
    monkeypatch.setattr(hm, "_G_REF", None)
    monkeypatch.setattr(hm, "_last_meta_prune", time.time())  # no prune in this test
    saves = []
    real = hm._save_meta
    monkeypatch.setattr(hm, "_save_meta", lambda m: (saves.append(1), real(m)))
    r = hm.record_accesses(["a", "b", "c", "a"])
    assert r["ok"] and len(r["results"]) == 3
    assert len(saves) == 1
    single = hm.record_access("a")
    assert single["ok"] and single["access_count"] == 2 and "importance_boost" in single
    meta = json.loads((tmp_path / "state" / "iris_memory_meta.json").read_text())
    assert meta["a"]["access_count"] == 2 and meta["b"]["access_count"] == 1


def test_meta_prune_is_conservative(tmp_path: Path) -> None:
    from brain import iris_human_memory as hm
    now = time.time()
    old = now - 200 * 86400
    meta = {f"live{i}": {"last_accessed_ts": old} for i in range(5)}
    meta["dead_old"] = {"last_accessed_ts": old}
    meta["dead_recent"] = {"last_accessed_ts": now - 86400}
    live = {f"live{i}" for i in range(5)}
    assert hm._prune_meta(meta, now, live) == 1
    assert "dead_old" not in meta and "dead_recent" in meta and "live0" in meta
    # a prune that would wipe most rows (e.g. a bad read) is refused
    meta2 = {f"x{i}": {"last_accessed_ts": old} for i in range(10)}
    assert hm._prune_meta(meta2, now, {"x0"}) == 0 and len(meta2) == 10
    # unreadable / bad-lined log => no prune
    hm.configure(tmp_path)
    (tmp_path / "state" / "iris_memory.jsonl").write_bytes(b"not json\n")
    assert hm._prune_meta(dict(meta2), now) == 0


# ── scripts/memory_decay.py ─────────────────────────────────────────────────

def test_memory_decay_commit_keeps_concurrent_adds(tmp_path: Path) -> None:
    mem_p = tmp_path / "iris_memory.jsonl"
    arch_p = tmp_path / "iris_memory_archive.jsonl"
    old = [{"id": f"old{i}", "ts": 1.0, "importance": 0.0, "base_importance": 0.01,
            "text": f"old {i}"} for i in range(300)]
    keep = [{"id": "perm", "ts": 1.0, "importance_level": "permanent", "text": "keep"}]
    js.append_jsonl(mem_p, old + keep)
    added: list[str] = []
    stop = threading.Event()

    def adder() -> None:
        i = 0
        while not stop.is_set():
            js.append_jsonl(mem_p, [{"id": f"new{i}", "ts": time.time(), "text": "new"}])
            added.append(f"new{i}")
            i += 1
            time.sleep(0.002)

    t = threading.Thread(target=adder)
    t.start()
    env = dict(os.environ, IRIS_DISABLE_LOGGING="1")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "memory_decay.py"), "--commit",
                        "--memory-path", str(mem_p), "--archive-path", str(arch_p)],
                       capture_output=True, text=True, timeout=120, env=env)
    time.sleep(0.05)
    stop.set()
    t.join(5)
    assert r.returncode == 0, r.stderr
    live = {e["id"] for e in js.read_jsonl(mem_p).entries}
    arch = {e["id"] for e in js.read_jsonl(arch_p).entries}
    assert set(added) <= live                 # nothing written during decay was lost
    assert {f"old{i}" for i in range(300)} == arch
    assert "perm" in live and not (live & arch)
    assert not [x for x in tmp_path.iterdir() if x.name.endswith(".tmp")]


# ── iris_semantic_memory reconcile ──────────────────────────────────────────

class _FakeEF:
    """Deterministic, CPU-trivial embedding so the test never loads MiniLM."""

    def __call__(self, input):  # noqa: A002 - chroma's signature
        out = []
        for t in input:
            v = [0.0] * 8
            for k, ch in enumerate(str(t)):
                v[k % 8] += (ord(ch) % 13) / 13.0
            out.append(v)
        return out

    def embed_query(self, input):  # noqa: A002
        return self(input)

    @staticmethod
    def name() -> str:
        return "fake-test-ef"

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config):
        return _FakeEF()


@pytest.fixture()
def fake_chroma(tmp_path: Path, monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    from brain import iris_semantic_memory as sm
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection("memtest", embedding_function=_FakeEF())
    monkeypatch.setattr(sm, "_BASE", tmp_path)
    monkeypatch.setattr(sm, "_CLIENT", client)
    monkeypatch.setattr(sm, "_COLLECTION", col)
    monkeypatch.setattr(sm, "_ID_PAGE", 3)      # force several pages
    (tmp_path / "state").mkdir(exist_ok=True)
    return sm, col, tmp_path


def test_upsert_tolerates_non_string_tags(fake_chroma) -> None:
    sm, col, _ = fake_chroma
    assert sm.upsert({"id": "t1", "text": "hello", "tags": ["a", 7, None, ""]}) is True
    assert col.get(ids=["t1"])["metadatas"][0]["tags"] == "a,7,None"


def test_reconcile_pages_deletes_known_stale_and_keeps_orphans(fake_chroma) -> None:
    sm, col, base = fake_chroma
    state = base / "state"
    rows = [_row(i) for i in range(10)]
    rows.append(dict(_row(10), supersedes="id0001"))       # id0001 superseded
    js.append_jsonl(state / "iris_memory.jsonl", rows)
    # Chroma already has 0..7 plus: archived, deleted, and an unexplained orphan
    for e in rows[:8] + [_row(50), _row(51), _row(52)]:
        assert sm.upsert(e)
    js.append_jsonl(state / "iris_memory_archive.jsonl", [_row(50)])
    js.append_jsonl(state / "iris_memory_deleted.jsonl", [{"id": "id0051"}])
    assert sm._all_ids() == {r["id"] for r in rows[:8]} | {"id0050", "id0051", "id0052"}
    added = sm.reconcile_from_jsonl()
    assert added == 3                                      # 8, 9, 10
    ids = sm._all_ids()
    assert ids == ({r["id"] for r in rows} - {"id0001"}) | {"id0052"}
    orphans = js.read_jsonl(state / "iris_memory.chroma_orphans.jsonl").entries
    assert [o["id"] for o in orphans] == ["id0052"] and orphans[0]["text"] == "memory 52"
    assert sm.reconcile_from_jsonl() == 0                  # idempotent, no re-embed


def test_reconcile_deletes_nothing_when_listing_fails(fake_chroma, monkeypatch) -> None:
    sm, col, base = fake_chroma
    js.append_jsonl(base / "state" / "iris_memory.jsonl", [_row(1)])
    sm.upsert(_row(99))
    js.append_jsonl(base / "state" / "iris_memory_archive.jsonl", [_row(99)])
    monkeypatch.setattr(sm, "_all_ids", lambda: None)
    sm.reconcile_from_jsonl()
    assert "id0099" in set(col.get(include=[])["ids"])


def test_search_caps_k_and_batches_access(fake_chroma, monkeypatch) -> None:
    sm, col, _ = fake_chroma
    from brain import iris_human_memory as hm
    seen: list[list[str]] = []
    monkeypatch.setattr(hm, "record_accesses", lambda ids, g=None: seen.append(list(ids)))
    for i in range(60):
        sm.upsert(_row(i))
    out = sm.search("memory", k=10_000)
    assert len(out) == sm._MAX_K
    assert len(seen) == 1 and len(seen[0]) == sm._MAX_K
