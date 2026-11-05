"""memory_correct — in-place correction of a memory in the JSONL log + Chroma.

Everything runs on a tmp JSONL + tmp Chroma dir with a fake embedder — the
live state/ and memory/chroma are never touched.

    /home/iris/venvs/iris-v100/bin/python -m pytest tests/test_memory_correct.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import _jsonl_store as js  # noqa: E402
from tests.test_memory_store import _FakeEF  # noqa: E402


@pytest.fixture()
def store(tmp_path: Path, monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    from brain import iris_semantic_memory as sm, memory_correct as mc
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection("memtest", embedding_function=_FakeEF())
    monkeypatch.setattr(sm, "_BASE", tmp_path)
    monkeypatch.setattr(sm, "_CLIENT", client)
    monkeypatch.setattr(sm, "_COLLECTION", col)
    (tmp_path / "state").mkdir()
    path = tmp_path / "state" / "iris_memory.jsonl"
    rows = [{"id": f"m{i}", "ts": 1000.0 + i, "text": f"memory number {i}",
             "person_id": "zeke", "tags": ["t"]} for i in range(4)]
    js.append_jsonl(path, rows)
    for r in rows:
        assert sm.upsert(r)
    return mc, sm, col, tmp_path, path


def _doc(col, eid):
    got = col.get(ids=[eid], include=["documents", "embeddings"])
    if not got["ids"]:
        return None, None
    return got["documents"][0], list(got["embeddings"][0])


def test_show_side_by_side(store) -> None:
    mc, sm, col, base, path = store
    r = mc.show("m1", base=base)
    assert r["ok"] and r["match"] and r["jsonl_text"] == r["chroma_text"] == "memory number 1"
    missing = mc.show("nope", base=base)
    assert missing["ok"] and not missing["in_jsonl"] and not missing["in_chroma"]


def test_correct_updates_both_and_keeps_audit(store) -> None:
    mc, sm, col, base, path = store
    _, emb_before = _doc(col, "m2")
    r = mc.correct("m2", "the corrected memory", "invented premise", base=base)
    assert r["ok"] and r["jsonl_changed"] and r["chroma_reembedded"]
    rows = {e["id"]: e for e in js.read_jsonl(path).entries}
    assert rows["m2"]["text"] == "the corrected memory"
    audit = rows["m2"]["corrections"]
    assert audit[-1]["old_text"] == "memory number 2" and audit[-1]["reason"] == "invented premise"
    assert rows["m1"]["text"] == "memory number 1" and len(rows) == 4   # nothing else touched
    doc, emb_after = _doc(col, "m2")
    assert doc == "the corrected memory" and emb_after != emb_before    # re-embedded
    assert mc.show("m2", base=base)["match"]
    # second correction appends to the trail
    mc.correct("m2", "corrected again", "typo", base=base)
    audit = {e["id"]: e for e in js.read_jsonl(path).entries}["m2"]["corrections"]
    assert [a["old_text"] for a in audit] == ["memory number 2", "the corrected memory"]


def test_correct_requires_reason_and_refuses_superseded(store) -> None:
    mc, sm, col, base, path = store
    assert not mc.correct("m0", "x", "", base=base)["ok"]
    js.append_jsonl(path, [{"id": "m0v2", "ts": 2000.0, "text": "v2", "supersedes": "m0"}])
    r = mc.correct("m0", "x", "why", base=base)
    assert not r["ok"] and "m0v2" in r["error"]
    assert not mc.correct("ghost", "x", "why", base=base)["ok"]


def test_resync_one_fixes_stale_chroma_when_jsonl_already_fixed(store) -> None:
    """The live gap: JSONL hand-corrected, Chroma still holds the old text."""
    mc, sm, col, base, path = store
    rows = js.read_jsonl(path).entries
    for e in rows:
        if e["id"] == "m3":
            e["text"] = "fixed by hand"
    js.rewrite_jsonl(path, rows)
    assert not mc.show("m3", base=base)["match"]
    dry = mc.resync("m3", base=base, dry_run=True)
    assert dry["resynced"] == 1 and _doc(col, "m3")[0] == "memory number 3"   # dry = no write
    r = mc.resync("m3", base=base)
    assert r["ok"] and r["resynced"] == 1 and r["items"][0]["was"] == "differs"
    assert r["items"][0]["chroma_old"] == "memory number 3"
    assert _doc(col, "m3")[0] == "fixed by hand"
    assert mc.resync("m3", base=base)["resynced"] == 0                         # idempotent


def test_resync_all_fixes_drift_and_missing_skips_superseded(store) -> None:
    mc, sm, col, base, path = store
    rows = js.read_jsonl(path).entries
    rows[0]["text"] = "drifted 0"
    js.rewrite_jsonl(path, rows)
    col.delete(ids=["m1"])                                                     # missing
    js.append_jsonl(path, [{"id": "m2v2", "ts": 3000.0, "text": "v2 of 2", "supersedes": "m2"}])
    r = mc.resync("all", base=base)
    assert r["ok"]
    got = {i["id"]: i["was"] for i in r["items"]}
    assert got == {"m0": "differs", "m1": "missing", "m2v2": "missing"}
    assert _doc(col, "m0")[0] == "drifted 0" and _doc(col, "m1")[0] == "memory number 1"
    assert _doc(col, "m2")[0] == "memory number 2"   # superseded row left for boot reconcile
    assert not mc.resync("m2", base=base)["ok"]
    assert not mc.resync("", base=base)["ok"]        # 'all' must be explicit


def test_tool_registers_and_dispatches(store, monkeypatch) -> None:
    mc, sm, col, base, path = store
    from brain import iris_memory
    monkeypatch.setattr(iris_memory, "_SINGLETON", iris_memory.IrisMemory(base))
    import importlib
    from tools import tool_registry as tr
    importlib.import_module("tools.system.memory_correct_tool")
    h = tr._REGISTRY["memory_correct"].handler
    assert h({"action": "show", "id": "m1"}, {})["match"]
    assert h({"action": "correct", "id": "m1", "new_text": "new", "reason": "r"}, {})["ok"]
    assert _doc(col, "m1")[0] == "new"
    assert h({"action": "resync", "id": "all"}, {})["resynced"] == 0
    assert not h({"action": "bogus"}, {})["ok"]
