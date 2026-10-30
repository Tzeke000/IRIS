"""
brain/iris_semantic_memory.py — semantic search layer over iris_memory.

iris_memory.jsonl is the durable canonical log. This module mirrors entries
into a ChromaDB collection so I can do similarity search ("what did I learn
about Zeke's sleep schedule?") rather than just substring match.

Embeddings: Chroma's bundled all-MiniLM-L6-v2 ONNX (CPU). 80MB cached at
~/.cache/chroma/. No Ollama, no API keys, no extra pip installs.

Sync model: write-through. Every iris_memory.add() / record_revisit() also
upserts into Chroma (and iris_memory.delete() deletes). On boot, we reconcile
both ways — JSONL entries missing from Chroma get backfilled, and Chroma IDs
that are no longer live in the JSONL (deleted, archived by memory_decay, or
superseded by a reconsolidated version) get deleted; unexplained orphans
are kept and exported for review (see reconcile_from_jsonl).

Public API:
  configure(base_dir)              — bind paths, idempotent
  upsert(entry)                    — add one memory (called by hook)
  search(query, k=5) -> list[dict] — semantic search, returns iris_memory rows
  reconcile_from_jsonl()           — rebuild collection from JSONL on boot
  delete(memory_id)                — remove from collection (mirrors iris_memory.delete)
  count()                          — collection size

Why not mem0: mem0 wants either Ollama or OpenAI for both LLM extraction
AND embedder config. We don't have Ollama on this machine and we don't
want a paid API. Chroma's bundled embedder gives us semantic search on
day one with zero extra deps. The "extraction" half (deciding what to
remember) is me — through inner_monologue and explicit memory_remember
MCP calls.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional


_BASE: Path | None = None
_CLIENT: Any = None  # chromadb.PersistentClient
_COLLECTION: Any = None  # chromadb.Collection
_LOCK = threading.Lock()
_COLLECTION_NAME = "iris_memories"
_MAX_K = 50            # search() result cap (callers pass k straight through)
_ID_PAGE = 1000        # page size for listing Chroma IDs in reconcile


def _chroma_dir() -> Path:
    base = _BASE if _BASE is not None else Path(".")
    return base / "memory" / "chroma"


def configure(base_dir: Path | str) -> None:
    """Bind paths. Call once from iris_bootstrap. Idempotent."""
    global _BASE
    _BASE = Path(base_dir)
    _chroma_dir().mkdir(parents=True, exist_ok=True)


def _ensure_client() -> bool:
    """Lazy-init the chroma client + collection. Returns True if ready."""
    global _CLIENT, _COLLECTION
    if _COLLECTION is not None:
        return True
    if _BASE is None:
        return False
    try:
        import chromadb  # type: ignore
    except ImportError:
        print("[iris_semantic_memory] chromadb not installed — semantic search disabled")
        return False
    try:
        with _LOCK:
            if _COLLECTION is not None:
                return True
            _CLIENT = chromadb.PersistentClient(path=str(_chroma_dir()))
            _COLLECTION = _CLIENT.get_or_create_collection(_COLLECTION_NAME)
            return True
    except Exception as e:
        print(f"[iris_semantic_memory] init error: {e!r}")
        return False


def upsert(entry: dict[str, Any]) -> bool:
    """Add a memory entry to the semantic index. Idempotent — calling twice
    with the same id replaces."""
    if not _ensure_client():
        return False
    try:
        eid = str(entry.get("id") or "")
        text = str(entry.get("text") or "")
        if not eid or not text:
            return False
        meta = {
            "ts": float(entry.get("ts") or 0.0),
            "iso": str(entry.get("iso") or ""),
            "person_id": str(entry.get("person_id") or ""),
            "category": str(entry.get("category") or ""),
            "importance": float(entry.get("importance") or 0.0),
            "source": str(entry.get("source") or ""),
            # str() each tag: one non-string tag used to raise here, so the
            # whole memory was never indexed.
            "tags": ",".join(str(t) for t in (entry.get("tags") or [])
                             if str(t).strip()),
        }
        with _LOCK:
            _COLLECTION.upsert(
                ids=[eid],
                documents=[text],
                metadatas=[meta],
            )
        return True
    except Exception as e:
        print(f"[iris_semantic_memory] upsert error: {e!r}")
        return False


def search(query: str, k: int = 5,
           person_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Semantic similarity search. Returns iris_memory-shaped rows
    (id, text, ts, etc.) ordered by relevance.

    Phase 45: every hit triggers record_access — retrieval strengthens
    the memory per Bjork's New Theory of Disuse (each successful recall
    bumps both retrieval-strength and storage-strength).
    """
    if not _ensure_client():
        return []
    if not (query or "").strip():
        return []
    try:
        where = {"person_id": person_id} if person_id else None
        with _LOCK:
            res = _COLLECTION.query(
                query_texts=[str(query)],
                n_results=int(max(1, min(_MAX_K, int(k)))),
                where=where,
            )
        # Chroma returns dict of lists; we get the first row.
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[dict[str, Any]] = []
        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            tags_str = str(meta.get("tags") or "")
            tags = [t for t in tags_str.split(",") if t]
            out.append({
                "id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "ts": float(meta.get("ts") or 0.0),
                "iso": str(meta.get("iso") or ""),
                "person_id": str(meta.get("person_id") or ""),
                "category": str(meta.get("category") or ""),
                "importance": float(meta.get("importance") or 0.0),
                "source": str(meta.get("source") or ""),
                "tags": tags,
                "distance": float(dists[i]) if i < len(dists) else 0.0,
            })
        # Iris_fixes #7: memory usefulness → retrieval ORDER. Blend chroma distance with
        # each hit's effective (access-boosted, decayed) importance; blend strength is the
        # learned memory_usefulness weight, and with no evidence the order is untouched.
        try:
            from brain import adaptive_iris
            out = adaptive_iris.rerank_memories(out)
        except Exception:
            pass
        # Phase 45: retrieval strengthens. Bump access metadata for each
        # hit. Doesn't fail the search if record_access errors.
        try:
            from brain import iris_human_memory
            hit_ids = [entry["id"] for entry in out if entry.get("id")]
            if hit_ids:
                # One meta load + one atomic save for the whole search.
                iris_human_memory.record_accesses(hit_ids)
        except Exception:
            pass
        return out
    except Exception as e:
        print(f"[iris_semantic_memory] search error: {e!r}")
        return []


def delete(memory_id: str) -> bool:
    """Remove from collection. Returns True if anything was deleted."""
    if not _ensure_client():
        return False
    try:
        with _LOCK:
            _COLLECTION.delete(ids=[str(memory_id)])
        return True
    except Exception as e:
        print(f"[iris_semantic_memory] delete error: {e!r}")
        return False


def count() -> int:
    if not _ensure_client():
        return 0
    try:
        with _LOCK:
            return int(_COLLECTION.count())
    except Exception:
        return 0


def _all_ids() -> Optional[set[str]]:
    """Every ID in the collection, paged (get(include=[]) with offset) so it
    neither caps out at 10k nor pulls documents/embeddings. None on error —
    callers must NOT treat an error as an empty collection."""
    ids: set[str] = set()
    offset = 0
    try:
        while True:
            with _LOCK:
                got = _COLLECTION.get(include=[], limit=_ID_PAGE, offset=offset)
            page = list(got.get("ids") or [])
            ids.update(str(i) for i in page)
            if len(page) < _ID_PAGE:
                return ids
            offset += len(page)
    except Exception as e:
        print(f"[iris_semantic_memory] id listing error: {e!r}")
        return None


def _ids_in(path: Path) -> set[str]:
    from brain import _jsonl_store
    return {str(e.get("id")) for e in _jsonl_store.read_jsonl(path, log=False).entries
            if e.get("id")}


def reconcile_from_jsonl() -> int:
    """Bring the collection in line with iris_memory.jsonl. Used at boot.

    - backfills live JSONL entries missing from Chroma (landed before chroma
      was ready);
    - deletes Chroma IDs that are no longer live FOR A KNOWN REASON:
        * superseded by a reconsolidated version (record_revisit keeps both
          in the log for audit; only the newest should rank),
        * archived by scripts/memory_decay.py (state/iris_memory_archive.jsonl),
        * deleted through iris_memory.delete (tombstone in
          state/iris_memory_deleted.jsonl — covers a delete whose Chroma
          mirror call failed).
    - Chroma IDs missing from the JSONL with NO known reason are KEPT and
      exported to state/iris_memory.chroma_orphans.jsonl. Non-fsync'd appends
      lost to power cuts (NUL-filled lines) and the old decay/rewrite race
      left memories whose only intact copy is in Chroma; deleting those
      would destroy them. Review that file and re-append what's real.

    If the JSONL can't be read at all, or the ID listing errored, nothing is
    deleted. Returns the number of entries ADDED (unchanged contract);
    deletions and orphans are logged."""
    if not _ensure_client():
        return 0
    if _BASE is None:
        return 0
    state = _BASE / "state"
    jsonl = state / "iris_memory.jsonl"
    if not jsonl.is_file():
        return 0
    try:
        from brain import _jsonl_store
        res = _jsonl_store.read_jsonl(jsonl)
        entries = res.entries
        superseded = {str(e.get("supersedes")) for e in entries if e.get("supersedes")}
        all_jsonl = {str(e.get("id")) for e in entries if e.get("id")}
        live: dict[str, dict[str, Any]] = {}
        for e in entries:
            eid = str(e.get("id") or "")
            if eid and eid not in superseded:
                live[eid] = e
        # Pull existing IDs from chroma to avoid re-embedding everything.
        existing = _all_ids()
        added = 0
        for eid, entry in live.items():
            if existing is not None and eid in existing:
                continue
            if upsert(entry):
                added += 1
        removed = 0
        orphans: list[str] = []
        if existing is not None and not res.error and entries:
            gone = existing - set(live)
            known_gone = ((gone & superseded)
                          | (gone & (_ids_in(state / "iris_memory_archive.jsonl")
                                     | _ids_in(state / "iris_memory_deleted.jsonl"))))
            # never delete something that's still a (non-superseded) JSONL row
            known_gone -= (all_jsonl - superseded)
            orphans = sorted(gone - known_gone)
            stale = sorted(known_gone)
            for i in range(0, len(stale), 500):
                chunk = stale[i:i + 500]
                try:
                    with _LOCK:
                        _COLLECTION.delete(ids=chunk)
                    removed += len(chunk)
                except Exception as e:
                    print(f"[iris_semantic_memory] reconcile delete error: {e!r}")
                    break
            if orphans:
                _export_orphans(orphans, state / "iris_memory.chroma_orphans.jsonl")
        if added or removed or orphans:
            print(f"[iris_semantic_memory] reconciled: +{added} from JSONL, "
                  f"-{removed} superseded/archived/deleted, "
                  f"{len(orphans)} orphan(s) kept + exported")
        return added
    except Exception as e:
        print(f"[iris_semantic_memory] reconcile error: {e!r}")
        return 0


def _export_orphans(ids: list[str], out_path: Path) -> None:
    """Write the full Chroma copy (id, text, metadata) of every orphan to a
    review file, atomically. Best-effort; never raises."""
    try:
        from brain import _jsonl_store
        rows: list[dict[str, Any]] = []
        for i in range(0, len(ids), 500):
            with _LOCK:
                got = _COLLECTION.get(ids=ids[i:i + 500],
                                      include=["documents", "metadatas"])
            gids = got.get("ids") or []
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            for j, gid in enumerate(gids):
                meta = dict(metas[j] or {}) if j < len(metas) else {}
                rows.append({"id": gid,
                             "text": docs[j] if j < len(docs) else "",
                             "chroma_metadata": meta})
        rows.sort(key=lambda r: float((r.get("chroma_metadata") or {}).get("ts") or 0.0))
        _jsonl_store.rewrite_jsonl(out_path, rows)
    except Exception as e:
        print(f"[iris_semantic_memory] orphan export error: {e!r}")


def bootstrap_iris_semantic_memory(g: dict[str, Any]) -> None:
    """Wire into the global state dict. Idempotent."""
    base = Path(g.get("BASE_DIR") or ".")
    configure(base)
    if _ensure_client():
        n = reconcile_from_jsonl()
        g["_iris_semantic_memory_ready"] = True
        print(f"[iris_semantic_memory] ready (count={count()}, reconciled={n})")
    else:
        g["_iris_semantic_memory_ready"] = False
