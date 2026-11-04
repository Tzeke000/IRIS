"""11-04: V3 4.1 - the diagnostic evidence index (scripts/server/diag_corpus.py), on a tiny fake tree.

No Ollama, no real state/: a tmp src root with docs/, tests/, a bundle, a health history and a fake spec.
"""
import importlib.util
import json
from pathlib import Path

SRV = Path(__file__).resolve().parents[1] / "scripts" / "server"


def _mod():
    spec = importlib.util.spec_from_file_location("diag_corpus_t", SRV / "diag_corpus.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _tree(tmp_path):
    src = tmp_path / "src"
    (src / "docs").mkdir(parents=True)
    (src / "docs" / "RUNBOOK.md").write_text("# Runbook\n\n## Restore\nrestore from the tower backup with rsync\n\n"
                                              "## Mouth\nthe mouth speaks through the audio sink\n")
    (src / "docs" / "OTHER.md").write_text("# Other\n\n## Cats\ncats sleep a lot\n")
    (src / "tests").mkdir()
    (src / "tests" / "test_widget.py").write_text('"""widget admission tests"""\n'
                                                  'def test_refuses_when_full():\n    """full = no"""\n')
    b = src / "state" / "incidents" / "20261104T000000_ears-red"
    b.mkdir(parents=True)
    (b / "manifest.json").write_text(json.dumps({"id": b.name, "reason": "ears red: ping failed", "subsystem": "ears"}))
    (b / "journal_iris-ears.log").write_text("ConnectionRefusedError talking to the mic\n")
    (b / "procs.txt").write_text("cats cats cats\n")                 # skipped: noise
    (src / "state" / "incidents" / ".tmp_half").mkdir()
    (src / "state" / "incidents" / ".tmp_half" / "manifest.json").write_text("{}")
    (src / "state" / "health").mkdir(parents=True)
    (src / "state" / "health" / "history.jsonl").write_text(
        '{"ts": 1, "red": [], "warn": ["perception"]}\n{"ts": 2, "red": ["ears"], "warn": []}\n')
    spec = tmp_path / "spec.md"
    spec.write_text("# spec\n\n## Liveness baseline\nruntime rss 900 MB, 3 processes\n\n## Run note x\nall green\n")
    sandbox = tmp_path / "iris-fi-abc" / "fallback" / "20261104T000001_storage-red"
    sandbox.mkdir(parents=True)
    (sandbox / "manifest.json").write_text(json.dumps({"id": sandbox.name, "reason": "storage red: 100% full",
                                                       "subsystem": "storage", "primary_failed": "ENOSPC"}))
    return src, spec


def _index(tmp_path):
    m = _mod()
    src, spec = _tree(tmp_path)
    docs = m.collect(src=src, spec=spec, sandbox_glob=str(tmp_path / "iris-fi-*"))
    return m, docs, m.build_index(docs)


def test_tokenizer_keeps_names_whole_and_split():
    m = _mod()
    t = m.tokenize("gpu_admit refused the loads")
    assert "gpu_admit" in t and "gpu" in t and "admit" in t and "load" in t and "the" not in t


def test_collects_every_source_and_skips_half_bundles(tmp_path):
    _, docs, idx = _index(tmp_path)
    assert idx["sources"]["docs"] == 3 and idx["sources"]["tests"] == 1
    assert idx["sources"]["incident"] == 2 and idx["sources"]["sandbox_incident"] == 1   # summary + evidence; summary only
    assert idx["sources"]["selfcheck"] == 3                                               # 2 spec sections + history
    assert not any(".tmp_half" in d["id"] for d in docs)
    ev = next(d for d in docs if d["id"].endswith("#evidence"))
    assert "ConnectionRefusedError" in ev["text"] and "cats" not in ev["text"]
    hist = next(d for d in docs if d["id"] == "selfcheck:health_history")
    assert "probe ears went red 1 times" in hist["text"]


def test_query_finds_right_doc_and_collapses_per_path(tmp_path):
    m, _, idx = _index(tmp_path)
    assert m.search(idx, "how do I restore from the tower backup")[0]["path"] == "docs/RUNBOOK.md"
    assert m.search(idx, "mic connection refused, ears ping failed")[0]["path"].startswith("bundle:20261104T000000")
    assert m.search(idx, "state disk full no space")[0]["path"].startswith("bundle:20261104T000001")
    assert m.search(idx, "which test refuses when full")[0]["path"] == "tests/test_widget.py"
    paths = [h["path"] for h in m.search(idx, "runbook restore mouth tower", k=5)]
    assert len(paths) == len(set(paths))
    assert len(m.search(idx, "runbook restore mouth tower", k=5, per_path=False)) >= 2


def test_save_load_and_eval(tmp_path):
    m, _, idx = _index(tmp_path)
    p = m.save(idx, tmp_path / "out" / "corpus.json")
    back = m.load(p)
    assert back["n"] == idx["n"]
    rep = m.evaluate(back, [{"q": "restore from tower backup", "expect": ["RUNBOOK"]},
                            {"q": "zebras", "expect": ["nothing"]}])
    assert rep["hit_at_1"] == 1 and rep["hit_at_5"] == 1 and rep["rows"][1]["rank"] is None


def test_embed_gate_never_pulls_and_respects_gpu_admit():
    m = _mod()
    no_model = lambda: {"models": [{"name": "qwen2.5:0.5b"}]}
    ok, why = m.embed_gate(tags_fn=no_model, admit_fn=lambda *a, **k: (True, {}))
    assert not ok and "not pulled" in why
    have = lambda: {"models": [{"name": "nomic-embed-text:latest"}]}
    ok, why = m.embed_gate(tags_fn=have, admit_fn=lambda *a, **k: (False, {"free_mib": 10}))
    assert not ok and "gpu_admit refused" in why
    ok, _ = m.embed_gate(tags_fn=have, admit_fn=lambda *a, **k: (True, {}))
    assert ok


def test_embedder_refuses_other_models():
    m = _mod()
    for bad in ("iris-little-v16", "qwen2.5:7b"):
        try:
            m.ollama_embedder(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_rrf_fusion_with_fake_embedder(tmp_path):
    m, docs, idx = _index(tmp_path)
    # fake vectors: only the Cats section points the same way as the query
    idx["embed"] = {"model": "fake", "vectors": [[1.0, 0.0] if "Cats" in d["title"] else [0.0, 1.0] for d in docs]}
    hits = m.search(idx, "feline naps", embedder=lambda texts: [[1.0, 0.0]])
    assert hits[0]["path"] == "docs/OTHER.md"           # BM25 alone finds nothing for this question
    assert m.search(idx, "feline naps") == []
