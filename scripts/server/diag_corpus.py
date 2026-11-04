#!/usr/bin/env python3
"""scripts/server/diag_corpus.py - the diagnostic helper's evidence index (V3 plan Phase 4 step 4.1, 2026-11-04).

Retrieval with NO LLM: a CPU-only BM25 index over everything a fault diagnosis would want to read.

    sources (all READ-ONLY)                         one document =
      incident bundles  state/incidents/<id>/        one bundle (manifest reason + health + journals + logs + events
                        + --bundle-dir / --sandbox-glob   + system head); fault_inject.py --keep sandbox bundles too
      docs              docs/**/*.md                  one heading section (## / ###), long ones split
      tests             tests/test_*.py               one test file: docstring + test names + last results
                        (+ .pytest_cache lastfailed, state/v3/baselines.json pytest, state/diag/last_junit.xml)
      selfcheck         memory self_cron_prompt_spec.md   one "## " section (liveness baselines, run notes, GPU watch)
                        + state/health/history.jsonl  one summary doc: which probes went warn/red, how often, when

    diag_corpus.py build [--sandbox-glob '/tmp/iris-fi-*'] [--bundle-dir D ...] [--embed]
    diag_corpus.py query "QUESTION" [-k 5] [--all-chunks] [--json] [--embed]
    diag_corpus.py eval [scripts/server/diag_eval_questions.json] [-k 5] [--json]
    diag_corpus.py stats

The index is ONE json file, state/diag/corpus.json (git-ignored with the rest of state/). `query` builds it first
if it is missing. Ranking: Okapi BM25 (k1 1.2, b 0.75) over a light tokenizer (lowercase, snake/dotted names kept
whole AND split, a tiny suffix stemmer); a doc's title counts 3x. By default hits are collapsed to the best chunk
per document path so the top-5 is five different things to read.

--embed (optional, OFF by default): adds nomic-embed-text vectors from the LOCAL Ollama and fuses its ranking with
BM25 (reciprocal-rank fusion). It is used ONLY if nomic-embed-text is ALREADY pulled AND gpu_admit.py admits
EMBED_NEED_MIB; otherwise it says why and stays BM25. This script never pulls a model and refuses any model name
that is not nomic-embed-text (iris-little-* in particular are never touched).

Env: IRIS_DIAG_SRC_ROOT (repo whose docs/tests/state are READ; default this repo), IRIS_DIAG_INDEX (index file),
IRIS_DIAG_SPEC (the self-cron spec), OLLAMA_HOST. Stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(os.environ.get("IRIS_DIAG_SRC_ROOT") or ROOT)
INDEX = Path(os.environ.get("IRIS_DIAG_INDEX") or ROOT / "state" / "diag" / "corpus.json")
SPEC = Path(os.environ.get("IRIS_DIAG_SPEC")
            or Path.home() / ".claude" / "projects" / "-home-iris-IRIS" / "memory" / "self_cron_prompt_spec.md")
EVAL_FILE = HERE / "diag_eval_questions.json"
VERSION = 1

K1, B = 1.2, 0.75
TITLE_WEIGHT = 3
CHUNK_CHARS = 3500          # a docs section longer than this is split on paragraphs
BUNDLE_FILE_CHARS = 6000    # per bundle file kept in the doc (tail: the end of a log is the interesting part)
SPEC_SECTION_CHARS = 6000

EMBED_MODEL = "nomic-embed-text"
EMBED_NEED_MIB = 700        # nomic-embed-text v1.5 f16 ~274 MB weights + context/KV + runtime; generous on purpose
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA

STOP = set("""a an the and or of to in on at for from by with is are was were be been being it its this that these
those i me my we our you your he she they them his her their what which who whom why how when where do does did done
not no but if then than so as into out up down over under again there here can could should would will just only
also very any all some more most other such own same too s t about after before once ever even""".split())


# ── tokenizer ───────────────────────────────────────────────────────────────
_WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]*[a-z0-9]|[a-z0-9]")


def _stem(w: str) -> str:
    if len(w) <= 4 or w.isdigit():
        return w
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: len(w) - len(suf)] + rep
            break
    return w


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _WORD.finditer(text.lower()):
        tok = m.group(0)
        parts = [p for p in re.split(r"[_.\-]+", tok) if p]
        if len(parts) > 1 and len(tok) <= 40:
            out.append(tok)                     # gpu_admit / worker.py kept whole ...
        for p in parts:                          # ... and split
            if p not in STOP and not (p.isdigit() and len(p) > 6):
                out.append(_stem(p))
    return out


# ── sources ─────────────────────────────────────────────────────────────────
def _read(p: Path, limit: int | None = None, tail: bool = False) -> str:
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if limit and len(t) > limit:
        t = t[-limit:] if tail else t[:limit]
    return t


def _doc(id_: str, source: str, path: str, title: str, text: str, **meta) -> dict:
    return {"id": id_, "source": source, "path": path, "title": title.strip(), "text": text, **meta}


def _split_long(text: str, cap: int) -> list[str]:
    if len(text) <= cap:
        return [text]
    chunks, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > cap:
            chunks.append(cur)
            cur = ""
        cur = (cur + "\n\n" + para) if cur else para
        while len(cur) > cap:                    # a single huge paragraph (a table)
            chunks.append(cur[:cap])
            cur = cur[cap:]
    if cur.strip():
        chunks.append(cur)
    return chunks


def md_sections(text: str, levels: tuple[int, ...] = (2, 3)) -> list[tuple[str, str]]:
    """[(heading path, body)] split on the given heading levels; text before the first heading is '(top)'."""
    pat = re.compile(r"^(#{1,6})\s+(.*)$")
    out, head, body, stack = [], "(top)", [], {}
    for line in text.splitlines():
        m = pat.match(line)
        if m and len(m.group(1)) in levels:
            if "".join(body).strip():
                out.append((head, "\n".join(body)))
            lvl = len(m.group(1))
            stack = {k: v for k, v in stack.items() if k < lvl}
            stack[lvl] = m.group(2).strip()
            head = " > ".join(stack[k] for k in sorted(stack))
            body = []
        else:
            body.append(line)
    if "".join(body).strip():
        out.append((head, "\n".join(body)))
    return out


def docs_docs(src: Path) -> list[dict]:
    out = []
    for p in sorted((src / "docs").rglob("*.md")):
        rel = p.relative_to(src).as_posix()
        text = _read(p)
        h1 = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), p.stem)
        for i, (head, body) in enumerate(md_sections(text)):
            if not re.sub(r"(?m)^# .*$", "", body).strip():
                continue                         # just the H1 title line: already in every chunk's title
            for j, chunk in enumerate(_split_long(body, CHUNK_CHARS)):
                out.append(_doc(f"{rel}#{i}.{j}", "docs", rel, f"{p.stem} {h1} {head}", chunk, section=head))
    return out


BUNDLE_SKIP = {"procs.txt", "git.txt"}          # top-RSS lists + git status: noise for retrieval, still in the bundle


def bundle_docs_one(bdir: Path, source: str) -> list[dict]:
    """A bundle becomes TWO chunks under one path: a short SUMMARY (manifest reason, subsystem, primary_failed,
    collector errors, the red/warn probes of its health.json) and the EVIDENCE (the file tails). Without the split
    the reason line drowns in the machine snapshot every bundle shares (system.txt), and BM25's length
    normalisation buries exactly the line that names the fault."""
    man_p = bdir / "manifest.json"
    if not man_p.is_file():
        return []
    try:
        man = json.loads(_read(man_p))
    except ValueError:
        man = {}
    bid = man.get("id") or bdir.name
    path = f"bundle:{bid}"
    title = f"incident {man.get('subsystem', '')} {man.get('reason', '')}"
    summary = [f"reason: {man.get('reason', '')}", f"subsystem: {man.get('subsystem', '')}",
               f"role: {man.get('role', '')}  iso: {man.get('iso', '')}"]
    if man.get("primary_failed"):
        summary.append(f"primary bundle dir failed: {man['primary_failed']}")
    errs = man.get("collector_errors") or man.get("errors")
    if errs:
        summary.append("collector errors: " + json.dumps(errs)[:800])
    try:
        health = json.loads(_read(bdir / "health.json") or "{}")
    except ValueError:
        health = {}
    for lvl in ("red", "warn"):
        if health.get(lvl):
            summary.append(f"health {lvl}: {json.dumps(health[lvl])[:600]}")
    evidence = []
    for f in sorted(bdir.iterdir()):
        if f.name == "manifest.json" or f.name in BUNDLE_SKIP or not f.is_file():
            continue
        body = _read(f, BUNDLE_FILE_CHARS, tail=True).strip()
        if body:
            evidence.append(f"--- {f.name}\n{body}")
    meta = {"bundle_dir": str(bdir), "ts": man.get("ts")}
    out = [_doc(f"{path}#summary", source, path, title, "\n".join(summary), **meta)]
    if evidence:
        out.append(_doc(f"{path}#evidence", source, path, f"incident {man.get('subsystem', '')} evidence",
                        "\n".join(evidence), **meta))
    return out


def find_bundle_dirs(roots: list[Path]) -> list[Path]:
    """Every dir under each root that holds a manifest.json (bundles sit 1-3 levels deep: incidents/<id>,
    a sandbox's state/incidents/<id> or fallback/<id>). .tmp_* half-bundles are skipped."""
    seen, out = set(), []
    for r in roots:
        if not r.is_dir():
            continue
        for man in sorted(r.glob("**/manifest.json")):
            d = man.parent
            if d.name.startswith(".tmp") or len(d.relative_to(r).parts) > 4 or d in seen:
                continue
            seen.add(d)
            out.append(d)
    return out


def bundle_docs(src: Path, extra_dirs: list[Path], sandbox_glob: str | None) -> list[dict]:
    out = []
    for d in find_bundle_dirs([src / "state" / "incidents"]):
        out.extend(bundle_docs_one(d, "incident"))
    sandbox_roots = list(extra_dirs) + ([Path(p) for p in sorted(glob.glob(sandbox_glob))] if sandbox_glob else [])
    for d in find_bundle_dirs(sandbox_roots):
        out.extend(bundle_docs_one(d, "sandbox_incident"))
    return out


def _junit_results(xml_path: Path) -> dict[str, str]:
    """{'tests/test_x.py::test_y': 'passed'|'failed'|'error'|'skipped'} from a junit xml, if present."""
    res: dict[str, str] = {}
    if not xml_path.is_file():
        return res
    try:
        import xml.etree.ElementTree as ET
        for tc in ET.parse(xml_path).getroot().iter("testcase"):
            cls, name = tc.get("classname", ""), tc.get("name", "")
            mod = cls.split(".")
            # pytest junit: classname "tests.test_x" or "tests.test_x.TestClass"
            fi = next((i for i, m in enumerate(mod) if m.startswith("test_")), len(mod) - 1)
            path = "/".join(mod[: fi + 1]) + ".py"
            key = f"{path}::{'::'.join(mod[fi + 1:] + [name])}"
            state = "passed"
            for child in tc:
                if child.tag in ("failure", "error", "skipped"):
                    state = {"failure": "failed"}.get(child.tag, child.tag)
            res[key] = state
    except Exception:
        pass
    return res


def test_docs(src: Path) -> list[dict]:
    import ast
    lastfailed: dict = {}
    try:
        lastfailed = json.loads(_read(src / ".pytest_cache" / "v" / "cache" / "lastfailed") or "{}")
    except ValueError:
        pass
    junit = _junit_results(ROOT / "state" / "diag" / "last_junit.xml")
    if not junit and SRC_ROOT != ROOT:
        junit = _junit_results(SRC_ROOT / "state" / "diag" / "last_junit.xml")
    summary = ""
    try:
        pt = json.loads(_read(src / "state" / "v3" / "baselines.json") or "{}").get("pytest") or {}
        if pt:
            summary = (f"suite baseline (regress_gate): {pt.get('passed')} passed / {pt.get('tests')} tests, "
                       f"{pt.get('skipped')} skipped, recorded {pt.get('recorded_iso')}")
    except ValueError:
        pass
    out = []
    for p in sorted((src / "tests").glob("test_*.py")):
        rel = p.relative_to(src).as_posix()
        text = _read(p)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = [ast.get_docstring(tree) or ""]
        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                names.append(node.name)
                ds = ast.get_docstring(node)
                key_res = [v for k, v in junit.items() if k.startswith(rel + "::") and k.endswith("::" + node.name)]
                state = key_res[0] if key_res else ("failed (lastfailed)" if any(
                    k.startswith(rel) and k.endswith(node.name) for k in lastfailed) else "")
                lines.append(f"{node.name} {node.name.replace('_', ' ')}"
                             + (f" [{state}]" if state else "") + (f" - {ds.splitlines()[0]}" if ds else ""))
        if not names:
            continue
        if junit:
            mine = [v for k, v in junit.items() if k.startswith(rel + "::")]
            c = Counter(mine)
            lines.append("last results (junit): " + ", ".join(f"{n} {s}" for s, n in sorted(c.items())))
        if summary:
            lines.append(summary)
        out.append(_doc(f"test:{rel}", "tests", rel, f"{p.stem} {p.stem.replace('_', ' ')}", "\n".join(lines),
                        n_tests=len(names)))
    return out


def selfcheck_docs(src: Path, spec: Path) -> list[dict]:
    out = []
    text = _read(spec)
    for i, (head, body) in enumerate(md_sections(text, levels=(2,))):
        if head == "(top)":
            continue
        body = body[:SPEC_SECTION_CHARS]
        out.append(_doc(f"selfcheck:spec#{i}", "selfcheck", f"selfcheck:spec#{i}", f"self-check {head}", body,
                        section=head))
    hist = src / "state" / "health" / "history.jsonl"
    if hist.is_file():
        probes: dict[tuple[str, str], list[float]] = {}
        n = 0
        with hist.open(encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                n += 1
                for lvl in ("red", "warn"):
                    for p in d.get(lvl) or []:
                        probes.setdefault((lvl, str(p)[:80]), []).append(float(d.get("ts") or 0))

        def iso(t: float) -> str:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "?"
        lines = [f"health history: {n} snapshots (state/health/history.jsonl, one per watchdog tick)"]
        for (lvl, p), ts in sorted(probes.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"probe {p} went {lvl} {len(ts)} times, first {iso(min(ts))}, last {iso(max(ts))}")
        if not probes:
            lines.append("no probe went warn or red in this window")
        out.append(_doc("selfcheck:health_history", "selfcheck", "selfcheck:health_history",
                        "self-check health history probes red warn", "\n".join(lines)))
    return out


def collect(src: Path = None, spec: Path = None, bundle_dirs: list[Path] | None = None,
            sandbox_glob: str | None = None) -> list[dict]:
    src = src or SRC_ROOT
    return (bundle_docs(src, bundle_dirs or [], sandbox_glob) + docs_docs(src) + test_docs(src)
            + selfcheck_docs(src, spec or SPEC))


# ── BM25 ────────────────────────────────────────────────────────────────────
def build_index(docs: list[dict]) -> dict:
    df: Counter = Counter()
    rows = []
    for d in docs:
        toks = tokenize(d["text"]) + tokenize(d["title"]) * TITLE_WEIGHT
        tf = Counter(toks)
        df.update(tf.keys())
        rows.append({k: d[k] for k in d if k != "text"} | {"len": len(toks), "tf": dict(tf),
                                                              "preview": d["text"][:400]})
    n = len(rows)
    avgdl = (sum(r["len"] for r in rows) / n) if n else 0.0
    return {"version": VERSION, "built_ts": time.time(), "n": n, "avgdl": avgdl, "df": dict(df), "docs": rows,
            "sources": dict(Counter(r["source"] for r in rows))}


def bm25_scores(index: dict, query: str) -> list[tuple[float, int]]:
    q = [t for t in dict.fromkeys(tokenize(query))]
    n, avgdl, df = index["n"], index["avgdl"] or 1.0, index["df"]
    idf = {t: math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in q if df.get(t)}
    out = []
    for i, r in enumerate(index["docs"]):
        tf, dl, s = r["tf"], r["len"], 0.0
        for t, w in idf.items():
            f = tf.get(t)
            if f:
                s += w * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
        if s > 0:
            out.append((s, i))
    out.sort(reverse=True)
    return out


def rrf(*rankings: list[int], k: int = 60) -> list[tuple[float, int]]:
    acc: dict[int, float] = {}
    for rank in rankings:
        for pos, i in enumerate(rank):
            acc[i] = acc.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(((s, i) for i, s in acc.items()), reverse=True)


def search(index: dict, query: str, k: int = 5, per_path: bool = True, embedder=None) -> list[dict]:
    ranked = bm25_scores(index, query)
    if embedder is not None and index.get("embed"):
        qv = embedder([query])[0]
        vecs = index["embed"]["vectors"]
        cos = sorted(((_cos(qv, v), i) for i, v in enumerate(vecs) if v), reverse=True)
        ranked = rrf([i for _, i in ranked[:200]], [i for _, i in cos[:200]])
    hits, seen = [], set()
    for s, i in ranked:
        r = index["docs"][i]
        if per_path and r["path"] in seen:
            continue
        seen.add(r["path"])
        hits.append({"score": round(s, 4), "id": r["id"], "path": r["path"], "source": r["source"],
                     "title": r["title"][:160], "preview": r["preview"][:240]})
        if len(hits) >= k:
            break
    return hits


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# ── optional embeddings (nomic-embed-text, gated) ───────────────────────────
def _http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def embed_gate(tags_fn=None, admit_fn=None) -> tuple[bool, str]:
    """(usable, why). Usable only if nomic-embed-text is ALREADY pulled and gpu_admit admits EMBED_NEED_MIB.
    Never pulls. Injectable for tests."""
    try:
        tags = (tags_fn or (lambda: _http_json(f"{OLLAMA}/api/tags", timeout=4)))()
    except Exception as e:
        return False, f"ollama unreachable ({type(e).__name__}) - BM25 only"
    names = {m.get("name", "") for m in tags.get("models", [])}
    if not any(n.split(":")[0] == EMBED_MODEL for n in names):
        return False, f"{EMBED_MODEL} not pulled (and this script never pulls) - BM25 only"
    if admit_fn is None:
        sys.path.insert(0, str(HERE))
        from gpu_admit import admit as admit_fn  # type: ignore
    ok, info = admit_fn(EMBED_NEED_MIB, who="diag_corpus_embed")
    if not ok:
        return False, f"gpu_admit refused {EMBED_NEED_MIB} MiB (free {info.get('free_mib')}) - BM25 only"
    return True, "ok"


def ollama_embedder(model: str = EMBED_MODEL):
    if model.split(":")[0] != EMBED_MODEL:
        raise ValueError(f"diag_corpus only ever embeds with {EMBED_MODEL}, refusing {model!r}")

    def embed(texts: list[str]) -> list[list[float]]:
        # keep_alive 0: unload as soon as the batch is done - the GPU belongs to the mouth and ears
        r = _http_json(f"{OLLAMA}/api/embed", {"model": model, "input": [t[:2000] for t in texts],
                                               "keep_alive": 0}, timeout=120)
        return r.get("embeddings") or []
    return embed


def add_embeddings(index: dict, docs: list[dict], embedder, batch: int = 32) -> None:
    vecs: list[list[float]] = []
    for i in range(0, len(docs), batch):
        part = docs[i: i + batch]
        vecs.extend(embedder([f"search_document: {d['title']}\n{d['text']}" for d in part]))
    index["embed"] = {"model": EMBED_MODEL, "vectors": vecs}


# ── IO ──────────────────────────────────────────────────────────────────────
def save(index: dict, path: Path = None) -> Path:
    path = path or INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, separators=(",", ":")))
    os.replace(tmp, path)
    return path


def load(path: Path = None) -> dict | None:
    path = path or INDEX
    try:
        idx = json.loads(path.read_text())
        return idx if idx.get("version") == VERSION else None
    except (OSError, ValueError):
        return None


def cmd_build(a) -> dict:
    docs = collect(bundle_dirs=[Path(p) for p in a.bundle_dir or []], sandbox_glob=a.sandbox_glob)
    idx = build_index(docs)
    idx["build_args"] = {"sandbox_glob": a.sandbox_glob, "bundle_dir": a.bundle_dir or [], "src": str(SRC_ROOT)}
    if getattr(a, "embed", False):
        ok, why = embed_gate()
        idx["embed_status"] = why
        if ok:
            add_embeddings(idx, docs, ollama_embedder())
    save(idx)
    return idx


def evaluate(index: dict, questions: list[dict], k: int = 5, embedder=None) -> dict:
    rows, hits1, hitk, rr = [], 0, 0, 0.0
    for q in questions:
        res = search(index, q["q"], k=k, embedder=embedder)
        rank = next((i + 1 for i, h in enumerate(res) if any(e in h["path"] or e in h["id"] or e in h["title"] for e in q["expect"])),
                    None)
        hits1 += rank == 1
        hitk += rank is not None
        rr += 1.0 / rank if rank else 0.0
        rows.append({"q": q["q"], "expect": q["expect"], "rank": rank, "top": [h["path"] for h in res]})
    n = len(questions) or 1
    return {"n": len(questions), "hit_at_1": hits1, f"hit_at_{k}": hitk, "mrr": round(rr / n, 3), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="diagnostic evidence index (BM25, CPU, no LLM)")
    sub = ap.add_subparsers(dest="op", required=True)
    for name in ("build", "query", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--sandbox-glob", default=None,
                       help="also index fault_inject --keep staging dirs, e.g. '/tmp/iris-fi-*'")
        p.add_argument("--bundle-dir", action="append", help="another dir holding bundles (repeatable)")
        p.add_argument("--embed", action="store_true", help="nomic-embed-text fusion, only if pulled + gpu_admit ok")
        p.add_argument("--json", action="store_true")
        if name == "query":
            p.add_argument("question")
            p.add_argument("-k", type=int, default=5)
            p.add_argument("--all-chunks", action="store_true", help="don't collapse to one hit per path")
            p.add_argument("--rebuild", action="store_true")
        if name == "eval":
            p.add_argument("file", nargs="?", default=str(EVAL_FILE))
            p.add_argument("-k", type=int, default=5)
            p.add_argument("--rebuild", action="store_true")
    sub.add_parser("stats")
    a = ap.parse_args(argv)

    if a.op == "build":
        idx = cmd_build(a)
        info = {"index": str(INDEX), "n": idx["n"], "sources": idx["sources"], "embed": idx.get("embed_status", "off")}
        print(json.dumps(info) if a.json else f"built {INDEX}: {idx['n']} chunks {idx['sources']} "
                                                f"embed={info['embed']}")
        return 0
    if a.op == "stats":
        idx = load()
        if not idx:
            print("no index (run build)")
            return 1
        print(json.dumps({"index": str(INDEX), "n": idx["n"], "sources": idx["sources"],
                          "built": time.strftime("%F %T", time.localtime(idx["built_ts"])),
                          "embed": bool(idx.get("embed")), "build_args": idx.get("build_args")}, indent=1))
        return 0

    idx = None if a.rebuild else load()
    if idx is None or a.sandbox_glob or a.bundle_dir:
        idx = cmd_build(a)
    embedder = None
    if a.embed:
        ok, why = embed_gate()
        if ok and idx.get("embed"):
            embedder = ollama_embedder()
        else:
            print(f"[embed off: {why if not ok else 'index has no vectors (build --embed)'}]", file=sys.stderr)
    if a.op == "query":
        hits = search(idx, a.question, k=a.k, per_path=not a.all_chunks, embedder=embedder)
        if a.json:
            print(json.dumps(hits, indent=1))
        else:
            for n, h in enumerate(hits, 1):
                print(f"{n}. [{h['score']:.2f}] {h['id']}  ({h['source']})\n   {h['title'][:110]}")
        return 0 if hits else 1
    # eval
    qs = json.loads(Path(a.file).read_text())["questions"]
    rep = evaluate(idx, qs, k=a.k, embedder=embedder)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        for i, r in enumerate(rep["rows"], 1):
            print(f"Q{i:<2} {'HIT @' + str(r['rank']) if r['rank'] else 'MISS  '}  {r['q']}")
            for t in r["top"]:
                print(f"       - {t}")
        print(f"\nhit@1 {rep['hit_at_1']}/{rep['n']}  hit@{a.k} {rep[f'hit_at_{a.k}']}/{rep['n']}  MRR {rep['mrr']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
