"""Async lookups for the conversation layer (brain/iris_lookups.py).

  lookup.py start "QUESTION" [--origin voice|discord|orb|vr] [--context TEXT] [--dir D]
        -> prints {"id", "say"}: launches a detached runner and returns at once
  lookup.py run JOB_ID [--dir D]        (the runner; spawned by start)
  lookup.py list [--dir D]              pending + recent jobs
Context defaults to the last few lines of state/transcript.jsonl.
"""
import argparse, json, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(_REPO))
from brain import iris_lookups as L  # noqa: E402


def _transcript_tail(n=6):
    try:
        lines = (_REPO / "state" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for ln in lines:
            d = json.loads(ln)
            who = "Iris" if d.get("role") == "assistant" else "Zeke"
            out.append(f"{who}: {str(d.get('content') or '')[:240]}")
        return "\n".join(out)
    except Exception:
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("question"); s.add_argument("--origin", default="voice")
    s.add_argument("--context", default=None); s.add_argument("--dir", default=None)
    r = sub.add_parser("run"); r.add_argument("job_id"); r.add_argument("--dir", default=None)
    li = sub.add_parser("list"); li.add_argument("--dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "start":
        try:
            job = L.start(a.question, origin=a.origin,
                          context=_transcript_tail() if a.context is None else a.context, d=a.dir)
        except ValueError as e:
            print(json.dumps({"ok": False, "error": str(e)})); return 1
        print(json.dumps({"ok": True, "id": job["id"], "say": "Checking - give me a second.",
                          "note": "the answer comes back as its own turn; the orb shows thinking meanwhile"}))
        return 0
    if a.cmd == "run":
        job = L.run_job(a.job_id, d=a.dir)
        print(json.dumps({k: job.get(k) for k in ("id", "status", "error")} if job else {"id": a.job_id}))
        return 0
    if a.cmd == "list":
        print(json.dumps({"summary": L.summary(a.dir), "jobs": L._jobs(a.dir)[-10:]}, indent=1)); return 0


if __name__ == "__main__":
    sys.exit(main())
