#!/usr/bin/env bash
# job_notify.sh NAME OUTFILE -- CMD...   (11-08) run CMD (stdout+stderr → OUTFILE), then wake Iris with an LLM-bridge
# request (kind job_done) carrying rc + the last lines. Why: worker jobs (land, flaky, regress) finished/failed UNSEEN
# for ~1 h twice on 11-08 — a job that can't wake me is a job I forget. Wrap long worker commands in this.
NAME=$1; OUT=$2; shift 2; [ "$1" = "--" ] && shift
t0=$(date +%s); "$@" > "$OUT" 2>&1; rc=$?
cd /home/iris/IRIS && /home/iris/venvs/iris-v100/bin/python - "$NAME" "$rc" "$t0" "$OUT" <<'PYEOF'
import sys, time
sys.path.insert(1, ".")
from brain import iris_llm
name, rc, t0, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
try: tail = "".join(open(out, errors="replace").readlines()[-12:])[-1500:]
except Exception as e: tail = f"(can't read {out}: {e})"
iris_llm.submit(
    f"[JOB DONE - automated, not Zeke] Worker job '{name}' exited rc={rc} after {(time.time()-t0)/60:.0f} min. "
    f"Output: {out}. Last lines:\n{tail}\nRead it, act on it (next step / tell Zeke if it matters), then reply here "
    "with one line via llm_reply.", kind="job_done", requester="job:" + name, deadline_s=24 * 3600, priority=2)
PYEOF
exit $rc
