#!/usr/bin/env bash
# scripts/server/bg_job.sh - run a long shell job OUTSIDE my session and wake me when it ends (10-30).
# Like helper.sh but for a plain command (a regress, a build, a render). A background Bash inside my session
# risks the same thing as an in-session background Agent: the CLI stays "mid-turn" and Zeke's messages wait.
#
# usage: bg_job.sh NAME [--mem 8G] [--timeout 7200] [--cwd DIR] -- command args...
#   output -> state/jobs/NAME.out   ·   on exit: a pending LLM-bridge request (kind job_done) with rc + tail
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PY="${IRIS_PYTHON:-$HOME/venvs/iris-v100/bin/python}"
[ $# -ge 3 ] || { sed -n 2,8p "$0"; exit 2; }
NAME=$1; shift
MEM=8G; TIMEOUT=7200; CWD="$ROOT"
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case "$1" in --mem) MEM=$2; shift;; --timeout) TIMEOUT=$2; shift;; --cwd) CWD=$2; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift
done
[ "${1:-}" = "--" ] || { echo "missing -- before the command" >&2; exit 2; }
shift
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad NAME" >&2; exit 2; }
OUT="$ROOT/state/jobs"; mkdir -p "$OUT"
INNER="$OUT/$NAME.run.sh"
{
  echo '#!/usr/bin/env bash'
  printf 'cd %q\n' "$CWD"
  echo 't0=$(date +%s)'
  printf '%q ' "$@"; printf '> %q 2>&1\n' "$OUT/$NAME.out"
  echo 'rc=$?'
  printf 'cd %q && %q - %q "$rc" "$t0" <<'"'"'PYEOF'"'"'\n' "$ROOT" "$PY" "$NAME"
  cat <<'PYEOF'
import sys, time
sys.path.insert(1, ".")
from brain import iris_llm
name, rc, t0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
out = f"state/jobs/{name}.out"
try:
    tail = "".join(open(out, encoding="utf-8", errors="replace").readlines()[-15:])[-1500:]
except Exception as e:
    tail = repr(e)
iris_llm.submit(
    f"[JOB DONE - automated, not Zeke] Background job '{name}' exited rc={rc} after {(time.time()-t0)/60:.0f} min. "
    f"Full output: /home/iris/IRIS/{out}\nTail:\n{tail}\n\nAct on it as usual (tell Zeke on Discord if it matters), "
    "then reply here with one line via llm_reply.",
    kind="job_done", requester="job:" + name, deadline_s=24 * 3600, priority=2)
PYEOF
  echo 'PYEOF'
  echo 'exit $rc'
} > "$INNER"
chmod 700 "$INNER"
nohup "$PY" "$ROOT/scripts/server/worker.py" run --kind job --name "job-$NAME" --mem "$MEM" --timeout "$TIMEOUT" \
  --side-effects reversible -- bash "$INNER" >> "$OUT/$NAME.worker.log" 2>&1 &
echo "job '$NAME' started (unit job-$NAME). output -> $OUT/$NAME.out"
