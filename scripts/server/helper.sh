#!/usr/bin/env bash
# scripts/server/helper.sh - run a helper agent OUT OF MY SESSION (10-30).
#
# Why: an in-session background Agent keeps the CLI "mid-turn" until it finishes, so the body host
# never gets my turn's result and Zeke's Discord/orb/voice messages sit frozen in its queue (85 min on
# 10-30). A helper here is its own `claude -p` process inside a worker unit (iris-workers.slice, task in
# the event store via worker.py), so my turns end normally and messages keep flowing.
#
# When the helper exits, a pending LLM-bridge request (kind helper_done) is written; the Stop hook /
# idle nudge delivers it to me like any brain/* request, with the report path.
#
# usage: helper.sh NAME PROMPT_FILE [--mem 8G] [--timeout 7200] [--cwd DIR] [--model M]
#   report  -> state/helpers/NAME.report.md   (the helper's final answer)
#   log     -> state/helpers/NAME.log         (stderr)
#   status  -> scripts/server/worker.py status / history --kind helper ; stop: worker.py stop helper-NAME
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PY="${IRIS_PYTHON:-$HOME/venvs/iris-v100/bin/python}"
[ $# -ge 2 ] || { sed -n 2,18p "$0"; exit 2; }
NAME=$1; PROMPT_FILE=$2; shift 2
MEM=8G; TIMEOUT=7200; CWD="$ROOT"; MODEL="${IRIS_HELPER_MODEL:-claude-opus-5-5}"
while [ $# -gt 0 ]; do
  case "$1" in
    --mem) MEM=$2; shift;; --timeout) TIMEOUT=$2; shift;; --cwd) CWD=$2; shift;; --model) MODEL=$2; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac; shift
done
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad NAME (letters, digits, _ . - only)" >&2; exit 2; }
[ -s "$PROMPT_FILE" ] || { echo "empty/missing prompt file $PROMPT_FILE" >&2; exit 2; }
OUT="$ROOT/state/helpers"; mkdir -p "$OUT"
cp "$PROMPT_FILE" "$OUT/$NAME.prompt.md"
CLAUDE="$(command -v claude || echo "$HOME/.local/bin/claude")"

# the unit runs this inner script: claude, then the wake-me request (even on failure)
INNER="$OUT/$NAME.run.sh"
cat > "$INNER" <<EOF
#!/usr/bin/env bash
cd "$CWD"
t0=\$(date +%s)
# ISOLATED: no hooks (the inbox-checkpoint hook would steal Zeke's messages into the helper), no MCP
# (.mcp.json would spawn a SECOND iris runtime; plugins a second Discord bot), no user settings.
"$CLAUDE" -p --model "$MODEL" --setting-sources project --strict-mcp-config \\
  --settings '{"disableAllHooks": true}' --dangerously-skip-permissions --output-format text \\
  < "$OUT/$NAME.prompt.md" > "$OUT/$NAME.report.md" 2> "$OUT/$NAME.log"
rc=\$?
cd "$ROOT" && "$PY" - "$NAME" "\$rc" "\$t0" <<'PYEOF'
import sys, time
sys.path.insert(1, ".")
from brain import iris_llm
name, rc, t0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
rep = f"state/helpers/{name}.report.md"
iris_llm.submit(
    f"[HELPER DONE - automated, not Zeke] Out-of-process helper '{name}' exited rc={rc} after "
    f"{(time.time()-t0)/60:.0f} min. Report: /home/iris/IRIS/{rep} (log: state/helpers/{name}.log). "
    "Read the report, review/verify the work as usual, tell Zeke on Discord if it matters, then reply here "
    "with one line (e.g. 'seen') via llm_reply.",
    kind="helper_done", requester="helper:" + name, deadline_s=24 * 3600, priority=2)
PYEOF
exit \$rc
EOF
chmod 700 "$INNER"
# --side-effects reversible: a helper edits repos/branches; it never deploys unless its prompt says so
nohup "$PY" "$ROOT/scripts/server/worker.py" run --kind helper --name "helper-$NAME" --mem "$MEM" \
  --timeout "$TIMEOUT" --side-effects reversible -- bash "$INNER" >> "$OUT/$NAME.worker.log" 2>&1 &
echo "helper '$NAME' started (worker unit helper-$NAME, launcher pid $!). report -> $OUT/$NAME.report.md"
