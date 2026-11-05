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
# Delegated != done (11-01): worker.py --ledger opens a "delegated" action in the act ledger at launch;
# it stays OPEN (pending verification) until I check the work and close it with evidence.
#
# usage: helper.sh NAME PROMPT_FILE [--mem 8G] [--timeout 7200] [--cwd DIR] [--model M]
#   LOCKED DOWN by default (11-02, V3 spec §6/§13): no sudo, no secrets, no ssh keys; --trusted-ssh keeps ~/.ssh only.
#   report  -> state/helpers/NAME.report.md   (the helper's final answer)
#   log     -> state/helpers/NAME.log         (stderr)
#   status  -> scripts/server/worker.py status / history --kind helper ; stop: worker.py stop helper-NAME
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PY="${IRIS_PYTHON:-$HOME/venvs/iris-v100/bin/python}"
[ $# -ge 2 ] || { sed -n 2,18p "$0"; exit 2; }
NAME=$1; PROMPT_FILE=$2; shift 2
MEM=8G; TIMEOUT=7200; CWD="$ROOT"; MODEL="${IRIS_HELPER_MODEL:-claude-opus-5-5}"; LOCK=helper; LEASES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --trusted-ssh) LOCK=helper-ssh;;   # keeps ~/.ssh (tower/temple builds); still NO sudo, NO secrets
    --mem) MEM=$2; shift;; --timeout) TIMEOUT=$2; shift;; --cwd) CWD=$2; shift;; --model) MODEL=$2; shift;;
    --leases) LEASES=$2; shift;;      # 11-03: comma-separated globs this helper may write (file leases, log-only)
    *) echo "unknown arg $1" >&2; exit 2;;
  esac; shift
done
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad NAME (letters, digits, _ . - only)" >&2; exit 2; }
[ -s "$PROMPT_FILE" ] || { echo "empty/missing prompt file $PROMPT_FILE" >&2; exit 2; }
OUT="$ROOT/state/helpers"; mkdir -p "$OUT"
cp "$PROMPT_FILE" "$OUT/$NAME.prompt.md"
# Standing rules every helper gets (11-02: a helper's FAKE Discord-shaped test token blocked a push - GitHub
# push protection can't tell fake from real, and the IRIS repo is PUBLIC).
cat >> "$OUT/$NAME.prompt.md" <<'RULES'

---
STANDING RULES (appended to every helper prompt): the IRIS repo is PUBLIC. Never put real IP addresses, personal
data, or secrets in tracked files or commit messages. Test fixtures must not contain LITERAL secret-shaped strings
either (Discord/GitHub/API-token shapes, private keys) even when fake - build them at runtime, e.g.
".".join([...]), or GitHub push protection will reject the push.
ZEKE'S TOWER (if you touch it): NEVER screenshot or capture his screens (private DMs/apps are on them — find windows by
process/window APIs instead), NEVER take focus from his game or apps (launch hidden/no-activate, restore focus), no
console windows flashing (hidden spawns only).
YOUR REPORT (11-05: twice a late "nothing new" note REPLACED a helper's whole report): keep the FULL report in
state/helpers/<your NAME>.full.md as you go, and make your FINAL message the COMPLETE report again — never a short
follow-up note. If something finishes after your report, append it to the .full.md and repeat the full report.
YOUR CLOCK (11-05: 4 of 6 helpers hit the hard timeout with their best work UNCOMMITTED): you are killed without
warning at the --timeout (default 2 h). COMMIT every working step as soon as its test passes (small commits, never a
pile of uncommitted edits), check the elapsed time (`date`) between steps, and STOP BUILDING at ~75 % of your time:
spend the rest on the suites + the final report. An uncommitted fix is a lost fix.
RULES
CLAUDE="$(command -v claude || echo "$HOME/.local/bin/claude")"

# the unit runs this inner script: claude, then the wake-me request (even on failure)
INNER="$OUT/$NAME.run.sh"
cat > "$INNER" <<EOF
#!/usr/bin/env bash
cd "$CWD"
# 11-04: a TIMED-OUT helper (RuntimeMaxSec -> SIGTERM to the cgroup) used to die here before the wake-me submit below,
# so I never heard (firemvp_1104). With a TERM trap, bash waits for claude to exit (it gets TERM too), then continues.
trap 'echo "[helper] SIGTERM (timeout/stop) - still sending helper_done" >&2' TERM
t0=\$(date +%s)
# ISOLATED: no hooks (the inbox-checkpoint hook would steal Zeke's messages into the helper), no MCP
# (.mcp.json would spawn a SECOND iris runtime; plugins a second Discord bot), no user settings.
"$CLAUDE" -p --model "$MODEL" --setting-sources project --strict-mcp-config \\
  --settings '{"disableAllHooks": true}' --dangerously-skip-permissions --output-format text \\
  < "$OUT/$NAME.prompt.md" > "$OUT/$NAME.report.md" 2> "$OUT/$NAME.log"
rc=\$?
cd "$ROOT" && "$PY" - "$NAME" "\$rc" "\$t0" <<'PYEOF'
import os, sys, time
sys.path.insert(1, ".")
from brain import iris_llm
name, rc, t0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
rep = f"state/helpers/{name}.report.md"
act = os.environ.get("IRIS_ACT_ID")
act_line = (f" Ledger action {act} (kind delegated) is OPEN = pending verification: after checking the work, "
            f"close it with act close id={act} (success needs evidence; failure wants a lesson)."
            if act else "")
iris_llm.submit(
    f"[HELPER DONE - automated, not Zeke] Out-of-process helper '{name}' exited rc={rc} after "
    f"{(time.time()-t0)/60:.0f} min. Report: /home/iris/IRIS/{rep} (log: state/helpers/{name}.log). "
    "Read the report, review/verify the work as usual, tell Zeke on Discord if it matters, then reply here "
    "with one line (e.g. 'seen') via llm_reply." + act_line,
    kind="helper_done", requester="helper:" + name, deadline_s=24 * 3600, priority=2)
PYEOF
exit \$rc
EOF
chmod 700 "$INNER"
# --side-effects reversible: a helper edits repos/branches; it never deploys unless its prompt says so
[ -f "$OUT/$NAME.worker.log" ] && mv -f "$OUT/$NAME.worker.log" "$OUT/$NAME.worker.log.prev"   # fresh log: the admission check below reads it
nohup "$PY" "$ROOT/scripts/server/worker.py" run --kind helper --name "helper-$NAME" --mem "$MEM" \
  --timeout "$TIMEOUT" --side-effects reversible --ledger --lockdown "$LOCK" ${LEASES:+--leases "$LEASES"} -- bash "$INNER" >> "$OUT/$NAME.worker.log" 2>&1 &
# 10-30: the launcher runs in the background, so a REFUSED admission used to be invisible ("started" printed
# anyway; the vrsecbatch helper silently never ran). Wait for worker.py's verdict and say it.
: > /dev/null
for _i in $(seq 1 30); do
  if grep -q '"admitted": true' "$OUT/$NAME.worker.log" 2>/dev/null; then break; fi
  if grep -q '"admitted": false' "$OUT/$NAME.worker.log" 2>/dev/null; then
    echo "REFUSED by worker admission:" >&2; grep -A4 '"why"' "$OUT/$NAME.worker.log" | tail -n +1 >&2; exit 3
  fi
  sleep 0.5
done
echo "helper '$NAME' started (worker unit helper-$NAME, launcher pid $!). report -> $OUT/$NAME.report.md"
