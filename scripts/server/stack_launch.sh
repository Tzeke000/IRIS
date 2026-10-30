#!/usr/bin/env bash
# scripts/server/stack_launch.sh opus|fable|cli - the v2/v3 STACK CHOOSER (V3 Phase 1 step 1.6, 2026-10-30).
#
# Called by ~/iris_start.sh (scripts/zorin_launcher/iris_start.sh) AFTER its ONE-OF-ME gate, in place of the
# old hard-wired "$HOME/IRIS/start_iris_v2.sh". It picks the code tree and execs that tree's launcher:
#
#   ~/.config/iris/active_stack = v2 | v3      (one token; anything else = garbled)
#     v2, missing, garbled                     -> ~/IRIS       (master)              <- the default, always
#     v3 and ~/IRIS-v3 is a launchable tree    -> ~/IRIS-v3    (branch v3) + flags from ~/.config/iris/v3.env
#     v3 but the tree is missing / not launchable / v3.env garbled -> ~/IRIS, logged (never a half-v3 start)
#
# ONE-OF-ME across versions: both trees' launchers take the SAME flock, IRIS_LAUNCHER_LOCK =
# ~/IRIS/.tmp/launcher.lock (resolved explicitly, never relative to the tree that is running). A v3 tree whose
# launcher does not honour IRIS_LAUNCHER_LOCK is refused (-> v2): it would take its own lock and be a second me.
#
# Every decision goes to state/v3/stack_launch.log; the launch record (which stack THIS start really runs) goes
# to state/v3/launched.env, which iris_supervise.sh reads to attribute a failed start (IRIS_STACK_START_ID).
# Canonical state is ~/IRIS/state for both stacks; this script never touches memory.
set -u
MODE="${1:-opus}"
V2_ROOT="$HOME/IRIS"
V3_ROOT="$HOME/IRIS-v3"
CFG="$HOME/.config/iris"
CHOOSER="$CFG/active_stack"
FLAGS="$CFG/v3.env"
V3STATE="$V2_ROOT/state/v3"
LOG="$V3STATE/stack_launch.log"
mkdir -p "$V3STATE" "$V2_ROOT/.tmp" 2>/dev/null
log() { echo "[$(date '+%F %T')] stack_launch: $*" | tee -a "$LOG" 2>/dev/null; }
clean() { printf '%s' "$1" | tr -c 'A-Za-z0-9 _.:/=+@,()-' '?' | head -c 160; }

case "$MODE" in
  opus)  ENTRY="start_iris_v2.sh" ;;
  fable) ENTRY="start_iris_v2_fable.sh" ;;
  cli)   ENTRY="start_iris.sh" ;;
  *) log "unknown mode: $(clean "$MODE") (opus|fable|cli)"; exit 2 ;;
esac

WANT="v2"; WHY=""
if [ ! -e "$CHOOSER" ]; then
  WHY="no active_stack file (missing = v2)"
else
  RAW="$(head -c 64 "$CHOOSER" 2>/dev/null | tr -d ' \t\r\n')"
  case "$RAW" in
    v2) WHY="active_stack=v2" ;;
    v3) WANT="v3" ;;
    *)  WHY="garbled active_stack ('$(clean "$RAW")') - falling back to v2" ;;
  esac
fi

STACK="v2"; ROOT_SEL="$V2_ROOT"
if [ "$WANT" = "v3" ]; then
  NOPE=""
  if [ ! -d "$V3_ROOT" ]; then
    NOPE="$V3_ROOT is missing"
  elif [ "$(cd "$V3_ROOT" && pwd -P)" = "$(cd "$V2_ROOT" && pwd -P)" ]; then
    NOPE="$V3_ROOT resolves to the v2 tree"
  elif [ ! -x "$V3_ROOT/$ENTRY" ]; then
    NOPE="$V3_ROOT/$ENTRY is not executable"
  elif ! grep -q 'IRIS_LAUNCHER_LOCK' "$V3_ROOT/scripts/server/start_iris_linux.sh" 2>/dev/null \
       || { [ "$MODE" = "cli" ] && ! grep -q 'IRIS_LAUNCHER_LOCK' "$V3_ROOT/scripts/server/start_iris_cli_linux.sh" 2>/dev/null; }; then
    NOPE="the v3 tree's launcher does not honour IRIS_LAUNCHER_LOCK (would not share the ONE-OF-ME lock)"
  fi
  # v3.env: KEY=VALUE lines only (comments/blank ok). Parsed, never sourced; a garbled file = no v3 start.
  V3_EXPORTS=()
  if [ -z "$NOPE" ] && [ -f "$FLAGS" ]; then
    LN=0
    while IFS= read -r line || [ -n "$line" ]; do
      LN=$((LN + 1))
      line="${line%$'\r'}"
      case "$line" in ''|'#'*) continue ;; esac
      if [[ "$line" =~ ^(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
        k="${BASH_REMATCH[2]}"; v="${BASH_REMATCH[3]}"
        if [[ "$v" =~ ^\"(.*)\"$ ]] || [[ "$v" =~ ^\'(.*)\'$ ]]; then v="${BASH_REMATCH[1]}"; fi
        V3_EXPORTS+=("$k=$v")
      else
        NOPE="garbled $FLAGS (line $LN is not KEY=VALUE)"; break
      fi
    done < "$FLAGS"
  fi
  if [ -n "$NOPE" ]; then
    WHY="active_stack=v3 but $NOPE - falling back to v2"
  else
    STACK="v3"; ROOT_SEL="$V3_ROOT"
    WHY="active_stack=v3 ($V3_ROOT, ${#V3_EXPORTS[@]} flag(s) from $([ -f "$FLAGS" ] && echo v3.env || echo 'no v3.env'))"
    for kv in "${V3_EXPORTS[@]:-}"; do [ -n "$kv" ] && export "$kv"; done
  fi
fi

export IRIS_STACK="$STACK" IRIS_STACK_ROOT="$ROOT_SEL"
export IRIS_LAUNCHER_LOCK="$V2_ROOT/.tmp/launcher.lock"

# launch record for the supervisor: written atomically (tmp + rename), one key per line
TMP="$V3STATE/.launched.env.$$"
{
  echo "start_id=${IRIS_STACK_START_ID:-}"
  echo "stack=$STACK"
  echo "wanted=$WANT"
  echo "root=$ROOT_SEL"
  echo "mode=$MODE"
  echo "ts=$(date +%s)"
  echo "pid=$$"
  echo "why=$(clean "$WHY")"
} > "$TMP" 2>/dev/null && mv -f "$TMP" "$V3STATE/launched.env" 2>/dev/null
rm -f "$TMP" 2>/dev/null

if [ "$WANT" = "v3" ] && [ "$STACK" = "v2" ]; then log "FALLBACK: $WHY"; else log "$STACK: $WHY"; fi
log "exec $ROOT_SEL/$ENTRY (mode=$MODE, lock=$IRIS_LAUNCHER_LOCK, start_id=${IRIS_STACK_START_ID:-none})"
cd "$ROOT_SEL" || exit 2
exec "$ROOT_SEL/$ENTRY"
