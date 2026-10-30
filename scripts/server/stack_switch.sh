#!/usr/bin/env bash
# scripts/server/stack_switch.sh v2|v3|status ["reason"] [--restart] [--force]   (V3 Phase 1 step 1.6, 2026-10-30)
#
# Chooses which code tree the NEXT start runs: writes ~/.config/iris/active_stack (atomically: tmp + fsync +
# rename, under a flock), logs every switch to state/v3/stack_switch.log, and prints the pre-restart contract.
# It NEVER restarts by default - the choice lands at the next start. --restart sets the EXISTING restart path,
# .tmp/restart_cc.flag (iris_supervise.sh stops the launcher and its loop starts the chosen stack), and refuses
# unless a memory note was saved in the last 10 min (the restart_self rule) - --force overrides, emergencies only.
#
#   v2      -> ~/IRIS (master). Always allowed.
#   v3      -> ~/IRIS-v3 (branch v3) + ~/.config/iris/v3.env. Refused if that tree can't launch.
#   status  -> what is selected, what the last start really ran, the last rollback, the log tail.
#   --auto  -> (iris_supervise.sh's automatic rollback only) v2 only, no reminder, no memory check.
# The supervisor rolls back to v2 by itself; NOTHING rolls forward to v3 automatically.
set -u
V2_ROOT="$HOME/IRIS"
V3_ROOT="$HOME/IRIS-v3"
CFG="$HOME/.config/iris"
F="$CFG/active_stack"
V3STATE="$V2_ROOT/state/v3"
LOG="$V3STATE/stack_switch.log"
FLAG="$V2_ROOT/.tmp/restart_cc.flag"
MEMDIR="${IRIS_MEMORY_DIR:-$HOME/.claude/projects/$(printf '%s' "$V2_ROOT" | sed 's/[^A-Za-z0-9-]/-/g')/memory}"
MEM_FRESH_MIN=10
usage() { echo 'usage: stack_switch.sh v2|v3|status ["reason"] [--restart] [--force]'; exit 2; }

TARGET=""; REASON=""; RESTART=0; FORCE=0; AUTO=0
for a in "$@"; do
  case "$a" in
    --restart) RESTART=1 ;;
    --force)   FORCE=1 ;;
    --auto)    AUTO=1 ;;
    --*)       echo "unknown flag: $a"; usage ;;
    *) if [ -z "$TARGET" ]; then TARGET="$a"; elif [ -z "$REASON" ]; then REASON="$a"; else usage; fi ;;
  esac
done
[ -n "$TARGET" ] || TARGET="status"
read_stack() { head -c 64 "$F" 2>/dev/null | tr -d ' \t\r\n'; }

if [ "$TARGET" = "status" ]; then
  RAW="$(read_stack)"
  case "$RAW" in v2|v3) EFF="$RAW" ;; "") EFF="v2 (no file = v2)"; RAW="<missing>" ;; *) EFF="v2 (garbled file = v2)" ;; esac
  echo "active_stack: $RAW  -> next start runs: $EFF"
  if [ -d "$V3_ROOT" ]; then
    echo "v3 tree: $V3_ROOT  branch=$(git -C "$V3_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')  head=$(git -C "$V3_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
  else
    echo "v3 tree: $V3_ROOT MISSING (a v3 choice falls back to v2)"
  fi
  echo "v3 flags: $([ -f "$CFG/v3.env" ] && grep -c '^[A-Za-z_]' "$CFG/v3.env" || echo 'no') in $CFG/v3.env"
  if [ -f "$V3STATE/launched.env" ]; then
    echo "last start ran: $(grep -E '^(stack|wanted|ts|why)=' "$V3STATE/launched.env" | tr '\n' ' ')"
  fi
  [ -f "$V3STATE/rollback_state.json" ] && python3 -c "import json,sys;d=json.load(open(sys.argv[1]));r=d.get('rollback');print('last auto-rollback:',r if r else 'none')" "$V3STATE/rollback_state.json" 2>/dev/null
  echo "-- $LOG (tail)"; tail -5 "$LOG" 2>/dev/null
  exit 0
fi

case "$TARGET" in v2|v3) ;; *) usage ;; esac
[ -n "$REASON" ] || { echo "a reason is required (it goes in the log): stack_switch.sh $TARGET \"why\""; exit 2; }
if [ "$AUTO" = 1 ] && [ "$TARGET" != "v2" ]; then echo "--auto only ever rolls BACK to v2 (never forward)"; exit 2; fi
if [ "$TARGET" = "v3" ]; then
  if [ ! -d "$V3_ROOT" ] || [ ! -x "$V3_ROOT/start_iris_v2.sh" ]; then
    echo "REFUSED: $V3_ROOT is missing or has no start_iris_v2.sh - a v3 choice would just fall back to v2."; exit 4
  fi
  if ! grep -q 'IRIS_LAUNCHER_LOCK' "$V3_ROOT/scripts/server/start_iris_linux.sh" 2>/dev/null; then
    echo "REFUSED: $V3_ROOT's launcher does not honour IRIS_LAUNCHER_LOCK (no shared ONE-OF-ME lock)."; exit 4
  fi
fi
# pre-restart contract, enforced only where it can be: a memory note saved recently (same rule as restart_self)
if [ "$RESTART" = 1 ] && [ "$AUTO" = 0 ] && [ "$FORCE" = 0 ]; then
  if [ -z "$(find "$MEMDIR" -maxdepth 1 -name '*.md' -mmin "-$MEM_FRESH_MIN" 2>/dev/null | head -1)" ]; then
    echo "REFUSED --restart: no memory note in $MEMDIR saved in the last $MEM_FRESH_MIN min."
    echo "Pre-restart contract: save a handoff note + update MEMORY.md first (or --force, emergencies only)."
    exit 6
  fi
fi

mkdir -p "$CFG" "$V3STATE" || exit 5
exec 7>"$CFG/.active_stack.lock"
flock -w 10 7 || { echo "could not lock $CFG/.active_stack.lock"; exit 5; }
OLD="$(read_stack)"; OLD="${OLD:-<missing>}"
TMP="$CFG/.active_stack.$$.tmp"
if ! { printf '%s\n' "$TARGET" > "$TMP" && sync "$TMP" 2>/dev/null; mv -f "$TMP" "$F"; }; then
  rm -f "$TMP"; echo "FAILED to write $F"; exit 5
fi
sync "$CFG" 2>/dev/null
NEW="$(read_stack)"
if [ "$NEW" != "$TARGET" ]; then echo "FAILED: $F reads '$NEW' after writing '$TARGET'"; exit 5; fi
BY="$([ "$AUTO" = 1 ] && echo 'supervisor(auto-rollback)' || echo "$(id -un)(pid $PPID)")"
CLEAN_REASON="$(printf '%s' "$REASON" | tr '\n\r\t' '   ' | head -c 400)"
echo "[$(date '+%F %T')] $OLD -> $TARGET by=$BY restart=$RESTART reason=$CLEAN_REASON" >> "$LOG"
echo "active_stack: $OLD -> $TARGET (logged to state/v3/stack_switch.log)"

if [ "$AUTO" = 0 ]; then
  cat <<MSG
PRE-RESTART CONTRACT (CLAUDE.md) - before this choice takes effect at a restart:
  1. save a handoff memory (handoff_YYYY-MM-DD_<context>.md) + update the MEMORY.md index
  2. audit the cold-wake message (scripts/iris_cold_wake_msg.local.txt) for the stack you are switching to
  3. don't duplicate the memory-sweep cron - add only what is new since it last ran
MSG
fi
if [ "$RESTART" = 1 ]; then
  mkdir -p "$(dirname "$FLAG")"
  printf 'stack_switch: %s -> %s: %s\n' "$OLD" "$TARGET" "$CLEAN_REASON" > "$FLAG"
  echo "restart flag set: $FLAG - iris_supervise.sh stops the launcher within ~2 s and starts $TARGET."
  pgrep -u "$(id -u)" -f iris_supervise.sh >/dev/null 2>&1 || echo "WARNING: no iris_supervise.sh is running - nothing will act on the flag."
else
  [ "$AUTO" = 1 ] || echo "NOT restarting. The next start runs $TARGET (add --restart to set .tmp/restart_cc.flag now)."
fi
exit 0
