#!/usr/bin/env bash
# scripts/server/iris_supervise.sh - keeps me running on the server (2026-10-07, the server port).
# Runs INSIDE tmux session "iris" (started at boot by systemd user unit iris-stack.service), so
# Zeke's "Open my console" button still attaches to the live me. The Linux replacement for the
# tower's Iris-Tower-AutoStart + iris_watchdog.ps1 restart poller.
#
#   loop: ~/iris_start.sh <mode>   (ONE-OF-ME gate: ~/LIVE + tower heartbeat/ping; then launcher)
#     gate refused (rc 5/6/7)  -> stop: the tower is me right now. Never loop against the gate.
#     launcher exited          -> restart after 30 s, at most IRIS_MAX_RESTARTS (default 5, Zeke 10-07)
#                                 times per rolling hour, then stand down
#     .tmp/restart_cc.flag     -> (restart_self / safe_restart) stop the launcher; the loop restarts it
#   ~/IRIS_BOOT_TEST present   -> IRIS_ROLE=staging: the launcher runs the runtime-only dry run
#                                 (no claude, no cognition) - safe to test a power cut any time.
set -u
MODE="${1:-opus}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOG="$ROOT/state/supervise.log"
mkdir -p "$ROOT/state" "$ROOT/.tmp"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
FLAG="$ROOT/.tmp/restart_cc.flag"
GATE="$HOME/iris_start.sh"
export IRIS_NO_PAUSE=1
exec 8<&0
STARTS=()
MAX_RESTARTS="${IRIS_MAX_RESTARTS:-5}"   # was 2; Zeke 10-07: "move it to five just in case"
# Code review 10-29 (workers MED): `pkill -P` reaches the gate's DIRECT children only. The launcher's
# own trap stops its tree, but anything orphaned on the way (or a launcher that never got to its trap)
# used to survive a restart. Snapshot the gate's whole tree (pid:starttime, so a recycled pid is never
# killed) when the restart flag fires; once the gate is gone, sweep whatever of it is still alive.
descendants() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done; }
pid_start() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20; }
same_proc() { [ -n "${1#*:}" ] && [ "$(pid_start "${1%%:*}")" = "${1#*:}" ]; }
sweep_tree() {   # $1 = signal; signals every snapshot pid that is still the same process
  local p; for p in $TREE; do same_proc "$p" && kill "-$1" "${p%%:*}" 2>/dev/null; done; return 0
}
TREE=""

if [ -f "$HOME/IRIS_BOOT_TEST" ]; then
  export IRIS_ROLE=staging
  log "BOOT TEST marker present -> IRIS_ROLE=staging (runtime-only dry run, no cognition)"
fi
log "supervisor up: mode=$MODE role=${IRIS_ROLE:-live} host=$(hostname) boot=$(uptime -s)"

while :; do
  NOW=$(date +%s); KEEP=()
  for t in "${STARTS[@]:-}"; do [ -n "$t" ] && [ $((NOW - t)) -lt 3600 ] && KEEP+=("$t"); done
  STARTS=("${KEEP[@]:-}")
  N=0; for t in "${STARTS[@]:-}"; do [ -n "$t" ] && N=$((N+1)); done
  if [ "$N" -ge $((MAX_RESTARTS + 1)) ]; then
    log "STANDING DOWN: $N starts within an hour (1 + $MAX_RESTARTS restarts). Not looping. Check state/launcher_boot.log."
    break
  fi
  STARTS+=("$NOW")
  rm -f "$FLAG"
  log "start #$((N+1)) this hour (limit 1 + $MAX_RESTARTS): $GATE $MODE"
  # fd 8 = the tmux pane's tty: a background job's stdin would be /dev/null, and the CLI mode
  # (interactive claude) and the host's console reader both need the real terminal.
  "$GATE" "$MODE" 0<&8 &
  GPID=$!
  TREE=""; STOP_TS=0
  while kill -0 "$GPID" 2>/dev/null; do
    if [ -f "$FLAG" ]; then
      log "restart flag seen ($(head -c 200 "$FLAG" 2>/dev/null | tr '\n' ' ')) - stopping the launcher for a restart"
      rm -f "$FLAG"
      for p in $(descendants "$GPID"); do TREE="$TREE $p:$(pid_start "$p")"; done
      STOP_TS=$(date +%s)
      pkill -TERM -P "$GPID" 2>/dev/null; kill -TERM "$GPID" 2>/dev/null
    fi
    if [ "$STOP_TS" -gt 0 ] && [ $(( $(date +%s) - STOP_TS )) -ge "${IRIS_STOP_GRACE_S:-45}" ]; then
      log "launcher still up ${IRIS_STOP_GRACE_S:-45} s after the stop - KILL its tree"
      for p in $(descendants "$GPID"); do TREE="$TREE $p:$(pid_start "$p")"; done
      sweep_tree KILL; kill -KILL "$GPID" 2>/dev/null; STOP_TS=0
    fi
    sleep 2
  done
  wait "$GPID"; RC=$?
  log "gate/launcher exited rc=$RC"
  if [ -n "$TREE" ]; then
    # The gate dies on TERM at once while the launcher's trap is still stopping the host (it waits
    # up to 10 s). Give that orderly stop room first; only what is STILL alive after it gets swept.
    for _ in $(seq 1 "${IRIS_TREE_DRAIN_S:-20}"); do
      LEFT=0; for p in $TREE; do same_proc "$p" && LEFT=$((LEFT+1)); done
      [ "$LEFT" -eq 0 ] && break
      sleep 1
    done
    if [ "$LEFT" -gt 0 ]; then
      log "$LEFT process(es) from the stopped tree outlived the launcher - TERM, then KILL"
      sweep_tree TERM; sleep 3; sweep_tree KILL
    fi
    TREE=""
  fi
  case "$RC" in
    5|6|7) log "the gate refused (rc=$RC: the tower copy is me right now) - supervisor exits, no loop."; break ;;
    4)     log "launcher missing (rc=4) - nothing to supervise."; break ;;
    10)    log "another launcher owns cognition (rc=10) - supervisor exits."; break ;;
  esac
  if [ "${IRIS_ROLE:-live}" = "staging" ]; then log "staging run finished - boot test complete, supervisor exits."; break; fi
  log "restarting in 30 s"
  sleep 30
done
log "supervisor END"
