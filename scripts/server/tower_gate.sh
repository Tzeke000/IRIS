#!/bin/bash
# tower_gate.sh — a LIVE gate for any tower/remote phase that may only run while a flag file exists on the server.
#   (devtools-1106; 11-05 the vrperf stop-watcher — a bare `while; ssh; sleep` loop — died silently at 22:57 and the stop never
#   fired: its ssh had no timeout, nothing showed it was alive, and it lived in the helper's cgroup.)
#
#   tower_gate.sh start REMOTE_DIR [--flag F] [--every S] [--host H] [--name N]   start the watcher (detached)
#   tower_gate.sh status [--name N]     ALIVE/DEAD/STOPPED + heartbeat age + last push; exit 0 only if ALIVE and fresh
#   tower_gate.sh check  [--name N]     the one-line gate to call BEFORE every remote step: exit 0 = go, 1 = stop now
#   tower_gate.sh stop   [--name N]     stop the remote phase NOW (push stop.flag) and end the watcher
#   tower_gate.sh selftest              the whole loop against a fake tower in /tmp (no ssh) — prints PASS/FAIL lines
#
# What the watcher does every S seconds (default 10) while F (default state/tower_free.flag) EXISTS:
#   • writes its HEARTBEAT  state/tower_gate/N.hb  (epoch + pushes ok/failed) — `status` calls it DEAD if older than 3×S
#   • pushes a LEASE        REMOTE_DIR\gate_lease.txt  (dead-man's switch: the TOWER script stops by itself when the lease
#                            is older than ~4×S — so a watcher that dies, hangs or loses the network STOPS the run, never
#                            lets it run on unwatched; see the PowerShell snippet below)
# When F DISAPPEARS (or `stop`): pushes REMOTE_DIR\stop.flag, retries until the tower confirms it exists, logs, exits STOPPED.
# When REMOTE_DIR\done.txt appears: exits DONE.  Every ssh is `timeout`-bounded (a hung ssh can't freeze the loop).
# REMOTE_DIR must be under D:\IrisDeliver\ (never C:). Spawns on the tower are plain `cmd /c echo`/`if exist` over ssh — no
# window, no focus change. Started from a HELPER the watcher dies with the helper's cgroup at its timeout — by design the
# lease then expires and the tower stops; start it from Iris's own session if the remote phase must outlive a helper.
#
# Tower-side snippet (put at the top of the remote script's step loop):
#   function Gate-Ok($dir, $maxAge=40) {        # $maxAge ≈ 4 × --every
#     if (Test-Path "$dir\stop.flag") { return $false }
#     $l = Get-Item "$dir\gate_lease.txt" -ErrorAction SilentlyContinue
#     if (-not $l) { return $false }
#     return ((Get-Date) - $l.LastWriteTime).TotalSeconds -lt $maxAge }
#   if (-not (Gate-Ok $W)) { "gate closed - stopping" | Add-Content "$W\run.log"; <restore>; exit 0 }
# Test/override: TOWER_GATE_SSH="cmd args" replaces `ssh -o BatchMode=yes -o ConnectTimeout=8 HOST` (selftest uses a shim).
set -u
IRIS=${IRIS_HOME:-/home/iris/IRIS}; SD=$IRIS/state/tower_gate; mkdir -p "$SD"
cmd=${1:-}; shift || true
DIR=""; FLAG=$IRIS/state/tower_free.flag; EVERY=10; HOST=tower; NAME=tower
[ "$cmd" = start ] && { DIR=${1:-}; shift || true; }
while [ $# -gt 0 ]; do case $1 in
  --flag) FLAG=$2; shift;; --every) EVERY=$2; shift;; --host) HOST=$2; shift;; --name) NAME=$2; shift;;
  *) echo "unknown arg $1"; exit 2;; esac; shift; done
HB=$SD/$NAME.hb; LOG=$SD/$NAME.log; CFG=$SD/$NAME.cfg; PIDF=$SD/$NAME.pid
rsh() { if [ -n "${TOWER_GATE_SSH:-}" ]; then timeout 25 $TOWER_GATE_SSH "$1"; else timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST" "$1"; fi; }
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
alive() { local p; p=$(cat "$PIDF" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && grep -q tower_gate "/proc/$p/cmdline" 2>/dev/null; }
push_stop() {   # until the tower confirms stop.flag (≤ 10 tries)
  local i; for i in $(seq 1 10); do
    rsh "echo stop> \"$DIR\\stop.flag\" & if exist \"$DIR\\stop.flag\" echo STOPOK" 2>/dev/null | grep -q STOPOK && { log "stop.flag CONFIRMED on the tower ($1)"; return 0; }
    log "stop push $i failed — retrying"; sleep 3; done
  log "✗ COULD NOT PUSH stop.flag after 10 tries ($1) — the lease will expire on the tower anyway"; return 1; }

case "$cmd" in
  start)
    [ -n "$DIR" ] || { echo "usage: tower_gate.sh start 'D:\\IrisDeliver\\…' [--flag F] [--every S] [--host H] [--name N]"; exit 2; }
    case "$DIR" in 'D:\IrisDeliver\'*|'d:\IrisDeliver\'*) ;; *) [ -n "${TOWER_GATE_SSH:-}" ] || { echo "✗ REMOTE_DIR must be under D:\\IrisDeliver\\ (got $DIR)"; exit 2; };; esac
    [ -f "$FLAG" ] || { echo "✗ gate flag $FLAG does not exist — the remote phase may not start"; exit 1; }
    alive && { echo "✗ a gate '$NAME' is already running (pid $(cat $PIDF)) — tower_gate.sh status/stop --name $NAME"; exit 1; }
    printf 'DIR=%q\nFLAG=%q\nEVERY=%q\nHOST=%q\nTOWER_GATE_SSH=%q\n' "$DIR" "$FLAG" "$EVERY" "$HOST" "${TOWER_GATE_SSH:-}" > "$CFG"
    rm -f "$SD/$NAME.state"
    setsid nohup bash "$(readlink -f "$0")" _loop --name "$NAME" < /dev/null >> "$LOG" 2>&1 &
    for i in $(seq 1 40); do [ -s "$HB" ] && [ "$(stat -c %Y "$HB")" -ge "$(( $(date +%s) - 2 ))" ] && break; sleep 0.25; done
    bash "$0" status --name "$NAME" ;;
  _loop)
    . "$CFG"; echo $$ > "$PIDF"; log "watcher up (pid $$): flag $FLAG, lease → $DIR\\gate_lease.txt every ${EVERY}s"
    trap 'log "watcher got SIGTERM — pushing stop (the lease would expire anyway)"; echo "STOPPED sigterm" > "$SD/$NAME.state"; push_stop sigterm; exit 0' TERM
    ok=0; bad=0
    while :; do
      if [ ! -f "$FLAG" ]; then log "flag $FLAG GONE → stop"; echo "STOPPED flag-gone $(date +%s)" > "$SD/$NAME.state"; push_stop flag-gone; exit 0; fi
      [ -f "$SD/$NAME.stopreq" ] && { rm -f "$SD/$NAME.stopreq"; log "stop requested"; echo "STOPPED requested $(date +%s)" > "$SD/$NAME.state"; push_stop requested; exit 0; }
      r=$(rsh "echo %DATE% %TIME% > \"$DIR\\gate_lease.txt\" & if exist \"$DIR\\done.txt\" (echo DONE) else (echo LEASEOK)" 2>/dev/null)
      case "$r" in *DONE*) log "tower says done.txt — gate ends"; echo "DONE $(date +%s)" > "$SD/$NAME.state"; echo "$(date +%s) ok=$ok bad=$bad last=done" > "$HB"; exit 0;;
        *LEASEOK*) ok=$((ok+1)); last=ok;; *) bad=$((bad+1)); last=FAILED; log "lease push failed (${r:0:80})";; esac
      echo "$(date +%s) ok=$ok bad=$bad last=$last" > "$HB"
      sleep "$EVERY"
    done ;;
  status|check)
    [ -f "$CFG" ] && . "$CFG"
    now=$(date +%s); hb=$(cat "$HB" 2>/dev/null); t=${hb%% *}; age=$(( now - ${t:-0} )); st=$(cat "$SD/$NAME.state" 2>/dev/null)
    if alive && [ "$age" -le $(( 3 * EVERY )) ] && [[ $hb == *last=ok* ]]; then s=ALIVE; elif alive; then s="STALE (alive, heartbeat ${age}s old or last push failed)"
    elif [ -n "$st" ]; then s="${st%% [0-9]*}"; else s=DEAD; fi
    if [ "$cmd" = check ]; then [ "$s" = ALIVE ] && [ -f "$FLAG" ] && { echo "GO"; exit 0; }; echo "STOP ($s; flag $([ -f "$FLAG" ] && echo present || echo GONE))"; exit 1; fi
    echo "gate '$NAME': $s · heartbeat ${age}s ago (${hb#* }) · flag $([ -f "$FLAG" ] && echo present || echo GONE) · remote ${DIR:-?}"
    tail -n 3 "$LOG" 2>/dev/null | sed 's/^/  /'
    [ "$s" = ALIVE ] ;;
  stop)
    [ -f "$CFG" ] && . "$CFG"
    if alive; then touch "$SD/$NAME.stopreq"; for i in $(seq 1 $(( EVERY * 4 + 60 ))); do alive || break; sleep 1; done; else
      log "stop (watcher not running) → pushing stop.flag directly"; echo "STOPPED requested $(date +%s)" > "$SD/$NAME.state"; push_stop requested-direct; fi
    bash "$0" status --name "$NAME"; true ;;
  selftest)
    T=$(mktemp -d /tmp/tower_gate_selftest.XXXXXX); mkdir -p "$T/remote"
    # fake ssh: run the cmd.exe one-liner's effects against $T/remote (only the 3 shapes the gate sends)
    cat > "$T/fakessh" <<'SH'
#!/bin/bash
R=$(dirname "$0")/remote; c=$1
[ -f "$(dirname "$0")/offline" ] && exit 255
case "$c" in *gate_lease.txt*) date > "$R/gate_lease.txt"; [ -f "$R/done.txt" ] && echo DONE || echo LEASEOK;;
  *stop.flag*) echo stop > "$R/stop.flag"; echo STOPOK;; esac
SH
    chmod +x "$T/fakessh"; export TOWER_GATE_SSH="$T/fakessh" IRIS_HOME=$T; mkdir -p "$T/state"; F=$T/state/tower_free.flag; touch "$F"
    pass() { echo "PASS $1 $2"; }; fail() { echo "FAIL $1 $2"; FAILED=1; }; FAILED=0; me=$(readlink -f "$0")
    bash "$me" start 'D:\IrisDeliver\selftest' --flag "$F" --every 1 --name st >/dev/null
    sleep 3; [ -f "$T/remote/gate_lease.txt" ] && pass G1 "lease pushed" || fail G1 "no lease on the fake tower"
    bash "$me" check --name st >/dev/null && pass G2 "check = GO while flag + watcher alive" || fail G2 "check said stop"
    rm -f "$F"; sleep 3
    [ -f "$T/remote/stop.flag" ] && pass G3 "flag removed → stop.flag pushed" || fail G3 "no stop.flag after the flag vanished"
    bash "$me" check --name st >/dev/null && fail G4 "check still GO after stop" || pass G4 "check = STOP after the flag vanished"
    grep -q 'CONFIRMED' "$T/state/tower_gate/st.log" && pass G5 "stop confirmed in the log" || fail G5 "no confirmation logged"
    # a watcher that is KILLED -9 (helper cgroup / OOM): status must say DEAD, check must say STOP
    rm -f "$T/remote/stop.flag" "$T/state/tower_gate/st.state"; touch "$F"; bash "$me" start 'D:\IrisDeliver\selftest' --flag "$F" --every 1 --name st >/dev/null; sleep 2
    kill -9 "$(cat "$T/state/tower_gate/st.pid")"; sleep 1
    bash "$me" status --name st | grep -q DEAD && pass G6 "killed watcher → status DEAD" || fail G6 "status didn't notice: $(bash "$me" status --name st | head -1)"
    bash "$me" check --name st >/dev/null && fail G7 "check GO with a dead watcher" || pass G7 "dead watcher → check STOP"
    # the network drops: heartbeat goes STALE (last push failed), check says STOP
    rm -f "$T/state/tower_gate/st.state"; bash "$me" start 'D:\IrisDeliver\selftest' --flag "$F" --every 1 --name st >/dev/null; sleep 2; touch "$T/offline"; sleep 3
    bash "$me" check --name st >/dev/null && fail G8 "check GO while pushes fail" || pass G8 "tower unreachable → check STOP (lease will expire there)"
    rm -f "$T/offline"; bash "$me" stop --name st >/dev/null; [ -f "$T/remote/stop.flag" ] && pass G9 "stop subcommand pushes stop.flag" || fail G9 "stop didn't push"
    kill -0 "$(cat "$T/state/tower_gate/st.pid")" 2>/dev/null && fail G10 "watcher still alive after stop" || pass G10 "watcher gone after stop"
    # REMOTE_DIR guard (no shim ⇒ real rule)
    out=$(TOWER_GATE_SSH= bash "$me" start 'C:\Users\x' --flag "$F" --name st2 2>&1); grep -q 'must be under D:' <<<"$out" && pass G11 "C:\\ refused" || fail G11 "C:\\ accepted: $out"
    rm -rf "$T"; exit $FAILED ;;
  *) sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 2 ;;
esac
