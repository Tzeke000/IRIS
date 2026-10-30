#!/usr/bin/env bash
# scripts/server/stack_rollback_drill.sh [--keep] - V3 Phase 1 EXIT GATE 5 rehearsal, in a SANDBOX HOME (2026-10-30).
#
#   gate 5: "stack_switch.sh v2 and an induced v3 boot failure both land on a working v2 within one supervisor
#            cycle, with memory intact (JSONL line count + Chroma count unchanged)."
#
# Runs the REAL iris_supervise.sh, the REAL ~/iris_start.sh gate (scripts/zorin_launcher/iris_start.sh with the
# chooser line), the REAL stack_launch.sh / stack_switch.sh / stack_rollback.py - against a throw-away HOME
# (mktemp) whose ~/IRIS and ~/IRIS-v3 hold STUB launchers (no body host, no claude, no ports). Supervisor sleeps
# are scaled down through a fake `sleep` on PATH; the DM goes to a stub (IRIS_STACK_DM_CMD). No live effects:
# nothing outside the sandbox is read for state or written. Prints PASS/FAIL per phase and overall; exit 0 = PASS.
#
#   A  manual switch: v3 running -> stack_switch.sh v2 --restart -> next start is v2, no rollback, no DM
#   B  induced v3 boot failure: v3 exits at once, twice -> auto-rollback -> next start is v2, one DM
#   C  v3 up but sdk_stream red (now.json) -> auto-rollback after the red window -> next start is v2, one DM
#   D  no roll-forward: v2 keeps crashing after a rollback -> supervisor stands down, active_stack stays v2
#   M  memory intact: JSONL lines + Chroma rows identical before/after (stack_rollback.py counts)
set -u
KEEP=0; [ "${1:-}" = "--keep" ] && KEEP=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PY="$(command -v python3)"
REAL_SLEEP="$(command -v sleep)"
SB="$(mktemp -d /tmp/iris_stack_drill.XXXXXX)"
D="$SB/drill"; mkdir -p "$D" "$SB/bin" "$SB/.config/iris"
FAILS=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }

# ── sandbox trees ─────────────────────────────────────────────────────────────
for T in IRIS IRIS-v3; do
  mkdir -p "$SB/$T/scripts/server" "$SB/$T/state" "$SB/$T/.tmp"
  for f in iris_supervise.sh stack_launch.sh stack_switch.sh stack_rollback.py health_snapshot.py \
           start_iris_linux.sh start_iris_cli_linux.sh; do
    cp "$REPO/scripts/server/$f" "$SB/$T/scripts/server/$f"
  done
done
cp "$REPO/scripts/zorin_launcher/iris_start.sh" "$SB/iris_start.sh"; chmod +x "$SB/iris_start.sh"
touch "$SB/LIVE"
# stub launchers (the entry each tree's chooser execs): take the shared lock, record the start, run or fail
for T in IRIS IRIS-v3; do
  S="v2"; [ "$T" = "IRIS-v3" ] && S="v3"
  cat > "$SB/$T/start_iris_v2.sh" <<STUB
#!/usr/bin/env bash
exec 9>"\$IRIS_LAUNCHER_LOCK"; flock -n 9 || exit 10
MODE="\$(cat "\$HOME/drill/${S}_mode" 2>/dev/null || echo ok)"
echo "\$(date +%s) $S up start_id=\${IRIS_STACK_START_ID:-} stack_env=\${IRIS_STACK:-} flag=\${IRIS_V3_DRILL_FLAG:-} mode=\$MODE" >> "\$HOME/drill/launches.log"
trap 'exit 143' TERM
if [ "\$MODE" = "fail" ]; then $REAL_SLEEP 0.3; exit 1; fi
while :; do
  if [ "\$MODE" = "red" ] || [ "\$MODE" = "green" ]; then
    ST=ok; [ "\$MODE" = "red" ] && ST=red
    "$PY" -c 'import json,sys,time,os
p=sys.argv[1]; st=sys.argv[2]
d={"ts":time.time(),"subsystems":{"sdk_stream":{"status":st,"last_error":"drill: no stream" if st=="red" else None},"events":{"status":"n/a"}}}
open(p+".tmp","w").write(json.dumps(d)); os.replace(p+".tmp",p)' "\$HOME/IRIS/state/health/now.json" "\$ST"
  fi
  $REAL_SLEEP 0.3
done
STUB
  chmod +x "$SB/$T/start_iris_v2.sh"
done
mkdir -p "$SB/IRIS/state/health"
printf 'IRIS_V3_DRILL_FLAG=1\n# comment\n' > "$SB/.config/iris/v3.env"
# memory fixtures: 7 JSONL lines + a Chroma-shaped sqlite with 7 rows
mkdir -p "$SB/IRIS/memory/chroma" "$SB/memnotes"
for i in 1 2 3 4 5 6 7; do echo "{\"id\": \"m$i\", \"text\": \"drill memory $i\"}"; done > "$SB/IRIS/state/iris_memory.jsonl"
"$PY" - "$SB/IRIS/memory/chroma/chroma.sqlite3" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.executescript("CREATE TABLE collections(id TEXT, name TEXT); CREATE TABLE segments(id TEXT, collection TEXT, scope TEXT);"
                "CREATE TABLE embeddings(id INTEGER PRIMARY KEY, segment_id TEXT); CREATE TABLE embeddings_queue(x);"
                "INSERT INTO collections VALUES('c1','iris_memories'); INSERT INTO segments VALUES('s1','c1','METADATA');")
c.executemany("INSERT INTO embeddings(segment_id) VALUES(?)", [("s1",)] * 7)
c.commit()
EOF
# fake sleep (supervisor + gate waits x0.05) and a DM stub
cat > "$SB/bin/sleep" <<EOF
#!/bin/sh
exec $REAL_SLEEP "\$(awk -v s="\${1:-1}" 'BEGIN{printf "%.3f", s*0.05}')"
EOF
cat > "$SB/bin/fake_dm" <<'EOF'
#!/bin/sh
printf '%s\n' "$1" >> "$HOME/drill/dm.log"
EOF
chmod +x "$SB/bin/sleep" "$SB/bin/fake_dm"

# the sandbox environment every phase runs in (nothing inherited that points at the live stack)
SBENV=(env -i "HOME=$SB" "PATH=$SB/bin:/usr/local/bin:/usr/bin:/bin" "USER=${USER:-iris}" "LANG=C.UTF-8"
       "TMUX=drill" "IRIS_TOWER_LAN=tower.invalid" "IRIS_TOWER_TS=tower-ts.invalid" "IRIS_PYTHON=$PY"
       "IRIS_STACK_DM_CMD=$SB/bin/fake_dm" "IRIS_MEMORY_DIR=$SB/memnotes"
       "IRIS_V3_FAILED_START_S=20" "IRIS_V3_FAIL_WINDOW_S=120" "IRIS_V3_RED_S=3" "IRIS_V3_HEALTH_POLL_S=1"
       "IRIS_V3_HEALTH_FRESH_S=30" "IRIS_STOP_GRACE_S=10" "IRIS_TREE_DRAIN_S=5")
counts() { "${SBENV[@]}" "$PY" "$SB/IRIS/scripts/server/stack_rollback.py" counts --root "$SB/IRIS"; }
stack() { head -c 64 "$SB/.config/iris/active_stack" 2>/dev/null | tr -d ' \t\r\n'; }
SUP_PID=""
sup_start() {   # $1 = phase tag, rest = extra env
  local tag="$1"; shift
  : > "$D/launches.log"; : > "$D/dm.log"; rm -f "$SB/IRIS/state/supervise.log" "$SB/IRIS/state/health/now.json"
  rm -f "$SB/IRIS/state/v3/rollback_state.json" "$SB/IRIS/.tmp/restart_cc.flag"
  setsid "${SBENV[@]}" "$@" bash "$SB/IRIS/scripts/server/iris_supervise.sh" opus </dev/null >"$D/sup_$tag.out" 2>&1 &
  SUP_PID=$!
}
sup_stop() {
  [ -n "$SUP_PID" ] || return 0
  kill -TERM -- "-$SUP_PID" 2>/dev/null; $REAL_SLEEP 0.5; kill -KILL -- "-$SUP_PID" 2>/dev/null
  wait "$SUP_PID" 2>/dev/null; SUP_PID=""
}
wait_for() {   # $1 = timeout s, rest = test command
  local t="$1"; shift; local end=$(( $(date +%s) + t ))
  while [ "$(date +%s)" -lt "$end" ]; do "$@" && return 0; $REAL_SLEEP 0.2; done; return 1
}
v2_after() {   # a v2 launch exists AFTER the first line matching $1 in launches.log order
  awk -v pat="$1" 'index($0,pat){seen=1; next} seen && / v2 up /{found=1} END{exit !found}' "$D/launches.log"
}
trap 'sup_stop' EXIT

echo "== stack rollback drill (exit gate 5) - sandbox $SB"
M0="$(counts)"; echo "memory before: $M0"

# ── A: manual switch v3 -> v2 ────────────────────────────────────────────────
printf 'v3\n' > "$SB/.config/iris/active_stack"; echo green > "$D/v3_mode"; echo ok > "$D/v2_mode"
sup_start A
if wait_for 20 grep -q " v3 up " "$D/launches.log"; then
  touch "$SB/memnotes/handoff_drill.md"
  "${SBENV[@]}" bash "$SB/IRIS/scripts/server/stack_switch.sh" v2 "drill A: manual switch back" --restart > "$D/switch_A.out" 2>&1
  if wait_for 30 v2_after " v3 up "; then
    N_BETWEEN=$(awk '/restart flag seen/{f=1; next} f && /start #/{n++} END{print n+0}' "$SB/IRIS/state/supervise.log")
    if [ "$(stack)" = "v2" ] && [ "$N_BETWEEN" = "1" ] && [ ! -s "$D/dm.log" ] \
       && ! grep -q "V3 ROLLBACK" "$SB/IRIS/state/supervise.log" \
       && grep -q "v3 -> v2 by=.*restart=1 reason=drill A" "$SB/IRIS/state/v3/stack_switch.log"; then
      pass "A manual stack_switch.sh v2 --restart: v3 -> v2 in one cycle, logged, no rollback/DM"
    else fail "A landed on v2 but: stack=$(stack) starts_after_flag=$N_BETWEEN dm=$(wc -l < "$D/dm.log")"; fi
  else fail "A no v2 start after the switch (see $D/sup_A.out)"; fi
else fail "A v3 never started (see $D/sup_A.out, $SB/IRIS/state/v3/stack_launch.log)"; fi
grep -q "stack_env=v3 flag=1" "$D/launches.log" && pass "A v3 start carried IRIS_STACK=v3 + the v3.env flag" \
  || fail "A v3 start did not get v3.env flags"
sup_stop

# ── B: induced v3 boot failure ───────────────────────────────────────────────
printf 'v3\n' > "$SB/.config/iris/active_stack"; echo fail > "$D/v3_mode"; echo ok > "$D/v2_mode"
sup_start B
if wait_for 40 v2_after " v3 up "; then
  NV3=$(grep -c " v3 up " "$D/launches.log")
  N_BETWEEN=$(awk '/V3 ROLLBACK/{f=1; next} f && /start #/{n++} END{print n+0}' "$SB/IRIS/state/supervise.log")
  wait_for 10 test -s "$D/dm.log"
  if [ "$NV3" = "2" ] && [ "$N_BETWEEN" = "1" ] && [ "$(stack)" = "v2" ] && [ "$(wc -l < "$D/dm.log")" = "1" ] \
     && grep -q "by=supervisor(auto-rollback)" "$SB/IRIS/state/v3/stack_switch.log"; then
    pass "B 2 failed v3 starts -> auto-rollback -> v2 in one cycle, active_stack=v2, 1 DM"
  else fail "B v3_starts=$NV3 starts_after_rollback=$N_BETWEEN stack=$(stack) dms=$(wc -l < "$D/dm.log")"; fi
else fail "B never landed on v2 (see $D/sup_B.out)"; fi
sup_stop

# ── C: v3 up but sdk_stream red ──────────────────────────────────────────────
printf 'v3\n' > "$SB/.config/iris/active_stack"; echo red > "$D/v3_mode"; echo ok > "$D/v2_mode"
sup_start C
if wait_for 40 v2_after " v3 up "; then
  NV3=$(grep -c " v3 up " "$D/launches.log")
  wait_for 10 test -s "$D/dm.log"
  if [ "$NV3" = "1" ] && [ "$(stack)" = "v2" ] && grep -q "health red" "$SB/IRIS/state/supervise.log" \
     && [ "$(wc -l < "$D/dm.log")" = "1" ] && grep -q "sdk_stream" "$D/dm.log"; then
    pass "C sdk_stream red past the window -> auto-rollback -> v2 in one cycle, 1 DM"
  else fail "C v3_starts=$NV3 stack=$(stack) dms=$(wc -l < "$D/dm.log")"; fi
else fail "C never landed on v2 (see $D/sup_C.out)"; fi
sup_stop

# ── D: no roll-forward ───────────────────────────────────────────────────────
echo fail > "$D/v2_mode"   # active_stack is v2 after C; v2 itself now crashes on every start
sup_start D "IRIS_MAX_RESTARTS=2"
if wait_for 40 grep -qs "supervisor END" "$SB/IRIS/state/supervise.log"; then
  if [ "$(stack)" = "v2" ] && ! grep -q " v3 up " "$D/launches.log" && [ "$(grep -c " v2 up " "$D/launches.log")" = "3" ] \
     && ! grep -q "rollback armed" "$SB/IRIS/state/supervise.log"; then
    pass "D v2 crash-looping never rolls forward: 3 v2 starts, stood down, active_stack still v2"
  else fail "D stack=$(stack) launches: $(tr '\n' '|' < "$D/launches.log")"; fi
else fail "D supervisor never stood down (see $D/sup_D.out)"; fi
sup_stop

# ── M: memory intact ─────────────────────────────────────────────────────────
M1="$(counts)"; echo "memory after:  $M1"
if [ -n "$M0" ] && [ "$M0" = "$M1" ] && echo "$M1" | grep -q '"chroma": 7' && echo "$M1" | grep -q '"jsonl": 7'; then
  pass "M memory intact (JSONL + Chroma counts unchanged)"
else fail "M memory counts changed or unreadable"; fi

trap - EXIT
if [ "$FAILS" -eq 0 ]; then
  echo "DRILL PASS (exit gate 5, sandbox)"; [ "$KEEP" = 1 ] && echo "sandbox kept: $SB" || rm -rf "$SB"; exit 0
fi
echo "DRILL FAIL ($FAILS check(s)) - sandbox kept for inspection: $SB"; exit 1
