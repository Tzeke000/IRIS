#!/usr/bin/env bash
# test_worker_e2e.sh (opsguard 11-08) — LIVE end-to-end proof of worker.py's --detach supervisor + --after + reconcile,
# with REAL transient units (tiny: sleep/echo, 100 M each) and TEMP DBs (never the live jobs.sqlite / event store).
# Needs the systemd user bus ⇒ run from Iris's own session, not a locked-down helper. ~1 min. No notify (temp DBs).
#   bash scripts/server/test_worker_e2e.sh [WORKER=scripts/server/worker.py]
set -u
W=${1:-$(dirname "$0")/worker.py}; PY=${PY:-/home/iris/venvs/iris-v100/bin/python}
E=$(mktemp -d /tmp/worker_e2e.XXXX); F=0
export IRIS_WORKER_DB=$E/jobs.sqlite IRIS_EVENTS_DB=$E/events.sqlite IRIS_WORKER_OUT_DIR=$E/out IRIS_WORKER_DEP_POLL_S=2
ok() { echo "PASS $1"; }; no() { echo "FAIL $1"; F=$((F+1)); }
q() { $PY -c "import sqlite3,sys; c=sqlite3.connect('$IRIS_WORKER_DB'); r=c.execute(sys.argv[1]).fetchall(); print(r[0][0] if r else '')" "$1"; }
# 1. a 20 s parent runs DETACHED (supervisor) …
$PY "$W" run --kind e2e --name e2e_parent --mem 100M --timeout 120 --force --no-notify --detach -- bash -c 'sleep 20; echo PARENT DONE' | tee $E/p.json
grep -q '"supervisor": "iris-wsup-e2e_parent' $E/p.json && ok "E1 --detach starts a supervisor unit" || no "E1"
for i in $(seq 1 30); do [ -n "$(q "select id from jobs where name='e2e_parent'")" ] && break; sleep 1; done
# 2. … and a child queued on it by NAME, detached, BEFORE the parent ends (the 11-07 case)
$PY "$W" run --kind e2e --name e2e_child --mem 100M --timeout 120 --force --no-notify --after name:e2e_parent --detach \
  -- bash -c "echo CHILD RAN > $E/child.txt" | tee $E/c.json
grep -q '"supervisor": "iris-wsup-e2e_child' $E/c.json && ok "E2 the dependant is queued under its own supervisor" || no "E2"
[ -e $E/child.txt ] && no "E3 the child must NOT run before the parent ends" || ok "E3 the child waits while the parent runs"
for i in $(seq 1 60); do [ -e $E/child.txt ] && break; sleep 1; done
[ -e $E/child.txt ] && ok "E4 the child STARTED by itself after the parent finished" || no "E4 child never ran"
[ "$(q "select outcome from jobs where name='e2e_parent'")" = ok ] && ok "E5 the detached parent's row has its REAL outcome (ok), not 'detached'" || no "E5 $(q "select outcome from jobs where name='e2e_parent'")"
for i in $(seq 1 20); do [ "$(q "select outcome from jobs where name='e2e_child'")" = ok ] && break; sleep 1; done
[ "$(q "select outcome from jobs where name='e2e_child'")" = ok ] && ok "E6 the child's row is ok" || no "E6"
# 3. a child of a FAILED parent never runs and says so
$PY "$W" run --kind e2e --name e2e_bad --mem 100M --timeout 60 --force --no-notify -- false >/dev/null 2>&1
$PY "$W" run --kind e2e --name e2e_orphan --mem 100M --timeout 60 --force --no-notify --after name:e2e_bad --wait-deps 5 -- touch $E/orphan.txt > /dev/null 2>$E/o.err; rc=$?
[ $rc = 65 ] && [ ! -e $E/orphan.txt ] && grep -q "NOT STARTED" $E/o.err && ok "E7 failed parent ⇒ child refused (65), never ran" || no "E7 rc=$rc"
# 4. the output file + eta
ls $E/out/*e2e_child*.log >/dev/null 2>&1 && ok "E8 output tee'd to the out dir" || no "E8"
$PY "$W" eta --kind e2e --name e2e_parent | grep -q '"n"' && ok "E9 eta answers from history" || no "E9"
echo "worker e2e: $([ $F = 0 ] && echo ALL PASS || echo "$F FAIL") (temp dir $E)"; [ $F = 0 ] && rm -rf "$E"; [ $F = 0 ]
