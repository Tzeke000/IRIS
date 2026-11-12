#!/usr/bin/env bash
# bgjob.sh NAME SCRIPT_OR_CMD... — start a long job in ITS OWN systemd user unit (iris-NAME) that survives a stack
# restart, logs to scratch/jobs/NAME.out and wakes me with job_done when it exits (via job_notify.sh).
# Why (11-11): I started `systemd-run … bash scripts/server/job_notify.sh …` from a cwd outside the repo TWICE in a
# day — systemd-run's cwd is $HOME, the RELATIVE path failed (status 127), and the job "ran" for 0 s while I waited
# on it. Everything here is resolved to ABSOLUTE paths; a relative SCRIPT arg is resolved against the CALLER's cwd.
#   bgjob.sh rerun8 /home/iris/IRIS/scratch/lands_1110/rerun8.sh      (a script: run with bash)
#   bgjob.sh flyre -- bash vr.sh mp flycarmp                           (any command; runs in the caller's cwd)
set -euo pipefail
NAME=${1:?NAME}; shift
IRIS=/home/iris/IRIS; OUT=$IRIS/scratch/jobs/$NAME.out; mkdir -p "$IRIS/scratch/jobs"
if [ "${1:-}" = "--" ]; then shift; CMD=("$@"); else S=$(readlink -f "$1"); shift; [ -f "$S" ] || { echo "no such script: $S"; exit 2; }; CMD=(bash "$S" "$@"); fi
systemctl --user is-active -q "iris-$NAME" && { echo "iris-$NAME is already running"; exit 1; }
systemd-run --user --unit="iris-$NAME" --collect --working-directory="$PWD" \
  bash "$IRIS/scripts/server/job_notify.sh" "$NAME" "$OUT" -- "${CMD[@]}" >/dev/null 2>&1
sleep 2
if systemctl --user is-active -q "iris-$NAME"; then echo "started iris-$NAME · out $OUT"
else echo "✗ iris-$NAME is NOT running 2 s after start:"; journalctl --user -u "iris-$NAME" -n 5 --no-pager | tail -5; exit 1; fi
