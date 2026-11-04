#!/usr/bin/env bash
# scripts/server/install_gpu_admit.sh - install the V3 3.8 GPU admission gate for the CORE voice units (11-04).
#   mouth: the gate lives in mouth_launch.sh (nothing to install - live at the mouth's next start once merged)
#   ears:  the tracked iris-ears.service runs `gpu_admit.py 1100 --core --exec -- <daemon>`
#   stream-asr: its 10-gpu-admit.conf drop-in (8ecdb41), installed here too if missing
# NEVER restarts a unit. Without --reload it only copies files; the new ExecStart is picked up at the next
# `systemctl --user daemon-reload` + ears restart, which Iris does at a PLANNED restart.
#   --dry-run   print what would happen, touch nothing
#   --reload    also daemon-reload (still no restart)
# PRECONDITION: the gpu_admit.py in ~/IRIS (the path the unit runs) must already have --core, or the ears would
# fail to start after the reload. Checked below; the install refuses otherwise.
set -eu
DRY=0; RELOAD=0
for a in "$@"; do
  case "$a" in --dry-run) DRY=1 ;; --reload) RELOAD=1 ;; *) echo "usage: $0 [--dry-run] [--reload]"; exit 2 ;; esac
done
run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIVE="${IRIS_LIVE:-$HOME/IRIS}"         # what the unit's %h/IRIS resolves to (override only for a dry-run test)
U="$HOME/.config/systemd/user"

# --max-wait 0 + --exec true: one admission round (forced if short), then exits 0 via `true` - proves --core works.
if ! IRIS_GPU_ADMIT=1 /usr/bin/python3 "$LIVE/scripts/server/gpu_admit.py" 0 --core --who install_check --max-wait 0 \
     --exec true >/dev/null 2>&1; then
  echo "REFUSE: $LIVE/scripts/server/gpu_admit.py has no working --core (merge the gpuadmit branch first)"; exit 3
fi
for f in scripts/server/systemd/iris-ears.service scripts/server/systemd/iris-mouth.service scripts/server/mouth_launch.sh; do
  cmp -s "$ROOT/$f" "$LIVE/$f" || { echo "REFUSE: $LIVE/$f differs from $ROOT/$f (install from the merged checkout)"; exit 3; }
done
grep -q 'gpu_gate ' "$LIVE/scripts/server/mouth_launch.sh" || { echo "REFUSE: mouth_launch.sh has no gpu_gate"; exit 3; }

for u in iris-ears.service iris-mouth.service; do
  [ -f "$U/$u" ] && run cp "$U/$u" "$U/$u.bak-pre-gpuadmit"
  run install -m 0644 "$ROOT/scripts/server/systemd/$u" "$U/$u"
done
D="$U/iris-stream-asr.service.d"
if [ -d "$D" ] && ! cmp -s "$ROOT/scripts/server/systemd/iris-stream-asr.service.d/10-gpu-admit.conf" "$D/10-gpu-admit.conf"; then
  run install -m 0644 "$ROOT/scripts/server/systemd/iris-stream-asr.service.d/10-gpu-admit.conf" "$D/10-gpu-admit.conf"
fi
if [ "$RELOAD" = 1 ]; then
  run systemctl --user daemon-reload
  echo "reloaded. Lands at the next restart of each unit (not done here): systemctl --user restart iris-ears"
else
  echo "files installed; NOT reloaded. At the planned restart: systemctl --user daemon-reload && systemctl --user restart iris-ears"
fi
echo "verify after the restart: systemctl --user show iris-ears -p ExecStart | grep -c gpu_admit ; tail -3 $LIVE/state/health/gpu_admission.jsonl"
