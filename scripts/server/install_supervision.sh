#!/usr/bin/env bash
# scripts/server/install_supervision.sh - install + enable the boot-time supervisor on iris-home.
# Idempotent. Needs passwordless sudo once for linger. (2026-10-07, the server port)
# --dry-run: print what would be installed/enabled, touch nothing (2026-10-29).
set -eu
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ "$(uname -s)" = "Linux" ] || { echo "Linux only"; exit 2; }
U="$HOME/.config/systemd/user"
run mkdir -p "$U"
for u in iris-stack.service iris-mouth.service iris-ears.service iris-quiet-hours.service iris-quiet-hours.timer \
         iris-workers.slice iris-workers-render.slice; do
  run install -m 0644 "$ROOT/scripts/server/systemd/$u" "$U/$u"
done
# 2026-10-29 code review CRITICAL: the watchdog + nightly backup were only ever installed by hand, and a
# timer that was never enabled never fails. Install AND enable them here.
for u in iris-watchdog.service iris-watchdog.timer iris-backup.service iris-backup.timer; do
  run install -m 0644 "$ROOT/scripts/server/$u" "$U/$u"
done
# the mouth's sink address comes from the git-ignored private config, never a tracked file
mkdir -p "$HOME/.config/iris"
if [ "$DRY" = 1 ]; then
  echo "[dry-run] would write ~/.config/iris/voice.env + stack.env from config/private.local.json (tower_lan)"
  echo "[dry-run] would: loginctl enable-linger; daemon-reload; enable stack/mouth/ears/quiet-hours; enable --now watchdog+backup timers"
  for u in iris-vector-brain.service iris-vector-nerves.service iris-wirepod.service iris-postoffice.service iris-presence.service; do
    echo "[dry-run] not managed here (hand-installed): $u = $(systemctl --user is-enabled "$u" 2>/dev/null || echo missing)"
  done
  exit 0
fi
TOWER="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("tower_lan",""))' "$ROOT/config/private.local.json")"
[ -n "$TOWER" ] || { echo "tower_lan missing from config/private.local.json"; exit 3; }
printf 'IRIS_AUDIO_SINK=%s:8775
PYTHONUNBUFFERED=1
IRIS_MIC_SOURCE=%s:8776
' "$TOWER" "$TOWER" > "$HOME/.config/iris/voice.env"
# the stack env (launcher sources it): app port on the LAN (firewalled tower-only by install_orb_firewall.sh)
# and letters via the TOWER post-office. From the private config - no address in the repo.
printf "IRIS_ORB_BIND=0.0.0.0
IRIS_POSTOFFICE_URL=http://%s:5877
" "$TOWER" > "$HOME/.config/iris/stack.env"
chmod +x "$ROOT/scripts/server/iris_supervise.sh" "$ROOT/scripts/server/start_iris_linux.sh" "$ROOT"/start_iris*.sh
sudo -n loginctl enable-linger "$(id -un)"
systemctl --user daemon-reload
systemctl --user enable iris-stack.service iris-mouth.service iris-ears.service iris-quiet-hours.timer
systemctl --user enable --now iris-watchdog.timer iris-backup.timer
echo "linger: $(loginctl show-user "$(id -un)" -p Linger)"
systemctl --user is-enabled iris-stack.service iris-mouth.service iris-ears.service iris-watchdog.timer iris-backup.timer
# the Vector / wire-pod / post-office units are hand-installed (their repo copies can lag the live ones):
# report them so a missing one is visible, but don't overwrite them from here.
for u in iris-vector-brain.service iris-vector-nerves.service iris-wirepod.service iris-postoffice.service iris-presence.service; do
  echo "hand-installed: $u = $(systemctl --user is-enabled "$u" 2>/dev/null || echo missing)"
done
