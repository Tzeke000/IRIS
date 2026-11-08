#!/usr/bin/env bash
# update_bin.sh (opsguard 11-08) — re-point my sandboxed VR body at a NEW game binary, idempotently. apply.sh is a one-time
# installer (preflights, patches, user, firewall, switch) and is NOT re-runnable, so the LUMEN rename (VRIrisServer →
# LUMENServer, 11-08) meant hand-editing /etc. This does only the binary:
#   ./update_bin.sh [--dry-run]          make the INSTALLED unit's ExecStart binary = the TRACKED unit's (iris-vr-presence.service)
#   ./update_bin.sh --bin PATH [--dry-run]  first rewrite the tracked files here (unit ExecStart, apply.sh/test_sandbox.sh BOT=)
#                                         to PATH (commit them), then the same
#   --restart                            also restart the unit if it is running (default: daemon-reload only; the new binary
#                                         is used at the next start — the body is off unless someone is in VR, Zeke 11-02)
# Idempotent: already pointing there ⇒ "nothing to do", exit 0. Backs the installed unit up to <unit>.pre-<stamp> first.
# Refuses when PATH isn't an executable file or the unit isn't installed (that's apply.sh's job). Never touches the drop-in
# (temple address) or anything else in /etc. Seams for test_update_bin.sh: UNIT_DST, SUDO, SYSTEMCTL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
U=iris-vr-presence.service
DST=${UNIT_DST:-/etc/systemd/system/$U}; SUDO=${SUDO-sudo}; SYSTEMCTL=${SYSTEMCTL:-systemctl}
DRY=0; RESTART=0; NEW=""
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --restart) RESTART=1;; --bin) NEW=${2:?--bin PATH}; shift;;
  *) echo "unknown arg $1"; exit 2;; esac; shift; done
run() { if [ $DRY = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
execbin() { sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' "$1" | head -1; }
if [ -n "$NEW" ]; then
  [ -x "$NEW" ] && [ -f "$NEW" ] || { echo "✗ $NEW is not an executable file — not touching anything"; exit 1; }
  old=$(execbin "$HERE/$U")
  if [ "$old" != "$NEW" ]; then
    for f in "$HERE/$U" "$HERE/apply.sh" "$HERE/test_sandbox.sh"; do
      grep -qF "$old" "$f" || continue
      if [ $DRY = 1 ]; then echo "[dry-run] tracked $(basename "$f"): $old → $NEW"
      else sed -i "s|$old|$NEW|g" "$f"; echo "· tracked $(basename "$f"): $old → $NEW (commit it)"; fi
    done
  fi
fi
WANT=${NEW:-$(execbin "$HERE/$U")}
[ -n "$WANT" ] || { echo "✗ no ExecStart in the tracked $U"; exit 1; }
[ -f "$DST" ] || { echo "✗ $DST is not installed — that's apply.sh's job (first install), not this"; exit 1; }
HAVE=$(execbin "$DST")
if [ "$HAVE" = "$WANT" ]; then echo "✓ installed unit already runs $WANT — nothing to do"; exit 0; fi
[ -x "$WANT" ] || { echo "✗ $WANT (the new binary) is not executable here — deploy it first"; exit 1; }
echo "· installed unit runs $HAVE → re-pointing at $WANT"
bak="$DST.pre-$(date +%m%d_%H%M%S)"
run $SUDO cp -p "$DST" "$bak"
if [ $DRY = 1 ]; then echo "[dry-run] sed ExecStart binary in $DST"
else $SUDO sed -i "s|^ExecStart=$HAVE\\b|ExecStart=$WANT|" "$DST"
  [ "$(execbin "$DST")" = "$WANT" ] || { echo "✗ the edit didn't take — restoring $bak"; $SUDO cp -p "$bak" "$DST"; exit 1; }
  echo "· backup: $bak"; fi
run $SUDO $SYSTEMCTL daemon-reload
if $SYSTEMCTL is-active --quiet $U 2>/dev/null; then
  if [ $RESTART = 1 ]; then run $SUDO $SYSTEMCTL restart $U; else echo "· $U is RUNNING the old binary until its next start (--restart to do it now)"; fi
fi
echo "✓ done$([ $DRY = 1 ] && echo ' (dry-run: nothing changed)')"
