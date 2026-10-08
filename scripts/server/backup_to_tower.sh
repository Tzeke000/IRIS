#!/usr/bin/env bash
# scripts/server/backup_to_tower.sh - nightly OFF-VM copy of everything git does not hold
# (hardening pass 2026-10-07: "nothing leaves this VM - a disk death tonight loses everything since
# the cutover"). One compressed tarball per night to the tower's D:\IrisBackups, last 7 kept.
# Contains secrets (sibling secret, private env, Proxmox stats key is NOT included) - it lands only
# on Zeke's own tower disk. Run by iris-backup.timer; safe to run by hand.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TOWER="${IRIS_TOWER_SSH:-tower}"
DEST_WIN='D:\IrisBackups'
STAMP=$(date +%Y%m%d_%H%M)
TMP=$(mktemp -d /tmp/iris_backup.XXXX)
trap 'rm -rf "$TMP"' EXIT
OUT="$TMP/iris-home_$STAMP.tar.zst"
cd "$HOME"
# paths relative to $HOME; missing ones are skipped
LIST=()
for p in IRIS/state IRIS/memory IRIS/profiles IRIS/faces IRIS/config/private.local.json \
         IRIS/ava_core/IDENTITY.local.md IRIS/ava_core/USER.local.md \
         IRIS/scripts/iris_cold_wake_msg.local.txt IRIS/apps/ava-control/.env.local \
         .claude/projects/-home-iris-IRIS/memory .config/iris .config/systemd/user \
         .iris_sibling_secret .iris_private.env iris_start.sh iris_console.sh LIVE; do
  [ -e "$p" ] && LIST+=("$p")
done
tar --zstd -cf "$OUT" --exclude='IRIS/state/vector/stock_backup' --exclude='IRIS/state/little_brain/adapter*' --exclude='IRIS/state/little_brain/*.gguf' --exclude='IRIS/state/little_brain/merged*' --exclude='IRIS/state/little_brain/base*' --exclude='*.tmp' \
    --warning=no-file-changed "${LIST[@]}" || [ $? -eq 1 ]   # rc 1 = a file changed while read: fine
SIZE=$(stat -c %s "$OUT")
ssh -o BatchMode=yes "$TOWER" "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force '$DEST_WIN' | Out-Null\""
scp -q -o BatchMode=yes "$OUT" "$TOWER:D:/IrisBackups/"
# verify the copy landed whole, then prune to the newest 7
REMOTE=$(ssh -o BatchMode=yes "$TOWER" "powershell -NoProfile -Command \"(Get-Item '$DEST_WIN\\$(basename "$OUT")').Length\"" | tr -d '\r')
[ "$REMOTE" = "$SIZE" ] || { echo "backup size mismatch: local $SIZE remote $REMOTE" >&2; exit 3; }
ssh -o BatchMode=yes "$TOWER" "powershell -NoProfile -Command \"Get-ChildItem '$DEST_WIN\\iris-home_*.tar.zst' | Sort-Object Name -Descending | Select-Object -Skip 7 | Remove-Item\""
echo "$(date '+%F %T') backup ok: $(basename "$OUT") $((SIZE/1048576)) MB -> tower $DEST_WIN" | tee -a "$ROOT/state/backup.log"
