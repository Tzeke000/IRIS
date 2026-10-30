#!/usr/bin/env bash
# scripts/server/backup_to_tower.sh - nightly OFF-VM copy of everything git does not hold
# (hardening pass 2026-10-07: "nothing leaves this VM - a disk death tonight loses everything since
# the cutover"). One compressed tarball per night to the tower's D:\IrisBackups. Run by
# iris-backup.timer; safe to run by hand. Restore: scripts/server/restore_from_tower.sh +
# docs/RESTORE_RUNBOOK.md.
#
# 2026-10-29 code review fixes:
#   - REQUIRED paths (memory, state, the Claude memory dir, identity files) - if one is missing the run
#     FAILS instead of logging "backup ok" (a moved memory dir used to be skipped silently, and a week
#     of "ok" runs would have pruned every copy that still had it).
#   - staged copy: SQLite files (Chroma's too) are snapshotted with the sqlite backup API, Chroma dirs
#     are re-copied until two passes agree, and a sha256 manifest of every file goes INSIDE the archive
#     (IRIS_BACKUP_MANIFEST.sha256).
#   - verify: zstd -t on the archive, sha256 of the copy on the tower (not just its size).
#   - refuse to prune when the new archive is under half the size of the previous one (and fail).
#   - retention 7 daily / 4 weekly / 6 monthly (backup_retention.py).
#   - optional age encryption: ON only when a recipient public key is configured in
#     ~/.config/iris/backup_age_recipients.txt (or $IRIS_BACKUP_AGE_RECIPIENTS). Configured but `age`
#     missing => FAIL (never silently upload plaintext). Extra secrets (~/.ssh, ~/.claude/channels,
#     ~/.claude/settings.json, wire-pod pairing) are added ONLY to encrypted archives.
#     Where the age PRIVATE key lives is Zeke's call - not on this VM, not on the tower.
# Contains secrets even unencrypted (sibling secret, ~/.iris_private.env incl. the Proxmox token) -
# it lands only on Zeke's own tower disk.
#
# usage: backup_to_tower.sh [--dry-run] [--keep-local DIR]
#   --dry-run      build + verify the archive locally and print the upload/retention plan (the tower is
#                  only LISTED, read-only); nothing is copied, deleted or logged as "backup ok".
#   --keep-local   also copy the finished archive (+ .sha256) into DIR (restore testing).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DRY=0; KEEP_LOCAL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1;;
    --keep-local) KEEP_LOCAL=$2; shift;;
    *) echo "usage: $0 [--dry-run] [--keep-local DIR]" >&2; exit 2;;
  esac; shift
done
TOWER="${IRIS_TOWER_SSH:-tower}"
DEST_WIN='D:\IrisBackups'
STAMP=$(date +%Y%m%d_%H%M)
PY="$HOME/venvs/iris-v100/bin/python"; [ -x "$PY" ] || PY=python3
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
ps_remote() { "${SSH[@]}" "$TOWER" "powershell -NoProfile -Command \"$1\"" | tr -d '\r'; }
say() { echo "[backup] $*" >&2; }
fail() {   # $1 = why, $2 = exit code. A failed run never writes "backup ok", so the watchdog's 36 h check fires.
  say "FAILED: $1"
  [ "$DRY" = 1 ] || echo "$(date '+%F %T') backup FAILED: $1" >> "$ROOT/state/backup.log"
  exit "${2:-3}"
}

# work dir on a real disk, never /tmp (a RAM-backed /tmp would put ~1 GB of staging in memory)
WORK_BASE="${IRIS_BACKUP_WORK:-/data/iris_backup_work}"
mkdir -p "$WORK_BASE" 2>/dev/null || WORK_BASE="$HOME/.cache/iris_backup_work"
mkdir -p "$WORK_BASE"
TMP=$(mktemp -d "$WORK_BASE/run.XXXX")
trap 'chmod -R u+w "$TMP" 2>/dev/null; rm -rf "$TMP"' EXIT   # stock_brain has read-only dirs
STAGE="$TMP/stage"; mkdir -p "$STAGE"
NAME="iris-home_$STAMP.tar.zst"
OUT="$TMP/$NAME"

# ---- encryption: off unless a recipient public key is configured ----------------------------------
RECIP="${IRIS_BACKUP_AGE_RECIPIENTS:-$HOME/.config/iris/backup_age_recipients.txt}"
ENC=0
if [ -f "$RECIP" ] && grep -qE '^[[:space:]]*age1' "$RECIP"; then
  ENC=1
  command -v age >/dev/null || fail "encryption is configured ($RECIP) but 'age' is not installed - refusing to upload plaintext" 4
fi

# ---- what goes in (paths relative to $HOME) ---------------------------------------------------------
REQUIRED=(IRIS/state IRIS/memory .claude/projects/-home-iris-IRIS/memory
          IRIS/ava_core/IDENTITY.local.md IRIS/ava_core/USER.local.md IRIS/config/private.local.json)
OPTIONAL=(IRIS/profiles IRIS/faces IRIS/scripts/iris_cold_wake_msg.local.txt IRIS/apps/ava-control/.env.local
          .config/iris .config/systemd/user .iris_sibling_secret .iris_private.env iris_start.sh iris_console.sh LIVE
          wire-pod/chipper/jdocs wire-pod/chipper/customIntents.json)
# little-brain: the final v16 adapter + Modelfile (hours to retrain); the multi-GB q8 GGUFs stay excluded
LITTLE=(IRIS/state/little_brain/base_modelfile.txt IRIS/state/little_brain/iris_little_adapter.gguf
        IRIS/state/little_brain/adapter_v16_bak)
SECRETS=(.ssh .claude/channels .claude/settings.json wire-pod/chipper/apiConfig.json
         wire-pod/chipper/session-certs wire-pod/chipper/certs)
EXCLUDES=(--exclude='IRIS/state/vector/stock_backup' --exclude='IRIS/state/little_brain/adapter*'
          --exclude='IRIS/state/little_brain/*.gguf' --exclude='IRIS/state/little_brain/merged*'
          --exclude='IRIS/state/little_brain/base*' --exclude='*.tmp' --exclude='IRIS/state/tzeke_songs')
cd "$HOME"
missing=()
for p in "${REQUIRED[@]}"; do [ -e "$p" ] || missing+=("$p"); done
[ ${#missing[@]} -eq 0 ] || fail "REQUIRED path(s) missing: ${missing[*]} - not backing up a copy without them" 3
LIST=("${REQUIRED[@]}")
for p in "${OPTIONAL[@]}"; do [ -e "$p" ] && LIST+=("$p"); done
EXTRA=()
for p in "${LITTLE[@]}"; do [ -e "$p" ] && EXTRA+=("$p"); done
if [ "$ENC" = 1 ]; then for p in "${SECRETS[@]}"; do [ -e "$p" ] && EXTRA+=("$p"); done; fi

# ---- stage: a copy, so the archive + manifest describe one consistent set ---------------------------
copy() {   # rsync -aR keeps the $HOME-relative path; rc 24 = a file vanished while copying: fine
  local rc=0
  rsync -aR "$@" "$STAGE/" || rc=$?
  [ "$rc" = 0 ] || [ "$rc" = 24 ] || fail "rsync rc=$rc while staging" 3
}
copy "${EXCLUDES[@]}" "${LIST[@]}"
[ ${#EXTRA[@]} -eq 0 ] || copy "${EXTRA[@]}"
# Chroma's HNSW segment files are written without a lock we can take: re-copy each Chroma dir until a
# checksum pass finds nothing changed (3 tries). Chroma is DERIVED (rebuilt from state/iris_memory.jsonl
# by iris_semantic_memory.reconcile_from_jsonl), so a still-moving copy is a warning, not a failure.
while IFS= read -r -d '' db; do
  d=$(dirname "$db")
  for try in 1 2 3; do
    changed=$(rsync -anc --out-format='%n' --exclude='*.sqlite3*' "$d/" "$STAGE/$d/" | grep -v '/$' || true)
    [ -z "$changed" ] && break
    say "chroma $d still changing (try $try): $(echo "$changed" | wc -l) file(s) - re-copying"
    rsync -a --exclude='*.sqlite3*' "$d/" "$STAGE/$d/" || true
  done
  [ -z "$changed" ] || say "WARNING: $d never settled - its HNSW files may be mid-write (rebuildable from the jsonl)"
done < <(find IRIS/memory IRIS/state -name chroma.sqlite3 -print0 2>/dev/null)
# SQLite: replace every staged copy with a consistent snapshot via the backup API (= sqlite3 .backup)
"$PY" - "$HOME" "$STAGE" <<'PYEOF' || fail "sqlite snapshot failed" 3
import os, sqlite3, sys, urllib.parse
home, stage = sys.argv[1], sys.argv[2]
n = bad = 0
for dp, _dn, fn in os.walk(stage):
    for f in fn:
        if not f.endswith((".sqlite3", ".sqlite", ".db")):
            continue
        dst = os.path.join(dp, f)
        src = os.path.join(home, os.path.relpath(dst, stage))
        if not os.path.isfile(src):
            continue
        with open(src, "rb") as fh:
            if fh.read(16) != b"SQLite format 3\x00":
                continue                        # a ".db" that isn't SQLite: keep the plain copy
        tmp = dst + ".snap"
        try:
            s = sqlite3.connect(f"file:{urllib.parse.quote(src)}?mode=ro", uri=True, timeout=30)
            d = sqlite3.connect(tmp)
            s.backup(d)
            d.close(); s.close()
        except Exception as e:                  # e.g. a browser profile holding an exclusive lock
            bad += 1
            print(f"[backup] WARNING sqlite snapshot failed, keeping the plain copy: {src}: {e!r}",
                  file=sys.stderr)
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            continue
        os.replace(tmp, dst)
        for side in ("-wal", "-shm", "-journal"):
            try:
                os.remove(dst + side)
            except FileNotFoundError:
                pass
        n += 1
print(f"[backup] sqlite snapshots: {n} ok, {bad} kept as plain copies", file=sys.stderr)
PYEOF
MANIFEST=IRIS_BACKUP_MANIFEST.sha256
( cd "$STAGE" && find . -type f ! -name "$MANIFEST" -print0 | sort -z | xargs -0 -r sha256sum > "$MANIFEST" )
NFILES=$(wc -l < "$STAGE/$MANIFEST")
tar -C "$STAGE" --zstd -cf "$OUT" . || fail "tar failed" 3
zstd -tq "$OUT" || fail "zstd -t says the new archive is corrupt" 3
tar --zstd -tf "$OUT" | grep -x "./$MANIFEST" >/dev/null || fail "manifest missing from the archive" 3
if [ "$ENC" = 1 ]; then
  age -R "$RECIP" -o "$OUT.age" "$OUT" || fail "age encryption failed" 4
  rm -f "$OUT"; OUT="$OUT.age"; NAME="$NAME.age"
fi
SIZE=$(stat -c %s "$OUT")
SUM=$(sha256sum "$OUT" | cut -d' ' -f1)
printf '%s  %s\n' "$SUM" "$NAME" > "$OUT.sha256"
say "built $NAME: $((SIZE/1048576)) MB, $NFILES files, encrypted=$ENC, sha256 ${SUM:0:16}..."
if [ -n "$KEEP_LOCAL" ]; then mkdir -p "$KEEP_LOCAL"; cp "$OUT" "$OUT.sha256" "$KEEP_LOCAL/"; say "kept a copy in $KEEP_LOCAL"; fi

# ---- the tower ---------------------------------------------------------------------------------------
LISTING=$(ps_remote "Get-ChildItem '$DEST_WIN\\iris-home_*.tar.zst*' -ErrorAction SilentlyContinue | Where-Object { \$_.Name -notlike '*.sha256' } | Sort-Object Name -Descending | ForEach-Object { \$_.Name + ' ' + \$_.Length }" 2>/dev/null) \
  || { [ "$DRY" = 1 ] && LISTING="" && say "(dry-run) tower not reachable - no listing"; } || fail "can't list $DEST_WIN on the tower" 3
PREV_SIZE=$(echo "$LISTING" | awk 'NF==2 {print $2; exit}')
SHRANK=0
if [ -n "${PREV_SIZE:-}" ] && [ "$PREV_SIZE" -gt 0 ] && [ $((SIZE * 2)) -lt "$PREV_SIZE" ]; then
  SHRANK=1; say "new archive $SIZE B is under HALF the previous one ($PREV_SIZE B) - will NOT prune"
fi
PLAN=$(printf '%s\n%s\n' "$NAME" "$LISTING" | awk 'NF' | "$PY" "$ROOT/scripts/server/backup_retention.py")
DELETE=$(echo "$PLAN" | awk '$1=="DELETE" {print $2}' | grep -E '^iris-home_[0-9]{8}_[0-9]{4}\.tar\.zst(\.age)?$' || true)
if [ "$DRY" = 1 ]; then
  say "DRY RUN - would upload $NAME (+ .sha256) to $TOWER:$DEST_WIN, verify sha256 there, then"
  if [ "$SHRANK" = 1 ]; then say "  NOT prune (shrank) and exit 5"; else echo "$PLAN" | sed 's/^/[backup]   /' >&2; fi
  exit 0
fi

ps_remote "New-Item -ItemType Directory -Force '$DEST_WIN' | Out-Null" >/dev/null
scp -q -o BatchMode=yes -o ConnectTimeout=20 "$OUT" "$OUT.sha256" "$TOWER:D:/IrisBackups/" || fail "scp to the tower failed" 3
REMOTE=$(ps_remote "(Get-FileHash -Algorithm SHA256 '$DEST_WIN\\$NAME').Hash" | tr 'A-F' 'a-f' | tr -d '[:space:]')
[ "$REMOTE" = "$SUM" ] || fail "sha256 mismatch on the tower: local $SUM remote ${REMOTE:-none}" 3
if [ "$SHRANK" = 1 ]; then
  fail "SUSPECT: $NAME is $((SIZE/1048576)) MB, under half the previous $((PREV_SIZE/1048576)) MB - uploaded but NOTHING pruned; look at what went missing" 5
fi
if [ -n "$DELETE" ]; then
  arr=""
  for n in $DELETE; do arr="$arr${arr:+,}'$DEST_WIN\\$n','$DEST_WIN\\$n.sha256'"; done
  ps_remote "Remove-Item -ErrorAction SilentlyContinue -LiteralPath $arr" >/dev/null || say "prune reported an error (will retry tomorrow)"
  say "pruned: $(echo "$DELETE" | tr '\n' ' ')"
fi
echo "$(date '+%F %T') backup ok: $NAME $((SIZE/1048576)) MB sha256 ${SUM:0:16} files $NFILES enc $ENC -> tower $DEST_WIN" | tee -a "$ROOT/state/backup.log"
