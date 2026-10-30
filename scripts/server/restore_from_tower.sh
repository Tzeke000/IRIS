#!/usr/bin/env bash
# scripts/server/restore_from_tower.sh - bring Iris's state back from a nightly backup
# (backup_to_tower.sh). 2026-10-29 code review HIGH: "no restore procedure is written down or tested".
# The full procedure (fresh VM -> running me) is docs/RESTORE_RUNBOOK.md; this script is its core step.
#
#   restore_from_tower.sh --into DIR [--latest | --archive NAME | --from-file PATH] [--identity AGE_KEY]
#   restore_from_tower.sh --list                      # what's on the tower
#   restore_from_tower.sh --live --yes [...]          # restore OVER this machine's $HOME (stack must be stopped)
#
# Steps: fetch (scp from the tower, or a local file) -> check the .sha256 sidecar -> decrypt (.age, needs
# --identity) -> zstd -t -> extract into DIR -> sha256sum -c the in-archive manifest -> PRAGMA
# integrity_check on every SQLite file. Exit 0 only when every check passed. Archives made before the
# 2026-10-29 fixes have no sidecar/manifest: those checks are skipped with a WARNING, not a failure.
# --live extracts into a staging dir FIRST, verifies, and only then rsyncs into $HOME (never a
# half-verified archive over a live tree). It refuses while iris-stack / mouth / ears are active.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TOWER="${IRIS_TOWER_SSH:-tower}"
DEST_WIN='D:\IrisBackups'
PY="$HOME/venvs/iris-v100/bin/python"; [ -x "$PY" ] || PY=python3
INTO=""; NAME=""; FROM=""; IDENT=""; LIVE=0; YES=0; LIST=0
usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --into) INTO=$2; shift;;
    --latest) NAME="";;
    --archive) NAME=$2; shift;;
    --from-file) FROM=$2; shift;;
    --identity) IDENT=$2; shift;;
    --live) LIVE=1;;
    --yes) YES=1;;
    --list) LIST=1;;
    -h|--help) usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac; shift
done
say() { echo "[restore] $*" >&2; }
die() { say "FAILED: $1"; exit "${2:-3}"; }
ps_remote() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$TOWER" "powershell -NoProfile -Command \"$1\"" | tr -d '\r'; }
NAME_RE='^iris-home_[0-9]{8}_[0-9]{4}\.tar\.zst(\.age)?$'

tower_list() {
  ps_remote "Get-ChildItem '$DEST_WIN\\iris-home_*.tar.zst*' -ErrorAction SilentlyContinue | Where-Object { \$_.Name -notlike '*.sha256' } | Sort-Object Name -Descending | ForEach-Object { \$_.Name + ' ' + \$_.Length }"
}
if [ "$LIST" = 1 ]; then tower_list; exit 0; fi

if [ "$LIVE" = 1 ]; then
  [ "$YES" = 1 ] || die "--live overwrites $HOME's state with the backup: add --yes if you mean it" 2
  for u in iris-stack iris-mouth iris-ears; do
    [ "$(systemctl --user is-active "$u" 2>/dev/null || true)" != active ] || die "$u is running - stop the stack first (a live restore under a running me corrupts both)" 2
  done
  INTO="${INTO:-}"
  [ -z "$INTO" ] || die "--live and --into are exclusive (--live restores into \$HOME)" 2
fi
[ "$LIVE" = 1 ] || [ -n "$INTO" ] || die "say where: --into DIR (a scratch dir) or --live --yes" 2

WORK_BASE="${IRIS_BACKUP_WORK:-/data/iris_backup_work}"
mkdir -p "$WORK_BASE" 2>/dev/null || WORK_BASE="$HOME/.cache/iris_backup_work"
mkdir -p "$WORK_BASE"
WORK=$(mktemp -d "$WORK_BASE/restore.XXXX")
trap 'chmod -R u+w "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

# ---- 1. fetch ------------------------------------------------------------------------------------------
if [ -n "$FROM" ]; then
  [ -f "$FROM" ] || die "no such file: $FROM" 2
  NAME=$(basename "$FROM")
  cp "$FROM" "$WORK/$NAME"
  [ -f "$FROM.sha256" ] && cp "$FROM.sha256" "$WORK/$NAME.sha256"
else
  if [ -z "$NAME" ]; then
    NAME=$(tower_list | awk 'NF==2 {print $1; exit}')
    [ -n "$NAME" ] || die "no backups found on the tower ($DEST_WIN)" 3
  fi
  [[ "$NAME" =~ $NAME_RE ]] || die "not a backup name: $NAME" 2
  say "fetching $NAME from $TOWER:$DEST_WIN"
  scp -q -o BatchMode=yes "$TOWER:D:/IrisBackups/$NAME" "$WORK/" || die "scp failed" 3
  scp -q -o BatchMode=yes "$TOWER:D:/IrisBackups/$NAME.sha256" "$WORK/" 2>/dev/null || true
fi
A="$WORK/$NAME"

# ---- 2. integrity of the file itself -------------------------------------------------------------------
if [ -f "$A.sha256" ]; then
  ( cd "$WORK" && sha256sum -c --quiet "$NAME.sha256" ) || die "sha256 of $NAME does not match its sidecar" 3
  say "sha256 sidecar: ok"
else
  say "WARNING: no .sha256 sidecar (pre-2026-10-29 archive?) - file checksum not verified"
fi
if [[ "$NAME" == *.age ]]; then
  [ -n "$IDENT" ] || die "$NAME is encrypted: pass --identity <age private key file>" 2
  command -v age >/dev/null || die "'age' is not installed" 3
  age -d -i "$IDENT" -o "$WORK/plain.tar.zst" "$A" || die "age decrypt failed (wrong key?)" 3
  rm -f "$A"; A="$WORK/plain.tar.zst"
fi
zstd -tq "$A" || die "zstd -t: the archive is corrupt" 3
say "zstd -t: ok"

# ---- 3. extract + verify the contents ------------------------------------------------------------------
if [ "$LIVE" = 1 ]; then DEST="$WORK/extract"; else DEST="$INTO"; fi
mkdir -p "$DEST"
[ -z "$(ls -A "$DEST" 2>/dev/null)" ] || die "$DEST is not empty - restore into an empty dir" 2
tar --zstd -xpf "$A" -C "$DEST" || die "tar extract failed" 3
rm -f "$A"
MANIFEST=IRIS_BACKUP_MANIFEST.sha256
if [ -f "$DEST/$MANIFEST" ]; then
  ( cd "$DEST" && sha256sum -c --quiet "$MANIFEST" ) || die "manifest check FAILED - some restored files differ from what was backed up" 3
  say "manifest: $(wc -l < "$DEST/$MANIFEST") files verified"
else
  say "WARNING: no $MANIFEST inside (pre-2026-10-29 archive) - per-file checksums not verified"
fi
"$PY" - "$DEST" <<'PYEOF' || die "SQLite integrity check failed" 3
import os, sqlite3, sys, urllib.parse
bad = n = 0
for dp, _d, fn in os.walk(sys.argv[1]):
    for f in fn:
        if not f.endswith((".sqlite3", ".sqlite", ".db")):
            continue
        p = os.path.join(dp, f)
        with open(p, "rb") as fh:
            if fh.read(16) != b"SQLite format 3\x00":
                continue
        n += 1
        try:
            c = sqlite3.connect(f"file:{urllib.parse.quote(p)}?mode=ro", uri=True)
            r = c.execute("PRAGMA integrity_check").fetchone()[0]
            c.close()
        except Exception as e:
            r = repr(e)
        if r != "ok":
            bad += 1
            print(f"[restore] SQLite NOT ok: {p}: {r}", file=sys.stderr)
print(f"[restore] sqlite integrity: {n - bad}/{n} ok", file=sys.stderr)
sys.exit(1 if bad else 0)
PYEOF
for need in IRIS/state IRIS/memory .claude/projects/-home-iris-IRIS/memory IRIS/ava_core/IDENTITY.local.md; do
  [ -e "$DEST/$need" ] || say "WARNING: $need is not in this backup"
done

# ---- 4. live: only now, copy the verified tree over $HOME ----------------------------------------------
if [ "$LIVE" = 1 ]; then
  say "verified - copying into $HOME (existing files are overwritten, nothing is deleted)"
  rsync -a --exclude="$MANIFEST" "$DEST/" "$HOME/"
  say "done. Next: docs/RESTORE_RUNBOOK.md step 'after the restore' (install_supervision.sh, staging_dryrun.py, memory_search)."
else
  say "restored + verified into $DEST. Nothing outside it was touched."
fi
