#!/usr/bin/env bash
# scripts/server/vault.sh - the continuity vault (ClaudeCodeMemory) from the SERVER (2026-10-07).
# The vault's ONLY git home is the tower's D:\ClaudeCodeMemory: the repo is private and only Zeke's
# desktop session holds GitHub credentials. The server keeps a read cache at ~/ClaudeCodeMemory.
# (The copy that came over at cutover had no usable git history and was moved aside to
# ~/staged/ClaudeCodeMemory.orphan-20261007 - identical content, kept, not deleted.)
#   vault.sh pull                         refresh the read cache from the tower
#   vault.sh put <vault-rel-path> [file]  write a file into the tower vault (stdin if no file) + cache
#   vault.sh commit "<message>"           commit on the tower and push (via git_push_desktop.ps1)
#   vault.sh status                       tower git status + last commit
set -euo pipefail
TOWER="${IRIS_TOWER_SSH:-tower}"
CACHE="$HOME/ClaudeCodeMemory"
cmd="${1:-status}"; shift || true
case "$cmd" in
  pull)
    tmp=$(mktemp -d); ssh -o BatchMode=yes "$TOWER" "tar -cf - -C D:/ClaudeCodeMemory --exclude=.git ." | tar -xf - -C "$tmp"
    mkdir -p "$CACHE"; cp -a "$tmp"/. "$CACHE"/; rm -rf "$tmp"; echo "cache refreshed: $CACHE" ;;
  put)
    rel="$1"; src="${2:-/dev/stdin}"
    case "$rel" in /*|*..*) echo "vault-relative path only" >&2; exit 2 ;; esac
    dir=$(dirname "$rel")
    ssh -o BatchMode=yes "$TOWER" "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force 'D:\\ClaudeCodeMemory\\${dir//\//\\}' | Out-Null\""
    tmpf=$(mktemp); cat "$src" > "$tmpf"
    scp -q -o BatchMode=yes "$tmpf" "$TOWER:D:/ClaudeCodeMemory/$rel"
    mkdir -p "$CACHE/$dir"; cp "$tmpf" "$CACHE/$rel"; rm -f "$tmpf"; echo "put: $rel" ;;
  commit)
    msg="${1:?commit message}"; msg="${msg//\"/\'}"
    ssh -o BatchMode=yes "$TOWER" "cd /d D:\\ClaudeCodeMemory && git add -A && git commit -q -m \"$msg\" && git log --oneline -1"
    ssh -o BatchMode=yes "$TOWER" "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\Wren-Companion\\scripts\\git_push_desktop.ps1 -Repo D:\\ClaudeCodeMemory" | tr -d '\r' | tail -3 ;;
  status)
    ssh -o BatchMode=yes "$TOWER" "cd /d D:\\ClaudeCodeMemory && git status -sb && git log --oneline -1" | tr -d '\r' ;;
  *) echo "usage: vault.sh pull|put|commit|status" >&2; exit 2 ;;
esac
