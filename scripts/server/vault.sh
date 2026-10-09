#!/usr/bin/env bash
# scripts/server/vault.sh - the continuity vault (ClaudeCodeMemory) on the SERVER.
# 2026-10-09 (Zeke: "everything moves to the server, the tower is backup"): the vault's git home is
# now ~/ClaudeCodeMemory on this server (full history copied from the tower's D:\ClaudeCodeMemory at
# 8b81757). It pushes to GitHub with its own deploy key (ssh alias github-vault). The tower's copy is
# a stale backup; refresh it with `vault.sh sync-tower` if ever needed.
#   vault.sh pull                         git pull --rebase from GitHub
#   vault.sh put <vault-rel-path> [file]  write a file into the vault (stdin if no file)
#   vault.sh commit "<message>"           commit + push to GitHub
#   vault.sh status                       git status + last commit
#   vault.sh sync-tower                   push the current tree to the tower's backup copy (no git there)
set -euo pipefail
VAULT="$HOME/ClaudeCodeMemory"
cmd="${1:-status}"; shift || true
cd "$VAULT"
case "$cmd" in
  pull)   git pull --rebase --quiet && git log --oneline -1 ;;
  put)
    rel="$1"; src="${2:-/dev/stdin}"
    case "$rel" in /*|*..*) echo "vault-relative path only" >&2; exit 2 ;; esac
    mkdir -p "$(dirname "$rel")"; cat "$src" > "$rel"; echo "put: $rel" ;;
  commit)
    msg="${1:?commit message}"
    git add -A
    git diff --cached --quiet && { echo "nothing to commit"; exit 0; }
    git commit -q -m "$msg" && git log --oneline -1
    git push --quiet 2>&1 | tail -3 || { echo "PUSH FAILED - commit is local only" >&2; exit 1; } ;;
  status) git status -sb && git log --oneline -1 ;;
  sync-tower)
    tar -cf - --exclude=.git . | ssh -o BatchMode=yes tower "tar -xf - -C D:/ClaudeCodeMemory" && echo "tower backup refreshed" ;;
  *) echo "usage: vault.sh pull|put|commit|status|sync-tower" >&2; exit 2 ;;
esac
