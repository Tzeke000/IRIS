#!/bin/sh
# Install the repo's git hooks (round-2 fix 3.13, 2026-10-01). Idempotent:
#   sh scripts/install_git_hooks.sh
# Writes the SAME hook that is live (privacy guard first, then the compile gate) — the old
# version of this script dropped the privacy guard and only knew the Windows .venv python.
# `git rev-parse --git-path hooks` resolves to the shared hooks dir from a worktree too.
hooks="$(git rev-parse --path-format=absolute --git-path hooks)"
mkdir -p "$hooks"
hook="$hooks/pre-commit"
cat > "$hook" <<'HOOK'
#!/bin/sh
# privacy guard (2026-10-07) — see scripts/privacy_guard.py. Fails closed.
top="$(git rev-parse --show-toplevel)"
py="$top/.venv/Scripts/python.exe"; [ -x "$py" ] || py="$(command -v python3 || command -v python)"
"$py" "$top/scripts/privacy_guard.py" pre-commit || exit 1
exec "$py" "$top/scripts/precommit_pycompile.py"
HOOK
chmod +x "$hook"
echo "installed $hook"
