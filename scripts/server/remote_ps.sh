#!/bin/bash
# remote_ps.sh HOST SCRIPT.ps1 [args...]  — run a LOCAL PowerShell file on a Windows box (ssh alias: tower | win101)
# without nesting quotes through bash → ssh → cmd → powershell (the source of most of 10-24's broken one-liners:
# eaten $vars, mangled backslashes, %ERRORLEVEL% expanded too early).
#   remote_ps.sh tower /tmp/x.ps1            → copies to C:\Users\<user>\AppData\Local\Temp\iris_<name>.ps1, runs, cleans up
#   echo 'Get-Date' | remote_ps.sh win101 -  → script from stdin
# Args are passed to the script as $args. Output comes back on stdout; exit code = the script's.
set -e
host=$1; src=$2; shift 2 || { echo "usage: remote_ps.sh tower|win101 FILE.ps1|- [args]"; exit 1; }
tmp=$(mktemp /tmp/rps_XXXX.ps1)
if [ "$src" = "-" ]; then cat > "$tmp"; else cp "$src" "$tmp"; fi
name=iris_$(basename "$tmp")
scp -q "$tmp" "$host:$name"                                  # lands in the remote user's home dir
rm -f "$tmp"
set +e
ssh "$host" "powershell -NoProfile -ExecutionPolicy Bypass -File $name $*"
rc=$?
ssh "$host" "del $name" >/dev/null 2>&1
exit $rc
