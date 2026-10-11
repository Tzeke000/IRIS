#!/bin/sh
# The ONLY way to launch a Blender render on the server (10-11 OOM crash fix).
# Runs the command as a transient user unit inside iris-render.slice (MemoryMax 20G, swap 4G)
# with OOMScoreAdjust=500, so a runaway render dies alone instead of taking the stack.
# usage: scripts/server/render.sh <unit-name> <command> [args...]
set -e
[ $# -ge 2 ] || { echo "usage: $0 <unit-name> <command> [args...]" >&2; exit 2; }
name=$1; shift
exec systemd-run --user --unit="$name" --slice=iris-render.slice \
  -p OOMScoreAdjust=500 -p MemoryMax=${MEM:-20G} -p Nice=10 "$@"
