#!/bin/sh
# The ONLY way to launch a Blender render on the server (10-11 OOM crash fix).
# Runs the command as a transient user unit inside iris-workers-render.slice - a child of
# iris-workers.slice, the ONE shared worker cap (MemoryMax 20G + swap 2G for ALL workers together,
# 2026-10-29 code review: the per-unit cap alone let three renders ask for 60 GB on a 47 GB VM) -
# with its own MemoryMax (MEM, default 20G), swap (SWAP, default 2G) and OOMScoreAdjust=500, so a
# runaway render dies alone instead of taking the stack.
# usage: scripts/server/render.sh <unit-name> <command> [args...]
set -e
[ $# -ge 2 ] || { echo "usage: $0 <unit-name> <command> [args...]" >&2; exit 2; }
name=$1; shift
exec systemd-run --user --unit="$name" --slice=iris-workers-render.slice \
  -p OOMScoreAdjust=500 -p MemoryMax=${MEM:-20G} -p MemorySwapMax=${SWAP:-2G} -p Nice=10 "$@"
