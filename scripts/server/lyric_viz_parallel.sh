#!/usr/bin/env bash
# Chunk-parallel lyric_viz render on the server — see lyric_viz_parallel.py.
#   scripts/server/lyric_viz_parallel.sh [--workers 16] <lyric_viz args> --out OUT.mp4
set -eu
PY=~/venvs/iris-v100/bin/python
export PYTHONIOENCODING=utf-8
exec "$PY" "$(dirname "$0")/lyric_viz_parallel.py" "$@"
