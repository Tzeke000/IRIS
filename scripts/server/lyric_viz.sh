#!/usr/bin/env bash
# Run scripts/lyric_viz.py on the server's V100 (Zeke 10-09: renders live on the server, not the tower).
# faster-whisper's ctranslate2 needs the CUDA 12 libs that ship inside the venv's nvidia/* wheels.
set -eu
PY=~/venvs/iris-v100/bin/python
NV=$("$PY" -c 'import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))')/nvidia
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:$NV/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../.."
exec "$PY" scripts/lyric_viz.py "$@"
