#!/usr/bin/env bash
# BOSS FIGHT arena — build / stills. ALWAYS through the capped slice:
#   scripts/server/render.sh <unit> scripts/lyric_scenes/real/boss_fight/render.sh build|stills [names] [OUT]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BLENDER="${BLENDER:-$HOME/.local/bin/blender}"
OUT_DIR="${OUT_DIR:-$REPO/scratch/scenes/boss_fight}"
BLEND="${BLEND:-$OUT_DIR/arena_v1.blend}"
LOCK="${LOCK:-/tmp/iris_gpu_render.lock}"
case "${1:-}" in
  build)
    mkdir -p "$OUT_DIR"
    "$BLENDER" -b --factory-startup --python-exit-code 1 --python "$HERE/build_arena.py" -- \
      --out "$BLEND" --report "$OUT_DIR/build_report.json" ;;
  stills)
    flock "$LOCK" nice -n 15 "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_stills.py" -- \
      --out "${3:-$OUT_DIR/stills_v1}" --names "${2:-a_wide,b_stairs,c_scale,d_overhead,e_pillar,f_roof,g_hoard}" \
      --scale "${PCT:-100}" --samples "${SPP:-0}" ;;
  *) sed -n 2,3p "$0"; exit 1 ;;
esac
