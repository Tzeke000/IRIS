#!/usr/bin/env bash
# BOSS FIGHT arena — build / stills. ALWAYS through the capped slice:
#   scripts/server/render.sh <unit> scripts/lyric_scenes/real/boss_fight/render.sh build|build2|build3|stills|stills2|stills3 [names] [OUT]
#   build   = v1 from nothing -> arena_v1.blend          build2  = v2 on top of arena_v1.blend -> arena_v2.blend
#   stills  = v1 stills from arena_v1.blend               stills2 = v2 stills from arena_v2.blend
#   build3  = v3 (interior cover + rubble) on arena_v2.blend -> arena_v3.blend   stills3 = v3 interior stills
#   stills3c = the v3c set (exterior shots + m_grass) from arena_v3.blend
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BLENDER="${BLENDER:-$HOME/.local/bin/blender}"
OUT_DIR="${OUT_DIR:-$REPO/scratch/scenes/boss_fight}"
BLEND="${BLEND:-$OUT_DIR/arena_v1.blend}"
BLEND2="${BLEND2:-$OUT_DIR/arena_v2.blend}"
BLEND3="${BLEND3:-$OUT_DIR/arena_v3.blend}"
LOCK="${LOCK:-/tmp/iris_gpu_render.lock}"
case "${1:-}" in
  build)
    mkdir -p "$OUT_DIR"
    "$BLENDER" -b --factory-startup --python-exit-code 1 --python "$HERE/build_arena.py" -- \
      --out "$BLEND" --report "$OUT_DIR/build_report.json" ;;
  build2)
    [ -f "$BLEND" ] || { echo "arena_v1.blend missing: run build first" >&2; exit 1; }
    flock "$LOCK" nice -n 10 "$BLENDER" -b --factory-startup "$BLEND" --python-exit-code 1 --python "$HERE/build_arena_v2.py" -- \
      --out "$BLEND2" --report "$OUT_DIR/build_report_v2.json" ${NOSIM:+--nosim} ;;
  stills)
    flock "$LOCK" nice -n 15 "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_stills.py" -- \
      --out "${3:-$OUT_DIR/stills_v1}" --names "${2:-a_wide,b_stairs,c_scale,d_overhead,e_pillar,f_roof,g_hoard}" \
      --scale "${PCT:-100}" --samples "${SPP:-0}" ;;
  stills2)
    flock "$LOCK" nice -n 15 "$BLENDER" -b --factory-startup "$BLEND2" --python-exit-code 1 --python "$HERE/render_stills.py" -- \
      --out "${3:-$OUT_DIR/stills_v2}" --names "${2:-a_compound,b_colossi,c_hall,d_overhead,d_plan,e_prize,f_colossus,g_hoard,h_aisle,c_scale}" \
      --scale "${PCT:-100}" --samples "${SPP:-0}" ;;
  build3)
    [ -f "$BLEND2" ] || { echo "arena_v2.blend missing: run build2 first" >&2; exit 1; }
    flock "$LOCK" nice -n 10 "$BLENDER" -b --factory-startup "$BLEND2" --python-exit-code 1 --python "$HERE/build_arena_v3.py" -- \
      --out "$BLEND3" --report "$OUT_DIR/build_report_v3.json" ${NOSIM:+--nosim} ${NOEXT:+--no-exterior} ;;
  stills3)
    flock "$LOCK" nice -n 15 "$BLENDER" -b --factory-startup "$BLEND3" --python-exit-code 1 --python "$HERE/render_stills.py" -- \
      --out "${3:-$OUT_DIR/stills_v3}" --names "${2:-c_hall,h_aisle,d_overhead,e_prize,e_hoard,i_cover,j_cover_run,d_hallplan,k_town,l_valley,a_compound,a_compound_se,b_colossi,f_colossus}" \
      --scale "${PCT:-100}" --samples "${SPP:-0}" ;;
  stills3c)
    flock "$LOCK" nice -n 15 "$BLENDER" -b --factory-startup "$BLEND3" --python-exit-code 1 --python "$HERE/render_stills.py" -- \
      --out "${3:-$OUT_DIR/stills_v3c}" --names "${2:-a_compound,a_compound_se,b_colossi,f_colossus,k_town,m_grass}" \
      --scale "${PCT:-100}" --samples "${SPP:-0}" ;;
  *) sed -n 2,7p "$0"; exit 1 ;;
esac
