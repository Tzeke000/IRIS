#!/usr/bin/env bash
# HIT A BUMP — arcade (Blender/EEVEE). Full pipeline, resumable, GPU-lock friendly.
#
#   scripts/lyric_scenes/real/arcade/render.sh [--start 1] [--end 3570] [--chunk 120] [--scale 100]
#                                              [--rebuild] [--mux-only] [--out <mp4>]
#
# 1. game_frames.py  (CPU) if the CRT image sequences / screen_stats.npz are missing
# 2. make_art.py     (CPU) if the printed art is missing
# 3. build_scene.py  (CPU) if arcade.blend is missing (or --rebuild)
# 4. render chunks of --chunk frames; EACH chunk takes `flock /tmp/iris_gpu_render.lock` and runs
#    under `nice -n 15`, so other GPU jobs interleave between chunks. Frames already on disk are
#    skipped (resume at frame granularity: just re-run the script).
# 5. when every frame 1..3570 exists: ffmpeg mux (H.264 CRF 17 + AAC 320k, light film grain).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
HERE=scripts/lyric_scenes/real/arcade2
A=$ROOT/assets/scenes/arcade
BLEND=$ROOT/assets/scenes/arcade2/arcade2.blend
FR=$ROOT/assets/scenes/arcade2/frames
PY="$HOME/venvs/iris-v100/bin/python"
BL="$HOME/.local/bin/blender"
LOCK=${LOCK:-/tmp/iris_gpu_render.lock}   # parallel workers: give each its own
TL=state/tzeke_songs/hit_a_bump/timeline_30fps.npz
WAV=state/tzeke_songs/hit_a_bump/hit_a_bump_half.wav
OUT=state/tzeke_songs/hit_a_bump/hit_a_bump_arcade2.mp4
START=1; END=3570; CHUNK=120; SCALE=100; REBUILD=0; MUXONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START=$2; shift 2;; --end) END=$2; shift 2;; --chunk) CHUNK=$2; shift 2;;
    --scale) SCALE=$2; shift 2;; --rebuild) REBUILD=1; shift;; --mux-only) MUXONLY=1; shift;;
    --out) OUT=$2; shift 2;; *) echo "unknown arg $1"; exit 2;;
  esac
done
log() { echo "[arcade $(date +%H:%M:%S)] $*"; }

if [[ $MUXONLY -eq 0 ]]; then
  if [[ ! -f $A/screen_stats.npz || ! -f $A/game/g_3570.png ]]; then
    log "game frames (CPU)"; nice -n 15 "$PY" $HERE/game_frames.py --timeline $TL --out $A --workers 16
  fi
  if [[ ! -f $A/art/marquee_hero.png ]]; then
    log "printed art (CPU)"; "$PY" $HERE/make_art.py --out $A/art
  fi
  if [[ ! -f $BLEND || $REBUILD -eq 1 ]]; then
    log "build scene"; nice -n 15 "$BL" -b --factory-startup --python-exit-code 1 \
      --python $HERE/build_scene.py -- --timeline $ROOT/$TL --assets $A --wav $ROOT/$WAV --out $BLEND
  fi
  mkdir -p "$FR"
  for ((s = START; s <= END; s += CHUNK)); do
    e=$(( s + CHUNK - 1 )); (( e > END )) && e=$END
    missing=0
    for ((f = s; f <= e; f++)); do [[ -f $(printf "%s/f_%04d.png" "$FR" "$f") ]] || { missing=1; break; }; done
    if [[ $missing -eq 0 ]]; then continue; fi
    log "chunk $s-$e (waiting for GPU lock)"
    flock "$LOCK" nice -n 15 "$BL" -b "$BLEND" --python-exit-code 1 --python $HERE/render_frames.py -- \
      --out "$FR" --frames "$s-$e" --scale "$SCALE" --times $ROOT/assets/scenes/arcade2/times.txt | grep -E "render_frames|Error|Traceback" || true
  done
fi

n=$(ls "$FR"/f_*.png 2>/dev/null | wc -l)
if (( n < 3570 )); then log "$n/3570 frames on disk — not muxing yet (re-run to resume)"; exit 0; fi
log "mux -> $OUT"
ffmpeg -y -loglevel error -framerate 30 -start_number 1 -i "$FR/f_%04d.png" -i "$WAV" \
  -vf "noise=alls=3:allf=t:all_seed=7" -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p \
  -c:a aac -b:a 320k -shortest -movflags +faststart "$OUT"
log "done: $OUT ($(du -h "$OUT" | cut -f1))"
