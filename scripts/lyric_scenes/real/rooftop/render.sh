#!/usr/bin/env bash
# BE MY ONE (3-D cyberpunk rooftop party) — build / stills / frames / mux. Resumable, chunked.
#   render.sh build | stills F1,F2 [PCT] [SAMPLES] | frames [S] [E] | mux
# ALWAYS launch through the capped slice:  scripts/server/render.sh <unit> scripts/lyric_scenes/real/rooftop/render.sh <cmd>
# env: OUT_DIR (scratch/scenes/bemyone_rooftop) CHUNK (60) PCT (100) LOCK (/tmp/iris_gpu_render.lock) MARKERS (0)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BLENDER="${BLENDER:-$HOME/.local/bin/blender}"
OUT_DIR="${OUT_DIR:-$REPO/scratch/scenes/bemyone_rooftop}"
BLEND="${BLEND:-$OUT_DIR/bemyone.blend}"
AUDIO="/data/iris_assets/bemyone/be_my_one_master.wav"
MP4="${MP4:-$OUT_DIR/be_my_one_rooftop_v1.mp4}"
CHUNK="${CHUNK:-60}"; PCT="${PCT:-100}"; LOCK="${LOCK:-/tmp/iris_gpu_render.lock}"
NF=3852
case "${1:-}" in
  build)
    extra=(); [[ "${MARKERS:-0}" == 1 ]] && extra=(--markers)
    "$BLENDER" -b --factory-startup --disable-autoexec --python-exit-code 1 --python "$HERE/build_scene.py" -- \
      --out "$BLEND" --audio "$OUT_DIR/audio.json" "${extra[@]}" ;;
  stills)
    flock "$LOCK" nice -n 15 "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_frames.py" -- \
      --out "${STILLS:-$OUT_DIR/stills}" --frames "$2" --scale "${3:-$PCT}" --samples "${4:-0}" --times "$OUT_DIR/still_times.txt" ;;
  frames)
    s=${2:-1}; e=${3:-$NF}; mkdir -p "$OUT_DIR/frames"
    for ((c = s; c <= e; c += CHUNK)); do
      ce=$((c + CHUNK - 1)); ((ce > e)) && ce=$e
      miss=0; for ((f = c; f <= ce; f++)); do [[ -s "$OUT_DIR/frames/$(printf 'f_%04d.png' "$f")" ]] || { miss=1; break; }; done
      ((miss)) || continue
      flock "$LOCK" nice -n 15 "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_frames.py" -- \
        --out "$OUT_DIR/frames" --frames "$c-$ce" --scale "$PCT" --times "$OUT_DIR/frame_times.txt" \
        | grep -E "render_frames|Error|Traceback" || true
    done ;;
  mux)
    n=$(ls "$OUT_DIR"/frames/f_*.png 2>/dev/null | wc -l)
    (( n >= NF )) || { echo "[mux] $n/$NF frames — not muxing"; exit 2; }
    ffmpeg -hide_banner -loglevel error -y -framerate 30 -start_number 1 -i "$OUT_DIR/frames/f_%04d.png" -i "$AUDIO" \
      -map 0:v -map 1:a -vf "noise=alls=5:allf=t" -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p \
      -c:a aac -b:a 320k -shortest -movflags +faststart "$MP4"
    echo "[mux] $MP4" ;;
  *) sed -n 2,5p "$0"; exit 1 ;;
esac
