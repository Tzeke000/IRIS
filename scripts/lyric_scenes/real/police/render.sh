#!/usr/bin/env bash
# POLICE (3-D night chase, song "police") — build / frames / stills / mux. Resumable, chunked.
#   render.sh build | frames [S] [E] | stills F1,F2 [PCT] | mux
# env: OUT_DIR (scratch/scenes/police3d) CHUNK (60) PCT (100) LOCK (/tmp/iris_gpu_render.lock)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BLENDER="${BLENDER:-$HOME/.local/bin/blender}"
OUT_DIR="${OUT_DIR:-$REPO/scratch/scenes/police3d}"
BLEND="${BLEND:-$OUT_DIR/police.blend}"
AUDIO="$REPO/state/tzeke_songs/police/police.wav"
MP4="${MP4:-$REPO/state/tzeke_songs/police/police_chase.mp4}"
CHUNK="${CHUNK:-60}"; PCT="${PCT:-100}"; LOCK="${LOCK:-/tmp/iris_gpu_render.lock}"
NF=6690
case "${1:-}" in
  build) "$BLENDER" -b --factory-startup --python-exit-code 1 --python "$HERE/build_scene.py" -- --out "$BLEND" ;;
  frames)
    s=${2:-1}; e=${3:-$NF}; mkdir -p "$OUT_DIR/frames"
    for ((c = s; c <= e; c += CHUNK)); do
      ce=$((c + CHUNK - 1)); ((ce > e)) && ce=$e
      miss=0; for ((f = c; f <= ce; f++)); do [[ -s "$OUT_DIR/frames/$(printf 'f_%04d.png' "$f")" ]] || { miss=1; break; }; done
      ((miss)) || continue
      flock "$LOCK" nice -n 15 "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_frames.py" -- \
        --out "$OUT_DIR/frames" --frames "$c-$ce" --scale "$PCT" | grep -E "render_frames|Error|Traceback" || true
    done ;;
  stills) "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_frames.py" -- --out "${STILLS:-$OUT_DIR/stills}" --frames "$2" --scale "${3:-$PCT}" ;;
  mux)
    n=$(ls "$OUT_DIR"/frames/f_*.png 2>/dev/null | wc -l)
    (( n >= NF )) || { echo "[mux] $n/$NF frames — not muxing"; exit 2; }
    ffmpeg -hide_banner -loglevel error -y -framerate 30 -start_number 1 -i "$OUT_DIR/frames/f_%04d.png" -i "$AUDIO" \
      -map 0:v -map 1:a -vf "noise=alls=4:allf=t" -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p \
      -c:a aac -b:a 320k -shortest -movflags +faststart "$MP4"
    echo "[mux] $MP4" ;;
  *) sed -n 2,4p "$0"; exit 1 ;;
esac
