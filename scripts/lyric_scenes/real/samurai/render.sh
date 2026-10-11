#!/usr/bin/env bash
# NEON SAMURAI DUEL — build / render / mux for "Pull Me Close" (see README.md in this folder).
#
#   render.sh build                       (re)build samurai.blend from the timeline npz (~10 s, CPU)
#   render.sh frames [START] [END] [PCT]  render Blender frames START..END (1-based; default the
#                                         whole song) -> $OUT_DIR/frames/f_####.png.
#                                         RESUMABLE: finished frames are skipped; each frame is
#                                         written to a temp name then renamed (no torn PNGs).
#   render.sh stills F1,F2,... [PCT]      look-dev stills -> $OUT_DIR/stills/
#   render.sh mux [START] [END]           frames -> MP4 with the song, audio offset to START
#                                         (refuses if frames are missing; FORCE=1 overrides)
#   render.sh all                         build if missing, frames 1..N, mux
#
# env: OUT_DIR (scratch/scenes/samurai)  CHUNK (frames per Blender launch, 60)
#      SAMPLES (EEVEE TAA samples; 0 = what the build set)  PCT (resolution %, 100)
#      MP4 (output, $OUT_DIR/pull_me_close_samurai.mp4)  VRAM_LOG (file: sample the Blender
#      process's GPU memory once a second while it renders)
# GPU etiquette: every Blender launch runs under `flock /tmp/iris_gpu_render.lock nice -n 15`,
# one chunk at a time, so other GPU jobs can interleave between chunks.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BLENDER="${BLENDER:-$HOME/.local/bin/blender}"
OUT_DIR="${OUT_DIR:-$REPO/scratch/scenes/samurai}"
BLEND="${BLEND:-$REPO/scratch/scenes/samurai/samurai.blend}"   # independent of OUT_DIR
SONG="$REPO/state/tzeke_songs/pull_me_close"
AUDIO="${AUDIO:-$SONG/pull_me_close.wav}"
TIMELINE="${TIMELINE:-$SONG/timeline_30fps.npz}"
CHUNK="${CHUNK:-60}"
PCT="${PCT:-100}"
SAMPLES="${SAMPLES:-0}"
MP4="${MP4:-$OUT_DIR/pull_me_close_samurai.mp4}"
LOCK="${LOCK:-/tmp/iris_gpu_render.lock}"   # parallel workers: give each its own
FPS=30

gpu() { flock "$LOCK" nice -n 15 "$@"; }

vram_watch() {   # sample GPU memory of our blender processes (by pid) into $VRAM_LOG
  [[ -n "${VRAM_LOG:-}" ]] || return 0
  ( while sleep 1; do
      for p in $(pgrep -f "blender -b $BLEND" || true); do
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', ' -v p="$p" '$1==p{print strftime("%H:%M:%S"), $2}'
      done
    done ) >> "$VRAM_LOG" &
  WATCH_PID=$!
  trap 'kill $WATCH_PID 2>/dev/null || true' EXIT
}

nframes() {
  "$BLENDER" -b --factory-startup --python-expr \
    "import numpy as np; print('NFRAMES', len(np.load('$TIMELINE')['rms']))" 2>/dev/null \
    | awk '/^NFRAMES/{print $2}'
}

build() {
  mkdir -p "$OUT_DIR"
  "$BLENDER" -b --factory-startup --python-exit-code 1 \
    --python "$HERE/build_scene.py" -- --timeline "$TIMELINE" --audio "$AUDIO" --out "$BLEND" \
    | grep -E '^\[(choreo|rig|build|verify|samurai)\]' || true
  [[ -f "$BLEND" ]]
}

render_range() {  # START END
  local s=$1 e=$2 c ce
  mkdir -p "$OUT_DIR/frames"
  [[ "${NO_TMP_CLEAN:-0}" == 1 ]] || rm -f "$OUT_DIR"/frames/.tmp_f_*.png   # parallel workers must not wipe each other
  for ((c = s; c <= e; c += CHUNK)); do
    ce=$((c + CHUNK - 1)); ((ce > e)) && ce=$e
    local missing=0 f
    for ((f = c; f <= ce; f++)); do
      [[ -s "$OUT_DIR/frames/$(printf 'f_%04d.png' "$f")" ]] || { missing=1; break; }
    done
    ((missing)) || continue
    echo "[render] frames $c..$ce  $(date +%H:%M:%S)"
    gpu "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_frames.py" -- \
      --start "$c" --end "$ce" --out "$OUT_DIR/frames" --pct "$PCT" --samples "$SAMPLES" \
      | grep -E '^(TIME|\[render_frames\])' || true
  done
}

mux() {  # [START END]
  local s=${1:-1} e=${2:-$(nframes)} n missing=0 f
  n=$((e - s + 1))
  for ((f = s; f <= e; f++)); do
    [[ -s "$OUT_DIR/frames/$(printf 'f_%04d.png' "$f")" ]] || missing=$((missing + 1))
  done
  if ((missing)) && [[ "${FORCE:-0}" != 1 ]]; then
    echo "[mux] $missing of $n frames missing in $OUT_DIR/frames — render them first (or FORCE=1)"; exit 2
  fi
  local off dur
  off=$(awk -v s="$s" -v f="$FPS" 'BEGIN{printf "%.4f", (s-1)/f}')
  dur=$(awk -v n="$n" -v f="$FPS" 'BEGIN{printf "%.4f", n/f}')
  # explicit audio window + maps; no -frames:v (it would stop ALL streams early)
  ffmpeg -hide_banner -loglevel error -y -framerate "$FPS" -start_number "$s" \
    -i "$OUT_DIR/frames/f_%04d.png" -ss "$off" -t "$dur" -i "$AUDIO" -map 0:v -map 1:a \
    -t "$dur" -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p \
    -movflags +faststart -c:a aac -b:a 320k "$MP4"
  echo "[mux] $MP4 ($n frames from $s, audio offset ${off}s)"
}

cmd=${1:-}
case "$cmd" in
  build) build ;;
  frames)
    [[ -f "$BLEND" ]] || build
    vram_watch
    render_range "${2:-1}" "${3:-$(nframes)}"
    ;;
  stills)
    [[ -f "$BLEND" ]] || build
    vram_watch
    gpu "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_frames.py" -- \
      --frames "$2" --out "$OUT_DIR/stills" --pct "${3:-$PCT}" --samples "$SAMPLES" --overwrite \
      | grep -E '^TIME' || true
    ;;
  mux) mux "${2:-1}" "${3:-$(nframes)}" ;;
  all)
    [[ -f "$BLEND" ]] || build
    N=$(nframes)
    render_range 1 "$N"
    mux 1 "$N"
    ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
