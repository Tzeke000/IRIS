#!/usr/bin/env bash
# ROCKET LAUNCH v2 — prep / build / render / mux for "Hit a Bump" (see README.md in this folder).
#
#   render.sh prep                        downloads → /data/iris_assets/rocket2/rocket2_lib.blend
#                                         (once; see ASSETS.md for what must be on disk first)
#   render.sh build                       (re)build rocket2.blend from the timeline npz
#   render.sh frames [START] [END] [PCT]  render Blender frames START..END (1-based; default
#                                         the whole song) → $OUT_DIR/frames/f_####.png.
#                                         RESUMABLE: finished frames are skipped; each frame is
#                                         written to a temp name then renamed (no torn PNGs).
#   render.sh stills F1,F2,... [PCT]      look-dev stills → $OUT_DIR/stills/
#   render.sh mux [START] [END]           frames → MP4 with the song, audio offset to START
#                                         (refuses if frames are missing; FORCE=1 overrides)
#   render.sh all                         build if missing, frames 1..N, mux
#
# env: OUT_DIR (scratch/scenes/rocket2)  CHUNK (frames per Blender launch, 60)  VRAM=1 (log GPU MB)
#      SAMPLES (EEVEE TAA samples; default = what the build set)  PCT (resolution %, 100)
#      MP4 (output path, $OUT_DIR/hit_a_bump_rocket2.mp4)  ASSETS (/data/iris_assets)
# GPU etiquette: every Blender launch runs under `flock /tmp/iris_gpu_render.lock nice -n 15`,
# one chunk at a time, so other GPU jobs can interleave between chunks.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BLENDER="${BLENDER:-$HOME/.local/bin/blender}"
OUT_DIR="${OUT_DIR:-$REPO/scratch/scenes/rocket2}"
BLEND="${BLEND:-$OUT_DIR/rocket2.blend}"
ASSETS="${ASSETS:-/data/iris_assets}"
LIB="${LIB:-$ASSETS/rocket2/rocket2_lib.blend}"
AUDIO="${AUDIO:-$REPO/state/tzeke_songs/hit_a_bump/hit_a_bump_half.wav}"
TIMELINE="${TIMELINE:-$REPO/state/tzeke_songs/hit_a_bump/timeline_30fps.npz}"
CHUNK="${CHUNK:-60}"
PCT="${PCT:-100}"
SAMPLES="${SAMPLES:-0}"
MP4="${MP4:-$OUT_DIR/hit_a_bump_rocket2.mp4}"
VRAMFLAG=""; [[ "${VRAM:-0}" == 1 ]] && VRAMFLAG="--vram"
LOCK=/tmp/iris_gpu_render.lock
FPS=30

gpu() { flock "$LOCK" nice -n 15 "$@"; }

nframes() {   # number of frames in the song (from the timeline)
  "$BLENDER" -b --factory-startup --python-expr \
    "import numpy as np; print('NFRAMES', len(np.load('$TIMELINE')['rms']))" 2>/dev/null \
    | awk '/^NFRAMES/{print $2}'
}

prep() {   # downloaded models are DATA: --disable-autoexec, the NASA .blend is read without scripts
  "$BLENDER" -b --factory-startup --disable-autoexec --python-exit-code 1 \
    --python "$HERE/prep_assets.py" -- --assets "$ASSETS" --out "$LIB"
}

build() {
  mkdir -p "$OUT_DIR"
  [[ -f "$LIB" ]] || prep
  "$BLENDER" -b --factory-startup --python-exit-code 1 \
    --python "$HERE/build_scene.py" -- --timeline "$TIMELINE" --audio "$AUDIO" --out "$BLEND" \
    --assets "$ASSETS" --lib "$LIB"
}

render_range() {  # START END
  local s=$1 e=$2 c ce
  mkdir -p "$OUT_DIR/frames"
  rm -f "$OUT_DIR"/frames/.tmp_f_*.png                 # leftovers of a killed run
  for ((c = s; c <= e; c += CHUNK)); do
    ce=$((c + CHUNK - 1)); ((ce > e)) && ce=$e
    # skip whole chunks that are already complete (no Blender launch at all)
    local missing=0 f
    for ((f = c; f <= ce; f++)); do
      [[ -s "$OUT_DIR/frames/$(printf 'f_%04d.png' "$f")" ]] || { missing=1; break; }
    done
    ((missing)) || continue
    echo "[render] frames $c..$ce  $(date +%H:%M:%S)"
    gpu "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_frames.py" -- \
      --start "$c" --end "$ce" --out "$OUT_DIR/frames" --pct "$PCT" --samples "$SAMPLES" $VRAMFLAG \
      | grep -E '^(TIME|VRAM|\[render_frames\])' || true
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
  local off
  off=$(awk -v s="$s" -v f="$FPS" 'BEGIN{printf "%.4f", (s-1)/f}')
  local dur
  dur=$(awk -v n="$n" -v f="$FPS" 'BEGIN{printf "%.4f", n/f}')
  # explicit audio window + maps; NO -frames:v (it stops ALL streams early → audio cut short);
  # the output -t bounds the video to n frames too
  ffmpeg -hide_banner -loglevel error -y -framerate "$FPS" -start_number "$s" \
    -i "$OUT_DIR/frames/f_%04d.png" -ss "$off" -t "$dur" -i "$AUDIO" -map 0:v -map 1:a \
    -t "$dur" -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p \
    -movflags +faststart -c:a aac -b:a 320k "$MP4"
  echo "[mux] $MP4 ($n frames from $s, audio offset ${off}s)"
}

cmd=${1:-}
case "$cmd" in
  prep) prep ;;
  build) build ;;
  frames)
    [[ -f "$BLEND" ]] || build
    render_range "${2:-1}" "${3:-$(nframes)}"
    ;;
  stills)
    [[ -f "$BLEND" ]] || build
    gpu "$BLENDER" -b "$BLEND" --python-exit-code 1 --python "$HERE/render_frames.py" -- \
      --frames "$2" --out "${STILLS:-$OUT_DIR/stills}" --pct "${3:-$PCT}" --samples "$SAMPLES" --overwrite $VRAMFLAG \
      | grep -E '^(TIME|VRAM)' || true
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
