#!/usr/bin/env bash
# ROCKET LAUNCH v4 ("Blast Off") — prep / build / render / HUD post pass / mux (see README.md in this folder).
# v4: source scratch/scenes/blastoff4_env.sh first. EVERY Blender render must run inside the capped slice:
#     scripts/server/render.sh <unit> scripts/lyric_scenes/real/rocket4/render.sh frames ...
#
#   render.sh prep                        downloads → /data/iris_assets/rocket2/rocket2_lib.blend (the v2 library)
#                                         (once; see ASSETS.md for what must be on disk first)
#   render.sh build                       (re)build rocket4.blend from the timeline npz
#   render.sh frames [START] [END] [PCT]  render Blender frames START..END (1-based; default
#                                         the whole song) → $OUT_DIR/frames/f_####.png.
#                                         RESUMABLE: finished frames are skipped; each frame is
#                                         written to a temp name then renamed (no torn PNGs).
#   render.sh stills F1,F2,... [PCT]      look-dev stills → $OUT_DIR/stills/
#   render.sh hud [START] [END]           v4 warning-HUD post pass (post_hud.py) over the finished frames that the
#                                         build's hud_plan.json marks → $OUT_DIR/hud/f_####.png (beauty untouched)
#   render.sh mux [START] [END]           frames (hud/ version where it exists) → MP4 with the song, audio offset to START
#                                         (refuses if frames are missing; FORCE=1 overrides)
#   render.sh all                         build if missing, frames 1..N, mux
#
# env: OUT_DIR (scratch/scenes/rocket4_blastoff)  CHUNK (frames per Blender launch, 60)  VRAM=1 (log GPU MB)
#      SAMPLES (EEVEE TAA samples; default = what the build set)  PCT (resolution %, 100)
#      MP4 (output path, $OUT_DIR/blast_off_rocket_v4.mp4)  ASSETS (/data/iris_assets)
# GPU etiquette: every Blender launch runs under `flock /tmp/iris_gpu_render.lock nice -n 15`,
# one chunk at a time, so other GPU jobs can interleave between chunks.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BLENDER="${BLENDER:-$HOME/.local/bin/blender}"
OUT_DIR="${OUT_DIR:-$REPO/scratch/scenes/rocket4_blastoff}"
BLEND="${BLEND:-$OUT_DIR/rocket4.blend}"
ASSETS="${ASSETS:-/data/iris_assets}"
LIB="${LIB:-$ASSETS/rocket2/rocket2_lib.blend}"                # the v2 library (unchanged)
AUDIO="${AUDIO:-$REPO/state/tzeke_songs/blast_off/blast_off.wav}"
TIMELINE="${TIMELINE:-$REPO/state/tzeke_songs/blast_off/timeline_30fps.npz}"
CHUNK="${CHUNK:-60}"
PCT="${PCT:-100}"
SAMPLES="${SAMPLES:-0}"
MP4="${MP4:-$OUT_DIR/blast_off_rocket_v4.mp4}"
VRAMFLAG=""; [[ "${VRAM:-0}" == 1 ]] && VRAMFLAG="--vram"
LOCK="${LOCK:-/tmp/iris_gpu_render.lock}"   # parallel workers: give each its own
FPS=30

gpu() { flock "$LOCK" nice -n 15 "$@"; }

nframes() {   # number of frames in the song (from the timeline)
  "$BLENDER" -b --factory-startup --python-expr \
    "import numpy as np; print('NFRAMES', len(np.load('$TIMELINE')['rms']))" 2>/dev/null \
    | awk '/^NFRAMES/{print $2}'
}

prep() {   # downloaded models are DATA: --disable-autoexec, the NASA .blend is read without scripts
  "$BLENDER" -b --factory-startup --disable-autoexec --python-exit-code 1 \
    --python "$HERE/prep_assets.py" -- --assets "$ASSETS" --out "$LIB"   # (writes the rocket2 library)
}

build() {
  mkdir -p "$OUT_DIR"
  [[ -f "$LIB" ]] || prep
  "$BLENDER" -b --factory-startup --python-exit-code 1 \
    --python "$HERE/build_scene.py" -- --timeline "$TIMELINE" --audio "$AUDIO" --out "$BLEND" \
    --assets "$ASSETS" --lib "$LIB" ${BUILD_ARGS:-}
}

render_range() {  # START END
  local s=$1 e=$2 c ce
  mkdir -p "$OUT_DIR/frames"
  [[ "${NO_TMP_CLEAN:-0}" == 1 ]] || rm -f "$OUT_DIR"/frames/.tmp_f_*.png   # leftovers of a killed run (parallel workers: skip)
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

hud() {  # [START END] — the warning HUD as a separate compositor pass (text edits never need a 3-D re-render)
  local s=${1:-1} e=${2:-$(nframes)}
  mkdir -p "$OUT_DIR/hud"
  gpu "$BLENDER" -b --factory-startup --python-exit-code 1 --python "$HERE/post_hud.py" -- \
    --plan "$OUT_DIR/hud_plan.json" --src "$OUT_DIR/frames" --out "$OUT_DIR/hud" --start "$s" --end "$e" \
    --pct "${HUD_PCT:-100}" | grep -E '^(POST|\[post_hud\])' || true
}

mux() {  # [START END]
  local s=${1:-1} e=${2:-$(nframes)} n missing=0 f src
  n=$((e - s + 1))
  for ((f = s; f <= e; f++)); do
    [[ -s "$OUT_DIR/frames/$(printf 'f_%04d.png' "$f")" ]] || missing=$((missing + 1))
  done
  # the HUD frames replace their beauty frames in the cut (symlink farm, nothing copied)
  local farm="$OUT_DIR/.mux_farm"
  rm -rf "$farm"; mkdir -p "$farm"
  for ((f = s; f <= e; f++)); do
    src="$OUT_DIR/frames/$(printf 'f_%04d.png' "$f")"
    [[ -s "$OUT_DIR/hud/$(printf 'f_%04d.png' "$f")" ]] && src="$OUT_DIR/hud/$(printf 'f_%04d.png' "$f")"
    ln -s "$src" "$farm/$(printf 'f_%04d.png' "$f")"
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
    -i "$farm/f_%04d.png" -ss "$off" -t "$dur" -i "$AUDIO" -map 0:v -map 1:a \
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
  hud) hud "${2:-1}" "${3:-$(nframes)}" ;;
  mux) mux "${2:-1}" "${3:-$(nframes)}" ;;
  all)
    [[ -f "$BLEND" ]] || build
    N=$(nframes)
    render_range 1 "$N"
    hud 1 "$N"
    mux 1 "$N"
    ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
