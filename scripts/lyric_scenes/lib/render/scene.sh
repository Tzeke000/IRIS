#!/usr/bin/env bash
# lib/render/scene.sh — build / stills / frames / par / wait / mux / full / parts for ANY scene, driven by its scene.env.
# Replaces the per-scene render.sh + par_frames.sh + render_all.sh + full_render.sh + chain scripts.
#
#   scene.sh /ABS/real/<scene>/scene.env VERB [args]       (run from your shell — it launches the units itself)
#
#   build                     run BUILD_SCRIPT → BLEND                        (blocking unit <NAME>-build)
#   verify [ARGS]             lib/qa/verify_saved.py on the SAVED BLEND in a fresh process → OUT_DIR/verify_saved.json;
#                             exit 1 on a failed check; ARGS + VERIFY_ARGS from scene.env (blocking unit <NAME>-verify)
#   stills SPEC [PCT] [SPP]   SPEC = frames "721,1021,1500-1502" or camera names "a_wide,c_hall" (CAM_PREFIX applied)
#                             → STILLS_DIR (+ sheet.jpg, safe to Read)       (blocking unit <NAME>-stills)
#   frames [S] [E]            one worker over S..E, 60-frame chunks          (DETACHED unit <NAME>-frames; FG=1 blocks)
#   par [S] [E] [W] [TAG]     W workers (default WORKERS=2), chunks interleaved, each its own GPU lock
#                             (DETACHED units <TAG>-w0..; TAG defaults to NAME)
#   wait [TAG]                block until no <TAG>-* unit is active (systemd unit state — never pgrep, see 10-10 deadlock)
#   status                    active units, missing/torn frames
#   mux [S] [E]               complete frames + the song → MP4 (refuses on missing frames unless FORCE=1)
#   full [S] [E]              par → wait → gap pass → mux → "<NAME>_DONE" line   (DETACHED unit <NAME>-full)
#   parts [MP4]               Discord parts: PARTS × ~PART_MB MB at PART_W px (lib/render/discord_parts.sh)
#   sheet [DIR]               contact sheet of DIR (default STILLS_DIR) → DIR/sheet.jpg
#
# scene.env (sourced with SCENE_DIR=<its folder> and REPO=<repo root> already set; ALL PATHS ABSOLUTE):
#   NAME=bossfight                     unit prefix [a-z0-9-]
#   BLEND=/abs/x.blend  OUT_DIR=/abs/dir         frames → $OUT_DIR/frames/f_####.png, stills → $OUT_DIR/stills
#   AUDIO=/abs/song.wav NF=3852 FPS=30           for frames/mux/full defaults
#   BUILD_SCRIPT=$SCENE_DIR/build_scene.py  BUILD_ARGS="--out $BLEND ..."  BUILD_FROM=/abs/base.blend (layered builds)
#   CAM_PREFIX=CAM_  CAM_PROPS='{"c_hall":{"exposure_offset":0.3}}'  STILLS_HOOK=$SCENE_DIR/stills_hook.py  PATCH='{...}'
#   VERIFY_ARGS="--ground Terrain --grounded Props --ignore HazeBox"   (verify_saved.py options, word-split)
#   optional: STILLS_DIR MP4 GRAIN(5; 0=off) CRF(17) CHUNK(60) WORKERS(2) PCT(100) SPP(0=scene) LOCK OVERLAY_DIR
#             PARTS(3) PART_MB(7.5) PART_W(960) MEM(render.sh MemoryMax) BLENDER SHEET_PY
#             ANIM(1: chunks render as ONE animation render per run of missing frames — frames.py --anim; spooky v3
#             measured 6.5 → 2.9 s/frame, pixels equal. ANIM=0 = the old one-still-render-per-frame loop)
#             RENDER_FLAGS (default "--factory-startup": no user add-ons in render runs, deterministic + quieter; set ""
#             if the scene needs an add-on at render time)  BUILD_FLAGS (e.g. --disable-autoexec)
# Command-line env overrides win over scene.env for: PCT SPP CHUNK WORKERS LOCK FORCE GRAIN CRF MP4 STILLS_DIR
#   STILL_AT CAM_PROPS PATCH MEM OUT_DIR FG ANIM — and are forwarded into the units with -E (units start in $HOME with a bare env).
# Every Blender launch runs inside a scripts/server/render.sh unit (iris-workers-render.slice under the shared iris-workers.slice cap, MemoryMax, OOMScoreAdjust)
# and under `flock $LOCK nice -n 15`.
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
LIBR="$(dirname "$SELF")"
LIB="$(dirname "$LIBR")"
REPO="$(cd "$LIB/../../.." && pwd)"
RENDER_SH="$REPO/scripts/server/render.sh"
OVERRIDABLE="PCT SPP CHUNK WORKERS LOCK FORCE GRAIN CRF MP4 STILLS_DIR STILL_AT CAM_PROPS PATCH MEM OUT_DIR FG ANIM"

die() { echo "[scene] $*" >&2; exit 2; }
[[ $# -ge 2 ]] || { sed -n '2,38p' "$SELF"; exit 1; }
SCENE_ENV="$(readlink -f "$1")"; shift
VERB=$1; shift
[[ -f "$SCENE_ENV" ]] || die "no scene.env at $SCENE_ENV"

# --- load scene.env, letting command-line overrides win
declare -A PRE=()
for v in $OVERRIDABLE; do [[ -n "${!v+x}" ]] && PRE[$v]="${!v}"; done
SCENE_DIR="$(dirname "$SCENE_ENV")"
# shellcheck disable=SC1090
source "$SCENE_ENV"
for v in "${!PRE[@]}"; do printf -v "$v" '%s' "${PRE[$v]}"; done

: "${NAME:?scene.env must set NAME}"; : "${BLEND:?scene.env must set BLEND}"; : "${OUT_DIR:?scene.env must set OUT_DIR}"
[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "NAME must be [a-z0-9-] (systemd unit prefix): $NAME"
for v in BLEND OUT_DIR AUDIO BUILD_SCRIPT BUILD_FROM STILLS_DIR MP4 STILLS_HOOK OVERLAY_DIR; do
  [[ -z "${!v:-}" || "${!v}" == /* ]] || die "$v must be an ABSOLUTE path (units start in \$HOME): ${!v}"
done
BLENDER="${BLENDER:-$HOME/.local/bin/blender}"
SHEET_PY="${SHEET_PY:-$HOME/venvs/iris-v100/bin/python}"
FPS="${FPS:-30}"; CHUNK="${CHUNK:-60}"; WORKERS="${WORKERS:-2}"; PCT="${PCT:-100}"; SPP="${SPP:-0}"
GRAIN="${GRAIN:-5}"; CRF="${CRF:-17}"; LOCK="${LOCK:-/tmp/iris_gpu_render.lock}"; ANIM="${ANIM:-1}"
STILLS_DIR="${STILLS_DIR:-$OUT_DIR/stills}"; FRAMES_DIR="$OUT_DIR/frames"; MP4="${MP4:-$OUT_DIR/$NAME.mp4}"
RENDER_FLAGS="${RENDER_FLAGS---factory-startup}"
PARTS="${PARTS:-3}"; PART_MB="${PART_MB:-7.5}"; PART_W="${PART_W:-960}"

gpu() { flock "$LOCK" nice -n 15 "$@"; }
# run Blender under the lock, keep only the useful stdout lines, RETURN BLENDER'S EXIT CODE (the old runners'
# `| grep … || true` swallowed failures; stderr — tracebacks — passes through unfiltered)
gpu_filtered() {
  local rc
  set +e
  gpu "$@" | grep --line-buffered -E '^(TIME|VRAM|\[frames\]|\[time_ab\]|Error|Traceback)'
  rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}
active_units() { systemctl --user list-units --no-legend --plain --state=active,activating,deactivating "$1" 2>/dev/null \
                   | awk '{print $1}' | grep . || true; }
need_nf() { [[ -n "${NF:-}" ]] || die "set NF (frame count) in scene.env or pass S E"; }
missing_count() { python3 "$LIBR/pngcheck.py" missing "$FRAMES_DIR" "$1" "$2" | awk '/^MISSING/{print $2}'; }

# launch "$@" (a scene.sh internal verb) in a capped unit. $1 = unit name, $2 = wait|detach
launch() {
  local unit=$1 mode=$2; shift 2
  [[ -z "$(active_units "$unit.service")" ]] || die "unit $unit is already running (systemctl --user status $unit)"
  local envs=(-E SCENE_IN_UNIT=1 -E "SCENE_ENV=$SCENE_ENV")
  for v in "${!PRE[@]}"; do envs+=(-E "$v=${PRE[$v]}"); done
  [[ -n "${LOCK_W:-}" ]] && envs+=(-E "LOCK=$LOCK_W")
  if [[ $mode == wait ]]; then
    MEM="${MEM:-}" "$RENDER_SH" "$unit" --wait --collect -P "${envs[@]}" "$SELF" "$SCENE_ENV" "$@"
  else
    MEM="${MEM:-}" "$RENDER_SH" "$unit" --collect "${envs[@]}" "$SELF" "$SCENE_ENV" "$@" >/dev/null
    echo "[scene] launched $unit — log: journalctl --user -u $unit -f"
  fi
}

# ------------------------------------------------------------------ internal verbs (run INSIDE a unit)
_build() {
  [[ -n "${BUILD_SCRIPT:-}" ]] || die "scene.env has no BUILD_SCRIPT (this scene is built elsewhere)"
  mkdir -p "$(dirname "$BLEND")"
  local from=(); [[ -n "${BUILD_FROM:-}" ]] && from=("$BUILD_FROM")
  # shellcheck disable=SC2086
  gpu "$BLENDER" -b --factory-startup ${BUILD_FLAGS:-} "${from[@]}" --python-exit-code 1 --python "$BUILD_SCRIPT" -- ${BUILD_ARGS:-}
  [[ -f "$BLEND" ]] || die "build finished but $BLEND does not exist (does BUILD_ARGS pass --out \$BLEND?)"
}

_verify() {   # extra verify_saved.py args
  [[ -f "$BLEND" ]] || die "no $BLEND — build first"
  local run=("$BLENDER" -b --factory-startup "$BLEND" --python-exit-code 1 --python "$LIB/qa/verify_saved.py" --
             --out "$OUT_DIR/verify_saved.json")
  # shellcheck disable=SC2206
  local extra=(${VERIFY_ARGS:-} "$@")
  mkdir -p "$OUT_DIR"
  if [[ " ${extra[*]} " == *" --render "* ]]; then gpu "${run[@]}" "${extra[@]}"; else nice -n 10 "${run[@]}" "${extra[@]}"; fi
}

_stills() {   # SPEC PCT SPP
  local spec=$1 pct=${2:-$PCT} spp=${3:-$SPP} sel=() names=()
  [[ -f "$BLEND" ]] || die "no $BLEND — build first"
  mkdir -p "$STILLS_DIR"
  if [[ "$spec" =~ ^[0-9,-]+$ ]]; then
    sel=(--frames "$spec" --overwrite)
    for f in $(python3 -c "import sys; sys.path.insert(0,'$LIBR'); import frames; print(' '.join(str(x) for x in frames.parse_frames('$spec')))"); do
      names+=("$STILLS_DIR/$(printf 'f_%04d.png' "$f")"); done
  else
    sel=(--cams "$spec" --cam-prefix "${CAM_PREFIX:-}")
    [[ -n "${CAM_PROPS:-}" ]] && sel+=(--cam-props "$CAM_PROPS")
    [[ -n "${STILLS_HOOK:-}" ]] && sel+=(--hook "$STILLS_HOOK")
    [[ -n "${STILL_AT:-}" ]] && sel+=(--at "$STILL_AT")
    IFS=, read -ra cs <<< "$spec"; for c in "${cs[@]}"; do names+=("$STILLS_DIR/$c.png"); done
  fi
  [[ -n "${PATCH:-}" ]] && sel+=(--patch "$PATCH")
  # shellcheck disable=SC2086
  gpu_filtered "$BLENDER" -b $RENDER_FLAGS "$BLEND" --python-exit-code 1 --python "$LIBR/frames.py" -- --out "$STILLS_DIR" "${sel[@]}" \
    --pct "$pct" --samples "$spp" --times "$STILLS_DIR/times.tsv" || die "stills render failed (exit $?)"
  local ok=() n; for n in "${names[@]}"; do python3 -c "import sys; sys.path.insert(0,'$LIBR'); import pngcheck; sys.exit(0 if pngcheck.is_complete_png('$n') else 1)" && ok+=("$n"); done
  [[ ${#ok[@]} -eq ${#names[@]} ]] || die "only ${#ok[@]}/${#names[@]} stills rendered"
  if [[ -x "$SHEET_PY" ]]; then "$SHEET_PY" -I "$LIB/qa/sheet.py" "$STILLS_DIR/sheet.jpg" "${ok[@]}" --title "$NAME stills $(date +%m-%d\ %H:%M)"; fi
}

_chunks() {   # S E W IDX  — worker IDX of W takes chunks IDX, IDX+W, ...
  local s=$1 e=$2 w=${3:-1} i=${4:-0} c ce
  mkdir -p "$FRAMES_DIR"
  [[ -f "$BLEND" ]] || die "no $BLEND — build first"
  for ((c = s + i * CHUNK; c <= e; c += w * CHUNK)); do
    ce=$((c + CHUNK - 1)); ((ce > e)) && ce=$e
    [[ "$(missing_count "$c" "$ce")" == 0 ]] && continue          # complete chunk: no Blender launch at all
    echo "[scene] $NAME frames $c..$ce $(date +%H:%M:%S)"
    local p=(); [[ -n "${PATCH:-}" ]] && p=(--patch "$PATCH")
    [[ "$ANIM" == 1 ]] && p+=(--anim)
    # shellcheck disable=SC2086
    gpu_filtered "$BLENDER" -b $RENDER_FLAGS "$BLEND" --python-exit-code 1 --python "$LIBR/frames.py" -- --out "$FRAMES_DIR" \
      --start "$c" --end "$ce" --pct "$PCT" --samples "$SPP" --times "$OUT_DIR/frame_times.tsv" "${p[@]}" \
      || die "frames $c..$ce failed (exit $?)"
  done
}

_par() {      # S E W TAG — launch W detached worker units
  local s=$1 e=$2 w=$3 tag=$4 k
  for ((k = 0; k < w; k++)); do
    LOCK_W="$LOCK.w$k" launch "$tag-w$k" detach _chunks "$s" "$e" "$w" "$k"
  done
}

_wait() {     # TAG-pattern
  local pat=$1
  while [[ -n "$(active_units "$pat")" ]]; do sleep 10; done
}

_mux() {      # S E
  local s=$1 e=$2 n=$(($2 - $1 + 1)) miss out=$MP4
  [[ -n "${AUDIO:-}" && -f "$AUDIO" ]] || die "AUDIO missing: ${AUDIO:-unset}"
  miss=$(missing_count "$s" "$e")
  if [[ "$miss" != 0 && "${FORCE:-0}" != 1 ]]; then
    die "$miss of $n frames missing/torn in $FRAMES_DIR ($(python3 "$LIBR/pngcheck.py" missing "$FRAMES_DIR" "$s" "$e" | tail -1)) — render them (or FORCE=1)"
  fi
  [[ "$s" == 1 && "$e" == "${NF:-$e}" ]] || out="${MP4%.mp4}_f${s}-${e}.mp4"
  local src="$FRAMES_DIR" farm f
  if [[ -n "${OVERLAY_DIR:-}" ]]; then     # overlay frames (HUD pass) replace their beauty frame; symlink farm, no copies
    farm="$OUT_DIR/.mux_farm"; rm -rf "$farm"; mkdir -p "$farm"; src=$farm
    for ((f = s; f <= e; f++)); do
      local fn; fn=$(printf 'f_%04d.png' "$f")
      if [[ -s "$OVERLAY_DIR/$fn" ]]; then ln -s "$OVERLAY_DIR/$fn" "$farm/$fn"; else ln -s "$FRAMES_DIR/$fn" "$farm/$fn"; fi
    done
  fi
  local off dur vf=()
  off=$(awk -v s="$s" -v f="$FPS" 'BEGIN{printf "%.4f", (s-1)/f}')
  dur=$(awk -v n="$n" -v f="$FPS" 'BEGIN{printf "%.4f", n/f}')
  [[ "$GRAIN" != 0 ]] && vf=(-vf "noise=alls=$GRAIN:allf=t")
  # explicit audio window + maps; NO -frames:v (it stops ALL streams early → audio cut short); output -t bounds both
  nice -n 10 ffmpeg -hide_banner -loglevel error -y -framerate "$FPS" -start_number "$s" -i "$src/f_%04d.png" \
    -ss "$off" -t "$dur" -i "$AUDIO" -map 0:v -map 1:a -t "$dur" "${vf[@]}" -c:v libx264 -preset slow -crf "$CRF" \
    -pix_fmt yuv420p -c:a aac -b:a 320k -movflags +faststart "$out"
  echo "[scene] mux $out ($n frames from $s, audio offset ${off}s, grain $GRAIN)"
}

_full() {     # S E
  local s=$1 e=$2
  _par "$s" "$e" "$WORKERS" "$NAME"
  sleep 5
  _wait "$NAME-w*"
  echo "[scene] workers done; gap pass $(date +%H:%M:%S)"
  _chunks "$s" "$e" 1 0                                    # anything a dead worker left behind
  _mux "$s" "$e"
  echo "$(date -Is) $NAME frames $s-$e complete, muxed → $MP4" | tee -a "$OUT_DIR/${NAME}_DONE"
}

# ------------------------------------------------------------------ dispatch
if [[ "${SCENE_IN_UNIT:-0}" == 1 ]]; then
  case "$VERB" in
    _build|_verify|_stills|_chunks|_full|_mux) "$VERB" "$@" ;;
    *) die "internal verb expected inside a unit, got $VERB" ;;
  esac
  exit 0
fi

case "$VERB" in
  build) launch "$NAME-build" wait _build ;;
  verify) launch "$NAME-verify" wait _verify "$@" ;;
  stills) [[ $# -ge 1 ]] || die "stills SPEC [PCT] [SPP]"; launch "$NAME-stills" wait _stills "$@" ;;
  frames)
    if [[ $# -lt 2 ]]; then need_nf; fi
    launch "$NAME-frames" "$([[ "${FG:-0}" == 1 ]] && echo wait || echo detach)" _chunks "${1:-1}" "${2:-$NF}" 1 0 ;;
  par)
    if [[ $# -lt 2 ]]; then need_nf; fi
    _par "${1:-1}" "${2:-$NF}" "${3:-$WORKERS}" "${4:-$NAME}" ;;
  wait) _wait "${1:-$NAME}-*"; echo "[scene] no ${1:-$NAME}-* units active" ;;
  status)
    echo "[scene] $NAME units:"; active_units "$NAME-*" | sed 's/^/  /'
    if [[ -n "${NF:-}" ]]; then python3 "$LIBR/pngcheck.py" missing "$FRAMES_DIR" 1 "$NF" | sed 's/^/  /'; fi
    [[ -d "$FRAMES_DIR" ]] && python3 "$LIBR/pngcheck.py" torn "$FRAMES_DIR" | tail -1 | sed 's/^/  /' ;;
  mux) if [[ $# -lt 2 ]]; then need_nf; fi; _mux "${1:-1}" "${2:-$NF}" ;;
  full) if [[ $# -lt 2 ]]; then need_nf; fi; launch "$NAME-full" detach _full "${1:-1}" "${2:-$NF}" ;;
  parts) "$LIBR/discord_parts.sh" "${1:-$MP4}" "$OUT_DIR/discord" "$PARTS" "$PART_MB" "$PART_W" ;;
  sheet) d=${1:-$STILLS_DIR}; "$SHEET_PY" -I "$LIB/qa/sheet.py" "$d/sheet.jpg" --dir "$d" --glob '*.png' ;;
  *) sed -n '2,38p' "$SELF"; exit 1 ;;
esac
