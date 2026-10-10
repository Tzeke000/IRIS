#!/bin/bash
# Test-render a scene plugin over a time window of "Hit a Bump" at 960x540.
# usage: scripts/lyric_scenes/test.sh SCENE TAG START:END [extra lyric_viz args]
#   -> scratch/scenes/SCENE/TAG.mp4 (+ TAG.log). Runs at nice 15 so the live
#      voice + vision on this box stay responsive.
# Contact sheet afterwards (≤900 px, <150 KB — safe to Read):
#   ~/venvs/iris-v100/bin/python scripts/lyric_scenes/sheet_at.py \
#       scratch/scenes/SCENE/TAG.mp4 scratch/scenes/SCENE/TAG.jpg 4 0.1,0.5,1,1.5
#   (times are relative to the clip start; 5th arg = label offset, e.g. START)
set -o pipefail
cd "$(dirname "$0")/../.."
scene=$1; tag=$2; win=$3; shift 3
D=state/tzeke_songs/hit_a_bump; O=scratch/scenes/$scene
mkdir -p "$O"
nice -n 15 scripts/server/lyric_viz.sh --audio $D/hit_a_bump_half.wav \
  --logo state/tzeke_songs/_assets/tzeke000_symbol.png --title "HIT A BUMP" \
  --look "$scene" --punch-words bump,bomb --aspect 16:9 --size 960x540 \
  --window "$win" --out "$O/$tag.mp4" "$@" > "$O/$tag.log" 2>&1
rc=$?
grep -v -i warn "$O/$tag.log" | grep -E "done|Error|Traceback|error|timeline|scene|File|line [0-9]" | tail -15
exit $rc
