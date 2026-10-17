#!/bin/bash
# ONE command for the boss-fight animatic (RENDERS — only with Iris's go): frames (2 capped workers) → mux with the
# song → labelled event contact sheet.   usage: animatic.sh [S E]   (default: the whole song 1..7628)
#   vertical slice 0:00–1:33 only:  animatic.sh 1 2790
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
SCENE="$REPO/scripts/lyric_scenes/lib/render/scene.sh"
ENV="$HERE/animatic_v2.env"
S=${1:-1}; E=${2:-7628}
"$SCENE" "$ENV" full "$S" "$E"
"$SCENE" "$ENV" wait
~/venvs/iris-v100/bin/python -I "$HERE/sheet_events.py" "$REPO/scratch/scenes/boss_fight/animatic_v2/frames" \
  "$REPO/scripts/lyric_scenes/real/boss_fight/choreo_v2.json" "$REPO/scratch/scenes/boss_fight/animatic_v2_sheet.jpg" \
  --from "$S" --to "$E"
ls -la "$REPO/scratch/scenes/boss_fight/"animatic_v2*.mp4 "$REPO/scratch/scenes/boss_fight/animatic_v2_sheet.jpg"
~/venvs/iris-v100/bin/python -I "$HERE/notes_sheet.py" "$REPO/scratch/scenes/boss_fight/animatic_v2/frames" \
  "$REPO/scratch/scenes/boss_fight/animatic_v2_notes_sheet.jpg"
