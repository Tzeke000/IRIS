#!/usr/bin/env bash
# End-to-end test of lib/render/scene.sh on a tiny 48-frame scene (all Blender runs go through scripts/server/render.sh
# because scene.sh launches them that way). ~2–4 min. Prints PASS/FAIL lines, exits 1 on any failure.
#   scripts/lyric_scenes/lib/tests/test_scene_sh.sh [WORKDIR=/tmp/libscene_test]
set -uo pipefail
LIB="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
SCENE="$LIB/render/scene.sh"
W=${1:-/tmp/libscene_test}
rm -rf "$W"; mkdir -p "$W"
FAILS=0
pass() { echo "[test_scene_sh] PASS $*"; }
fail() { echo "[test_scene_sh] FAIL $*"; FAILS=$((FAILS + 1)); }
chk() { local msg=$1; shift; if "$@"; then pass "$msg"; else fail "$msg"; fi; }
ffmpeg -v error -y -f lavfi -i "sine=frequency=440:duration=3" -ar 48000 "$W/song.wav"
cat > "$W/scene.env" <<EOF
NAME=libtest
BLEND=$W/tiny.blend
OUT_DIR=$W/out
AUDIO=$W/song.wav
NF=48
FPS=30
CHUNK=8
WORKERS=2
CAM_PREFIX=CAM_
CAM_PROPS='{"front": {"exposure_offset": -3.0}}'
BUILD_SCRIPT=$LIB/tests/bpy_tiny_scene.py
BUILD_ARGS="--out \$BLEND --nf 48"
PARTS=2
PART_MB=0.2
PART_W=128
EOF
E="$W/scene.env"
mean() { "$HOME/venvs/iris-v100/bin/python" -I -c "import sys,numpy as n;from PIL import Image;print(n.asarray(Image.open(sys.argv[1]).convert('L')).mean())" "$1"; }

# 1. build (blocking unit libtest-build)
"$SCENE" "$E" build >"$W/build.log" 2>&1
chk "build → blend exists" test -f "$W/tiny.blend"

# 2. stills by frame and by camera (+ sheet)
"$SCENE" "$E" stills 1,24 >"$W/stills1.log" 2>&1
chk "stills by frame" test -f "$W/out/stills/f_0001.png" -a -f "$W/out/stills/f_0024.png"
"$SCENE" "$E" stills front,top 100 >"$W/stills2.log" 2>&1
chk "stills by camera" test -f "$W/out/stills/front.png" -a -f "$W/out/stills/top.png"
chk "stills sheet written, < 150 KB" test -f "$W/out/stills/sheet.jpg" -a "$(stat -c %s "$W/out/stills/sheet.jpg" 2>/dev/null || echo 999999)" -lt 150000
# CAM_PROPS exposure −3 on 'front' must darken it vs the same view rendered as frame 1 by frame (scene camera = front)
m_f1=$(mean "$W/out/stills/f_0001.png"); m_front=$(mean "$W/out/stills/front.png")
chk "CAM_PROPS exposure_offset applied (frame1 $m_f1 > front $m_front)" python3 -c "import sys; sys.exit(0 if $m_f1 > 2 * $m_front else 1)"

# 3. frames, foreground, one worker
FG=1 "$SCENE" "$E" frames 1 12 >"$W/frames1.log" 2>&1
chk "frames 1..12 complete" test "$(python3 "$LIB/render/pngcheck.py" missing "$W/out/frames" 1 12 | head -1)" == "MISSING 0"
# 4. the torn-PNG bug: truncate f_0005 (still > 1 KB? make it non-empty) + a stale tmp → resume re-renders exactly that one
truncate -s 600 "$W/out/frames/f_0005.png"; echo junk > "$W/out/frames/.tmp_f_0006.99999.png"
FG=1 "$SCENE" "$E" frames 1 12 >"$W/frames2.log" 2>&1
chk "resume re-rendered ONLY the torn frame" grep -q "rendered 1, skipped 7" "$W/frames2.log"
chk "torn frame now complete" python3 -c "import sys; sys.path.insert(0,'$LIB/render'); import pngcheck; sys.exit(0 if pngcheck.is_complete_png('$W/out/frames/f_0005.png') else 1)"
chk "status/torn reports the stale tmp left by a killed run" grep -q "TORN 1" <(python3 "$LIB/render/pngcheck.py" torn "$W/out/frames")
# 5. mux refuses with frames missing
"$SCENE" "$E" mux >"$W/mux0.log" 2>&1; rc=$?
chk "mux refuses on missing frames (rc=$rc)" test "$rc" -ne 0
# 6. par 2 workers over 13..48 + wait
"$SCENE" "$E" par 13 48 >"$W/par.log" 2>&1
sleep 2
"$SCENE" "$E" wait >"$W/wait.log" 2>&1
chk "par+wait → all 48 frames complete" test "$(python3 "$LIB/render/pngcheck.py" missing "$W/out/frames" 1 48 | head -1)" == "MISSING 0"
chk "both workers rendered (2 units launched)" test "$(grep -c 'launched libtest-w' "$W/par.log")" -eq 2
# 7. mux + duration + audio
"$SCENE" "$E" mux >"$W/mux.log" 2>&1
d=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$W/out/libtest.mp4" 2>/dev/null || echo 0)
chk "mux duration 1.6 s (got $d)" python3 -c "import sys; sys.exit(0 if abs($d - 1.6) < 0.07 else 1)"
chk "mux has an audio stream" test "$(ffprobe -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 "$W/out/libtest.mp4")" == aac
# 8. partial mux names its range
"$SCENE" "$E" mux 13 24 >"$W/mux2.log" 2>&1
chk "partial mux file" test -f "$W/out/libtest_f13-24.mp4"
# 9. full from scratch on a second output dir (par → wait → gap → mux → DONE), with a frame pre-torn
mkdir -p "$W/out2/frames"; cp "$W/out/frames/f_0001.png" "$W/out2/frames/"; truncate -s 700 "$W/out2/frames/f_0001.png"
OUT_DIR="$W/out2" "$SCENE" "$E" full >"$W/full.log" 2>&1
sleep 3
"$SCENE" "$E" wait >>"$W/full.log" 2>&1
chk "full → DONE line" grep -q "libtest frames 1-48 complete" "$W/out2/libtest_DONE"
chk "full → mp4" test -f "$W/out2/libtest.mp4"
chk "full's gap/worker pass re-rendered the pre-torn f_0001" python3 -c "import sys; sys.path.insert(0,'$LIB/render'); import pngcheck; sys.exit(0 if pngcheck.is_complete_png('$W/out2/frames/f_0001.png') else 1)"
chk "frame logs carry no user add-on noise (factory startup)" bash -c "! grep -q MIDIAnimator '$W/frames2.log'"
# 10. parts
"$SCENE" "$E" parts >"$W/parts.log" 2>&1
chk "parts → 2 files" test "$(ls "$W"/out/discord/libtest_p*.mp4 2>/dev/null | wc -l)" -eq 2
# 11. guard rails
sed 's#^BLEND=.*#BLEND=relative.blend#' "$E" > "$W/bad.env"
"$LIB/render/scene.sh" "$W/bad.env" status >/dev/null 2>&1; rc=$?
chk "relative path in scene.env rejected (rc=$rc)" test "$rc" -eq 2
echo "[test_scene_sh] $([[ $FAILS == 0 ]] && echo 'ALL PASS' || echo "$FAILS FAILED") — logs in $W"
exit $((FAILS > 0))
