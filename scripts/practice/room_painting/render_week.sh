#!/bin/sh
# Render one week's room painting through the capped render slice (never a bare blender).
# usage: scripts/practice/room_painting/render_week.sh <WEEK e.g. 2026-W43> [pct] [samples] [extra build args...]
# Output: scratch/practice/room_painting/<WEEK>/room_<WEEK>[_pctNN].png (+ build.log). Returns immediately;
# poll `systemctl --user is-active iris-roompaint-<WEEK>-<pct>` or the PNG.
set -e
WEEK=${1:?week like 2026-W43}; PCT=${2:-100}; SAMPLES=${3:-64}
[ $# -ge 3 ] && shift 3 || shift $#
REPO=/home/iris/IRIS
OUTDIR=$REPO/scratch/practice/room_painting/$WEEK
mkdir -p "$OUTDIR"
SUFFIX=""; [ "$PCT" != 100 ] && SUFFIX="_pct$PCT"
OUT=$OUTDIR/room_$WEEK$SUFFIX.png
UNIT=iris-roompaint-$WEEK-$PCT-$(date +%H%M%S)
echo "unit: $UNIT"; echo "out:  $OUT"
exec "$REPO/scripts/server/render.sh" "$UNIT" /bin/sh -c "flock /tmp/iris_gpu_render.lock \
  /home/iris/.local/bin/blender --background --factory-startup \
  --python $REPO/scripts/practice/room_painting/build_room_painting.py --python-exit-code 1 -- \
  --out $OUT --pct $PCT --samples $SAMPLES $* > $OUTDIR/build$SUFFIX.log 2>&1"
