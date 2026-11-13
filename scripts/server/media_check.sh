#!/usr/bin/env bash
# media_check.sh FILE... — CHECK AUDIO/VIDEO BEFORE IT GOES TO ZEKE (11-13).
# Why: the first soundpass audition I sent (11-13) was CORRUPT — he got a broken file and the round trip was wasted.
# A file that exists and has a size is not a file that plays. For each FILE this does:
#   1. ffprobe: has an audio and/or video stream, duration > 0.2 s
#   2. a FULL decode (ffmpeg -v error … -f null -) — any decode error ⇒ ✗ (this is what catches truncated/corrupt files)
#   3. audio not silent: max_volume > -50 dB (a silent mix "plays" fine and still wastes his time)
#   4. size vs Discord's limit (~10 MB for a bot attachment; warn > 9.5 MB ⇒ split, see the music-scene skill)
# Exit 0 only if every file passes. Use it as the LAST step before discord reply files=[…]:
#   scripts/server/media_check.sh scratch/audition.ogg && <send>
set -uo pipefail
[ $# -ge 1 ] || { sed -n 2,10p "$0"; exit 2; }
bad=0
for f in "$@"; do
  if [ ! -s "$f" ]; then echo "✗ $f: missing or empty"; bad=1; continue; fi
  pj=$(ffprobe -v error -show_entries format=duration:stream=codec_type,codec_name -of json "$f" 2>&1) || { echo "✗ $f: ffprobe can't read it: $(tr -s "\n{} " " " <<<"$pj" | cut -c1-120)"; bad=1; continue; }
  read -r dur kinds < <(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); s=d.get("streams",[]); print(float(d.get("format",{}).get("duration") or 0), ",".join(x["codec_type"]+":"+x.get("codec_name","?") for x in s) or "none")' "$pj")
  errs=$(ffmpeg -nostdin -v error -i "$f" -f null - 2>&1 | head -5)
  sz=$(stat -c %s "$f"); mb=$(python3 -c "print(f'{$sz/1048576:.1f}')")
  msg="$f: ${dur%.*}s ${kinds} ${mb} MB"
  if [ "$kinds" = none ]; then echo "✗ $msg — no streams"; bad=1; continue; fi
  if python3 -c "import sys; sys.exit(0 if $dur > 0.2 else 1)"; then :; else echo "✗ $msg — duration ~0"; bad=1; continue; fi
  if [ -n "$errs" ]; then echo "✗ $msg — DECODE ERRORS: $(tr '\n' ' ' <<<"$errs" | cut -c1-200)"; bad=1; continue; fi
  if grep -q audio <<<"$kinds"; then
    mv=$(ffmpeg -nostdin -i "$f" -map 0:a:0 -af volumedetect -f null - 2>&1 | grep -o 'max_volume: [-0-9.]*' | awk '{print $2}')
    if python3 -c "import sys; sys.exit(0 if float('${mv:--99}') > -50 else 1)"; then msg+=" peak ${mv} dB"
    else echo "✗ $msg — audio is SILENT (peak ${mv:-?} dB)"; bad=1; continue; fi
  fi
  python3 -c "import sys; sys.exit(0 if $sz < 9.5*1048576 else 1)" || msg+=" ⚠ > 9.5 MB: Discord may refuse it — split/compress"
  echo "✓ $msg"
done
exit $bad
