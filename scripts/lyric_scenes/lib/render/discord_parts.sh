#!/usr/bin/env bash
# Split a finished music-scene master into Discord-sized parts (two-pass H.264 at a computed bitrate).
#   discord_parts.sh MASTER.mp4 OUTDIR [PARTS=3] [TARGET_MB=7.5] [WIDTH=960]
# Be My One (128.4 s) → 3 × ~43 s, 960×540, ~1.27 Mbps video + 128 kbps AAC ≈ 7.5 MB each (2026-10-14).
# 2026-10-15 (moved here from the pipeline skill): a part that lands over LIMIT_MB (default 8.0, Discord's cap) is
# re-encoded at 90 % of the bitrate, up to 3 times — Be My One's part 2 came out at 7.70 MB, too close to the cap.
set -euo pipefail
src=$1; out=$2; parts=${3:-3}; mb=${4:-7.5}; w=${5:-960}; limit=${LIMIT_MB:-8.0}
mkdir -p "$out"
dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$src")
seg=$(python3 -c "import math;print(math.ceil($dur/$parts))")
vk0=$(python3 -c "print(max(100, int($mb*8*1000*1000/$seg/1000 - 128 - 20)))")     # kbps video (−128 audio, −20 mux)
base=$(basename "${src%.*}")
for ((i = 0; i < parts; i++)); do
  ss=$((i * seg)); o="$out/${base}_p$((i + 1)).mp4"; log="$out/.2pass_$i"; vk=$vk0
  for try in 1 2 3 4; do
    ffmpeg -hide_banner -loglevel error -y -ss "$ss" -t "$seg" -i "$src" -vf "scale=$w:-2" -c:v libx264 -preset slow \
      -b:v "${vk}k" -pass 1 -passlogfile "$log" -an -f mp4 /dev/null
    ffmpeg -hide_banner -loglevel error -y -ss "$ss" -t "$seg" -i "$src" -vf "scale=$w:-2" -c:v libx264 -preset slow \
      -b:v "${vk}k" -pass 2 -passlogfile "$log" -c:a aac -b:a 128k -movflags +faststart "$o"
    rm -f "$log"*
    sz=$(python3 -c "import os;print(os.path.getsize('$o')/1e6)")
    python3 -c "import sys; sys.exit(0 if $sz <= $limit else 1)" && break
    echo "[parts] $o is $sz MB > $limit MB — retry at 90 % bitrate" >&2
    vk=$((vk * 9 / 10))
  done
  printf '%s  %.2f MB  (%s kbps video)\n' "$o" "$sz" "$vk"
done
