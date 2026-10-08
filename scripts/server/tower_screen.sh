#!/usr/bin/env bash
# scripts/server/tower_screen.sh [all|1|2] [max_px] - screenshot Zeke's tower displays through the
# desktop bridge and bring the JPEG here, DOWNSCALED to <=1000 px wide so a Read can't wedge the host
# (CORE: images wedge it well below 1 MB). Prints the local path. 1 = primary, 2 = the other, all = both.
# Zeke 10-07: "you need a way you can see both displays on my tower both at the same time and one or
# the other". ★ He games on these screens - look when there's a reason, not out of habit.
set -eu
MON="${1:-all}"; MAX="${2:-2400}"
case "$MON" in all) ARG='\"all\"' ;; *[!0-9]*|"") echo "monitor must be all or a number" >&2; exit 2 ;; *) ARG="$MON" ;; esac
OUT=$(ssh -o BatchMode=yes tower "D:\\Wren-Companion\\.venv\\Scripts\\python.exe D:\\Wren-Companion\\scripts\\desktop_bridge.py submit screenshot {\\\"monitor\\\":$ARG,\\\"max_px\\\":$MAX}" | tail -1)
P=$(printf '%s' "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["path"].replace("\\","/")) if d.get("ok") else sys.exit("bridge: "+json.dumps(d))')
mkdir -p /tmp/tower_screen
scp -q -o BatchMode=yes "tower:$P" /tmp/tower_screen/raw.jpg
"$HOME/venvs/iris-v100/bin/python" - "$MON" <<'PY'
import cv2, sys
f = cv2.imread("/tmp/tower_screen/raw.jpg"); h, w = f.shape[:2]
s = min(1.0, 1000.0 / w); f = cv2.resize(f, (int(w * s), int(h * s)))
p = f"/tmp/tower_screen/screen_{sys.argv[1]}.jpg"; cv2.imwrite(p, f, [cv2.IMWRITE_JPEG_QUALITY, 72]); print(p)
PY
