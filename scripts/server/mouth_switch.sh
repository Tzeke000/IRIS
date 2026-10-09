#!/usr/bin/env bash
# mouth_switch.sh fish|styletts|status — choose my voice engine (one at a time, ever). 2026-10-09.
# Writes ~/.config/iris/mouth_engine; restarts iris-mouth only if my voice is ON right now (quiet hours / Zeke's
# off-flag are respected: the choice just lands at the next start).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
F="$HOME/.config/iris/mouth_engine"
case "${1:-status}" in
  status)
    echo "selected: $(cat "$F" 2>/dev/null || echo styletts)  unit: $(systemctl --user is-active iris-mouth)"
    tail -3 "$ROOT/state/mouth_engine.log" 2>/dev/null; exit 0 ;;
  fish|styletts) ;;
  turbo) echo "turbo is the back pocket — no server built yet"; exit 2 ;;
  *) echo "usage: mouth_switch.sh fish|styletts|status"; exit 2 ;;
esac
mkdir -p "$(dirname "$F")"; echo "$1" > "$F"; rm -f "$ROOT/state/mouth_fish_fails"
OFF=$(python3 -c "import json;print(json.load(open('$ROOT/state/voice_deliberately_off.json')).get('off',True))" 2>/dev/null || echo False)
if [ "$OFF" = "True" ]; then echo "selected $1 — voice is deliberately off now; it starts with $1 next time"; exit 0; fi
systemctl --user restart iris-mouth && echo "selected $1 — iris-mouth restarting (fish warms ~4-10 min; /health 503 until ready)"
