#!/usr/bin/env bash
# scripts/server/voice_quiet_hours.sh - the server twin of scripts/voice_quiet_hours.ps1 (2026-10-07).
# Runs every 5 min from iris-quiet-hours.timer. Inside the window (default 22:30-05:10, from
# state/voice_quiet_hours.json) it writes state/voice_deliberately_off.json (room MUTED; since 10-29 the
# mouth + ears stay loaded - IRIS_QUIET_STOPS_UNITS=1 restores stopping them); when the window closes it removes the flag ONLY if this script set it (auto_off_active),
# then starts them again. override_night = a night he overrode: stand down for that night.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$ROOT/state/voice_quiet_hours.json"
FLAG="$ROOT/state/voice_deliberately_off.json"
LOG="$ROOT/state/voice_quiet_hours.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
PY="$(command -v python3)"
read -r ENABLED START END ACTIVE OVERRIDE < <("$PY" - "$CFG" <<'EOF'
import json, sys
d = {"enabled": True, "start": "22:30", "end": "05:10", "auto_off_active": False, "override_night": ""}
try:
    d.update(json.load(open(sys.argv[1])))
except Exception:
    pass
print(int(bool(d["enabled"])), d["start"], d["end"], int(bool(d["auto_off_active"])), d["override_night"] or "-")
EOF
)
NOW=$(date +%H:%M)
# the "night" a time belongs to: before the end time it is still last night
if [[ "$NOW" < "$END" ]]; then NIGHT=$(date -d yesterday +%F); else NIGHT=$(date +%F); fi
in_window() { [[ "$NOW" > "$START" || "$NOW" == "$START" || "$NOW" < "$END" ]]; }
setcfg() { "$PY" - "$CFG" "$1" "$2" <<'EOF'
import json, sys, time
p, k, v = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(p))
except Exception:
    d = {}
d[k] = (v == "1") if k == "auto_off_active" else v
d["last_action_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(d, open(p, "w"), indent=1)
EOF
}
[ "$ENABLED" = "1" ] || exit 0
if in_window; then
  [ "$OVERRIDE" = "$NIGHT" ] && exit 0
  if [ ! -f "$FLAG" ]; then
    printf '{"off": true, "why": "quiet hours %s-%s (server)", "since": "%s"}\n' "$START" "$END" "$(date +%FT%T)" > "$FLAG"
    # Zeke 10-29: "mouth and ears on, but MUTED until the morning". The flag mutes the ROOM (host routes
    # voice replies to Discord, the ears loop sleeps); the units stay LOADED so there is no cold ~3 min
    # warm-up at 05:10 and my VR voice (same mouth) keeps working at night. Stopping them is opt-in.
    if [ "${IRIS_QUIET_STOPS_UNITS:-0}" = "1" ]; then systemctl --user stop iris-ears.service iris-mouth.service; fi
    setcfg auto_off_active 1; setcfg last_action "auto_off"
    log "window open ($START-$END): voice MUTED (flag written; mouth + ears stay loaded)"
  fi
else
  if [ "$ACTIVE" = "1" ] && [ -f "$FLAG" ]; then
    rm -f "$FLAG"
    systemctl --user start iris-mouth.service iris-ears.service
    setcfg auto_off_active 0; setcfg last_action "auto_on"
    log "window closed: voice ON (flag removed, mouth + ears started)"
  fi
fi
