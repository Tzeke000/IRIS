#!/usr/bin/env bash
# test_sandbox.sh — prove the bot STARTS under the unit's sandbox WITHOUT touching the live iris-vr-presence:
# a throwaway transient unit (DynamicUser; its own ctl port; NO --join ⇒ never talks to the temple server),
# /data/irisvr re-pointed at a scratch dir.
#   ./test_sandbox.sh [--with-firewall] [seconds=60] [extra -p overrides, e.g. -p MemoryDenyWriteExecute=no]
#   --with-firewall: run as the idle `nobody` uid with a TEMPORARY egress-rule copy for it (proves the socket replies pass)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/lib.sh"
BOT=/data/projects/vr_iris/bot/VRIrisServer.x86_64
FW=0; [ "${1:-}" = --with-firewall ] && { FW=1; shift; }
SECS=${1:-60}; shift || true
U=iris-vr-sandbox-test; PORT=17790
T=$(mktemp -d /data/sandbox/irisvr_test.XXXX); mkdir -p "$T"/{hear,gestures,in,voice}; chmod -R 777 "$T"
trap 'sudo systemctl stop $U 2>/dev/null; sudo systemctl reset-failed $U 2>/dev/null; rm -rf "$T"; [ $FW = 1 ] && sudo "$HERE/egress_rules.sh" remove $(id -u nobody) x IRIS_VR_TEST' EXIT
who=(-p DynamicUser=yes -p StateDirectory=$U -p Environment=HOME=/var/lib/$U -p WorkingDirectory=/var/lib/$U)
if [ $FW = 1 ]; then
  [ -z "$(ps -u nobody -o pid=)" ] || { echo "something runs as nobody — not testing"; exit 2; }
  sudo "$HERE/egress_rules.sh" install "$(id -u nobody)" "$(temple_host)" IRIS_VR_TEST $PORT
  who=(-p User=nobody -p Group=nogroup -p Environment=HOME=/tmp -p WorkingDirectory=/tmp)
fi
sandbox_props "${UNIT_FILE:-$HERE/iris-vr-presence.service}" "$T"
sudo systemctl stop $U 2>/dev/null; sudo systemctl reset-failed $U 2>/dev/null
SINCE=$(date '+%Y-%m-%d %H:%M:%S')
sudo systemd-run --quiet --unit=$U "${who[@]}" -p Nice=5 \
  -p Environment=IRIS_VR_HEAR_DIR=/data/irisvr/hear -p Environment=IRIS_HEARD_PATH=/data/irisvr/gestures/gestures.jsonl \
  "${PROPS[@]}" "$@" \
  $BOT --headless -- --bot --hear=on --ctl-port=$PORT || { echo "systemd-run failed"; exit 1; }
ok=0
for i in $(seq 1 "$SECS"); do
  sleep 1
  [ "$(systemctl is-active $U)" = active ] || { echo "unit died after ${i}s"; break; }
  r=$(python3 -I - "$PORT" <<'PY' 2>/dev/null
import json, socket, sys
s = socket.create_connection(("localhost", int(sys.argv[1])), timeout=2); s.sendall(b'{"cmd":"state"}\n'); b = b""
while not b.endswith(b"\n"):
    c = s.recv(65536)
    if not c: break
    b += c
d = json.loads(b); print("SOCKET OK (%d keys)" % len(d))
PY
)
  [ -n "$r" ] && { echo "after ${i}s: $r"; ok=1; break; }
done
if [ $ok = 1 ]; then   # exposure score of the running copy (DynamicUser stands in for irisvr; the real unit scores the same or better)
  sudo systemd-analyze security $U --no-pager > "$HERE/analyze_test_copy.txt" 2>&1; echo "systemd-analyze security: $(tail -1 "$HERE/analyze_test_copy.txt")"
fi
echo "--- journal: sandbox-relevant lines ---"
sudo journalctl -u $U --since "$SINCE" --no-pager -o cat | grep -iE "denied|not permitted|EPERM|seccomp|read-only|cannot|failed to|\[iris\]|\[hear\]|\[voice\]|Godot Engine" | grep -v OpenXR | mask_ips | sort | uniq -c | head -20
[ $ok = 1 ] && echo "PASS: bot ran under the sandbox and its control socket answered" || { echo "FAIL"; exit 1; }
