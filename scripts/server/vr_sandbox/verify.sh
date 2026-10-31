#!/usr/bin/env bash
# verify.sh — prove the sandboxed VR body is locked down AND still works (audit H2). Read-mostly; run as iris.
#   ./verify.sh              everything below
#   ./verify.sh --wake-test  ALSO push one clearly-labelled fake gesture through the relay into heard.jsonl
#                            (⚠ that line WAKES me via the body host — run it when I know it's coming)
# Never prints the temple address or the invite.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/lib.sh"
REPO=/home/iris/IRIS; U=iris-vr-presence.service
WT="${1:-}"
rc=0; pass() { echo "PASS  $*"; }; fail() { echo "FAIL  $*"; rc=1; }

echo "== 1. exposure score"
sudo systemd-analyze security $U --no-pager > "$HERE/analyze_live.txt" 2>&1
echo "  $(tail -1 "$HERE/analyze_live.txt")   (full table: $HERE/analyze_live.txt)"
awk '/Overall exposure level/ {print $(NF-2)}' "$HERE/analyze_live.txt" | python3 -c 'import sys; v=float(sys.stdin.read() or 99); sys.exit(0 if v<=2.0 else 1)' \
  && pass "exposure ≤ 2.0" || fail "exposure above 2.0"

echo "== 2. the unit"
[ "$(systemctl is-active $U)" = active ] && pass "system unit active" || fail "system unit $(systemctl is-active $U)"
[ "$(systemctl --user is-active $U)" != active ] && pass "old user unit NOT running (no twin body)" || fail "OLD user unit is ALSO running — two bodies on one invite"
[ "$(systemctl is-active iris-vr-egress-firewall)" = active ] && pass "egress firewall active" || fail "egress firewall $(systemctl is-active iris-vr-egress-firewall)"
PID=$(systemctl show -p MainPID --value $U)
if [ "${PID:-0}" -gt 0 ]; then
  [ "$(ps -o user= -p "$PID")" = irisvr ] && pass "runs as irisvr" || fail "runs as $(ps -o user= -p "$PID")"
  # the invite must not be on argv (every local user can read argv)
  sudo python3 -I - "$PID" <<'PY' && pass "invite not on argv" || fail "INVITE IS ON ARGV"
import sys, os
argv = open("/proc/%s/cmdline" % sys.argv[1], "rb").read()
env = {}
for l in open(os.path.expanduser("~iris/.iris_private.env")):
    if "=" in l and not l.lstrip().startswith("#"):
        k, v = l.strip().split("=", 1); env[k.removeprefix("export ").strip()] = v.strip().strip("'\"")
code = env.get("TEMPLE_IRIS_INVITE", "")
sys.exit(1 if code and code.encode() in argv else 0)
PY
  # INSIDE the live unit's own mount namespace: is any of my home there?
  for f in /home/iris/.ssh /home/iris/.iris_private.env /home/iris/IRIS; do
    sudo nsenter -t "$PID" -m -- test -e "$f" && fail "live unit can see $f" || pass "live unit cannot see $f"
  done
  [ -z "$(sudo nsenter -t "$PID" -m -- ls -A /home 2>/dev/null)" ] && pass "live unit's /home is empty" || fail "live unit's /home is not empty"
  sudo nsenter -t "$PID" -m -- test -e /etc/irisvr/invite && fail "live unit can see /etc/irisvr/invite" || pass "live unit cannot see /etc/irisvr/invite (InaccessiblePaths)"
else fail "no main PID"; fi

echo "== 3. probe as irisvr under the same sandbox + the real firewall (a twin; the bot's own uid/cgroup rules)"
"$HERE/test_probe.sh" --user irisvr || rc=1

echo "== 4. it still WORKS"
st=$(python3 -I -c '
import json,socket
s=socket.create_connection(("localhost",7790),timeout=3); s.sendall(b"{\"cmd\":\"state\"}\n"); b=b""
while not b.endswith(b"\n"): b+=s.recv(65536)
d=json.loads(b); print(d.get("ok"), d.get("online"), len(d.get("people") or []))' 2>/dev/null)
set -- $st
[ "${1:-}" = True ] && pass "control socket answers" || fail "control socket: ${st:-no answer}"
[ "${2:-}" = True ] && pass "online in the temple (reached the temple server through the sandbox)" || fail "NOT online in the temple"
python3 -I -c '
import json,socket
s=socket.create_connection(("localhost",7790),timeout=3); s.sendall(b"{\"cmd\":\"hearing\"}\n"); b=b""
while not b.endswith(b"\n"): b+=s.recv(65536)
print(json.dumps(json.loads(b))[:300])' 2>/dev/null | grep -q '/data/irisvr/hear' && pass "bot's ears spool to /data/irisvr/hear" || echo "INFO  couldn't confirm the spool dir via {cmd:hearing} (journal: $(sudo journalctl -u $U -o cat --no-pager | grep -m1 -o '\[hear\] ears ready (spool [^,]*'))"
[ "$(systemctl --user is-active iris-vr-hear)" = active ] && pass "my ears service (iris-vr-hear) active" || fail "iris-vr-hear $(systemctl --user is-active iris-vr-hear)"
[ -L /tmp/iris_vr_voice ] && [ "$(readlink /tmp/iris_vr_voice)" = /data/irisvr/voice ] && pass "/tmp/iris_vr_voice → /data/irisvr/voice (my speech reaches the bot)" || fail "/tmp/iris_vr_voice is not the link"
m=/data/irisvr/in/iris_mood.json
[ -f $m ] && cmp -s $m $REPO/state/iris_mood.json && pass "mood mirror current" || fail "mood mirror missing/stale (systemctl --user status iris-vr-mood-mirror.path)"
# spool READ path: a junk item written AS irisvr must be consumed (read + deleted) by my ears service — silent, wakes nobody
J=/data/irisvr/hear/0_verify_0.json
sudo -u irisvr sh -c "umask 022; printf '{\"verify\":true}' > $J.tmp && mv $J.tmp $J"
for i in $(seq 1 20); do [ -e $J ] || break; sleep 0.5; done
[ -e $J ] && { fail "my ears did not consume a spool item written by irisvr (perm?)"; sudo rm -f $J; } || pass "a spool item written by irisvr was read + removed by my ears"
if [ "$WT" = --wake-test ] || [ "${WAKE_TEST:-0}" = 1 ]; then
  H=$REPO/state/vr_temple/heard.jsonl; tag="sandbox-verify-$(date +%s)"
  sudo -u irisvr sh -c "umask 022; printf '%s\n' '{\"kind\":\"gesture\",\"speaker\":\"verify.sh\",\"text\":\"($tag — sandbox relay test, ignore)\"}' >> /data/irisvr/gestures/gestures.jsonl"
  for i in $(seq 1 20); do grep -q "$tag" $H 2>/dev/null && break; sleep 0.5; done
  grep -q "$tag" $H && pass "a gesture line written by irisvr reached heard.jsonl (my reader)" || fail "gesture relay did not reach heard.jsonl"
fi
echo; [ $rc = 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"; exit $rc
