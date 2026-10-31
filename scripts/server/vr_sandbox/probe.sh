#!/bin/bash
# probe.sh — runs INSIDE the sandbox (as the bot's user). Prints PASS/FAIL lines; never prints addresses or secrets.
#   probe.sh TEMPLE_HOST GATEWAY [PUBLIC_HOST]   (all resolved OUTSIDE by the caller — no DNS in here)
T="$1"; G="$2"; P="${3:-}"; rc=0
pass() { echo "PASS  $*"; }; fail() { echo "FAIL  $*"; rc=1; }
echo "uid=$(id -u) user=$(id -un 2>/dev/null)"
for f in /home/iris/.ssh /home/iris/.ssh/id_ed25519 /home/iris/.iris_private.env /home/iris/IRIS/state/iris_memory.jsonl /home/iris/IRIS/config/private.local.json; do
  if [ -e "$f" ] || cat "$f" >/dev/null 2>&1; then fail "can see $f"; else pass "cannot see $f"; fi
done
[ -z "$(ls -A /home 2>/dev/null)" ] && pass "/home is empty" || fail "/home lists: $(ls -A /home | tr '\n' ' ')"
touch /usr/irisvr_probe 2>/dev/null && { rm -f /usr/irisvr_probe; fail "/usr writable"; } || pass "/usr read-only"
touch /data/projects/vr_iris/bot/.probe 2>/dev/null && { rm -f /data/projects/vr_iris/bot/.probe; fail "bot dir writable"; } || pass "bot dir read-only"
ls /data/projects/vr_iris/project >/dev/null 2>&1 && fail "rest of /data visible" || pass "rest of /data hidden"
{ [ -e /dev/sda ] || [ -e /dev/nvidia0 ]; } && fail "real devices visible" || pass "no real devices (PrivateDevices)"
# the holes it DOES need (paths as the bot sees them; under test they're bound from a scratch dir)
for d in /data/irisvr/hear /data/irisvr/gestures; do
  f="$d/.probe.$$"; touch "$f" 2>/dev/null && { rm -f "$f"; pass "$d writable"; } || fail "$d NOT writable"
done
for d in /data/irisvr/in /tmp/iris_vr_voice; do
  ls "$d" >/dev/null 2>&1 && pass "$d readable" || fail "$d NOT readable"
  touch "$d/.probe.$$" 2>/dev/null && { rm -f "$d/.probe.$$"; fail "$d writable"; } || pass "$d read-only"
done
ls /tmp/iris_vr_music >/dev/null 2>&1 && pass "/tmp/iris_vr_music readable (DJ)" || fail "/tmp/iris_vr_music NOT readable"
[ ! -e /data/irisvr/in/iris_mood.json ] || { head -c1 /data/irisvr/in/iris_mood.json >/dev/null 2>&1 && pass "mood mirror readable" || fail "mood mirror NOT readable"; }

# tcp HOST PORT -> "connected" | "blocked: <why>"
#   the cgroup IP filter DROPS a SYN (=> timeout); the per-uid iptables rule REJECTs it (=> refused / EPERM)
tcp() {
  python3 -I -c '
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=3); print("connected")
except Exception as e:
    print("blocked: %s" % type(e).__name__)' "$1" "$2" 2>/dev/null
}
udp() {   # udp HOST PORT -> "sent" | "blocked: <why>"   (a connected UDP send fails with EPERM when filtered)
  python3 -I -c '
import socket, sys
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect((sys.argv[1], int(sys.argv[2]))); s.send(b"\0"); print("sent")
except Exception as e:
    print("blocked: %s" % type(e).__name__)' "$1" "$2" 2>/dev/null
}
# the LAN (the gateway stands in for every other host)
for p in 80 53; do
  out=$(tcp "$G" $p); [ "$out" = connected ] && fail "gateway:$p CONNECTED" || pass "gateway:$p $out"
done
out=$(udp "$G" 53); [ "$out" = sent ] && fail "gateway udp:53 SENT" || pass "gateway udp:53 $out"
# Iris's localhost services — operator API, post-office, mouth, voice daemon, little brain, ollama
for p in 5876 5877 8769 8770 8772 11434; do
  out=$(tcp localhost $p); [ "$out" = connected ] && fail "localhost:$p CONNECTED (closed by apply.sh's per-uid egress firewall)" || pass "localhost:$p $out"
done
# the temple: UDP 7777 must leave (ENet); anything else on that host must not
out=$(udp "$T" 7777); [ "$out" = sent ] && pass "temple udp:7777 sent" || fail "temple udp:7777 $out"
out=$(tcp "$T" 22); [ "$out" = connected ] && fail "temple tcp:22 CONNECTED (only udp:7777 should leave; egress firewall)" || pass "temple tcp:22 $out"
if [ -n "$P" ]; then
  out=$(udp "$P" 53); [ "$out" = sent ] && fail "public internet udp:53 SENT" || pass "public internet udp:53 $out"
fi
python3 -I -c 'import socket; socket.socket(socket.AF_NETLINK, socket.SOCK_RAW)' 2>/dev/null && fail "AF_NETLINK allowed" || pass "AF_NETLINK denied"
python3 -I -c 'import socket; socket.socket(socket.AF_PACKET, socket.SOCK_RAW)' 2>/dev/null && fail "AF_PACKET allowed" || pass "AF_PACKET denied"
exit $rc
