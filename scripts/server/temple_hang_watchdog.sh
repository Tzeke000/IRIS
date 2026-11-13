#!/bin/bash
# temple_hang_watchdog.sh — runs ON the temple CT (as root, systemd unit vr-temple-watchdog). DK-120 (11-12 23:22:42):
# the LUMEN server froze — process alive, unit "active", main thread + a WorkerThread spinning ~94 % CPU, UDP :7777 no
# longer read (Recv-Q 213 KB) ⇒ every client dropped; no debugger on the box ⇒ no stacks. This watches for exactly that:
# the server's UDP receive queue stays above RECVQ_MAX bytes for HANG_S seconds ⇒ dump every thread's stack with gdb
# (+ top -H, socket, /proc status) to /root/hang_<ts>/, then restart vr-temple. A busy-but-healthy server drains its
# queue every tick, so a backlog that never drains is a hang, not load.
set -u
RECVQ_MAX=${RECVQ_MAX:-4096}      # bytes (a live server drains every tick; 11-12 hang sat at 213 KB; a stopped test process showed 11.5 KB)
HANG_S=${HANG_S:-15}              # seconds the backlog must persist
PORT=${PORT:-7777}
since=0
while :; do
  q=$(ss -uanH "sport = :$PORT" 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
  q=${q:-0}
  if [ "$q" -gt "$RECVQ_MAX" ]; then
    [ "$since" -eq 0 ] && since=$(date +%s)
    if [ $(( $(date +%s) - since )) -ge "$HANG_S" ]; then
      P=$(pgrep -f LUMENServer | head -1); D=/root/hang_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$D"
      echo "recvq=$q for >=${HANG_S}s pid=$P" > "$D/why.txt"
      ss -uanp > "$D/sock.txt"; top -H -b -n 1 -p "$P" > "$D/top.txt" 2>&1; cp /proc/"$P"/status "$D/" 2>/dev/null
      timeout 60 gdb -p "$P" -batch -ex "set pagination off" -ex "thread apply all bt 30" > "$D/bt.txt" 2>&1
      journalctl -u vr-temple -n 300 --no-pager > "$D/journal.txt" 2>&1
      logger -t temple-watchdog "HANG: recvq=$q — stacks in $D — restarting vr-temple"
      systemctl restart vr-temple; since=0; sleep 30
    fi
  else
    since=0
  fi
  sleep 2
done
