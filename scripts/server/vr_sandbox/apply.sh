#!/usr/bin/env bash
# apply.sh — move my VR temple body into its sandbox (audit H2). Idempotent; run as iris (uses sudo).
#   ./apply.sh --dry-run     print every step, change nothing
#   ./apply.sh               do it (the body is OFF for ~10–30 s during the switch)
#   ./apply.sh --no-switch   install everything but leave the OLD user unit running (stage now, switch later)
# Needs FIRST: the bot rebuilt + deployed with game_code_file.patch (preflight proves it; refuses otherwise).
# Undo: ./rollback.sh       Check: ./verify.sh
# Never writes the temple address or the invite into a tracked file: they go to root-only files under /etc/irisvr
# and the generated drop-in, read from ~/.iris_private.env at apply time.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/lib.sh"
REPO=/home/iris/IRIS
BOT=/data/projects/vr_iris/bot/VRIrisServer.x86_64
U=iris-vr-presence.service
DRY=0; SWITCH=1
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --no-switch) SWITCH=0;; *) echo "unknown arg $a"; exit 2;; esac; done
[ "$(id -un)" = iris ] || { echo "run as iris"; exit 2; }
run() { if [ $DRY = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
step() { echo; echo "== $*"; }
die() { echo "ABORT: $*"; exit 1; }

# ---------------------------------------------------------------- preflight (read-only)
step "preflight"
TEMPLE_HOST=$(temple_host)
INVITE_SET=$(bash -c 'set -a; . ~/.iris_private.env; [ -n "${TEMPLE_IRIS_INVITE:-}" ] && echo yes')
[[ "$TEMPLE_HOST" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || die "TEMPLE_HOST in ~/.iris_private.env must be a literal IPv4 (the IP filter needs it)"
[ "$INVITE_SET" = yes ] || die "TEMPLE_IRIS_INVITE missing from ~/.iris_private.env"
for f in iris-vr-presence.service egress_rules.sh probe.sh iris_readers.patch; do [ -f "$HERE/$f" ] || die "missing $HERE/$f"; done
sudo -n true || die "needs passwordless sudo"
echo "  temple host + invite present (not shown)"

# the bot must understand --code-file. A throwaway sandboxed copy (no network but localhost, own ctl port) is
# pointed at a MISSING code file: a patched build logs "[join] --code-file: cannot read" and quits 3.
PU=iris-vr-codefile-preflight
sudo systemctl stop $PU 2>/dev/null || true; sudo systemctl reset-failed $PU 2>/dev/null || true
SINCE=$(date '+%Y-%m-%d %H:%M:%S')
sudo systemd-run --quiet --unit=$PU -p DynamicUser=yes -p IPAddressDeny=any -p IPAddressAllow=localhost \
  -p PrivateTmp=yes -p ProtectHome=tmpfs -p ProtectSystem=strict \
  "$BOT" --headless -- --bot --hear=off --ctl-port=17792 --join=localhost:9 --code-file=/nonexistent/invite
for i in $(seq 1 60); do
  sleep 1
  sudo journalctl -u $PU --since "$SINCE" -o cat --no-pager | grep -- "--code-file: cannot read" >/dev/null && { CF=ok; break; }
  [ "$(systemctl is-active $PU)" = active ] || [ $i -lt 5 ] || break
done
# 10-31: the bot quits (status 3) right after logging, and the loop could break on "inactive" before journald had
# the line -> a patched build was reported as unpatched. One last look after a short flush wait.
[ "${CF:-}" = ok ] || { sleep 2; sudo journalctl -u $PU --since "$SINCE" -o cat --no-pager | grep -- "--code-file: cannot read" >/dev/null && CF=ok; }
sudo systemctl stop $PU 2>/dev/null || true; sudo systemctl reset-failed $PU 2>/dev/null || true
if [ "${CF:-}" = ok ]; then echo "  bot build supports --code-file ✓"
elif [ $DRY = 1 ]; then echo "  ⚠ the deployed bot does NOT support --code-file yet (a real run would ABORT here) — continuing the dry-run"
else die "the deployed bot doesn't support --code-file yet — apply game_code_file.patch to the game repo, rebuild + deploy the bot, then rerun"; fi

# ---------------------------------------------------------------- 1. user
step "1. system user irisvr (no shell, home /var/lib/irisvr — nothing under /home)"
if id irisvr >/dev/null 2>&1; then echo "  exists"; else
  run sudo useradd --system --user-group --no-create-home --home-dir /var/lib/irisvr --shell /usr/sbin/nologin irisvr
fi

# ---------------------------------------------------------------- 2. dirs
step "2. /data/irisvr: mine, group irisvr — hear+gestures the bot writes (2770), in+voice it only reads (2750)"
run sudo install -d -o iris -g irisvr -m 0750 /data/irisvr
for d in hear gestures; do run sudo install -d -o iris -g irisvr -m 2770 /data/irisvr/$d; done
for d in in voice;      do run sudo install -d -o iris -g irisvr -m 2750 /data/irisvr/$d; done

# ---------------------------------------------------------------- 3. credential + address
step "3. invite → /etc/irisvr/invite (root 0600, handed to the unit as a credential); address → drop-in"
run sudo install -d -o root -g root -m 0700 /etc/irisvr
if [ $DRY = 0 ]; then
  bash -c 'set -a; . ~/.iris_private.env; printf "%s" "$TEMPLE_IRIS_INVITE"' | sudo tee /etc/irisvr/invite.new >/dev/null
  sudo chmod 0600 /etc/irisvr/invite.new; sudo mv -f /etc/irisvr/invite.new /etc/irisvr/invite
  printf 'TEMPLE_HOST=%s\n' "$TEMPLE_HOST" | sudo tee /etc/irisvr/egress.env >/dev/null; sudo chmod 0600 /etc/irisvr/egress.env
  sudo install -d -m 0755 /etc/systemd/system/$U.d
  printf '# generated by scratch/vr_sandbox/apply.sh — the temple address (never in a tracked file)\n[Service]\nEnvironment=TEMPLE_HOST=%s\nIPAddressAllow=%s\n' "$TEMPLE_HOST" "$TEMPLE_HOST" \
    | sudo tee /etc/systemd/system/$U.d/temple.conf >/dev/null
  sudo chmod 0640 /etc/systemd/system/$U.d/temple.conf
else echo "[dry-run] write /etc/irisvr/invite, /etc/irisvr/egress.env, /etc/systemd/system/$U.d/temple.conf"; fi

# ---------------------------------------------------------------- 4. egress firewall
step "4. per-uid egress firewall (udp → temple:7777 + control-socket replies; nothing else)"
run sudo install -o root -g root -m 0755 "$HERE/egress_rules.sh" /usr/local/sbin/iris-vr-egress.sh
FW_UNIT='[Unit]
Description=Egress firewall for my sandboxed VR body (user irisvr: udp to the temple :7777 + control-socket replies only)
After=network-pre.target
Before=iris-vr-presence.service

[Service]
Type=oneshot
RemainAfterExit=yes
EnvironmentFile=/etc/irisvr/egress.env
ExecStart=/usr/local/sbin/iris-vr-egress.sh install irisvr ${TEMPLE_HOST}
ExecStop=/usr/local/sbin/iris-vr-egress.sh remove irisvr ${TEMPLE_HOST}

[Install]
WantedBy=multi-user.target'
if [ $DRY = 0 ]; then echo "$FW_UNIT" | sudo tee /etc/systemd/system/iris-vr-egress-firewall.service >/dev/null
else echo "[dry-run] write /etc/systemd/system/iris-vr-egress-firewall.service"; fi

# ---------------------------------------------------------------- 5. the unit
step "5. install the sandboxed system unit"
run sudo install -o root -g root -m 0644 "$HERE/iris-vr-presence.service" /etc/systemd/system/$U
run sudo systemctl daemon-reload
run sudo systemctl enable iris-vr-egress-firewall.service
run sudo systemctl restart iris-vr-egress-firewall.service

# ---------------------------------------------------------------- 6. readers
step "6. my side: reader patch (vr_hear spool + gesture relay, vr_temple tool, selfcheck, hear unit), mood mirror, voice link"
if (cd $REPO && git apply --check -R "$HERE/iris_readers.patch" 2>/dev/null); then echo "  reader patch already applied"
else (cd $REPO && run git apply "$HERE/iris_readers.patch"); echo "  (applied to the working tree — NOT committed)"; fi
run install -m 0644 $REPO/scripts/server/systemd/iris-vr-hear.service ~/.config/systemd/user/iris-vr-hear.service
run install -m 0644 $REPO/scripts/server/systemd/iris-vr-mood-mirror.path ~/.config/systemd/user/
run install -m 0644 $REPO/scripts/server/systemd/iris-vr-mood-mirror.service ~/.config/systemd/user/
TF=~/.config/user-tmpfiles.d/iris-vr.conf
if grep -q '^L+ /tmp/iris_vr_voice' $TF; then echo "  voice link already in $TF"; else
  run cp -n $TF $TF.pre-h2
  run sed -i 's|^d /tmp/iris_vr_voice .*|# audit H2: my voice wavs live where the sandboxed bot can read them\nL+ /tmp/iris_vr_voice - - - - /data/irisvr/voice|' $TF
fi
[ -L /tmp/iris_vr_voice ] || run rm -rf /tmp/iris_vr_voice           # only cached TTS wavs (the tool prunes them anyway)
run systemd-tmpfiles --user --create
run systemctl --user daemon-reload
run systemctl --user enable --now iris-vr-mood-mirror.path
run systemctl --user start iris-vr-mood-mirror.service

# ---------------------------------------------------------------- 7. switch
if [ $SWITCH = 0 ]; then echo; echo "== staged (--no-switch): the OLD user unit still runs. Switch later with ./apply.sh"; exit 0; fi
step "7. switch: old user unit off, sandboxed system unit on"
run systemctl --user disable --now iris-vr-presence.service
run systemctl --user restart iris-vr-hear.service                    # my ears now read /data/irisvr/hear + relay gestures
run sudo systemctl enable --now $U
[ $DRY = 1 ] && { echo; echo "dry-run done — nothing changed"; exit 0; }
echo "  waiting for the body to join…"
ok=0
for i in $(seq 1 90); do
  sleep 1
  st=$(python3 -I -c '
import json,socket
s=socket.create_connection(("localhost",7790),timeout=2); s.sendall(b"{\"cmd\":\"state\"}\n"); b=b""
while not b.endswith(b"\n"): b+=s.recv(65536)
print(json.loads(b).get("online"))' 2>/dev/null || true)
  [ "$st" = True ] && { ok=1; echo "  joined the temple after ${i}s ✓"; break; }
done
if [ $ok = 0 ]; then
  echo "⚠ the sandboxed body did NOT join within 90 s — journal tail:"; sudo journalctl -u $U -n 30 --no-pager -o cat | mask_ips
  echo "⚠ rolling back to the old user unit"; "$HERE/rollback.sh"; exit 1
fi
step "8. verify"
"$HERE/verify.sh" || echo "⚠ verify reported failures — read them; ./rollback.sh restores the old unit"
