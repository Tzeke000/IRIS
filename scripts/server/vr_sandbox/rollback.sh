#!/usr/bin/env bash
# rollback.sh — back to the pre-H2 body: the old USER unit iris-vr-presence (runs as me, invite on argv). Idempotent.
#   ./rollback.sh [--dry-run] [--purge]     --purge also deletes user irisvr, /data/irisvr and /var/lib/irisvr
# Leaves the reader patch in place on purpose: it works both ways (vr_temple tool + selfcheck fall back to the user
# unit when /etc/systemd/system/iris-vr-presence.service is gone; vr_hear just gets its old spool dir back below).
set -uo pipefail
DRY=0; PURGE=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --purge) PURGE=1;; *) echo "unknown arg $a"; exit 2;; esac; done
run() { if [ $DRY = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
U=iris-vr-presence.service
echo "== stop + remove the sandboxed system unit and its firewall"
run sudo systemctl disable --now $U 2>/dev/null
run sudo systemctl disable --now iris-vr-egress-firewall.service 2>/dev/null   # ExecStop removes the iptables rules
run sudo rm -rf /etc/systemd/system/$U /etc/systemd/system/$U.d /etc/systemd/system/iris-vr-egress-firewall.service
run sudo rm -rf /etc/irisvr                                                     # the invite's root-only copy
run sudo rm -f /usr/local/sbin/iris-vr-egress.sh
run sudo systemctl daemon-reload
run sudo systemctl reset-failed $U iris-vr-egress-firewall.service 2>/dev/null
if [ $DRY = 0 ] && id irisvr >/dev/null 2>&1; then     # belt + braces: no stray rule left behind
  sudo iptables -S OUTPUT | grep -q IRIS_VR_OUT && { echo "  removing leftover iptables rules"; sudo /bin/sh -c '
    while iptables -D OUTPUT -m owner --uid-owner irisvr -j IRIS_VR_OUT 2>/dev/null; do :; done; iptables -F IRIS_VR_OUT; iptables -X IRIS_VR_OUT
    while ip6tables -D OUTPUT -m owner --uid-owner irisvr -j IRIS_VR_OUT6 2>/dev/null; do :; done; ip6tables -F IRIS_VR_OUT6; ip6tables -X IRIS_VR_OUT6' 2>/dev/null; }
fi

echo "== my side: old voice dir, old hear spool, mood mirror off"
TF=~/.config/user-tmpfiles.d/iris-vr.conf
[ -f $TF.pre-h2 ] && run mv -f $TF.pre-h2 $TF
[ -L /tmp/iris_vr_voice ] && run rm -f /tmp/iris_vr_voice
run systemd-tmpfiles --user --create
run systemctl --user disable --now iris-vr-mood-mirror.path 2>/dev/null
HU=~/.config/systemd/user/iris-vr-hear.service
if grep -q 'IRIS_VR_HEAR_DIR=/data/irisvr/hear' $HU 2>/dev/null; then
  run sed -i 's|^Environment=IRIS_VR_HEAR_DIR=/data/irisvr/hear|Environment=IRIS_VR_HEAR_DIR=/tmp/iris_vr_hear|' $HU
fi
run systemctl --user daemon-reload
run systemctl --user restart iris-vr-hear.service

echo "== the old user unit back on"
run systemctl --user enable --now $U
if [ $PURGE = 1 ]; then
  echo "== purge"
  run sudo userdel irisvr; run sudo rm -rf /data/irisvr /var/lib/irisvr
fi
[ $DRY = 1 ] && { echo "dry-run done — nothing changed"; exit 0; }
sleep 3
echo "user unit: $(systemctl --user is-active $U)   system unit: $(systemctl is-active $U 2>/dev/null)   hear: $(systemctl --user is-active iris-vr-hear)"
echo "NOTE: the repo's scripts/server/systemd/iris-vr-hear.service (reader patch) still says /data/irisvr/hear —"
echo "      if you roll back for good, revert that line or reverse the patch: git apply -R scratch/vr_sandbox/iris_readers.patch"
