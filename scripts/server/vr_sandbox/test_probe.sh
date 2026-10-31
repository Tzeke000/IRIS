#!/usr/bin/env bash
# test_probe.sh — run probe.sh inside a throwaway transient unit with the unit's EXACT sandbox block.
#   ./test_probe.sh                  DynamicUser, cgroup filter only — shows what the egress firewall must close
#   ./test_probe.sh --with-firewall  the idle `nobody` uid + a TEMPORARY copy of the egress rules for that uid only
#   ./test_probe.sh --user irisvr    after apply: the real user + the real firewall (verify.sh uses this)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/lib.sh"
TEMPLE_HOST=$(temple_host)
GW=$(ip -4 route show default | awk '{print $3; exit}')
PUB=$(getent ahostsv4 one.one.one.one | awk '{print $1; exit}')
[ -n "$TEMPLE_HOST" ] && [ -n "$GW" ] || { echo "need TEMPLE_HOST + a default gateway"; exit 2; }
U=iris-vr-sandbox-probe
who=(-p DynamicUser=yes); dir=""
case "${1:-}" in
  --with-firewall)
    # systemd needs an NSS user; `nobody` owns no process on this box (checked) ⇒ the temporary rule bites nothing else
    TUID=$(id -u nobody); [ -z "$(ps -u nobody -o pid=)" ] || { echo "something runs as nobody — not testing"; exit 2; }
    sudo "$HERE/egress_rules.sh" install $TUID "$TEMPLE_HOST" IRIS_VR_TEST
    trap 'sudo "$HERE/egress_rules.sh" remove $TUID x IRIS_VR_TEST' EXIT
    who=(-p User=nobody -p Group=nogroup)
    dir=$(mktemp -d /data/sandbox/irisvr_probe.XXXX); mkdir -p "$dir"/{hear,gestures,in,voice}; chmod -R 777 "$dir"
    trap 'sudo "$HERE/egress_rules.sh" remove $TUID x IRIS_VR_TEST; rm -rf "$dir"' EXIT ;;
  --user) who=(-p "User=$2" -p "Group=$2") ;;
esac
if [ -z "$dir" ] && [ ! -d /data/irisvr/hear ]; then   # pre-apply: the binds need targets
  dir=$(mktemp -d /data/sandbox/irisvr_probe.XXXX); mkdir -p "$dir"/{hear,gestures,in,voice}; chmod -R 777 "$dir"
  trap 'rm -rf "$dir"' EXIT
fi
sandbox_props "$HERE/iris-vr-presence.service" "${dir:-/data/irisvr}"
sudo systemctl stop $U 2>/dev/null; sudo systemctl reset-failed $U 2>/dev/null
sudo systemd-run --quiet --wait --pipe --unit=$U "${who[@]}" "${PROPS[@]}" \
  -p "IPAddressAllow=$TEMPLE_HOST" -p "BindReadOnlyPaths=$HERE/probe.sh:/run/irisvr_probe.sh" \
  /bin/bash /run/irisvr_probe.sh "$TEMPLE_HOST" "$GW" "$PUB" 2>&1 | mask_ips
r=${PIPESTATUS[0]}; sudo systemctl reset-failed $U 2>/dev/null; exit $r
