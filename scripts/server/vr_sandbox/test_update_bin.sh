#!/usr/bin/env bash
# test_update_bin.sh (opsguard 11-08) — update_bin.sh on a temp copy: never /etc, never sudo, never systemctl.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; F=0
ok() { echo "PASS $1"; }; no() { echo "FAIL $1"; F=$((F+1)); }
cp "$HERE"/update_bin.sh "$HERE"/iris-vr-presence.service "$HERE"/apply.sh "$HERE"/test_sandbox.sh $T/
printf '#!/bin/sh\n' > $T/NewServer.x86_64; chmod +x $T/NewServer.x86_64
cur=$(sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' $T/iris-vr-presence.service)
printf '#!/bin/sh\necho "$@" >> %s/sysctl.log\n[ "$1" = is-active ] && exit 3; exit 0\n' $T > $T/sysctl; chmod +x $T/sysctl
sed "s|$cur|/old/VRIrisServer.x86_64|" $T/iris-vr-presence.service > $T/installed.service
E="UNIT_DST=$T/installed.service SUDO= SYSTEMCTL=$T/sysctl"
o=$(env $E bash $T/update_bin.sh --bin $T/NewServer.x86_64 --dry-run 2>&1)
grep -q "/old/VRIrisServer" $T/installed.service && grep -q "dry-run" <<<"$o" && ok "U1 --dry-run changes nothing" || no "U1 dry-run: $o"
o=$(env $E bash $T/update_bin.sh --bin $T/NewServer.x86_64 2>&1)
grep -q "^ExecStart=$T/NewServer.x86_64 --headless" $T/installed.service && ok "U2 installed ExecStart re-pointed, args kept" || no "U2 $o"
grep -q "ExecStart=$T/NewServer.x86_64" $T/iris-vr-presence.service && grep -q "BOT=$T/NewServer.x86_64" $T/apply.sh && ok "U3 tracked unit + apply.sh BOT= rewritten" || no "U3"
ls $T/installed.service.pre-* >/dev/null 2>&1 && grep -q daemon-reload $T/sysctl.log && ok "U4 backup made + daemon-reload" || no "U4"
o=$(env $E bash $T/update_bin.sh 2>&1); grep -q "nothing to do" <<<"$o" && ok "U5 idempotent: second run = nothing to do" || no "U5 $o"
o=$(env $E bash $T/update_bin.sh --bin $T/missing 2>&1); [ $? != 0 ] && grep -q "not an executable" <<<"$o" && ok "U6 refuses a missing binary" || no "U6"
o=$(UNIT_DST=$T/nope SUDO= SYSTEMCTL=$T/sysctl bash $T/update_bin.sh 2>&1); grep -q "apply.sh's job" <<<"$o" && ok "U7 not installed ⇒ refuses (apply.sh)" || no "U7"
grep -q "TEMPLE_HOST" $T/installed.service && ok "U8 the rest of the unit is untouched" || no "U8"
echo "update_bin: $([ $F = 0 ] && echo ALL PASS || echo "$F FAIL")"; [ $F = 0 ]
