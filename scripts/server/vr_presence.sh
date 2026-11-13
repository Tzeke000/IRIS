#!/usr/bin/env bash
# vr_presence.sh on|off|status ["reason"] — my body in the VR temple, ALL of it, in one switch (11-12).
# Why: 11-12 22:2x I started iris-vr-presence by hand for Zeke's session but not iris-vr-hear (the ears' speech-to-text),
# so I could talk to him but not hear him — the bot recorded 17 clips that nothing transcribed. Body + ears (+ mood mirror)
# go on and off TOGETHER, and the deliberate-off flag (Zeke 11-02: on only while someone is in VR) moves with them.
set -u
cd /home/iris/IRIS || exit 1
FLAG=state/vr_temple/presence_deliberately_off.json
case "${1:-status}" in
  on)
    rm -f "$FLAG"
    sudo -n systemctl start iris-vr-presence || echo "✗ could not start iris-vr-presence"
    systemctl --user start iris-vr-hear iris-vr-mood-mirror.path || echo "✗ could not start ears/mood mirror"
    ;;
  off)
    reason=${2:-"nobody in VR"}
    printf '{"off": true, "reason": "Zeke 11-02: only while someone is in VR; %s (%s)"}\n' "$reason" "$(date '+%m-%d %H:%M')" > "$FLAG"
    sudo -n systemctl stop iris-vr-presence
    systemctl --user stop iris-vr-hear
    ;;
  status) ;;
  *) echo "usage: vr_presence.sh on|off|status [\"reason\"]"; exit 2 ;;
esac
printf 'body=%s ears=%s mood_mirror=%s flag=%s\n' "$(systemctl is-active iris-vr-presence)" \
  "$(systemctl --user is-active iris-vr-hear)" "$(systemctl --user is-active iris-vr-mood-mirror.path)" \
  "$([ -e "$FLAG" ] && echo OFF || echo absent)"
