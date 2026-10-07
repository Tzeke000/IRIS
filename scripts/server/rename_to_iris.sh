#!/usr/bin/env bash
# rename_to_iris.sh - one-shot (Zeke 2026-10-07): move the server harness from
# ~/staged/Wren-Companion to ~/IRIS. Run DETACHED (it stops the stack it is launched from):
#   systemd-run --user --unit=iris-rename bash ~/staged/Wren-Companion/scripts/server/rename_to_iris.sh
# Leaves ~/staged/Wren-Companion as a SYMLINK because the tower app's start button hardcodes
# that path until the app is rebuilt (main.rs now says ~/IRIS).
set -u
OLD="$HOME/staged/Wren-Companion"; NEW="$HOME/IRIS"; LOG="$HOME/iris_rename.log"
exec >>"$LOG" 2>&1; echo "== rename start $(date -Is)"
[ -L "$OLD" ] && { echo "already a symlink, nothing to do"; exit 0; }
[ -e "$NEW" ] && { echo "ABORT: $NEW exists"; exit 3; }
systemctl --user stop iris-stack iris-ears iris-mouth iris-quiet-hours.timer
for i in $(seq 1 40); do pgrep -f "$OLD/(iris_runtime|iris_body_host)" >/dev/null || break; sleep 1; done
pkill -f "$OLD/scripts/zeke_presence.py" ; sleep 1
pgrep -af "$OLD" && echo "WARN: processes still on old path (listed above)"
mv "$OLD" "$NEW" && ln -s "$NEW" "$OLD" || { echo "ABORT: mv failed"; systemctl --user start iris-mouth iris-ears iris-stack; exit 4; }
# the old path survives as a symlink, so these edits are belt-and-braces
sed -i "s#staged/Wren-Companion#IRIS#g" "$HOME"/.config/systemd/user/iris-*.service "$HOME/iris_start.sh" \
    "$HOME/.claude/settings.json" "$NEW/.mcp.json" "$NEW/.claude/settings.local.json"
P="$HOME/.claude/projects"
if [ ! -e "$P/-home-iris-IRIS" ]; then cp -a "$P/-home-iris-staged-Wren-Companion" "$P/-home-iris-IRIS"; fi
grep -rlI "/home/iris/staged/Wren-Companion" "$P/-home-iris-IRIS/memory" | xargs -r sed -i "s#/home/iris/staged/Wren-Companion#/home/iris/IRIS#g"
systemctl --user daemon-reload
systemctl --user start iris-mouth iris-ears iris-quiet-hours.timer iris-stack
echo "== rename done $(date -Is)"; systemctl --user is-active iris-mouth iris-ears iris-stack
