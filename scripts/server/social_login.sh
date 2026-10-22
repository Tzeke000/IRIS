#!/bin/sh
# Zeke logs into TikTok/YouTube HIMSELF through noVNC; I never handle the password.
# start: Xvfb :77 + headed browser + x11vnc (password, localhost only) + websockify/noVNC on :6080
# stop:  tears all of it down (the saved login stays in state/cloak_profiles/social)
D=:77; PW=$HOME/.iris_social_vncpw
case "$1" in
start)
  [ -f "$PW" ] || { head -c 9 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 8 > "$PW.txt"; x11vnc -storepasswd "$(cat $PW.txt)" "$PW" >/dev/null 2>&1; chmod 600 "$PW" "$PW.txt"; }
  systemd-run --user --unit social-xvfb Xvfb $D -screen 0 1280x860x24 -nolisten tcp
  sleep 2
  systemd-run --user --unit social-browser --setenv=DISPLAY=$D /home/iris/apps/cloak-mcp/.venv/bin/python /home/iris/IRIS/scripts/server/social_login_browser.py
  systemd-run --user --unit social-vnc x11vnc -display $D -rfbauth "$PW" -localhost -forever -shared -rfbport 5977 -quiet
  sleep 2
  systemd-run --user --unit social-novnc websockify --web /usr/share/novnc 6080 localhost:5977
  echo "VNC password: $(cat $PW.txt)";;
stop)
  for u in social-novnc social-vnc social-browser social-xvfb; do systemctl --user stop $u 2>/dev/null; done; echo stopped;;
status)
  for u in social-xvfb social-browser social-vnc social-novnc; do printf "%s " $u; systemctl --user is-active $u; done;;
esac
