#!/bin/bash
# selfcheck.sh — every SHELL-side probe of the 6-hourly self-check in one compact report (Iris, 10-24).
# The MCP-side ones stay tool calls (iris_health, voice_delivery_check, attention_smooth status, identity_stability,
# malloc_trim info, wake_verdict recent, adaptive_learning learn). Prints ✓/⚠ lines + the raw baseline row to append.
R=/home/iris/IRIS
ok() { echo "✓ $*"; }; warn() { echo "⚠ $*"; }
# ONE-OF-ME
n_tmux=$(tmux ls 2>/dev/null | wc -l)
host=$(pgrep -f "iris_body_host.py" | head -1); n_host=$(pgrep -fc "iris_body_host.py")
claude=$(pgrep -P "$host" -x claude | head -1); n_rt=$(pgrep -fc "iris_runtime.py")
rt=$(pgrep -f "iris_runtime.py" | head -1)
[ "$n_tmux" = 1 ] && [ "$n_host" = 1 ] && [ -n "$claude" ] && [ "$n_rt" = 1 ] \
  && ok "one of me: tmux 1, host $host, claude $claude (etime $(ps -o etime= -p $claude | tr -d ' ')), runtime $rt" \
  || warn "ONE-OF-ME: tmux=$n_tmux host=$n_host claude=${claude:-none} runtime=$n_rt — check ppids before killing anything"
fa_age=$(( $(date +%s) - $(stat -c %Y ~/FAILOVER_ACTIVE 2>/dev/null || echo 0) ))
[ -e ~/LIVE ] && [ $fa_age -lt 900 ] && ok "LIVE + FAILOVER_ACTIVE (${fa_age}s old)" || warn "LIVE/FAILOVER_ACTIVE missing or stale (${fa_age}s)"
model=$(ps -o args= -p "$claude" 2>/dev/null | grep -oP -- '--model \K\S+')
[ "$model" = claude-opus-5-5 ] && ok "model $model" || warn "MODEL '$model' (expected claude-opus-5-5) — tell Zeke; restart_self is banned"
# UNITS
for u in iris-stack iris-mouth iris-ears iris-vr-presence; do s=$(systemctl --user is-active $u); printf "%s=%s " $u $s; done
# audit H2: the VR body is a sandboxed SYSTEM unit once scratch/vr_sandbox/apply.sh ran
[ -e /etc/systemd/system/iris-vr-presence.service ] && printf "iris-vr-presence(system)=%s" "$(systemctl is-active iris-vr-presence)"; echo
[ -e $R/state/voice_deliberately_off.json ] && echo "  voice flag: $(cat $R/state/voice_deliberately_off.json) (night/deliberate ⇒ don't heal)" || echo "  voice flag absent = ON"
[ -e $R/state/attention/servo_deliberately_off.json ] && echo "  servo flag SET: $(cat $R/state/attention/servo_deliberately_off.json)" || echo "  servo flag absent (servo should run while Zeke is home)"
python3 -c "import json;d=json.load(open('$R/scratch/voice_control.json'));print('  mic_muted',d.get('mic_muted'),'mouth_muted',d.get('mouth_muted'))" 2>/dev/null
# PORTS + DOORS
ports=$(ss -ltn | awk '{print $4}' | grep -E ":(5876|8769|8770|5877)$" | tr '\n' ' '); echo "  listening: $ports"
code=$(curl -sk -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:5876/api/v1/vision/latest_frame)
[ "$code" = 200 ] && ok "orb :5876 frame 200" || warn "orb :5876 → $code (204 at night = eyes resting; listener lost ⇒ restart_orb_http FIRST)"
po=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:5877/health); [ "$po" = 200 ] && ok "post-office 200" || warn "post-office $po (Wren's lifeline — never bounce unilaterally)"
sk=$(grep -h "^IRIS_AUDIO_SINK=" ~/.config/iris/voice.env 2>/dev/null | cut -d= -f2 | tr -d '"'); if [ -n "$sk" ] && timeout 3 bash -c "</dev/tcp/${sk%:*}/${sk#*:}" 2>/dev/null; then ok "tower audio sink reachable (my voice can play)"; else warn "tower audio sink :8775 UNREACHABLE — my voice plays NOWHERE (the orb app owns the sink: is it open on the tower?) (10-26)"; fi
ts=$(tailscale ip -4 2>&1 | head -1); [ -n "$ts" ] && [[ "$ts" != *NoState* ]] && ok "ssh $(systemctl is-active ssh), tailscale up" || warn "tailscale: $ts (his re-auth)"
kex=$(journalctl -u ssh --since -6h --no-pager 2>/dev/null | grep -c kex_exchange_identification); echo "  kex noise last 6h: $kex"
# PRESENCE
python3 - <<EOF
import json,datetime
d=json.load(open('$R/state/zeke_presence.json'))
age=(datetime.datetime.now()-datetime.datetime.fromisoformat(d['last_check'])).total_seconds()
print(('✓' if age<300 else '⚠'),'presence watcher', int(age),'s old; present' if d.get('present') else '; AWAY', 'since', d.get('since'))
EOF
# CHAT FLAG (a stale .pending kept chat_pending:true for 10 h on 10-24)
pf=$R/state/iris_chat/.pending; [ -e $pf ] && echo "  chat .pending exists ($(( ($(date +%s) - $(stat -c %Y $pf)) / 60 )) min old) — stale if every json is answered/expired"
# TEMPLE
TH=$(grep -E "^TEMPLE_HOST=" ~/.iris_private.env | cut -d= -f2)
t=$(timeout 15 ssh -i ~/.ssh/id_ed25519_temple -o BatchMode=yes root@$TH "systemctl is-active vr-temple nftables | tr '\n' ' '" 2>&1); echo "  temple: $t"
# RESOURCES (raw baseline row)
gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null)
mem=$(free -g | awk '/Mem:/{print $3"/"$2" G"}'); disk=$(df -h / | awk 'NR==2{print $3"/"$2" "$5}')
rss=$(ps -o rss= -p $rt | awk '{printf "%.2f GB", $1/1048576}'); hrss=$(ps -o rss= -p $host | awk '{printf "%d MB", $1/1024}')
echo "BASELINE $(date '+%F %H:%M') | GPU $gpu | RAM $mem | / $disk | runtime RSS $rss | host $hrss"
# HEALTH SNAPSHOT (V3 step 1.5): state/health/now.json, written every minute by the watchdog timer from OUTSIDE
# my process tree. Fresh (<180 s) => one line per subsystem; stale/missing => say so (the timer or the writer is down).
python3 - "$R/state/health/now.json" <<'EOF'
import json, sys, time
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("⚠ health now.json unreadable/missing (%s) — is iris-watchdog.timer running?" % type(e).__name__); sys.exit()
age = time.time() - float(d.get("ts") or 0)
if age > float(d.get("fresh_s") or 180):
    print("⚠ health now.json is STALE (%d s old) — the watchdog timer/writer stopped; don't trust it" % age); sys.exit()
s = d.get("summary") or {}
bad = ["%s=%s" % (k, ",".join(s[k])) for k in ("red", "warn", "off", "parked") if s.get(k)]
print(("✓" if not s.get("red") else "⚠"), "health now.json (%d s old, %s ms):" % (age, d.get("took_ms")),
      "all ok" if not bad else " ".join(bad))
for k, v in (d.get("subsystems") or {}).items():
    if v.get("status") != "ok":
        why = (v.get("detail") or {}).get("why") if v.get("status") in ("off", "parked", "n/a") else v.get("last_error")
        err = (why or "")[:110]
        lag = "-" if v.get("lag_s") is None else "%ss" % v.get("lag_s")
        print("   %-6s %-13s lag=%s %s" % (v.get("status"), k, lag, err))
EOF
# PAINTING
wk=$(date +%G-W%V); [ -d $R/scratch/practice/room_painting/$wk ] && ok "painting $wk exists" || { [ $(date +%u) -ge 4 ] && warn "painting $wk missing (Thu+ ⇒ paint it, after quiet hours / conversation)" || echo "  painting $wk not due yet"; }
