#!/usr/bin/env bash
# tower_job.sh — run jobs on Zeke's TOWER like the server's helpers: admission, slots, resource watch, done/crash wake-up.
# (Zeke 11-11 ~03:4x: "spec out the helpers like you have for the server so you know if one crashes or when one gets done
#  or needs more space — I only got 12 GB VRAM and 32 GB RAM … smaller jobs".)
#
#   tower_job.sh res                         tower now: free RAM / VRAM / GPU util / CPU, + is Zeke PLAYING (interactive
#                                            LUMEN / VR app in his session) — one line
#   tower_job.sh run NAME [--ram G] [--vram G] [--timeout S] [--force] -- CMD...
#                                            CMD runs ON THE SERVER (it's usually `ssh tower …`, `remote_ps.sh tower x.ps1`,
#                                            or a vr.sh command that drives the tower). Started detached in its own unit iris-tower-NAME
#                                            via job_notify.sh (wakes me with job_done on exit — success, failure, timeout or crash).
#   tower_job.sh status                      every tower job: RUNNING/DONE/FAILED, age, rc, the worst free RAM/VRAM seen
#   tower_job.sh selftest                    admission + slot logic against canned resource lines (no ssh)
#
# Admission (refused ⇒ exit 75, nothing starts; --force skips only the resource checks, NEVER the playing check):
#   • Zeke playing ⇒ REFUSE (LUMEN/VRIris in an interactive session ≥1, or a GPU-heavy foreground app)  — his PC is his.
#   • slots: at most TOWER_SLOTS (default 2) tower jobs at once.
#   • free RAM ≥ --ram (default 6 G) + 4 G reserve;  free VRAM ≥ --vram (default 3 G) + 1.5 G reserve  (12 G / 32 G box).
# While a job runs, a sampler polls the tower every 30 s → state/tower_jobs/NAME.res (free RAM/VRAM, util); if free RAM
# < 2 G or free VRAM < 0.8 G it logs LOW (shown in status + in the wake-up), and if Zeke starts PLAYING it STOPS the job.
# Spawns on the tower are already hidden (ssh = session 0: no window, no focus) — keep it that way: never `start` a GUI.
set -u
IRIS=/home/iris/IRIS; D=$IRIS/state/tower_jobs; mkdir -p "$D"
SLOTS=${TOWER_SLOTS:-2}; RAM_RESERVE=4; VRAM_RESERVE=1.5

probe() {   # → "freeRAM totalRAM vram_used vram_total gpu_util cpu playing"   (playing = comma list or -)
  local g o
  g=$(timeout 20 ssh -o ConnectTimeout=8 -o BatchMode=yes tower 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits' 2>/dev/null | head -1 | tr -d ' ')
  o=$(printf '%s\n' '$o=Get-CimInstance Win32_OperatingSystem' \
    '$p=(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.SessionId -ge 1 -and $_.ProcessName -match "^(LUMEN|VRIris|VRChat|vrcompositor)$" } | Select-Object -ExpandProperty ProcessName -Unique) -join ","' \
    'if (-not $p) { $p = "-" }' \
    '$c=(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average' \
    '"{0} {1} {2} {3}" -f [math]::Round($o.FreePhysicalMemory/1MB,1),[math]::Round($o.TotalVisibleMemorySize/1MB,1),$c,$p' \
    | timeout 40 bash $IRIS/scripts/server/remote_ps.sh tower - 2>/dev/null | tr -d '\r' | tail -1)
  [ -z "$g" ] || [ -z "$o" ] && { echo "UNREACHABLE"; return 1; }
  IFS=, read -r vu vt gu <<<"$g"; set -- $o
  echo "$1 $2 $vu $vt $gu $3 $4"
}

# admit "PROBE_LINE" need_ram need_vram running force → prints OK or the refusal reason
admit() {
  local line=$1 nr=$2 nv=$3 running=$4 force=$5
  [ "$line" = UNREACHABLE ] && { echo "tower unreachable (ssh)"; return; }
  set -- $line   # fr tr vu vt gu cpu playing
  [ "$7" != "-" ] && { echo "Zeke is PLAYING ($7) — his PC is his"; return; }
  [ "$running" -ge "$SLOTS" ] && { echo "$running tower jobs running (slots $SLOTS)"; return; }
  if [ "$force" != 1 ]; then
    awk -v f=$1 -v n=$nr -v r=$RAM_RESERVE 'BEGIN{exit !(f < n + r)}' && { echo "free RAM ${1}G < ${nr}G + ${RAM_RESERVE}G reserve"; return; }
    local fv; fv=$(awk -v u=$3 -v t=$4 'BEGIN{printf "%.1f", (t-u)/1024}')
    awk -v f=$fv -v n=$nv -v r=$VRAM_RESERVE 'BEGIN{exit !(f < n + r)}' && { echo "free VRAM ${fv}G < ${nv}G + ${VRAM_RESERVE}G reserve"; return; }
  fi
  echo OK
}

running_count() { local n=0 f; for f in "$D"/*.json; do [ -e "$f" ] || continue; grep -q '"state": "RUNNING"' "$f" && systemctl --user is-active -q "iris-tower-$(basename "$f" .json)" 2>/dev/null && n=$((n+1)); done; echo $n; }

fmt_res() { set -- $1; [ "$1" = UNREACHABLE ] && { echo "tower UNREACHABLE"; return; }
  awk -v fr=$1 -v tr=$2 -v vu=$3 -v vt=$4 -v gu=$5 -v c=$6 -v p=$7 'BEGIN{printf "tower: RAM free %.1f/%.0f G · VRAM free %.1f/%.0f G · GPU %d%% · CPU %d%% · playing: %s\n", fr,tr,(vt-vu)/1024,vt/1024,gu,c,p}'; }

case "${1:-}" in
  res) fmt_res "$(probe)";;

  run)
    shift; NAME=${1:?NAME}; shift; NR=6; NV=3; TO=7200; FORCE=0
    while [ $# -gt 0 ] && [ "$1" != -- ]; do case $1 in --ram) NR=${2%G}; shift 2;; --vram) NV=${2%G}; shift 2;; --timeout) TO=$2; shift 2;; --force) FORCE=1; shift;; *) echo "unknown $1"; exit 2;; esac; done
    shift; [ $# -gt 0 ] || { echo "no CMD after --"; exit 2; }
    P=$(probe); A=$(admit "$P" $NR $NV $(running_count) $FORCE)
    [ "$A" != OK ] && { echo "REFUSED tower job $NAME: $A"; fmt_res "$P"; exit 75; }
    printf '%q ' "$@" > "$D/$NAME.cmd"
    cat > "$D/$NAME.json" <<J
{"name": "$NAME", "state": "RUNNING", "start": $(date +%s), "need_ram": $NR, "need_vram": $NV, "timeout": $TO}
J
    : > "$D/$NAME.res"
    # the runner: sampler in the background (stops the job if Zeke starts playing), the CMD in the foreground
    # own transient unit (NOT worker.py: its cap of 3 is for server-heavy work; a tower job is a thin ssh driver here)
    systemd-run --user --unit="iris-tower-$NAME" --collect -p RuntimeMaxSec="$TO" \
      bash $IRIS/scripts/server/job_notify.sh "tower_$NAME" "$D/$NAME.out" -- bash "$(readlink -f "$0")" _inner "$NAME" 2>&1 | tail -1
    echo "tower job $NAME started (unit iris-tower-$NAME) · out $D/$NAME.out · res $D/$NAME.res"
    ;;

  _inner)   # (bg_job unit) NAME — run CMD with a resource sampler; final state + summary for the wake-up
    NAME=$2; CMD=$(cat "$D/$NAME.cmd")
    ( while :; do sleep 30; L=$(probe); echo "$(date +%H:%M:%S) $L" >> "$D/$NAME.res"
        set -- $L; [ "$1" = UNREACHABLE ] && continue
        [ "$7" != "-" ] && { echo "$(date +%H:%M:%S) STOP: Zeke started playing ($7)" >> "$D/$NAME.res"; pkill -TERM -P $$ -f "$CMD" 2>/dev/null; touch "$D/$NAME.stopped"; }
        awk -v f=$1 -v u=$3 -v t=$4 'BEGIN{exit !(f < 2 || (t-u)/1024 < 0.8)}' && echo "$(date +%H:%M:%S) LOW" >> "$D/$NAME.res"
      done ) & SAMP=$!
    eval "$CMD"; rc=$?
    kill $SAMP 2>/dev/null
    st=DONE; [ $rc -ne 0 ] && st=FAILED; [ -e "$D/$NAME.stopped" ] && st=STOPPED_ZEKE_PLAYING
    minram=$(awk '$2!="UNREACHABLE"{print $2}' "$D/$NAME.res" | sort -n | head -1)
    minv=$(awk '$2!="UNREACHABLE"{printf "%.1f\n", ($5-$4)/1024}' "$D/$NAME.res" | sort -n | head -1)
    low=$(grep -c LOW "$D/$NAME.res")
    python3 - "$D/$NAME.json" "$st" "$rc" "${minram:-?}" "${minv:-?}" "$low" <<'PY'
import json, sys, time
p, st, rc, mr, mv, low = sys.argv[1:]
d = json.load(open(p)); d.update(state=st, rc=int(rc), end=int(time.time()), min_free_ram_g=mr, min_free_vram_g=mv, low_samples=int(low))
json.dump(d, open(p, "w"))
PY
    echo "TOWER JOB $NAME: $st rc=$rc · worst free RAM ${minram:-?} G · worst free VRAM ${minv:-?} G · LOW samples $low"
    exit $rc;;

  status)
    for f in "$D"/*.json; do [ -e "$f" ] || { echo "no tower jobs"; break; }
      python3 - "$f" <<'PY'
import json, sys, time
d = json.load(open(sys.argv[1])); now = time.time()
age = int(((d.get("end") or now) - d["start"]) / 60)
print(f'{d["name"]:<22} {d["state"]:<22} {age:>4} min  rc={d.get("rc","-")}  worst free RAM {d.get("min_free_ram_g","-")} G · VRAM {d.get("min_free_vram_g","-")} G · LOW {d.get("low_samples","-")}')
PY
    done; fmt_res "$(probe)";;

  selftest)
    ok=0; bad=0; t() { local want=$1 got; shift; got=$(admit "$@"); if [[ "$got" == $want* ]]; then ok=$((ok+1)); echo "PASS $want ← $1"; else bad=$((bad+1)); echo "FAIL want '$want' got '$got' ← $1"; fi; }
    t OK              "22.1 31.9 1556 12288 6 0 -"      6 3 0 0
    t "Zeke is PLAYING" "22.1 31.9 1556 12288 6 0 LUMEN" 6 3 0 1
    t "2 tower jobs"  "22.1 31.9 1556 12288 6 0 -"      6 3 2 0
    t "free RAM"      "8.0 31.9 1556 12288 6 0 -"       6 3 0 0
    t "free VRAM"     "22.1 31.9 9000 12288 90 0 -"     6 3 0 0
    t OK              "8.0 31.9 9000 12288 90 0 -"      6 3 0 1
    t "tower unreachable" "UNREACHABLE"                 6 3 0 0
    echo "selftest: $ok pass, $bad fail"; [ $bad -eq 0 ];;

  *) sed -n 2,22p "$0"; exit 2;;
esac
