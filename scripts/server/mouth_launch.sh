#!/usr/bin/env bash
# mouth_launch.sh — iris-mouth.service runs THIS; it starts whichever voice engine is selected (2026-10-09).
# Zeke 10-08: "fish as the main, then two on standby just in case, but they should never be on at the same time."
# ONE unit (iris-mouth) on ONE port (8769) ⇒ two mouths can never be on at once, and quiet hours, the watchdog,
# the ears' dependency and the status board keep working unchanged. Choose with scripts/server/mouth_switch.sh.
#   engine file: ~/.config/iris/mouth_engine   fish | styletts   (turbo = back pocket, no server built yet)
# FAILSAFE: if Fish dies twice within 10 minutes of starting, fall back to StyleTTS2 for this run (logged) so a
# broken Fish can never leave me mute.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE_FILE="$HOME/.config/iris/mouth_engine"
FAILS="$ROOT/state/mouth_fish_fails"
LOG="$ROOT/state/mouth_engine.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
ENGINE="$(tr -d '[:space:]' < "$ENGINE_FILE" 2>/dev/null || true)"
[ -n "$ENGINE" ] || ENGINE=styletts
export WREN_VOICE_PORT=8769

if [ "$ENGINE" = "fish" ]; then
  now=$(date +%s)
  recent=$(awk -v n="$now" '$1 > n-600' "$FAILS" 2>/dev/null | wc -l)
  if [ "$recent" -ge 2 ]; then
    log "fish failed $recent times in 10 min — FALLING BACK to styletts for this run"
    ENGINE=styletts
  else
    log "starting fish on :8769"
    export FISH_MAXSEQ=4096 FISH_COMPILE=reduce-overhead FISH_INT8=1
    # Zeke 10-09: 3-word pieces sounded choppy (each piece is voiced on its own) ⇒ start with 8 words, then grow.
    export FISH_CHUNK_SCHEDULE=6,14,24 FISH_FIRST_FLUSH_S=1.6   # Zeke 10-09: 6 words to start; flush deadline outlasts a 6-word first piece (~1.2 s compute)
    export FISH_SEED=-1   # research 10-09: a fixed seed made every repeat identical ("hmm" always the same)
    # Zeke 10-09: first blind test picked refA_32s; then after "still monotone" he made a lively ElevenLabs take and picked #3 = x3
    # (45 s montage: excited + mischievous + sad + comforting, from iris-expresive.mp3) — pitch movement ~2x refA
    # Zeke 10-31: x3 read "sultry/alluring" (worst with the tenderness tag); new ElevenLabs v3 voice B (light British, clear, "not
    # breathy" in the design prompt) — Zeke: "sounds like you just with the British accent… much clearer". Revert = the x3 line.
    # export IRIS_FISH_REF_WAV="$HOME/voice_lab/ref_expressive/x3.wav" IRIS_FISH_REF_TXT="$HOME/voice_lab/ref_expressive/x3.txt"
    export IRIS_FISH_REF_WAV="$HOME/voice_lab/ref_candidates/elB_20s.wav" IRIS_FISH_REF_TXT="$HOME/voice_lab/ref_candidates/elB_20s.txt"
    export IRIS_FISH_ROOT="$HOME/voice_lab/fish" TMPDIR="$HOME/voice_lab/tmp"
    # 11-02 efficiency pass (audit #6): inductor forked 32 compile workers at warm-up and kept them for the
    # life of the mouth (~9.9 GB private RAM, 0% CPU, the reason iris-oom-demote exists). 1 = compile in-process,
    # no pool. Warm starts read the inductor cache and barely compile, so they cost ~nothing; a COLD cache
    # (torch/Fish upgrade, cache wiped) compiles serially and the ~10 min first warm-up gets longer. Override
    # per start with IRIS_FISH_COMPILE_THREADS=32 (the old behaviour) in the unit env.
    export TORCHINDUCTOR_COMPILE_THREADS="${IRIS_FISH_COMPILE_THREADS:-1}"
    t0=$(date +%s)
    "$HOME/voice_lab/fish/venv/bin/python" "$ROOT/voice/wren_fish_server.py"
    rc=$?
    log "fish exited rc=$rc after $(( $(date +%s) - t0 )) s"
    case "$rc" in 0|130|137|143) ;; *) echo "$(date +%s)" >> "$FAILS" ;; esac   # a stop (SIGTERM/INT/KILL) is not a failure
    exit "$rc"
  fi
fi
if [ "$ENGINE" = "styletts" ]; then
  log "starting styletts on :8769"
  exec "$HOME/venvs/iris-style/bin/python" "$ROOT/voice/wren_styletts_server.py"
fi
log "unknown engine '$ENGINE' — starting styletts"
exec "$HOME/venvs/iris-style/bin/python" "$ROOT/voice/wren_styletts_server.py"
