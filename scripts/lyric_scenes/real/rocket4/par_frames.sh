#!/usr/bin/env bash
# Parallel, resumable frame render for rocket v4: W workers, each in its OWN capped systemd unit (the server's
# render rule: scripts/server/render.sh = iris-render.slice, MemoryMax, OOMScoreAdjust), chunks interleaved
# (worker w takes chunks w, w+W, w+2W, ...), each worker under its own GPU lock (as v3's 4-worker run did).
#
#   par_frames.sh START END [WORKERS=3] [PCT=100] [TAG=r4]       → launches the units and returns
#   par_frames.sh wait [TAG=r4]                                  → blocks until every worker unit has exited
#
# env: the scene env must be exported first (source scratch/scenes/blastoff4_env.sh); OUT_DIR/BLEND/... are
# forwarded to the workers. CHUNK (default 30). Frames already on disk are skipped (resume = just re-run).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
if [[ "${1:-}" == wait ]]; then
  tag=${2:-r4}
  while systemctl --user list-units --state=active --no-legend "${tag}-w*" 2>/dev/null | grep -q .; do sleep 15; done
  exit 0
fi
S=$1; E=$2; W=${3:-3}; P=${4:-100}; TAG=${5:-r4}; C=${CHUNK:-30}
: "${OUT_DIR:?source scratch/scenes/blastoff4_env.sh first}"
for ((w = 0; w < W; w++)); do
  script=$(mktemp /tmp/${TAG}_w${w}_XXXX.sh)
  {
    echo "#!/usr/bin/env bash"
    echo "export OUT_DIR='$OUT_DIR' BLEND='$BLEND' AUDIO='$AUDIO' TIMELINE='$TIMELINE' MP4='$MP4'"
    echo "export LOCK=/tmp/iris_gpu_render.lock.w$w NO_TMP_CLEAN=1 CHUNK=$C PCT=$P"
    echo "for ((c = $S + $w * $C; c <= $E; c += $W * $C)); do"
    echo "  ce=\$((c + $C - 1)); ((ce > $E)) && ce=$E"
    echo "  '$HERE/render.sh' frames \$c \$ce $P"
    echo "done"
  } > "$script"
  chmod +x "$script"
  "$REPO/scripts/server/render.sh" "${TAG}-w$w" "$script" >/dev/null
  echo "[par_frames] worker $w → unit ${TAG}-w$w ($script); log: journalctl --user -u ${TAG}-w$w"
done
