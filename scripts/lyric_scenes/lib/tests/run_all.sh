#!/usr/bin/env bash
# Run every lib/ test suite. Pure-python suites run directly; Blender suites go through scripts/server/render.sh
# (the capped slice) like every Blender launch. ~3–5 min. Exit 1 if any suite fails.
#   scripts/lyric_scenes/lib/tests/run_all.sh [--quick]     (--quick: pure-python only, no Blender, no GPU)
set -uo pipefail
T="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
LIB="$(dirname "$T")"; REPO="$(cd "$LIB/../../.." && pwd)"
PY="$HOME/venvs/iris-v100/bin/python"; BL="$HOME/.local/bin/blender"
fails=0
run() { local name=$1; shift; if "$@" >"/tmp/libtest_$name.log" 2>&1; then echo "PASS $name"; else echo "FAIL $name (log /tmp/libtest_$name.log)"; fails=$((fails + 1)); fi; }
cd "$LIB"
for t in test_qa test_song test_importer_pure test_bpyx_pure test_checks_report; do
  run "$t" "$PY" -I -m unittest discover -s "$T" -p "$t.py"
done
for t in test_render_pure test_assets; do run "$t" python3 -I -m unittest discover -s "$T" -p "$t.py"; done
if [[ "${1:-}" != --quick ]]; then
  for t in bpy_test_qa bpy_test_importer bpy_test_bpyx bpy_test_checks; do
    run "$t" "$REPO/scripts/server/render.sh" "t-lib-${t#bpy_test_}" --wait --collect -P "$BL" -b --factory-startup \
      --python-exit-code 1 --python "$T/$t.py"
  done
  run test_scene_sh "$T/test_scene_sh.sh"
fi
echo "[run_all] $([[ $fails == 0 ]] && echo 'ALL SUITES PASS' || echo "$fails suite(s) FAILED")"
exit $((fails > 0))
