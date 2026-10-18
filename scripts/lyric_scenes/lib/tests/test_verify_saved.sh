#!/usr/bin/env bash
# End-to-end test of lib/qa/verify_saved.py: build a GOOD and a BAD tiny scene (bpy_verify_fixture.py), save them, then
# verify each in a FRESH Blender process — every launch through scripts/server/render.sh. ~1 min.
#   scripts/lyric_scenes/lib/tests/test_verify_saved.sh [WORKDIR=/tmp/libverify_test]
set -uo pipefail
T="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
LIB="$(dirname "$T")"; REPO="$(cd "$LIB/../../.." && pwd)"
RS="$REPO/scripts/server/render.sh"; BL="$HOME/.local/bin/blender"
W=${1:-/tmp/libverify_test}
rm -rf "$W"; mkdir -p "$W"
FAILS=0
pass() { echo "[test_verify_saved] PASS $*"; }
fail() { echo "[test_verify_saved] FAIL $*"; FAILS=$((FAILS + 1)); }
chk() { local msg=$1; shift; if "$@"; then pass "$msg"; else fail "$msg"; fi; }
bl() { local unit=$1; shift; "$RS" "$unit" --wait --collect -P "$BL" -b --factory-startup "$@"; }
for m in good bad; do
  bl "t-libverify-fx-$m" --python-exit-code 1 --python "$T/bpy_verify_fixture.py" -- --out "$W/$m.blend" --mode $m \
    >"$W/fixture_$m.log" 2>&1
  chk "fixture $m built" test -f "$W/$m.blend"
done
cat > "$W/extra.py" <<'PY'
def verify(scene, R, args):
    R.check(scene.camera is not None and scene.camera.name == "Cam", "extra_scene_rule", "scene camera is Cam")
PY
ARGS=(--ground Ground --grounded Prop_ --direction Car_:+Y --view-min 0.5 --ignore HazeBox --render 1,30 --pct 100
      --extra "$W/extra.py")
bl t-libverify-good "$W/good.blend" --python-exit-code 1 --python "$LIB/qa/verify_saved.py" -- --out "$W/good.json" \
  "${ARGS[@]}" >"$W/good.log" 2>&1; rc_good=$?
bl t-libverify-bad "$W/bad.blend" --python-exit-code 1 --python "$LIB/qa/verify_saved.py" -- --out "$W/bad.json" \
  "${ARGS[@]}" >"$W/bad.log" 2>&1; rc_bad=$?
chk "good scene: exit 0 (rc=$rc_good)" test "$rc_good" = 0
chk "bad scene: exit non-zero (rc=$rc_bad)" test "$rc_bad" != 0
PY="$HOME/venvs/iris-v100/bin/python"
"$PY" -I - "$W" <<'PY'
import json, sys
W = sys.argv[1]
fails = 0
def chk(msg, cond, info=""):
    global fails
    print(f"[test_verify_saved] {'PASS' if cond else 'FAIL'} {msg} {info}")
    fails += not cond
g, b = json.load(open(f"{W}/good.json")), json.load(open(f"{W}/bad.json"))
gk = {a["key"]: a["ok"] for a in g["asserts"]}
bk = {a["key"]: a["ok"] for a in b["asserts"]}
want = ["images", "finite", "camera", "view_hit", "cuts", "instancers", "grounding", "direction", "render_nan"]
chk("good: every check ran and passed (incl. --extra rule)", all(gk.get(k) for k in want + ["extra_scene_rule"]), str(gk))
chk("JSON comes from the SAVED file in a fresh process", g["values"]["file"]["path"] == f"{W}/good.blend", g["values"]["file"]["path"])
chk("cuts sampled (24, 25 in the sample frames)", {24, 25} <= set(g["values"]["file"]["samples"]), str(g["values"]["file"]["samples"]))
chk("good: the Lamps instancer counted 5", all(c == 5 for c in g["values"]["details"]["instancers"][0]["counts"].values()))
for k in ["images", "finite", "camera", "cuts", "instancers", "grounding", "direction", "render_nan"]:
    chk(f"bad: '{k}' check fails", bk.get(k) is False,
        str(b["values"]["details"].get({"camera": "camera_inside", "render_nan": "render"}.get(k, k)))[:300])
bi = b["values"]["details"]
chk("bad: the camera-inside hit is shot 2 in the Wall", any(h["frame"] >= 25 and "Wall" in h["near"] for h in bi["camera_inside"]))
chk("bad: the relative texture is named", any("relative" in w for _, w in bi["images"]["bad_images"]))
chk("bad: the hovering crate gap ≈ 1.5 m", abs(bi["grounding"]["floating"][0][1] - 1.5) < 0.01, str(bi["grounding"]))
rows = bi["render"]["rows"]
chk("bad: NaN signature = raw clean, composited non-finite", all(r["nonfinite"] == 0 for r in rows if not r["compositing"])
    and all(r["nonfinite"] > 0 for r in rows if r["compositing"]), str(rows))
chk("good: absolute texture path survived the save (bpyx.save_blend)", not g["values"]["details"]["images"]["bad_images"])
chk("bad: report status recorded", b["status"].startswith("ok with") or b["status"].startswith("failed"), b["status"])
sys.exit(1 if fails else 0)
PY
[[ $? == 0 ]] || FAILS=$((FAILS + 1))
echo "[test_verify_saved] $([[ $FAILS == 0 ]] && echo ALL PASS || echo "$FAILS FAILED") ($W)"
exit $((FAILS > 0))
