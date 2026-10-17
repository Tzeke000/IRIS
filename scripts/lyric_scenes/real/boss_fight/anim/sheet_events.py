"""Contact sheet of the animatic: one frame per choreo event, labelled with its time + text (≤900 px, <150 KB).
PIL venv, isolated:  ~/venvs/iris-v100/bin/python -I sheet_events.py FRAMES_DIR CHOREO.json OUT.jpg [--from F --to F]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib", "qa")))
import sheet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("frames")
ap.add_argument("choreo")
ap.add_argument("out")
ap.add_argument("--from", dest="f0", type=int, default=1)
ap.add_argument("--to", dest="f1", type=int, default=10 ** 9)
ap.add_argument("--cols", type=int, default=8)
a = ap.parse_args()
ev = json.load(open(a.choreo))["events"]
files, labels = [], []
for e in ev:
    bf = int(e["frame"]) + 1
    if not a.f0 <= bf <= a.f1:
        continue
    p = os.path.join(a.frames, f"f_{bf:04d}.png")
    if os.path.isfile(p):
        files.append(p)
        m, s = divmod(e["t_snap"], 60)
        labels.append(f"{int(m)}:{s:04.1f} {e['what'][:26]}".replace(",", ";"))
if not files:
    sys.exit("no event frames rendered in range")
sheet.make_sheet(files, a.out, labels=labels, cols=a.cols, title=f"boss fight animatic {os.environ.get('ANIMATIC_VER', 'v2')} — {len(files)} events")
print(a.out)
