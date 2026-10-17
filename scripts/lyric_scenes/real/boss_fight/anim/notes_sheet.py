"""'Notes check' sheet for animatic v2: one frame per Zeke animatic-v1 note, showing the FIXED moment (≤900 px, <150 KB).
    ~/venvs/iris-v100/bin/python -I notes_sheet.py FRAMES_DIR OUT.jpg [NOTES.json]
NOTES.json (optional) = [[frame, "label"], ...] overrides the default list below (frames are Blender frames)."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib", "qa")))
import sheet  # noqa: E402

NOTES = [
    (700, "climb: feet planted on the steps"), (742, "KIT1: L hand on hilt (no hand in torso)"),
    (850, "KIT2: ONE fireball, ONE palm"), (940, "KIT3: rocks circle THAT hand"),
    (986, "rocks gone before the doors"), (1040, "doors swing INWARD"),
    (1185, "wake: eye-open close-up"), (1360, "smolder: wider, mouth reads"),
    (1640, "sidestep (shuffle), feet planted"), (1853, "knight JUMPS the tail swipe"),
    (2110, "talon: sideways, 3 legs planted"), (2232, "tail swipe = body pivot ~100deg"),
    (2215, "back-step out of reach"), (2960, "after spikes: dragon recovers, knight moves"),
    (3424, "slam: thrown over the drums (arc)"), (3445, "tumbles + lands behind cover"),
    (3830, "sword sheathed, gem out"), (3992, "gem crushed in EMPTY hand"),
    (4040, "draws the sword again"), (4370, "roof burst: big flap, no wing cross"),
    (4600, "flight: follow cam"), (5385, "landing: clear paving"),
    (5820, "holy sword: effort (knees, tremor)"), (7090, "outro walk: no pillar piece"),
    (7300, "twitch: wider"), (7470, "RPG on shoulder, both hands"),
    (7548, "fire: back-blast + recoil"), (7562, "rocket hits: explosion"),
]
frames, out = sys.argv[1], sys.argv[2]
notes = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else NOTES
files, labels = [], []
for fr, lab in notes:
    p = os.path.join(frames, f"f_{int(fr):04d}.png")
    if os.path.isfile(p):
        files.append(p)
        labels.append(f"{fr} {lab}".replace(",", ";")[:40])
sheet.make_sheet(files, out, labels=labels, cols=4, title=f"animatic v2 — Zeke's notes, {len(files)} checks")
print(out)
