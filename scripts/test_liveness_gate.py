"""Adversarial checks for brain/liveness_gate.py (2026-10-20). Run with the venv python."""
import json, sys, tempfile, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from brain import liveness_gate as lg

d = Path(tempfile.mkdtemp()); now = time.time()
flag, snap = d / "f.json", d / "n.json"
W = lambda p, o: p.write_text(json.dumps(o))
tr = lambda pid, would, bbox=(0, 0, 100, 100), src="live", age=0.5: {
    "source": src, "live_pid": pid, "would": would, "bbox": list(bbox), "last_ts": now - age}
J = lambda: lg.judge_unknown_wake(now_path=snap, flag_path=flag, now=now)["suppress"]
fails = []
def chk(name, got, exp):
    print(("PASS " if got == exp else "FAIL ") + name)
    if got != exp: fails.append(name)

W(flag, {"on": True, "gate": False}); W(snap, {"ts": now, "tracks": [tr("unknown", "face_like")]})
chk("gate off never suppresses", J(), False)
W(flag, {"on": True, "gate": True})
chk("lone photo suppressed", J(), True)
W(snap, {"ts": now, "tracks": [tr("zeke", "person:zeke"), tr("unknown", "face_like", (300, 0, 400, 100))]})
chk("zeke + photo suppressed", J(), True)
W(snap, {"ts": now, "tracks": [tr("unknown", "face_like"), tr("unknown", "visitor", (300, 0, 400, 100))]})
chk("photo + real visitor WAKES", J(), False)
W(snap, {"ts": now, "tracks": [tr("unknown", "uncertain")]})
chk("uncertain WAKES", J(), False)
W(snap, {"ts": now - 10, "tracks": [tr("unknown", "face_like")]})
chk("stale snapshot WAKES", J(), False)
W(snap, {"ts": now, "tracks": [tr("unknown", "decoy:helmet")]})
chk("decoy suppressed", J(), True)
W(snap, {"ts": now, "tracks": [tr("unknown", "face_like", src="probe")]})
chk("probe-only track ignored", J(), False)
snap.write_text("garbage")
chk("unreadable snapshot WAKES", J(), False)
W(snap, {"ts": now, "tracks": [tr("unknown", "face_like")]})
chk("is_fake_face box match", lg.is_fake_face([5, 5, 105, 105], now_path=snap, flag_path=flag, now=now), True)
chk("is_fake_face other box", lg.is_fake_face([300, 0, 400, 100], now_path=snap, flag_path=flag, now=now), False)
W(flag, {"on": True, "gate": False})
chk("is_fake_face gate off", lg.is_fake_face([5, 5, 105, 105], now_path=snap, flag_path=flag, now=now), False)
print(f"RESULT: {12 - len(fails)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
