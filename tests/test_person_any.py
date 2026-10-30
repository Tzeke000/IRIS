"""brain/person_any — the person:any target picker + the frontal-unknown stranger gate.

Synthetic detections only: no camera, no models, no state files, no network, no env.
Fake/decoy predicates are injected so the live liveness/static-shape files are never read.

    /home/iris/venvs/iris-v100/bin/python -m pytest tests/test_person_any.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import person_any as pa  # noqa: E402

NO = staticmethod(lambda bb: False)


def face(pid="unknown", cx=320, cy=200, size=100, *, yaw=1.5, pitch=-2.0, det=0.9,
         conf=0.10, pose=True, kps=None):
    h = size / 2.0
    f = {"person_id": pid, "bbox": [cx - h, cy - h, cx + h, cy + h],
         "det_score": det, "confidence": conf,
         "pose": [pitch, yaw, 0.0] if pose else [0.0, 0.0, 0.0]}
    if kps is not None:
        f["kps"] = kps
    return f


def track(tid, x, y, w, h, vel=(0.0, 0.0)):
    return {"id": tid, "tlwh": [x, y, w, h], "score": 0.9, "vel": list(vel), "vh": 0.0}


def picker(**kw):
    return pa.AnyPicker(fake_fn=kw.get("fake", lambda bb: False),
                        decoy_fn=kw.get("decoy", lambda bb: False))


def gate(**kw):
    return pa.StrangerGate(fake_fn=kw.get("fake", lambda bb: False),
                           decoy_fn=kw.get("decoy", lambda bb: False))


# ── target picker ───────────────────────────────────────────────────────────────

def test_fresh_pick_prefers_owner_over_bigger_stranger():
    p = picker()
    c = p.pick([face("unknown", 100, 200, 160), face("zeke", 500, 200, 80)], [], now=0.0)
    assert c is not None and c["pid"] == "zeke"


def test_owner_takes_head_only_after_debounce():
    p = picker()
    stranger = face("unknown", 100, 200, 120)
    assert p.pick([stranger], [], now=0.0)["pid"] == "unknown"
    both = [stranger, face("zeke", 500, 200, 80)]
    assert p.pick(both, [], now=0.1)["pid"] == "unknown"          # one glimpse: no flip
    assert p.pick(both, [], now=0.1 + pa.OWNER_SWITCH_S * 0.5)["pid"] == "unknown"
    assert p.pick(both, [], now=0.1 + pa.OWNER_SWITCH_S + 0.01)["pid"] == "zeke"
    # once on the owner, a much bigger stranger never steals the head
    big = [face("unknown", 100, 200, 300), face("zeke", 500, 200, 80)]
    for i in range(20):
        assert p.pick(big, [], now=1.0 + i * 0.2)["pid"] == "zeke"


def test_largest_face_wins_and_hysteresis_resists_small_changes():
    p = picker()
    a, b = face("unknown", 100, 200, 100), face("unknown", 500, 200, 120)
    assert p.pick([a, b], [], now=0.0)["bbox"] == b["bbox"]
    a2 = face("unknown", 100, 200, 150)          # bigger, but < CHALLENGE_RATIO x 120
    for i in range(20):
        assert p.pick([a2, b], [], now=0.1 + i * 0.1)["bbox"] == b["bbox"]


def test_challenger_must_hold_to_switch():
    p = picker()
    a, b = face("unknown", 100, 200, 100), face("unknown", 500, 200, 120)
    p.pick([a, b], [], now=0.0)
    big_a = face("unknown", 100, 200, 220)       # >= 1.6 x 120
    t = 1.0
    assert p.pick([big_a, b], [], now=t)["bbox"] == b["bbox"]
    assert p.pick([big_a, b], [], now=t + pa.CHALLENGE_S * 0.5)["bbox"] == b["bbox"]
    # the challenger shrinks back for a tick -> its timer resets
    p.pick([a, b], [], now=t + pa.CHALLENGE_S * 0.8)
    assert p.pick([big_a, b], [], now=t + pa.CHALLENGE_S * 1.1)["bbox"] == b["bbox"]
    t2 = t + pa.CHALLENGE_S * 1.1
    assert p.pick([big_a, b], [], now=t2 + pa.CHALLENGE_S + 0.01)["bbox"] == big_a["bbox"]
    assert p.switches == 2


def test_no_flip_flop_between_two_equal_people():
    p = picker()
    a, b = face("unknown", 100, 200, 110), face("unknown", 500, 200, 110)
    first = p.pick([a, b], [], now=0.0)["bbox"]
    for i in range(60):   # sizes jitter +-10% every frame
        s = 100 if i % 2 else 120
        cur = p.pick([face("unknown", 100, 200, s), face("unknown", 500, 200, 220 - s)],
                     [], now=0.033 * (i + 1))
        assert _same_x(cur["bbox"], first)
    assert p.switches == 1


def _same_x(bb1, bb2):
    return abs((bb1[0] + bb1[2]) / 2 - (bb2[0] + bb2[2]) / 2) < 5


def test_lost_current_holds_then_repicks():
    p = picker()
    a, b = face("unknown", 100, 200, 100), face("unknown", 500, 200, 150)
    assert p.pick([a, b], [], now=0.0)["bbox"] == b["bbox"]
    # b vanishes: HOLD (None), do not jump to a
    assert p.pick([a], [], now=0.2) is None
    assert p.pick([a], [], now=pa.LOST_GRACE_S * 0.9) is None
    assert p.pick([a], [], now=pa.LOST_GRACE_S + 0.05)["bbox"] == a["bbox"]


def test_nobody_returns_none():
    assert picker().pick([], [], now=0.0) is None


def test_body_track_alone_is_never_followed():
    """The statue/helmet scar: a body box with no face is not somebody."""
    p = picker()
    for i in range(30):
        assert p.pick([], [track(7, 200, 100, 150, 350)], now=i * 0.1) is None


def test_helmet_decoy_and_fake_faces_rejected():
    helmet = face("unknown", 400, 450, 90, det=0.8, conf=0.11)
    is_helmet = lambda bb: abs((bb[0] + bb[2]) / 2 - 400) < 5   # noqa: E731
    assert picker(fake=is_helmet).pick([helmet], [], now=0.0) is None
    assert picker(decoy=is_helmet).pick([helmet], [], now=0.0) is None
    # the decoy layers never demote a RECOGNISED face at the same spot
    zeke = face("zeke", 400, 450, 90)
    assert picker(fake=is_helmet, decoy=is_helmet).pick([zeke], [], now=0.0)["pid"] == "zeke"


def test_weak_and_tiny_detections_rejected():
    p = picker()
    assert p.pick([face("unknown", size=100, det=0.3)], [], now=0.0) is None
    assert p.pick([face("unknown", size=20)], [], now=0.0) is None


def test_face_corroborated_body_keeps_following_a_turned_away_person():
    p = picker()
    trk = track(3, 250, 100, 140, 380)          # face centre (320,200) inside
    c = p.pick([face("unknown", 320, 200, 90)], [trk], now=0.0)
    assert c["kind"] == "face" and c["track_id"] == 3
    # they turn away: face gone, body still tracked -> follow the body, aim at the head
    c = p.pick([], [trk], now=0.5)
    assert c is not None and c["kind"] == "body" and c["track_id"] == 3
    assert abs(c["aim"][1] - 200) < 1.0          # remembered face height on the body
    # ...but only while the face corroboration is recent
    assert p.pick([], [trk], now=pa.CORROB_S + 0.6) is None


def test_face_returning_on_same_track_is_not_a_switch():
    p = picker()
    trk = track(3, 250, 100, 140, 380)
    p.pick([face("unknown", 320, 200, 90)], [trk], now=0.0)
    p.pick([], [trk], now=0.5)
    c = p.pick([face("unknown", 330, 205, 90)], [trk], now=1.0)
    assert c["kind"] == "face" and p.switches == 1


def test_recognition_catching_up_on_followed_person_keeps_them():
    p = picker()
    trk = track(3, 250, 100, 140, 380)
    p.pick([face("unknown", 320, 200, 90)], [trk], now=0.0)
    c = p.pick([face("zeke", 320, 200, 90)], [trk], now=0.1)
    assert c["track_id"] == 3


def test_aim_offset_math():
    c = {"aim": (480.0, 120.0), "kind": "face", "track": track(1, 0, 0, 10, 10, vel=(4.0, 0.0)),
         "track_id": 1, "pid": "unknown"}
    off = pa.aim_offset(c, (480, 640), fps=30.0)
    assert abs(off["dx"] - 0.5) < 1e-9 and abs(off["dy"] - (-0.5)) < 1e-9
    assert abs(off["vx"] - 4.0 * 30.0 / 320.0) < 1e-9 and off["source"] == "any_face"


def test_canonical_aliases():
    from brain import visual_attention as va
    for s in ("anyone", "Anyone", "person:any", "any", "anybody", "person:anyone"):
        assert va.canonical_target(s)[0] == "person:any", s
    assert va.canonical_target("zeke") == ("person:zeke", "zeke")
    assert va.canonical_target("person:any")[0].startswith("person:")   # person rails apply


# ── frontal-unknown stranger gate ──────────────────────────────────────────────

def _run(g, faces_at, times, sharp=lambda bb: 100.0):
    out = []
    for t in times:
        ev = g.observe(faces_at(t), now=t, sharp_fn=sharp)
        if ev:
            out.append((t, ev))
    return out


def test_frontal_unknown_triggers_after_5_obs_over_2s():
    g = gate()
    hits = _run(g, lambda t: [face("unknown", size=110, yaw=5, pitch=-4)],
                [i * 0.5 for i in range(6)])
    assert len(hits) == 1
    t, ev = hits[0]
    assert t == 2.0 and ev["observations"] == 5 and ev["span_s"] >= 2.0
    assert ev["pose_source"] == "pose3d" and ev["best_sim"] < pa.STRANGER_MAX_SIM


def test_five_obs_in_under_2s_is_not_enough():
    g = gate()
    assert not _run(g, lambda t: [face("unknown", size=110)], [i * 0.3 for i in range(7)])
    assert _run(g, lambda t: [face("unknown", size=110)], [2.1])


def test_angled_unknown_is_not_a_stranger():
    g = gate()
    assert not _run(g, lambda t: [face("unknown", size=110, yaw=35)],
                    [i * 0.5 for i in range(20)])
    assert not _run(g, lambda t: [face("unknown", size=110, pitch=-30)],
                    [i * 0.5 for i in range(20)])


def test_unfair_attempts_never_trigger():
    times = [i * 0.5 for i in range(20)]
    assert not _run(gate(), lambda t: [face("unknown", size=50)], times)           # small
    assert not _run(gate(), lambda t: [face("unknown", size=110, det=0.5)], times)  # weak
    assert not _run(gate(), lambda t: [face("unknown", size=110, conf=0.35)], times)  # ambiguous
    assert not _run(gate(), lambda t: [face("unknown", size=110)], times,
                    sharp=lambda bb: 5.0)                                          # blurry
    assert not _run(gate(fake=lambda bb: True), lambda t: [face("unknown", size=110)], times)
    assert not _run(gate(decoy=lambda bb: True), lambda t: [face("unknown", size=110)], times)
    assert not _run(gate(), lambda t: [face("zeke", size=110)], times)             # known


def test_recognition_in_between_resets_the_streak():
    g = gate()
    seq = {0.0: "unknown", 0.5: "unknown", 1.0: "unknown", 1.5: "zeke",
           2.0: "unknown", 2.5: "unknown", 3.0: "unknown"}
    hits = _run(g, lambda t: [face(seq[t], size=110)], sorted(seq))
    assert not hits


def test_missing_pose_falls_back_to_landmark_symmetry():
    # eyes at y=180, mouth at y=240, nose centred between -> frontal
    sym = [[290, 180], [350, 180], [320, 210], [295, 240], [345, 240]]
    hits = _run(gate(), lambda t: [face("unknown", size=110, pose=False, kps=sym)],
                [i * 0.5 for i in range(5)])
    assert len(hits) == 1 and hits[0][1]["pose_source"] == "kps"
    # nose pushed far to one side -> turned head -> no trigger
    turned = [[290, 180], [350, 180], [345, 210], [295, 240], [345, 240]]
    assert not _run(gate(), lambda t: [face("unknown", size=110, pose=False, kps=turned)],
                    [i * 0.5 for i in range(10)])
    # neither pose nor kps -> no claim of frontal
    assert not _run(gate(), lambda t: [face("unknown", size=110, pose=False)],
                    [i * 0.5 for i in range(10)])


def test_brief_turn_tolerated_long_gap_resets():
    g = gate()
    yaw = {0.0: 2, 0.5: 2, 1.0: 40, 1.5: 2, 2.0: 2, 2.5: 2}   # one glance away
    hits = _run(g, lambda t: [face("unknown", size=110, yaw=yaw[t])], sorted(yaw))
    assert len(hits) == 1 and hits[0][0] == 2.5
    g = gate()
    pts = [0.0, 0.5, 1.0, 1.5, 3.5, 4.0, 4.5]               # 2 s gap > reset
    assert not _run(g, lambda t: [face("unknown", size=110)], pts)


def test_study_cooldown_after_trigger():
    g = gate()
    hits = _run(g, lambda t: [face("unknown", size=110)], [i * 0.5 for i in range(40)])
    assert len(hits) == 1
    later = pa.STRANGER_STUDY_COOLDOWN_S + 10
    assert _run(g, lambda t: [face("unknown", size=110)],
                [later + i * 0.5 for i in range(6)])


def test_alert_limiter_one_per_stranger_per_window():
    lim = pa.AlertLimiter(600)
    assert lim.allow("seen_a", 0.0)
    assert not lim.allow("seen_a", 300.0)
    assert lim.allow("seen_b", 300.0)
    assert lim.allow("seen_a", 601.0)


def test_dm_text_says_silent():
    ev = {"span_s": 2.5, "yaw": 3.0, "pitch": -2.0, "best_sim": 0.12}
    study = {"frames_saved": 4, "dir": "/x/study_1", "crop": {"path": "/x/study_1/crop.jpg"},
             "seen": {"ok": True, "seen_before": False, "seen_id": "seen_1"}}
    txt = pa.stranger_dm_text(ev, study, {"reason": "phone on wifi"})
    assert "SILENT" in txt and "seen_1" in txt and "crop.jpg" in txt


# ── watcher wiring (stubbed study + DM; nothing real is called) ──────────────────

def test_watcher_stranger_tick_studies_dms_once_and_never_speaks(monkeypatch):
    from brain import unknown_capture as uc
    monkeypatch.setattr(uc, "STRANGER_DM", True)

    fired = []

    class Bus:
        def fire(self, kind, data=None, priority=None):
            fired.append((kind, data))

    g = {"_face_results": [face("unknown", size=110)], "_signal_bus": Bus()}
    w = uc.UnknownCaptureWatcher(g)
    w.__dict__["_stranger_gate"] = gate()
    calls, dms = [], []

    def study(p):
        calls.append(p)
        if p["action"] != "start":
            return {"ok": True}
        return {"ok": True, "study_id": "study_x", "dir": "/tmp/study_x", "frames_saved": 4,
                "crop": {"path": "/tmp/study_x/crop.jpg"}, "sheet": {"path": "/tmp/s.jpg"},
                "seen": {"ok": True, "seen_before": False, "seen_id": "seen_q"},
                "known_present": [], "zeke_presence": {"reason": "test"}}

    def dm(text):
        dms.append(text)
        return {"ok": True}

    handled = [w._stranger_tick(i * 0.5, study_fn=study, dm_fn=dm, sharp_fn=lambda bb: 99.0)
               for i in range(6)]
    assert handled.count(True) == 1
    assert [c["action"] for c in calls] == ["start"]          # servo not running: no track
    assert calls[0]["track"] is False
    assert len(dms) == 1 and "SILENT" in dms[0]
    assert fired and fired[0][0] == "unknown_capture"
    assert fired[0][1]["rule"] == "stranger-frontal" and fired[0][1]["dm"]["sent"]

    # the same stranger again after the study cooldown: studied, but NOT re-DM'd/re-woken
    t0 = pa.STRANGER_STUDY_COOLDOWN_S + 5
    for i in range(6):
        w._stranger_tick(t0 + i * 0.5, study_fn=study, dm_fn=dm, sharp_fn=lambda bb: 99.0)
    assert len([c for c in calls if c["action"] == "start"]) == 2
    assert len(dms) == 1 and len(fired) == 1
    assert "voice" not in " ".join(str(k) for k in g)        # nothing touched a mouth
