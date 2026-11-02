"""2026-11-02 efficiency pass (profiles/server/waste_audit_2026-11-02.md):
event-driven frame waits (#3), MediaPipe step rate (#1), thread caps (#8),
InsightFace module trim + landmark carry (#5), mouth compile workers (#6),
battery poll back-off (#9).

    ~/venvs/iris-v100/bin/python -m pytest tests/test_efficiency_1102.py -q
"""
from __future__ import annotations

import ast
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import frame_store  # noqa: E402


# ── #3 event-driven frame waits ─────────────────────────────────────────────

def test_frame_signal_wakes_on_publish() -> None:
    sig = frame_store.FrameSignal()
    got: dict = {}

    def waiter() -> None:
        t0 = time.monotonic()
        got["ok"] = sig.wait_newer(0.0, timeout=2.0)
        got["dt"] = time.monotonic() - t0

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.05)
    sig.publish(123.0)
    th.join(3.0)
    assert got["ok"] is True
    assert got["dt"] < 1.0                       # woke on the publish, not the timeout


def test_frame_signal_times_out_and_never_hangs() -> None:
    sig = frame_store.FrameSignal()
    sig.publish(10.0)
    t0 = time.monotonic()
    assert sig.wait_newer(10.0, timeout=0.05) is False      # nothing newer
    assert time.monotonic() - t0 < 0.5
    assert sig.wait_newer(5.0, timeout=0.05) is True        # already newer: no wait


def test_legacy_poll_knob(monkeypatch) -> None:
    monkeypatch.setattr(frame_store, "FRAME_EVENT_WAIT", False)
    sig = frame_store.FrameSignal()
    t0 = time.monotonic()
    assert sig.wait_newer(0.0, timeout=5.0) is False         # 4 ms sleep, timeout ignored
    assert time.monotonic() - t0 < 0.5


def test_push_frame_wakes_buffer_waiters(monkeypatch) -> None:
    monkeypatch.setattr(frame_store, "BUFFER_SIGNAL", frame_store.FrameSignal())
    last = frame_store.BUFFER_SIGNAL.ts
    res: dict = {}
    th = threading.Thread(target=lambda: res.setdefault(
        "ok", frame_store.wait_for_buffered_frame(last, timeout=2.0)))
    th.start()
    time.sleep(0.05)
    frame_store.push_frame(np.zeros((4, 4, 3), np.uint8))
    th.join(3.0)
    assert res["ok"] is True
    # the buffer and its signal share ONE clock (person_track's stale-frame guard relies on it)
    assert frame_store.get_buffered_frame().capture_ts == frame_store.BUFFER_SIGNAL.ts


def test_person_track_loop_sleeps_between_frames(monkeypatch) -> None:
    """With no new frames the loop must idle on the signal (~10 checks/s), not
    spin at 250/s; each pushed frame is stepped exactly once."""
    from brain import person_track
    monkeypatch.setattr(frame_store, "BUFFER_SIGNAL", frame_store.FrameSignal())
    steps: list = []
    monkeypatch.setattr(person_track, "step", lambda frame, faces, ts: steps.append(ts))
    monkeypatch.setattr(person_track, "_LOOP_BUDGET_S", 0.0)
    frame_store.push_frame(np.zeros((4, 4, 3), np.uint8))
    stop = threading.Event()
    stats_before = person_track._loop_stats["ticks"]
    th = threading.Thread(target=person_track._loop, args=({}, stop), daemon=True)
    th.start()
    time.sleep(0.6)                                  # one frame, then silence
    ticks_idle = person_track._loop_stats["ticks"] - stats_before
    for _ in range(3):
        frame_store.push_frame(np.zeros((4, 4, 3), np.uint8))
        time.sleep(0.03)
    time.sleep(0.1)
    stop.set()
    th.join(2.0)
    assert not th.is_alive()                         # stop lands within the bounded wait
    assert len(steps) == 4                           # every frame once, none twice
    assert ticks_idle < 30, ticks_idle               # old 4 ms poll ≈ 150 in 0.6 s


# ── #1 MediaPipe rate (source-level: the runtime can't be imported in tests) ─

def test_runtime_wires_mediapipe_rate_and_raw_signal() -> None:
    src = (ROOT / "iris_runtime.py").read_text(encoding="utf-8")
    assert 'os.environ.get("IRIS_MEDIAPIPE_HZ", "10")' in src
    assert '("face", _face_step, None, step_period)' in src           # face stays 30 Hz
    for w in ("hands", "expr", "attn"):
        assert f'("{w}", _{w}_step, ' in src and "mediapipe_period)" in src
    assert "_raw_signal.publish(_pub_ts)" in src
    assert "_raw_signal.wait_newer(last_ts)" in src
    assert "time.sleep(0.004)            # wait for a NEW frame" not in src


# ── #8 thread caps ──────────────────────────────────────────────────────────

def test_apply_env_never_overrides_launcher(monkeypatch) -> None:
    from brain import thread_caps
    monkeypatch.setenv("IRIS_THREAD_CAPS", "1")
    monkeypatch.setenv("IRIS_THREAD_CAP", "3")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    out = thread_caps.apply_env()
    assert out["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "3"


def test_thread_caps_kill_switch(monkeypatch) -> None:
    from brain import thread_caps
    monkeypatch.setenv("IRIS_THREAD_CAPS", "0")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    assert thread_caps.apply_env() == {}
    assert "OPENBLAS_NUM_THREADS" not in os.environ
    assert thread_caps.apply_libs() == {"enabled": False}


def test_session_options_capped_but_caller_wins() -> None:
    ort = pytest.importorskip("onnxruntime")
    from brain import thread_caps
    so = ort.SessionOptions()
    thread_caps.configure_session_options(so, threads=2, spin=False)
    assert so.intra_op_num_threads == 2
    assert so.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    mine = ort.SessionOptions()
    mine.intra_op_num_threads = 7
    thread_caps.configure_session_options(mine, threads=2, spin=False)
    assert mine.intra_op_num_threads == 7


def test_ort_patch_is_idempotent_and_runs_models(monkeypatch) -> None:
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    # undo the process-wide patch after this test
    monkeypatch.setattr(ort.InferenceSession, "__init__", ort.InferenceSession.__init__)
    from onnx import TensorProto, helper
    from brain import thread_caps
    assert thread_caps.patch_onnxruntime(ort)
    init = ort.InferenceSession.__init__
    assert thread_caps.patch_onnxruntime(ort)
    assert ort.InferenceSession.__init__ is init                        # no double wrap
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    m = helper.make_model(helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [x], [y]),
                          opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    out = s.run(None, {"x": np.array([[-1, 0, 2]], np.float32)})[0]
    assert out.tolist() == [[0, 0, 2]]


# ── #5 InsightFace ──────────────────────────────────────────────────────────

def test_face_modules_default_drops_genderage(monkeypatch) -> None:
    from brain import insight_face_engine as ife
    monkeypatch.delenv("IRIS_FACE_MODULES", raising=False)
    mods = ife._face_modules()
    assert "genderage" not in mods
    for need in ("detection", "recognition", "landmark_3d_68"):   # pose feeds person:any
        assert need in mods
    monkeypatch.setenv("IRIS_FACE_MODULES", "all")
    assert ife._face_modules() is None
    monkeypatch.setenv("IRIS_FACE_MODULES", "recognition")
    assert ife._face_modules() == ["detection", "recognition"]


def test_shift_landmarks_follows_the_box() -> None:
    from brain import insight_face_engine as ife
    lm = np.array([[10.0, 10.0], [20.0, 20.0]])
    out = ife._shift_landmarks(lm, (0, 0, 20, 20), np.array([100, 50, 140, 90], float))
    # centre (10,10) -> (120,70), scale x2
    assert out.tolist() == [[120.0, 70.0], [140.0, 90.0]]
    assert lm.tolist() == [[10.0, 10.0], [20.0, 20.0]]              # input untouched


class _FakeModel:
    def __init__(self, name: str, calls: list):
        self.name, self.calls = name, calls

    def get(self, frame, face):
        self.calls.append(self.name)
        if self.name == "recognition":
            face["embedding"] = np.ones(4)
        elif self.name == "landmark_2d_106":
            x1, y1, x2, y2 = face["bbox"]
            face["landmark_2d_106"] = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
        elif self.name == "landmark_3d_68":
            face["pose"] = np.array([1.0, 2.0, 3.0])


def _fake_engine(monkeypatch, boxes: list):
    from brain import insight_face_engine as ife
    calls: list = []

    class Det:
        def __init__(self):
            self.i = 0

        def detect(self, frame, max_num=0, metric="default"):
            b = np.array([boxes[min(self.i, len(boxes) - 1)] + [0.9]], float)
            self.i += 1
            return b, np.zeros((1, 5, 2))

    class App:
        det_model = Det()
        models = {"detection": object(),
                  "recognition": _FakeModel("recognition", calls),
                  "landmark_3d_68": _FakeModel("landmark_3d_68", calls),
                  "landmark_2d_106": _FakeModel("landmark_2d_106", calls)}

    eng = ife.InsightFaceEngine.__new__(ife.InsightFaceEngine)
    eng._app = App()
    monkeypatch.setattr(ife, "_FULL_EVERY_N", 3)
    return ife, eng, calls


def test_fast_get_carries_landmarks_between_identity_passes(monkeypatch) -> None:
    pytest.importorskip("insightface")
    ife, eng, calls = _fake_engine(monkeypatch, [[0, 0, 100, 100], [10, 0, 110, 100], [20, 0, 120, 100]])
    monkeypatch.setattr(ife, "_LM106_EVERY_FRAME", False)
    f1 = eng._fast_get(None)            # n=1, no cache -> full models
    assert sorted(calls) == ["landmark_2d_106", "landmark_3d_68", "recognition"]
    calls.clear()
    f2 = eng._fast_get(None)            # n=2, IoU-matched -> NOTHING but detection
    assert calls == []
    # the carried landmark moved with the box (centre 50 -> 60)
    assert f2[0]["landmark_2d_106"].tolist() == [[60.0, 50.0]]
    assert f2[0]["pose"].tolist() == [1.0, 2.0, 3.0]
    assert f1[0]["landmark_2d_106"].tolist() == [[50.0, 50.0]]
    f3 = eng._fast_get(None)            # n=3 -> full pass again
    assert "recognition" in calls and f3[0]["landmark_2d_106"].tolist() == [[70.0, 50.0]]


def test_fast_get_lm106_every_frame_rollback(monkeypatch) -> None:
    pytest.importorskip("insightface")
    ife, eng, calls = _fake_engine(monkeypatch, [[0, 0, 100, 100], [10, 0, 110, 100]])
    monkeypatch.setattr(ife, "_LM106_EVERY_FRAME", True)
    eng._fast_get(None)
    calls.clear()
    eng._fast_get(None)
    assert calls == ["landmark_2d_106"]


# ── #6 mouth compile workers ────────────────────────────────────────────────

def test_mouth_caps_inductor_compile_workers() -> None:
    src = (ROOT / "scripts" / "server" / "mouth_launch.sh").read_text(encoding="utf-8")
    assert 'export TORCHINDUCTOR_COMPILE_THREADS="${IRIS_FISH_COMPILE_THREADS:-1}"' in src
    # set before the fish python starts
    assert src.index("TORCHINDUCTOR_COMPILE_THREADS") < src.index("wren_fish_server.py\"")


# ── #9 battery poll back-off (function lifted out by AST: importing the daemon
#     configures live paths, which tests must never do) ──────────────────────

def _battery_delay_fn(**consts):
    src = (ROOT / "scripts" / "vector_inhabit_daemon.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_battery_next_delay")
    ns = {"BATTERY_POLL_S": 15.0, "BATTERY_BACKOFF": True,
          "BATTERY_MAX_BACKOFF_S": 120.0, "BATTERY_OFF_POLL_S": 300.0, **consts}
    exec(compile(ast.Module([fn], []), "daemon", "exec"), ns)
    return ns["_battery_next_delay"]


def test_battery_backoff_schedule() -> None:
    f = _battery_delay_fn()
    assert f(True, 0, False) == 15.0
    assert [f(False, n, False) for n in (1, 2, 3, 4, 9)] == [30.0, 60.0, 120.0, 120.0, 120.0]
    assert f(True, 5, False) == 15.0                 # a good read snaps back
    assert f(False, 1, True) == 300.0                # deliberately off
    assert f(True, 0, True) == 300.0


def test_battery_backoff_rollback() -> None:
    f = _battery_delay_fn(BATTERY_BACKOFF=False)
    assert f(False, 7, True) == 15.0
