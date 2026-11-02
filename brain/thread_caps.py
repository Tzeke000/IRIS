"""Thread-pool caps for the runtime (2026-11-02 efficiency pass, audit #8).

The server has 36 vCPUs and nothing in the runtime capped a thread pool, so
every library sized itself to all 36: 705 runtime threads, ~36 per ONNX Runtime
session (5 InsightFace + YOLOX + MiniLM), torch/OpenMP/BLAS pools on top. Most
of those sessions run on the CUDA EP, so their CPU intra-op pools only ever
spin-wait — and ORT spins by default.

Two entry points, both idempotent and fail-open (a cap that can't be applied is
logged and skipped, never raised):

  apply_env()   — BEFORE numpy/torch/onnxruntime are imported. Sets the
                  OMP/MKL/OpenBLAS/numexpr/rayon env vars with setdefault, so a
                  value the launcher exported always wins.
  apply_libs()  — AFTER the heavy imports. torch.set_num_threads, cv2.setNumThreads,
                  and a wrapper on onnxruntime.InferenceSession.__init__ that gives
                  every session (including the ones InsightFace and chromadb build
                  internally — neither exposes SessionOptions) an intra-op cap and
                  turns intra-op spinning off, unless the caller set its own.

Knobs (all env, all read once at startup — changes need a stack restart):
  IRIS_THREAD_CAPS=0        disable everything here (exact old behaviour)
  IRIS_THREAD_CAP=4         OMP/MKL/BLAS/torch/cv2 pool size
  IRIS_ORT_THREADS=2        ONNX Runtime intra-op threads per session
  IRIS_ORT_SPIN=0           1 = leave ORT's default spin-waiting on

MediaPipe (hands / face mesh) has no thread knob in its Python tasks API; its
XNNPACK pools still size from the CPU count. The lever there is the step rate
(IRIS_MEDIAPIPE_HZ in iris_runtime) or a systemd CPUAffinity= on the unit.
"""
from __future__ import annotations

import os
import sys
from typing import Any

_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")

_state: dict[str, Any] = {"env": False, "libs": False, "ort_patched": False,
                          "ort_sessions": 0, "notes": []}


def _int_env(name: str, default: int) -> int:
    try:
        v = int(str(os.environ.get(name, default)).strip())
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return os.environ.get("IRIS_THREAD_CAPS", "1").strip().lower() not in ("0", "off", "false", "no")


def cap() -> int:
    return _int_env("IRIS_THREAD_CAP", 4)


def ort_threads() -> int:
    return _int_env("IRIS_ORT_THREADS", 2)


def ort_spin() -> bool:
    return os.environ.get("IRIS_ORT_SPIN", "0").strip().lower() in ("1", "on", "true", "yes")


def _log(msg: str) -> None:
    _state["notes"].append(msg)
    print(f"[thread_caps] {msg}", file=sys.stderr, flush=True)


def apply_env() -> dict[str, str]:
    """Set the pool-size env vars (setdefault). Call before numpy/torch import."""
    if not enabled():
        return {}
    n = str(cap())
    applied = {}
    for k in _ENV_VARS:
        applied[k] = os.environ.setdefault(k, n)
    _state["env"] = True
    return applied


def configure_session_options(so: Any, threads: int | None = None, spin: bool | None = None) -> Any:
    """Cap one ort.SessionOptions in place. Leaves any value the caller set alone
    (intra_op_num_threads == 0 means 'ORT default' = all cores)."""
    threads = ort_threads() if threads is None else threads
    spin = ort_spin() if spin is None else spin
    try:
        if int(getattr(so, "intra_op_num_threads", 0) or 0) == 0:
            so.intra_op_num_threads = threads
        if int(getattr(so, "inter_op_num_threads", 0) or 0) == 0:
            so.inter_op_num_threads = 1
    except Exception as e:  # noqa: BLE001
        _log(f"session thread cap skipped: {e!r}")
    if not spin:
        try:
            have = None
            try:
                have = so.get_session_config_entry("session.intra_op.allow_spinning")
            except Exception:
                have = None
            if not have:
                so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        except Exception as e:  # noqa: BLE001
            _log(f"session spin-off skipped: {e!r}")
    return so


def patch_onnxruntime(ort: Any = None) -> bool:
    """Wrap ort.InferenceSession.__init__ so every session gets capped options.
    Subclasses (InsightFace's PickableInferenceSession) call super().__init__,
    so they're covered too. Idempotent."""
    if ort is None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:  # noqa: BLE001
            _log(f"onnxruntime not importable, no ORT caps: {e!r}")
            return False
    cls = ort.InferenceSession
    if getattr(cls.__init__, "_iris_thread_capped", False):
        return True
    orig = cls.__init__

    def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):  # noqa: N807
        try:
            if sess_options is None:
                sess_options = ort.SessionOptions()
            configure_session_options(sess_options)
            _state["ort_sessions"] += 1
        except Exception as e:  # noqa: BLE001 — never block a model load on a cap
            _log(f"ORT cap failed for one session (loading uncapped): {e!r}")
        return orig(self, path_or_bytes, sess_options, *args, **kwargs)

    __init__._iris_thread_capped = True  # type: ignore[attr-defined]
    __init__._iris_orig = orig  # type: ignore[attr-defined]
    cls.__init__ = __init__
    _state["ort_patched"] = True
    return True


def apply_libs() -> dict[str, Any]:
    """torch / cv2 / onnxruntime caps. Call after the heavy imports."""
    if not enabled():
        return {"enabled": False}
    n = cap()
    out: dict[str, Any] = {"enabled": True, "cap": n, "ort_threads": ort_threads(),
                           "ort_spin": ort_spin()}
    try:
        import torch  # type: ignore
        torch.set_num_threads(n)
        out["torch"] = torch.get_num_threads()
        try:
            torch.set_num_interop_threads(min(n, 2))
        except RuntimeError:
            pass  # only settable before the first inter-op parallel work
    except Exception as e:  # noqa: BLE001
        out["torch"] = f"skipped: {e!r}"
    try:
        import cv2  # type: ignore
        cv2.setNumThreads(n)
        out["cv2"] = cv2.getNumThreads()
    except Exception as e:  # noqa: BLE001
        out["cv2"] = f"skipped: {e!r}"
    out["ort_patched"] = patch_onnxruntime()
    _state["libs"] = True
    _log(f"applied: {out}")
    return out


def status() -> dict[str, Any]:
    return {"enabled": enabled(), "cap": cap(), "ort_threads": ort_threads(),
            "ort_spin": ort_spin(), **{k: v for k, v in _state.items() if k != "notes"},
            "notes": list(_state["notes"][-5:])}
