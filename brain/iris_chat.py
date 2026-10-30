"""
brain/iris_chat.py — pending-chat-request store.

Cross-process bridge between the orb's POST /api/v1/chat (HTTP, blocking
long-poll) and Iris's CC session (MCP, async). The orb writes a pending
request to disk; the Stop hook detects it on the next CC turn and rewakes
the model with a handle-this-chat directive; Iris generates a reply and
calls chat_reply(id, text) which writes the response file; the orb's
long-poll picks it up.

Files:
  state/iris_chat/<id>.json   per-request, status=pending|answered|expired
  state/iris_chat/.pending    flag file: exists iff at least one pending
                              request. Cheap for the Stop hook to test.

Why a flag file *and* status fields: the Stop hook fires after every CC
turn — even when voice mode is off, even when no chat is pending. We
need an O(1) "is there work to do" test that doesn't require listing
the chat directory. The flag is updated whenever submit/mark_answered
run.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl as _fcntl
except Exception:            # non-POSIX (the old Windows tower): thread lock only, as before
    _fcntl = None


# 10-29 code review (message bridges): the orb server, the runtime (chat_reply), the body host and
# the Stop hook are separate PROCESSES writing state/iris_chat. Every write + flag refresh now runs
# under _xlock(): this re-entrant thread lock PLUS an fcntl.flock on state/iris_chat/.lock. If the
# flock can't be had within ~2 s it fails OPEN (the old behaviour) rather than wedging the caller.
_LOCK = threading.RLock()
_TL = threading.local()
_XLOCK_WAIT_S = 2.0
_LOCK_NAME = ".lock"
# Claim lease (10-29 code review): the host poller and the Stop hook could BOTH deliver one chat.
# Whoever delivers first claims it (status stays "pending" so chat_reply/wait_for_reply are
# unchanged); a different owner skips it until the lease runs out.
_CLAIM_LEASE_S = 180.0


@contextlib.contextmanager
def _xlock(dir_path: Optional[Path] = None):
    with _LOCK:
        depth = getattr(_TL, "depth", 0)
        fh = None
        if depth == 0 and _fcntl is not None:
            try:
                d = Path(dir_path) if dir_path is not None else _chat_dir()
                d.mkdir(parents=True, exist_ok=True)
                fh = open(d / _LOCK_NAME, "a+")
                t_end = time.monotonic() + _XLOCK_WAIT_S
                while True:
                    try:
                        _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() > t_end:
                            raise
                        time.sleep(0.01)
            except Exception:
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
                fh = None        # fail-open
        _TL.depth = depth + 1
        try:
            yield
        finally:
            _TL.depth = depth
            if fh is not None:
                try:
                    _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    fh.close()
                except Exception:
                    pass


def _atomic_write_json(path: Path, data: dict) -> None:
    """Unique tmp name + fsync + os.replace (readers never see half-written JSON)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.stem, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False))
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise
_BASE: Path | None = None
_DIR_NAME = "state/iris_chat"
_FLAG_NAME = ".pending"
_REQUEST_TTL_S = 600.0  # requests older than this are considered abandoned (10x HTTP timeout)


def configure(base_dir: Path | str) -> None:
    global _BASE
    _BASE = Path(base_dir)
    # Also configure brain/iris_paths so other modules see the same root.
    try:
        from brain.iris_paths import paths as _paths
        _paths.configure(_BASE)
    except Exception:
        pass
    _chat_dir().mkdir(parents=True, exist_ok=True)


def _chat_dir() -> Path:
    """Canonical chat dir. Prefer brain/iris_paths.paths.chat_dir if configured."""
    try:
        from brain.iris_paths import paths as _paths
        if _paths._root is not None:
            return _paths.chat_dir
    except Exception:
        pass
    base = _BASE if _BASE is not None else Path(".")
    return base / _DIR_NAME


def _flag_path() -> Path:
    try:
        from brain.iris_paths import paths as _paths
        if _paths._root is not None:
            return _paths.chat_pending_flag
    except Exception:
        pass
    return _chat_dir() / _FLAG_NAME


def _request_path(request_id: str) -> Path:
    return _chat_dir() / f"{request_id}.json"


def _refresh_flag() -> None:
    """Set or clear the .pending flag file based on current state on disk. Under _xlock(), so a
    submit in another process can't land between the scan and the unlink (10-29 flag race)."""
    with _xlock():
        _refresh_flag_locked()


def _refresh_flag_locked() -> None:
    has_pending = False
    for path in _chat_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("status") == "pending":
                ts = float(data.get("ts") or 0.0)
                if (time.time() - ts) <= _REQUEST_TTL_S:
                    has_pending = True
                    break
        except Exception:
            continue
    flag = _flag_path()
    if has_pending and not flag.exists():
        flag.write_text("1", encoding="utf-8")
    elif (not has_pending) and flag.exists():
        try:
            flag.unlink()
        except Exception:
            pass


def has_pending() -> bool:
    """O(1) check used by Stop hook."""
    return _flag_path().exists()


def submit(user_text: str, source: str = "orb", trusted: bool = True,
           sender: Optional[str] = None, note: Optional[str] = None) -> str:
    """Write a new pending request, return request_id.

    10-29 (code review, security): `source`/`trusted`/`sender` are STRUCTURED fields. Untrusted senders
    (e.g. a Discord DM relayed from someone who isn't Zeke) get their own wrapper + a talk-only turn in
    the host, and the Stop hook leaves them to the host - their text can no longer pose as Zeke's.
    `note` = trusted delivery instructions written by OUR code (never the sender's words)."""
    request_id = uuid.uuid4().hex[:12]
    entry = {
        "id": request_id,
        "ts": time.time(),
        "user_text": str(user_text or "")[:8000],
        "source": str(source or "orb"),
        "trusted": bool(trusted),
        "sender": sender,
        "note": note,
        "status": "pending",
        "reply": None,
        "answered_ts": None,
    }
    path = _request_path(request_id)
    with _xlock():
        _atomic_write_json(path, entry)
        _refresh_flag_locked()
    return request_id


def _set_status_if_pending(path: Path, status: str, reason: Optional[str] = None) -> bool:
    """Compare-and-set under the cross-process lock: pending -> status, atomic replace. An answer
    that landed in between is never overwritten (the 10-29 lost-reply race)."""
    path = Path(path)
    with _xlock(path.parent):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(data, dict) or data.get("status") != "pending":
            return False
        data["status"] = status
        data[status + "_ts"] = time.time()
        if reason:
            data[status + "_reason"] = str(reason)
        try:
            _atomic_write_json(path, data)
        except Exception:
            return False
        if path.parent == _chat_dir():
            _refresh_flag_locked()
    return True


def claim(request_id: str, owner: str, lease_s: float = _CLAIM_LEASE_S) -> bool:
    """Atomically claim a pending request for delivery: True iff it is still pending AND unclaimed,
    claimed by this same owner (renewal), or its lease has run out. Status stays "pending", so
    chat_reply / wait_for_reply / is_pending behave exactly as before."""
    path = _request_path(str(request_id))
    with _xlock():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(data, dict) or data.get("status") != "pending":
            return False
        now = time.time()
        holder = data.get("claimed_by")
        try:
            until = float(data.get("claimed_until") or 0.0)
        except Exception:
            until = 0.0
        if holder and holder != owner and until > now:
            return False
        data["claimed_by"] = str(owner)
        data["claimed_until"] = now + float(lease_s)
        try:
            _atomic_write_json(path, data)
        except Exception:
            return True      # fail-open: deliver (the pre-claim behaviour) rather than drop
    return True


def claimed_by_other(data: dict, owner: str) -> bool:
    """For scanners: is this request under a live lease held by someone else?"""
    try:
        holder = data.get("claimed_by")
        return bool(holder) and holder != owner and float(data.get("claimed_until") or 0.0) > time.time()
    except Exception:
        return False


def next_pending() -> Optional[dict[str, Any]]:
    """Return the OLDEST pending request (FIFO across all modalities)."""
    pend = list_pending()
    return pend[0] if pend else None


def list_pending() -> list[dict[str, Any]]:
    """Every live pending request, oldest first (10-29 code review: the host used to see only the
    oldest one, so one stale request it had already 'seen' hid every newer one). Expires (CAS,
    atomic) anything past the TTL."""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in _chat_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("status") != "pending":
            continue
        ts = float(data.get("ts") or 0.0)
        if (time.time() - ts) > _REQUEST_TTL_S:
            # mark as expired so we don't keep trying (compare-and-set: an answer wins)
            _set_status_if_pending(path, "expired", "ttl")
            continue
        candidates.append((ts, data))
    candidates.sort(key=lambda kv: kv[0])
    return [d for _ts, d in candidates]


def is_pending(request_id: str) -> bool:
    """True iff this request exists and is still status=pending.

    2026-07-14: fire-time staleness check for the host's orb queue — orb
    requests ride BOTH the host poller and the Stop hook; long host turns
    starve the queue, so items fire here after the Stop hook already
    answered them (a 23:57 cliff ghost re-fired at 00:44). The host drops
    non-pending items at fire time instead of replaying them at me."""
    try:
        path = _request_path(request_id)
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, dict) and data.get("status") == "pending"
    except Exception:
        return False


def mark_answered(request_id: str, reply: str) -> bool:
    """Flip the request to answered and write the reply. Atomic via tmp+rename.

    Phase 41: existence check is now INSIDE the lock to prevent the race where
    two concurrent mark_answered calls on the same id both succeed.
    """
    path = _request_path(request_id)
    with _xlock():
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(data, dict):
            return False
        # Idempotency — if already answered, return False so caller knows.
        if data.get("status") == "answered":
            return False
        data["status"] = "answered"
        data["reply"] = str(reply or "")
        data["answered_ts"] = time.time()
        _atomic_write_json(path, data)
        _refresh_flag_locked()
    return True


def wait_for_reply(request_id: str, timeout_s: float = 60.0) -> Optional[str]:
    """Block up to timeout_s for the request to flip to answered. Returns
    the reply string on success, None on timeout or expired.

    Phase 41: also returns None immediately if the request status flips
    to expired (e.g. iris_runtime down, or stale-cleanup ran). Avoids
    the waiter blocking the full timeout when the answer will never come.
    """
    deadline = time.time() + max(0.5, float(timeout_s))
    path = _request_path(request_id)
    while time.time() < deadline:
        if path.is_file():
            data = None
            for _ in range(3):
                try:
                    raw = path.read_text(encoding="utf-8")   # writes are atomic replaces
                    data = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    time.sleep(0.05)
                    continue
                except Exception:
                    break
            if isinstance(data, dict):
                status = data.get("status")
                if status == "answered":
                    return str(data.get("reply") or "")
                if status == "expired":
                    return None
        time.sleep(0.2)
    return None


def get(request_id: str) -> Optional[dict[str, Any]]:
    path = _request_path(request_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
