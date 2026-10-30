"""
brain/_jsonl_store.py — one shared, cross-process-safe way to touch the memory files.

Why (code review 2026-10-29, fix-first #3): three writers shared canonical memory
(iris_memory, iris_human_memory.record_revisit, scripts/memory_decay.py) with three
separate in-process locks, and two of them used the same `<file>.tmp` name. A memory
appended while decay ran was lost on its rewrite, and two concurrent rewrites could
corrupt the file. Appends never checked for a trailing newline (one torn line then
swallowed the NEXT memory too) and never fsynced, and one bad UTF-8 byte made a whole
store read back as [] with nothing logged.

Everything here is stdlib-only and import-safe on every platform:

  locked(path)              — reentrant per-thread + fcntl.flock on a SIDECAR `<path>.lock`
                              (a lock on the data file itself would be lost when an
                              atomic rewrite swaps the inode). Cross-process AND
                              cross-thread. No fcntl (Windows) => in-process lock only.
  read_jsonl(path)          — per-line tolerant decode. Returns a ReadResult with the
                              entries plus counts (and raw bytes) of the bad lines; the
                              count is logged, never silent.
  append_jsonl(path, rows)  — O_APPEND, prefixes a newline if the file doesn't end in
                              one (isolates a torn tail), one write, fsync.
  rewrite_jsonl(path, rows, salvage=...)
                            — mkstemp in the same dir + fsync + os.replace + dir fsync;
                              keeps the original file mode; raw bad lines are appended
                              to a salvage sidecar first so a rewrite never silently
                              drops bytes it could not parse.
  write_json_atomic(path, obj)
                            — same atomic recipe for a single JSON document.

Callers doing read-modify-write MUST hold locked(path) across the read and the
rewrite; the lock is reentrant so the helpers can take it again inside.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:  # POSIX only; on Windows we degrade to the in-process lock.
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore

_DEFAULT_TIMEOUT_S = 30.0

_REG_LOCK = threading.Lock()
_STATE: dict[str, "_PathLock"] = {}


class _PathLock:
    __slots__ = ("rlock", "depth", "fd", "owner")

    def __init__(self) -> None:
        self.rlock = threading.RLock()
        self.depth = 0
        self.fd: Optional[int] = None
        self.owner: Optional[int] = None


def _key(path: Path | str) -> str:
    return os.path.abspath(os.fspath(path))


def lock_path(path: Path | str) -> Path:
    p = Path(_key(path))
    return p.with_name(p.name + ".lock")


def _log(msg: str) -> None:
    try:
        print(f"[jsonl_store] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


@contextmanager
def locked(path: Path | str, timeout: float = _DEFAULT_TIMEOUT_S,
           fail_open: bool = False) -> Iterator[None]:
    """Exclusive lock for `path`, shared by every thread and process that uses this module.

    Reentrant within a thread. On timeout: raise TimeoutError, or (fail_open=True, for
    readers) carry on unlocked. If the sidecar lock file can't be opened at all
    (permissions, read-only fs) we log and carry on with the in-process lock only —
    the write itself would fail the same way, so there's nothing safer to do.
    """
    k = _key(path)
    with _REG_LOCK:
        st = _STATE.get(k)
        if st is None:
            st = _STATE[k] = _PathLock()
    deadline = time.monotonic() + max(0.0, float(timeout))
    if not st.rlock.acquire(timeout=max(0.0, float(timeout))):
        if fail_open:
            _log(f"in-process lock timeout on {Path(k).name}; reading unlocked")
            yield
            return
        raise TimeoutError(f"lock timeout on {k}")
    took_flock = False
    try:
        if st.depth == 0 and fcntl is not None:
            fd = None
            try:
                lp = lock_path(k)
                lp.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(lp), os.O_RDWR | os.O_CREAT, 0o666)
            except Exception as e:
                _log(f"cannot open lock file for {Path(k).name}: {e!r}; in-process lock only")
            if fd is not None:
                got = False
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        got = True
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(0.02)
                    except OSError as e:  # e.g. fs without flock support
                        _log(f"flock unsupported for {Path(k).name}: {e!r}; in-process lock only")
                        break
                if got:
                    st.fd = fd
                    took_flock = True
                else:
                    os.close(fd)
                    if time.monotonic() >= deadline and not fail_open:
                        raise TimeoutError(f"flock timeout on {k}")
                    if time.monotonic() >= deadline:
                        _log(f"flock timeout on {Path(k).name}; reading unlocked")
        st.depth += 1
        try:
            yield
        finally:
            st.depth -= 1
            if st.depth == 0 and took_flock and st.fd is not None:
                try:
                    fcntl.flock(st.fd, fcntl.LOCK_UN)  # type: ignore[union-attr]
                except Exception:
                    pass
                try:
                    os.close(st.fd)
                except Exception:
                    pass
                st.fd = None
    finally:
        st.rlock.release()


# ── reading ─────────────────────────────────────────────────────────────────

@dataclass
class ReadResult:
    entries: list[dict[str, Any]] = field(default_factory=list)
    nul_repaired: int = 0      # lines with a NUL-filled head (power cut on a non-fsync'd append) that
                               # parsed once the NULs were stripped — recovered, not "bad"
    bad_decode: int = 0        # lines that weren't valid UTF-8 (kept, with U+FFFD, if JSON parsed)
    bad_json: int = 0          # lines that weren't a JSON object (skipped)
    bad_raw: list[bytes] = field(default_factory=list)  # raw bytes of every bad line, for salvage
    error: Optional[str] = None  # whole-file read failure (entries is then [])

    @property
    def bad(self) -> int:
        return self.bad_decode + self.bad_json


def read_jsonl(path: Path | str, log: bool = True) -> ReadResult:
    """Read a JSONL file one line at a time. A bad line never hides the good ones."""
    res = ReadResult()
    p = Path(path)
    try:
        if not p.is_file():
            return res
        data = p.read_bytes()
    except Exception as e:
        res.error = repr(e)
        if log:
            _log(f"read error on {p.name}: {e!r}")
        return res
    for raw in data.split(b"\n"):
        nul = False
        if b"\x00" in raw:
            # A power cut after a non-fsync'd append leaves a run of NULs where
            # data never reached disk; the next append then lands right after
            # it on the same "line". Dropping the NULs recovers that next row.
            stripped = raw.replace(b"\x00", b"")
            nul = stripped != raw
            raw = stripped
        if not raw.strip():
            continue
        decode_bad = False
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            decode_bad = True
            text = raw.decode("utf-8", errors="replace")
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            res.entries.append(obj)
            if nul:
                res.nul_repaired += 1
            if decode_bad:
                res.bad_decode += 1
                res.bad_raw.append(raw)
        else:
            res.bad_json += 1
            res.bad_raw.append(raw)
    if log and (res.bad or res.nul_repaired):
        _log(f"{p.name}: {res.nul_repaired} NUL-damaged line(s) recovered, "
             f"{res.bad_decode} undecodable line(s) kept with U+FFFD, "
             f"{res.bad_json} unparseable line(s) skipped")
    return res


# ── writing ─────────────────────────────────────────────────────────────────

def _dumps_line(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _write_all(fd: int, buf: bytes) -> None:
    view = memoryview(buf)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _fsync_dir(d: Path) -> None:
    if os.name != "posix":
        return
    try:
        dfd = os.open(str(d), os.O_RDONLY)
    except Exception:
        return
    try:
        os.fsync(dfd)
    except Exception:
        pass
    finally:
        os.close(dfd)


def append_jsonl(path: Path | str, entries: Iterable[dict[str, Any]],
                 timeout: float = _DEFAULT_TIMEOUT_S) -> int:
    """Append rows as one O_APPEND write + fsync. Returns the number of rows written."""
    rows = [e for e in entries]
    if not rows:
        return 0
    p = Path(path)
    payload = b"".join(_dumps_line(e) for e in rows)
    with locked(p, timeout=timeout):
        p.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        fd = os.open(str(p), flags, 0o666)
        try:
            size = os.fstat(fd).st_size
            if size > 0:
                if hasattr(os, "pread"):
                    last = os.pread(fd, 1, size - 1)
                else:  # pragma: no cover - Windows
                    os.lseek(fd, size - 1, os.SEEK_SET)
                    last = os.read(fd, 1)
                if last != b"\n":
                    # A torn tail: start on a fresh line so it can't swallow this row.
                    payload = b"\n" + payload
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
    return len(rows)


def _atomic_replace_bytes(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = os.stat(p).st_mode & 0o7777
    except FileNotFoundError:
        mode = None
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        try:
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        if mode is not None:
            try:
                os.chmod(tmp, mode)
            except Exception:
                pass
        else:
            try:  # mkstemp makes 0600; match what a plain open() would have made.
                um = os.umask(0)
                os.umask(um)
                os.chmod(tmp, 0o666 & ~um)
            except Exception:
                pass
        os.replace(tmp, str(p))
        tmp = ""
        _fsync_dir(p.parent)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass


def salvage_path(path: Path | str) -> Path:
    p = Path(path)
    stem = p.name[:-len(".jsonl")] if p.name.endswith(".jsonl") else p.name
    return p.with_name(f"{stem}.salvage_{datetime.now():%Y-%m-%d}_unparseable.jsonl")


def rewrite_jsonl(path: Path | str, entries: Iterable[dict[str, Any]],
                  salvage: Optional[Iterable[bytes]] = None,
                  timeout: float = _DEFAULT_TIMEOUT_S) -> None:
    """Atomically replace `path` with `entries`. Raw `salvage` lines (the bad lines a
    read_jsonl returned) are appended to a dated salvage sidecar FIRST, so bytes we
    couldn't parse are kept rather than silently dropped by the rewrite."""
    p = Path(path)
    data = b"".join(_dumps_line(e) for e in entries)
    with locked(p, timeout=timeout):
        bad = [b for b in (salvage or []) if b and b.strip()]
        if bad:
            sp = salvage_path(p)
            fd = os.open(str(sp), os.O_WRONLY | os.O_APPEND | os.O_CREAT
                         | getattr(os, "O_BINARY", 0), 0o666)
            try:
                _write_all(fd, b"".join(b.rstrip(b"\n") + b"\n" for b in bad))
                os.fsync(fd)
            finally:
                os.close(fd)
            _log(f"{p.name}: kept {len(bad)} bad line(s) in {sp.name} before rewrite")
        _atomic_replace_bytes(p, data)


def write_json_atomic(path: Path | str, obj: Any, indent: Optional[int] = 2,
                      timeout: float = _DEFAULT_TIMEOUT_S) -> None:
    p = Path(path)
    data = json.dumps(obj, ensure_ascii=False, indent=indent).encode("utf-8")
    with locked(p, timeout=timeout):
        _atomic_replace_bytes(p, data)
