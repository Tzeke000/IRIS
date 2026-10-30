"""
brain/iris_events.py — V3 Phase 1 durable event store + atomic task claims.

Steps 1.1 + 1.3 of the V3 Phase 1 plan. ONE SQLite file in WAL mode
(default ``state/v3/events.sqlite``; override with the ``path`` arg or the
``IRIS_EVENTS_DB`` env var) holding:

  events       one row per input (Discord DM, orb chat, voice utterance, VR, letter ...)
               with a lifecycle and a fence
  transitions  append-only audit of every status change (events AND tasks);
               UPDATE/DELETE are refused by triggers
  tasks        delegated work with fenced claims, heartbeats, checkpoint_json and
               a side_effects class
  leases       advisory file leases (path globs) owned by a task + fence

Lifecycle (spec §5):
    received -> admitted -> assigned -> running -> output_pending -> verified -> committed
    plus failed | cancelled | needs_review.
``committed`` and ``cancelled`` are terminal. ``failed`` / ``needs_review`` can be
re-queued (an ``external`` side-effect task only with ``force=True`` — it must be
reconciled against the real world first, never auto-retried).

Every status write is a compare-and-set inside ``BEGIN IMMEDIATE``: the caller
says which status + fence it believes the row has; if the row moved on, the write
loses and returns None/False (re-read and decide). Fences:
  * events: the fence increments on every move INTO ``assigned`` (a new owner);
  * tasks:  ``claim()`` increments the fence and returns it; ``heartbeat``,
    ``complete``, ``fail`` and file leases only succeed with the CURRENT fence, so
    a stale worker can never overwrite a newer result. A lease timeout alone does
    not reject a worker (the fence is the safety, the lease is liveness): an
    expired lease is turned into ``needs_review`` by ``sweep_expired()``, and only
    after that can someone else claim (with a higher fence).

FAIL OPEN: the store must never crash its caller. Every public method catches
everything and returns a falsy default (None / False / [] / {}), recording
``store.last_error``. A corrupt or unopenable DB file is moved aside
(``events.sqlite.corrupt-<stamp>`` + its -wal/-shm), recreated empty, and a
``needs_review`` incident row (source ``store``) is written — the store is
rebuildable from the channel cursors. File-lease checks fail OPEN too (allowed),
because leases are advisory.

Priorities follow the host's ``_SOURCE_PRIORITY``: LOWER number = more urgent.
Timestamps are epoch seconds (float). Stdlib only.
"""
from __future__ import annotations

import fcntl
import fnmatch
import functools
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Optional

log = logging.getLogger("iris_events")

SCHEMA_VERSION = 1

LIFECYCLE = ("received", "admitted", "assigned", "running",
             "output_pending", "verified", "committed")
STATUSES = LIFECYCLE + ("failed", "cancelled", "needs_review")
TERMINAL = frozenset({"committed", "cancelled"})
OPEN_STATUSES = tuple(s for s in STATUSES if s not in TERMINAL)
SIDE_EFFECTS = ("none", "reversible", "external")
_DONE_STAGES = ("output_pending", "verified", "committed")

INCIDENT_SOURCE = "store"

_EVENT_FIELDS = frozenset({"owner", "lease_until", "reply_ref", "note", "priority",
                           "deadline_ts", "trusted", "sender", "text", "channel_msg_id"})
_TASK_FIELDS = frozenset({"owner", "lease_until", "note", "result_ref",
                          "verification_ref", "goal"})


class InsertResult(NamedTuple):
    id: str
    created: bool      # False ⇒ the idempotency key already existed; ``id`` is the original row
    status: str


def flag_enabled() -> bool:
    """The host's dual-write gate (``IRIS_V3_EVENTS=1``)."""
    return (os.environ.get("IRIS_V3_EVENTS") or "").strip() == "1"


def default_db_path() -> Path:
    env = os.environ.get("IRIS_EVENTS_DB")
    if env:
        return Path(env)
    try:
        from brain.iris_paths import paths
        state = paths.state_dir
    except Exception:  # noqa: BLE001 — paths must never block the store
        state = Path(__file__).resolve().parent.parent / "state"
    return Path(state) / "v3" / "events.sqlite"


def _default_lease_root() -> Path:
    try:
        from brain.iris_paths import paths
        return Path(paths.root)
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parent.parent


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY,
  trace_id TEXT, parent_id TEXT,
  source TEXT NOT NULL, channel_msg_id TEXT, sender TEXT,
  trusted INTEGER NOT NULL DEFAULT 0, priority INTEGER NOT NULL DEFAULT 5,
  text TEXT, received_ts REAL NOT NULL, deadline_ts REAL,
  status TEXT NOT NULL CHECK(status IN ({",".join(repr(s) for s in STATUSES)})),
  owner TEXT, lease_until REAL, fence INTEGER NOT NULL DEFAULT 0,
  idempotency_key TEXT UNIQUE, reply_ref TEXT, updated_ts REAL NOT NULL,
  note TEXT, meta_json TEXT);
CREATE INDEX IF NOT EXISTS events_open ON events(status, source, received_ts);
CREATE INDEX IF NOT EXISTS events_lease ON events(lease_until) WHERE lease_until IS NOT NULL;

CREATE TABLE IF NOT EXISTS transitions(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  entity TEXT NOT NULL CHECK(entity IN ('event','task')),
  entity_id TEXT NOT NULL, from_status TEXT, to_status TEXT NOT NULL,
  fence INTEGER, ts REAL NOT NULL, actor TEXT, trace_id TEXT, note TEXT);
CREATE INDEX IF NOT EXISTS transitions_entity ON transitions(entity, entity_id, seq);
CREATE INDEX IF NOT EXISTS transitions_trace ON transitions(trace_id);
CREATE TRIGGER IF NOT EXISTS transitions_no_update BEFORE UPDATE ON transitions
  BEGIN SELECT RAISE(ABORT, 'transitions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS transitions_no_delete BEFORE DELETE ON transitions
  BEGIN SELECT RAISE(ABORT, 'transitions is append-only'); END;

CREATE TABLE IF NOT EXISTS tasks(
  task_id TEXT PRIMARY KEY, parent_event_id TEXT, trace_id TEXT,
  kind TEXT, goal TEXT,
  status TEXT NOT NULL CHECK(status IN ({",".join(repr(s) for s in STATUSES)})),
  owner TEXT, lease_until REAL, lease_s REAL, heartbeat_ts REAL,
  fence INTEGER NOT NULL DEFAULT 0, idempotency_key TEXT UNIQUE,
  result_ref TEXT, verification_ref TEXT, checkpoint_json TEXT,
  side_effects TEXT NOT NULL DEFAULT 'none'
      CHECK(side_effects IN ({",".join(repr(s) for s in SIDE_EFFECTS)})),
  attempts INTEGER NOT NULL DEFAULT 0,
  created_ts REAL NOT NULL, updated_ts REAL NOT NULL, note TEXT);
CREATE INDEX IF NOT EXISTS tasks_open ON tasks(status, lease_until);

CREATE TABLE IF NOT EXISTS leases(
  lease_id INTEGER PRIMARY KEY AUTOINCREMENT,
  path_glob TEXT NOT NULL, task_id TEXT NOT NULL, fence INTEGER, owner TEXT,
  until REAL NOT NULL, created_ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS leases_task ON leases(task_id);
CREATE INDEX IF NOT EXISTS leases_until ON leases(until);
"""


# ── corruption classification ────────────────────────────────────────────────

_CORRUPT_MARKERS = ("not a database", "malformed", "corrupt", "file is encrypted")


def _is_corrupt(e: BaseException) -> bool:
    if isinstance(e, sqlite3.IntegrityError):
        return False
    msg = str(e).lower()
    if isinstance(e, sqlite3.DatabaseError) and any(m in msg for m in _CORRUPT_MARKERS):
        return True
    return isinstance(e, sqlite3.OperationalError) and "unable to open database file" in msg


def _default_value(default: Any) -> Any:
    return default() if callable(default) else default


def _guard(default: Any) -> Callable:
    """Fail-open wrapper: never raise to the caller. Corruption ⇒ recover + retry once."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self: "EventStore", *a: Any, **kw: Any) -> Any:
            with self._lock:
                for attempt in (0, 1):
                    try:
                        self._ensure_conn()
                        return fn(self, *a, **kw)
                    except Exception as e:  # noqa: BLE001 — FAIL OPEN is the contract
                        self.last_error = f"{fn.__name__}: {e!r}"
                        if attempt == 0 and _is_corrupt(e):
                            log.error("iris_events: %s hit a corrupt store (%r); recovering",
                                      fn.__name__, e)
                            try:
                                self._recover(f"{fn.__name__}: {e!r}")
                                continue
                            except Exception as e2:  # noqa: BLE001
                                self.degraded = True
                                self.last_error = f"recover failed: {e2!r}"
                                log.error("iris_events: recovery failed: %r", e2)
                        else:
                            log.warning("iris_events: %s failed open: %r", fn.__name__, e)
                        return _default_value(default)
                return _default_value(default)
        return wrapper
    return deco


# ── glob overlap (file leases) ───────────────────────────────────────────────

_WILD = re.compile(r"[*?\[]")


def _norm_glob(p: str, root: Optional[Path]) -> str:
    s = str(p).replace("\\", "/").strip()
    if root is not None and s.startswith("/"):
        r = str(root).replace("\\", "/").rstrip("/") + "/"
        if s.startswith(r):
            s = s[len(r):]
    trailing = s.endswith("/")
    absolute = s.startswith("/")
    parts = [x for x in s.split("/") if x not in ("", ".")]
    out: list[str] = []
    for x in parts:
        if x == ".." and out and out[-1] not in ("..", "**"):
            out.pop()
        else:
            out.append(x)
    if trailing:
        out.append("**")
    return ("/" if absolute else "") + "/".join(out)


def _segs(g: str) -> tuple[str, ...]:
    return (("/",) if g.startswith("/") else ()) + tuple(x for x in g.split("/") if x)


def _seg_overlap(x: str, y: str) -> bool:
    wx, wy = bool(_WILD.search(x)), bool(_WILD.search(y))
    if not wx and not wy:
        return x == y
    if wx and not wy:
        return fnmatch.fnmatchcase(y, x)
    if wy and not wx:
        return fnmatch.fnmatchcase(x, y)
    # both wild: refuse only when fixed prefixes or fixed suffixes clearly disagree
    # (conservative — an advisory lease should over-refuse, never under-refuse)
    px, py = _WILD.split(x, 1)[0], _WILD.split(y, 1)[0]
    if not (px.startswith(py) or py.startswith(px)):
        return False
    sx, sy = re.split(r"[*?\]]", x)[-1], re.split(r"[*?\]]", y)[-1]
    return sx.endswith(sy) or sy.endswith(sx)


@functools.lru_cache(maxsize=4096)
def _overlap(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    if not a and not b:
        return True
    if a and a[0] == "**":
        return _overlap(a[1:], b) or (bool(b) and _overlap(a, b[1:]))
    if b and b[0] == "**":
        return _overlap(a, b[1:]) or (bool(a) and _overlap(a[1:], b))
    if not a or not b:
        return False
    return _seg_overlap(a[0], b[0]) and _overlap(a[1:], b[1:])


def globs_overlap(a: str, b: str, root: Optional[Path] = None) -> bool:
    """True if some path could match both globs (``**`` = any number of segments)."""
    return _overlap(_segs(_norm_glob(a, root)), _segs(_norm_glob(b, root)))


# ── the store ────────────────────────────────────────────────────────────────

class EventStore:
    def __init__(self, path: Optional[os.PathLike | str] = None, *,
                 synchronous: str = "FULL", busy_timeout_s: float = 30.0,
                 clock: Callable[[], float] = time.time,
                 lease_root: Optional[os.PathLike | str] = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.lease_root = Path(lease_root) if lease_root else _default_lease_root()
        self._sync = synchronous if synchronous in ("OFF", "NORMAL", "FULL", "EXTRA") else "FULL"
        self._busy = float(busy_timeout_s)
        self._clock = clock
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._pid: Optional[int] = None
        self._ino: Optional[int] = None
        self._forked_conns: list = []   # never close a connection inherited across fork
        self.degraded = False
        self.last_error: Optional[str] = None
        self.recovered_to: Optional[Path] = None
        with self._lock:
            try:
                self._ensure_conn()
            except Exception as e:  # noqa: BLE001
                self.degraded = True
                self.last_error = f"open: {e!r}"
                log.error("iris_events: store unavailable, failing open: %r", e)

    # ── connection management ──
    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._busy
        while True:
            c = sqlite3.connect(str(self.path), timeout=self._busy,
                                isolation_level=None, check_same_thread=False)
            try:
                c.row_factory = sqlite3.Row
                c.execute(f"PRAGMA busy_timeout={int(self._busy * 1000)}")
                mode = c.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if str(mode).lower() != "wal":
                    log.warning("iris_events: journal_mode is %s, not wal", mode)
                c.execute(f"PRAGMA synchronous={self._sync}")
                ver = c.execute("PRAGMA user_version").fetchone()[0]
                if ver < SCHEMA_VERSION:
                    c.execute("BEGIN IMMEDIATE")
                    try:
                        for stmt in _SCHEMA.split(";\n"):
                            if stmt.strip():
                                c.execute(stmt)
                        c.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                        c.execute("COMMIT")
                    except BaseException:
                        c.execute("ROLLBACK")
                        raise
                elif ver > SCHEMA_VERSION:
                    log.warning("iris_events: DB schema v%s is newer than this code (v%s)",
                                ver, SCHEMA_VERSION)
                c.execute("SELECT count(*) FROM events").fetchone()
                return c
            except sqlite3.OperationalError as e:
                c.close()
                if "locked" in str(e).lower() and time.monotonic() < deadline:
                    time.sleep(0.02)
                    continue
                raise
            except BaseException:
                c.close()
                raise

    def _open(self) -> None:
        self._conn = self._connect()
        self._pid = os.getpid()
        try:
            self._ino = os.stat(self.path).st_ino
        except OSError:
            self._ino = None
        self.degraded = False

    def _drop_conn(self) -> None:
        c, self._conn = self._conn, None
        if c is None:
            return
        if self._pid != os.getpid():
            self._forked_conns.append(c)
            return
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_conn(self) -> None:
        if self._conn is not None and self._pid == os.getpid():
            try:
                ino = os.stat(self.path).st_ino
            except FileNotFoundError:
                ino = None
            if ino is not None and ino == self._ino:
                return
            # the file was moved aside (another process recovered it) or deleted
        self._drop_conn()
        try:
            self._open()
        except Exception as e:  # noqa: BLE001
            if _is_corrupt(e) or (self.path.exists() and not self.path.is_file()):
                self._recover(f"open: {e!r}")
            else:
                raise

    def _probe_ok(self) -> bool:
        if not self.path.exists():
            return True
        if not self.path.is_file():
            return False
        try:
            c = sqlite3.connect(str(self.path), timeout=self._busy, isolation_level=None)
            try:
                r = c.execute("PRAGMA quick_check").fetchone()
                c.execute("SELECT count(*) FROM events").fetchone()
                return bool(r) and r[0] == "ok"
            finally:
                c.close()
        except Exception:  # noqa: BLE001
            return False

    def _recover(self, reason: str) -> None:
        """Move a corrupt/unopenable DB aside, recreate it, write an incident row.
        Serialised across processes with a flock; re-probes under the lock so two
        processes that both saw the corruption don't move a freshly rebuilt file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".recover.lock")
        with open(lock_path, "a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            self._drop_conn()
            if self._probe_ok():
                self._open()
                return
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{uuid.uuid4().hex[:6]}"
            moved = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(self.path) + suffix)
                if p.exists() or p.is_symlink():
                    os.replace(p, Path(str(moved) + suffix))
            self._open()
            self.recovered_to = moved
            now = self._clock()
            eid = "incident_" + uuid.uuid4().hex
            note = f"event store unreadable ({reason[:300]}); moved aside to {moved.name}; rebuilt empty"
            with self._tx() as c:
                c.execute(
                    "INSERT INTO events(event_id, trace_id, source, sender, trusted, priority, text,"
                    " received_ts, status, fence, idempotency_key, updated_ts, note, meta_json)"
                    " VALUES(?,?,?,?,1,0,?,?, 'needs_review', 0, ?, ?, ?, ?)",
                    (eid, eid, INCIDENT_SOURCE, "iris_events", note, now,
                     f"incident:store_recreated:{stamp}", now, reason[:500],
                     json.dumps({"kind": "store_recreated", "moved_to": str(moved)})))
                self._audit(c, "event", eid, None, "needs_review", 0, now, "iris_events", eid, note)
            log.error("iris_events: %s", note)

    # ── transaction + helpers ──
    class _Tx:
        def __init__(self, store: "EventStore") -> None:
            self.c = store._conn

        def __enter__(self) -> sqlite3.Connection:
            self.c.execute("BEGIN IMMEDIATE")
            return self.c

        def __exit__(self, et, ev, tb) -> bool:
            if et is None:
                self.c.execute("COMMIT")
            else:
                try:
                    self.c.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
            return False

    def _tx(self) -> "EventStore._Tx":
        return EventStore._Tx(self)

    @staticmethod
    def _audit(c: sqlite3.Connection, entity: str, eid: str, frm: Optional[str], to: str,
               fence: Optional[int], ts: float, actor: Optional[str], trace: Optional[str],
               note: Optional[str] = None) -> None:
        c.execute("INSERT INTO transitions(entity, entity_id, from_status, to_status, fence, ts,"
                  " actor, trace_id, note) VALUES(?,?,?,?,?,?,?,?,?)",
                  (entity, eid, frm, to, fence, ts, actor, trace, note))

    @staticmethod
    def _dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
        if row is None:
            return None
        d = dict(row)
        if "meta_json" in d:
            mj = d.pop("meta_json")
            try:
                d["meta"] = json.loads(mj) if mj else {}
            except Exception:  # noqa: BLE001
                d["meta"] = {"_unparsed": mj}
        if "checkpoint_json" in d:
            cj = d.pop("checkpoint_json")
            try:
                d["checkpoint"] = json.loads(cj) if cj else None
            except Exception:  # noqa: BLE001
                d["checkpoint"] = {"_unparsed": cj}
        return d

    @staticmethod
    def _expect(expect_status: Any) -> tuple[str, ...]:
        if expect_status is None:
            return OPEN_STATUSES
        if isinstance(expect_status, str):
            exp = (expect_status,)
        else:
            exp = tuple(expect_status)
        bad = [s for s in exp if s not in STATUSES]
        if bad:
            raise ValueError(f"unknown status {bad}")
        return tuple(s for s in exp if s not in TERMINAL)

    def _cas(self, table: str, key: str, entity: str, row_id: str, expect_status: Any,
             expect_fence: Optional[int], new_status: str, actor: Optional[str],
             fields: dict, allowed: frozenset, bump: bool,
             lease_s: Optional[float]) -> Optional[dict]:
        if new_status not in STATUSES or new_status == "received":
            raise ValueError(f"bad target status {new_status!r}")
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown fields {sorted(bad)}")
        exp = self._expect(expect_status)
        now = self._clock()
        if lease_s is not None:
            fields = dict(fields, lease_until=now + float(lease_s))
        with self._tx() as c:
            row = c.execute(f"SELECT status, fence, trace_id FROM {table} WHERE {key}=?",
                            (row_id,)).fetchone()
            if row is None or row["status"] not in exp:
                return None
            if expect_fence is not None and row["fence"] != int(expect_fence):
                return None
            new_fence = row["fence"] + 1 if bump else row["fence"]
            sets = {"status": new_status, "fence": new_fence, "updated_ts": now, **fields}
            cols = ", ".join(f"{k}=?" for k in sets)
            cur = c.execute(f"UPDATE {table} SET {cols} WHERE {key}=? AND status=? AND fence=?",
                            (*sets.values(), row_id, row["status"], row["fence"]))
            if cur.rowcount != 1:
                return None
            self._audit(c, entity, row_id, row["status"], new_status, new_fence, now, actor,
                        row["trace_id"], fields.get("note"))
            return self._dict(c.execute(f"SELECT * FROM {table} WHERE {key}=?",
                                        (row_id,)).fetchone())

    # ══ EVENTS ═══════════════════════════════════════════════════════════════
    @_guard(None)
    def insert(self, source: str, text: Optional[str] = None, *,
               idempotency_key: Optional[str] = None, status: str = "received",
               event_id: Optional[str] = None, trace_id: Optional[str] = None,
               parent_id: Optional[str] = None, channel_msg_id: Optional[str] = None,
               sender: Optional[str] = None, trusted: bool = False, priority: int = 5,
               received_ts: Optional[float] = None, deadline_ts: Optional[float] = None,
               note: Optional[str] = None, meta: Optional[dict] = None,
               actor: Optional[str] = None) -> Optional[InsertResult]:
        """Idempotent insert. A repeated ``idempotency_key`` returns the ORIGINAL row
        (``created=False``) and writes nothing. ``trace_id`` defaults to the parent's
        trace (if ``parent_id``) else this event's own id (a root input)."""
        if status not in ("received", "admitted"):
            raise ValueError("an event is inserted as 'received' or 'admitted'")
        if not source:
            raise ValueError("source is required")
        now = self._clock()
        eid = event_id or "ev_" + uuid.uuid4().hex
        with self._tx() as c:
            if idempotency_key:
                row = c.execute("SELECT event_id, status FROM events WHERE idempotency_key=?",
                                (idempotency_key,)).fetchone()
                if row:
                    return InsertResult(row["event_id"], False, row["status"])
            trace = trace_id
            if trace is None and parent_id:
                p = c.execute("SELECT trace_id FROM events WHERE event_id=?",
                              (parent_id,)).fetchone()
                trace = p["trace_id"] if p and p["trace_id"] else parent_id
            trace = trace or eid
            c.execute(
                "INSERT INTO events(event_id, trace_id, parent_id, source, channel_msg_id, sender,"
                " trusted, priority, text, received_ts, deadline_ts, status, fence, idempotency_key,"
                " updated_ts, note, meta_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?)",
                (eid, trace, parent_id, source, channel_msg_id, sender, 1 if trusted else 0,
                 int(priority), text, received_ts if received_ts is not None else now,
                 deadline_ts, status, idempotency_key, now, note,
                 json.dumps(meta, default=str) if meta else None))
            self._audit(c, "event", eid, None, status, 0, now, actor, trace, note)
        return InsertResult(eid, True, status)

    @_guard(None)
    def transition(self, event_id: str, expect_status: Any, expect_fence: Optional[int],
                   new_status: str, *, actor: Optional[str] = None,
                   lease_s: Optional[float] = None, **fields: Any) -> Optional[dict]:
        """CAS status write. ``expect_status``: a status, a collection of statuses, or
        None (= any non-terminal). ``expect_fence``: int, or None to skip the fence
        check. Moving INTO ``assigned`` increments the fence. Returns the updated row,
        or None if the row isn't in the expected state (lost the race — re-read)."""
        return self._cas("events", "event_id", "event", event_id, expect_status,
                         expect_fence, new_status, actor, fields, _EVENT_FIELDS,
                         new_status == "assigned", lease_s)

    @_guard(None)
    def get(self, event_id: str) -> Optional[dict]:
        return self._dict(self._conn.execute("SELECT * FROM events WHERE event_id=?",
                                             (event_id,)).fetchone())

    @_guard(None)
    def find_by_key(self, idempotency_key: str) -> Optional[dict]:
        return self._dict(self._conn.execute("SELECT * FROM events WHERE idempotency_key=?",
                                             (idempotency_key,)).fetchone())

    @_guard(list)
    def unanswered(self, source: Optional[str] = None, *, limit: int = 500,
                   include_incidents: bool = False) -> list[dict]:
        """Events not yet ``committed`` (and not ``cancelled``), most urgent first
        (priority ASC), then oldest first. Store incidents are excluded unless asked
        for (or ``source='store'``)."""
        q = "SELECT * FROM events WHERE status NOT IN ('committed','cancelled')"
        args: list = []
        if source is not None:
            q += " AND source=?"
            args.append(source)
        elif not include_incidents:
            q += " AND source<>?"
            args.append(INCIDENT_SOURCE)
        q += " ORDER BY priority ASC, received_ts ASC LIMIT ?"
        args.append(int(limit))
        return [self._dict(r) for r in self._conn.execute(q, args).fetchall()]

    @_guard(list)
    def incidents(self, *, open_only: bool = True) -> list[dict]:
        q = "SELECT * FROM events WHERE source=?"
        if open_only:
            q += " AND status NOT IN ('committed','cancelled')"
        return [self._dict(r) for r in
                self._conn.execute(q + " ORDER BY received_ts", (INCIDENT_SOURCE,)).fetchall()]

    @_guard(False)
    def mark_replied(self, event_id: str, reply_ref: str, *, actor: Optional[str] = None,
                     commit: bool = True) -> bool:
        """Record that a reply for this event WENT OUT (the reply is a fact, so no
        status CAS): stores ``reply_ref`` (comma-joined if several) and, with
        ``commit=True``, moves the event to ``committed`` from any status. Without
        commit it moves a pre-output event to ``output_pending``. Idempotent."""
        if not reply_ref:
            raise ValueError("reply_ref is required")
        now = self._clock()
        ref = str(reply_ref)
        with self._tx() as c:
            row = c.execute("SELECT status, fence, trace_id, reply_ref FROM events"
                            " WHERE event_id=?", (event_id,)).fetchone()
            if row is None:
                return False
            refs = [r for r in (row["reply_ref"] or "").split(",") if r]
            new_refs = refs if ref in refs else refs + [ref]
            if commit:
                new_status = "committed"
            elif row["status"] in ("received", "admitted", "assigned", "running"):
                new_status = "output_pending"
            else:
                new_status = row["status"]
            if new_status == row["status"] and new_refs == refs:
                return True
            c.execute("UPDATE events SET status=?, reply_ref=?, updated_ts=? WHERE event_id=?",
                      (new_status, ",".join(new_refs), now, event_id))
            self._audit(c, "event", event_id, row["status"], new_status, row["fence"], now,
                        actor, row["trace_id"], f"reply {ref}")
        return True

    @_guard(list)
    def history(self, entity_id: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM transitions WHERE entity_id=? ORDER BY seq", (entity_id,)).fetchall()]

    @_guard(list)
    def trace(self, trace_id: str) -> list[dict]:
        """Every transition stamped with a trace (Step 1.4 stage timing)."""
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM transitions WHERE trace_id=? ORDER BY seq", (trace_id,)).fetchall()]

    # ══ TASKS ════════════════════════════════════════════════════════════════
    @_guard(None)
    def create_task(self, goal: str, *, kind: Optional[str] = None,
                    parent_event_id: Optional[str] = None,
                    idempotency_key: Optional[str] = None, side_effects: str = "none",
                    task_id: Optional[str] = None, trace_id: Optional[str] = None,
                    checkpoint: Optional[dict] = None, note: Optional[str] = None,
                    actor: Optional[str] = None) -> Optional[InsertResult]:
        """New task in ``admitted`` (claimable). Idempotent on ``idempotency_key``."""
        if side_effects not in SIDE_EFFECTS:
            raise ValueError(f"side_effects must be one of {SIDE_EFFECTS}")
        now = self._clock()
        tid = task_id or "task_" + uuid.uuid4().hex
        with self._tx() as c:
            if idempotency_key:
                row = c.execute("SELECT task_id, status FROM tasks WHERE idempotency_key=?",
                                (idempotency_key,)).fetchone()
                if row:
                    return InsertResult(row["task_id"], False, row["status"])
            trace = trace_id
            if trace is None and parent_event_id:
                p = c.execute("SELECT trace_id FROM events WHERE event_id=?",
                              (parent_event_id,)).fetchone()
                trace = p["trace_id"] if p and p["trace_id"] else parent_event_id
            trace = trace or tid
            c.execute(
                "INSERT INTO tasks(task_id, parent_event_id, trace_id, kind, goal, status, fence,"
                " idempotency_key, checkpoint_json, side_effects, created_ts, updated_ts, note)"
                " VALUES(?,?,?,?,?, 'admitted', 0, ?,?,?,?,?,?)",
                (tid, parent_event_id, trace, kind, goal, idempotency_key,
                 json.dumps(checkpoint, default=str) if checkpoint is not None else None,
                 side_effects, now, now, note))
            self._audit(c, "task", tid, None, "admitted", 0, now, actor, trace, note)
        return InsertResult(tid, True, "admitted")

    @_guard(None)
    def get_task(self, task_id: str) -> Optional[dict]:
        return self._dict(self._conn.execute("SELECT * FROM tasks WHERE task_id=?",
                                             (task_id,)).fetchone())

    @_guard(None)
    def find_task_by_key(self, idempotency_key: str) -> Optional[dict]:
        return self._dict(self._conn.execute("SELECT * FROM tasks WHERE idempotency_key=?",
                                             (idempotency_key,)).fetchone())

    @_guard(list)
    def open_tasks(self, *, kind: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM tasks WHERE status NOT IN ('committed','cancelled')"
        args: list = []
        if kind is not None:
            q += " AND kind=?"
            args.append(kind)
        return [self._dict(r) for r in self._conn.execute(q + " ORDER BY created_ts",
                                                           args).fetchall()]

    @_guard(None)
    def claim(self, task_id: str, owner: str, lease_s: float = 300.0, *,
              actor: Optional[str] = None) -> Optional[int]:
        """Claim an ``admitted`` task → ``running``. Increments and returns the fence;
        None if someone else holds it / it isn't claimable. (An expired claim is NOT
        taken over here — ``sweep_expired`` + ``requeue_task`` first.)"""
        if not owner:
            raise ValueError("owner is required")
        now = self._clock()
        with self._tx() as c:
            row = c.execute("SELECT status, fence, trace_id FROM tasks WHERE task_id=?",
                            (task_id,)).fetchone()
            if row is None or row["status"] != "admitted":
                return None
            fence = row["fence"] + 1
            cur = c.execute(
                "UPDATE tasks SET status='running', owner=?, lease_until=?, lease_s=?,"
                " heartbeat_ts=?, fence=?, attempts=attempts+1, updated_ts=?"
                " WHERE task_id=? AND status='admitted' AND fence=?",
                (owner, now + float(lease_s), float(lease_s), now, fence, now,
                 task_id, row["fence"]))
            if cur.rowcount != 1:
                return None
            self._audit(c, "task", task_id, "admitted", "running", fence, now,
                        actor or owner, row["trace_id"], f"claimed by {owner}")
        return fence

    @_guard(False)
    def heartbeat(self, task_id: str, fence: int, lease_s: Optional[float] = None) -> bool:
        """Extend the claim (and this fence's file leases). False ⇒ you no longer own
        the task — stop working and don't write results."""
        now = self._clock()
        with self._tx() as c:
            row = c.execute("SELECT status, fence, lease_s FROM tasks WHERE task_id=?",
                            (task_id,)).fetchone()
            if row is None or row["status"] != "running" or row["fence"] != int(fence):
                return False
            ls = float(lease_s if lease_s is not None else (row["lease_s"] or 300.0))
            until = now + ls
            c.execute("UPDATE tasks SET heartbeat_ts=?, lease_until=?, lease_s=?, updated_ts=?"
                      " WHERE task_id=? AND fence=?", (now, until, ls, now, task_id, int(fence)))
            c.execute("UPDATE leases SET until=MAX(until, ?) WHERE task_id=? AND fence=?",
                      (until, task_id, int(fence)))
        return True

    @_guard(False)
    def complete(self, task_id: str, fence: int, result_ref: Optional[str] = None, *,
                 status: str = "output_pending", verification_ref: Optional[str] = None,
                 actor: Optional[str] = None, note: Optional[str] = None) -> bool:
        """Record a result. Succeeds only with the CURRENT fence. ``status`` is one of
        output_pending | verified | committed and may advance a completed task (same
        fence, never backwards). ``verified`` needs a ``verification_ref`` (an ``act``
        ledger id); so does ``committed`` for an ``external`` side-effect task.
        Repeating a completion (same fence, same or no result_ref, already at/after that
        stage) is an idempotent True; a different result for the same fence is False."""
        if status not in _DONE_STAGES:
            raise ValueError(f"complete status must be one of {_DONE_STAGES}")
        now = self._clock()
        with self._tx() as c:
            row = c.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None or row["fence"] != int(fence):
                return False
            cur_st = row["status"]
            same_result = result_ref is None or row["result_ref"] in (None, result_ref)
            if cur_st in _DONE_STAGES:
                if not same_result:
                    return False
                if LIFECYCLE.index(cur_st) >= LIFECYCLE.index(status):
                    return result_ref is None or row["result_ref"] == result_ref
            elif cur_st != "running":
                return False
            vref = verification_ref or row["verification_ref"]
            if status == "verified" and not vref:
                return False
            if status == "committed" and row["side_effects"] == "external" and not vref:
                return False
            cur = c.execute(
                "UPDATE tasks SET status=?, result_ref=COALESCE(?, result_ref),"
                " verification_ref=COALESCE(?, verification_ref), lease_until=NULL,"
                " updated_ts=?, note=COALESCE(?, note) WHERE task_id=? AND fence=? AND status=?",
                (status, result_ref, verification_ref, now, note, task_id, int(fence), cur_st))
            if cur.rowcount != 1:
                return False
            self._audit(c, "task", task_id, cur_st, status, int(fence), now,
                        actor or row["owner"], row["trace_id"],
                        note or (f"result {result_ref}" if result_ref else None))
            c.execute("DELETE FROM leases WHERE task_id=?", (task_id,))
        return True

    @_guard(False)
    def fail(self, task_id: str, fence: int, note: Optional[str] = None, *,
             actor: Optional[str] = None) -> bool:
        r = self._cas("tasks", "task_id", "task", task_id, ("running", "output_pending"),
                      fence, "failed", actor, {"note": note, "lease_until": None},
                      _TASK_FIELDS, False, None)
        if r is not None:
            self._release(task_id)
        return r is not None

    @_guard(False)
    def cancel_task(self, task_id: str, note: Optional[str] = None, *,
                    actor: Optional[str] = None) -> bool:
        r = self._cas("tasks", "task_id", "task", task_id, None, None, "cancelled", actor,
                      {"note": note, "lease_until": None}, _TASK_FIELDS, False, None)
        if r is not None:
            self._release(task_id)
        return r is not None

    @_guard(False)
    def requeue_task(self, task_id: str, *, force: bool = False, actor: Optional[str] = None,
                     note: Optional[str] = None) -> bool:
        """failed | needs_review → admitted (claimable again). An ``external``
        side-effect task needs ``force=True`` — reconcile against the real world
        first ("never assume an issued command succeeded", spec §5)."""
        t = self._conn.execute("SELECT side_effects FROM tasks WHERE task_id=?",
                               (task_id,)).fetchone()
        if t is None or (t["side_effects"] == "external" and not force):
            return False
        r = self._cas("tasks", "task_id", "task", task_id, ("failed", "needs_review"), None,
                      "admitted", actor, {"owner": None, "lease_until": None,
                                          "note": note or "requeued"}, _TASK_FIELDS, False, None)
        return r is not None

    @_guard(None)
    def transition_task(self, task_id: str, expect_status: Any, expect_fence: Optional[int],
                        new_status: str, *, actor: Optional[str] = None,
                        **fields: Any) -> Optional[dict]:
        """Low-level CAS for tasks (prefer claim/complete/fail/requeue)."""
        return self._cas("tasks", "task_id", "task", task_id, expect_status, expect_fence,
                         new_status, actor, fields, _TASK_FIELDS, False, None)

    @_guard(False)
    def write_checkpoint(self, task_id: str, checkpoint: dict, *,
                         fence: Optional[int] = None) -> bool:
        """Overwrite the task's handoff record (goal, steps done, outstanding, files,
        running jobs, locks, next safe action). With ``fence``, only the current
        holder may write."""
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dict")
        now = self._clock()
        body = json.dumps(dict(checkpoint, _ts=now), default=str)
        with self._tx() as c:
            if fence is None:
                cur = c.execute("UPDATE tasks SET checkpoint_json=?, updated_ts=? WHERE task_id=?",
                                (body, now, task_id))
            else:
                cur = c.execute("UPDATE tasks SET checkpoint_json=?, updated_ts=?"
                                " WHERE task_id=? AND fence=?", (body, now, task_id, int(fence)))
            return cur.rowcount == 1

    @_guard(None)
    def read_checkpoint(self, task_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT checkpoint_json FROM tasks WHERE task_id=?",
                                 (task_id,)).fetchone()
        if row is None or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except Exception:  # noqa: BLE001
            return {"_unparsed": row[0]}

    # ══ SWEEP ════════════════════════════════════════════════════════════════
    @_guard(dict)
    def sweep_expired(self, now: Optional[float] = None, *, auto_retry: bool = False,
                      max_attempts: int = 3, actor: str = "sweep") -> dict:
        """Expired claims never stay silently ``running``:
          * tasks running past ``lease_until`` → ``needs_review`` (or, with
            ``auto_retry``, back to ``admitted`` if side_effects != external and
            attempts < max_attempts; ``external`` ALWAYS goes to needs_review);
          * events assigned/running past ``lease_until`` → ``needs_review``;
          * expired file leases are deleted.
        Returns counts."""
        now = self._clock() if now is None else float(now)
        out = {"tasks_needs_review": 0, "tasks_requeued": 0, "events_needs_review": 0,
               "leases_dropped": 0}
        with self._tx() as c:
            for t in c.execute("SELECT task_id, fence, owner, side_effects, attempts, trace_id"
                               " FROM tasks WHERE status='running' AND lease_until IS NOT NULL"
                               " AND lease_until < ?", (now,)).fetchall():
                retry = (auto_retry and t["side_effects"] != "external"
                         and t["attempts"] < int(max_attempts))
                to = "admitted" if retry else "needs_review"
                note = (f"lease expired (owner {t['owner']}, fence {t['fence']},"
                        f" side_effects {t['side_effects']})")
                c.execute("UPDATE tasks SET status=?, lease_until=NULL, updated_ts=?, note=?"
                          + (", owner=NULL" if retry else "")
                          + " WHERE task_id=? AND status='running' AND fence=?",
                          (to, now, note, t["task_id"], t["fence"]))
                self._audit(c, "task", t["task_id"], "running", to, t["fence"], now, actor,
                            t["trace_id"], note)
                c.execute("DELETE FROM leases WHERE task_id=?", (t["task_id"],))
                out["tasks_requeued" if retry else "tasks_needs_review"] += 1
            for e in c.execute("SELECT event_id, status, fence, owner, trace_id FROM events"
                               " WHERE status IN ('assigned','running') AND lease_until IS NOT NULL"
                               " AND lease_until < ?", (now,)).fetchall():
                note = f"lease expired (owner {e['owner']}, fence {e['fence']})"
                c.execute("UPDATE events SET status='needs_review', lease_until=NULL,"
                          " updated_ts=?, note=? WHERE event_id=? AND status=? AND fence=?",
                          (now, note, e["event_id"], e["status"], e["fence"]))
                self._audit(c, "event", e["event_id"], e["status"], "needs_review", e["fence"],
                            now, actor, e["trace_id"], note)
                out["events_needs_review"] += 1
            out["leases_dropped"] = c.execute("DELETE FROM leases WHERE until < ?",
                                              (now,)).rowcount
        return out

    # ══ FILE LEASES ══════════════════════════════════════════════════════════
    def _live_leases(self, c: sqlite3.Connection, now: float) -> list[sqlite3.Row]:
        """Leases still in force: not expired, and the holder task (if it exists) still
        holds the same fence and isn't finished."""
        return c.execute(
            "SELECT l.* FROM leases l LEFT JOIN tasks t ON t.task_id = l.task_id"
            " WHERE l.until >= ? AND (t.task_id IS NULL OR ("
            "   (l.fence IS NULL OR t.fence = l.fence)"
            "   AND t.status NOT IN ('committed','cancelled','failed')))",
            (now,)).fetchall()

    @_guard(lambda: {"ok": True, "degraded": True, "lease_ids": [], "conflicts": []})
    def acquire_leases(self, task_id: str, globs: Iterable[str], *,
                       fence: Optional[int] = None, ttl_s: Optional[float] = None,
                       owner: Optional[str] = None) -> dict:
        """Declare the paths a task may write. Refused (``ok=False`` + ``conflicts``)
        if ANY glob overlaps a live lease of another task. All-or-nothing. Lease
        lifetime = ``ttl_s`` or the task's own lease (default 600 s); heartbeats extend
        it. Fails OPEN (``ok=True, degraded=True``) if the store is broken."""
        gl = [_norm_glob(g, self.lease_root) for g in globs if str(g).strip()]
        if not gl:
            raise ValueError("no globs")
        now = self._clock()
        with self._tx() as c:
            t = c.execute("SELECT status, fence, lease_until, owner FROM tasks WHERE task_id=?",
                          (task_id,)).fetchone()
            if t is not None:
                if t["status"] in ("committed", "cancelled", "failed"):
                    return {"ok": False, "lease_ids": [], "conflicts": [],
                            "reason": f"task is {t['status']}"}
                if fence is not None and t["fence"] != int(fence):
                    return {"ok": False, "lease_ids": [], "conflicts": [],
                            "reason": f"stale fence {fence} (current {t['fence']})"}
                if fence is None:
                    fence = t["fence"]
            until = now + float(ttl_s) if ttl_s is not None else (
                t["lease_until"] if t is not None and t["lease_until"] else now + 600.0)
            conflicts = []
            for lease in self._live_leases(c, now):
                if lease["task_id"] == task_id:
                    continue
                for g in gl:
                    if _overlap(_segs(g), _segs(lease["path_glob"])):
                        conflicts.append({"glob": g, "held_glob": lease["path_glob"],
                                          "held_by": lease["task_id"], "owner": lease["owner"],
                                          "until": lease["until"]})
            if conflicts:
                return {"ok": False, "lease_ids": [], "conflicts": conflicts,
                        "reason": "overlaps a live lease"}
            ids = []
            for g in gl:
                cur = c.execute("INSERT INTO leases(path_glob, task_id, fence, owner, until,"
                                " created_ts) VALUES(?,?,?,?,?,?)",
                                (g, task_id, fence, owner or (t["owner"] if t else None),
                                 until, now))
                ids.append(cur.lastrowid)
        return {"ok": True, "lease_ids": ids, "conflicts": [], "until": until}

    def _release(self, task_id: str, fence: Optional[int] = None) -> int:
        with self._tx() as c:
            if fence is None:
                return c.execute("DELETE FROM leases WHERE task_id=?", (task_id,)).rowcount
            return c.execute("DELETE FROM leases WHERE task_id=? AND fence=?",
                             (task_id, int(fence))).rowcount

    @_guard(0)
    def release_leases(self, task_id: str, fence: Optional[int] = None) -> int:
        return self._release(task_id, fence)

    @_guard(list)
    def lease_holders(self, path: str) -> list[dict]:
        """Live leases whose glob covers ``path``."""
        p = _segs(_norm_glob(path, self.lease_root))
        now = self._clock()
        return [dict(l) for l in self._live_leases(self._conn, now)
                if _overlap(p, _segs(l["path_glob"]))]

    @_guard(lambda: {"allowed": True, "degraded": True, "reason": "store unavailable"})
    def check_write(self, task_id: str, path: str) -> dict:
        """For the PreToolUse hook: may ``task_id`` write ``path``? Allowed iff one of
        the task's own live leases covers it and no OTHER task's live lease does."""
        p = _segs(_norm_glob(path, self.lease_root))
        now = self._clock()
        mine, theirs = [], []
        for l in self._live_leases(self._conn, now):
            if _overlap(p, _segs(l["path_glob"])):
                (mine if l["task_id"] == task_id else theirs).append(dict(l))
        if theirs:
            return {"allowed": False, "reason": "leased by another task",
                    "held_by": sorted({l["task_id"] for l in theirs})}
        if not mine:
            return {"allowed": False, "reason": "outside this task's declared globs"}
        return {"allowed": True, "reason": "covered by own lease"}

    # ══ HEALTH ═══════════════════════════════════════════════════════════════
    @_guard(lambda: {"ok": False})
    def stats(self, now: Optional[float] = None) -> dict:
        """For the Step 1.5 health snapshot."""
        now = self._clock() if now is None else float(now)
        c = self._conn
        by_status = {r[0]: r[1] for r in c.execute(
            "SELECT status, count(*) FROM events GROUP BY status")}
        tasks = {r[0]: r[1] for r in c.execute(
            "SELECT status, count(*) FROM tasks GROUP BY status")}
        oldest = {r[0]: round(now - r[1], 3) for r in c.execute(
            "SELECT source, MIN(received_ts) FROM events"
            " WHERE status NOT IN ('committed','cancelled') GROUP BY source")}
        expired = c.execute("SELECT count(*) FROM tasks WHERE status='running'"
                            " AND lease_until < ?", (now,)).fetchone()[0]
        return {"ok": not self.degraded, "path": str(self.path), "events": by_status,
                "tasks": tasks, "oldest_open_age_s": oldest, "expired_task_leases": expired,
                "open_incidents": c.execute(
                    "SELECT count(*) FROM events WHERE source=? AND status NOT IN"
                    " ('committed','cancelled')", (INCIDENT_SOURCE,)).fetchone()[0],
                "last_error": self.last_error}

    @_guard("unavailable")
    def integrity_check(self) -> str:
        return str(self._conn.execute("PRAGMA integrity_check").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._drop_conn()


# ── per-process singleton ────────────────────────────────────────────────────

_STORES: dict = {}
_STORES_LOCK = threading.Lock()


def get_store(path: Optional[os.PathLike | str] = None) -> EventStore:
    """One EventStore per (resolved path, pid). Never raises."""
    p = Path(path) if path else default_db_path()
    key = (str(p.absolute()), os.getpid())
    with _STORES_LOCK:
        s = _STORES.get(key)
        if s is None:
            s = EventStore(p)
            _STORES[key] = s
        return s
