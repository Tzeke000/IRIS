"""shm_ring — single-writer / many-reader frame ring in /dev/shm (eyes-process prototype).

Design: docs/eyes_process_design.md §4.2. The eyes process is the ONLY writer; the
runtime (orb latest_frame, scene_memory keyframes, study_face, servo) are readers.

Why a raw mmap of a /dev/shm file and not multiprocessing.shared_memory:
  * Python <3.13's resource_tracker registers every ATTACH and unlinks the segment
    when an attaching process exits (bpo-39959) — a reader dying would delete the
    writer's ring. A plain file in /dev/shm has no tracker.
  * Readers map it PROT_READ: a consumer bug cannot scribble on the frames.

Layout (little-endian, all fields 8-byte aligned):

  [0, 128)           ring header
  [128 + i*SLOT_HDR, ...)   per-slot headers (NSLOTS of them)
  [DATA_OFF + i*slot_bytes]  per-slot pixel data

Ring header:
  magic(8s) version(u32) nslots(u32) slot_bytes(u64) latest_seq(u64)
  writer_pid(u64) heartbeat_ts(f64) state(u32) pad(u32) started_ts(f64)

Slot header (one per slot, a SEQLOCK):
  lock(u64)  — even = stable, odd = writer mid-copy. Bumped twice per write.
  seq(u64)   — frame sequence number stored in this slot (1-based, monotonic)
  capture_ts(f64) pub_ts(f64) h(u32) w(u32) c(u32) pad(u32) nbytes(u64)

Correctness: the writer bumps `lock` to odd, writes meta + pixels, bumps `lock` to
even, THEN publishes `latest_seq`. A reader snapshots lock (must be even), copies,
re-reads lock; any change ⇒ the slot was overwritten mid-copy ⇒ retry/newer slot.
This relies on store→store and load→load ordering, which x86-64 (TSO) guarantees;
the server is x86-64. (On a weakly-ordered CPU this would need real fences.)

Staleness: every frame carries capture_ts; the header carries a writer heartbeat
that keeps ticking while the eyes are PARKED (state=2), so a reader can tell
"eyes alive but parked" from "eyes process dead" from "frames stopped arriving".
"""
from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAGIC = b"IRISEYE1"
VERSION = 1
HDR_SIZE = 128
SLOT_HDR = 64
_HDR_FMT = "<8sIIQQQdIId"          # magic ver nslots slot_bytes latest writer_pid hb state pad started
_SLOT_FMT = "<QQddIIIIQ"           # lock seq capture_ts pub_ts h w c pad nbytes
_OFF_LATEST = 24
_OFF_PID = 32
_OFF_HB = 40
_OFF_STATE = 48
_OFF_STARTED = 56

STATE_STARTING, STATE_STREAMING, STATE_PARKED, STATE_STOPPED = 0, 1, 2, 3
STATE_NAMES = {0: "starting", 1: "streaming", 2: "parked", 3: "stopped"}

# Freshness bands — same numbers brain/frame_store.py uses, so the orb's
# fresh/aging/stale labels mean the same thing across the process boundary.
FRESH_MAX_AGE_SEC = 0.35
AGING_MAX_AGE_SEC = 1.20
# Writer heartbeat older than this ⇒ the eyes process is gone (not just parked).
WRITER_DEAD_SEC = 2.0

DEFAULT_NAME = "iris_eyes_frames"
DEFAULT_NSLOTS = 4
DEFAULT_SLOT_BYTES = 1280 * 720 * 3


def shm_path(name: str) -> str:
    if "/" in name:
        raise ValueError("ring name must be a bare name, not a path")
    return f"/dev/shm/{name}"


def _data_off(nslots: int) -> int:
    return HDR_SIZE + nslots * SLOT_HDR


def classify(age_sec: Optional[float]) -> str:
    if age_sec is None or age_sec < 0:
        return "unavailable"
    if age_sec <= FRESH_MAX_AGE_SEC:
        return "fresh"
    if age_sec <= AGING_MAX_AGE_SEC:
        return "aging"
    return "stale"


@dataclass
class FrameMeta:
    seq: int
    capture_ts: float
    pub_ts: float
    shape: tuple
    age_sec: float
    freshness: str


class RingWriter:
    """The eyes process's half. Not thread-safe: exactly ONE thread writes."""

    def __init__(self, name: str = DEFAULT_NAME, nslots: int = DEFAULT_NSLOTS,
                 slot_bytes: int = DEFAULT_SLOT_BYTES) -> None:
        if nslots < 2:
            raise ValueError("need >=2 slots so readers can copy while the writer moves on")
        self.name, self.nslots, self.slot_bytes = name, nslots, slot_bytes
        self.path = shm_path(name)
        size = _data_off(nslots) + nslots * slot_bytes
        # O_CREAT|O_TRUNC: a crashed predecessor's ring is replaced, never trusted.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, size)
            self._mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self._buf = memoryview(self._mm)
        self._seq = 0
        now = time.time()
        struct.pack_into(_HDR_FMT, self._buf, 0, MAGIC, VERSION, nslots, slot_bytes,
                         0, os.getpid(), now, STATE_STARTING, 0, now)

    # -- header fields -----------------------------------------------------
    def heartbeat(self, state: Optional[int] = None) -> None:
        struct.pack_into("<d", self._buf, _OFF_HB, time.time())
        if state is not None:
            struct.pack_into("<I", self._buf, _OFF_STATE, int(state))

    def set_state(self, state: int) -> None:
        self.heartbeat(state)

    # -- frames ------------------------------------------------------------
    def write(self, frame: np.ndarray, capture_ts: float) -> int:
        """Copy one frame into the next slot. Returns its seq."""
        if frame.dtype != np.uint8:
            raise TypeError("frames are uint8")
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        nbytes = frame.nbytes
        if nbytes > self.slot_bytes:
            raise ValueError(f"frame {frame.shape} = {nbytes}B exceeds slot {self.slot_bytes}B")
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim == 3 else 1
        seq = self._seq + 1
        i = seq % self.nslots
        sh = HDR_SIZE + i * SLOT_HDR
        lock = struct.unpack_from("<Q", self._buf, sh)[0]
        struct.pack_into("<Q", self._buf, sh, lock + 1)                 # odd: writing
        d0 = _data_off(self.nslots) + i * self.slot_bytes
        dst = np.frombuffer(self._buf, dtype=np.uint8, count=nbytes, offset=d0)
        np.copyto(dst, frame.reshape(-1))
        pub = time.time()
        struct.pack_into(_SLOT_FMT, self._buf, sh, lock + 1, seq, float(capture_ts), pub,
                         h, w, c, 0, nbytes)
        struct.pack_into("<Q", self._buf, sh, lock + 2)                 # even: stable
        struct.pack_into("<Q", self._buf, _OFF_LATEST, seq)             # publish
        struct.pack_into("<d", self._buf, _OFF_HB, pub)
        self._seq = seq
        return seq

    def close(self, unlink: bool = True) -> None:
        try:
            struct.pack_into("<I", self._buf, _OFF_STATE, STATE_STOPPED)
        except Exception:
            pass
        self._buf.release()
        self._mm.close()
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


class RingReader:
    """A consumer's half. Read-only mapping; safe to construct before the writer
    exists (attach() retries lazily)."""

    def __init__(self, name: str = DEFAULT_NAME) -> None:
        self.name = name
        self.path = shm_path(name)
        self._mm = None
        self._buf = None
        self._ino = None
        self.nslots = 0
        self.slot_bytes = 0
        self.retries = 0      # torn-read retries (observability; should stay ~0)

    def attach(self) -> bool:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            self._detach()
            return False
        if self._mm is not None and st.st_ino == self._ino:
            return True
        # A restarted writer O_TRUNC-recreates the file ⇒ new inode ⇒ remap.
        self._detach()
        if st.st_size < HDR_SIZE:
            return False
        fd = os.open(self.path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, st.st_size, mmap.MAP_SHARED, mmap.PROT_READ)
        finally:
            os.close(fd)
        buf = memoryview(mm)
        magic, ver, nslots, slot_bytes = struct.unpack_from("<8sIIQ", buf, 0)
        if magic != MAGIC or ver != VERSION or st.st_size < _data_off(nslots) + nslots * slot_bytes:
            buf.release()
            mm.close()
            return False
        self._mm, self._buf, self._ino = mm, buf, st.st_ino
        self.nslots, self.slot_bytes = nslots, slot_bytes
        return True

    def _detach(self) -> None:
        if self._buf is not None:
            self._buf.release()
        if self._mm is not None:
            self._mm.close()
        self._mm = self._buf = self._ino = None

    close = _detach

    def writer_status(self) -> dict:
        """Is anyone home? {alive, state, heartbeat_age_s, latest_seq, writer_pid}."""
        if not self.attach():
            return {"alive": False, "state": "absent", "heartbeat_age_s": None,
                    "latest_seq": 0, "writer_pid": None}
        latest, pid, hb, state = struct.unpack_from("<QQdI", self._buf, _OFF_LATEST)
        age = max(0.0, time.time() - hb)
        sname = STATE_NAMES.get(state, "?")
        alive = age <= WRITER_DEAD_SEC and state != STATE_STOPPED
        return {"alive": alive, "state": sname if alive else ("stopped" if state == STATE_STOPPED else "dead"),
                "heartbeat_age_s": round(age, 3), "latest_seq": latest, "writer_pid": pid}

    def latest_seq(self) -> int:
        if not self.attach():
            return 0
        return struct.unpack_from("<Q", self._buf, _OFF_LATEST)[0]

    def read_latest(self, out: Optional[np.ndarray] = None, *, max_age_sec: Optional[float] = None,
                    newer_than: int = 0, max_tries: int = 8):
        """Copy the newest stable frame. Returns (frame, FrameMeta) or (None, reason_str).

        out: optional preallocated uint8 buffer (>= frame size) to copy into — a
             consumer polling at 30 Hz should reuse one instead of allocating 2.7 MB
             per read (allocation churn is the very thing this split is fighting).
        max_age_sec: frames older than this are refused (reason 'stale').
        newer_than: refuse a frame whose seq <= this (reason 'no_new_frame') —
             the cross-process version of person_track's stale-frame guard.
        """
        if not self.attach():
            return None, "absent"
        for _ in range(max_tries):
            seq = struct.unpack_from("<Q", self._buf, _OFF_LATEST)[0]
            if seq == 0:
                return None, "no_frame_yet"
            if seq <= newer_than:
                return None, "no_new_frame"
            i = seq % self.nslots
            sh = HDR_SIZE + i * SLOT_HDR
            lock1, sseq, cts, pts, h, w, c, _pad, nbytes = struct.unpack_from(_SLOT_FMT, self._buf, sh)
            if lock1 & 1 or sseq != seq:
                self.retries += 1          # writer is mid-copy / already lapped this slot
                continue
            if nbytes > self.slot_bytes or nbytes != h * w * c:
                self.retries += 1
                continue
            age = time.time() - cts
            if max_age_sec is not None and age > max_age_sec:
                return None, "stale"
            d0 = _data_off(self.nslots) + i * self.slot_bytes
            src = np.frombuffer(self._buf, dtype=np.uint8, count=nbytes, offset=d0)
            if out is not None and out.nbytes >= nbytes and out.dtype == np.uint8:
                dst = out.reshape(-1)[:nbytes]
                np.copyto(dst, src)
            else:
                dst = src.copy()
            lock2 = struct.unpack_from("<Q", self._buf, sh)[0]
            if lock2 != lock1:
                self.retries += 1          # overwritten during our copy — discard
                continue
            shape = (h, w, c) if c > 1 else (h, w)
            meta = FrameMeta(seq=seq, capture_ts=cts, pub_ts=pts, shape=shape,
                             age_sec=age, freshness=classify(age))
            return dst.reshape(shape), meta
        return None, "contended"
