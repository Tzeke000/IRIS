"""voice/iris_duplex_watch.py - the tiered barge-in watch: VAD frames -> tier decisions -> mouth actions.

Full-duplex step (c) wiring, 2026-10-31. wren_voice_core's binary watch (_bargein_accumulate: ~256 ms
voiced = stop the mouth) stays the default. When the HOST asks for tiers (bargein_watch args
{"tiers": true}, host flag IRIS_DUPLEX_TIERS) the daemon runs tiered_accumulate() instead, which feeds
voice/iris_interrupt_tiers.OverlapClassifier every 32 ms and applies its decisions through
MouthActions: duck my volume while it is unclear, pause (stop + keep the reply) on real speech, barge
on a stop word / the floor being taken, and RESUME on a backchannel, a cough or my own echo.

Pure: no audio device, no model, no ctx. The daemon passes callables. Tests:
tests/test_duplex_watch.py.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

try:
    from iris_interrupt_tiers import Observation, OverlapClassifier, Tier, TierConfig
except Exception:  # noqa: BLE001
    from .iris_interrupt_tiers import Observation, OverlapClassifier, Tier, TierConfig  # type: ignore

CHUNK_MS = 32.0                 # one Silero chunk (512 samples @ 16 kHz)
DUCK_REFRESH_S = 1.0            # the mouth's duck has a TTL; refresh while still ducked


def _dbfs(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20.0 * float(np.log10(max(r, 1e-9)))


class MouthActions:
    """Applies tier decisions to the mouth idempotently. state: normal | ducked | paused."""

    def __init__(self, duck=None, unduck=None, pause=None, resume=None, log=None):
        self._duck, self._unduck = duck, unduck
        self._pause, self._resume = pause, resume
        self._log = log or (lambda m: None)
        self.state = "normal"
        self._duck_ts = 0.0
        self.calls: list = []        # (action, tier) for tests / the episode log

    def _call(self, fn, name, *a):
        self.calls.append(name)
        if fn is None:
            return
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001  (a failed duck must never break the watch)
            self._log(f"mouth action {name} failed: {e!r}")

    def apply(self, d) -> None:
        if d.resume:
            if self.state == "paused":
                self._call(self._resume, "resume")
            elif self.state == "ducked":
                self._call(self._unduck, "unduck")
            self.state = "normal"
            return
        if d.tier in (Tier.PAUSE, Tier.BARGE):
            if self.state != "paused":
                if self.state == "ducked":
                    self._call(self._unduck, "unduck")      # resumed audio must come back at full volume
                self._call(self._pause, "pause")
                self.state = "paused"
            return
        if d.gain_db < 0 and self.state != "paused":
            now = time.monotonic()
            if self.state != "ducked" or now - self._duck_ts > DUCK_REFRESH_S:
                self._call(self._duck, "duck", d.gain_db)
                self._duck_ts = now
                self.state = "ducked"

    def settle(self, d) -> None:
        """Episode over: leave the mouth in the state the FINAL decision implies."""
        if d.tier in (Tier.BARGE,) or (d.tier == Tier.PAUSE and not d.resume):
            if self.state != "paused":
                self.apply(d)
            return
        # NONE / ACK / DEFER: I keep (or go back to) talking at full volume
        if self.state == "paused":
            self._call(self._resume, "resume")
        elif self.state == "ducked":
            self._call(self._unduck, "unduck")
        self.state = "normal"


def tiered_accumulate(frame_prob_iter, is_speaking, actions: MouthActions, *,
                      cfg: "TierConfig | None" = None, critical: bool = False,
                      threshold: float = 0.5, preroll_chunks: int = 31,
                      end_silence_chunks: int = 19, max_chunks: int = 625,
                      min_rms: float = 0.0, residual_fn=None, text_fn=None,
                      on_capture=None, on_episode=None, stats: "dict | None" = None):
    """Watch while I speak; run ONE overlap episode through the classifier.

    frame_prob_iter yields (float32 chunk, speech prob). is_speaking() False with no episode -> None.
    residual_fn() -> predicted echo residual dBFS or None. text_fn() -> latest live partial or None.
    on_episode(clf) is called once when an episode starts (the caller creates its LiveWords there).

    Returns (audio | None, classifier | None). audio is None when nothing needs a transcript: no
    episode, or an episode resolved to NONE before end-of-speech (noise/too faint/too short) - in
    that case the watch simply continues (episodes that fizzle never end the window)."""
    cfg = cfg or TierConfig()
    onset_chunks = max(1, int(round(cfg.onset_ms / CHUNK_MS)))
    preroll = deque(maxlen=max(preroll_chunks, onset_chunks))
    run = 0
    clf = None
    collected: list = []
    voiced_lvls: list = []
    voiced_ms = 0.0
    silence = 0

    def _tap(c):
        if on_capture is not None:
            try:
                on_capture(c)
            except Exception:
                pass

    for f, prob in frame_prob_iter:
        if clf is None:
            if not is_speaking():
                return None, None
            preroll.append(f)
            v = prob >= threshold
            if v and min_rms > 0.0 and float(np.sqrt(np.mean(f * f))) < min_rms:
                v = False
                if stats is not None:
                    stats["faint_voiced"] = stats.get("faint_voiced", 0) + 1
            run = run + 1 if v else 0
            if run >= onset_chunks:
                clf = OverlapClassifier(cfg=cfg, mouth_speaking=True, critical=critical)
                collected = list(preroll)
                voiced_lvls = [_dbfs(c) for c in list(preroll)[-run:]]
                voiced_ms = run * CHUNK_MS
                silence = 0
                if on_episode is not None:
                    try:
                        on_episode(clf)
                    except Exception:
                        pass
                for c in collected:
                    _tap(c)
                d = clf.update(Observation(voiced_ms=voiced_ms, level_dbfs=float(np.mean(voiced_lvls)),
                                           residual_dbfs=residual_fn() if residual_fn else None))
                actions.apply(d)
                if stats is not None:
                    stats["episodes"] = stats.get("episodes", 0) + 1
            continue
        # ---- inside an episode
        collected.append(f)
        _tap(f)
        if prob >= threshold:
            voiced_ms += CHUNK_MS
            voiced_lvls.append(_dbfs(f))
            silence = 0
        else:
            silence += 1
        clf.mouth_speaking = bool(is_speaking()) or actions.state == "paused"
        txt = text_fn() if text_fn else None
        d = clf.update(Observation(voiced_ms=voiced_ms,
                                   level_dbfs=float(np.mean(voiced_lvls[-30:])) if voiced_lvls else None,
                                   residual_dbfs=residual_fn() if residual_fn else None,
                                   text=txt or None))
        actions.apply(d)
        if clf.done:
            # decided before end-of-speech (explicit stop / addressed / taking the floor): keep
            # capturing his words to the end, the decision itself stands
            pass
        if silence >= end_silence_chunks or len(collected) >= max_chunks:
            if clf.tier == Tier.NONE and not clf.done:
                # fizzled: never escalated past NONE - not worth a transcript; reset and keep watching
                d = clf.update(Observation(ended=True))
                actions.settle(d)
                if stats is not None:
                    stats["fizzled"] = stats.get("fizzled", 0) + 1
                clf, collected, voiced_lvls, run, preroll = None, [], [], 0, deque(maxlen=preroll.maxlen)
                continue
            return np.concatenate(collected), clf
    if clf is not None and collected:
        return np.concatenate(collected), clf
    return None, None


def speaker_class(verdict: "dict | None", self_voice: bool, owner_names=("zeke",)) -> "str | None":
    """Daemon voice-id verdict -> classifier speaker: 'self' | 'owner' | 'unknown' | None (no verdict)."""
    if self_voice:
        return "self"
    v = verdict or {}
    st = v.get("status")
    if st == "matched":
        return "owner" if str(v.get("speaker") or "").lower() in owner_names else "unknown"
    if st == "unknown":
        return "unknown"
    return None


def final_marker(d) -> str:
    """The return-string PREFIX for a final decision. The host's contract:
    '[barge-in]' -> his turn, interrupt mine; '[barge-defer]' -> his turn, do NOT interrupt mine;
    '[bargein_watch] ...' -> no turn (ack / noise / self), just a closed window."""
    if d.tier == Tier.BARGE:
        return "[barge-in]"
    if d.enqueue:
        return "[barge-defer]"
    if d.tier == Tier.ACK:
        return "[bargein_watch] ack"
    return "[bargein_watch] no_turn"


class MouthDucker:
    """Mouth-side volume duck (the DUCK tier). set(db, ttl_s) from the /duck endpoint; apply(block)
    on every block written to the speaker. Gain ramps linearly over ramp_ms (no clicks) and a duck
    EXPIRES after ttl_s unless refreshed - the 10-08 lesson (a barge mute that outlived its window
    silenced a whole tour): a lost un-duck must never leave me quiet. Thread-safe enough for one
    writer thread + HTTP setters (plain float assignment)."""

    def __init__(self, sr: int, ramp_ms: float = 30.0, floor_db: float = -40.0, max_ttl_s: float = 10.0):
        self.sr = int(sr)
        self._ramp = max(1, int(sr * ramp_ms / 1000.0))
        self._floor_db = float(floor_db)
        self._max_ttl = float(max_ttl_s)
        self._target_db = 0.0
        self._until = 0.0
        self._g = 1.0

    def set(self, db: float, ttl_s: float = 3.0, now: "float | None" = None) -> dict:
        now = time.monotonic() if now is None else now
        db = max(self._floor_db, min(0.0, float(db)))
        self._target_db = db
        self._until = now + max(0.1, min(self._max_ttl, float(ttl_s))) if db < 0 else 0.0
        return {"db": db, "ttl_s": round(self._until - now, 2) if db < 0 else 0.0}

    def target_gain(self, now: "float | None" = None) -> float:
        now = time.monotonic() if now is None else now
        if self._target_db < 0 and now >= self._until:
            self._target_db = 0.0                    # expired
        return float(10.0 ** (self._target_db / 20.0))

    def apply(self, blk: np.ndarray, now: "float | None" = None) -> np.ndarray:
        tgt = self.target_gain(now)
        if self._g == 1.0 and tgt == 1.0:
            return blk                               # the common case: untouched, zero cost
        n = blk.shape[0]
        frac = np.minimum(np.arange(1, n + 1, dtype=np.float32) / self._ramp, 1.0)
        ramp = (self._g + (tgt - self._g) * frac).astype(np.float32)
        self._g = tgt if n >= self._ramp else float(ramp[-1])
        return (blk * (ramp if blk.ndim == 1 else ramp[:, None])).astype(np.float32)
