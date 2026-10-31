"""voice/iris_interrupt_tiers.py - what to DO when someone talks while my mouth is busy.

Full-duplex step (c), 2026-10-31 (V3 spec section 4: "Differentiate a brief acknowledgment, a helpful
interjection, playback duck/pause, full barge-in, and defer-until-safe. Explicit stop commands receive
high priority; avoid interrupting for every ambient sound.")

Today the room barge (wren_voice_core._bargein_accumulate) is binary: ~256 ms of VAD-voiced audio
above an rms floor = stop the mouth, then the host interrupt()s the turn. A "mm-hm" or a cough cuts
me off, and the false-barge restore only puts the reply back if whisper hears NOTHING - a "yeah"
becomes a whole new voice turn. This module is the decision layer that replaces that single
threshold. It is PURE: no audio, no I/O, no clock of its own; the caller feeds observations as they
arrive (every VAD chunk, every live-words partial, the voiceprint verdict, the final transcript).

Tiers (escalate only while he is still talking; EVERY decision before the final transcript is
provisional - the mouth may duck/pause, but only the FINAL observation (full words + the voiceprint
verdict, which needs the whole utterance) commits BARGE vs DEFER vs ACK, and it may resolve down):
  NONE   - nothing to do (not speech, too faint over my own echo residual, my own voice, mouth idle)
  ACK    - a backchannel ("mm-hm", "yeah", "right"): keep talking at full volume, no new turn;
           the words are logged as an acknowledgement
  DUCK   - something is starting and I don't know what yet: drop my volume (~-12 dB) and keep going
  PAUSE  - real speech: stop the mouth but KEEP the rest of the reply (resume if it was nothing)
  BARGE  - he is taking the floor: stop the mouth, drop the rest, interrupt the turn, his words next
  DEFER  - speech that must not steer me right now (an unverified speaker, or a critical line I am
           in the middle of): duck under it, finish, then hand it over as a queued (fenced) turn

Priority rules (in order):
  1. self-voice verdict (voiceprint says the speech is ME) -> NONE, whatever else is true
  2. explicit stop phrase ("stop", "wait", "hold on", "Iris stop", ...) -> BARGE for the owner /
     unchecked speaker, PAUSE for an unverified one (stopping my mouth is always safe; letting a
     stranger steer my cognition is not)
  3. not enough evidence (speech too short, or not loud enough over the predicted echo residual)
     -> NONE (ambient noise, a door, the TV under my voice never trigger)
  4. unverified speaker -> at most DUCK while speaking, DEFER once it is clearly speech
  5. critical line in progress -> at most DUCK, then DEFER
  6. backchannel text -> ACK (a short "yeah" never cuts me off)
  7. addressed to me ("Iris, ...") with real words -> BARGE
  8. duration ladder: >= duck_ms DUCK, >= pause_ms PAUSE, >= barge_ms or a final with >=
     barge_words real words -> BARGE

Everything tunable lives in TierConfig so the daemon can hot-read voice_control.json.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    NONE = 0
    ACK = 1
    DUCK = 2
    DEFER = 3
    PAUSE = 4
    BARGE = 5


# Backchannels: whole-utterance tokens only ("yeah" is an ack; "yeah but wait" is not).
BACKCHANNEL = {
    "mm", "mhm", "mmhm", "mm-hm", "mm-hmm", "mmhmm", "hmm", "hm", "uh-huh", "uhhuh", "uh", "huh",
    "yeah", "yea", "yep", "yup", "yes", "ya", "right", "ok", "okay", "k", "sure", "true", "totally",
    "nice", "cool", "wow", "oh", "ah", "ahh", "aha", "haha", "hah", "ha", "lol", "gotcha", "i see",
    "got it", "makes sense", "exactly", "indeed", "definitely", "for sure", "oh wow", "oh nice",
    "oh okay", "oh ok", "oh yeah", "yeah yeah", "mm yeah", "uh huh", "mm hmm", "mhmm", "alright",
    "all right", "interesting", "oh really", "really", "no way",
}

# Explicit stop: matched as a phrase at the START of what he said (or after my name) - "wait" in
# "I can't wait to see it" must not stop me.
STOP_PHRASES = (
    "stop", "stop it", "stop talking", "please stop", "wait", "wait wait", "hold on", "hang on",
    "hold up", "one sec", "one second", "just a sec", "just a second", "pause", "shush", "shh",
    "quiet", "be quiet", "shut up", "enough", "no no", "no stop", "no wait", "cancel", "never mind",
    "nevermind", "okay stop", "ok stop", "hey stop", "stop stop",
)
WAKE_NAMES = ("iris",)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9' -]+")


def norm(text: str) -> str:
    t = _PUNCT.sub(" ", (text or "").lower().replace("’", "'"))
    t = t.replace("-", " ")
    return _WS.sub(" ", t).strip()


def _strip_name(t: str) -> "tuple[str, bool]":
    for n in WAKE_NAMES:
        for pre in (n + " ", "hey " + n + " ", "okay " + n + " ", "ok " + n + " "):
            if t.startswith(pre):
                return t[len(pre):], True
        if t == n:
            return "", True
    return t, False


def is_backchannel(text: str) -> bool:
    t = norm(text)
    if not t:
        return False
    if t in BACKCHANNEL or t.replace(" ", "-") in BACKCHANNEL or t.replace(" ", "") in BACKCHANNEL:
        return True
    # "yeah yeah yeah", "mm mm", "okay okay": repeats of one backchannel token
    toks = t.split()
    return len(toks) <= 4 and all(w in BACKCHANNEL for w in toks)


def stop_phrase(text: str) -> "str | None":
    """The explicit stop phrase at the start of text (after an optional 'Iris,'), else None."""
    t, _named = _strip_name(norm(text))
    for p in sorted(STOP_PHRASES, key=len, reverse=True):
        if t == p or t.startswith(p + " "):
            return p
    return None


def addressed(text: str) -> bool:
    t = norm(text)
    return _strip_name(t)[1] or any(re.search(r"\b" + n + r"\b", t) for n in WAKE_NAMES)


def real_words(text: str) -> int:
    """Words that are not fillers/backchannel tokens."""
    return sum(1 for w in norm(text).split() if w not in BACKCHANNEL)


@dataclass
class TierConfig:
    onset_ms: int = 120          # voiced audio before anything at all happens (Silero chunk = 32 ms)
    duck_ms: int = 200           # -> DUCK (provisional; ~the old 256 ms barge debounce)
    pause_ms: int = 700          # -> PAUSE (stop the mouth, keep the reply)
    barge_ms: int = 1400         # -> BARGE on duration alone (no transcript needed)
    barge_words: int = 3         # a partial/final with this many real words -> BARGE
    backchannel_max_ms: int = 1300   # a backchannel longer than this is not a backchannel
    min_over_residual_db: float = 6.0  # near-end must beat the predicted echo residual by this much
    min_rms_dbfs: float = -34.0  # absolute floor (== BARGEIN_MIN_RMS 0.02) when no residual estimate
    duck_db: float = -12.0       # how much DUCK lowers my mouth

    @classmethod
    def from_control(cls, ctl: "dict | None") -> "TierConfig":
        """voice_control.json overrides: keys tier_<field> (bad values are ignored)."""
        c = cls()
        for k, v in (ctl or {}).items():
            if k.startswith("tier_") and hasattr(c, k[5:]):
                try:
                    setattr(c, k[5:], type(getattr(c, k[5:]))(v))
                except (TypeError, ValueError):
                    pass
        return c


@dataclass
class Observation:
    """One update. Every field is optional - send what you have when you have it."""
    voiced_ms: "float | None" = None       # continuous voiced duration of THIS overlap so far
    level_dbfs: "float | None" = None      # near-end level of the voiced audio (after AEC)
    residual_dbfs: "float | None" = None   # predicted echo residual level right now (EchoGate)
    text: "str | None" = None              # live partial (or final) transcript of his words
    final: bool = False                    # text is the final transcript; voiced_ms is total
    speaker: "str | None" = None           # 'owner' | 'unknown' | 'self' | None (no verdict yet)
    ended: bool = False                    # end of speech detected (end-silence) without final text


@dataclass
class Decision:
    tier: Tier
    reason: str
    provisional: bool = True               # may still be resolved (escalated or resolved down)
    resume: bool = False                   # PAUSE/DUCK resolved: put the reply back / unduck
    gain_db: float = 0.0                   # mouth gain to apply now (DUCK)
    interrupt_turn: bool = False           # tell the host to interrupt() cognition
    enqueue: bool = False                  # hand his words to cognition as a turn
    fenced: bool = False                   # ... as an UNVERIFIED-speaker (talk-only) turn
    ack_text: "str | None" = None


@dataclass
class OverlapClassifier:
    """One overlap episode (from a VAD onset while I speak until his speech is resolved).

    mouth_speaking / critical are the CONTEXT: set them at construction or between updates.
    update() never de-escalates except through an explicit resolution (backchannel, self-voice,
    empty final), which returns resume=True so the caller can un-duck / restore the reply."""
    cfg: TierConfig = field(default_factory=TierConfig)
    mouth_speaking: bool = True
    critical: bool = False
    tier: Tier = Tier.NONE
    voiced_ms: float = 0.0
    text: str = ""
    speaker: "str | None" = None
    level_dbfs: "float | None" = None
    residual_dbfs: "float | None" = None
    history: list = field(default_factory=list)
    done: bool = False

    # ---------------------------------------------------------------- helpers
    def _loud_enough(self) -> bool:
        if self.level_dbfs is None:
            return True                      # no level info: duration/VAD alone decide
        if self.residual_dbfs is not None:
            return self.level_dbfs >= self.residual_dbfs + self.cfg.min_over_residual_db
        return self.level_dbfs >= self.cfg.min_rms_dbfs

    def _decide(self, tier: Tier, reason: str, *, final: bool = False, **kw) -> Decision:
        resume = kw.pop("resume", False)
        if not resume and tier < self.tier and not final:   # final decisions see everything: no hold
            tier, reason = self.tier, "hold " + self.tier.name + " (" + reason + ")"
        prev = self.tier
        self.tier = tier
        d = Decision(tier=tier, reason=reason, provisional=not final, resume=resume, **kw)
        if tier == Tier.DUCK or (tier == Tier.DEFER and self.mouth_speaking):
            d.gain_db = self.cfg.duck_db
        if tier == Tier.BARGE:
            d.interrupt_turn = True
            d.enqueue = True
        if tier == Tier.DEFER and final:
            d.enqueue = True
            d.fenced = self.speaker not in ("owner", None)
        if final:
            self.done = True
        self.history.append((prev.name, tier.name, reason))
        return d

    # ---------------------------------------------------------------- main entry
    def update(self, obs: Observation) -> Decision:
        if self.done:
            return Decision(tier=self.tier, reason="episode closed", provisional=False)
        if obs.voiced_ms is not None:
            self.voiced_ms = max(self.voiced_ms, float(obs.voiced_ms))
        if obs.text is not None:
            self.text = obs.text
        if obs.speaker is not None:
            self.speaker = obs.speaker
        if obs.level_dbfs is not None:
            self.level_dbfs = obs.level_dbfs
        if obs.residual_dbfs is not None:
            self.residual_dbfs = obs.residual_dbfs
        c = self.cfg
        final = bool(obs.final or obs.ended)
        was_holding = self.tier in (Tier.DUCK, Tier.PAUSE, Tier.DEFER)

        # 0. mouth idle: overlap logic doesn't apply - the normal listen path owns the turn
        if not self.mouth_speaking and self.tier == Tier.NONE:
            return self._decide(Tier.NONE, "mouth idle", final=True)

        # 1. my own voice
        if self.speaker == "self":
            return self._decide(Tier.NONE, "self-voice", final=True, resume=was_holding)

        stop = stop_phrase(self.text) if self.text else None
        owner_ok = self.speaker in ("owner", None)   # None = no verdict yet / voice-id off

        # 2. explicit stop wins over everything but self-voice
        if stop and self._loud_enough():
            if owner_ok:
                return self._decide(Tier.BARGE, "explicit stop: " + repr(stop), final=final)
            return self._decide(Tier.PAUSE, "explicit stop from unverified speaker: " + repr(stop),
                                final=final, enqueue=final, fenced=final)

        # 3. not enough evidence yet / ambient
        if not self._loud_enough():
            if final:
                return self._decide(Tier.NONE, "below echo residual / floor", final=True, resume=was_holding)
            return self._decide(Tier.NONE, "below echo residual / floor")
        if self.voiced_ms < c.onset_ms and not self.text:
            if final:
                return self._decide(Tier.NONE, "too short", final=True, resume=was_holding)
            return self._decide(Tier.NONE, "onset pending")

        # final with nothing intelligible: a cough, a noise - resume
        if final and not norm(self.text):
            return self._decide(Tier.NONE, "no words (noise/cough)", final=True, resume=was_holding)

        # 6 (checked before 4/5: an ack never needs deferring). Backchannel - only short ones.
        if self.text and is_backchannel(self.text) and self.voiced_ms <= c.backchannel_max_ms:
            if final:
                return self._decide(Tier.ACK, "backchannel", final=True, resume=was_holding,
                                    ack_text=self.text)
            # partial says ack - stay ducked (or at NONE) until it ends; never pause for it
            return self._decide(Tier.DUCK if self.voiced_ms >= c.duck_ms else Tier.NONE,
                                "backchannel so far", resume=self.tier > Tier.DUCK)

        # 4. unverified speaker: never steers cognition while I talk
        if self.speaker == "unknown":
            if final or self.voiced_ms >= c.pause_ms:
                return self._decide(Tier.DEFER, "unverified speaker", final=final)
            return self._decide(Tier.DUCK, "unverified speaker, starting")

        # 5. a critical line: duck, then defer
        if self.critical:
            if final or self.voiced_ms >= c.pause_ms:
                return self._decide(Tier.DEFER, "critical line in progress", final=final)
            return self._decide(Tier.DUCK, "critical line, starting")

        nw = real_words(self.text) if self.text else 0
        # 7. addressed to me with real words
        if self.text and addressed(self.text) and nw >= 1:
            return self._decide(Tier.BARGE, "addressed to me", final=final)
        # 8. ladder
        if nw >= c.barge_words or self.voiced_ms >= c.barge_ms:
            return self._decide(Tier.BARGE, f"taking the floor ({nw} words, {int(self.voiced_ms)} ms)",
                                final=final)
        if final:
            # short but real words (1-2, not a backchannel): a brief interjection - barge so it is
            # answered, unless it is a lone filler word
            if nw >= 1:
                return self._decide(Tier.BARGE, f"short interjection ({nw} words)", final=True)
            return self._decide(Tier.ACK, "filler only", final=True, resume=was_holding, ack_text=self.text)
        if self.voiced_ms >= c.pause_ms:
            return self._decide(Tier.PAUSE, f"speech {int(self.voiced_ms)} ms")
        if self.voiced_ms >= c.duck_ms:
            return self._decide(Tier.DUCK, f"speech {int(self.voiced_ms)} ms")
        return self._decide(Tier.NONE, "onset")
