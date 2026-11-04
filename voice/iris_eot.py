"""Semantic end-of-turn judge (Zeke 11-02, built 11-04 on eot-1104).

The ears end Zeke's turn on ACOUSTICS (VAD end-silence + smart-turn + the grace window). A drawn-out
"is it..." mid-question got cut off: the waveform looked finished, the words did not. This module
reads the transcript and says whether the SENTENCE is finished; the daemon combines that with the
acoustic verdict to stretch or shorten the grace window (see plan_grace and wren_voice_core Step 4).

Two layers, CPU only, no GPU:
  1. lexical_verdict(text) - deterministic rules, ~0.05 ms. Primary. Knows Zeke-style trailing
     ("is it", "and uh", "do you currently..."), open question frames, dangling conjunctions /
     prepositions / articles, and short complete answers ("yes", "no", "okay", "thanks").
  2. ModelJudge - OPTIONAL tie-breaker for what the rules call "unsure": Namo-Turn-Detector-v1-English
     (videosdk-live, Apache-2.0, DistilBERT, fp32 ONNX ~540 MB on disk, ~5-12 ms on 2 CPU threads).
     Measured 11-04: their model_quant.onnx is BROKEN (constant logits ~[0.84,-1.01] for every input -
     use model.onnx), and the fp32 model is overconfident and wrong on short answers ("no", "okay",
     "that's all", an unpunctuated "what time is it" all score ~0.000 finished). So it is never trusted
     alone: it only runs on lexically-unsure text of >= MODEL_MIN_WORDS words and can only move the
     verdict into the MEDIUM bands, never the strong ones that override the acoustic signal.
     (LiveKit's turn-detector was checked first: its licence forbids use outside LiveKit Agents.)

Verdict.p_finished bands (the contract plan_grace reads):
  P_UNFINISHED_STRONG 0.05  dangling word / explicit continuation mark  -> grace + the full extension
  P_UNFINISHED_MED    0.20  open frame, aux tail, model says unfinished  -> grace + half the extension
  P_UNSURE            0.50  no opinion                                   -> legacy behaviour exactly
  P_FINISHED_MED      0.75  sentence-final '.', model says finished      -> legacy behaviour exactly
  P_FINISHED_STRONG   0.95  '?'/'!' close, short complete answer        -> end promptly even on HOLD

Pure module: no I/O at import, no daemon imports - unit-testable and hot-reloadable.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field

P_UNFINISHED_STRONG = 0.05
P_UNFINISHED_MED = 0.20
P_UNSURE = 0.50
P_FINISHED_MED = 0.75
P_FINISHED_STRONG = 0.95

UNFINISHED_BAR = 0.35        # p <= this -> "unfinished"
FINISHED_STRONG_BAR = 0.90   # p >= this -> "finished" strongly enough to override an acoustic HOLD

DEFAULT_MAX_EXTEND_S = 1.5   # Zeke's bound: an unfinished-looking turn waits at most this much MORE
DEFAULT_FINISHED_GRACE_S = 0.3   # clearly-finished text + acoustic HOLD -> this short a grace (0 = none)
MODEL_MIN_WORDS = 3          # the model never judges 1-2 word utterances (it gets "no"/"okay" wrong)
MODEL_UNFINISHED_P = 0.02    # model p_finished at/below this -> medium unfinished
MODEL_FINISHED_P = 0.98      # model p_finished at/above this -> medium finished

_ELLIPSIS = ("...", "…", "..")
_DASHES = ("-", "—", "–")

# hesitations - the turn is still being built
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mmm"}
# conjunctions / subordinators - a clause is still coming
CONJ = {"and", "but", "or", "nor", "because", "cause", "cuz", "since", "although", "while", "plus",
        "if", "unless", "until", "whether", "whereas", "when", "where", "which", "who", "whose",
        "like", "than"}
# articles / determiners / possessives - a noun is still coming
DETERMINERS = {"the", "a", "an", "my", "your", "our", "their", "his", "its", "this", "these",
               "those", "every", "each", "some", "any", "no", "another", "such"}
# prepositions that almost never end an English clause outside a wh-question
PREPS = {"of", "to", "for", "with", "about", "from", "at", "by", "into", "onto", "than", "through",
         "toward", "towards", "between", "during", "without", "within", "upon", "via", "per"}
# particles: "turn it on", "log in", "check it out" end clauses all the time -> only weak signals
PARTICLES = {"on", "off", "in", "out", "up", "down", "over", "back", "around", "away"}
# contracted subjects / intent verbs that dangle
DANGLE_HEADS = {"gonna", "wanna", "gotta", "i'm", "you're", "we're", "they're", "it's", "that's",
                "there's", "here's", "what's", "i'd", "i'll", "we'll", "you'll", "let's", "i've"}
SUBJECTS = {"i", "you", "we", "they", "he", "she", "it", "that", "there", "this"}
WH = {"what", "who", "where", "when", "why", "how", "which", "what's", "who's", "where's", "how's",
      "when's", "why's", "whats", "whos", "wheres", "whose", "whom"}
AUX = {"is", "are", "was", "were", "am", "be", "been", "being", "do", "does", "did", "have", "has",
       "had", "can", "could", "should", "would", "will", "shall", "may", "might", "must"}
# open question / statement frames: the WHOLE final clause is one of these -> the complement is missing
OPEN_FRAMES = {
    "is it", "is that", "is this", "is there", "is he", "is she", "are you", "are they", "are we",
    "are there", "was it", "was that", "were you", "do you", "does it", "does that", "did you",
    "did it", "can you", "can i", "can we", "could you", "could i", "would you", "will you",
    "should i", "should we", "have you", "has it", "do you know", "do you think", "do you remember",
    "do you currently", "can you tell me", "what if", "what about", "how about", "i think",
    "i mean", "i want", "i need", "i was", "i wanted", "i guess", "i feel like", "you know what",
    "the thing is", "my question is", "the question is", "i was thinking", "i was wondering",
    "i don't", "i didn't", "i can't", "it's like", "so basically", "basically", "what is",
    "what are", "what was", "where is", "where are", "how do", "how does", "how did", "how is",
    "why is", "why did", "why does", "who is", "which one is", "tell me", "let me", "i'm gonna",
    "we should", "you should", "maybe we", "let's see",
}
# whole-utterance short answers / commands / closers that ARE a complete turn
SHORT_COMPLETE = {
    "yes", "yeah", "yep", "yup", "yea", "ya", "no", "nope", "nah", "okay", "ok", "k", "sure", "thanks",
    "thank you", "thank you so much", "right", "correct", "exactly", "perfect", "great", "cool",
    "nice", "awesome", "stop", "wait", "go ahead", "never mind", "nevermind", "got it", "sounds good",
    "of course", "absolutely", "definitely", "maybe", "i don't know", "i dunno", "good night",
    "goodnight", "good morning", "hello", "hi", "hey", "hey iris", "hi iris", "bye", "alright",
    "all right", "uh huh", "uh-huh", "mm hmm", "mhm", "yes please", "no thanks", "no thank you",
    "do it", "go for it", "love you", "i love you", "me too", "same", "true", "fair", "fair enough",
    "nothing", "not really", "not yet", "kind of", "sort of", "i guess", "i think so", "yes ma'am",
    "that's it", "that's all", "that's right", "that's fine", "that's good", "good", "fine", "done",
    "go", "continue", "keep going", "pause", "resume", "repeat that", "say that again", "what",
    "huh", "really", "seriously", "oh", "wow", "ok cool", "okay cool", "please",
}
# endings that LOOK dangling but close a sentence ("I think so", "I like that", "let's do it then")
COMPLETE_TAILS = {
    ("think", "so"), ("guess", "so"), ("hope", "so"), ("believe", "so"), ("even", "so"),
    ("or", "so"), ("said", "so"), ("is", "so"), ("not", "so"), ("so", "so"), ("that", "so"),
    ("and", "stuff"), ("or", "something"), ("and", "whatnot"), ("or", "whatever"),
    ("like", "that"), ("do", "that"), ("about", "that"),
    ("of", "that"), ("with", "that"), ("on", "that"), ("into", "that"), ("for", "that"),
    ("than", "that"), ("at", "that"), ("see", "that"), ("try", "that"), ("hear", "that"),
}
# after these a final "that" is a complementizer ("I think that ...") -> unfinished
THAT_COMPLEMENTIZERS = {"think", "said", "say", "know", "believe", "mean", "sure", "hope", "guess",
                        "realize", "feel", "saying", "thought", "knew", "so", "such", "noticed",
                        "thinking", "means", "told", "glad", "worried", "hoping", "figured"}
OBJECT_PRONOUNS = {"it", "them", "me", "him", "her", "us", "you", "that", "this", "everything",
                   "something", "anything", "nothing", "one"}

_WORD_RE = re.compile(r"[a-z0-9']+")
_CLAUSE_SPLIT = re.compile(r"[.?!;:]+\s+|,\s*")


@dataclass
class Verdict:
    label: str                 # "unfinished" | "unsure" | "finished"
    p_finished: float          # band value (see module doc)
    reason: str                # which rule fired - goes in the decision log
    source: str = "lexical"    # "lexical" | "model" | "empty"
    ms: float = 0.0            # judge latency
    model_p: "float | None" = None
    extra: dict = field(default_factory=dict)


def _norm_words(text: str) -> list:
    return _WORD_RE.findall((text or "").lower().replace("’", "'"))


def _final_clause(text: str) -> str:
    """The text after the last clause boundary that has words after it ("Thinking. My question is,
    is it" -> "is it"). A trailing boundary (the final '?') does not count as a split."""
    t = (text or "").strip()
    parts = [p for p in _CLAUSE_SPLIT.split(t) if _norm_words(p)]
    return parts[-1] if parts else t


def _strip_lead(words: list) -> list:
    """Drop leading fillers / discourse markers so "so uh is it" reads as the frame "is it"."""
    lead = FILLERS | {"so", "and", "but", "well", "okay", "ok", "alright", "like", "hey", "iris",
                      "yeah", "oh", "now", "then", "also", "or"}
    i = 0
    while i < len(words) - 1 and words[i] in lead:
        i += 1
    return words[i:]


def lexical_verdict(text: str) -> Verdict:
    """Rules over the transcript. Returns a Verdict; never raises."""
    raw = (text or "").strip()
    words = _norm_words(raw)
    if not words:
        return Verdict("unsure", P_UNSURE, "empty", source="empty")
    tail = raw.rstrip()
    closed = tail.endswith(("?", "!"))   # the ASR heard a question/exclamation close
    last = words[-1]
    prev = words[-2] if len(words) >= 2 else ""

    # 1. explicit continuation marks the ASR kept ("do you currently...", "is it -")
    if tail.endswith(_ELLIPSIS):
        return Verdict("unfinished", P_UNFINISHED_STRONG, "trailing-ellipsis")
    if tail.endswith(_DASHES) or tail.endswith(","):
        return Verdict("unfinished", P_UNFINISHED_STRONG, "trailing-comma-or-dash")

    # 2. the whole utterance is a short complete answer / command
    whole = " ".join(words)
    if whole in SHORT_COMPLETE or " ".join(_strip_lead(words)) in SHORT_COMPLETE and len(words) <= 4:
        return Verdict("finished", P_FINISHED_STRONG, f"short-complete:{whole}")

    # 3. dangling final word (strongest lexical cue)
    if last in FILLERS:
        return Verdict("unfinished", P_UNFINISHED_STRONG, f"filler-tail:{last}")
    if (prev, last) in COMPLETE_TAILS:
        pass  # "I think so", "like that" - fall through to the closing checks
    elif last == "that":
        if prev in THAT_COMPLEMENTIZERS:
            return Verdict("unfinished", P_UNFINISHED_STRONG, f"complementizer-that:{prev}")
        if prev in {"is", "was"} and len(words) >= 3 and words[0] not in WH and not closed:
            # "the reason is that" / "the problem was that" - but "what is that" asks a question
            return Verdict("unfinished", P_UNFINISHED_MED, f"copula-that:{prev}")
    elif last == "so":
        if prev in {"and", "but", "or"} or len(words) > 2:
            return Verdict("unfinished", P_UNFINISHED_MED, "so-tail")
    elif last == "then":
        if prev in {"and", "but", "or", "so"}:
            return Verdict("unfinished", P_UNFINISHED_STRONG, f"{prev}-then")
    elif last in CONJ:
        # "what is it like" / "more than" / "who" in "guess who" - stranded in a question = weaker
        p = P_UNFINISHED_MED if (last in {"like", "who", "which", "when", "where"}) else P_UNFINISHED_STRONG
        return Verdict("unfinished", p, f"conj-tail:{last}")
    elif last in DETERMINERS:
        if not (last in {"this", "these", "those", "some", "any", "no", "another", "such"}
                and len(words) >= 2):
            return Verdict("unfinished", P_UNFINISHED_STRONG, f"determiner-tail:{last}")
        # "look at this" / "I want some" end clauses; only a weak signal
        if not closed:
            return Verdict("unsure", P_UNSURE, f"demonstrative-tail:{last}")
    elif last in PREPS:
        clause_words = _strip_lead(_norm_words(_final_clause(raw)))
        if clause_words and clause_words[0] in WH:
            # preposition stranding in a wh-question: "who are you talking to", "what's it about"
            if not closed:
                return Verdict("unsure", P_UNSURE, f"wh-stranded:{last}")
        else:
            return Verdict("unfinished", P_UNFINISHED_STRONG, f"prep-tail:{last}")
    elif last in PARTICLES:
        if prev in OBJECT_PRONOUNS or prev in AUX or closed or tail.endswith("."):
            pass  # "turn it on", "check it out", "what's up?", "Turn the lights off." - complete
        else:
            return Verdict("unsure", P_UNSURE, f"particle-tail:{last}")
    elif last in DANGLE_HEADS:
        return Verdict("unfinished", P_UNFINISHED_STRONG, f"dangling-head:{last}")

    # 4. open frame: the final clause is ONLY a question/statement opener ("is it", "my question is")
    clause = _strip_lead(_norm_words(_final_clause(raw)))
    clause_s = " ".join(clause)
    if clause_s in OPEN_FRAMES and not closed:
        return Verdict("unfinished", P_UNFINISHED_MED, f"open-frame:{clause_s}")
    if closed and clause_s in OPEN_FRAMES and len(clause) <= 2:
        # "is it?" alone can be a real question ("Is it?") - but in Zeke's drawn-out case the ASR
        # often writes the '?' anyway. Medium: give him the extra beat, not the full extension.
        return Verdict("unfinished", P_UNFINISHED_MED, f"open-frame-q:{clause_s}")

    # 5. auxiliary tail right after a wh-word / subject: "what is", "where are", "I was", "you can"
    if last in AUX and not tail.endswith(("?", "!", ".")):
        if prev in WH or prev in SUBJECTS:
            if not (len(words) >= 3 and words[-3] in {"yes", "no", "yeah", "nope", "know", "where",
                                                      "what", "how", "who", "matter", "wonder"}):
                return Verdict("unfinished", P_UNFINISHED_MED, f"aux-tail:{prev}-{last}")

    # 6. closing punctuation the ASR wrote
    if closed:
        return Verdict("finished", P_FINISHED_STRONG, "closing-?!")
    if tail.endswith(".") and len(words) >= 3:
        # whisper appends '.' mid-utterance too, so a period alone is only medium evidence
        return Verdict("finished", P_FINISHED_MED, "closing-period")
    return Verdict("unsure", P_UNSURE, "no-cue")


class ModelJudge:
    """Optional Namo-Turn-Detector-v1-English tie-breaker, CPU via onnxruntime + tokenizers.

    Lazy: loads on first predict (~1-2 s, ~600 MB RSS). Thread-safe. predict() -> p_finished or None
    on any failure (missing files, import error) - a broken model can never change a verdict."""

    def __init__(self, model_dir: str, threads: int = 2):
        self.model_dir = model_dir
        self.threads = max(1, int(threads))
        self._sess = None
        self._tok = None
        self._lock = threading.Lock()
        self._failed: "str | None" = None

    def _load(self) -> bool:
        if self._sess is not None:
            return True
        if self._failed:
            return False
        try:
            import numpy as np  # noqa: F401
            import onnxruntime as ort
            from tokenizers import Tokenizer
            path = os.path.join(self.model_dir, "model.onnx")
            tok = Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))
            tok.no_padding()          # the shipped tokenizer.json pads to 128 - wasted compute
            tok.enable_truncation(128)
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            self._tok = tok
            return True
        except Exception as e:
            self._failed = repr(e)
            return False

    def predict(self, text: str) -> "float | None":
        with self._lock:
            if not self._load():
                return None
            try:
                import numpy as np
                enc = self._tok.encode(text)
                ids = np.array([enc.ids], dtype=np.int64)
                mask = np.array([enc.attention_mask], dtype=np.int64)
                lo = self._sess.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
                ex = np.exp(lo - lo.max())
                return float(ex[1] / ex.sum())
            except Exception as e:
                self._failed = repr(e)
                return None


_MODEL: "ModelJudge | None" = None
_MODEL_KEY: "tuple | None" = None


def model_from_env() -> "ModelJudge | None":
    """IRIS_SEMANTIC_EOT_MODEL=<dir with model.onnx + tokenizer.json> turns the tie-breaker on
    (default off); IRIS_SEMANTIC_EOT_THREADS (default 2). Cached per (dir, threads)."""
    global _MODEL, _MODEL_KEY
    d = os.environ.get("IRIS_SEMANTIC_EOT_MODEL", "").strip()
    if not d:
        return None
    key = (d, os.environ.get("IRIS_SEMANTIC_EOT_THREADS", "2"))
    if _MODEL is None or _MODEL_KEY != key:
        try:
            th = int(key[1])
        except ValueError:
            th = 2
        _MODEL, _MODEL_KEY = ModelJudge(d, th), key
    return _MODEL


def judge(text: str, model: "ModelJudge | None" = None) -> Verdict:
    """Lexical first; the model only refines an UNSURE verdict on >= MODEL_MIN_WORDS words, and only
    into the medium bands. Never raises."""
    t0 = time.perf_counter()
    try:
        v = lexical_verdict(text)
    except Exception as e:  # rules must never break a turn
        v = Verdict("unsure", P_UNSURE, f"lexical-error:{e!r}")
    if model is not None and v.label == "unsure" and v.source != "empty" \
            and len(_norm_words(text)) >= MODEL_MIN_WORDS:
        mp = model.predict(text)
        if mp is not None:
            v.model_p = round(mp, 4)
            if mp <= MODEL_UNFINISHED_P:
                v = Verdict("unfinished", P_UNFINISHED_MED, f"model<= {MODEL_UNFINISHED_P} ({v.reason})",
                            source="model", model_p=v.model_p)
            elif mp >= MODEL_FINISHED_P:
                v = Verdict("finished", P_FINISHED_MED, f"model>= {MODEL_FINISHED_P} ({v.reason})",
                            source="model", model_p=v.model_p)
    v.ms = round((time.perf_counter() - t0) * 1000, 3)
    return v


@dataclass
class GracePlan:
    complete: bool      # True -> the turn ends now (no grace round)
    grace_s: float      # grace window to wait for him to resume when not complete
    why: str


def plan_grace(acoustic_hold: bool, verdict: Verdict, base_grace_s: float,
               max_extend_s: float = DEFAULT_MAX_EXTEND_S,
               finished_grace_s: float = DEFAULT_FINISHED_GRACE_S) -> GracePlan:
    """Combine the acoustic verdict (smart-turn HOLD) with the semantic one.

      unfinished          -> grace = base + extension (strong: the full max_extend_s, medium: half)
      finished (strong)   -> acoustic ok: end now; acoustic HOLD: grace = min(base, finished_grace_s),
                             0 -> end now ("clearly finished -> end promptly")
      everything else     -> legacy: HOLD -> base grace, ok -> end now
    """
    base = max(0.0, float(base_grace_s))
    ext_cap = max(0.0, float(max_extend_s))
    p = verdict.p_finished
    if p <= UNFINISHED_BAR:
        frac = min(1.0, max(0.0, (UNFINISHED_BAR - p) / (UNFINISHED_BAR - P_UNFINISHED_STRONG)))
        frac = 1.0 if frac >= 0.999 else (0.5 if frac > 0 else 0.0)
        g = base + ext_cap * frac
        return GracePlan(False, round(g, 3), f"semantic-unfinished(+{ext_cap * frac:.2f}s)")
    if p >= FINISHED_STRONG_BAR:
        if not acoustic_hold:
            return GracePlan(True, 0.0, "both-finished")
        fg = min(base, max(0.0, float(finished_grace_s)))
        if fg <= 0.0:
            return GracePlan(True, 0.0, "semantic-finished-overrides-hold")
        return GracePlan(False, round(fg, 3), "semantic-finished-short-grace")
    if acoustic_hold:
        return GracePlan(False, round(base, 3), "legacy-acoustic-hold")
    return GracePlan(True, 0.0, "legacy-acoustic-ok")


def env_enabled() -> bool:
    """IRIS_SEMANTIC_EOT=1 (in ~/.config/iris/voice.env, read by iris-ears) turns this on; default OFF."""
    return os.environ.get("IRIS_SEMANTIC_EOT", "0").strip().lower() in ("1", "true", "on", "yes")


def knobs(ctl: "dict | None" = None) -> dict:
    """Hot knobs from voice_control.json (same file as grace_resume_s), clamped:
      semantic_eot_max_extend_s     (default 1.5, 0-3)
      semantic_eot_finished_grace_s (default 0.3, 0-2)"""
    c = ctl or {}

    def _f(key, dflt, lo, hi):
        try:
            v = c.get(key)
            return dflt if v is None else min(max(float(v), lo), hi)
        except (TypeError, ValueError):
            return dflt
    return {"max_extend_s": _f("semantic_eot_max_extend_s", DEFAULT_MAX_EXTEND_S, 0.0, 3.0),
            "finished_grace_s": _f("semantic_eot_finished_grace_s", DEFAULT_FINISHED_GRACE_S, 0.0, 2.0)}
