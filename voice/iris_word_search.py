"""voice/iris_word_search.py - jump in with the word he can't find (conversation layer, 10-31).

Zeke 10-31: *"if mid-sentence I say 'oh, I can't remember what it's called', you jump in with the
word and we carry on."* People do this all the time (co-operative word search: the speaker shows
he is searching - "what's it called", "uh, the ... thing" - and the listener offers a candidate,
usually with a try-marked, questioning intonation; Goodwin & Goodwin 1986, Hayashi 2003, Lerner
1996 on co-completion). The offer is cheap to be wrong about because it is shaped as a GUESS
("geometry nodes?"), and he carries on either way.

Three pieces:
  * detect_cue(text)        - pure: is the live partial an explicit word search, and where does it
                              start? Regex on the cue phrases people actually use; hesitation alone
                              ("uh") is NOT enough (too common, mostly not a search).
  * WordSearchPolicy        - pure: one offer per utterance, only once he has actually PAUSED on the
                              cue (the partial stopped growing for pause_s) - if he finds the word
                              himself ("what's it called - oh, geometry nodes") we stay quiet.
  * suggest(...)            - one call to a FAST local LLM (Ollama on the V100) with the recent
                              conversation as context; returns a short candidate or None. Never the
                              main cognition: a Claude turn is seconds, the gap is ~1 s.

The policy and the parser are pure (unit-tested); suggest() takes an injectable post_fn.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request as _req
from pathlib import Path

# ── cue detection (pure) ────────────────────────────────────────────────────────
# Explicit "I'm searching for a word" phrases. Ordered loosely by how often people use them.
_CUES = [
    r"what(?:'s| is| was) (?:it|that|this|the (?:thing|word|name|one)) called",
    r"what(?:'s| is| was) the (?:word|name|term)",
    r"what do (?:you|they|we) call (?:it|that|them|those|the)",
    r"what(?:'s| is) (?:his|her|their) name",
    r"i (?:can(?:no|')t|cannot|don'?t) (?:remember|recall|think of) (?:the |what |its |it'?s |his |her )?"
    r"(?:word|name|term|it'?s? called|what it'?s? called|the thing)",
    r"i (?:forgot|forget) (?:the |what )?(?:word|name|term|it'?s called|what it'?s called)",
    r"(?:it'?s|it is) on the tip of my tongue",
    r"tip of my tongue",
    r"whatchamacallit|what-?cha-?ma-?call-?it|thingamajig|thingamabob",
    r"(?:oh|uh|um),? (?:what|how) do you say",
    r"what(?:'s| is) it,? (?:uh|um)",
]
_CUE_RE = re.compile(r"\b(?:" + "|".join(_CUES) + r")\b", re.I)


def detect_cue(text: str) -> "dict | None":
    """{'cue': matched phrase, 'start': idx, 'before': words before the cue, 'after': words after}
    for the LAST cue in the partial, else None. Pure."""
    if not text:
        return None
    last = None
    for m in _CUE_RE.finditer(text):
        last = m
    if last is None:
        return None
    return {"cue": last.group(0), "start": last.start(),
            "before": text[:last.start()].strip(), "after": text[last.end():].strip(" ,.?!-")}


def _words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", s or ""))


# ── policy (pure) ───────────────────────────────────────────────────────────────
DEFAULTS = {
    "pause_s": 0.6,          # trailing SILENCE in his audio after the cue (he is searching)
    "fallback_pause_s": 1.4, # no tail_silence_s from the writer: partial unchanged this long instead
                             # (partials refresh every ~0.6-0.8 s, so "unchanged for 0.6 s" is NOT a
                             # pause - the 10-31 e2e run offered a word mid-sentence that way)
    "resumed_s": 0.2,        # fresh tail silence below this after the LLM = he is talking again
    "max_after_words": 3,    # more than this many words after the cue = he moved on / found it
    "min_before_words": 2,   # need SOME context before the cue to guess from
    "max_age_s": 8.0,        # an offer later than this after the cue is noise, drop it
}


class WordSearchPolicy:
    """Per-utterance. Feed every new live partial with observe(); call due() on a timer. Exactly
    one offer per utterance; after a suggestion comes back, still_relevant() re-checks the live
    words (he may have found it himself meanwhile)."""

    def __init__(self, cfg: "dict | None" = None):
        self.cfg = dict(DEFAULTS if cfg is None else cfg)
        self.utterance_id = None
        self.text = ""
        self.changed_t = 0.0
        self.cue = None
        self.cue_t = None
        self.fired = False
        self.tail_sil = None

    def observe(self, utterance_id: str, text: str, now: float, tail_sil=None) -> None:
        if utterance_id != self.utterance_id:
            self.__init__(self.cfg)
            self.utterance_id = utterance_id
        self.tail_sil = tail_sil
        if text != self.text:
            self.text = text or ""
            self.changed_t = now
            cue = detect_cue(self.text)
            if cue and (self.cue is None or cue["start"] != self.cue["start"]):
                self.cue_t = now
            self.cue = cue

    def due(self, now: float) -> "dict | None":
        """The cue to act on now, or None. Marks the utterance as fired (one offer max)."""
        c, cfg = self.cue, self.cfg
        if self.fired or c is None or self.cue_t is None:
            return None
        if _words(c["after"]) > cfg["max_after_words"]:
            return None                              # he kept going - he found it or moved on
        if _words(c["before"]) < cfg["min_before_words"]:
            return None
        if self.tail_sil is not None:
            if self.tail_sil < cfg["pause_s"]:
                return None                          # his audio isn't quiet yet - let him find it
        elif now - self.changed_t < cfg["fallback_pause_s"]:
            return None                              # old writer: only the (slow) partial clock
        if now - self.cue_t > cfg["max_age_s"]:
            return None
        self.fired = True
        return c

    def still_relevant(self, text: str, cue: dict, word: str, tail_sil=None) -> bool:
        """After the LLM answered: is the offer still welcome? Not if he said the word himself, has
        carried on well past the cue, or his audio shows he is talking again."""
        if tail_sil is not None and tail_sil < self.cfg["resumed_s"]:
            return False
        t = (text or "").lower()
        if word and word.lower() in t[cue["start"]:]:
            return False
        i = t.find(cue["cue"].lower())
        after = t[i + len(cue["cue"]):] if i >= 0 else ""
        return _words(after) <= self.cfg["max_after_words"]


# ── the fast LLM ────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"

PROMPT = (
    "Live spoken conversation. The speaker is mid-sentence, searching for a word or name they can't "
    "remember. Name the exact thing they mean: the specific proper name or technical term, not a "
    "generic description (1-4 words). Use the recent conversation. If you can't make a confident "
    "guess, answer NONE. Reply with ONLY the word/name.\n\n"
    "Example: \"the painter who did the starry night, what's his name\" -> Van Gogh\n"
    "Example: \"in python, the thing that installs packages, what's it called\" -> pip\n"
    "Example: \"you know, the, uh, what's it called\" -> NONE\n\n"
    "{glossary}Recent conversation:\n{context}\n\nSpeaker, right now: \"{partial}\" ->"
)
GLOSSARY_FILE = Path(__file__).with_name("convo_glossary.txt")


def load_glossary(path=None) -> "list[str]":
    """Names/terms from our own world the model can't know (one per line, # comments). Missing = []."""
    try:
        lines = Path(path or GLOSSARY_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def build_prompt(partial: str, context_lines: "list[str] | None" = None,
                 glossary: "list[str] | None" = None) -> str:
    ctx = "\n".join((context_lines or [])[-6:]) or "(none)"
    gl = ("Names and terms from our own world (the answer may be one of these): "
          + "; ".join(glossary)[:1200] + "\n\n") if glossary else ""
    return PROMPT.format(glossary=gl, context=ctx[-1500:], partial=(partial or "")[-400:])


def parse_suggestion(raw: str) -> "str | None":
    """First line, stripped of quotes/punctuation/'Word:' echo; None for NONE/empty/too long."""
    if not raw:
        return None
    s = raw.strip().splitlines()[0].strip()
    s = re.sub(r"^(?:word|answer|guess)\s*:\s*", "", s, flags=re.I)
    s = s.strip(" \t\"'`*.,!?;:()[]")
    if not s or s.upper().startswith("NONE") or s.lower() in ("i don't know", "unknown", "n/a"):
        return None
    if _words(s) > 4 or len(s) > 40:
        return None
    return s


def _default_post(url: str, payload: dict, timeout: float) -> dict:
    req = _req.Request(url, data=json.dumps(payload).encode(), method="POST",
                       headers={"Content-Type": "application/json"})
    with _req.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def suggest(partial: str, context_lines=None, *, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
            timeout: float = 2.5, keep_alive: str = "30m", glossary=None, num_ctx: int = 2048,
            post_fn=None) -> "tuple[str | None, float]":
    """(candidate or None, seconds). Short greedy decode; any error = (None, t)."""
    t0 = time.time()
    payload = {"model": model, "prompt": build_prompt(partial, context_lines, glossary), "stream": False,
               "keep_alive": keep_alive,
               "options": {"temperature": 0, "num_predict": 8, "stop": ["\n"], "num_ctx": num_ctx}}
    try:
        out = (post_fn or _default_post)(url, payload, timeout)
        return parse_suggestion(str(out.get("response") or "")), time.time() - t0
    except Exception:
        return None, time.time() - t0


def offer_line(word: str) -> str:
    """Try-marked: a guess, said like a question, so being wrong costs nothing."""
    return word.rstrip("?") + "?"
