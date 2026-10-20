"""Styled, word-timed captions (ASS) + emoji cues from a faster-whisper transcript.

Styles (STYLES below) are the editing looks we A/B across platforms:
  tiktok_gaming  big pop-in words, 3 at a time, active word in colour, hype words red + emoji
  tiktok_clean   same timing, calmer: white text, active word yellow, no emoji
  music_bts      Bebas Neue, small lower captions, 4-word groups
  youtube        sentence captions at the bottom of a 16:9 frame, no animation

usage (library): captions.build(transcript, t0, t1, style, W, H) -> (ass_text, emoji_events)
"""
from __future__ import annotations

import re

# hex BGR for ASS (&HBBGGRR&)
WHITE, BLACK, YELLOW, GREEN, CYAN, PINK, RED = ("&H00FFFFFF&", "&H00000000&", "&H0000E5FF&",
                                                 "&H0055FF55&", "&H00FFE04D&", "&H00B469FF&", "&H003030FF&")

STYLES = {
    "tiktok_gaming": dict(font="Luckiest Guy", size=96, outline=7, shadow=3, group=3, pop=True,
                          palette=[YELLOW, GREEN, CYAN, PINK], hype_color=RED, emoji=True,
                          pos_y=0.64, upper=True, layout="vertical"),
    "tiktok_clean": dict(font="Anton", size=92, outline=6, shadow=2, group=3, pop=True,
                         palette=[YELLOW], hype_color=YELLOW, emoji=False, pos_y=0.66, upper=True,
                         layout="vertical"),
    "music_bts": dict(font="Bebas Neue", size=84, outline=4, shadow=0, group=4, pop=False,
                      palette=[WHITE], hype_color=WHITE, emoji=False, pos_y=0.80, upper=True,
                      layout="vertical"),
    "youtube": dict(font="DejaVu Sans", size=46, outline=3, shadow=1, group=0, pop=False,
                    palette=[WHITE], hype_color=WHITE, emoji=False, pos_y=0.90, upper=False,
                    layout="landscape"),
}

# reaction word -> Twemoji codepoint (graphics CC-BY 4.0, see scratch/video_edit/emoji/CREDITS.txt)
EMOJI = [(r"\b(ha(ha)+|lmao|lol)\b", "1f602"), (r"\blet'?s go\b", "1f525"), (r"\b(no way|what)\b", "1f631"),
         (r"\b(dead|died|killed)\b", "1f480"), (r"\b(insane|crazy)\b", "1f92f"), (r"\bclutch\b", "1f3c6"),
         (r"\b(oh my god|omg|holy)\b", "1f62d"), (r"\b(gg|win|won)\b", "1f389"), (r"\b(sheesh|nice)\b", "1f60e")]
HYPE_RE = re.compile(r"\b(let'?s|go|no|way|oh|my|god|omg|holy|clutch|insane|crazy|dead|ha(ha)+|lmao|gg|"
                     r"sheesh|what|bro|wow|boom)\b", re.I)


def _ts(t: float) -> str:
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def _clean(w: str) -> str:
    return w.replace("{", "").replace("}", "").replace("\\", "")


def header(st: dict, W: int, H: int) -> str:
    return ("[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
            f"PlayResX: {W}\nPlayResY: {H}\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Cap,{st['font']},{st['size']},{WHITE},{WHITE},{BLACK},&H80000000&,-1,0,0,0,100,100,1,0,1,"
            f"{st['outline']},{st['shadow']},5,60,60,60,1\n\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


def build(tr: dict, t0: float, t1: float, style: str = "tiktok_gaming", W: int = 1080, H: int = 1920,
          speed: float = 1.0) -> tuple[str, list[dict]]:
    """ASS text for the words inside [t0, t1], timed relative to t0 (÷speed). Also returns emoji
    events [{t, dur, code}] relative to the clip."""
    st = STYLES[style]
    words = [dict(w, spk=s.get("speaker")) for s in tr.get("segments", []) for w in s.get("words", [])
             if w["end"] > t0 and w["start"] < t1 and w["word"].strip()]
    # per-speaker colours (speakers.py tagged the segments): Zeke always yellow, friends in order
    spk_order = [x for x in dict.fromkeys(w["spk"] for w in words) if x]
    spk_col = {}
    others = [CYAN, PINK, GREEN, WHITE]
    for sp in spk_order:
        spk_col[sp] = YELLOW if sp == "zeke" else others[len([k for k in spk_col if k != "zeke"]) % len(others)]
    by_speaker = len(spk_col) >= 2
    rel = lambda t: (min(max(t, t0), t1) - t0) / speed  # noqa: E731
    x, y = W // 2, int(H * st["pos_y"])
    out, emo = [header(st, W, H)], []
    if st["group"] == 0:   # sentence captions (youtube)
        for s in tr.get("segments", []):
            if s["end"] > t0 and s["start"] < t1:
                out.append(f"Dialogue: 0,{_ts(rel(s['start']))},{_ts(rel(s['end']))},Cap,,0,0,0,,"
                           f"{{\\an2\\pos({x},{y})}}{_clean(s['text'])}\n")
        return "".join(out), emo
    groups, cur = [], []
    for w in words:   # break groups on size or on a pause > 0.45 s
        if cur and (len(cur) >= st["group"] or w["start"] - cur[-1]["end"] > 0.45 or w["spk"] != cur[-1]["spk"]):
            groups.append(cur); cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    k = 0
    for gi, g in enumerate(groups):
        nxt_start = rel(groups[gi + 1][0]["start"]) if gi + 1 < len(groups) else None
        texts = [_clean(w["word"].upper() if st["upper"] else w["word"]) for w in g]
        for i, w in enumerate(g):
            a = rel(w["start"])
            b = rel(g[i + 1]["start"]) if i + 1 < len(g) else rel(max(w["end"], w["start"] + 0.25))
            if nxt_start is not None:   # never overlap the next group (two caption lines on screen at once)
                b = min(b, nxt_start)
            if b <= a:
                continue
            hype = bool(HYPE_RE.search(w["word"]))
            if by_speaker:   # who's talking decides the colour; hype only makes it bigger
                col = spk_col.get(w["spk"], WHITE)
            else:
                col = st["hype_color"] if hype else st["palette"][k % len(st["palette"])]
            parts = []
            for j, t in enumerate(texts):
                if j == i:
                    pop = "\\fscx118\\fscy118\\t(0,90,\\fscx100\\fscy100)" if st["pop"] else ""
                    big = "\\fscx125\\fscy125\\t(0,120,\\fscx108\\fscy108)" if (st["pop"] and hype) else pop
                    parts.append(f"{{\\c{col}{big}}}{t}{{\\c{WHITE}\\fscx100\\fscy100}}")
                else:
                    parts.append(t)
            out.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{\\an5\\pos({x},{y})}}{' '.join(parts)}\n")
            k += 1
    if st["emoji"]:
        last = -9.0
        for s in tr.get("segments", []):
            if not (s["end"] > t0 and s["start"] < t1):
                continue
            for pat, code in EMOJI:
                m = re.search(pat, s["text"].lower())
                if m:
                    tw = next((w["start"] for w in s["words"] if m.group(0).split()[0][:3] in w["word"].lower()),
                              s["start"])
                    tr_ = rel(tw)
                    if tr_ - last >= 1.2:
                        emo.append({"t": round(tr_, 2), "dur": 1.1, "code": code})
                        last = tr_
                    break
    return "".join(out), emo
