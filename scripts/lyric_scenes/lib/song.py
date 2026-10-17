"""Song timing for scene builders: song_map.json (scripts/music/analyze_song.py) → constant beat grid, frames,
sections, drops, hits, envelopes, curves. numpy only (Blender's python, or any venv with numpy) — no bpy.

    from lib.song import Song                        # sys.path must contain <repo>/scripts/lyric_scenes
    S = Song.load("/data/iris_assets/<song>/analysis/song_map.json", fps=30)      # picks up ear.json next to it
    S.bpm, S.beat0, S.n                              # fitted grid, frame count (index i ↔ Blender frame i+1)
    S.bt(k) → s · S.fr(t) → Blender frame · S.idx(t) → array index · S.snap(t, "downbeat"|"beat"|"bar") → s
    S.sections, S.drops, S.section_at(t), S.mask("drop") / S.mask((t0, t1)) → bool[n]
    S.hits("kick") → array indices · S.env("kick", tau=4) → per-frame decay · S.curve("bass", smooth=1.5) → [n]
    S.check_order(IGN=f1, BUILD=f2, DROP=f3)         # rocket4-style chain-of-events assert
CLI (prints the grid report to send Zeke):  ~/venvs/iris-v100/bin/python -I song.py SONG_MAP [--ear EAR] [--marks 44,58]

THE GRID. song_map's tempo_bpm/beats are beat-tracker output, not a constant grid. Song fits one (kick_grid.py's
maths, moved here): a complex comb over the analyser's KICK times — Σ exp(2πi·t·bpm/60), bpm searched in bpm_range
(default ±6 % around the map's tempo; widen it if you suspect an octave error) — then beat 0 = the circular MODE of the
on-beat kicks (refine_phase; the comb's own phase is dragged by off-beat kicks: 1.5 frames off on Be My One).
Measured on Be My One (all kicks, 10-15): 134.073 BPM / beat0 0.08599 → max 0.38 frames from the hand-fitted
134.095 / 0.09864 grid over all 287 beats (tests/test_song.py). Falls back to the tracker beats if < 16 kicks.

ZEKE'S EAR OUTRANKS THE ANALYSIS. ear.json (next to song_map.json, or pass ear=path|dict), every key optional:
    {"drops": [29.0, 57.8],                          # times (s) or {"time":..,"label":..}; drops[0] anchors the DOWNBEAT
     "sections": [{"label": "bridge", "start": 86.0, "end": 101.0}, ...],   # REPLACES the analysed sections
     "bpm": 134.095, "beat0": 0.09864,               # pin the grid outright (skips the fit)
     "bpm_range": [120, 150], "fit_windows": [[29, 43.5], [58, 72]],        # steer the fit instead
     "notes": "first drop 0:29 (Zeke 10-13)"}
"""
import json
import math
import os
import sys

import numpy as np

# ------------------------------------------------------------------------------------------------ pure helpers


def gsmooth(x, sigma):
    """Gaussian smooth (edge-padded, same length). sigma in samples; ≤0 returns a float copy."""
    x = np.asarray(x, np.float64)
    if sigma <= 0:
        return x.copy()
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def pnorm(x, lo=5, hi=98):
    """Percentile-normalise to 0..1 (robust to outliers)."""
    x = np.asarray(x, np.float64)
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(b - a, 1e-9), 0, 1)


def hit_age(hit_idx, n):
    """Frames since the most recent hit at each index (inf before the first hit)."""
    age = np.full(n, np.inf)
    h = np.unique(np.asarray(hit_idx, int))
    h = h[(h >= 0) & (h < n)]
    if len(h):
        i = np.arange(n)
        last = np.searchsorted(h, i, side="right") - 1
        ok = last >= 0
        age[ok] = i[ok] - h[last[ok]]
    return age


def age_env(age, tau):
    """exp(−age/tau), 0 where age is inf (before the first hit)."""
    a = np.asarray(age, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(np.isfinite(a), np.exp(-np.minimum(a, 1e6) / tau), 0.0)


def fit_grid(events, bpm_lo, bpm_hi, windows=None, step=0.002):
    """Constant-tempo comb fit to event times (s). Returns (bpm, beat0 in [0, period), strength 0..1)."""
    t = np.asarray(events, np.float64)
    if windows:
        t = t[np.any([(t >= a) & (t <= b) for a, b in windows], axis=0)]
    if len(t) < 4:
        raise ValueError(f"too few events to fit a grid ({len(t)})")
    bpms = np.arange(bpm_lo, bpm_hi + step / 2, step)
    best = None
    for c in range(0, len(bpms), 512):                   # chunked: bounded memory on long songs
        b = bpms[c:c + 512]
        z = np.exp(2j * np.pi * np.outer(b / 60.0, t)).sum(1)
        i = int(np.argmax(np.abs(z)))
        if best is None or abs(z[i]) > abs(best[1]):
            best = (b[i], z[i])
    bpm, z = best
    period = 60.0 / bpm
    return float(bpm), refine_phase(t, period, np.angle(z) / (2 * np.pi)), float(abs(z) / len(t))


def refine_phase(t, period, phase0=None, win=0.08):
    """beat0 from the ON-BEAT events only. The comb's own phase is a mean over ALL events, so syncopated / off-beat
    kicks pull it (Be My One: 1.5 frames off). Take the circular mode of the event phases instead, then the circular
    mean of events within ±win beats of it. Returns beat0 in [0, period)."""
    r = (np.asarray(t, np.float64) / period) % 1.0
    h, _ = np.histogram(r, bins=50, range=(0, 1))
    hs = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5), "valid")         # circular 5-bin smoothing
    c = (int(np.argmax(hs)) + 0.5) / 50
    sel = np.abs((r - c + 0.5) % 1.0 - 0.5) < win
    if sel.sum() < 4:                                                       # too sparse: trust the comb
        ph = (phase0 if phase0 is not None else c) % 1.0
    else:
        ph = (np.angle(np.exp(2j * np.pi * r[sel]).sum()) / (2 * np.pi)) % 1.0
    return float(ph * period % period)


def _drop_times(drops):
    out = []
    for d in drops or []:
        out.append(float(d["time"] if isinstance(d, dict) else d))
    return out


# ------------------------------------------------------------------------------------------------ Song


class Song:
    def __init__(self, song_map, fps=30, ear=None, source=None, curves=None):
        self.map = song_map
        self.fps = float(fps)
        self.source = source
        self.ear = ear or {}
        self.duration = float(song_map["duration"])
        self.n = int(round(self.duration * self.fps))
        self.t = np.arange(self.n) / self.fps                          # time of each array index
        self.bars_per = int(song_map.get("beats_per_bar", 4))
        self._curves = curves
        self._fit()
        self.drops = _drop_times(self.ear.get("drops")) or _drop_times(song_map.get("drops"))
        self.sections = [dict(s, start=float(s["start"]), end=float(s["end"]))
                         for s in (self.ear.get("sections") or song_map.get("sections") or [])]
        self.sections_from = "ear" if self.ear.get("sections") else "analysis"
        self._downbeat_phase()

    # ---- construction
    @classmethod
    def load(cls, song_map_json, fps=30, ear="auto"):
        with open(song_map_json) as f:
            m = json.load(f)
        if ear == "auto":
            p = os.path.join(os.path.dirname(os.path.abspath(song_map_json)), "ear.json")
            ear = p if os.path.isfile(p) else None
        if isinstance(ear, str):
            with open(ear) as f:
                ear = json.load(f)
        curves = None
        cp = (m.get("curves") or {}).get("path")
        if not cp or not os.path.isfile(cp):
            cp = os.path.join(os.path.dirname(os.path.abspath(song_map_json)), "curves.json")
        if os.path.isfile(cp):
            curves = cp
        return cls(m, fps, ear, source=os.path.abspath(song_map_json), curves=curves)

    def _fit(self):
        e = self.ear
        if e.get("bpm") and e.get("beat0") is not None:
            self.bpm, self.beat0, self.fit_strength, self.fit_from = float(e["bpm"]), float(e["beat0"]), None, "ear"
        else:
            tempo = float(self.map.get("tempo_bpm") or 120)
            lo, hi = e.get("bpm_range") or (tempo * 0.94, tempo * 1.06)
            kicks = (self.map.get("drum_hits") or {}).get("kick") or []
            ev, src = (kicks, "kick") if len(kicks) >= 16 else (self.map.get("beats") or [], "beats")
            self.bpm, self.beat0, self.fit_strength = fit_grid(ev, lo, hi, e.get("fit_windows"))
            self.fit_from = src
            if e.get("bpm"):                                  # bpm pinned, phase fitted
                self.bpm = float(e["bpm"])
                self.beat0 = refine_phase(ev, 60.0 / self.bpm)
        self.period = 60.0 / self.bpm

    def _downbeat_phase(self):
        """Which beat index ≡ downbeat (mod beats_per_bar): ear drop[0] if given, else vote with analysed downbeats."""
        if self.ear.get("drops"):
            self.downbeat_mod = self.beat_index(self.drops[0]) % self.bars_per
            self.downbeat_from = "ear drop"
        else:
            db = self.map.get("downbeats") or []
            votes = np.zeros(self.bars_per)
            for t in db:
                votes[self.beat_index(t) % self.bars_per] += 1
            self.downbeat_mod = int(np.argmax(votes)) if len(db) else 0
            self.downbeat_from = "analysis downbeats" if len(db) else "default"

    # ---- grid
    def bt(self, k):
        """time (s) of grid beat k"""
        return self.beat0 + k * self.period

    def beat_index(self, t):
        """nearest grid beat index to time t"""
        return int(round((t - self.beat0) / self.period))

    def beat_pos(self, t):
        """fractional beat position of time t (beat 0 = 0.0)"""
        return (t - self.beat0) / self.period

    @property
    def beats(self):
        k = np.arange(0, int((self.duration - self.beat0) / self.period) + 1)
        return self.bt(k)

    @property
    def downbeats(self):
        k = np.arange(0, int((self.duration - self.beat0) / self.period) + 1)
        k = k[k % self.bars_per == self.downbeat_mod]
        return self.bt(k)

    def bar_of(self, t):
        """bar number (0-based, bar 0 = the first full downbeat) and beat-in-bar of time t"""
        k = self.beat_index(t) - self.downbeat_mod
        return k // self.bars_per, k % self.bars_per

    def snap(self, t, to="downbeat"):
        """nearest grid time to t: to = 'beat' | 'downbeat' | 'bar' (= downbeat) | 'half' (half-beat)"""
        if to == "beat":
            return self.bt(self.beat_index(t))
        if to == "half":
            return self.beat0 + round((t - self.beat0) / (self.period / 2)) * self.period / 2
        if to in ("downbeat", "bar"):
            k = self.beat_index(t)
            r = (k - self.downbeat_mod) % self.bars_per
            k = k - r if r <= self.bars_per // 2 else k + (self.bars_per - r)
            return self.bt(k)
        raise ValueError(to)

    # ---- frames
    def fr(self, t):
        """Blender frame for time t (frame 1 = t 0)"""
        return 1 + int(round(t * self.fps))

    def idx(self, t):
        """array index for time t (index i ↔ frame i+1), clamped to [0, n-1]"""
        return min(self.n - 1, max(0, int(round(t * self.fps))))

    def snap_fr(self, t, to="downbeat"):
        return self.fr(self.snap(t, to))

    # ---- structure
    def section_at(self, t):
        for s in self.sections:
            if s["start"] <= t < s["end"]:
                return s
        return None

    def mask(self, what):
        """bool[n]: a section label (all sections with that label), a (t0, t1) window, or a list of either"""
        if isinstance(what, list):
            m = np.zeros(self.n, bool)
            for w in what:
                m |= self.mask(w)
            return m
        if isinstance(what, tuple):
            return (self.t >= what[0]) & (self.t < what[1])
        m = np.zeros(self.n, bool)
        found = False
        for s in self.sections:
            if s["label"] == what:
                m |= (self.t >= s["start"]) & (self.t < s["end"])
                found = True
        if not found:
            raise KeyError(f"no section labelled {what!r}; have {sorted({s['label'] for s in self.sections})}")
        return m

    # ---- events + curves
    def hits(self, kind="kick"):
        """array indices of analysed drum hits: kind = kick | snare | hat"""
        ts = (self.map.get("drum_hits") or {}).get(kind)
        if ts is None:
            raise KeyError(f"no drum_hits[{kind!r}] in song_map")
        return np.unique(np.clip(np.round(np.asarray(ts) * self.fps).astype(int), 0, self.n - 1))

    def env(self, kind="kick", tau=4.0, hits=None):
        """per-frame decay envelope exp(−frames_since_hit/tau) (0 before the first hit)"""
        return age_env(hit_age(self.hits(kind) if hits is None else hits, self.n), tau)

    def beat_env(self, tau=4.0, every=1):
        """the same envelope on the GRID (every=1 beats, =bars_per downbeats) — steady pulse, no tracker jitter"""
        k = np.arange(0, int((self.duration - self.beat0) / self.period) + 1)
        if every > 1:
            k = k[(k - self.downbeat_mod) % every == 0]
        return age_env(hit_age(np.round(self.bt(k) * self.fps).astype(int), self.n), tau)

    def curve(self, name, smooth=0.0, norm=False):
        """curves.json curve resampled to n frames (0..1); smooth = gaussian sigma in frames; norm → pnorm()"""
        if self._curves is None:
            raise FileNotFoundError("no curves.json next to song_map.json")
        if isinstance(self._curves, str):
            with open(self._curves) as f:
                self._curves = json.load(f)
        c = self._curves["curves"].get(name)
        if c is None:
            raise KeyError(f"no curve {name!r}; have {list(self._curves['curves'])}")
        c = np.asarray(c, np.float64)
        cf = float(self._curves.get("fps", self.fps))
        if len(c) != self.n or cf != self.fps:
            c = np.interp(self.t, np.arange(len(c)) / cf, c)
        c = gsmooth(c, smooth) if smooth else c
        return pnorm(c) if norm else c

    # ---- asserts + report
    @staticmethod
    def check_order(**named):
        """assert the named values are strictly increasing in the order given (frames or seconds)"""
        items = list(named.items())
        for (a, va), (b, vb) in zip(items, items[1:]):
            if not va < vb:
                raise AssertionError(f"order: {a}={va} must be < {b}={vb}  (full: {named})")
        return True

    def report(self, marks=()):
        d0 = self.drops[0] if self.drops else None
        lines = [f"[song] {os.path.basename(self.source or '?')}: {self.duration:.2f} s = {self.n} frames @ {self.fps:g} fps",
                 f"[song] grid {self.bpm:.3f} BPM (period {self.period:.5f} s), beat0 {self.beat0:.5f} s — from "
                 f"{self.fit_from}{'' if self.fit_strength is None else f' (comb strength {self.fit_strength:.2f})'}; "
                 f"analyser said {self.map.get('tempo_bpm')}",
                 f"[song] downbeats: beat ≡ {self.downbeat_mod} mod {self.bars_per} ({self.downbeat_from}); "
                 f"sections from {self.sections_from}; drops {['%.2f' % x for x in self.drops]}"
                 + (f"; first drop = beat {self.beat_index(d0)} at {self.bt(self.beat_index(d0)):.3f} s" if d0 else "")]
        for s in self.sections:
            b = self.snap(s["start"])
            lines.append(f"  {s['label']:10s} {s['start']:7.2f}–{s['end']:7.2f} s → downbeat {b:7.3f} s "
                         f"(frame {self.fr(b)}, {1000 * (s['start'] - b):+.0f} ms)")
        for m in marks:
            k = self.beat_index(m)
            bar, beat = self.bar_of(m)
            lines.append(f"  mark {m:7.2f} s → beat {k} at {self.bt(k):.3f} s ({1000 * (m - self.bt(k)):+.0f} ms) "
                         f"bar {bar} beat {beat + 1} → frame {self.fr(self.bt(k))}")
        return "\n".join(lines)


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="grid report for a song_map.json")
    ap.add_argument("song_map")
    ap.add_argument("--ear", default="auto")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--marks", default="")
    a = ap.parse_args(argv)
    s = Song.load(a.song_map, a.fps, a.ear if a.ear != "none" else None)
    print(s.report([float(x) for x in a.marks.split(",") if x]))
    return s


if __name__ == "__main__":
    _main(sys.argv[1:])
