#!/usr/bin/env python3
"""analyze_song.py — turn a song into a machine-readable map for animation planning.

Run with the music venv, isolated:
    ~/venvs/musicai/bin/python -I scripts/music/analyze_song.py SONG.wav OUTDIR [options]

GPU: the default is CPU everywhere (VRAM is shared with the live stack). To use the GPU for
demucs / beat_this / crepe, launch through the capped render slice and pass --device cuda:
    scripts/server/render.sh music-<name> ~/venvs/musicai/bin/python -I <abs path>/analyze_song.py SONG.wav OUTDIR --device cuda

Writes OUTDIR/song_map.json (the one file a scene builder reads) plus:
    stems/htdemucs/<name>/{drums,bass,other,vocals}.wav   demucs htdemucs
    midi/melody_crepe.mid        monophonic lead melody (torchcrepe on the vocal stem)
    midi/vocals_basicpitch.mid   basic-pitch on the vocal stem (polyphonic, catches harmonies/chops)
    midi/bass.mid  midi/other.mid  basic-pitch on those stems
    midi/drums.mid               kick/snare/hat onsets from the drum stem (GM 36/38/42)
    midi/song_all.mid            melody + bass + other + drums in one file (for M2V / MIDIAnimator / nla_midi_copier)
    melody_contour.csv           time_s, f0_hz, midi, periodicity  (10 ms hop)
    curves.json                  per-frame 0..1 energy curves at --fps (full, stems, bands, onset)
    song_map.png                 small timeline plot (<150 KB) — safe to Read
    allin1/                      raw all-in-one output (only with --allin1)

Tools (licences): demucs (MIT), beat_this (MIT code + weights), basic-pitch (Apache-2.0, ONNX model),
torchcrepe (MIT), essentia (AGPL-3, algorithmic key profiles, no ML weights), librosa (ISC),
all-in-one (MIT code + weights; needs NATTEN, built CPU-only here). madmom is installed (allin1 needs its DBN
code) but its bundled NN models are CC BY-NC-SA — this script does NOT use them.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
TIMINGS = {}


def log(*a):
    print("[analyze_song]", *a, flush=True)


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        log(f"-> {self.name}")

    def __exit__(self, *exc):
        TIMINGS[self.name] = round(time.time() - self.t, 2)
        log(f"   {self.name} done in {TIMINGS[self.name]} s")


def rnd(x, n=3):
    return [round(float(v), n) for v in x]


# ----------------------------------------------------------------------------- stems
def run_demucs(wav, outdir, device):
    name = os.path.splitext(os.path.basename(wav))[0]
    sdir = os.path.join(outdir, "stems", "htdemucs", name)
    stems = {s: os.path.join(sdir, s + ".wav") for s in ("drums", "bass", "other", "vocals")}
    if all(os.path.isfile(p) for p in stems.values()):
        log("   stems already present, reusing")
        return stems, sdir
    cmd = [sys.executable, "-m", "demucs.separate", "-n", "htdemucs", "-d", device,
           "-o", os.path.join(outdir, "stems"), wav]
    subprocess.run(cmd, check=True, cwd=outdir)
    return stems, sdir


# ----------------------------------------------------------------------------- beats
def run_beats(wav, device):
    from beat_this.inference import File2Beats
    f2b = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = f2b(wav)
    beats = np.asarray(beats, float)
    downbeats = np.asarray(downbeats, float)
    # beat_this works on a 50 fps grid, so single inter-beat intervals are quantised to 20 ms
    # (0.44 s = 136.4 BPM, 0.46 s = 130.4). Median of 16-beat spans brings that down to ~1 ms and is
    # robust to a missed beat.
    ibi = np.diff(beats)
    if len(beats) > 20:
        span = (beats[16:] - beats[:-16]) / 16.0
        tempo = 60.0 / float(np.median(span))
    else:
        tempo = 60.0 / float(np.median(ibi)) if len(ibi) else 0.0
    # beats per bar = median count of beats between consecutive downbeats
    if len(downbeats) > 2:
        counts = [int(((beats >= a - 1e-3) & (beats < b - 1e-3)).sum()) for a, b in zip(downbeats[:-1], downbeats[1:])]
        bpb = int(np.median(counts))
    else:
        bpb = 4
    return beats, downbeats, tempo, bpb


# ----------------------------------------------------------------------------- bar features + structure
def db(x):
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-6) if len(x) else -120.0


def bar_table(signals, sr, downbeats, duration):
    edges = [0.0] + [float(d) for d in downbeats if 0.05 < d < duration - 0.05] + [duration]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 0.25:
            continue
        ia, ib = int(a * sr), int(b * sr)
        rows.append({"start": a, "end": b, **{k: db(v[ia:ib]) for k, v in signals.items()}})
    return rows


def label_bars(rows):
    """EDM-aware bar labels from stem loudness: drop / build / breakdown / other.

    drop      = mix within 3 dB of its loudest bar AND drums within 7 dB of their loudest bar
    build     = non-drop bars right before a drop where the bass has dropped out (> 30 dB under its max)
                but drums are still there (riser / snare roll)
    breakdown = drums > 25 dB under their max (drums gone)
    """
    fmax = max(r["full"] for r in rows)
    dmax = max(r["drums"] for r in rows)
    bmax = max(r["bass"] for r in rows)
    lab = []
    for r in rows:
        if r["full"] >= fmax - 3.0 and r["drums"] >= dmax - 7.0:
            lab.append("drop")
        elif r["drums"] <= dmax - 25.0:
            lab.append("breakdown")
        else:
            lab.append("other")
    # builds: walk back from each drop start
    for i in range(1, len(rows)):
        if lab[i] == "drop" and lab[i - 1] != "drop":
            j = i - 1
            while j >= 0 and lab[j] != "drop" and rows[j]["bass"] <= bmax - 30.0 and rows[j]["drums"] > dmax - 25.0:
                lab[j] = "build"
                j -= 1
    return lab


def runs(labels):
    out = []
    for i, l in enumerate(labels):
        if out and out[-1][0] == l:
            out[-1][2] = i
        else:
            out.append([l, i, i])
    return out


def structure_from_bars(rows, labels, duration):
    # 1-bar 'other' or 'breakdown' blips merge into the following run (or previous at the end)
    lab = list(labels)
    changed = True
    while changed:
        changed = False
        rr = runs(lab)
        for k, (l, a, b) in enumerate(rr):
            if b == a and l in ("other", "breakdown") and len(rr) > 1:
                tgt = rr[k + 1][0] if k + 1 < len(rr) else rr[k - 1][0]
                if tgt != l:
                    lab[a] = tgt
                    changed = True
                    break
    rr = runs(lab)
    drop_idx = [k for k, r in enumerate(rr) if r[0] == "drop"]
    first_drop = drop_idx[0] if drop_idx else len(rr)
    last_drop = drop_idx[-1] if drop_idx else -1
    parts = []
    for k, (l, a, b) in enumerate(rr):
        nb = b - a + 1
        name = l
        if l != "drop" and l != "build":
            if k < first_drop:
                name = "intro"
            elif k > last_drop:
                name = "outro"
            elif l == "other":
                name = "bridge" if nb >= 4 else "break"
            else:
                name = "breakdown" if nb >= 3 else "break"
        parts.append({"label": name, "start": round(rows[a]["start"], 3), "end": round(rows[b]["end"], 3), "bars": nb,
                      "bar_index": [a, b]})
    # coarse sections: merge everything between two drops into one region
    sections = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p["label"] == "drop":
            sections.append(dict(p))
            i += 1
            continue
        j = i
        while j + 1 < len(parts) and parts[j + 1]["label"] != "drop":
            j += 1
        grp = parts[i:j + 1]
        a, b = grp[0]["bar_index"][0], grp[-1]["bar_index"][1]
        nb = b - a + 1
        if i == 0:
            name = "intro"
        elif j == len(parts) - 1:
            name = "outro"
        elif nb < 4:
            name = "break"
        else:
            bmax = max(r["bass"] for r in rows)
            bass_gone = np.mean([rows[x]["bass"] <= bmax - 30.0 for x in range(a, b + 1)])
            name = "breakdown" if bass_gone >= 0.5 else "bridge"
        sections.append({"label": name, "start": grp[0]["start"], "end": grp[-1]["end"], "bars": nb,
                         "bar_index": [a, b], "parts": [g["label"] for g in grp]})
        i = j + 1
    # drops: numbered; 'major' when the region before it lasted >= 4 bars
    drops = []
    for k, s in enumerate(sections):
        if s["label"] != "drop":
            continue
        prev = sections[k - 1] if k else None
        jump = rows[s["bar_index"][0]]["full"] - (rows[s["bar_index"][0] - 1]["full"] if s["bar_index"][0] else -120)
        drops.append({"n": len(drops) + 1, "time": s["start"], "end": s["end"],
                      "major": bool(prev is None or prev["bars"] >= 4),
                      "after": prev["label"] if prev else None, "loudness_jump_db": round(float(jump), 1)})
    # an 'outro' that is itself drop-loud is reported as a drop above; the tail after it stays 'outro'
    return parts, sections, drops


# ----------------------------------------------------------------------------- key + chords
def chord_templates():
    names, T = [], []
    for root in range(12):
        for q, iv in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            v = np.zeros(12)
            for x in iv:
                v[(root + x) % 12] = 1.0
            v[root] += 0.5  # root weight
            T.append(v / np.linalg.norm(v))
            names.append(FLAT_NAMES[root] + q)
    return names, np.array(T)


def diatonic(key_root, minor):
    """Chord names (FLAT_NAMES + '' / 'm') diatonic to a key; minor keys also get the harmonic-minor V."""
    if minor:
        degs = [(0, "m"), (3, ""), (5, "m"), (7, "m"), (7, ""), (8, ""), (10, "")]
    else:
        degs = [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m")]
    return {FLAT_NAMES[(key_root + d) % 12] + q for d, q in degs}


def key_from_chords(seg):
    dur = {}
    for c in seg:
        if c["chord"] != "N":
            dur[c["chord"]] = dur.get(c["chord"], 0.0) + (c["end"] - c["start"]) * (0.5 + c.get("conf", 0.5))
    best = None
    for r in range(12):
        for minor in (False, True):
            tonic = FLAT_NAMES[r] + ("m" if minor else "")
            sc = sum(v for k, v in dur.items() if k in diatonic(r, minor)) + 0.5 * dur.get(tonic, 0.0)
            if best is None or sc > best[0]:
                best = (sc, r, minor)
    tot = sum(dur.values()) + 1e-9
    return {"key": f"{FLAT_NAMES[best[1]]} {'minor' if best[2] else 'major'}", "root": best[1], "minor": best[2],
            "diatonic_share": round(best[0] / (1.5 * tot), 2)}


def run_chords(harm, sr, beats, duration):
    """Beat-synchronous CQT chroma on the harmonic stems (other+bass), 24 major/minor templates + 'N',
    Viterbi-smoothed (sticky transitions). Pass 2 adds a small prior for chords diatonic to the key that
    pass 1 implies. Each segment gets conf = mean template cosine (≈0.85+ clean, <0.6 smeared, e.g. a
    distorted/gliding drop bass). Licence-clean (no learned weights)."""
    import librosa
    hop = 512
    C = librosa.feature.chroma_cqt(y=harm, sr=sr, hop_length=hop, bins_per_octave=36)
    frames = librosa.time_to_frames(np.concatenate([[0.0], beats, [duration]]), sr=sr, hop_length=hop)
    frames = librosa.util.fix_frames(np.unique(np.clip(frames, 0, C.shape[1])), x_min=0, x_max=C.shape[1])
    Cs = librosa.util.sync(C, frames, aggregate=np.median)
    frames = frames[: Cs.shape[1] + 1]
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    energy = np.array([np.sqrt(np.mean(harm[int(a * sr):int(b * sr)] ** 2) + 1e-12)
                       for a, b in zip(times[:-1], times[1:])])
    names, T = chord_templates()
    Cn = Cs / (np.linalg.norm(Cs, axis=0, keepdims=True) + 1e-9)
    sim = T @ Cn  # (24, n)
    quiet = energy < (np.max(energy) * 10 ** (-40 / 20))
    K = len(names) + 1
    trans = np.full((K, K), (1 - 0.85) / (K - 1))
    np.fill_diagonal(trans, 0.85)

    def decode(prior):
        S = np.vstack([sim + prior[:, None], np.where(quiet, 2.0, 0.5)[None, :]])
        prob = np.exp(10.0 * S)
        prob /= prob.sum(axis=0, keepdims=True)
        path = librosa.sequence.viterbi(prob, trans)
        seg = []
        for i, k in enumerate(path):
            a = float(times[i])
            b = float(times[i + 1]) if i + 1 < len(times) else duration
            nm = (names + ["N"])[k]
            cf = float(sim[k, i]) if k < len(names) else 0.0
            if seg and seg[-1]["chord"] == nm:
                seg[-1]["end"] = round(b, 3)
                seg[-1]["_c"].append(cf)
            else:
                seg.append({"start": round(a, 3), "end": round(b, 3), "chord": nm, "_c": [cf]})
        for s_ in seg:
            s_["conf"] = round(float(np.mean(s_.pop("_c"))), 2)
        return seg

    seg1 = decode(np.zeros(len(names)))
    k1 = key_from_chords(seg1)
    dia = diatonic(k1["root"], k1["minor"])
    prior = np.array([0.06 if n in dia else 0.0 for n in names])
    seg = decode(prior)
    return seg, key_from_chords(seg)


def run_key(wav, chords, melody_midi, chord_key=None):
    import essentia.standard as es
    audio = es.MonoLoader(filename=wav, sampleRate=44100)()
    cands = {}
    for prof in ("edma", "krumhansl", "temperley", "bgate"):
        k, s, strength = es.KeyExtractor(profileType=prof)(audio)
        cands[prof] = {"key": f"{k} {s}", "strength": round(float(strength), 3)}
    # chord-duration histogram -> tonal centre
    dur = {}
    for c in chords:
        if c["chord"] != "N":
            dur[c["chord"]] = dur.get(c["chord"], 0.0) + c["end"] - c["start"]
    top = sorted(dur.items(), key=lambda kv: -kv[1])[:6]
    centre = top[0][0] if top else None
    pcs = None
    if melody_midi is not None and len(melody_midi):
        h = np.bincount(np.round(melody_midi).astype(int) % 12, minlength=12).astype(float)
        h /= h.sum()
        pcs = {FLAT_NAMES[i]: round(float(h[i]), 3) for i in np.argsort(-h)[:7]}
    votes = {}
    for v in cands.values():
        votes[v["key"]] = votes.get(v["key"], 0) + 1
    est_profiles = max(votes.items(), key=lambda kv: (kv[1], -list(votes).index(kv[0])))[0]
    # prefer the key implied by the chord progression (harmonic stems, drums/vocals removed); the whole-mix
    # profiles get pulled toward whatever note the melody sits on.
    est = chord_key["key"] if chord_key and chord_key.get("diatonic_share", 0) >= 0.6 else est_profiles
    return {"estimate": est, "from_chords": chord_key, "essentia_profiles": cands, "essentia_vote": est_profiles,
            "chord_centre": centre, "top_chords_seconds": {k: round(v, 1) for k, v in top},
            "melody_pitch_classes": pcs,
            "note": "estimate = key implied by the chord progression when >=60% of chord time is diatonic to it, "
                    "else the essentia profile vote. Profiles disagreeing => treat the key as ambiguous."}


# ----------------------------------------------------------------------------- melody + MIDI
def run_crepe(vocals_wav, device, model):
    import librosa
    import torch
    import torchcrepe
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    y, _ = librosa.load(vocals_wav, sr=16000, mono=True)
    x = torch.tensor(y)[None]
    f0, per = torchcrepe.predict(x, 16000, hop_length=160, fmin=65, fmax=1100, model=model, batch_size=1024,
                                 device=device, return_periodicity=True, decoder=torchcrepe.decode.viterbi)
    per = torchcrepe.filter.median(per, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0 = f0[0].cpu().numpy()
    per = per[0].cpu().numpy()
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=160)[0][: len(f0)]
    if len(rms) < len(f0):
        rms = np.pad(rms, (0, len(f0) - len(rms)))
    t = np.arange(len(f0)) * 0.01
    return t, f0, per, rms


def contour_to_notes(t, f0, per, rms, per_thr=0.5, min_len=0.08):
    import librosa
    gate = 10 ** (-45 / 20) * (np.max(rms) + 1e-9)
    voiced = (per >= per_thr) & (rms >= gate) & (f0 > 0)
    midi = np.where(voiced, librosa.hz_to_midi(np.maximum(f0, 1.0)), np.nan)
    notes = []
    cur = None
    for i in range(len(t)):
        m = midi[i]
        if np.isnan(m):
            if cur:
                notes.append(cur)
                cur = None
            continue
        r = int(round(m))
        if cur and abs(r - cur["p"]) <= 0:
            cur["end"] = i
            cur["vals"].append(m)
            cur["amp"].append(rms[i])
        elif cur and len(cur["vals"]) and abs(m - np.median(cur["vals"])) < 0.7:
            cur["end"] = i
            cur["vals"].append(m)
            cur["amp"].append(rms[i])
        else:
            if cur:
                notes.append(cur)
            cur = {"start": i, "end": i, "p": r, "vals": [m], "amp": [rms[i]]}
    if cur:
        notes.append(cur)
    out = []
    amax = max([max(n["amp"]) for n in notes] + [1e-9])
    for n in notes:
        s, e = t[n["start"]], t[n["end"]] + 0.01
        if e - s < min_len:
            continue
        pitch = int(round(np.median(n["vals"])))
        vel = int(np.clip(40 + 87 * (np.mean(n["amp"]) / amax) ** 0.5, 1, 127))
        out.append((float(s), float(e), pitch, vel))
    # merge same-pitch notes separated by < 40 ms
    merged = []
    for n in out:
        if merged and merged[-1][2] == n[2] and n[0] - merged[-1][1] < 0.04:
            merged[-1] = (merged[-1][0], n[1], n[2], max(merged[-1][3], n[3]))
        else:
            merged.append(n)
    return merged, midi


def notes_to_midi(notes, path, tempo, program=0, name="melody", is_drum=False):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo) if tempo else 120.0)
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    for s, e, p, v in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(s), end=float(e)))
    pm.instruments.append(inst)
    pm.write(path)
    return pm


def run_basic_pitch(wav, path, **kw):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    _, midi, notes = predict(wav, ICASSP_2022_MODEL_PATH, **kw)
    midi.write(path)
    return midi, len(notes)


def run_drums(drums_wav, sr=44100):
    """Kick/snare/hat onsets from the drum stem: onset strength in three mel bands, peak-picked."""
    import librosa
    y, sr = librosa.load(drums_wav, sr=sr, mono=True)
    hop = 256
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop, n_mels=128, fmax=sr / 2)
    mel_f = librosa.mel_frequencies(n_mels=128, fmax=sr / 2)
    bands = {"kick": (30, 150, 36), "snare": (150, 4000, 38), "hat": (6000, 16000, 42)}
    hits = {}
    notes = []
    for name, (lo, hi, pitch) in bands.items():
        idx = np.where((mel_f >= lo) & (mel_f < hi))[0]
        env = librosa.onset.onset_strength(S=librosa.power_to_db(S[idx]), sr=sr, hop_length=hop)
        env = env / (env.max() + 1e-9)
        pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=0.12, wait=int(0.07 * sr / hop))
        # loudness gate in the band: ignore peaks where the band is near silent
        band_e = librosa.power_to_db(S[idx].sum(axis=0), ref=np.max)
        pk = [p for p in pk if band_e[p] > -35]
        tt = librosa.frames_to_time(pk, sr=sr, hop_length=hop)
        hits[name] = rnd(tt, 3)
        for p, ti in zip(pk, tt):
            notes.append((float(ti), float(ti) + 0.08, pitch, int(np.clip(50 + 77 * env[p], 1, 127))))
    return hits, sorted(notes)


# ----------------------------------------------------------------------------- curves
def run_curves(signals, sr, duration, fps):
    import librosa
    n = int(np.ceil(duration * fps))
    hop = sr / fps
    out = {}
    for k, y in signals.items():
        idx = (np.arange(n) * hop).astype(int)
        win = int(hop)
        vals = np.array([np.sqrt(np.mean(y[i:i + win] ** 2) + 1e-12) for i in idx])
        dbv = 20 * np.log10(vals + 1e-9)
        out[k] = np.clip((dbv + 60.0) / 60.0, 0, 1)  # -60 dB..0 dB -> 0..1
    full = signals["full"]
    S = np.abs(librosa.stft(full, n_fft=2048, hop_length=512))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    tS = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    tF = np.arange(n) / fps
    for name, lo, hi in (("low", 20, 150), ("mid", 150, 2500), ("high", 2500, 16000)):
        e = S[(f >= lo) & (f < hi)].sum(axis=0)
        e = np.interp(tF, tS, e)
        out[name] = e / (np.percentile(e, 99) + 1e-9)
    on = librosa.onset.onset_strength(y=full, sr=sr, hop_length=512)
    on = np.interp(tF, tS[: len(on)], on)
    out["onset"] = on / (np.percentile(on, 99) + 1e-9)
    return {k: rnd(np.clip(v, 0, 1.5), 4) for k, v in out.items()}


# ----------------------------------------------------------------------------- allin1 (optional)
def run_allin1(wav, outdir, stems_root, model):
    import natten.functional as nf
    # all-in-one 1.1.0 calls the NATTEN <=0.15 names; NATTEN 0.17 renamed them (same tensor layout)
    nf.natten1dqkrpb = lambda q, k, rpb, ks, dil: nf.na1d_qk(q, k, kernel_size=ks, dilation=dil, rpb=rpb)
    nf.natten1dav = lambda a, v, ks, dil: nf.na1d_av(a, v, kernel_size=ks, dilation=dil)
    nf.natten2dqkrpb = lambda q, k, rpb, ks, dil: nf.na2d_qk(q, k, kernel_size=ks, dilation=dil, rpb=rpb)
    nf.natten2dav = lambda a, v, ks, dil: nf.na2d_av(a, v, kernel_size=ks, dilation=dil)
    import allin1
    od = os.path.join(outdir, "allin1")
    r = allin1.analyze(wav, out_dir=od, demix_dir=stems_root, spec_dir=os.path.join(od, "spec"), device="cpu",
                       model=model, keep_byproducts=True, multiprocess=False, overwrite=True)
    return {"model": model, "bpm": r.bpm,
            "segments": [{"start": round(s.start, 3), "end": round(s.end, 3), "label": s.label} for s in r.segments],
            "n_beats": len(r.beats), "downbeats": rnd(r.downbeats, 3)}


# ----------------------------------------------------------------------------- lip sync (optional)
RHUBARB = os.path.expanduser("~/apps/rhubarb-1.14.0/rhubarb")


def run_lipsync(vocals_wav, outdir):
    """Rhubarb Lip Sync (phonetic recogniser = language-independent, the better choice for singing).
    Mouth shapes A-H + X (rest), Preston Blair set; see Rhubarb's README for the shape sheet."""
    import librosa
    import soundfile as sf
    if not os.path.isfile(RHUBARB):
        raise FileNotFoundError(RHUBARB)
    y, _ = librosa.load(vocals_wav, sr=16000, mono=True)
    tmp = os.path.join(outdir, "_vocals16k.wav")
    sf.write(tmp, y, 16000, subtype="PCM_16")
    out = os.path.join(outdir, "lipsync_rhubarb.json")
    subprocess.run([RHUBARB, "-r", "phonetic", "-f", "json", "-q", tmp, "-o", out], check=True, cwd=outdir)
    os.remove(tmp)
    with open(out) as fh:
        cues = json.load(fh).get("mouthCues", [])
    return {"path": out, "n_cues": len(cues), "recognizer": "phonetic"}


# ----------------------------------------------------------------------------- plot
def plot_map(path, curves, fps, sections, drops, beats_down, duration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"intro": "#9db4c0", "build": "#f2c14e", "drop": "#e4572e", "breakdown": "#76b041", "bridge": "#5c80bc",
           "break": "#c9c9c9", "outro": "#9db4c0"}
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=100)
    t = np.arange(len(curves["full"])) / fps
    for s in sections:
        ax.axvspan(s["start"], s["end"], color=col.get(s["label"], "#ddd"), alpha=0.35, lw=0)
        ax.text((s["start"] + s["end"]) / 2, 1.02, s["label"], ha="center", va="bottom", fontsize=7)
    ax.plot(t, curves["full"], lw=0.6, color="#222", label="mix")
    ax.plot(t, curves["drums"], lw=0.5, color="#e4572e", alpha=0.8, label="drums")
    ax.plot(t, curves["bass"], lw=0.5, color="#5c80bc", alpha=0.8, label="bass")
    for d in drops:
        ax.axvline(d["time"], color="k", lw=1.2 if d["major"] else 0.6, ls="-" if d["major"] else "--")
    for x in beats_down:
        ax.axvline(x, color="#888", lw=0.2, alpha=0.4)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(np.arange(0, duration + 1, 10))
    ax.set_xticklabels([f"{int(x // 60)}:{int(x % 60):02d}" for x in np.arange(0, duration + 1, 10)], fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=6, loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("outdir")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda — cuda ONLY via scripts/server/render.sh")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of curves.json (match the scene)")
    ap.add_argument("--crepe-model", default="full", choices=["full", "tiny"])
    ap.add_argument("--allin1", default="off", choices=["off", "harmonix-fold0", "harmonix-all"],
                    help="add all-in-one section labels (CPU only: fold0 ~1-2 min, all = 8-fold ensemble ~8 min)")
    ap.add_argument("--lipsync", action="store_true",
                    help="mouth cues from the vocal stem with Rhubarb Lip Sync (MIT CLI, ~/apps/rhubarb-1.14.0)")
    ap.add_argument("--skip", default="", help="comma list of stages to skip: chords,melody,midi,drums,curves,plot,key")
    a = ap.parse_args()
    skip = set(x for x in a.skip.split(",") if x)
    wav = os.path.abspath(a.wav)
    outdir = os.path.abspath(a.outdir)
    os.makedirs(os.path.join(outdir, "midi"), exist_ok=True)
    import librosa
    import soundfile as sf

    info = sf.info(wav)
    duration = float(info.frames) / info.samplerate
    song = {"source": wav, "duration": round(duration, 3), "samplerate": info.samplerate,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": a.device}

    with Stage("stems"):
        stems, sdir = run_demucs(wav, outdir, a.device)
    song["stems"] = stems

    with Stage("beats"):
        beats, downbeats, tempo, bpb = run_beats(wav, a.device)
    song.update({"tempo_bpm": round(tempo, 2), "beats_per_bar": bpb, "beats": rnd(beats), "downbeats": rnd(downbeats)})

    sr = 22050
    with Stage("load"):
        sig = {k: librosa.load(p, sr=sr, mono=True)[0] for k, p in stems.items()}
        sig["full"] = librosa.load(wav, sr=sr, mono=True)[0]

    with Stage("structure"):
        rows = bar_table(sig, sr, downbeats, duration)
        labels = label_bars(rows)
        parts, sections, drops = structure_from_bars(rows, labels, duration)
    song["sections"] = sections
    song["parts"] = parts
    song["drops"] = drops
    song["bars"] = [{"start": round(r["start"], 3), "label": l,
                     **{k: round(r[k], 1) for k in ("full", "drums", "bass", "other", "vocals")}}
                    for r, l in zip(rows, labels)]

    chords, chord_key = [], None
    if "chords" not in skip:
        with Stage("chords"):
            harm = sig["other"] + sig["bass"]
            chords, chord_key = run_chords(harm, sr, beats, duration)
        song["chords"] = chords

    melody_midi_vals = None
    if "melody" not in skip:
        with Stage("melody_crepe"):
            t, f0, per, rms = run_crepe(stems["vocals"], a.device, a.crepe_model)
            notes, midi_curve = contour_to_notes(t, f0, per, rms)
            melody_midi_vals = midi_curve[~np.isnan(midi_curve)]
            cpath = os.path.join(outdir, "melody_contour.csv")
            with open(cpath, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["time_s", "f0_hz", "midi", "periodicity"])
                for i in range(len(t)):
                    m = midi_curve[i]
                    w.writerow([f"{t[i]:.2f}", f"{f0[i]:.2f}" if not np.isnan(m) else "", f"{m:.2f}" if not np.isnan(m) else "",
                                f"{per[i]:.3f}"])
            mpath = os.path.join(outdir, "midi", "melody_crepe.mid")
            notes_to_midi(notes, mpath, tempo, program=0, name="melody")
        song["melody"] = {"contour_csv": cpath, "midi": mpath, "n_notes": len(notes),
                          "range_midi": [int(np.percentile(melody_midi_vals, 5)), int(np.percentile(melody_midi_vals, 95))]
                          if len(melody_midi_vals) else None,
                          "first_notes": [{"t": round(s, 2), "dur": round(e - s, 2), "note": librosa.midi_to_note(p)}
                                          for s, e, p, v in notes[:24]]}

    midi_paths = {}
    if "midi" not in skip:
        with Stage("basic_pitch"):
            for stem, kw in (("vocals", {}),
                             ("bass", {"maximum_frequency": 400.0, "minimum_note_length": 100.0}),
                             ("other", {"minimum_note_length": 90.0})):
                p = os.path.join(outdir, "midi", ("vocals_basicpitch" if stem == "vocals" else stem) + ".mid")
                _, nn = run_basic_pitch(stems[stem], p, **kw)
                midi_paths[stem] = {"path": p, "n_notes": nn}

    drum_notes = []
    if "drums" not in skip:
        with Stage("drums"):
            hits, drum_notes = run_drums(stems["drums"])
            dp = os.path.join(outdir, "midi", "drums.mid")
            notes_to_midi(drum_notes, dp, tempo, name="drums", is_drum=True)
        midi_paths["drums"] = {"path": dp, "n_notes": len(drum_notes)}
        song["drum_hits"] = hits

    if midi_paths or "melody" in song:
        import pretty_midi
        allpm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo) if tempo else 120.0)
        srcs = []
        if "melody" in song:
            srcs.append(("melody", song["melody"]["midi"], 0, False))
        for k, prog in (("bass", 38), ("other", 81), ("drums", 0)):
            if k in midi_paths:
                srcs.append((k, midi_paths[k]["path"], prog, k == "drums"))
        for name, p, prog, isd in srcs:
            pm = pretty_midi.PrettyMIDI(p)
            inst = pretty_midi.Instrument(program=prog, is_drum=isd, name=name)
            for ins in pm.instruments:
                inst.notes.extend(ins.notes)
            allpm.instruments.append(inst)
        ap_ = os.path.join(outdir, "midi", "song_all.mid")
        allpm.write(ap_)
        midi_paths["all"] = {"path": ap_, "tracks": [s[0] for s in srcs]}
    song["midi"] = midi_paths

    if "key" not in skip:
        with Stage("key"):
            song["key"] = run_key(wav, chords, melody_midi_vals, chord_key)

    curves = None
    if "curves" not in skip:
        with Stage("curves"):
            curves = run_curves(sig, sr, duration, a.fps)
            cp = os.path.join(outdir, "curves.json")
            with open(cp, "w") as fh:
                json.dump({"fps": a.fps, "range": "0..1 (stems/full: -60..0 dBFS; bands/onset: /99th pct)",
                           "curves": curves}, fh, separators=(",", ":"))
        song["curves"] = {"path": cp, "fps": a.fps, "names": sorted(curves)}

    if a.lipsync:
        with Stage("lipsync"):
            song["lipsync"] = run_lipsync(stems["vocals"], outdir)

    if a.allin1 != "off":
        with Stage("allin1"):
            song["allin1"] = run_allin1(wav, outdir, os.path.join(outdir, "stems"), a.allin1)

    if curves and "plot" not in skip:
        with Stage("plot"):
            pp = os.path.join(outdir, "song_map.png")
            plot_map(pp, curves, a.fps, sections, drops, downbeats, duration)
        song["plot"] = pp

    song["timings_s"] = TIMINGS
    vers = {}
    for m in ("torch", "demucs", "beat_this", "basic_pitch", "torchcrepe", "librosa", "essentia", "allin1", "natten"):
        try:
            from importlib.metadata import version
            vers[m] = version(m.replace("_", "-"))
        except Exception:
            vers[m] = None
    song["tool_versions"] = vers
    out = os.path.join(outdir, "song_map.json")
    with open(out, "w") as fh:
        json.dump(song, fh, indent=1)
    log("wrote", out)
    log("tempo", song["tempo_bpm"], "| key", song.get("key", {}).get("estimate"), "| drops",
        [f"{int(d['time'] // 60)}:{d['time'] % 60:05.2f}{'' if d['major'] else '(minor)'}" for d in drops])
    for s in sections:
        log(f"   {int(s['start'] // 60)}:{s['start'] % 60:05.2f} - {int(s['end'] // 60)}:{s['end'] % 60:05.2f}  {s['label']}")


if __name__ == "__main__":
    main()
