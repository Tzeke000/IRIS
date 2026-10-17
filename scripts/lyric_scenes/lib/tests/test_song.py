"""lib/song.py tests (numpy venv):  ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_song.py'
The Be My One check needs /data/iris_assets/bemyone/analysis/song_map.json (skipped if absent)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(LIB))
from lib import song as S  # noqa: E402

BMO = "/data/iris_assets/bemyone/analysis/song_map.json"
REF_BPM, REF_T0 = 134.095, 0.09864          # rooftop/rt_util.py, hand-fitted with kick_grid.py from the audio


def synthetic_map(bpm=128.0, t0=0.21, dur=60.0, jitter=0.004, seed=1):
    rng = np.random.default_rng(seed)
    p = 60 / bpm
    beats = t0 + p * np.arange(int((dur - t0) / p) + 1)
    kicks = np.sort(np.r_[beats[::1] + rng.normal(0, jitter, len(beats)), beats[2::8] + p / 2])   # + offbeat kicks
    return {"duration": dur, "tempo_bpm": round(bpm * 0.998, 2), "beats_per_bar": 4,
            "beats": list(np.round(beats + 0.01, 3)), "downbeats": list(np.round(beats[1::4], 3)),
            "drum_hits": {"kick": list(np.round(kicks, 4)), "snare": [1.0, 2.0, 2.5]},
            "sections": [{"label": "intro", "start": 0.0, "end": 20.0}, {"label": "drop", "start": 20.0, "end": 40.0},
                         {"label": "outro", "start": 40.0, "end": 60.0}],
            "drops": [{"n": 1, "time": 20.05}]}


class TestPure(unittest.TestCase):
    def test_gsmooth_preserves_length_and_mean(self):
        x = np.zeros(101)
        x[50] = 1
        y = S.gsmooth(x, 3)
        self.assertEqual(len(y), 101)
        self.assertAlmostEqual(y.sum(), 1.0, places=6)
        self.assertTrue(np.array_equal(S.gsmooth(x, 0), x))

    def test_hit_age_and_env(self):
        a = S.hit_age([2, 5], 8)
        self.assertTrue(np.isinf(a[:2]).all())
        self.assertEqual(list(a[2:]), [0, 1, 2, 0, 1, 2])
        e = S.age_env(a, 2.0)
        self.assertEqual(e[0], 0.0)
        self.assertAlmostEqual(e[3], np.exp(-0.5))

    def test_pnorm(self):
        y = S.pnorm(np.r_[np.arange(100), 1e6])
        self.assertEqual(y.max(), 1.0)
        self.assertEqual(y.min(), 0.0)

    def test_fit_grid_recovers_synthetic_tempo(self):
        m = synthetic_map()
        bpm, t0, st = S.fit_grid(m["drum_hits"]["kick"], 120, 136)
        self.assertAlmostEqual(bpm, 128.0, delta=0.01)
        self.assertAlmostEqual(t0, 0.21, delta=0.006)


class TestSong(unittest.TestCase):
    def setUp(self):
        self.s = S.Song(synthetic_map(), fps=30)

    def test_frames_and_grid(self):
        s = self.s
        self.assertEqual(s.n, 1800)
        self.assertEqual(s.fr(0.0), 1)
        self.assertEqual(s.idx(1.0), 30)
        self.assertAlmostEqual(s.bt(4) - s.bt(0), 4 * 60 / 128, delta=1e-3)
        self.assertEqual(s.fit_from, "kick")

    def test_downbeat_from_analysis_vote_then_ear(self):
        s = self.s
        self.assertEqual(s.downbeat_from, "analysis downbeats")
        self.assertEqual(s.downbeat_mod, 1)                    # synthetic downbeats = beats[1::4]
        ear = S.Song(synthetic_map(), 30, ear={"drops": [s.bt(42)]})   # beat 42 ≡ 2 mod 4
        self.assertEqual(ear.downbeat_mod, 2)
        self.assertEqual(ear.downbeat_from, "ear drop")

    def test_snap(self):
        s = self.s
        t = s.bt(9) + 0.05                                     # beat 9 ≡ 1 → a downbeat
        self.assertAlmostEqual(s.snap(t), s.bt(9))
        self.assertAlmostEqual(s.snap(s.bt(11)), s.bt(9))      # exactly between downbeats 9 and 13: tie → EARLIER
        self.assertAlmostEqual(s.snap(s.bt(10)), s.bt(9))
        self.assertAlmostEqual(s.snap(s.bt(10) + 0.01, "beat"), s.bt(10))
        self.assertTrue(all((s.beat_index(d) - s.downbeat_mod) % 4 == 0 for d in s.downbeats))

    def test_ear_outranks_sections_drops_and_grid(self):
        ear = {"sections": [{"label": "bridge", "start": 10, "end": 30}], "drops": [{"time": 30.0}],
               "bpm": 127.5, "beat0": 0.3}
        s = S.Song(synthetic_map(), 30, ear=ear)
        self.assertEqual([x["label"] for x in s.sections], ["bridge"])
        self.assertEqual(s.drops, [30.0])
        self.assertEqual((s.bpm, s.beat0, s.fit_from), (127.5, 0.3, "ear"))

    def test_mask_and_errors(self):
        s = self.s
        m = s.mask("drop")
        self.assertEqual(m.sum(), 600)
        self.assertEqual(s.mask((0.0, 1.0)).sum(), 30)
        self.assertEqual(s.mask(["intro", (40.0, 41.0)]).sum(), 630)
        with self.assertRaises(KeyError):
            s.mask("chorus")

    def test_hits_env(self):
        s = self.s
        self.assertEqual(list(s.hits("snare")), [30, 60, 75])
        e = s.env("snare", 3)
        self.assertEqual(e[29], 0.0)
        self.assertEqual(e[60], 1.0)
        b = s.beat_env(2, every=4)
        self.assertEqual(int((b == 1.0).sum()), len(s.downbeats))

    def test_check_order(self):
        self.assertTrue(S.Song.check_order(A=1, B=5, C=9))
        with self.assertRaises(AssertionError):
            S.Song.check_order(IGN=10, BUILD=10)

    def test_load_with_ear_file_and_curves(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "song_map.json")
        json.dump(synthetic_map(), open(p, "w"))
        json.dump({"drops": [25.0]}, open(os.path.join(d, "ear.json"), "w"))
        json.dump({"fps": 10.0, "curves": {"bass": list(np.linspace(0, 1, 600))}}, open(os.path.join(d, "curves.json"), "w"))
        s = S.Song.load(p)
        self.assertEqual(s.drops, [25.0])
        c = s.curve("bass")
        self.assertEqual(len(c), s.n)                           # 10 fps curve resampled to 30 fps
        self.assertAlmostEqual(c[900], 0.5, delta=0.01)
        self.assertIn("grid", s.report([25.0]))


@unittest.skipUnless(os.path.isfile(BMO), "Be My One song_map not on disk")
class TestBeMyOne(unittest.TestCase):
    """The audit's acceptance test: reproduce rt_util's hand-fitted grid within ONE frame across the whole song."""

    def test_grid_matches_rt_util_within_one_frame(self):
        s = S.Song.load(BMO, 30, ear={"drops": [29.0]})       # Zeke's ear: first drop 0:29
        ref = REF_T0 + np.arange(int((128.4 - REF_T0) * REF_BPM / 60) + 1) * 60 / REF_BPM
        ours = np.array([s.bt(s.beat_index(t)) for t in ref])
        off = np.abs(ours - ref).max() * 30
        print(f"\n[test_song] Be My One: {s.bpm:.3f} BPM beat0 {s.beat0:.5f} vs {REF_BPM}/{REF_T0}: "
              f"max offset {off:.2f} frames over {len(ref)} beats")
        self.assertLess(off, 1.0)
        self.assertEqual(s.n, 3852)                             # rt_util NF
        k = s.beat_index(29.0)                                  # downbeat anchored to the ear drop
        self.assertEqual((k - s.downbeat_mod) % 4, 0)
        self.assertAlmostEqual(s.curve("full").shape[0], 3852)


if __name__ == "__main__":
    unittest.main()
