"""Pure parts of lib/qa/verify_saved.py and lib/assets/probe.py (numpy venv):
    ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_verify_probe_pure.py'"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.assets import probe  # noqa: E402
from lib.qa import verify_saved as V  # noqa: E402


class TestVerifyHelpers(unittest.TestCase):
    def test_frames(self):
        self.assertEqual(V.parse_frames("1,5-7, 10"), [1, 5, 6, 7, 10])
        self.assertEqual(V.sample_frames(1, 100, cuts=[31, 61]), [1, 30, 31, 50, 60, 61, 100])
        self.assertEqual(V.sample_frames(1, 10, extra=[0, 4, 99], every=5), [1, 4, 5, 6, 10])
        self.assertEqual(len(V.sample_frames(1, 10000, every=1, cap=50)), 50)

    def test_direction_spec(self):
        self.assertEqual(V.parse_direction("Car_:+Y, Drone:-x"), [("Car_", (0, 1, 0)), ("Drone", (-1, 0, 0))])
        with self.assertRaises(ValueError):
            V.parse_direction("Car_:forward")

    def test_bad_pixels(self):
        a = np.array([0.1, np.nan, np.inf, 2e4, -3e4, 5.0], np.float32)
        self.assertEqual(V.bad_pixels(a), (2, 2))


class TestProbeHelpers(unittest.TestCase):
    def test_scale_flags(self):
        self.assertEqual(probe.scale_flags([1.8, 0.5, 0.3]), [])
        self.assertIn("centimetre", probe.scale_flags([180.0, 50.0, 30.0])[0])
        self.assertIn("tiny", probe.scale_flags([0.002, 0.001, 0.001])[0])
        self.assertEqual(probe.scale_flags(None), ["no mesh geometry"])

    def test_summary_line(self):
        rec = dict(size_m=[1, 2, 3], totals=dict(polys=12), meshes=[{}], materials=[], armatures=[], actions=[],
                   longest_axis="Z", flags=["unlit (x)"])
        line = probe.summary_line("/data/a/slug/model.glb", rec)
        self.assertIn("slug/model.glb", line)
        self.assertIn("FLAGS: unlit", line)


if __name__ == "__main__":
    unittest.main()
