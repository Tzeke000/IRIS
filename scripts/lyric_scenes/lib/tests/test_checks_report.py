"""lib/checks.py (pure) + lib/report.py (numpy venv):
    ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_checks_report.py'"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import checks as K  # noqa: E402
from lib.report import Report  # noqa: E402


class TestChecks(unittest.TestCase):
    def test_direction_catches_a_backwards_car(self):
        t = np.linspace(0, 10, 50)
        pos = np.c_[t, np.zeros(50), np.zeros(50)]                 # drives +X
        self.assertTrue(K.direction_check(pos, (1, 0, 0))["ok"])
        r = K.direction_check(pos, (-1, 0, 0))                      # model's nose points backwards
        self.assertFalse(r["ok"])
        self.assertAlmostEqual(r["worst_dot"], -1.0)
        parked = np.zeros((10, 3))
        self.assertEqual(K.direction_check(parked, (0, 1, 0))["n_checked"], 0)

    def test_box_distance_and_clearance(self):
        boxes = [(0, 0, 0, 1, 1, 1), (10, 0, 0, 11, 1, 5)]
        d = K.box_distance([(0.5, 0.5, 0.5), (2, 0.5, 0.5), (10.5, 3, 6)], boxes)
        self.assertEqual(d[0, 0], 0.0)
        self.assertAlmostEqual(d[1, 0], 1.0)
        self.assertAlmostEqual(d[2, 1], np.hypot(2, 1))
        dmin, pi, bi = K.clearance([(5, 0.5, 0.5), (9.5, 0.5, 2)], boxes)
        self.assertEqual((round(dmin, 6), pi, bi), (0.5, 1, 1))
        self.assertEqual(K.inside_count([(0.5, 0.5, 0.5), (3, 3, 3), (1.05, 0.5, 0.5)], boxes, pad=0.1), 2)

    def test_floating_and_finite_and_spacing(self):
        pts = [(0, 0, 0.1), (1, 0, 2.0), (2, 0, 40.0), (3, 0, -1.0)]
        self.assertEqual(list(K.floating(pts, lambda x, y: np.zeros_like(x))), [1])
        self.assertTrue(K.finite("ok", [1, 2]))
        with self.assertRaises(ValueError):
            K.finite("cam", [[0, 0, 0], [0, np.nan, 0]])
        sw = np.zeros((2, 3, 3))
        sw[:, 1, 0] = 5
        sw[:, 2, 0] = [10, 6.5]
        self.assertEqual(K.min_spacing(sw)[:2], (1.5, 1))
        with self.assertRaises(AssertionError):
            K.check_order(IGN=5, LIFT=4)


class TestReport(unittest.TestCase):
    def test_saves_on_success_soft_fail_and_exception(self):
        p = os.path.join(tempfile.mkdtemp(), "r", "build_report.json")
        with Report(p, "t", quiet=True) as R:
            R.add("hoard.coins", np.int64(12))
            R.add("cam", np.array([1.0, 2.0]))
            R.tick("a")
            R.check(False, "soft", "not fatal")
        d = json.load(open(p))
        self.assertEqual(d["values"]["hoard"]["coins"], 12)
        self.assertEqual(d["status"], "ok with 1 soft failures")
        self.assertIn("a", d["timings"])
        with self.assertRaises(AssertionError):
            with Report(p, "t", quiet=True) as R:
                R.assert_(False, "cam_clearance", "camera 0.1 m from a wall")
        d = json.load(open(p))
        self.assertEqual(d["status"], "failed: cam_clearance")
        with self.assertRaises(KeyError):
            with Report(p, "t", quiet=True) as R:
                {}["x"]
        d = json.load(open(p))
        self.assertTrue(d["status"].startswith("error: KeyError"))
        self.assertIn("traceback", d)

    def test_carries_importer_log_for_credits(self):
        from lib import importer
        importer.LOG.append({"path": "/data/iris_assets/x/model.glb"})
        importer.USED_FILES.add("/data/iris_assets/x/tex.png")
        try:
            p = os.path.join(tempfile.mkdtemp(), "b.json")
            Report(p, quiet=True).save()
            d = json.load(open(p))
            self.assertEqual(d["files"], ["/data/iris_assets/x/tex.png"])
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"))
            import credits
            self.assertEqual(credits.used_files([p]), ["/data/iris_assets/x/model.glb", "/data/iris_assets/x/tex.png"])
        finally:
            importer.LOG.clear()
            importer.USED_FILES.clear()


if __name__ == "__main__":
    unittest.main()
