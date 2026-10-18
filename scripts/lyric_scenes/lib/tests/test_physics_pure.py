"""lib/physics.py pure parts (numpy venv): ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_physics_pure.py'"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import bpyx as X  # noqa: E402
from lib import physics as PH  # noqa: E402


class TestDecompose(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(2)
        e = rng.uniform(-3, 3, (20, 3))
        s = rng.uniform(0.2, 3, (20, 3))
        t = rng.normal(size=(20, 3)) * 10
        M = np.tile(np.eye(4), (20, 1, 1))
        M[:, :3, :3] = X.euler_to_mat(e) * s[:, None, :]
        M[:, :3, 3] = t
        loc, R, sc = PH.decompose(M)
        self.assertTrue(np.allclose(loc, t) and np.allclose(sc, s) and np.allclose(R, X.euler_to_mat(e)))

    def test_mirror_goes_into_x_scale(self):
        M = np.eye(4)
        M[0, 0] = -2.0
        loc, R, s = PH.decompose(M[None])
        self.assertAlmostEqual(np.linalg.det(R[0]), 1.0)
        self.assertTrue(np.allclose(R[0] * s[0][None, :], M[:3, :3]))


class TestHoldAndGround(unittest.TestCase):
    def test_hold_extend(self):
        out = PH.hold_extend(np.array([1.0, 2.0, 3.0]), start=4, n=8)
        self.assertEqual(list(out), [1, 1, 1, 1, 2, 3, 3, 3])
        with self.assertRaises(ValueError):
            PH.hold_extend(np.zeros(5), start=6, n=8)

    def test_lowest_columns(self):
        pts = [(0, 0, 5), (0, 0, 1), (0.01, 0, 0.5), (3, 0, 2), (3, 0, 9)]
        cols = PH.lowest_columns(pts, cell=0.1)
        self.assertEqual(sorted(map(tuple, cols.tolist())), [(0.01, 0, 0.5), (3, 0, 2)])

    def test_gap_of(self):
        slope = lambda x, y: 0.1 * np.asarray(x)                   # noqa: E731
        g, i = PH.gap_of([(0, 0, 1.0), (10, 0, 1.2), (5, 0, 3)], slope)
        self.assertAlmostEqual(g, 0.2)
        self.assertEqual(i, 1)
        none = lambda x, y: np.full(np.shape(x), np.nan)            # noqa: E731
        g, i = PH.gap_of([(0, 0, 1)], none)
        self.assertTrue(math.isnan(g) and i == -1)


if __name__ == "__main__":
    unittest.main()
