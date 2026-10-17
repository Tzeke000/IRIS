"""lib/importer.py pure parts (numpy venv): ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_importer_pure.py'"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import importer as imp  # noqa: E402  (no bpy at import time)


class TestPure(unittest.TestCase):
    def test_label_islands(self):
        lab = imp.label_islands(7, [(0, 1), (1, 2), (3, 4), (5, 4)])
        self.assertEqual(list(lab), [0, 0, 0, 3, 3, 3, 6])
        self.assertEqual(list(imp.label_islands(3, np.zeros((0, 2)))), [0, 1, 2])
        # a long chain converges (pointer jumping)
        n = 1000
        self.assertTrue((imp.label_islands(n, [(i, i + 1) for i in range(n - 1)]) == 0).all())

    def test_normalize_matrix(self):
        M, s = imp.normalize_matrix((2, 4, 1), (4, 8, 5), height=2.0)
        self.assertEqual(s, 0.5)
        lo = M @ np.array([2, 4, 1, 1.0])
        hi = M @ np.array([4, 8, 5, 1.0])
        self.assertTrue(np.allclose(lo[:3], (-0.5, -1.0, 0.0)) and np.allclose(hi[:3], (0.5, 1.0, 2.0)))
        M2, s2 = imp.normalize_matrix((0, 0, -3), (1, 1, 3), scale=3.0, center=False)
        self.assertEqual(s2, 3.0)
        self.assertTrue(np.allclose((M2 @ np.array([0, 0, -3, 1.0]))[:3], (0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
