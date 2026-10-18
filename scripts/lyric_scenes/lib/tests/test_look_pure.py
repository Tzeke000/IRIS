"""lib/look.py pure parts (numpy venv): ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_look_pure.py'"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import look as L  # noqa: E402

MSG = ("bpy_struct: item.attr = val: enum \"Punchy\" not found in ('None', 'AgX - Punchy', 'AgX - Greyscale', "
       "'AgX - Very High Contrast', 'AgX - High Contrast', 'AgX - Medium High Contrast', 'AgX - Base Contrast', "
       "'AgX - Medium Low Contrast', 'AgX - Low Contrast', 'AgX - Very Low Contrast')")    # Blender 5.2.1 verbatim


class TestLookNames(unittest.TestCase):
    def test_error_parse_matches_the_documented_list(self):
        valid = L._valid_from_error(MSG)
        self.assertEqual(set(valid) - {"None"}, set(L.AGX_LOOKS))

    def test_resolve(self):
        valid = L._valid_from_error(MSG)
        self.assertEqual(L.resolve_look("AgX", "Punchy", valid), "AgX - Punchy")
        self.assertEqual(L.resolve_look("AgX", "AgX - Base Contrast", valid), "AgX - Base Contrast")
        self.assertEqual(L.resolve_look("AgX", None, valid), "None")
        with self.assertRaises(ValueError):
            L.resolve_look("AgX", "Medium Contrast", valid)


class TestHdriMaths(unittest.TestCase):
    def test_equirect_dir(self):
        self.assertTrue(np.allclose(L.equirect_dir(0.5, 0.5), (1, 0, 0), atol=1e-12))   # image centre = +X (measured)
        self.assertTrue(np.allclose(L.equirect_dir(0.25, 0.5), (0, 1, 0), atol=1e-12))  # left quarter = +Y
        self.assertTrue(np.allclose(L.equirect_dir(10.5 / 64, 0.5)[:2] @ (np.cos(np.radians(121.0)),
                                                                          np.sin(np.radians(121.0))), 1.0, atol=1e-3))
        self.assertTrue(np.allclose(L.equirect_dir(0.3, 1.0), (0, 0, 1), atol=1e-12))
        for u, v in np.random.default_rng(1).uniform(0, 1, (50, 2)):
            self.assertAlmostEqual(np.linalg.norm(L.equirect_dir(u, v)), 1.0)

    def test_brightest_and_rotation(self):
        w, h = 64, 32
        px = np.zeros((h, w, 4), np.float32)
        px[24, 40, :3] = 50.0                                                     # row 24 from the BOTTOM
        az, el, (ix, iy) = L.brightest(px.ravel(), w, h)
        self.assertEqual((ix, iy), (40, 24))
        d = L.equirect_dir((40.5) / w, (24.5) / h)
        self.assertAlmostEqual(az, math.atan2(d[1], d[0]))
        self.assertAlmostEqual(el, math.asin(d[2]))
        rot = L.hdri_rotation(az, math.radians(118))
        self.assertTrue(-math.pi <= rot < math.pi)
        self.assertAlmostEqual((az - rot - math.radians(118) + math.pi) % (2 * math.pi) - math.pi, 0.0)


if __name__ == "__main__":
    unittest.main()
