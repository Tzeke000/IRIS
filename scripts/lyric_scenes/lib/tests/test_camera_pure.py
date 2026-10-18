"""lib/camera.py pure parts (numpy venv): ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_camera_pure.py'"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import bpyx as X  # noqa: E402
from lib import camera as CM  # noqa: E402


def dolly(i, u, n):
    return dict(cam=(0.0, -20.0 + 10.0 * u / n, 2.0), tgt=(0.0, 0.0, 2.0), lens=35.0, fstop=4.0, haze=0.5 * u / n)


def orbit(i, u, n):
    a = 4 * math.pi * u / n
    return dict(cam=(10 * math.cos(a), 10 * math.sin(a), 3.0), tgt=(0.0, 0.0, 1.0), lens=50.0, fstop=2.8, focus=9.0,
                haze=1.0)


SHOTS = [CM.Shot(31, "orbit", orbit), CM.Shot(1, "dolly", dolly)]          # unsorted on purpose


class TestPlan(unittest.TestCase):
    def test_sorted_index_and_cuts(self):
        shots, idx, cuts = CM.plan(SHOTS, 60)
        self.assertEqual([s.name for s in shots], ["dolly", "orbit"])
        self.assertEqual(cuts, [31])
        self.assertEqual(idx[29], 0)
        self.assertEqual(idx[30], 1)                                      # index 30 = Blender frame 31 = first of orbit

    def test_errors(self):
        f = SHOTS[0].fn
        for bad in ([CM.Shot(2, "a", f)], [CM.Shot(1, "a", f), CM.Shot(1, "b", f)], [CM.Shot(1, "a", f), CM.Shot(5, "a", f)],
                    [CM.Shot(1, "a", f), CM.Shot(99, "b", f)], []):
            with self.assertRaises(ValueError):
                CM.plan(bad, 60)


class TestEval(unittest.TestCase):
    A = CM.eval_shots(SHOTS, 60)

    def test_values_and_defaults(self):
        A = self.A
        self.assertTrue(np.allclose(A["loc"][0], (0, -20, 2)) and np.allclose(A["loc"][29], (0, -10, 2)))
        self.assertAlmostEqual(A["focus"][0], 20.0)                       # default |tgt − cam|
        self.assertEqual(A["focus"][40], 9.0)                             # explicit focus
        self.assertEqual(A["lens"][30], 50.0)
        self.assertEqual(A["cuts"], [31])
        self.assertEqual(A["extra"]["haze"].shape, (60,))
        self.assertEqual(A["extra"]["haze"][30], 1.0)
        self.assertTrue(np.allclose(X.euler_to_mat(A["rot"]), A["R"], atol=1e-9))

    def test_euler_resets_at_cut_and_is_continuous_inside(self):
        rot = self.A["rot"]
        self.assertTrue(np.allclose(rot[30], X.mat_to_euler(self.A["R"][30])))
        self.assertLess(np.abs(np.diff(rot[30:], axis=0)).max(), 0.6)    # two turns in 30 frames, no flips
        self.assertGreater(rot[-1, 2] - rot[30, 2], 3 * math.pi)

    def test_bake_plan_holds_every_channel_at_the_cut(self):
        for key in ("lens", "focus"):
            f, v, names = X.bake_plan(self.A[key], cuts=self.A["cuts"])
            self.assertEqual(names[list(f).index(30.0)], "CONSTANT")

    def test_tuple_form_post_hook_and_no_dof(self):
        shots = [CM.Shot(1, "t", lambda i, u, n: ((0, -5, 1), (0, 0, 1), 24.0))]
        calls = []

        def post(i, d, sh):
            calls.append(sh.name)
            d["cam"] = tuple(np.asarray(d["cam"]) + (0, 0, 0.1 * i))
            return d
        A = CM.eval_shots(shots, 10, post=post)
        self.assertIsNone(A["fstop"])
        self.assertEqual(len(calls), 10)
        self.assertAlmostEqual(A["loc"][9, 2], 1.9)

    def test_mixed_fstop_and_nan_raise(self):
        with self.assertRaises(ValueError):
            CM.eval_shots([CM.Shot(1, "a", dolly), CM.Shot(5, "b", lambda i, u, n: ((0, -5, 1), (0, 0, 1), 24.0))], 10)
        with self.assertRaises(ValueError):
            CM.eval_shots([CM.Shot(1, "a", lambda i, u, n: ((0, -5, float("nan")), (0, 0, 1), 24.0))], 4)
        with self.assertRaises(KeyError):
            CM.eval_shots([CM.Shot(1, "a", lambda i, u, n: dict(cam=(0, 0, 0), lens=20))], 4)

    def test_on_screen(self):
        A = self.A
        self.assertTrue(CM.on_screen(A, (0.0, 0.0, 2.0))[:30].all())     # the dolly target sits at frame centre
        self.assertFalse(CM.on_screen(A, (0.0, -40.0, 2.0))[:30].any())  # behind the camera
        side = np.c_[np.full(60, 100.0), np.zeros(60), np.full(60, 2.0)]   # far off to the side during the dolly
        self.assertFalse(CM.on_screen(A, side)[:30].any())
        fr = CM.on_screen_by_shot(A, (0.0, 0.0, 1.0))
        self.assertEqual(fr["orbit"], 1.0)

    def test_half_tangents_and_dof(self):
        tx, ty = CM.half_tangents(36.0)
        self.assertAlmostEqual(tx, 0.5)
        self.assertAlmostEqual(ty, 0.5 * 9 / 16)
        tx, ty = CM.half_tangents(36.0, aspect=9 / 16)                    # portrait: vertical spans the sensor
        self.assertAlmostEqual(ty, 0.5)
        near, far = CM.dof_limits(50, 2.8, 5.0)
        self.assertTrue(near < 5.0 < far < math.inf)
        H = (0.05 ** 2) / (8 * 0.00003) + 0.05
        self.assertEqual(CM.dof_limits(50, 8, H + 1)[1], math.inf)


if __name__ == "__main__":
    unittest.main()
