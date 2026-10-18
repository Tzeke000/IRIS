"""lib/bpyx.py pure parts (numpy venv): ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_bpyx_pure.py'"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import bpyx as X  # noqa: E402


def rocket_reference(v, cut_idx):
    """rocket4/police scene_helpers.bake key selection, verbatim (N = len(v); CUT_SET[c-1] for 0-based cut index c)."""
    N = len(v)
    CUT_SET = np.zeros(N, bool)
    for c in cut_idx:
        if 0 < c < N:
            CUT_SET[c - 1] = True
    keep = np.ones(N, bool)
    keep[1:-1] = (v[1:-1] != v[:-2]) | (v[1:-1] != v[2:])
    keep |= CUT_SET
    keep[np.minimum(np.nonzero(CUT_SET)[0] + 1, N - 1)] = True
    ix = np.nonzero(keep)[0]
    const = CUT_SET[ix]
    return ix + 1, v[ix], const


class TestBakePlan(unittest.TestCase):
    def test_matches_rocket_on_random_plateaus_and_cuts(self):
        rng = np.random.default_rng(7)
        for trial in range(50):
            n = int(rng.integers(5, 400))
            v = np.repeat(rng.integers(0, 4, n), rng.integers(1, 6, n))[:n].astype(float)   # lots of plateaus
            cut_idx = sorted(set(rng.integers(1, n, int(rng.integers(0, 6))).tolist()))
            rf, rv, rc = rocket_reference(v, cut_idx)
            f, vv, names = X.bake_plan(v, interp="LINEAR", cuts=[c + 1 for c in cut_idx])   # Blender frames
            self.assertTrue(np.array_equal(f, rf), f"trial {trial}")
            self.assertTrue(np.array_equal(vv, rv))
            self.assertEqual([x == "CONSTANT" for x in names], list(rc))

    def test_rocket_const_true_is_all_constant(self):
        f, v, names = X.bake_plan([0, 1, 1, 1, 2], interp="CONSTANT")
        self.assertEqual(set(names), {"CONSTANT"})
        self.assertEqual(list(f), [1, 2, 4, 5])          # interior plateau key (frame 3) thinned

    def test_rooftop_behaviour_with_thin_false(self):
        f, v, names = X.bake_plan([3, 3, 3, 3], thin=False, interp="BEZIER")
        self.assertEqual(list(f), [1, 2, 3, 4])
        self.assertEqual(names, ["BEZIER"] * 4)

    def test_explicit_frames_and_cut_hold(self):
        f, v, names = X.bake_plan([0, 5, 5, 9], frames=[10, 20, 30, 40], cuts=[30])
        self.assertEqual(list(f), [10, 20, 30, 40])
        self.assertEqual(names, ["LINEAR"] * 4)          # no key at frame 29 → nothing to hold (sparse keys)
        f, v, names = X.bake_plan(np.zeros(10), cuts=[6])
        self.assertEqual(list(f), [1, 5, 6, 10])
        self.assertEqual(names, ["LINEAR", "CONSTANT", "LINEAR", "LINEAR"])

    def test_scalar_and_errors(self):
        f, v, names = X.bake_plan(2.5)
        self.assertEqual((list(f), list(v)), ([1.0], [2.5]))
        with self.assertRaises(ValueError):
            X.bake_plan([1, 2], frames=[1, 2, 3])
        with self.assertRaises(ValueError):
            X.bake_plan([1, 2], interp="SINE")


class TestMath(unittest.TestCase):
    def test_hash01_stable(self):
        self.assertEqual(X.hash01(1, 2, 3), X.hash01(1, 2, 3))
        self.assertNotEqual(X.hash01(1, 2, 3), X.hash01(1, 2, 4))
        vals = [X.hash01(i) for i in range(2000)]
        self.assertTrue(0 <= min(vals) and max(vals) <= 1 and abs(np.mean(vals) - 0.5) < 0.03)

    def test_steps_and_ramp(self):
        self.assertEqual(float(X.smoothstep(0.5)), 0.5)
        self.assertEqual(float(X.smootherstep(1.5)), 1.0)
        self.assertEqual(list(X.ramp([0, 5, 10, 20], 0, 10)), [0, 0.5, 1, 1])


class TestRotations(unittest.TestCase):
    def test_track_matrix_is_a_camera_rotation(self):
        R = X.track_matrix((0.0, 1.0, 0.0))                       # looking along +Y: right = +X, up = +Z
        self.assertTrue(np.allclose(R, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-12))
        rng = np.random.default_rng(3)
        f = rng.normal(size=(200, 3))
        Rs = X.track_matrix(f, roll=rng.uniform(-1, 1, 200))
        self.assertTrue(np.allclose(Rs @ Rs.transpose(0, 2, 1), np.eye(3), atol=1e-9))
        self.assertTrue(np.allclose(np.linalg.det(Rs), 1.0))
        self.assertTrue(np.allclose(-Rs[:, :, 2], f / np.linalg.norm(f, axis=1, keepdims=True)))   # −Z = forward

    def test_straight_up_does_not_nan(self):
        R = X.track_matrix((0.0, 0.0, 5.0))
        self.assertTrue(np.isfinite(R).all() and abs(np.linalg.det(R) - 1) < 1e-9)

    def test_euler_round_trip(self):
        rng = np.random.default_rng(5)
        for e in rng.uniform(-3, 3, (300, 3)):
            R = X.euler_to_mat(e)
            self.assertTrue(np.allclose(X.euler_to_mat(X.mat_to_euler(R)), R, atol=1e-9))
            self.assertTrue(np.allclose(X.euler_to_mat(X.mat_to_euler(R, prev=e + 0.01)), R, atol=1e-9))
            self.assertTrue(np.allclose(X.mat_to_euler(R, prev=e), e, atol=1e-9))     # same angles when prev agrees

    def test_euler_seq_is_continuous_through_turns_and_resets_at_cuts(self):
        a = np.linspace(0, 6 * np.pi, 400)                         # orbit: three full turns, slight look-down
        fwd = np.stack([-np.cos(a), -np.sin(a), np.full_like(a, -0.3)], 1)
        R = X.track_matrix(fwd)
        e = X.euler_seq(R)
        self.assertLess(np.abs(np.diff(e, axis=0)).max(), 0.1)    # no ±2π / ±π jumps between frames
        self.assertGreater(e[-1, 2] - e[0, 2], 5.5 * np.pi)       # z keeps winding past π (keys interpolate the short way)
        self.assertTrue(np.allclose(X.euler_to_mat(e), R, atol=1e-9))
        e2 = X.euler_seq(R, resets=[200])
        self.assertTrue(np.allclose(e2[200], X.mat_to_euler(R[200])))   # the cut starts from the canonical solution
        self.assertTrue(np.allclose(X.euler_to_mat(e2), R, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
