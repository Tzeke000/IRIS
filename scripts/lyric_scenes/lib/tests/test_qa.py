"""lib/qa pure-Python tests (PIL venv):  ~/venvs/iris-v100/bin/python -I -m unittest discover -s <lib>/tests -p 'test_qa.py'
The bpy half (time_ab + render/patch inside Blender) is tests/bpy_test_qa.py."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LIB, "qa"))
sys.path.insert(0, os.path.join(LIB, "render"))
import crops  # noqa: E402
import patch  # noqa: E402
import sheet  # noqa: E402
import time_ab  # noqa: E402


def noise(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


class TestSheet(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def test_fit_jpeg_guarantees_limits_on_worst_case(self):
        # noise at a 12 KB cap does not fit at q36 / 900 px — the old copies gave up there; this must keep shrinking
        out = os.path.join(self.td, "big.jpg")
        r = sheet.fit_jpeg(noise(4000, 3000), out, max_bytes=12_000)
        self.assertLess(os.path.getsize(out), 12_000)
        self.assertLessEqual(max(Image.open(out).size), 900)
        self.assertLess(r["scale"], 900 / 4000)        # had to shrink below the plain 900-px fit
        sheet.fit_jpeg(noise(4000, 3000), out)         # default limits
        self.assertLess(os.path.getsize(out), 150_000)
        self.assertLessEqual(max(Image.open(out).size), 900)

    def test_sheet_from_files_labels_and_limits(self):
        files = []
        for k in range(7):
            p = os.path.join(self.td, f"shot_{k}.png")
            noise(1920, 1080, k).save(p)
            files.append(p)
        out = os.path.join(self.td, "sheet.jpg")
        r = sheet._main([out, "--cols", "3", *files])
        self.assertEqual(r["n"], 7)
        im = Image.open(out)
        self.assertLessEqual(max(im.size), 900)
        self.assertLess(os.path.getsize(out), 150_000)
        self.assertGreater(im.height, 3 * 900 // 3 * 1080 // 1920 * 0.9)   # 3 rows of 16:9 cells

    def test_dir_glob_skips_tmp_and_every(self):
        for k in range(6):
            noise(320, 180, k).save(os.path.join(self.td, f"f_{k:04d}.png"))
        noise(320, 180).save(os.path.join(self.td, ".tmp_f_0009.png"))
        out = os.path.join(self.td, "d.jpg")
        r = sheet._main([out, "--dir", self.td, "--glob", "*f_*.png", "--every", "2", "--label", "index"])
        self.assertEqual(r["n"], 3)

    def test_mixed_aspect_letterboxed(self):
        ims = [noise(400, 400), noise(1600, 400)]
        g = sheet.grid(ims, ["sq", "wide"], cols=2, width=800)
        self.assertEqual(g.size, (800, 400))           # cell height from the tallest aspect (1:1 → 400)

    def test_shrink_cli(self):
        src = os.path.join(self.td, "huge.png")
        noise(3000, 2000).save(src)
        r = sheet._main(["shrink", src])
        self.assertTrue(r["path"].endswith("huge.safe.jpg"))
        self.assertLess(r["bytes"], 150_000)

    def test_video_times(self):
        v = os.path.join(self.td, "t.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=4",
                        "-pix_fmt", "yuv420p", v], check=True)
        frs = sheet.video_frames(v, [0.5, 2.0, 3.5])
        self.assertEqual(len(frs), 3)
        self.assertAlmostEqual(frs[1][0], 2.0, delta=1 / 30 + 1e-6)
        out = os.path.join(self.td, "v.jpg")
        r = sheet._main([out, "--video", v, "--n", "4", "--offset", "60"])
        self.assertEqual(r["n"], 4)


class TestCrops(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        a = np.zeros((360, 640, 3), np.uint8)
        b = a.copy()
        b[100:120, 500:530] = 200          # a 20x30 change at x 500..530, y 100..120
        b[300, 10] = 5                     # below threshold
        self.a, self.b = os.path.join(self.td, "a.png"), os.path.join(self.td, "b.png")
        Image.fromarray(a).save(self.a)
        Image.fromarray(b).save(self.b)

    def test_diff_stats(self):
        s = crops.diff_stats(self.a, self.b, 8)
        self.assertEqual(s["changed"], 600)
        self.assertEqual(s["bbox"], [500, 100, 530, 120])
        self.assertEqual(s["max"], 200)
        self.assertEqual(crops.diff_stats(self.a, self.a)["psnr"], None)

    def test_diff_cli_json_and_heat(self):
        heat = os.path.join(self.td, "h.jpg")
        s = crops._main(["diff", self.a, self.b, "--heat", heat, "--json"])
        self.assertEqual(s["changed"], 600)
        self.assertTrue(os.path.getsize(heat) < 150_000)

    def test_auto_box_finds_change(self):
        A, B = crops.load_rgb(self.a), crops.load_rgb(self.b)
        (x0, y0, x1, y1), = crops.auto_boxes(A, B, 1, 100)
        self.assertTrue(x0 <= 500 and x1 >= 530 and y0 <= 100 and y1 >= 120, (x0, y0, x1, y1))

    def test_compare_is_one_to_one_when_it_fits(self):
        out = os.path.join(self.td, "c.jpg")
        r = crops._main(["compare", out, self.a, self.b, "--box", "450,50,750,350", "--labels", "before,after"])
        self.assertEqual(r["crop_scale"], 1.0)
        self.assertEqual(Image.open(out).size, (600, 300))


class TestPatchPure(unittest.TestCase):
    def test_parse_path(self):
        self.assertEqual(patch.parse_path("objects['A b'].modifiers[0].show_render"),
                         [("attr", "objects"), ("key", "A b"), ("attr", "modifiers"), ("idx", 0),
                          ("attr", "show_render")])
        self.assertEqual(patch.parse_path('eevee.volumetric_end'), [("attr", "eevee"), ("attr", "volumetric_end")])
        for bad in ("objects['x']", "__import__('os').system('x')", "a..b", "a[foo]"):
            with self.assertRaises(ValueError, msg=bad):
                patch.parse_path(bad)

    def test_resolve_on_plain_objects(self):
        class O:
            pass
        root = O()
        root.objects = {"Cube": O()}
        root.objects["Cube"].hide_render = False
        owner, attr = patch.resolve(root, "objects['Cube'].hide_render")
        self.assertIs(owner, root.objects["Cube"])
        self.assertEqual(attr, "hide_render")

    def test_normalize_rejects_typos_and_splits(self):
        with self.assertRaises(KeyError):
            patch.normalize_spec({"hide_colections": ["X"]})
        self.assertEqual(patch.normalize_spec({"hide_prefix": "A_,B_"})["hide_prefix"], ["A_", "B_"])

    def test_undo_restores_in_reverse(self):
        class O:
            v = 1
        o = O()
        u = patch.Undo()
        u.setattr(o, "v", 2)
        u.setattr(o, "v", 3)
        u()
        self.assertEqual(o.v, 1)

    def test_load_spec_forms(self):
        self.assertEqual(patch.load_spec('{"a": 1}'), {"a": 1})
        p = os.path.join(tempfile.mkdtemp(), "v.json")
        json.dump({"x": {}}, open(p, "w"))
        self.assertEqual(patch.load_spec(p), {"x": {}})
        self.assertEqual(patch.load_spec(""), {})


class TestTimeAbPure(unittest.TestCase):
    def test_parse_frames(self):
        self.assertEqual(time_ab.parse_frames("1-3,10, 12"), [1, 2, 3, 10, 12])

    def test_plan_base_first_and_bookend(self):
        p = time_ab.plan({"x": {}, "base": {}}, ["s1", "s2"], repeat=2, bookend=True)
        self.assertEqual([v for _, v, _ in p[:4]], ["base", "base", "x", "x"])
        self.assertEqual(len(p), 2 * 2 * 2 + 2)
        self.assertEqual(p[-1], (2, "base", "s2"))

    def test_summarize_median_and_delta(self):
        runs = [{"variant": "base", "shot": "a", "s": 2.0}, {"variant": "base", "shot": "a", "s": 4.0},
                {"variant": "base", "shot": "a", "s": 3.0}, {"variant": "x", "shot": "a", "s": 1.0}]
        s = time_ab.summarize(runs)
        self.assertEqual(s["variants"]["base"]["a"], 3.0)
        self.assertEqual(s["delta"]["x"], -2.0)
        self.assertIn("Δbase", time_ab.table(s, ["a"]))


if __name__ == "__main__":
    unittest.main()
