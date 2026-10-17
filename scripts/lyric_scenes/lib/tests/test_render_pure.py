"""lib/render pure tests (stdlib):  python3 -I -m unittest discover -s <lib>/tests -p 'test_render_pure.py'"""
import os
import sys
import tempfile
import unittest
import zlib
import struct

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LIB, "render"))
import frames  # noqa: E402  (bpy is only imported inside main())
import pngcheck  # noqa: E402


def tiny_png(path, w=4, h=4):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + b"\x80" * (3 * w) for _ in range(h))
    data = (pngcheck.SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)
    return data


class TestPngCheck(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_complete_vs_torn(self):
        p = os.path.join(self.d, "f_0001.png")
        data = tiny_png(p)
        self.assertTrue(pngcheck.is_complete_png(p))
        with open(p, "wb") as f:                 # the half-written-PNG bug: big enough for the old 1 KB / -s tests
            f.write(data[:-5])
        self.assertFalse(pngcheck.is_complete_png(p))
        with open(p, "wb") as f:
            f.write(b"\x00" * 4096)
        self.assertFalse(pngcheck.is_complete_png(p))
        self.assertFalse(pngcheck.is_complete_png(os.path.join(self.d, "nope.png")))

    def test_missing_and_ranges(self):
        for f in (1, 2, 3, 6):
            tiny_png(pngcheck.frame_path(self.d, f))
        with open(pngcheck.frame_path(self.d, 2), "r+b") as fh:
            fh.truncate(30)
        self.assertEqual(pngcheck.missing(self.d, 1, 7), [2, 4, 5, 7])
        self.assertEqual(pngcheck.ranges([2, 4, 5, 7, 8, 9]), "2,4-5,7-9")
        open(os.path.join(self.d, ".tmp_f_0004.123.png"), "w").close()
        self.assertEqual(len(pngcheck.torn(self.d)), 2)

    def test_parse_frames(self):
        self.assertEqual(frames.parse_frames("1-3, 7,10-11"), [1, 2, 3, 7, 10, 11])
        self.assertEqual(frames.parse_frames(""), [])


if __name__ == "__main__":
    unittest.main()
