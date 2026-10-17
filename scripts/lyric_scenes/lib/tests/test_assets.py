"""lib/assets tests (stdlib, network mocked):  python3 -I -m unittest discover -s <lib>/tests -p 'test_assets.py'"""
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import zipfile

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LIB, "assets"))
import credits  # noqa: E402
import fetch  # noqa: E402
import licence  # noqa: E402
import search  # noqa: E402


def zbytes(members):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return b.getvalue()


class TestLicence(unittest.TestCase):
    def test_policy_constant(self):
        self.assertEqual(licence.ALLOWED_LICENCES, ("CC0", "CC-BY"))

    def test_gate_table(self):
        allow = ["CC Attribution", "CC0 1.0", "CC0 Public Domain", "by", "cc0", "CC BY 4.0",
                 "CC0 (CGHEVEN free tier) — per https://cgheven.com/licensing"]
        refuse = {"CC Attribution-ShareAlike": "ShareAlike", "by-sa": "ShareAlike",
                  "CC Attribution-NonCommercial": "NonCommercial", "CC Attribution-NonCommercial-ShareAlike": "NonCommercial",
                  "CC Attribution-NoDerivs": "NoDerivatives", "by-nd": "NoDerivatives", "Free Standard": "not recognised",
                  None: "not recognised", "ODbL 1.0": "map/terrain data"}
        for x in allow:
            self.assertTrue(licence.gate(x)[0], x)
        for x, why in refuse.items():
            ok, c, msg = licence.gate(x)
            self.assertFalse(ok, x)
            self.assertIn(why, msg, x)
            self.assertIn("REFUSED", msg)
        self.assertTrue(licence.gate("ODbL 1.0 — © OpenStreetMap contributors", data_ok=True)[0])
        self.assertTrue(licence.gate("Copernicus DEM licence", data_ok=True)[0])
        self.assertFalse(licence.gate("CC Attribution-ShareAlike", data_ok=True)[0])

    def test_normalize_all_legacy_shapes(self):
        sf = {"slug": "x", "uid": "U1", "name": "Dragon", "author": "jdoe", "author_name": "J Doe",
              "author_url": "https://sketchfab.com/jdoe", "license": "CC Attribution",
              "license_url": "http://creativecommons.org/licenses/by/4.0/", "url": "https://sketchfab.com/3d-models/U1"}
        ph = {"source": "https://polyhaven.com/a/aircon", "id": "aircon", "res": "2k", "licence": "CC0 1.0"}
        acg = {"kind": "ambientcg", "id": "PaintedMetal017", "res": "2K", "licence": "CC0 1.0"}
        acg2 = {"source": "https://ambientcg.com/view?id=Ground068", "licence": "CC0 1.0", "res": "2K-JPG"}
        cgh = {"name": "Flamethrower 07 - VDB", "author": "CGHEVEN (Ammar Khan)", "source": "https://cgheven.com (assets)",
               "download_url": "https://cdn.cgheven.com/x.zip", "license": "CC0 (CGHEVEN free tier)", "license_note": "FLAG"}
        osm = {"source": "OpenStreetMap via Overpass API", "licence": "ODbL 1.0 — © OpenStreetMap contributors",
               "url": "https://www.openstreetmap.org/copyright"}
        n = licence.normalize(sf)
        self.assertEqual((n["licence"], n["author"], n["site"], n["id"]), ("CC-BY", "J Doe", "sketchfab", "U1"))
        n = licence.normalize(ph)
        self.assertEqual((n["licence"], n["site"], n["url"], n["author"]), ("CC0", "polyhaven", ph["source"], "Poly Haven"))
        n = licence.normalize(acg)
        self.assertEqual((n["site"], n["url"]), ("ambientcg", "https://ambientcg.com/view?id=PaintedMetal017"))
        n = licence.normalize(acg2)
        self.assertEqual((n["site"], n["id"]), ("ambientcg", "Ground068"))
        n = licence.normalize(cgh)
        self.assertEqual((n["licence"], n["site"], n["note"]), ("CC0", "cgheven", "FLAG"))
        n = licence.normalize(osm)
        self.assertEqual((n["licence"], n["site"], n["attribution"]), ("ODbL", "osm", "© OpenStreetMap contributors"))
        self.assertEqual(licence.normalize({"schema": 1, "name": "k"})["name"], "k")

    def test_find_source_walks_up_but_not_past_root(self):
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "a", "textures"))
        json.dump({"licence": "CC0"}, open(os.path.join(root, "a", "SOURCE.json"), "w"))
        self.assertEqual(licence.find_source(os.path.join(root, "a", "textures", "x.png"), root),
                         os.path.join(root, "a", "SOURCE.json"))
        os.makedirs(os.path.join(root, "b"))
        self.assertIsNone(licence.find_source(os.path.join(root, "b", "y.png"), root))


class FakeNet:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, headers=None, *a, **k):
        self.calls.append(url)
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                if isinstance(resp, Exception):
                    raise resp
                return resp if isinstance(resp, bytes) else json.dumps(resp).encode()
        raise AssertionError(f"unexpected URL {url}")


class TestFetch(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self._get = fetch.GET
        os.environ["SKETCHFAB_TOKEN"] = "test-token"

    def tearDown(self):
        fetch.GET = self._get

    def meta(self, label):
        return {"name": "Old Well", "license": {"label": label, "url": "https://creativecommons.org/licenses/by/4.0/"},
                "user": {"username": "jdoe", "displayName": "J Doe", "profileUrl": "https://sketchfab.com/jdoe"},
                "viewerUrl": "https://sketchfab.com/3d-models/well-U1", "faceCount": 1000}

    def test_sketchfab_by_sa_refused_before_any_download(self):
        net = FakeNet({"https://api.sketchfab.com/v3/models/U1": self.meta("CC Attribution-ShareAlike")})
        fetch.GET = net
        with self.assertRaises(fetch.Refused) as cm:
            fetch.sketchfab("well", "U1", self.root)
        self.assertIn("ShareAlike", str(cm.exception))
        self.assertEqual(len(net.calls), 1)                    # metadata only — never the download endpoint
        self.assertEqual(os.listdir(self.root), [])

    def test_sketchfab_cc_by_glb(self):
        net = FakeNet({"https://api.sketchfab.com/v3/models/U1/download": {"glb": {"url": "https://cdn/x.glb", "size": 3}},
                       "https://api.sketchfab.com/v3/models/U1": self.meta("CC Attribution"),
                       "https://cdn/x.glb": b"glTF"})
        fetch.GET = net
        m = fetch.sketchfab("well", "U1", self.root)
        d = os.path.join(self.root, "well")
        self.assertTrue(os.path.isfile(os.path.join(d, "model.glb")))
        s = json.load(open(os.path.join(d, "SOURCE.json")))
        self.assertEqual((s["schema"], s["licence"], s["author"], m["site"]), (1, "CC-BY", "J Doe", "sketchfab"))
        self.assertFalse(os.path.exists(d + ".partial"))
        with self.assertRaises(fetch.Refused):                  # second fetch into the same slug
            fetch.sketchfab("well", "U1", self.root)

    def test_safe_zip_extraction(self):
        data = zbytes({"scene.gltf": "{}", "textures/a.png": "png", "../evil.png": "x", "/abs.png": "x",
                       "run.exe": "x", "C:/win.png": "x", "sub/../../up.png": "x"})
        d = tempfile.mkdtemp()
        got = fetch.extract_zip(data, d)
        self.assertEqual(sorted(got), ["scene.gltf", "textures/a.png"])
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(d), "evil.png")))

    def test_ambientcg_flatten_and_failed_fetch_stays_partial(self):
        fetch.GET = FakeNet({"https://ambientcg.com/get?file=Rock1_2K-JPG.zip": zbytes({"x/Rock1_2K_Color.jpg": "j"})})
        m = fetch.ambientcg("Rock1", "2K", self.root)
        self.assertEqual(m["files"], ["Rock1_2K_Color.jpg"])
        fetch.GET = FakeNet({"https://ambientcg.com/get?file=Rock2_2K-JPG.zip": OSError("network down")})
        with self.assertRaises(OSError):
            fetch.ambientcg("Rock2", "2K", self.root)
        self.assertTrue(os.path.isdir(os.path.join(self.root, "Rock2.partial")))
        self.assertFalse(os.path.exists(os.path.join(self.root, "Rock2")))

    def test_429_backoff(self):
        import urllib.request
        calls = {"n": 0}
        slept = []

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.HTTPError(req.full_url, 429, "slow down", {"Retry-After": "7"}, None)
            return Resp(b"ok")
        orig = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            self.assertEqual(fetch.http_get("https://x/y", sleep=slept.append), b"ok")
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(slept, [7, 7])


class TestSearch(unittest.TestCase):
    def test_drops_disallowed_licences(self):
        res = {"results": [{"uid": "A", "name": "a", "license": {"label": "CC Attribution"}, "user": {"username": "u"}},
                           {"uid": "B", "name": "b", "license": {"label": "CC Attribution-NonCommercial"}, "user": {}},
                           {"uid": "C", "name": "c", "license": {"label": "CC0 Public Domain"}, "user": {}}]}
        old = fetch.GET
        fetch.GET = FakeNet({"https://api.sketchfab.com/v3/search": res})
        try:
            r = search.search(["well"])
        finally:
            fetch.GET = old
        self.assertEqual(sorted(x["uid"] for x in r), ["A", "C"])


class TestCredits(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

        def asset(slug, src, files=("model.glb",)):
            d = os.path.join(self.root, slug)
            os.makedirs(d)
            json.dump(src, open(os.path.join(d, "SOURCE.json"), "w"))
            for f in files:
                open(os.path.join(d, f), "w").close()
            return [os.path.join(d, f) for f in files]
        self.files = asset("dragon", {"name": "Dragon", "author": "jdoe", "author_name": "J Doe", "license": "CC Attribution",
                                      "url": "https://sketchfab.com/3d-models/d"})
        self.files += asset("rock_tex", {"source": "https://polyhaven.com/a/rock", "id": "rock", "licence": "CC0 1.0"},
                            ("rock_diff.jpg",))
        self.files += asset("osm", {"source": "OpenStreetMap", "licence": "ODbL 1.0 — © OpenStreetMap contributors"},
                            ("city.json",))
        self.own = "/tmp/generated_texture.png"
        self.log = os.path.join(self.root, "import_log.json")
        json.dump({"imports": [{"path": self.files[0]}], "files": self.files[1:] + [self.own]}, open(self.log, "w"))
        self.out = os.path.join(tempfile.mkdtemp(), "ASSETS.md")

    def run_credits(self, *extra):
        return credits._main(["--used", self.log, "--root", self.root, "--out", self.out, "--title", "Test",
                              "--use", "dragon=the boss", *extra])

    def test_generates_table_and_credits_block(self):
        md = self.run_credits()
        self.assertIn('"Dragon" by J Doe — https://sketchfab.com/3d-models/d', md)
        self.assertIn("© OpenStreetMap contributors", md)
        self.assertIn("CC0 assets (no credit required — thank you): Poly Haven", md)
        self.assertIn("| the boss |", md)
        self.assertIn("Own/generated files used (not credited): 1", md)
        self.assertTrue(open(self.out).read().startswith(credits.MARK))
        self.run_credits()                                    # regenerating its own file is fine

    def test_refuses_hand_written_assets_md(self):
        open(self.out, "w").write("# hand written\n")
        with self.assertRaises(SystemExit):
            self.run_credits()
        self.assertEqual(open(self.out).read(), "# hand written\n")

    def test_blocks_by_sa_and_cc_by_without_author_and_unsourced(self):
        d = os.path.join(self.root, "banner")
        os.makedirs(d)
        json.dump({"name": "Banner", "license": "CC Attribution-ShareAlike", "author": "x", "url": "u"},
                  open(os.path.join(d, "SOURCE.json"), "w"))
        d2 = os.path.join(self.root, "noauthor")
        os.makedirs(d2)
        json.dump({"name": "Anon", "license": "CC Attribution"}, open(os.path.join(d2, "SOURCE.json"), "w"))
        os.makedirs(os.path.join(self.root, "mystery"))
        stray = os.path.join(self.root, "mystery", "m.glb")
        open(stray, "w").close()
        json.dump({"imports": [], "files": self.files + [os.path.join(d, "a.glb"), os.path.join(d2, "b.glb"), stray]},
                  open(self.log, "w"))
        sources, unsourced, own = credits.collect(credits.used_files([self.log]), root=self.root)
        errs = credits.problems(sources, unsourced)
        joined = "\n".join(errs)
        self.assertIn("ShareAlike", joined)
        self.assertIn("CC-BY needs author AND url", joined)
        self.assertIn("NO SOURCE.json", joined)
        with self.assertRaises(SystemExit) as cm:
            self.run_credits()
        self.assertEqual(cm.exception.code, 2)
        self.assertFalse(os.path.exists(self.out))


if __name__ == "__main__":
    unittest.main()
