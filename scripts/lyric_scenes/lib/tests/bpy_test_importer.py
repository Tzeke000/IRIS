"""Headless Blender test for lib/importer.py. Through the slice:
    scripts/server/render.sh t-libimp --wait --collect -P ~/.local/bin/blender -b --factory-startup \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/tests/bpy_test_importer.py [-- --assets]
Hand-written .gltf files (KHR_materials_unlit + a missing texture) prove detection/fix/report; --assets also imports the
boss_fight scans and checks find_unlit flags exactly the materials build_arena_v3 fixed by hand on 10-15."""
import base64
import json
import os
import struct
import sys
import tempfile
import zlib

import bpy

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(LIB))
from lib import importer as imp  # noqa: E402

FAILS = []
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def check(name, cond, info=""):
    print(f"[bpy_test_importer] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAILS.append(name)


def png_bytes(w=4, h=4, rgb=(200, 180, 160)):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def gltf(path, unlit, image_uri, offset=(0.0, 0.0, 0.0), scale=1.0):
    """one quad (2 tris) with a textured material; optional KHR_materials_unlit."""
    pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    pos = [tuple(scale * c + o for c, o in zip(p, offset)) for p in pos]
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    idx = [0, 1, 2, 0, 2, 3]
    b = b"".join(struct.pack("<3f", *p) for p in pos) + b"".join(struct.pack("<2f", *t) for t in uv) + \
        b"".join(struct.pack("<H", i) for i in idx)
    mat = {"name": "scanmat", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0}}
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "Quad"}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
           "materials": [mat], "textures": [{"source": 0}], "images": [{"uri": image_uri}],
           "buffers": [{"byteLength": len(b), "uri": "data:application/octet-stream;base64," + base64.b64encode(b).decode()}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 32},
                           {"buffer": 0, "byteOffset": 80, "byteLength": 12}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                          "min": list(map(min, zip(*pos))), "max": list(map(max, zip(*pos)))},
                         {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
                         {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}]}
    if unlit:
        mat["extensions"] = {"KHR_materials_unlit": {}}
        doc["extensionsUsed"] = ["KHR_materials_unlit"]
    json.dump(doc, open(path, "w"))


def main():
    td = tempfile.mkdtemp(prefix="libimp_")
    tex = os.path.join(td, "albedo.png")
    open(tex, "wb").write(png_bytes())
    gltf(os.path.join(td, "unlit.gltf"), True, "albedo.png")
    gltf(os.path.join(td, "lit_missing.gltf"), False, "missing_tex.png", offset=(5, 0, 2), scale=2.0)
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    C = bpy.data.collections.new("IMPORT")
    bpy.context.scene.collection.children.link(C)

    a = imp.import_model(os.path.join(td, "unlit.gltf"), coll=C)
    rec = imp.LOG[-1]
    check("import returns the new objects, linked into coll", len(a) == 1 and a[0].users_collection[0] == C)
    check("KHR_materials_unlit detected (report mode)", rec["unlit"] == ["scanmat"], str(rec["unlit"]))
    check("report mode leaves the material unlit", imp.is_unlit(a[0].material_slots[0].material))
    n_obj = len(bpy.data.objects)
    a2 = imp.import_model(os.path.join(td, "unlit.gltf"), coll=C)
    check("cache: second import returns the same objects, no new ones", a2 == a and len(bpy.data.objects) == n_obj)
    b = imp.import_model(os.path.join(td, "unlit.gltf"), coll=C, cache=False, fix_unlit="lit",
                         unlit_kw={"target_mean": 0.3, "normal_tex": tex})
    m = b[0].material_slots[0].material
    check("fix_unlit='lit' → Principled, no longer unlit", not imp.is_unlit(m) and
          any(n.type == "BSDF_PRINCIPLED" for n in m.node_tree.nodes), str(imp.LOG[-1]["unlit_fixed"]))
    k = imp.LOG[-1]["unlit_fixed"][m.name][1]
    check("albedo scaled to target mean (texture mean 0.706 → k = 0.30/0.706 ≈ 0.425)", abs(k - 0.425) < 0.01, str(k))
    c = imp.import_model(os.path.join(td, "lit_missing.gltf"), coll=C)
    bad = imp.LOG[-1]["bad_images"]
    check("missing texture reported", any("missing" in why for _, why in bad), str(bad))
    check("lit glTF not flagged unlit", imp.LOG[-1]["unlit"] == [])
    try:
        imp.import_model(os.path.join(td, "lit_missing.gltf"), coll=C, cache=False, strict_images=True, quiet=True)
        check("strict_images raises", False)
    except FileNotFoundError:
        check("strict_images raises", True)
    check("size_m logged in metres", imp.LOG[-1]["size_m"] is not None)

    # bake_world + normalize + islands + sub_mesh + decimate
    me = imp.bake_world(a + c, "Merged")
    lo, hi = imp.bbox(me)
    check("bake_world: 8 verts, world transforms baked", len(me.vertices) == 8 and abs(hi[0] - 7) < 1e-4 and abs(hi[2] - 2) < 1e-4,
          f"{len(me.vertices)} {hi}")
    check("bake_world de-dups materials", len(me.materials) == 2, str([x.name for x in me.materials]))
    lab = imp.islands(me)
    check("islands: two quads → 2 labels", len(set(lab.tolist())) == 2)
    sm = imp.sub_mesh(me, lab == lab[0], "One")
    check("sub_mesh keeps one island", len(sm.vertices) == 4)
    s = imp.normalize(me, height=4.0)
    lo, hi = imp.bbox(me)
    check("normalize: ground z0, height 4, centred XY", abs(lo[2]) < 1e-5 and abs(hi[2] - 4) < 1e-4 and
          abs(lo[0] + hi[0]) < 1e-4, f"s={s} lo={lo} hi={hi}")
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    sph = bpy.context.active_object
    n0 = len(sph.data.polygons)
    dm = imp.decimate(sph.data.copy(), 0.25)
    check("decimate reduces polys", len(dm.polygons) < 0.5 * n0, f"{n0} → {len(dm.polygons)}")
    imp.write_log(os.path.join(td, "import_log.json"))
    lg = json.load(open(os.path.join(td, "import_log.json")))
    check("write_log: imports + every file used (models + load_img images)", len(lg["imports"]) == len(imp.LOG) and
          os.path.join(td, "unlit.gltf") in lg["files"] and tex in lg["files"], str(lg["files"]))

    if "--assets" in ARGS:
        R3, SF, R4 = "/data/iris_assets/medieval/r3/", "/data/iris_assets/medieval/sketchfab/", "/data/iris_assets/medieval/r4_v3b/"
        want_unlit = [R3 + "statue_gargoyle_photogrammetry_pigfinite", R3 + "bld_stone_house_1scanaday",
                      R3 + "bld_st_alphage_ruin_artfletch", SF + "env_stones_pile"]
        want_lit = [R3 + "statue_gargoyle_column_alban", R3 + "statue_knight_effigy_southwark_natesid",
                    R4 + "prop_bow_cgdude0007"]
        for d in want_unlit + want_lit:
            imp.import_model(d + "/model.glb", coll=C)
            hit = imp.LOG[-1]["unlit"]
            check(f"boss asset {os.path.basename(d)} unlit={bool(hit)} (v3c fixed it: {d in want_unlit})",
                  bool(hit) == (d in want_unlit), str(hit))
    print(f"[bpy_test_importer] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  ({td})", flush=True)
    sys.exit(1 if FAILS else 0)


main()
