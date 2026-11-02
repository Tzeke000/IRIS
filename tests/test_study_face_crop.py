"""study_face crop: the box and the pixels must come from the SAME frame (studyface-fix 11-01).

The bug: with no unknown in the latest face results, the crop used the bbox from the START of the study on the live frame
— after the servo had panned onto the person, so the crop was a piece of wall."""
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from tools.system import study_face_tool as sf  # noqa: E402


class _Face:
    def __init__(self, bbox, emb):
        self.bbox = np.array(bbox, dtype="float32")
        self.embedding = np.array(emb, dtype="float32")


def _engine(by_mark, known_embs=()):
    """fake InsightFace: the face it 'finds' is keyed on the frame's top-left pixel value"""
    def get(img):
        return list(by_mark.get(int(img[0, 0, 0]), []))
    def match(emb):
        return ("zeke", 0.9) if any(np.allclose(emb, k) for k in known_embs) else ("unknown", 0.1)
    return types.SimpleNamespace(_app=types.SimpleNamespace(get=get), _lock=None, _match=match)


def _frame(mark):
    img = np.zeros((480, 640, 3), dtype="uint8"); img[0, 0, 0] = mark
    return img


def test_crop_uses_the_box_found_in_the_saved_frame_not_the_start_box(tmp_path):
    # the person, after the servo panned, is centred in the burst frames; the live results are empty (the bug's trigger)
    p1, p2 = str(tmp_path / "a.jpg"), str(tmp_path / "b.png")
    cv2.imwrite(p1, _frame(10)); cv2.imwrite(p2, _frame(20))
    g = {"_insight_face": _engine({20: [_Face([280, 180, 360, 280], [1, 0])]})}
    frame, bbox, source = sf._crop_source(g, cv2, [p1, p2], _frame(99), unknowns_now=[])
    assert bbox == [280, 180, 360, 280] and "b.png" in source and int(frame[0, 0, 0]) == 20


def test_a_known_face_in_the_burst_is_never_the_crop(tmp_path):
    p = str(tmp_path / "a.png"); cv2.imwrite(p, _frame(30))
    g = {"_insight_face": _engine({30: [_Face([0, 0, 200, 200], [0, 1]), _Face([400, 100, 450, 160], [1, 0])]}, known_embs=[[0, 1]])}
    _f, bbox, _s = sf._crop_source(g, cv2, [p], None, unknowns_now=[])
    assert bbox == [400, 100, 450, 160]          # the smaller UNKNOWN, not Zeke's bigger face


def test_engine_down_uses_the_live_frame_only_with_its_own_results():
    live = _frame(5)
    f, bbox, source = sf._crop_source({}, cv2, [], live, unknowns_now=[{"person_id": "unknown", "bbox": [10, 20, 60, 90]}])
    assert f is live and bbox == [10, 20, 60, 90] and source.startswith("live frame +")


def test_no_unknown_anywhere_means_no_crop_not_the_wall():
    f, bbox, source = sf._crop_source({}, cv2, [], _frame(5), unknowns_now=[])
    assert bbox is None and "no unknown face" in source
