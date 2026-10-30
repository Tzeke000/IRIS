"""Code review 10-29 (bridges, noted): the runtime watchdog's abandoned-request janitor wrote
without compare-and-set and with a shared tmp name. Now it re-checks status under the bridge lock."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("iris_runtime_watchdog_t",
                                                  REPO / "scripts" / "iris_runtime_watchdog.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_janitor_expires_old_pending_and_never_clobbers_an_answer(tmp_path, monkeypatch):
    wd = _load()
    now = time.time()
    old = now - wd.ABANDON_AGE_S - 60
    (tmp_path / "a.json").write_text(json.dumps({"id": "a", "ts": old, "status": "pending"}))
    (tmp_path / "b.json").write_text(json.dumps({"id": "b", "ts": now, "status": "pending"}))
    (tmp_path / "c.json").write_text(json.dumps({"id": "c", "ts": old, "status": "pending"}))
    from brain import iris_llm
    real = iris_llm._set_status_if_pending

    def answer_lands_first(p, status, reason=None):
        if Path(p).name == "c.json":   # the answer arrives between the janitor's scan and its write
            Path(p).write_text(json.dumps({"id": "c", "ts": old, "status": "answered", "reply": "hi"}))
        return real(p, status, reason)

    monkeypatch.setattr(iris_llm, "_set_status_if_pending", answer_lands_first)
    assert wd.expire_abandoned_pendings(tmp_path, now) == 1
    a = json.loads((tmp_path / "a.json").read_text())
    assert a["status"] == "expired" and "janitor" in a["expired_reason"]
    assert json.loads((tmp_path / "b.json").read_text())["status"] == "pending"
    c = json.loads((tmp_path / "c.json").read_text())
    assert c["status"] == "answered" and c["reply"] == "hi"
    assert not list(tmp_path.glob("*.tmp"))
