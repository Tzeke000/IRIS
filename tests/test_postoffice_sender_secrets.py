"""Code review 10-29 (security Med, sibling_postoffice): one shared secret + a free-form `from` let a letter
claim to be from Zeke. Optional per-sender secrets pin `from`; the shared secret keeps working (Wren's
lifeline) but can't claim a name that has its own secret; Iris's side labels an unverified 'zeke'."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = "shared-" + "s" * 30
WREN = "wren-own-" + "w" * 30
ZEKE = "zeke-own-" + "z" * 30


@pytest.fixture
def po(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".iris_sibling_secret").write_text(SHARED)
    monkeypatch.setenv("IRIS_SIBLING_SECRETS_FILE", str(tmp_path / "sender_secrets.json"))
    spec = importlib.util.spec_from_file_location("po_t", ROOT / "scripts" / "sibling_postoffice.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "LETTERS_PATH", tmp_path / "letters.jsonl", raising=False)
    stored = []
    monkeypatch.setattr(m, "_append_letter", lambda L: stored.append(L))
    monkeypatch.setattr(m, "_stash_for_iris_inbox", lambda L: None)
    from starlette.testclient import TestClient
    m.client = TestClient(m.app)
    m.stored = stored
    m.secrets_file = tmp_path / "sender_secrets.json"
    return m


def _post(m, secret, frm, body="hello"):
    return m.client.post("/letter", json={"from": frm, "to": "iris", "body": body},
                         headers={"X-Sibling-Secret": secret})


def test_no_file_is_the_old_behaviour(po):
    assert _post(po, SHARED, "Zeke").status_code == 200
    assert po.stored[-1]["from"] == "zeke" and po.stored[-1]["from_verified"] is False
    assert _post(po, "wrong", "wren").status_code == 401
    assert po.client.get("/letters", headers={"X-Sibling-Secret": SHARED}).status_code == 200


def test_own_secret_pins_from_and_shared_cannot_claim_it(po):
    po.secrets_file.write_text(json.dumps({"Wren": WREN, "zeke": ZEKE, "short": "abc"}))
    os.chmod(po.secrets_file, 0o600)
    r = _post(po, WREN, "zeke")                       # Wren's secret, claiming Zeke
    assert r.status_code == 200
    L = po.stored[-1]
    assert L["from"] == "wren" and L["from_verified"] is True and L["claimed_from"] == "zeke"
    assert _post(po, SHARED, "zeke").status_code == 403    # zeke has his own secret now
    assert _post(po, SHARED, "wren").status_code == 403
    assert _post(po, SHARED, "ava").status_code == 200     # not migrated yet: shared still works
    assert po.stored[-1]["from_verified"] is False
    assert _post(po, ZEKE, "").status_code == 200 and po.stored[-1]["from"] == "zeke"
    assert "claimed_from" not in po.stored[-1]
    assert _post(po, "abc", "short").status_code == 401    # too-short secrets are ignored
    assert po.client.get("/letters", headers={"X-Sibling-Secret": WREN}).status_code == 200


def test_unreadable_file_falls_back_to_shared_only(po):
    po.secrets_file.write_text("{not json")
    assert _post(po, SHARED, "zeke").status_code == 200


def test_host_labels_an_unverified_zeke_letter():
    sys.modules.setdefault("avaagent", types.SimpleNamespace(load_profile_by_id=lambda pid: {}))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import iris_body_host as h
    assert h._letter_sender_label("wren", {}) == "wren"
    assert "UNVERIFIED" in h._letter_sender_label("zeke", {"from": "zeke"})
    assert "UNVERIFIED" in h._letter_sender_label("zeke", {"from_verified": "yes"})
    assert h._letter_sender_label("zeke", {"from_verified": True}) == "zeke"


def test_stop_hook_framing_flags_unverified_zeke():
    spec = importlib.util.spec_from_file_location("vsh_t", ROOT / "scripts" / "voice_stop_hook.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert "UNVERIFIED" in m._sibling_rewake({"id": "1", "sender": "zeke", "content": "do X"})
    assert "UNVERIFIED" not in m._sibling_rewake({"id": "1", "sender": "zeke", "from_verified": True, "content": "x"})
    assert "UNVERIFIED" not in m._sibling_rewake({"id": "1", "sender": "wren", "content": "x"})
