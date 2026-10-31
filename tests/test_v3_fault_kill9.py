"""V3 Phase 1 exit gate 2 (fault injection), the kill -9 half: a REAL SIGKILL of a host process after it
replied, against a scratch event store + scratch discord_inbox.json — never the live host or state/.

test_host_event_dualwrite.py simulates the crash in-process (drops the handles). This one kills a real
process with SIGKILL, so nothing in it gets a chance to flush or close: if the reply marker were not durable
at the moment the reply tool returned, the restart would re-answer blind.

  1. child A: DM 333 queued (durable inbox + store row), turn begins, the Discord reply tool returns
     -> PostToolUse marker; then os.kill(self, SIGKILL) before the turn end (inbox still lists 333 unacked).
  2. child B (fresh process, same files): exactly what main() does at boot — re-queue the JSON unacked items
     with _ev_discord_replay_prefix, then _ev_boot_replay. One item comes back, labelled "already answered".
  3. stress: a child marks replies in a tight loop and is SIGKILLed at random moments; the DB always opens
     with integrity ok and no row is ever output_pending without its reply_ref (status + ref are one txn).
"""
from __future__ import annotations

import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_PRELUDE = textwrap.dedent("""
    import asyncio, json, os, signal, sys, time, types
    sys.path.insert(0, {root!r})
    _fake_av = types.ModuleType("avaagent"); _fake_av.load_profile_by_id = lambda pid: {{}}
    sys.modules.setdefault("avaagent", _fake_av)
    import iris_body_host as h
    h.DISCORD_INBOX_PATH = {inbox!r}
    h.HOST_STREAM_PATH = {stream!r}
    h._blog = lambda *a, **k: None
    h._orb_claim = lambda mid: True
    DM = str(h.DISCORD_CHANNEL_ID or "")

    def reply(msg_id, sent_id):
        return {{"tool_name": "mcp__plugin_discord_discord__reply",
                 "tool_input": {{"chat_id": DM, "text": "hi", "reply_to": str(msg_id)}},
                 "tool_response": [{{"type": "text", "text": "sent (id: " + str(sent_id) + ")"}}]}}

    def begin_turn(q):
        entry = q.get_nowait()
        eids = h._ev_take(entry)
        item = entry[2]
        h._ev_turn_begin(eids, item[0], item[2], item[3] if len(item) > 3 else None)
        h._INBOX.update(queue=q, turn=types.SimpleNamespace(active=True, guest=False))
        return eids
""")

_CHILD_A = textwrap.dedent("""
    async def go():
        q = asyncio.PriorityQueue()
        # the poller's two durable writes for a new DM: inbox JSON (unacked) + the store row
        h._DISCORD_INBOX.update(last_id="333", unacked=[{"id": "333", "content": "what time is it"}])
        h._discord_inbox_save()
        await h._enqueue(q, ("discord", "what time is it", "333"))
        begin_turn(q)
        h._ev_mark_reply(reply("333", "4444"))     # the reply tool returned: Discord has the message
    asyncio.run(go())
    sys.stdout.write("replied\\n"); sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)            # no turn end, no close, no atexit
""")

_CHILD_B = textwrap.dedent("""
    async def boot():
        q = asyncio.PriorityQueue()
        start = time.time() + 1
        h._discord_inbox_load()
        for u in list(h._DISCORD_INBOX["unacked"]):
            await h._enqueue(q, ("discord", h._ev_discord_replay_prefix(u["id"]) + str(u.get("content") or ""), u["id"]))
        await h._ev_boot_replay(q, start, [u.get("id") for u in h._DISCORD_INBOX["unacked"]],
                                h._DISCORD_INBOX["last_id"])
        out = []
        while not q.empty():
            out.append(list(q.get_nowait()[2][:3]))
        return out
    print(json.dumps(asyncio.run(boot())))
""")

_CHILD_STRESS = textwrap.dedent("""
    async def go():
        n = 0
        while True:
            q = asyncio.PriorityQueue()
            mid = str(10_000 + n)
            await h._enqueue(q, ("discord", "msg " + mid, mid))
            begin_turn(q)
            h._ev_mark_reply(reply(mid, "9" + mid))
            h._ev_turn_end(True)
            n += 1
            if n == 1:
                sys.stdout.write("go\\n"); sys.stdout.flush()
    asyncio.run(go())
""")


def _env(tmp: Path) -> tuple[dict, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRIS_")}
    env.update(IRIS_V3_EVENTS="1", IRIS_EVENTS_DB=str(tmp / "v3" / "events.sqlite"), PYTHONDONTWRITEBYTECODE="1")
    prelude = _PRELUDE.format(root=str(ROOT), inbox=str(tmp / "discord_inbox.json"),
                              stream=str(tmp / "host_stream.json"))
    return env, prelude


def test_kill9_after_reply_restart_does_not_reanswer(tmp_path):
    env, prelude = _env(tmp_path)
    a = subprocess.run([sys.executable, "-c", prelude + _CHILD_A], env=env, capture_output=True, text=True,
                       timeout=120, cwd=str(tmp_path))
    assert a.returncode == -signal.SIGKILL, (a.returncode, a.stderr[-2000:])
    assert "replied" in a.stdout
    inbox = json.loads((tmp_path / "discord_inbox.json").read_text())
    assert [u["id"] for u in inbox["unacked"]] == ["333"], "killed before the turn end: still unacked"

    b = subprocess.run([sys.executable, "-c", prelude + _CHILD_B], env=env, capture_output=True, text=True,
                       timeout=120, cwd=str(tmp_path))
    assert b.returncode == 0, b.stderr[-2000:]
    items = json.loads(b.stdout.strip().splitlines()[-1])
    assert len(items) == 1, items
    src, text, mid = items[0]
    assert (src, mid) == ("discord", "333")
    assert text.startswith("[I already answered this before the restart: discord:4444"), text
    assert "unhandled" not in text

    c = sqlite3.connect(str(tmp_path / "v3" / "events.sqlite"))
    try:
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert c.execute("SELECT count(*) FROM events").fetchone()[0] == 1, "replay reused the row"
    finally:
        c.close()


def test_kill9_at_random_moments_never_leaves_a_reply_without_its_ref(tmp_path):
    env, prelude = _env(tmp_path)
    rnd = random.Random(1031)
    for _ in range(6):
        p = subprocess.Popen([sys.executable, "-c", prelude + _CHILD_STRESS], env=env, cwd=str(tmp_path),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert p.stdout.readline().strip() == "go", p.stderr.read()[-2000:]
            time.sleep(rnd.uniform(0.0, 0.4))
        finally:
            p.send_signal(signal.SIGKILL)
            p.wait(timeout=30)
        assert p.returncode == -signal.SIGKILL
    c = sqlite3.connect(str(tmp_path / "v3" / "events.sqlite"))
    try:
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        n = c.execute("SELECT count(*) FROM events").fetchone()[0]
        assert n >= 6
        bad = c.execute("SELECT event_id, status FROM events WHERE status IN ('output_pending','committed') "
                        "AND (reply_ref IS NULL OR reply_ref = '')").fetchall()
        assert bad == [], bad
        torn = c.execute("SELECT e.event_id FROM events e JOIN transitions t ON t.entity='event' "
                         "AND t.entity_id=e.event_id AND t.to_status='output_pending' "
                         "WHERE e.reply_ref IS NULL").fetchall()
        assert torn == [], "a reply transition exists for a row whose ref was lost"
    finally:
        c.close()
