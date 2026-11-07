"""mcphang_1107: a child process that inherits the MCP stdin pipe STEALS request lines (the 180 s hang), and the
stdio guard stops it. Runs without the live runtime: each case is a throwaway subprocess over real pipes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

pytestmark = pytest.mark.skipif(os.name != "posix", reason="fd isolation is posix-only")

# A stand-in for `ssh host cmd` without -n: a child that reads whatever arrives on its (inherited) stdin.
_PARENT = textwrap.dedent(r"""
    import os, select, subprocess, sys, time
    sys.path.insert(0, sys.argv[2])
    guard = sys.argv[1] == "1"
    in_fd = 0
    if guard:
        from brain import stdio_guard
        in_fd = stdio_guard.isolate()[0]
    child = subprocess.Popen([sys.executable, "-c",
        "import os,select,time; t=time.time()\n"
        "while time.time()-t<1.5:\n"
        "    if select.select([0],[],[],0.05)[0] and not os.read(0, 65536): break\n"],
        stdout=subprocess.DEVNULL)
    sys.stderr.write("READY\n"); sys.stderr.flush()
    child.wait()
    got = b""
    end = time.time() + 1.0
    while time.time() < end:
        r, _, _ = select.select([in_fd], [], [], 0.1)
        if r:
            chunk = os.read(in_fd, 65536)
            if not chunk:
                break
            got += chunk
    os.write(2, ("GOT %d\n" % got.count(b"\n")).encode())
""")


def _run_parent(guard: bool) -> int:
    p = subprocess.Popen([PY, "-c", _PARENT, "1" if guard else "0", str(REPO)], stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert p.stderr.readline().strip() == b"READY"
    time.sleep(0.2)                                    # the child is now blocked reading the inherited stdin
    for i in range(5):
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call"}).encode() + b"\n")
        p.stdin.flush()
        time.sleep(0.05)
    out = p.stderr.read().decode()
    p.wait(timeout=10)
    return int(out.split("GOT")[-1].split()[0])


def test_unguarded_child_steals_requests() -> None:
    # the bug, reproduced: the requests went to the child, the "server" never saw them
    assert _run_parent(guard=False) < 5


def test_guarded_child_cannot_steal() -> None:
    assert _run_parent(guard=True) == 5


def test_child_stdout_goes_to_stderr_not_protocol() -> None:
    code = textwrap.dedent(r"""
        import os, subprocess, sys
        sys.path.insert(0, sys.argv[1])
        from brain import stdio_guard
        _, out_fd = stdio_guard.isolate()
        subprocess.run(["echo", "Loading model"])          # a child / C library writing to fd 1
        print("python print to sys.stdout", file=sys.stdout, flush=True)
        os.write(out_fd, b'{"jsonrpc":"2.0"}\n')
    """)
    r = subprocess.run([PY, "-c", code, str(REPO)], capture_output=True, timeout=20)
    assert r.stdout == b'{"jsonrpc":"2.0"}\n'               # protocol stream holds ONLY the frame
    assert b"Loading model" in r.stderr and b"python print" in r.stderr


def test_isolate_idempotent_and_kill_switch() -> None:
    code = textwrap.dedent(r"""
        import os, sys
        sys.path.insert(0, sys.argv[1])
        from brain import stdio_guard
        print("OFF" if stdio_guard.isolate() is None else "ON", file=sys.stderr)
    """)
    r = subprocess.run([PY, "-c", code, str(REPO)], capture_output=True, timeout=20,
                       env={**os.environ, "IRIS_STDIO_ISOLATE": "0"})
    assert r.stderr.strip().endswith(b"OFF")
    code2 = code.replace('print("OFF"', 'a = stdio_guard.isolate(); print("SAME" if a == stdio_guard.isolate() else "DIFF", file=sys.stderr); print("OFF"')
    r2 = subprocess.run([PY, "-c", code2, str(REPO)], capture_output=True, timeout=20)
    assert b"SAME" in r2.stderr


# End to end: a real FastMCP server run through stdio_guard.run(); one tool starts an stdin-reading child in the
# background (like the status board's ssh loop), and every later call must still be answered.
_SERVER = textwrap.dedent(r"""
    import subprocess, sys, threading
    sys.path.insert(0, sys.argv[1])
    from brain import stdio_guard
    stdio_guard.isolate()
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("t")

    @mcp.tool()
    def spawn_reader() -> str:
        def _bg():
            subprocess.run([sys.executable, "-c",
                "import os,select,time; t=time.time()\n"
                "while time.time()-t<3:\n"
                "    if select.select([0],[],[],0.05)[0] and not os.read(0, 65536): time.sleep(0.05)\n"],
                stdout=subprocess.DEVNULL)
        threading.Thread(target=_bg, daemon=True).start()
        return "started"

    @mcp.tool()
    def ping(n: int) -> str:
        print("stray print", n)                    # must not reach the protocol stream
        return "pong %d" % n

    stdio_guard.run(mcp)
""")


def test_fastmcp_end_to_end_with_stdin_reading_child() -> None:
    p = subprocess.Popen([PY, "-c", _SERVER, str(REPO)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)

    def send(obj: dict) -> None:
        p.stdin.write((json.dumps(obj) + "\n").encode())
        p.stdin.flush()

    def recv() -> dict:
        line = p.stdout.readline()
        assert line, "server closed stdout"
        return json.loads(line)                    # every stdout line must be JSON

    try:
        send({"jsonrpc": "2.0", "id": 0, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "t", "version": "0"}}})
        assert recv()["id"] == 0
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "spawn_reader", "arguments": {}}})
        assert recv()["id"] == 1
        time.sleep(0.3)
        for i in range(2, 12):
            send({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": "ping", "arguments": {"n": i}}})
            r = recv()
            assert r["id"] == i and ("pong %d" % i) in json.dumps(r)
            time.sleep(0.1)
    finally:
        p.stdin.close()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def test_runtime_wires_the_guard_before_imports_and_at_run() -> None:
    src = (REPO / "iris_runtime.py").read_text(encoding="utf-8")
    iso = src.index("_stdio_guard.isolate()")
    assert iso < src.index("from mcp.server.fastmcp import FastMCP")      # before anything heavy can spawn a child
    assert iso < src.index("SHADOW-AVA KILL SWITCH")
    assert "_stdio_guard_run.run(mcp)" in src
    head = src[:iso]
    assert 'if __name__ == "__main__":' in head[-1200:]                   # never on import (tests)
