"""STDIO GUARD (2026-11-07, mcphang_1107) — the MCP pipe belongs to the MCP transport and NOTHING else.

The intermittent 180 s MCP hang (11-02 .. 11-06: wake_verdict, time_check, llm_reply, vr_temple speak/follow): the
11-05 call ledger (brain/mcp_call_ledger.py) proved the hung requests NEVER reached a handler and were never processed
late either — the request bytes simply vanished from the stdin pipe. Cause: every child process the runtime spawns
inherits fd 0 (the CLI -> runtime JSON-RPC pipe) and fd 1 (runtime -> CLI). The status board (brain/app_status_board.py,
inside this process via orb_http) runs `ssh` to the Proxmox host every ~6 s and to the tower every 30-60 s without
`-n`/stdin=DEVNULL; ssh reads its stdin and forwards it to the remote command. A request the CLI writes while such an
ssh is alive can be read by ssh instead of by the MCP reader thread -> shipped to the remote shell -> gone; the CLI
waits 180 s, the next request (no ssh alive) works instantly. Children (and C libraries) writing to fd 1 also put
non-JSON lines into the response stream ("Ignoring non-JSON line on stdout: Loading/Using" in the CLI's log).

Fix, once, for every spawn site: move the MCP pipe to PRIVATE non-inheritable fds, then point fd 0 at /dev/null and
fd 1 at stderr. Children now read EOF and write to the log; only the transport can touch the protocol.
Kill switch: IRIS_STDIO_ISOLATE=0.
"""
from __future__ import annotations

import os
import sys
from io import TextIOWrapper
from typing import Any

_ISOLATED: "tuple[int, int] | None" = None


def enabled() -> bool:
    return os.name == "posix" and os.environ.get("IRIS_STDIO_ISOLATE", "1") != "0"


def isolate(in_fd: int = 0, out_fd: int = 1, err_fd: int = 2) -> "tuple[int, int] | None":
    """Dup the protocol fds to private (O_CLOEXEC, PEP 446) copies, then re-point in_fd at /dev/null and out_fd at
    err_fd. Returns (private_in, private_out); idempotent; None when disabled or on failure (nothing changed)."""
    global _ISOLATED
    if _ISOLATED is not None:
        return _ISOLATED
    if not enabled():
        return None
    try:
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        priv_in = os.dup(in_fd)             # os.dup -> non-inheritable: no child ever gets these
        priv_out = os.dup(out_fd)
        dn = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(dn, in_fd)              # dup2 targets stay inheritable: children get /dev/null + stderr
        finally:
            os.close(dn)
        os.dup2(err_fd, out_fd)
    except Exception as e:  # noqa: BLE001
        print(f"[stdio_guard] isolate failed, running unguarded: {e!r}", file=sys.stderr, flush=True)
        return None
    _ISOLATED = (priv_in, priv_out)
    return _ISOLATED


def private_fds() -> "tuple[int, int] | None":
    return _ISOLATED


def streams(fds: "tuple[int, int]") -> "tuple[Any, Any]":
    """anyio AsyncFile wrappers over the private fds, built exactly like mcp.server.stdio.stdio_server's defaults."""
    import anyio
    rin = os.fdopen(fds[0], "rb", closefd=False)
    rout = os.fdopen(fds[1], "wb", closefd=False)
    stdin = anyio.wrap_file(TextIOWrapper(rin, encoding="utf-8", errors="replace"))
    stdout = anyio.wrap_file(TextIOWrapper(rout, encoding="utf-8"))
    return stdin, stdout


async def run_fastmcp_stdio(mcp: Any, fds: "tuple[int, int]") -> None:
    """FastMCP.run_stdio_async (mcp 1.27.1), but on the private fds."""
    from mcp.server.stdio import stdio_server
    stdin, stdout = streams(fds)
    async with stdio_server(stdin=stdin, stdout=stdout) as (read_stream, write_stream):
        await mcp._mcp_server.run(read_stream, write_stream, mcp._mcp_server.create_initialization_options())


def run(mcp: Any) -> None:
    """Drop-in for mcp.run(): guarded transport when isolate() succeeded, the stock one otherwise."""
    fds = _ISOLATED
    if fds is None:
        mcp.run()
        return
    import anyio
    anyio.run(run_fastmcp_stdio, mcp, fds)
