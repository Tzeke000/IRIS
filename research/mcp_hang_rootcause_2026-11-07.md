# The intermittent 180 s MCP hang: root cause (2026-11-07, helper mcphang_1107)

## Verdict
**Child processes were stealing JSON-RPC requests off the MCP stdin pipe.** Every child process the runtime spawns
inherits fd 0, which is the pipe the CLI writes requests into. The status board (`brain/app_status_board.py`) runs
inside the runtime through orb_http, and it runs `ssh` the whole time: Proxmox `stats` about every 6 s, and the tower
(`echo iris-ok` every 30 s, `desktop_bridge` and `mic_watchdog` every 60 s). None of those calls pass `-n` or
`stdin=DEVNULL`. Without `-n`, ssh reads its stdin and forwards it to the remote command. If a request line arrives
while one of those ssh processes is reading the pipe, ssh can get the bytes before the MCP reader thread does. The
request then goes to the remote shell and is lost. The CLI waits until its 180 s timeout. The next request usually
arrives when no ssh is reading, so it works straight away.

This is a race, so it is intermittent. Any tool can be hit. vr_temple speak/follow showed up most only because they
were the calls being made most often while Zeke was in VR.

## Evidence
1. **The hung requests never reached a handler.** `brain/mcp_call_ledger.py` (added 11-05) writes a `start` row
   whenever FastMCP's tool manager receives a call. The CLI log
   (`~/.cache/claude-cli-nodejs/-home-iris-IRIS/mcp-logs-iris/`) has 3 timeouts after the ledger went in, and the
   ledger has **no row for any of them**:
   | CLI sent (UTC) | ledger around it |
   |---|---|
   | 11-06 05:53:36.667 | seq 26 ends 05:53:35.575, and then nothing. No row for the hung call |
   | 11-07 03:07:24.923 | seq 81 ends 03:07:04.9, seq 82 starts 03:10:30.046 (= the CLI's next send at .035) |
   | 11-07 03:55:27.425 | seq 181 ends 03:55:03.2, seq 182 starts 03:58:32.080 (= the CLI's next send at .071) |
   The sequence numbers have no gaps, and the next call's start row matches the CLI's next send to the millisecond.
   So the hung request was not slow and not processed late. The server never received it.
   No call longer than 60 s appears in the ledger (`state/mcp_slow_calls.log` was never created).
2. **That rules out a, b and d.** Everything between `stdin_reader` and `call_tool` in mcp 1.27.1 is pure async
   (`mcp/server/stdio.py`, `shared/session.py::_receive_loop`, `server/lowlevel/server.py::run/_handle_request`).
   If the loop or a lock had blocked, the request would have been processed late and left a start row. There is no
   such row. Thread count has been flat at about 180–250 for days (`state/runtime_rss.jsonl`). The 704 seen on 11-02
   was before the efficiency fixes.
3. **The thieves are live right now.** Sampling the children of the runtime pid (3640) at 50 ms intervals for 60 s:
   about 3.6 % of samples had an `ssh` child alive: `ssh -i …pve_stats … stats` every ~6 s, plus
   `ssh … tower echo iris-ok`, `… desktop_bridge.py submit ping`, and `… type …mic_watchdog.json`. The code that
   spawns them is `brain/app_status_board.py:45-58` (`_run`/`_ssh_tower`, no `stdin=`) and `:286` (the Proxmox loop,
   which uses ControlMaster, so the mux master reads the stdin fd that the client passes to it).
   `tools/system/voice_delivery_check_tool.py:29` (ssh to the tower, no `stdin=`) ran 0.6 s before the 11-06 hang.
   23 more files under brain/ and tools/ spawn processes with no `stdin=` at all.
4. **Reproduced.** In `tests/test_stdio_guard_1107.py::test_unguarded_child_steals_requests`, a child that inherits
   stdin takes the request lines and the "server" receives fewer than 5 of 5. With the guard on, it receives 5 of 5.
5. **Hypothesis c is also real, but a different bug.** The CLI log shows `Ignoring non-JSON line on stdout: "Loading"
   / "Using"` once per boot, and once mid-session (11-07 14:50Z). Something (a child process or a C library) is
   writing to fd 1, which is the response stream. A partial line written just before a frame would corrupt that
   response and cause the same 180 s symptom. Python `print` was already sent to stderr (`iris_runtime.py:34-47`), but
   fd-level writes were not.

## Fix (branch `mcphang-1107`)
- **`brain/stdio_guard.py`**: as the very first thing in the server process, `os.dup` fds 0 and 1 to private copies
  (PEP 446: not inherited by children). Then point fd 0 at `/dev/null` and fd 1 at stderr. The MCP transport runs
  `stdio_server(stdin=…, stdout=…)` on the private fds, the same way `FastMCP.run_stdio_async` does. As a result:
  children see EOF on stdin and write to the log; Python `sys.stdout.write`, C `printf` and child output can no longer
  reach the protocol stream. This one change covers every spawn site, including ones added later.
  Kill switch: `IRIS_STDIO_ISOLATE=0`. If it is disabled or fails, the server falls back to the stock `mcp.run()`.
- **`iris_runtime.py`**: calls `isolate()` right after the print reroute, inside `if __name__ == "__main__"`, so it
  runs before any import can spawn a child and never runs when tests import the module. `mcp.run()` is replaced with
  `stdio_guard.run(mcp)`.
- **Extra safety:** added `stdin=subprocess.DEVNULL` to the in-runtime ssh/scp/systemctl calls (status board `_run`,
  voice_delivery_check `_run`, app_guide_routes ssh+scp, vr_temple `_unit`).
- Already on master and unchanged: vr_temple runs off the event loop with a 150 s deadline (10-29 / 11-05), and the
  ledger stays as the tripwire.

## Tests
`tests/test_stdio_guard_1107.py`: 6 tests, all pass, about 6 s, and none need the live runtime:
- the unguarded child steals requests (the bug reproduced)
- the guarded child cannot steal them
- child and C-level stdout goes to stderr, and the protocol stream contains only JSON frames
- `isolate()` is idempotent and the kill switch works
- **end to end:** a real FastMCP server run through `stdio_guard.run()`, with a background child reading stdin and a
  tool that prints. 10 calls during that window are all answered, with only JSON on stdout. With
  `IRIS_STDIO_ISOLATE=0` this test fails.
- the runtime calls `isolate()` before its heavy imports, only under `__main__`, and uses `stdio_guard.run`

## What it takes to land
Merge `mcphang-1107` and then **restart the runtime**, i.e. a body-host / CC session restart that respawns
`iris_runtime.py`. `iris_tool_reload` and hot-swap are not enough: the isolation has to happen at process start.
The `stdin=DEVNULL` edits land with the same restart.
**After it lands, check:** the ledger shows calls flowing; the CLI log has no new "Ignoring non-JSON line"; there
are no 180 s timeouts over a busy VR night; and the status board still shows green for the tower and Proxmox
(ssh with stdin=/dev/null works the same for these commands).
**Rollback:** set `IRIS_STDIO_ISOLATE=0` in the runtime's environment, or revert the two commits.
