#!/usr/bin/env python3
"""PreToolUse guard for Read: block images too big to pass through the SDK pipe.

Why: a tool result over 1 MB (base64 JSON) kills the body host
(CLIJSONDecodeError, buffer 1048576). 2026-11-02 01:46 a Read of a 2.8 MB cover
PNG produced a 1.27 MB result line and took me down for 12 min. Images also
wedge the host well below 1 MB (08-29: 476 KB PNG). So: images > 150 KB are
denied with a downscale recipe. Fails OPEN on any error.
"""
import json, os, sys

LIMIT = 150 * 1024
IMG = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff')
try:
    ev = json.load(sys.stdin)
    if ev.get('tool_name') != 'Read':
        sys.exit(0)
    p = (ev.get('tool_input') or {}).get('file_path') or ''
    if p.lower().endswith(IMG) and os.path.isfile(p):
        size = os.path.getsize(p)
        if size > LIMIT:
            base = os.path.splitext(p)[0]
            msg = (f"BLOCKED: {p} is {size // 1024} KB; images over {LIMIT // 1024} KB can crash the host "
                   f"(>1 MB tool result = death, 11-02). Downscale first, then Read the copy:\n"
                   f"/home/iris/venvs/iris-v100/bin/python -c \"from PIL import Image; im=Image.open('{p}'); "
                   f"im.thumbnail((800,800)); im.convert('RGB').save('{base}_small.jpg', quality=80)\"")
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                  "permissionDecision": "deny", "permissionDecisionReason": msg}}))
except Exception:
    pass
sys.exit(0)
