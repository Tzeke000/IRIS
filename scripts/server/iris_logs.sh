#!/bin/bash
# iris_logs.sh PATTERN [--days N] [--ctx C] [--max M] [--all]
#   Search my own Claude Code transcripts (main sessions + subagents) for PATTERN — newest first, JSON DECODED, so it matches
#   the real message text / tool inputs / tool outputs, not the escaped JSON. One line per hit:
#     <session8>[/<agent6>] <local MM-DD HH:MM> <role|→Tool|←Tool> …±C chars of context…
#   PATTERN = case-insensitive regex (falls back to a literal if it isn't valid regex). Defaults: --ctx 120 --max 15, all days.
#   Identical snippets are shown once; output is capped at ~4 KB; files are streamed one line at a time (multi-MB lines OK).
#   Hits inside an iris_logs.sh call itself are skipped (the search would find itself); --all keeps them.
#   (Built 10-28, Zeke: "tools you need — simpler to use, fewer tokens".)
exec python3 - "$@" <<'PY'
import sys, os, re, json, glob, time, heapq, collections
from datetime import datetime, timezone

a = sys.argv[1:]
if not a or a[0] in ('-h', '--help'):
    print("usage: iris_logs.sh PATTERN [--days N] [--ctx C] [--max M] [--all]"); sys.exit(0 if a else 1)
pat, days, ctx, mx, keep_self = None, None, 120, 15, False
i = 0
while i < len(a):
    x = a[i]
    if x == '--days': days = float(a[i + 1]); i += 2
    elif x == '--ctx': ctx = int(a[i + 1]); i += 2
    elif x == '--max': mx = int(a[i + 1]); i += 2
    elif x == '--all': keep_self = True; i += 1
    elif pat is None: pat = x; i += 1
    else: print("unexpected arg: %s" % x); sys.exit(1)
if not pat: print("no PATTERN"); sys.exit(1)
try: rx = re.compile(pat, re.I)
except re.error: rx = re.compile(re.escape(pat), re.I)
# cheap raw-line prefilter, only when it is SAFE: a plain literal (JSON escaping never touches these characters)
lit = pat.lower() if re.fullmatch(r"[A-Za-z0-9 _.,:=/@+-]+", pat) else None

D = os.path.expanduser('~/.claude/projects/-home-iris-IRIS')
files = glob.glob(D + '/*.jsonl') + glob.glob(D + '/*/subagents/*.jsonl')
now = time.time()
if days is not None: files = [f for f in files if os.path.getmtime(f) >= now - days * 86400]
files.sort(key=os.path.getmtime, reverse=True)
cut_ts = None if days is None else datetime.fromtimestamp(now - days * 86400, timezone.utc)

def short(f):
    rel = os.path.relpath(f, D)
    if '/subagents/' in rel:
        sess, _, ag = rel.split('/'); return sess[:8] + '/' + ag.replace('agent-', '').replace('.jsonl', '')[:6]
    return rel[:8]

def texts(o, names):
    """(label, text) pieces of one transcript line."""
    m = o.get('message') or {}
    c = m.get('content'); role = o.get('type') or m.get('role') or '?'
    if isinstance(c, str): yield role, c; return
    if not isinstance(c, list): return
    for b in c:
        if not isinstance(b, dict): continue
        t = b.get('type')
        if t == 'text': yield role, b.get('text', '')
        elif t == 'tool_use':
            names[b.get('id')] = b.get('name', '?')
            inp = b.get('input', {})
            s = inp.get('command') if isinstance(inp, dict) and isinstance(inp.get('command'), str) else json.dumps(inp, ensure_ascii=False)
            yield '→' + b.get('name', '?'), s
        elif t == 'tool_result':
            r = b.get('content')
            if isinstance(r, list): r = '\n'.join(x.get('text', '') for x in r if isinstance(x, dict) and x.get('type') == 'text')
            yield '←' + names.get(b.get('tool_use_id'), 'tool'), r if isinstance(r, str) else ''

hits = []          # (iso_ts, file_short, label, snippet)
seen = set()
scanned = 0
for f in files:
    # every timestamp in a file is <= its mtime: once we hold mx hits all newer than this file, stop
    if len(hits) >= mx:
        nth = sorted(h[0] for h in hits)[-mx]
        if datetime.fromtimestamp(os.path.getmtime(f), timezone.utc).isoformat() < nth: break
    scanned += 1
    per = collections.deque(maxlen=mx)     # newest mx hits of THIS file (we read it forward)
    names = {}
    try: fh = open(f, 'r', encoding='utf-8', errors='replace')
    except OSError: continue
    with fh:
        for line in fh:
            if lit is not None and lit not in line.lower() and '"tool_use"' not in line: continue
            try: o = json.loads(line)
            except ValueError: continue
            ts = o.get('timestamp') or ''
            if cut_ts and ts and ts < cut_ts.isoformat().replace('+00:00', 'Z'): continue
            for label, s in texts(o, names):
                if not s: continue
                if not keep_self and 'iris_logs' in s and label.startswith(('→', '←')): continue
                mm = rx.search(s)
                if not mm: continue
                a0, b0 = max(0, mm.start() - ctx), min(len(s), mm.end() + ctx)
                snip = ('…' if a0 else '') + re.sub(r'\s+', ' ', s[a0:b0]) + ('…' if b0 < len(s) else '')
                key = re.sub(r'\s+', ' ', s[max(0, mm.start() - 40):mm.end() + 40]).strip()
                per.append((ts, short(f), label, snip, key))
    for h in per:
        if h[4] in seen: continue
        seen.add(h[4]); hits.append(h[:4])

hits.sort(key=lambda h: h[0], reverse=True)
hits = hits[:mx]
out = ["%d hit(s) for /%s/ · %d of %d files scanned · newest first · local time" % (len(hits), pat, scanned, len(files))]
for ts, fs, label, snip in hits:
    try: lt = datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone().strftime('%m-%d %H:%M')
    except ValueError: lt = '?'
    out.append("%s %s %s %s" % (fs, lt, label[:14], snip))
body, total = [], 0
for l in out:
    if total + len(l.encode()) + 1 > 4000: body.append("… (output capped at 4 KB — narrow the pattern or lower --ctx/--max)"); break
    body.append(l); total += len(l.encode()) + 1
print('\n'.join(body))
PY
