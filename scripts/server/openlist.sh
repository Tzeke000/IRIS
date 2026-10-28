#!/bin/bash
# openlist.sh — the temple's ALIVE open list (profiles/vr_temple/open_list.md) without reading 35 KB of it.
#   openlist.sh show [open]              every unchecked "- [ ]" line (line no. + text ≤160 chars), under its section
#   openlist.sh show SECTION-substring   the same, one section only
#   openlist.sh sections                 "## " headings: line no., open count, title
#   openlist.sh add "SECTION-substr" "text"     append "- [ ] text" at the end of that ## section (must match ONE section)
#   openlist.sh tick "item-substr" ["note"]     "- [ ] X" → "- [x] ✅ (MM-DD) note — X" (must match ONE open item)
#   Matching is case-insensitive. Edits also bump the front-matter "updated:" date. OPENLIST=path overrides the file (tests).
#   (Built 10-28, Zeke: "tools you need — simpler to use, fewer tokens".)
exec python3 - "$@" <<'PY'
import sys, os, re, datetime
F = os.environ.get('OPENLIST') or '/home/iris/IRIS/profiles/vr_temple/open_list.md'
a = sys.argv[1:]
cmd = a[0] if a else 'show'
OPEN = re.compile(r'^(\s*)- \[ \] ?')
lines = open(F, encoding='utf-8').read().split('\n')

def sections():
    """[(title, start_idx, end_idx_exclusive)] for every '## ' heading."""
    hs = [i for i, l in enumerate(lines) if l.startswith('## ')]
    return [(lines[s][3:].strip(), s, (hs[k + 1] if k + 1 < len(hs) else len(lines))) for k, s in enumerate(hs)]

def cut(s, n): return s if len(s) <= n else s[:n - 1] + '…'

def find_sec(sub):
    m = [x for x in sections() if sub.lower() in x[0].lower()]
    if len(m) != 1:
        print('✗ %d sections match "%s"%s' % (len(m), sub, ':' if m else ' — `openlist.sh sections`'))
        for t, s, e in m: print('  L%d %s' % (s + 1, cut(t, 100)))
        sys.exit(1)
    return m[0]

def save():
    today = datetime.date.today().isoformat()
    if lines and lines[0] == '---':
        for i in range(1, min(len(lines), 15)):
            if lines[i] == '---': break
            if lines[i].startswith('updated:'): lines[i] = 'updated: ' + today; break
    tmp = F + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f: f.write('\n'.join(lines))
    os.replace(tmp, F)

if cmd == 'sections':
    for t, s, e in sections():
        n = sum(1 for l in lines[s:e] if OPEN.match(l))
        print('L%-4d %2d open  %s' % (s + 1, n, cut(t, 90)))
elif cmd == 'show':
    want = a[1] if len(a) > 1 and a[1] != 'open' else None
    secs = [find_sec(want)] if want else [('(top)', 0, sections()[0][1] if sections() else len(lines))] + sections()
    total = 0
    for t, s, e in secs:
        items = [(i, l) for i, l in enumerate(lines[s:e], s) if OPEN.match(l)]
        if not items: continue
        print('## ' + cut(t, 70))
        for i, l in items: print('%4d %s' % (i + 1, cut(OPEN.sub(r'\1', l).rstrip(), 160)))
        total += len(items)
    print('(%d open)' % total)
elif cmd == 'add':
    if len(a) < 3 or not a[2].strip(): print('usage: openlist.sh add "SECTION-substring" "text"'); sys.exit(1)
    t, s, e = find_sec(a[1])
    j = e
    while j > s + 1 and not lines[j - 1].strip(): j -= 1      # after the section's last non-blank line
    lines.insert(j, '- [ ] ' + a[2].strip()); save()
    print('✓ added L%d under "%s"' % (j + 1, cut(t, 60)))
elif cmd == 'tick':
    if len(a) < 2 or not a[1].strip(): print('usage: openlist.sh tick "item-substring" ["note"]'); sys.exit(1)
    m = [i for i, l in enumerate(lines) if OPEN.match(l) and a[1].lower() in l.lower()]
    if len(m) != 1:
        print('✗ %d open items match "%s"%s' % (len(m), a[1], ':' if m else ''))
        for i in m[:12]: print('  %4d %s' % (i + 1, cut(OPEN.sub(r'\1', lines[i]), 140)))
        sys.exit(1)
    i = m[0]; ind = OPEN.match(lines[i]).group(1); rest = OPEN.sub('', lines[i])
    note = a[2].strip() if len(a) > 2 else ''
    lines[i] = '%s- [x] ✅ (%s) %s%s' % (ind, datetime.date.today().strftime('%m-%d'), (note + ' — ') if note else '', rest)
    save(); print('✓ ticked L%d: %s' % (i + 1, cut(lines[i], 140)))
else:
    print('usage: openlist.sh show [open|SECTION] | sections | add "SECTION" "text" | tick "item" ["note"]'); sys.exit(1)
PY
