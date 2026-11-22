#!/usr/bin/env python3
"""Pure-Python reference of rules.md. Slow but literal. Writes CSVs (for fixture checking)."""
import glob, json, os, re, sys, csv
from datetime import datetime, timezone, date, timedelta

DAY = 86400000
ASCII_DIG = re.compile(r'^[0-9]+$')
F1 = re.compile(r'^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z$')
F2 = re.compile(r'^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})([+-])([0-9]{2}):([0-9]{2})$')


def parse_line(line, n):
    parts = line.split(',')
    if len(parts) != n:
        return None
    out = []
    for p in parts:
        if p.startswith('"'):
            if len(p) < 2 or not p.endswith('"') or '"' in p[1:-1]:
                return None
            p = p[1:-1]
        elif '"' in p:
            return None
        out.append(p if p != '' else None)
    return out


def read_csv(path, n):
    with open(path, 'rb') as f:
        txt = f.read().decode('utf-8')
    lines = txt.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    rows = []
    for ln in lines[1:]:
        r = parse_line(ln, n)
        if r is not None:
            rows.append(r)
    return rows


def days_from_civil(y, m, d):
    return (date(y, m, d) - date(1970, 1, 1)).days


def to_ms(s):
    if s is None:
        return None
    if ASCII_DIG.match(s):
        return int(s)
    m = F1.match(s)
    off = 0
    if m:
        g = m.groups()
    else:
        m = F2.match(s)
        if not m:
            return None
        g = m.groups()[:6]
        sign, oh, om = m.group(7), int(m.group(8)), int(m.group(9))
        if oh > 23 or om > 59:
            return None
        off = (oh * 60 + om) * 60000 * (1 if sign == '+' else -1)
    y, mo, d, h, mi, se = map(int, g[:6])
    if y < 1 or mo < 1 or mo > 12 or h > 23 or mi > 59 or se > 59:
        return None
    try:
        date(y, mo, d)
    except ValueError:
        return None
    return ((days_from_civil(y, mo, d) * 24 + h) * 60 + mi) * 60000 + se * 1000 - off


def main(data, out):
    man = json.load(open(os.path.join(data, 'manifest.json')))
    ws = to_ms(man['window_start']); we = to_ms(man['window_end'])
    cat = {r[0]: r[1] for r in read_csv(os.path.join(data, 'products.csv'), 2) if r[0] is not None}
    camp = {r[0]: r[1] for r in read_csv(os.path.join(data, 'campaigns.csv'), 2) if r[0] is not None}

    # links
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in sorted(glob.glob(os.path.join(data, 'links', '*.csv'))):
        for a, b in read_csv(p, 2):
            if a is None or b is None:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    members = {}
    for x in list(parent):
        members.setdefault(find(x), []).append(x)
    name = {}
    for r, ms in members.items():
        mn = min(ms)
        for x in ms:
            name[x] = mn
    person = lambda u: name.get(u, u)

    # events
    best = {}
    for p in sorted(glob.glob(os.path.join(data, 'events', '*.csv'))):
        for r in read_csv(p, 7):
            if r[0] is None:
                continue
            ing = to_ms(r[3])
            k = r[0]
            if k not in best or ing < best[k][0]:
                best[k] = (ing, r)
    evs = []
    for k, (ing, r) in best.items():
        _, uid, ts, _, et, pid, price = r
        if uid is None: continue
        t = to_ms(ts)
        if t is None or not (ws <= t < we): continue
        if et not in ('view', 'cart', 'purchase'): continue
        if pid is None: continue
        pc = 0
        if et == 'purchase':
            if price is None or not ASCII_DIG.match(price): continue
            pc = int(price)
            if not (1 <= pc <= 10000000): continue
        if ing - t > DAY: continue
        evs.append((person(uid), t, et, pid, pc))

    # clicks
    clicks = {}
    for p in sorted(glob.glob(os.path.join(data, 'clicks', '*.csv'))):
        for r in read_csv(p, 4):
            if r[0] is None: continue
            cid, uid, cmp_, ts = r
            if uid is None: continue
            t = to_ms(ts)
            if t is None or not (ws - 7 * DAY <= t < we): continue
            if cmp_ not in camp: continue
            clicks[cid] = (person(uid), t, cmp_)
    byp = {}
    for cid, (pe, t, c) in clicks.items():
        byp.setdefault(pe, []).append((t, cid, c))

    # sessions
    evp = {}
    for e in evs:
        evp.setdefault(e[0], []).append(e)
    sessions = []  # (person, start, events, revenue, catrev dict)
    for pe, es in evp.items():
        es.sort(key=lambda e: e[1])
        cur = None; last = None
        for e in es:
            if cur is None or e[1] - last > 1800000:
                cur = [pe, e[1], 0, 0, {}]
                sessions.append(cur)
            last = e[1]
            cur[2] += 1
            if e[2] == 'purchase':
                cur[3] += e[4]
                c = cat.get(e[3], 'unknown')
                cur[4][c] = cur[4].get(c, 0) + e[4]
    ud = {}; cd = {}
    for pe, st, n, rev, cr in sessions:
        d = (date(1970, 1, 1) + timedelta(days=st // DAY)).isoformat()
        u = ud.setdefault((pe, d), [0, 0, 0, {}])
        u[0] += 1; u[1] += n; u[2] += rev
        for c, v in cr.items():
            u[3][c] = u[3].get(c, 0) + v
        cl = [x for x in byp.get(pe, []) if x[0] <= st and st - x[0] <= 7 * DAY]
        if cl:
            mt = max(x[0] for x in cl)
            top = min((x for x in cl if x[0] == mt), key=lambda x: x[1].encode())
            cc = top[2]
        else:
            cc = 'organic'
        row = cd.setdefault((cc, d), [0, 0, 0, 0])
        row[0] += 1; row[1] += 1 if rev > 0 else 0; row[2] += rev
        for c in {x[2] for x in cl}:
            r2 = cd.setdefault((c, d), [0, 0, 0, 0]); r2[3] += 1
    os.makedirs(os.path.join(out, 'user_daily'), exist_ok=True)
    os.makedirs(os.path.join(out, 'campaign_daily'), exist_ok=True)
    with open(os.path.join(out, 'user_daily', 'o.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['user_id', 'day', 'sessions', 'events', 'revenue_cents', 'top_category'])
        for (pe, d), (s, n, rev, cr) in ud.items():
            tc = ''
            if cr:
                mv = max(cr.values())
                tc = min(c for c, v in cr.items() if v == mv)
            w.writerow([pe, d, s, n, rev, tc])
    with open(os.path.join(out, 'campaign_daily', 'o.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['campaign_id', 'channel', 'day', 'sessions', 'purchasing_sessions', 'revenue_cents', 'assisted_sessions'])
        for (c, d), v in cd.items():
            w.writerow([c, 'organic' if c == 'organic' else camp[c], d] + v)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
