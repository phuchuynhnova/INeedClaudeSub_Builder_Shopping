#!/usr/bin/env python3
"""
Shopping Sessions: DuckDB pipeline, v2 (faster).

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)

Design notes
- CSVs are read with DuckDB's native parser into VARCHAR columns with quoting switched off.
  Lines with the wrong field count are skipped by the parser (ignore_errors); quote rules of
  rules.md (quotes only around a whole field) are checked in SQL (qok / unq).
  If the native reader raises, the files are re-read as plain text lines and parsed by a
  regex (slow path, always correct, used e.g. for files with a \\r).
- Validity of every event row is decided BEFORE dedup, then the dedup keeps the row with the
  smallest ingest_ts and its verdict: that is "dedup, then validate the kept copy".
- Persons are keyed by a 64-bit hash of their name (smallest linked ID) in all big tables;
  names are restored at the very end.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", os.cpu_count() or 1))
MEM_GB = int(os.environ.get("MEM_GB", "8"))

# the fixtures runner passes DATA_DIR and OUTPUT_DIR as the last two arguments too
if len(sys.argv) >= 3:
    DATA, OUT = sys.argv[-2], sys.argv[-1]

DAY = 86_400_000
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.2f}s] {msg}", file=sys.stderr, flush=True)


con = duckdb.connect()
con.execute(f"SET threads = {NCPU}")
con.execute(f"SET memory_limit = '{max(1, int(MEM_GB * 0.8))}GB'")
os.makedirs(os.path.join(SCRATCH, "duckdb_tmp"), exist_ok=True)
con.execute(f"SET temp_directory = '{os.path.join(SCRATCH, 'duckdb_tmp')}'")
con.execute("SET preserve_insertion_order = false")

with open(os.path.join(DATA, "manifest.json")) as f:
    manifest = json.load(f)


def iso_ms(s):
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


WS = iso_ms(manifest["window_start"])
WE = iso_ms(manifest["window_end"])

# ---------------------------------------------------------------- sources

FIELD = r'(?:"[^",]*"|[^",]*)'
VIEWS = {}


def row_regex(n):
    return FIELD + "(?:," + FIELD + "){" + str(n - 1) + "}"


# qok: a field is acceptable iff it has no quote, or is wrapped in quotes with none inside.
# unq: remove the wrapping quotes; an empty field (quoted or not) is NULL.
con.execute(
    """
CREATE OR REPLACE MACRO qok(x) AS
  (x IS NULL OR position('"' IN x) = 0
   OR (length(x) >= 2 AND substr(x, 1, 1) = '"' AND substr(x, length(x), 1) = '"'
       AND position('"' IN substr(x, 2, length(x) - 2)) = 0))
"""
)
con.execute(
    """
CREATE OR REPLACE MACRO unq(x) AS
  CASE WHEN x IS NOT NULL AND substr(x, 1, 1) = '"' THEN nullif(substr(x, 2, length(x) - 2), '') ELSE x END
"""
)


def src(view, pattern, n, slow=False):
    """View `view` with columns f1..fn: the well-formed lines of the files matching `pattern`,
    quotes removed, empty fields NULL."""
    VIEWS[view] = (pattern, n)
    cols = ", ".join(f"'f{i}': 'VARCHAR'" for i in range(1, n + 1))
    sel = ", ".join(f"unq(f{i}) AS f{i}" for i in range(1, n + 1))
    if not slow:
        ok = " AND ".join(f"qok(f{i})" for i in range(1, n + 1))
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT {sel} FROM read_csv('{pattern}', header = true, columns = {{{cols}}},
                delim = ',', quote = '', escape = '', new_line = '\\n',
                auto_detect = false, ignore_errors = true, parallel = true)
            WHERE {ok}
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view}__lines AS
            SELECT unnest(string_split(content, chr(10))[2:]) AS line FROM read_text('{pattern}')
            """
        )
        pick = ", ".join(f"nullif(trim(f[{i}], '\"'), '') AS f{i}" for i in range(1, n + 1))
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT {pick}
            FROM (SELECT string_split(line, ',') AS f FROM {view}__lines
                  WHERE regexp_full_match(line, '{row_regex(n)}'))
            """
        )


def run(sql, *views):
    """Execute `sql`; if reading one of `views` fails in the fast path, switch them to the
    slow path and run it again."""
    try:
        return con.execute(sql)
    except duckdb.Error as error:
        log(f"fast CSV path failed ({str(error).splitlines()[0]}); using the slow line reader")
        for v in views:
            src(v, *VIEWS[v], slow=True)
        return con.execute(sql)


# ts -> epoch milliseconds, NULL when bad. The strict regexes decide the FORM (digits only,
# hours 00-23, minutes/seconds 00-59, offset range, exact 'Z'); strptime then converts and
# rejects impossible calendar dates (Feb 30, Feb 29 outside leap years). F3 values too big
# for BIGINT are NULL (they are outside every window anyway).
con.execute(
    r"""
CREATE OR REPLACE MACRO ts_ms(x) AS
  CASE
    WHEN x IS NULL THEN NULL
    WHEN regexp_full_match(x, '[0-9]+') THEN TRY_CAST(ltrim(x, '0') AS BIGINT)
    WHEN regexp_full_match(x, '[0-9]{4}-[0-9]{2}-[0-9]{2}T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z')
      THEN CAST(epoch(try_strptime(x, '%Y-%m-%dT%H:%M:%SZ')) * 1000 AS BIGINT)
    WHEN regexp_full_match(x, '[0-9]{4}-[0-9]{2}-[0-9]{2}T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9][+-]([01][0-9]|2[0-3]):[0-5][0-9]')
      THEN CAST(epoch(try_strptime(x, '%Y-%m-%dT%H:%M:%S%z')) * 1000 AS BIGINT)
    ELSE NULL
  END
"""
)

# ingest_ts is always F1 (fixed width): skip the regexes, fall back to ts_ms if it is not.
con.execute(
    r"""
CREATE OR REPLACE MACRO ing_ms(x) AS
  coalesce(
    CASE WHEN length(x) = 20 AND substr(x, 20, 1) = 'Z'
         THEN CAST(epoch(try_strptime(x, '%Y-%m-%dT%H:%M:%SZ')) * 1000 AS BIGINT) END,
    ts_ms(x))
"""
)

# ---------------------------------------------------------------- small tables

src("products_raw", os.path.join(DATA, "products.csv"), 2)
src("campaigns_raw", os.path.join(DATA, "campaigns.csv"), 2)
run("CREATE TABLE prod AS SELECT f1 AS product_id, f2 AS category FROM products_raw WHERE f1 IS NOT NULL", "products_raw")
run("CREATE TABLE camp AS SELECT f1 AS campaign_id, f2 AS channel FROM campaigns_raw WHERE f1 IS NOT NULL", "campaigns_raw")

# ---------------------------------------------------------------- linked IDs

src("links_raw", os.path.join(DATA, "links", "*.csv"), 2)
run(
    """
    CREATE TABLE lk AS
    SELECT DISTINCT least(f1, f2) AS a, greatest(f1, f2) AS b
    FROM links_raw
    WHERE f1 IS NOT NULL AND f2 IS NOT NULL AND f1 <> f2
    """,
    "links_raw",
)
con.execute("CREATE TABLE ids AS SELECT id, (row_number() OVER ()) - 1 AS code FROM (SELECT a AS id FROM lk UNION SELECT b FROM lk)")
n_ids = con.execute("SELECT count(*) FROM ids").fetchone()[0]
con.execute("CREATE TABLE lk_map (id VARCHAR, person VARCHAR)")
if n_ids:
    ed = con.execute(
        "SELECT ia.code AS a, ib.code AS b FROM lk JOIN ids ia ON lk.a = ia.id JOIN ids ib ON lk.b = ib.id"
    ).fetchnumpy()
    ea = np.asarray(ed["a"], dtype=np.int64)
    eb = np.asarray(ed["b"], dtype=np.int64)
    graph = csr_matrix((np.ones(len(ea), dtype=np.int32), (ea, eb)), shape=(n_ids, n_ids))
    _, labels = connected_components(graph, directed=False)
    lab_df = pd.DataFrame({"code": np.arange(n_ids, dtype=np.int64), "comp": labels.astype(np.int64)})
    con.register("lab_df", lab_df)
    con.execute(
        """
        INSERT INTO lk_map
        SELECT i.id, m.person
        FROM ids i JOIN lab_df l ON i.code = l.code
        JOIN (SELECT l2.comp AS comp, min(i2.id) AS person
              FROM ids i2 JOIN lab_df l2 ON i2.code = l2.code GROUP BY l2.comp) m ON l.comp = m.comp
        """
    )
    con.unregister("lab_df")
log(f"links done: {n_ids} linked ids")

# ---------------------------------------------------------------- events

src("events_raw", os.path.join(DATA, "events", "*.csv"), 7)
# Per row: parse ts, decide validity (rules 3 and 4), then keep, per event_id, the row with the
# smallest ingest_ts together with its verdict `ok`. A later valid copy never rescues an
# invalid kept copy. Only valid rows carry their strings (uid, and pid for purchases).
# The event_id is grouped by two independent 64-bit hashes (collision odds ~1e-23 per dataset).
EV_ROWS = f"""
    SELECT hash(f1) AS h1, hash(f1 || '|') AS h2, f2, f6, t, ing, pv,
           coalesce(f2 IS NOT NULL AND t >= {WS} AND t < {WE}
                    AND f5 IN ('view', 'cart', 'purchase') AND f6 IS NOT NULL
                    AND (f5 <> 'purchase' OR pv BETWEEN 1 AND 10000000)
                    AND ing - t <= {DAY}, false) AS ok
    FROM (
      SELECT f1, f2, f5, f6, ts_ms(f3) AS t, ing_ms(f4) AS ing,
             CASE WHEN f5 = 'purchase'
                  THEN CASE WHEN f7 IS NOT NULL AND regexp_full_match(f7, '[0-9]+')
                            THEN TRY_CAST(ltrim(f7, '0') AS BIGINT) END
                  ELSE 0 END AS pv
      FROM events_raw WHERE f1 IS NOT NULL
    )
"""


def ev_dedup(source):
    return f"""
    CREATE TABLE ev_v AS
    SELECT r.uid AS uid, r.t AS t, r.pid AS pid, r.pv AS pv
    FROM (
      SELECT arg_min(struct_pack(ok := ok, uid := CASE WHEN ok THEN f2 END, t := t,
                                 pid := CASE WHEN ok AND pv > 0 THEN f6 END, pv := pv), ing) AS r
      FROM {source}
      GROUP BY h1, h2
    )
    WHERE r.ok
    """


if os.environ.get("DIAG"):
    # diagnostics: time the parse/validate step and the dedup step separately
    run(f"CREATE TABLE ev_r AS {EV_ROWS}", "events_raw")
    log("DIAG events parsed and validated per row")
    con.execute(ev_dedup("ev_r"))
    con.execute("DROP TABLE ev_r")
else:
    run(ev_dedup(f"({EV_ROWS})"), "events_raw")
log("events deduped and validated")

# 5. sessions. person key = hash of the person's name. The default RANGE frame gives events
#    with equal ts the same running sum, so they always share a session.
con.execute(
    """
    CREATE TABLE ev_s AS
    SELECT ph, t, pid, pv,
           CAST(sum(CASE WHEN prev IS NULL OR t - prev > 1800000 THEN 1 ELSE 0 END)
                OVER (PARTITION BY ph ORDER BY t) AS INTEGER) AS sid
    FROM (
      SELECT ph, t, pid, pv, lag(t) OVER (PARTITION BY ph ORDER BY t) AS prev
      FROM (SELECT hash(coalesce(m.person, v.uid)) AS ph, v.t AS t, v.pid AS pid, v.pv AS pv
            FROM ev_v v LEFT JOIN lk_map m ON v.uid = m.id)
    )
    """
)
# names for the person hashes: every person is either a linked person or an unlinked ID
con.execute(
    """
    CREATE TABLE names AS
    SELECT hash(name) AS h, any_value(name) AS name
    FROM (SELECT person AS name FROM lk_map UNION ALL SELECT DISTINCT uid FROM ev_v)
    GROUP BY 1
    """
)
con.execute("DROP TABLE ev_v")
# 6. one row per session: start, events, revenue, UTC day of the start
con.execute(
    """
    CREATE TABLE sess AS
    SELECT ph, sid, min(t) AS st, count(*)::BIGINT AS n, sum(pv)::BIGINT AS rev,
           (DATE '1970-01-01' + CAST(min(t) // 86400000 AS INTEGER)) AS day
    FROM ev_s GROUP BY ph, sid
    """
)
log("sessions built")

# 7. top category per (person, day): purchases summed over the day's sessions
con.execute(
    """
    CREATE TABLE top_cat AS
    SELECT ph, day, cat FROM (
      SELECT s.ph AS ph, s.day AS day, coalesce(p.category, 'unknown') AS cat, sum(e.pv) AS v
      FROM (SELECT ph, sid, pid, pv FROM ev_s WHERE pv > 0) e
      JOIN sess s ON e.ph = s.ph AND e.sid = s.sid
      LEFT JOIN prod p ON e.pid = p.product_id
      GROUP BY s.ph, s.day, coalesce(p.category, 'unknown')
    )
    QUALIFY row_number() OVER (PARTITION BY ph, day ORDER BY v DESC, cat ASC) = 1
    """
)
con.execute("DROP TABLE ev_s")

# ---------------------------------------------------------------- clicks

src("clicks_raw", os.path.join(DATA, "clicks", "*.csv"), 4)
run(
    f"""
    CREATE TABLE clk_v AS
    SELECT hash(coalesce(m.person, c.uid)) AS ph, c.t AS t, c.campaign_id AS campaign_id, c.click_id AS click_id
    FROM (
      SELECT click_id, uid, campaign_id, ts_ms(ts) AS t FROM (
        SELECT f1 AS click_id, any_value(f2) AS uid, any_value(f3) AS campaign_id, any_value(f4) AS ts
        FROM clicks_raw
        WHERE f1 IS NOT NULL
        GROUP BY f1
      )
    ) c
    JOIN camp ON c.campaign_id = camp.campaign_id
    LEFT JOIN lk_map m ON c.uid = m.id
    WHERE c.uid IS NOT NULL AND c.t >= {WS - 7 * DAY} AND c.t < {WE}
    """,
    "clicks_raw",
)
# credit: per (person, ts) the click with the smallest click_id (byte-wise) wins ties
con.execute(
    """
    CREATE TABLE cu AS
    SELECT ph, t AS ts, arg_min(campaign_id, click_id) AS campaign_id
    FROM clk_v GROUP BY ph, t
    """
)
con.execute("CREATE TABLE clk AS SELECT DISTINCT ph, campaign_id, t AS ts FROM clk_v")
con.execute("DROP TABLE clk_v")
log("clicks ready")

# credited campaign: latest click at or before the session start, at most 7 days earlier
con.execute(
    f"""
    CREATE TABLE cred AS
    SELECT CASE WHEN c.ts IS NOT NULL AND s.st - c.ts <= {7 * DAY} THEN c.campaign_id ELSE 'organic' END AS campaign_id,
           s.day AS day,
           count(*)::BIGINT AS sessions,
           (count(*) FILTER (WHERE s.rev > 0))::BIGINT AS purchasing_sessions,
           sum(s.rev)::BIGINT AS revenue_cents
    FROM sess s ASOF LEFT JOIN cu c ON s.ph = c.ph AND s.st >= c.ts
    GROUP BY 1, 2
    """
)
# assists: every campaign with a valid click in [start - 7 days, start], once per session
con.execute(
    f"""
    CREATE TABLE asst AS
    SELECT campaign_id, day, count(*)::BIGINT AS assisted_sessions FROM (
      SELECT s.ph, s.sid, s.day, c.campaign_id
      FROM sess s JOIN clk c ON s.ph = c.ph AND c.ts <= s.st AND c.ts >= s.st - {7 * DAY}
      GROUP BY s.ph, s.sid, s.day, c.campaign_id
    ) GROUP BY campaign_id, day
    """
)
log("attribution done")

# ---------------------------------------------------------------- outputs

os.makedirs(os.path.join(OUT, "user_daily"), exist_ok=True)
os.makedirs(os.path.join(OUT, "campaign_daily"), exist_ok=True)
con.execute(
    f"""
    COPY (
      SELECT nm.name AS user_id, u.day AS day, u.sessions AS sessions, u.events AS events,
             u.revenue_cents AS revenue_cents, t.cat AS top_category
      FROM (SELECT ph, day, count(*)::BIGINT AS sessions, sum(n)::BIGINT AS events, sum(rev)::BIGINT AS revenue_cents
            FROM sess GROUP BY ph, day) u
      JOIN names nm ON u.ph = nm.h
      LEFT JOIN top_cat t ON u.ph = t.ph AND u.day = t.day
    ) TO '{os.path.join(OUT, "user_daily", "part-0.parquet")}' (FORMAT parquet)
    """
)
con.execute(
    f"""
    COPY (
      SELECT coalesce(a.campaign_id, b.campaign_id) AS campaign_id,
             CASE WHEN coalesce(a.campaign_id, b.campaign_id) = 'organic' THEN 'organic' ELSE camp.channel END AS channel,
             coalesce(a.day, b.day) AS day,
             coalesce(a.sessions, 0)::BIGINT AS sessions,
             coalesce(a.purchasing_sessions, 0)::BIGINT AS purchasing_sessions,
             coalesce(a.revenue_cents, 0)::BIGINT AS revenue_cents,
             coalesce(b.assisted_sessions, 0)::BIGINT AS assisted_sessions
      FROM cred a
      FULL OUTER JOIN asst b ON a.campaign_id = b.campaign_id AND a.day = b.day
      LEFT JOIN camp ON coalesce(a.campaign_id, b.campaign_id) = camp.campaign_id
    ) TO '{os.path.join(OUT, "campaign_daily", "part-0.parquet")}' (FORMAT parquet)
    """
)
log("done")
