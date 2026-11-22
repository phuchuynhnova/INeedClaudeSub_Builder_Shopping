#!/usr/bin/env python3
"""
Shopping Sessions: DuckDB pipeline (correctness-first version).

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)

Design notes
- CSV files are read as one-column text lines (no quote handling by DuckDB), then each
  line is validated with ONE regex that encodes the dialect of rules.md and split on ','.
  This avoids DuckDB's own CSV repair / null / quote behaviour.
- Linked IDs: connected components with scipy, person name = byte-wise smallest ID.
- Sessions: window functions per person. Credit: ASOF join. Assists: range join.
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

# ---------------------------------------------------------------- helpers

FIELD = r'(?:"[^",]*"|[^",]*)'


def row_regex(n):
    """A whole line is well formed iff it has exactly n fields, each either unquoted without
    a quote, or wrapped in quotes with no quote inside. No field holds a comma."""
    return FIELD + "(?:," + FIELD + "){" + str(n - 1) + "}"


VIEWS = {}


def lines(view, pattern, slow=False):
    """Expose the files matching `pattern` as one VARCHAR column `line` (header skipped).
    Only \\n ends a line; DuckDB's own quote/escape handling is switched off.
    Fast path: DuckDB's CSV reader with one column. Slow path (used only if the fast
    reader rejects the files, e.g. because of a \\r): read whole files and split on \\n."""
    VIEWS[view] = pattern
    if not slow:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT line FROM read_csv('{pattern}', header = true, columns = {{'line': 'VARCHAR'}},
                delim = '\x01', quote = '', escape = '', new_line = '\\n',
                auto_detect = false, parallel = true)
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT unnest(string_split(content, chr(10))[2:]) AS line FROM read_text('{pattern}')
            """
        )


def run(sql, *views):
    """Execute `sql`; if reading one of `views` fails in the fast path, switch them to the
    slow path and run it again."""
    try:
        return con.execute(sql)
    except duckdb.InvalidInputException as error:
        log(f"fast CSV path failed ({str(error).splitlines()[0]}); using the slow line reader")
        for v in views:
            lines(v, VIEWS[v], slow=True)
        return con.execute(sql)


def parsed(view, n):
    """Columns f1..fn of the well-formed lines of `view`; quotes removed, empty -> NULL."""
    cols = ", ".join(f"nullif(trim(f[{i}], '\"'), '') AS f{i}" for i in range(1, n + 1))
    return f"""
        SELECT {cols}
        FROM (SELECT string_split(line, ',') AS f FROM {view}
              WHERE regexp_full_match(line, '{row_regex(n)}'))
    """


# ts -> epoch milliseconds, NULL when bad. F3 values too big for BIGINT are NULL (they are
# outside every window anyway). Calendar validity comes from TRY_CAST to DATE.
con.execute(
    r"""
CREATE OR REPLACE MACRO ts_ms(x) AS
  CASE
    WHEN x IS NULL THEN NULL
    WHEN regexp_full_match(x, '[0-9]+') THEN TRY_CAST(ltrim(x, '0') AS BIGINT)
    WHEN regexp_full_match(x, '[0-9]{4}-[0-9]{2}-[0-9]{2}T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])')
         AND TRY_CAST(substr(x, 1, 10) AS DATE) IS NOT NULL THEN
      (TRY_CAST(substr(x, 1, 10) AS DATE) - DATE '1970-01-01')::BIGINT * 86400000
      + CAST(substr(x, 12, 2) AS BIGINT) * 3600000
      + CAST(substr(x, 15, 2) AS BIGINT) * 60000
      + CAST(substr(x, 18, 2) AS BIGINT) * 1000
      - CASE WHEN substr(x, 20, 1) = 'Z' THEN 0
             ELSE (CAST(substr(x, 21, 2) AS BIGINT) * 3600000 + CAST(substr(x, 24, 2) AS BIGINT) * 60000)
                  * (CASE WHEN substr(x, 20, 1) = '+' THEN 1 ELSE -1 END)
        END
    ELSE NULL
  END
"""
)

# ---------------------------------------------------------------- small tables

lines("products_raw", os.path.join(DATA, "products.csv"))
lines("campaigns_raw", os.path.join(DATA, "campaigns.csv"))
run(f"CREATE TABLE prod AS SELECT f1 AS product_id, f2 AS category FROM ({parsed('products_raw', 2)}) WHERE f1 IS NOT NULL", "products_raw")
run(f"CREATE TABLE camp AS SELECT f1 AS campaign_id, f2 AS channel FROM ({parsed('campaigns_raw', 2)}) WHERE f1 IS NOT NULL", "campaigns_raw")

# ---------------------------------------------------------------- linked IDs

lines("links_raw", os.path.join(DATA, "links", "*.csv"))
run(
    f"""
    CREATE TABLE lk AS
    SELECT DISTINCT least(f1, f2) AS a, greatest(f1, f2) AS b
    FROM ({parsed('links_raw', 2)})
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

lines("events_raw", os.path.join(DATA, "events", "*.csv"))
# 2. dedup on event_id, keeping the copy with the smallest ingest_ts. ingest_ts is always
#    F1 (fixed width), so comparing the strings compares the instants.
run(
    f"""
    CREATE TABLE ev_k AS
    SELECT f1 AS event_id,
           arg_min(struct_pack(uid := f2, ts := f3, ing := f4, et := f5, pid := f6, pr := f7), f4) AS r
    FROM ({parsed('events_raw', 7)})
    WHERE f1 IS NOT NULL
    GROUP BY f1
    """,
    "events_raw",
)
log("events deduped")
# 3. validate the kept copy, 4. drop late events
con.execute(
    f"""
    CREATE TABLE ev_v AS
    SELECT uid, t, pid, pv FROM (
      SELECT r.uid AS uid, ts_ms(r.ts) AS t, ts_ms(r.ing) AS ing, r.et AS et, r.pid AS pid,
             CASE WHEN r.et = 'purchase'
                  THEN CASE WHEN r.pr IS NOT NULL AND regexp_full_match(r.pr, '[0-9]+')
                            THEN TRY_CAST(ltrim(r.pr, '0') AS BIGINT) END
                  ELSE 0 END AS pv
      FROM ev_k
    )
    WHERE uid IS NOT NULL AND t >= {WS} AND t < {WE}
      AND et IN ('view', 'cart', 'purchase') AND pid IS NOT NULL
      AND (et <> 'purchase' OR pv BETWEEN 1 AND 10000000)
      AND ing - t <= {DAY}
    """
)
con.execute("DROP TABLE ev_k")
log("events validated")

# link every event to its person
con.execute(
    """
    CREATE TABLE ev_p AS
    SELECT coalesce(m.person, v.uid) AS person, v.t AS t, v.pid AS pid, v.pv AS pv
    FROM ev_v v LEFT JOIN lk_map m ON v.uid = m.id
    """
)
con.execute("DROP TABLE ev_v")

# 5. sessions. The default RANGE frame gives events with equal ts the same running sum, so
#    they always share a session whatever order they are read in.
con.execute(
    """
    CREATE TABLE ev_s AS
    SELECT person, t, pid, pv,
           sum(CASE WHEN prev IS NULL OR t - prev > 1800000 THEN 1 ELSE 0 END)
               OVER (PARTITION BY person ORDER BY t) AS sid
    FROM (SELECT person, t, pid, pv, lag(t) OVER (PARTITION BY person ORDER BY t) AS prev FROM ev_p)
    """
)
con.execute("DROP TABLE ev_p")
# 6. one row per session: start, events, revenue, UTC day of the start
con.execute(
    """
    CREATE TABLE sess AS
    SELECT person, sid, min(t) AS st, count(*)::BIGINT AS n, sum(pv)::BIGINT AS rev,
           (DATE '1970-01-01' + CAST(min(t) // 86400000 AS INTEGER)) AS day
    FROM ev_s GROUP BY person, sid
    """
)
log("sessions built")

# 7. top category per (person, day): purchases summed over the day's sessions
con.execute(
    """
    CREATE TABLE top_cat AS
    SELECT person, day, cat FROM (
      SELECT s.person AS person, s.day AS day, coalesce(p.category, 'unknown') AS cat, sum(e.pv) AS v
      FROM (SELECT person, sid, pid, pv FROM ev_s WHERE pv > 0) e
      JOIN sess s ON e.person = s.person AND e.sid = s.sid
      LEFT JOIN prod p ON e.pid = p.product_id
      GROUP BY s.person, s.day, coalesce(p.category, 'unknown')
    )
    QUALIFY row_number() OVER (PARTITION BY person, day ORDER BY v DESC, cat ASC) = 1
    """
)
con.execute("DROP TABLE ev_s")

# ---------------------------------------------------------------- clicks

lines("clicks_raw", os.path.join(DATA, "clicks", "*.csv"))
run(
    f"""
    CREATE TABLE clk_v AS
    SELECT coalesce(m.person, c.uid) AS person, c.t AS t, c.campaign_id AS campaign_id, c.click_id AS click_id
    FROM (
      SELECT click_id, uid, campaign_id, ts_ms(ts) AS t FROM (
        SELECT f1 AS click_id, any_value(f2) AS uid, any_value(f3) AS campaign_id, any_value(f4) AS ts
        FROM ({parsed('clicks_raw', 4)})
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
    SELECT person, t AS ts, arg_min(campaign_id, click_id) AS campaign_id
    FROM clk_v GROUP BY person, t
    """
)
con.execute("CREATE TABLE clk AS SELECT DISTINCT person, campaign_id, t AS ts FROM clk_v")
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
    FROM sess s ASOF LEFT JOIN cu c ON s.person = c.person AND s.st >= c.ts
    GROUP BY 1, 2
    """
)
# assists: every campaign with a valid click in [start - 7 days, start], once per session
con.execute(
    f"""
    CREATE TABLE asst AS
    SELECT campaign_id, day, count(*)::BIGINT AS assisted_sessions FROM (
      SELECT s.person, s.sid, s.day, c.campaign_id
      FROM sess s JOIN clk c ON s.person = c.person AND c.ts <= s.st AND c.ts >= s.st - {7 * DAY}
      GROUP BY s.person, s.sid, s.day, c.campaign_id
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
      SELECT u.person AS user_id, u.day AS day, u.sessions AS sessions, u.events AS events,
             u.revenue_cents AS revenue_cents, t.cat AS top_category
      FROM (SELECT person, day, count(*)::BIGINT AS sessions, sum(n)::BIGINT AS events, sum(rev)::BIGINT AS revenue_cents
            FROM sess GROUP BY person, day) u
      LEFT JOIN top_cat t ON u.person = t.person AND u.day = t.day
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
