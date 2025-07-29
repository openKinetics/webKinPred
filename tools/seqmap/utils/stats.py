import json
from datetime import datetime
from .db import open_db

def _fetchone(con, sql, params=()):
    return con.execute(sql, params).fetchone()[0]

def _percentile_len(con, q: float) -> int:
    """
    q in (0,1]; uses window functions to select the q-th percentile by length.
    Median is q=0.5. For even N, this picks the upper-middle; that's fine for a quick stat.
    """
    # Clamp q
    if q <= 0: q = 0.000001
    if q > 1: q = 1.0
    # Row number & count, then pick the ceil(q * cnt)-th row
    sql = """
    WITH ranked AS (
      SELECT len,
             ROW_NUMBER() OVER (ORDER BY len) AS rn,
             COUNT(*)     OVER ()             AS cnt
      FROM sequences
    )
    SELECT len
    FROM ranked
    WHERE rn = CAST(CEIL(? * cnt) AS INTEGER)
    LIMIT 1;
    """
    # SQLite before 3.46 may not have CEIL; emulate: CAST((? * cnt + 0.999999) AS INTEGER)
    sql = sql.replace("CEIL(? * cnt)", "(? * cnt + 0.999999)")
    row = con.execute(sql, (q,)).fetchone()
    return int(row[0]) if row and row[0] is not None else 0

def _median_len(con) -> float:
    # True median using the avg of the two middle values for even counts.
    sql = """
    WITH t AS (
      SELECT len,
             ROW_NUMBER() OVER (ORDER BY len) AS rn,
             COUNT(*)     OVER ()             AS cnt
      FROM sequences
    )
    SELECT AVG(len) AS median
    FROM t
    WHERE rn IN ( (cnt + 1) / 2, (cnt + 2) / 2 );
    """
    row = con.execute(sql).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0

def _recent_counts(con, column: str, days_list):
    out = {}
    for d in days_list:
        row = con.execute(
            f"SELECT COUNT(*) FROM sequences WHERE {column} >= datetime('now', ?)",
            (f"-{int(d)} days",)
        ).fetchone()
        out[int(d)] = int(row[0])
    return out

def cmd_stats(args):
    con = open_db(args.db)

    # Core totals
    total = _fetchone(con, "SELECT COUNT(*) FROM sequences")
    total_uses = _fetchone(con, "SELECT COALESCE(SUM(uses_count),0) FROM sequences")
    never_used = _fetchone(con, "SELECT COUNT(*) FROM sequences WHERE uses_count=0")
    avg_uses = (total_uses / total) if total else 0.0

    # Length stats
    row_short = con.execute("SELECT id, len FROM sequences ORDER BY len ASC, id ASC LIMIT 1").fetchone()
    row_long  = con.execute("SELECT id, len FROM sequences ORDER BY len DESC, id ASC LIMIT 1").fetchone()
    avg_len   = _fetchone(con, "SELECT COALESCE(AVG(len),0) FROM sequences")
    median    = _median_len(con)
    p90       = _percentile_len(con, 0.90)
    p99       = _percentile_len(con, 0.99)

    # Recent activity windows
    days = [int(x) for x in (args.days.split(",") if args.days else ["1","10","30"])]
    added_recent = _recent_counts(con, "created_at", days)
    used_recent  = _recent_counts(con, "last_seen_at", days)

    # Integrity: id base vs sha256 prefix
    id_mismatch = _fetchone(con, """
    WITH b AS (
      SELECT
        id,
        CASE WHEN instr(id,'_')>0 THEN substr(id,1,instr(id,'_')-1) ELSE id END AS id_base,
        substr(sha256,1,12) AS sha_base
      FROM sequences
    )
    SELECT COUNT(*) FROM b WHERE id_base<>sha_base;
    """)
    # Suffix & collision metrics
    suffixed_ids = _fetchone(con, "SELECT COUNT(*) FROM sequences WHERE instr(id,'_')>0;")
    row = con.execute("""
      WITH g AS (
        SELECT substr(sha256,1,12) AS sha_base, COUNT(*) AS n
        FROM sequences
        GROUP BY sha_base
      )
      SELECT SUM(CASE WHEN n>1 THEN 1 ELSE 0 END) AS bases_with_collisions,
             COALESCE(SUM(CASE WHEN n>1 THEN n-1 ELSE 0 END),0) AS extra_ids
      FROM g;
    """).fetchone()
    bases_with_collisions = int(row[0] or 0)
    extra_ids_due_to_collisions = int(row[1] or 0)

    # Top lists
    top_n = int(args.top)
    top_used = con.execute("""
      SELECT id, len, uses_count, last_seen_at
      FROM sequences
      ORDER BY uses_count DESC, last_seen_at DESC
      LIMIT ?
    """, (top_n,)).fetchall()

    top_recent = con.execute("""
      SELECT id, len, uses_count, last_seen_at
      FROM sequences
      ORDER BY last_seen_at DESC
      LIMIT ?
    """, (top_n,)).fetchall()

    con.close()

    report = {
        "totals": {
            "sequences": int(total),
            "total_uses": int(total_uses),
            "never_used": int(never_used),
            "avg_uses_per_sequence": round(avg_uses, 4),
        },
        "lengths": {
            "shortest": {"id": (row_short[0] if row_short else None), "len": (row_short[1] if row_short else None)},
            "longest" : {"id": (row_long[0]  if row_long  else None), "len": (row_long[1]  if row_long  else None)},
            "average_len": round(float(avg_len), 3) if avg_len is not None else 0.0,
            "median_len": round(float(median), 3),
            "p90_len": int(p90),
            "p99_len": int(p99),
        },
        "recent": {
            "added_in_last_days": added_recent,
            "used_in_last_days":  used_recent,
        },
        "integrity": {
            "id_sha12_mismatches": int(id_mismatch),
            "suffixed_ids": int(suffixed_ids),
            "bases_with_collisions": bases_with_collisions,
            "extra_ids_due_to_collisions": extra_ids_due_to_collisions,
        },
        "top": {
            "by_uses": [
                {"id": r[0], "len": int(r[1]), "uses": int(r[2]), "last_seen_at": r[3]} for r in top_used
            ],
            "recently_used": [
                {"id": r[0], "len": int(r[1]), "uses": int(r[2]), "last_seen_at": r[3]} for r in top_recent
            ],
        },
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    # Pretty text output
    print("== seqmap stats ==")
    print(f"Generated at (UTC): {report['generated_at']}")
    print("\nTotals")
    print("------")
    print(f"Sequences           : {report['totals']['sequences']:,}")
    print(f"Total uses          : {report['totals']['total_uses']:,}")
    print(f"Never used          : {report['totals']['never_used']:,}")
    print(f"Avg uses / sequence : {report['totals']['avg_uses_per_sequence']:.4f}")

    print("\nLengths")
    print("-------")
    print(f"Shortest            : {report['lengths']['shortest']}")
    print(f"Longest             : {report['lengths']['longest']}")
    print(f"Average length      : {report['lengths']['average_len']}")
    print(f"Median length       : {report['lengths']['median_len']}")
    print(f"P90 length          : {report['lengths']['p90_len']}")
    print(f"P99 length          : {report['lengths']['p99_len']}")

    print("\nRecent activity (counts)")
    print("------------------------")
    print("Added in last days  :", ", ".join(f"{d}d={c}" for d, c in report["recent"]["added_in_last_days"].items()))
    print("Used  in last days  :", ", ".join(f"{d}d={c}" for d, c in report["recent"]["used_in_last_days"].items()))

    print("\nIntegrity")
    print("---------")
    print(f"id_base vs sha256[:12] mismatches : {report['integrity']['id_sha12_mismatches']}  (0 is correct)")
    print(f"Suffixed IDs                      : {report['integrity']['suffixed_ids']}")
    print(f"Hash-base collisions (groups)     : {report['integrity']['bases_with_collisions']}")
    print(f"Extra IDs due to collisions       : {report['integrity']['extra_ids_due_to_collisions']}")

    def _print_table(title, rows):
        print(f"\n{title}")
        print("-" * len(title))
        if not rows:
            print("(none)")
            return
        print(f"{'id':<14} {'len':>6} {'uses':>6}  last_seen_at")
        for r in rows:
            print(f"{r['id']:<14} {r['len']:>6} {r['uses']:>6}  {r['last_seen_at']}")

    _print_table(f"Top {top_n} by uses", report["top"]["by_uses"])
    _print_table(f"Top {top_n} most recently used", report["top"]["recently_used"])

