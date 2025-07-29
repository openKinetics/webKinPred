import hashlib, os, sqlite3

DEFAULT_DB = os.environ.get("SEQMAP_DB", "/home/saleh/webKinPred/media/sequence_info/seqmap.sqlite3")

def sha256_hex(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()

def base12(sha: str) -> str:
    return sha[:12]

def open_db(db_path: str) -> sqlite3.Connection:
    # autocommit; keep connections short-lived; no explicit transactions
    # NOTE: default busy_timeout (~5s) is fine; to fail-fast set PRAGMA busy_timeout=0
    con = sqlite3.connect(db_path, timeout=5, isolation_level=None, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    # con.execute("PRAGMA busy_timeout=0;")  # uncomment for truly fail-fast behaviour
    return con

def get_or_create_id(con: sqlite3.Connection, sequence: str) -> str:
    """
    Very short critical section per sequence:
      1) Try UPDATE by sha256 (existing row): increments uses_count, updates last_seen_at, then SELECT id.
      2) If no row was updated, INSERT a new row with a free id (12-hex base, suffixing as needed).
    """
    sha = sha256_hex(sequence)
    # Step 1: try to update existing row (single-statement write)
    cur = con.execute(
        "UPDATE sequences "
        "SET last_seen_at=CURRENT_TIMESTAMP, uses_count=uses_count+1 "
        "WHERE sha256=?",
        (sha,),
    )
    if cur.rowcount:  # found existing
        row = con.execute("SELECT id FROM sequences WHERE sha256=?", (sha,)).fetchone()
        return row[0]

    # Step 2: need to insert; choose an available id (base 12-hex, suffixing)
    base = base12(sha)
    suffix = 0
    while True:
        sid = base if suffix == 0 else f"{base}_{suffix}"
        try:
            con.execute(
                "INSERT INTO sequences(id, seq, sha256, len, created_at, last_seen_at, uses_count) "
                "VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)",
                (sid, sequence, sha, len(sequence)),
            )
            return sid
        except sqlite3.IntegrityError as e:
            msg = str(e)
            # If id is taken by some other row, try next suffix.
            if "UNIQUE constraint failed: sequences.id" in msg:
                suffix += 1
                continue
            # If another process inserted this same sequence concurrently, fall back to SELECT id.
            if "UNIQUE constraint failed: sequences.seq" in msg or "sequences.sha256" in msg:
                row = con.execute("SELECT id FROM sequences WHERE sha256=?", (sha,)).fetchone()
                if row:
                    # bump usage now that the row exists (single short UPDATE)
                    con.execute(
                        "UPDATE sequences SET last_seen_at=CURRENT_TIMESTAMP, uses_count=uses_count+1 WHERE id=?",
                        (row[0],),
                    )
                    return row[0]
            raise