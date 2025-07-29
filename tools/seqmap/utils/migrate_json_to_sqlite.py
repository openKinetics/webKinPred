#!/usr/bin/env python3
"""
Migrate seq_id_to_seq.json -> SQLite (WAL) sequences table.

Table (must exist or will be created if missing):
  sequences(
    id TEXT PRIMARY KEY,
    seq TEXT UNIQUE NOT NULL,
    sha256 TEXT UNIQUE NOT NULL,
    len INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    last_seen_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    uses_count INTEGER NOT NULL DEFAULT 0
  )

Behaviour:
- Preserves existing rows, inserts new ones.
- If an ID exists with a different sequence, logs a conflict and keeps the DB value.
- If a sequence exists but with a different ID, logs a mismatch and keeps the DB value.
- New rows get uses_count=0, created_at/last_seen_at=CURRENT_TIMESTAMP.
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from typing import Dict, Tuple

DEFAULT_DB = "/home/saleh/webKinPred/media/sequence_info/seqmap.sqlite3"

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS sequences (
  id           TEXT PRIMARY KEY,
  seq          TEXT NOT NULL UNIQUE,
  sha256       TEXT NOT NULL UNIQUE,
  len          INTEGER NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  last_seen_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  uses_count   INTEGER NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sequences_seq    ON sequences(seq);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sequences_sha256 ON sequences(sha256);
CREATE INDEX IF NOT EXISTS idx_sequences_len           ON sequences(len);
"""

def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA_SQL)
    # If the table pre-existed without uses_count, try to add it.
    try:
        con.execute("ALTER TABLE sequences ADD COLUMN uses_count INTEGER NOT NULL DEFAULT 0;")
    except sqlite3.OperationalError:
        # Column already exists; ignore.
        pass

def compute_sha_and_len(seq: str) -> Tuple[str, int]:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest(), len(seq)

def load_json(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("JSON must be an object mapping {id: sequence}.")
    return data

def migrate(json_path: str, db_path: str, dry_run: bool, batch_size: int) -> None:
    if not os.path.exists(json_path):
        raise SystemExit(f"JSON not found: {json_path}")

    # Optional: create a backup of the DB
    # (Handled by --backup in main)

    # Open DB (autocommit off; we control transactions)
    con = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    ensure_schema(con)

    mapping = load_json(json_path)
    total = len(mapping)
    print(f"Loaded {total:,} pairs from JSON")

    insert_sql = (
        "INSERT INTO sequences(id, seq, sha256, len, created_at, last_seen_at, uses_count) "
        "VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)"
    )

    # Counters
    inserted = 0
    id_exists_same = 0
    id_exists_diff = 0
    seq_exists_other_id = 0
    malformed = 0

    t0 = time.time()
    processed = 0

    # Begin outer transaction; we’ll commit periodically if batch_size > 0
    if not dry_run:
        con.execute("BEGIN;")
    try:
        for sid, seq in mapping.items():
            processed += 1
            if not isinstance(sid, str) or not isinstance(seq, str) or not seq:
                malformed += 1
                continue

            sha, slen = compute_sha_and_len(seq)

            # Check by id first
            row = con.execute("SELECT seq FROM sequences WHERE id=?", (sid,)).fetchone()
            if row:
                if row[0] == seq:
                    id_exists_same += 1
                else:
                    # Conflict: same id, different sequence
                    id_exists_diff += 1
                # Do not modify existing rows on migration
            else:
                # See if the sequence already exists (maybe under a different id)
                row2 = con.execute("SELECT id FROM sequences WHERE sha256=? OR seq=?", (sha, seq)).fetchone()
                if row2:
                    seq_exists_other_id += 1
                    # Keep existing mapping; do not insert a duplicate
                else:
                    if dry_run:
                        inserted += 1
                    else:
                        try:
                            con.execute(insert_sql, (sid, seq, sha, slen))
                            inserted += 1
                        except sqlite3.IntegrityError as e:
                            # Race or unexpected constraint: re-check presence, else log and continue
                            row3 = con.execute("SELECT id FROM sequences WHERE sha256=? OR seq=? OR id=?", (sha, seq, sid)).fetchone()
                            if row3:
                                # Treated as existing (concurrent insert)
                                id_exists_same += 1
                            else:
                                print(f"[WARN] IntegrityError for id={sid}: {e}", file=sys.stderr)

            # Periodic progress + batch commit
            if processed % 100 == 0:
                rate = processed / max(1e-9, (time.time() - t0))
                print(f"Processed {processed:,}/{total:,} ({rate:,.0f}/s)")
                if not dry_run and batch_size > 0:
                    con.execute("COMMIT;")
                    con.execute("BEGIN;")

        # Final commit
        if not dry_run:
            con.execute("COMMIT;")
    except Exception:
        if not dry_run:
            con.execute("ROLLBACK;")
        raise
    finally:
        con.close()

    dt = time.time() - t0
    print("\nMigration summary")
    print("-----------------")
    print(f"Inserted new rows           : {inserted:,}")
    print(f"Existing id, same sequence  : {id_exists_same:,}")
    print(f"Existing id, different seq  : {id_exists_diff:,}  (kept DB value)")
    print(f"Seq existed under other id  : {seq_exists_other_id:,}  (kept DB value)")
    print(f"Malformed entries skipped   : {malformed:,}")
    print(f"Elapsed                     : {dt:.1f}s")

def main():
    ap = argparse.ArgumentParser(description="Migrate seq_id_to_seq.json into SQLite sequences table.")
    ap.add_argument("--json", required=True, help="Path to seq_id_to_seq.json")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"Path to SQLite DB (default: {DEFAULT_DB})")
    ap.add_argument("--dry-run", action="store_true", help="Do not write; just report actions")
    ap.add_argument("--batch-size", type=int, default=50000, help="Rows per commit (0=single commit at end)")
    ap.add_argument("--backup", action="store_true", help="Create DB backup before writing")
    args = ap.parse_args()

    if args.backup and os.path.exists(args.db) and not args.dry_run:
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = f"{args.db}.bak.{ts}"
        print(f"Creating backup: {bak}")
        shutil.copy2(args.db, bak)

    migrate(args.json, args.db, dry_run=args.dry_run, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
