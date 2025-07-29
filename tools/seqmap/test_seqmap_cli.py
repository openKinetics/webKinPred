#!/usr/bin/env python3
import os
import sys
import csv
import time
import json
import hashlib
import sqlite3
import tempfile
import unittest
import subprocess
import threading
from pathlib import Path

# Path to your CLI
CLI = "/home/saleh/webKinPred/tools/seqmap/main.py"

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

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def base12(h: str) -> str:
    return h[:12]

def run_cli(args, *, stdin_str=None, db_path=None, check=True):
    """Run the CLI and return (rc, stdout, stderr)."""
    cmd = [sys.executable, CLI] + args
    if db_path:
        cmd = [sys.executable, CLI, "--db", db_path] + args
    proc = subprocess.run(
        cmd,
        input=stdin_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"CLI failed rc={proc.returncode}\ncmd={cmd}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc.returncode, proc.stdout, proc.stderr

class SeqMapCLITests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="seqmap_cli_test_")
        self.db = os.path.join(self.tmpdir.name, "seqmap.sqlite3")
        con = sqlite3.connect(self.db, timeout=60, isolation_level=None)
        con.executescript(SCHEMA_SQL)
        con.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def fetch_row(self, sid=None, seq=None):
        con = sqlite3.connect(self.db, timeout=60, isolation_level=None)
        cur = con.cursor()
        if sid:
            row = cur.execute("SELECT id, seq, sha256, len, uses_count FROM sequences WHERE id=?", (sid,)).fetchone()
        elif seq:
            row = cur.execute("SELECT id, seq, sha256, len, uses_count FROM sequences WHERE seq=?", (seq,)).fetchone()
        else:
            row = None
        con.close()
        return row

    def test_single_insert_and_retrieve(self):
        seq = "MKIVLVLYDAGKHA"
        rc, out, _ = run_cli(["get-or-create", "--seq", seq], db_path=self.db)
        sid = out.strip()
        self.assertTrue(sid)
        row = self.fetch_row(sid=sid)
        self.assertIsNotNone(row)
        _id, _seq, _sha, _len, _uses = row
        self.assertEqual(_seq, seq)
        self.assertEqual(_len, len(seq))
        self.assertEqual(_uses, 1)
        self.assertTrue(sid.startswith(base12(_sha)))

    def test_repeat_increments_uses_count(self):
        seq = "MVDGNYSVASNVMV"
        # First call
        rc, out1, _ = run_cli(["get-or-create", "--seq", seq], db_path=self.db)
        sid = out1.strip()
        # Second call
        rc, out2, _ = run_cli(["get-or-create", "--seq", seq], db_path=self.db)
        self.assertEqual(sid, out2.strip())
        row = self.fetch_row(sid=sid)
        self.assertEqual(row[-1], 2)  # uses_count

    def test_batch_stdin_order_and_counts(self):
        seqs = ["AAA", "BBB", "AAA", "", "CCC", "BBB", "AAA"]
        stdin_payload = "\n".join(seqs) + "\n"
        rc, out, _ = run_cli(["batch-get-or-create", "--stdin"], stdin_str=stdin_payload, db_path=self.db)
        ids = [line for line in out.strip().splitlines()]
        # Empty lines are ignored -> expected 6 outputs
        self.assertEqual(len(ids), 6)
        # uses_count should be AAA:3 (positions 0,2,6), BBB:2, CCC:1
        a_row = self.fetch_row(seq="AAA")
        b_row = self.fetch_row(seq="BBB")
        c_row = self.fetch_row(seq="CCC")
        self.assertEqual(a_row[-1], 3)
        self.assertEqual(b_row[-1], 2)
        self.assertEqual(c_row[-1], 1)
        # Order preservation: identical seqs must resolve to the same id
        self.assertEqual(ids[1], ids[4])  # BBB id
        # Also assert AAA is consistent across its three occurrences
        self.assertEqual(ids[0], ids[2])  # AAA id
        self.assertEqual(ids[0], ids[5])  # AAA id

    def test_suffix_collision_on_id(self):
        # Choose a sequence and compute its base id
        seq = "QWERTYQWERTYQWERTY"
        sha = sha256_hex(seq)
        b12 = base12(sha)

        # Insert a blocker row with id=b12 but different sequence
        blocker_seq = "BLOCKERSEQXYZ"
        blocker_sha = sha256_hex(blocker_seq)
        con = sqlite3.connect(self.db, timeout=60, isolation_level=None)
        con.execute(
            "INSERT INTO sequences(id, seq, sha256, len, uses_count) VALUES(?, ?, ?, ?, 0)",
            (b12, blocker_seq, blocker_sha, len(blocker_seq)),
        )
        con.close()

        # Now calling CLI for 'seq' must suffix the id
        rc, out, _ = run_cli(["get-or-create", "--seq", seq], db_path=self.db)
        sid = out.strip()
        self.assertTrue(sid.startswith(b12))
        self.assertNotEqual(sid, b12)  # must be b12_1 (or higher)
        row = self.fetch_row(sid=sid)
        self.assertIsNotNone(row)
        self.assertEqual(row[1], seq)
        self.assertEqual(row[-1], 1)

    def test_concurrent_writes_same_sequence(self):
        seq = "CONCURRENTSEQ"
        N_PROCS = 5
        REPEATS = 10

        def worker():
            payload = ("\n".join([seq] * REPEATS)) + "\n"
            run_cli(["batch-get-or-create", "--stdin"], stdin_str=payload, db_path=self.db)

        threads = [threading.Thread(target=worker) for _ in range(N_PROCS)]
        for t in threads: t.start()
        for t in threads: t.join()

        row = self.fetch_row(seq=seq)
        self.assertIsNotNone(row)
        _id, _seq, _sha, _len, _uses = row
        self.assertEqual(_uses, N_PROCS * REPEATS)

        # Only one row for this sha/seq should exist
        con = sqlite3.connect(self.db, timeout=60, isolation_level=None)
        cnt = con.execute("SELECT COUNT(*) FROM sequences WHERE sha256=? OR seq=?", (sha256_hex(seq), seq)).fetchone()[0]
        con.close()
        self.assertEqual(cnt, 1)

    def test_csv_missing_column_error(self):
        # Create a temp CSV without the expected column
        csv_path = os.path.join(self.tmpdir.name, "input.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["NotTheColumn"])
            w.writerow(["AAA"])
        # Expect non-zero exit code
        rc, out, err = run_cli(
            ["batch-get-or-create", "--csv", csv_path, "--col", "Protein Sequence"],
            db_path=self.db, check=False
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("Column 'Protein Sequence' not found", (out + err))

    def test_very_long_sequence(self):
        seq = "A" * 5000
        rc, out, _ = run_cli(["get-or-create", "--seq", seq], db_path=self.db)
        sid = out.strip()
        row = self.fetch_row(sid=sid)
        self.assertEqual(row[3], 5000)  # len

if __name__ == "__main__":
    unittest.main(verbosity=2)
