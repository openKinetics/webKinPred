import argparse
from utils.db import open_db, get_or_create_id, DEFAULT_DB
from utils.cli import cmd_get_or_create, cmd_batch_get_or_create
from utils.stats import cmd_stats

def main():
    p = argparse.ArgumentParser(description="Sequence ↔ ID resolver backed by SQLite (WAL).")
    p.add_argument("--db", default=DEFAULT_DB, help=f"Path to SQLite DB (default: {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("get-or-create", help="Resolve one sequence; increments uses_count.")
    p1.add_argument("--seq", required=True, help="Amino-acid sequence")
    p1.set_defaults(func=cmd_get_or_create)

    p2 = sub.add_parser("batch-get-or-create", help="Resolve many sequences in order; increments uses_count per occurrence.")
    g = p2.add_mutually_exclusive_group(required=True)
    g.add_argument("--stdin", action="store_true", help="Read sequences, one per line, from STDIN")
    g.add_argument("--csv", help="Read sequences from a CSV file")
    p2.add_argument("--col", help="CSV column name containing sequences (required when using --csv)")
    p2.set_defaults(func=cmd_batch_get_or_create)

    p3 = sub.add_parser("stats", help="Show database statistics (totals, lengths, recency, top lists).")
    p3.add_argument("--days", default="1,10,30",
                    help="Comma-separated day windows for recency stats (default: 1,10,30)")
    p3.add_argument("--top", default="10",
                    help="How many rows to show in top lists (default: 10)")
    p3.add_argument("--json", action="store_true",
                    help="Output JSON instead of text")
    p3.set_defaults(func=cmd_stats)

    args = p.parse_args()
    if getattr(args, "csv", None) and not getattr(args, "col", None):
        p.error("--col is required when using --csv")
    args.func(args)

if __name__ == "__main__":
    main()
