import sys
from .db import open_db, get_or_create_id
from .io import iter_sequences_from_stdin, iter_sequences_from_csv
def cmd_get_or_create(args):
    con = open_db(args.db)
    try:
        sid = get_or_create_id(con, args.seq)
        print(sid)
    finally:
        con.close()
        
def cmd_batch_get_or_create(args):
    con = open_db(args.db)
    try:
        if args.stdin:
            source = iter_sequences_from_stdin()
        else:
            source = iter_sequences_from_csv(args.csv, args.col)

        # No explicit BEGIN/COMMIT; each call is one statement -> minimal lock duration
        out = sys.stdout
        for seq in source:
            sid = get_or_create_id(con, seq)
            out.write(sid + "\n")
        out.flush()
    finally:
        con.close()