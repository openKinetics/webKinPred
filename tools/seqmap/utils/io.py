import csv, sys

def iter_sequences_from_stdin():
    for line in sys.stdin:
        s = line.rstrip("\n\r")
        if s != "":
            yield s

def iter_sequences_from_csv(path, col):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if col not in reader.fieldnames:
            raise SystemExit(f"Column '{col}' not found in CSV header: {reader.fieldnames}")
        for row in reader:
            seq = (row[col] or "").strip()
            if seq != "":
                yield seq