"""Bound parsing before pandas or expansion can allocate an entire large request."""

import csv
import io

import pandas as pd


def frames(source, *, max_rows=128, max_bytes=8 * 1024**2, max_units=4096):
    csv.field_size_limit(max_bytes)
    with source.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        batch, size, units = [], 0, 0
        sequence = header.index("Protein Sequence")
        substrate = header.index("Substrates") if "Substrates" in header else None
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Malformed input CSV row")
            row_size = sum(len(value.encode()) for value in row)
            expanded = (row[sequence].count(";") + 1) * (
                row[substrate].count(";") + 1 if substrate is not None else 1
            )
            if row_size > max_bytes or expanded > max_units:
                raise ValueError("Input requires a smaller child-unit execution plan")
            if batch and (
                len(batch) == max_rows
                or size + row_size > max_bytes
                or units + expanded > max_units
            ):
                yield _frame(header, batch)
                batch, size, units = [], 0, 0
            batch.append(row)
            size += row_size
            units += expanded
        if batch:
            yield _frame(header, batch)


def _frame(header, rows):
    buffer = io.StringIO()
    csv.writer(buffer).writerows([header, *rows])
    buffer.seek(0)
    return pd.read_csv(buffer)
