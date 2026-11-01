import csv

import pytest
from tools.executor.binding_table import read_binding_table


def test_binding_table_preserves_exponent_like_ids_and_serialized_scores(tmp_path):
    path = tmp_path / "binding.tsv"
    rows = [
        ["0e6681122986", ",".join(["0.000445"] * 1024)],
        ["000000000001", "0.0,0.5,1.0"],
        ["deadbeef1234", "1.000000"],
    ]
    with path.open("w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["PDB", "Pred_BS_Scores"])
        writer.writerows(rows)
    frame = read_binding_table(path)
    assert frame.values.tolist() == rows
    assert [float(x) for x in frame.iloc[1]["Pred_BS_Scores"].split(",")] == [0, 0.5, 1]


@pytest.mark.parametrize(
    "text",
    ["wrong\theader\na\tb\n", "PDB\tPred_BS_Scores\na\t1\na\t2\n", "PDB\tPred_BS_Scores\n\t1\n"],
)
def test_binding_table_rejects_ambiguous_identities(tmp_path, text):
    path = tmp_path / "binding.tsv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_binding_table(path)
