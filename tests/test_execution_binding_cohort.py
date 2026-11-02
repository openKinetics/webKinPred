import csv

import pytest
from tools.executor.kinform import split_binding_outputs
from tools.executor.validate_artifacts import metadata


def test_binding_cohort_preserves_original_partitions_and_validates_each_context(tmp_path):
    mapping = {"a": "MA", "b": "MC", "c": "MD", "d": "ME"}
    combined = tmp_path / "combined.tsv"
    with combined.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["PDB", "Pred_BS_Scores"])
        writer.writerows((sid, "0.2,0.8") for sid in mapping)
    outputs = [
        {"id": str(i), "path": f"parts/{i}.tsv", "identity": {"context": {"sequences": ids}}}
        for i, ids in enumerate([["a", "b"], ["c", "d"]])
    ]
    payload = {"seq_id_to_seq": mapping, "outputs": outputs, "step": "kinform_pseq2sites"}
    split_binding_outputs(combined, payload, tmp_path)
    assert not combined.exists()
    for output in outputs:
        assert metadata(tmp_path / output["path"], {"payload": payload}, output) == {
            "rows": 2,
            "dtype": "binding_site_scores",
            "finite": True,
        }
        with (tmp_path / output["path"]).open(newline="") as stream:
            assert [r["PDB"] for r in csv.DictReader(stream, delimiter="\t")] == output["identity"][
                "context"
            ]["sequences"]
    with pytest.raises(ValueError, match="Missing binding"):
        metadata(tmp_path / outputs[0]["path"], {"payload": payload})


def test_binding_cohort_rejects_missing_rows_without_publishing_partitions(tmp_path):
    combined = tmp_path / "combined.tsv"
    combined.write_text("PDB\tPred_BS_Scores\na\t0.2,0.8\n")
    payload = {
        "seq_id_to_seq": {"a": "MA", "b": "MC"},
        "outputs": [{"path": "part.tsv", "identity": {"context": {"sequences": ["a", "b"]}}}],
    }
    with pytest.raises(ValueError, match="coverage"):
        split_binding_outputs(combined, payload, tmp_path)
    assert not (tmp_path / "part.tsv").exists()
