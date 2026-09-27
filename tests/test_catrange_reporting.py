"""Generic output-frame assembly and completion counting without model loading."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.services.catrange_reporting import build_prediction_result_frame, completed_reaction_count
from api.services.result_service import serialize_result_csv


def _result(preds, sources=None, extra=None, output_col="Predicted kcat"):
    return {
        "preds": preds,
        "sources": sources or ["Prediction from CatRange"] * len(preds),
        "extra": extra or [""] * len(preds),
        "output_col": output_col,
    }


def test_production_frame_assembly_preserves_other_method_and_input_rows(tmp_path):
    inputs = pd.DataFrame({"Protein Sequence": ["AAA", "BBB"], "Substrate": ["O", "CC"]}, index=[9, 3])
    original = inputs.copy()
    catrange = _result(
        [math.sqrt(10), None],
        ["Prediction from CatRange", "too long"],
        ["Predicted range: 1 to 10 s^-1", "too long"],
    )
    other = _result([0.25, 2.5], ["Prediction from UniKP"] * 2, ["first", "second"], output_col="KM (mM)")
    frame = build_prediction_result_frame(inputs, ["kcat", "Km"], {"kcat": catrange, "Km": other})
    assert list(frame.columns) == [
        "Predicted kcat", "Source kcat", "Extra Info kcat",
        "KM (mM)", "Source Km", "Extra Info Km",
        "Protein Sequence", "Substrate",
    ]
    assert frame.index.tolist() == [9, 3]
    assert frame["Predicted kcat"].tolist()[0] == pytest.approx(catrange["preds"][0])
    assert pd.isna(frame["Predicted kcat"].tolist()[1])
    assert frame["Extra Info kcat"].tolist() == catrange["extra"]
    assert frame["KM (mM)"].tolist() == other["preds"]
    assert frame["Extra Info Km"].tolist() == other["extra"]
    pd.testing.assert_frame_equal(inputs, original)
    output = tmp_path / "mixed-output.csv"
    frame.to_csv(output, index=False)
    payload = serialize_result_csv(str(output))
    assert payload["data"][0]["Predicted kcat"] == pytest.approx(math.sqrt(10))
    assert payload["data"][0]["Extra Info kcat"] == "Predicted range: 1 to 10 s^-1"
    assert payload["data"][0]["KM (mM)"] == 0.25
    assert payload["data"][1]["Predicted kcat"] is None


def test_array_result_passes_through_unchanged(tmp_path):
    value = math.sqrt(1e-5 * 1e-4)
    items = [
        {"substrateIndex": 1, "prediction": value, "source": "Prediction from CatRange", "details": "Predicted range: 1e-05 to 0.0001 M"},
        {"substrateIndex": 2, "prediction": None, "source": "", "error": "bad substrate"},
    ]
    result = _result(
        [json.dumps([value, None])],
        ["Mixed per-substrate sources; see Extra Info"],
        [json.dumps(items)],
        output_col="Predicted Km",
    )
    frame = build_prediction_result_frame(pd.DataFrame({"Substrates": ["O;C"]}), ["Km"], {"Km": result})
    output = tmp_path / "array-output.csv"
    frame.to_csv(output, index=False)
    payload = serialize_result_csv(str(output))
    assert json.loads(payload["data"][0]["Predicted Km"]) == [value, None]
    assert json.loads(payload["data"][0]["Extra Info Km"]) == items


def test_completion_counts_measured_rows_and_excludes_failures():
    result = _result([math.sqrt(10), 2.0, None, "", float("nan")], ["Prediction from CatRange", "BRENDA", "failed", "failed", "failed"])
    assert completed_reaction_count(["kcat"], {"kcat": result}) == 2


def test_completion_requires_every_requested_target_by_row_position():
    kcat = _result([2.0, None, 4.0, 0.0])
    km = _result([0.2, 0.3, "", 0.4])
    # Index labels must not change positional result alignment.
    kcat["preds"] = pd.Series(kcat["preds"], index=[9, 3, 2, 0])
    assert completed_reaction_count(["kcat", "Km"], {"kcat": kcat, "Km": km}) == 2


def test_completion_rejects_misaligned_target_lengths():
    with pytest.raises(ValueError, match="same number of rows"):
        completed_reaction_count(["kcat", "Km"], {"kcat": _result([1]), "Km": _result([])})
