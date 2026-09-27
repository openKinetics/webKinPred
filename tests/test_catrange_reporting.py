"""Class reporting, provenance, and CSV/API serialization without model loading."""

import ast
import copy
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.services.catrange_reporting import BIN_EDGES, build_catrange_report_columns, build_prediction_result_frame, completed_reaction_count
from api.services.result_service import serialize_result_csv


def _result(preds, sources=None, extra=None):
    return {
        "preds": preds,
        "sources": sources or ["Prediction from CatRange"] * len(preds),
        "extra": extra or [""] * len(preds),
        "output_col": "Predicted range",
    }


@pytest.mark.parametrize("target", ["kcat", "Km"])
def test_every_class_is_reported_before_derived_values(target):
    edges = BIN_EDGES[target]
    values = [math.sqrt(a * b) if a else b / 2 for a, b in zip(edges, edges[1:])]
    result = _result(values)
    original = copy.deepcopy(result)
    report = build_catrange_report_columns(target, result)
    unit = "s^-1" if target == "kcat" else "M"
    assert next(iter(report)) == "Predicted range"
    assert report["Predicted range"] == [f"{a:g} to {b:g} {unit}" for a, b in zip(edges, edges[1:])]
    assert list(report.values())[3] == values
    assert list(report.values())[4] == [""] * len(values)
    assert result == original


def test_reporting_edges_match_model_without_importing_runtime():
    tree = ast.parse((REPO_ROOT / "models/CatRange/inference/catrange_inference.py").read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BIN_EDGES" for t in node.targets))
    model_edges = {ast.literal_eval(key): ast.literal_eval(value.args[0]) for key, value in zip(assignment.value.keys, assignment.value.values)}
    assert BIN_EDGES["kcat"] == tuple(model_edges["kcat"])
    assert BIN_EDGES["Km"] == tuple(model_edges["km"])


def test_measurement_equal_to_midpoint_is_never_labeled_as_prediction():
    value = math.sqrt(10)
    result = _result([value, value, 0, ""], ["Prediction from CatRange", "BRENDA", "Experimental", "Prediction could not be made"], ["range details", "measurement details", "", "too long"])
    report = build_catrange_report_columns("kcat", result)
    assert report["Predicted range"] == ["1 to 10 s^-1", "", "", ""]
    assert report["Derived kcat range midpoint (1/s)"] == [value, "", "", ""]
    assert report["Experimental kcat (1/s)"] == ["", value, 0, ""]
    assert report["Source kcat"] == result["sources"]
    assert report["Extra Info kcat"] == result["extra"]


@pytest.mark.parametrize("value", [1.234, 0.0, math.inf, True])
def test_arbitrary_model_value_is_rejected(value):
    with pytest.raises(ValueError):
        build_catrange_report_columns("kcat", _result([value]))


def test_same_source_arrays_preserve_slot_order_without_metadata():
    values = [math.sqrt(1e-5 * 1e-4), None, math.sqrt(1e-2 * 1e-1)]
    report = build_catrange_report_columns("Km", _result([json.dumps(values)], ["Prediction from CatRange (per substrate)"]))
    assert json.loads(report["Predicted range"][0]) == ["1e-05 to 0.0001 M", None, "0.01 to 0.1 M"]
    assert json.loads(report["Derived KM range midpoint (M)"][0]) == values
    assert json.loads(report["Experimental KM (source units)"][0]) == [None, None, None]


def test_mixed_substrate_array_keeps_measured_values_separate():
    value = math.sqrt(1e-5 * 1e-4)
    items = [
        {"substrateIndex": 1, "prediction": value, "source": "Prediction from CatRange", "error": None},
        {"substrateIndex": 2, "prediction": value, "source": "Sabio-RK", "error": None},
        {"substrateIndex": 3, "prediction": None, "source": "", "error": "bad substrate"},
    ]
    result = _result([json.dumps([value, value, None])], ["Mixed per-substrate sources; see Extra Info"], [json.dumps(items)])
    report = build_catrange_report_columns("Km", result)
    assert json.loads(report["Predicted range"][0]) == ["1e-05 to 0.0001 M", None, None]
    assert json.loads(report["Derived KM range midpoint (M)"][0]) == [value, None, None]
    assert json.loads(report["Experimental KM (source units)"][0]) == [None, value, None]
    assert report["Extra Info Km"] == result["extra"]


def test_nested_sequence_metadata_uses_selected_source_for_each_slot():
    value = math.sqrt(1e-5 * 1e-4)
    items = [
        {"sequenceIndex": 1, "substrates": [
            {"substrateIndex": 1, "prediction": value, "source": "BRENDA", "selected": False},
            {"substrateIndex": 2, "prediction": value, "source": "UniProt", "selected": True},
        ]},
        {"sequenceIndex": 2, "substrates": [
            {"substrateIndex": 1, "prediction": value, "source": "Prediction from CatRange", "selected": True},
            {"substrateIndex": 2, "prediction": value, "source": "Prediction from CatRange", "selected": False},
        ]},
    ]
    result = _result([json.dumps([value, value])], ["Mixed per-substrate sources; see Extra Info"], [json.dumps(items)])
    report = build_catrange_report_columns("Km", result)
    assert json.loads(report["Predicted range"][0]) == ["1e-05 to 0.0001 M", None]
    assert json.loads(report["Experimental KM (source units)"][0]) == [None, value]


def test_mixed_source_array_without_provenance_is_rejected():
    with pytest.raises(ValueError, match="provenance for substrate 1"):
        build_catrange_report_columns("Km", _result(["[0.2]"], ["Mixed per-substrate sources; see Extra Info"]))


def test_csv_and_json_export_preserve_primary_range_and_numeric_secondary(tmp_path):
    value = math.sqrt(10)
    report = build_catrange_report_columns("kcat", _result([value, None, value], ["Prediction from CatRange", "too long", "BRENDA"]))
    frame = pd.DataFrame(report)
    frame["Protein Sequence"] = ["AAA", "BBB", "CCC"]
    output = tmp_path / "output.csv"
    frame.to_csv(output, index=False)
    payload = serialize_result_csv(str(output))
    assert payload["columns"][0] == "Predicted range"
    assert payload["data"][0]["Predicted range"] == "1 to 10 s^-1"
    assert payload["data"][0]["Derived kcat range midpoint (1/s)"] == pytest.approx(value)
    assert payload["data"][1]["Predicted range"] is None
    assert payload["data"][2]["Experimental kcat (1/s)"] == pytest.approx(value)
    assert [row["Protein Sequence"] for row in payload["data"]] == ["AAA", "BBB", "CCC"]
    json.dumps(payload, allow_nan=False)


def test_production_frame_assembly_preserves_other_method_and_input_rows(tmp_path):
    inputs = pd.DataFrame({"Protein Sequence": ["AAA", "BBB"], "Substrate": ["O", "CC"]}, index=[9, 3])
    original = inputs.copy()
    catrange = _result([math.sqrt(10), None], ["Prediction from CatRange", "too long"])
    other = _result([0.25, 2.5], ["Prediction from UniKP"] * 2, ["first", "second"])
    other["output_col"] = "KM (mM)"
    frame = build_prediction_result_frame(inputs, ["kcat", "Km"], {"kcat": catrange, "Km": other}, {"kcat": "CatRange", "Km": "UniKP"})
    assert list(frame.columns) == [
        "Predicted range", "Source kcat", "Extra Info kcat",
        "Derived kcat range midpoint (1/s)", "Experimental kcat (1/s)",
        "KM (mM)", "Source Km", "Extra Info Km", "Protein Sequence", "Substrate",
    ]
    assert frame.index.tolist() == [9, 3]
    assert frame["KM (mM)"].tolist() == other["preds"]
    assert frame["Extra Info Km"].tolist() == other["extra"]
    pd.testing.assert_frame_equal(inputs, original)
    output = tmp_path / "mixed-output.csv"
    frame.to_csv(output, index=False)
    payload = serialize_result_csv(str(output))
    assert payload["data"][0]["Predicted range"] == "1 to 10 s^-1"
    assert payload["data"][0]["KM (mM)"] == 0.25
    assert payload["data"][1]["Predicted range"] is None


def test_array_csv_json_roundtrip_preserves_ordered_range_strings(tmp_path):
    value = math.sqrt(1e-5 * 1e-4)
    result = _result([json.dumps([value, None])], ["Prediction from CatRange (per substrate)"])
    frame = build_prediction_result_frame(pd.DataFrame({"Substrates": ["O;C"]}), ["Km"], {"Km": result}, {"Km": "CatRange"})
    output = tmp_path / "array-output.csv"
    frame.to_csv(output, index=False)
    payload = serialize_result_csv(str(output))
    assert json.loads(payload["data"][0]["Predicted range"]) == ["1e-05 to 0.0001 M", None]
    assert json.loads(payload["data"][0]["Derived KM range midpoint (M)"]) == [value, None]


def test_completion_counts_measured_rows_with_blank_range_and_excludes_failures():
    result = _result([math.sqrt(10), 2.0, None, "", float("nan")], ["Prediction from CatRange", "BRENDA", "failed", "failed", "failed"])
    report = build_catrange_report_columns("kcat", result)
    assert report["Predicted range"][1] == ""
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
