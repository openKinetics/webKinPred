"""CatRange output contracts without model loading or inference."""

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

from api.methods.catrange import descriptor
from api.services.catrange_reporting import (
    BIN_EDGES,
    build_prediction_result_frame,
    catrange_range_label,
    completed_reaction_count,
)
from api.services.result_service import serialize_result_csv
from api.utils.extra_info import build_extra_info

CATRANGE = "Prediction from CatRange"
MIXED = "Mixed per-substrate sources; see Extra Info"
KCAT_COL = descriptor.output_cols["kcat"]
KM_COL = descriptor.output_cols["Km"]


def _result(preds, sources=None, extra=None, output_col=KCAT_COL):
    return {
        "preds": preds,
        "sources": sources if sources is not None else [CATRANGE] * len(preds),
        "extra": extra if extra is not None else [""] * len(preds),
        "output_col": output_col,
    }


def _serialize(frame, tmp_path):
    output = tmp_path / "output.csv"
    frame.to_csv(output, index=False)
    return serialize_result_csv(str(output))


def test_reporting_edges_match_runtime_without_importing_model():
    source = REPO_ROOT / "models/CatRange/inference/catrange_inference.py"
    tree = ast.parse(source.read_text())
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "BIN_EDGES" for target in node.targets)
    )
    runtime_edges = {
        ast.literal_eval(key): tuple(ast.literal_eval(value.args[0]))
        for key, value in zip(assignment.value.keys, assignment.value.values)
    }
    assert BIN_EDGES == {"kcat": runtime_edges["kcat"], "Km": runtime_edges["km"]}


@pytest.mark.parametrize("target,low,high,label", [
    ("kcat", 0, 1e-8, "0 to 1e-08 s^-1"),
    ("kcat", 1e-8, 1e-2, "1e-08 to 0.01 s^-1"),
    ("kcat", 1e-2, 1e-1, "0.01 to 0.1 s^-1"),
    ("kcat", 1e-1, 1, "0.1 to 1 s^-1"),
    ("kcat", 1, 10, "1 to 10 s^-1"),
    ("kcat", 10, 100, "10 to 100 s^-1"),
    ("kcat", 100, 1000, "100 to 1000 s^-1"),
    ("kcat", 1000, 1e8, "1000 to 1e+08 s^-1"),
    ("Km", 1e-14, 1e-5, "1e-14 to 1e-05 M"),
    ("Km", 1e-5, 1e-4, "1e-05 to 0.0001 M"),
    ("Km", 1e-4, 1e-3, "0.0001 to 0.001 M"),
    ("Km", 1e-3, 1e-2, "0.001 to 0.01 M"),
    ("Km", 1e-2, 1e-1, "0.01 to 0.1 M"),
    ("Km", 1e-1, 1e4, "0.1 to 10000 M"),
])
def test_every_class_is_reported_as_its_runtime_range(target, low, high, label):
    value = math.sqrt(low * high) if low else high / 2
    assert catrange_range_label(target, value) == label


@pytest.mark.parametrize("value", [2.0, 0, -1.0, float("inf"), True])
def test_unknown_midpoint_does_not_create_a_range(value):
    with pytest.raises(ValueError, match="not a known class midpoint"):
        catrange_range_label("kcat", value)


def test_production_frame_preserves_other_method_input_order_and_numeric_results(tmp_path):
    inputs = pd.DataFrame({"Protein Sequence": ["AAA", "BBB"], "Substrate": ["O", "CC"]}, index=[9, 3])
    original = inputs.copy()
    catrange = _result(
        [math.sqrt(10), None], [CATRANGE, "too long"],
        ["Predicted range: 1 to 10 s^-1", "too long"],
    )
    other = _result([0.25, 2.5], ["Prediction from UniKP"] * 2, ["first", "second"], output_col="KM (mM)")
    target_results = {"kcat": catrange, "Km": other}
    before = copy.deepcopy(target_results)
    frame = build_prediction_result_frame(inputs, ["kcat", "Km"], target_results)
    assert list(frame.columns) == [
        KCAT_COL, "Source kcat", "Extra Info kcat",
        "KM (mM)", "Source Km", "Extra Info Km",
        "Protein Sequence", "Substrate",
    ]
    assert frame.index.tolist() == [9, 3]
    assert frame[KCAT_COL].tolist() == ["1 to 10 s^-1", None]
    assert frame["Extra Info kcat"].tolist() == catrange["extra"]
    assert frame["KM (mM)"].tolist() == other["preds"]
    assert frame["Extra Info Km"].tolist() == other["extra"]
    assert target_results == before
    pd.testing.assert_frame_equal(inputs, original)
    payload = _serialize(frame, tmp_path)
    assert payload["data"][0][KCAT_COL] == "1 to 10 s^-1"
    assert payload["data"][0]["KM (mM)"] == 0.25
    assert payload["data"][1][KCAT_COL] is None


def test_measurement_equal_to_midpoint_stays_numeric_in_csv_json(tmp_path):
    value = math.sqrt(10)
    sources = [CATRANGE, "BRENDA", "Sabio-RK", "UniProt", "Experimental"]
    result = _result([value] * len(sources), sources)
    frame = build_prediction_result_frame(pd.DataFrame(index=range(len(sources))), ["kcat"], {"kcat": result})
    assert frame[KCAT_COL].tolist() == ["1 to 10 s^-1"] + [value] * 4
    data = _serialize(frame, tmp_path)["data"]
    assert data[0][KCAT_COL] == "1 to 10 s^-1"
    for record in data[1:]:
        assert isinstance(record[KCAT_COL], float)
        assert record[KCAT_COL] == pytest.approx(value)


@pytest.mark.parametrize("source", [CATRANGE + " (per substrate)", MIXED])
def test_array_keeps_null_slots_and_existing_json_encoding(tmp_path, source):
    value = math.sqrt(1e-5 * 1e-4)
    items = [
        {"substrateIndex": 1, "prediction": value, "source": CATRANGE, "details": "Predicted range: 1e-05 to 0.0001 M"},
        {"substrateIndex": 2, "prediction": None, "source": "", "error": "bad substrate"},
    ]
    result = _result([json.dumps([value, None])], [source], [json.dumps(items)], output_col=KM_COL)
    before = copy.deepcopy(result)
    frame = build_prediction_result_frame(pd.DataFrame({"Substrates": ["O;C"]}), ["Km"], {"Km": result})
    record = _serialize(frame, tmp_path)["data"][0]
    assert isinstance(record[KM_COL], str)
    assert json.loads(record[KM_COL]) == ["1e-05 to 0.0001 M", None]
    expected = copy.deepcopy(items)
    expected[0]["prediction"] = "1e-05 to 0.0001 M"
    assert json.loads(record["Extra Info Km"]) == expected
    assert result == before


def test_nested_mixed_source_array_uses_selected_provenance_for_ties(tmp_path):
    value = math.sqrt(1e-5 * 1e-4)
    metadata = [
        {"sequenceIndex": 1, "sequence": "AAA", "substrates": [
            {"substrateIndex": 1, "prediction": value, "source": CATRANGE, "selected": False},
            {"substrateIndex": 2, "prediction": value, "source": "BRENDA", "selected": False},
            {"substrateIndex": 3, "prediction": None, "source": "", "selected": False, "error": "invalid"},
        ]},
        {"sequenceIndex": 2, "sequence": "BBB", "substrates": [
            {"substrateIndex": 1, "prediction": value, "source": "BRENDA", "selected": True},
            {"substrateIndex": 2, "prediction": value, "source": CATRANGE, "selected": True},
            {"substrateIndex": 3, "prediction": None, "source": "", "selected": False, "error": "invalid"},
        ]},
    ]
    result = _result([json.dumps([value, value, None])], [MIXED], [json.dumps(metadata)], output_col=KM_COL)
    before = copy.deepcopy(result)
    frame = build_prediction_result_frame(pd.DataFrame(index=[0]), ["Km"], {"Km": result})
    record = _serialize(frame, tmp_path)["data"][0]
    assert json.loads(record[KM_COL]) == [value, "1e-05 to 0.0001 M", None]
    expected = copy.deepcopy(metadata)
    expected[0]["substrates"][0]["prediction"] = "1e-05 to 0.0001 M"
    expected[1]["substrates"][1]["prediction"] = "1e-05 to 0.0001 M"
    assert json.loads(record["Extra Info Km"]) == expected
    assert result == before


def test_ambiguous_array_source_does_not_guess_measurement_or_prediction():
    value = math.sqrt(1e-5 * 1e-4)
    items = [
        {"substrateIndex": 1, "prediction": value, "source": CATRANGE},
        {"substrateIndex": 1, "prediction": value, "source": "BRENDA"},
    ]
    result = _result([json.dumps([value])], [MIXED], [json.dumps(items)], output_col=KM_COL)
    with pytest.raises(ValueError, match="Cannot identify CatRange result source"):
        build_prediction_result_frame(pd.DataFrame(index=[0]), ["Km"], {"Km": result})


@pytest.mark.parametrize("nested", [False, True])
def test_experimental_override_prose_retains_model_range_and_measured_value(tmp_path, nested):
    value = math.sqrt(10)
    detail = build_extra_info({"found": True, "from_brenda": 1, "protein_ID": "P12345"}, "kcat", value, "CatRange")
    assert "Prediction by CatRange is 1 to 10 s^-1" in detail
    assert str(value) not in detail
    metadata = [{"sequenceIndex": 1, "substrates": [
        {"substrateIndex": 1, "prediction": 2.0, "source": "BRENDA", "details": detail, "selected": True},
        {"substrateIndex": 2, "prediction": value, "source": CATRANGE, "selected": False},
    ]}]
    extra = json.dumps(metadata) if nested else detail
    result = _result([2.0], ["BRENDA"], [extra])
    frame = build_prediction_result_frame(pd.DataFrame(index=[0]), ["kcat"], {"kcat": result})
    record = _serialize(frame, tmp_path)["data"][0]
    assert record[KCAT_COL] == 2.0
    if nested:
        expected = copy.deepcopy(metadata)
        expected[0]["substrates"][1]["prediction"] = "1 to 10 s^-1"
        assert json.loads(record["Extra Info kcat"]) == expected
    else:
        assert record["Extra Info kcat"] == detail
    assert result["extra"] == [extra]


def test_other_model_experimental_details_keep_point_prediction():
    detail = build_extra_info({"found": True}, "kcat", 2.5, "UniKP")
    assert detail.endswith("Prediction by UniKP is 2.5")


def test_completion_counts_measured_rows_and_excludes_failures():
    result = _result([math.sqrt(10), 2.0, None, "", float("nan")], [CATRANGE, "BRENDA", "failed", "failed", "failed"])
    assert completed_reaction_count(["kcat"], {"kcat": result}) == 2


def test_completion_requires_every_requested_target_by_row_position():
    kcat = _result([2.0, None, 4.0, 0.0])
    km = _result([0.2, 0.3, "", 0.4])
    kcat["preds"] = pd.Series(kcat["preds"], index=[9, 3, 2, 0])
    assert completed_reaction_count(["kcat", "Km"], {"kcat": kcat, "Km": km}) == 2


def test_completion_rejects_misaligned_target_lengths():
    with pytest.raises(ValueError, match="same number of rows"):
        completed_reaction_count(["kcat", "Km"], {"kcat": _result([1]), "Km": _result([])})


def test_reporting_rejects_misaligned_source_lengths():
    result = _result([math.sqrt(10)], sources=[])
    with pytest.raises(ValueError, match="must have matching lengths"):
        build_prediction_result_frame(pd.DataFrame(index=[0]), ["kcat"], {"kcat": result})
