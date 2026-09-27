"""Unsupported substrates must not erase other rows in an API batch."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from test_catrange_normalization import _load_adapter
from test_catrange_normalization import inference_module as inference_module


@pytest.fixture
def adapter(monkeypatch, inference_module):
    return _load_adapter(monkeypatch, inference_module)


def _install_predictor(monkeypatch, adapter, inference_module):
    calls = {"protein_rows": [], "pairs": [], "setup": []}

    class Predictor:
        def _load_model(self, parameter):
            calls["setup"].append(parameter)

        def _load_chemberta(self):
            calls["setup"].append("ChemBERTa")

        def embed_smiles(self, smiles):
            if smiles == "unsupported":
                raise inference_module.CatRangeInputError("Substrate has 559 tokens; maximum 510")
            return np.array([len(smiles)], dtype=np.float32)

        def predict_from_embeddings(self, sequences, substrates, parameter):
            calls["pairs"].append((sequences.copy(), substrates.copy()))
            return pd.DataFrame({f"{parameter}_pred_range": [
                "1 to 10 s^-1", "100 to 1000 s^-1"
            ][:len(sequences)]})

    predictor = Predictor()
    monkeypatch.setattr(adapter, "_load_inference", lambda: predictor)

    def load_proteins(rows, sequences):
        calls["protein_rows"].append([row["seq_id"] for row in rows])
        return np.array([[len(sequence)] for sequence in sequences], dtype=np.float32)

    monkeypatch.setattr(adapter, "_load_protein_embeddings", load_proteins)
    return predictor, calls


def _mixed_rows():
    return [
        {"seq_id": "first", "sequence": "MAAA", "substrates": "CC"},
        {"seq_id": "middle", "sequence": "MAAAAAA", "substrates": "unsupported"},
        {"seq_id": "last", "sequence": "MAAAAA", "substrates": "CCCC"},
    ]


@pytest.mark.parametrize("target", ["kcat", "Km"])
def test_unsupported_substrate_preserves_neighbor_predictions(
    monkeypatch, adapter, inference_module, target, capsys
):
    _, calls = _install_predictor(monkeypatch, adapter, inference_module)
    predictions, invalid, extra = adapter.predict_rows(_mixed_rows(), target)
    assert predictions[0] == pytest.approx(np.sqrt(10))
    assert predictions[1] is None
    assert predictions[2] == pytest.approx(np.sqrt(100_000))
    assert invalid == [1]
    assert extra[1] == "Substrate has 559 tokens; maximum 510"
    assert calls["protein_rows"] == [["first", "last"]]
    np.testing.assert_array_equal(calls["pairs"][0][0], [[4], [6]])
    np.testing.assert_array_equal(calls["pairs"][0][1], [[2], [4]])
    assert "Progress: 3/3" in capsys.readouterr().out


def test_all_invalid_rows_skip_classifier_and_protein_embedding(
    monkeypatch, adapter, inference_module
):
    _, calls = _install_predictor(monkeypatch, adapter, inference_module)
    rows = [_mixed_rows()[1], {"sequence": "MAAA", "substrates": ""}]
    predictions, invalid, extra = adapter.predict_rows(rows, "kcat")
    assert predictions == [None, None]
    assert invalid == [0, 1]
    assert all(extra)
    assert not calls["protein_rows"]
    assert not calls["pairs"]
    assert calls["setup"] == ["kcat", "ChemBERTa"]


def test_cli_retains_actionable_row_failure_reason(
    monkeypatch, adapter, inference_module, tmp_path
):
    _install_predictor(monkeypatch, adapter, inference_module)
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps({"rows": _mixed_rows(), "target": "kcat"}))
    monkeypatch.setattr(sys, "argv", ["predict", "--input", str(input_path),
                                     "--output", str(output_path)])
    adapter.main()
    payload = json.loads(output_path.read_text())
    assert payload["invalid_indices"] == [1]
    assert payload["invalid_reasons"] == {"1": "Substrate has 559 tokens; maximum 510"}
    assert payload["extra_info"][1] == payload["invalid_reasons"]["1"]
    assert len(payload["predictions"]) == 3


def test_setup_failure_is_batch_fatal(monkeypatch, adapter, inference_module, tmp_path, capsys):
    predictor, calls = _install_predictor(monkeypatch, adapter, inference_module)

    def fail_setup(_parameter):
        raise FileNotFoundError("Missing required CatRange standardization statistics")

    monkeypatch.setattr(predictor, "_load_model", fail_setup)
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps({"rows": _mixed_rows(), "target": "kcat"}))
    monkeypatch.setattr(sys, "argv", ["predict", "--input", str(input_path),
                                     "--output", str(output_path)])
    with pytest.raises(SystemExit) as error:
        adapter.main()
    assert error.value.code == 1
    assert "Missing required CatRange standardization statistics" in capsys.readouterr().err
    assert not output_path.exists()
    assert not calls["pairs"]


def test_runtime_failure_is_not_misreported_as_bad_substrate(
    monkeypatch, adapter, inference_module
):
    predictor, _ = _install_predictor(monkeypatch, adapter, inference_module)

    def fail_runtime(_smiles):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(predictor, "embed_smiles", fail_runtime)
    with pytest.raises(RuntimeError, match="out of memory"):
        adapter.predict_rows(_mixed_rows(), "kcat")


def _stub_encoder(inference_module, token_count, tokenizer_limit=10**30, nonfinite=False):
    calls = {"tokenizer": [], "forward": []}

    class Tokenizer:
        model_max_length = tokenizer_limit

        def __call__(self, texts, **kwargs):
            calls["tokenizer"].append(kwargs)
            return {"input_ids": torch.zeros((1, token_count), dtype=torch.long)}

    class Encoder:
        config = SimpleNamespace(model_type="roberta", max_position_embeddings=512)
        embeddings = SimpleNamespace(padding_idx=1)

        def __call__(self, **kwargs):
            calls["forward"].append(kwargs)
            value = float("nan") if nonfinite else 2.0
            return SimpleNamespace(last_hidden_state=torch.full((1, token_count, 3), value))

    predictor = inference_module.CatRangeInference.__new__(inference_module.CatRangeInference)
    predictor.device = torch.device("cpu")
    predictor._chem_tokenizer = Tokenizer()
    predictor._chem_model = Encoder()
    return predictor, calls


@pytest.mark.parametrize("token_count", [511, 512, 559])
def test_roberta_rejects_over_limit_tokens_before_forward(inference_module, token_count):
    predictor, calls = _stub_encoder(inference_module, token_count)
    with pytest.raises(inference_module.CatRangeInputError, match=f"{token_count} tokens.*510"):
        predictor.embed_smiles("CC")
    assert not calls["forward"]
    assert calls["tokenizer"][0]["truncation"] is False


def test_roberta_accepts_exact_token_limit(inference_module):
    predictor, calls = _stub_encoder(inference_module, 510)
    np.testing.assert_array_equal(predictor.embed_smiles("CC"), [2.0, 2.0, 2.0])
    assert len(calls["forward"]) == 1
    assert calls["tokenizer"][0]["truncation"] is False


def test_smaller_tokenizer_limit_is_respected(inference_module):
    predictor, calls = _stub_encoder(inference_module, 129, tokenizer_limit=128)
    with pytest.raises(inference_module.CatRangeInputError, match="maximum 128"):
        predictor.embed_smiles("CC")
    assert not calls["forward"]


def test_nonfinite_substrate_embedding_is_a_row_error(inference_module):
    predictor, _ = _stub_encoder(inference_module, 10, nonfinite=True)
    with pytest.raises(inference_module.CatRangeInputError, match="non-finite"):
        predictor.embed_smiles("CC")
