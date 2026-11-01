"""The prediction subprocess must consume the same text as embedding planning."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from api.execution.context import StageContext, execution_job, stage_context
from api.methods.registry import get
from api.prediction_engines import generic_subprocess as engine
from api.tasks import _execute_target_batch
from api.utils.convert_to_mol import substrate_as_smiles


@pytest.mark.parametrize("target", ["kcat", "Km"])
@pytest.mark.parametrize("canonicalize", [False, True])
@pytest.mark.parametrize("batch_size", [1, 128])
def test_catrange_payload_matches_planned_encoder_text(
    monkeypatch, tmp_path, target, canonicalize, batch_size
):
    raw = ["OCC", "InChI=1S/H2O/h1H2", "not-a-molecule", "C[C@H](O)C(=O)O"]
    sequences = ["MAAA", "MAAC", "MAAG", "MAAT"]
    expected = [
        substrate_as_smiles(s, canonicalize=canonicalize, preserve_raw_smiles_when_possible=True)
        for s in raw
    ]
    assert expected[0] == ("CCO" if canonicalize else "OCC")
    assert expected[1] == "O"
    assert expected[2] is None
    received = []

    def predict(*, command, **kwargs):
        payload = json.loads(open(command[-3]).read())
        received.extend((r["sequence"], r["substrates"]) for r in payload["rows"])
        with open(command[-1], "w") as output:
            json.dump(
                {
                    "predictions": [1.0] * len(payload["rows"]),
                    "extra_info": ["Predicted range: 1 to 10"] * len(payload["rows"]),
                },
                output,
            )

    monkeypatch.setattr(engine, "MEDIA_ROOT", str(tmp_path))
    (tmp_path / "jobs/context-match").mkdir(parents=True)
    monkeypatch.setattr(engine, "_resolve_subprocess_paths", lambda _desc: ("python", "probe"))
    monkeypatch.setattr(engine, "_build_subprocess_env", lambda _desc: {})
    monkeypatch.setattr(engine, "_attach_seq_ids_to_rows", lambda **_kwargs: None)
    monkeypatch.setattr(
        engine,
        "run_gpu_precompute_if_available",
        lambda **_kwargs: SimpleNamespace(attempted=False, failed=False),
    )
    monkeypatch.setattr(engine, "run_prediction_subprocess", predict)
    frame = pd.DataFrame({"Protein Sequence": sequences, "Substrate": raw})
    with stage_context(StageContext("context-match", unit_batch_size=batch_size)):
        result = _execute_target_batch(
            job=execution_job("context-match"),
            desc=get("CatRange"),
            df=frame,
            target=target,
            sequences=sequences,
            processed_by_reaction=sequences,
            valid_reaction_indices=list(range(len(raw))),
            experimental_results=[],
            canonicalize_substrates=canonicalize,
            disable_gpu_precompute=False,
            extra_call_kwargs={},
        )
    assert received == [(seq, text) for seq, text in zip(sequences, expected, strict=True) if text]
    assert result["preds"][2] == ""
    assert "Invalid substrate" in result["sources"][2]
    assert set(result["failed_reactions"]) == {2}
    assert all(result["extra"][i] == "Predicted range: 1 to 10" for i in [0, 1, 3])
