"""Regression checks for the production artifact path and required scaling."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.methods.catrange import descriptor


@pytest.fixture
def inference_module(monkeypatch):
    path = REPO_ROOT / "models/CatRange/inference/catrange_inference.py"
    spec = importlib.util.spec_from_file_location("catrange_normalization_impl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Unit fixtures supply their own artifacts; never download model weights.
    monkeypatch.setattr(module, "_ensure_model_artifacts", lambda _path: None)
    return module


def _load_adapter(monkeypatch, inference_module):
    package = types.ModuleType("inference")
    package.catrange_inference = inference_module
    monkeypatch.setitem(sys.modules, "inference", package)
    monkeypatch.setitem(sys.modules, "inference.catrange_inference", inference_module)
    path = REPO_ROOT / "models/CatRange/predict.py"
    spec = importlib.util.spec_from_file_location("catrange_normalization_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def _build_data_paths(monkeypatch):
    # Import the real configuration without starting the Celery application.
    package = types.ModuleType("webKinPred")
    package.__path__ = [str(REPO_ROOT / "webKinPred")]
    monkeypatch.setitem(sys.modules, "webKinPred", package)
    path = REPO_ROOT / "webKinPred/config_base.py"
    spec = importlib.util.spec_from_file_location("catrange_config_base", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.build_data_paths(REPO_ROOT)


@pytest.mark.parametrize("parameter", ["kcat", "km"])
def test_deployed_environment_uses_tracked_stats(monkeypatch, inference_module, parameter):
    paths = _build_data_paths(monkeypatch)
    for env_var, data_key in descriptor.subprocess.data_path_env.items():
        monkeypatch.setenv(env_var, paths[data_key])
    adapter = _load_adapter(monkeypatch, inference_module)
    predictor = adapter._load_inference()

    assert descriptor.subprocess.data_path_env["CATRANGE_MODELS_DIR"] == "CatRange"
    assert predictor.models_dir == REPO_ROOT / "models/CatRange"
    # Existing deployments may have downloaded weights at the repo root;
    # normalization must still resolve to the tracked inference/models files.
    model_path = predictor.models_dir / "model_weights" / f"{parameter}_model_v1b.pkl"
    stats_path = inference_module._resolve_stats_path(predictor.models_dir, model_path, parameter)
    assert stats_path == predictor.models_dir / "inference/models" / f"{parameter}_esmc_FINAL_stats.pt"
    assert stats_path.is_file()
    stats = inference_module._validate_stats(
        torch.load(stats_path, map_location="cpu", weights_only=False), stats_path
    )
    assert stats["std_1"] != 1.0
    assert descriptor.model_version != "1"


def test_explicit_artifact_directory_remains_supported(monkeypatch, inference_module, tmp_path):
    monkeypatch.setenv("CATRANGE_MODELS_DIR", str(tmp_path / "from_environment"))
    adapter = _load_adapter(monkeypatch, inference_module)
    assert adapter._load_inference().models_dir == tmp_path / "from_environment"
    assert adapter._load_inference(tmp_path / "explicit").models_dir == tmp_path / "explicit"


def _valid_stats():
    return {"mean_1": torch.tensor(2.0), "std_1": torch.tensor(4.0),
            "mean_2": torch.tensor(-1.0), "std_2": torch.tensor(2.0)}


@pytest.mark.parametrize("parameter", ["kcat", "km"])
@pytest.mark.parametrize("layout", [
    "flat", "nested_weights", "nested_weights_and_stats", "repo_root",
    "repo_root_nested_weights", "repo_root_legacy_weights", "repo_root_flat_weights",
])
def test_classifier_receives_standardized_features(
    monkeypatch, inference_module, tmp_path, parameter, layout
):
    tracked_dir = tmp_path / "inference/models"
    weights_dir = {
        "flat": tmp_path,
        "nested_weights": tmp_path / "model_weights",
        "nested_weights_and_stats": tmp_path / "model_weights",
        "repo_root": tracked_dir,
        "repo_root_nested_weights": tracked_dir / "model_weights",
        "repo_root_legacy_weights": tmp_path / "model_weights",
        "repo_root_flat_weights": tmp_path,
    }[layout]
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / f"{parameter}_model_v1b.pkl").write_bytes(b"stub model")
    stats_dir = tracked_dir if layout.startswith("repo_root") else (
        weights_dir if layout == "nested_weights_and_stats" else tmp_path
    )
    stats_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_valid_stats(), stats_dir / f"{parameter}_esmc_FINAL_stats.pt")
    seen = []

    class Classifier:
        def predict_proba(self, features):
            seen.append(features.copy())
            probabilities = np.zeros((2, len(inference_module.BIN_EDGES[parameter]) - 1))
            probabilities[0, 4] = probabilities[1, 1] = 1.0
            return probabilities

    monkeypatch.setattr(inference_module.joblib, "load", lambda _path: Classifier())
    predictor = inference_module.CatRangeInference(tmp_path, device="cpu")
    output = predictor.predict_from_embeddings(
        np.array([[6, 10], [-2, 2]], dtype=np.float32),
        np.array([[1, 3], [-3, -1]], dtype=np.float32),
        parameter=parameter,
    )

    np.testing.assert_array_equal(seen[0], [[1, 2, 1, 2], [-1, 0, -1, 0]])
    assert seen[0].dtype == np.float32
    assert output[f"{parameter}_pred_bin"].tolist() == [4, 1]


def test_repo_root_artifacts_do_not_trigger_download(monkeypatch, tmp_path):
    path = REPO_ROOT / "models/CatRange/inference/catrange_inference.py"
    spec = importlib.util.spec_from_file_location("catrange_artifact_ensure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights_dir = tmp_path / "inference/models/model_weights"
    weights_dir.mkdir(parents=True)
    for parameter in ("kcat", "km"):
        (weights_dir / f"{parameter}_model_v1b.pkl").write_bytes(b"stub model")

    def reject_download(*args, **kwargs):
        pytest.fail("Existing CatRange artifacts must not trigger a download")

    monkeypatch.setattr(module.urllib.request, "urlopen", reject_download)
    module._ensure_model_artifacts(tmp_path)
    assert not (tmp_path / "model_weights.zip").exists()


def test_stats_beside_selected_weights_take_precedence(inference_module, tmp_path):
    weights_dir = tmp_path / "inference/models/model_weights"
    weights_dir.mkdir(parents=True)
    name = "kcat_esmc_FINAL_stats.pt"
    for directory in (tmp_path, tmp_path / "inference/models", weights_dir):
        torch.save(_valid_stats(), directory / name)
    model_path = weights_dir / "kcat_model_v1b.pkl"
    assert inference_module._resolve_stats_path(tmp_path, model_path, "kcat") == weights_dir / name


def test_explicit_root_weights_take_precedence_over_bundled_weights(inference_module, tmp_path):
    root_model = tmp_path / "kcat_model_v1b.pkl"
    bundled_model = tmp_path / "inference/models/kcat_model_v1b.pkl"
    bundled_model.parent.mkdir(parents=True)
    root_model.write_bytes(b"explicit model")
    bundled_model.write_bytes(b"bundled model")
    assert inference_module._resolve_model_path(tmp_path, "kcat") == root_model


@pytest.mark.parametrize("parameter", ["kcat", "km"])
def test_esm2_weights_are_not_an_esmc_fallback(inference_module, tmp_path, parameter):
    for name in (f"{parameter}_model.pkl", f"{parameter}.pkl"):
        (tmp_path / name).write_bytes(b"incompatible ESM-2 weights")
    with pytest.raises(FileNotFoundError, match="Missing CatRange model artifact"):
        inference_module._resolve_model_path(tmp_path, parameter)


@pytest.mark.parametrize("parameter", ["kcat", "km"])
def test_v1b_is_preferred_across_supported_directories(inference_module, tmp_path, parameter):
    for name in (f"{parameter}_esmc_FINAL.pkl", f"{parameter}_model.pkl"):
        (tmp_path / name).write_bytes(b"other weights")
    released = tmp_path / "inference/models/model_weights" / f"{parameter}_model_v1b.pkl"
    released.parent.mkdir(parents=True)
    released.write_bytes(b"released ESM-C weights")
    assert inference_module._resolve_model_path(tmp_path, parameter) == released


def test_missing_stats_fail_before_loading_weights(monkeypatch, inference_module, tmp_path):
    (tmp_path / "kcat_model_v1b.pkl").write_bytes(b"stub model")
    load_calls = []
    monkeypatch.setattr(inference_module.joblib, "load", lambda path: load_calls.append(path))
    predictor = inference_module.CatRangeInference(tmp_path, device="cpu")
    with pytest.raises(FileNotFoundError, match="Missing required CatRange standardization"):
        predictor._load_model("kcat")
    assert not load_calls
    assert not predictor._models
    assert not predictor._stats


@pytest.mark.parametrize(
    "key,value",
    [("mean_1", float("nan")), ("mean_2", float("inf")),
     ("std_1", 0.0), ("std_1", -1.0), ("std_2", float("nan")),
     ("std_2", float("inf")), ("mean_1", [2.0]), ("std_2", "2.0")],
)
def test_invalid_stats_fail_before_prediction(
    monkeypatch, inference_module, tmp_path, key, value
):
    (tmp_path / "kcat_model_v1b.pkl").write_bytes(b"stub model")
    stats = _valid_stats()
    stats[key] = value
    torch.save(stats, tmp_path / "kcat_esmc_FINAL_stats.pt")
    load_calls = []
    monkeypatch.setattr(inference_module.joblib, "load", lambda path: load_calls.append(path))
    predictor = inference_module.CatRangeInference(tmp_path, device="cpu")
    with pytest.raises(ValueError, match=key):
        predictor._load_model("kcat")
    assert not load_calls
    assert not predictor._models
    assert not predictor._stats


@pytest.mark.parametrize("key", ["mean_1", "std_1", "mean_2", "std_2"])
def test_incomplete_stats_are_rejected(inference_module, tmp_path, key):
    stats = _valid_stats()
    del stats[key]
    with pytest.raises(ValueError, match=f"missing {key}"):
        inference_module._validate_stats(stats, tmp_path / "stats.pt")


def test_non_mapping_stats_are_rejected(inference_module, tmp_path):
    with pytest.raises(ValueError, match="expected a dict"):
        inference_module._validate_stats([0.0, 1.0, 0.0, 1.0], tmp_path / "stats.pt")
