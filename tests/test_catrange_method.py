import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.methods.registry import get


def _load_catrange_predict_module():
    module_path = REPO_ROOT / "models" / "CatRange" / "predict.py"
    spec = importlib.util.spec_from_file_location("catrange_predict", module_path)
    assert spec is not None and spec.loader is not None

    fake_inference_module = types.ModuleType("inference.catrange_inference")

    class CatRangeInference:  # pragma: no cover - stub for import-time dependency isolation
        def __init__(self, *args, **kwargs):
            self.models_dir = kwargs.get("models_dir")

        def _resolve_model_path(self, models_dir, parameter):
            return models_dir / f"{parameter}_model_v1b.pkl"

    fake_inference_module.CatRangeInference = CatRangeInference
    fake_package = types.ModuleType("inference")
    fake_package.catrange_inference = fake_inference_module
    sys.modules["inference"] = fake_package
    sys.modules["inference.catrange_inference"] = fake_inference_module

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_catrange_inference_module():
    module_path = REPO_ROOT / "models" / "CatRange" / "inference" / "catrange_inference.py"
    spec = importlib.util.spec_from_file_location("catrange_inference_impl", module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catrange_descriptor_is_registered_and_runnable() -> None:
    desc = get("CatRange")

    assert desc.key == "CatRange"
    assert desc.display_name == "CatRange"
    assert desc.supports == ["kcat", "Km"]
    assert desc.output_cols["kcat"].startswith("Predicted kcat")
    assert desc.output_cols["Km"].startswith("Predicted KM")
    assert desc.subprocess is not None
    assert desc.subprocess.python_path_key == "CatRange"
    assert desc.subprocess.script_key == "CatRange"


def test_realkcat_is_not_registered() -> None:
    try:
        get("RealKcat")
    except KeyError:
        return
    raise AssertionError("RealKcat should no longer be registered")


def test_catrange_predict_uses_local_inference_models_dir() -> None:
    module_path = REPO_ROOT / "models" / "CatRange" / "predict.py"
    module = _load_catrange_predict_module()

    inference = module._load_inference()
    expected_models_dir = (module_path.parent / "inference" / "models").resolve()

    assert inference.models_dir == expected_models_dir
    assert inference._resolve_model_path(inference.models_dir, "kcat") == (
        inference.models_dir / "kcat_model_v1b.pkl"
    )
    assert inference._resolve_model_path(inference.models_dir, "km") == (
        inference.models_dir / "km_model_v1b.pkl"
    )


def test_catrange_prediction_payload_uses_range_median_and_range_text() -> None:
    module = _load_catrange_predict_module()

    frame = pd.DataFrame({"kcat_pred_range": ["1e-8 to 1e-2 s^-1"]})
    payload = module._build_prediction_payload(frame, target="kcat")

    assert payload["predictions"] == pytest.approx([1e-5])
    assert payload["extra_info"] == ["Predicted range: 1e-8 to 1e-2 s^-1"]


def test_catrange_model_lookup_finds_nested_model_weights_dir(tmp_path: Path) -> None:
    module = _load_catrange_inference_module()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    nested_dir = model_dir / "model_weights"
    nested_dir.mkdir()
    (nested_dir / "kcat_model_v1b.pkl").write_bytes(b"stub")

    assert module._resolve_model_path(model_dir, "kcat") == nested_dir / "kcat_model_v1b.pkl"
