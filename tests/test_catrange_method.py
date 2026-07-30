import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.methods.registry import get


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
    spec = importlib.util.spec_from_file_location("catrange_predict", module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    inference = module._load_inference()
    expected_models_dir = (module_path.parent / "inference" / "models").resolve()

    assert inference.models_dir == expected_models_dir
    assert inference._resolve_model_path(inference.models_dir, "kcat") == (
        inference.models_dir / "kcat_model_v1b.pkl"
    )
    assert inference._resolve_model_path(inference.models_dir, "km") == (
        inference.models_dir / "km_model_v1b.pkl"
    )
