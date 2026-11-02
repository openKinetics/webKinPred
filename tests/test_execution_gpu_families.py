from copy import deepcopy
from types import SimpleNamespace

import pytest
from api.execution.control import _gpu_profile_allowed
from api.execution.gpu_families import families


def profiles():
    parent = SimpleNamespace(
        fingerprint="small",
        validated=True,
        suspended=False,
        stage_kind="embedding",
        method="KinForm-H",
        image="pinned-image",
        resources={"vram": 1},
        settings={
            "step": "kinform_esm2_residues",
            "batch_size": 1,
            "model_identity": {"weights": "a"},
        },
        limits={"sequences": 16, "max_length": 1500},
    )
    child = deepcopy(parent)
    child.fingerprint = "large"
    child.limits = {"sequences": 128, "max_length": 1024}
    child.settings["cohort_sources"] = ["small"]
    return parent, child


def test_work_unit_growth_inherits_only_a_measured_resident_model_mixture():
    parent, child = profiles()
    mapping = families([parent, child])
    assert mapping == {"large": "small"}
    host = SimpleNamespace(capabilities={"validated_gpu_combinations": [["other", "small"]]})
    active = [{"resources": {"vram": 1}, "profile_fingerprint": "other"}]
    assert _gpu_profile_allowed(child, active, host, mapping)
    active.append({"resources": {"vram": 1}, "profile_fingerprint": "large"})
    assert not _gpu_profile_allowed(child, active, host, mapping)


@pytest.mark.parametrize(
    "change", ["image", "batch", "model", "length", "molecular", "unvalidated", "suspended"]
)
def test_unmeasured_resident_models_never_inherit_a_mixture(change):
    parent, child = profiles()
    if change == "image":
        child.image = "different"
    if change == "batch":
        child.settings["batch_size"] = 2
    if change == "model":
        child.settings["model_identity"] = {"weights": "b"}
    if change == "length":
        child.limits["max_length"] = 1501
    if change == "molecular":
        parent.settings["step"] = child.settings["step"] = "molecular"
    if change == "unvalidated":
        child.validated = False
    if change == "suspended":
        parent.suspended = True
    assert families([parent, child]) == {}
