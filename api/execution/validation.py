"""Explicit numerical waivers never waive resource, shape, or data-integrity checks."""


def gpu_numerical_waiver(evidence):
    waiver = evidence.get("gpu_numerical_waiver", {})
    return (
        waiver.get("approved_by_user") is True
        and waiver.get("scope") == "gpu_numerical_parity"
        and bool(waiver.get("request_reference"))
        and waiver.get("shape_dtype_finite_passed") is True
    )


def parity_accepted(evidence, *, gpu):
    return evidence.get("parity_passed") is True or (gpu and gpu_numerical_waiver(evidence))
