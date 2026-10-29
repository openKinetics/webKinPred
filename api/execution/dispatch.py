from pathlib import Path

from django.conf import settings

from api.execution.artifacts import sha256_file
from api.execution.control import ingest
from api.models import Job


def submit(
    public_id,
    targets,
    methods,
    experimental_results=None,
    canonicalize_substrates=True,
    include_similarity_columns=True,
    disable_gpu_precompute=False,
    recon_xkg=False,
):
    job = Job.objects.get(public_id=public_id)
    if disable_gpu_precompute:
        raise ValueError("Local embedding execution is disabled under the GPU-only policy")
    from api.methods.registry import all_methods
    from api.utils.job_utils import canonicalise_targets

    versions = settings.EXECUTION_MODEL_VERSIONS
    requested_targets, requested_methods = list(targets), dict(methods)
    targets = canonicalise_targets(targets)
    methods = {target: methods[target] for target in targets}
    if any(method not in versions for method in methods.values()):
        raise ValueError("Pinned model versions must be configured before ingestion")
    specification = {
        "targets": targets,
        "methods": methods,
        "requested_targets": requested_targets,
        "requested_methods": requested_methods,
        "experimental_results": experimental_results or {},
        "canonicalize_substrates": canonicalize_substrates,
        "include_similarity_columns": include_similarity_columns,
        "handle_long_sequences": job.handle_long_sequences,
        "recon_xkg": recon_xkg,
        "model_versions": {method: versions[method] for method in methods.values()},
        "execution_versions": getattr(settings, "EXECUTION_STAGE_VERSIONS", {}),
        "method_registry": {
            key: getattr(value, "model_version", "1")
            for key, value in all_methods().items()
            if key in methods.values()
        },
    }
    source = Path(settings.MEDIA_ROOT) / "jobs" / public_id / "input.csv"
    return ingest(
        public_id, specification, sha256_file(source), paused=settings.EXECUTION_START_PAUSED
    )
