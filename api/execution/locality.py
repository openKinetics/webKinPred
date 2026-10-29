"""Briefly prefer an idle compatible host that already owns large dependencies."""

from datetime import timedelta

from django.db.models import Max, Sum

from api.execution.models import ArtifactCleanup, ArtifactLocation, ExecutorHost, ResourceProfile
from api.execution.resources import LIVE_STATES, admit


def prefer_peer(stage, host, profile, input_bytes, now):
    if stage.kind != "predict" or profile.resources.get("vram") or input_bytes < 8 * 1024**2:
        return False
    ready_at = (
        stage.dependencies.aggregate(latest=Max("completed_at"))["latest"] or stage.created_at
    )
    # Keep backfilling possible when the preferred agent is slow or loses contact.
    if now - ready_at >= timedelta(seconds=3):
        return False
    locations = dict(
        ArtifactLocation.objects.filter(
            artifact_id__in=stage.artifacts.values("artifact_id"), ready=True
        )
        .values("host_id")
        .annotate(size=Sum("artifact__size"))
        .values_list("host_id", "size")
    )
    local = locations.get(host.pk, 0)
    for peer in ExecutorHost.objects.filter(
        paused=False, reconciled=True, heartbeat_at__gte=now - timedelta(seconds=10)
    ).exclude(pk=host.pk):
        if locations.get(peer.pk, 0) <= local + 8 * 1024**2 or peer.telemetry.get(
            "legacy_consumers"
        ):
            continue
        if ArtifactCleanup.objects.filter(host=peer, completed_at__isnull=True).exists():
            continue
        candidates = ResourceProfile.objects.filter(
            stage_kind=stage.kind,
            method=stage.method,
            validated=True,
            suspended=False,
            fingerprint__in=peer.capabilities.get("profiles", []),
        )
        for candidate in candidates:
            compatibility = candidate.evidence.get("runtime", {}).get("hosts", {})
            if (
                candidate.settings.get("model_identity") != profile.settings.get("model_identity")
                or candidate.settings.get("step") != stage.payload.get("step")
                or stage.target not in candidate.settings.get("targets", [stage.target])
                or candidate.resources.get("vram")
                or (
                    compatibility
                    and compatibility.get(peer.pk) != peer.capabilities.get("runtime_fingerprint")
                )
                or any(
                    stage.payload.get("shape", {}).get(k, float("inf")) > v
                    for k, v in candidate.limits.items()
                )
            ):
                continue
            resources = {
                k: max(v, candidate.runtime_floor.get(k, v)) for k, v in candidate.resources.items()
            }
            resources["disk"] = resources.get("disk", 0) + 2 * input_bytes
            active = [
                {
                    "resources": attempt.resources,
                    "observed": peer.telemetry.get("attempts", {}).get(str(attempt.pk), {}),
                }
                for attempt in peer.attempts.filter(state__in=LIVE_STATES)
            ]
            if admit(resources, active, peer.budgets, peer.telemetry, gpu=False).fits:
                return True
    return False
