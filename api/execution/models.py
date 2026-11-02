from __future__ import annotations

import uuid

from django.db import models
from django.utils import timezone


class ExecutionSpec(models.Model):
    job = models.OneToOneField(
        "api.Job", null=True, on_delete=models.PROTECT, related_name="execution"
    )
    specification = models.JSONField()
    checksum = models.CharField(max_length=64)
    input_checksum = models.CharField(max_length=64)
    created_at = models.DateTimeField(default=timezone.now)
    paused = models.BooleanField(default=True)
    priority = models.PositiveIntegerField(default=0, db_default=0, db_index=True)
    background = models.BooleanField(default=False, db_default=False, db_index=True)
    policy_version = models.CharField(max_length=32, default="v2-gpu-only")


class SchedulingRule(models.Model):
    kind = models.CharField(
        max_length=20, choices=[("ip", "IP address"), ("api_key_prefix", "API key prefix")]
    )
    value = models.CharField(max_length=128)
    enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["kind", "value"], name="scheduling_rule_selector")
        ]


class ExecutorHost(models.Model):
    id = models.SlugField(primary_key=True, max_length=64)
    capabilities = models.JSONField(default=dict)
    budgets = models.JSONField(default=dict)
    telemetry = models.JSONField(default=dict)
    heartbeat_at = models.DateTimeField(null=True)
    paused = models.BooleanField(default=True)
    reconciled = models.BooleanField(default=False)
    gpu_paused = models.BooleanField(default=False, db_default=False)


class EmbeddingSubmission(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    request_key = models.CharField(max_length=64, unique=True)
    payload = models.JSONField()
    execution = models.OneToOneField(ExecutionSpec, on_delete=models.PROTECT)
    created_at = models.DateTimeField(default=timezone.now)


class ResourceProfile(models.Model):
    id = models.CharField(primary_key=True, max_length=128)
    fingerprint = models.CharField(max_length=64)
    image = models.CharField(max_length=256)
    stage_kind = models.CharField(max_length=32)
    method = models.CharField(max_length=64, blank=True)
    settings = models.JSONField(default=dict)
    resources = models.JSONField(default=dict)
    runtime_floor = models.JSONField(default=dict)
    limits = models.JSONField(default=dict)
    evidence = models.JSONField(default=dict)
    validated = models.BooleanField(default=False)
    suspended = models.BooleanField(default=False)


class WorkStage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    execution = models.ForeignKey(ExecutionSpec, on_delete=models.PROTECT, related_name="stages")
    priority = models.PositiveIntegerField(default=0, db_default=0, db_index=True)
    key = models.CharField(max_length=160)
    oldest_consumer_at = models.DateTimeField(null=True)
    kind = models.CharField(max_length=32)
    method = models.CharField(max_length=64, blank=True)
    target = models.CharField(max_length=16, blank=True)
    payload = models.JSONField(default=dict)
    dependencies = models.ManyToManyField("self", symmetrical=False, related_name="dependants")
    state = models.CharField(max_length=32, default="ready", db_index=True)
    waiting_reason = models.CharField(max_length=128, blank=True)
    profile = models.ForeignKey(ResourceProfile, null=True, on_delete=models.PROTECT)
    available_at = models.DateTimeField(default=timezone.now, db_index=True)
    checkpoint = models.JSONField(default=dict)
    progress = models.PositiveBigIntegerField(default=0)
    failures = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["execution", "key"], name="execution_stage_key")
        ]
        indexes = [
            models.Index(
                fields=["profile", "available_at"],
                condition=models.Q(state="ready"),
                name="stage_ready_profile_time",
            ),
            models.Index(fields=["completed_at"], name="stage_completed_time"),
        ]


class RecoveryBatch(models.Model):
    """Current recovery generation for an immutable producer input context."""

    id = models.CharField(primary_key=True, max_length=64)
    stage = models.OneToOneField(WorkStage, null=True, on_delete=models.PROTECT)
    source = models.ForeignKey(WorkStage, on_delete=models.PROTECT, related_name="recovery_batches")
    generation = models.PositiveIntegerField(default=0)


class ExecutionAttempt(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    stage = models.ForeignKey(WorkStage, on_delete=models.PROTECT, related_name="attempts")
    host = models.ForeignKey(ExecutorHost, on_delete=models.PROTECT, related_name="attempts")
    token = models.UUIDField(default=uuid.uuid4)
    state = models.CharField(max_length=32, default="leased", db_index=True)
    resources = models.JSONField(default=dict)
    profile_fingerprint = models.CharField(max_length=64)
    container_id = models.CharField(max_length=128, blank=True)
    stop_requested = models.BooleanField(default=False)
    started_at = models.DateTimeField(default=timezone.now)
    heartbeat_at = models.DateTimeField(default=timezone.now)
    lease_until = models.DateTimeField()
    finished_at = models.DateTimeField(null=True)
    scratch_released_at = models.DateTimeField(null=True)
    outcome = models.JSONField(default=dict)

    class Meta:
        indexes = [
            models.Index(models.F("outcome__category"), name="attempt_outcome_category"),
            models.Index(fields=["finished_at"], name="attempt_finished_time"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["stage"],
                condition=models.Q(state__in=["leased", "running", "uncertain"]),
                name="one_live_stage_attempt",
            )
        ]


class Artifact(models.Model):
    id = models.CharField(primary_key=True, max_length=64)
    identity = models.JSONField()
    checksum = models.CharField(max_length=64, db_index=True)
    size = models.PositiveBigIntegerField()
    metadata = models.JSONField(default=dict)
    regenerable = models.BooleanField(default=True)
    # Unclassified/legacy files are retained until provenance is established.
    retention = models.CharField(max_length=16, default="protected", db_default="protected")
    producer = models.ForeignKey(
        WorkStage, null=True, on_delete=models.PROTECT, related_name="outputs"
    )

    class Meta:
        indexes = [
            models.Index(
                fields=["retention", "checksum"],
                condition=models.Q(retention__in=["transient", "cache"]),
                name="artifact_collectible_policy",
            )
        ]


class ArtifactLocation(models.Model):
    artifact = models.ForeignKey(Artifact, on_delete=models.PROTECT, related_name="locations")
    host = models.ForeignKey(ExecutorHost, on_delete=models.PROTECT)
    verified_at = models.DateTimeField(default=timezone.now)
    ready = models.BooleanField(default=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["artifact", "host"], name="artifact_host_location")
        ]


class ArtifactDependency(models.Model):
    artifact = models.ForeignKey(Artifact, on_delete=models.PROTECT, related_name="dependencies")
    stage = models.ForeignKey(WorkStage, on_delete=models.PROTECT, related_name="artifacts")
    path = models.CharField(max_length=512)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["stage", "path"], name="stage_artifact_path")
        ]


class ArtifactSweep(models.Model):
    host = models.OneToOneField(ExecutorHost, primary_key=True, on_delete=models.PROTECT)
    cursor = models.PositiveBigIntegerField(default=0)
    pending = models.JSONField(default=list)


class ArtifactCleanup(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    host = models.ForeignKey(ExecutorHost, on_delete=models.PROTECT)
    token = models.UUIDField(default=uuid.uuid4)
    blobs = models.JSONField(default=list)
    attempt_ids = models.JSONField(default=list, db_default=[])
    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["host"],
                condition=models.Q(completed_at__isnull=True),
                name="one_pending_artifact_cleanup",
            )
        ]


class ExecutionOutbox(models.Model):
    key = models.CharField(max_length=192, unique=True)
    event = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)
    published_at = models.DateTimeField(null=True)


class QuotaDay(models.Model):
    subject = models.CharField(max_length=128)
    day = models.DateField()
    used = models.PositiveBigIntegerField(default=0)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["subject", "day"], name="quota_subject_day")]


class QuotaEntry(models.Model):
    key = models.CharField(max_length=192, unique=True)
    account = models.ForeignKey(QuotaDay, on_delete=models.PROTECT)
    job = models.ForeignKey("api.Job", null=True, on_delete=models.PROTECT)
    amount = models.BigIntegerField()
    reserved_amount = models.PositiveBigIntegerField(null=True)
    created_at = models.DateTimeField(default=timezone.now)


class SequenceIdentity(models.Model):
    id = models.CharField(primary_key=True, max_length=255)
    sha256 = models.CharField(max_length=64, unique=True)
    # PostgreSQL cannot put arbitrarily long sequences in a btree unique index.
    # The unique SHA-256 plus explicit collision verification is the identity key.
    seq = models.TextField()
    length = models.PositiveIntegerField(db_column="len")
    uses_count = models.PositiveBigIntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    last_seen_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "sequences"
