# models.py
from django.db import models
from django.utils import timezone
import secrets
import string
import random


def generate_public_id(length=7):
    chars = string.ascii_letters + string.digits
    return "".join(random.choices(chars, k=length))


class ApiUser(models.Model):
    ip_address = models.GenericIPAddressField(unique=True)
    custom_daily_limit = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Custom daily limit for this IP. Leave blank to use default.",
    )
    is_blocked = models.BooleanField(default=False)
    first_seen = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    notes = models.TextField(blank=True, help_text="Admin notes about this user")

    class Meta:
        ordering = ["-last_seen"]
        verbose_name = "API User"
        verbose_name_plural = "API Users"

    def __str__(self):
        return f"{self.ip_address} ({'blocked' if self.is_blocked else 'active'})"

    @property
    def total_jobs(self):
        return self.job_set.count()

    @property
    def jobs_today(self):
        today = timezone.now().date()
        return self.job_set.filter(submission_time__date=today).count()

    @property
    def effective_daily_limit(self):
        from api.utils.quotas import DAILY_LIMIT

        return self.custom_daily_limit or DAILY_LIMIT


class Job(models.Model):
    job_id = models.AutoField(primary_key=True)
    public_id = models.CharField(max_length=10, unique=True)
    prediction_type = models.CharField(max_length=32)
    ip_address = models.CharField(max_length=45, blank=True, default="")  # IPv4/IPv6
    quota_subject = models.CharField(
        max_length=128,
        blank=True,
        default="",
        help_text="Identifier used for quota accounting (IP or API-key subject).",
    )
    requested_rows = models.PositiveIntegerField(default=0)
    kcat_method = models.CharField(max_length=50, null=True, blank=True)
    km_method = models.CharField(max_length=50, null=True, blank=True)
    kcat_km_method = models.CharField(max_length=50, null=True, blank=True)
    status = models.CharField(max_length=20)
    submission_time = models.DateTimeField(default=timezone.now)
    start_time = models.DateTimeField(null=True, blank=True)
    completion_time = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    output_file = models.FileField(upload_to="jobs/%Y/%m/%d/", null=True, blank=True)
    handle_long_sequences = models.CharField(
        max_length=100,
        default="truncate",
        choices=[
            ("truncate", "truncate"),
            ("skip", "skip"),
        ],
    )
    canonicalize_substrates = models.BooleanField(default=True)
    recon_xkg = models.BooleanField(
        default=False,
        help_text=(
            "Internal: this job was served via the ReconXKG memoization store "
            "(allowlisted keys only). Recorded for audit."
        ),
    )

    total_molecules = models.IntegerField(default=0)
    molecules_processed = models.IntegerField(default=0)
    invalid_rows = models.IntegerField(default=0)
    total_predictions = models.IntegerField(default=0)
    predictions_made = models.IntegerField(default=0)
    user = models.ForeignKey(ApiUser, on_delete=models.SET_NULL, null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.public_id:
            while True:
                pid = generate_public_id()
                if not Job.objects.filter(public_id=pid).exists():
                    self.public_id = pid
                    break
        super().save(*args, **kwargs)


class JobProgressStage(models.Model):
    STATUS_CHOICES = [
        ("pending", "pending"),
        ("running", "running"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("skipped", "skipped"),
    ]

    EMBEDDING_STATE_CHOICES = [
        ("", ""),
        ("not_required", "not_required"),
        ("pending", "pending"),
        ("running", "running"),
        ("done", "done"),
        ("error", "error"),
    ]

    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="progress_stages")
    stage_index = models.PositiveIntegerField()
    target = models.CharField(max_length=32)
    method_key = models.CharField(max_length=50)
    method_display_name = models.CharField(max_length=100, blank=True, default="")

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    message = models.TextField(blank=True, default="")

    molecules_total = models.IntegerField(default=0)
    molecules_processed = models.IntegerField(default=0)
    invalid_rows = models.IntegerField(default=0)
    predictions_total = models.IntegerField(default=0)
    predictions_made = models.IntegerField(default=0)

    embedding_enabled = models.BooleanField(default=False)
    embedding_state = models.CharField(
        max_length=20,
        choices=EMBEDDING_STATE_CHOICES,
        default="",
        blank=True,
    )
    embedding_method_key = models.CharField(max_length=50, blank=True, default="")
    embedding_target = models.CharField(max_length=32, blank=True, default="")
    embedding_total = models.IntegerField(default=0)
    embedding_cached_already = models.IntegerField(default=0)
    embedding_need_computation = models.IntegerField(default=0)
    embedding_computed = models.IntegerField(default=0)
    embedding_remaining = models.IntegerField(default=0)

    class Meta:
        ordering = ["job_id", "stage_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["job", "stage_index"],
                name="api_jobprogressstage_unique_job_stage_index",
            ),
            models.UniqueConstraint(
                fields=["job", "target"],
                name="api_jobprogressstage_unique_job_target",
            ),
        ]

    def __str__(self):
        return f"{self.job.public_id}#{self.stage_index}:{self.target} ({self.status})"


def generate_api_key():
    """Generate a cryptographically secure API key with an 'ak_' prefix."""
    return "ak_" + secrets.token_hex(32)  # 67 chars total, 256 bits of entropy


class ApiKey(models.Model):
    """
    A long-lived API key that grants programmatic access to the prediction API.

    Keys are tied to an ApiUser (identified by IP address). Quota/blocking
    policy is inherited from the owning ApiUser, while usage counters are
    tracked by API key subject.

    The full key is only returned once at creation time (via the management
    command). Afterwards, only the first 10 characters are surfaced in the admin
    so that accidental exposure is minimised.
    """

    key = models.CharField(
        max_length=67,
        unique=True,
        default=generate_api_key,
        db_index=True,
        help_text="The secret token sent in the Authorization: Bearer header.",
    )
    user = models.OneToOneField(
        ApiUser,
        on_delete=models.CASCADE,
        related_name="api_key",
        help_text="The API user (IP address) this key belongs to.",
    )
    label = models.CharField(
        max_length=100,
        blank=True,
        help_text="A human-readable name for this key, e.g. 'Lab Python Script'.",
    )
    custom_daily_limit = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text=(
            "Optional per-key daily limit override. "
            "Effective limit is max(user limit, key limit)."
        ),
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Revoke a key by setting this to False.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp of the most recent authenticated request.",
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"

    def __str__(self):
        status = "active" if self.is_active else "revoked"
        label = self.label or "Unnamed"
        return f"{label} ({self.key[:10]}…) [{status}]"

    @property
    def key_prefix(self):
        """Returns only the first 10 characters for safe display in the admin."""
        return self.key[:10] + "…"

    @property
    def effective_daily_limit(self):
        """
        Effective daily limit for authenticated requests using this key.

        If the owning user is blocked, effective limit is 0 regardless of key.
        Otherwise use the higher of the user's IP limit and this key's limit.
        """
        if self.user.is_blocked:
            return 0
        return max(self.user.effective_daily_limit, self.custom_daily_limit or 0)


class AboutStatsCache(models.Model):
    """
    Persistent cache for the About-page metrics payload.

    A singleton-style row keyed by ``key='about_stats'`` is used by
    ``api.services.about_stats_service``.
    """

    key = models.CharField(max_length=64, unique=True)
    payload = models.TextField(blank=True, default="")
    generated_at = models.DateTimeField(null=True, blank=True)
    is_stale = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "About Stats Cache"
        verbose_name_plural = "About Stats Cache"

    def __str__(self):
        state = "stale" if self.is_stale else "fresh"
        return f"{self.key} [{state}]"


class ReconXkgAllowedKey(models.Model):
    """
    Allowlist entry permitting one API key to activate the (undocumented)
    ``recon_xkg`` submit parameter.

    Membership in this table is the *only* thing that authorizes ReconXKG mode;
    keys that are not present (or whose entry is inactive) silently fall back to
    a normal job. The raw key is never stored here — only a reference to the
    existing :class:`ApiKey` row.
    """

    api_key = models.OneToOneField(
        ApiKey,
        on_delete=models.CASCADE,
        related_name="recon_xkg_allow",
        help_text="API key permitted to enable recon_xkg.",
    )
    label = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="Optional human-readable note for this allowlist entry.",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Disable without deleting by unchecking this.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "ReconXKG Allowed Key"
        verbose_name_plural = "ReconXKG Allowed Keys"

    def __str__(self):
        state = "active" if self.is_active else "inactive"
        return f"recon_xkg:{self.api_key.key_prefix} [{state}]"


class PredictionStore(models.Model):
    """
    Append/upsert memoization of a single raw model prediction unit.

    One row caches either the raw predicted value or a deterministic row-level
    validation failure for one prediction unit. A unit is (post-truncation
    sequence, canonical substrate(s), canonical products, target, method, model
    version, params fingerprint). Successful values are stored before RealKcat
    formatting, substrate reduction, or experimental overrides so downstream
    assembly is identical to a freshly computed row.

    Routed to the dedicated ``prediction_store`` database (see
    api/dbrouters.py). ``lookup_key`` is a SHA-256 hex digest over all
    prediction-affecting fields and is the only column on the hot lookup path.
    """

    prediction_store_db = True

    lookup_key = models.CharField(max_length=64, unique=True)
    target = models.CharField(max_length=16)
    method = models.CharField(max_length=64)
    model_version = models.CharField(max_length=32)
    params_fingerprint = models.CharField(max_length=64)
    sequence_sha256 = models.CharField(max_length=64)
    substrate_canon = models.TextField()
    products_canon = models.TextField(blank=True, default="")
    value = models.FloatField(null=True, blank=True)
    failure_reason = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        verbose_name = "Prediction Store Entry"
        verbose_name_plural = "Prediction Store Entries"
        indexes = [
            models.Index(fields=["model_version"], name="predstore_modelver_idx"),
            models.Index(fields=["method", "target"], name="predstore_method_tgt_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(value__isnull=False, failure_reason="")
                    | (models.Q(value__isnull=True) & ~models.Q(failure_reason=""))
                ),
                name="predstore_value_or_failure",
            )
        ]

    def __str__(self):
        return f"{self.method}/{self.target}@{self.model_version}:{self.lookup_key[:12]}"


class SimilarityStore(models.Model):
    """
    Memoization of per-sequence MMseqs2 similarity to a method's training set.

    Similarity is a property of (sequence, training dataset) and is independent
    of substrates or the kinetic value, so it is cached separately. This lets a
    fully cached kcat job skip MMseqs2 entirely. ``NULL`` mean/max values are a
    negative cache hit and render as blank output cells. Routed to
    ``prediction_store``.
    """

    prediction_store_db = True

    lookup_key = models.CharField(max_length=64, unique=True)
    sequence_sha256 = models.CharField(max_length=64, db_index=True)
    dataset_label = models.CharField(max_length=128)
    mean_similarity = models.FloatField(null=True, blank=True)
    max_similarity = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        verbose_name = "Similarity Store Entry"
        verbose_name_plural = "Similarity Store Entries"

    def __str__(self):
        return f"sim:{self.dataset_label}:{self.lookup_key[:12]}"
