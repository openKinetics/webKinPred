from db_models.seqmap_models import Sequence
from django import forms
from django.contrib import admin
from django.urls import reverse
from django.utils import timezone
from django.utils.html import format_html

from api.utils.quotas import (
    _key,
    get_all_user_quota_subjects,
    get_quota_usage,
    get_user_quota_daily_limit,
    get_user_quota_subject,
)

from .models import (
    ApiKey,
    ApiUser,
    Job,
    JobProgressStage,
    PredictionStore,
    ReconXkgAllowedKey,
    SimilarityStore,
)


def _format_duration(total_seconds):
    """Format a duration as a compact, readable value for the admin list."""
    total_seconds = max(0, int(total_seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


@admin.register(ApiUser)
class ApiUserAdmin(admin.ModelAdmin):
    list_display = [
        "ip_address",
        "quota_status",
        "effective_daily_limit",
        "total_jobs",
        "jobs_today",
        "last_seen",
        "is_blocked",
    ]
    list_filter = ["is_blocked", "first_seen", "last_seen"]
    search_fields = ["ip_address", "notes"]
    readonly_fields = ["first_seen", "last_seen", "quota_info", "job_summary"]
    fieldsets = (
        ("User Information", {"fields": ("ip_address", "first_seen", "last_seen")}),
        (
            "Quota Management",
            {"fields": ("custom_daily_limit", "is_blocked", "quota_info")},
        ),
        ("Job Summary", {"fields": ("job_summary",)}),
        ("Admin Notes", {"fields": ("notes",)}),
    )

    @admin.display(description="Today's Usage")
    def quota_status(self, obj):
        usage = get_quota_usage(
            get_user_quota_subject(obj),
            daily_limit=get_user_quota_daily_limit(obj),
        )
        used = usage["used"]
        limit = usage["limit"]
        remaining = usage["remaining"]

        if remaining == 0:
            color = "red"
        elif remaining < limit * 0.1:  # Less than 10% remaining
            color = "orange"
        else:
            color = "green"

        return format_html(
            '<span style="color: {};">{}/{} ({} left)</span>',
            color,
            used,
            limit,
            remaining,
        )

    @admin.display(description="Live Quota Status")
    def quota_info(self, obj):
        if obj.pk:
            usage = get_quota_usage(
                get_user_quota_subject(obj),
                daily_limit=get_user_quota_daily_limit(obj),
            )
            reset_hours = usage["reset_in_seconds"] // 3600
            reset_minutes = (usage["reset_in_seconds"] % 3600) // 60

            return format_html(
                '<div style="background: #f8f8f8; padding: 10px; border-radius: 4px;">'
                "<strong>Current Usage:</strong> {used}/{limit}<br>"
                "<strong>Remaining:</strong> {remaining}<br>"
                "<strong>Resets in:</strong> {hours}h {minutes}m<br>"
                "<strong>Status:</strong> {status}"
                "</div>",
                used=usage["used"],
                limit=usage["limit"],
                remaining=usage["remaining"],
                hours=reset_hours,
                minutes=reset_minutes,
                status="Blocked" if obj.is_blocked else "Active",
            )
        return "Save user first to see quota information"

    @admin.display(description="Job Summary")
    def job_summary(self, obj):
        if obj.pk:
            total_jobs = obj.total_jobs
            jobs_today = obj.jobs_today
            recent_jobs = obj.job_set.order_by("-submission_time")[:5]

            html = f'<div style="background: #f8f8f8; padding: 10px; border-radius: 4px;">'
            html += f"<strong>Total Jobs:</strong> {total_jobs}<br>"
            html += f"<strong>Jobs Today:</strong> {jobs_today}<br>"

            if recent_jobs:
                html += "<br><strong>Recent Jobs:</strong><ul>"
                for job in recent_jobs:
                    job_url = reverse("admin:api_job_change", args=[job.pk])
                    html += f'<li><a href="{job_url}">{job.public_id}</a> - {job.status} ({job.submission_time.strftime("%Y-%m-%d %H:%M")})</li>'
                html += "</ul>"

            html += "</div>"
            return format_html(html)
        return "Save user first to see job summary"

    actions = ["block_users", "unblock_users", "reset_quotas"]

    @admin.action(description="Block selected users")
    def block_users(self, request, queryset):
        count = queryset.update(is_blocked=True)
        self.message_user(request, f"{count} users blocked.")

    @admin.action(description="Unblock selected users")
    def unblock_users(self, request, queryset):
        count = queryset.update(is_blocked=False)
        self.message_user(request, f"{count} users unblocked.")

    @admin.action(description="Reset today's quota for selected users")
    def reset_quotas(self, request, queryset):
        from django_redis import get_redis_connection

        r = get_redis_connection("default")
        count = 0
        for user in queryset:
            for subject in get_all_user_quota_subjects(user):
                key = _key(subject)
                if r.delete(key):
                    count += 1

        self.message_user(request, f"Reset {count} quota counter(s).")


# Update existing JobAdmin to use existing download URLs for both input and output
@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = [
        "public_id",
        "user_ip",
        "prediction_type",
        "kcat_method",
        "km_method",
        "kcat_km_method_display",
        "canonicalize_substrates",
        "status",
        "elapsed_time",
        "time_in_queue",
        "submission_time",
        "requested_rows",
        "download_links",
    ]
    list_filter = [
        "status",
        "prediction_type",
        "kcat_method",
        "km_method",
        "kcat_km_method",
        "canonicalize_substrates",
        "submission_time",
    ]
    search_fields = ["public_id", "ip_address", "user__ip_address"]
    readonly_fields = ["public_id", "submission_time", "download_links"]

    @admin.display(description="User IP")
    def user_ip(self, obj):
        if obj.user:
            user_url = reverse("admin:api_apiuser_change", args=[obj.user.pk])
            return format_html('<a href="{}">{}</a>', user_url, obj.user.ip_address)
        return obj.ip_address

    @admin.display(description="kcat/Km method", ordering="kcat_km_method")
    def kcat_km_method_display(self, obj):
        return obj.kcat_km_method

    @admin.display(description="Time elapsed")
    def elapsed_time(self, obj):
        if obj.status not in {"Processing", "Completed"} or not obj.start_time:
            return "—"

        end_time = obj.completion_time if obj.status == "Completed" else timezone.now()
        if end_time is None:
            return "—"
        return _format_duration((end_time - obj.start_time).total_seconds())

    @admin.display(description="Time in queue")
    def time_in_queue(self, obj):
        if obj.status != "Pending":
            return "—"
        return _format_duration((timezone.now() - obj.submission_time).total_seconds())

    @admin.display(description="Downloads")
    def download_links(self, obj):
        links = []

        # Always show input download link
        input_url = reverse("download_job_input", args=[obj.public_id])
        links.append(f'<a href="{input_url}" class="button">Download Input</a>')

        # Show output download link if job is completed and has output
        if obj.status == "Completed" and obj.output_file:
            output_url = reverse("download_job_output", args=[obj.public_id])
            links.append(f'<a href="{output_url}" class="button">Download Results</a>')
        elif obj.status == "Completed":
            links.append('<span style="color: #666;">No output file</span>')
        else:
            links.append('<span style="color: #666;">Job not completed</span>')

        return format_html(" | ".join(links))


@admin.register(JobProgressStage)
class JobProgressStageAdmin(admin.ModelAdmin):
    list_display = [
        "job",
        "stage_index",
        "target",
        "method_key",
        "status",
        "molecules_processed",
        "molecules_total",
        "predictions_made",
        "predictions_total",
        "embedding_state",
        "updated_at",
    ]
    list_filter = ["status", "target", "method_key", "embedding_state", "updated_at"]
    search_fields = ["job__public_id", "target", "method_key", "method_display_name"]
    readonly_fields = ["updated_at"]


class ApiKeyAdminForm(forms.ModelForm):
    """ApiKey form with an inline ReconXKG allowlist toggle."""

    recon_xkg_enabled = forms.BooleanField(
        required=False,
        label="ReconXKG enabled",
        help_text=(
            "Allow this key to use the (undocumented) recon_xkg cached-prediction "
            "mode. Changes take effect within ~30s."
        ),
    )

    class Meta:
        model = ApiKey
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            self.fields["recon_xkg_enabled"].initial = ReconXkgAllowedKey.objects.filter(
                api_key=self.instance, is_active=True
            ).exists()


@admin.register(ApiKey)
class ApiKeyAdmin(admin.ModelAdmin):
    """
    Admin interface for API keys.

    The full key is never displayed here — only the first 10 characters are
    shown so that accidental screen-sharing cannot leak credentials.  The full
    key is printed exactly once when it is created via the management command
    ``python manage.py create_api_key``.
    """

    form = ApiKeyAdminForm

    list_display = [
        "key_prefix",
        "label",
        "user_ip",
        "custom_daily_limit",
        "effective_daily_limit_display",
        "is_active",
        "recon_xkg_status",
        "created_at",
        "last_used",
    ]
    list_filter = ["is_active", "created_at"]
    search_fields = ["label", "user__ip_address"]
    readonly_fields = ["key_prefix", "effective_daily_limit_display", "created_at", "last_used"]
    ordering = ["-created_at"]

    fieldsets = (
        (
            "Key Details",
            {
                "fields": (
                    "key_prefix",
                    "label",
                    "is_active",
                    "custom_daily_limit",
                    "effective_daily_limit_display",
                ),
                "description": (
                    "The full key is shown only once at creation time "
                    "(via the create_api_key management command). "
                    "To revoke a key, set 'Active' to false."
                ),
            },
        ),
        ("ReconXKG", {"fields": ("recon_xkg_enabled",)}),
        ("Ownership", {"fields": ("user",)}),
        ("Timestamps", {"fields": ("created_at", "last_used")}),
    )

    actions = ["revoke_keys", "activate_keys", "enable_recon_xkg", "disable_recon_xkg"]

    @admin.display(description="User IP")
    def user_ip(self, obj):
        user_url = reverse("admin:api_apiuser_change", args=[obj.user.pk])
        return format_html('<a href="{}">{}</a>', user_url, obj.user.ip_address)

    @admin.display(description="Effective Daily Limit")
    def effective_daily_limit_display(self, obj):
        return obj.effective_daily_limit

    @admin.display(description="ReconXKG", boolean=True)
    def recon_xkg_status(self, obj):
        return ReconXkgAllowedKey.objects.filter(api_key=obj, is_active=True).exists()

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        self._set_recon_xkg(obj, form.cleaned_data.get("recon_xkg_enabled", False))

    @staticmethod
    def _set_recon_xkg(api_key, enabled: bool) -> None:
        if enabled:
            ReconXkgAllowedKey.objects.update_or_create(
                api_key=api_key, defaults={"is_active": True}
            )
        else:
            ReconXkgAllowedKey.objects.filter(api_key=api_key).update(is_active=False)
        # Drop the short-lived in-process allowlist cache so the change is
        # reflected immediately on the next submission.
        try:
            from api.utils.recon_xkg import _allow_cache

            _allow_cache.pop(api_key.pk, None)
        except Exception:
            pass

    @admin.action(description="Revoke selected API keys")
    def revoke_keys(self, request, queryset):
        count = queryset.update(is_active=False)
        self.message_user(request, f"{count} API key(s) revoked.")

    @admin.action(description="Re-activate selected API keys")
    def activate_keys(self, request, queryset):
        count = queryset.update(is_active=True)
        self.message_user(request, f"{count} API key(s) re-activated.")

    @admin.action(description="Enable ReconXKG for selected keys")
    def enable_recon_xkg(self, request, queryset):
        for api_key in queryset:
            self._set_recon_xkg(api_key, True)
        self.message_user(request, f"ReconXKG enabled for {queryset.count()} key(s).")

    @admin.action(description="Disable ReconXKG for selected keys")
    def disable_recon_xkg(self, request, queryset):
        for api_key in queryset:
            self._set_recon_xkg(api_key, False)
        self.message_user(request, f"ReconXKG disabled for {queryset.count()} key(s).")


@admin.register(Sequence)
class SequenceAdmin(admin.ModelAdmin):
    list_display = ("id", "len", "uses_count", "last_seen_at")
    search_fields = ("id", "seq", "sha256")
    readonly_fields = [f.name for f in Sequence._meta.fields]


@admin.register(ReconXkgAllowedKey)
class ReconXkgAllowedKeyAdmin(admin.ModelAdmin):
    """Allowlist of API keys permitted to enable recon_xkg."""

    list_display = ["api_key", "label", "is_active", "created_at"]
    list_filter = ["is_active", "created_at"]
    search_fields = ["api_key__label", "api_key__user__ip_address", "label"]
    readonly_fields = ["created_at"]
    autocomplete_fields = ["api_key"]


@admin.register(PredictionStore)
class PredictionStoreAdmin(admin.ModelAdmin):
    list_display = [
        "lookup_key",
        "method",
        "target",
        "model_version",
        "value",
        "updated_at",
    ]
    list_filter = ["method", "target", "model_version"]
    search_fields = ["lookup_key", "sequence_sha256"]
    readonly_fields = [f.name for f in PredictionStore._meta.fields]


@admin.register(SimilarityStore)
class SimilarityStoreAdmin(admin.ModelAdmin):
    list_display = ["lookup_key", "dataset_label", "mean_similarity", "max_similarity", "updated_at"]
    list_filter = ["dataset_label"]
    search_fields = ["lookup_key", "sequence_sha256"]
    readonly_fields = [f.name for f in SimilarityStore._meta.fields]
