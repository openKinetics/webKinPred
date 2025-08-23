# api/admin.py
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from .models import Job, ApiUser
from api.utils.quotas import get_quota_usage
from db_models.seqmap_models import Sequence


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

    def quota_status(self, obj):
        usage = get_quota_usage(obj.ip_address)
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

    quota_status.short_description = "Today's Usage"

    def quota_info(self, obj):
        if obj.pk:
            usage = get_quota_usage(obj.ip_address)
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

    quota_info.short_description = "Live Quota Status"

    def job_summary(self, obj):
        if obj.pk:
            total_jobs = obj.total_jobs
            jobs_today = obj.jobs_today
            recent_jobs = obj.job_set.order_by("-submission_time")[:5]

            html = (
                f'<div style="background: #f8f8f8; padding: 10px; border-radius: 4px;">'
            )
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

    job_summary.short_description = "Job Summary"

    actions = ["block_users", "unblock_users", "reset_quotas"]

    def block_users(self, request, queryset):
        count = queryset.update(is_blocked=True)
        self.message_user(request, f"{count} users blocked.")

    block_users.short_description = "Block selected users"

    def unblock_users(self, request, queryset):
        count = queryset.update(is_blocked=False)
        self.message_user(request, f"{count} users unblocked.")

    unblock_users.short_description = "Unblock selected users"

    def reset_quotas(self, request, queryset):
        from django_redis import get_redis_connection
        from api.utils.quotas import _key

        r = get_redis_connection("default")
        count = 0
        for user in queryset:
            key = _key(user.ip_address)
            if r.delete(key):
                count += 1

        self.message_user(request, f"Reset quotas for {count} users.")

    reset_quotas.short_description = "Reset today's quota for selected users"


# Update existing JobAdmin to use existing download URLs for both input and output
@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = [
        "public_id",
        "user_ip",
        "prediction_type",
        "status",
        "submission_time",
        "requested_rows",
        "download_links",
    ]
    list_filter = ["status", "prediction_type", "submission_time"]
    search_fields = ["public_id", "ip_address", "user__ip_address"]
    readonly_fields = ["public_id", "submission_time", "download_links"]

    def user_ip(self, obj):
        if obj.user:
            user_url = reverse("admin:api_apiuser_change", args=[obj.user.pk])
            return format_html('<a href="{}">{}</a>', user_url, obj.user.ip_address)
        return obj.ip_address

    user_ip.short_description = "User IP"

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

    download_links.short_description = "Downloads"


@admin.register(Sequence)
class SequenceAdmin(admin.ModelAdmin):
    list_display = ("id", "len", "uses_count", "last_seen_at")
    search_fields = ("id", "seq", "sha256")
    readonly_fields = [f.name for f in Sequence._meta.fields]
