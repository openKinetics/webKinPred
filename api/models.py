# models.py
from django.db import models
from django.utils import timezone
import string, random


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
    prediction_type = models.CharField(max_length=10)
    ip_address = models.CharField(max_length=45, blank=True, default="")  # IPv4/IPv6
    requested_rows = models.PositiveIntegerField(default=0)
    kcat_method = models.CharField(max_length=50, null=True, blank=True)
    km_method = models.CharField(max_length=50, null=True, blank=True)
    status = models.CharField(max_length=20)
    submission_time = models.DateTimeField(default=timezone.now)
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

    total_molecules = models.IntegerField(default=0)
    molecules_processed = models.IntegerField(default=0)
    invalid_molecules = models.IntegerField(default=0)
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
