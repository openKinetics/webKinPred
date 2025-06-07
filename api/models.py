# models.py
from django.db import models
import uuid
from django.utils import timezone
import string, random

def generate_public_id(length=7):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

class Job(models.Model):
    job_id = models.AutoField(primary_key=True)
    public_id = models.CharField(max_length=10, unique=True)
    prediction_type = models.CharField(max_length=10)
    kcat_method = models.CharField(max_length=50, null=True, blank=True)
    km_method = models.CharField(max_length=50, null=True, blank=True)
    status = models.CharField(max_length=20)
    submission_time = models.DateTimeField(default=timezone.now)
    completion_time = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    output_file = models.FileField(upload_to='jobs/%Y/%m/%d/', null=True, blank=True)
    handle_long_sequences = models.CharField(max_length=100, default='truncate', choices=[
        ('truncate', 'truncate'),
        ('skip', 'skip'),
    ])
    # New fields for progress tracking
    total_molecules = models.IntegerField(default=0)
    molecules_processed = models.IntegerField(default=0)
    invalid_molecules = models.IntegerField(default=0)
    total_predictions = models.IntegerField(default=0)
    predictions_made = models.IntegerField(default=0)

    def save(self, *args, **kwargs):
        if not self.public_id:
            while True:
                pid = generate_public_id()
                if not Job.objects.filter(public_id=pid).exists():
                    self.public_id = pid
                    break
        super().save(*args, **kwargs)

