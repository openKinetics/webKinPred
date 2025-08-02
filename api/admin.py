# api/admin.py
from django.contrib import admin
from .models import Job
from .seqmap_models import Sequence

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = (
        'job_id', 'public_id', 'prediction_type', 'kcat_method', 'km_method', 
        'status', 'submission_time', 'completion_time')
    list_filter = ('status', 'prediction_type', 'kcat_method', 'km_method')
    search_fields = ('public_id', 'error_message')
    readonly_fields = ('submission_time', 'completion_time', 'public_id')
    ordering = ('-submission_time',)

@admin.register(Sequence)
class SequenceAdmin(admin.ModelAdmin):
    list_display    = ("id", "len", "uses_count", "last_seen_at")
    search_fields   = ("id", "seq", "sha256")
    readonly_fields = [f.name for f in Sequence._meta.fields]
