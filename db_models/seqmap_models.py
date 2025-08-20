# api/seqmap_models.py
from django.db import models

class Sequence(models.Model):
    id           = models.CharField(primary_key=True, max_length=255)
    seq          = models.TextField()
    sha256       = models.CharField(max_length=64)
    len          = models.PositiveIntegerField()
    created_at   = models.DateTimeField()
    last_seen_at = models.DateTimeField()
    uses_count   = models.PositiveIntegerField()

    class Meta:
        managed  = False
        db_table = "sequences"
        verbose_name_plural = "Sequences"
        app_label = 'seqmap'

    # flag for the router (outside Meta)
    seqmap_db = True
