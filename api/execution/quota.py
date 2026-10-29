from __future__ import annotations

import uuid
from contextvars import ContextVar

from django.db import transaction
from django.utils import timezone

from api.execution.models import QuotaDay, QuotaEntry

pending_reservation = ContextVar("pending_quota_reservation", default=None)


@transaction.atomic
def reserve(subject, amount, daily_limit):
    if amount < 0:
        raise ValueError("Negative quota reservation")
    day = timezone.now().date()
    account, _ = QuotaDay.objects.get_or_create(subject=subject, day=day)
    account = QuotaDay.objects.select_for_update().get(pk=account.pk)
    if account.used + amount > daily_limit:
        pending_reservation.set(None)
        return False, max(0, daily_limit - account.used)
    account.used += amount
    account.save(update_fields=["used"])
    entry = QuotaEntry.objects.create(
        account=account, amount=amount, reserved_amount=amount, key="reserve:" + uuid.uuid4().hex
    )
    pending_reservation.set(entry.pk)
    return True, daily_limit - account.used


def bind(job):
    ident = pending_reservation.get()
    if ident is not None:
        QuotaEntry.objects.filter(pk=ident, job__isnull=True).update(job=job)
        pending_reservation.set(None)


@transaction.atomic
def refund(job, amount):
    reservation = QuotaEntry.objects.filter(job=job, amount__gte=0).order_by("created_at").first()
    if not reservation:
        # Old reservations may have expired before migration; never credit today's bucket.
        return 0
    key = "refund:" + str(job.pk)
    account = QuotaDay.objects.select_for_update().get(pk=reservation.account_id)
    if QuotaEntry.objects.filter(key=key).exists():
        return 0
    maximum = (
        reservation.reserved_amount
        if reservation.reserved_amount is not None
        else reservation.amount
    )
    amount = max(0, min(int(amount), maximum, account.used))
    QuotaEntry.objects.create(key=key, account=account, job=job, amount=-amount)
    account.used -= amount
    account.save(update_fields=["used"])
    return amount
