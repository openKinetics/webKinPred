"""Re-enable GPU cohort profiles suspended by shared-GPU OOMs and let their split pieces merge again.

Run in the backend shell (26 November 2026). Each profile below was suspended by one OOM
after 200-2,209 clean cohort runs, every one while the user's research training held GPU
memory (11, 13 and 24 November). That training ended before 02:14 on 26 November. Cohorts
reserve no more VRAM than the single-protein attempts they replace (CatPred 4.03 GB vs 5.15 GB).

Splitting marks every piece do_not_coalesce (reopened originals and new partitions alike),
so the pieces would run one protein per attempt
(about 20 s each, mostly model load) even with the cohort profile restored. Untried pieces
of the restored profiles get that flag cleared; tried or failed pieces keep it.
"""
import json

from django.db import transaction
from django.db.models import Exists, OuterRef
from django.utils import timezone

from api.execution.models import ExecutionAttempt, ResourceProfile, WorkStage

COHORTS = [
    "gpu76-cohort-gpu16-CatPred-catpred_embed-kcat",
    "gpu76-cohort-gpu16-CatPred-catpred_embed-Km",
    "gpu76-cohort-gpu16-CataPro-prot_t5_mean",
    "gpu76-cohort-gpu16-UniKP-prot_t5_mean",
    "gpu68-cohort-gpu16-KinForm-H-kinform_t5_residues",
]
REASON = (
    "Suspended by a single OOM while research training held GPU memory on the shared host; "
    "re-enabled 2026-11-26 after that training ended."
)

with transaction.atomic():
    restored, sources = [], set()
    for profile in ResourceProfile.objects.select_for_update().filter(pk__in=COHORTS):
        sources.update(profile.settings.get("cohort_sources", []))
        if not profile.suspended:
            continue
        profile.suspended = False
        profile.evidence.setdefault("restorations", []).append(
            {"at": timezone.now().isoformat(), "reason": REASON}
        )
        profile.save(update_fields=["suspended", "evidence"])
        restored.append(profile.pk)

    # cohort_sources lists the single-protein profiles' fingerprints, not their IDs.
    sources = set(
        ResourceProfile.objects.filter(fingerprint__in=sources).values_list("pk", flat=True)
    )
    pieces = (
        WorkStage.objects.select_for_update(skip_locked=True)
        .filter(
            profile_id__in=sources,
            state="ready",
            failures=0,
            # Splitting reopens the cohort's original stages when it can, so pieces
            # keep their embedding:<...> keys; only new partitions get split:<...>.
            payload__do_not_coalesce=True,
        )
        .annotate(tried=Exists(ExecutionAttempt.objects.filter(stage_id=OuterRef("pk"))))
        .filter(tried=False)
    )
    cleared = []
    for stage in pieces:
        stage.payload.pop("do_not_coalesce", None)
        stage.save(update_fields=["payload"])
        cleared.append(str(stage.pk))

with open("/tmp/restore_cohorts_1126.cleared.json", "w") as handle:
    json.dump(cleared, handle)
print(json.dumps({"restored": restored, "sources": sorted(sources), "cleared": len(cleared)}))
