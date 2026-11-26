# Execution operations, 26 November 2026

## Cohort profiles restored after the research training ended

From 24 November until about 02:10 on 26 November, the owner's research training kept the
shared GPU above the 40 % back-off threshold, so 15 requests waited on GPU embeddings. The
training then ended (`train.py` exited and the GPU dropped to 848 MiB), and webKinPred
resumed at 02:14. By then five cohort profiles were suspended:

| Profile | Clean cohort runs | Suspending OOM |
|---|---|---|
| `gpu76-cohort-gpu16-CatPred-catpred_embed-kcat` | 1,333 | 24 Nov 13:43 |
| `gpu76-cohort-gpu16-CataPro-prot_t5_mean` | 833 | 13 Nov 02:41 |
| `gpu76-cohort-gpu16-UniKP-prot_t5_mean` | 200 | 13 Nov 02:39 |
| `gpu68-cohort-gpu16-KinForm-H-kinform_t5_residues` | 2,209 | 11 Nov 23:16 |
| `gpu76-cohort-gpu16-CatPred-catpred_embed-Km` | 1,068 | 11 Nov 15:58 |

In each case a single OOM suspended the profile, and in each case the research training held
GPU memory at the time. Suspension is permanent (`control.py`, the cohort OOM branch of
attempt failure), and `cohorts.coalesce` then splits that profile's ready cohorts back into
single-protein stages marked `do_not_coalesce`. As a result, 563 CatPred kcat embeddings were
running one protein per attempt: 23 s each, mostly model load, three at a time. Each single
attempt reserves 5.15 GB of VRAM, more than the cohort's 4.03 GB.

`tools/restore_shared_gpu_cohorts.py` was run in the backend shell at 02:19-02:21. It
re-enabled the five profiles and cleared `do_not_coalesce` on the 542 pieces that were ready,
had no attempts and had no failures. Pieces that had already been tried keep the flag. The
cleared stage IDs are in `staging/restore_cohorts_1126.cleared.json` on production.

cohorts.coalesce matches a profile's `cohort_sources` against fingerprints, not profile IDs.
Splitting also reopens the cohort's original `embedding:<...>` stages where it can, and only
creates `split:<...>` stages otherwise. The script's first two runs missed both points and
changed no stages.

Within two minutes, the pieces had been merged into cohorts of 8. Cohort attempts took 50 s
for 8 proteins, about 6 s per protein, and four ran at once. Ten of the 15 requests completed
by 02:26. The only refusals were two `waiting_vram` launches ("Capacity changed during
artifact transfer"), which were retried and do not suspend anything.

Open issue: on a shared GPU, any research burst can permanently suspend a cohort profile,
and nothing restores it automatically. The KinForm-H T5 cohort stayed suspended for 14 days.
