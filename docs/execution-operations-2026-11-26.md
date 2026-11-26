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

## Loki kept every log since April, and its own log was unbounded

Production's free space fell 1.1 GiB in the half hour after the backlog drained, and the
webKinPred stores did not account for it. Two monitoring leaks did:

- `monitoring/loki/config.yml` set `retention_period: 168h`, but Loki applies retention only
  through the compactor (`retention_enabled`), which was not configured. Every chunk since
  April 2026 was kept: 183,000 files and 217 daily index tables, 3.0 GB. Of those files,
  179,000 (1.7 GiB) were more than 7 days old.
- Loki logged each periodic index-table upload at info level: about 3,500 lines every
  10 minutes. The host's Docker daemon has no log rotation, so Loki's container log had reached
  1.44 GB since 9 November. Promtail also shipped those lines back into Loki.

Commit 9d32443 changes three things:
- It enables compactor retention: 10-minute passes, a 24 h delete delay and one delete worker.
- It sets Loki's `log_level: warn`.
- It caps Loki's container log at 3 × 50 MB in `docker-compose.prod.yml`.

The config passed `-verify-config` on the production `grafana/loki:2.9.8` image. At 02:55,
`docker compose -f docker-compose.prod.yml up -d --no-deps loki` recreated only Loki;
Promtail kept its positions, and only its in-flight batches were retried. The old container's
log went with it, freeing 1.4 GB. Promtail and the other services keep the daemon's
unlimited default. Recreating Promtail loses its read positions, which live in `/tmp` inside
the container, and it would then re-read every container log.

The compactor's sweeper unlinks without pacing, and on this disk 750 unlinks a second pushed
one-minute full I/O pressure to 8-10 %, past the supervisors' 5 % launch gate.
`tools/cleanup_loki_chunks.py` (`staging/loki_chunk_cleanup_1126.py`, log
`staging/loki-chunk-cleanup-1126.jsonl`) therefore removes chunk files older than 8 days at
200 operations a second, only while no request is unfinished and full pressure stays under 2 %.
The sweeper's 24 h delay gives it time to finish first. The sweeper treats chunks that are
already gone as deleted.

Scanning the chunk tree with `find` (even at idle I/O priority) raised one-minute full
pressure to 8 %. Avoid unpaced walks of this directory.
