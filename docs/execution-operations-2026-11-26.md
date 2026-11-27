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

The cleanup started at 02:56 and waited for the last request until about 03:15. It finished at
03:48, having removed 179,166 chunk files (1.80 GB) and 146,183 emptied stream directories.
It paused for 1,180 s in total. One-minute full pressure stayed between 1.5 and 3.5 % while it
deleted, and fell to 0.4 % afterwards. Production went from 72.1 GiB free before the Loki
changes to 75.8 GiB.

## The manual restore restarted the controller; release 159 automates restoration safely

Clearing `do_not_coalesce` by hand at 02:21 let `cohorts.coalesce` regroup some units into
exactly the cohorts they had been split from. `replace()` keys a cohort by its member stage
IDs, and the canceled cohort still held that key, so creating the new one violated
`execution_stage_key`. That `IntegrityError` is not caught by the controller loop, so
`execution_controller` exited seven times between 02:21:11 and 02:21:48 (restart count 7).
It ran cleanly after 02:21:48, once claims had changed the candidate groups. The controller
was down for about 37 s in total, and nothing was lost: each failed pass rolled back.

Commit fadd13a (`webkinpred-control-fix:r159`, `sha256:738ad35e…`, built from r158):

- Cohort keys: when a stage with the same `cohort:<digest>` already exists in the execution,
  the new cohort gets a generation suffix (`:2`, `:3`, ...).
- `control.fail` records `evidence.shared_gpu_suspension` (time, host and the host's
  `gpu_external`) when a cohort OOM happens while other users' processes use at least 1 %
  mean SM. Other cohort OOMs suspend the profile as before, with no automatic restoration.
- Each cohort pass calls `cohorts.restore_shared_gpu_suspensions`. It keeps `quiet_since` on
  the profile and lifts the suspension after 15 minutes during which every live GPU host
  reported `gpu_external` below 1 %. Any external compute, or a missing pmon sample, restarts
  the period. Each restoration is logged as `execution.cohort_restored` and recorded in
  `evidence.restorations`.
- A `reopen_cohort_pieces` outbox event then clears `do_not_coalesce`, in transactions of 500,
  on the source units that are ready, untried and have no failures. This runs outside the
  cohort pass and its 3 s statement limit.
- `tests/test_execution_cohorts.py` adds five tests: restoration after the quiet period, a
  reset by external work, missing telemetry, a non-shared OOM, and re-merging the same
  units. The re-merge test reproduces the production `UniqueViolation` without the key change.
  The full suite gave 760 passed, with the same 5 setup errors.

The image's copies of the three replaced files matched the repository before the change.
`staging/release159/deploy_control159.sh` recreated the backend, controller and control
worker at 04:24:31. Preservation (`preservation113/*159*`): the one unfinished request
(xrghY3g, 7,354 rows, processing) was present afterwards with unchanged checksums. The Redis
snapshots were identical, and the legacy queue still held 5,002 messages. MWi2hIV was then
submitted through the new backend.

**Build the next control release from `webkinpred-control-fix:r159`.**

The compactor's sweeper started on the marked backlog when the 24 h delay expired, at about
03:05 on 27 November, and finished by 03:44. Most marked chunks (180,768) were already gone;
it deleted the 413 that were 7-8 days old. While it ran, production's one-minute full pressure
was about 2 % with no requests; it fell back to 0.1 % once the sweeper finished. From now on
the compactor expires about one day of chunks per day.
