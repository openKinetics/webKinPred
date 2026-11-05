# Execution throughput rollout, 4 November 2026

Releases 113–119 removed the scheduling and control-plane limits that kept both hosts
under-used. Scientific methods, model images, weights, environments and input/output
semantics are unchanged; every change is in admission, supervision, control queries or
host configuration. No job was lost: all 7,493 jobs unfinished at the first snapshot are
present with unchanged specification/input checksums.

## What was wrong

| Symptom | Cause | Fix |
|---|---|---|
| Zero completions from ~02:00 to 14:15 UTC | Production free disk (14.7 GiB) below its 15 GiB admission floor; assembly only runs on production. Docker console logs had no rotation: Loki 9.3 GB, Postgres 1.8 GB | Logs gzip-archived and verified before truncation (`staging/log-archive/`); 24 GiB free afterwards |
| Production admissions flapping (`host_pressure`, `waiting_cpu`) | Postgres capped at 4 GiB with 128 MB `shared_buffers` against a 15 GB database: constant reads, I/O PSI ~8% "full" (threshold 5%), Postgres at ~170% of its 2 CPUs | `docker update --memory 10g` (no restart; `EXECUTION_DB_MEMORY=10g` pinned for recreation) and `effective_cache_size=10GB`: I/O PSI 0.4%, Postgres ~67% CPU |
| GPU-host CPUs idle while ~1,600 predictions were dependency-ready | Single-threaded supervisor admitted one attempt per ~6 s (per-job `docker inspect`, three `nvidia-smi` calls per tick, one synchronous HTTP heartbeat per attempt, synchronous 1–8 s completions; claiming stopped whenever any lease was due) | Batched claims and heartbeats, asynchronous idempotent completion, one `docker inspect` per tick, cached cgroups, rate-limited runtime checks |
| Reservations capped real use | Peak CPU reservations filled the 19-CPU ceiling at ~5 cores of real use; finished containers kept CPU/RAM/VRAM/slots while publishing | `cpu_overcommit` (spare CPUs × factor), reservations released when a container exits (disk kept until publication ends) |
| GPU starvation behind CPU work | `cpu_external` inflated by container churn; GPU CPU share carved from the nominal budget; batches filled with CPU work first | 20 s rolling-minimum `cpu_external`, GPU share carved from effective capacity, GPU candidates examined first within a class |
| Claims ~1.9 s each | `DISTINCT ON` over the whole backlog (1.0 s), recursive foreground CTE seq-scanning stages (0.3 s) | Two top-1 head queries; explicit normal-execution array (0.3 s → 6 ms). Claims ~0.6–0.9 s |
| Public status polls ~200 ms each | Producer `UNION` planned as a parallel seq scan of `api_workstage` | Indexed lookups |
| Two 17,010-row normal jobs stuck for days | Inputs beyond measured envelopes (SMILES ≤ 801, 17,010-row similarity/assembly) | Cloned envelope-extension profiles (below); both jobs completed 16:13/16:17 |

## Code changes

* `api/execution/control.py`: `claim_batch()` (≤ 8 admissions per locked, ranked scan;
  `claim()` wraps it); exact per-stage checks against a ledger that includes the batch;
  never preempts after admitting in the same batch; `job_has_active` re-ranking; GPU-first
  examination; production serves assembly first; `heartbeats()`; GPU CPU share from
  `cpu_capacity()`; alternative profiles chosen by smallest fitting reservation; top-1
  queue-head queries; host-row locks `FOR NO KEY UPDATE`.
* `api/execution/resources.py`: `cpu_overcommit` (1–4, scales spare CPUs only),
  `cpu_capacity()`, released attempts keep only their disk reservation.
* `api/execution/views.py`: `claim` with `limit` returns `attempts`; `heartbeats`
  operation. Older agents keep the single-attempt protocol.
* `api/execution/background.py`, `status.py`, `metrics.py`, `retention.py`,
  `telemetry.py`: query plans above; artifact inventory scan off unless
  `EXECUTION_METRICS_ARTIFACT_INVENTORY`; cached cgroups; pruned CPU samples.
* `tools/executor/agent.py`: batched claims/renewals, async completion with resubmission
  on transport or server errors (403/409 still reconcile), `publish` assembly without a
  container (same outcome as the runner), exited-container release flag, refused
  launches drop their materialized inputs, per-minute `executor.loop` summaries.
* Tests: 693 passed, 4 skipped (opt-in container tests); new
  `test_execution_claim_batch.py`, `test_execution_agent_supervision.py`,
  `test_execution_cpu_overcommit.py`.

## Deployed images and configuration

| Component | Release 119 image | Previous (rollback) |
|---|---|---|
| Backend, controller, control worker | `sha256:3a0247d03236c797fafad5af55c5cfa2959e713cbc25b2dbe9944acdf8561f0d` | `sha256:7c6c85977ceb…` |
| Production supervisor | `sha256:9f85f15be1ab0b18ffaebfb67a934c7550900c2e9327b7af363b3ced3357903a` | `sha256:9be60d35034b…` |
| GPU supervisor | `sha256:d7ced8d72f6c280a10c118c7e7733e596f9a5cd70ada0d9379a86bb16289315c` | `sha256:51e428587157…` |

Each supervisor replacement drained its host first and kept a stopped
`*-rollbackNNN` container. Private configuration changes (all with `*.beforeNNN`
backups in `live20/`):

* GPU budgets (server and `gpu.json`): `cpu` 30, `cpu_slots` 24, `cpu_headroom` 1,
  `cpu_overcommit` 1.6, `gpu_cpu_reserve` 10, `gpu_slots` 6, `ram` 112 GiB. Production
  budgets unchanged.
* Backend gunicorn workers 3 → 6; `EXECUTION_PREPARED_BACKGROUND_LIMIT` 512;
  `EXECUTION_DB_MEMORY=10g`; Prometheus scrape interval 60 s
  (`release100/deploy/execution/prometheus.yml`).
* Envelope-extension profiles cloned from measured ones (same image, fingerprint,
  settings and model identity): `gpu16-UniKP-molecular-len4096` (the legacy SMILES
  encoder always uses a 220-token window), `*-UniKP/EITLEM-*kcat-mol1024` (RAM ×1.25),
  `*-UniKP/EITLEM-similarity-rows20000` and `production34-assembly-rows20000` (×1.5).
* 834 stale exited attempt containers removed on the GPU host (exited > 10 min, not live).

**Do not run `deploy/execution/deploy.sh` as is.** Its first step, `up -d execution-db`,
would recreate PostgreSQL because the running container's labels reference older
`active-release/` paths. Deploy control services with
`docker compose … up -d --no-build --no-deps backend execution-controller execution-control-worker`
from `release112/`, after a `--dry-run`.

## Measurements

Ten-minute windows with 2 s host samples and 14 s database snapshots.

| Measurement | 3 Nov, before allowlist removal | 3 Nov, after | 4 Nov 16:57–17:07 |
|---|---:|---:|---:|
| Mean GPU utilization | 40.3% | 5.3% | **62.9%** |
| GPU samples at 0% | 44.7% | 62.2% | **17.0%** |
| GPU-host CPU, cores / 20 | 2.71 | 1.86 | **10.68** |
| Production CPU, cores / 4 | 2.56 | 2.10 | **3.24** |
| Jobs completed in the window | 0 | 0 | **72 (28,516 rows)** |

Completed jobs per hour on 4 November: 0 until the disk fix; 14:00–15:00 113 (partial);
15:00–16:00 254 (92,287 rows); 16:00–17:00 295 (156,198 rows). In the final window the
GPU host completed 57 embedding, 97 prediction and 49 similarity attempts; production
completed 115 assemblies. No HTTP 5xx and no deadlock appeared after release 119.

An unrelated ThermoKinetics evaluation (`eval_thermo_metrics.py`) used ~18.5 of the GPU
host's 20 CPUs from 16:14 until shortly before 16:57. Admission treats such load as unavailable and
yields, so CPU work on that host paused during it; the 16:21–16:31 window is not
representative.

## Incident during rollout

Releases 113–118 submitted completions asynchronously but re-raised a cached failed
completion on every tick. Concurrent completions could deadlock against a claim holding
the host row (`FOR UPDATE` versus the foreign-key `KEY SHARE`); the HTTP 500 then stopped
the GPU supervisor's loop for ~13 minutes (16:33–16:46) until its replacement. Leases
expired and the affected stages returned to the queue; no job failed or was lost.
Release 119 resubmits on server errors and takes host rows `FOR NO KEY UPDATE`; a
regression test proves a held host lock no longer blocks the insert.

## Preservation and health

Snapshots `staging/preservation113/` (CSV of every job with checksums, Redis RDB copies)
were taken before each release. Final check: 7,493 baseline jobs present, 0 missing,
0 checksum changes, 599 completed since 14:42, 0 failed. Legacy Redis queue 5,002
messages, control queue 0. No OOM, held, uncertain or expired-live attempts.

## Remaining limits

* GPU concurrency is bound by VRAM reservations (profile peak + 25%, plus a 2 GiB guard)
  held from container start through model loading: about three large encoders at once.
  Shape-aware reservations or warm encoder workers are the next large step.
* The GPU host's free disk fell ~118 → 94 GiB during peak throughput (blob store and
  retained attempt journals; ~19 GB in `attempts/`); its admission floor is 50 GiB.
  Watch it alongside production's 25 GiB versus 15 GiB.
* Every attempt still starts a container and loads its model; prediction attempts take
  at least ~8 s even for tiny inputs.
