# Execution operations, 9 November 2026

Monitoring after the GPU host came back online. Release 125 removes storage that
grew without being needed; the rule now is that only permanent pooled vectors (and
users' job inputs/results) should lower free disk over time. No job was lost: all
4,717 jobs unfinished before release 125 are present with unchanged checksums, and the
legacy Redis queue still holds 5,002 messages.

## State found

* The GPU host's network dropped repeatedly (Tailscale "offline", no ping from either
  side, heartbeat 3+ minutes stale at 10:12); its attempts were stopped and retried.
* Once reachable, both hosts claimed nothing although thousands of stages were ready.
  A traced claim showed no runnable candidates: every prediction, similarity and
  assembly stage waits for embeddings, and every embedding needs the GPU, which two
  research `train.py` runs used at a 51–67 % one-minute mean (back-off limit 40 %) with
  12.4 GiB of its memory. This is the chosen sharing policy, not a fault.

## Storage that should not grow

| Item | Size | Fix |
|---|---|---|
| Collector discovery query timing out | 18,773 cancellations/day in the PostgreSQL log (266 MB/day, full statement text each); every cancelled call rolled back, so nothing was collected | Pages join only to their artifacts (0.06–0.08 s instead of 0.6–0.7 s warm); job ownership is resolved for the few transient rows |
| Transient residue matrices with no unfinished consumer | 63 GB on the GPU host (28,406 files, some since 5 Nov) | Collected again after the fix; 4,096 queued as hints to start at once |
| Finished attempts' journals (`request.json`, `seqmap.json`, `molecular.json`) | 2.9 GB / 21,816 dirs on production, 20 GB / 64,244 dirs on the GPU host | Supervisor removes a directory once not running, without scratch, unreferenced by a checkpoint index and unchanged for 24 h (500 every 2 minutes); checkpoint indexes older than 3 days are dropped |
| Container logs without limits | PostgreSQL 636 MB, backend 441 MB, Loki 372 MB, promtail 87 MB | Archived (gzip, verified) to `staging/log-archive/` and truncated; backend, controller and control worker now run with `json-file` max 50 MB × 2 (`live20/compose.local.json`) |

A user-facing 500 at 10:39 (`/api/v1/status/cp4SgoK/`, "could not resize shared memory
segment" in a parallel worker) coincided with heavy diagnostic queries; the database
container has the default 64 MB `/dev/shm`. Diagnostics now run with
`max_parallel_workers_per_gather = 0`.

## GPU drain barrier (release 126)

The barrier that holds smaller GPU admissions back so a long-waiting GPU stage can fit
only checked that the waiter fits an empty GPU. While other users' processes held the
memory it could never fit, so the barrier would have held smaller GPU work back
indefinitely. It now also requires that the waiter fits in free memory plus memory held
by this host's own attempts (commit `cccad66`, control image
`sha256:e5ee462be5cea9bf0003f29607ad12d0f0e67089d640ca553478af0d9b8ccb5c`). Correction to
that commit message: the 28 candidates that fit at 12:15 were profiles, not queued
stages. All 3,204 runnable GPU embeddings needed 4–6 GiB plus the 2 GiB guard while
5.0 GiB was free, so on 9 November the GPU was limited by memory held by the research
jobs, not by the barrier.

## Deployment (release 125)

| Component | Image |
|---|---|
| Backend, controller, control worker | `sha256:02cdf9287bc0494e10a5f9968f6d485570afda721abe96ddd236456f56d50a43` |
| GPU supervisor | `sha256:9959c7bd770424ef5bdbf3686820f5fba7d42dcca73f0dcf995b2226f3889f04` |
| Production supervisor | `sha256:eec04d7b939a408419a858e07a84c257afa0515f4978880666668170800e9717` |

Rollback containers `*-rollback125`; environment backups `live20/*.before125`.

## Duplicate catalogue indexes (release 129, prepared)

Over 13 hours to 23:50 the execution database grew 99 MB/h. Six indexes duplicate
other indexes and held 3.0 GB, and every catalogue insert writes them: Django's
`varchar_pattern_ops` copies of `api_artifact.id`, `api_artifact.checksum`,
`api_artifactdependency.artifact_id`, `api_artifactlocation.artifact_id` and
`api_artifactlocation.host_id` (nothing searches these by prefix), and
`api_artifactlocation.artifact_id`, which the `artifact_host_location` unique index
covers. Migration `0045_drop_duplicate_artifact_indexes` (commit `76b6228`) drops them
concurrently; reversing it recreates them. Control image
`sha256:55f5068481a967125bfbccfb9d5987dbda79e5969b6b95c661c8eff4edd2d5bf`
(`webkinpred-control-fix:r129`) adds only the migration and the model flag. Running
services stay on release 128, since nothing changes at runtime. Apply on production;
this does not recreate any service:

```bash
sudo sh -c 'cd /home/saleh/webkinpred-execution-staging/release129 && ./migrate129.sh apply'
```

`./migrate129.sh plan` lists pending migrations (only 0045 on 9 November). Roll back
with `migrate api 0044` from the same image.

## Still growing

* **Execution database, 99 MB/h on 9 November** (41 MB/h on 6 November, at lower
  throughput). No catalogue rows are ever deleted:
  `api_artifact` 12.3 MB/h (8,225 inserts and 28,589 non-HOT updates per hour),
  `api_artifactdependency` 9.5 MB/h, `api_workstage` 8.4 MB/h, `api_artifactlocation`
  7.8 MB/h. Pruning history of finished jobs (rows of deleted transient artifacts and
  their dependencies/locations) needs a decision because it removes provenance.
* **Duplicate permanent copies on production, ~0.6 GB/day.** 4.5 GB of production's
  5.5 GB of permanent artifacts also have a ready copy on the GPU host (2.5 GB
  embedding vectors, 1.8 GB predictions); only 0.9 GB of assembled results exist solely
  on production. They are never collected.
* **Loki** keeps every log line: `retention_period: 168h` has no effect without a
  compactor with `retention_enabled` (~30 MB/day).
* 1,441 exited `wk-profile-*` profiling containers on the GPU host (397 MB of logs plus
  writable layers) are leftovers from earlier profiling runs.
