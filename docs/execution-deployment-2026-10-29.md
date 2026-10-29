# Resource-aware execution deployment record

This record describes the deployment in progress on 29 October 2026. Public
submissions remain paused until the canary passes. Status and completed-result
downloads remain available. This is not a completed rollout acceptance report. Deployments use versioned
source bundles with file checksums and pinned image digests. The implementation
is now committed in Git; original dirty remote checkouts and model files are
preserved rather than overwritten with `git pull`.

## Deployed architecture

Production runs the dedicated PostgreSQL 17 instance, Django backend, durable
controller, short Celery control worker, executor agent, artifact gateway,
Prometheus and Grafana. The GPU host has an executor agent, artifact gateway,
and durable compatibility proxy for the embedding service. Both hosts can run
CPU prediction containers; embeddings require the assigned GPU. Existing model
environments were reused. Application fixes use thin layers on preserved images.

The legacy prediction/cache workers and beat remain stopped. The old GPU
executor's systemd service is disabled. Unrelated applications remain running.
The original Redis broker database 0 is frozen migration evidence. Database 3
carries the versioned control queue. PostgreSQL owns prediction scheduling, so
Redis message counts are not expected to equal the number of unfinished jobs.

## Preservation and cutover evidence

* 10,324 original jobs were migrated with their public identifiers and related
  data. All original fields were compared; unfinished progress counters were
  deliberately reset for restart.
* 4,999 unfinished requests have one immutable specification each. Import was
  repeated successfully to establish idempotency. Quotas were not charged again.
* The 5,002 frozen broker messages were reconciled to jobs, including duplicate
  and terminal-message dispositions. The active/reserved requests were captured
  before stopping the old workers. No purge/resubmission command was used.
* 488,585 sequence identifiers, 244,549 prediction-cache rows and 7,843 similarity
  records were preserved. Input checksums and request coverage were rechecked for
  all 4,999 imported requests.
* The previously processing job (identified in the private canary manifest) retains its identifier and was included
  in the restart canary.
* Final SQLite/Redis backups were verified off production, including actual
  SQLite restoration and cold Redis restoration. Fresh custom-format PostgreSQL
  backups were also checksum-verified and independently restored on the development host.

Private evidence is under `webkinpred-execution-staging/cutover19` on production,
with the original off-host copy under `production-backup/cutover19` on the GPU
host. Credentials, database dumps, user inputs and request exports must remain
outside source control.

## Canary and availability

The 100-job canary began at `2026-10-29T10:07:54.206058+00:00`. It covers every
queued method, long inputs, multiple targets and the interrupted job. Initial
limits are one CPU attempt per host and one GPU allocation; 4,899 other jobs
remain paused. The gate requires 100 completed jobs and at least one hour of
observation before increasing admissions or reopening submissions.

The GPU host became unreachable at approximately 10:20 UTC. Both local and
production checks report its Tailscale peer offline; SSH, embedding and artifact
ports are unreachable. Connectivity returned at approximately 13:12 UTC. Outstanding reservations were
retained until the executor reconciled. A pre-launch transfer lease that remained
stuck after reconnecting exposed a recovery defect; the supervisor now drains
such transfers and confirms absence before releasing their reservations. Completed artifacts are not invalidated, and GPU-dependent
work does not fall back to CPU embedding computation. At the subsequent check,
57 GPU-host attempts had completed and no imported job had status `Failed`.

Production admissions were drained temporarily for additional isolated CPU
profiling and have resumed for the canary. Disk headroom increased from approximately 15 GiB to 24 GiB by
hard-linking identical files within inactive rehearsal/profiling copies.
All paths and file contents were preserved during deduplication; authoritative
media and model trees were excluded. Subsequently, 28,945 files from inactive
rehearsal/profiling copies were archived on the GPU host and verified against
the complete manifest before those production copies were removed. The manifests
and reclamation record are retained.

The canary exposed an all-invalid-input OmniESI bug: the generic adapter returned
an empty annotation list, which the orchestration indexed as though it contained
one entry per row. The fix returns absent annotations; invalid rows retain their
existing warnings and no model is loaded. Four regression cases cover both
OmniESI variants and two chunk sizes. Patched prediction images are reprofiled
before advertising their new fingerprints. All four replacement CatPred/CatRange
profiles on production have now passed three fresh and five warm runs. The next
batch covers the other prediction heads on production; the first OmniESI `Km`
run matched the saved CPU baseline and took approximately 554 seconds for the
128-row boundary fixture. That profiling batch stopped safely at the disk-headroom guard. Two additional
OmniESI-family CPU profiles completed all required runs before the stop. All
eight replacement GPU-host prediction profiles subsequently passed three fresh
and five warm runs against their saved CPU results.
A guarded activation script waits for calibration to stop, imports only complete
validated profiles, restarts the drained production supervisor, and resumes only
the existing CPU canary. It cannot reopen public submissions or lift GPU limits.

## Verification completed during this continuation

* 372 tests and three subtests pass against isolated PostgreSQL, including real
  Docker execution/OOM recovery, atomic admission, ownership fencing, migration,
  quota, artifact and API contracts. Test output:
  `/tmp/webkinpred-full-tests29.txt` on the development host.
* Offline-artifact admission waits without starting a container, incrementing
  failures or invalidating a verified location. Tests cover recovery through a
  returning peer and through a verified local copy. A partial local transfer
  does not make the stage eligible. A page of 513 stages waiting for offline
  artifacts cannot hide younger work whose inputs are reachable.
* The production control image containing that check is
  `sha256:eb3f3ef5ab1ead8cb82ff45d60a1463398c2fc44e2142d8ff31bfcaed9d950fb`.
  Backend and frontend health checks pass with submissions paused.
* Bidirectional artifact transfer was verified before the outage using checksums.
  Existing completed downloads were checked against their saved checksums.
* Prometheus successfully scrapes the authenticated control endpoint. Nine alert
  rules are loaded; host-unavailable and uncertain-ownership alerts detect the
  real outage. Grafana's provisioned dashboard has 18 panels. Services bind only
  to production loopback ports 9093 and 3002 respectively.

GPU embedding numerical comparisons were waived explicitly by the user.
Shape/dtype/finiteness, positional context, CPU prediction compatibility,
resource envelopes and recovery remain required. A completed artifact transfer
also requires a fresh successful ownership renewal before compute may launch;
controller disconnection and stale ownership both prevent that launch. Initial profiling measurements
are documented separately; whole-platform throughput must be measured after the
canary and must not be inferred from free VRAM alone.

## Remaining release gates

Activate the newly validated GPU profiles, retry the remaining diagnosed
OmniESI stage, complete the 100-job canary and
one-hour healthy observation after connectivity and profile recovery, then
increase admissions only to measured configurations.
Before reopening submissions, set the new-request pause policy appropriately
and rerun `execution_readiness --phase reopen`. Retain rollback images and
migration evidence for 30 days; do not prune images or restart legacy consumers.

A single parent requiring more than the current 4,096 expanded-unit planning
envelope remains an explicit hold; the imported backlog's maximum expansion is
two and was checked to fit. Unversioned legacy embedding files remain preserved
but are regenerated when their representation identity cannot be verified.
