# Resource-aware execution: implementation and rollout

This replaces the serial executor after the migration and acceptance procedure
below. Fresh installations must keep `EXECUTION_V2_ENABLED=0` until profiling,
compatibility, migration rehearsal, and recovery gates pass. No universal
resource registry ships with the code: resources must be measured on each final
runtime. An empty registry admits no model work. See the
[deployment record](execution-deployment-2026-10-29.md) for the live rollout.

## Architecture and code

PostgreSQL owns immutable requests, dependency stages, attempts, host budgets,
leases, artifact identities/locations, quota events, and the notification outbox.
Redis is not the authoritative prediction queue. The controller reconciles
PostgreSQL every second, including after notification loss.

* `api/execution/control.py`: transactional claims, aggregate reservations,
  fencing, completion, retries and confirmed-termination reconciliation.
* `api/execution/planning.py`: stable sequence resolution and deterministic
  prepare → embedding dependencies → target chunks → similarity → publication.
* `tools/executor/agent.py`: host-local supervisor, durable request journals,
  Docker identity/limits, asynchronous transfers/publication, lease enforcement.
* `tools/executor/runner.py`: isolated stage adapters and streamed assembly.
* `tools/executor/molecular.py`: CUDA molecular encoding with preserved legacy
  UniKP/KinForm batch positions. The positional-encoding bug is deliberately
  preserved; no molecule-only cache reuse is allowed for those representations.
* `tools/executor/kinform.py`: explicit residue → binding-site → CPU pooling
  stages. Compound legacy workers are rejected under the new execution policy.
* `api/execution/quota.py`: idempotent reservations and original-day refunds.
* `api/execution/recovery.py`: dependency repair after missing embeddings, including
  shared embeddings originally produced for a completed job. Repair belongs to
  the unfinished consumer; completed jobs retain their results and status.

The controller takes no model-execution lease. Preparation uses bounded CSV
frames: at most 128 parent rows, 8 MiB input text, and 4096 estimated expanded
units. A shape beyond the implemented planning envelope is held for attention,
never silently truncated. Profile limits may be smaller and are enforced again
at admission. A single reaction requiring child-unit subdivision beyond this
envelope still requires the child-partition planner before that request can run.

CPU containers have no GPU access. Missing embeddings cannot load an embedding
model in a prediction container. GPU workers explicitly require CUDA. RAM/CPU,
PID, swap and disk-headroom checks apply to all work on a host, with a second
availability check at the executor. GPU admission includes unspent reservations
of allocations still loading their models and at least 2 GiB/10% GPU headroom.
Work requiring an artifact whose verified locations are all on offline hosts
waits before reserving a container. Locations remain verified through an outage;
a returning host or a verified local copy makes the work eligible again.
Additional GPU processes require a measured compatible combination, not merely
free VRAM. Default simultaneous GPU allocation count is one.

The supervisor checks running attempt scratch usage and stops its own container
when it exceeds its disk reservation or host headroom is breached. Transfers
recheck free space while streaming. These checks are sampled containment; host
filesystem project quotas are still needed for a hard per-attempt disk boundary
against an instantaneous unexpected write burst. Committed blobs survive cleanup
of successful private scratch; interrupted attempt scratch remains available for
checkpoint recovery.

Existing submission/start/completion timing fields keep their historical meaning.
The added active/resource-wait durations measure this executor's lifetime, so
unmeasured legacy computation is not incorrectly reported as resource waiting.

Each attempt gets private inputs, temporary paths and output files. Artifacts
are SHA-256 blobs published atomically after size/checksum verification. Active
computation uses host-local disk. Only immutable blobs move between authenticated
Tailscale artifact gateways; production retains authoritative original inputs
and downloadable results. Attempt containers have no network access or database
credentials. Model weights/environment mounts are read-only.

Generated embedding arrays, tensors and binding-site rows are checked for
nonempty floating values and finiteness before publication. Their shape/dtype
metadata is stored with the checksum. Explicit CatRange input rejection markers
remain distinct from infrastructure failures.

Regenerable blob collection runs only while the local executor is idle and free
space is below its cache target (disk headroom plus 10 GiB by default). A durable
cleanup record fences new launches and local preparation writes until deletion
is acknowledged. All checksum aliases are checked for unfinished dependencies,
active attempts and non-regenerable data. Files must be at least an hour old by
default. Original inputs, final results and legacy cache files are retained.
Lost cleanup responses and agent restarts resume the same deletion record.

Heartbeats renew a 30-second lease. An agent stops its container after 20 seconds
without renewal. Expiry alone never frees a reservation: a reconciled host must
prove termination. Old ownership tokens cannot publish. SIGKILL and container
OOM are observed through Docker. Resource shortages retry/back off or hold work;
they never mark a job `Failed`. Three repeated failures at the smallest available
configuration hold the stage and suspend its profile.

## Build and private configuration

Generate templates outside source control:

```sh
python -m tools.executor.configuration .execution-private
python -m tools.executor.release_bundle /path/on/staging/execution-source.tar.gz
docker build -f Dockerfile.executor -t webkinpred-executor:release .
```

Templates start disabled/paused with empty images, profiles and model versions.
Record image digests, exact model/tokenizer/source checksums and host runtime
fingerprints. Populate `runtime_files`, read-only `model_mounts`, approved image
digests, profile fingerprints and model version identities. Never overwrite the
dirty production/GPU checkouts or their model files. Use separate release trees.
`settings.python_paths` and `embedding_environment` support validated host conda
paths mounted read-only; the complete environment must be included in the runtime
identity and profiling evidence. Rebuilding or upgrading dependencies invalidates
the matching profiles.

The production overlay is `docker-compose.execution.yml`; GPU host services use
`docker-compose.executor-host.yml`. They are intentionally absent from the legacy
deployment path. Do not use `deploy.sh` for cutover: it rebuilds/restarts legacy
consumers and prunes images. Preserve rollback images and evidence for 30 days.
Do not create a temporary 103 GB image archive on production; use the small
source bundle and reuse model environments/layers or stream an image transfer.
`Dockerfile.execution-runtime` adds application code to the exact existing web
or prediction image without running `Dockerfile.envs`. `Dockerfile.gpu-native`
uses the GPU host's mounted environments on a Debian 12 base, which also supports
the older PyTorch libraries used by that deployment. Pin and profile the resulting
image; never infer resource compatibility from a successful import alone.

For a GPU host lacking Docker GPU integration,
`deploy/execution/install-gpu-runtime.sh` installs the pinned NVIDIA toolkit,
preserves daemon configuration, registers the OCI runtime, validates the Docker
configuration, and sends SIGHUP. It checks existing container start times and
does not restart Docker. Docker 27 may require explicit `gpu_runtime: nvidia`
with the assigned GPU UUID because `--gpus` device-driver registration predates
the configuration reload. [Docker runtime reload documentation](https://docs.docker.com/reference/cli/dockerd/#configuration-reload-behavior),
[NVIDIA installation documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Profiling and scientific gates

Run probes against copied inputs in a new isolated scratch namespace. Do not
mount production media read-write or invoke submission/quota/cache-write APIs.
`tools.executor.baseline` samples host, external GPU and container activity for
one hour. `tools.executor.profile` runs an actual adapter in the pinned image,
records cgroup peaks/events, process-tree RSS/PSS where available, disk usage,
GPU/process observations and framework allocator high-water marks at ~100 ms.
`tools.executor.profiling_suite` fingerprints copied data, runs three fresh and
five repeated/warm probes, and compiles conservative envelopes. Record
`warm_scope=process` when a warm command reuses loaded models, or
`warm_scope=filesystem` for adapters that launch a new model subprocess for
every attempt. The latter measures repeated production behavior with warm file
caches; it must not be reported as model reuse. `molecular_probe` supports
process reuse for the legacy SMILES transformer.

Every registry entry must include the source/image/model/tokenizer/dependency,
precision/thread/batch-context, hardware/driver identities; tested shape limits;
three fresh and five warm measurements; and successful scientific parity.
`execution_profiles` rejects incomplete evidence or mutable image references.
The compiler adds peak + max(25%, 512 MiB RAM / 256 MiB VRAM / 1 GiB disk),
and at least 128/twice-observed PIDs. Unknown shapes or runtimes wait.

Cover **every registered method and target**, both CPU hosts, 1/2/4 threads,
one/small/median/large/max inputs, protein and molecular length boundaries,
multiple sequences/substrates, complete/missing/partial/corrupt/shared caches,
interrupted transfers and checkpoints. GPU microbatch sizes advance in powers
of two from one. Stop escalation before an unsafe allocation. Measure all
individual KinForm workers/file-backed combinations, all protein families and
the four molecular encoders. Then test same/mixed-profile pairs and larger
combinations only within measured envelopes. Retain single-process execution
where extra processes reduce throughput. Record external GPU contention.

Use `tools.executor.parity` on exported baseline/staged artifacts. Molecular
FP32 comparisons use `rtol=1e-5, atol=1e-6`; dtype/shape must agree. Compare CSV
row ordering, invalid/skipped rows, units, source annotations, experimental
overrides, CatRange ranges/classes and method predictions. Test transformer
positions 0, 1, 99, 100, 127, 128 and changed retry microbatch boundaries. Failed
parity blocks that configuration; do not automatically loosen tolerance.

The user explicitly waived **GPU numerical parity** in the 28 October 2026
deployment instruction. Profile evidence may therefore record
`gpu_numerical_waiver` with `approved_by_user=true`,
`scope="gpu_numerical_parity"`, an attributable `request_reference`, and
`shape_dtype_finite_passed=true`. Retain the actual numerical comparison outcome;
do not mark it passed. This waiver does not waive output structure, finiteness,
CPU compatibility, resource profiling, migration, or recovery gates.

Example commands (all paths are isolated evidence):

```sh
python -m tools.executor.profiling_suite manifest copied-inputs dataset.json
python -m tools.executor.profiling_suite run benchmark.json new-evidence-directory
python -m tools.executor.parity baseline staged comparison.json parity.json
python -m tools.executor.profiling_suite compile benchmark.json registry.json --runs runs.json --parity parity-by-probe.json
python manage.py execution_profiles registry.json
```

Runtime reservations can increase after an OOM and validated smaller batches
can be selected. Reducing an envelope or adding a GPU combination requires new
measurements. Profiles bind to semantic model identity, target step, input shape,
approved host fingerprint and immutable image. A runtime-file change removes
the agent's advertised eligible profiles.

## Cutover: preserve the original queue

Do not begin maintenance until the implementation and the following rehearsal
have passed. The export/import commands **never purge broker messages**.

1. Inventory both dirty repositories, image IDs, mounts, service definitions,
   databases and model files. Preserve source patches and unique modified model
   binaries outside production's root disk. Freeze release identities.
2. Activate the maintenance middleware before stopping consumers. Block new
   submissions/validation/similarity before quota reservation; keep status and
   completed downloads working. Verify 503 + `Retry-After` on blocked routes.
3. Stop beat and cancel consumption on every legacy prediction/cache worker.
   Verify acknowledgements; capture active/reserved/scheduled requests before
   stopping those workers. Export all priority lists and unacknowledged data.
4. Run `execution_export /private/off-root/export`. Reconciliation must explain
   every source request and recover complete immutable options for every
   Pending/Processing job. Conflicting requests or missing evidence stop cutover.
   GPU-service state from the old in-memory service must also be captured or its
   requests drained before replacement. Do not restart it while requests exist.
5. Stop only identified platform workers/descendants. Confirm no remaining
   writers. The Processing job retains its public ID and is restarted under the
   new executor. Unrelated GPU processes and services remain running.
6. Take final SQLite backups using SQLite's backup API (including WAL-visible
   data), preserve Redis DUMP/RDB + TTL evidence and all original inputs/results,
   patches, unique model weights and deployment metadata. Store off-root, verify
   checksums and rehearse restoration. `tools.executor.backups` supports copied
   file/SQLite manifests, verification and isolated restore rehearsal. Archive
   Redis itself as well as the human-readable queue export.
7. Start the dedicated PostgreSQL 17 instance. Run the one-shot migration service
   once for default, sequence and prediction-store aliases. Do not run migrations
   from each backend startup.
8. Import each verified SQLite backup with `execution_migrate_database BACKUP
   --database ALIAS`. Target databases must be empty except migration seed rows.
   This preserves primary/foreign keys and sequence collision suffixes, resets
   PostgreSQL serial sequences and checks constraints/counts. Validate completed
   download checksums and permissions against the source inventory.
9. Run `execution_import reconciliation.json --quota-snapshot quota.json`.
   Repeating it is idempotent. It imports one request/stage root per unfinished
   job, preserves public IDs/options/submission times and seeds original-day
   quota reservations without charging again. Retain the old queues/exports as
   frozen evidence; new notifications use `webkinpred-control-v2`.
10. Deploy control services, gateways and agents with all hosts paused. Verify
    authentication, local storage, model checksums and container reconciliation.
    Prepare imported graphs while host admissions remain disabled. Missing
    profiles, unsupported child partitions or unavailable artifacts block resume.
11. Complete final-runtime profiling/parity/recovery gates. Run
    `execution_readiness EVIDENCE --phase canary`. Start with at most one CPU
    attempt per host and one GPU allocation. Select an explicit cohort covering
    methods, long inputs, cache misses, multi-target jobs and the interrupted job.
12. After at least 100 imported jobs complete and one hour of observation, run
    `execution_readiness EVIDENCE --phase reopen`. Lift only validated resource
    limits/combinations, then reopen submissions. Reconcile all original job IDs
    until each has an explained state. Report measured serial/concurrent
    throughput on the same workload, including external contention.

The evidence JSON requires true verification fields listed by
`execution_readiness`, the migration manifest path and, for reopening,
`canary_started_at` with timezone and method/shape coverage confirmation. The
command reports missing gates and never resumes services itself.

Before reopening, rollback stops every new attempt, proves termination and
restores frozen databases/broker/images/configuration. After reopening, first
preserve all newly accepted requests/results/quota events; never overwrite them
with the old snapshot. Prefer pausing admissions, reducing concurrency, or
disabling a problematic profile while retaining the durable control plane.

## Operator controls and verification

```sh
python manage.py execution_admin status
python manage.py execution_admin pause-submissions
python manage.py execution_admin pause
python manage.py execution_admin drain --host gpu
python manage.py execution_admin stop --host gpu
python manage.py execution_admin explain --stage STAGE_UUID
python manage.py execution_admin disable-profile --profile PROFILE_ID
python manage.py execution_admin retry --stage STAGE_UUID
python manage.py execution_admin bind-profiles
python manage.py execution_admin resume --host gpu
python manage.py execution_admin resume
python manage.py execution_admin resume-submissions
```

`drain` stops admissions and lets active work finish. `stop` requests termination
of that host's labelled attempts and retains reservations until confirmed.
Agent restart inventories/stops its own surviving containers before reconciliation.
Retrying a stage does not automatically reactivate a suspended profile. Inspect
the cause, provide a newly validated profile and bind it before retrying.

## Monitoring

The authenticated `/internal/v2/metrics` endpoint exposes aggregate job/unit
throughput, queue age and reasons, reservations versus observations, free host
resources, retained artifact bytes, retries/OOMs, stale ownership and host health.
It does not expose public job IDs, input sequences or ownership tokens. Private
configuration generation creates a separate `metrics-token` file and matching
`EXECUTION_METRICS_TOKEN` setting. Keep that file private.

The optional Compose `monitoring` profile provisions Prometheus with
`deploy/execution/execution-alerts.yml` and Grafana with the dashboard and data
source in `deploy/execution/grafana/`. Pin both monitoring images by digest in
the private environment before enabling that profile. Their CPU/RAM/storage
overheads must be included in the final baseline and admission calibration.
Grafana and Prometheus bind only to localhost, on ports 3001 and 9093. Connect
through an SSH tunnel. Configure the existing operator Alertmanager destination
in `prometheus.yml` before relying on delivered notifications; the shipped rules
make alert states visible in Prometheus without contacting anyone.

Request scrape duration and metric-query duration distinguish control-plane
latency from database collection time. Completed dependency transfers record
duration, transferred bytes and local cache hits in the durable attempt outcome,
with hourly aggregates in metrics. Historical phase-level resource peaks remain
profiling evidence rather than complete per-attempt operational time series.

Tests use a dedicated disposable PostgreSQL database. The opt-in container tests
exercise cached prediction → blob publication → final CSV through real Docker
containers and a real 64 MiB cgroup OOM/requeue, without exhausting the host/GPU:

```sh
DJANGO_SECRET_KEY=test-only python -m pytest tests --ds=tests.execution_settings
WEBKINPRED_TEST_EXECUTOR_IMAGE=sha256:YOUR_LOCAL_IMAGE DJANGO_SECRET_KEY=test-only python -m pytest tests/test_execution_container.py --ds=tests.execution_settings
```

Release acceptance requires measured profiles on every advertised runtime,
the migration/recovery checks, and the canary gates above. Unmeasured shapes and
host/profile combinations wait rather than borrowing another host's envelope.
Child-unit subdivision beyond the current planning envelope and verified lazy
adoption of unversioned legacy embedding caches remain implementation limits.
Retained legacy files are not automatically trusted as current representations.
External alert delivery requires an operator-configured destination; Prometheus
and Grafana expose alerts locally without sending messages to third parties.

## Source and runtime verification

Release bundles record their Git commit and file checksums and include the
committed model adapters. They exclude model weights, user media and environment
secrets. Thin images reuse the installed environments; deployment-specific
weights remain in preserved read-only release trees.

The executor caches verified model digests against device, inode, size, mtime
and ctime in its private state directory. Changed metadata forces rehashing;
unchanged hardlinked files are verified once. A preserved mtime does not bypass
ctime checks. To prepare this cache without admitting work, run the executor
with `--verify-runtime-files --verification-threads 4`.

The reopen gate counts completed jobs from initial canary admission, but requires
one healthy hour after the latest executor recovery or explicit
`canary_healthy_since` timestamp. An outage cannot satisfy the observation hour.
Elapsed time beyond an unrenewed attempt lease is resource waiting rather than
active execution.

### Scheduled backups after the database migration

The legacy SQLite backup cron must be replaced during cutover. The execution
backup service runs `tools/executor/backup.py` with a private JSON configuration
at `/etc/webkinpred/backup.json`. It requires the dedicated database container,
role, three active database names, media root, scratch directory, minimum free
bytes, existing S3 prefix, AWS user and AWS binary path. The AWS user's existing
credential chain is used; credentials do not enter source control.

Install the provided `webkinpred-backup.service` and timer after preserving the
legacy script and cron. The service caps CPU at half a core and memory at 512 MiB.
Database dumps use bounded scratch space; authoritative `media/jobs` streams to
S3 without creating a full local archive. Shared embedding caches remain
regenerable. Archives larger than the validated 128 GiB streaming envelope hold
the backup for operator review. Every object is downloaded and checksum-verified;
the completion manifest is written last. Restore only prefixes with a verified
`manifest.json` and `complete: true`. No existing backup is automatically deleted.

Restore the three custom-format dumps into dedicated PostgreSQL databases,
restore `jobs.tar.gz` under the media root, restore the private deployment/model
configuration from the retained cutover evidence, then reconcile stopped attempts
before enabling admissions. Do not restore an old snapshot over jobs accepted
since that snapshot. The cutover evidence also retains the independent PostgreSQL
restoration rehearsal.
