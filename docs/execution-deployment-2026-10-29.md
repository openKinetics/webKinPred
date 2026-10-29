# Resource-aware execution deployment — 29 October 2026

The production interface is open and all unfinished imported requests are
unpaused under the new scheduler. This rollout reused the existing model
execution environments; `Dockerfile.envs` was not rebuilt. Application and
supervisor fixes use small source layers on retained runtime images.

Production and GPU deployments use committed source bundles with file checksums,
source labels and immutable image IDs. The deployed application/supervisor source
is commit `74965f0`; scheduled backup operations were added in `c941484`.
Original dirty remote checkouts, deployment-specific model files and rollback
images remain preserved. Deployment does not require overwriting those checkouts
with a pull. Exact private configurations, image IDs and manifests are under
`webkinpred-execution-staging` on each host.

## Running services and admission limits

Production runs dedicated PostgreSQL 17, Django, the durable controller, a short
Celery control worker, an executor, an artifact gateway, Prometheus and Grafana.
The GPU host runs its executor, artifact gateway and durable embedding-service
compatibility proxy. Both hosts execute CPU prediction stages. Protein and neural
molecular embeddings require the assigned GPU; shortages wait instead of falling
back to local CPU embedding computation. Completed artifacts transfer by checksum
to production for final result publication.

| Budget | Production | GPU host |
|---|---:|---:|
| Allocated logical CPUs | 3 | 16 |
| Attempt RAM reservations | 10 GiB | 96 GiB |
| CPU attempt ceiling | 3 | 8 |
| GPU allocation ceiling | — | 5 |
| RAM headroom | 4 GiB | 16 GiB |
| Disk headroom | 15 GiB | 50 GiB |

These are upper bounds. Current availability, other workloads, unspent
reservations, input-transfer space, pressure signals, profile limits and the
validated GPU-combination allowlist can reduce admissions. GPU headroom is at
least 2 GiB or 10% of the device, whichever is larger. There are 24 advertised
production profiles, 62 GPU-host profiles and 73 validated GPU combinations. The
[measured resource table](execution-resource-profiles-2026-10-29.md) lists their
CPU/RAM/VRAM/disk envelopes and supported shapes.

Actual residency was observed for two CPU attempts on production, four CPU
attempts on the GPU host, and two simultaneous GPU embeddings. Parallel proof
checks require observed resource use and subsequently committed successful
attempts. No OOM, CPU-resource failure or reservation-envelope breach occurred in
the recorded parallel ramp. Other applications and unrelated GPU workloads remain
running.

The legacy prediction/cache workers and beat are disabled. Redis database 0
retains the original broker evidence; database 3 carries the versioned short
control queue. PostgreSQL owns scheduling and recovery. Broker-message counts
and unfinished-job counts describe different records and need not be equal.

## Preservation and verification

The private `cutover19/final-audit39.json` records the final audit at
`2026-10-29T16:36:49Z`:

* All 10,324 original job IDs, owners, submission times and immutable job metadata
  match the frozen source. Original completed-job identities remain unchanged.
* All 4,999 unfinished source requests have exactly one immutable execution
  specification. Original targets, methods, experimental overrides and options
  match their reconciled requests. All source input checksums were rechecked.
* All 5,002 source broker messages retain their exact envelope checksums and
  positions. Duplicates and terminal messages have explicit reconciliation
  dispositions. No queue purge or delete/recreate resubmission was used.
* All 4,999 legacy quota reservations have zero new charge. Ledger balances match
  their entries; refunds use the original reservation day.
* At the audit, 186 imported jobs were completed, 14 processing and 4,799 pending.
  No imported job was failed, no stage held and no attempt uncertain. These counts
  continue to change as the backlog runs.
* The interrupted source job retained its public ID and completed its restarted
  prediction chunk. Genuine boundary chunks and all 22 queued method/target
  combinations completed prediction stages during validation.
* A fresh CataPro HTTP submission completed in approximately 36.94 seconds. Both
  protein and molecular encoders ran on the GPU, the prediction ran on the GPU
  host, and production published the result. The downloaded CSV matched the
  authoritative file; API and interface status contracts passed. Quota was
  charged once. The synthetic probe credential was revoked after verification.
* The full isolated suite passes 394 tests and three subtests, including real
  PostgreSQL/Docker execution, OOM recovery, atomic admission, ownership fencing,
  idempotency, migration, quotas, artifact transfers and API contracts.

GPU numerical comparisons were explicitly waived by the user. CPU prediction
compatibility, representation identity, positional context, output shape/dtype/
finiteness and resource measurements remain required. Replacement profiles use
three fresh and five warm runs before activation.

## Measured serial and concurrent stage throughput

The following medians compare the same stage probes over three repetitions,
including process startup and publication. They do not predict whole-job speedups.
External shared-GPU activity was recorded and retained in the raw measurements.

| Stage | Concurrent allocations | Serial seconds | Concurrent seconds | Throughput ratio |
|---|---:|---:|---:|---:|
| UniKP ProtT5 | 3 | 62.98 | 31.29 | 2.01× |
| EITLEM protein encoder | 4 | 82.09 | 29.79 | 2.76× |
| OmniESI protein encoder | 3 | 53.08 | 27.80 | 1.91× |
| TurNup protein encoder | 4 | 98.52 | 33.76 | 2.92× |
| UniKP molecular encoder | 5 | 23.74 | 8.81 | 2.69× |
| KinForm molecular encoder | 5 | 15.88 | 5.59 | 2.84× |
| CataPro molecular encoder | 5 | 18.82 | 6.97 | 2.70× |

The selected caps come from those measurements. More processes were slower in
some configurations, so available VRAM alone does not determine concurrency.
Raw reports and combination registries remain on the GPU staging host under
`probes/concurrency12`, `concurrency16` and `concurrency18`.

## Recovery and rollout observations

The operator accepted 27 initial completed jobs as sufficient to move to actual
parallel execution. Reopening used the explicit `--parallel-ramp` acceptance
policy with recorded authorization and successful observed parallel execution.
It did not wait for 100 jobs at single-allocation limits. The default 100-job/hour
policy remains available for deployments without that authorization.

A GPU-host outage from approximately 10:20 to 13:12 UTC exercised fencing and
reconciliation. Completed artifacts survived. A stuck pre-launch transfer lease
was corrected so abandoned transfers are drained and confirmed absent before
reservations are released. Compute requires a fresh successful ownership renewal.

A missing assembly image caused a recoverable hold. The unchanged assembler was
reprofiled in a retained runtime: three fresh and five warm 10,000-row runs all
produced byte-identical CSV to the serial baseline. Approved runtime images now
have running, isolated keepers so pruning cannot remove them between attempts.

Backlog testing also corrected queue-page starvation and excessive admission
queries. Fitting CPU/GPU work is selected before the bounded scan while preserving
an older GPU request's fairness barrier. Eligible GPU waiting excludes dependency
waiting. Existing verified local embeddings no longer require redundant network
acknowledgements. Temporary canary deferrals of 1,534 stages were restored before
resuming the full backlog.

## Backups and operations

Frozen SQLite/Redis backups were verified off production, including actual
restoration. Custom PostgreSQL backups were independently restored on the
development host. Original sequence identifiers, including collision suffixes,
remain unchanged. Inactive rehearsal copies were archived and checksum-verified
before reclaiming production space; inputs, final results and model files were
excluded from that reclamation.

The legacy SQLite backup cron was preserved and replaced with
`webkinpred-backup.timer`, daily at 04:10 UTC. Its service backs up the three live
PostgreSQL databases and streams `media/jobs` to the existing S3 destination with
512 MiB memory and half a CPU of allowance. Every object is downloaded and
checksum-verified before publishing the completion manifest. No full local media
archive is created and existing backups are not automatically deleted. The
initial live run completed successfully: all five objects passed full download
and checksum verification in 8 minutes 39 seconds, with approximately 384.5 MiB
peak memory and no swap. Its completion evidence is retained in the private
cutover directory.

Prometheus scrapes the authenticated metrics endpoint; nine alert rules and an
18-panel Grafana dashboard are active. Monitoring binds to production loopback
ports 9093 and 3002. Routine deployments use the guarded execution deployment
script, one-shot migrations, pinned images, nginx reload and no automatic image
pruning. Original legacy consumers remain disabled.

Use `execution_admin` to pause submissions, pause/resume/drain a host, explain a
stage, retry held work or disable a profile. Preserve rollback images and cutover
evidence for 30 days. After accepting new requests, prefer reducing concurrency
or disabling a profile; never restore a frozen old snapshot over new jobs.

Requests outside measured envelopes enter an operator-visible hold. In particular,
very large per-parent expansions beyond the current 4,096-unit planning envelope
require further partitioning/profiling; the imported backlog's maximum expansion
is two. Unversioned legacy embedding caches remain preserved and are regenerated
when their representation identity cannot be verified.
