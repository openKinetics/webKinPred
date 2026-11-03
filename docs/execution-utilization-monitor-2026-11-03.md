# Live utilization and throughput monitoring, 3 November 2026

## Conclusion

Both hosts are executing work concurrently, but sustained utilization remains
below the available capacity, especially on the GPU host. Free memory and a
brief 95–100% GPU reading do not establish that the pipeline is saturated.
Production is substantially busier, including the database and web/control
services. Increasing its prediction-worker limit would compete with that work.

This monitoring followed the executor reconciliation repair documented in
[`execution-stall-audit-2026-11-03.md`](execution-stall-audit-2026-11-03.md).
No recurrence of the unreconciled-host latch was observed in the monitoring
samples. That does not prove the absence of every transient interruption.

## Measurements

Direct GPU-host samples use `nvidia-smi` and host CPU measurements at approximately
one-second intervals. Production CPU figures come from executor telemetry sampled
approximately every 20–25 seconds. CPU figures are logical-core equivalents for
the **whole host**, including other applications. They are not prediction-only
CPU consumption. The coarser GPU-host telemetry and the direct CPU samples are
different sampling series and should not be interchanged.

| Window | GPU mean | GPU-host CPU mean / 20 | Production CPU mean / 4 |
|---|---:|---:|---:|
| Initial 30 minutes | 69.0% | 3.41 | 2.73 |
| Subsequent 20 minutes | 42.3% | 3.30 | 2.97 |
| Batching rollout, approximately 12 minutes | 31.9% | 4.41 | 2.96 |
| Retry-guard window, approximately 10 minutes | 20.0% | 3.12 | 3.05 |
| Final heartbeat optimization, approximately 10 minutes | 22.4% | 2.86 | 3.02 |

These are changing real workloads, not an A/B benchmark on identical inputs.
The lower later GPU figures are material: this session has **not** demonstrated
maximum sustained utilization or a general throughput speedup from the changes.
The windows include imports, model loading, stage transitions, transfers and
publication, plus brief control-service deployments.

The initial window observed up to three GPU reservations, six CPU-only attempts
on the GPU host and two CPU-only attempts on production. The next window observed
up to four GPU reservations. Ownership includes setup and publication; those
counts do not prove simultaneous GPU kernels. A separate direct check found two
IECata embedding containers alongside three CPU prediction containers at 93% GPU
utilization. GPU execution stayed on the GPU host.

In the second window, **26 of 52 snapshots had no dependency-ready CPU stages**;
the largest observed ready-CPU frontier was eight stages. Available CPU cores
therefore do not imply a ready CPU prediction backlog. CPU demand arrives in
bursts as embeddings complete. Normal-request precedence, the measured mixture
allowlist, artifact locality and production-only final assembly also constrain
placement. A sampled active GPU mixture had no further validated compatible
allocation, even though some individual profiles passed the numeric capacity
check.

Completed public jobs in fixed database windows:

| Interval, UTC | Jobs | Submitted rows |
|---|---:|---:|
| 11:32:45–12:02:45 | 22 | 2,519 |
| 12:09:09–12:29:09 | 18 | 1,930 |
| 12:35:00–12:47:00 | 25 | 4,955 |
| 12:55:50–13:05:46 | 9 | 2,688 |

Submitted rows are not expanded prediction-unit counts. Completion rates are
affected by method mix, target count, caching and work begun before each window;
they must not be extrapolated into a confident queue-clearance estimate.

## Changes implemented and deployed

All source changes were committed and pushed before deployment. Only thin
control-plane images were rebuilt. Scientific worker images, model environments,
precision, batch-position context and resource envelopes were retained. Executor
agents and running scientific containers were not restarted by these deployments.

### Admission-query overhead

Commit `3823fa3` removes unnecessary work from resource admission:

- Resolve the small set of offline host IDs before testing artifact locations.
  An empty offline set no longer requires a negative scan over the location
  catalogue.
- Calculate verified cached-output bytes using an existence check. The previous
  join/distinct aggregate unnecessarily deduplicated complete artifact rows,
  including large identity and metadata JSON.
- Fetch only output IDs, checksums and sizes for ready-output serialization.
- Combine the oldest CPU and GPU ranking queries into one query while preserving
  CPU eligibility, priority and the GPU aging barrier.
- Add the partial `stage_unfinished_id` index through concurrent PostgreSQL
  migration `0044`. The live index was valid and approximately 2.7 MiB.

A comparison on the same repeatable-read database snapshot returned identical
byte totals. For 350 candidate artifact IDs, the old aggregate took 188 ms and
the replacement 29 ms; the warm repetitions took 84 ms and 8 ms respectively.
This demonstrates a query-level improvement, not a whole-job speedup. The earlier
3.64-second successful claim and a later 1.69-second no-fit claim are different
operations and are not presented as a controlled speedup.

### Packing compatible embedding work

Commit `37e76b9` fixes fragmentation in the existing measured cohort planner.
Long sequences outside the larger profile's validated length limit previously
interrupted greedy packing of compatible shorter stages. Such long stages are
now skipped during that packing pass and retain their original measured profiles.
The bounded candidate scan increases from 16 to 64 IDs, while the pass still
creates at most four batches and rotates profile families.

An early post-deployment sample contained 116 newly created shared batches,
averaging **114.9 sequences**, with a maximum of 128. This establishes improved
packing, not that all those batches had executed. Subsequent state counts are
recorded below. Sources already attempted, molecular positional batches,
representation identities, consumer links and scheduling class are preserved.

### Unchanged heartbeats

Commit `89e42b0` renews every ownership heartbeat but projects public job progress
only on the first running heartbeat or when the monotonic progress counter
advances. Completion and error paths continue to project their state changes.

Previously even unchanged heartbeats rewrote progress and, for shared batches,
recomputed multiple source jobs' summaries. PostgreSQL was observed at its
two-CPU container ceiling, with approximately 3.5 GiB accounted memory inside
its 4 GiB limit. An unchanged shared-batch heartbeat comparison, with both
variants rolled back, used **20 queries before and four after**. That particular
probe took 0.559 seconds versus 0.009 seconds. A separate ordinary-job probe used
13 versus four queries. These are small metadata probes, not scientific profiling
workloads or a sustained database-capacity benchmark.

### Shared intermediate retention

Commit `45b0e72` makes transient residues produced by shared protein batches
eligible for the existing fenced collector. Shared batch specifications have no
single owning job; the previous job-owned-only filter excluded them indefinitely.

Collection still requires no unfinished or active consumer and checks every
checksum alias. Permanent vectors and protected/project artifacts retain their
protection. This change does not traverse or delete legacy T5, ESM2 or ESM-C
research directories. Consumer-completion hints and the bounded sweep drive
collection through the authenticated control plane.

### Retry loops identified during monitoring

Commit `bc53d87` prevents repeated `missing_embedding` attempts from silently
looping when the declared graph contains no missing or pending artifact that
recovery can repair. After three consecutive stopped attempts with the same
profile fingerprint and error, that stage enters
`dependency_plan_requires_attention`. The job and request remain preserved;
the healthy execution profile is not suspended. Real missing declared artifacts
continue through ordinary recovery, even after repeated misses.

The inspected case supplied semicolon-separated molecules in the singular
`Substrate` column. Planning rejected the complete text as a single molecule;
the generic wrapper accepted its separately validated fragments and requested
undeclared embeddings. The request had only its CSV input attached. Retrying that
unchanged plan could not resolve the mismatch. Several held stages were observed
after installing the guard. **The guard contains the retry waste; it does not
resolve the underlying validation/schema mismatch or complete those requests.**
Resolving that mismatch requires aligning submission validation, planning and
the runtime adapter, including existing-request recovery. The original immutable
requests were not rewritten into a different substrate schema.

## Remaining opportunities, in order

1. **Measure the stage phases that account for GPU idle gaps.** Add lightweight
   timestamps for process/import startup, model load, forward execution, residue
   writes, transfer and publication. Correlate those with active database
   ownership. Stopped containers retained for recovery are not evidence that
   their reservations are still held. The existing agent already publishes
   asynchronously; a synchronous-publication diagnosis would be incorrect.
2. **Partition long outliers, then validate larger microbatches by length.**
   The bounded queue sample found many original 16-sequence units outside the
   128-sequence profile's length envelope. The deployed fix leaves those units
   intact. A further planner change could separate long proteins from their
   shorter neighbors, retaining the original envelope for the long partition
   and packing the short partition into existing larger profiles. It must
   atomically preserve every output producer and consumer edge and exclude
   context-dependent molecular/binding partitions. This has not been deployed.
    Packing 128 sequences into
   one work unit amortizes startup; it is distinct from increasing the number
   forwarded through a model at once. Benchmark microbatches 1, 2 and 4 for the
   high-volume CatPred and KinForm encoder families, including long-input
   boundaries and existing mixed workloads. Retain measured peak reservations,
   GPU headroom and numerical-context rules. Reuse the isolated profiling
   harness and copied inputs, without updating user jobs or scientific caches.
3. **Consider bounded warm model workers where startup dominates.** A supervised
   process can serve successive compatible protein work units while its full
   model reservation remains owned. It needs explicit draining/eviction for
   older or higher-priority requests, failure isolation, immutable encoder
   identities and no GPU ownership while awaiting another model's dependency.
   Keeping every model resident would consume the shared GPU and is not the
   intended design. This is not implemented by this session's control fixes.
4. **Reduce remaining control-plane work.** Profile the current ranking queries
   and public progress joins after the heartbeat change. An indexed durable
   ready-work frontier may eventually replace repeated ranking of a large
   dependency graph, provided graph rewrites, fairness, normal-request priority
   and crash recovery remain transactional. Avoid weakening the admission ledger
   to obtain lower dispatch latency.
5. **Resolve the validation mismatch described above.** This removes preserved
   attention holds as well as preventing future malformed single-molecule
   requests from consuming retry capacity. Do not silently reinterpret historical
   inputs or turn infrastructure cache misses into invalid scientific results.

No additional GPU mixtures, larger microbatch settings or CPU ceilings were
enabled without measurements. Production disk headroom is also limited: a later
snapshot reached approximately 17.8 GiB free against the 15 GiB safety floor.
One coarse I/O-pressure sample occurred. GPU-host model mounts were verified
local; no SSHFS mount appeared in the host mount inventory. A remaining mounted
SSHFS coordination path was therefore not established as the current bottleneck.

## Verification and final state

At approximately 13:06 UTC both hosts were online, unpaused and reconciled.
There were no uncertain attempts and the durable outbox was drained. Four stages
(one CatRange and three CataPro) remained in the new dependency-plan attention
hold. These are an explicit remaining limitation, not completed repairs.

All **9,431 original unfinished-job IDs** were present; all **11,676 original
request and input checksum pairs** were unchanged. At the preservation snapshot,
1,028 source jobs were Completed, 80 Processing and 8,323 Pending. No source job
was missing or newly marked Failed. Redis retained all **5,002 frozen legacy
messages**; the new control broker was database 3 with an empty notification
queue. An empty notification queue is compatible with a nonempty durable work
backlog: PostgreSQL and periodic reconciliation remain authoritative.

The five observation windows issued **620 successful public status/health
requests**, with no failed sampled requests. Five completed-result downloads
also returned HTTP 200 with their expected 78, 98 or 1,000 rows. The About endpoint
returned current statistics, the authenticated GPU health endpoint reported
online, and Django system checks passed. Prometheus continued scraping the
execution target successfully. Existing old-queue-age warnings remain relevant.

Of the shared batches created after the packing rollout, **three had completed
and 332 were ready** at the final batch inventory. Many improved batches remain
queued; this is another reason not to attribute a general throughput gain to
packing yet. A later direct check also found a long-protein stage of 16 sequences
and length 1,027 still using its original work-unit size, rather than being
forced into a shorter-length cohort profile.

The isolated suite passed **624 tests and three subtests**. New coverage includes
cached-output byte accounting across hosts, CPU eligibility under GPU waiting,
packing across out-of-envelope long stages, shared residue retention and research
alias protection, unrecoverable retry loops, and lease/public-progress correctness
when unchanged heartbeats skip projection. Migration checks found no ungenerated
model changes when the new index was introduced.

Final control source: `89e42b0783affc03bbb5de804982cdd7a5f1522c`.
Control image:
`sha256:d5affcdbf7f1db4baa465bcafe348a6ef492c7641b201a0dd4099bf2b084e45e`.
The production and GPU supervisors remain the release-100 images documented in
the incident report. There were no queue purges or model-environment rebuilds.

Private evidence remains in the hosts' `webkinpred-execution-staging` directories:
`capacity101/102/103/104/105.samples.jsonl`, matching HTTP and GPU sample files,
`cached-sum102.json`, heartbeat comparison records, progress/preservation audits,
release manifests and build/deployment logs. GPU originals are under
`/var/lib/webkinpred/gpu*.samples.jsonl`. No input sequences, credentials or public
download identifiers are included in this report.
