# Capacity and concurrency audit, 2 November 2026

The 14:45:42 UTC/Dublin inventory was read-only and took 6.56 seconds. It covered
current hosts/profiles/attempts, grouped dependency-ready stages and a bounded
sample per profile. It exported 898 attempts started in the preceding two hours,
68 processing jobs and their unfinished stages. No job was restarted, merged,
reprioritized or requeued, and no capacity/profile setting was changed.

## What is happening

The inventory contained 9,374 Pending, 68 Processing and 7,521 Completed jobs.
Only three attempts were owned at that observation: two GPU embedding stages and
one production CPU prediction stage. Attempt ownership includes materialization,
model startup and publication; it does not mean the GPU or CPU is executing a
model for that whole interval.

| Resource/policy | GPU host | Production |
|---|---:|---:|
| Logical CPU count | 20 | 4 |
| Configured aggregate attempt CPU ceiling | 16 | 3 |
| Configured CPU-only attempt slots | 8 | 3 |
| Configured GPU attempt slots | 5 | 1, no GPU profiles advertised |
| Attempt RAM ceiling | 96 GiB | 10 GiB |
| Observed available host RAM | 118.9 GiB | 15.9 GiB |
| Observed free disk | 175.5 GiB | 22.1 GiB |

There are 45,710 dependency-ready embedding stages. “Dependency-ready” means their
stage prerequisites have completed, not that every physical blob has already been
rechecked or that the reservation/compatibility rules permit admission.

There is concurrency, but considerably less than the configured ceilings:

| Two-hour ownership measure | GPU host | Production |
|---|---:|---:|
| Average owned attempts, all types | 2.12 | 0.38 |
| Average GPU attempts | 1.66 | 0 |
| Peak GPU attempts | 4 | 0 |
| Average CPU-only attempts | 0.45 | 0.38 |
| Peak CPU-only attempts | 3 | 2 |
| Average reserved logical CPUs, all attempt types | 3.99 | 0.36 |

At least two GPU attempts overlapped during 61.8% of the two-hour window, including
setup and publication. These figures do not measure GPU kernel overlap. There
were 579 successful embedding attempts in that interval, roughly 290 per hour.

A separate subsequent 60-second, one-second-resolution GPU-host probe observed
5.67% average GPU utilization, a 72% peak, and 43 zero-utilization samples. Mean
VRAM use was 2.49 GiB with a 6.99 GiB peak. Total host CPU consumption averaged
approximately 1.85 logical CPUs. This short probe establishes substantial gaps in
that minute; the user's earlier 95% reading establishes a busy compute interval.
Neither observation alone describes the full day's utilization.

## 1. Exact GPU-combination approvals prevent otherwise fitting work

The GPU host advertises 23 usable GPU profiles. Its 73 approved combinations
include only 17 mixed pairs and six mixed triples; many approvals repeat the same
profile two to five times.

The captured active pair was:

- KinForm-H ESM-C residues: 6.24 GiB VRAM reservation.
- KinForm-L ESM2 residues: 4.28 GiB VRAM reservation.

The pair was approved. No approved three-or-more-profile combination contained
that exact pair. A further Pseq2Sites request, requiring approximately 1.15 GiB,
passed the numerical CPU/RAM/disk/VRAM admission calculation in the captured
state, but failed the exact combination whitelist. EITLEM, OmniESI and several
other fitting requests were likewise excluded by that policy.

Raw free VRAM is not all available for new claims: reservations already issued
for models still loading must also be honored. This is correctly included in
`api/execution/resources.py`. Even after applying that rule and GPU headroom,
there were fitting requests rejected solely by combination compatibility.
Raising `gpu_slots` beyond its existing value of five would not remove this gate.

## 2. A dependency failure suspended CPU-only KinForm-L pooling globally

`gpu16-KinForm-L-kinform_pool` is validated but suspended. It consumes one CPU and
about 1.09 GiB reserved RAM, with no VRAM allocation. At the inventory there were
3,330 ready stages referencing it, of which 14 had completed their prerequisites.
Those 14 therefore offered CPU work, but could not pass profile eligibility.

Job `RRIdoOq` has 100 rows. Its pooling stage failed at 13:29, 13:32 and 13:34 while
materializing required artifacts, with `missing_embedding` and “Required artifact
must be rescheduled.” The exception originates before model launch. The failure
handler treats three matching failures as sufficient to hold the stage and
suspend its whole profile, even for this dependency error. Earlier attempts using
the same profile completed successfully; this evidence does not identify an OOM
or an invalid CPU resource envelope.

The profile suspension prevents unrelated consumers from making progress. For
example, `jEOm0w0` is a 100-row KinForm-L job with a dependency-ready pooling stage.
`SglY0HR`, a 1,000-row KinForm-L job, had a processing age of approximately 72 hours
and 32 stages reporting `profiling_required`.

The repair should distinguish missing-data recovery from profile execution
failure: recover/verify the specific missing dependencies, preserve checkpoints,
and keep healthy consumers eligible. A profile-wide circuit breaker should not
be tripped by this pre-execution condition. Re-enabling the profile should follow
verification of the affected materialization/recovery path; this audit did not
blindly unsuspend it.

## 3. Artifact RPC overhead makes production CPU execution unnecessarily slow

A 100-row CatPred prediction attempt on production took 368.6 seconds. Its
materialization alone took 341.2 seconds for **6,468,000 bytes**, with 1,001 artifact
references and only one local hit. A second 95-row attempt took 360.1 seconds,
including 330.0 seconds transferring 6,144,600 bytes across 950 remote artifacts.

The executor processes artifacts sequentially: for each missing artifact it
fetches the blob, verifies/publishes it, then performs a control-plane location
acknowledgement. Hundreds or thousands of tiny files incur large HTTP/database
round-trip overhead even when their total byte volume is small.

GPU-host CPU attempts with the same 95/100-row shapes and lengths used the
locally generated embeddings. Their observed durations were approximately
14–17 seconds, with about 0.3–1.3 seconds materializing inputs. These are
observational examples rather than a controlled inference benchmark; the
production transfer overhead is directly recorded and dominates those attempts.

The current locality preference applies only above 8 MiB total input bytes and
expires after three seconds. Consequently, a thousand-file, 6.5 MB request gets
no locality preference. A slightly slower GPU-host claim can lose that work to
production and create minutes of transfer overhead while holding reservations.

Recommended fixes are stronger placement based on file count and measured
transfer cost, batched location acknowledgements, and a bounded immutable
artifact-pack transfer protocol. Keep per-file checksums, authorization,
publication fencing and retryable partial progress. Send the small completed
results to production for normal assembly and downloads.

## 4. Setup, microbatching and admission cadence leave gaps

The active protein profiles contain at most 16 sequences per stage and use model
microbatch size one. Each attempt launches a new container/interpreter/model.
Successful GPU-host embedding attempts typically took about 15–28 seconds
including setup and publication. GPU-host CPU merge/pooling tasks also paid
roughly 8–16 seconds per isolated attempt.

The agent claims at most one additional stage per outer loop. In a checked
15-minute backend log window, GPU claim requests had a 2.745-second median and
4.486-second p95. Production claims had a 1.297-second median, with five retryable
503 responses among 325 requests. Those are control requests, not model profiling
runs. Repeated queries and launch/setup latency matter for these short stages.

Keep admitted models resident across bounded batches, profile larger model
microbatches, and fill host capacity through bounded batch admissions without
blocking heartbeat supervision. Reuse compatible representations across
KinForm-H/L and OmniESI/O2DENet, as identified in the earlier bulk audit. Preserve
original job-specific results, options and molecular positional context.

## Recommended rollout order

1. Repair artifact-recovery/profile-suspension isolation and restore verified
   CPU pooling work. Investigate all missing inputs together rather than allow
   sequential misses to disable a healthy method.
2. Prefer GPU-host CPU execution for inputs already local there, using input
   count/round-trip cost as well as byte volume. Batch transfer and location RPCs.
3. Profile the high-value mixed GPU triples, especially the captured ESM-C/ESM2
   pair plus Pseq2Sites. Approve only measured combinations and improve packing
   within the required foreground/background priority policy.
4. Introduce model residency and length/batch-specific measured profiles. Compare
   throughput and job latency for batching versus extra processes; 95% GPU busy
   during one forward pass does not prove either strategy will be faster.
5. Optimize admission batching and small CPU stage execution. Preserve atomic
   reservations, per-attempt tokens/checkpoints and five-second heartbeats.

A useful initial measured target is three fitting GPU allocations alongside up
to four two-thread CPU prediction attempts on the GPU host. That example uses
about 14 reserved logical CPUs, within the existing ceiling of 16. Actual
concurrency remains determined by measured VRAM/RAM/disk envelopes and throughput.
Smaller profiles may fit more; three large GPU profiles may not fit together.
There is no need to increase the existing worker-slot ceilings to start this work.
Do not relax RAM/VRAM guards or restore CPU embedding fallback.

Full residue matrices remain transient work products. Clearing downstream CPU
stalls before admitting many more producers also prevents their outputs from
accumulating behind blocked consumers. Permanent vector caches and protected
research residue files retain their existing policies.

## Monitoring regression corrected during this audit

The About statistics worker introduced in the previous change used advisory lock
481726395, which was already the metrics scrape mutex. Its session-level lock
made every scrape return 503 immediately. This caused a metrics-history gap;
therefore this report uses database attempt history and the direct GPU probe,
not an invented one-hour GPU-utilization average.

The worker now uses its own lock, 481726396. Source `1e98b52`, thin image r65,
was deployed by restarting only the About worker. The scientific executor agents,
active attempts and scheduling/profile budgets were unchanged. A regression test
holds the worker lock on a separate PostgreSQL connection while successfully
scraping metrics. The full suite passed 518 tests and three subtests. Prometheus
subsequently reported UP with no scrape error at 15:06:17 UTC.

Evidence: `/home/saleh/webkinpred-execution-staging/audit64*`, release65 image/build/
deployment records on production, and
`/home/msp/webkinpred-execution-staging/gpu64-samples.json` on the GPU host. Audit
scripts were bounded, read-only against job data, and performed no model probes
or queue mutations. The only runtime change was the monitoring-lock correction.
