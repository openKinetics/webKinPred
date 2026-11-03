# Execution follow-up, 3 November 2026, 14:40–14:55 UTC

The platform continued completing work after the previous monitoring report.
Both executors remained online, reconciled and unpaused. There was no observed
recurrence of the prolonged unreconciled-host dispatch stall. This was an
observational pass; no model workload was added, service restarted, job requeued,
profile changed or database maintenance applied.

## Throughput and concurrency

From 13:06 UTC to 14:55:26 UTC, **134 jobs / 18,427 submitted rows** completed.
The final 30-minute interval completed 36 jobs / 3,445 rows.
No newly failed job or OOM attempt was found in the inspected interval.

The initial history inventory, ending at 14:40:18, contained 1,019 successful
attempts: 739 on the GPU host and 280 on production. Allocation ownership averaged
2.09 GPU attempts, peaking at four; at least two GPU allocations overlapped for
82% of that interval. GPU-host CPU-only ownership averaged 1.15 attempts, peaking
at seven; production averaged 0.55, peaking at three. These include startup,
materialization and publication, and do not establish simultaneous GPU kernels.

## Actual resource use

| Measurement | Average |
|---|---:|
| GPU, historical 90-minute Prometheus window | 32.3% |
| GPU, fresh approximately 15-minute direct samples | 22.0% |
| GPU-host CPU, same direct samples | 22.0%CPU / 20 logical cores |
| Production CPU, separate contained 10-minute direct samples | 2.35 / 4 logical cores |

GPU utilization reached 100% and production CPU reached 3.99 cores. The means
establish substantial idle capacity despite those busy snapshots. CPU figures
are host-wide, including control services and other applications. Historical
90-minute observed attempt CPU use averaged 2.15 cores on the GPU host and
0.29 on production; these are different windows and sampling series from the
fresh direct measurements.

In the fresh 39 capacity snapshots, 29 had no dependency-ready CPU stages.
Available CPU capacity therefore often had no eligible scientific work to run.
No unreconciled host appeared in those snapshots.

No RAM/VRAM reservation breach appeared in the historical metrics. Brief CPU
reservation overshoots did appear; the largest captured in the early fresh
samples was approximately 0.2%. Running container CPU quotas remained enforced.
A 20-second cgroup check found no OOM kills or service restarts. GPU-supervisor
CPU use was about 0.23 cores in that check, so one separate near-100%-of-one-core
Docker reading is insufficient to diagnose a sustained supervisor CPU bottleneck.

## Recoveries and unresolved holds

At 14:40, the interval contained two interrupted attempts and one reconciliation
recovery. Two affected jobs had subsequently completed; the latest interrupted
KinForm stage had restarted. Missing-artifact recovery also completed four
previously deferred stages. Queue preservation and eventual recovery are working,
but these transient interruptions mean the period was not error-free.

**Six prediction stages on two existing jobs remain held** with
`dependency_plan_requires_attention`: two CatRange and four CataPro targets.
These are the same two requests involved in the previously identified
validation/planning mismatch. The increase from four to six held stages reflects
additional targets reaching the guard. The requests remain preserved, and the
underlying mismatch has not been repaired by this monitoring pass.

The final inventory had zero uncertain attempts, zero expired
live attempts and zero unpublished outbox events. Both hosts' most recent
successful completions were recent. Existing old-queue-age warnings remain active.

## Control-plane costs worth fixing next

The request-log sample since 13:06 contained:

- 3,034 successful production claim requests, averaging 0.743 seconds; these
  HTTP-200 responses include both admitted attempts and empty polls.
- 38 production claim requests returning HTTP 503, averaging 3.58 seconds.
- 1,716 successful GPU-host claim requests, averaging 1.11 seconds.
- 26 GPU heartbeat conflicts; the executor reconciles stale ownership rather
  than accepting stale commits.
- GPU completion requests averaging 1.05 seconds, with a 12.16-second maximum.
  Ordinary supervisor requests have a five-second client timeout. Slow commits
  can therefore lose their response even when the database eventually commits;
  idempotent completion and reconciliation are essential. File publication is
  already asynchronous; the database completion transaction still costs time.

In a 20-second sample PostgreSQL consumed 1.75 cores against its two-core limit.
It was CPU-throttled in 129 of 201 accounting periods, with 10.6 accumulated
throttled seconds. Multiple throttled processes can contribute to that duration;
it is not an extra ten seconds of whole-host outage. The public backend was
only briefly CPU-throttled in the same sample. Planner statistics were receiving
automatic updates; stale statistics alone were not established as the cause.

All 92 CPU-capacity deferrals counted at 14:47 occurred **before a container was
recorded**. Local rechecks are rejecting changed capacity, not failing running
scientific computations. This protects resources but also incurs unnecessary
admission/materialization overhead when repeated.

The initial batch inventory had 21 completed / 675 ready new shared batches;
by 14:52 it had 26 completed / 841 ready / one running. Planning was creating
future batches much faster than execution consumed them. The next targeted
control improvements should be:

1. Bound the per-profile backlog of ready cohorts before more graph rewrites,
   preserving urgent-request eligibility and existing queued batches.
2. Back off empty admission polls, resetting promptly when local work finishes
   or ready-work notification arrives, while keeping lease renewal independent.
3. Reduce completion-transaction and ranking-query costs before increasing
   production worker counts or simply extending supervisor HTTP timeouts.

At the final inventory, 27 of these batches had completed and 889 remained ready.

The length-outlier partitioning, measured larger microbatches and bounded warm
model workers described in the preceding report remain further opportunities.
No throughput gain from these unimplemented changes is claimed.

## Storage, interface and queue integrity

There were 316 acknowledged GPU-host cleanup operations and 148 on production
since 13:06 at the 14:47 check, with no pending cleanup then. Free disk stayed
above the configured safety floors: historical minima were approximately
166.9 GiB on the GPU host and 18.3 GiB on production. No manual cache/project-file
deletion was performed.

All **9,431 original unfinished-job IDs** remained present and all **11,676
original specification/input checksum pairs** were unchanged. The frozen legacy
Redis queue retained 5,002 messages. The active control broker used database 3;
its notification queue and durable outbox were empty at the check. Durable work
continues to reside in PostgreSQL, so Redis message counts need not equal the
number of queued prediction jobs.

The fresh sampling window issued **116 successful public status/health checks**,
with zero failures, 0.37-second average latency and a 1.43-second maximum.
A separate final health request returned HTTP 200 in 0.13 seconds.

Five completed-result downloads returned HTTP 200 with the expected row counts;
the About endpoint returned refreshed statistics. Authenticated GPU health
reported online. Prometheus recorded successful scraping throughout the preceding
90-minute window. No control images, resource budgets or scientific profiles were
changed during this follow-up.

Private evidence is retained in the hosts' `webkinpred-execution-staging`
directories: `audit106-initial/final.json`, `prom106.json`, `recovery106.json`,
`http-log106-summary.json`, `cgroups106.json`, `preservation106.json`,
`broker106.json`, `public106.json`, and the capacity/HTTP/direct CPU/GPU sampling
files. Source code remains at the deployed `89e42b0` runtime; this report does
not require another application deployment.
