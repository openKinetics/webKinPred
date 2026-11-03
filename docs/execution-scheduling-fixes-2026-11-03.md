# Validation and scheduling fixes, 3 November 2026

## Deployed changes

Source commits `21cf49a`, `01056b6`, `977846b`, `c47cf48` and `f5ae007` fix the two
preserved validation holds, excessive advance batch planning, repeated empty admission
polls, slow artifact publication and expensive progress updates in lease-sensitive
requests. Source was committed and pushed before deployment. Scientific runtime images,
model environments, weights, resource profiles, reservation ceilings, GPU-only embedding
policy and bulk-request precedence were retained.

### Single-molecule input validation

Both affected requests contained a semicolon-separated pair of molecules in the singular
`Substrate` column. Planning correctly did not declare embeddings for that unsupported
single-molecule text. The legacy generic runner instead validated each fragment, then
requested the missing embeddings. Recovery could not repair that graph, because there
were no declared missing artifacts.

Preparation now gives generic single-molecule runners a private, immutable input copy
with unsupported chemistry cells blanked. Their existing validation path then emits the
ordinary invalid-substrate row result before requesting embeddings or loading a model.
Valid chemistry, including dot-separated components, is unchanged. Explicit plural
`Substrates` inputs retain expansion/reduction behavior. This is a compatibility
boundary for the already deployed scientific images; it does not require replacing those
environments.

Original uploaded files and checksums remain authoritative. Final assembly is explicitly
attached to original chunks rather than validated runner copies, so result downloads
retain original input text, row count and ordering. The input validation policy and
affected local row positions are recorded in stage payloads.

The new operator command repairs only selected inactive jobs held with
`dependency_plan_requires_attention`. It verifies the original input checksum, checks
that unsupported single-molecule inputs explain the hold, refuses live attempts and
unrelated holds, and changes no immutable request or quota values. The repair is
transactional and repeatable:

```sh
docker exec webkinpred-backend-1 python manage.py execution_repair_validation "$JOB_ID" --dry-run
docker exec webkinpred-backend-1 python manage.py execution_repair_validation "$JOB_ID"
```

Both jobs are now **Completed**. Each download returned HTTP 200 and contained all 13
original rows and substrate strings, 13 invalid-substrate reasons and zero numeric
kinetic predictions. Requested similarity columns are still present. Each original
13-row quota reservation received exactly one 13-row refund from its original account
bucket. No fabricated prediction or silent reinterpretation of the two molecules was
used. To request predictions for each molecule, submit the supported plural `Substrates`
schema.

### Bounded advance planning

The coalescer now maintains a target buffer of eight ready/running enlarged stages per
measured profile before creating more background batches. The check reads at most eight
IDs. Paused or completed owning executions do not count.

When the buffer is full, normal requests and their inherited prerequisites can still
coalesce; ordinary smaller producer stages remain eligible. When a batch finishes, the
background buffer can refill. The existing four-batches-per-pass limit, profile
rotation, representation/context checks, untouched-source rule and ownership/consumer
rewrites remain in force.

Existing queued cohorts were preserved, including those above the new buffer size. The
change stops unnecessary new rewrites while those queues drain; it neither deletes
batches nor restores canceled source producers.

### Empty admission polling

Executors back off after consecutive empty claim replies by 1, 2, 4 and then at most 5
seconds. A local ownership change, successful claim or resume clears the idle delay.
Saturated hosts continue polling for foreground preemption within that bound.
Five-second lease supervision, local fencing deadlines, telemetry and cancellation
handling continue independently of claim backoff.

### Artifact publication and progress costs

Successful completion requests for shared batches were taking as long as 33.97 seconds
for 1,188 output artifacts and 29.90 seconds for 990 artifacts. Publication now
validates the complete manifest, locks existing records in deterministic order and bulk
creates/updates artifact records and verified locations in one transaction. Checksums,
ownership fencing, producer identity and retention rules remain enforced. A late
conflicting checksum rolls back the entire publication. The 600-output regression test
bounds database work below 35 queries.

Shared-batch progress fan-out now uses a durable, coalesced outbox event instead of
scanning every consumer during completion or first-running heartbeat. Ordinary requests
above 1,024 input rows use the same mechanism, with a 15-second refresh cooldown; final
assembly bypasses that delay. This follows a live 17,010-row request whose full progress
scan was adding several seconds to repeated small molecular-embedding updates. No
scientific profiling was run to diagnose this: the measurements were control-request and
SQL timings against existing work.

Projection reads stage counters before briefly locking the public job row. It holds no
outbox lock during the scan, uses generation checks to retain updates arriving
meanwhile, and refuses snapshots superseded by newer progress or final publication.
Optional database contention leaves the durable event pending. The controller remains
the only projection consumer; normal outbox processing excludes these events. Small job
progress retains its existing immediate path. JIT and parallel planning are disabled
locally for lease-sensitive queries.

A saturated host now checks whether any advertised profile can fit either its current
reservations or the capacity remaining after eligible background work stops, before
ranking queued stages. If neither can fit, the claim returns empty without a global
candidate sort. The full priority/preemption path remains in force whenever capacity is
possible; the query-bound regression test covers this short circuit.

## Deployment and preservation

The control plane was deployed first, then the selected validation holds were repaired.
Supervisor replacement was gated on zero live database attempts and an empty
running-container inventory for that host. Both replacements recorded zero
running-attempt interruptions, retaining rollback containers and images.

Production admissions remained paused longer than intended during the first supervisor
drain: approximately **15:21:10–15:57:46 UTC**. Existing attempts were allowed to finish
and GPU-host execution continued, but production could not admit final assembly or other
CPU work during that interval. Admissions were restored. The remaining GPU-host drain
used a deadline with automatic resume in `finally`; GPU admissions resumed at
approximately 16:00:31. Both hosts were subsequently verified online, reconciled and
unpaused. These pauses must not be included as a steady-state throughput benchmark.

| Component | Pinned image |
|---|---|
| Backend, controller, control worker | `sha256:257407c8b5067a513920c8eb669100329792995f11df2700e1c64eb5cd5d0257` |
| Production supervisor | `sha256:9be60d35034b49d5a64b63378f7233f2911c035328534197d99c9bbaab90520c` |
| GPU-host supervisor | `sha256:51e42858715791ae8b65b1de27eabace6829f0e481e26216655545cb02b95f12` |

No broker queue was purged or quota reservation charged again. All 9,431 originally
unfinished job IDs remain present and all 11,676 original specification/input checksum
pairs are unchanged. The frozen legacy Redis queue still contains 5,002 messages; the
active database-3 control queue is a notification channel, not the authoritative
prediction backlog.

## Verification

The isolated PostgreSQL/container suite passed **644 tests and three subtests**; focused
lint and whitespace checks passed. New tests cover full background buffers,
normal-request precedence, bounded refill, idle-poll timing and reset, all-invalid
prediction without model/embedding access, mixed valid/invalid rows, plural inputs,
original assembly inputs, repair idempotency and refusal of live ownership. Additional
tests cover bounded publication queries, atomic checksum conflict rollback, durable
shared progress, updates arriving during a scan, large job cooldown, forced final
refresh and rejection of stale progress snapshots.

In the 16:50:00–16:58:15 UTC interval, two jobs containing 1,001 submitted rows
completed, alongside 114 successful stage attempts. GPU allocation ownership peaked at
five, with two or more allocations overlapping for 85% of the window. Both hosts
performed CPU work. Allocation ownership includes setup and artifact publication; it
does not establish simultaneous GPU kernels.

For 79 molecular-UniKP completion requests in the checked sample, mean callback latency
was 0.214 seconds and the maximum 0.512 seconds, compared with repeated 5–6-second
callbacks before large-job progress was deferred. One 1,188-output KinForm-pool
completion took 5.57 seconds, down from the earlier 33.97-second example. These are
observational examples under changing load, not a controlled throughput benchmark. The
large publication still exceeded the ordinary five-second client deadline; durable
completion/reconciliation remains required.

After 15:20, only one additional shared-batch specification was created through 16:58,
while existing batches continued to complete. The inspected cohort series had 56
completed and 1,170 ready batches at that point, including work planned before the
buffer limit was deployed. Existing overfull queues were deliberately retained rather
than unpacked or discarded.

The final database observation, **17:00:00–17:07:02 UTC**, recorded five completed jobs
/ 3,123 submitted rows and 99 successful stage attempts. GPU ownership again peaked at
five allocations; GPU-host CPU-only ownership peaked at five attempts, and production
CPU-only ownership peaked at two. Two or more GPU allocations overlapped for 83% of the
interval. Direct six-minute samples averaged 16.2% GPU busy, 3.6 of 20 GPU-host CPU
cores and 3.02 of four production CPU cores. Workload mix changed between windows, so no
controlled speedup is inferred.

Across 1,062 internal executor requests in that final interval, one production claim
returned retryable HTTP 503. There were no heartbeat ownership conflicts. GPU heartbeats
averaged 0.122 seconds; GPU completion callbacks averaged 0.307 seconds with a
1.001-second maximum. CPU-capacity deferrals and priority preemptions persisted. There
was no new permanently failed job or OOM attempt.

At the final inventory both hosts were reconciled, unpaused and recently heartbeating,
with zero held stages, zero uncertain attempts and zero expired live leases. Three
durable outbox events were pending. Profile-required waiting work is counted separately
below and is not included in the zero-held figure. Disk was approximately 18.5 GiB free
on production and 172.2 GiB on the GPU host, above configured safety floors. All 48
bounded public health/status checks succeeded; mean latency was 0.60 seconds and maximum
latency 3.25 seconds. Authenticated GPU health returned HTTP 200 and online. Both
repaired downloads and five other completed downloads passed their row checks; About
statistics returned a fresh successful response.

The measured changes remove the identified unnecessary work. Remaining GPU utilization
is still affected by dependency readiness, stage startup, measured microbatches and
approved mixtures; this rollout does not establish maximum sustained utilization. The
earlier interval included ten retryable production claim timeouts, preemptions, one
unknown-classified stopped similarity attempt and two reconciled attempts. All six
checked `interrupted` attempts had `stop_requested=true`, confirming deliberate
foreground preemption. No new permanently failed job or OOM attempt appeared in that
interval. Local CPU rechecks also returned work to waiting before container creation
when capacity changed during transfer.

Direct GPU samples spanning approximately nine minutes around the progress fix averaged
only 4.2% GPU busy and 2.6 CPU cores on the 20-core GPU host. A large foreground
molecular request dominated this window, using short isolated encoder stages. Thus low
sustained GPU utilization remains a separate performance constraint: resident encoders
and validated larger bounded work units are still worth pursuing. More allocated
attempts alone do not prove effective GPU use. The stricter foreground/background policy
remains in effect; background work may be deliberately stopped to serve ordinary
requests.

A separate active 17,010-row UniKP request has **26 stages without matching validated
profiles**: seven molecular embedding stages with longer molecules, 16 CPU prediction
chunks exceeding the measured 257-character molecule bound, one similarity stage above
its 10,000-row envelope and two assembly stages. These are waiting for profiling, rather
than failed jobs. The scheduler must not silently widen these envelopes. Validating
those shapes or safely decomposing these stages is additional required work before that
request can fully complete; no claim of complete backlog/profile coverage is made by
this rollout.

Private evidence is stored under each host's `webkinpred-execution-staging`:
release/build/deployment manifests, `tests111.txt`, validation repair and download
checks, quota ledger inspection, preservation checks, `audit110/111-*.json`,
`http-log110/111-*.json`, completion-size and SQL timing checks, and bounded direct
GPU/CPU and capacity/HTTP samples. No private input sequences or credentials are
included in this report.
