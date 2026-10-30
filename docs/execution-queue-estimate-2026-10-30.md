# Queue completion estimate, 30 October 2026

The best current estimate for the **already queued batch** is approximately
**two weeks**, with a practical **10–20 day planning range** under the current
execution configuration. This is a workload forecast, not a promised finish date
or a statistical confidence interval. The live queue cannot empty if new work
continues arriving faster than it can be completed.

## Scope and measurements

A consistent read-only PostgreSQL snapshot at **14:47:57 Dublin time** covered
all 6,600 durable jobs, 90,252 stages, 184,084 dependency edges, 2,122 execution
attempts and 99 resource profiles. It contained **6,227 unfinished jobs**:
6,190 pending and 37 processing, representing **1,139,381 input rows**.

A lightweight refresh at **15:06:27** found **6,313 unfinished jobs**:
6,259 pending and 54 processing, representing **1,148,983 input rows**.
The stage model below uses the earlier complete graph; the later row-volume
increase is under one percent and is smaller than the forecast uncertainty.
No job state, scheduling limits, scientific options or quotas were changed by
this analysis.

The 19-hour outage is excluded from throughput calibration. The final repair
window starts at 14:18:55 on 30 October. Some method/step combinations had few
successful attempts in that short window, so their earlier successful stage
measurements were also used. Failed attempts and time merely spent waiting in
the public queue were not treated as successful model execution time.

## Why a jobs-per-hour extrapolation is too optimistic

The first 93 completed jobs after the final repair included 56 DLKcat and
29 MMISA-KM jobs: **91% were these faster CPU methods**. GPU-heavy methods
constitute most of the remaining queue. The observed 172–186 jobs/hour would
suggest roughly 34–36 hours for the snapshot batch, but that does not account
for the different remaining method mix.

A fresh 30-minute sample ending at 14:59 still showed 81 of 93 completions were
DLKcat/MMISA-KM. It completed 130 embedding stages, or approximately **260 stages
per hour**. Their summed successful attempt durations divided by the wall-clock
window gave an effective overlap of approximately **1.43 embedding attempts**.
This includes process startup and execution; it is not a count of independent
physical GPUs or a measurement of exclusive GPU-busy hours.

## Work remaining and calculation

The complete graph contains **53,671 unfinished embedding stages**, plus
18,352 prediction/pooling stages, 3,050 similarity stages and 6,628 assembly
stages. **2,648 preparation stages** are unfinished. Some preparation is partial;
already declared prediction rows were subtracted before estimating undeclared
work.

For each method and stage type, use the mean duration of successful post-repair
attempts when at least three are available; otherwise use successful attempts
from the concurrency rollout. Count existing stage IDs once, including their
shared producers and dependencies. Estimate not-yet-declared work from each
method's observed remaining embedding cost per planned input row and its
unplanned rows. Thus unprepared jobs are not counted as zero work, and already
declared partial chunks are not added twice.

| Component | Summed attempt runtime |
|---|---:|
| Already declared remaining embedding work | 301.5 hours |
| Estimated additional embedding work from unfinished preparation | 166.0 hours |
| Total embedding work | **467.5 hours** |
| CPU prediction, pooling, similarity and assembly work | 238.1 hours |

Attempt-hours can overlap on the same GPU. Applying the observed overlap gives:

```text
467.5 embedding attempt-hours / 1.43 effective overlap
= 327 elapsed hours
= approximately 13.6 days
```

CPU work overlaps GPU work. Its observed effective attempt overlap was about
2.2, giving a separate approximate 4–5 day CPU workload. Consequently, the GPU
pipeline dominates the overall estimate; the two durations are not added.

A separate count-based check projects approximately **83,400 embedding stages**
after completing preparation. At 260 stages/hour this is approximately
**13.4 days**, consistent with the duration-weighted calculation.

## Remaining method mix

This table uses the complete graph snapshot. The one mixed CatRange/UniKP job
is grouped with CatRange for workload attribution. Input rows are not expanded
prediction-unit counts.

| Method | Unfinished jobs | Input rows | Projected embedding attempt-hours |
|---|---:|---:|---:|
| KinForm-L | 602 | 101,797 | 116.2 |
| KinForm-H | 602 | 101,811 | 113.4 |
| CatPred | 590 | 86,399 | 56.7 |
| IECata | 599 | 101,120 | 35.0 |
| OmniESI | 601 | 101,412 | 32.8 |
| OmniESI-O2DENet | 599 | 101,217 | 30.3 |
| EITLEM | 599 | 101,408 | 28.1 |
| TurNup | 584 | 85,716 | 24.4 |
| CatRange | 602 | 101,632 | 19.1 |
| CataPro | 60 | 61,981 | 10.8 |
| UniKP | 50 | 47,286 | 0.7 |
| DLKcat | 458 | 85,882 | 0.0 |
| MMISA-KM | 281 | 61,720 | 0.0 |

KinForm-H and KinForm-L together account for approximately **49%** of projected
embedding work. Per-method hours here describe work volumes, not independent
per-method completion dates: all methods share the same GPU and scheduling
budgets.

## Planning scenarios

| Scenario | Assumptions | Approximate elapsed time |
|---|---|---:|
| Optimistic | 15% less work through reuse/shorter cases; overlap improves to 1.8 | 9–10 days |
| Central | Measured per-stage durations; current overlap of 1.43 | 13–14 days |
| Conservative | 20% more work/overhead; overlap drops to 1.2 | 19–20 days |

For planning, round this to **10–20 days**, centred on **two weeks**: roughly
**9–19 November**, with a central estimate around **12–13 November**. This assumes
normal host availability, comparable external GPU contention, no new submissions
added to this batch and resolution of the existing preparation hold. It does not
assume higher concurrency settings or future optimization work.

The main uncertainties are cache reuse and expansion in unprepared requests,
the limited recent sample for slower methods, model loading/transfer overhead,
and future shared-GPU availability. The scenario bounds are sensitivity checks,
not guarantees.

## New arrivals and the meaning of an empty queue

At the 15:06 refresh, the preceding hour had **370 submissions** and
**166 completions**. If that continues, the live queue grows rather than empties.
The forecast above concerns the batch already present; it is not an empty-queue
date under continuing arrivals. Late submissions can also compete with that batch
for shared artifacts and resource-dependent work.

One preparation job was already held before the incident repair. A literal
all-jobs-finished date also requires resolving that hold; its operator waiting
time cannot be predicted from normal execution throughput.

Raw private observations remain on production under
`/home/saleh/webkinpred-execution-staging/eta46/`. Local analysis intermediates
are under `/tmp/webkinpred-eta46/`. The full queue graph and method-level duration
samples were used, rather than projecting only the latest completed jobs.
