# Resource-based GPU admission, 3 November 2026

The mandatory GPU model-combination allowlist was removed at the user's request.
Admission uses each individually validated profile's full peak reservation, remaining
reservations of already admitted workers, and host/device headroom. Historical
combination metadata is retained for compatibility with existing supervisors but has no
effect on admission.

## Implementation and deployment

Runtime source: `1f15812b76cf5af10cd7b5658e91acca3f0c1dbe`. Control image:
`sha256:7c6c85977ceb8765c5ddb191fe43956523057d617a1ffccc3b97299f6458bda9`. Source was
committed and pushed before deploying the thin control image.

The gate was removed from candidate selection, final admission and foreground preemption
calculations. CPU, RAM, VRAM, temporary-disk and GPU-slot budgets, individual
profile/shape/runtime validation, ownership fencing, GPU-only embeddings and
foreground/background priority rules remain enforced. The GPU allocation cap stays at
five for this rollout; no capacity ceiling was raised. The resource admission
calculation, including unallocated future model peaks and at least 2 GiB/10% GPU
headroom, is unchanged.

Only the backend, execution controller and control worker were replaced. Neither host
supervisor nor scientific model image was replaced; no admission drain, queue purge or
model-environment rebuild was requested. One transient GPU-agent connection reset
occurred during backend replacement. Subsequent health checks found both hosts
reconciled/unpaused, zero expired leases and zero held stages. The retired
`unvalidated_gpu_combination` waiting-label cleanup matched zero rows.

The isolated PostgreSQL/container suite passed **647 tests and three subtests**. New
integration coverage admits unrelated GPU profiles with missing, empty or irrelevant
historical mixture metadata; refuses startup-peak overcommit under concurrent claims;
permits CPU work alongside GPU allocations; and retains the individual-profile
validation gate. GPU fairness/priority tests now exercise actual VRAM/slot limits
instead of allowlist exclusions. Focused lint and whitespace checks passed.

## Measurements

A fresh ten-minute baseline completed before building or deploying. Post-change sampling
covers fifteen minutes. Both periods use one-second device/host CPU samples,
approximately twenty-second capacity snapshots and bounded public status/health probes.
No scientific benchmark workload was added. Job/stage history is measured over the
production CPU-sample timestamps; GPU-host clock offset does not affect the
duration-based GPU averages.

Baseline: **17:25:50–17:35:50 UTC** (600 direct samples per host). After deployment:
**17:38:22–17:53:22 UTC** (900 direct samples per host).

| Measurement | Before, 10 minutes | After, 15 minutes |
|---|---:|---:|
| Mean GPU utilization | 40.3% | 5.3% |
| GPU samples at 0% busy | 44.7% | 62.2% |
| GPU-host CPU, mean cores / 20 | 2.71 | 1.86 |
| Production CPU, mean cores / 4 | 2.56 | 2.10 |
| GPU-host observed attempt CPU, mean cores | 1.98 | 0.93 |
| Production observed attempt CPU, mean cores | 0.22 | 0.00 |
| Maximum GPU allocations observed | 5 | 5 |
| Capacity snapshots with different GPU methods allocated | 0 / 26 | 3 / 39 |
| Snapshots without dependency-ready CPU work | 11 / 26 | 34 / 39 |
| Successful stage attempts | 133 | 283 |
| Embedding output artifacts committed | 4,005 | 3,731 |
| Prediction work units processed | 6,656 | 3,841 |
| Completed jobs during the exact interval | 0 | 0 |
| New stopped/error attempts | 15 | 0 |
| Public health/status checks, failed / total | 0 / 76 | 0 / 116 |

**Removing the allowlist alone did not improve measured utilization in this interval.**
The before period included heavier protein work; the after period was dominated by small
molecular stages. These totals are deliberately reported without selecting a later
GPU-busy snapshot as a replacement for the full window. Output-artifact and work-unit
counts span different methods and are not comparable scientific throughput units.
Successful attempt counts are not job counts; a large request can advance without
reaching final assembly.

The after-period snapshots explicitly contained CatPred and molecular-UniKP allocations
together, which demonstrates mixed admission without an allowlist. These allocations
include setup and transfers: the snapshots do not establish simultaneous GPU kernels or
that both models were resident at that moment. A live check of the backend confirmed
that `_gpu_profile_allowed` is absent.

The remaining five-slot cap can be occupied by short molecular attempts that spend
substantial time starting processes and loading models. A checked 49-sequence protein
cohort from the same request passed profile shape, target, runtime and
advertised-capability checks; older molecular stages still existed in that request's
queue. FIFO stage ordering within that request can therefore fill allocations with those
small stages while a heavier embedding remains eligible. Many lower-priority embedding
stages also fit the base resource checks, but the requested normal-over-bulk policy
still applies. Keeping encoders resident and improving stage choice within the same job
are further performance work; neither is silently included in this policy change.

No stopped/error/OOM attempt or newly Failed job appeared in the exact after window.
Both hosts stayed reconciled and unpaused in all 39 capacity snapshots, with no
memory/I/O-pressure flag. The final interval inventory had zero held, uncertain or
expired-live attempts. The separate out-of-profile waiting stages are not included in
the zero-held count. Minimum sampled free disk was 18.53 GiB on production and 171.45
GiB on the GPU host, above admission safety floors.

Public checks averaged 0.37 seconds after deployment, with a 1.51-second maximum. The
post-deployment request-log summary extended slightly beyond the direct sampling window:
it contained one retryable production claim HTTP 503, no ownership conflicts, GPU
heartbeats averaging 0.065 seconds and GPU completion callbacks averaging 0.154 seconds
(0.387-second maximum).

The existing 17,010-row foreground UniKP request changes execution phase during these
windows, so this is observational before/after evidence rather than a controlled speedup
benchmark. Total host CPU includes databases, control services and unrelated
applications; observed attempt CPU is reported separately.

## Preservation and remaining constraints

All **9,431 original unfinished job IDs** remain present and all **11,676 original
specification/input checksum pairs** are unchanged. The frozen legacy Redis queue
retains 5,002 messages; the active notification broker remains separate from the
authoritative PostgreSQL work queue. Authenticated GPU health returned HTTP 200 and
online. Five existing completed-result downloads passed row checks and About statistics
returned successfully. No original input, completed result, model file or protected
research embedding was removed.

The separate 26 stages outside validated input-shape envelopes are unaffected. Removing
a model-mixture gate does not validate longer molecular inputs or larger
assembly/similarity shapes. The existing five-allocation cap, repeated model startup and
strict normal-over-bulk priority policy can still limit utilization. No warm model
service, profile-envelope expansion or priority-policy change is part of this release.

Private evidence is retained under both hosts' `webkinpred-execution-staging`
directories: release112 source/image/build/deployment manifests, `tests112.txt`,
`audit112-before/after.json`, admission explanation snapshots, bounded CPU/GPU/
capacity/HTTP samples, request-log summaries, queue-preservation and health checks. No
input sequences, job access identifiers or credentials are included here.
