# Embedding recovery and admission correction — 2 November 2026

## Recovery defect and correction

Legacy recovery copied a producer's full input batch but selected only one output.
For the 663-row EITLEM request XUCLM7I, the audit found 661 recovery stages with
10,461 sequence input slots for 661 retained outputs. Full residue matrices are
transient by design, so regenerating an evicted matrix can be necessary. Repeating
the original batch for each individual artifact was unnecessary.

Recovery now groups compatible missing outputs by their immutable producer context.
A durable RecoveryBatch record coordinates the current generation, including
across consumers. Protein mappings and measured shapes are narrowed to the
requested sequence outputs. Molecular rows retain their original positional and
encoder partition context, and binding tables retain their complete coverage.
Pinned profiles, weights, representation identities and execution images remain
unchanged.

Only a never-claimed ready batch can be extended. Live attempts, retry plans and
held stages remain immutable. Additional missing outputs use a subsequent
generation. Source templates have protected database references. All unfinished
consumers retain dependencies on the correct producer. Unspecified runtime cache
errors no longer invalidate every verified artifact on a host; the executor's
specific invalid-location reports drive recovery.

The operator command `execution_reconcile_recovery --dry-run --limit 128` previews
the conversion of legacy pending recoveries. Without `--dry-run`, it retires only
eligible pending stages, rewires dependencies and preserves completed/running work.
Already verified outputs are reused. Retired nodes remain as evidence and are
excluded from public outstanding-work/progress counts.

## Admission and monitoring

The old CPU admission query repeated large profile JSON comparisons per stage;
one captured query was approximately 120 KB. Compatible profile IDs are now
resolved from the small validated registry before stage selection. Atomic
reservations, row locks, target/shape/runtime checks, CPU/RAM/VRAM/disk guards,
normal-request precedence and background classification remain enforced.

Migration 0039 adds the recovery registry. Migration 0040 concurrently creates
indexes for ready stages by profile/time, completed-stage time, attempt completion
time and attempt outcome category. The latter removes the historical JSON scan
that was repeatedly timing out during metrics scrapes. No model environment or
weight was rebuilt. PostgreSQL table writes stayed available during index builds.

## Deployment and observed validation

The implementation is committed in 0a22a95, 495106e and 6927a49. Thin control-plane
images r57/r58 reuse the existing web image. Additive migrations ran before the
new backend, controller and short-task worker started. Model runtimes, executor
supervisors and artifact gateways were not restarted. Their image identities and
start times were checked before/after rollout.

The full suite passed 505 tests and three subtests. Coverage includes concurrent
consumers, shared outputs, narrowing protein batches, molecular positions,
immutable live/retry plans, held stages, subsequent generations, verified copies,
completed-job preservation, dry-run rollback and legacy dependency rewiring.
Migration-state checks and generated concurrent-index SQL were verified.

The live dry run found 57 pending OmniESI-O2DENet single-output recoveries. The
applied reconciliation replaced them with five batches: 8/8, 16/16, 16/16, 16/16
and 1/1 input/output counts. All five subsequently completed. The old tasks
contained 897 sequence input slots; the new batches used 57 for the same 57
required outputs. This is a measured plan/work-unit reduction, not a promise of
an equivalent end-to-end speedup.

Before rollout, 139 of 142 CPU claim requests in a sampled 15-minute window
returned HTTP 503. A checked post-rollout window had 98 successful CPU claim
responses and one retryable 503, with a 1.605-second median among successes.
GPU claims in that window had 42 successful responses. The independent 3-second
SQL timeout remains in place; rare contention still backs off safely. Prometheus's
execution target recovered from DOWN/503 to UP.

The preservation inventory verified all 9,579 previously unfinished jobs and all
11,656 original execution specification/input checksums. Both captured live
attempts completed naturally. Both hosts were unpaused and reporting fresh
heartbeats. The frozen legacy Redis queue retained 5,002 messages. Quota
reservations were not recreated and queues were not purged.

Permanent pooled-vector and protected research-file retention was unchanged.
Full residue outputs remain transient and protected while unfinished consumers
or live attempts require them.

Private evidence is retained on production under
`/home/saleh/webkinpred-execution-staging/`: preservation57.json,
verification57.json, claim-timeout57.sql, claim-plan57 files, build/deploy57/58
logs and source manifests. The GPU host retains the supervisor start-time check.
