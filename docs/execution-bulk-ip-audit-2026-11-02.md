# Bulk submission audit, 2 November 2026

Read-only production investigation of submissions from `161.114.115.187`.
The job inventory is from 09:16 UTC/Dublin time. Detailed stage samples were
collected at 09:21 and dependency/artifact checks later in the same investigation.
These are successive observations of a running system, not a frozen migration
snapshot. No jobs, specifications, priorities, artifacts, quotas or execution
limits were changed.

## Inventory and workload

| Item | Count |
|---|---:|
| Jobs ever submitted from this IP | 10,038 |
| Completed | 2,287 |
| Pending | 7,730 |
| Processing | 21 |
| Unfinished | 7,751 |
| Unfinished input rows | 764,955 |
| Distinct protein sequences in unfinished inputs | 89,056 |
| Distinct input-file checksums among unfinished jobs | 2,626 |
| Unfinished jobs with 99 or 100 rows | 4,299 |

The IP accounts for approximately 81.2% of the 9,551 unfinished platform jobs
at the inventory observation. All 7,751 executions are unpaused and in the
background class, consistent with the requested lowest-priority policy. That
policy contributes to waiting whenever normal work needs resources. Its
unfinished submissions span 27–31 October.
Each queued row contains one protein sequence; there are no semicolon-expanded
protein lists in this cohort. Protein lengths have a median of 470 residues,
90th percentile 759, 95th percentile 785, 99th percentile 893 and maximum 1,098.
Lengths vary by method because the submitted datasets already obey their limits.
All unfinished jobs have substrate canonicalization disabled and no experimental
overrides in the saved specifications. Representative saved specifications enable
similarity columns and disable ReconXKG prediction memoization.

Inputs look like a large protein screening exercise across multiple prediction
methods. Exact chemical-field hashes identify four input encodings: the dominant
substrate in the single-substrate schema (605,160 rows), the same substrate in the
CatPred multi-substrate schema (78,009 rows), TurNup's full reaction including water
and products (77,786 rows), and a different substrate used by four older 1,000-row
KinForm jobs (4,000 rows). One representative row per chemical hash was checked
against its original CSV. Input `Name` values are opaque hexadecimal
identifiers, so organism, project purpose and whether proteins are variants cannot
be determined from those names. Similar-looking proteins must not be treated as
identical embeddings.

## Repetition by method

Counts below cover Pending and Processing jobs, count each selected method once
per job, and compare exact trimmed protein strings using full SHA-256 hashes.
Two prediction targets selecting the same method do not count as two distinct
protein requests. All rows here contain a single protein.

| Method | Jobs | Rows / distinct sequences | Repeated sequence slots within method |
|---|---:|---:|---:|
| CatPred | 793 | 78,009 | 0 |
| CatRange | 836 | 82,133 | 0 |
| DLKcat | 586 | 57,601 | 0 |
| EITLEM | 788 | 77,314 | 0 |
| IECata | 793 | 77,782 | 0 |
| KinForm-H | 794 | 79,913 | 0 |
| KinForm-L | 794 | 79,911 | 0 |
| OmniESI | 786 | 77,107 | 0 |
| OmniESI-O2DENet | 789 | 77,399 | 0 |
| TurNup | 792 | 77,786 | 0 |

There are also no exact repeated protein/reaction pairs or identical input files
within any one unfinished method group. Comparing these pending sequence sets
with completed jobs from this same IP found no same-method overlap either. That
comparison does not rule out reusable work from other IPs or standalone embedding
requests.

Globally the 764,955 row occurrences comprise 89,056 sequences, about 8.59 method
requests per sequence. That repetition is primarily across different methods,
whose encoder representations are not all interchangeable.

Machine-readable method counts: [CSV](reports/bulk-ip-2026-11-02.csv).

## A verified cross-method sharing gap

| Method pair | Shared exact sequences | Union | Potential duplicate share of paired requests |
|---|---:|---:|---:|
| KinForm-H / KinForm-L | 79,619 | 80,205 | 49.82% |
| OmniESI / OmniESI-O2DENet | 77,009 | 77,497 | 49.84% |

The saved representative model-version dictionaries for each pair differ only
in their `method` field. They have equal production/GPU inventory fingerprints,
precision and representation-policy metadata. They have no explicit
`representations` mapping.

`api/execution/planning.py:artifact_identity` deliberately includes the whole
method identity when such a mapping is absent. Consequently, compatible shared
encoder outputs receive different artifact IDs and producers across those method
names. Within-method sharing, including non-CatPred target sharing, already exists.
CatPred's target-specific attentive pooling remains genuinely target-specific.

This was verified against actual artifacts for two shared sequences in live jobs
`SglY0HR` (KinForm-L) and `odoXDcj` (KinForm-H). Each method had a separate completed
producer for the T5 last-layer residue matrix and ESM2 layer-26 mean vector. For
each sampled representation, the resulting SHA-256 checksum was identical across
the two methods. This proves duplicated computation in that sample, not merely
an identity-key difference.

The percentages describe input overlap, not promised remaining runtime savings:
some embeddings of unfinished jobs have already completed, other heads still
need separate evaluation, and representation compatibility must be established
for every shared output family. CatRange and KinForm ESM-C representations must
remain distinct. No cross-method aliasing was performed during this audit.

## Waiting, utilization and execution granularity

A 97-row TurNup job, `7oBd1Un`, had approximately 6.38 hours since its recorded
processing start but only 1.2 minutes in the union of its recorded attempt lease
intervals. Three embedding stages were completed and four remained. Attempt time
includes setup/transfers and is not a measurement of GPU kernel time.

The four sampled older 1,000-row KinForm jobs likewise had long wall-clock ages
and roughly 83–99 minutes of accumulated lease-covered attempt intervals. Some
remaining pooling stages depend on binding-site merges owned by other executions;
those merges in turn wait for Pseq2Sites work. Their remaining dependencies must
be considered before interpreting a job's own completed embedding count.

Prometheus observations from 09:04–09:24 UTC, 78 available samples:

| Signal | Sample mean | Sample maximum |
|---|---:|---:|
| GPU utilization, including unrelated activity | 5.63% | 94% |
| Free VRAM | 17.28 GiB | 17.73 GiB |
| GPU-host live attempts, all kinds | 2.08 | 5 |
| GPU-host observed platform CPU consumption | 0.77 logical CPUs | 4.12 |
| Production live attempts | 0.22 | 1 |
| Production observed platform CPU consumption | 0.05 logical CPUs | 1.00 |

Available RAM averaged approximately 119.6 GiB on the GPU host and 16.6 GiB on
production. Production had about 23.9 GiB disk free. These samples establish
substantial unused capacity, but 15-second telemetry can miss short GPU bursts;
they do not measure the exact fraction spent loading models versus scheduling,
I/O or inference.

The active GPU protein profiles admit at most 16 sequences per attempt and use
model batch size one. Each attempt starts a separate container/interpreter and
loads its model. In successful attempts beginning after 08:43 and finishing before
the inventory observation, typical whole-attempt embedding medians were about
15–24 seconds depending on method. Small isolated attempts repeatedly incur
startup and weight-loading cost. Increasing the input envelope or model batch
size requires profiling; spare VRAM alone is insufficient evidence.

## Recommended changes

1. **Share encoders by representation, keeping prediction heads separate.**
   Define explicit identities for pinned weights, tokenizer, preprocessing,
   layer, pooling and precision. Start with verified KinForm-H/L protein families
   and OmniESI/O2DENet. For the existing backlog, use an idempotent previewable
   reconciliation that aliases verified outputs or redirects only unstarted work
   to one compatible producer. Preserve immutable request specs, running/retrying
   execution plans, dependencies, checksums and result annotations.
2. **Keep admitted models loaded across bounded batches.** Introduce a supervised
   session per compatible encoder that processes several queued batches under
   one continuously held RAM/VRAM reservation. Preserve per-batch ownership,
   checkpoints and output commits. Measure cold-load time separately from forward
   time, then profile batching at 1/2/4/etc. Resume smaller safe batches after OOM;
   do not raise concurrency or memory limits on the basis of this sample alone.
   Normal-priority requests must still preempt/yield ahead of this background run.
3. **Use a one-time internal execution cohort rather than merge public jobs.**
   Export the eligible unfinished work, group by exact compatible representation,
   and stream a bounded window of unique sequences through the admitted encoder.
   Publish artifacts through the existing fenced completion path, then fan results
   out to each job's existing prediction and assembly stages. Keep all IDs, row
   order, downloads, timings and quota effects independent. Run dependent CPU
   stages promptly on the GPU host, avoiding unnecessary large transfers.
4. **Schedule critical shared prerequisites and small CPU stages efficiently.**
   Account for jobs blocked on other executions' binding/Pseq2Sites producers;
   inherit urgency from waiting consumers within the background class while
   preserving strict foreground precedence. Profile combining lightweight binding
   merge/pooling work to reduce repeated container/import overhead. Measure CPU
   admission and readiness bottlenecks before raising worker counts.

A bounded cohort can improve model reuse even when all proteins within a method
are unique. Simply concatenating public jobs, reusing predictions across different
methods, or precomputing every full residue matrix onto production would not solve
this safely. Full residue intermediates remain transient until the bounded window's
consumers finish; permanent mean/weighted vectors retain their existing policy.
Molecular encodings must retain existing batch-position context.

## Audit cost and evidence

The initial read-only metadata queries finished in 7.64 seconds. A broad detailed
stage query reached its eight-second statement limit and was abandoned; the
replacement sampled 1,971 stages and 1,211 attempt records in 3.66 seconds.
There were no ORM queries inside input-file row loops.

Each distinct input checksum across this IP's completed and unfinished jobs was
read once: 3,334 files, 221,963,856 bytes, 56.5 seconds, zero missing-file errors.
The file reader ran at nice level 19 with a 5 MiB/s read-rate ceiling and fixed
per-file/total read budgets. Exported row records contain hashes, lengths and
counts rather than protein strings. Comparisons ran locally. Later artifact and
dependency queries were restricted to specific indexed IDs.

Evidence and scripts are under `/home/saleh/webkinpred-execution-staging/bulk59-*`
on production and `/tmp/webkinpred-bulk59-*` locally. The original inputs remain
unchanged. This audit deployed no runtime changes and did not merge/requeue jobs.
