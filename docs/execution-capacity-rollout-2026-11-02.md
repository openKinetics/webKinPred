# Resource-aware capacity rollout, 2 November 2026

This implements the changes identified in [the capacity audit](execution-capacity-audit-2026-11-02.md). It preserves existing jobs and immutable requests, uses GPU execution for embeddings, and retains admission and failure-recovery safeguards. Model environments and `Dockerfile.envs` were not rebuilt.

## Changes deployed

- Missing embedding files now trigger dependency recovery rather than disabling a healthy CPU execution profile. Recovery schedules all missing dependencies and retains committed outputs. Previously suspended KinForm-L pooling work was repaired.
- Artifact transfers use bounded packs and batch location acknowledgements, with per-file checksums, ownership fencing and atomic publication. CPU execution can use inputs local to either host. A production CatPred attempt transferred 1,001 artifact references / 6,468,000 bytes in **5.77 seconds** after deployment. Earlier attempts with the same reference count and byte size recorded **341–375 seconds**. These are observed transfers, not a controlled network benchmark.
- Untouched protein embedding stages can be combined into measured work units of up to **128 sequences**. The model remains loaded while executing their existing microbatches. Original artifact identities, job boundaries and consumer dependencies remain intact. Molecular requests retain original positional context and partition boundaries.
- Compatible untouched protein stages from separate bulk jobs can share a bounded GPU batch. Scheduling class, priority, policy, method, target, model profile and non-sequence options must match. Foreground prerequisites, molecular requests and input-dependent binding/pooling stages are excluded from cross-job batching. Original job specifications remain immutable, outputs retain their semantic IDs, and each consumer receives only its own results. Shared progress counts only a job's required outputs. Unknown batch errors split to individual sequences; OOM recovery splits to smaller measured units. Newly generated raw residues stay transient, while jobless legacy research-service outputs remain protected. `EXECUTION_CROSS_JOB_PROTEIN_BATCHES=0` disables creation of further shared batches without interrupting existing attempts.
- KinForm Pseq2Sites and CPU pooling also accept measured bounded work units. Pseq2Sites retains each original input TSV's partition context. Recovery can split an enlarged failed unit back into its original smaller units without losing verified outputs.
- GPU admission accepts startup subsets of measured mixtures, including multiplicities. A separately measured larger work unit may inherit its unchanged resident-model mixture when image, model settings and supported per-sequence shapes match. Changes to model image, precision, molecular context or execution settings do not acquire approval automatically.
- Exact, representation-compatible KinForm-H/L and OmniESI/O2DENet outputs can be shared. Cache identity still includes representation-affecting settings. KinForm planning checks final vector outputs before rebuilding expired residue intermediates, and does not replace another producer's active or completed stage.
- Cohort planning selects bounded stage IDs before fetching manifests. Claim queries disable analytical JIT/parallel plans, use bounded work memory and retain resource/ownership checks. Heartbeats run at five-second cadence. Public status requests fetch counters and timestamps rather than large manifests and attempt logs.
- Preparation has separate bounded windows for GPU-dependent and CPU-only requests. The original single window hid **586 DLKcat jobs** while CPU executors were idle. The controller now looks ahead to independent CPU jobs, initially admitted **32 prepared CPU-only requests**, then increased the live controller window to **128** after that smaller window filled with jobs waiting for production publication, verifies their immutable method choices, and retains the disk guard and foreground priority. `EXECUTION_PREPARED_CPU_LIMIT` configures this window (code default 32, live controller 128). This is planning lookahead, not 128 simultaneous attempts. Mixed-method requests do not obtain a CPU-only exemption.
- GPU-host CPU-only attempt slots increased from 8 to 16. Aggregate CPU, RAM and storage reservations remain authoritative; this is not permission to run 16 arbitrary models simultaneously.
- Shared prerequisites inherit the oldest unfinished consumer's queue age, including transitive dependencies. Previously, a job submitted on 27 October at 12:03 depended on work owned by a later job submitted at 15:51, and that later owner's queue position delayed the earlier job. A nullable scheduling timestamp fixes ordering without changing inputs, quotas, resources or public job IDs. Initial backfill visited 3,312 prepared requests in bounded SQL pages; subsequent sweeps have a time budget, and new preparation refreshes its own dependencies immediately. Cohorts preserve inherited age.
- Executors fill capacity in bounded admission bursts, up to four requests while starting no more claims after four seconds or when a heartbeat is due. Pending transfers remain capped at four. Every claim still reserves resources transactionally and is checked locally against already admitted, not-yet-allocated model peaks. A local rejection ends the burst until telemetry refreshes. Saturated hosts still poll for foreground preemption.
- Batch graph rewrites are capped at four new batches per reconciliation pass, with method-family rotation. Prediction-unit monitoring aggregates numeric counters in PostgreSQL rather than returning full scientific manifests to Python.
- PostgreSQL's original 1.5 GiB memory cap caused avoidable cache churn and disk waits while the host had spare RAM. Its live memory/swap ceiling was raised to **4 GiB without restarting PostgreSQL**. `EXECUTION_DB_MEMORY` persists this setting; filesystem cache is included in the container limit.
- PostgreSQL's one-CPU Docker cap was raised to a configurable **two CPUs** after observing throttling and control-request timeouts. The live limit was updated without restarting PostgreSQL. `EXECUTION_DB_CPUS` controls the persistent Compose setting. Host admission accounts for actual external CPU use.
- Retired CataPro CPU profiles were suspended and replaced with the corrected worker images. A copied production input passed all three target regressions, including alignment of its 78 invalid rows. The preserved job then completed all three targets and final assembly.

Temporary residue retention remains reference-based and transient. Permanent pooled vectors and protected research residue files were not deleted. Cleanup during this rollout removed only this rollout's copied profiling inputs, after archiving and verifying them off production.

## Final operating limits

Effective admission is the lower of these ceilings and currently available resources after external workloads and headroom. CPU containers have no GPU device access. Embedding stages require CUDA and wait rather than falling back to CPU.

| Setting | Production | GPU host |
|---|---:|---:|
| Aggregate attempt CPU ceiling | 3 logical CPUs | 16 logical CPUs |
| CPU-only attempt slots | 3 | 16 |
| GPU allocation slots | No GPU profiles | 5 |
| Aggregate attempt RAM ceiling | 10 GiB | 96 GiB |
| Aggregate attempt scratch reservations | 8 GiB | 128 GiB |
| Host RAM headroom | 4 GiB | 16 GiB |
| Disk headroom | 15 GiB | 50 GiB |
| CPU headroom against measured external use | 0.25 CPU | 2 CPUs |

GPU headroom is the larger of 2 GiB or 10% of device memory. Reservations include admitted models that have not yet reached their allocation peaks. The temporary profiling reductions and extra VRAM guards were removed. When GPU work could fit, `gpu_cpu_reserve=4` keeps background CPU-only reservations below 12 CPUs so they cannot consume every GPU-worker startup CPU. The total ceiling remains 16. CPU-only work may use the remainder when no GPU work can run; foreground requests retain their full resource/preemption rights.

Strict precedence for normal users over the configured bulk IP/API-key class remains enabled. An early view of prepared stages missed the independent CPU backlog: the new preparation window fixes that omission. Effective utilization still depends on dependency readiness, model startup, input shape and measured resource envelopes; free VRAM alone does not establish spare GPU compute capacity.

## Profiling and measured improvements

Profiling ran in isolated containers with copied/synthetic inputs, disabled network access for CPU scientific probes, separate scratch directories and no writes to live requests, quotas or user results. Runtime images, source, weights, tokenizers, hardware and execution context were recorded. Each promoted configuration has three fresh-process and five repeated runs; repeated runs include model loading and benefit from the filesystem cache. GPU numerical parity was waived by the user; shape, dtype, finite values and output completeness remained checked. CPU prediction comparisons retained `rtol=1e-5`, `atol=1e-6`.

The larger protein work units were measured on the same 128 inputs as eight legacy 16-sequence cold processes. The legacy reference is one serial pass; the enlarged result is the median of eight runs. This is a stage benchmark, not a promised whole-backlog speedup.

| Encoder | Legacy serial seconds | Bounded 128-sequence seconds | Ratio |
|---|---:|---:|---:|
| KinForm ESM-C residues | 58.67 | 17.50 | 3.35× |
| KinForm ESM2 residues | 130.89 | 60.50 | 2.16× |
| OmniESI ESM2 | 135.04 | 48.45 | 2.79× |
| EITLEM ESM1v | 139.36 | 44.81 | 3.11× |
| KinForm T5 residues | 184.05 | 68.18 | 2.70× |
| CatRange ESM-C | 60.92 | 17.01 | 3.58× |
| TurNup ESM1b | 178.39 | 52.28 | 3.41× |
| IECata T5 residues | 169.41 | 57.14 | 2.97× |
| Shared UniKP/CataPro T5 means | 165.57 | 42.24 | 3.92× |
| CatPred kcat protein encoder | 146.95 | 43.87 | 3.35× |
| CatPred Km protein encoder | 145.39 | 43.76 | 3.32× |

Length envelopes remain method-specific: 1,000, 1,022 or 1,024 residues for the newly measured encoders. Pseq2Sites and pooling retain their separately measured 1,500-residue envelope. A smaller or different envelope does not authorize extrapolation beyond its validated limits.

Five cold-stage mixed configurations were compared serially and concurrently for three repetitions each. Measured stage speed ratios were **1.27–1.93×**. Two additional mixtures using the updated 128-sequence Pseq2Sites worker measured **1.30×** and **1.32×**. Two additional cold mixtures with the updated binding-worker image measured **1.64×** (ESM-C + ESM2 + Pseq2Sites) and **1.73×** (T5 + OmniESI + Pseq2Sites). The four-model combination that could not pass capacity checks was rejected. Alias expansions of these measured configurations produce **129 exact combination approvals**; that number does not mean 129 independent benchmarks. Unrelated GPU work was left running, and each comparison recorded its external baseline.

Production gained 13 measured CPU method/target profiles on the existing environment image: CataPro's three targets, EITLEM's two, KinForm-H's two, KinForm-L kcat, MMISA-KM Km, TurNup kcat, UniKP's two, and IECata kcat/Km. All 104 CPU comparison runs passed. GPU-host CataPro's three replacement CPU profiles passed another 24 comparisons. The large KinForm pooling profile passed eight NumPy reducer comparisons. Existing compatible profiles remain available for other supported methods.

The hosts advertise **34 production profiles** and **84 GPU-host profiles**, including CPU work on the GPU host. All promoted profiles matched the intended host runtime identity. Unknown or incompatible shapes remain waiting for measured coverage.

## Release and preservation evidence

Control-plane release `r95` uses source `700e2f4d462cc04697f668ad463519d27654a37a` and image `sha256:54c0fc525efb73a5225da0c5676ecd56e2d1f4326fc2b6cdc5d8fc8b0f33b5da`. Source changes were committed and pushed before deployment. Releases use source bundles and pinned images; deployment did not reset the dirty server repositories or overwrite their model changes.

Executor supervisors use independently pinned source `1e68687` (release 92); worker model images remain at their profiled versions. Supervisor images are `sha256:d746f58e8d3745b50d0fdae6c49aaed5da43f3ed1442e28b61a117c6c78e2f1e` on production and `sha256:9f436ce24ec21e801f5dd4eacec910aa1d6dc25043543c43ac42ce5902499330` on the GPU host. Production CPU image `sha256:32998ce13de9b6ae06798e7c203c725f2117743fc109234470c808e6647a3671`, Pseq2Sites image `sha256:3eb4eebf2d5fdad8cb92f5ba13136c1e4fedab01eb2fccad0ccab736248d509a`, and GPU-host CataPro CPU image `sha256:52612c14abc997d95e2c650b272de5d2ed43264bc73c3990b438389825f7ce43` reuse the existing model environments. Profile evidence pins the image and mounted model/adaptor checksums. No Conda environment rebuild was performed.

Agents were drained before their replacements, and real attempt containers were checked stopped before replacement. Later control-plane-only deployments did not restart the agents or model containers. Redis queues were never purged or consumed by legacy workers. Immutable inputs, output files, request options and quota reservations were not recreated.

The preservation inventory checks all **9,431 originally unfinished job IDs** and **11,676 immutable execution specifications**, including checksums and input checksums. The frozen legacy Redis DB 0 queue is retained as migration evidence; active short control tasks use broker DB 3. These are distinct queues and must not be compared as if they were the same broker namespace.

The complete code test suite passed **611 tests and 3 subtests**, including queue-age inheritance and strict-priority regressions. Compose configuration and live resource limits were also checked. Additional acceptance evidence includes actual public status/download checks, authenticated GPU-service health, profile imports and the post-deployment observation below.

## Post-deployment observation

A live canary combined **78 proteins from 16 bulk jobs** into one measured CatRange stage. It completed in **25.24 seconds** and published all 78 verified outputs. Every original request checksum and input checksum remained unchanged. The canary received a temporary priority within the background class for validation; normal-user precedence was unchanged, and that priority was restored afterward. General cross-job batching was then enabled.

A subsequent observation caught a candidate-filter regression that read large payloads before selecting bounded IDs. The filter now runs after selection; a SQL regression test covers this. Final measurements below start after that correction, rather than combining the slow intermediate deployment with the final runtime.

The final control-plane observation ran from **00:51:24 to 01:06:24 UTC on 3 November**, with 51 host snapshots and 196 public health/status requests. The CPU preparation window increased from 32 to 128 during the window. The separate one-second GPU probe recorded 900 samples over 15 minutes, with no sampling errors. Host clocks differ; probe durations and statistics use each probe's own clock rather than subtracting timestamps from different hosts.

- **36 jobs / 6,244 submitted rows / 10,118 reported predictions** completed by 01:06:08. These counts include existing work; they are not all newly submitted canary jobs.
- Production CPU use averaged **3.02 / 4 logical CPUs**, peaking at 3.98. Up to **three production CPU attempts**, **five GPU-host CPU attempts**, and **three GPU allocations** were observed. These maxima need not occur in the same snapshot.
- Direct GPU utilization averaged **58.75%**, with a 100% peak. VRAM peaked at approximately **15.21 GiB**. This demonstrates sustained useful work with remaining gaps, rather than continuous full saturation.
- Direct GPU-host CPU use averaged **3.31 / 20 logical CPUs**, peaking at 9.5. The coarser executor telemetry averaged 4.51. The GPU host still has substantial CPU headroom: dispatch latency, model startup, dependency readiness and production publication constrain throughput. The rollout does **not** establish maximal utilization of every CPU. Claim RPCs averaged about 1.5 seconds; a claim audit with rolled-back reservations confirmed fitting CPU work remained eligible.
- **No OOMs, held stages or uncertain attempts** were recorded in this window. There were 34 CPU-capacity deferrals and three host-pressure deferrals, all recoverable. Production had two I/O-pressure snapshots; neither host had a memory-pressure snapshot. The GPU host had no I/O-pressure snapshots.
- All **196 public checks returned HTTP 200**. Mean latency was 0.90 seconds and p95 was **2.81 seconds**; the maximum was 5.64 seconds. The interface was available, with further latency headroom. Twelve internal claim requests timed out and were retried; two stale heartbeats were rejected by ownership fencing. These are not permanently failed jobs.
- Five completed downloads returned HTTP 200 with the expected **1,000 / 98 / 78 / 1,000 / 1,000 rows**. About statistics returned HTTP 200 with a fresh snapshot, the authenticated GPU-service health check returned 200, and Prometheus reported UP.

Of the 61 jobs in the supplied processing-job list, **32 were completed and 29 remained processing** at 01:10 UTC, with no held stages.

The final preservation inventory retained **all 9,431 source job IDs**, with **484 completed / 103 processing / 8,844 pending** at that snapshot. All **11,676 original execution checksums and input checksums** were unchanged. The frozen Redis queue still contained **5,002 messages**; the active control queue was empty at inspection. Outbox entries can briefly be nonzero while live work finishes and are reconciled by the controller.

An earlier observational comparison counted 14 completed jobs in 30 minutes before the rollout and 129 in approximately 135.5 minutes afterward (about **2.04× the job rate**). Completed encoder-sequence units per second were about **2.38×** higher. Workload mix, cache state and external contention differed, so these are operating observations, not controlled speedup estimates. The controlled stage comparisons are listed above.

## Evidence and operator checks

Production evidence is under `/home/saleh/webkinpred-execution-staging/`: `preservation66.json`, `progress85-*.json`, `capacity76.samples.jsonl`, `http85.samples.jsonl`, `window85.json`, `admission85.json`, `window95-final.json`, `preservation95.json`, `broker95-final.json`, `health95-final.json`, `capacity95.samples.jsonl`, `http95.samples.jsonl`, `canary93-result.json`, `public95.json`, `tests95.txt`, `cpu-coverage70/`, release image/deployment records, and `evidence85/`. GPU evidence is under `/home/msp/webkinpred-execution-staging/probes/` in `capacity66`, `cohorts68`, `kinform72`, `pool72`, `mixtures73`, `cohorts76`, `cpu83`, `invalid83` and `mixtures88`; copied harnesses and summaries are in `evidence85/`. The final one-second GPU observation is `/var/lib/webkinpred/gpu95.samples.jsonl`, with `gpu95-summary.json` in the GPU staging directory. Earlier windows remain retained separately. No profiling workloads remain running.

Copied production profiling inputs were archived, read back, and SHA-256 verified off production before removing the private copies and their production archive. The retained archive is `/home/saleh/.local/share/webkinpred/rollout-2026-11-02/production-cpu70-fixtures.tar.gz`, SHA-256 `c45330f1ce2c53d70067a2262ba30c8d3288f4f46ad2aecd703c596235098e86`. The manifest is retained in production `evidence85/production-cpu70-fixtures.json`. Restoring it into an isolated `cpu-coverage70` directory recovers the original relative fixture paths. Original GPU profiling source fixtures remain retained.

Useful production commands, executed with Docker privileges:

```sh
docker exec webkinpred-backend-1 python manage.py execution_admin explain --stage STAGE_UUID
docker exec webkinpred-backend-1 python manage.py execution_admin drain --host gpu
docker exec webkinpred-backend-1 python manage.py execution_admin resume --host gpu
docker exec webkinpred-backend-1 python manage.py execution_admin drain-gpu --host gpu
docker exec webkinpred-backend-1 python manage.py execution_admin resume-gpu --host gpu
docker exec webkinpred-backend-1 python manage.py execution_repair_dependencies --dry-run
docker exec webkinpred-backend-1 python manage.py prioritize_job PUBLIC_ID --priority 100
```

Draining stops new admissions and preserves existing ownership; `stop` is a different operation. Do not unsuspend a profile merely to increase concurrency. Inspect the failure category and validate its resource envelope first. Preserve the queue and committed outputs when reducing concurrency or disabling a profile.
