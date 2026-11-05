# Execution operations, 5 November 2026

Releases 120–122 fixed an 11.5-hour GPU-host stall and the disk growth on both executor
hosts, and let the GPU host keep working at low priority beside research jobs.
Scientific methods, model images, weights and input/output semantics are unchanged. No job was lost: all 6,207 jobs unfinished at the pre-deploy snapshot
are present with unchanged specification/input checksums, and the frozen legacy Redis
queue still holds 5,002 messages.

## GPU host idle for 11.5 hours

| Time (UTC, 4–5 Nov) | Event |
|---|---|
| 20:00–23:15 | A workload outside webKinPred used all 20 GPU-host CPUs, and the GPU from about 20:30 to 22:30. Admission yielded as designed (`cpu_external` ≈ 20). |
| 21:31:56 | That workload loaded ESM C. `huggingface_hub` rewrote `models--EvolutionaryScale--esmc-600m-2024-12/refs/main` with the same 40-byte commit hash. |
| from 21:32 | The supervisor's pinned-file check saw a new inode stamp, treated it as a model change and withdrew every profile until restart. Claims returned nothing although about 3,900 dependency-ready embedding stages were waiting. |
| 23:30–11:08 | GPU at 0 %, no attempts. Job completions: 48 at 21:00, 10 at 22:00, then 3 until 11:00. |
| 11:08 | Supervisor restarted; work resumed within a minute. |

The ESM C snapshot (`e4d83bc7…`) was unchanged; only the stamp moved.

**Fix (release 120 supervisor, commit `96f23e2`).** A changed stamp re-hashes that file
on a background thread. Profiles are withheld only while that verification runs and
come back when the digest still matches. Different bytes or a different GPU identity
still stop advertising until restart. A failed `nvidia-smi` probe withholds profiles
without aborting the supervision tick. Transitions are logged as
`executor.runtime_unverified`, `executor.runtime_verified` and
`executor.runtime_changed`, and every `executor.loop` summary now includes
`"runtime"`.

## Disk growth: finished residue matrices kept for hours

Transient per-residue embedding matrices (`residue_vecs`, ESM1v, OmniESI ESM2, IECata
ProtT5) are meant to be deleted as soon as every consumer of their bytes has finished.
Before release 120:

* Production held 4.3 GB of them, 2.1 GB with no unfinished consumer. Production copies
  these inputs whenever it runs prediction stages; most arrived 21:00–23:00 while the
  external workload occupied the GPU host's CPUs. Production free disk fell
  25.4 → 18 GiB, three GiB above its 15 GiB admission floor.
* The GPU host held 106 GB of them, **92 GB with no unfinished consumer**. The oldest had
  been deletable since 16:05 the day before.

**Cause.** Each collection call examined the next 512 ready locations of the host.
Almost all locations are permanent vectors that can never be collected: 463,000 on
production and 2.59 million on the GPU host. A full pass took about 8 and 43 hours
respectively. Completion hints did not cover copies registered under alias artifact
IDs with the same checksum, so most copies waited for that pass.

**Fix (release 120 control plane, commit `6ef2cb0`).** Pages of 4,096 locations, with
further pages scanned within 0.5 s while a page yields nothing (0.2 s per page on the
live catalogue). Completion hints cover every transient copy of the consumed bytes,
including aliases. Each call processes 512 hints and deletes up to 256 blobs (4 GiB
cap unchanged). Protection is unchanged: any unfinished consumer of the checksum, any
live attempt, and permanent, protected or legacy artifacts still block deletion.

Result in the first 24 minutes: 43 GB deleted on the GPU host (free 97 → 138 GiB) and
3.4 GB on production (19.2 → 22.9 GiB). Production completed a full pass in that time.

## Exited attempt containers accumulating

A completion request can take longer than the supervisor's 5 s client timeout
(observed up to 6.6 s). The supervisor then renewed the lease before resubmitting. When
the first request had already committed, the renewal was rejected, the attempt took the
lease-lost path, and that path never removed the container. Committed results were not
affected (all 281 such attempts are `completed` or reconciled), but 281 containers
accumulated on the GPU host after 834 were removed the day before.

**Fix (release 121 supervisor, commit `7983931`).** After a lost reply the identical,
idempotent completion is resubmitted without renewal. After a server error the lease
is still renewed first, because nothing was committed. The lease-lost path removes the
container it fenced. The 281 stale containers were removed after confirming that none
of their attempts was live.

## Sharing the GPU host with research jobs

Research jobs on the GPU host (4 Nov 16:14–16:56 and 20:00–23:15; 5 Nov from 12:13,
`code/remote_kcat/*.py` as `msp`) occupy up to all 20 CPUs. Admission subtracted that
use from the spare CPUs and admitted nothing, including GPU work, so the GPU idled at
about 14 % while those jobs used it only lightly. Chosen policy: webKinPred runs at low
priority beside them (release 122, commit `8184f40`).

* `/etc/systemd/system/webkinpred.slice` (`CPUWeight=10`, `IOWeight=10`) on the GPU
  host; the supervisor creates attempt containers with
  `--cgroup-parent webkinpred.slice` and `--oom-score-adj 500` (`gpu.json`:
  `attempt_cgroup_parent`, `attempt_oom_score_adj`). User sessions keep weight 100,
  so they win every CPU contention, and under host memory pressure the kernel kills a
  retryable attempt first. The supervisor itself stays in `system.slice`.
* `gpu_cpu_floor: 6`: GPU attempts may reserve 6 CPUs (three embedding attempts)
  however busy other work keeps the host. CPU-only attempts still use only spare CPUs.
* `gpu_external_limit: 40`: a supervisor thread samples `nvidia-smi pmon` every 10 s;
  GPU use by processes outside `webkinpred.slice` is averaged over a minute
  (`telemetry.gpu_external`). At 40 % or more no new GPU attempt is admitted
  (`gpu_unavailable`); running ones finish.

Measured at 13:06 with the research jobs running (load average 41): `user.slice`
18.1 cores, `webkinpred.slice` 1.7 cores, other services 0.2. The research jobs' GPU
use was 3–10 %. Three webKinPred GPU attempts kept running; with about 1.7 cores between
them, embeddings averaged 62 s (8 attempts, maximum 93 s) instead of about 16 s on an
idle host, roughly 17 % of the idle-host rate, where the previous policy ran none.
Over 13:08–13:13 the GPU averaged 40.6 % with no idle samples while the host CPUs stayed
fully used (19.9 of 20).
`CPUWeight` is the single knob for a different balance. Remove the two settings from
`gpu.json` and the two budgets from `EXECUTION_HOSTS_JSON` to return to full yielding.

## Deployment

| Component | Image | Rollback container |
|---|---|---|
| Backend, controller, control worker (r122) | `sha256:409583f7694fd8eabb3e57c18da0ae2ed2c1b33e18ef42741665286b7e13185a` | r120 `sha256:8380a30b0270…` (`live20/*.before122`), r119 `sha256:3a0247d03236…` (`*.before120`) |
| GPU supervisor (r122) | `sha256:40b68d91d257af63f412dfdaad975e33b84550831627200f85ddc4087343f927` | `webkinpred-gpu-executor-rollback122` (r121), `-rollback121` (r120), `-rollback120` (r119); `gpu.json.before122` |
| Production supervisor (r121) | `sha256:aaee6533daa3aaf0d92faa4e76a81794c1e0fc3e0f238ff8a42f30d02256c6a1` | `webkinpred-execution-agent-1-rollback121` (r120), `-rollback120` (r119) |

Images are thin overlays on the release-119 images (`FROM … COPY` of the changed files
only). The control plane was deployed with `up -d --no-build --no-deps` after a dry run
showing that only the three control services would be recreated; PostgreSQL and Redis
were not touched. Each supervisor was drained before replacement. The production
release-121 roll waited 25 minutes for a single-threaded UniKP similarity attempt of the
17,010-row foreground job instead of discarding its work. Production had no stale
containers and needs none of the release-122 settings, so it stays on release 121.
Preservation snapshots are in `staging/preservation113/` (`*-before120*`,
`*-after120ctl*`, `*-before122*`, `*-after122ctl*`); both checks found every
unfinished job present with unchanged checksums.

The production checkout `/home/saleh/webKinPred` was fast-forwarded to these commits;
its pre-existing local modifications (none in the changed files) are unchanged. The
GPU host has no deployment checkout. Its personal clone `/home/msp/saleh/webKinPred`
(March 2025) refused the fast-forward because upstream changes two EITLEM files that
have uncommitted local edits there; it was left untouched.

## Measurements

Ten-minute windows with 2 s host samples.

| Measurement | 4 Nov 16:57–17:07 (r119) | 5 Nov 11:57–12:07 (r120) |
|---|---:|---:|
| Mean GPU utilization | 62.9 % | **75.3 %** |
| GPU samples at 0 % | 17.0 % | **1.0 %** |
| GPU-host CPU, cores / 20 | 10.68 | 8.81 |
| Production CPU, cores / 4 | 3.24 | 3.24 |

The queue is now dominated by 6,189 small background jobs that need new embeddings, so
the GPU is the bottleneck and fewer CPU prediction stages are dependency-ready.
11:00–12:00: 115 jobs (19,384 rows) completed.

## What still grows, and remaining limits

* **Execution database: about 72 MB/h (~1.7 GB/day).** No rows are ever deleted from
  `api_artifact`, `api_artifactlocation`, `api_artifactdependency`, `api_workstage` or
  `api_executionattempt`: the catalogue records every per-sequence artifact and copy.
  It is 15.5 GiB now. Production has about 8 GiB above its 15 GiB floor, so pruning the
  history of finished, deleted transient copies is the next storage task.
* **Permanent copies on production: 3.2 GB.** Pooled vectors and predictions copied for
  production-run stages are never removed, although the GPU host keeps the originals.
* **Legacy cache:** `/home/saleh/webKinPred/media/sequence_info` holds 169 GB of
  pre-migration embeddings (CatPred ESM2 37 GB, ESM2 layer 33 35 GB, ESM C layer 32
  34 GB, ProtT5 33 GB, ESM1v 17 GB). It does not grow; the ESM1v cleanup
  (`docs/legacy-esm1v-cleanup.md`) remains blocked by the
  `wk-maintenance-web19-candidate` reader.
* **Research jobs on the GPU host** now leave webKinPred about 9 % of contended CPU and
  three GPU attempts; heavy external GPU use (≥ 40 % for a minute) still pauses new GPU
  work.
* **Collection calls** now take 0.7–1.2 s (median) and the supervisor still makes one
  inline every 30 s beside the standalone collector, about 5–8 % of its loop.
* **GPU concurrency** remains bound by VRAM reservations (about five embedding
  attempts); every attempt still starts a container and loads its model.
