# Initial execution measurements — 28 October 2026

**Historical initial measurements.** These measurements describe initial
calibration and compatibility evidence, not a validated production profile
registry or a completed migration/canary. The live queue, live repositories,
application databases and running production workers were not modified during
this initial measurement phase. The subsequent migration and live canary are
tracked in [the deployment record](execution-deployment-2026-10-29.md).

Raw measurements are retained locally under
`execution-evidence/2026-10-28/` (ignored by Git), with the remote originals in
`/home/saleh/webkinpred-execution-staging` on production and
`/home/msp/webkinpred-execution-staging` on the GPU host. Failed probes are
retained alongside successful probes. No SSH passwords are included.

## One-hour host baseline

Each host produced 360 samples at ten-second intervals over a one-hour sampler
run. The first-to-last sample spans approximately 3590 seconds. Figures include
the live legacy platform and other host workloads. GPU-host build/bootstrap and
small isolated probe activity near the end are also included; these observations
must not be interpreted as a clean measurement of unrelated workloads alone.

| Measurement | Production | GPU host |
|---|---:|---:|
| Available RAM range | 8.25–17.25 GiB | 102.41–121.27 GiB |
| Free root-disk range | 27.49–35.22 GiB | 513.49–515.35 GiB |
| Mean aggregate CPU use | 96.96% | 67.03% |
| Maximum aggregate CPU use | 100% | 100% |
| Free GPU memory | — | 12,566–18,818 MiB |
| Mean / maximum GPU utilization | — | 1.22% / 100% |

Production was CPU-saturated during this observation. Available capacity cannot
be inferred from installed CPU/RAM totals. Admission therefore accounts for live
external CPU use, host pressure, current free RAM/VRAM/disk, and unspent platform
reservations as well as configured ceilings.

## Container runtime bootstrap

The GPU host runs Docker Engine 27.1.1 and initially lacked NVIDIA container
integration. CUDA could not be requested with Docker. NVIDIA Container Toolkit
1.20.1-1 was installed from NVIDIA's signed repository; its runtime was registered
and Docker was reloaded with SIGHUP. Existing containers retained their original
start times. No Docker restart, GPU-driver upgrade or production/GPU service stop
was performed. A container using the existing read-only UniKP environment then
reported PyTorch `1.13.1+cu117` and CUDA available.

On this daemon, explicit `--runtime=nvidia` plus the assigned GPU UUID works;
the `--gpus` selector still requires device-driver registration from daemon
startup. The executor supports explicit runtime selection, avoiding that restart.

## UniKP molecular encoder compatibility probe

Copied synthetic molecular inputs cover 130 original positions, including
0, 1, 99, 100, 127, 128, repeated molecules and a long SMILES case exercising
legacy truncation. The CPU baseline uses the original encoder's outer batches of
128 and internal partitions of 100. CUDA uses microbatches of eight with explicit
legacy positional context. Both use the same original weights/vocabulary and
FP32, in isolated read-only containers with 4 GiB RAM and two CPU allocations.
CUDA probe allocation was capped at 7.5% of device memory. Neither production
media nor application/quota/cache APIs were used.

Pinned input artifacts:

* Vocabulary SHA-256:
  `21a66c850a3222547ec0fbd30c05fe587d66d22d3de2ee2195c58250fe486fb7`.
* Transformer checkpoint SHA-256:
  `6b56c8c05d048e7c7d143c4e3ba2bc6f76e5eda2358798cf636210406a700eb2`.
* Staging base image:
  `sha256:6ae77dcc961bfdeede74d7a6a0f6bbe1b681fbaea879434291c4f7194038f994`.

The experiment used source mounts recorded in the raw configurations. The image
is a staging supervisor/runtime, not a frozen final production model image.

Three fresh CUDA runs and five repeated-process runs succeeded operationally;
each repeated-process run encoded the dataset five times with the loaded model.
Fresh CUDA process wall times were approximately 4.65–4.86 seconds. GPU encoding
after initialization took approximately 0.066–0.085 seconds per 130-item pass.
The single legacy CPU encoding pass took approximately 1.325 seconds. These are
different startup/warm states and are **not** a platform throughput comparison.

Across CUDA probes, sampled cgroup RAM peaked at approximately 2.03 GiB. Device-wide
GPU usage peaked at approximately 1.89 GiB, **including external activity**.
PyTorch reported approximately 41.5 MiB peak allocated and 62 MiB peak reserved.
The profiler now records owned process IDs and separates external allocations in
subsequent measurements; these older device-wide numbers remain conservative.

**Scientific parity failed.** All eight CUDA runs exceeded the approved
`rtol=1e-5, atol=1e-6` comparison against the legacy CPU embedding array: 312 of
133,120 values failed; the maximum absolute difference was approximately
`1.502 × 10^-5`. Original positional context was retained. No tolerance was
loosened and no profile was activated. See
`execution-evidence/2026-10-28/molecular-parity.json` and the saved arrays.

The user subsequently explicitly waived GPU numerical parity. This discrepancy
is retained as evidence and no longer blocks activation on numerical grounds.
Final-runtime resource measurements, shape/dtype/finiteness checks, and explicit
waiver provenance are still required before this configuration can serve jobs.
The other methods, prediction heads, molecular encoders, input
boundaries, mixed GPU combinations, complete recovery matrix, serial/concurrent
workload comparison and production canary still require final-runtime validation.

A separate diagnostic experiment promoted GPU arithmetic to FP64 and cast the
result back to FP32. It also failed the unchanged comparison (2,184 values failed;
maximum absolute difference approximately `1.287 × 10^-5`). Higher arithmetic
precision alone does not reproduce the legacy CPU's FP32 rounding. This was one
diagnostic run, not an approved execution configuration; production encoding
precision was not changed.

Two additional FP32 diagnostic runs used CUDA batch sizes 100 and 128. They also
failed the same unchanged tolerance: respectively 309 and 312 values failed,
with maximum absolute difference approximately `1.502 × 10^-5` in both cases.
Peak framework allocations were approximately 279 MiB and 350 MiB. These were
single diagnostics, not full profiling configurations; neither is eligible for
activation. Merely increasing the CUDA batch size does not resolve the parity
failure. Arrays are retained as `unikp-batch-100.npy` and `unikp-batch-128.npy` in
the local evidence directory; full diagnostic logs remain on the GPU staging host.

## Implementation verification

The test suite covers PostgreSQL claims/fencing/reservations, durable request
reconciliation, original-day quota refunds, artifact integrity, batch context,
maintenance/API contracts and an old SQLite-schema import/restore rehearsal into
disposable PostgreSQL. Opt-in real Docker tests exercise cached prediction through
final downloadable CSV and a contained 64 MiB container OOM/requeue. The frontend
production build passes. These checks do not substitute for model parity or the
100-job, one-hour production canary.

Initial-phase local validation: **286 tests and three subtests pass**, including the two
real Docker tests, against the disposable PostgreSQL instance. The tested executor
image is `sha256:ec0267f35d300a37b21e35d2b5c6499e6c02f52a4c581e431ac75208427f0845`.
Django system/migration checks and execution-code lint pass. All 122 changed
Python files parse. The monitoring Compose configuration disables legacy consumers;
its YAML/dashboard files parse, but monitoring services have not been started and
their images remain deliberately unconfigured pending release pinning.
