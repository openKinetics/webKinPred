# Execution operations, 6 November 2026

Jobs stopped progressing overnight for three independent reasons. Releases 123–124 and
new envelope-extension profiles fixed the first two; the third is a host-sharing
policy that stays as chosen. Scientific methods, images, weights and input/output
semantics are unchanged. No job was lost: preservation checks before releases 123 and
124 found all 5,566 and 5,561 unfinished jobs present with unchanged checksums, and the
legacy Redis queue still holds 5,002 messages.

## What stopped jobs

| Cause | Effect | Fix |
|---|---|---|
| GPU-host free disk fell to 51 GiB (admission floor 50 GiB) | Nothing started on the GPU host after 01:41 | 120 GiB of old research files deleted at the user's request (`/home/msp/saleh/kinetics/results`: `msas/`, eight `.pkl` files); 171 GiB free |
| Inputs just beyond measured profile envelopes | 45 normal-priority stages waited as `profiling_required`, some for 15 h: DLKcat SMILES 269 > 257, KinForm-L prediction SMILES ≤ 801 > 257, KinForm molecular SMILES ≤ 839 > 512, similarity 17,010 / 18,000 rows > 10,000 | Envelope-extension clones (same image, fingerprint, settings, model identity) |
| `bind_profiles()` loaded every unbound stage with its full payload and job specification | A 9-job submission at 10:00 published ~1,900 unbound stages; the controller (1 GiB) was killed for running out of memory, so none could be bound or claimed | Release 123: binding reads only the compared fields; 1,900 stages bound in ~15 s |

The three 153-row DLKcat jobs (submitted 02:18–03:25) completed at 10:55–10:56, about a
minute after their stages were bound.

## Envelope extensions

KinForm's SMILES transformer always encodes a fixed 220-token window (first 109 and
last 109 tokens), KinForm prediction uses fixed-size precomputed features, DLKcat builds
a small per-molecule graph, and similarity streams 128-row chunks, so these inputs do
not change the computation's memory shape. Clones:
`gpu17-KinForm-{L,H}-molecular-len4096` (`max_length` 4096),
`gpu54-KinForm-L-predict-kcat-mol1024` and `production70-KinForm-L-kcat-mol1024`,
`{gpu16,production16}-DLKcat-predict-kcat-mol1024` (`max_molecule_length` 1024, RAM
×1.25), and a `-rows20000` similarity clone (RAM and disk ×1.5) for every method on both
hosts, so the next large job of any method does not stall the same way.

## CPU share on the shared GPU host

Attempts there run in `webkinpred.slice` at CPU weight 10 beside research jobs (user
sessions, weight 100). Admission had subtracted all of the research jobs' CPU use, so no
CPU-only stage could start while they used ~18 of 20 cores. Release 124 adds
`cpu_external_share` (0–1, default 1): the fraction of other work's CPU use that
admission and `cpu_capacity()` treat as unavailable. It is 0.91 on the GPU host, matching
the kernel's guaranteed 10/110 share.

With the research jobs at ~17 cores, that share (~1.8 cores) is fully used by three GPU
embedding attempts (~62 s each instead of ~16 s on an idle host); 12 GiB of VRAM stays
free. Asked whether to raise the weight (50 or 100), the user chose to keep weight 10, so
webKinPred on the GPU host runs at that share until the research jobs finish. Production
(4 cores) runs at most two 1-CPU attempts beside its services.

## Deployment

| Component | Image | Rollback |
|---|---|---|
| Backend, controller, control worker (r124) | `sha256:098f53b13c2aaab0008ef553de5d94f093d3c4bbcc4a049c707458d9c08b9d7e` | r123 `sha256:045692564b3f…`, r122 `sha256:409583f7694f…` (`live20/*.before123`, `*.before124`) |
| GPU supervisor (r124) | `sha256:14e56e223e35e10a0016c12fec27760fcb458471adcf2c823cb22132b2b48f86` | `webkinpred-gpu-executor-rollback124` (r122); `gpu.json.before124` |

Both are thin overlays (`control.py`, `resources.py`). Control services were recreated
with `up -d --no-build --no-deps` after a dry run; PostgreSQL and Redis were untouched.

## Remaining limits

* While research jobs occupy the GPU host's CPUs, webKinPred there is bound by the slice
  weight (the user's choice); `CPUWeight` in `/etc/systemd/system/webkinpred.slice` and
  `cpu_external_share` (= 1 − weight/(weight + 100)) are the knobs.
* Production free disk is 20.4 GiB against its 15 GiB floor. The legacy ESM1v cleanup
  (16.7 GiB, `docs/legacy-esm1v-cleanup.md`) is ready to apply with
  `--trust-reader openkinetics-data-backend-1`.
* Envelope gaps for other inputs (e.g. very long SMILES for methods that featurize
  molecules inside prediction) still surface as `profiling_required` and need a measured
  or reasoned extension.
