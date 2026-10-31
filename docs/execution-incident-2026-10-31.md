# Production storage stall, 31 October 2026

## Cause and immediate recovery

The last whole-job completion before inspection was 30 October at 22:23 UTC.
GPU stages continued completing while production could not admit final assembly.
Production had fallen below the disk reservation needed to preserve its 15 GiB
headroom; preparation also reported `Artifact storage budget exhausted` from
01:00 UTC. Public `Processing` elapsed time included this resource wait.

The storage audit found approximately 13 GiB in the dedicated PostgreSQL volume,
9.9 GiB of isolated profiling fixtures, 3.9 GiB of executor working data, and
retained migration/model backups. These are observations of the current
filesystem, not a reconstruction of an earlier 40 GiB-free snapshot.

The requested `docker system prune -a -f` reclaimed only 55.25 MB. Volumes were
not pruned. Production admissions were briefly drained for pruning, then resumed;
no job queue was purged and no model environment was rebuilt.

Checksum-verified compression preserved 1,354 profiling CSVs and 1,280 synthetic
`cpu_*.pt` profiling files. The latter had only 16 distinct original contents;
verified equivalent compressed archives were deduplicated with hard links,
preserving each archive path. Compression/deduplication reclaimed about 9.4 GiB.
An additional 634,098,627 bytes of planning CSV copies were removed only after
matching their committed immutable blobs by checksum and size. Original input
CSVs were untouched. Production subsequently had approximately 24 GiB free.

The interface and execution recovered before retention deployment. The reported
normal-priority jobs `C21Lq3w`, `54C78XV`, and `OgNGnPn` completed. The existing
priority policy remains active for bulk-client requests.

## Retention correction and future safeguards

The initial executor collected generic artifacts only under storage pressure and
only when its host was idle. That did not preserve the former short lifetime of
full residue matrices, and generic eviction could also remove pooled vectors.

Artifacts now have explicit policies. Unknown/legacy files default to protected.
Known pooled vectors are permanent. Only managed `execution-v2` residue artifacts
with platform-job provenance become transient. Standalone embedding requests and
original legacy/project filesystem trees stay protected. In particular, older
T5, ESM2 and ESM-C project files are not traversed or deleted.

A separate host-local collector checks approximately every 30 seconds, with
jitter to avoid repeatedly colliding with executor claims; it drains confirmed
work more frequently. Deletion waits for every unfinished consumer, live or
uncertain reader/producer, and protected/permanent checksum alias. Durable
cleanup manifests fence new admissions until unlinking is acknowledged. Finished
stages' private scratch can be retired, including old retry directories;
unfinished checkpoints and journals remain intact.

Consumer completion enqueues durable cleanup hints in the same transaction.
Background reconciliation scans bounded pages of ready location IDs, then checks
retention and ownership, without sorting entire embedding manifests. The
retention backfill uses keyset pages and constant-policy updates, avoiding a
large held server cursor on production storage. Both helpers are resumable.

Background preparation stops expanding the backlog once 128 eligible background
jobs are prepared, or disk space is within 2 GiB of the headroom guard. Normal
requests retain precedence; held/unprofiled work does not monopolize that window.
Successful planning removes redundant scratch copies. Completion-stall and
below-headroom alerts now detect this failure even when GPU stages stay active.

The new services reuse existing supervisor/model environments. Running inference
supervisors are preserved; collection has no Docker socket or legacy model/cache
mounts. Migrations 0037 and 0038 are additive. No submissions are charged again.

Private operational evidence, compression/deduplication manifests, job inventory,
classification logs and recovery checks are retained on production under
`/home/saleh/webkinpred-execution-staging/incident50/` and adjacent
`incident50-profile-*.jsonl` files. The regression suite passes 475 tests and
three subtests, including shared-consumer protection, permanent vectors, legacy
project preservation, checkpoint retention, sweep progress and planning limits.
