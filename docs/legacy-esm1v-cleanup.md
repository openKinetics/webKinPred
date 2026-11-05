# Manual cleanup of the old ESM1v residue cache

Run `tools/cleanup_legacy_esm1v.py` **on the production host as root**.
It uses Python's standard library and the running backend's database connection;
it does not rebuild images, update job state, stop attempts, or touch Redis.

One-time database preparation (run on production on 4 November, and again on 5 November
to add the dependency index):

```bash
sudo python3 /home/saleh/webkinpred-execution-staging/cleanup_legacy_esm1v.py --prepare-index
```

This builds two operator-owned partial indexes containing only ESM1v paths:
`wk_legacy_esm1v_path` on artifact identities and `wk_legacy_esm1v_dependency`
on dependency paths (with their stage). The concurrent builds run outside the
graph-publication lock, use one maintenance worker and 32 MiB of maintenance
memory, and permit normal database writes. Preparation can be repeated safely;
an existing index with an unexpected definition stops it. Preview/apply refuse
to scan an unindexed catalogue if either index is missing or invalid. No job or
artifact records are changed by index preparation.

On 5 November the preview failed with `Database safety snapshot refused:
OperationalError (SQLSTATE 57014)`: the dependency snapshot read the whole
8-million-row dependency table, and a cold read of the ESM1v artifact rows took
4.5 s, both against the guard's five-second statement limit. With the dependency
index the dependency snapshot takes 0.4 s (built in 12 s, 15.5 MB). The guard now
also reads the ESM1v artifact rows once before taking the graph lock (20 s
limit, result unused), so the guarded snapshot reads warm pages, and it retries
the shared graph lock for up to 10 s, because routine artifact collection holds
it briefly every few seconds. The whole snapshot step took 4.6–7.1 s on
production, including container start-up. Protection rules and the guarded
statement limit are unchanged.

Preview (no deletion):

```bash
sudo python3 /home/saleh/webkinpred-execution-staging/cleanup_legacy_esm1v.py
```

Progress uses one updating line on terminals. Redirected output gets short phase
messages and scan/deletion counts at most every two seconds, plus final counts.
Progress goes to stderr; stdout retains the final JSON summary for scripts.

Apply after reviewing the preview and resolving any `deletion_blockers`:

```bash
sudo python3 /home/saleh/webkinpred-execution-staging/cleanup_legacy_esm1v.py \
  --apply --limit 1000 \
  --report "/home/saleh/webkinpred-execution-staging/esm1v-cleanup-$(date -u +%Y%m%dT%H%M%S).jsonl"
```

Each run creates a new audit report. Repeat to collect further eligible files.
The default deletion limit is 1,000; the maximum is 10,000. Every apply takes a
new database and filesystem snapshot rather than trusting an old preview.

Only direct files in `/home/saleh/webKinPred/media/sequence_info/esm1v` qualify.
The script checks the expected sequence-ID filename and Numpy header, exact
matrix byte length, FP32 dtype, residue-by-1280 shape, a modification time before
29 October 2026, and a single hard link. It rejects symlink paths, pooled vectors,
unknown formats, changed files, and files open or memory mapped by a process.
It never recursively deletes a directory.

All unfinished artifact consumers and producers, live/uncertain attempts, and
protected, permanent, or non-regenerable catalog entries protect matching paths.
Both executor hosts must have fresh, reconciled telemetry and no legacy workers.
An unfinished job without a compatible execution specification blocks cleanup.
The graph-publication lock prevents new dependencies appearing during the short
safety window. Database queries time out after five seconds; deletion stops if
the guard dies or the window expires. Running computation continues.

Unexpected containers with access to the cache block apply. In the 4 November
inspection, `wk-maintenance-web19-candidate` was one such container: it mounted
the legacy media read-only and had `EXECUTION_V2_ENABLED=False`. The script does
not stop it or bypass the check. Retire that obsolete reader through the normal
deployment process before applying cleanup; any other readers must also be
resolved. An open file check alone cannot prove a legacy service will not open
another file later.

On 5 November the unused cutover pair `wk-maintenance-web19-candidate` and
`wk-maintenance-frontend17-candidate` was stopped (not removed; its only traffic
was its own health check). The next reported reader is `openkinetics-data-backend-1`,
the OpenKinetics data portal. It mounts `media/sequence_info` read-only, and its
code opens only `seqmap.sqlite3` and `<root>/<sequence_id>.npy` for the fixed
roots `esm2_layer_33/residue_vecs`, `esmc_layer_32/residue_vecs`,
`prot_t5_last/residue_vecs` and `pseq2sites_scores` (each path is confined to its
root); the repository never references ESM1v. For a reader reviewed like this,
pass `--trust-reader CONTAINER` (repeatable) on preview and apply. It allows only
read-only mounts by that container, never a legacy worker, and is recorded in
the summary and the audit report; the open-file audit still runs before
deletion. Review again if the reader's code or mounts change.

```bash
sudo python3 /home/saleh/webkinpred-execution-staging/cleanup_legacy_esm1v.py \
  --apply --limit 10000 --trust-reader openkinetics-data-backend-1 \
  --report "/home/saleh/webkinpred-execution-staging/esm1v-cleanup-$(date -u +%Y%m%dT%H%M%S).jsonl"
```

The original preview timed out while scanning about four million catalogue
records. The partial index addresses that lookup without weakening any retention
checks or increasing the query timeout. Legacy-reader blockers are independent
of this database fix and still prevent apply.

Verified production preview after the fix on 4 November: 8,958 directory entries,
8,954 candidates (17,974,081,792 bytes), four unrecognized filenames preserved,
and zero deletions. The valid partial index occupies 4,218,880 bytes. The legacy
maintenance container still blocks apply; no production deletion was performed.
These counts are a snapshot; every subsequent run checks current dependencies.

This is a cleanup for the managed platform after migration. It cannot infer
unrecorded future research requirements or coordinate arbitrary external scripts
that start reading files during cleanup. Stop such external readers first. To
preserve additional ESM1v files for another project, create a text file of their
exact `.npy` basenames and pass `--keep-file /absolute/path/keep.txt` on every run.

The T5/ESM2/ESMC research directories, pooled-vector caches, managed checksum
blobs, job inputs/results, and model files are outside this command's scope.
An atomically claimed candidate is rechecked in a private directory before
unlinking. Unexpected replacements are restored without overwriting another
file, or preserved in a `.cleanup-*` directory for inspection. Such directories
are never recursively removed, even on an error. Keep the JSONL audit reports.

The script deliberately keeps files when a matching managed consumer exists,
even if that consumer currently has a separate checksum-addressed copy. It
therefore favors preservation over recovering every possible byte.
