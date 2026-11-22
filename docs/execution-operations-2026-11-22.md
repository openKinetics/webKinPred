# Execution operations, 22 November 2026

## Submissions with experimental values failed (release 155)

From 12:06 to 12:11, `/api/submit-job/` returned HTTP 500 eleven times. Every traceback ended
in `api.execution.control.digest` with `ValueError: Out of range float values are not JSON
compliant: nan`.

When a user asks for experimental values, `get_experimental.lookup_experimental` returns every
column of the first matching row of the experimental CSV (`row.to_dict()`). Blank cells become
`NaN`, for example a missing standard deviation, pH or mutation. The execution-v2 ingestion
checksums the specification with `allow_nan=False`, and a jsonb column cannot store `NaN`
either, so any request whose experimental match had a blank field could not be submitted.
The pre-migration Celery path serialized `NaN` without complaint.

Commit b0563d3 turns `NaN` cells into `None` (JSON null) in each matched row. The Extra
Information text treats both as not reported (`extra_info._nullish`), and `_source` compares
flags with `== 1`, so the output text is unchanged. The one exception is a blank `protein_ID`,
which would now print `None` instead of `nan`. Predictions are unaffected. A new test,
`tests/test_experimental_lookup_json.py`, fails without the change. The full suite gave 749
passed, with the same 5 setup errors (missing `seqmap` database) as before the change.

Image `webkinpred-control-fix:r155` (`sha256:2b3e5ec0…`) is built from r154, so it keeps
migrations 0046 and 0047, and adds only `api/utils/get_experimental.py`. The image's copy
matched the repository before the change. At 12:18:19-12:18:31,
`staging/release155/deploy_control155.sh` recreated the backend, controller and control
worker. The database and Redis were not touched.

Preservation (`preservation113/*155*`):

* There were no unfinished requests at the baseline, and none were missing afterwards.
* The legacy queue still held 5,002 messages; the Redis snapshots differed only in one cache
  key's TTL.

In its first minute the backend served 99 requests without an error. Inside the new backend,
rows from the Km table (56,976 rows) and the kcat table (23,757 rows) with blank fields were
looked up again. Each returned 4 null fields, and the specification checksum succeeded.

Release 156 (below) replaced it at 18:24.

## Restore batches traded outputs and could not commit (release 156)

At 18:08 a request's EITLEM predictions needed ESM1v residue embeddings that had already been
collected, so recovery scheduled `restore-batch` stages. From 18:09 to 18:15, 16 GPU attempts
of two batches of one recovery group (`restore-batch:5166119d…:1` and `…:2`) were stopped as
`executor_reconciled`. Each ran 20-40 s. Its completion was then rejected with 409, and the
supervisor reported a lost lease. Seventy-three predictions waited on them. Batch `:2`
completed at 18:15:57 and `:1` at 18:18:16. The previous eight days had 11 such stops in all.

`recovery.missing_dependencies` waited for a missing output's producer only if the producer
was running, held, or belonged to another request. When the producer was a restore batch of
the same request that was waiting to retry, the output was moved to a new batch. The older
batch then no longer owned every output it declared, and `control.finish` rejects such a
completion: "Artifact producer ownership mismatch" while the new owner had not committed,
"Artifact identity conflict" after it had. `executor_reconciled` counts as a temporary
failure, so a batch could retry indefinitely.

Commit 7d5b2d3:

* Recovery now also waits for any unfinished restore batch that owns the output.
* `finish` keeps an output that another stage owns, logging `execution.output_reassigned`,
  and commits the attempt's own outputs instead of rejecting all of them.

Two new tests in `tests/test_execution_recovery_batches.py` fail without the change. The full
suite gave 751 passed, with the same 5 setup errors as before.

Image `webkinpred-control-fix:r156` (`sha256:c16689a8…`) is built from r155 and replaces
`api/execution/control.py` and `api/execution/recovery.py`; the image's copies matched the
repository before the change. `staging/release156/deploy_control156.sh` recreated the backend,
controller and control worker at 18:23:49-18:24:01.

Preservation (`preservation113/*156*`): the one unfinished request was present afterwards with
an unchanged checksum, and the legacy queue still held 5,002 messages. In the first two minutes
the backend served 161 requests without an error. One GPU attempt stopped as `interrupted`
around the restart; that category is temporary, so its stage retries.

**Build the next control release from `webkinpred-control-fix:r156`.**
