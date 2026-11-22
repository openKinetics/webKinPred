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

**Build the next control release from `webkinpred-control-fix:r155`.**
