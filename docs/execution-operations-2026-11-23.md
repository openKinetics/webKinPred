# Execution operations, 23 November 2026

## Downloading an unfinished job's output returned 500 (release 157)

At 16:15:49, `/api/jobs/Q3ixgk9/download/` raised `IsADirectoryError: '/app/media/'`.
Q3ixgk9 (14,950 CataPro rows) was still assembling and completed at 16:37:43. Its
`output_file` was still empty, so `download_job_output` built `MEDIA_ROOT + "/"`. That
directory exists, so the view tried to open it. The bug predates execution v2.

Commit 5e6c77d raises 404 ("No output file for this job.") unless the job has an output name
and the path is a regular file. `tests/test_download_unfinished_job.py` covers an unfinished
job (fails without the change) and a completed download. The full suite gave 753 passed, with
the same 5 setup errors as before.

Image `webkinpred-control-fix:r157` (`sha256:a03970c4…`) is built from r156 and replaces only
`api/views/file_views.py`; the image's copy matched the repository before the change.
`staging/release157/deploy_control157.sh` recreated the backend, controller and control worker
at 16:49:51-16:50:03, while no request was unfinished. The legacy queue still held 5,002
messages. In the new backend, Q3ixgk9's download returned its 10.4 MB CSV, and an unknown job
returned 404. In its first three minutes the backend served 55 requests without an error.

The same day, two CataPro requests (13:57) needed 714 ProtT5 embedding stages. The GPU ran 3-4
at a time, limited by VRAM reservations, at 150-290 stages per half hour. Short stages spend
much of their time starting a container and loading the model. Both requests had completed
by 16:37.

**Build the next control release from `webkinpred-control-fix:r157`.**
