# Execution operations, 24 November 2026

## JSON API submissions above 2.5 MB were refused (release 158)

At 07:39 and 07:44, `POST /api/v1/submit/` returned 400 with
`RequestDataTooBig: Request body exceeded settings.DATA_UPLOAD_MAX_MEMORY_SIZE`. The v1 API
accepts JSON bodies of up to 10,000 rows (`_parse_json_body`, which points larger datasets to
CSV upload). However, the project never set `DATA_UPLOAD_MAX_MEMORY_SIZE`, so Django's 2.5 MB
default refused bodies of roughly 2,500 typical protein rows or more, with a bare HTML 400,
before the view ran.

Commit ec4af8e sets `DATA_UPLOAD_MAX_MEMORY_SIZE = 64 * 1024 * 1024` in `webKinPred/settings.py`.
That is at most 768 MB in memory across the 12 gunicorn workers; production had 15 GB
available. Both JSON parsers (`submit` and `validate`) now return a JSON 413 that names the
limit and points to CSV upload when a body is still too large.
`tests/test_api_json_body_size.py` covers a 3,000-row body above 2.5 MB and the 413 response;
both tests fail without the change. The full suite gave 755 passed, with the same 5 setup
errors as before.

Image `webkinpred-control-fix:r158` (`sha256:66471043…`) is built from r157 and replaces
`api/views/v1_views.py` and `webKinPred/settings.py`; the image's copies matched the repository
before the change. `staging/release158/deploy_control158.sh` recreated the backend, controller
and control worker at 07:48:20-07:48:32.

Preservation (`preservation113/*158*`): all 15 unfinished requests (6 processing, 9 pending)
were present afterwards with unchanged checksums, and the legacy queue still held 5,002
messages. The new backend reports a 67,108,864-byte limit.

Those 15 requests had waited on GPU embeddings since 02:40. From about 03:15 the owner's
research used the GPU above the 40 % back-off threshold (it was at 60-88 % when sampled),
so no webKinPred GPU work started. That is the configured priority.

**Build the next control release from `webkinpred-control-fix:r158`.**
