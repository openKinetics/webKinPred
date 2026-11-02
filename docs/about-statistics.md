# About page usage statistics

The public endpoint serves a small persisted snapshot. It never reads result CSVs
or performs an all-job/protein aggregation during a web request.

## Counting rules

- Jobs: records currently marked Completed, including completed jobs whose files
  are temporarily unavailable. Unavailable results are disclosed separately.
- Reactions: output rows with at least one successfully returned parameter value.
- Parameters: finite returned values, including supplied experimental overrides.
  Substrate arrays count their successful scalar slots individually; a CatRange
  interval counts as one estimate, not its two bounds. Empty, null, NaN, infinity
  and error text do not count. The total is the sum of the three parameter types.
- Distinct proteins: exact trimmed original sequences with a successful result,
  deduplicated across jobs/methods. Multi-sequence reduction metadata identifies
  successful candidates and excludes failed candidates; older output without that
  metadata falls back to splitting the successful row's protein list.

Header aliases come from the method registry, including CatRange and IECata's
current units and the older CataPro kcat/Km spelling and RealKcat's `Km (M)` output.
The job's requested targets and selected method determine which output columns
count: extra user-supplied measurement columns are excluded. Version the summary format
when changing these rules so historical results are reconciled again.

## Background reconciliation and freshness

`python manage.py about_stats_worker` runs as one dedicated, low-resource Compose
service. A PostgreSQL advisory lock prevents duplicate workers. Each iteration
selects at most 20 new/changed completed jobs, streams their CSVs, and stores a
summary plus unique SHA-256 protein memberships. Parsing runs outside database
transactions. Before committing, it rechecks the job's status/completion identity
under a row lock. Repeated delivery is idempotent; changed results replace their
old contributions. Database reads/upserts/deletes are batched across up to 20 jobs; the small-job
test uses nine queries for 20 jobs. Protein inserts are bounded to 1,000 per SQL
statement and buffered memberships are capped between files.

Completed status and output name/completion time are reconciled from the database;
notifications are only hints. Lost broker events do not lose statistics. Status
changes are excluded at snapshot time. Job deletion cascades summary/membership
rows, so restarting a public job cannot count it twice. Unreadable files retry
after 20 minutes, do not block other summaries, and never change job state or quota.

The initial historical backfill must finish before the first new snapshot is
published. Every five minutes the worker publishes a consistent PostgreSQL
repeatable-read snapshot from the summaries, with exact distinct-protein counting
in SQL. This scan is background-only and uses bounded PostgreSQL work memory and
no parallel query workers. Files already summarised are not reread on each refresh.

The endpoint uses a 60-second shared-cache entry and HTTP caching with ETag
revalidation. Both server and browser reject snapshots older than 20 minutes.
Missing/expired snapshots return 503/Retry-After rather than perform a full scan
or show permanently frozen numbers.

The About page revalidates on mount, refreshes every five minutes while visible,
and refreshes on focus/visibility restoration. Requests are coalesced, aborted on
unmount, and rate-limited to once per minute per mounted page. Local storage is
only a preview, expires using the original snapshot timestamp, and old cache
schemas are ignored. The page displays the snapshot update time.

## Operations and validation

- `python manage.py about_stats_worker --once --batch-size 20` reconciles one
  bounded batch and attempts publication. It leaves the prediction queue alone.
- Inspect `AboutJobStats.error` for unavailable results. After restoring a file,
  clear its `retry_after` to retry immediately, or wait for the retry interval.
- Keep the Compose `about-stats-worker` service running; it is independent of
  Redis delivery, Celery prediction tasks and executor model environments.

Tests cover scalar/array/range counts, current and historical headers, invalid
values, shared/multiple proteins, retries, changed results, status changes,
restarts/deletion, incomplete backfills, missing files, HTTP query count/ETag and
expiry. Browser cache tests verify freshness is not extended by page visits.

## Production validation, 2 November 2026

Backend/statistics worker release r62 uses source `072f853`; the frontend r63
uses `c6d6a85`. Migration 0041 adds only the summary/membership tables. Existing
model environments, executor images and supervisor processes were retained.

The initial historical backfill covered 7,470 completed jobs with zero pending
summaries and zero unavailable/unreadable results. A header-only audit established
RealKcat's historical `Km (M)` column and distinguished extra input measurement
columns from requested result columns. Previously summarised results were reused
only when all counted columns were identical; changed cases were reparsed.

The first snapshot was published at 12:35:45 UTC. Normal background operation then
summarised newly completed jobs within approximately 6–28 seconds in the checked
examples. The next automatic snapshot at 12:40:59 included four more completed
jobs, with no manual counter updates:

| Metric | Published value at 12:40:59 UTC |
|---|---:|
| Completed jobs | 7,474 |
| Successful reaction rows | 6,629,149 |
| Distinct successful protein sequences | 538,347 |
| Parameter values, total | 8,959,541 |
| kcat | 4,279,085 |
| Km | 3,798,134 |
| kcat/Km | 882,322 |
| Unavailable result files | 0 |

The parameter total equals the three component totals. Historical values can move
up or down because the previous implementation missed formats and counted invalid
values/unrequested input columns; continued job completions also change totals.

Validation passed 517 backend tests plus three subtests, three browser-cache unit
tests, and the frontend production build. Tests include a nine-query reconciliation
of 20 small completed jobs. Production query capture verified one simple row query
on a cold stats read and zero queries on a cached read; neither path opened CSVs
or ran aggregates. Cached Python view processing measured approximately 0.92 ms
(excluding web-server queuing, TLS and network time). A public HTTPS request to
`predictor.openkinetics.org/api/about-stats/` returned 200 with the new snapshot
in approximately 264 ms. Conditional GET returned 304 with ETag and max-age=60.

Headless Chrome exercised the deployed bundle with controlled responses and a
controlled clock: fetch on mount, reject expired local previews, timer refresh,
focus refresh, and coalescing focus/visibility events. A separate browser visit
rendered the genuine production counts and update timestamp.

Preservation checks retained all 9,498 jobs that were unfinished at rollout and
all 11,671 original immutable execution specification/input checksums. All three
attempts running before rollout completed normally. Both hosts kept fresh
heartbeats; production executor/artifact supervisor images and start times were
unchanged. Frozen legacy Redis prediction messages remained at 5,002. No quota,
job input, result file or model weight was rewritten by the statistics work.

The worker used a temporary 0.5-CPU ceiling during the one-time backfill and was
returned to its configured 0.25 CPU afterward, with no restarts after its final
rollout. Its observed RSS during the backfill was approximately 88 MiB. Production
retained approximately 23 GiB free disk after the summary tables were populated.

Operational evidence is under
`/home/saleh/webkinpred-execution-staging/`: `preservation60.json`,
`verification60.json`, `statistics-verification63.json`, `promotion61.json`
(in the backend temporary directory), release image manifests, deployment logs
and the preserved predeployment environment files. No credentials are recorded
in this document.
