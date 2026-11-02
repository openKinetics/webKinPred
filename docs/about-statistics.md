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
