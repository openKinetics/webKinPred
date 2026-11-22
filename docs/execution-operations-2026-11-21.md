# Execution operations, 21 November 2026

## Two polling queries read whole tables (migrations 0046 and 0047)

On 20 November at 14:16, and again on 21 November at 08:15, production's one-minute full I/O
pressure reached 7-10 % with no request running and no cleanup or backup active. In 30 s
Postgres read 435,000-531,000 blocks (3.4-4.3 GB) from outside its 128 MB shared buffers.
Those spikes turned out to be the monitoring pass itself (see the next section). The reads
were a separate, constant load. Per-table counters (`pg_statio_user_tables`,
`pg_stat_user_tables`) over 60 s showed two causes:

* **Outbox.** The progress projector asks for the oldest unpublished event
  (`published_at IS NULL ... ORDER BY id LIMIT 1`) about 5.5 times a second. With no index
  on `published_at`, every poll was a sequential scan of all 227,357 rows (47 MB), although
  none was unpublished. That was 476,000 blocks a minute, 4.66 million scans since
  28 October.
* **Scratch release.** Every cleanup claim (`retention._claim_cleanup`) selects a host's
  finished attempts whose scratch is not yet released. Only 7 of 206,490 attempts qualified,
  but the query read every attempt the host had run, sequentially on the GPU host (155,000
  attempts). That was 160,000 blocks a minute.

Migration `0046_outbox_unpublished_index` adds `outbox_unpublished_id`, `(id) WHERE
published_at IS NULL` (commit 8ccd984, image `webkinpred-control-fix:r153`,
`sha256:30b51d59dcc8…`). Migration `0047_attempt_unreleased_scratch_index` adds
`attempt_unreleased_scratch`, `(host_id, started_at) WHERE scratch_released_at IS NULL`
(commit 680404d, image `r154`, `sha256:80b8c894…`, built from r153). Both build concurrently
on PostgreSQL. `staging/release153/migrate153.sh apply` and `release154/migrate154.sh apply`
ran each in a one-off container at 08:24 and 08:32; each took 3 s. The backend, controller,
control worker and supervisors were not restarted, so they still run r151 code, and the
running code does not depend on the indexes. A queue snapshot (`preservation113/*-before153`)
was taken first.

After the indexes:

* The outbox had 1 sequential scan in 30 s, down from 164.
* The cleanup claim's estimated cost fell from 24,417 to 156.
* Idle block reads fell from 435,000 to 132,000 per 30 s.

The rest comes mostly from the retention sweep's artifact index lookups (140,000 blocks a
minute) and the about-page statistics worker scanning `api_aboutjobprotein` (58,000 a
minute). `api_job` is scanned about 3 times a second by the controller's preparation and
active-execution queries, but it fits in shared buffers.

**Build the next control release from `webkinpred-control-fix:r154`, so that it keeps
migrations 0046 and 0047.**

## The monitoring pass caused the idle pressure spikes

The spikes continued after the indexes (6.7 % at 09:45). A sampler of disk stall time and
per-process I/O ran from 09:47 to 10:17. In 27 minutes no 5 s interval stalled for more than
400 ms until 10:14, the next monitoring pass, when `monitor_sizes.sh` stalled the disk for
0.4-1.2 s per interval. Each pass ran `du` over the blob store (millions of small files),
job media, the legacy cache and staging, plus `docker system df`, which walks the volumes.
On the rotating disk that was the 7-10 % minute at :14-:16 and :44-:46 of every hour.

Since 10:15 `staging/monitor_sizes.sh` measures only small directories on every pass. Since
12:46 Loki's data volume (328,488 chunk files) also counts as a large tree. Since 19:46 the
database volume does too. Its 3,500 files' metadata is evicted between passes, and reading it again
from the rotating disk still gave 2-4 % minutes. The status query reports the database size anyway. The three-hourly walk itself still
gives one 4-5 % minute, so since 23:16 it also waits while any request is pending or processing,
unless its values are 12 hours old. On 22 November it reached 6.6 % for a minute at 09:44, so since then
it runs every six hours instead of three. The large trees
and `docker system df` are measured at most every three hours, at idle CPU and I/O
priority, and the last values are repeated in between (`/var/tmp/webkinpred-monitor-heavy.txt`).
The size step of a pass now takes under a second instead of about 30 s. The previous script is kept as
`monitor_sizes.sh.before1121`.

## Similarity summaries longer than 300 s are cut off

At 02:11 a 5,391-row request's pre-submission `POST /api/sequence-similarity-summary/` ran
past gunicorn's 300 s worker timeout. gunicorn killed the worker (`WORKER TIMEOUT`, then
SIGKILL), together with the matching `/api/progress-stream/` request. The view streams a space
every 10 s, but a sync worker is killed by its timeout however much it writes. The request
xNAZjKm was submitted at 02:26 and completed at about 03:40. The pre-migration deployment used
the same sync workers and timeout. These were the only two worker timeouts in the week's log.
Raising the limit, or moving to threaded workers, waits for the owner's decision.
