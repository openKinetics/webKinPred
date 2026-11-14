# Execution operations, 14 November 2026

## Collector starved by long preparation chunks (control release 144)

A background API batch of 5,000-row requests (submitted 13 November, 03:57-04:00) was being
prepared continuously. Each preparation chunk holds the graph advisory lock (`481726392`) for
its whole transaction; sampling `pg_locks` showed the controller holding it for 27-45 s per
chunk, often idle in transaction between queries, with gaps of about a second. The artifact
collector waits at most 2.5 s for that lock, so production created no cleanup from 03:18 to
03:57 (a rolled-back trace of `_claim_cleanup("production")` failed with a lock timeout after
2.5 s). Its 896 pending hints were consumed OmniESI residue inputs (1.17 GB) with no
unfinished consumer; production's free space fell to 18.7 GiB, and launches were refused for
disk space (`waiting_disk`, up to 195 per half hour) because admission keeps 15 GiB free.

Commit `4c890aa` adds `_yield_to_cleanup` after each committed chunk: while a host has pending
deletion hints and no cleanup created in the last minute (or one still open), preparation waits
(1 s polls) for at most 20 s before taking the lock for the next chunk. Supervisors ask for
cleanups about every 16 s, so each starved host gets a turn; preparation pauses at most about
once a minute per host and only while deletions are pending.

Control image `webkinpred-control-fix:r144`
(`sha256:da2456fc3c213f79594199996623c6e6e3113cc06f44845afa42f25f26febd1e`, overlay on r142)
runs since 03:57:50; all 167 unfinished requests were present afterwards with unchanged
checksums and the legacy queue still held 5,002 messages. Within three minutes production's
pending hints fell from 896 to 0 (two cleanups, 256 blobs) and its free space rose to 19.1 GiB.

## Nightly backup no longer fit production's disk (backup tool)

The verified nightly backup (`webkinpred-backup.timer`, 04:10 UTC) wrote each `pg_dump` to
`/var/lib/webkinpred/backup-scratch` before uploading it. The live database (25 GB) now dumps
to about 5 GB, and with production at about 19 GiB free the run failed at 04:32 with "Backup
reached its scratch headroom guard" (15 GiB floor). systemd retries failed runs every 30 min
(`Restart=on-failure`, `RestartSec=30min`); the 05:02 retry dumped 5.3 GB by 05:20 and held
production at 15 GiB free, below the headroom that admits execution work, so production ran
nothing while it dumped.

Commit `d8423ba` streams each dump from `pg_dump` into the existing verified upload (the media
archive was already streamed): scratch holds only the manifest, the round-trip
download/checksum verification is unchanged, and a non-zero `pg_dump` exit fails the run
before any completion manifest is written. Installed at
`/usr/local/lib/webkinpred-execution/backup.py` at 05:21 (previous version kept as
`backup.py.before-stream`); the old-code retry was left to fail at its guard and clean up its
own scratch, and the next systemd retry runs the streaming version.

The streaming retry started at 05:52:23 and finished at 06:27:41 with
`{"complete": true, "objects_verified": 5}` (three database dumps, the job-media archive and
the manifest, each downloaded and checksum-verified) under
`s3://openkinetics-backup/backups/execution-v2/2026-11-14T05-52-23Z`. Its scratch directory
stayed empty and production's free space held at 18.4-19.0 GiB throughout.

## Backend workers exhausted by completions waiting on preparation (release 145)

At 07:01-07:02 six attempt completions (`.../attempts/<id>/complete`) took 28-32 s at the
backend, and 145 took over 5 s in 90 minutes. `finish` locks the output artifacts with
`select_for_update`, and a preparation chunk of the 5,000-row batch that attaches the same
artifacts holds them for its whole 30-45 s transaction. The backend ran 6 sync gunicorn
workers while the two supervisors commit up to 4 completions each, so the waiting completions
occupied every worker; the GPU supervisor's batched heartbeats then timed out (07:03:13 and
07:03:26) and two leases expired.

Release 145 changes only the private override `live20/compose.local.json`: the backend runs
12 workers (`--workers 12`) with a 2 GiB memory limit (was 6 and 1.5 GiB), on the same control
image as release 144 (`sha256:da2456fc...`). Only the backend was recreated (07:20:21-26;
`staging/release145/deploy_control145.sh`); all 151 unfinished requests were present
afterwards with unchanged checksums and the legacy queue still held 5,002 messages. Five
minutes later the backend used 1.1 GiB of 2 GiB, completions still waited up to 11 s on
preparation locks, and no heartbeat timed out or lease was lost.

## Completion commits scanned every artifact location (control release 146)

`pg_stat_database` showed the live database writing temporary files at 45 per minute and
65.6 MB/s (8.8 TB since its statistics were last reset) on production's rotating disk.
Logging temporary files over 20 MB for 75 s (`log_temp_files`, reset afterwards) attributed all
of them to `release_consumed_residues`, which runs inside every attempt completion: its join of
ready locations to the artifact table was planned as a sequential scan of all 7.4 M locations
with a hash that spilled to disk (8.0 s, 1 GB read and 522 MB of temporary files for one stage
with 129 inputs). `work_mem` is the default 4 MB and `shared_buffers` 160 MB.

Commit `254a7ed` resolves the stage's transient checksums, their aliases and their ready
copies with three indexed queries (0.03 s warm, 1.0 s cold for the same stage) and skips
unproduced (empty) checksums. Control image `webkinpred-control-fix:r146`
(`sha256:fab4170f4f15b1182b2e84cade66a70bc9c74498816a2d68986c0e1e4680aba4`, overlay on r144,
backend still 12 workers) runs since 11:18:47; all 129 unfinished requests were present
afterwards with unchanged checksums and the legacy queue still held 5,002 messages. Five
minutes later temporary files fell to 1 per minute and 1.9 MB/s, completions took 164 ms
(median, 159 calls; 95th percentile 637 ms, against 5-32 s before), and production's I/O
pressure fell to 2.5 % (avg60).
