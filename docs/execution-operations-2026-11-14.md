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

## Progress and preparation queries that still loaded production (control releases 147-149)

After release 146, production's I/O pressure was still 8-11 % (avg60). Sampling
`pg_stat_activity` showed the progress projector's shared-embedding count
(`shared_embedding_summaries`) in 130 of 240 samples: release 142 spaced each large request by
nine times its scan, but about 40 background requests of the 5,000-row batch were large at
once, so the projector ran continuously. Commit `d4de94c` (release 147) makes a background
request also wait for every other large projection pending (`_spacing`), so together they stay
near a ninth of the projector; normal requests keep their own spacing. Afterwards the query
appeared in 1 of 240 samples, 20 large projections were spaced 9-25 minutes apart, and I/O
pressure fell below 1 % (avg60).

Temporary files then rose to 81 per minute (12.8 MB/s). Logging temporary files over 5 MB
(reset afterwards) attributed them to the controller's per-tick query for due preparations,
which ordered whole request rows, job specifications included. Commit `7c51b00` (release 148)
removed its redundant DISTINCT (one stage per key), which did not stop the spill; commit
`791faae` (release 149) ranks the due requests' identifiers only and loads the chosen rows by
key. Control image `webkinpred-control-fix:r149`
(`sha256:b74c2576e01690f2cbe233e190acc853020944ae26ab3acf6381d82d53b0ae98`) runs since
17:52:28; all 345 unfinished requests were present afterwards with unchanged checksums and the
legacy queue still held 5,002 messages. Four minutes later temporary files were 1 per minute
(0.1 MB/s) and I/O pressure 0.75 % (avg60).

## Cleanup batches and production's disk (control release 150)

With production's free space (16.3 GiB) under its cache target (headroom 15 GiB + 10 GiB),
older cache copies become collectible, but a sweep call stopped at the first page that held
any candidate. Commit `df1c0a5` keeps paging within the same 0.5 s budget until 256 candidates
are found and ends the pass at the last location instead of wrapping once it holds candidates.
Control image `webkinpred-control-fix:r150`
(`sha256:5e188b5cef8759483f05f39372537a0fedf7b46c1fb3a857fe87ff325a0f6a3f`, overlay on r149)
runs since 18:57:53; all 342 unfinished requests were present afterwards with unchanged
checksums and the legacy queue still held 5,002 messages.

The gain on production is small. Of its 13,949 ready regenerable cache copies (1.68 GiB),
13,103 share their bytes with a permanent or non-regenerable artifact and 96 with unfinished
stages, so only 344 (21 MiB) are collectible; the rest are correctly retained. By distinct
checksum, production's blob store holds 6.76 GiB of permanent regenerable embeddings (pooled
vectors), 2.48 GiB of permanent non-regenerable files, 0.21 GiB protected and under 0.1 GiB of
cache and transient files.

Production's free space fell from 28.8 GiB (9 November) to 16.3 GiB, 3.8 GiB of it since
midnight: blobs +1.45 GiB (permanent embeddings), the database volume +1.33 GiB and job media
+0.48 GiB; the legacy `media/sequence_info` (136.6 GiB) is unchanged. Admission keeps the 15 GiB
headroom free, so production runs one attempt at a time (a CPU prediction reserves 1.01 GiB,
the 1 GiB disk floor) and completed 1-7 attempts per 10 minutes against about 150 on the GPU
host. In the database, `api_workstage` is 6.5 GB, almost all of it TOAST holding live stage
payloads (18 KB average compressed: completed predictions 40 KB, completed embeddings 28 KB,
canceled embeddings 12 KB); `api_artifact` is 9.9 GB and `api_artifactdependency` 5.8 GB.
The disk is 97 % used (422 of 439 GB); images take 127 GB and the staging directory 42 GB
(36 GB of it `models-release`).
