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
