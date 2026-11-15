# Execution operations, 15 November 2026

## Collection deleted bytes an attempt was publishing again (supervisor release 152)

At 01:32-01:33 three KinForm-H binding-site stages on the GPU host stopped with
`missing_embedding` ("Required artifacts must be rescheduled"): the executor confirmed that
three of their t5 residue inputs were absent from its blob store, and recovery scheduled
`restore-batch` stages to rebuild them. These were the only such outcomes in 48 hours. The
producing attempt had completed at 01:28:50 and registered the three residues as ready on the
GPU host; a GPU-host cleanup created and acknowledged at 01:28:49 had deleted blobs with exactly
those three checksums. The residues had been produced before, and the earlier copies were
collectible because their consumers had finished.

Control protects a checksum while any alias has an unfinished consumer or a live producer,
but a running attempt's output artifact has no checksum until its commit, so the new copies
could not be matched. The executor publishes an attempt's outputs on a transfer thread (and
returns at once when an identical verified blob is already stored), then commits them on a
control thread, while collection deletes blobs on the supervision thread. Commit `5ea7605`
records the checksums each supervised attempt publishes before the store looks for an existing
copy, and collection skips them until that attempt leaves supervision (committed or
abandoned). An abandoned attempt can leave such a blob on disk without a location.

Supervisor image `webkinpred-supervisor:r152-gpu`
(`sha256:75b24fa74984f3e1cdd8844ac2386e4d1fc62527bb0244fe324ffc1b104991ce`, overlay on r143,
rollback container `webkinpred-gpu-executor-rollback152`) runs on the GPU host since 01:55:31.
The drain finished in 26 s without stopping any attempt; all 332 unfinished requests were
present afterwards with unchanged checksums and the legacy queue still held 5,002 messages.
Production's supervisor (r138) is unchanged.

## Production's disk held finished requests back

From 21:51 on 14 November no request completed. Production had 16.1-16.2 GiB free against its
15 GiB headroom, so its 1.19 GiB assembly reservation (the 1 GiB disk floor over a measured
207 MB peak) fitted only for the smallest inputs, and background preparation (which waits for
headroom plus 2 GiB) stayed paused for 295 queued requests. By 03:14 the GPU host had finished
every prepared embedding and prediction and was idle. At 03:22 and 03:28 two normal requests
(gW5xsFA, 9 rows, and OzlZMcA, 309 rows) computed every prediction within seconds, but their
final assemblies waited for disk: 1,176 MiB was available above the headroom against 1,222 MiB
reserved.

The preservation snapshots taken around each control release (`preservation113`, 242 MB) were
all verified, and the database is the record they checked. Removing 270 intermediate snapshot
files (222 MB), keeping the first baseline (`*-before.*`) and the snapshots of releases
150-152, left 16.36 GiB free. Assemblies resumed at 03:48:55, and both normal requests
completed at 03:50:05 and 03:50:09. Assemblies with larger inputs still wait (`waiting_disk`);
preparation stays paused until production has 17 GiB free.
