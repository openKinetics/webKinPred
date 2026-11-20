# Execution operations, 20 November 2026

## Long-SMILES envelopes for CatPred, TurNup and IECata

Three normal requests waited at prediction stages marked `profiling_required` because their
substrate SMILES exceed the 257-character envelope (largest 290-439 characters); protein
length, rows, units and input bytes were within their profiles' limits:

* h1Bm1wh (IECata kcat/Km): 1 stage, waiting since 17 November.
* uwZRo1r (CatPred kcat): 5 stages, waiting since 19 November.
* kEjievW (TurNup kcat): 5 stages, waiting since 19 November.

None of these methods changes its memory shape with SMILES length beyond a small bound:

* TurNup encodes each reaction as a fixed-size difference fingerprint.
* IECata pads each substrate graph to 290 nodes.
* CatPred builds one small message-passing graph per substrate (one node per heavy atom), a few
  MB per 128-row chunk beside a multi-GiB reservation.

`staging/extend_envelopes_1120.py` cloned the following profiles to `-mol1024`
(`max_molecule_length` 1024, RAM ×1.25; same image, fingerprint, settings and model identity):

* `gpu29-CatPred-{kcat,Km}` and `production29-CatPred-{kcat,Km}`
* `gpu16-TurNup-predict-kcat` and `production70-TurNup-kcat`
* `gpu54-IECata-predict-kcat_Km` and `production70-IECata-kcat_Km`

All 11 stages were bound at once and no stage remained unbound.

## Production disk

Production had 16.06 GiB free (1,090 MiB above its 15 GiB headroom) against about 1,222 MiB per
assembly reservation, so assemblies waited again (`waiting_disk`). Since 16 November the
execution database grew 1.4 GiB, blobs 0.7 GiB, job media 0.3 GiB and container logs 0.3 GiB.
A `reconstructor-web:latest` image built on 17 November (2.38 GB) left its previous image
untagged (2.37 GB, `c3d863daa574`), and Docker's build cache grew to 4.0 GB (0.78 GB
reclaimable). The journal had grown back to 0.62 GiB.

## Where production's disk went, and what was removed

The owner approved removing the old `reconstructor-web` image and asked for anything else that
could go without consequences. At 11:24 the untagged image (`c3d863daa574`, 2.37 GB) and
Docker's unused build cache (0.78 GB) were removed: free space rose from 17.2 GB to 20.6 GB
(19.2 GiB). The assemblies that had been waiting ran at once. h1Bm1wh, uwZRo1r and kEjievW
completed at 11:24-11:26, and by 11:41 no request was pending or processing.

Of production's 422 GiB in use, the largest part (152 GiB) was the pre-migration sequence cache
`/home/saleh/webKinPred/media/sequence_info`. Execution v2 never reads it:

* Attempts stage their inputs from the blob store into their own work directory.
* Locations are (artifact, host) rows that point into the blob store, never at host paths.
* `execution_import` imported jobs at cutover, not caches.
* No cache file had a second hard link, so none was shared with the blob store.

Its only reader is the OpenKinetics data portal (`openkinetics-data-backend-1`, mounted read
only), which serves `seqmap.sqlite3`, the residue roots `esm2_layer_33/residue_vecs` (35 GB),
`esmc_layer_32/residue_vecs` (32 GB) and `prot_t5_last/residue_vecs` (29 GB), and
`pseq2sites_scores`. Its release scripts read the same roots, plus `media/pseq2sites`. No
process had a file under the cache open, and no legacy worker was running.

`tools/cleanup_legacy_sequence_cache.py` removes everything else, about 55 GiB in 7.1 million
files:

* `catpred_esm2` (37 GB, 4.76 M files)
* the pooled `mean_vecs` and `weighted_vecs` of every KinForm root (about 14 GB)
* the residue caches the portal does not serve (`esm2_layer_26`, `prot_t5_layer_19`)
* `esm1b_turnup`, `catrange_esmc`, `esm2_layer_last_mean`, IECata, OmniESI and ESM1v
* three old seqmap copies (`seqmap.corrupt.1790169998`, `seqmap.sqlite3.bak`, `.bak2`; 1.8 GB)

It refuses any path inside the portal's roots. It runs at idle CPU and I/O priority from
`staging/legacy_cache_cleanup_1120.py` and logs each directory to
`staging/legacy-cache-cleanup-1120.jsonl`. Re-running it continues where it stopped.

Throttling matters on this rotating disk. Production's own load keeps full I/O pressure near
1 %. The first, unthrottled run raised the one-minute average to 5 %, the supervisors'
host-pressure limit, within minutes, because each burst's journal writes land after it. The
script waits while full pressure exceeds 2 % (10 s) or 2.5 % (60 s). At up to 800 files a
second it still produced one-minute peaks of 8-10 % at 12:15 and 12:45, because each burst ran
until the 10 s average caught up. Since 12:48 it unlinks at most 100 files every 0.5 s, so the
remaining 5.3 million files take most of a day.

On the GPU host the owner's research downloaded `esm1v_t33_650M_UR90S_2.pt` (7.8 GB) to
`/home/msp/.cache/torch` at 11:20. That cache belongs to the research, and it was left alone.
`webkinpred-execution-staging/probes`, profiling probe outputs from 28 October to 2 November
that nothing mounts or reads, was removed (listing in `probes-deleted-20261120.txt`). The GPU
host's free space rose from 128 GB to 213 GB.

What still grows on production (9 to 20 November):

* Execution database: +9.4 GiB (31 GB now; about 0.85 GiB a day).
  * `api_artifact`: 10 GB for 8.6 M rows. Each identity repeats about 570 bytes of model
    identity.
  * `api_workstage` payloads: 6.9 GB. Finished embedding and prediction stages keep 26-36 KB
    each.
  * Dependency and location rows: 8.7 GB.
* Blobs: +5.2 GiB (permanent pooled vectors and prediction copies).
* Job inputs and results: +2.1 GiB.

Left in place:

* `production-backup/media` (13 GiB) on the GPU host: a cutover copy of job results.
* The other cutover leftovers on the GPU host (3.4 GiB): `production-backup/rehearsal`,
  `production-backup/container-model-backup` and `cutover20`.
* The rehearsal databases `webkinpred`, `webkinpred_seqmap` and `webkinpred_predictions`
  (about 800 MB): no connections, superseded by the `_live_20261028` databases.
* Release build directories in staging (1.5 GB).
* `production-backup/models` on the GPU host: mostly hard links to the model tree, and the
  source for `repair_model_copy.py`.
