# Execution operations, 11 November 2026

## Long substrate SMILES beyond measured envelopes

Three normal requests submitted at 07:27 (m9vDYWG, 13h6CwL, XN3lVBz) have substrate
SMILES of up to 813 characters. 38 prediction stages (CataPro, KinForm-H, CatRange
kcat) exceeded `max_molecule_length` 257 and 8 molecular embedding stages (CataPro,
CatRange) exceeded `max_length` 512, so they waited as `profiling_required`.

These steps do not grow with SMILES length beyond a fixed bound. CatRange's ChemBERTa
encoder rejects inputs over its 510-token limit with an explicit message
(`CatRangeInputError`); CataPro's MolT5 tokenizer truncates at 512 tokens
(`truncation=True`, batch size 1); CataPro, CatRange and KinForm-H prediction read
fixed-size precomputed embeddings. Neither method's behaviour changes.

Envelope-extension clones (same image, fingerprint, settings and model identity; the
reason is recorded in each profile's `evidence.envelope_extension`), created with
`staging/extend_envelopes_1111.py` at about 07:55:

* `gpu16-{CatRange,CataPro}-molecular-len1024`: `max_length` 1024, RAM and VRAM ×1.25.
* `*-mol1024` for every target of CataPro (`gpu83`, `production70`), CatRange
  (`gpu55`, `production55`) and KinForm-H (`gpu54`, `production70`):
  `max_molecule_length` 1024, RAM ×1.25.

All 46 stages were bound immediately; none remained `profiling_required` or unbound.

## Status polls saturated production I/O (control release 139)

From about 11:20 the 6,707-row normal request aLMfftl waited at its final assembly
while production's I/O pressure stayed near 30 % (`full avg300`) and its assembly
launches were refused as `host_pressure`. The load was the job's own status page: each
`/api/job-status/aLMfftl/` request took about 19 s, polled every ~20 s by the
submitter's browser. To report active execution time, `execution_status` joined every
input artifact of the request's stages to find their producers: 134,460 dependency
rows joined to the 5.7-million-row artifact table, read at about 390 MB/s per poll.

Attaching an input links its consumer to the producer stage, and cohort and recovery
rewrites move those links; the scheduler waits on the same links. Commit `c74e3cd`
reads the producers from that stage graph instead (381 producers in 0.5 s for this
request). Control image `webkinpred-control-fix:r139`
(`sha256:c6314a3cad683833b78799481521631af63db82d463c05e68bc34a8aa4b2f0c6`, also
carrying release 138's verified-blob store) runs since 11:54. Both unfinished jobs were
present afterwards with unchanged checksums and the legacy queue still held 5,002
messages. Status requests for that job then took 108 ms (median) and I/O pressure fell
to 1.4 % within two minutes.

## History pruning (release 140, user-approved)

`execution_prune_history` (commits `9c8c15e`, `7bab5d0`, `8d3a5e6`; dry run unless
`--apply`) removes, for requests completed at least 24 h ago without a pending progress
projection, the input links of their finished prediction, similarity and assembly stages,
and the catalogue and location rows of residue matrices already deleted from disk that
nothing references and whose producer finished. Embedding stages and the KinForm pool and
binding steps keep their inputs (recovery regenerates lost outputs from them); stage
payloads, job, stage and attempt records, results and pooled vectors are kept. It runs in a
one-off container (`staging/release140/prune140.sh`, image `webkinpred-control-fix:r140`);
no service is recreated.

* Dry run (15:50, 12.5 min): 10,460,656 of 13,899,232 input links qualified.
* 15:04–15:30: input links 13.9 M → 3.41 M. Autovacuum then cleaned
  `api_artifactdependency` (15:29) and `api_artifact` (15:43), so the space is reused.
* The residue phase raised I/O pressure on production's rotating disk to ~15 %, above the
  5 % at which launches are refused, and nine requests waited at their final assembly; the
  run was stopped at 15:38 (about 130,000 residue rows removed, all nine requests
  completed by 15:47). Commit `8d3a5e6` makes the command wait while I/O pressure exceeds
  3 %; the residue phase was resumed at 15:49 with `--skip-links --batch 2000 --pause 1`
  and trickles at about 1,000 rows per minute (1.42 M transient rows to examine).

The database files do not shrink; deleted rows leave space that new requests reuse.
