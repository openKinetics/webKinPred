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
