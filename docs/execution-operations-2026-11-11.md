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
