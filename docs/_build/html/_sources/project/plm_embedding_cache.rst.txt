PLM Embedding Cache and Remote GPU Offload
==========================================

Use this guide when your method depends on protein language model
embeddings.

1. Purpose
----------

The PLM step is expensive. You should run it once per sequence, store
the result, and reuse it across jobs.

The system already supports this with:

- file-based cache artefacts under ``media/sequence_info``
- stable sequence IDs from ``tools/seqmap/main.py``
- remote GPU precompute with local fail-open fallback
- file-level embedding progress tracking through inotify

2. Core concepts
----------------

2.1 ``seq_id`` is the cache key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The cache key is ``seq_id``, not raw sequence text.
- ``seq_id`` is shared across methods.
- One cached artefact can serve many jobs and many methods.

2.2 Planner is sparse
~~~~~~~~~~~~~~~~~~~~~

``api/services/embedding_plan_service.py`` computes:

- expected cache paths per ``seq_id``
- missing paths per ``seq_id``
- step-level missing work
- watch directories for progress tracking

Only missing work is sent to the GPU service.

2.3 Prediction and embedding are separate stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Embedding generation is precompute.
- Prediction inference consumes cached artefacts.
- If GPU precompute fails, local prediction continues and computes
  missing artefacts.

3. Current method-by-method behaviour
-------------------------------------

This section reflects the current code paths.

3.1 KinForm-H and KinForm-L
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Engine and planner:

- Engine: ``api/prediction_engines/kinform.py``
- Planner profile: ``kinform_full``
- GPU precompute call: yes

Cache artefacts:

- ``media/sequence_info/esm2_layer_26/{mean_vecs,weighted_vecs}/{seq_id}.npy``
- ``media/sequence_info/esm2_layer_29/{mean_vecs,weighted_vecs}/{seq_id}.npy``
- ``media/sequence_info/esmc_layer_24/{mean_vecs,weighted_vecs}/{seq_id}.npy``
- ``media/sequence_info/esmc_layer_32/{mean_vecs,weighted_vecs}/{seq_id}.npy``
- ``media/sequence_info/prot_t5_layer_19/{mean_vecs,weighted_vecs}/{seq_id}.npy``
- ``media/sequence_info/prot_t5_last/{mean_vecs,weighted_vecs}/{seq_id}.npy``
- Binding site table: ``media/pseq2sites/binding_sites_all.tsv``

GPU steps:

- ``kinform_t5_full``
- ``kinform_esm2_layers``
- ``kinform_esmc_layers``

KinForm rule:

- ``kinform_t5_full`` covers both binding sites and ProtT5 outputs.
- A sequence enters this step if binding sites are missing, ProtT5 files
  are missing, or both.

3.2 CataPro
~~~~~~~~~~~

Engine and planner:

- Engine: generic subprocess
  ``api/prediction_engines/generic_subprocess.py``
- Planner profile: ``prot_t5_mean``
- GPU precompute call: yes, from generic engine

Cache artefact:

- ``media/sequence_info/prot_t5_last/mean_vecs/{seq_id}.npy``

3.3 UniKP
~~~~~~~~~

Engine and planner:

- Engine: ``api/prediction_engines/unikp.py``
- Planner profile: ``prot_t5_mean``
- GPU precompute call: yes

Cache artefact:

- ``media/sequence_info/prot_t5_last/mean_vecs/{seq_id}.npy``

3.4 TurNup
~~~~~~~~~~

Engine and planner:

- Engine: ``api/prediction_engines/turnup.py``
- Planner profile: ``turnup_esm1b``
- GPU precompute call: yes

Cache artefact:

- ``media/sequence_info/esm1b_turnup/{seq_id}.npy``

3.5 CatPred
~~~~~~~~~~~

Engine and planner:

- Engine: generic subprocess
  ``api/prediction_engines/generic_subprocess.py``
- Planner profile: ``catpred_embed``
- GPU precompute call: yes, from generic engine

Cache artefact:

- ``media/sequence_info/catpred_esm2/{parameter}/{model_key}/{seq_id}.pt``
- ``parameter`` is ``kcat`` or ``km``
- ``model_key`` comes from checkpoint path structure

GPU steps:

- ``catpred_embed_kcat``
- ``catpred_embed_km``

CatPred uses a deterministic reduction path. The cached ``.pt`` file is
the checkpoint specific pooled tensor.

3.6 EITLEM
~~~~~~~~~~

Planner and GPU runner:

- Planner profile: ``eitlem_esm1v``
- GPU step key: ``eitlem_esm1v``
- Cache path: ``media/sequence_info/esm1v/{seq_id}.npy``

Current configured prediction script:

- ``webKinPred/config_base.py`` points to
  ``models/EITLEM/Code/eitlem_prediction_script_batch.py``

Also present in the repo:

- ``models/EITLEM/Code/eitlem_prediction_script.py``
- This script shows the preferred full-matrix ephemeral cleanup pattern.
- It removes touched ``esm1v`` files after prediction.

3.7 OmniESI
~~~~~~~~~~~

Engine and planner:

- Engine: generic subprocess
  ``api/prediction_engines/generic_subprocess.py``
- Planner profile: ``omniesi_esm2``
- GPU precompute call: yes, from generic engine

Staged artefact:

- ``media/sequence_info/omniesi_esm2/{seq_id}.pt``

GPU step:

- ``omniesi_esm2``

OmniESI stages the full per-residue ``esm2_t33_650M_UR50D`` layer-33
tensor, shape ``[seq_len, 1280]``, as a CPU ``.pt`` file. Prediction
deletes these files after the run, matching EITLEM's ephemeral ``esm1v``
behaviour. This is intentionally separate from CatPred's
checkpoint-specific pooled ESM2 cache.

3.8 RealKcat
~~~~~~~~~~~~

Engine and planner:

- Engine: generic subprocess
  ``api/prediction_engines/generic_subprocess.py``
- Planner profile: ``realkcat_esm2_last_mean``
- GPU precompute call: yes, from generic engine

Cache artefact:

- ``media/sequence_info/esm2_layer_last_mean/{seq_id}.npy``

GPU step:

- ``realkcat_esm2_last_mean``

RealKcat uses a persistent ESM2 layer-33 mean-vector cache keyed by
``seq_id``. Files are reused across runs and are not deleted after
prediction.

3.9 IECata
~~~~~~~~~~

Engine and planner:

- Engine: generic subprocess
  ``api/prediction_engines/generic_subprocess.py``
- Planner profile: ``iecata_prot_t5_residues``
- GPU precompute call: yes, from generic engine

Staged artefact:

- ``media/sequence_info/iecata_prot_t5_residues/{seq_id}.npy``

GPU step:

- ``iecata_prot_t5_residues``

IECata stages full ProtT5 per-residue tensors as ephemeral files.
Prediction consumes these files and removes touched artefacts after the
run.

3.10 MMISA-KM
~~~~~~~~~~~~~

MMISA-KM does not use a PLM embedding cache and is not GPU-offload
eligible in the shared embedding system.

It uses a method-local contact-map cache:

- ``MMISA_KM_CACHE_DIR`` -> ``media/sequence_info/mmisa_km_contacts/``

The generic subprocess engine skips shared embedding planning and
progress tracking for MMISA-KM because its descriptor has
``embeddings_used=[]``.

4. Planner contract
-------------------

``build_embedding_plan(...)`` produces an ``EmbeddingPlan`` with:

- ``profile``
- ``seq_ids``
- ``seq_id_to_seq``
- ``expected_paths_by_seq``
- ``missing_paths_by_seq``
- ``watch_dirs``
- ``step_plans``
- file-level totals for ``total``, ``cached_already``,
  ``need_computation``

Important details:

- Counts are file based, not sequence based.
- One sequence may be complete for one step and missing in another.
- Step work must stay sparse by step and by sequence.

Key planner functions to update for new work:

- ``_profile_for_method``
- ``expected_paths_by_seq``
- ``_step_plans_for_profile``

5. Backend orchestration contract
---------------------------------

``api/services/gpu_embed_service.py`` runs the precompute flow.

Main entry point:

- ``run_gpu_precompute_if_available(...)``

Flow:

1.  Build embedding plan.
2.  Return if cache is complete.
3.  Return if method is unsupported or service URL is not set.
4.  Read GPU health with TTL cache.
5.  Build sparse ``step_work``.
6.  Start embedding tracker before remote submission.
7.  Submit ``POST /embed/jobs``.
8.  Poll ``GET /embed/jobs/{job_id}``.
9.  Rebuild plan after remote ``done`` and verify no missing files
    remain.
10. Continue to prediction subprocess.

Fallback behaviour:

- Default is fail-open.
- Local prediction continues on failure or unreachable GPU.
- Set ``GPU_EMBED_FAIL_CLOSED=1`` only if fail-fast is required.

6. GPU service API contract
---------------------------

Service file:

- ``tools/gpu_embed_service/app.py``

Endpoints:

- ``GET /health``
- ``POST /embed/jobs``
- ``GET /embed/jobs/{job_id}``

Submit payload:

.. code:: json

   {
     "method_key": "KinForm-H",
     "target": "kcat",
     "profile": "kinform_full",
     "step_work": {
       "kinform_t5_full": ["sid_1"],
       "kinform_esm2_layers": ["sid_2"]
     },
     "seq_id_to_seq": {
       "sid_1": "MPE...",
       "sid_2": "MQA..."
     }
   }

Optional auth:

- bearer token via ``GPU_EMBED_SERVICE_TOKEN``

Health payload includes:

- ``online``
- ``gpu_name``
- ``free_vram_gb``
- ``total_vram_gb``
- ``active_jobs``
- ``queued_jobs``

7. Step runner contract
-----------------------

Step runner file:

- ``tools/gpu_embed_service/run_step.py``

Current active step keys:

- ``kinform_t5_full``
- ``kinform_esm2_layers``
- ``kinform_esmc_layers``
- ``prot_t5_mean``
- ``turnup_esm1b``
- ``eitlem_esm1v``
- ``catpred_embed_kcat``
- ``catpred_embed_km``
- ``omniesi_esm2``
- ``realkcat_esm2_last_mean``
- ``iecata_prot_t5_residues``

Deprecated keys kept for compatibility:

- ``kinform_pseq2sites``
- ``kinform_prott5_layers``

Do not add new features on deprecated keys.

8. Progress tracking
--------------------

Tracker file:

- ``api/services/embedding_progress_service.py``

Rules:

- Progress is driven by expected missing file paths.
- Progress increments when files appear.
- Inotify is used on Linux.
- Reconciliation polling handles missed events.
- Existing tracker is reused for the same job stage.

UI impact:

- UI receives embedding progress without direct GPU progress streaming.
- Remote writes through shared storage still update UI progress.

9. Contributor playbook
-----------------------

Use this decision rule first:

1. If your method uses a PLM and artefact type already supported, you
   must reuse that existing embedding family.
2. If your method introduces a new PLM or artefact type, you must add a
   new embedding family.

Existing PLM caches:

- ``prot_t5_last/mean_vecs/{seq_id}.npy``: mean embedding of the last
  layer from ``Rostlab/prot_t5_xl_uniref50`` (shared by CataPro and
  UniKP, also used by KinForm).
- ``prot_t5_last/weighted_vecs/{seq_id}.npy``: weighted average of
  last-layer ``Rostlab/prot_t5_xl_uniref50`` residue embeddings, with
  weights from Pseq2Sites binding-site probabilities in
  ``media/pseq2sites/binding_sites_all.tsv`` (KinForm).
- ``prot_t5_layer_19/mean_vecs/{seq_id}.npy``: mean embedding of layer
  19 from ``Rostlab/prot_t5_xl_uniref50`` (KinForm).
- ``prot_t5_layer_19/weighted_vecs/{seq_id}.npy``: weighted average of
  layer 19 ``Rostlab/prot_t5_xl_uniref50`` residue embeddings, with
  Pseq2Sites binding-site probabilities (KinForm).
- ``esm2_layer_26/mean_vecs/{seq_id}.npy``: mean embedding from
  ``esm2_t33_650M_UR50D`` layer 26 (KinForm).
- ``esm2_layer_26/weighted_vecs/{seq_id}.npy``: weighted average of
  ``esm2_t33_650M_UR50D`` layer 26 residue embeddings, with Pseq2Sites
  binding-site probabilities (KinForm).
- ``esm2_layer_29/mean_vecs/{seq_id}.npy``: mean embedding from
  ``esm2_t33_650M_UR50D`` layer 29 (KinForm).
- ``esm2_layer_29/weighted_vecs/{seq_id}.npy``: weighted average of
  ``esm2_t33_650M_UR50D`` layer 29 residue embeddings, with Pseq2Sites
  binding-site probabilities (KinForm).
- ``esmc_layer_24/mean_vecs/{seq_id}.npy``: mean embedding from
  ``esmc_600m`` layer 24 (KinForm).
- ``esmc_layer_24/weighted_vecs/{seq_id}.npy``: weighted average of
  ``esmc_600m`` layer 24 residue embeddings, with Pseq2Sites
  binding-site probabilities (KinForm).
- ``esmc_layer_32/mean_vecs/{seq_id}.npy``: mean embedding from
  ``esmc_600m`` layer 32 (KinForm).
- ``esmc_layer_32/weighted_vecs/{seq_id}.npy``: weighted average of
  ``esmc_600m`` layer 32 residue embeddings, with Pseq2Sites
  binding-site probabilities (KinForm).
- ``esm1b_turnup/{seq_id}.npy``: TurNup protein vector derived from
  ``esm1b_t33_650M_UR50S`` plus the TurNup fine-tuned checkpoint
  (``model_ESM_binary_A100_epoch_1_new_split.pkl``).
- ``catpred_esm2/{kcat|km}/{model_key}/{seq_id}.pt``: CatPred
  checkpoint-specific pooled tensor, generated from
  ``esm2_t33_650M_UR50D`` residue features and attentive pooling.
- ``omniesi_esm2/{seq_id}.pt``: OmniESI ephemeral full per-residue
  ``esm2_t33_650M_UR50D`` layer-33 tensor.
- ``esm2_layer_last_mean/{seq_id}.npy``: RealKcat persistent ESM2
  layer-33 mean vector (``esm2_t33_650M_UR50D``).
- ``iecata_prot_t5_residues/{seq_id}.npy``: IECata ephemeral full ProtT5
  per-residue embedding tensor.

9.1 Reuse an existing embedding family
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Reuse an existing cache artefact shape in ``media/sequence_info``.
2. Reuse the related planner profile in ``_profile_for_method``.
3. Update expected path logic only when path shape differs.
4. Ensure method script reads cache first and computes missing only.
5. Ensure precompute hook runs before prediction subprocess.

   - Generic subprocess methods already do this.
   - Custom engines must call ``run_gpu_precompute_if_available(...)``.

6. Add tests for cache hit, partial miss, and full miss.

9.2 Add a new embedding family
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Define deterministic artefact paths keyed by ``seq_id``.
2. Add ``expected_paths_by_seq`` logic.
3. Add method to profile mapping.
4. Add sparse step partitioning logic.
5. Implement step execution in ``run_step.py``.
6. Add optional command override in ``gpu_service.env``.
7. Pass required env paths into prediction subprocess env.
8. Add planner, orchestration, and API tests.

10. Full matrix policy
----------------------

Default rule:

- Do not cache full residue matrices.
- Cache reduced artefacts when reduction is deterministic and substrate
  independent.

CatPred pattern:

- ESM2 residue level signals are reduced by checkpoint attentive
  pooling.
- The reduced tensor is deterministic for ``(seq_id, checkpoint_key)``.
- Persist and reuse
  ``catpred_esm2/{parameter}/{model_key}/{seq_id}.pt``.

Full matrix pattern:

- Use full matrix files only when your model consumes them directly.
- Treat those files as ephemeral job files.
- Create files, run prediction, then delete touched files.
- EITLEM is the reference pattern in
  ``models/EITLEM/Code/eitlem_prediction_script.py``.

11. Tests to add
----------------

Planner tests:

- mixed cache state with step-level partial misses
- KinForm mixed state where only one step is missing per sequence
- CatPred target-specific step mapping

Orchestration tests:

- tracker starts before remote submission
- request payload includes only sparse missing ``step_work``
- failure fallback
- post-check catches remote done with missing artefacts

API tests:

- ``/api/v1/gpu/status/`` for configured and unconfigured cases
- status payload includes ``gpuPrecompute`` events

12. Operations and configuration
--------------------------------

Backend environment keys:

- ``GPU_EMBED_SERVICE_URL``
- ``GPU_EMBED_SERVICE_TOKEN``
- ``GPU_EMBED_HEALTH_TTL_SECONDS``
- ``GPU_EMBED_JOB_TIMEOUT_SECONDS``
- ``GPU_EMBED_FAIL_CLOSED``

GPU host env setup:

- ``tools/gpu_embed_service/gpu_service.env``

If ``GPU_EMBED_SERVICE_URL`` is empty, GPU offload is skipped and local
behaviour remains unchanged.
