Contributing a New Prediction Method
====================================

| Start with the ``MethodDescriptor``.
| It is the contract between your method and the rest of the platform.

1. Define the Descriptor First
------------------------------

Create ``api/methods/your_method.py`` first, then implement the method
behind it.

.. code:: python

   from api.methods.base import MethodDescriptor, SubprocessEngineConfig
   from api.prediction_engines.your_method import your_method_predictions  # only for Path 2

   descriptor = MethodDescriptor(
       key="YourMethod",                  # unique ID used in API/UI
       display_name="Your Method",        # human-readable name
       authors="Author A, Author B",
       publication_title="Paper title",
       citation_url="https://doi.org/...",
       repo_url="https://github.com/...",

       supports=["kcat"],                 # e.g. ["kcat"], ["Km"], ["kcat/Km"], or combinations
       input_format="single",             # backend contract: "single" or "multi"
       # input_behavior_by_target={"kcat": "native_multi"},  # rare target override
       output_cols={"kcat": "kcat (1/s)"},
       max_seq_len=1024,

       col_to_kwarg={"Substrate": "substrates"},
       target_kwargs={"kcat": {}},

       # Engine selection rule:
       # - Use subprocess=SubprocessEngineConfig(...) by default.
       # - Use pred_func=your_method_predictions only when custom orchestration is required.

       embeddings_used=[],
   )

What these fields mean
~~~~~~~~~~~~~~~~~~~~~~

- ``supports``: which targets your method predicts.
- ``input_format``: backend CSV column contract expected by the method.
  Descriptors use ``single`` for the ``Substrate`` column contract and
  ``multi`` for the full-reaction ``Substrates`` + ``Products``
  contract. The orchestration layer automatically expands
  semicolon-separated ``Substrates`` values for every ``single``
  descriptor, so predictor integrations should continue declaring only
  their native one-substrate contract.
- ``input_behavior_by_target``: override the default only when a target
  consumes list input natively. CatPred declares ``native_multi`` for
  kcat and ``expanded_pair`` for Km; TurNup inherits
  ``native_full_reaction`` from its full-reaction descriptor. Keep
  target-specific behavior here instead of adding method-name branches
  to orchestration or UI code.
- ``col_to_kwarg``: maps CSV columns to kwargs passed into your method
  runtime.
- ``target_kwargs``: per-target switches (for shared kcat/Km scripts).
- ``subprocess`` or ``pred_func``: set exactly one. Use ``subprocess``
  by default. Use ``pred_func`` only when the shared subprocess engine
  cannot support your runtime flow.

2. Implement Your Method's Predictor
------------------------------------

Use this decision rule:

1. Use the shared subprocess engine by default.
2. Use a custom engine only when required by method-specific behaviour.

Source code of your method should be added to ``models/YourMethod/`` (this
can be a Git submodule).

General batching best practice:

- Batching is fine, but keep batch sizes realistic to avoid RAM spikes
  (generally no more than 32-64 rows/sequences per batch).

Path 1: Script + Shared Engine (default)
----------------------------------------

Use this if your model can run as one subprocess call.

You write:

- One prediction script.
- ``subprocess=SubprocessEngineConfig(...)`` in the descriptor.

The shared engine handles:

- Row validation (sequence and substrate/product chemistry).
- Temporary input/output files.
- Subprocess execution.
- Progress parsing (``Progress: x/y``).
- Output parsing and row mapping.

Your script must support:

.. code:: bash

   python your_script.py --input <input.json> --output <output.json>

Input JSON:

.. code:: json

   {
     "method": "YourMethod",
     "target": "kcat",
     "public_id": "abc1234",
     "rows": [
       {"sequence": "MKT...", "substrates": "CC(=O)O"}
     ],
     "params": {
       "kinetics_type": "KCAT"
     }
   }

Output JSON:

.. code:: json

   {
     "predictions": [12.3],
     "invalid_indices": []
   }

Rules:

- ``predictions`` length must equal ``rows`` length.
- ``invalid_indices`` is optional and is relative to ``rows``.
- Use ``null`` for missing predictions.
- If your script uses PyTorch, handle both GPU and CPU runtimes: use
  CUDA only when ``torch.cuda.is_available()`` is ``True``, and keep a
  CPU fallback.
- Emit prediction progress as ``Progress: x/y`` on stdout if the script
  can report it. The platform parses those lines for frontend progress
  and separately writes structured infrastructure logs.
- Do not add bare ``print()`` calls in ``api/`` runtime code. Use
  Python logging with a stable ``event`` key in ``extra``, and keep
  user-facing validation/session text on ``push_line()``.

Path config example:

.. code:: python

   subprocess=SubprocessEngineConfig(
       python_path_key="YourMethod",
       script_key="YourMethod",
       data_path_env={"YOUR_METHOD_DATA": "YourMethod"},
   )

Path 2: Script + Custom Engine (only when required)
---------------------------------------------------

Use this if you need custom behavior not covered by the shared engine.

Examples:

- Special validation rules.
- Non-standard file contracts.
- Multi-stage orchestration.
- Extra Python-side preprocessing/caching.

You write:

- ``api/prediction_engines/your_method.py``.
- ``pred_func=your_method_predictions`` in the descriptor.

Expected engine signature:

.. code:: python

   def your_method_predictions(
       sequences: list[str],
       public_id: str,
       **kwargs,
   ) -> tuple[list, list[int] | dict[int, str]]:
       ...

Return:

- ``predictions``: one value per input row.
- ``invalid_indices``: one of:

  - ``list[int]`` of failed row indices relative to input list.
  - ``dict[int, str]`` mapping failed row indices to clear reasons.

Recommendation:

- Return ``dict[int, str]`` for richer user feedback in job output and
  progress views.

3. Register Runtime Paths
-------------------------

If your method needs a new Python environment, you must update the full
worker image ``Dockerfile.envs``.

1. Add a requirements file:

.. code:: text

   docker-requirements/your_method_requirements.txt

2. Add a parallel env stage in ``Dockerfile.envs``.

The Dockerfile uses multi-stage builds so all envs are built in parallel
by BuildKit. Add two things:

**a) A new ``FROM base AS env-your_method`` stage** (alongside the other
``env-*`` stages):

.. code:: dockerfile

   # -- YourMethod ---------------------------------------------------------------
   FROM base AS env-your_method
   COPY docker-requirements/your_method_requirements.txt ./docker-requirements/
   RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
       --mount=type=cache,id=webkinpred-pip-py310,target=/root/.cache/pip,sharing=locked \
       mamba create -n your_method_env python=3.10 -c conda-forge -y \
       && conda run -n your_method_env pip install -r docker-requirements/your_method_requirements.txt

If your method needs extra conda packages (for example RDKit or XGBoost), install
them before ``pip install`` (see ``env-dlkcat`` and ``env-turnup``
stages for examples).

**b) A ``COPY --from`` line in the ``final`` stage** (alongside the
other env copies):

.. code:: dockerfile

   COPY --from=env-your_method /opt/conda/envs/your_method_env /opt/conda/envs/your_method_env

3. Add runtime keys in:

- ``webKinPred/config_docker.py``
- ``webKinPred/config_local.py`` (for local development) Both inherit
  common path shape from ``webKinPred/config_base.py``.

.. code:: python

   PYTHON_PATHS["YourMethod"] = "/opt/conda/envs/your_method_env/bin/python"
   PREDICTION_SCRIPTS["YourMethod"] = "/app/models/YourMethod/predict.py"
   DATA_PATHS["YourMethod"] = "/app/models/YourMethod/data"

If your method can reuse an existing env, skip steps 1-2 and only add
the config keys.

4. PLM Embeddings (Optional)
----------------------------

The embeddings cache stores reusable PLM outputs under
``media/sequence_info``, keyed by ``seq_id``. We use this to avoid
repeated PLM inference for the same sequence across jobs and methods.

GPU offload runs missing embedding work on a remote GPU before
prediction starts. We use this to reduce CPU load and improve
throughput. If the remote GPU path fails or is unavailable, prediction
falls back to local compute.

Read the full guide:

- :doc:`plm_embedding_cache`

5. Add MMseqs Similarity Dataset (Optional)
-------------------------------------------

If you want to include your method's training data in the
sequence-similarity validation, read:

- :doc:`mmseqs_similarity_datasets`

This includes:

- reusing an existing dataset by extending its label (for example
  ``DLKcat/UniKP/YourMethod``)
- adding a new FASTA + DB dataset
- setting ``method_keys`` in each dataset entry so backend method
  mapping works

6. Test Your Integration End-to-End
-----------------------------------

Setup:

.. code:: bash

   pip install -r requirements.txt
   python manage.py migrate

Run:

.. code:: bash

   python tools/test_method_integration.py --method YourMethod

What it tests:

- method registry discovery
- descriptor validity (runnable config checks)
- direct prediction execution through backend task helpers
- output CSV generation and output-shape checks
- all targets your method supports (``kcat``, ``Km``, and/or
  ``kcat/Km``)
- optional DLKcat sanity check first

If you use Path 1 (``subprocess=SubprocessEngineConfig(...)``), do this
before testing:

- create/install your method environment
- set ``PYTHON_PATHS["YourMethod"]`` in ``webKinPred/config_local.py``
  to that environment's Python executable
