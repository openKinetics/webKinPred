Input Data
==========

OpenKinetics Predictor accepts reaction data as CSV in the web interface.
The public API uses the same column names when data is submitted as JSON.

Column names are case-sensitive. Empty CSV rows are ignored.

Required columns
----------------

Every input row must contain a ``Protein Sequence`` value and either a
``Substrate`` value or a ``Substrates`` value.

- ``Protein Sequence``: full amino-acid sequence. Use uppercase one-letter
  amino-acid codes from ``ACDEFGHIKLMNPQRSTVWY``.
- ``Substrate``: one SMILES or InChI string.
- ``Substrates``: one or more SMILES or InChI strings separated by
  semicolons.
- ``Products``: one or more product SMILES or InChI strings separated by
  semicolons. This column is only valid with ``Substrates``.

Do not include both ``Substrate`` and ``Substrates`` in the same CSV. If a
``Products`` column is present, the frontend treats the file as a full-reaction
CSV.

CSV formats
-----------

Single-substrate CSV
~~~~~~~~~~~~~~~~~~~~

Use this format for one protein/substrate pair per row.

Required columns:

- ``Protein Sequence``
- ``Substrate``

Example:

.. code-block:: text

   Protein Sequence,Substrate
   MKTLLILAV...,CC(=O)O

This is the native format for CataPro, DLKcat, EITLEM-Kinetics, IECata,
KinForm-H, KinForm-L, MMISA-KM, OmniESI, OmniESI + O2DENet, RealKcat,
UniKP, and CatPred Km.

Multi-substrate CSV
~~~~~~~~~~~~~~~~~~~

Use this format when one reaction row contains an ordered list of substrates.

Required columns:

- ``Protein Sequence``
- ``Substrates``

Example:

.. code-block:: text

   Protein Sequence,Substrates
   MKTLLILAV...,CC(=O)O;O;C1CCCCC1

CatPred kcat consumes ``Substrates`` natively as one combined substrate set.
Single-substrate methods can also run on this format by expanding each
semicolon-separated substrate independently.

Full-reaction CSV
~~~~~~~~~~~~~~~~~

Use this format for methods that need both substrates and products.

Required columns:

- ``Protein Sequence``
- ``Substrates``
- ``Products``

Example:

.. code-block:: text

   Protein Sequence,Substrates,Products
   MKTLLILAV...,CC(=O)O;O,C(C(=O)O)O

TurNup kcat requires this format. Other compatible methods can accept a
full-reaction file, but only TurNup uses the ``Products`` values for
prediction. Supplied products are still validated.

Method compatibility
--------------------

The frontend builds the available method list from ``/api/v1/methods/``. The
table below reflects those method descriptors.

For methods whose minimum format is ``Protein Sequence`` + ``Substrate``, the
frontend also accepts ``Substrates`` and full-reaction files by adapting each
listed substrate to the method's single-substrate contract.

.. list-table::
   :header-rows: 1
   :widths: 24 22 34 20

   * - Method
     - Targets
     - Minimum or native input columns
     - Max sequence length
   * - CataPro
     - kcat, Km, kcat/Km
     - ``Protein Sequence``, ``Substrate``
     - 1,000
   * - CatPred
     - kcat, Km
     - kcat: ``Protein Sequence``, ``Substrates``; Km: ``Protein Sequence``,
       ``Substrate`` or ``Substrates``
     - 2,048
   * - DLKcat
     - kcat
     - ``Protein Sequence``, ``Substrate``
     - No method limit
   * - EITLEM-Kinetics
     - kcat, Km
     - ``Protein Sequence``, ``Substrate``
     - 1,024
   * - IECata
     - kcat/Km
     - ``Protein Sequence``, ``Substrate``
     - 1,000
   * - KinForm-H
     - kcat, Km
     - ``Protein Sequence``, ``Substrate``
     - 1,500
   * - KinForm-L
     - kcat
     - ``Protein Sequence``, ``Substrate``
     - 1,500
   * - MMISA-KM
     - Km
     - ``Protein Sequence``, ``Substrate``
     - 500
   * - OmniESI
     - kcat, Km
     - ``Protein Sequence``, ``Substrate``
     - 1,000
   * - OmniESI + O2DENet
     - kcat, Km
     - ``Protein Sequence``, ``Substrate``
     - 1,000
   * - RealKcat
     - kcat, Km
     - ``Protein Sequence``, ``Substrate``
     - 1,022
   * - TurNup
     - kcat
     - ``Protein Sequence``, ``Substrates``, ``Products``
     - 1,024
   * - UniKP
     - kcat, Km
     - ``Protein Sequence``, ``Substrate``
     - 1,000

Semicolon lists
---------------

``Substrates`` and ``Products`` use semicolons to separate molecules. Empty
fragments are ignored, so ``A;;B`` is read as ``A`` and ``B``.

For single-substrate methods running on a ``Substrates`` list:

- kcat output is reduced to the maximum successful per-substrate prediction.
- Km and kcat/Km outputs remain ordered JSON arrays, one value per substrate.
- Failed per-substrate positions are recorded as ``null`` in the array and in
  the result ``Extra Info``.

``Protein Sequence`` may also contain semicolon-separated candidate sequences.
The output remains one row per input row:

- kcat uses the maximum successful candidate-sequence prediction.
- Km uses the minimum successful candidate-sequence prediction.
- Direct kcat/Km uses the maximum successful candidate-sequence prediction.
- When both candidate sequences and ``Substrates`` are present, Km and
  kcat/Km keep ordered per-substrate arrays.

Validation
----------

The frontend checks CSV structure as soon as a file is selected. A valid file
is labelled as ``single-substrate``, ``multi-substrate``, or
``full-reaction`` and the compatible method selectors are enabled.

The optional validation step checks:

- missing ``Protein Sequence`` values;
- invalid amino-acid characters;
- invalid substrate or product SMILES/InChI strings;
- empty substrate or product lists;
- per-method protein sequence length limits;
- the overall server sequence limit of 10,000 residues;
- optional sequence similarity to method training data.

Invalid substrates or proteins are skipped during prediction. Invalid
``Products`` values block submission because full-reaction methods need valid
product chemistry.

Long sequences
--------------

When validation finds sequence-length warnings, the frontend lets you choose
how submission should handle long sequences:

- ``truncate``: default. The service keeps the first and last portions of the
  sequence. For example, with a 1,000-residue limit, a 1,200-residue sequence
  becomes the first 500 residues plus the last 500 residues.
- ``skip``: exclude rows or sequence candidates that exceed the applicable
  limit.

For multi-target jobs, the applied limit is the strictest selected method
limit, capped by the 10,000-residue server limit.

Submission options
------------------

The method-selection footer exposes input-related switches:

- ``Canonicalize substrates``: enabled by default. Equivalent SMILES are
  normalized before prediction. Disable it to use each method's native
  preprocessing.
- ``Prefer experimental data``: when a curated match from BRENDA, SABIO-RK, or
  UniProt is found, return that value instead of a model prediction.
- ``Include similarity columns``: enabled by default. Adds kcat training-set
  similarity columns to the output CSV.

Templates
---------

The frontend provides downloadable CSV templates:

- ``/templates/single_substrate_template.csv``
- ``/templates/multi_substrate_template.csv``
- ``/templates/full_reaction_template.csv``
