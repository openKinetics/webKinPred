MMseqs Similarity Datasets Guide
================================

Use this guide if your method needs sequence-similarity validation
(``runSimilarity=true``).

1. If Your Training Dataset Already Matches an Existing Dataset
---------------------------------------------------------------

Do not add a new FASTA/DB.

Just reuse the existing dataset and update its label in:

- ``webKinPred/similarity_dataset_registry.py``

Example:

- If your method uses the same training set as DLKcat, change
  ``"DLKcat/UniKP"`` to ``"DLKcat/UniKP/YourMethod"``.

This makes your method visible under that same similarity dataset option
without duplicating databases.

2. If You Need a New Dataset
----------------------------

1. Add a FASTA file of unique training-set sequences:

- ``fastas/your_dataset_sequences.fasta``

2. Register the dataset:

- ``webKinPred/similarity_dataset_registry.py``
- Add:

.. code:: python

   "Your Dataset Label": {
       "fasta_filename": "your_dataset_sequences.fasta",
       "db_name": "targetdb_your_dataset",
       "method_keys": ["YourMethod"],
   }

``method_keys`` is required for method to dataset mapping in kcat
similarity enrichment. Use one key for one method, or list multiple keys
when methods share one training set.

3. Build the MMseqs target DB:

.. code:: bash

   python tools/build_similarity_dbs.py --dataset "Your Dataset Label"

or build all:

.. code:: bash

   python tools/build_similarity_dbs.py --all

4. Confirm generated files exist:

- ``fastas/dbs/targetdb_your_dataset*``

After this, ``runSimilarity=true`` includes your dataset automatically
(backend + frontend).
