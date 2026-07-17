Input Data
==========

OpenKinetics Predictor accepts tabular input.

Core columns
------------

- Protein Sequence, the amino acid sequence.
- Substrate, a SMILES or InChI molecule string.

Extra columns
-------------

Some methods need extra fields.

- TurNup uses substrates and products.
- Mutant-aware workflows need sequence context.
- Batch jobs accept one row per prediction.

Validation
----------

The service validates input before prediction.

Validation checks include:

- Missing protein sequences.
- Invalid amino acid characters.
- Invalid molecule strings.
- Method-specific sequence length limits.
- Missing columns for the selected method.

Use validation before large jobs. You avoid failed runs and preserve quota.
