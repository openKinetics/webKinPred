"""Resource shapes shared by incremental planning and read-only coverage audits."""


def prediction_shape(frame, batch, input_bytes, method):
    lengths = [len(sequence) for sequence in batch.sequences]
    if method == "TurNup":
        # Both TurNup encoders preprocess to 1022. Raw CSV storage/parsing is
        # accounted for separately, including sequences before truncation.
        lengths = [min(length, 1022) for length in lengths]
    return {
        "rows": len(frame),
        "units": len(batch.sequences),
        "max_length": max(lengths, default=0),
        "max_molecule_length": max(
            (
                sum(len(str(item)) for item in value)
                if isinstance(value, (list, tuple))
                else len(str(value))
                for key in ("substrates", "products")
                for value in batch.call_kwargs.get(key, [])
            ),
            default=0,
        ),
        "input_bytes": input_bytes,
    }
