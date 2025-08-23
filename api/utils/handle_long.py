def get_valid_indices(sequences, limit, mode="skip"):
    """
    Returns indices of sequences that are valid (below both model and server limits).
    If mode is 'skip', sequences above limit are excluded from prediction.
    """
    if mode != "skip":
        return list(range(len(sequences)))  # in truncate mode, all allowed

    valid_indices = [i for i, seq in enumerate(sequences) if len(seq) <= limit]
    return valid_indices


def truncate_sequences(sequences, limit):
    """
    Truncates sequences that exceed the limit by keeping the first and last halves.
    For example, if limit=1000 and sequence length=1200, it returns seq[:500] + seq[-500:]

    Returns:
        - List of sequences (truncated if needed)
        - List of indices for sequences that were valid or truncated
    """
    truncated_sequences = []
    valid_indices = []

    for i, seq in enumerate(sequences):
        if len(seq) > limit:
            half = limit // 2
            truncated_seq = seq[:half] + seq[-half:]
            truncated_sequences.append(truncated_seq)
        else:
            truncated_sequences.append(seq)
        valid_indices.append(i)

    return truncated_sequences, valid_indices
