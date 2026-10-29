"""Bound model inputs while retaining original reduction and molecular positions."""


def invoke_bounded(invoke, context, descriptor, sequences, aligned_kwargs, options):
    predictions, invalid, details = [], {}, []
    has_details = False
    original_context, selector, base = (
        context.unit_context,
        context.select_units,
        context.progress_base,
    )
    try:
        for start in range(0, len(sequences), context.unit_batch_size):
            end = min(len(sequences), start + context.unit_batch_size)
            context.unit_context = {
                str(i - start): value
                for i in range(start, end)
                if (value := original_context.get(str(i))) is not None
            }
            context.progress_base = base + start
            if selector:
                context.select_units = lambda indices, offset=start: selector(
                    [offset + i for i in indices]
                )
                context.select_units(range(end - start))
            kwargs = {
                key: value[start:end]
                if key in descriptor.col_to_kwarg.values()
                and isinstance(value, (list, tuple))
                and len(value) == len(sequences)
                else value
                for key, value in aligned_kwargs.items()
            }
            values, errors, extra = invoke(
                desc=descriptor, sequences=sequences[start:end], **options, **kwargs
            )
            predictions.extend(values)
            invalid.update({start + index: reason for index, reason in errors.items()})
            has_details = has_details or extra is not None
            details.extend(extra if extra is not None else [""] * (end - start))
            context.emit("prediction_units", {"done": end - start})
    finally:
        context.unit_context, context.select_units, context.progress_base = (
            original_context,
            selector,
            base,
        )
    return predictions, invalid, details if has_details else None
