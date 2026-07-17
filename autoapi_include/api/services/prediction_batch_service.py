"""Pure positional planning for prediction engine batches.

The planner is deliberately free of database, cache, model, and progress side
effects.  Both the request-time ReconXKG preflight and the worker use these
objects, so a cache hit is proved against the exact units that execution will
consume.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from api.methods.base import PredictionError
from api.utils.substrate_expansion import SubstrateExpansionPlan
from api.utils.sequence_expansion import (
    SequenceExpansionPlan,
    TargetExpansionPlan,
    TargetPredictionUnit,
)

try:
    from webKinPred.config_docker import SERVER_LIMIT
except ImportError:
    from webKinPred.config_local import SERVER_LIMIT


@dataclass(frozen=True)
class SequenceBatchPlan:
    """Position-safe sequence handling shared by all selected targets."""

    original_sequences: tuple[Any, ...]
    processed_by_reaction: tuple[Any | None, ...]
    valid_reaction_indices: tuple[int, ...]
    skipped_reactions: dict[int, str]
    expansion: SequenceExpansionPlan


@dataclass(frozen=True)
class TargetBatchPlan:
    """One method/target engine call after input-schema adaptation."""

    target: str
    input_behavior: str
    sequences: tuple[Any, ...]
    call_kwargs: dict[str, Any]
    expansion: SubstrateExpansionPlan | None = None
    unit_expansion: TargetExpansionPlan | None = None


def build_sequence_batch_plan(
    dataframe: pd.DataFrame,
    descriptors: Iterable[Any],
    handle_long_sequences: str,
) -> SequenceBatchPlan:
    sequences = dataframe["Protein Sequence"].tolist()
    limits = [min(SERVER_LIMIT, desc.max_seq_len) for desc in descriptors]
    limit = min(limits) if limits else SERVER_LIMIT

    expansion = SequenceExpansionPlan.build(
        sequences,
        range(len(sequences)),
        limit=limit,
        handle_long_sequences=handle_long_sequences,
    )
    processed_by_reaction = list(expansion.valid_processed_by_reaction())
    valid_indices = list(expansion.valid_reaction_indices_for_legacy())
    skipped = dict(expansion.skipped_reactions)
    return SequenceBatchPlan(
        original_sequences=tuple(sequences),
        processed_by_reaction=tuple(processed_by_reaction),
        valid_reaction_indices=tuple(valid_indices),
        skipped_reactions=skipped,
        expansion=expansion,
    )


def build_target_batch_plan(
    descriptor: Any,
    target: str,
    dataframe: pd.DataFrame,
    sequence_plan: SequenceBatchPlan,
) -> TargetBatchPlan:
    """Build the exact positional engine batch for one selected target."""
    behavior = descriptor.input_behavior(target)
    valid_indices = sequence_plan.valid_reaction_indices

    if sequence_plan.expansion.requires_reduction:
        return _build_sequence_reduction_target_batch_plan(
            descriptor,
            target,
            behavior,
            dataframe,
            sequence_plan,
        )

    if (
        behavior == "expanded_pair"
        and "Substrate" in descriptor.col_to_kwarg
        and "Substrates" in dataframe.columns
    ):
        expansion = SubstrateExpansionPlan.build(
            dataframe["Substrates"].tolist(),
            valid_indices,
        )
        kwargs = _expanded_call_kwargs(descriptor, dataframe, expansion)
        kwargs.update(descriptor.target_kwargs.get(target, {}))
        return TargetBatchPlan(
            target=target,
            input_behavior=behavior,
            sequences=tuple(expansion.expanded_sequences(sequence_plan.processed_by_reaction)),
            call_kwargs=kwargs,
            expansion=expansion,
        )

    expansion = None
    if behavior == "native_multi" and "Substrates" in dataframe.columns:
        expansion = SubstrateExpansionPlan.build(
            dataframe["Substrates"].tolist(),
            valid_indices,
        )
        kwargs = _native_multi_call_kwargs(descriptor, dataframe, expansion)
    else:
        kwargs = _native_call_kwargs(descriptor, dataframe, valid_indices)

    kwargs.update(descriptor.target_kwargs.get(target, {}))
    return TargetBatchPlan(
        target=target,
        input_behavior=behavior,
        sequences=tuple(
            sequence_plan.processed_by_reaction[index] for index in valid_indices
        ),
        call_kwargs=kwargs,
        expansion=expansion,
    )


def _build_sequence_reduction_target_batch_plan(
    descriptor: Any,
    target: str,
    behavior: str,
    dataframe: pd.DataFrame,
    sequence_plan: SequenceBatchPlan,
) -> TargetBatchPlan:
    sequence_expansion = sequence_plan.expansion
    units: list[TargetPredictionUnit] = []
    slices: list[tuple[int, int, int]] = []
    substrate_tokens_by_reaction: list[tuple[str, ...]] = [tuple()] * len(dataframe)

    from api.utils.substrate_expansion import split_substrate_list

    uses_substrate_slots = (
        behavior == "expanded_pair"
        and "Substrate" in descriptor.col_to_kwarg
        and "Substrates" in dataframe.columns
    )

    for reaction_position, seq_start, seq_end in sequence_expansion.reaction_slices:
        start = len(units)
        substrate_tokens = (
            tuple(split_substrate_list(dataframe["Substrates"].iloc[reaction_position]))
            if "Substrates" in dataframe.columns
            else tuple()
        )
        substrate_tokens_by_reaction[reaction_position] = substrate_tokens
        for sequence_child_index in range(seq_start, seq_end):
            sequence_child = sequence_expansion.children[sequence_child_index]
            if sequence_child.processed_sequence is None:
                continue
            if uses_substrate_slots:
                for substrate_position, substrate in enumerate(substrate_tokens):
                    units.append(
                        TargetPredictionUnit(
                            reaction_position=reaction_position,
                            sequence_child_index=sequence_child_index,
                            sequence_position=sequence_child.sequence_position,
                            sequence=sequence_child.sequence,
                            substrate_position=substrate_position,
                            substrate=substrate,
                        )
                    )
            else:
                units.append(
                    TargetPredictionUnit(
                        reaction_position=reaction_position,
                        sequence_child_index=sequence_child_index,
                        sequence_position=sequence_child.sequence_position,
                        sequence=sequence_child.sequence,
                    )
                )
        slices.append((reaction_position, start, len(units)))

    unit_plan = TargetExpansionPlan(
        sequence_plan=sequence_expansion,
        units=tuple(units),
        reaction_slices=tuple(slices),
        substrate_tokens_by_reaction=tuple(substrate_tokens_by_reaction),
        uses_substrate_slots=uses_substrate_slots,
    )
    kwargs = _sequence_reduction_call_kwargs(
        descriptor,
        dataframe,
        behavior,
        unit_plan,
    )
    kwargs.update(descriptor.target_kwargs.get(target, {}))
    return TargetBatchPlan(
        target=target,
        input_behavior=behavior,
        sequences=tuple(
            sequence_expansion.children[unit.sequence_child_index].processed_sequence
            for unit in unit_plan.units
        ),
        call_kwargs=kwargs,
        unit_expansion=unit_plan,
    )


def _sequence_reduction_call_kwargs(
    descriptor: Any,
    dataframe: pd.DataFrame,
    behavior: str,
    unit_plan: TargetExpansionPlan,
) -> dict[str, list[Any]]:
    kwargs: dict[str, list[Any]] = {}
    for column, kwarg_name in descriptor.col_to_kwarg.items():
        if column == "Substrate" and unit_plan.uses_substrate_slots:
            kwargs[kwarg_name] = [unit.substrate for unit in unit_plan.units]
            continue
        if (
            column == "Substrate"
            and behavior == "native_multi"
            and "Substrates" in dataframe.columns
        ):
            kwargs[kwarg_name] = [
                list(unit_plan.substrate_tokens_by_reaction[unit.reaction_position])
                for unit in unit_plan.units
            ]
            continue
        source_column = (
            "Substrates"
            if column == "Substrate" and "Substrates" in dataframe.columns
            else column
        )
        _require_column(descriptor, dataframe, source_column)
        kwargs[kwarg_name] = [
            dataframe[source_column].iloc[unit.reaction_position]
            for unit in unit_plan.units
        ]
    return kwargs


def _expanded_call_kwargs(
    descriptor: Any,
    dataframe: pd.DataFrame,
    expansion: SubstrateExpansionPlan,
) -> dict[str, list[Any]]:
    kwargs: dict[str, list[Any]] = {}
    for column, kwarg_name in descriptor.col_to_kwarg.items():
        if column == "Substrate":
            kwargs[kwarg_name] = expansion.expanded_substrates()
        else:
            _require_column(descriptor, dataframe, column)
            kwargs[kwarg_name] = expansion.expanded_parent_values(
                dataframe[column].tolist()
            )
    return kwargs


def _native_multi_call_kwargs(
    descriptor: Any,
    dataframe: pd.DataFrame,
    expansion: SubstrateExpansionPlan,
) -> dict[str, list[Any]]:
    kwargs: dict[str, list[Any]] = {}
    for column, kwarg_name in descriptor.col_to_kwarg.items():
        if column == "Substrate":
            kwargs[kwarg_name] = [
                [expansion.children[index].substrate for index in range(start, end)]
                for _reaction_position, start, end in expansion.reaction_slices
            ]
        else:
            _require_column(descriptor, dataframe, column)
            kwargs[kwarg_name] = [
                dataframe[column].iloc[reaction_position]
                for reaction_position in expansion.reaction_positions
            ]
    return kwargs


def _native_call_kwargs(
    descriptor: Any,
    dataframe: pd.DataFrame,
    reaction_indices: tuple[int, ...],
) -> dict[str, list[Any]]:
    kwargs: dict[str, list[Any]] = {}
    for column, kwarg_name in descriptor.col_to_kwarg.items():
        _require_column(descriptor, dataframe, column)
        kwargs[kwarg_name] = [
            dataframe[column].iloc[reaction_index]
            for reaction_index in reaction_indices
        ]
    return kwargs


def _require_column(descriptor: Any, dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise PredictionError(
            f"Missing column required for {descriptor.display_name}: {column}"
        )
