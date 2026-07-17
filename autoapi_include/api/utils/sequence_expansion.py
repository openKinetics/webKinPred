"""Position-safe expansion and reduction for semicolon-separated protein sequences."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ExpandedSequence:
    """One protein-sequence candidate from an input row."""

    reaction_position: int
    sequence_position: int
    sequence: str
    processed_sequence: str | None
    skip_reason: str = ""


@dataclass(frozen=True)
class SequenceExpansionPlan:
    """A positional mapping from input rows to split protein sequences."""

    original_sequences: tuple[Any, ...]
    reaction_positions: tuple[int, ...]
    children: tuple[ExpandedSequence, ...]
    reaction_slices: tuple[tuple[int, int, int], ...]

    @classmethod
    def build(
        cls,
        sequence_values: Sequence[Any],
        reaction_positions: Iterable[int],
        *,
        limit: int,
        handle_long_sequences: str,
    ) -> "SequenceExpansionPlan":
        children: list[ExpandedSequence] = []
        slices: list[tuple[int, int, int]] = []
        positions = tuple(int(position) for position in reaction_positions)

        for reaction_position in positions:
            if reaction_position < 0 or reaction_position >= len(sequence_values):
                raise ValueError(
                    f"Reaction position {reaction_position} is outside the input data."
                )
            start = len(children)
            for sequence_position, sequence in enumerate(
                split_sequence_list(sequence_values[reaction_position])
            ):
                processed = sequence
                reason = ""
                if len(sequence) > limit:
                    if handle_long_sequences == "truncate":
                        half = limit // 2
                        processed = sequence[:half] + sequence[-half:]
                    else:
                        processed = None
                        reason = "Sequence too long — sequence candidate was excluded"
                children.append(
                    ExpandedSequence(
                        reaction_position=reaction_position,
                        sequence_position=sequence_position,
                        sequence=sequence,
                        processed_sequence=processed,
                        skip_reason=reason,
                    )
                )
            slices.append((reaction_position, start, len(children)))

        plan = cls(
            original_sequences=tuple(sequence_values),
            reaction_positions=positions,
            children=tuple(children),
            reaction_slices=tuple(slices),
        )
        plan._assert_invariants()
        return plan

    @property
    def valid_child_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, child in enumerate(self.children)
            if child.processed_sequence is not None
        )

    @property
    def skipped_children(self) -> dict[int, str]:
        return {
            index: child.skip_reason
            for index, child in enumerate(self.children)
            if child.skip_reason
        }

    @property
    def skipped_reactions(self) -> dict[int, str]:
        skipped: dict[int, str] = {}
        for reaction_position, start, end in self.reaction_slices:
            if start == end:
                skipped[reaction_position] = "Missing protein sequence"
                continue
            if all(self.children[index].processed_sequence is None for index in range(start, end)):
                skipped[reaction_position] = "Sequence too long — row was excluded"
        return skipped

    @property
    def requires_reduction(self) -> bool:
        for _reaction_position, start, end in self.reaction_slices:
            if end - start != 1:
                return True
        return False

    def valid_processed_by_reaction(self) -> tuple[Any | None, ...]:
        out: list[Any | None] = [None] * len(self.original_sequences)
        for _reaction_position, start, end in self.reaction_slices:
            valid_children = [
                self.children[index]
                for index in range(start, end)
                if self.children[index].processed_sequence is not None
            ]
            if len(valid_children) == 1:
                child = valid_children[0]
                out[child.reaction_position] = child.processed_sequence
        return tuple(out)

    def valid_reaction_indices_for_legacy(self) -> tuple[int, ...]:
        indices: list[int] = []
        for reaction_position, start, end in self.reaction_slices:
            valid_count = sum(
                1
                for index in range(start, end)
                if self.children[index].processed_sequence is not None
            )
            if valid_count == 1:
                indices.append(reaction_position)
        return tuple(indices)

    def sequence_count(self, reaction_position: int) -> int:
        for row_position, start, end in self.reaction_slices:
            if row_position == reaction_position:
                return end - start
        return 0

    def _assert_invariants(self) -> None:
        cursor = 0
        seen_reactions: list[int] = []
        for reaction_position, start, end in self.reaction_slices:
            if start != cursor or end < start:
                raise ValueError("Sequence expansion slices are not contiguous.")
            for child_index in range(start, end):
                child = self.children[child_index]
                if child.reaction_position != reaction_position:
                    raise ValueError("Sequence expansion mapped a child to the wrong reaction.")
                if child.sequence_position != child_index - start:
                    raise ValueError("Sequence positions are not contiguous within a reaction.")
            cursor = end
            seen_reactions.append(reaction_position)

        if cursor != len(self.children):
            raise ValueError("Sequence expansion did not account for every child.")
        if tuple(seen_reactions) != self.reaction_positions:
            raise ValueError("Sequence expansion changed reaction ordering.")


@dataclass(frozen=True)
class TargetPredictionUnit:
    """One flattened engine input unit after sequence/substrate adaptation."""

    reaction_position: int
    sequence_child_index: int
    sequence_position: int
    sequence: str
    substrate_position: int | None = None
    substrate: str | None = None


@dataclass(frozen=True)
class TargetExpansionPlan:
    """Position-safe mapping between engine units and original reaction rows."""

    sequence_plan: SequenceExpansionPlan
    units: tuple[TargetPredictionUnit, ...]
    reaction_slices: tuple[tuple[int, int, int], ...]
    substrate_tokens_by_reaction: tuple[tuple[str, ...], ...]
    uses_substrate_slots: bool = False

    def assert_result_count(self, values: Sequence[Any], label: str = "prediction") -> None:
        if len(values) != len(self.units):
            raise ValueError(
                f"Expanded batch produced {len(values)} {label}(s) for "
                f"{len(self.units)} prediction unit(s)."
            )


@dataclass(frozen=True)
class ReducedSequenceResults:
    predictions: list[Any]
    sources: list[str]
    extra_info: list[str]
    failed_reactions: dict[int, str]
    selected_sequences: list[str]


def split_sequence_list(value: Any) -> list[str]:
    """Split a semicolon-separated protein list, trimming empty fragments."""
    if value is None:
        return []
    try:
        if value != value:
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "<na>", "nat"}:
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def count_multi_sequence_rows(values: Iterable[Any]) -> int:
    """Return the number of rows containing more than one protein token."""
    return sum(1 for value in values if len(split_sequence_list(value)) > 1)


def reduce_sequence_predictions(
    *,
    plan: TargetExpansionPlan,
    target: str,
    child_predictions: Sequence[Any],
    child_sources: Sequence[str],
    child_errors: dict[int, str] | None,
    child_details: Sequence[str] | None,
    reaction_count: int,
) -> ReducedSequenceResults:
    """Reduce flattened sequence/substrate child results back to input rows."""
    plan.assert_result_count(child_predictions)
    plan.assert_result_count(child_sources, "source")
    if child_details is None:
        child_details = [""] * len(plan.units)
    plan.assert_result_count(child_details, "detail")

    errors = _normalise_child_errors(child_errors or {}, len(plan.units))
    predictions: list[Any] = [""] * reaction_count
    sources: list[str] = [""] * reaction_count
    extra_info: list[str] = [""] * reaction_count
    selected_sequences: list[str] = [""] * reaction_count
    failed_reactions: dict[int, str] = {}
    sequence_slice_by_reaction = {
        reaction_position: (start, end)
        for reaction_position, start, end in plan.sequence_plan.reaction_slices
    }
    skipped_reactions = plan.sequence_plan.skipped_reactions

    consumed = 0
    for reaction_position, start, end in plan.reaction_slices:
        sequence_items = _initial_sequence_items(
            plan,
            reaction_position,
            sequence_slice_by_reaction,
        )
        successful: list[tuple[int, float, str]] = []

        for unit_index in range(start, end):
            consumed += 1
            unit = plan.units[unit_index]
            error = errors.get(unit_index, "")
            numeric_value = None if error else _finite_number(child_predictions[unit_index])
            if numeric_value is None and not error:
                error = "Prediction could not be made"

            source = str(child_sources[unit_index] or "") if numeric_value is not None else ""
            item: dict[str, Any] = {
                "sequenceIndex": unit.sequence_position + 1,
                "sequence": unit.sequence,
                "prediction": numeric_value,
                "source": source,
                "error": error or None,
            }
            if unit.substrate_position is not None:
                item["substrateIndex"] = unit.substrate_position + 1
                item["substrate"] = unit.substrate
            detail = str(child_details[unit_index] or "")
            if detail:
                item["details"] = detail
            _merge_unit_item(sequence_items, unit, item)
            if numeric_value is not None:
                successful.append((unit_index, numeric_value, source))

        if plan.uses_substrate_slots and target != "kcat":
            _reduce_per_substrate_slots(
                plan=plan,
                target=target,
                reaction_position=reaction_position,
                successful=successful,
                predictions=predictions,
                sources=sources,
                selected_sequences=selected_sequences,
                failed_reactions=failed_reactions,
                sequence_items=sequence_items,
                skipped_reactions=skipped_reactions,
            )
        else:
            _reduce_scalar(
                plan=plan,
                target=target,
                reaction_position=reaction_position,
                successful=successful,
                predictions=predictions,
                sources=sources,
                selected_sequences=selected_sequences,
                failed_reactions=failed_reactions,
                sequence_items=sequence_items,
                skipped_reactions=skipped_reactions,
            )

        if sequence_items:
            extra_info[reaction_position] = json.dumps(
                sequence_items,
                allow_nan=False,
                separators=(",", ":"),
            )

    if consumed != len(plan.units):
        raise ValueError(
            f"Sequence reduction consumed {consumed} of {len(plan.units)} child result(s)."
        )

    return ReducedSequenceResults(
        predictions=predictions,
        sources=sources,
        extra_info=extra_info,
        failed_reactions=failed_reactions,
        selected_sequences=selected_sequences,
    )


def _initial_sequence_items(
    plan: TargetExpansionPlan,
    reaction_position: int,
    sequence_slice_by_reaction: dict[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    start_end = sequence_slice_by_reaction.get(reaction_position)
    if start_end is None:
        return items

    start, end = start_end
    for sequence_child_index in range(start, end):
        child = plan.sequence_plan.children[sequence_child_index]
        if not child.skip_reason:
            continue
        items.append(
            {
                "sequenceIndex": child.sequence_position + 1,
                "sequence": child.sequence,
                "prediction": None,
                "source": "",
                "error": child.skip_reason,
            }
        )
    return items


def _merge_unit_item(
    sequence_items: list[dict[str, Any]],
    unit: TargetPredictionUnit,
    item: dict[str, Any],
) -> None:
    if unit.substrate_position is None:
        sequence_items.append(item)
        return

    sequence_item = None
    for candidate in sequence_items:
        if candidate.get("sequenceIndex") == unit.sequence_position + 1:
            sequence_item = candidate
            break
    if sequence_item is None:
        sequence_item = {
            "sequenceIndex": unit.sequence_position + 1,
            "sequence": unit.sequence,
            "substrates": [],
        }
        sequence_items.append(sequence_item)

    substrate_item = {
        "substrateIndex": item["substrateIndex"],
        "substrate": item["substrate"],
        "prediction": item["prediction"],
        "source": item["source"],
        "error": item["error"],
    }
    if "details" in item:
        substrate_item["details"] = item["details"]
    sequence_item.setdefault("substrates", []).append(substrate_item)


def _reduce_scalar(
    *,
    plan: TargetExpansionPlan,
    target: str,
    reaction_position: int,
    successful: list[tuple[int, float, str]],
    predictions: list[Any],
    sources: list[str],
    selected_sequences: list[str],
    failed_reactions: dict[int, str],
    sequence_items: list[dict[str, Any]],
    skipped_reactions: dict[int, str],
) -> None:
    selected_unit: int | None = None
    if successful:
        selected_unit, selected_value, selected_source = _select_success(target, successful)
        unit = plan.units[selected_unit]
        predictions[reaction_position] = selected_value
        sources[reaction_position] = selected_source
        selected_sequences[reaction_position] = unit.sequence
    else:
        reason = _failure_reason(skipped_reactions, reaction_position, sequence_items)
        sources[reaction_position] = reason
        failed_reactions[reaction_position] = reason

    _mark_selected(sequence_items, selected_unit, plan)


def _reduce_per_substrate_slots(
    *,
    plan: TargetExpansionPlan,
    target: str,
    reaction_position: int,
    successful: list[tuple[int, float, str]],
    predictions: list[Any],
    sources: list[str],
    selected_sequences: list[str],
    failed_reactions: dict[int, str],
    sequence_items: list[dict[str, Any]],
    skipped_reactions: dict[int, str],
) -> None:
    substrate_tokens = plan.substrate_tokens_by_reaction[reaction_position]
    values: list[float | None] = []
    selected_units: set[int] = set()
    selected_sources: list[str] = []

    for substrate_position in range(len(substrate_tokens)):
        candidates = [
            result
            for result in successful
            if plan.units[result[0]].substrate_position == substrate_position
        ]
        if candidates:
            selected_unit, selected_value, selected_source = _select_success(target, candidates)
            values.append(selected_value)
            selected_units.add(selected_unit)
            selected_sources.append(selected_source)
            if not selected_sequences[reaction_position]:
                selected_sequences[reaction_position] = plan.units[selected_unit].sequence
        else:
            values.append(None)

    if any(value is not None for value in values):
        predictions[reaction_position] = json.dumps(
            values,
            allow_nan=False,
            separators=(",", ":"),
        )
        unique_sources = list(dict.fromkeys(source for source in selected_sources if source))
        if len(unique_sources) == 1:
            sources[reaction_position] = f"{unique_sources[0]} (per substrate)"
        else:
            sources[reaction_position] = "Mixed per-substrate sources; see Extra Info"
    else:
        reason = _failure_reason(skipped_reactions, reaction_position, sequence_items)
        sources[reaction_position] = reason
        failed_reactions[reaction_position] = reason

    _mark_selected(sequence_items, selected_units, plan)


def _select_success(
    target: str,
    successful: list[tuple[int, float, str]],
) -> tuple[int, float, str]:
    if target == "Km":
        return min(successful, key=lambda result: result[1])
    return max(successful, key=lambda result: result[1])


def _mark_selected(
    sequence_items: list[dict[str, Any]],
    selected: int | set[int] | None,
    plan: TargetExpansionPlan,
) -> None:
    selected_units = selected if isinstance(selected, set) else {selected}
    selected_lookup: set[tuple[int, int | None]] = set()
    for unit_index in selected_units:
        if unit_index is None:
            continue
        unit = plan.units[unit_index]
        selected_lookup.add((unit.sequence_position + 1, unit.substrate_position))

    for sequence_item in sequence_items:
        if "substrates" in sequence_item:
            for substrate_item in sequence_item["substrates"]:
                substrate_position = int(substrate_item["substrateIndex"]) - 1
                substrate_item["selected"] = (
                    sequence_item["sequenceIndex"],
                    substrate_position,
                ) in selected_lookup
        else:
            sequence_item["selected"] = (
                sequence_item.get("sequenceIndex"),
                None,
            ) in selected_lookup


def _failure_reason(
    skipped_reactions: dict[int, str],
    reaction_position: int,
    sequence_items: list[dict[str, Any]],
) -> str:
    sequence_reason = skipped_reactions.get(reaction_position)
    if sequence_reason:
        return sequence_reason
    child_errors = _collect_child_errors(sequence_items)
    if child_errors:
        return "; ".join(child_errors)
    return "Prediction could not be made for any sequence"


def _collect_child_errors(sequence_items: list[dict[str, Any]]) -> list[str]:
    """Return the distinct per-candidate error strings, preserving order."""
    unique: list[str] = []
    for item in sequence_items:
        substrates = item.get("substrates")
        candidates = substrates if isinstance(substrates, list) else [item]
        for candidate in candidates:
            error = candidate.get("error")
            if not error:
                continue
            error = str(error).strip()
            if error and error not in unique:
                unique.append(error)
    return unique


def _normalise_child_errors(errors: dict[int, str], child_count: int) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw_index, raw_reason in errors.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if 0 <= index < child_count:
            reason = str(raw_reason or "").strip()
            out[index] = reason or "Prediction could not be made"
    return out


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "nan", "inf", "+inf", "-inf"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    return number if math.isfinite(number) else None
