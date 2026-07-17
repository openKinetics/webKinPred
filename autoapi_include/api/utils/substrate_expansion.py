"""Position-safe expansion and reduction for ``Substrates`` CSV inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ExpandedSubstrate:
    """One flattened protein/substrate pair and its original positions."""

    reaction_position: int
    substrate_position: int
    substrate: str


@dataclass(frozen=True)
class SubstrateExpansionPlan:
    """A positional mapping between reaction rows and flattened substrates."""

    reaction_positions: tuple[int, ...]
    children: tuple[ExpandedSubstrate, ...]
    reaction_slices: tuple[tuple[int, int, int], ...]

    @classmethod
    def build(
        cls,
        substrate_values: Sequence[Any],
        reaction_positions: Iterable[int],
    ) -> "SubstrateExpansionPlan":
        children: list[ExpandedSubstrate] = []
        slices: list[tuple[int, int, int]] = []
        positions = tuple(int(position) for position in reaction_positions)

        for reaction_position in positions:
            if reaction_position < 0 or reaction_position >= len(substrate_values):
                raise ValueError(f"Reaction position {reaction_position} is outside the input data.")
            start = len(children)
            for substrate_position, substrate in enumerate(
                split_substrate_list(substrate_values[reaction_position])
            ):
                children.append(
                    ExpandedSubstrate(
                        reaction_position=reaction_position,
                        substrate_position=substrate_position,
                        substrate=substrate,
                    )
                )
            slices.append((reaction_position, start, len(children)))

        plan = cls(
            reaction_positions=positions,
            children=tuple(children),
            reaction_slices=tuple(slices),
        )
        plan._assert_invariants()
        return plan

    def expanded_sequences(self, sequences: Sequence[Any]) -> list[Any]:
        return [sequences[child.reaction_position] for child in self.children]

    def expanded_parent_values(self, values: Sequence[Any]) -> list[Any]:
        return [values[child.reaction_position] for child in self.children]

    def expanded_substrates(self) -> list[str]:
        return [child.substrate for child in self.children]

    def assert_result_count(self, values: Sequence[Any], label: str = "prediction") -> None:
        if len(values) != len(self.children):
            raise ValueError(
                f"Expanded batch produced {len(values)} {label}(s) for "
                f"{len(self.children)} substrate input(s)."
            )

    def _assert_invariants(self) -> None:
        cursor = 0
        seen_reactions: list[int] = []
        for reaction_position, start, end in self.reaction_slices:
            if start != cursor or end < start:
                raise ValueError("Substrate expansion slices are not contiguous.")
            for child_index in range(start, end):
                child = self.children[child_index]
                if child.reaction_position != reaction_position:
                    raise ValueError("Substrate expansion mapped a child to the wrong reaction.")
                if child.substrate_position != child_index - start:
                    raise ValueError("Substrate positions are not contiguous within a reaction.")
            cursor = end
            seen_reactions.append(reaction_position)

        if cursor != len(self.children):
            raise ValueError("Substrate expansion did not account for every child.")
        if tuple(seen_reactions) != self.reaction_positions:
            raise ValueError("Substrate expansion changed reaction ordering.")


@dataclass(frozen=True)
class ReducedSubstrateResults:
    predictions: list[Any]
    sources: list[str]
    extra_info: list[str]
    failed_reactions: dict[int, str]


def split_substrate_list(value: Any) -> list[str]:
    """Split a semicolon list, trimming and ignoring empty separator fragments."""
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


def reduce_substrate_predictions(
    *,
    plan: SubstrateExpansionPlan,
    target: str,
    child_predictions: Sequence[Any],
    child_sources: Sequence[str],
    child_errors: dict[int, str] | None,
    child_details: Sequence[str] | None,
    reaction_count: int,
) -> ReducedSubstrateResults:
    """Reduce flattened child results back to their original reaction rows."""
    plan.assert_result_count(child_predictions)
    plan.assert_result_count(child_sources, "source")
    if child_details is None:
        child_details = [""] * len(plan.children)
    plan.assert_result_count(child_details, "detail")

    errors = _normalise_child_errors(child_errors or {}, len(plan.children))
    predictions: list[Any] = [""] * reaction_count
    sources: list[str] = [""] * reaction_count
    extra_info: list[str] = [""] * reaction_count
    failed_reactions: dict[int, str] = {}

    consumed = 0
    for reaction_position, start, end in plan.reaction_slices:
        items: list[dict[str, Any]] = []
        successful: list[tuple[int, float, str]] = []

        for child_index in range(start, end):
            consumed += 1
            child = plan.children[child_index]
            error = errors.get(child_index, "")
            numeric_value = None if error else _finite_number(child_predictions[child_index])
            if numeric_value is None and not error:
                error = "Prediction could not be made"

            source = str(child_sources[child_index] or "") if numeric_value is not None else ""
            item: dict[str, Any] = {
                "substrateIndex": child.substrate_position + 1,
                "substrate": child.substrate,
                "prediction": numeric_value,
                "source": source,
                "error": error or None,
            }
            detail = str(child_details[child_index] or "")
            if detail:
                item["details"] = detail
            items.append(item)
            if numeric_value is not None:
                successful.append((child_index, numeric_value, source))

        if target == "kcat":
            selected_child: int | None = None
            if successful:
                # ``max`` returns the first occurrence for ties, which makes the
                # substrate-order tie policy deterministic.
                selected_child, selected_value, selected_source = max(
                    successful,
                    key=lambda result: result[1],
                )
                predictions[reaction_position] = selected_value
                sources[reaction_position] = selected_source
            else:
                reason = _failure_reason(items)
                sources[reaction_position] = reason
                failed_reactions[reaction_position] = reason

            for offset, item in enumerate(items, start=start):
                item["selected"] = offset == selected_child
        else:
            values = [item["prediction"] for item in items]
            if successful:
                predictions[reaction_position] = json.dumps(
                    values,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                unique_sources = list(
                    dict.fromkeys(source for _, _, source in successful if source)
                )
                if len(unique_sources) == 1:
                    sources[reaction_position] = f"{unique_sources[0]} (per substrate)"
                else:
                    sources[reaction_position] = "Mixed per-substrate sources; see Extra Info"
            else:
                reason = _failure_reason(items)
                sources[reaction_position] = reason
                failed_reactions[reaction_position] = reason

        extra_info[reaction_position] = json.dumps(
            items,
            allow_nan=False,
            separators=(",", ":"),
        )

    if consumed != len(plan.children):
        raise ValueError(
            f"Substrate reduction consumed {consumed} of {len(plan.children)} child result(s)."
        )

    return ReducedSubstrateResults(
        predictions=predictions,
        sources=sources,
        extra_info=extra_info,
        failed_reactions=failed_reactions,
    )


def _failure_reason(items: list[dict[str, Any]]) -> str:
    """Return the distinct per-substrate error(s), or a generic fallback."""
    unique: list[str] = []
    for item in items:
        error = item.get("error")
        if not error:
            continue
        error = str(error).strip()
        if error and error not in unique:
            unique.append(error)
    if unique:
        return "; ".join(unique)
    return "Prediction could not be made for any substrate"


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
