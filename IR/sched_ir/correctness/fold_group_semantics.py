"""Fold-group-local symbolic semantics checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .records import FoldGroupSemanticFailure, SymbolicTokenValue


@dataclass(frozen=True)
class _SliceExpr:
    parent: Any
    axis: int
    start: int
    stop: int
    token: SymbolicTokenValue


def _normalise_axis(axis: int, rank: int) -> int:
    if 0 <= axis < rank:
        return axis
    adjusted = axis - 1
    if 0 <= adjusted < rank:
        return adjusted
    return axis


def _normalise_declared_fold_axis(axis: int, rank: int) -> int:
    if axis > 0 and 0 <= axis - 1 < rank:
        return axis - 1
    return _normalise_axis(axis, rank)


def _strip_batch_shape(shape) -> tuple[int, ...] | None:
    if shape is None:
        return None
    try:
        dims = tuple(None if dim is None else int(dim) for dim in tuple(shape))
    except TypeError:
        return None
    if dims and dims[0] is None:
        dims = dims[1:]
    if any(dim is None for dim in dims):
        return None
    return dims


def _shape_from_output_shapes(output_shapes):
    if not output_shapes:
        return None
    if all(isinstance(dim, (int, np.integer, type(None))) for dim in output_shapes):
        return output_shapes
    return output_shapes[0]


def _node_output_shape(node: dict) -> tuple[int, ...] | None:
    params = node.get("op_params") or {}
    for shape in (
        params.get("output_shape"),
        _shape_from_output_shapes(params.get("output_shapes")),
        (node.get("evaluated_output_shapes") or [None])[0],
    ):
        stripped = _strip_batch_shape(shape)
        if stripped:
            return stripped
    return None


def _same_symbolic_parent(left, right) -> bool:
    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right))
        except Exception:
            return False
    try:
        equal = left == right
    except Exception:
        return False
    if isinstance(equal, bool):
        return equal
    if isinstance(equal, np.bool_):
        return bool(equal)
    return False


def _slice_interval(
    token: SymbolicTokenValue,
    *,
    expected_axis: int,
) -> tuple[int, int, int] | str:
    logical_slice = token.logical_slice
    if logical_slice is None:
        return (expected_axis, 0, 0)
    axis_slices = [
        (index, part)
        for index, part in enumerate(logical_slice)
        if isinstance(part, slice) and (part.start is not None or part.stop is not None)
    ]
    if len(axis_slices) != 1:
        return "logical slice must identify exactly one folded axis"
    axis, folded_slice = axis_slices[0]
    if axis != expected_axis:
        return f"expected fold axis {expected_axis} but token used axis {axis}"
    if folded_slice.step not in (None, 1):
        return "logical slice step must be 1"
    start = 0 if folded_slice.start is None else int(folded_slice.start)
    stop = int(folded_slice.stop) if folded_slice.stop is not None else None
    if stop is None:
        return "logical slice stop is required"
    return axis, start, stop


def _failure(
    group,
    source_node: int,
    reason: str,
    *,
    expected=None,
    actual=None,
    tokens: list[SymbolicTokenValue] | None = None,
) -> FoldGroupSemanticFailure:
    return FoldGroupSemanticFailure(
        group_id=group.group_id,
        source_nodes=(source_node,),
        reason=reason,
        expected=expected,
        actual=actual,
        provenance={
            "tokens": [
                {
                    "token_id": token.token_id,
                    "source_node": token.source_node,
                    "temporal_step": token.temporal_step,
                    "logical_slice": token.logical_slice,
                    "ready_cycle": token.ready_cycle,
                }
                for token in (tokens or [])
            ]
        },
    )


def _is_temporalized_reduction(node: dict) -> bool:
    return (
        node.get("op") == "reduce"
        and node.get("reduce_mode") in {"hybrid", "temporal_accumulate"}
    )


def _check_temporalized_reduction_member(
    group,
    source_node: int,
    tokens,
) -> list[FoldGroupSemanticFailure]:
    partials_by_step: dict[int, list[SymbolicTokenValue]] = {}
    accumulators = []
    for token in tokens:
        if token.task_kind == "temporal_accumulator":
            accumulators.append(token)
        elif token.temporal_step is not None:
            partials_by_step.setdefault(int(token.temporal_step), []).append(token)

    failures = []
    for step in range(int(group.temporal_steps_t)):
        step_tokens = partials_by_step.get(step, [])
        if not step_tokens:
            failures.append(
                _failure(
                    group,
                    source_node,
                    f"missing temporal step {step}",
                    expected=list(range(int(group.temporal_steps_t))),
                    actual=sorted(partials_by_step),
                    tokens=list(tokens),
                )
            )
        elif len(step_tokens) > 1:
            failures.append(
                _failure(
                    group,
                    source_node,
                    f"duplicate temporal step {step}",
                    expected=1,
                    actual=len(step_tokens),
                    tokens=step_tokens,
                )
            )
    if failures:
        return failures

    if not accumulators:
        return [
            _failure(
                group,
                source_node,
                "missing temporal reduction accumulator",
                expected=1,
                actual=0,
                tokens=list(tokens),
            )
        ]
    if len(accumulators) > 1:
        return [
            _failure(
                group,
                source_node,
                "duplicate temporal reduction accumulator",
                expected=1,
                actual=len(accumulators),
                tokens=accumulators,
            )
        ]
    return []


def _check_member(
    group,
    source_node: int,
    tokens,
    evaluation,
) -> list[FoldGroupSemanticFailure]:
    node = evaluation.graph.pmap[source_node]
    if _is_temporalized_reduction(node):
        return _check_temporalized_reduction_member(group, source_node, tokens)

    full_shape = _node_output_shape(node)
    if full_shape is None:
        return [
            _failure(
                group,
                source_node,
                "cannot recover full output shape",
                expected="known full output shape",
                actual=None,
                tokens=list(tokens),
            )
        ]
    if not group.axes:
        return [
            _failure(
                group,
                source_node,
                "fold group has no declared fold axis",
                expected="declared fold axis",
                actual=(),
                tokens=list(tokens),
            )
        ]

    fold_axis = _normalise_declared_fold_axis(int(group.axes[0]), len(full_shape))
    full_extent = int(full_shape[fold_axis])
    by_step: dict[int, list[SymbolicTokenValue]] = {}
    for token in tokens:
        if token.temporal_step is None:
            continue
        by_step.setdefault(int(token.temporal_step), []).append(token)

    failures = []
    for step in range(int(group.temporal_steps_t)):
        step_tokens = by_step.get(step, [])
        if not step_tokens:
            failures.append(
                _failure(
                    group,
                    source_node,
                    f"missing temporal step {step}",
                    expected=list(range(int(group.temporal_steps_t))),
                    actual=sorted(by_step),
                    tokens=list(tokens),
                )
            )
        elif len(step_tokens) > 1:
            failures.append(
                _failure(
                    group,
                    source_node,
                    f"duplicate temporal step {step}",
                    expected=1,
                    actual=len(step_tokens),
                    tokens=step_tokens,
                )
            )
    if failures:
        return failures

    ordered_tokens = [by_step[step][0] for step in sorted(by_step)]
    if int(group.temporal_steps_t) == 1 and ordered_tokens[0].logical_slice is None:
        return []

    parent = ordered_tokens[0].value if ordered_tokens else None
    if any(
        not _same_symbolic_parent(parent, token.value)
        for token in ordered_tokens[1:]
    ):
        return [
            _failure(
                group,
                source_node,
                "slices have different symbolic parents",
                expected=repr(parent),
                actual=[repr(token.value) for token in ordered_tokens],
                tokens=ordered_tokens,
            )
        ]

    slices = []
    for token in ordered_tokens:
        interval = _slice_interval(token, expected_axis=fold_axis)
        if isinstance(interval, str):
            return [
                _failure(
                    group,
                    source_node,
                    interval,
                    expected=fold_axis,
                    actual=token.logical_slice,
                    tokens=[token],
                )
            ]
        axis, start, stop = interval
        if start < 0 or stop > full_extent:
            return [
                _failure(
                    group,
                    source_node,
                    (
                        f"slice {start}:{stop} is out of bounds for folded "
                        f"dimension {full_extent}"
                    ),
                    expected=(0, full_extent),
                    actual=(start, stop),
                    tokens=[token],
                )
            ]
        if stop < start:
            return [
                _failure(
                    group,
                    source_node,
                    f"slice {start}:{stop} has negative extent",
                    expected="start <= stop",
                    actual=(start, stop),
                    tokens=[token],
                )
            ]
        slices.append(_SliceExpr(parent, axis, start, stop, token))

    slices = sorted(slices, key=lambda item: item.start)
    cursor = 0
    for item in slices:
        if item.start > cursor:
            return [
                _failure(
                    group,
                    source_node,
                    f"slice coverage gap between {cursor} and {item.start}",
                    expected=(cursor, item.start),
                    actual=[(expr.start, expr.stop) for expr in slices],
                    tokens=[expr.token for expr in slices],
                )
            ]
        if item.start < cursor:
            return [
                _failure(
                    group,
                    source_node,
                    f"slice overlap between {cursor} and {item.start}",
                    expected="non-overlapping slices",
                    actual=[(expr.start, expr.stop) for expr in slices],
                    tokens=[expr.token for expr in slices],
                )
            ]
        cursor = item.stop

    if cursor != full_extent:
        return [
            _failure(
                group,
                source_node,
                f"slice coverage gap between {cursor} and {full_extent}",
                expected=(0, full_extent),
                actual=[(expr.start, expr.stop) for expr in slices],
                tokens=[expr.token for expr in slices],
            )
        ]

    # Minimal canonicalisation rule:
    # Concat(Slice(Y, 0:a), Slice(Y, a:b), ..., axis=k) -> Y.
    canonical = (
        parent
        if slices and slices[0].start == 0 and cursor == full_extent
        else None
    )
    if not _same_symbolic_parent(canonical, parent):
        return [
            _failure(
                group,
                source_node,
                "canonical expression mismatch",
                expected=repr(parent),
                actual=repr(canonical),
                tokens=[expr.token for expr in slices],
            )
        ]
    return []


def check_fold_group_semantics(
    fold_plan,
    scheduled_trace,
    evaluation,
) -> list[FoldGroupSemanticFailure]:
    failures: list[FoldGroupSemanticFailure] = []
    tokens_by_group_node: dict[tuple[int, int], list[SymbolicTokenValue]] = {}
    for token in scheduled_trace.token_values.values():
        if token.fold_group is None or token.source_node is None:
            continue
        key = (int(token.fold_group), int(token.source_node))
        tokens_by_group_node.setdefault(key, []).append(token)

    for group in getattr(fold_plan, "groups", ()) or ():
        for source_node in group.members:
            tokens = tokens_by_group_node.get((int(group.group_id), int(source_node)), [])
            failures.extend(_check_member(group, int(source_node), tokens, evaluation))
    return failures
