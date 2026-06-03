"""Reconstruct symbolic token values for a scheduled folded design."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from .records import ScheduledTrace, SymbolicTokenValue
from .reductions import reduce_scheduled_values


@dataclass(frozen=True)
class NeutralQInt:
    bits: int = 0
    neutral: bool = True


def _shape_numel(shape: tuple[int, ...] | None) -> int | None:
    if shape is None or any(dim is None for dim in shape):
        return None
    return int(prod(shape))


def _normalise_axis(axis: int, shape: tuple[int, ...]) -> int:
    if 0 <= axis < len(shape):
        return axis
    adjusted = axis - 1
    if 0 <= adjusted < len(shape):
        return adjusted
    return 0


def _normalise_declared_fold_axis(axis: int, shape: tuple[int, ...]) -> int:
    if axis > 0 and 0 <= axis - 1 < len(shape):
        return axis - 1
    return _normalise_axis(axis, shape)


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


def _logical_output_shape(node: dict) -> tuple[int, ...] | None:
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


def _slice_for_step(node: dict, step: int) -> tuple[Any, ...] | None:
    steps = max(int(node.get("temporal_steps_T") or 1), 1)
    if (
        node.get("op") == "reduce"
        and node.get("reduce_mode") in {"hybrid", "temporal_accumulate"}
    ):
        return None
    shape = _logical_output_shape(node)
    if steps == 1 or shape is None:
        return None
    axes = node.get("fold_axes") or []
    axis = _normalise_declared_fold_axis(int(axes[0]), shape) if axes else 0
    lanes = max(int(node.get("lanes_P") or 1), 1)
    start = step * lanes
    stop = min(start + lanes, int(shape[axis]))
    result = [slice(None)] * len(shape)
    result[axis] = slice(start, stop)
    return tuple(result)


def _select_symbolic_output(symbolic_outputs: list[Any], step: int) -> Any:
    if not symbolic_outputs:
        return None
    return symbolic_outputs[min(step, len(symbolic_outputs) - 1)]


def _qints_for_step(node: dict, step: int) -> list[Any]:
    all_qints = list(((node.get("evaluated_output_qints") or [[]])[0]) or [])
    steps = max(int(node.get("temporal_steps_T") or 1), 1)
    if steps == 1:
        return all_qints
    shape = (node.get("evaluated_output_shapes") or [None])[0]
    axes = node.get("fold_axes") or []
    axis = _normalise_axis(int(axes[0]), tuple(shape)) if axes and shape else 0
    lanes = max(int(node.get("lanes_P") or 1), 1)

    if shape and len(all_qints) == shape[axis]:
        start = step * lanes
        stop = min(start + lanes, len(all_qints))
        qints = all_qints[start:stop]
        missing = lanes - len(qints)
    elif shape and _shape_numel(tuple(shape)) == len(all_qints):
        block = int(prod(tuple(shape)[axis + 1 :])) if axis + 1 < len(shape) else 1
        start = step * lanes * block
        stop = min(start + lanes * block, len(all_qints))
        qints = all_qints[start:stop]
        missing = lanes * block - len(qints)
    else:
        start = step * lanes
        stop = min(start + lanes, len(all_qints))
        qints = all_qints[start:stop]
        missing = lanes - len(qints)

    if missing > 0:
        qints.extend(NeutralQInt() for _ in range(missing))
    return qints


def _symbolic_add(left, right):
    return left + right


def _qints_from_symbolic_value(value) -> list[Any]:
    if hasattr(value, "qint"):
        return [value.qint]
    try:
        arr = np.asarray(value, dtype=object)
    except Exception:
        return []
    if arr.shape == ():
        item = arr.item()
        return [item.qint] if hasattr(item, "qint") else []
    return [item.qint for item in arr.reshape(-1) if hasattr(item, "qint")]


def _terminal_tokens(task_graph, produced_tokens):
    consumed = {
        token
        for task in task_graph.tasks.values()
        for token in task.input_tokens
    }
    return [token for token in produced_tokens if token not in consumed]


def _assembled_symbolic_outputs(evaluation, token_values, output_tokens):
    outputs = [token_values[token_id].value for token_id in output_tokens]
    by_node: dict[int, list[SymbolicTokenValue]] = {}
    for token_id in output_tokens:
        token = token_values[token_id]
        if token.source_node is not None:
            by_node.setdefault(token.source_node, []).append(token)

    for node_id, tokens in by_node.items():
        node = evaluation.graph.pmap[node_id]
        if (
            node.get("op") == "reduce"
            and node.get("reduce_mode") in {"hybrid", "temporal_accumulate"}
            and len(tokens) > 1
        ):
            ordered = sorted(tokens, key=lambda token: token.temporal_step or 0)
            reduced = reduce_scheduled_values(
                [token.value for token in ordered],
                mode=node.get("reduce_mode"),
                reducer=_symbolic_add,
            )
            outputs = [
                value
                for value, token_id in zip(outputs, output_tokens)
                if token_values[token_id].source_node != node_id
            ]
            outputs.append(reduced)
    return outputs


def build_scheduled_trace(evaluation, task_graph, schedule) -> ScheduledTrace:
    token_values = {}
    produced_tokens = []
    for scheduled in schedule.tasks.values():
        task = scheduled.task
        node = evaluation.graph.pmap[task.source_node]
        if task.task_kind == "temporal_accumulator":
            input_tokens = [token_values[token_id] for token_id in task.input_tokens]
            ordered = sorted(input_tokens, key=lambda token: token.temporal_step or 0)
            accumulated = reduce_scheduled_values(
                [token.value for token in ordered],
                mode=node.get("reduce_mode"),
                reducer=_symbolic_add,
            )
            qints = _qints_from_symbolic_value(accumulated)
            if not qints:
                qints = [
                    qint
                    for token in ordered
                    for qint in token.qints
                    if not getattr(qint, "neutral", False)
                ]
            shape = (
                ordered[0].shape
                if ordered
                else (node.get("evaluated_output_shapes") or [None])[0]
            )
            for token_id in task.output_tokens:
                produced_tokens.append(token_id)
                token_values[token_id] = SymbolicTokenValue(
                    token_id=token_id,
                    source_node=task.source_node,
                    temporal_step=None,
                    logical_slice=None,
                    value=accumulated,
                    qints=qints,
                    shape=shape,
                    ready_cycle=scheduled.end,
                    fold_group=node.get("fold_group"),
                    task_kind=task.task_kind,
                )
            continue

        trace_id = node.get("backend_trace_id")
        artifact = evaluation.runtime_artifacts[trace_id]
        outputs = list(getattr(artifact, "symbolic_outputs", []) or [])
        symbolic_value = _select_symbolic_output(outputs, task.temporal_step)
        shape = (node.get("evaluated_output_shapes") or [None])[0]
        qints = _qints_for_step(node, task.temporal_step)
        for token_id in task.output_tokens:
            produced_tokens.append(token_id)
            token_values[token_id] = SymbolicTokenValue(
                token_id=token_id,
                source_node=task.source_node,
                temporal_step=task.temporal_step,
                logical_slice=_slice_for_step(node, task.temporal_step),
                value=symbolic_value,
                qints=qints,
                shape=shape,
                ready_cycle=scheduled.end,
                fold_group=node.get("fold_group"),
                task_kind=task.task_kind,
            )

    output_tokens = _terminal_tokens(task_graph, produced_tokens)
    return ScheduledTrace(
        token_values=token_values,
        output_tokens=output_tokens,
        symbolic_outputs=_assembled_symbolic_outputs(
            evaluation,
            token_values,
            output_tokens,
        ),
        output_qints=[
            qint
            for token_id in output_tokens
            for qint in token_values[token_id].qints
            if not getattr(qint, "neutral", False)
        ],
        metadata={"task_count": len(schedule.tasks)},
    )
