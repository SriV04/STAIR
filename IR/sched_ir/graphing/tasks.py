"""Heterograph rendering input for scheduled task executions."""

from __future__ import annotations

from heterograph import HGraph

from .styling import apply_sched_style, sched_vx_label


ACCUMULATOR_COLOR = "#8E24AA"
ACCUMULATOR_OUTPUT_COLOR = "#6A1B9A"


def _fmt_shape(shape) -> str:
    if shape is None:
        return "?"
    return "x".join("?" if dim is None else str(dim) for dim in shape)


def _strip_batch(shape):
    if shape is None:
        return None
    dims = tuple(shape)
    return dims[1:] if dims and dims[0] is None else dims


def _normalise_fold_axis(axis: int, shape) -> int:
    adjusted = axis - 1
    if 0 <= adjusted < len(shape):
        return adjusted
    if 0 <= axis < len(shape):
        return axis
    return 0


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _infer_reduce_output_shape(properties):
    params = properties.get("op_params") or {}
    input_shape = _strip_batch(_first_present(
        params.get("input_shape"),
        params.get("in_shape"),
    ))
    axes = params.get("axes") or []
    if input_shape is None:
        return None
    axes = [
        _normalise_fold_axis(int(axis), input_shape)
        for axis in axes
    ]
    keepdims = bool(params.get("keepdims"))
    if keepdims:
        return tuple(1 if index in axes else dim for index, dim in enumerate(input_shape))
    return tuple(dim for index, dim in enumerate(input_shape) if index not in axes)


def _node_output_shape(properties):
    params = properties.get("op_params") or {}
    if properties.get("op") == "reduce":
        inferred = _infer_reduce_output_shape(properties)
        if inferred is not None:
            return inferred
    evaluated = properties.get("evaluated_output_shapes") or []
    return _strip_batch(_first_present(
        evaluated[0] if evaluated else None,
        params.get("output_shape"),
        params.get("out_shape"),
    ))


def _task_output_shape(properties):
    if properties.get("task_kind") == "temporal_accumulator":
        input_shapes = properties.get("input_shapes") or []
        return input_shapes[0] if input_shapes else _node_output_shape(properties)
    evaluated = properties.get("evaluated_output_shapes") or []
    if evaluated:
        return _strip_batch(evaluated[0])
    shape = _node_output_shape(properties)
    if shape is None:
        return None
    if properties.get("op") == "reduce":
        return shape
    temporal_step = properties.get("temporal_step")
    steps = int(properties.get("temporal_steps_T") or 1)
    if temporal_step is None or steps <= 1:
        return shape
    axes = properties.get("fold_axes") or []
    if not axes:
        return shape
    axis = _normalise_fold_axis(int(axes[0]), shape)
    lanes = int(properties.get("lanes_P") or 1)
    if not (0 <= axis < len(shape)) or shape[axis] is None:
        return shape
    start = int(temporal_step) * lanes
    width = max(0, min(lanes, int(shape[axis]) - start))
    dims = list(shape)
    dims[axis] = width
    return tuple(dims)


def task_vx_label(graph, vertex) -> str:
    """Label a scheduled execution with its source operation and allocation."""
    properties = graph.pmap[vertex]
    base_label = sched_vx_label(graph, vertex)
    lines = [base_label]
    if properties.get("task_kind") == "temporal_accumulator":
        lines.append("[temporal accumulator]")
        lines.append(f"partials: {len(properties.get('input_tokens') or [])}")
    input_shapes = properties.get("input_shapes")
    if input_shapes is not None:
        lines.append(
            "task inputs: "
            + ", ".join(_fmt_shape(shape) for shape in input_shapes)
        )
    lines.extend(
        [
            f"task: {properties['task_id']}",
            f"resource: {properties['resource_id']}",
        ]
    )
    return "\n".join(lines)


def task_edge_label(graph, edge) -> str:
    token = str(graph.pmap[edge].get("token") or "")
    if graph.pmap[edge].get("edge_kind") == "temporal_accumulator_input":
        return f"acc {token}"
    if graph.pmap[edge].get("edge_kind") == "temporal_accumulator_output":
        return f"out {token}"
    return token


def task_vx_fillcolor(graph, vertex) -> str:
    if graph.pmap[vertex].get("task_kind") == "temporal_accumulator":
        return ACCUMULATOR_COLOR
    return graph.vstyle["_sched_fillcolor"](graph, vertex)


def task_vx_shape(graph, vertex) -> str:
    if graph.pmap[vertex].get("task_kind") == "temporal_accumulator":
        return "component"
    return graph.vstyle["_sched_shape"](graph, vertex)


def task_vx_style(graph, vertex) -> str:
    if graph.pmap[vertex].get("task_kind") == "temporal_accumulator":
        return "filled,bold"
    return graph.vstyle["_sched_style"](graph, vertex)


def task_vx_fontcolor(graph, vertex) -> str:
    if graph.pmap[vertex].get("task_kind") == "temporal_accumulator":
        return "white"
    return graph.vstyle["_sched_fontcolor"](graph, vertex)


def task_edge_color(graph, edge) -> str:
    kind = graph.pmap[edge].get("edge_kind")
    if kind == "temporal_accumulator_input":
        return ACCUMULATOR_COLOR
    if kind == "temporal_accumulator_output":
        return ACCUMULATOR_OUTPUT_COLOR
    return "#666666"


def task_edge_style(graph, edge) -> str:
    kind = graph.pmap[edge].get("edge_kind")
    if kind == "temporal_accumulator_input":
        return "dashed"
    if kind == "temporal_accumulator_output":
        return "bold"
    return "solid"


def apply_task_style(graph) -> None:
    """Attach task-execution labels while retaining Sched-IR operation colors."""
    apply_sched_style(graph)
    graph.vstyle["_sched_fillcolor"] = graph.vstyle["fillcolor"]
    graph.vstyle["_sched_shape"] = graph.vstyle["shape"]
    graph.vstyle["_sched_style"] = graph.vstyle["style"]
    graph.vstyle["_sched_fontcolor"] = graph.vstyle["fontcolor"]
    graph.vstyle["fillcolor"] = task_vx_fillcolor
    graph.vstyle["shape"] = task_vx_shape
    graph.vstyle["style"] = task_vx_style
    graph.vstyle["fontcolor"] = task_vx_fontcolor
    graph.vstyle["label"] = task_vx_label
    graph.estyle["label"] = task_edge_label
    graph.estyle["color"] = task_edge_color
    graph.estyle["style"] = task_edge_style


def task_schedule_to_hgraph(task_graph, schedule, *, source_graph=None):
    graph = HGraph()
    graph.pmap["name"] = "sched_ir_tasks"
    graph.pmap["view"] = "task_schedule"
    vertex_by_task = {}
    producer_by_token = {}
    shape_by_token = {}

    for task_id, task in task_graph.tasks.items():
        scheduled = schedule.tasks[task_id]
        vertex = graph.add_vx()
        vertex_by_task[task_id] = vertex
        source_metadata = (
            dict(source_graph.pmap[task.source_node])
            if source_graph is not None
            else {}
        )
        source_metadata.update({
            "task_id": task_id,
            "source_node": task.source_node,
            "temporal_step": task.temporal_step,
            "task_kind": task.task_kind,
            "resource_id": task.resource_id,
            "input_tokens": list(task.input_tokens),
            "output_tokens": list(task.output_tokens),
            "input_shapes": [shape_by_token.get(token) for token in task.input_tokens],
            "start": scheduled.start,
            "end": scheduled.end,
            "t_start": scheduled.start,
            "t_end": scheduled.end,
            "latency": task.latency,
            "ii": task.ii,
        })
        graph.pmap[vertex] = source_metadata
        output_shape = _task_output_shape(source_metadata)
        for token in task.output_tokens:
            producer_by_token[token] = vertex
            shape_by_token[token] = output_shape

    for task_id, task in task_graph.tasks.items():
        destination = vertex_by_task[task_id]
        for token in task.input_tokens:
            source = producer_by_token.get(token)
            if source is None:
                continue
            edge = graph.add_edge(source, destination)[0]
            source_kind = graph.pmap[source].get("task_kind")
            destination_kind = graph.pmap[destination].get("task_kind")
            if destination_kind == "temporal_accumulator":
                edge_kind = "temporal_accumulator_input"
            elif source_kind == "temporal_accumulator":
                edge_kind = "temporal_accumulator_output"
            else:
                edge_kind = "data"
            graph.pmap[edge] = {"token": token, "edge_kind": edge_kind}

    apply_task_style(graph)
    return graph
