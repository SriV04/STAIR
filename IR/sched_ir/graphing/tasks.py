"""Heterograph rendering input for scheduled task executions."""

from __future__ import annotations

from heterograph import HGraph

from .styling import apply_sched_style, sched_vx_label


def task_vx_label(graph, vertex) -> str:
    """Label a scheduled execution with its source operation and allocation."""
    properties = graph.pmap[vertex]
    base_label = sched_vx_label(graph, vertex)
    return "\n".join(
        [
            base_label,
            f"task: {properties['task_id']}",
            f"resource: {properties['resource_id']}",
        ]
    )


def task_edge_label(graph, edge) -> str:
    return str(graph.pmap[edge].get("token") or "")


def apply_task_style(graph) -> None:
    """Attach task-execution labels while retaining Sched-IR operation colors."""
    apply_sched_style(graph)
    graph.vstyle["label"] = task_vx_label
    graph.estyle["label"] = task_edge_label


def task_schedule_to_hgraph(task_graph, schedule, *, source_graph=None):
    graph = HGraph()
    graph.pmap["name"] = "sched_ir_tasks"
    graph.pmap["view"] = "task_schedule"
    vertex_by_task = {}
    producer_by_token = {}

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
            "resource_id": task.resource_id,
            "input_tokens": list(task.input_tokens),
            "output_tokens": list(task.output_tokens),
            "start": scheduled.start,
            "end": scheduled.end,
            "t_start": scheduled.start,
            "t_end": scheduled.end,
            "latency": task.latency,
            "ii": task.ii,
        })
        graph.pmap[vertex] = source_metadata
        for token in task.output_tokens:
            producer_by_token[token] = vertex

    for task_id, task in task_graph.tasks.items():
        destination = vertex_by_task[task_id]
        for token in task.input_tokens:
            source = producer_by_token.get(token)
            if source is None:
                continue
            edge = graph.add_edge(source, destination)[0]
            graph.pmap[edge] = {"token": token, "edge_kind": "data"}

    apply_task_style(graph)
    return graph
