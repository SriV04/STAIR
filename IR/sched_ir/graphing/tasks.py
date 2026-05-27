"""Heterograph rendering input for scheduled task executions."""

from __future__ import annotations

from heterograph import HGraph


def task_schedule_to_hgraph(task_graph, schedule):
    graph = HGraph()
    graph.pmap["name"] = "sched_ir_tasks"
    graph.pmap["view"] = "task_schedule"
    vertex_by_task = {}
    producer_by_token = {}

    for task_id, task in task_graph.tasks.items():
        scheduled = schedule.tasks[task_id]
        vertex = graph.add_vx()
        vertex_by_task[task_id] = vertex
        graph.pmap[vertex] = {
            "task_id": task_id,
            "source_node": task.source_node,
            "temporal_step": task.temporal_step,
            "resource_id": task.resource_id,
            "input_tokens": list(task.input_tokens),
            "output_tokens": list(task.output_tokens),
            "start": scheduled.start,
            "end": scheduled.end,
            "latency": task.latency,
            "ii": task.ii,
        }
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

    return graph
