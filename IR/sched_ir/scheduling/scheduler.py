"""Static list scheduler for evaluated fixed-resource task graphs."""

from __future__ import annotations

from .task_ir import ScheduledTask, TaskGraph, TaskSchedule


def topological_tasks(task_graph: TaskGraph):
    ready_tokens = set(task_graph.initial_tokens)
    remaining = dict(task_graph.tasks)
    ordered = []
    while remaining:
        candidate = next(
            (
                task
                for task in remaining.values()
                if set(task.input_tokens).issubset(ready_tokens)
            ),
            None,
        )
        if candidate is None:
            missing = {token for task in remaining.values() for token in task.input_tokens} - ready_tokens
            raise ValueError(f"Task graph has unresolved or cyclic token dependencies: {sorted(missing)}")
        ordered.append(candidate)
        ready_tokens.update(candidate.output_tokens)
        del remaining[candidate.task_id]
    return ordered


def schedule_tasks(task_graph: TaskGraph) -> TaskSchedule:
    token_ready = {token: 0 for token in task_graph.initial_tokens}
    resource_next = {}
    scheduled = {}
    for task in topological_tasks(task_graph):
        input_ready = max((token_ready[token] for token in task.input_tokens), default=0)
        resource_ready = resource_next.get(task.resource_id, 0)
        start = max(input_ready, resource_ready)
        end = start + task.latency
        resource_next[task.resource_id] = start + task.ii
        for token in task.output_tokens:
            token_ready[token] = end
        scheduled[task.task_id] = ScheduledTask(task, start, end)
    return TaskSchedule(scheduled, token_ready, resource_next)
