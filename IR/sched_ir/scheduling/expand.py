"""Expand evaluated folded hardware into static executions and data tokens."""

from __future__ import annotations

from .task_ir import ResourceInstance, Task, TaskGraph


def _topological_order(graph) -> list[int]:
    indegree = {node: 0 for node in graph.vertices}
    successors = {node: [] for node in graph.vertices}
    for source, destination in graph.edges:
        indegree[destination] += 1
        successors[source].append(destination)
    ready = [node for node in graph.vertices if indegree[node] == 0]
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(order) != len(graph.vertices):
        raise ValueError("Cannot expand tasks for cyclic Sched-IR")
    return order


def _tokens_for(base: str, count: int) -> list[str]:
    return [base] if count == 1 else [f"{base}:t{step}" for step in range(count)]


def expand_tasks(evaluated_graph) -> TaskGraph:
    """Create task executions for fixed resources using graph edge identity."""
    tasks = {}
    resources = {}
    initial_tokens: set[str] = set()
    output_tokens: dict[int, list[str]] = {}

    for node_id in _topological_order(evaluated_graph):
        node = evaluated_graph.pmap[node_id]
        steps = max(int(node.get("temporal_steps_T") or 1), 1)
        cost = dict(node.get("cost") or {})
        resource_id = f"node:{node_id}"
        latency = max(int(cost.get("latency_cycles") or 1), 1)
        ii = max(int(cost.get("ii") or 1), 1)
        resources[resource_id] = ResourceInstance(resource_id, node_id, cost, latency, ii)

        incoming = [
            (source, evaluated_graph.pmap[(source, destination)])
            for source, destination in evaluated_graph.edges
            if destination == node_id
        ]
        outgoing = [
            evaluated_graph.pmap[(source, destination)]
            for source, destination in evaluated_graph.edges
            if source == node_id
        ]
        output_base = (
            outgoing[0].get("value_id")
            if outgoing and outgoing[0].get("value_id") is not None
            else f"{node_id}:out"
        )
        node_outputs = _tokens_for(output_base, steps)
        output_tokens[node_id] = node_outputs

        for step in range(steps):
            if not incoming:
                inputs = [f"input:{node_id}:t{step}" if steps > 1 else f"input:{node_id}"]
                initial_tokens.update(inputs)
            else:
                inputs = []
                for source, edge in incoming:
                    source_tokens = output_tokens[source]
                    consume_mode = edge.get("consume_mode") or "stepwise"
                    if consume_mode == "all":
                        inputs.extend(source_tokens)
                    elif len(source_tokens) == steps and steps > 1:
                        inputs.append(source_tokens[step])
                    else:
                        inputs.extend(source_tokens)
            outputs = [node_outputs[step]]
            task_id = f"{resource_id}:t{step}" if steps > 1 else resource_id
            tasks[task_id] = Task(
                task_id,
                node_id,
                step,
                resource_id,
                inputs,
                outputs,
                latency,
                ii,
            )

    return TaskGraph(tasks, resources, initial_tokens)
