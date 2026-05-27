"""Validation for typed FoldPlan records."""

from __future__ import annotations

import math

from .fold_plan import FoldPlan


def validate_fold_plan(graph, plan: FoldPlan) -> None:
    for group in plan.groups:
        if group.temporal_steps_t != math.ceil(group.parallelism_n / group.lanes_p):
            raise ValueError(f"fold group {group.group_id} violates T == ceil(N/P)")
        for node_id in group.members:
            node = graph.pmap[node_id]
            if (
                node.get("parallelism_N"),
                node.get("lanes_P"),
                node.get("temporal_steps_T"),
            ) != (group.parallelism_n, group.lanes_p, group.temporal_steps_t):
                raise ValueError(f"fold group {group.group_id} disagrees with node {node_id}")
