"""Notebook-equivalent metrics for statically scheduled task reuse."""

from __future__ import annotations

import math


def schedule_metrics(schedule, task_graph, *, fclk_hz=300e6) -> dict:
    if not schedule.tasks:
        return {}
    latency_cycles = max(item.end for item in schedule.tasks.values())
    area_proxy = sum(float(resource.cost.get("lut") or 0) for resource in task_graph.resources.values())
    work_cost = sum(
        float(task_graph.resources[item.task.resource_id].cost.get("lut") or 0)
        for item in schedule.tasks.values()
    )
    issue_cycles = {}
    for item in schedule.tasks.values():
        issue_cycles[item.task.resource_id] = (
            issue_cycles.get(item.task.resource_id, 0) + item.task.ii
        )
    sample_ii_cycles = max(issue_cycles.values())
    utilisation = {
        resource: cycles / latency_cycles for resource, cycles in issue_cycles.items()
    }
    return {
        "area_proxy": area_proxy,
        "work_cost": work_cost,
        "work_to_area_ratio": work_cost / area_proxy if area_proxy else math.nan,
        "latency_cycles": latency_cycles,
        "sample_ii_cycles": sample_ii_cycles,
        "throughput_samples_per_sec": fclk_hz / sample_ii_cycles,
        "resource_utilisation": utilisation,
        "max_resource_utilisation": max(utilisation.values()),
    }
