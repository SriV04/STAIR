"""Post-schedule synchronisation diagnostics for static task schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .task_ir import ScheduledTask, TaskSchedule


@dataclass(frozen=True)
class SyncPointDiagnostic:
    task_name: str
    layer_type: str
    start_cycle: int
    latest_input_arrival_cycle: int
    earliest_input_arrival_cycle: int
    sync_wait_cycles: int
    max_buffer_wait_cycles: int
    input_arrival_cycles: dict[str, int]
    input_buffer_wait_cycles: dict[str, int]


@dataclass(frozen=True)
class SyncAnalysisReport:
    sync_point_count: int
    max_sync_wait_cycles: int
    total_sync_wait_cycles: int
    mean_sync_wait_cycles: float
    max_buffer_wait_cycles: int
    total_input_buffer_wait_cycles: int
    sync_legality_passed: bool
    sync_violations: list[str] = field(default_factory=list)
    sync_points: list[SyncPointDiagnostic] = field(default_factory=list)


def _scheduled_tasks(
    schedule_or_tasks: TaskSchedule | Iterable[ScheduledTask],
) -> list[ScheduledTask]:
    if isinstance(schedule_or_tasks, TaskSchedule):
        return list(schedule_or_tasks.tasks.values())
    return list(schedule_or_tasks)


def _task_layer_type(scheduled_task: ScheduledTask) -> str:
    task = scheduled_task.task
    return (
        getattr(task, "layer_type", None)
        or getattr(task, "op_kind", None)
        or task.task_kind
    )


def analyse_sync_points(
    scheduled_tasks: TaskSchedule | Iterable[ScheduledTask],
    source_arrival_cycle: int = 0,
) -> SyncAnalysisReport:
    """Analyse branch-alignment pressure after a legal schedule is produced."""
    schedule_items = _scheduled_tasks(scheduled_tasks)

    producer_end_by_output: dict[str, int] = {}
    for scheduled_task in schedule_items:
        for output in scheduled_task.task.output_tokens:
            producer_end_by_output[output] = scheduled_task.end

    diagnostics: list[SyncPointDiagnostic] = []
    violations: list[str] = []
    total_input_buffer_wait = 0

    for scheduled_task in schedule_items:
        task = scheduled_task.task
        if len(task.input_tokens) <= 1:
            continue

        input_arrivals = {
            input_token: producer_end_by_output.get(input_token, source_arrival_cycle)
            for input_token in task.input_tokens
        }
        arrivals = list(input_arrivals.values())
        if not arrivals:
            continue

        latest = max(arrivals)
        earliest = min(arrivals)
        sync_wait = latest - earliest
        input_buffer_waits = {
            input_token: scheduled_task.start - arrival
            for input_token, arrival in input_arrivals.items()
        }
        max_buffer_wait = max(input_buffer_waits.values(), default=0)
        total_input_buffer_wait += sum(input_buffer_waits.values())

        if scheduled_task.start < latest:
            violations.append(
                f"{task.task_id} starts at {scheduled_task.start}, "
                f"but latest input arrives at {latest}"
            )
        else:
            assert all(wait >= 0 for wait in input_buffer_waits.values())

        diagnostics.append(
            SyncPointDiagnostic(
                task_name=task.task_id,
                layer_type=_task_layer_type(scheduled_task),
                start_cycle=scheduled_task.start,
                latest_input_arrival_cycle=latest,
                earliest_input_arrival_cycle=earliest,
                sync_wait_cycles=sync_wait,
                max_buffer_wait_cycles=max_buffer_wait,
                input_arrival_cycles=input_arrivals,
                input_buffer_wait_cycles=input_buffer_waits,
            )
        )

    waits = [diagnostic.sync_wait_cycles for diagnostic in diagnostics]
    return SyncAnalysisReport(
        sync_point_count=len(diagnostics),
        max_sync_wait_cycles=max(waits, default=0),
        total_sync_wait_cycles=sum(waits),
        mean_sync_wait_cycles=(sum(waits) / len(waits)) if waits else 0.0,
        max_buffer_wait_cycles=max(
            (diagnostic.max_buffer_wait_cycles for diagnostic in diagnostics),
            default=0,
        ),
        total_input_buffer_wait_cycles=total_input_buffer_wait,
        sync_legality_passed=not violations,
        sync_violations=violations,
        sync_points=diagnostics,
    )
