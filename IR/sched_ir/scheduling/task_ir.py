"""Static task, resource and scheduled-task records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceInstance:
    resource_id: str
    source_node: int
    cost: dict
    latency: int
    ii: int


@dataclass(frozen=True)
class Task:
    task_id: str
    source_node: int
    temporal_step: int
    resource_id: str
    input_tokens: list[str]
    output_tokens: list[str]
    latency: int
    ii: int


@dataclass
class TaskGraph:
    tasks: dict[str, Task]
    resources: dict[str, ResourceInstance]
    initial_tokens: set[str]


@dataclass(frozen=True)
class ScheduledTask:
    task: Task
    start: int
    end: int


@dataclass
class TaskSchedule:
    tasks: dict[str, ScheduledTask]
    token_ready: dict[str, int]
    resource_next_issue: dict[str, int]
