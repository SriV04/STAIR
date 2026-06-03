"""Records used by symbolic correctness checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SymbolicTokenValue:
    token_id: str
    source_node: int | None
    temporal_step: int | None
    logical_slice: tuple[Any, ...] | None
    value: Any
    qints: list[Any]
    shape: tuple[int, ...] | None
    ready_cycle: int | None
    fold_group: int | None = None
    task_kind: str = "compute"


@dataclass(frozen=True)
class ReferenceTrace:
    symbolic_inputs: list[Any]
    symbolic_outputs: list[Any]
    output_qints: list[Any]
    output_shapes: list[tuple[int, ...]]
    comb_logic: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduledTrace:
    token_values: dict[str, SymbolicTokenValue]
    output_tokens: list[str]
    symbolic_outputs: list[Any]
    output_qints: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrectnessFailure:
    output_index: int
    path: str
    expected: Any
    actual: Any
    reason: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldGroupSemanticFailure:
    group_id: int
    source_nodes: tuple[int, ...]
    reason: str
    expected: Any
    actual: Any
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrectnessReport:
    reference_qints: list[Any]
    scheduled_qints: list[Any]
    failures: list[CorrectnessFailure]
    checked_output_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    fold_group_failures: list[FoldGroupSemanticFailure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures and not self.fold_group_failures
