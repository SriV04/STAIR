"""Runtime and snapshot records for DA4ML Sched-IR evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineEvalConfig:
    enabled: bool = True
    latency_cutoff: float = 5.0
    retiming: bool = True
    verbose: bool = False


@dataclass
class SymbolicTensorState:
    value_id: str
    shape: tuple[int, ...]
    qints: list
    kifs: list
    latency: float
    producer_node: int | None
    symbolic_value: Any


@dataclass
class DA4MLNodeArtifact:
    trace_id: str
    node_id: int
    symbolic_inputs: list[Any]
    symbolic_outputs: list[Any]
    comb_logic: Any
    pipeline: Any | None


@dataclass
class PrimitiveEvaluation:
    symbolic_inputs: list[Any]
    symbolic_outputs: list[Any]
    output_shapes: list[tuple[int, ...]]
    output_qints: list
    output_kifs: list
    output_latency: float
    cost: dict
    n_ops: int
    comb_logic: Any
    pipeline: Any | None
    kernel_meta: dict


@dataclass
class NodeEvaluation:
    node_id: int
    trace_id: str
    input_states: list[SymbolicTensorState]
    output_states: list[SymbolicTensorState]
    cost: dict
    latency: tuple[float, float]
    n_ops: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    graph: Any
    node_results: list[NodeEvaluation]
    tensor_states: dict[str, SymbolicTensorState]
    runtime_artifacts: dict[str, DA4MLNodeArtifact]


def snapshot_node_evaluation(evaluation: NodeEvaluation) -> dict:
    """Return the graph-safe portion of a node's evaluated backend state."""
    return {
        "backend": "da4ml",
        "backend_trace_id": evaluation.trace_id,
        "evaluated_input_shapes": [state.shape for state in evaluation.input_states],
        "evaluated_output_shapes": [state.shape for state in evaluation.output_states],
        "evaluated_input_qints": [state.qints for state in evaluation.input_states],
        "evaluated_input_kifs": [state.kifs for state in evaluation.input_states],
        "evaluated_output_qints": [state.qints for state in evaluation.output_states],
        "evaluated_output_kifs": [state.kifs for state in evaluation.output_states],
        "evaluated_input_latency": [state.latency for state in evaluation.input_states],
        "evaluated_output_latency": [state.latency for state in evaluation.output_states],
        "cost": evaluation.cost,
        "evaluated_n_ops": evaluation.n_ops,
    }
