"""DA4ML backend for folded Sched-IR evaluation."""

from .records import (
    DA4MLNodeArtifact,
    EvaluationResult,
    NodeEvaluation,
    PipelineEvalConfig,
    PrimitiveEvaluation,
    SymbolicTensorState,
)


def __getattr__(name):
    if name == "DA4MLBackend":
        from .plugin import DA4MLBackend

        return DA4MLBackend
    raise AttributeError(name)

__all__ = [
    "DA4MLNodeArtifact",
    "EvaluationResult",
    "NodeEvaluation",
    "PipelineEvalConfig",
    "PrimitiveEvaluation",
    "SymbolicTensorState",
    "DA4MLBackend",
]
