"""Orchestrate symbolic correctness comparison for evaluated designs."""

from __future__ import annotations

from .compare import compare_output_widths
from .da4ml_reference import trace_unfolded_reference
from .fold_group_semantics import check_fold_group_semantics
from .scheduled_trace import build_scheduled_trace


def _provenance_for(scheduled_trace):
    """Where this result came from, in terms of the scheduled trace."""
    provenance = []
    for token_id in scheduled_trace.output_tokens:
        token = scheduled_trace.token_values[token_id]
        provenance.append(
            {
                "token_id": token_id,
                "source_node": token.source_node,
                "fold_group": token.fold_group,
                "temporal_step": token.temporal_step,
                "ready_cycle": token.ready_cycle,
                "logical_slice": token.logical_slice,
            }
        )
    return provenance


def check_symbolic_correctness(design, *, model, config: dict | None = None):
    reference = trace_unfolded_reference(model, config=config)
    scheduled = build_scheduled_trace(
        design.evaluation,
        design.task_ir,
        design.task_schedule,
    )
    report = compare_output_widths(
        reference_qints=reference.output_qints,
        scheduled_qints=scheduled.output_qints,
        provenance=_provenance_for(scheduled),
    )
    fold_group_failures = check_fold_group_semantics(
        design.fold_plan,
        scheduled,
        design.evaluation,
    )
    report.fold_group_failures.extend(fold_group_failures)
    report.metadata.update(
        {
            "reference": reference.metadata,
            "scheduled": scheduled.metadata,
            "fold_group_count": len(getattr(design.fold_plan, "groups", ()) or ()),
            "fold_group_semantic_failure_count": len(fold_group_failures),
        }
    )
    return report
