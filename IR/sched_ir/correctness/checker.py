"""Orchestrate symbolic correctness comparison for evaluated designs."""

from __future__ import annotations

from .compare import compare_output_widths
from .da4ml_reference import trace_unfolded_reference
from .scheduled_trace import build_scheduled_trace


def _provenance_for(scheduled_trace):
    provenance = []
    for token_id in scheduled_trace.output_tokens:
        token = scheduled_trace.token_values[token_id]
        provenance.append(
            {
                "token_id": token_id,
                "source_node": token.source_node,
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
    report.metadata.update(
        {
            "reference": reference.metadata,
            "scheduled": scheduled.metadata,
            "fold_group_count": len(getattr(design.fold_plan, "groups", ()) or ()),
        }
    )
    return report
