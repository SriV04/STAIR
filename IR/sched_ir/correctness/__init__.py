"""Symbolic correctness checks for scheduled Sched-IR designs."""

from .checker import check_symbolic_correctness
from .compare import compare_output_widths, normalize_output_widths
from .da4ml_reference import trace_unfolded_reference
from .records import (
    CorrectnessFailure,
    CorrectnessReport,
    ReferenceTrace,
    ScheduledTrace,
    SymbolicTokenValue,
)
from .reductions import reduce_scheduled_values
from .scheduled_trace import build_scheduled_trace

__all__ = [
    "CorrectnessFailure",
    "CorrectnessReport",
    "ReferenceTrace",
    "ScheduledTrace",
    "SymbolicTokenValue",
    "build_scheduled_trace",
    "check_symbolic_correctness",
    "compare_output_widths",
    "normalize_output_widths",
    "reduce_scheduled_values",
    "trace_unfolded_reference",
]
