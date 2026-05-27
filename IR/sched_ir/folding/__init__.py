"""Folding utilities for Sched-IR."""

from .fold_precision import apply_fold_aware_precision
from .folder import apply_timing_from_costs, stamp_fold_plan

__all__ = ["apply_fold_aware_precision", "apply_timing_from_costs", "stamp_fold_plan"]
