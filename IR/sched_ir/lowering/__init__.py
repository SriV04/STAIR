"""Stable namespace for NN-IR to Sched-IR lowering."""

from .decomposer import decompose_nn_to_sched

__all__ = ["decompose_nn_to_sched"]
