"""Folding policy records used by FoldPlan construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoldByFactor:
    temporal_steps: int


@dataclass(frozen=True)
class FoldByLanes:
    lanes: int
