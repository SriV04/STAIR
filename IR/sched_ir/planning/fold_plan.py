"""Typed facade for backend-independent folding policy decisions."""

from __future__ import annotations

from dataclasses import dataclass

from ..folding.folder import stamp_fold_plan


@dataclass(frozen=True)
class FoldGroup:
    group_id: int
    axes: tuple[int, ...]
    parallelism_n: int
    lanes_p: int
    temporal_steps_t: int
    members: tuple[int, ...]
    reductions_temporalised: tuple[int, ...]


@dataclass(frozen=True)
class FoldPlan:
    groups: tuple[FoldGroup, ...]


def _group_from_snapshot(snapshot: dict) -> FoldGroup:
    return FoldGroup(
        group_id=int(snapshot["group_id"]),
        axes=tuple(snapshot.get("axes") or ()),
        parallelism_n=int(snapshot["parallelism"]),
        lanes_p=int(snapshot["lanes"]),
        temporal_steps_t=int(snapshot["temporal_steps"]),
        members=tuple(snapshot.get("members") or ()),
        reductions_temporalised=tuple(snapshot.get("reductions_temporalised") or ()),
    )


def make_fold_plan(graph, *, factor: int | None = None, lanes: int | None = None) -> FoldPlan:
    """Stamp a graph fold plan and return its typed backend-independent view."""
    stamp_fold_plan(graph, factor=factor, lanes=lanes)
    return FoldPlan(
        groups=tuple(_group_from_snapshot(group) for group in graph.pmap.get("fold_plan") or ())
    )
