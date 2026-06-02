"""Fold grouping and backend-independent folding policy records."""

from __future__ import annotations

from dataclasses import dataclass
import math


class _UnionFind:
    def __init__(self, items):
        self._parent = {item: item for item in items}

    def find(self, item):
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, left, right) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[left_root] = right_root

    def groups(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = {}
        for item in self._parent:
            grouped.setdefault(self.find(item), []).append(item)
        return list(grouped.values())


def _shared_fold_axis(source: dict, destination: dict, shape) -> int | None:
    if not source.get("foldable", True) or not destination.get("foldable", True):
        return None
    if shape is None:
        return None
    common_axes = set(source.get("fold_axes") or ()) & set(
        destination.get("fold_axes") or ()
    )
    for axis in common_axes:
        if 0 <= axis < len(shape) and shape[axis] not in (None, 1):
            return axis
    return None


def _group_parallelism(graph, members: list[int]) -> int | None:
    member_set = set(members)
    largest = None
    for source, destination in graph.edges:
        if source not in member_set or destination not in member_set:
            continue
        shape = graph.pmap[(source, destination)].get("tensor_shape")
        axis = _shared_fold_axis(
            graph.pmap[source],
            graph.pmap[destination],
            shape,
        )
        if axis is not None:
            largest = max(largest or 0, int(shape[axis]))
    return largest


def _reduce_consumes_fold_axis(node: dict, axes: list[int]) -> bool:
    if node.get("op") != "reduce":
        return False
    reduction_axes = (node.get("op_params") or {}).get("axes") or ()
    return bool(set(reduction_axes) & set(axes))


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


def stamp_fold_plan(
    graph,
    *,
    factor: int | None = None,
    lanes: int | None = None,
):
    """Stamp fold groups and the N/P/T policy needed by backend evaluation."""
    if (factor is None) == (lanes is None):
        raise ValueError("stamp_fold_plan(...) requires exactly one of factor= or lanes=")
    if factor is not None and factor < 1:
        raise ValueError(f"fold factor must be >= 1, got {factor}")
    if lanes is not None and lanes < 1:
        raise ValueError(f"fold lanes must be >= 1, got {lanes}")

    union_find = _UnionFind(graph.vertices)
    grouped_nodes: set[int] = set()
    for source, destination in graph.edges:
        edge = graph.pmap[(source, destination)]
        if (
            _shared_fold_axis(
                graph.pmap[source],
                graph.pmap[destination],
                edge.get("tensor_shape"),
            )
            is not None
        ):
            union_find.union(source, destination)
            grouped_nodes.update((source, destination))

    groups = [
        sorted(node for node in members if node in grouped_nodes)
        for members in union_find.groups()
    ]
    groups = [members for members in groups if members]

    fold_plan = []
    group_by_node = {}
    for group_id, members in enumerate(groups):
        parallelism = _group_parallelism(graph, members)
        if not parallelism:
            continue
        if lanes is not None:
            group_lanes = max(1, min(int(lanes), parallelism))
        else:
            requested_steps = max(1, min(int(factor), parallelism))
            group_lanes = max(1, math.ceil(parallelism / requested_steps))
        temporal_steps = max(1, math.ceil(parallelism / group_lanes))

        common_axes = None
        for node_id in members:
            axes = set(graph.pmap[node_id].get("fold_axes") or ())
            common_axes = axes if common_axes is None else common_axes & axes
        axes = sorted(common_axes or ())
        temporalised = [
            node_id
            for node_id in members
            if _reduce_consumes_fold_axis(graph.pmap[node_id], axes)
        ]
        snapshot = {
            "group_id": group_id,
            "axes": axes,
            "parallelism": parallelism,
            "lanes": group_lanes,
            "temporal_steps": temporal_steps,
            "members": members,
            "reductions_temporalised": temporalised,
        }
        fold_plan.append(snapshot)
        for node_id in members:
            group_by_node[node_id] = snapshot

    for node_id in graph.vertices:
        node = graph.pmap[node_id]
        group = group_by_node.get(node_id)
        if group is None:
            node["parallelism_N"] = 1
            node["lanes_P"] = 1
            node["temporal_steps_T"] = 1
            node["fold_group"] = None
            if node.get("op") == "reduce":
                node["reduce_mode"] = "spatial"
            continue

        node["parallelism_N"] = group["parallelism"]
        node["lanes_P"] = group["lanes"]
        node["temporal_steps_T"] = group["temporal_steps"]
        node["fold_group"] = group["group_id"]
        if node.get("op") == "reduce":
            if (
                node_id in group["reductions_temporalised"]
                and group["temporal_steps"] > 1
            ):
                node["reduce_mode"] = (
                    "temporal_accumulate" if group["lanes"] == 1 else "hybrid"
                )
            else:
                node["reduce_mode"] = "spatial"

    graph.pmap["fold_plan"] = fold_plan
    return graph


def make_fold_plan(graph, *, factor: int | None = None, lanes: int | None = None) -> FoldPlan:
    """Stamp a graph fold plan and return its typed backend-independent view."""
    stamp_fold_plan(graph, factor=factor, lanes=lanes)
    return FoldPlan(
        groups=tuple(_group_from_snapshot(group) for group in graph.pmap.get("fold_plan") or ())
    )
