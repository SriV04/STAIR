"""Scheduled reduction semantics for symbolic checking."""

from __future__ import annotations


def reduce_scheduled_values(values, *, mode: str, reducer):
    values = list(values)
    if not values:
        raise ValueError("cannot reduce an empty scheduled value list")
    if mode == "spatial":
        if len(values) != 1:
            raise ValueError(f"spatial reduction expects one value, got {len(values)}")
        return values[0]
    if mode not in {"hybrid", "temporal_accumulate"}:
        raise ValueError(f"unsupported scheduled reduction mode {mode!r}")
    result = values[0]
    for value in values[1:]:
        result = reducer(result, value)
    return result
