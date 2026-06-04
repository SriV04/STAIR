"""Cost proxy for temporal accumulator resources created during task expansion."""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Any


def _ceil_log2_at_least_one(value: int) -> int:
    return max(1, math.ceil(math.log2(max(int(value), 1))))


def _numeric_max(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = [_numeric_max(item) for item in value]
        values = [item for item in values if item is not None]
        return max(values) if values else None
    if hasattr(value, "item") and getattr(value, "shape", ()) == ():
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_scalar_numeric(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (dict, list, tuple)):
        return False
    if hasattr(value, "shape") and getattr(value, "shape", ()) != ():
        return False
    return _numeric_max(value) is not None


def _flatten_kifs(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == 3 and all(_is_scalar_numeric(part) for part in value):
            return [value]
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_kifs(item))
        return flattened
    return [value]


def _width_from_kif(kif: Any) -> int | None:
    if kif is None:
        return None
    if isinstance(kif, dict):
        bits = _numeric_max(kif.get("bits"))
        if bits is not None:
            return int(math.ceil(bits))
        i = _numeric_max(kif.get("i"))
        f = _numeric_max(kif.get("f"))
        if i is None or f is None:
            return None
        k = _numeric_max(kif.get("k")) or 0
        return int(math.ceil(k + i + f))
    if isinstance(kif, (list, tuple)) and len(kif) >= 3:
        values = [_numeric_max(part) for part in kif[:3]]
        if any(value is None for value in values):
            return None
        return int(math.ceil(sum(values)))
    bits = _numeric_max(getattr(kif, "bits", None))
    return int(math.ceil(bits)) if bits is not None else None


def _max_width_from_kifs(value: Any) -> int | None:
    widths = [_width_from_kif(kif) for kif in _flatten_kifs(value)]
    widths = [width for width in widths if width is not None]
    return max(widths) if widths else None


def _shape_numel(shape: Any) -> int | None:
    if shape is None:
        return None
    if (
        isinstance(shape, list)
        and shape
        and not isinstance(shape[0], (int, type(None)))
    ):
        shape = shape[0]
    dims = tuple(shape)
    if dims and dims[0] is None:
        dims = dims[1:]
    if not dims or any(dim is None for dim in dims):
        return None
    return int(reduce(mul, (int(dim) for dim in dims), 1))


def estimate_temporal_accumulator_cost(node: dict) -> dict:
    """Estimate one reused temporal accumulator resource for a folded reduction."""

    temporal_steps = max(int(node.get("temporal_steps_T") or 1), 1)
    guard_bits = math.ceil(math.log2(temporal_steps)) if temporal_steps > 1 else 0
    counter_width = _ceil_log2_at_least_one(temporal_steps + 1)

    input_width = _max_width_from_kifs(node.get("evaluated_input_kifs"))
    output_width = _max_width_from_kifs(node.get("evaluated_output_kifs"))
    elements = _shape_numel(node.get("evaluated_output_shapes"))

    missing_metadata = input_width is None or output_width is None or elements is None
    base_input_width = input_width if input_width is not None else 1
    base_output_width = output_width if output_width is not None else 1
    accumulator_width = max(base_output_width, base_input_width + guard_bits)
    accumulator_elements = max(int(elements or 1), 1)
    sum_bits = accumulator_elements * accumulator_width

    return {
        "lut": int(sum_bits + counter_width + 2),
        "ff": int(sum_bits + counter_width + 1),
        "dsp": 0,
        "bram": 0,
        "latency_cycles": 1,
        "ii": 1,
        "cost_mode": (
            "synthetic_temporal_accumulator_missing_metadata"
            if missing_metadata
            else "synthetic_temporal_accumulator_width_proxy"
        ),
        "accumulator_width_bits": int(accumulator_width),
        "accumulator_elements": int(accumulator_elements),
        "temporal_steps_T": int(temporal_steps),
        "guard_bits": int(guard_bits),
        "counter_width_bits": int(counter_width),
    }
