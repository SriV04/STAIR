"""Fast analytic cost proxies for attention barrier operations.

These estimates are intentionally local and deterministic. hls4ml HLS reports
can calibrate the constants later, but scheduler sweeps should not invoke HLS.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Any


DEFAULT_INPUT_WIDTH = 8
DEFAULT_OUTPUT_WIDTH = 8
LUT_INPUTS = 6
LUTRAM_PACK_BITS = 64


def _ceil_log2(value: int) -> int:
    return int(math.ceil(math.log2(max(int(value), 1)))) if value > 1 else 0


def _ceil_div(numerator: int, denominator: int) -> int:
    return int(math.ceil(int(numerator) / max(int(denominator), 1)))


def _numeric_max(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return _numeric_max(value.get("bits"))
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


def _width_from_kif(value: Any, default: int) -> int:
    if isinstance(value, dict):
        bits = _numeric_max(value.get("bits"))
        if bits is not None:
            return max(1, int(math.ceil(bits)))
        i = _numeric_max(value.get("i"))
        f = _numeric_max(value.get("f"))
        k = _numeric_max(value.get("k")) or 0
        if i is not None and f is not None:
            return max(1, int(math.ceil(k + i + f)))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        parts = [_numeric_max(item) for item in value[:3]]
        if all(item is not None for item in parts):
            return max(1, int(math.ceil(sum(parts))))
    bits = _numeric_max(value)
    return max(1, int(math.ceil(bits))) if bits is not None else int(default)


def _max_input_width(params: dict, index: int) -> int:
    kifs = params.get("input_kifs") or []
    if index < len(kifs):
        return _width_from_kif(kifs[index], DEFAULT_INPUT_WIDTH)
    widths = params.get("in_bws") or []
    if index < len(widths):
        width = _numeric_max(widths[index])
        if width is not None:
            return max(1, int(math.ceil(width)))
    return DEFAULT_INPUT_WIDTH


def _output_width(params: dict) -> int:
    return _width_from_kif(
        params.get("output_kif"),
        int(math.ceil(_numeric_max(params.get("out_bw")) or DEFAULT_OUTPUT_WIDTH)),
    )


def _shape_dims(shape: Any) -> tuple[int | None, ...]:
    if shape is None:
        return ()
    return tuple(None if dim is None else int(dim) for dim in tuple(shape))


def _normalise_shape(value: Any) -> tuple[int | None, ...]:
    return _shape_dims(value)


def _normalise_shapes(value: Any) -> tuple[tuple[int | None, ...], ...]:
    return tuple(_normalise_shape(shape) for shape in (value or ()))


def _concrete_product(values: list[int | None] | tuple[int | None, ...]) -> int:
    concrete = [int(value) for value in values if value not in (None, 1)]
    return int(reduce(mul, concrete, 1))


def _split_equation(equation: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    equation = "".join(str(equation).split())
    if "..." in equation or "->" not in equation:
        raise ValueError(f"attention cost oracle requires explicit einsum equation, got {equation!r}")
    lhs, output = equation.split("->", 1)
    inputs = lhs.split(",")
    if len(inputs) != 2:
        raise ValueError(f"attention cost oracle supports two-input einsums, got {equation!r}")
    return tuple(inputs[0]), tuple(inputs[1]), tuple(output)


def _axis_sizes_by_label(
    input_axes: tuple[str, ...],
    weight_axes: tuple[str, ...],
    input_shapes: list[tuple[int | None, ...]],
) -> dict[str, int | None]:
    dims: dict[str, int | None] = {}
    for axes, shape in zip((input_axes, weight_axes), input_shapes, strict=True):
        for axis, dim in zip(axes, shape, strict=True):
            current = dims.get(axis)
            if current is None:
                dims[axis] = dim
            elif dim is not None and current != dim:
                raise ValueError(f"einsum axis {axis!r} has inconsistent dimensions {current} and {dim}")
    return dims


def _einsum_work(params: dict) -> tuple[int, int]:
    equation = params.get("equation")
    input_shapes = [_shape_dims(shape) for shape in (params.get("input_shapes") or [])]
    output_shape = _shape_dims(params.get("output_shape"))
    if not equation or len(input_shapes) != 2:
        return _concrete_product(output_shape), 1

    input_axes, weight_axes, output_axes = _split_equation(equation)
    if len(input_axes) != len(input_shapes[0]) or len(weight_axes) != len(input_shapes[1]):
        return _concrete_product(output_shape), 1
    dims = _axis_sizes_by_label(input_axes, weight_axes, input_shapes)
    output_elements = _concrete_product([dims.get(axis) for axis in output_axes])
    input_axis_set = set(input_axes) | set(weight_axes)
    reduction_terms = _concrete_product(
        [dims.get(axis) for axis in input_axis_set if axis not in set(output_axes)]
    )
    return max(output_elements, 1), max(reduction_terms, 1)


def estimate_einsum_cost(node_pmap: dict) -> dict:
    """Estimate a fully spatial two-input dynamic einsum barrier."""

    params = node_pmap.get("op_params") or {}
    output_elements, reduction_terms = _einsum_work(params)
    multiply_count = output_elements * reduction_terms
    left_width = _max_input_width(params, 0)
    right_width = _max_input_width(params, 1)
    output_width = _output_width(params)
    accumulator_width = max(output_width, left_width + right_width + _ceil_log2(reduction_terms))

    multiplier_lut = multiply_count * _ceil_div(left_width * right_width, LUT_INPUTS)
    adder_count = output_elements * max(reduction_terms - 1, 0)
    adder_lut = adder_count * accumulator_width
    output_register_bits = output_elements * output_width
    latency_cycles = max(1, 1 + _ceil_log2(reduction_terms))

    return {
        "lut": int(multiplier_lut + adder_lut),
        "ff": int(output_register_bits + output_elements * latency_cycles),
        "dsp": 0,
        "bram": 0,
        "latency_cycles": int(latency_cycles),
        "ii": 1,
        "pipeline_stages": int(latency_cycles),
        "cost_mode": "analytic_einsum_attention_oracle",
        "einsum_output_elements": int(output_elements),
        "einsum_reduction_terms": int(reduction_terms),
        "einsum_multiply_count": int(multiply_count),
        "input_width_bits": [int(left_width), int(right_width)],
        "output_width_bits": int(output_width),
        "accumulator_width_bits": int(accumulator_width),
    }


def _normalise_axes(axis: Any, rank: int) -> tuple[int, ...]:
    if axis is None or axis == ():
        axis = (-1,)
    axes = axis if isinstance(axis, (list, tuple)) else (axis,)
    result = []
    for item in axes:
        axis_index = int(item)
        if axis_index < 0:
            axis_index += rank
        if 0 <= axis_index < rank:
            result.append(axis_index)
    return tuple(result or [rank - 1])


def estimate_softmax_cost(node_pmap: dict) -> dict:
    """Estimate a stable lookup-table softmax barrier."""

    params = node_pmap.get("op_params") or {}
    shape = _shape_dims(params.get("input_shape") or params.get("output_shape"))
    rank = len(shape)
    axes = _normalise_axes(params.get("axis"), rank) if rank else ()
    axis_size = _concrete_product([shape[index] for index in axes]) if axes else 1
    element_count = _concrete_product(shape)
    group_count = max(_ceil_div(element_count, axis_size), 1)
    input_width = _width_from_kif(params.get("input_kif"), int(_numeric_max(params.get("in_bw")) or DEFAULT_INPUT_WIDTH))
    output_width = _width_from_kif(
        params.get("output_kif"),
        int(_numeric_max(params.get("out_bw")) or input_width),
    )
    stable = bool(params.get("stable", True))
    exp_entries = int(params.get("exp_lut_entries") or 1024)
    inv_entries = int(params.get("inv_lut_entries") or 1024)
    table_width = max(output_width, input_width)

    compare_lut = group_count * max(axis_size - 1, 0) * input_width if stable else 0
    subtract_lut = element_count * input_width if stable else 0
    exp_lut = element_count * _ceil_div(exp_entries * table_width, LUTRAM_PACK_BITS)
    sum_width = output_width + _ceil_log2(axis_size)
    sum_lut = group_count * max(axis_size - 1, 0) * sum_width
    reciprocal_lut = group_count * _ceil_div(inv_entries * table_width, LUTRAM_PACK_BITS)
    normalise_lut = element_count * _ceil_div(output_width * output_width, LUT_INPUTS)
    latency_cycles = (
        (1 + _ceil_log2(axis_size) if stable else 0)
        + 1
        + _ceil_log2(axis_size)
        + 1
        + 1
    )

    return {
        "lut": int(compare_lut + subtract_lut + exp_lut + sum_lut + reciprocal_lut + normalise_lut),
        "ff": int(element_count * output_width + group_count * sum_width),
        "dsp": 0,
        "bram": 0,
        "latency_cycles": int(max(latency_cycles, 1)),
        "ii": 1,
        "pipeline_stages": int(max(latency_cycles, 1)),
        "cost_mode": "analytic_softmax_attention_oracle",
        "softmax_axis_size": int(axis_size),
        "softmax_group_count": int(group_count),
        "softmax_element_count": int(element_count),
        "softmax_stable": stable,
        "input_width_bits": int(input_width),
        "output_width_bits": int(output_width),
        "exp_lut_entries": int(exp_entries),
        "inv_lut_entries": int(inv_entries),
    }


def _calibration_records(calibration: Any) -> list[dict]:
    if calibration is None:
        return []
    if isinstance(calibration, list):
        return [record for record in calibration if isinstance(record, dict)]
    if not isinstance(calibration, dict):
        return []
    records = (
        calibration.get("hls4ml_reports")
        or calibration.get("reports")
        or calibration.get("calibration")
        or calibration.get("records")
        or []
    )
    return [record for record in records if isinstance(record, dict)]


def _normalise_layer_name(record: dict) -> Any:
    return record.get("layer_name") or record.get("nn_layer_name") or record.get("name")


def _record_matches_node(record: dict, node_pmap: dict) -> bool:
    op = node_pmap.get("op")
    params = node_pmap.get("op_params") or {}
    if record.get("op") != op:
        return False

    node_name = node_pmap.get("nn_layer_name")
    record_name = _normalise_layer_name(record)
    if record_name is not None:
        return record_name == node_name

    if op == "einsum":
        return (
            record.get("equation") == params.get("equation")
            and _normalise_shapes(record.get("input_shapes")) == _normalise_shapes(params.get("input_shapes"))
            and _normalise_shape(record.get("output_shape")) == _normalise_shape(params.get("output_shape"))
        )
    if op == "softmax":
        return (
            tuple(record.get("axis") or ()) == tuple(params.get("axis") or ())
            and _normalise_shape(record.get("input_shape")) == _normalise_shape(params.get("input_shape"))
            and _normalise_shape(record.get("output_shape")) == _normalise_shape(params.get("output_shape"))
        )
    return False


def _integer_field(value: Any) -> int | None:
    numeric = _numeric_max(value)
    return int(math.ceil(numeric)) if numeric is not None else None


def _report_cost(record: dict) -> dict:
    nested_cost = record.get("cost") if isinstance(record.get("cost"), dict) else {}
    resources = record.get("resources") if isinstance(record.get("resources"), dict) else {}
    sources = (nested_cost, resources, record)
    cost: dict[str, Any] = {}
    for key in ("lut", "ff", "dsp", "bram", "latency_cycles", "ii"):
        for source in sources:
            if key in source:
                value = _integer_field(source[key])
                if value is not None:
                    cost[key] = value
                break
    if "latency_cycles" not in cost:
        for alias in ("latency", "Latency", "latency_min", "latency_max"):
            if alias in record:
                value = _integer_field(record[alias])
                if value is not None:
                    cost["latency_cycles"] = value
                    break
    if "ii" not in cost:
        for alias in ("interval", "Interval", "initiation_interval", "initiationInterval"):
            if alias in record:
                value = _integer_field(record[alias])
                if value is not None:
                    cost["ii"] = value
                    break
    return cost


def _apply_hls4ml_report(base_cost: dict, record: dict) -> dict:
    reported = _report_cost(record)
    if not reported:
        return base_cost
    cost = dict(base_cost)
    cost.update(
        {
            "lut": int(reported.get("lut", cost.get("lut", 0))),
            "ff": int(reported.get("ff", cost.get("ff", 0))),
            "dsp": int(reported.get("dsp", cost.get("dsp", 0))),
            "bram": int(reported.get("bram", cost.get("bram", 0))),
            "latency_cycles": max(1, int(reported.get("latency_cycles", cost.get("latency_cycles", 1)))),
            "ii": max(1, int(reported.get("ii", cost.get("ii", 1)))),
            "cost_mode": "hls4ml_report_calibrated",
            "ii_source": "hls4ml_report",
            "latency_source": "hls4ml_report",
        }
    )
    cost["pipeline_stages"] = int(cost["latency_cycles"])
    if _normalise_layer_name(record) is not None:
        cost["hls4ml_report_layer_name"] = _normalise_layer_name(record)
    return cost


def _matching_calibration_record(node_pmap: dict, calibration: Any) -> dict | None:
    for record in _calibration_records(calibration):
        if _record_matches_node(record, node_pmap):
            return record
    return None


def estimate_attention_cost(node_pmap: dict, calibration: Any = None) -> dict | None:
    op = node_pmap.get("op")
    if op == "einsum":
        cost = estimate_einsum_cost(node_pmap)
    if op == "softmax":
        cost = estimate_softmax_cost(node_pmap)
    if op not in {"einsum", "softmax"}:
        return None
    record = _matching_calibration_record(node_pmap, calibration)
    return _apply_hls4ml_report(cost, record) if record is not None else cost
