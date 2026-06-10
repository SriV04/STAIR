"""Equation-aware fold-axis analysis for dense/einsum Sched-IR nodes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EinsumSpec:
    equation: str
    input_axes: tuple[str, ...]
    weight_axes: tuple[str, ...]
    output_axes: tuple[str, ...]
    input_shape: tuple[int | None, ...]
    weight_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    layer_name: str


def _split_equation(equation: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    equation = "".join(equation.split())
    if "..." in equation:
        raise ValueError(f"fold-axis analysis requires explicit axes, got {equation!r}")
    if "->" not in equation:
        raise ValueError(f"einsum equation must include an output, got {equation!r}")
    inputs, output = equation.split("->", 1)
    parts = inputs.split(",")
    if len(parts) != 2:
        raise ValueError(f"fold-axis analysis supports two-input einsums, got {equation!r}")
    input_axes, weight_axes = (tuple(part.strip()) for part in parts)
    output_axes = tuple(output.strip())
    if not input_axes or not weight_axes or not output_axes:
        raise ValueError(f"einsum equation has empty axis list, got {equation!r}")
    return input_axes, weight_axes, output_axes


def _as_shape(shape) -> tuple[int | None, ...] | None:
    if shape is None:
        return None
    return tuple(None if dim is None else int(dim) for dim in shape)


def parse_einsum_spec(
    *,
    equation: str | None,
    input_shape,
    weight_shape,
    output_shape,
    layer_name: str | None = None,
) -> EinsumSpec | None:
    """Build an explicit axis spec from a layer equation and tensor shapes."""
    if equation is None:
        return None
    input_axes, weight_axes, output_axes = _split_equation(str(equation).strip())
    input_shape = _as_shape(input_shape)
    weight_shape = _as_shape(weight_shape)
    output_shape = _as_shape(output_shape)
    if input_shape is None or weight_shape is None or output_shape is None:
        return None
    if len(input_axes) != len(input_shape):
        return None
    if len(weight_axes) != len(weight_shape):
        return None
    if len(output_axes) != len(output_shape):
        return None
    return EinsumSpec(
        equation=str(equation).strip(),
        input_axes=input_axes,
        weight_axes=weight_axes,
        output_axes=output_axes,
        input_shape=input_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        layer_name=layer_name or "<unknown>",
    )


def detect_fold_axes(spec: EinsumSpec) -> list[str]:
    """Return preserved non-batch input axes that do not index the weights."""
    fold_axes: list[str] = []
    output_axis_set = set(spec.output_axes)
    weight_axis_set = set(spec.weight_axes)
    for index, axis in enumerate(spec.input_axes):
        size = spec.input_shape[index]
        is_batch = axis in {"b", "batch"} or size is None
        if is_batch:
            continue
        if axis in output_axis_set and axis not in weight_axis_set and size > 1:
            fold_axes.append(axis)
    return fold_axes


def detect_fold_axis_indices(spec: EinsumSpec) -> list[int]:
    """Return Sched-IR integer fold axes for the equation-aware candidates."""
    axis_names = set(detect_fold_axes(spec))
    return [index for index, axis in enumerate(spec.input_axes) if axis in axis_names]


def derive_local_equation(spec: EinsumSpec, fold_axis: str) -> str:
    """Drop batch and one structural fold axis from the parent equation."""
    remove = {fold_axis, "b", "batch"}
    local_input = "".join(axis for axis in spec.input_axes if axis not in remove)
    local_weight = "".join(spec.weight_axes)
    local_output = "".join(axis for axis in spec.output_axes if axis not in remove)
    return f"{local_input},{local_weight}->{local_output}"


def fold_analysis_from_params(params: dict, *, layer_name: str | None = None) -> dict:
    """Return fold-axis metadata for a lowered dense/einsum parameter record."""
    spec = parse_einsum_spec(
        equation=params.get("equation"),
        input_shape=params.get("input_shape") or _first(params.get("input_shapes")),
        weight_shape=params.get("kernel_shape"),
        output_shape=params.get("output_shape"),
        layer_name=layer_name,
    )
    if spec is None:
        return {
            "einsum_spec": None,
            "fold_axes": None,
            "fold_axis_names": None,
            "local_equations": None,
        }
    fold_axis_names = detect_fold_axes(spec)
    fold_axes = detect_fold_axis_indices(spec)
    return {
        "einsum_spec": spec,
        "fold_axes": fold_axes or None,
        "fold_axis_names": fold_axis_names or None,
        "local_equations": {
            axis: derive_local_equation(spec, axis) for axis in fold_axis_names
        }
        or None,
    }


def _first(values):
    return values[0] if values else None
