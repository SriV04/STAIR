"""Unfolded DA4ML reference tracing for correctness checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .records import ReferenceTrace


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _trace_model(model, verbose=False):
    from da4ml.converter import trace_model

    return trace_model(model, verbose=verbose)


def _comb_trace(inputs, outputs):
    from da4ml.trace import comb_trace

    return comb_trace(inputs, outputs)


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            return tuple(np.asarray(value, dtype=object).shape)
        except Exception:
            return None
    return tuple(int(dim) for dim in shape if dim is not None)


def trace_unfolded_reference(model, *, config: dict | None = None) -> ReferenceTrace:
    config = config or {}
    symbolic_inputs, symbolic_outputs = _trace_model(
        model,
        verbose=bool(config.get("verbose", False)),
    )
    inputs = _as_list(symbolic_inputs)
    outputs = _as_list(symbolic_outputs)
    comb = _comb_trace(symbolic_inputs, symbolic_outputs)
    return ReferenceTrace(
        symbolic_inputs=inputs,
        symbolic_outputs=outputs,
        output_qints=list(getattr(comb, "out_qint", []) or []),
        output_shapes=[
            shape
            for shape in (_shape_of(output) for output in outputs)
            if shape is not None
        ],
        comb_logic=comb,
        metadata={"model_name": getattr(model, "name", None)},
    )
