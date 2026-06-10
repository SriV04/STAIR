"""DA4ML symbolic primitive evaluation used by the sequential executor."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .attention_cost_oracle import estimate_attention_cost
from .records import PipelineEvalConfig, PrimitiveEvaluation, SymbolicTensorState
from .folded_layers import build_folded_layer_model


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _numel(shape: tuple[int, ...]) -> int:
    if any(dim is None for dim in shape):
        raise ValueError(f"Cannot create symbolic input for dynamic shape {shape}")
    return int(np.prod(shape))


def _latency_end(latency) -> float:
    if isinstance(latency, (tuple, list, np.ndarray)):
        return float(max(latency))
    return float(latency)


def _tensor_shape(tensor) -> tuple[int, ...]:
    return tuple(None if dim is None else int(dim) for dim in tuple(tensor.shape[1:]))


def _traced_output_shapes(symbolic_output) -> list[tuple[int, ...]]:
    return [
        tuple(np.asarray(value, dtype=object).shape)
        for value in _as_list(symbolic_output)
    ]


def _strip_batch_shape(shape) -> tuple[int, ...]:
    if shape is None:
        return ()
    dims = tuple(None if dim is None else int(dim) for dim in tuple(shape))
    if dims and dims[0] is None:
        dims = dims[1:]
    return dims


def _shape_numel_or_zero(shape) -> int:
    if not shape or any(dim is None for dim in shape):
        return 0
    return int(np.prod(shape))


def _repeat_precision(value, count: int) -> list:
    if value is None or count <= 0:
        return []
    return [value for _ in range(count)]


def _qintervals_from_summary(qint, shape) -> list:
    if qint is None:
        return []
    count = _shape_numel_or_zero(shape)
    if count <= 0:
        return []
    if hasattr(qint, "min") and hasattr(qint, "max") and hasattr(qint, "step"):
        return [qint for _ in range(count)]
    if not isinstance(qint, dict):
        return [qint for _ in range(count)]

    from da4ml.types import QInterval

    mins = np.array(qint.get("min"), dtype=float)
    maxs = np.array(qint.get("max"), dtype=float)
    steps = np.array(qint.get("step"), dtype=float)
    if mins.shape == ():
        return [QInterval(float(mins), float(maxs), float(steps)) for _ in range(count)]

    mins = np.broadcast_to(mins, shape).reshape(-1) if mins.size != count else mins.reshape(-1)
    maxs = np.broadcast_to(maxs, shape).reshape(-1) if maxs.size != count else maxs.reshape(-1)
    steps = np.broadcast_to(steps, shape).reshape(-1) if steps.size != count else steps.reshape(-1)
    return [
        QInterval(*_normalise_qinterval_values(float(min_v), float(max_v), float(step_v)))
        for min_v, max_v, step_v in zip(mins, maxs, steps, strict=True)
    ]


def _normalise_qinterval_values(min_v: float, max_v: float, step_v: float) -> tuple[float, float, float]:
    step_v = abs(step_v) if step_v else 1.0
    if max_v >= min_v:
        return min_v, max_v, step_v
    if min_v == 0 and max_v < 0:
        return 0.0, 0.0, step_v
    return max_v, min_v, step_v


def _qinterval_like(template, min_v: float, max_v: float, step_v: float):
    try:
        return template.__class__(min_v, max_v, step_v)
    except Exception:
        from da4ml.types import QInterval

        return QInterval(min_v, max_v, step_v)


def _merge_qintervals(qints) -> Any:
    qints = list(qints)
    if not qints:
        raise ValueError("Cannot merge an empty qint block")
    mins = [float(getattr(qint, "min")) for qint in qints]
    maxs = [float(getattr(qint, "max")) for qint in qints]
    steps = [abs(float(getattr(qint, "step"))) for qint in qints]
    return _qinterval_like(
        qints[0],
        min(mins),
        max(maxs),
        min(step for step in steps if step > 0) if any(step > 0 for step in steps) else 1.0,
    )


def _adapt_qints_for_shape(qints, source_shape, target_shape) -> list:
    source_shape = tuple(int(dim) for dim in source_shape)
    target_shape = tuple(int(dim) for dim in target_shape)
    flat_qints = list(qints)
    if source_shape == target_shape:
        return flat_qints
    if _numel(source_shape) != len(flat_qints):
        raise ValueError(
            f"Expected {source_shape} to contain {len(flat_qints)} qints, "
            f"got {_numel(source_shape)} elements"
        )
    if len(source_shape) == len(target_shape) and source_shape[1:] == target_shape[1:]:
        source_n = source_shape[0]
        target_n = target_shape[0]
        source_array = np.empty(source_shape, dtype=object)
        source_array.reshape(-1)[:] = flat_qints
        if target_n > source_n and target_n % source_n == 0:
            return np.repeat(source_array, target_n // source_n, axis=0).reshape(-1).tolist()
        if source_n > target_n and source_n % target_n == 0:
            fold_factor = source_n // target_n
            grouped = source_array.reshape(target_n, fold_factor, *source_shape[1:])
            folded = np.empty(target_shape, dtype=object)
            for index in np.ndindex(target_shape):
                folded[index] = _merge_qintervals(grouped[(index[0], slice(None), *index[1:])])
            return folded.reshape(-1).tolist()
    raise ValueError(f"Cannot adapt qints from shape {source_shape} to {target_shape}")


def _symbolic_input_from_state(state: SymbolicTensorState, target_shape, hwconf):
    qints = _adapt_qints_for_shape(state.qints, state.shape, target_shape)
    return qints_to_symbolic_input(
        qints,
        target_shape,
        hwconf,
        latency=state.latency,
    )


def _layer_input_shapes(layer, input_states) -> list[tuple[int, ...]]:
    try:
        shapes = [_tensor_shape(tensor) for tensor in _as_list(layer.input)]
    except Exception:
        shapes = []
    if len(shapes) != len(input_states):
        return [state.shape for state in input_states]
    return shapes


def _op_param_input_shapes(node_pmap, count: int) -> list[tuple[int, ...]]:
    params = (node_pmap or {}).get("op_params") or {}
    raw_shapes = params.get("input_shapes") or params.get("in_shapes")
    if raw_shapes is None and params.get("input_shape") is not None:
        raw_shapes = [params.get("input_shape")]
    if raw_shapes is None:
        return []
    shapes = [_strip_batch_shape(shape) for shape in _as_list(raw_shapes)]
    return shapes if len(shapes) == count else []


def reconcile_output_shapes(
    layer_name,
    input_states,
    keras_outputs,
    symbolic_output,
    output_qints,
) -> list[tuple[int, ...]]:
    """Preserve folded tensor layout while validating DA4ML's flat output precision."""
    keras_shapes = [_tensor_shape(tensor) for tensor in _as_list(keras_outputs)]
    n_qints = len(output_qints)
    if sum(_numel(shape) for shape in keras_shapes) == n_qints:
        return keras_shapes
    if len(keras_shapes) == 1 and len(keras_shapes[0]) >= 2:
        tail = keras_shapes[0][1:]
        tail_count = _numel(tail)
        if tail_count and n_qints % tail_count == 0:
            return [(n_qints // tail_count, *tail)]
    traced_shapes = _traced_output_shapes(symbolic_output)
    if sum(_numel(shape) for shape in traced_shapes) == n_qints:
        if (
            len(traced_shapes) == 1
            and len(traced_shapes[0]) == 1
            and input_states
            and n_qints == _numel(input_states[0].shape)
        ):
            return [input_states[0].shape]
        return traced_shapes
    raise ValueError(
        f"{layer_name} produced {n_qints} qints, but neither Keras shapes "
        f"{keras_shapes} nor DA4ML shapes {traced_shapes} partition them"
    )


def qints_to_symbolic_input(qints, shape, hwconf, latency=0):
    """Construct the next DA4ML input from predecessor combinational QIntervals."""
    from da4ml.trace import FixedVariableArray
    from da4ml.types import minimal_kif

    flat_qints = list(qints)
    expected = _numel(shape)
    if len(flat_qints) != expected:
        raise ValueError(f"Expected {expected} qints for shape {shape}, got {len(flat_qints)}")
    kif = np.array([minimal_kif(qint) for qint in flat_qints], dtype=np.int8)
    return FixedVariableArray.from_kif(
        kif[:, 0].reshape(shape),
        kif[:, 1].reshape(shape),
        kif[:, 2].reshape(shape),
        hwconf=hwconf,
        latency=latency,
    )


def _pipeline(comb, config: PipelineEvalConfig | dict | None):
    if isinstance(config, dict):
        config = config.get("pipeline_config")
    if config is None or not config.enabled:
        return None
    from da4ml.trace import to_pipeline

    return to_pipeline(
        comb,
        latency_cutoff=config.latency_cutoff,
        retiming=config.retiming,
        verbose=config.verbose,
    )


def _output_kifs(qints) -> list:
    try:
        from da4ml.types import minimal_kif

        return [minimal_kif(qint) for qint in qints]
    except Exception:
        return []


def primitive_from_comb(
    *,
    symbolic_inputs: list[Any],
    symbolic_outputs: list[Any],
    output_shapes: list[tuple[int, ...]],
    comb,
    pipeline,
    kernel_meta: dict | None = None,
) -> PrimitiveEvaluation:
    """Convert actual DA4ML combinational state into the backend contract."""
    evaluated = pipeline or comb
    raw_cost = getattr(evaluated, "cost", 0)
    reg_bits = getattr(evaluated, "reg_bits", getattr(comb, "reg_bits", 0))
    latency = getattr(evaluated, "latency", getattr(comb, "latency", 1))
    out_qints = list(getattr(comb, "out_qint", []) or [])
    stages = getattr(evaluated, "solutions", None)
    cost = {
        "lut": int(round(float(raw_cost))),
        "ff": int(reg_bits or 0),
        "dsp": 0,
        "bram": 0,
        "latency_cycles": max(1, int(math.ceil(_latency_end(latency)))),
        "ii": 1,
        "pipeline_stages": len(stages) if stages is not None else None,
    }
    return PrimitiveEvaluation(
        symbolic_inputs=symbolic_inputs,
        symbolic_outputs=symbolic_outputs,
        output_shapes=output_shapes,
        output_qints=out_qints,
        output_kifs=_output_kifs(out_qints),
        output_latency=_latency_end(getattr(comb, "latency", latency)),
        cost=cost,
        n_ops=len(getattr(comb, "ops", []) or []),
        comb_logic=comb,
        pipeline=pipeline,
        kernel_meta=kernel_meta or {},
    )


def synthetic_primitive_from_op(
    *,
    node_pmap,
    input_states: list[SymbolicTensorState],
    cost: dict | None = None,
    output_shape=None,
) -> PrimitiveEvaluation:
    """Create a dependency-preserving primitive for non-DA4ML attention ops."""
    params = node_pmap.get("op_params") or {}
    full_shape = output_shape or params.get("output_shape") or params.get("output_shapes")
    state_shape = _strip_batch_shape(full_shape)
    if not state_shape and input_states:
        state_shape = input_states[0].shape
    placeholder_shape = tuple(1 if dim is None else int(dim) for dim in state_shape)
    symbolic_output = np.empty(placeholder_shape, dtype=object)
    symbolic_output.fill(f"{node_pmap.get('nn_layer_name') or node_pmap.get('op')}:synthetic")

    output_qint = params.get("output_qint")
    output_kif = params.get("output_kif")
    count = _shape_numel_or_zero(state_shape)
    output_qints = _qintervals_from_summary(output_qint, state_shape)
    output_kifs = _repeat_precision(output_kif, count)
    if node_pmap.get("op") == "transport" and input_states:
        if not output_qints and len(input_states[0].qints) == count:
            output_qints = list(input_states[0].qints)
        if not output_qints:
            input_shapes = _op_param_input_shapes(node_pmap, 1)
            if input_shapes:
                adapted_qints = _adapt_qints_for_shape(
                    input_states[0].qints,
                    input_states[0].shape,
                    input_shapes[0],
                )
                if len(adapted_qints) == count:
                    output_qints = adapted_qints
        if not output_kifs and len(input_states[0].kifs) == count:
            output_kifs = list(input_states[0].kifs)
    default_cost = {
        "lut": 0,
        "ff": 0,
        "dsp": 0,
        "bram": 0,
        "latency_cycles": 1,
        "ii": 1,
        "pipeline_stages": None,
        "cost_mode": node_pmap.get("cost_mode") or params.get("cost_mode") or "synthetic_zero",
    }
    if cost:
        default_cost.update(cost)
    input_latency = max([state.latency for state in input_states] or [0.0])
    return PrimitiveEvaluation(
        symbolic_inputs=[state.symbolic_value for state in input_states],
        symbolic_outputs=[symbolic_output],
        output_shapes=[state_shape],
        output_qints=output_qints,
        output_kifs=output_kifs,
        output_latency=input_latency + float(default_cost["latency_cycles"]),
        cost=default_cost,
        n_ops=0,
        comb_logic=None,
        pipeline=None,
        kernel_meta={
            "op": node_pmap.get("op"),
            "cost_model": default_cost["cost_mode"],
            "synthetic": True,
        },
    )


def trace_layer(
    *,
    node_pmap=None,
    layer,
    input_states: list[SymbolicTensorState],
    config: dict | None,
) -> PrimitiveEvaluation:
    """Trace one model operation using evaluated predecessor precision."""

    config = config or {}
    hwconf = config.get("hwconf")
    verbose = bool(config.get("verbose", False))
    import keras
    from da4ml.converter import trace_model
    from da4ml.trace import comb_trace

    if input_states:
        expected_shapes = _op_param_input_shapes(node_pmap, len(input_states)) or _layer_input_shapes(
            layer,
            input_states,
        )
        keras_inputs = [
            keras.Input(shape=shape, name=f"{layer.name}_input_{index}")
            for index, shape in enumerate(expected_shapes)
        ]
        layer_arg = keras_inputs[0] if len(keras_inputs) == 1 else keras_inputs
        model = keras.Model(keras_inputs, layer(layer_arg), name=f"tmp_{layer.name}")
        symbolic_inputs = [
            _symbolic_input_from_state(state, shape, hwconf)
            for state, shape in zip(input_states, expected_shapes, strict=True)
        ]
        trace_inputs = symbolic_inputs[0] if len(symbolic_inputs) == 1 else tuple(symbolic_inputs)
        inp, out = trace_model(model, inputs=trace_inputs, verbose=verbose)
    else:
        model = keras.Model(layer.input, layer.output, name=f"tmp_{layer.name}")
        inp, out = trace_model(model, verbose=verbose)
        symbolic_inputs = _as_list(inp)
    comb = comb_trace(inp, out)
    symbolic_outputs = _as_list(out)
    output_shapes = reconcile_output_shapes(
        layer.name,
        input_states,
        model.outputs,
        out,
        comb.out_qint,
    )
    return primitive_from_comb(
        symbolic_inputs=symbolic_inputs,
        symbolic_outputs=symbolic_outputs,
        output_shapes=output_shapes,
        comb=comb,
        pipeline=_pipeline(comb, config),
        kernel_meta={"op": layer.__class__.__name__},
    )


def trace_folded_layer(
    *,
    node_pmap,
    layer,
    input_states: list[SymbolicTensorState],
    config,
) -> PrimitiveEvaluation:
    """Trace rebuilt folded hardware before sequential precision handoff."""

    config = config or {}
    from da4ml.converter import trace_model
    from da4ml.trace import comb_trace

    factor = int(node_pmap.get("temporal_steps_T") or 1)
    folded_model, weight_copy_log = build_folded_layer_model(
        layer,
        fold_factor=factor,
    )
    if input_states:
        expected_shapes = [_tensor_shape(tensor) for tensor in _as_list(folded_model.input)]
        symbolic_inputs = [
            _symbolic_input_from_state(state, shape, config.get("hwconf"))
            for state, shape in zip(input_states, expected_shapes, strict=True)
        ]
        trace_inputs = symbolic_inputs[0] if len(symbolic_inputs) == 1 else tuple(symbolic_inputs)
        inp, out = trace_model(
            folded_model,
            inputs=trace_inputs,
            verbose=bool(config.get("verbose", False)),
        )
    else:
        inp, out = trace_model(folded_model, verbose=bool(config.get("verbose", False)))
        symbolic_inputs = _as_list(inp)
    comb = comb_trace(inp, out)
    symbolic_outputs = _as_list(out)
    output_shapes = reconcile_output_shapes(
        folded_model.layers[-1].name,
        input_states,
        folded_model.outputs,
        out,
        comb.out_qint,
    )
    return primitive_from_comb(
        symbolic_inputs=symbolic_inputs,
        symbolic_outputs=symbolic_outputs,
        output_shapes=output_shapes,
        comb=comb,
        pipeline=_pipeline(comb, config),
        kernel_meta={
            "op": "dense",
            "cost_model": "da4ml_trace_model_folded_layer",
            "fold_factor": factor,
            "weight_copy_log": weight_copy_log,
        },
    )


def trace_folded_entry_layer(*, node_pmap, layer, config) -> PrimitiveEvaluation:
    """Trace rebuilt folded entry hardware before sequential precision handoff."""
    return trace_folded_layer(
        node_pmap=node_pmap,
        layer=layer,
        input_states=[],
        config=config,
    )


def trace_reduce(*, node_pmap, input_states, config) -> PrimitiveEvaluation:
    """Trace a Sched-IR reduction directly over evaluated symbolic inputs."""

    from da4ml.trace import comb_trace

    if len(input_states) != 1:
        raise ValueError(f"reduce expects one input state, got {len(input_states)}")
    config = config or {}
    symbolic_input = qints_to_symbolic_input(
        input_states[0].qints,
        input_states[0].shape,
        config.get("hwconf"),
        latency=input_states[0].latency,
    )
    params = node_pmap.get("op_params") or {}
    axes = tuple(int(axis) - 1 for axis in (params.get("axes") or []) if int(axis) >= 1)
    mode = (params.get("mode") or "sum").lower()
    keepdims = bool(params.get("keepdims"))
    if mode == "sum":
        symbolic_output = np.sum(symbolic_input, axis=axes, keepdims=keepdims)
    elif mode == "max":
        symbolic_output = np.amax(symbolic_input, axis=axes, keepdims=keepdims)
    elif mode == "min":
        symbolic_output = np.amin(symbolic_input, axis=axes, keepdims=keepdims)
    else:
        raise NotImplementedError(f"DA4ML direct reduction does not support mode {mode!r}")
    comb = comb_trace(list(np.ravel(symbolic_input)), list(np.ravel(symbolic_output)))
    return primitive_from_comb(
        symbolic_inputs=[symbolic_input],
        symbolic_outputs=[symbolic_output],
        output_shapes=[tuple(np.asarray(symbolic_output, dtype=object).shape)],
        comb=comb,
        pipeline=_pipeline(comb, config),
        kernel_meta={"op": "reduce", "mode": mode, "axes": list(axes)},
    )


def evaluate_node(node_pmap, *, input_states, keras_layer, config):
    """Evaluate one supported compute node through a DA4ML single-layer trace."""
    config = config or {}
    if node_pmap.get("op") in {"einsum", "softmax"}:
        return synthetic_primitive_from_op(
            node_pmap=node_pmap,
            input_states=input_states,
            cost=estimate_attention_cost(
                node_pmap,
                calibration=config.get("attention_cost_oracle"),
            ),
        )
    if node_pmap.get("op") == "transport":
        return synthetic_primitive_from_op(
            node_pmap=node_pmap,
            input_states=input_states,
        )
    if node_pmap.get("op") not in {"dense", "reduce", "elementwise", "activation"}:
        raise NotImplementedError(f"DA4ML backend does not evaluate {node_pmap.get('op')!r}")
    if node_pmap.get("op") == "reduce":
        return trace_reduce(
            node_pmap=node_pmap,
            input_states=input_states,
            config=config,
        )
    if keras_layer is None:
        raise ValueError(f"No Keras layer supplied for node {node_pmap.get('nn_layer_name')!r}")
    if (
        node_pmap.get("op") == "dense"
        and not input_states
        and int(node_pmap.get("temporal_steps_T") or 1) > 1
    ):
        return trace_folded_entry_layer(
            node_pmap=node_pmap,
            layer=keras_layer,
            config=config,
        )
    if node_pmap.get("op") in {"dense", "elementwise"} and int(node_pmap.get("temporal_steps_T") or 1) > 1:
        return trace_folded_layer(
            node_pmap=node_pmap,
            layer=keras_layer,
            input_states=input_states,
            config=config,
        )
    return trace_layer(
        node_pmap=node_pmap,
        layer=keras_layer,
        input_states=input_states,
        config=config,
    )
