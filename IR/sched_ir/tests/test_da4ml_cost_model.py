import numpy as np

from IR.sched_ir.backends.da4ml.cost_model import (
    evaluate_node,
    primitive_from_comb,
    reconcile_output_shapes,
)


class FakeComb:
    cost = 11
    reg_bits = 7
    shape = (1,)
    latency = (0, 2)
    ops = [object(), object()]
    out_qint = ["out_qint"]


class FakeTensor:
    shape = (None, 4, 64)


def test_primitive_from_comb_retains_actual_symbolic_inputs_and_outputs():
    symbolic_input = object()
    symbolic_output = object()

    result = primitive_from_comb(
        symbolic_inputs=[symbolic_input],
        symbolic_outputs=[symbolic_output],
        output_shapes=[(1,)],
        comb=FakeComb(),
        pipeline=None,
    )

    assert result.symbolic_inputs == [symbolic_input]
    assert result.symbolic_outputs == [symbolic_output]
    assert result.output_qints == ["out_qint"]
    assert result.cost["lut"] == 11
    assert result.cost["ff"] == 7


def test_evaluate_node_dispatches_to_trace_layer_for_supported_compute(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        "IR.sched_ir.backends.da4ml.cost_model.trace_layer",
        lambda **kwargs: expected,
    )

    result = evaluate_node(
        {"op": "dense"},
        input_states=[],
        keras_layer=object(),
        config={},
    )

    assert result is expected


def test_folded_entry_dense_is_materialised_before_tracing(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        "IR.sched_ir.backends.da4ml.cost_model.trace_folded_entry_layer",
        lambda **kwargs: expected,
    )

    result = evaluate_node(
        {"op": "dense", "temporal_steps_T": 2, "fold_axes": [1]},
        input_states=[],
        keras_layer=object(),
        config={},
    )

    assert result is expected


def test_reduce_node_traces_sched_ir_operation_without_calling_original_hgq_layer(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        "IR.sched_ir.backends.da4ml.cost_model.trace_reduce",
        lambda **kwargs: expected,
        raising=False,
    )

    result = evaluate_node(
        {"op": "reduce", "op_params": {"axes": [1]}},
        input_states=[object()],
        keras_layer=object(),
        config={},
    )

    assert result is expected


def test_reconcile_output_shapes_preserves_folded_keras_layout_for_flat_da4ml_output():
    shape = reconcile_output_shapes(
        "folded_dense",
        [],
        [FakeTensor()],
        np.empty((256,), dtype=object),
        [object()] * 256,
    )

    assert shape == [(4, 64)]
