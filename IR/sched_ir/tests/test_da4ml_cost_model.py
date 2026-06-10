import sys
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from IR.sched_ir.backends.da4ml.cost_model import (
    _adapt_qints_for_shape,
    _op_param_input_shapes,
    _qintervals_from_summary,
    evaluate_node,
    primitive_from_comb,
    reconcile_output_shapes,
    synthetic_primitive_from_op,
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


@dataclass
class FakeQInterval:
    min: float
    max: float
    step: float


@dataclass
class FakeIterableQInterval:
    min: float
    max: float
    step: float

    def __iter__(self):
        yield self.min
        yield self.max
        yield self.step


def _install_fake_da4ml_qinterval(monkeypatch):
    fake_da4ml = ModuleType("da4ml")
    fake_types = ModuleType("da4ml.types")
    fake_types.QInterval = FakeQInterval
    monkeypatch.setitem(sys.modules, "da4ml", fake_da4ml)
    monkeypatch.setitem(sys.modules, "da4ml.types", fake_types)


def test_qinterval_summary_normalises_reversed_negative_ranges_for_da4ml(monkeypatch):
    _install_fake_da4ml_qinterval(monkeypatch)

    qints = _qintervals_from_summary(
        {
            "min": [-0.02518070101176188],
            "max": [-0.1817974671035733],
            "step": [0.20697816811533518],
        },
        (1,),
    )

    assert qints[0].min == -0.1817974671035733
    assert qints[0].max == -0.02518070101176188


def test_qinterval_summary_collapses_unsigned_impossible_range_to_zero(monkeypatch):
    _install_fake_da4ml_qinterval(monkeypatch)

    qints = _qintervals_from_summary(
        {
            "min": [0],
            "max": [-1],
            "step": [1],
        },
        (1,),
    )

    assert qints[0].min == 0
    assert qints[0].max == 0
    assert qints[0].step == 1


def test_adapt_qints_expands_folded_state_to_full_consumer_shape():
    qints = [
        FakeQInterval(0, 1, 0.5),
        FakeQInterval(1, 2, 0.25),
    ]

    adapted = _adapt_qints_for_shape(qints, (2, 1), (4, 1))

    assert [qint.min for qint in adapted] == [0, 0, 1, 1]
    assert [qint.max for qint in adapted] == [1, 1, 2, 2]


def test_adapt_qints_treats_iterable_qintervals_as_scalar_objects():
    qints = [
        FakeIterableQInterval(0, 1, 0.5),
        FakeIterableQInterval(1, 2, 0.25),
    ]

    adapted = _adapt_qints_for_shape(qints, (2, 1), (4, 1))

    assert [qint.min for qint in adapted] == [0, 0, 1, 1]


def test_adapt_qints_folds_full_state_to_folded_consumer_shape():
    qints = [
        FakeQInterval(0, 1, 0.5),
        FakeQInterval(-2, 3, 0.25),
        FakeQInterval(4, 5, 0.125),
        FakeQInterval(6, 9, 1.0),
    ]

    adapted = _adapt_qints_for_shape(qints, (4, 1), (2, 1))

    assert [(qint.min, qint.max, qint.step) for qint in adapted] == [
        (-2, 3, 0.25),
        (4, 9, 0.125),
    ]


def test_op_param_input_shapes_strip_batch_dimension():
    shapes = _op_param_input_shapes(
        {"op_params": {"input_shape": (None, 32, 16)}},
        1,
    )

    assert shapes == [(32, 16)]


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


def test_transport_expands_folded_precision_before_flattening():
    state = type(
        "State",
        (),
        {
            "shape": (2, 1),
            "qints": [FakeQInterval(0, 1, 0.5), FakeQInterval(1, 2, 0.25)],
            "kifs": [],
            "latency": 3.0,
            "symbolic_value": object(),
        },
    )()

    result = synthetic_primitive_from_op(
        node_pmap={
            "op": "transport",
            "op_params": {
                "input_shape": (None, 4, 1),
                "output_shape": (None, 4),
            },
        },
        input_states=[state],
    )

    assert result.output_shapes == [(4,)]
    assert [qint.min for qint in result.output_qints] == [0, 0, 1, 1]


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


def test_folded_dense_with_predecessor_state_is_materialised_before_tracing(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        "IR.sched_ir.backends.da4ml.cost_model.trace_folded_layer",
        lambda **kwargs: expected,
        raising=False,
    )

    result = evaluate_node(
        {"op": "dense", "temporal_steps_T": 2, "fold_axes": [1]},
        input_states=[object()],
        keras_layer=object(),
        config={},
    )

    assert result is expected


def test_folded_elementwise_with_predecessor_state_is_materialised_before_tracing(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        "IR.sched_ir.backends.da4ml.cost_model.trace_folded_layer",
        lambda **kwargs: expected,
        raising=False,
    )

    result = evaluate_node(
        {"op": "elementwise", "temporal_steps_T": 2, "fold_axes": [1]},
        input_states=[object(), object()],
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


def test_evaluate_node_uses_attention_oracle_for_einsum():
    result = evaluate_node(
        {
            "op": "einsum",
            "op_params": {
                "equation": "bshd,bthd->bhst",
                "input_shapes": [(None, 8, 1, 16), (None, 2, 1, 16)],
                "output_shape": (None, 1, 8, 2),
                "input_kifs": [{"bits": 6}, {"bits": 5}],
                "output_kif": {"bits": 12},
            },
        },
        input_states=[],
        keras_layer=None,
        config={},
    )

    assert result.cost["cost_mode"] == "analytic_einsum_attention_oracle"
    assert result.cost["einsum_multiply_count"] == 256
    assert result.cost["lut"] > 0


def test_evaluate_node_uses_attention_oracle_for_softmax():
    result = evaluate_node(
        {
            "op": "softmax",
            "op_params": {
                "axis": (-1,),
                "input_shape": (None, 1, 8, 2),
                "output_shape": (None, 1, 8, 2),
                "input_kif": {"bits": 10},
                "output_kif": {"bits": 8},
            },
        },
        input_states=[],
        keras_layer=None,
        config={},
    )

    assert result.cost["cost_mode"] == "analytic_softmax_attention_oracle"
    assert result.cost["softmax_element_count"] == 16
    assert result.cost["lut"] > 0


def test_evaluate_node_uses_hls4ml_reported_ii_from_config():
    result = evaluate_node(
        {
            "op": "softmax",
            "nn_layer_name": "mha1_softmax",
            "op_params": {
                "axis": (-1,),
                "input_shape": (None, 1, 8, 2),
                "output_shape": (None, 1, 8, 2),
            },
        },
        input_states=[],
        keras_layer=None,
        config={
            "attention_cost_oracle": {
                "hls4ml_reports": [
                    {
                        "op": "softmax",
                        "layer_name": "mha1_softmax",
                        "cost": {
                            "lut": 70,
                            "ff": 20,
                            "dsp": 0,
                            "bram": 1,
                            "latency_cycles": 11,
                            "ii": 3,
                        },
                    }
                ]
            }
        },
    )

    assert result.cost["cost_mode"] == "hls4ml_report_calibrated"
    assert result.cost["latency_cycles"] == 11
    assert result.cost["ii"] == 3
    assert result.cost["lut"] == 70


def test_reconcile_output_shapes_preserves_folded_keras_layout_for_flat_da4ml_output():
    shape = reconcile_output_shapes(
        "folded_dense",
        [],
        [FakeTensor()],
        np.empty((256,), dtype=object),
        [object()] * 256,
    )

    assert shape == [(4, 64)]
