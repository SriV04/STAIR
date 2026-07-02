"""End-to-end smoke test: build -> fold -> schedule -> symbolic correctness."""

import importlib.util

import pytest


def _has(name):
    return importlib.util.find_spec(name) is not None


pytestmark = pytest.mark.skipif(
    not (_has("da4ml") and _has("keras") and _has("hgq")),
    reason="DA4ML/Keras/HGQ integration dependencies are not installed",
)


def test_symbolic_correctness_smoke_for_small_model():
    import keras
    from hgq.layers import QDense

    from IR.nn_ir.builder import build_nn_ir
    from IR.sched_ir import api

    # The NN-IR builder only lowers HGQ2 quantized layers, so the smoke model
    # must use QDense rather than a plain keras Dense.
    inputs = keras.Input(shape=(4,), name="input")
    outputs = QDense(2, use_bias=False, name="dense")(inputs)
    model = keras.Model(inputs, outputs, name="small_dense")

    nn_graph = build_nn_ir(model)
    design = api.evaluate_folded_design(nn_graph, model=model, factor=1)
    report = api.check_symbolic_correctness(design, model=model)

    assert report.passed is True
