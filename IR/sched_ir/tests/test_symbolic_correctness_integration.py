import importlib.util

import pytest


def _has(name):
    return importlib.util.find_spec(name) is not None


pytestmark = pytest.mark.skipif(
    not (_has("da4ml") and _has("keras")),
    reason="DA4ML/Keras integration dependencies are not installed",
)


def test_symbolic_correctness_smoke_for_small_model():
    import keras

    from IR.nn_ir.builder import build_nn_ir
    from IR.sched_ir import api

    inputs = keras.Input(shape=(4,), name="input")
    outputs = keras.layers.Dense(2, use_bias=False, name="dense")(inputs)
    model = keras.Model(inputs, outputs, name="small_dense")

    nn_graph = build_nn_ir(model)
    design = api.evaluate_folded_design(nn_graph, model=model, factor=1)
    report = api.check_symbolic_correctness(design, model=model)

    assert report.passed is True
