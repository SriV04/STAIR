import numpy as np
import pytest

from IR.sched_ir.backends.da4ml.folded_layers import (
    copy_folded_layer_weights,
    fold_spatial_max,
)


class FakeVar:
    def __init__(self, name, path, value):
        self.name = name
        self.path = path
        self._value = np.asarray(value)

    def numpy(self):
        return self._value


class FakeLayer:
    def __init__(self, name, weights):
        self.name = name
        self.weights = weights
        self.received = None

    def set_weights(self, weights):
        self.received = [np.asarray(value) for value in weights]


def test_fold_spatial_max_matches_notebook_precision_collapse():
    source = np.arange(1 * 8 * 3).reshape(1, 8, 3)

    folded = fold_spatial_max(source, fold_factor=2)

    np.testing.assert_array_equal(
        folded,
        source.reshape(1, 4, 2, 3).max(axis=2),
    )


def test_copy_folded_layer_weights_matches_relative_quantizer_path():
    source = FakeLayer(
        "dense",
        [
            FakeVar("k", "dense/input_quantizer/k", np.arange(24).reshape(1, 8, 3)),
            FakeVar("k", "dense/output_quantizer/k", np.full((1, 8, 3), 99)),
        ],
    )
    folded = FakeLayer(
        "dense_folded",
        [FakeVar("k", "dense_folded/input_quantizer/k", np.zeros((1, 4, 3)))],
    )

    log = copy_folded_layer_weights(source, folded, fold_factor=2)

    assert log[0]["transform"] == "fold_spatial_max"
    np.testing.assert_array_equal(
        folded.received[0],
        np.arange(24).reshape(1, 4, 2, 3).max(axis=2),
    )


def test_copy_folded_layer_weights_rejects_ambiguous_equal_name_matches():
    source = FakeLayer(
        "dense",
        [
            FakeVar("k", "dense/a/k", np.ones((1, 8, 3))),
            FakeVar("k", "dense/b/k", np.ones((1, 8, 3))),
        ],
    )
    folded = FakeLayer(
        "dense_folded",
        [FakeVar("k", "dense_folded/c/k", np.zeros((1, 4, 3)))],
    )

    with pytest.raises(ValueError, match="Ambiguous folded weight copy"):
        copy_folded_layer_weights(source, folded, fold_factor=2)
