import numpy as np
import pytest

from IR.sched_ir.backends.da4ml.folded_layers import (
    copy_folded_layer_weights,
    fold_spatial_max,
    standard_dense_equation,
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


def test_standard_dense_equation_matches_qdense_last_axis_contract():
    assert standard_dense_equation((4,), (2,)) == "bc,cC->bC"
    assert standard_dense_equation((8, 3), (8, 64)) == "bnc,cC->bnC"


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


def test_copy_folded_layer_weights_copies_ambiguous_equal_scalar_state():
    source = FakeLayer(
        "dense",
        [
            FakeVar("i_decay_speed", None, np.array(0.5)),
            FakeVar("i_decay_speed", None, np.array(0.5)),
        ],
    )
    folded = FakeLayer(
        "dense_folded",
        [FakeVar("i_decay_speed", None, np.array(0.0))],
    )

    log = copy_folded_layer_weights(source, folded, fold_factor=2)

    assert log[0]["transform"] == "copy_equal_scalar"
    assert log[0]["old_index"] is None
    np.testing.assert_array_equal(folded.received[0], np.array(0.5))


def test_copy_folded_layer_weights_matches_scalar_quantizer_role_before_value():
    source = FakeLayer(
        "dense",
        [
            FakeVar(
                "i_decay_speed",
                "dense/dense_iq/fixed_point_quantizer_kif/i_decay_speed",
                np.array(0.01),
            ),
            FakeVar(
                "i_decay_speed",
                "dense/dense_kq/fixed_point_quantizer_kbi/i_decay_speed",
                np.array(np.inf),
            ),
            FakeVar(
                "i_decay_speed",
                "dense/dense_bq/fixed_point_quantizer_kbi_1/i_decay_speed",
                np.array(0.2),
            ),
        ],
    )
    folded = FakeLayer(
        "dense_folded",
        [
            FakeVar(
                "i_decay_speed",
                "dense_folded/dense_folded_bq/fixed_point_quantizer_kbi_28/i_decay_speed",
                np.array(0.01),
            )
        ],
    )

    log = copy_folded_layer_weights(source, folded, fold_factor=2)

    assert log[0]["transform"] == "copy"
    assert log[0]["old_index"] == 2
    np.testing.assert_array_equal(folded.received[0], np.array(0.2))


def test_copy_folded_layer_weights_matches_multi_input_quantizer_index():
    source = FakeLayer(
        "add",
        [
            FakeVar(
                "f",
                "add/multiple_quantizers/quantizer/fixed_point_quantizer_kif_1/f",
                np.ones((1, 8, 3)),
            ),
            FakeVar(
                "f",
                "add/multiple_quantizers/quantizer_1/fixed_point_quantizer_kif_2/f",
                np.full((1, 8, 3), 2),
            ),
        ],
    )
    folded = FakeLayer(
        "add_folded",
        [
            FakeVar(
                "f",
                "add_folded/multiple_quantizers_3/quantizer_6/fixed_point_quantizer_kif_8/f",
                np.zeros((1, 4, 3)),
            ),
            FakeVar(
                "f",
                "add_folded/multiple_quantizers_3/quantizer_7/fixed_point_quantizer_kif_9/f",
                np.zeros((1, 4, 3)),
            ),
        ],
    )

    log = copy_folded_layer_weights(source, folded, fold_factor=2)

    assert log[0]["old_index"] == 0
    assert log[1]["old_index"] == 1
    np.testing.assert_array_equal(folded.received[1], np.full((1, 4, 3), 2))
