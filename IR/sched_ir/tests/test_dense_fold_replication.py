from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO))


class FakeHGraph:
    def __init__(self, *, vinit=None, einit=None, ginit=None):
        self._vinit = vinit
        self._einit = einit
        self._ginit = ginit
        self.vertices = []
        self.edges = []
        self.pmap = {}
        if self._ginit is not None:
            self._ginit(self)

    def add_vx(self):
        vx = len(self.vertices)
        self.vertices.append(vx)
        if self._vinit is not None:
            self._vinit(self, vx)
        return vx

    def add_edge(self, src, dst):
        edge = (src, dst)
        self.edges.append(edge)
        if self._einit is not None:
            self._einit(self, edge)
        return [edge]

    def in_vx(self, vx):
        return [src for src, dst in self.edges if dst == vx]

    def out_vx(self, vx):
        return [dst for src, dst in self.edges if src == vx]


fake_heterograph = ModuleType("heterograph")
fake_heterograph.HGraph = FakeHGraph
sys.modules["heterograph"] = fake_heterograph
try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    fake_yaml = ModuleType("yaml")
    fake_yaml.safe_load = lambda *_args, **_kwargs: {}
    sys.modules["yaml"] = fake_yaml

from IR.sched_ir import schema  # noqa: E402
from IR.sched_ir.costing import kernels  # noqa: E402
from IR.sched_ir.scheduling import infrastructure  # noqa: E402


class FakeVar:
    def __init__(self, name: str, value):
        self.name = name
        self._value = np.asarray(value)
        self.shape = self._value.shape

    def numpy(self):
        return self._value


class FakeLayer:
    def __init__(self, weights):
        self.weights = weights
        self.received = None

    def set_weights(self, weights):
        self.received = [np.asarray(w) for w in weights]


class FakeWeights:
    def __init__(self, layer):
        self.layer = layer

    def get_kernel(self, layer_name):
        return np.ones((3, 2), dtype=np.float32)

    def get_layer(self, layer_name):
        return self.layer


class DenseFoldReplicationTests(unittest.TestCase):
    def test_copy_weights_matches_names_and_spatially_folds_3d_metadata(self):
        old_spatial = np.arange(1 * 8 * 3).reshape(1, 8, 3)
        old = FakeLayer(
            [
                FakeVar("kernel", np.ones((3, 2))),
                FakeVar("k", old_spatial),
                FakeVar("dup", np.array([1])),
                FakeVar("dup", np.array([2])),
            ]
        )
        new = FakeLayer(
            [
                FakeVar("kernel", np.zeros((3, 2))),
                FakeVar("k", np.zeros((1, 4, 3))),
                FakeVar("dup", np.array([0])),
                FakeVar("dup", np.array([0])),
                FakeVar("missing", np.array([7])),
            ]
        )

        log = kernels.copy_folded_layer_weights_by_name(old, new, fold_factor=2)

        expected_folded = old_spatial.reshape(1, 4, 2, 3).max(axis=2)
        self.assertEqual(log[1]["transform"], "fold_spatial_max")
        np.testing.assert_array_equal(new.received[0], np.ones((3, 2)))
        np.testing.assert_array_equal(new.received[1], expected_folded)
        np.testing.assert_array_equal(new.received[2], np.array([1]))
        np.testing.assert_array_equal(new.received[3], np.array([2]))
        np.testing.assert_array_equal(new.received[4], np.array([7]))

    def test_folded_dense_cost_uses_replicated_layer_granularity(self):
        layer = object()
        calls = []

        def fake_folded_result(p, layer_arg, fpga):
            calls.append((p, layer_arg, fpga))
            return {
                "cost": {
                    "lut": 11,
                    "ff": 3,
                    "dsp": 0,
                    "bram": 0,
                    "latency_cycles": 2,
                    "ii": 1,
                },
                "input_qints": None,
                "input_kifs": None,
                "output_qints": None,
                "output_kifs": None,
                "input_tensor_width_bits": None,
                "output_tensor_width_bits": None,
                "precision_source": "da4ml",
                "kernel_meta": {"dense_cost_granularity": "folded_replicated_layer"},
            }

        prev = kernels._folded_dense_layer_result
        try:
            kernels._folded_dense_layer_result = fake_folded_result
            p = {
                "op": "dense",
                "nn_layer_name": "dense_0",
                "fold_axes": [1],
                "parallelism_N": 8,
                "lanes_P": 4,
                "temporal_steps_T": 2,
                "op_params": {
                    "input_shape": (None, 8, 3),
                    "output_shape": (None, 8, 2),
                },
            }

            result = kernels.da4ml_dense_cost(p, FakeWeights(layer), {"latency_cutoff": 2})
        finally:
            kernels._folded_dense_layer_result = prev

        self.assertEqual(calls[0][1], layer)
        self.assertEqual(result["kernel_meta"]["dense_cost_granularity"], "folded_replicated_layer")

        g = FakeHGraph(vinit=schema.vinit_sched, einit=schema.einit_sched, ginit=schema.ginit_sched)
        vx = g.add_vx()
        g.pmap[vx].update(
            {
                "op": "dense",
                "physical_instances": 4,
                "cost": result["cost"],
                "kernel_result": result,
            }
        )
        infrastructure._rollup(g)

        self.assertEqual(g.pmap["total_luts"], 11)
        self.assertEqual(g.pmap["total_ffs"], 3)


if __name__ == "__main__":
    unittest.main()
