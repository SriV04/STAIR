from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


HERE = Path(__file__).resolve().parent


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
        if edge in self.edges:
            return []
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
from IR.sched_ir import binder
from IR.sched_ir import schema


class _Model:
    def get_layer(self, layer_name):
        raise LookupError(layer_name)


class FakeQIntervalType:
    @staticmethod
    def from_kif(k, i, f):
        return ("from_kif", k, i, f)

    def __new__(cls, qmin, qmax, step):
        return ("from_dict", qmin, qmax, step)


class BindKernelResultTests(unittest.TestCase):
    def _write_resource_yaml(self, cost_query_name: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(
            f"""
fpga:
  device: VU13P
  latency_cutoff: 2
kernels:
  fake_dense:
    supported_ops: [dense]
    constraints:
      weight_source: constant
    instances: unlimited
    cost_query: {cost_query_name}
"""
        )
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def _build_graph(self) -> FakeHGraph:
        g = FakeHGraph(vinit=schema.vinit_sched, einit=schema.einit_sched, ginit=schema.ginit_sched)
        vx = g.add_vx()
        p = g.pmap[vx]
        p["op"] = "dense"
        p["nn_layer_name"] = "dense_0"
        p["inserted_by"] = "decomposer"
        p["precision_source"] = "hgq"
        p["output_tensor_width_bits"] = 4
        p["op_params"] = {
            "in_bw": 4.0,
            "input_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "input_kif": {"k": True, "i": 2, "f": 1, "bits": 4},
            "output_qint": {"min": -1.0, "max": 1.0, "step": 0.5},
            "output_kif": {"k": True, "i": 2, "f": 1, "bits": 4},
            "out_bw": 4.0,
        }
        return g

    def test_bind_stores_full_kernel_result_and_updates_output_precision(self):
        def _fake_full_result(p, weights, fpga):
            return {
                "cost": {
                    "lut": 17,
                    "ff": 9,
                    "dsp": 0,
                    "bram": 0,
                    "latency_cycles": 3,
                    "ii": 1,
                },
                "input_qints": [{"min": -1.0, "max": 1.0, "step": 0.5}],
                "input_kifs": [{"k": True, "i": 2, "f": 1, "bits": 4}],
                "output_qints": [{"min": -4.0, "max": 3.0, "step": 1.0}],
                "output_kifs": [{"k": True, "i": 3, "f": 0, "bits": 4}],
                "input_tensor_width_bits": 4,
                "output_tensor_width_bits": 4,
                "precision_source": "da4ml",
                "da4ml": {"solution_type": "FakePipeline"},
            }

        binder.REGISTRY["fake_dense_result"] = _fake_full_result
        cfg = self._write_resource_yaml("fake_dense_result")
        g = self._build_graph()

        out = binder.bind(g, _Model(), cfg)
        p = out.pmap[0]

        self.assertEqual(p["kernel_type"], "fake_dense")
        self.assertEqual(p["kernel_instance"], 0)
        self.assertEqual(p["cost"]["lut"], 17)
        self.assertEqual(p["cost"]["latency_cycles"], 3)
        self.assertEqual(p["precision_source"], "da4ml")
        self.assertEqual(p["kernel_result"]["da4ml"]["solution_type"], "FakePipeline")
        self.assertEqual(p["output_qints"][0]["step"], 1.0)
        self.assertEqual(p["output_kifs"][0]["bits"], 4)
        self.assertEqual(p["output_tensor_width_bits"], 4)
        self.assertEqual(p["op_params"]["output_qint"]["step"], 1.0)
        self.assertEqual(p["op_params"]["output_kif"]["bits"], 4)
        self.assertEqual(p["op_params"]["out_bw"], 4)

    def test_bind_normalizes_legacy_cost_dict(self):
        def _fake_cost_only(p, weights, fpga):
            return {
                "lut": 5,
                "ff": 2,
                "dsp": 0,
                "bram": 0,
                "latency_cycles": 1,
                "ii": 1,
            }

        binder.REGISTRY["fake_cost_only"] = _fake_cost_only
        cfg = self._write_resource_yaml("fake_cost_only")
        g = self._build_graph()

        out = binder.bind(g, _Model(), cfg)
        p = out.pmap[0]

        self.assertEqual(p["cost"]["lut"], 5)
        self.assertEqual(p["kernel_result"]["cost"]["ff"], 2)
        self.assertEqual(p["kernel_result"]["precision_source"], "closed_form")
        self.assertIsNone(p["kernel_result"]["output_qints"])
        self.assertEqual(p["precision_source"], "closed_form")

    def test_bind_and_propagate_feeds_da4ml_output_precision_to_next_dense(self):
        cfg = self._write_resource_yaml("da4ml_dense_cost")
        g = FakeHGraph(vinit=schema.vinit_sched, einit=schema.einit_sched, ginit=schema.ginit_sched)
        dense0 = g.add_vx()
        dense1 = g.add_vx()
        g.add_edge(dense0, dense1)

        p0 = g.pmap[dense0]
        p0["op"] = "dense"
        p0["nn_layer_name"] = "dense_0"
        p0["inserted_by"] = "decomposer"
        p0["op_params"] = {
            "qkernel_values": np.ones((3, 2), dtype=np.float32),
            "input_qint": {
                "min": np.array([[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
                "max": np.array([[[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]]),
                "step": np.array([[[0.25, 0.25, 0.25], [0.125, 0.25, 0.5]]]),
            },
            "input_kif": {"k": True, "i": 1, "f": 2, "bits": 4},
            "in_bw": 4.0,
        }

        stale_hgq_qint = {
            "min": np.full((1, 8, 2), -9.0),
            "max": np.full((1, 8, 2), 9.0),
            "step": np.full((1, 8, 2), 1.0),
        }
        p1 = g.pmap[dense1]
        p1["op"] = "dense"
        p1["nn_layer_name"] = "dense_1"
        p1["inserted_by"] = "decomposer"
        p1["op_params"] = {
            "qkernel_values": np.ones((2, 1), dtype=np.float32),
            "input_qint": stale_hgq_qint,
            "input_kif": {"k": True, "i": 4, "f": 0, "bits": 5},
            "in_bw": 5.0,
        }

        edge_p = g.pmap[(dense0, dense1)]
        edge_p["tensor_shape"] = (None, 2)

        prev_ok = binder._kernels._da4ml._DA4ML_OK
        prev_qinterval = binder._kernels._da4ml.QInterval
        prev_solve = binder._kernels._da4ml.solve_dense_result
        calls = []

        def _fake_qint_to_dict(q):
            if isinstance(q, tuple) and len(q) == 4 and q[0] == "from_dict":
                return {"min": q[1], "max": q[2], "step": q[3]}
            return binder._kernels._da4ml.qint_to_dict(q)

        def _fake_solve(kernel, *, input_qints, **kwargs):
            calls.append(input_qints)
            out_count = int(np.asarray(kernel).shape[1])
            output_qints = [
                {"min": -float(idx + 1), "max": float(idx + 1), "step": 0.25}
                for idx in range(out_count)
            ]
            output_kifs = [
                {"k": True, "i": idx + 1, "f": 2, "bits": idx + 4}
                for idx in range(out_count)
            ]
            return {
                "cost": {
                    "lut": 10 + out_count,
                    "ff": 5 + out_count,
                    "dsp": 0,
                    "bram": 0,
                    "latency_cycles": 1,
                    "ii": 1,
                },
                "input_qints": [_fake_qint_to_dict(q) for q in input_qints],
                "input_kifs": None,
                "output_qints": output_qints,
                "output_kifs": output_kifs,
                "input_tensor_width_bits": None,
                "output_tensor_width_bits": sum(k["bits"] for k in output_kifs),
                "precision_source": "da4ml",
            }

        try:
            binder._kernels._da4ml._DA4ML_OK = True
            binder._kernels._da4ml.QInterval = FakeQIntervalType
            binder._kernels._da4ml.solve_dense_result = _fake_solve
            out = binder.bind_and_propagate(g, _Model(), cfg)
        finally:
            binder._kernels._da4ml._DA4ML_OK = prev_ok
            binder._kernels._da4ml.QInterval = prev_qinterval
            binder._kernels._da4ml.solve_dense_result = prev_solve

        dense0_p = out.pmap[dense0]
        dense1_p = out.pmap[dense1]
        edge_p = out.pmap[(dense0, dense1)]

        self.assertTrue(dense0_p["kernel_result"]["kernel_meta"]["input_precision_collapse"]["widened"])
        self.assertEqual(edge_p["src_qint"], dense0_p["output_qints"])
        self.assertEqual(dense1_p["op_params"]["input_qint"], edge_p["src_qint"])
        self.assertNotEqual(dense1_p["op_params"]["input_qint"], stale_hgq_qint)
        self.assertEqual(calls[1], [("from_dict", -1.0, 1.0, 0.25), ("from_dict", -2.0, 2.0, 0.25)])


if __name__ == "__main__":
    unittest.main()
