from __future__ import annotations

import sys
from types import ModuleType

import pytest

from IR.sched_ir import schema


class FakeHGraph:
    def __init__(self, *, vinit=None, einit=None, ginit=None):
        self.vertices = []
        self.edges = []
        self.pmap = {}
        self.vstyle = {}
        self.estyle = {}
        self._vinit = vinit
        self._einit = einit
        if ginit is not None:
            ginit(self)

    @property
    def num_vx(self):
        return len(self.vertices)

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


@pytest.fixture
def build_dense_reduce_graph():
    def build(n=8):
        graph = FakeHGraph(
            vinit=schema.vinit_sched,
            einit=schema.einit_sched,
            ginit=schema.ginit_sched,
        )
        dense = graph.add_vx()
        reduction = graph.add_vx()
        graph.pmap[dense].update(
            {"op": "dense", "fold_axes": [1], "nn_layer_name": "dense"}
        )
        graph.pmap[reduction].update(
            {
                "op": "reduce",
                "fold_axes": [1],
                "nn_layer_name": "reduce",
                "op_params": {"axes": [1], "in_shape": (None, n, 4), "mode": "sum"},
                "reduce_mode": "spatial",
            }
        )
        edge = graph.add_edge(dense, reduction)[0]
        graph.pmap[edge]["tensor_shape"] = (None, n, 4)
        return graph, dense, reduction

    return build


@pytest.fixture
def evaluated_dense_reduce_graph(build_dense_reduce_graph):
    graph, dense, reduction = build_dense_reduce_graph()
    for node in (dense, reduction):
        graph.pmap[node].update(
            {
                "temporal_steps_T": 2,
                "parallelism_N": 8,
                "lanes_P": 4,
                "cost": {"lut": 10 + node, "ff": 2, "latency_cycles": 2, "ii": 1},
                "backend": "da4ml",
                "backend_trace_id": f"trace:{node}",
            }
        )
    graph.pmap[reduction]["reduce_mode"] = "hybrid"
    graph.pmap[(dense, reduction)]["value_id"] = "dense:out"
    graph.pmap["backend_evaluated"] = True
    return graph
