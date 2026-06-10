from __future__ import annotations

import sys
from types import ModuleType


class FakeHGraph:
    def __init__(self, *, vinit=None, einit=None, ginit=None):
        self._vinit = vinit
        self._einit = einit
        self.vertices = []
        self.edges = []
        self.pmap = {}
        self.vstyle = {}
        self.estyle = {}
        if ginit is not None:
            ginit(self)

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


fake_heterograph = ModuleType("heterograph")
fake_heterograph.HGraph = FakeHGraph
sys.modules["heterograph"] = fake_heterograph

from IR.sched_ir.analysis.einsum_axes import (  # noqa: E402
    derive_local_equation,
    detect_fold_axes,
    parse_einsum_spec,
)
from IR.sched_ir.lowering.decomposer import decompose_nn_to_sched  # noqa: E402


def test_detect_fold_axes_preserves_structural_axis_absent_from_weight():
    spec = parse_einsum_spec(
        equation="bnc,cC->bnC",
        input_shape=(None, 8, 3),
        weight_shape=(3, 64),
        output_shape=(None, 8, 64),
        layer_name="q_dense",
    )

    assert detect_fold_axes(spec) == ["n"]
    assert derive_local_equation(spec, "n") == "c,cC->C"


def test_detect_fold_axes_rejects_axis_that_indexes_weight_tensor():
    spec = parse_einsum_spec(
        equation="bnc,ncC->bnC",
        input_shape=(None, 8, 3),
        weight_shape=(8, 3, 64),
        output_shape=(None, 8, 64),
        layer_name="q_dense_unshared",
    )

    assert detect_fold_axes(spec) == []


def test_detect_fold_axes_rejects_batch_and_size_one_axes():
    vector_spec = parse_einsum_spec(
        equation="bc,cC->bC",
        input_shape=(None, 64),
        weight_shape=(64, 5),
        output_shape=(None, 5),
        layer_name="classifier",
    )
    size_one_spec = parse_einsum_spec(
        equation="bnc,cC->bnC",
        input_shape=(None, 1, 64),
        weight_shape=(64, 64),
        output_shape=(None, 1, 64),
        layer_name="post_sum_dense",
    )

    assert detect_fold_axes(vector_spec) == []
    assert detect_fold_axes(size_one_spec) == []


def test_decomposer_uses_einsum_semantics_instead_of_shape_only_fold_axes():
    nn_g = FakeHGraph()
    nn_g.pmap["name"] = "fake_nn"
    source = nn_g.add_vx()
    dense = nn_g.add_vx()
    nn_g.pmap[source] = {
        "layer_idx": 0,
        "layer_name": "input",
        "op_kind": "input",
        "out_shapes": [(None, 8, 3)],
    }
    nn_g.pmap[dense] = {
        "layer_idx": 1,
        "layer_name": "unshared_dense",
        "op_kind": "einsum_dense",
        "equation": "bnc,ncC->bnC",
        "in_shapes": [(None, 8, 3)],
        "out_shapes": [(None, 8, 64)],
        "kernel_shape": (8, 3, 64),
        "foldable": True,
    }
    edge = nn_g.add_edge(source, dense)[0]
    nn_g.pmap[edge] = {"tensor_shape": (None, 8, 3)}

    sched_g = decompose_nn_to_sched(nn_g)
    dense_vx = sched_g.vertices[0]

    assert sched_g.pmap[dense_vx]["fold_axes"] is None
