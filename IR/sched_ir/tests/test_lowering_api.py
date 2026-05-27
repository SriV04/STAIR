from IR.sched_ir.lowering.decomposer import decompose_nn_to_sched


def test_canonical_lowering_namespace_exports_decomposer():
    assert callable(decompose_nn_to_sched)
