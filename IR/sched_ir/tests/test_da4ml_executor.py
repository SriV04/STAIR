from IR.sched_ir.backends.da4ml.executor import FoldedCostChainExecutor
from IR.sched_ir.backends.da4ml.records import PrimitiveEvaluation
from IR.sched_ir.planning.fold_plan import make_fold_plan


def _primitive(symbolic_input, symbolic_output, qint):
    return PrimitiveEvaluation(
        symbolic_inputs=[] if symbolic_input is None else [symbolic_input],
        symbolic_outputs=[symbolic_output],
        output_shapes=[(4,)],
        output_qints=[qint],
        output_kifs=[{"bits": 4}],
        output_latency=1.0,
        cost={"lut": 2, "ff": 1, "ii": 1, "latency_cycles": 1},
        n_ops=1,
        comb_logic=object(),
        pipeline=None,
        kernel_meta={},
    )


def test_executor_retains_next_symbolic_input_built_from_predecessor_output(
    build_dense_reduce_graph,
    monkeypatch,
):
    graph, _, _ = build_dense_reduce_graph()
    plan = make_fold_plan(graph, factor=2)
    produced = object()
    rebuilt_input = object()
    calls = []

    def fake_evaluate(node_pmap, *, input_states, keras_layer, config):
        calls.append(input_states)
        if len(calls) == 1:
            return _primitive(None, produced, "q0")
        assert input_states[0].symbolic_value is produced
        return _primitive(rebuilt_input, object(), "q1")

    monkeypatch.setattr(
        "IR.sched_ir.backends.da4ml.executor.evaluate_node",
        fake_evaluate,
    )

    result = FoldedCostChainExecutor(model=None, fold_plan=plan, config={}).run(graph)

    second = result.node_results[1]
    assert second.input_states[0].symbolic_value is produced
    assert result.runtime_artifacts[second.trace_id].symbolic_inputs == [rebuilt_input]
    assert result.graph.pmap[second.node_id]["backend_trace_id"] == second.trace_id
    assert result.graph.pmap["backend_evaluated"] is True
