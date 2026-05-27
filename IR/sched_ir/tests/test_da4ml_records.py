from IR.sched_ir.backends.da4ml.records import (
    NodeEvaluation,
    PrimitiveEvaluation,
    SymbolicTensorState,
    snapshot_node_evaluation,
)


def test_runtime_state_retains_symbolic_input_without_serialising_it():
    symbolic = object()
    state = SymbolicTensorState(
        value_id="v0",
        shape=(4, 64),
        qints=["q0"],
        kifs=["k0"],
        latency=2.0,
        producer_node=0,
        symbolic_value=symbolic,
    )
    evaluation = NodeEvaluation(
        node_id=1,
        trace_id="trace:1",
        input_states=[state],
        output_states=[],
        cost={"lut": 2},
        latency=(2.0, 3.0),
        n_ops=4,
    )

    snapshot = snapshot_node_evaluation(evaluation)

    assert evaluation.input_states[0].symbolic_value is symbolic
    assert snapshot["backend_trace_id"] == "trace:1"
    assert snapshot["evaluated_input_shapes"] == [(4, 64)]
    assert "symbolic_value" not in snapshot


def test_primitive_evaluation_is_the_cost_model_executor_contract():
    symbolic_input = object()
    symbolic_output = object()
    result = PrimitiveEvaluation(
        symbolic_inputs=[symbolic_input],
        symbolic_outputs=[symbolic_output],
        output_shapes=[(4, 64)],
        output_qints=["qout"],
        output_kifs=["kout"],
        output_latency=3.0,
        cost={"lut": 2, "ii": 1},
        n_ops=4,
        comb_logic=object(),
        pipeline=None,
        kernel_meta={},
    )

    assert result.symbolic_inputs == [symbolic_input]
    assert result.symbolic_outputs == [symbolic_output]
    assert result.output_kifs == ["kout"]
