from IR.sched_ir.correctness.records import (
    CorrectnessFailure,
    CorrectnessReport,
    FoldGroupSemanticFailure,
    SymbolicTokenValue,
)


def test_symbolic_token_value_carries_schedule_and_logical_position():
    token = SymbolicTokenValue(
        token_id="dense:out:t1",
        source_node=3,
        temporal_step=1,
        logical_slice=(slice(4, 8), slice(None)),
        value="symbolic",
        qints=["q0", "q1"],
        shape=(4, 64),
        ready_cycle=12,
    )

    assert token.token_id == "dense:out:t1"
    assert token.source_node == 3
    assert token.temporal_step == 1
    assert token.value == "symbolic"
    assert token.qints == ["q0", "q1"]
    assert token.ready_cycle == 12


def test_correctness_report_passed_property_tracks_failures():
    passed = CorrectnessReport(
        reference_qints=["ref-qint"],
        scheduled_qints=["sched-qint"],
        failures=[],
        checked_output_count=1,
        metadata={"fold_factor": 2},
    )
    failed = CorrectnessReport(
        reference_qints=["ref-qint"],
        scheduled_qints=["sched-qint"],
        failures=[
            CorrectnessFailure(
                output_index=0,
                path="output_widths",
                expected=(4, 6),
                actual=(4, 7),
                reason="output bit-width mismatch",
                provenance={"task_id": "node:3:t1"},
            )
        ],
        checked_output_count=1,
        metadata={},
    )

    assert passed.passed is True
    assert failed.passed is False
    assert failed.failures[0].provenance["task_id"] == "node:3:t1"


def test_correctness_report_fails_on_fold_group_semantic_failures():
    report = CorrectnessReport(
        reference_qints=[],
        scheduled_qints=[],
        failures=[],
        checked_output_count=0,
        fold_group_failures=[
            FoldGroupSemanticFailure(
                group_id=2,
                source_nodes=(3,),
                reason="missing temporal step 1",
                expected=(0, 1),
                actual=(0,),
            )
        ],
    )

    assert report.passed is False
