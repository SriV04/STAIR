from types import SimpleNamespace

from IR.sched_ir.correctness.fold_group_semantics import check_fold_group_semantics
from IR.sched_ir.correctness.records import SymbolicTokenValue
from IR.sched_ir.planning.fold_plan import FoldGroup, FoldPlan


class FakeGraph:
    vertices = [3]
    edges = []

    def __init__(self, node):
        self.pmap = {3: node}


def _group(**overrides):
    values = {
        "group_id": 7,
        "axes": (0,),
        "parallelism_n": 8,
        "lanes_p": 4,
        "temporal_steps_t": 2,
        "members": (3,),
        "reductions_temporalised": (),
    }
    values.update(overrides)
    return FoldGroup(**values)


def _token(
    token_id,
    *,
    step,
    logical_slice,
    value="Y",
    source_node=3,
    task_kind="compute",
):
    return SymbolicTokenValue(
        token_id=token_id,
        source_node=source_node,
        temporal_step=step,
        logical_slice=logical_slice,
        value=value,
        qints=[],
        shape=(8, 4),
        ready_cycle=step,
        fold_group=7,
        task_kind=task_kind,
    )


def _trace(tokens):
    return SimpleNamespace(
        token_values={token.token_id: token for token in tokens},
        output_tokens=[],
        metadata={},
    )


def _evaluation(node=None):
    return SimpleNamespace(
        graph=FakeGraph(
            node
            or {
                "op": "dense",
                "op_params": {"output_shape": (8, 4)},
                "fold_axes": [0],
                "fold_group": 7,
                "parallelism_N": 8,
                "lanes_P": 4,
                "temporal_steps_T": 2,
            }
        )
    )


def test_fold_group_semantics_accepts_contiguous_slices_for_each_member_node():
    trace = _trace(
        [
            _token("3:out:t0", step=0, logical_slice=(slice(0, 4), slice(None))),
            _token("3:out:t1", step=1, logical_slice=(slice(4, 8), slice(None))),
        ]
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(),)),
        trace,
        _evaluation(),
    )

    assert failures == []


def test_fold_group_semantics_accepts_batch_indexed_fold_axis_after_batch_strip():
    trace = _trace(
        [
            _token("3:out:t0", step=0, logical_slice=(slice(0, 4), slice(None))),
            _token("3:out:t1", step=1, logical_slice=(slice(4, 8), slice(None))),
        ]
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(axes=(1,)),)),
        trace,
        _evaluation(
            {
                "op": "dense",
                "op_params": {"output_shape": (None, 8, 4)},
                "fold_axes": [1],
                "fold_group": 7,
                "parallelism_N": 8,
                "lanes_P": 4,
                "temporal_steps_T": 2,
            }
        ),
    )

    assert failures == []


def test_fold_group_semantics_accepts_single_step_unsliced_group_as_trivial():
    token = _token("3:out", step=0, logical_slice=None)

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(lanes_p=8, temporal_steps_t=1),)),
        _trace([token]),
        _evaluation(
            {
                "op": "dense",
                "op_params": {"output_shape": (8, 4)},
                "fold_axes": [0],
                "fold_group": 7,
                "parallelism_N": 8,
                "lanes_P": 8,
                "temporal_steps_T": 1,
            }
        ),
    )

    assert failures == []


def test_fold_group_semantics_recovers_full_shape_from_single_output_shapes_tuple():
    trace = _trace(
        [
            _token("3:out:t0", step=0, logical_slice=(slice(0, 4), slice(None))),
            _token("3:out:t1", step=1, logical_slice=(slice(4, 8), slice(None))),
        ]
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(),)),
        trace,
        _evaluation(
            {
                "op": "dense",
                "op_params": {"output_shapes": (8, 4)},
                "fold_axes": [0],
                "fold_group": 7,
                "parallelism_N": 8,
                "lanes_P": 4,
                "temporal_steps_T": 2,
            }
        ),
    )

    assert failures == []


def test_fold_group_semantics_accepts_temporalized_reduction_partials_with_accumulator():
    trace = _trace(
        [
            _token("3:partial:t0", step=0, logical_slice=(slice(0, 1),), value="p0"),
            _token("3:partial:t1", step=1, logical_slice=(slice(4, 1),), value="p1"),
            _token(
                "3:out",
                step=None,
                logical_slice=None,
                value="reduced",
                task_kind="temporal_accumulator",
            ),
        ]
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(reductions_temporalised=(3,)),)),
        trace,
        _evaluation(
            {
                "op": "reduce",
                "reduce_mode": "hybrid",
                "op_params": {"output_shape": (1,)},
                "fold_axes": [0],
                "fold_group": 7,
                "parallelism_N": 8,
                "lanes_P": 4,
                "temporal_steps_T": 2,
            }
        ),
    )

    assert failures == []


def test_fold_group_semantics_reports_missing_temporal_reduction_accumulator():
    trace = _trace(
        [
            _token("3:partial:t0", step=0, logical_slice=(slice(0, 1),), value="p0"),
            _token("3:partial:t1", step=1, logical_slice=(slice(4, 1),), value="p1"),
        ]
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(reductions_temporalised=(3,)),)),
        trace,
        _evaluation(
            {
                "op": "reduce",
                "reduce_mode": "hybrid",
                "op_params": {"output_shape": (1,)},
                "fold_axes": [0],
                "fold_group": 7,
                "parallelism_N": 8,
                "lanes_P": 4,
                "temporal_steps_T": 2,
            }
        ),
    )

    assert len(failures) == 1
    assert failures[0].reason == "missing temporal reduction accumulator"


def test_fold_group_semantics_reports_missing_temporal_step():
    trace = _trace(
        [
            _token("3:out:t0", step=0, logical_slice=(slice(0, 4), slice(None))),
        ]
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(),)),
        trace,
        _evaluation(),
    )

    assert len(failures) == 1
    assert failures[0].group_id == 7
    assert failures[0].source_nodes == (3,)
    assert failures[0].reason == "missing temporal step 1"


def test_fold_group_semantics_reports_duplicate_temporal_step():
    trace = _trace(
        [
            _token("3:out:t0:a", step=0, logical_slice=(slice(0, 4), slice(None))),
            _token("3:out:t0:b", step=0, logical_slice=(slice(0, 4), slice(None))),
            _token("3:out:t1", step=1, logical_slice=(slice(4, 8), slice(None))),
        ]
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(),)),
        trace,
        _evaluation(),
    )

    assert failures[0].reason == "duplicate temporal step 0"


def test_fold_group_semantics_reports_wrong_axis_gap_overlap_and_oob():
    cases = [
        (
            "expected fold axis 0 but token used axis 1",
            [
                _token("3:out:t0", step=0, logical_slice=(slice(None), slice(0, 4))),
                _token("3:out:t1", step=1, logical_slice=(slice(None), slice(4, 8))),
            ],
        ),
        (
            "slice coverage gap between 3 and 4",
            [
                _token("3:out:t0", step=0, logical_slice=(slice(0, 3), slice(None))),
                _token("3:out:t1", step=1, logical_slice=(slice(4, 8), slice(None))),
            ],
        ),
        (
            "slice overlap between 5 and 4",
            [
                _token("3:out:t0", step=0, logical_slice=(slice(0, 5), slice(None))),
                _token("3:out:t1", step=1, logical_slice=(slice(4, 8), slice(None))),
            ],
        ),
        (
            "slice 4:9 is out of bounds for folded dimension 8",
            [
                _token("3:out:t0", step=0, logical_slice=(slice(0, 4), slice(None))),
                _token("3:out:t1", step=1, logical_slice=(slice(4, 9), slice(None))),
            ],
        ),
    ]

    for expected_reason, tokens in cases:
        failures = check_fold_group_semantics(
            FoldPlan(groups=(_group(),)),
            _trace(tokens),
            _evaluation(),
        )

        assert failures[0].reason == expected_reason


def test_fold_group_semantics_reports_different_symbolic_parents():
    trace = _trace(
        [
            _token("3:out:t0", step=0, logical_slice=(slice(0, 4), slice(None)), value="Y0"),
            _token("3:out:t1", step=1, logical_slice=(slice(4, 8), slice(None)), value="Y1"),
        ]
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(),)),
        trace,
        _evaluation(),
    )

    assert failures[0].reason == "slices have different symbolic parents"


def test_fold_group_semantics_validates_each_source_node_inside_group_independently():
    trace = _trace(
        [
            _token("3:out:t0", step=0, logical_slice=(slice(0, 4), slice(None)), source_node=3),
            _token("3:out:t1", step=1, logical_slice=(slice(4, 8), slice(None)), source_node=3),
            _token("4:out:t0", step=0, logical_slice=(slice(0, 4), slice(None)), source_node=4),
        ]
    )
    evaluation = SimpleNamespace(
        graph=SimpleNamespace(
            pmap={
                3: {"op_params": {"output_shape": (8, 4)}, "fold_axes": [0], "fold_group": 7},
                4: {"op_params": {"output_shape": (8, 2)}, "fold_axes": [0], "fold_group": 7},
            }
        )
    )

    failures = check_fold_group_semantics(
        FoldPlan(groups=(_group(members=(3, 4)),)),
        trace,
        evaluation,
    )

    assert len(failures) == 1
    assert failures[0].source_nodes == (4,)
    assert failures[0].reason == "missing temporal step 1"
