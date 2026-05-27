from IR.sched_ir.planning.fold_plan import FoldPlan, make_fold_plan


def test_make_fold_plan_groups_connected_shared_axis_and_stamps_npt(build_dense_reduce_graph):
    graph, dense, reduction = build_dense_reduce_graph()
    plan = make_fold_plan(graph, factor=2)

    assert isinstance(plan, FoldPlan)
    assert len(plan.groups) == 1
    group = plan.groups[0]
    assert group.parallelism_n == 8
    assert group.lanes_p == 4
    assert group.temporal_steps_t == 2
    assert graph.pmap[dense]["temporal_steps_T"] == 2
    assert graph.pmap[reduction]["reduce_mode"] == "hybrid"


def test_make_fold_plan_marks_fully_temporal_reduction(build_dense_reduce_graph):
    graph, _, reduction = build_dense_reduce_graph()
    plan = make_fold_plan(graph, lanes=1)

    assert plan.groups[0].temporal_steps_t == 8
    assert graph.pmap[reduction]["reduce_mode"] == "temporal_accumulate"
