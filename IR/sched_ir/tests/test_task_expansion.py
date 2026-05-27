from IR.sched_ir.scheduling.expand import expand_tasks


def test_folded_nodes_create_temporal_tasks_reusing_one_physical_resource(
    evaluated_dense_reduce_graph,
):
    task_graph = expand_tasks(evaluated_dense_reduce_graph)
    dense_tasks = [
        task for task in task_graph.tasks.values() if task.source_node == 0
    ]

    assert len(dense_tasks) == 2
    assert len({task.resource_id for task in dense_tasks}) == 1
    assert {task.temporal_step for task in dense_tasks} == {0, 1}


def test_hybrid_reduce_consumes_edge_identified_temporal_tokens(
    evaluated_dense_reduce_graph,
):
    task_graph = expand_tasks(evaluated_dense_reduce_graph)
    reduce_tasks = [
        task for task in task_graph.tasks.values() if task.source_node == 1
    ]

    assert len(reduce_tasks) == 2
    assert reduce_tasks[0].input_tokens == ["dense:out:t0"]
    assert reduce_tasks[1].input_tokens == ["dense:out:t1"]
