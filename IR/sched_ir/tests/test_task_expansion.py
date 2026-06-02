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
        task
        for task in task_graph.tasks.values()
        if task.source_node == 1 and task.task_kind == "compute"
    ]

    assert len(reduce_tasks) == 2
    assert reduce_tasks[0].input_tokens == ["dense:out:t0"]
    assert reduce_tasks[1].input_tokens == ["dense:out:t1"]


def test_hybrid_reduce_creates_temporal_accumulator_task(
    evaluated_dense_reduce_graph,
):
    task_graph = expand_tasks(evaluated_dense_reduce_graph)

    assert "node:1:acc" in task_graph.tasks
    accumulator = task_graph.tasks["node:1:acc"]

    assert accumulator.task_kind == "temporal_accumulator"
    assert accumulator.source_node == 1
    assert accumulator.temporal_step is None
    assert accumulator.resource_id == "node:1:acc"
    assert accumulator.input_tokens == ["1:partial:t0", "1:partial:t1"]
    assert accumulator.output_tokens == ["1:out"]
    assert task_graph.resources["node:1:acc"].cost["cost_mode"] == "synthetic_temporal_accumulator"
