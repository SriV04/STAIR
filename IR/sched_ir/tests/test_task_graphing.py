from IR.sched_ir.graphing.styling import sched_vx_label
from IR.sched_ir.graphing.tasks import task_schedule_to_hgraph
from IR.sched_ir.scheduling.expand import expand_tasks
from IR.sched_ir.scheduling.scheduler import schedule_tasks


def test_evaluated_label_includes_backend_trace_and_npt(evaluated_dense_reduce_graph):
    label = sched_vx_label(evaluated_dense_reduce_graph, 0)

    assert "N=8 P=4 T=2" in label
    assert "da4ml" in label
    assert "trace:0" in label


def test_task_schedule_view_contains_executions_and_token_dependency_edges(
    evaluated_dense_reduce_graph,
):
    task_graph = expand_tasks(evaluated_dense_reduce_graph)
    schedule = schedule_tasks(task_graph)

    graph = task_schedule_to_hgraph(task_graph, schedule)

    assert len(graph.vertices) == 4
    reduce_vertices = [
        vertex for vertex in graph.vertices if graph.pmap[vertex]["source_node"] == 1
    ]
    assert len(reduce_vertices) == 2
    assert all(graph.in_vx(vertex) for vertex in reduce_vertices)
