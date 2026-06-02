from IR.sched_ir.graphing.styling import SCHED_COLORS, sched_vx_label
from IR.sched_ir.graphing.tasks import task_schedule_to_hgraph
from IR.sched_ir.scheduling.expand import expand_tasks
from IR.sched_ir.scheduling.scheduler import schedule_tasks


class StyledFakeHGraph:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.pmap = {}
        self.vstyle = {}
        self.estyle = {}

    def add_vx(self):
        vertex = len(self.vertices)
        self.vertices.append(vertex)
        return vertex

    def add_edge(self, source, destination):
        edge = (source, destination)
        self.edges.append(edge)
        return [edge]

    def in_vx(self, vertex):
        return [source for source, destination in self.edges if destination == vertex]


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

    assert len(graph.vertices) == 5
    reduce_vertices = [
        vertex
        for vertex in graph.vertices
        if graph.pmap[vertex]["source_node"] == 1
        and graph.pmap[vertex]["task_kind"] == "compute"
    ]
    assert len(reduce_vertices) == 2
    assert all(graph.in_vx(vertex) for vertex in reduce_vertices)


def test_task_schedule_view_styles_executions_from_source_node_metadata(
    monkeypatch,
    evaluated_dense_reduce_graph,
):
    monkeypatch.setattr(
        "IR.sched_ir.graphing.tasks.HGraph",
        StyledFakeHGraph,
    )
    task_graph = expand_tasks(evaluated_dense_reduce_graph)
    schedule = schedule_tasks(task_graph)

    graph = task_schedule_to_hgraph(
        task_graph,
        schedule,
        source_graph=evaluated_dense_reduce_graph,
    )
    dense_vertex = next(
        vertex
        for vertex in graph.vertices
        if graph.pmap[vertex]["task_id"] == "node:0:t0"
    )
    label = graph.vstyle["label"](graph, dense_vertex)

    assert graph.vstyle["fillcolor"](graph, dense_vertex) == SCHED_COLORS["dense"]
    assert "dense" in label
    assert "[dense]" in label
    assert "task inputs: ?" in label
    assert "task: node:0:t0" in label
    assert "resource: node:0" in label
    assert "t=[0..2]" in label


def test_task_schedule_view_styles_temporal_accumulator_as_distinct_vertex(
    monkeypatch,
    evaluated_dense_reduce_graph,
):
    monkeypatch.setattr(
        "IR.sched_ir.graphing.tasks.HGraph",
        StyledFakeHGraph,
    )
    task_graph = expand_tasks(evaluated_dense_reduce_graph)
    schedule = schedule_tasks(task_graph)

    graph = task_schedule_to_hgraph(
        task_graph,
        schedule,
        source_graph=evaluated_dense_reduce_graph,
    )
    accumulator_vertex = next(
        vertex
        for vertex in graph.vertices
        if graph.pmap[vertex]["task_id"] == "node:1:acc"
    )
    incoming_edges = [
        edge for edge in graph.edges if edge[1] == accumulator_vertex
    ]
    label = graph.vstyle["label"](graph, accumulator_vertex)

    assert graph.vstyle["fillcolor"](graph, accumulator_vertex) == "#8E24AA"
    assert graph.vstyle["shape"](graph, accumulator_vertex) == "component"
    assert graph.vstyle["style"](graph, accumulator_vertex) == "filled,bold"
    assert graph.vstyle["fontcolor"](graph, accumulator_vertex) == "white"
    assert "[temporal accumulator]" in label
    assert "partials: 2" in label
    assert "task inputs: 4" in label
    assert "task: node:1:acc" in label
    assert graph.estyle["color"](graph, incoming_edges[0]) == "#8E24AA"
    assert graph.estyle["style"](graph, incoming_edges[0]) == "dashed"
    assert graph.estyle["label"](graph, incoming_edges[0]).startswith("acc ")
