from IR.sched_ir.types.edges import default_edge_properties
from IR.sched_ir.types.graph import default_graph_properties
from IR.sched_ir.types.nodes import default_node_properties


def test_node_schema_contains_serialisable_backend_evaluation_fields():
    node = default_node_properties()
    assert node["backend"] is None
    assert node["backend_trace_id"] is None
    assert node["evaluated_input_shapes"] is None
    assert node["evaluated_output_shapes"] is None
    assert node["evaluated_input_qints"] is None
    assert node["evaluated_input_kifs"] is None
    assert node["evaluated_output_qints"] is None
    assert node["evaluated_output_kifs"] is None
    assert node["evaluated_input_latency"] is None
    assert node["evaluated_output_latency"] is None
    assert node["evaluated_comb_shape"] is None
    assert node["evaluated_n_ops"] is None
    assert node["evaluated_pipeline_stages"] is None


def test_edge_schema_contains_backend_value_snapshot_fields():
    edge = default_edge_properties()
    assert edge["value_id"] is None
    assert edge["evaluated_qints"] is None
    assert edge["evaluated_kifs"] is None
    assert edge["evaluated_shape"] is None
    assert edge["evaluated_latency"] is None


def test_graph_schema_marks_backend_evaluation_and_task_schedule():
    graph = default_graph_properties()
    assert graph["fold_plan"] is None
    assert graph["backend"] is None
    assert graph["backend_evaluated"] is False
    assert graph["backend_warnings"] is None
    assert graph["task_schedule"] is None
