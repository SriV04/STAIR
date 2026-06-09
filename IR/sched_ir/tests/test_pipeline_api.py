from types import SimpleNamespace

from IR.sched_ir import api


def test_api_runs_plan_backend_tasks_and_schedule(monkeypatch):
    evaluation = SimpleNamespace(graph=object())
    rendered_task_graph = object()
    task_view_source = {}
    sync_report = SimpleNamespace(
        sync_point_count=0,
        max_sync_wait_cycles=0,
        total_sync_wait_cycles=0,
        mean_sync_wait_cycles=0.0,
        max_buffer_wait_cycles=0,
        total_input_buffer_wait_cycles=0,
        sync_legality_passed=True,
        sync_violations=[],
    )

    class FakeBackend:
        def evaluate(self, graph, **kwargs):
            return evaluation

    monkeypatch.setattr(api, "decompose_nn_to_sched", lambda graph: "sched")
    monkeypatch.setattr(api, "make_fold_plan", lambda graph, **kwargs: "plan")
    monkeypatch.setattr(api, "DA4MLBackend", FakeBackend)
    monkeypatch.setattr(api, "expand_tasks", lambda graph: "tasks")
    monkeypatch.setattr(api, "schedule_tasks", lambda graph: "schedule")
    monkeypatch.setattr(api, "analyse_sync_points", lambda schedule: sync_report)

    def render_task_graph(task_ir, schedule, *, source_graph, sync_report):
        task_view_source["graph"] = source_graph
        task_view_source["sync_report"] = sync_report
        return rendered_task_graph

    monkeypatch.setattr(api, "task_schedule_to_hgraph", render_task_graph)
    monkeypatch.setattr(
        api,
        "schedule_metrics",
        lambda *args, **kwargs: {"throughput_samples_per_sec": 1.0},
    )

    design = api.evaluate_folded_design(
        object(),
        model=object(),
        backend="da4ml",
        factor=2,
        target_fmax_hz=300e6,
    )

    assert design.fold_plan == "plan"
    assert design.evaluation is evaluation
    assert design.task_ir == "tasks"
    assert design.task_graph is rendered_task_graph
    assert task_view_source["graph"] is evaluation.graph
    assert task_view_source["sync_report"] is sync_report
    assert design.task_schedule == "schedule"
    assert design.sync_report is sync_report
    assert design.metrics["throughput_samples_per_sec"] > 0
