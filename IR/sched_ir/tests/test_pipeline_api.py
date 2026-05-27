from types import SimpleNamespace

from IR.sched_ir import api


def test_api_runs_plan_backend_tasks_and_schedule(monkeypatch):
    evaluation = SimpleNamespace(graph=object())
    rendered_task_graph = object()
    task_view_source = {}
    monkeypatch.setattr(api, "decompose_nn_to_sched", lambda graph: "sched")
    monkeypatch.setattr(api, "make_fold_plan", lambda graph, **kwargs: "plan")
    monkeypatch.setattr(
        api.DA4MLBackend,
        "evaluate",
        lambda self, graph, **kwargs: evaluation,
    )
    monkeypatch.setattr(api, "expand_tasks", lambda graph: "tasks")
    monkeypatch.setattr(api, "schedule_tasks", lambda graph: "schedule")
    def render_task_graph(task_ir, schedule, *, source_graph):
        task_view_source["graph"] = source_graph
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
    assert design.task_schedule == "schedule"
    assert design.metrics["throughput_samples_per_sec"] > 0
