from IR.sched_ir.backends.da4ml.plugin import DA4MLBackend


def test_plugin_delegates_complete_evaluation_to_sequential_executor(monkeypatch):
    graph = object()
    plan = object()
    model = object()
    seen = {}

    def fake_run(self, passed_graph):
        seen["graph"] = passed_graph
        seen["config"] = self.config
        return "evaluated"

    monkeypatch.setattr(
        "IR.sched_ir.backends.da4ml.plugin.FoldedCostChainExecutor.run",
        fake_run,
    )

    result = DA4MLBackend().evaluate(
        graph,
        model=model,
        fold_plan=plan,
        resource_config={"hwconf": "hw", "pipeline_config": None},
    )

    assert result == "evaluated"
    assert seen["graph"] is graph
    assert seen["config"]["hwconf"] == "hw"
