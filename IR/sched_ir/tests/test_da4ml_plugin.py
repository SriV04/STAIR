import sys
from types import ModuleType

from IR.sched_ir.backends.da4ml.plugin import DA4MLBackend
from IR.sched_ir.backends.da4ml.plugin import load_resource_config


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


def test_load_resource_config_preserves_attention_oracle_from_yaml(tmp_path, monkeypatch):
    class FakeHWConfig:
        def __init__(self, adder_size, carry_size, _unused):
            self.adder_size = adder_size
            self.carry_size = carry_size

    fake_da4ml = ModuleType("da4ml")
    fake_trace = ModuleType("da4ml.trace")
    fake_trace.HWConfig = FakeHWConfig
    monkeypatch.setitem(sys.modules, "da4ml", fake_da4ml)
    monkeypatch.setitem(sys.modules, "da4ml.trace", fake_trace)

    resource_yaml = tmp_path / "resource.yaml"
    resource_yaml.write_text(
        "\n".join(
            [
                "fpga:",
                "  adder_size: 2",
                "  carry_size: 4",
                "attention_cost_oracle:",
                "  hls4ml_reports:",
                "    - op: softmax",
                "      layer_name: mha1_softmax",
                "      cost:",
                "        lut: 70",
                "        ff: 20",
                "        latency_cycles: 11",
                "        ii: 3",
            ]
        )
    )

    config = load_resource_config(resource_yaml)

    assert config["hwconf"].adder_size == 2
    assert config["hwconf"].carry_size == 4
    assert config["attention_cost_oracle"]["hls4ml_reports"][0]["cost"]["ii"] == 3
