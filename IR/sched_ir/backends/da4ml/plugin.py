"""Public DA4ML backend plugin entry point."""

from __future__ import annotations

from pathlib import Path

import yaml

from .executor import FoldedCostChainExecutor
from .records import PipelineEvalConfig


def _default_runtime_config() -> dict:
    from da4ml.trace import HWConfig

    return {
        "hwconf": HWConfig(1, -1, -1),
        "pipeline_config": PipelineEvalConfig(
            enabled=True,
            latency_cutoff=5.0,
            retiming=True,
            verbose=False,
        ),
        "verbose": False,
    }


def load_resource_config(resource_config=None) -> dict:
    if isinstance(resource_config, dict):
        if "hwconf" in resource_config:
            return dict(resource_config)
        data = dict(resource_config)
    elif resource_config is None:
        return _default_runtime_config()
    else:
        data = yaml.safe_load(Path(resource_config).read_text()) or {}

    fpga = data.get("fpga") or {}
    from da4ml.trace import HWConfig

    cutoff = fpga.get("latency_cutoff", 5.0)
    if isinstance(cutoff, str):
        cutoff = 5.0
    return {
        "hwconf": HWConfig(
            int(fpga.get("adder_size", 1)),
            int(fpga.get("carry_size", -1)),
            -1,
        ),
        "pipeline_config": PipelineEvalConfig(
            enabled=True,
            latency_cutoff=float(cutoff),
            retiming=True,
            verbose=False,
        ),
        "verbose": False,
    }


class DA4MLBackend:
    name = "da4ml"

    def evaluate(self, sched_graph, *, model, fold_plan, resource_config=None):
        config = load_resource_config(resource_config)
        return FoldedCostChainExecutor(model, fold_plan, config).run(sched_graph)
