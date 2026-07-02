"""Default graph-level properties for a Sched-IR graph."""

from __future__ import annotations


def default_graph_properties() -> dict:
    return {
        "name": None,
        "source_nn_ir": None,
        "schema_version": 2,
        "target_device": None,
        "target_fmax": None,
        "target_slrs": None,
        "fpga_config": None,
        "objective": None,
        "area_budget": None,
        "latency_budget": None,
        "fold_plan": None,
        "backend": None,
        "backend_evaluated": False,
        "backend_warnings": None,
    }
