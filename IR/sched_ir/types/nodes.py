from __future__ import annotations

from .precision import default_precision_interface
from .schedule import default_timing_fields


def default_node_properties() -> dict:
    base = {
        "nn_layer_idx": None,
        "nn_layer_name": None,
        "nn_op_kind": None,
        "decomp_index": None,
        "inserted_by": None,
        "op": None,
        "op_params": None,
        "kernel_type": None,
        "kernel_instance": None,
        "cost": None,
        "kernel_result": None,
        "reduce_mode": None,
        "schema_version": 2,
        "schema_notes": None,
    }
    base.update(default_precision_interface())
    base.update(default_timing_fields())
    return base
