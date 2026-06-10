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
        "foldable": True,
        "fold_axes": None,
        "fold_axis_names": None,
        "einsum_spec": None,
        "local_equations": None,
        "cost_mode": None,
        "keras_layer": None,
        "cost": None,
        "reduce_mode": None,
        "backend": None,
        "backend_trace_id": None,
        "evaluated_input_shapes": None,
        "evaluated_output_shapes": None,
        "evaluated_input_qints": None,
        "evaluated_input_kifs": None,
        "evaluated_output_qints": None,
        "evaluated_output_kifs": None,
        "evaluated_input_latency": None,
        "evaluated_output_latency": None,
        "evaluated_comb_shape": None,
        "evaluated_n_ops": None,
        "evaluated_pipeline_stages": None,
        "schema_version": 2,
        "schema_notes": None,
    }
    base.update(default_precision_interface())
    base.update(default_timing_fields())
    return base
