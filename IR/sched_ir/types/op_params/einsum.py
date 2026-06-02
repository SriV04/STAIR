from __future__ import annotations


def default_einsum_params() -> dict:
    return {
        "op_type": "einsum",
        "equation": None,
        "input_shapes": None,
        "output_shape": None,
        "input_qints": None,
        "input_kifs": None,
        "output_qint": None,
        "output_kif": None,
        "reduction_axes": None,
        "batch_axes": None,
        "dynamic_weight": True,
        "foldable": False,
        "cost_model": "synthetic_zero",
        "cost_mode": "synthetic_zero",
        "in_bws": None,
        "out_bw": None,
    }
