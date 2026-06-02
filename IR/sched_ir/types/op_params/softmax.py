from __future__ import annotations


def default_softmax_params() -> dict:
    return {
        "op_type": "softmax",
        "axis": None,
        "input_shape": None,
        "output_shape": None,
        "input_qint": None,
        "input_kif": None,
        "output_qint": None,
        "output_kif": None,
        "implementation": "lookup_table",
        "exp_lut_entries": None,
        "inv_lut_entries": None,
        "stable": True,
        "foldable": False,
        "cost_model": "synthetic_zero",
        "cost_mode": "synthetic_zero",
        "in_bw": None,
        "out_bw": None,
    }
