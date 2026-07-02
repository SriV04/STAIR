"""Op-param record for data-movement (transport) primitives."""

from __future__ import annotations


def default_transport_params() -> dict:
    return {
        "op_type": "transport",
        "mode": None,
        "input_shape": None,
        "output_shape": None,
        "input_qint": None,
        "input_kif": None,
        "output_qint": None,
        "output_kif": None,
        "foldable": False,
        "cost_model": "synthetic_zero",
        "cost_mode": "synthetic_zero",
        "in_bw": None,
        "out_bw": None,
    }
