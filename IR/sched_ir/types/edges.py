"""Default property record for Sched-IR data/control edges."""

from __future__ import annotations


def default_edge_properties() -> dict:
    return {
        "tensor_shape": None,
        "edge_kind": "data",
        "value_id": None,
        "qint": None,
        "kif": None,
        "bitwidth": None,
        "element_bitwidth_bits": None,
        "element_qint": None,
        "element_kif": None,
        "tensor_width_bits": None,
        "volume_bits": None,
        "volume_bits_exact": None,
        "src_qint": None,
        "src_kif": None,
        "src_bitwidth_bits": None,
        "dst_qint": None,
        "dst_kif": None,
        "dst_bitwidth_bits": None,
        "has_quantization_boundary": False,
        "producer_quantizer": None,
        "consumer_quantizer": None,
        "consume_mode": "stepwise",
        "needs_cast": False,
        "cast_mode": None,
        "evaluated_qints": None,
        "evaluated_kifs": None,
        "evaluated_shape": None,
        "evaluated_latency": None,
    }
