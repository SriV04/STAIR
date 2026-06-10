from IR.sched_ir.backends.da4ml.attention_cost_oracle import (
    estimate_attention_cost,
    estimate_einsum_cost,
    estimate_softmax_cost,
)


def test_einsum_cost_counts_output_elements_and_reduction_terms():
    node = {
        "op": "einsum",
        "op_params": {
            "equation": "bshd,bthd->bhst",
            "input_shapes": [(None, 8, 1, 16), (None, 2, 1, 16)],
            "output_shape": (None, 1, 8, 2),
            "input_kifs": [{"bits": 6}, {"bits": 5}],
            "output_kif": {"bits": 12},
        },
    }

    cost = estimate_einsum_cost(node)

    assert cost["cost_mode"] == "analytic_einsum_attention_oracle"
    assert cost["einsum_output_elements"] == 16
    assert cost["einsum_reduction_terms"] == 16
    assert cost["einsum_multiply_count"] == 256
    assert cost["lut"] > 0
    assert cost["latency_cycles"] >= 2


def test_softmax_cost_scales_with_axis_groups_and_widths():
    node = {
        "op": "softmax",
        "op_params": {
            "axis": (-1,),
            "input_shape": (None, 1, 8, 2),
            "output_shape": (None, 1, 8, 2),
            "input_kif": {"bits": 10},
            "output_kif": {"bits": 8},
            "stable": True,
            "exp_lut_entries": 256,
            "inv_lut_entries": 256,
        },
    }

    cost = estimate_softmax_cost(node)

    assert cost["cost_mode"] == "analytic_softmax_attention_oracle"
    assert cost["softmax_axis_size"] == 2
    assert cost["softmax_group_count"] == 8
    assert cost["softmax_element_count"] == 16
    assert cost["softmax_stable"] is True
    assert cost["lut"] > 0
    assert cost["latency_cycles"] >= 4


def test_attention_cost_uses_matching_hls4ml_reported_ii_and_resources():
    node = {
        "op": "einsum",
        "nn_layer_name": "mha1_qk",
        "op_params": {
            "equation": "bshd,bthd->bhst",
            "input_shapes": [(None, 8, 1, 16), (None, 2, 1, 16)],
            "output_shape": (None, 1, 8, 2),
            "input_kifs": [{"bits": 6}, {"bits": 5}],
            "output_kif": {"bits": 12},
        },
    }
    calibration = [
        {
            "op": "einsum",
            "layer_name": "mha1_qk",
            "cost": {
                "lut": 123,
                "ff": 45,
                "dsp": 6,
                "bram": 1,
                "latency_cycles": 17,
                "ii": 4,
            },
        }
    ]

    cost = estimate_attention_cost(node, calibration=calibration)

    assert cost["cost_mode"] == "hls4ml_report_calibrated"
    assert cost["lut"] == 123
    assert cost["ff"] == 45
    assert cost["dsp"] == 6
    assert cost["bram"] == 1
    assert cost["latency_cycles"] == 17
    assert cost["ii"] == 4
    assert cost["ii_source"] == "hls4ml_report"
    assert cost["einsum_multiply_count"] == 256


def test_attention_cost_matches_hls4ml_report_by_signature_without_layer_name():
    node = {
        "op": "softmax",
        "op_params": {
            "axis": (-1,),
            "input_shape": (None, 1, 8, 2),
            "output_shape": (None, 1, 8, 2),
        },
    }
    calibration = {
        "hls4ml_reports": [
            {
                "op": "softmax",
                "axis": [-1],
                "input_shape": [None, 1, 8, 2],
                "output_shape": [None, 1, 8, 2],
                "resources": {"lut": 88, "ff": 77, "dsp": 0, "bram": 0},
                "latency_cycles": 9,
                "ii": 2,
            }
        ]
    }

    cost = estimate_attention_cost(node, calibration=calibration)

    assert cost["cost_mode"] == "hls4ml_report_calibrated"
    assert cost["lut"] == 88
    assert cost["ff"] == 77
    assert cost["latency_cycles"] == 9
    assert cost["ii"] == 2
