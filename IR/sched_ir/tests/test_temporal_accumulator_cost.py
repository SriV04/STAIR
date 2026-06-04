from IR.sched_ir.scheduling.temporal_accumulator_cost import (
    estimate_temporal_accumulator_cost,
)


def test_accumulator_width_uses_output_width_when_output_is_wider():
    node = {
        "temporal_steps_T": 4,
        "evaluated_input_kifs": [[[0, 7, 0], [0, 7, 0]]],
        "evaluated_output_kifs": [[[0, 10, 0], [0, 10, 0]]],
        "evaluated_output_shapes": [(2,)],
    }

    cost = estimate_temporal_accumulator_cost(node)

    assert cost["cost_mode"] == "synthetic_temporal_accumulator_width_proxy"
    assert cost["accumulator_width_bits"] == 10
    assert cost["accumulator_elements"] == 2
    assert cost["temporal_steps_T"] == 4
    assert cost["lut"] == 2 * 10 + 3 + 2
    assert cost["ff"] == 2 * 10 + 3 + 1
    assert cost["latency_cycles"] == 1
    assert cost["ii"] == 1


def test_accumulator_width_adds_guard_bits_when_input_growth_is_wider():
    node = {
        "temporal_steps_T": 8,
        "evaluated_input_kifs": [[[0, 8, 0]]],
        "evaluated_output_kifs": [[[0, 6, 0]]],
        "evaluated_output_shapes": [(1,)],
    }

    cost = estimate_temporal_accumulator_cost(node)

    assert cost["accumulator_width_bits"] == 11
    assert cost["accumulator_elements"] == 1
    assert cost["lut"] == 11 + 4 + 2
    assert cost["ff"] == 11 + 4 + 1


def test_accumulator_element_count_multiplies_sum_storage_and_adder_cost():
    node = {
        "temporal_steps_T": 2,
        "evaluated_input_kifs": [[[0, 4, 0] for _ in range(6)]],
        "evaluated_output_kifs": [[[0, 4, 0] for _ in range(6)]],
        "evaluated_output_shapes": [(2, 3)],
    }

    cost = estimate_temporal_accumulator_cost(node)

    assert cost["accumulator_width_bits"] == 5
    assert cost["accumulator_elements"] == 6
    assert cost["lut"] == 6 * 5 + 2 + 2
    assert cost["ff"] == 6 * 5 + 2 + 1


def test_accumulator_flattens_three_kif_entries_as_elements_not_one_triplet():
    node = {
        "temporal_steps_T": 2,
        "evaluated_input_kifs": [[[0, 4, 0] for _ in range(3)]],
        "evaluated_output_kifs": [[[0, 4, 0] for _ in range(3)]],
        "evaluated_output_shapes": [(3,)],
    }

    cost = estimate_temporal_accumulator_cost(node)

    assert cost["accumulator_width_bits"] == 5
    assert cost["accumulator_elements"] == 3
    assert cost["lut"] == 3 * 5 + 2 + 2


def test_accumulator_width_accepts_dict_kif_bits_metadata():
    node = {
        "temporal_steps_T": 4,
        "evaluated_input_kifs": [[{"bits": 5}, {"bits": 7}]],
        "evaluated_output_kifs": [[{"bits": 6}, {"bits": 8}]],
        "evaluated_output_shapes": [(2,)],
    }

    cost = estimate_temporal_accumulator_cost(node)

    assert cost["cost_mode"] == "synthetic_temporal_accumulator_width_proxy"
    assert cost["accumulator_width_bits"] == 9
    assert cost["accumulator_elements"] == 2


def test_accumulator_cost_missing_metadata_falls_back_conservatively():
    node = {"temporal_steps_T": 4}

    cost = estimate_temporal_accumulator_cost(node)

    assert cost["cost_mode"] == "synthetic_temporal_accumulator_missing_metadata"
    assert cost["accumulator_width_bits"] == 3
    assert cost["accumulator_elements"] == 1
    assert cost["lut"] == 3 + 3 + 2
    assert cost["ff"] == 3 + 3 + 1
