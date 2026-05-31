from __future__ import annotations


def default_timing_fields() -> dict:
    return {
        "parallelism_N": None,
        "lanes_P": None,
        "temporal_steps_T": None,
        "fold_axes": None,
        "fold_group": None,
    }
