"""Compare final output bit-width signatures derived from DA4ML qints."""

from __future__ import annotations

import numpy as np

from .records import CorrectnessFailure, CorrectnessReport


def qint_bit_width(qint):
    if getattr(qint, "neutral", False):
        return None
    if hasattr(qint, "bits"):
        value = qint.bits
        if callable(value):
            value = value()
        return int(value)
    if isinstance(qint, dict) and "bits" in qint:
        return int(qint["bits"])
    if isinstance(qint, (tuple, list, np.ndarray)):
        values = np.asarray(qint, dtype=object).reshape(-1).tolist()
        if len(values) >= 3:
            return int(values[0]) + int(values[1]) + int(values[2])
        if len(values) == 1:
            return int(values[0])
    return int(qint)


def normalize_output_widths(qints):
    widths = []
    for qint in qints:
        width = qint_bit_width(qint)
        if width is not None:
            widths.append(width)
    return tuple(widths)


def compare_output_widths(
    *,
    reference_qints,
    scheduled_qints,
    provenance=None,
) -> CorrectnessReport:
    provenance = list(provenance or [{}])
    failures = []
    expected = normalize_output_widths(reference_qints)
    actual = normalize_output_widths(scheduled_qints)
    checked = min(len(expected), len(actual))
    if len(expected) != len(actual):
        failures.append(
            CorrectnessFailure(
                output_index=-1,
                path="output_widths",
                expected=expected,
                actual=actual,
                reason="output width count mismatch",
                provenance=provenance[0] if provenance else {},
            )
        )
    elif expected != actual:
        failures.append(
            CorrectnessFailure(
                output_index=0,
                path="output_widths",
                expected=expected,
                actual=actual,
                reason="output bit-width mismatch",
                provenance=provenance[0] if provenance else {},
            )
        )
    return CorrectnessReport(
        reference_qints=list(reference_qints),
        scheduled_qints=list(scheduled_qints),
        failures=failures,
        checked_output_count=checked,
        metadata={},
    )
