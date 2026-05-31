from IR.sched_ir.correctness.compare import (
    compare_output_widths,
    normalize_output_widths,
)


class FakeQInt:
    def __init__(self, bits):
        self.bits = bits


def test_normalize_output_widths_reads_bits_from_out_qints():
    assert normalize_output_widths([FakeQInt(3), FakeQInt(5)]) == (3, 5)


def test_compare_output_widths_passes_identical_width_signatures():
    report = compare_output_widths(
        reference_qints=[FakeQInt(4), FakeQInt(6)],
        scheduled_qints=[FakeQInt(4), FakeQInt(6)],
        provenance=[{"token_id": "out"}],
    )

    assert report.passed is True
    assert report.checked_output_count == 2


def test_compare_output_widths_reports_mismatch_with_provenance():
    report = compare_output_widths(
        reference_qints=[FakeQInt(4), FakeQInt(6)],
        scheduled_qints=[FakeQInt(4), FakeQInt(7)],
        provenance=[{"token_id": "out:t1", "task_id": "node:0:t1"}],
    )

    assert report.passed is False
    assert report.failures[0].path == "output_widths"
    assert report.failures[0].provenance["task_id"] == "node:0:t1"
