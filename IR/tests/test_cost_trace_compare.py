import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from IR import cost_trace_compare as MODULE

format_delta = MODULE.format_delta
render_section = MODULE.render_section
render_metric_row = MODULE.render_metric_row
rollup_multiplier = MODULE.rollup_multiplier
rolled_cost_value = MODULE.rolled_cost_value
ff_without_local_reg_bits = MODULE.ff_without_local_reg_bits


class CostTraceCompareTests(unittest.TestCase):
    def test_format_delta_reports_signed_value_and_percent(self):
        self.assertEqual(format_delta(10, 12), "-2 (-16.7%)")
        self.assertEqual(format_delta(12, 10), "+2 (+20.0%)")

    def test_render_section_wraps_title_and_body(self):
        out = render_section("Example", ["line one", "line two"])
        self.assertIn("Example", out)
        self.assertIn("line one", out)
        self.assertIn("line two", out)
        self.assertTrue(out.startswith("=" * 80))

    def test_render_metric_row_aligns_sched_ir_and_ground_truth(self):
        row = render_metric_row("LUTs", 12358, 129316)
        self.assertIn("LUTs", row)
        self.assertIn("12,358", row)
        self.assertIn("129,316", row)
        self.assertIn("-116,958", row)

    def test_rollup_multiplier_matches_dense_vs_reduce_convention(self):
        self.assertEqual(rollup_multiplier("dense", 8), 8)
        self.assertEqual(rollup_multiplier("reduce", 8), 1)
        self.assertEqual(rollup_multiplier("elementwise", 8), 1)

    def test_rolled_cost_value_applies_multiplier(self):
        row = {"op": "dense", "physical_instances": 8, "lut": 101}
        self.assertEqual(rolled_cost_value(row, "lut"), 808)

    def test_ff_without_local_reg_bits_strips_compute_reg_bits_but_keeps_buffers(self):
        dense_row = {"op": "dense", "ff": 3795, "reg_bits": 3795}
        buffer_row = {"op": "buffer", "ff": 1606, "reg_bits": None}
        self.assertEqual(ff_without_local_reg_bits(dense_row), 0)
        self.assertEqual(ff_without_local_reg_bits(buffer_row), 1606)


if __name__ == "__main__":
    unittest.main()
