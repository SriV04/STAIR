import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from IR import evaluate_helpers as MODULE
summarize_precision_payload = MODULE.summarize_precision_payload
apply_k1_ground_truth_override = MODULE.apply_k1_ground_truth_override
build_k1_validation_rows = MODULE.build_k1_validation_rows


class EvaluateHelpersTests(unittest.TestCase):
    def test_summarize_scalar_qint_payload(self):
        payload = {"min": -1.0, "max": 1.0, "step": 0.5}
        self.assertEqual(summarize_precision_payload(payload), "qint")

    def test_summarize_list_of_qints_reports_length(self):
        payload = [
            {"min": -1.0, "max": 1.0, "step": 0.5},
            {"min": -2.0, "max": 2.0, "step": 0.25},
        ]
        self.assertEqual(summarize_precision_payload(payload), "list[2]<qint>")

    def test_summarize_nested_list_reports_inner_shape(self):
        payload = [[{"min": -1.0, "max": 1.0, "step": 0.5}] for _ in range(35)]
        self.assertEqual(summarize_precision_payload(payload), "list[35]<list[1]<qint>>")

    def test_summarize_kif_payload(self):
        payload = {"k": True, "i": 3, "f": 1, "bits": 5}
        self.assertEqual(summarize_precision_payload(payload), "kif")

    def test_summarize_none_payload(self):
        self.assertEqual(summarize_precision_payload(None), "None")

    def test_apply_k1_ground_truth_override_preserves_sched_ir_metrics(self):
        metrics = {
            "K": 1,
            "total_luts": 100,
            "total_ffs": 200,
            "makespan": 10,
            "pipeline_depth": 8,
            "batches_in_flight": 8,
            "II": 1,
            "throughput_hz": 300e6,
            "throughput_mhz": 300.0,
        }
        ground_truth = {"lut": 120, "ff": 240, "stages": 12}

        out = apply_k1_ground_truth_override(metrics, ground_truth)

        self.assertEqual(out["sched_ir_total_luts"], 100)
        self.assertEqual(out["sched_ir_total_ffs"], 200)
        self.assertEqual(out["sched_ir_makespan"], 10)
        self.assertEqual(out["sched_ir_pipeline_depth"], 8)
        self.assertEqual(out["sched_ir_batches_in_flight"], 8)
        self.assertEqual(out["total_luts"], 120)
        self.assertEqual(out["total_ffs"], 240)
        self.assertEqual(out["makespan"], 12)
        self.assertEqual(out["pipeline_depth"], 12)
        self.assertEqual(out["batches_in_flight"], 12)
        self.assertTrue(out["ground_truth"])

    def test_build_k1_validation_rows_computes_delta_percent(self):
        metrics = {
            "K": 1,
            "sched_ir_total_luts": 100,
            "sched_ir_total_ffs": 200,
            "sched_ir_makespan": 10,
            "sched_ir_pipeline_depth": 10,
            "sched_ir_batches_in_flight": 10,
            "II": 1,
            "throughput_mhz": 300.0,
        }
        ground_truth = {"lut": 120, "ff": 240, "stages": 12}

        rows = build_k1_validation_rows(metrics, ground_truth, 300e6)

        self.assertEqual(rows[0]["metric"], "LUTs")
        self.assertEqual(rows[0]["sched_ir"], 100)
        self.assertEqual(rows[0]["ground_truth"], 120)
        self.assertEqual(rows[0]["delta"], -20)
        self.assertAlmostEqual(rows[0]["delta_pct"], -16.6666666667)
        self.assertEqual(rows[3]["metric"], "II")
        self.assertEqual(rows[3]["ground_truth"], 1)
        self.assertEqual(rows[4]["metric"], "Throughput (MHz)")
        self.assertEqual(rows[4]["ground_truth"], 300.0)


if __name__ == "__main__":
    unittest.main()
