"""Helpers for evaluate.py that are small enough to unit test directly."""

from __future__ import annotations


def summarize_precision_payload(payload) -> str:
    if payload is None:
        return "None"
    if isinstance(payload, dict):
        if {"min", "max", "step"}.issubset(payload):
            return "qint"
        if {"k", "i", "f"}.issubset(payload):
            return "kif"
        return f"dict[{','.join(sorted(payload.keys()))}]"
    if isinstance(payload, (list, tuple)):
        if not payload:
            return f"{type(payload).__name__}[0]"
        return (
            f"{type(payload).__name__}[{len(payload)}]"
            f"<{summarize_precision_payload(payload[0])}>"
        )
    return type(payload).__name__


def apply_k1_ground_truth_override(metrics: dict, ground_truth: dict | None) -> dict:
    out = dict(metrics)
    if out.get("K") != 1 or ground_truth is None:
        return out

    out["sched_ir_total_luts"] = out.get("total_luts")
    out["sched_ir_total_ffs"] = out.get("total_ffs")
    out["sched_ir_makespan"] = out.get("makespan")
    out["sched_ir_pipeline_depth"] = out.get("pipeline_depth")
    out["sched_ir_batches_in_flight"] = out.get("batches_in_flight")

    out["total_luts"] = int(ground_truth["lut"])
    out["total_ffs"] = int(ground_truth["ff"])
    out["makespan"] = int(ground_truth["stages"])
    out["pipeline_depth"] = int(ground_truth["stages"])
    out["batches_in_flight"] = int(ground_truth["stages"])
    out["ground_truth"] = True
    return out


def build_k1_validation_rows(metrics: dict, ground_truth: dict | None, target_fmax_hz: float) -> list[dict]:
    if metrics.get("K") != 1 or ground_truth is None:
        return []
    if "sched_ir_total_luts" not in metrics:
        return []

    gt_tput_mhz = float(target_fmax_hz) / 1e6 if target_fmax_hz else None
    rows = [
        ("LUTs", metrics["sched_ir_total_luts"], int(ground_truth["lut"])),
        ("FFs", metrics["sched_ir_total_ffs"], int(ground_truth["ff"])),
        ("Makespan (cyc)", metrics["sched_ir_makespan"], int(ground_truth["stages"])),
        ("II", metrics.get("II"), 1),
        ("Throughput (MHz)", metrics.get("throughput_mhz"), gt_tput_mhz),
    ]

    out = []
    for metric_name, sched_ir_value, gt_value in rows:
        delta = None
        delta_pct = None
        if sched_ir_value is not None and gt_value not in (None, 0):
            delta = sched_ir_value - gt_value
            delta_pct = (delta / gt_value) * 100.0
        out.append(
            {
                "metric": metric_name,
                "sched_ir": sched_ir_value,
                "ground_truth": gt_value,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )
    return out
