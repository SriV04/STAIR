"""Generate a side-by-side DA4ML vs Sched-IR K=1 cost trace report.

Run from the repo root:

    KERAS_BACKEND=jax conda run -n jedi-linear python ir/cost_trace_compare.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_REPORT = HERE / "cost_trace_report.txt"

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "JEDI-linear" / "src"))
sys.path.insert(0, str(REPO / "heterograph"))

N_CONSTITUENTS = 8
USE_PERMINV = True
LOAD_TRAINED_WEIGHTS = True
TARGET_FMAX_HZ = 300e6

from IR.evaluate_helpers import summarize_precision_payload


def format_delta(actual: int | float | None, expected: int | float | None) -> str:
    if actual is None or expected in (None, 0):
        return "?"
    delta = actual - expected
    pct = (delta / expected) * 100.0
    if isinstance(delta, float) or isinstance(expected, float) or isinstance(actual, float):
        return f"{delta:+.1f} ({pct:+.1f}%)"
    return f"{delta:+,} ({pct:+.1f}%)"


def render_section(title: str, lines: list[str]) -> str:
    body = "\n".join(lines) if lines else "(empty)"
    return f"{'=' * 80}\n{title}\n{'=' * 80}\n{body}\n"


def render_metric_row(metric: str, sched_ir: int | float | None, ground_truth: int | float | None) -> str:
    def _fmt(value: int | float | None) -> str:
        if value is None:
            return "?"
        if isinstance(value, float):
            return f"{value:,.1f}"
        return f"{value:,}"

    delta = "?" if sched_ir is None or ground_truth is None else format_delta(sched_ir, ground_truth)
    return f"{metric:>18} | {_fmt(sched_ir):>12} | {_fmt(ground_truth):>12} | {delta:>18}"


def rollup_multiplier(op: str | None, physical_instances: int | None) -> int:
    if op in ("reduce", "elementwise", "buffer", "mux"):
        return 1
    return max(int(physical_instances or 1), 1)


def rolled_cost_value(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0) * rollup_multiplier(row.get("op"), row.get("physical_instances"))


def ff_without_local_reg_bits(row: dict[str, Any]) -> int:
    ff = int(row.get("ff") or 0)
    reg_bits = row.get("reg_bits")
    if reg_bits is None:
        return ff
    return max(ff - int(reg_bits), 0)

def _runtime() -> SimpleNamespace:
    import yaml as _yaml
    import keras
    import hgq  # noqa: F401
    from IR.nn_ir.builder import build_nn_ir
    from IR.sched_ir import binder as sched_engine
    from IR.sched_ir import decomposer as sched_decomp
    from IR.sched_ir import precision as sched_precision
    from IR.sched_ir.costing import da4ml as sched_da4ml
    from IR.sched_ir.folding import fold_precision as sched_fold_precision
    from IR.sched_ir.folding import folder as sched_folder
    from IR.sched_ir.resource import DA4ML_RESOURCE_YAML as resource_yaml
    from IR.sched_ir.scheduling import infrastructure as sched_infra
    from IR.sched_ir.scheduling import scheduler_p3 as sched_p3
    from heterograph import HGraph
    from model import get_gnn
    from da4ml.converter.hgq2.parser import parse_model, trace_model
    from da4ml.trace import FixedVariableArrayInput, HWConfig, comb_trace
    from da4ml.trace.pipeline import to_pipeline
    from da4ml.cmvm.types import CascadedSolution

    return SimpleNamespace(
        build_nn_ir=build_nn_ir,
        sched_decomp=sched_decomp,
        sched_engine=sched_engine,
        sched_precision=sched_precision,
        sched_fold_precision=sched_fold_precision,
        sched_folder=sched_folder,
        sched_p3=sched_p3,
        sched_infra=sched_infra,
        sched_da4ml=sched_da4ml,
        resource_yaml=resource_yaml,
        yaml=_yaml,
        keras=keras,
        HGraph=HGraph,
        get_gnn=get_gnn,
        parse_model=parse_model,
        trace_model=trace_model,
        FixedVariableArrayInput=FixedVariableArrayInput,
        HWConfig=HWConfig,
        comb_trace=comb_trace,
        to_pipeline=to_pipeline,
        CascadedSolution=CascadedSolution,
    )


def _checkpoint_glob() -> Path:
    variant_dir = "3-feature-perminv" if USE_PERMINV else "3-feature"
    return (
        REPO / "official_models" / variant_dir / f"jet_classifier_large_{N_CONSTITUENTS}" / "models"
    )


def _load_model(rt: SimpleNamespace):
    if LOAD_TRAINED_WEIGHTS:
        ckpts = sorted(_checkpoint_glob().glob("*.keras"))
        if ckpts:
            return rt.keras.models.load_model(ckpts[0]), ckpts[0].name
    conf = SimpleNamespace(n_constituents=N_CONSTITUENTS, pt_eta_phi=True)
    return rt.get_gnn(conf, uq1=USE_PERMINV), "fresh"


def _format_float(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:.1f}"


def _summarize_stage(stage, stage_index: int, rt: SimpleNamespace) -> dict[str, Any]:
    single_stage = rt.CascadedSolution((stage,))
    lat_min, lat_max = stage.latency
    return {
        "index": stage_index,
        "shape": tuple(stage.shape),
        "cost": float(stage.cost),
        "latency_min": float(lat_min),
        "latency_max": float(lat_max),
        "n_ops": len(stage.ops),
        "n_out": len(stage.out_idxs),
        "reg_bits": int(single_stage.reg_bits),
    }


def _safe_pipeline(sol, cutoff: int, rt: SimpleNamespace):
    try:
        return rt.to_pipeline(sol, cutoff, retiming=True, verbose=False), "retiming=True"
    except Exception:
        return rt.to_pipeline(sol, cutoff, retiming=False, verbose=False), "retiming=False"


def _capture_da4ml_trace(model, fpga_cfg: dict[str, Any], rt: SimpleNamespace) -> dict[str, Any]:
    cutoff = int(fpga_cfg.get("latency_cutoff", -1))
    hwconf = rt.HWConfig(1, -1, -1)
    solver_options = {"hard_dc": 2}
    explicit_inputs = tuple(
        rt.FixedVariableArrayInput(inp.shape[1:], hwconf=hwconf, solver_options=solver_options)
        for inp in model.inputs
    )

    raw_stdout = io.StringIO()
    with contextlib.redirect_stdout(raw_stdout):
        rt.trace_model(
            model,
            solver_options=solver_options,
            hwconf=hwconf,
            verbose=True,
            inputs=explicit_inputs,
        )
    raw_verbose = raw_stdout.getvalue().splitlines()

    dumped = rt.trace_model(
        model,
        solver_options=solver_options,
        hwconf=hwconf,
        verbose=False,
        dump=True,
        inputs=explicit_inputs,
    )
    parser_mod = sys.modules[rt.parse_model.__module__]
    flatten_arr = parser_mod._flatten_arr
    flat_inputs = flatten_arr(explicit_inputs)
    flat_outputs = flatten_arr([dumped[tensor.name] for tensor in model.outputs])

    full_sol = rt.comb_trace(flat_inputs, flat_outputs)
    full_pipe, full_pipe_mode = _safe_pipeline(full_sol, cutoff, rt)

    ops_summary = []
    for depth, ops in enumerate(rt.parse_model(model)):
        for op_index, op in enumerate(ops):
            op_name = op.operation.name
            op_class = op.operation.__class__.__name__
            if op_class == "InputLayer":
                continue
            produced_names = [tensor.name for tensor in op.produces]
            required_names = [tensor.name for tensor in op.requires]
            produced = [dumped[name] for name in produced_names]
            required = [dumped[name] for name in required_names]

            cumulative_sol = rt.comb_trace(flat_inputs, flatten_arr(produced))
            cumulative_pipe, cumulative_pipe_mode = _safe_pipeline(cumulative_sol, cutoff, rt)

            local_sol = None
            local_result = None
            local_pipe_mode = None
            if required:
                try:
                    local_sol = rt.comb_trace(flatten_arr(required), flatten_arr(produced))
                    local_result = rt.sched_da4ml.solution_to_result(local_sol, cutoff)
                    local_pipe_mode = "adapter"
                except Exception as exc:
                    local_result = {"error": str(exc)}

            ops_summary.append(
                {
                    "depth": depth,
                    "op_index": op_index,
                    "name": op_name,
                    "class": op_class,
                    "requires": required_names,
                    "produces": produced_names,
                    "cumulative_shape": tuple(cumulative_sol.shape),
                    "cumulative_cost": float(cumulative_sol.cost),
                    "cumulative_latency": tuple(map(float, cumulative_sol.latency)),
                    "cumulative_pipeline_stages": len(cumulative_pipe.solutions),
                    "cumulative_pipeline_cost": float(cumulative_pipe.cost),
                    "cumulative_pipeline_reg_bits": int(cumulative_pipe.reg_bits),
                    "cumulative_pipeline_mode": cumulative_pipe_mode,
                    "local_shape": tuple(local_sol.shape) if local_sol is not None else None,
                    "local_result": local_result,
                    "local_pipe_mode": local_pipe_mode,
                }
            )

    stage_summaries = [
        _summarize_stage(stage, idx, rt) for idx, stage in enumerate(full_pipe.solutions)
    ]

    return {
        "raw_verbose": raw_verbose,
        "full_solution": full_sol,
        "full_pipeline": full_pipe,
        "full_pipe_mode": full_pipe_mode,
        "full_solution_debug": rt.sched_da4ml._solution_debug_summary(full_sol),
        "full_pipeline_debug": rt.sched_da4ml._solution_debug_summary(full_pipe),
        "stage_summaries": stage_summaries,
        "ops_summary": ops_summary,
    }


def _capture_sched_trace(model, g_nnir, fpga_cfg: dict[str, Any], rt: SimpleNamespace) -> dict[str, Any]:
    cutoff = int(fpga_cfg.get("latency_cutoff", -1))
    g = rt.sched_decomp.decompose_nn_to_sched(g_nnir)
    g = rt.sched_folder.stamp_fold_plan(g, factor=1)
    os.environ["CMIR_DEBUG_DA4ML"] = "1"

    cfg = rt.yaml.safe_load(rt.resource_yaml.read_text())
    fpga = rt.sched_engine.normalize_fpga(cfg.get("fpga") or {})
    cfg["fpga"] = fpga
    library = rt.sched_engine.build_kernel_library(cfg)
    weights = rt.sched_engine.WeightProvider(model)
    g.pmap["resource_yaml"] = str(rt.resource_yaml.resolve())
    g.pmap["target_device"] = fpga.get("device")
    g.pmap["fpga_config"] = fpga

    raw_lines: list[str] = []
    next_instance: dict[str, int] = {}
    vertex_rows: list[dict[str, Any]] = []

    for vx in rt.sched_engine._topo_order(g):
        p = g.pmap[vx]
        prim = p.get("op")
        if prim not in rt.sched_engine._NEEDS_BIND:
            continue

        rt.sched_engine._ingest_inputs_from_edges(g, vx)
        in_qint = summarize_precision_payload(p.get("input_qints"))
        in_kif = summarize_precision_payload(p.get("input_kifs"))
        layer = p.get("nn_layer_name")

        candidates = library.get(prim) or []
        chosen = rt.sched_engine._select_kernel(p, candidates)
        raw_result = chosen.cost_query(p, weights, fpga)
        result = rt.sched_engine._kernel_result.normalize_kernel_result(raw_result, source="closed_form")
        da4ml_info = result.get("da4ml") or {}
        solve_debug = da4ml_info.get("solve_debug")
        pipe_debug = da4ml_info.get("pipe_debug")
        if solve_debug or pipe_debug:
            raw_lines.append(
                f"[bind da4ml] vx={vx} layer={layer} op={prim} "
                f"solve={solve_debug} pipe={pipe_debug}"
            )

        p["kernel_type"] = chosen.name
        p["kernel_instance"] = next_instance.setdefault(chosen.name, 0)
        next_instance[chosen.name] += 1
        rt.sched_engine._apply_kernel_result(p, result)
        rt.sched_engine._propagate_outputs_to_edges(g, vx)
        out_qint = summarize_precision_payload(p.get("output_qints"))
        out_kif = summarize_precision_payload(p.get("output_kifs"))
        raw_lines.append(
            f"[bind] vx={vx} layer={layer} op={prim} "
            f"in_qint={in_qint} in_kif={in_kif} out_qint={out_qint} out_kif={out_kif}"
        )

    rt.sched_engine._validate_bind(g)

    g = rt.sched_precision.propagate_precision(g)
    g = rt.sched_fold_precision.apply_fold_aware_precision(g)
    g = rt.sched_folder.apply_timing_from_costs(g)
    g = rt.sched_p3.schedule(g)
    g = rt.sched_p3.steady_state(g, fmax=TARGET_FMAX_HZ)
    g = rt.sched_infra.insert_buffers(g)

    for vx in rt.sched_p3._topo_sort(g):
        p = g.pmap[vx]
        if p.get("op") is None:
            continue
        cost = p.get("cost") or {}
        kr = p.get("kernel_result") or {}
        da4ml_info = kr.get("da4ml") or {}
        reg_bits = cost.get("reg_bits")
        vertex_rows.append(
            {
                "vx": vx,
                "layer": p.get("nn_layer_name"),
                "op": p.get("op"),
                "kernel_type": p.get("kernel_type"),
                "physical_instances": int(p.get("physical_instances") or 1),
                "fold_factor": int(p.get("fold_factor") or 1),
                "latency_cycles": int(cost.get("latency_cycles") or 0),
                "lut": int(cost.get("lut") or 0),
                "ff": int(cost.get("ff") or 0),
                "dsp": int(cost.get("dsp") or 0),
                "bram": int(cost.get("bram") or 0),
                "reg_bits": int(reg_bits) if reg_bits is not None else None,
                "t_start": p.get("t_start"),
                "t_end": p.get("t_end"),
                "critical_path": bool(p.get("critical_path")),
                "input_qints": summarize_precision_payload(p.get("input_qints")),
                "output_qints": summarize_precision_payload(p.get("output_qints")),
                "da4ml": da4ml_info,
            }
        )

    return {
        "graph": g,
        "raw_lines": raw_lines,
        "vertex_rows": vertex_rows,
        "totals": {
            "lut": int(g.pmap.get("total_luts") or 0),
            "ff": int(g.pmap.get("total_ffs") or 0),
            "makespan": int(g.pmap.get("makespan") or 0),
            "ii": int(g.pmap.get("initiation_interval") or 0),
            "throughput_mhz": (g.pmap.get("sustained_throughput_hz") or 0.0) / 1e6,
            "buffers": sum(1 for v in g.vertices if g.pmap[v].get("op") == "buffer"),
            "muxes": sum(1 for v in g.vertices if g.pmap[v].get("op") == "mux"),
        },
        "derived_totals": {
            "rolled_lut": sum(rolled_cost_value(row, "lut") for row in vertex_rows),
            "rolled_ff": sum(rolled_cost_value(row, "ff") for row in vertex_rows),
            "rolled_ff_no_reg_bits": sum(
                ff_without_local_reg_bits(row) * rollup_multiplier(row.get("op"), row.get("physical_instances"))
                for row in vertex_rows
            ),
        },
    }


def _da4ml_layer_map(da4ml_trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in da4ml_trace["ops_summary"]:
        out[row["name"]] = row
    return out


def _render_da4ml_full_summary(da4ml_trace: dict[str, Any]) -> str:
    full_sol = da4ml_trace["full_solution"]
    full_pipe = da4ml_trace["full_pipeline"]
    lat_min, lat_max = full_pipe.latency
    lines = [
        f"raw solution: shape={tuple(full_sol.shape)} cost={full_sol.cost:.1f} latency={full_sol.latency}",
        f"pipelined: stages={len(full_pipe.solutions)} cost={full_pipe.cost:.1f} reg_bits={int(full_pipe.reg_bits)} "
        f"latency=({_format_float(lat_min)}, {_format_float(lat_max)}) mode={da4ml_trace['full_pipe_mode']}",
        f"solution debug: {da4ml_trace['full_solution_debug']}",
        f"pipeline debug: {da4ml_trace['full_pipeline_debug']}",
        "",
        "stage summaries:",
    ]
    for stage in da4ml_trace["stage_summaries"]:
        lines.append(
            f"  stage {stage['index']:>2}: shape={stage['shape']} ops={stage['n_ops']:>4} out={stage['n_out']:>4} "
            f"cost={stage['cost']:>10.1f} reg_bits={stage['reg_bits']:>8} "
            f"lat=({_format_float(stage['latency_min'])}, {_format_float(stage['latency_max'])})"
        )
    return render_section("DA4ML FULL MODEL SUMMARY", lines)


def _render_da4ml_per_op(da4ml_trace: dict[str, Any]) -> str:
    lines = []
    for row in da4ml_trace["ops_summary"]:
        local_result = row["local_result"] or {}
        local_cost = local_result.get("cost") or {}
        lines.append(
            f"{row['name']} [{row['class']}] depth={row['depth']} op_index={row['op_index']}"
        )
        lines.append(f"  requires: {row['requires']}")
        lines.append(f"  produces: {row['produces']}")
        lines.append(
            f"  cumulative: shape={row['cumulative_shape']} comb_cost={row['cumulative_cost']:.1f} "
            f"comb_latency={row['cumulative_latency']} pipe_stages={row['cumulative_pipeline_stages']} "
            f"pipe_cost={row['cumulative_pipeline_cost']:.1f} pipe_reg_bits={row['cumulative_pipeline_reg_bits']} "
            f"pipe_mode={row['cumulative_pipeline_mode']}"
        )
        if "error" in local_result:
            lines.append(f"  local: ERROR {local_result['error']}")
        else:
            lines.append(
                f"  local: shape={row['local_shape']} lut={int(local_cost.get('lut') or 0)} "
                f"ff={int(local_cost.get('ff') or 0)} lat={int(local_cost.get('latency_cycles') or 0)} "
                f"reg_bits={local_cost.get('reg_bits')} mode={row['local_pipe_mode']}"
            )
    return render_section("DA4ML PER-KERAS-OP TRACE", lines)


def _render_sched_raw(sched_trace: dict[str, Any]) -> str:
    return render_section("SCHED-IR RAW BIND TRACE", sched_trace["raw_lines"])


def _render_sched_vertices(sched_trace: dict[str, Any]) -> str:
    lines = []
    for row in sched_trace["vertex_rows"]:
        raw_ff_no_reg = ff_without_local_reg_bits(row)
        mult = rollup_multiplier(row.get("op"), row.get("physical_instances"))
        lines.append(
            f"vx={row['vx']:>2} layer={row['layer']} op={row['op']} kernel={row['kernel_type']} "
            f"P={row['physical_instances']} T={row['fold_factor']} "
            f"lut={row['lut']} ff={row['ff']} dsp={row['dsp']} bram={row['bram']} "
            f"lat={row['latency_cycles']} t=[{row['t_start']}, {row['t_end']}] "
            f"critical={row['critical_path']}"
        )
        lines.append(
            f"  rolled: mult={mult} lut={rolled_cost_value(row, 'lut')} ff={rolled_cost_value(row, 'ff')} "
            f"ff_no_local_reg_bits={raw_ff_no_reg * mult}"
        )
        lines.append(
            f"  precision: in={row['input_qints']} out={row['output_qints']}"
        )
        if row["da4ml"]:
            lines.append(
                f"  da4ml: n_in={row['da4ml'].get('n_inputs')} n_out={row['da4ml'].get('n_outputs')} "
                f"shape={row['da4ml'].get('shape')} stages={row['da4ml'].get('pipeline_stages')} "
                f"reg_bits={row['da4ml'].get('reg_bits')} raw_reg_bits={row['reg_bits']}"
            )
            if row["da4ml"].get("logical_debug"):
                lines.append(f"  da4ml logical_debug: {row['da4ml']['logical_debug']}")
            if row["da4ml"].get("pipe_debug"):
                lines.append(f"  da4ml pipe_debug: {row['da4ml']['pipe_debug']}")
    return render_section("SCHED-IR VERTEX TRACE", lines)


def _render_comparison(da4ml_trace: dict[str, Any], sched_trace: dict[str, Any]) -> str:
    layer_map = _da4ml_layer_map(da4ml_trace)
    derived = sched_trace["derived_totals"]
    lines = [
        f"{'Metric':>18} | {'Sched-IR':>12} | {'DA4ML GT':>12} | {'Delta':>18}",
        "-" * 80,
    ]
    full_pipe = da4ml_trace["full_pipeline"]
    totals = sched_trace["totals"]
    lines.append(render_metric_row("LUTs", totals["lut"], int(round(float(full_pipe.cost)))))
    lines.append(render_metric_row("FFs", totals["ff"], int(full_pipe.reg_bits)))
    lines.append(render_metric_row("FFs (no regbits)", derived["rolled_ff_no_reg_bits"], int(full_pipe.reg_bits)))
    lines.append(render_metric_row("Makespan", totals["makespan"], len(full_pipe.solutions)))
    lines.append(render_metric_row("II", totals["ii"], 1))
    lines.append(render_metric_row("Throughput (MHz)", totals["throughput_mhz"], TARGET_FMAX_HZ / 1e6))
    lines.append("")
    lines.append(
        f"rolled check: graph_lut={totals['lut']:,} graph_ff={totals['ff']:,} "
        f"derived_lut={derived['rolled_lut']:,} derived_ff={derived['rolled_ff']:,}"
    )
    lines.append("")
    lines.append(
        f"{'Layer':<32} | {'Sched LUT raw':>13} | {'Sched LUT roll':>14} | {'DA local LUT':>12} | "
        f"{'Sched FF raw':>12} | {'Sched FF roll':>13} | {'Sched FF noRB':>13} | {'DA local FF':>11} | "
        f"{'Sched lat':>9} | {'DA local lat':>12}"
    )
    lines.append("-" * 170)

    for row in sched_trace["vertex_rows"]:
        layer = row["layer"]
        da_row = layer_map.get(layer)
        local_cost = ((da_row or {}).get("local_result") or {}).get("cost") or {}
        ff_no_reg = ff_without_local_reg_bits(row)
        lines.append(
            f"{layer:<32} | {row['lut']:>13,} | {rolled_cost_value(row, 'lut'):>14,} | {int(local_cost.get('lut') or 0):>12,} | "
            f"{row['ff']:>12,} | {rolled_cost_value(row, 'ff'):>13,} | "
            f"{ff_no_reg * rollup_multiplier(row.get('op'), row.get('physical_instances')):>13,} | {int(local_cost.get('ff') or 0):>11,} | "
            f"{row['latency_cycles']:>9} | {int(local_cost.get('latency_cycles') or 0):>12}"
        )
    return render_section("SIDE-BY-SIDE COMPARISON", lines)


def _render_runtime_header(model_name: str, fpga_cfg: dict[str, Any]) -> str:
    lines = [
        f"repo: {REPO}",
        f"resource_yaml: {rt.resource_yaml}",
        f"model: {model_name}",
        f"target_fmax_hz: {float(fpga_cfg.get('target_fmax_hz') or TARGET_FMAX_HZ)}",
        f"latency_cutoff: {int(fpga_cfg.get('latency_cutoff', -1))}",
        f"weights: {'trained' if LOAD_TRAINED_WEIGHTS else 'fresh'}",
    ]
    return render_section("TRACE CONFIG", lines)


def generate_report(output_path: Path) -> Path:
    rt = _runtime()
    model, model_name = _load_model(rt)
    g_nnir = rt.build_nn_ir(model, name="jedi_gnn")

    cfg = rt.yaml.safe_load(rt.resource_yaml.read_text())
    fpga_cfg = rt.sched_engine.normalize_fpga(cfg.get("fpga") or {})

    da4ml_trace = _capture_da4ml_trace(model, fpga_cfg, rt)
    sched_trace = _capture_sched_trace(model, g_nnir, fpga_cfg, rt)

    report = "\n".join(
        [
            _render_runtime_header(model_name, fpga_cfg),
            render_section("RAW DA4ML TRACE_MODEL(verbose=True)", da4ml_trace["raw_verbose"]),
            _render_da4ml_full_summary(da4ml_trace),
            _render_da4ml_per_op(da4ml_trace),
            _render_sched_raw(sched_trace),
            _render_sched_vertices(sched_trace),
            _render_comparison(da4ml_trace, sched_trace),
        ]
    )
    output_path.write_text(report)
    return output_path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to the text report to write.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = generate_report(args.output.resolve())
    print(f"Wrote report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
