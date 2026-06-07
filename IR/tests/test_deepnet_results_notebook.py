import json
from pathlib import Path


def test_deepnet_results_notebook_displays_stair_lut_reduction_by_fold_factor():
    notebook = json.loads(Path("notebooks/deepnet_results.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "fold_lut_reduction_table" in source
    assert "sys.path.insert(0, str(REPO_ROOT))" in source
    assert 'stair_baseline_lut = stair_results[stair_results["fold_factor"] == 1]' in source
    assert 'stair_lut_reduction_pct' in source
    assert '1 - fold_lut_reduction["estimated_lut"] / fold_lut_reduction["baseline_stair_lut"]' in source
    assert 'groupby("fold_factor", as_index=False)' in source
    assert 'display(fold_lut_reduction_table)' in source


def test_deepnet_results_notebook_plots_real_and_estimated_luts_by_fold_factor():
    notebook = json.loads(Path("notebooks/deepnet_results.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "lut_by_fold_df" in source
    assert 'group["fold_factor"]' in source
    assert 'group["estimated_lut"]' in source
    assert 'real_group["real_lut"]' in source
    assert 'linestyle="--"' in source
    assert "luts_vs_fold_factor_path" in source
    assert 'fig.savefig(luts_vs_fold_factor_path, dpi=200)' in source


def test_deepnet_results_notebook_records_schedule_latency_from_metrics():
    notebook = json.loads(Path("notebooks/deepnet_results.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert '"schedule_latency_cycles": metrics.get("latency_cycles")' in source
    assert "da4ml_combtrace_latency_proxy" not in source
    assert "da4ml_combtrace_latency_proxy_cycles" not in source
    assert "da4ml_combtrace_resource_count" not in source


def test_deepnet_results_notebook_plots_schedule_latency_against_fold_groups():
    notebook = json.loads(Path("notebooks/deepnet_results.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "schedule_latency_by_fold_df" in source
    assert '"fold_group_count"' in source
    assert '"schedule_latency_cycles"' in source
    assert 'ax.set_xlabel("Fold groups")' in source
    assert 'ax.set_ylabel("Schedule latency cycles (t0 to t_end)")' in source
    assert 'schedule_latency_vs_fold_groups_path' in source
    assert 'fig.savefig(schedule_latency_vs_fold_groups_path, dpi=200)' in source
