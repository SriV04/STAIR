import json
from pathlib import Path


DEEPSET_NOTEBOOK = Path("notebooks/deepset_results.ipynb")


def test_deepnet_results_notebook_displays_stair_lut_reduction_by_fold_factor():
    notebook = json.loads(DEEPSET_NOTEBOOK.read_text())
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
    notebook = json.loads(DEEPSET_NOTEBOOK.read_text())
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
    notebook = json.loads(DEEPSET_NOTEBOOK.read_text())
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
    notebook = json.loads(DEEPSET_NOTEBOOK.read_text())
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


def test_deepnet_results_notebook_plots_lut_latency_ii_and_throughput_proxies():
    notebook = json.loads(DEEPSET_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "deepset_tradeoff_df" in source
    assert 'deepset_tradeoff_df["ii_proxy_per_temporal_step"]' in source
    assert 'deepset_tradeoff_df["normalised_throughput_proxy"]' in source
    assert 'x_col="estimated_lut"' in source
    assert 'y_col="schedule_latency_cycles"' in source
    assert 'y_col="ii_proxy_per_temporal_step"' in source
    assert 'y_col="normalised_throughput_proxy"' in source
    assert 'xlabel="STAIR-estimated LUTs"' in source
    assert 'ylabel="Scheduled latency cycles"' in source
    assert 'ylabel="Effective II proxy / temporal steps"' in source
    assert 'ylabel="Normalised throughput proxy (1 / II proxy)"' in source
    assert "deepset_tradeoff_plot_paths" in source
    assert "deepset_lut_vs_scheduled_latency_path" in source
    assert "deepset_lut_vs_ii_proxy_per_temporal_step_path" in source
    assert "deepset_lut_vs_normalised_throughput_proxy_path" in source


def test_deepnet_results_notebook_has_e3_validity_checking_tables():
    notebook = json.loads(DEEPSET_NOTEBOOK.read_text())
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "## E3 Validity Checking" in markdown
    assert "e3_fold_plan_legality_table" in source
    assert "e3_schedule_legality_table" in source
    assert "e3_output_quantization_table" in source
    assert "e3_fold_group_semantics_table" in source
    assert "e3_representative_slice_reconstruction_table" in source
    assert "e3_sync_analysis_table" in source
    assert '"T == ceil(N/P)"' in source
    assert '"legal_schedule"' in source
    assert '"reference_output_widths"' in source
    assert '"scheduled_output_widths"' in source
    assert '"fold_group_semantic_failure_count"' in source
    assert '"sync_point_count"' in source
    assert '"total_input_buffer_wait_cycles"' in source
    assert '"sync_legality_passed"' in source
    assert '"sync_violations"' in source
