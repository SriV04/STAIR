import json
from pathlib import Path


def test_stair_evaluation_notebook_uses_real_pipeline_not_scaffold():
    notebook = json.loads(Path("notebooks/stair_evaluation_graphs.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "make_scaffold_stair_sweep" not in source
    assert "api.evaluate_folded_design" in source
    assert "api.check_symbolic_correctness" in source
    assert "fold_factor == 1" in source
    assert "compile=False" in source
    assert "pipeline_status" in source
    assert "except Exception as exc" in source


def test_stair_evaluation_pareto_uses_scheduled_length_t_axis():
    notebook = json.loads(Path("notebooks/stair_evaluation_graphs.ipynb").read_text())
    pareto_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "Pareto frontier" in "".join(cell.get("source", []))
    )

    assert 'plot_df["schedule_length_t"] = plot_df["latency_cycles"]' in pareto_source
    assert 'pareto_mask(plot_df, x="schedule_length_t")' in pareto_source
    assert 'frontier = plot_df[plot_df["pareto"]].sort_values("schedule_length_t")' in pareto_source
    assert 'ax.set_xlabel("Schedule length (t)")' in pareto_source


def test_stair_evaluation_notebook_summarises_all_model_graph_counts():
    notebook = json.loads(Path("notebooks/stair_evaluation_graphs.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "graph_topology_summary_all_models" in source
    assert "_summarise_all_graph_topologies" in source
    assert "for _, model_row in real_df.iterrows()" in source
    assert '"Benchmark"' in source
    assert '"Model size"' in source
    assert '"NN-IR nodes"' in source
    assert '"NN-IR edges"' in source
    assert '"Sched-IR nodes"' in source
    assert '"Sched-IR edges"' in source
    assert '"Fold groups"' in source
    assert '"Fold axes"' in source
    assert "task_count_by_fold" in source
    assert '"Min task count"' in source
    assert '"Min task count fold"' in source
    assert '"Max task count"' in source
    assert '"Max task count fold"' in source
    assert '"Max fold factor"' in source


def test_stair_evaluation_notebook_plots_jedi_luts_by_particles():
    notebook = json.loads(Path("notebooks/stair_evaluation_graphs.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "jedi_lut_by_particles_df" in source
    assert 'group["particles"]' in source
    assert 'group["stair_lut"]' in source
    assert 'real_group["lut"]' in source
    assert 'linestyle="--"' in source
    assert 'ax.set_xlabel("Particles")' in source
    assert "jedi_luts_vs_particles_path" in source
    assert 'fig.savefig(jedi_luts_vs_particles_path, dpi=200)' in source


def test_stair_evaluation_notebook_plots_jedi_lut_latency_ii_and_throughput_proxies():
    notebook = json.loads(Path("notebooks/stair_evaluation_graphs.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "jedi_tradeoff_df" in source
    assert 'jedi_tradeoff_df["ii_proxy_per_temporal_step"]' in source
    assert 'jedi_tradeoff_df["normalised_throughput_proxy"]' in source
    assert 'x_col="stair_lut"' in source
    assert 'y_col="latency_cycles"' in source
    assert 'y_col="ii_proxy_per_temporal_step"' in source
    assert 'y_col="normalised_throughput_proxy"' in source
    assert 'xlabel="STAIR-estimated LUTs"' in source
    assert 'ylabel="Scheduled latency cycles"' in source
    assert 'ylabel="Effective II proxy / temporal steps"' in source
    assert 'ylabel="Normalised throughput proxy (1 / II proxy)"' in source
    assert "jedi_tradeoff_plot_paths" in source
    assert "jedi_lut_vs_scheduled_latency_path" in source
    assert "jedi_lut_vs_ii_proxy_per_temporal_step_path" in source
    assert "jedi_lut_vs_normalised_throughput_proxy_path" in source


def test_stair_evaluation_notebook_plots_mean_sync_wait_by_fold_and_particles():
    notebook = json.loads(Path("notebooks/stair_evaluation_graphs.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "mean_sync_wait_by_fold_df" in source
    assert '.dropna(subset=["fold_factor", "particles", "mean_sync_wait_cycles"])' in source
    assert "def _feature_set_sort_key(feature_set)" in source
    assert "feature_sets = sorted(mean_sync_wait_by_fold_df[\"feature_set\"].dropna().unique(), key=_feature_set_sort_key)" in source
    assert "for feature_set, ax in zip(feature_sets, axes)" in source
    assert "for idx, particles in enumerate(sorted(feature_frame[\"particles\"].dropna().astype(int).unique()))" in source
    assert 'label=f"N={particles}"' in source
    assert 'group["fold_factor"]' in source
    assert 'group["mean_sync_wait_cycles"]' in source
    assert 'ax.set_xlabel("Fold factor")' in source
    assert 'axes[0].set_ylabel("Mean sync wait cycles")' in source
    assert "jedi_mean_sync_wait_vs_fold_factor_path" in source
    assert 'fig.savefig(jedi_mean_sync_wait_vs_fold_factor_path, dpi=200)' in source


def test_stair_evaluation_notebook_has_e3_validity_checking_tables():
    notebook = json.loads(Path("notebooks/stair_evaluation_graphs.ipynb").read_text())
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
