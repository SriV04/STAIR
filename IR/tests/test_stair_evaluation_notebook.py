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
