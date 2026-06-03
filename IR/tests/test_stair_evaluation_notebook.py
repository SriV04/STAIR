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
