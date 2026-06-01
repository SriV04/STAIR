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
