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
    assert 'stair_baseline_lut = stair_results[stair_results["fold_factor"] == 1]' in source
    assert 'stair_lut_reduction_pct' in source
    assert '1 - fold_lut_reduction["estimated_lut"] / fold_lut_reduction["baseline_stair_lut"]' in source
    assert 'groupby("fold_factor", as_index=False)' in source
    assert 'display(fold_lut_reduction_table)' in source
