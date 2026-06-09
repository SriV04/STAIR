"""
Full test pipeline for folded design evaluation.
Creates NN-IR, evaluates folded design with fold factor 2 (4 lanes),
and generates cost analysis per layer and task.
"""

import sys
import re
import tempfile
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import numpy as np

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from IR import nn_ir
from IR.nn_ir.styling import apply_nn_style
from IR.sched_ir import api
from IR.sched_ir.graphing.styling import apply_sched_style
import keras
import hgq

from heterograph.webview import WebView


def _slugify_filename(value):
    """Return a filesystem-safe filename stem."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "graph"


def _style_value(style_map, key, graph, item, default):
    value = style_map.get(key, default)
    if callable(value):
        return value(graph, item)
    return value


def _graph_layout(graph):
    import networkx as nx

    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(graph.vertices)
    nx_graph.add_edges_from(graph.edges)

    if not graph.vertices:
        return {}, nx_graph

    try:
        generations = list(nx.topological_generations(nx_graph))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(nx_graph, seed=7), nx_graph

    positions = {}
    for layer_index, layer in enumerate(generations):
        layer = list(layer)
        offset = (len(layer) - 1) / 2
        for row_index, node in enumerate(layer):
            positions[node] = (layer_index * 4.0, -(row_index - offset) * 2.25)
    return positions, nx_graph


def save_graph_png(graph, title, output_dir="graphs"):
    """Save a styled heterograph-like graph as a PNG and return the path."""
    import os

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "cmir-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_slugify_filename(title)}.png"

    positions, nx_graph = _graph_layout(graph)
    node_count = max(1, len(graph.vertices))
    layer_x_values = {x for x, _ in positions.values()} or {0}
    layer_count = len(layer_x_values)
    max_layer_size = max(
        sum(1 for x, _ in positions.values() if x == layer_x)
        for layer_x in layer_x_values
    )
    fig_width = min(40, max(10, layer_count * 3.2))
    fig_height = min(60, max(7, max_layer_size * 1.7, node_count * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.set_title(title)
    ax.axis("off")

    edge_colors = [
        _style_value(graph.estyle, "color", graph, edge, "#666666")
        for edge in nx_graph.edges
    ]
    nx.draw_networkx_edges(
        nx_graph,
        positions,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        edge_color=edge_colors,
        width=1.2,
        connectionstyle="arc3,rad=0.04",
    )

    edge_labels = {
        edge: _style_value(graph.estyle, "label", graph, edge, "")
        for edge in nx_graph.edges
    }
    edge_labels = {edge: label for edge, label in edge_labels.items() if label}
    if edge_labels:
        nx.draw_networkx_edge_labels(
            nx_graph,
            positions,
            edge_labels=edge_labels,
            font_size=6,
            ax=ax,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    for vertex in graph.vertices:
        x, y = positions[vertex]
        fillcolor = _style_value(graph.vstyle, "fillcolor", graph, vertex, "#FFFFFF")
        fontcolor = _style_value(graph.vstyle, "fontcolor", graph, vertex, "black")
        label = _style_value(graph.vstyle, "label", graph, vertex, str(vertex))
        ax.text(
            x,
            y,
            str(label),
            ha="center",
            va="center",
            fontsize=7,
            color=fontcolor,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": fillcolor,
                "edgecolor": "#333333",
                "linewidth": 0.8,
            },
        )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_pipeline_graphs(graphs, output_dir="graphs"):
    """Save all pipeline graphs and return their PNG paths."""
    paths = []
    for title, graph in graphs:
        path = save_graph_png(graph, title, output_dir)
        paths.append(path)
        print(f"Saved {title} PNG: {path}")
    return paths


def load_keras_model(model_path):
    """Load Keras model from disk."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\n{'='*70}")
    print(f"Loading Keras model from: {model_path}")
    print(f"{'='*70}")
    
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")
    print(f"\nModel Summary:")
    model.summary()
    
    return model


def build_nn_ir_graph(model):
    """Build NN-IR graph from Keras model."""
    print(f"\n{'='*70}")
    print(f"Building NN-IR from model")
    print(f"{'='*70}")
    
    nn_graph = nn_ir.build_nn_ir(model)
    
    print(f"✓ NN-IR built successfully")
    
    return nn_graph


def evaluate_folded_design(nn_graph, model, fold_factor=2):
    """Evaluate folded design with given fold factor."""

    print(f"\n{'='*70}")
    print(f"Evaluating Folded Design")
    print(f"{'='*70}")
    print(f"Fold Factor: {fold_factor}")
    
    design = api.evaluate_folded_design(
        nn_graph,
        model=model,
        backend="da4ml",
        factor=fold_factor,
        target_fmax_hz=300e6,
    )
    
    print(f"✓ Design evaluated successfully")
    print(f"\nMetrics:")
    for key, value in design.metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return design



def extract_cost_per_task(design, task_graph):
    """Extract cost information per task from the schedule."""
    print(f"\n{'='*70}")
    print(f"Extracting Cost Per Task")
    print(f"{'='*70}")
    
    task_costs = []
    schedule = design.task_schedule
    
    for task_id, item in schedule.tasks.items():
        task = item.task
        resource = task_graph.resources.get(task.resource_id, {})
        
        # Get cost from resource
        cost = getattr(resource, 'cost', {})
        if cost is None:
            cost = {}
        
        task_costs.append({
            "task_id": task_id,
            "task_name": getattr(task, 'name', str(task_id)),
            "resource_id": task.resource_id,
            "start_cycle": item.start,
            "end_cycle": item.end,
            "duration": item.end - item.start,
            "ii": task.ii,
            "lut": cost.get("lut", 0) if isinstance(cost, dict) else 0,
            "ff": cost.get("ff", 0) if isinstance(cost, dict) else 0,
            "dsp": cost.get("dsp", 0) if isinstance(cost, dict) else 0,
            "bram": cost.get("bram", 0) if isinstance(cost, dict) else 0,
            "uram": cost.get("uram", 0) if isinstance(cost, dict) else 0,
        })
    
    df_task_costs = pd.DataFrame(task_costs)
    
    print(f"\nTask Costs Summary:")
    print(df_task_costs.to_string())
    
    return df_task_costs


def check_correctness(design, model):
    """Check symbolic correctness of the design."""
    print(f"\n{'='*70}")
    print(f"Checking Symbolic Correctness")
    print(f"{'='*70}")
    
    correctness_report = api.check_symbolic_correctness(design, model=model)
    
    print(f"\nCorrectness Report:")
    if hasattr(correctness_report, "__dict__"):
        for key, value in vars(correctness_report).items():
            if isinstance(value, list):
                print(f"  {key}:")
                for idx, item in enumerate(value):
                    print(f"    [{idx}] {item}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    else:
        print(correctness_report)
    
    return correctness_report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Run folded design evaluation pipeline.")
    parser.add_argument(
        "--save-graphs",
        action="store_true",
        help="Save NN-IR, scheduled, and task graphs as PNG files.",
    )
    parser.add_argument(
        "--graph-output-dir",
        default="graphs",
        help="Directory used with --save-graphs. Defaults to ./graphs.",
    )
    args = parser.parse_args(argv)

    print(f"FOLDED DESIGN EVALUATION PIPELINE")
    
    # Model path from api.py
    model_path = Path("official_models/3-feature-perminv/jet_classifier_large_8/ckpts/epoch=1087-acc=66.97%-val_acc=66.60%-EBOPs=170586.keras")
    # model_path = Path("official_models/deepset/epoch=890-acc=69.51%-val_acc=70.03%-EBOPs=146212.keras")
    # model_path = Path("official_models/deepset/epoch=490-acc=68.93%-val_acc=69.57%-EBOPs=80935.keras")
    # linformer
    # model_path = Path("official_models/linformers/lin8part.keras")

    # 64 particle model 
    # model_path = Path("official_models/3-feature-perminv/jet_classifier_large_64/ckpts/epoch=1294-acc=81.65%-val_acc=81.64%-EBOPs=226791.keras")

    try:
        # Step 1: Load model
        model = load_keras_model(model_path)
        
        # Step 2: Build NN-IR
        nn_graph = build_nn_ir_graph(model)
        
        # Step 3: Evaluate folded design
        design = evaluate_folded_design(nn_graph, model, fold_factor=1)

        apply_nn_style(nn_graph)
        apply_sched_style(design.sched_graph)
        
        # Step 6: Print summary
        for k, v in design.metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        correctness = check_correctness(design, model)


        webview = WebView()
        graphs = [
            ("NN-IR Graph", nn_graph),
            ("Scheduled Graph", design.sched_graph),
            ("Task Graph", design.task_graph),
        ]
        if args.save_graphs:
            save_pipeline_graphs(graphs, args.graph_output_dir)

        for title, graph in graphs:
            webview.add_graph(graph, title=title)
        webview.run(host="127.0.0.1", port="8888")

        
        print(f" PIPELINE COMPLETED SUCCESSFULLY")
        
        return 0
        
    except Exception as e:
        print(f" PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
