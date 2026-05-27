"""
Full test pipeline for folded design evaluation.
Creates NN-IR, evaluates folded design with fold factor 2 (4 lanes),
and generates cost analysis per layer and task.
"""

import sys
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import numpy as np

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from IR import nn_ir
from IR.sched_ir import api
import keras
import hgq

from heterograph.webview import WebView


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
    hardware_lanes = fold_factor * 2  # fold_factor=2 -> 4 lanes
    
    print(f"\n{'='*70}")
    print(f"Evaluating Folded Design")
    print(f"{'='*70}")
    print(f"Fold Factor: {fold_factor}")
    print(f"Hardware Lanes: {hardware_lanes}")
    
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


    """Extract cost information per layer from the design."""
    print(f"\n{'='*70}")
    print(f"Extracting Cost Per Layer")
    print(f"{'='*70}")
    
    layer_costs = []
    
    # Extract from evaluation graph
    eval_graph = design.evaluation.graph
    
    for node_id, node in eval_graph.v.items():
        layer_name = node.labels.get("name", str(node_id))
        layer_op = node.labels.get("op", "unknown")
        
        # Get cost information
        cost = node.labels.get("cost", {})
        if cost is None:
            cost = {}
        
        layer_costs.append({
            "layer_id": node_id,
            "layer_name": layer_name,
            "op_type": layer_op,
            "lut": cost.get("lut", 0),
            "ff": cost.get("ff", 0),
            "dsp": cost.get("dsp", 0),
            "bram": cost.get("bram", 0),
            "uram": cost.get("uram", 0),
            "latency_cycles": cost.get("latency_cycles", 0),
            "ii": cost.get("ii", 1),
        })
    
    df_layer_costs = pd.DataFrame(layer_costs)
    
    print(f"\nLayer Costs Summary:")
    print(df_layer_costs.to_string())
    
    return df_layer_costs


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





def main():
    """Run full test pipeline."""
    print(f"\n{'='*70}")
    print(f"FOLDED DESIGN EVALUATION PIPELINE")
    print(f"{'='*70}")
    
    # Model path from api.py
    model_path = Path("official_models/3-feature-perminv/jet_classifier_large_8/ckpts/epoch=1087-acc=66.97%-val_acc=66.60%-EBOPs=170586.keras")
    
    try:
        # Step 1: Load model
        model = load_keras_model(model_path)
        
        # Step 2: Build NN-IR
        nn_graph = build_nn_ir_graph(model)
        
        # Step 3: Evaluate folded design (fold_factor=2 -> 4 lanes)
        design = evaluate_folded_design(nn_graph, model, fold_factor=2)

        
        # Step 6: Print summary
        for k, v in design.metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        webview = WebView()
        webview.add_graph(nn_graph, title="NN-IR Graph")
        webview.add_graph(design.sched_graph, title="Scheduled Graph")
        webview.add_graph(design.task_graph, title="Task Graph")
        webview.run(host="127.0.0.1", port="8888")

        
        print(f"\n{'='*70}")
        print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
