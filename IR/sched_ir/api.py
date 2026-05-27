"""Public orchestration API for folded backend evaluation and static scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from heterograph import HGraph

from .backends.da4ml.plugin import DA4MLBackend
from .graphing.tasks import task_schedule_to_hgraph
from .lowering.decomposer import decompose_nn_to_sched
from .planning.fold_plan import make_fold_plan
from .scheduling.expand import expand_tasks
from .scheduling.metrics import schedule_metrics
from .scheduling.scheduler import schedule_tasks


@dataclass
class EvaluatedDesign:
    sched_graph: HGraph
    fold_plan: object
    evaluation: object
    task_ir: object
    task_graph: HGraph
    task_schedule: object
    metrics: dict


def evaluate_folded_design(
    nn_graph,
    *,
    model,
    backend: str = "da4ml",
    factor: int | None = None,
    lanes: int | None = None,
    target_fmax_hz: float = 300e6,
    resource_config=None,
) -> EvaluatedDesign:
    if backend != "da4ml":
        raise ValueError(f"Unsupported Sched-IR backend: {backend!r}")
    sched_graph = decompose_nn_to_sched(nn_graph)
    fold_plan = make_fold_plan(sched_graph, factor=factor, lanes=lanes)
    evaluation = DA4MLBackend().evaluate(
        sched_graph,
        model=model,
        fold_plan=fold_plan,
        resource_config=resource_config,
    )
    task_ir = expand_tasks(evaluation.graph)
    task_schedule = schedule_tasks(task_ir)
    metrics = schedule_metrics(task_schedule, task_ir, fclk_hz=target_fmax_hz)
    task_graph = task_schedule_to_hgraph(task_ir, task_schedule)
    return EvaluatedDesign(
        sched_graph=sched_graph,
        fold_plan=fold_plan,
        evaluation=evaluation,
        task_ir=task_ir,
        task_graph=task_graph,
        task_schedule=task_schedule,
        metrics=metrics,
    )


if __name__ == "__main__":
    import IR.nn_ir as nn_ir
    import keras

    model_path = Path("official_models/3-feature-perminv/jet_classifier_large_8/ckpts/epoch=1087-acc=66.97%-val_acc=66.60%-EBOPs=170586.keras")
    model = keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    print("Model summary:")
    model.summary()

    nn_graph = nn_ir.build_nn_ir(model)
    print("\nEvaluating folded design with DA4ML backend...")
    design = evaluate_folded_design(nn_graph, model=model, factor=2)
    print("\nEvaluation metrics:")
    for k, v in design.metrics.items():
        print(f"  {k}: {v}")
    
