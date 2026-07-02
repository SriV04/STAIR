"""Public orchestration API for folded backend evaluation and static scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from heterograph import HGraph

from .graphing.tasks import task_schedule_to_hgraph
from .lowering.decomposer import decompose_nn_to_sched
from .planning.fold_plan import make_fold_plan
from .scheduling.expand import expand_tasks
from .scheduling.metrics import schedule_metrics
from .scheduling.scheduler import schedule_tasks
from .scheduling.sync_analysis import analyse_sync_points


DA4MLBackend = None


def _da4ml_backend_cls():
    global DA4MLBackend
    if DA4MLBackend is None:
        from .backends.da4ml.plugin import DA4MLBackend as _DA4MLBackend

        DA4MLBackend = _DA4MLBackend
    return DA4MLBackend


@dataclass
class EvaluatedDesign:
    sched_graph: HGraph
    fold_plan: object
    evaluation: object
    task_ir: object
    task_graph: HGraph
    task_schedule: object
    sync_report: object
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
    evaluation = _da4ml_backend_cls()().evaluate(
        sched_graph,
        model=model,
        fold_plan=fold_plan,
        resource_config=resource_config,
    )
    task_ir = expand_tasks(evaluation.graph)
    task_schedule = schedule_tasks(task_ir)
    metrics = schedule_metrics(task_schedule, task_ir, fclk_hz=target_fmax_hz)
    if "latency_cycles" in metrics:
        metrics.setdefault("scheduled_latency_cycles", metrics["latency_cycles"])
    sync_report = analyse_sync_points(task_schedule)
    metrics.update(
        {
            "sync_point_count": sync_report.sync_point_count,
            "max_sync_wait_cycles": sync_report.max_sync_wait_cycles,
            "total_sync_wait_cycles": sync_report.total_sync_wait_cycles,
            "mean_sync_wait_cycles": sync_report.mean_sync_wait_cycles,
            "max_buffer_wait_cycles": sync_report.max_buffer_wait_cycles,
            "total_input_buffer_wait_cycles": sync_report.total_input_buffer_wait_cycles,
            "sync_legality_passed": sync_report.sync_legality_passed,
            "sync_violations": len(sync_report.sync_violations),
        }
    )
    task_graph = task_schedule_to_hgraph(
        task_ir,
        task_schedule,
        source_graph=evaluation.graph,
        sync_report=sync_report,
    )
    return EvaluatedDesign(
        sched_graph=sched_graph,
        fold_plan=fold_plan,
        evaluation=evaluation,
        task_ir=task_ir,
        task_graph=task_graph,
        task_schedule=task_schedule,
        sync_report=sync_report,
        metrics=metrics,
    )


def check_symbolic_correctness(
    design: EvaluatedDesign,
    *,
    model,
    config=None,
):
    from .correctness.checker import check_symbolic_correctness as _check

    return _check(design, model=model, config=config)
