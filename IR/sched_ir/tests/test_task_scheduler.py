from IR.sched_ir.scheduling.metrics import schedule_metrics
from IR.sched_ir.scheduling.scheduler import schedule_tasks
from IR.sched_ir.scheduling.task_ir import ResourceInstance, Task, TaskGraph


def _two_step_task_graph():
    resource = ResourceInstance(
        resource_id="dense",
        source_node=0,
        cost={"lut": 12, "ff": 3},
        latency=2,
        ii=1,
    )
    return TaskGraph(
        tasks={
            "dense:t0": Task("dense:t0", 0, 0, "dense", ["in:t0"], ["out:t0"], 2, 1),
            "dense:t1": Task("dense:t1", 0, 1, "dense", ["in:t1"], ["out:t1"], 2, 1),
        },
        resources={"dense": resource},
        initial_tokens={"in:t0", "in:t1"},
    )


def test_shared_resource_serialises_issues_at_resource_ii():
    task_graph = _two_step_task_graph()
    schedule = schedule_tasks(task_graph)

    assert schedule.tasks["dense:t0"].start == 0
    assert schedule.tasks["dense:t1"].start == 1


def test_metrics_count_hardware_area_once_and_work_per_task_invocation():
    task_graph = _two_step_task_graph()
    schedule = schedule_tasks(task_graph)
    metrics = schedule_metrics(schedule, task_graph, fclk_hz=300e6)

    assert metrics["area_proxy"] == 12
    assert metrics["work_cost"] == 24
    assert metrics["latency_cycles"] == 3
    assert metrics["sample_ii_cycles"] == 2
    assert metrics["throughput_samples_per_sec"] == 150e6


def test_latency_cycles_spans_t0_to_latest_scheduled_task_end():
    task_graph = _two_step_task_graph()
    schedule = schedule_tasks(task_graph)
    metrics = schedule_metrics(schedule, task_graph, fclk_hz=300e6)

    assert min(item.start for item in schedule.tasks.values()) == 0
    assert max(item.end for item in schedule.tasks.values()) == 3
    assert metrics["latency_cycles"] == 3
