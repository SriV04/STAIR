from IR.sched_ir.scheduling.sync_analysis import analyse_sync_points
from IR.sched_ir.scheduling.task_ir import ScheduledTask, Task, TaskSchedule


def _task(task_id, inputs, outputs, *, latency=1, task_kind="compute"):
    return Task(
        task_id=task_id,
        source_node=0,
        temporal_step=None,
        resource_id=task_id,
        input_tokens=list(inputs),
        output_tokens=list(outputs),
        latency=latency,
        ii=1,
        task_kind=task_kind,
    )


def test_sync_analysis_reports_multi_input_arrivals_and_waits():
    schedule = TaskSchedule(
        tasks={
            "left": ScheduledTask(_task("left", ["source:left"], ["left:out"]), 0, 10),
            "right": ScheduledTask(_task("right", ["source:right"], ["right:out"]), 0, 14),
            "join": ScheduledTask(
                _task("join", ["left:out", "right:out"], ["join:out"]),
                14,
                15,
            ),
        },
        token_ready={},
        resource_next_issue={},
    )

    report = analyse_sync_points(schedule)

    assert report.sync_point_count == 1
    assert report.max_sync_wait_cycles == 4
    assert report.total_sync_wait_cycles == 4
    assert report.mean_sync_wait_cycles == 4.0
    assert report.max_buffer_wait_cycles == 4
    assert report.total_input_buffer_wait_cycles == 4
    assert report.sync_legality_passed is True
    assert report.sync_violations == []

    diagnostic = report.sync_points[0]
    assert diagnostic.task_name == "join"
    assert diagnostic.layer_type == "compute"
    assert diagnostic.start_cycle == 14
    assert diagnostic.latest_input_arrival_cycle == 14
    assert diagnostic.earliest_input_arrival_cycle == 10
    assert diagnostic.input_arrival_cycles == {"left:out": 10, "right:out": 14}
    assert diagnostic.input_buffer_wait_cycles == {"left:out": 4, "right:out": 0}


def test_sync_analysis_uses_scheduled_start_for_resource_delayed_buffer_wait():
    schedule = TaskSchedule(
        tasks={
            "left": ScheduledTask(_task("left", ["source:left"], ["left:out"]), 0, 10),
            "right": ScheduledTask(_task("right", ["source:right"], ["right:out"]), 0, 14),
            "join": ScheduledTask(
                _task("join", ["left:out", "right:out"], ["join:out"]),
                20,
                21,
            ),
        },
        token_ready={},
        resource_next_issue={},
    )

    report = analyse_sync_points(schedule)

    assert report.max_sync_wait_cycles == 4
    assert report.max_buffer_wait_cycles == 10
    assert report.total_input_buffer_wait_cycles == 16
    assert report.sync_points[0].input_buffer_wait_cycles == {
        "left:out": 10,
        "right:out": 6,
    }


def test_sync_analysis_treats_missing_producers_as_source_arrivals():
    schedule = [
        ScheduledTask(
            _task("join", ["external:a", "external:b"], ["join:out"]),
            3,
            4,
        )
    ]

    report = analyse_sync_points(schedule, source_arrival_cycle=2)

    assert report.sync_point_count == 1
    assert report.max_sync_wait_cycles == 0
    assert report.max_buffer_wait_cycles == 1
    assert report.total_input_buffer_wait_cycles == 2
    assert report.sync_points[0].input_arrival_cycles == {
        "external:a": 2,
        "external:b": 2,
    }


def test_sync_analysis_reports_dependency_violations():
    schedule = TaskSchedule(
        tasks={
            "left": ScheduledTask(_task("left", ["source:left"], ["left:out"]), 0, 10),
            "right": ScheduledTask(_task("right", ["source:right"], ["right:out"]), 0, 14),
            "join": ScheduledTask(
                _task("join", ["left:out", "right:out"], ["join:out"]),
                12,
                13,
            ),
        },
        token_ready={},
        resource_next_issue={},
    )

    report = analyse_sync_points(schedule)

    assert report.sync_legality_passed is False
    assert report.sync_violations == [
        "join starts at 12, but latest input arrives at 14"
    ]
    assert report.max_buffer_wait_cycles == 2


def test_sync_analysis_counts_temporal_accumulator_inputs_as_sync_point():
    schedule = TaskSchedule(
        tasks={
            "reduce:t0": ScheduledTask(
                _task("reduce:t0", ["source:t0"], ["partial:t0"]), 0, 2
            ),
            "reduce:t1": ScheduledTask(
                _task("reduce:t1", ["source:t1"], ["partial:t1"]), 1, 3
            ),
            "reduce:acc": ScheduledTask(
                _task(
                    "reduce:acc",
                    ["partial:t0", "partial:t1"],
                    ["reduce:out"],
                    task_kind="temporal_accumulator",
                ),
                3,
                4,
            ),
        },
        token_ready={},
        resource_next_issue={},
    )

    report = analyse_sync_points(schedule)

    assert report.sync_point_count == 1
    assert report.sync_points[0].task_name == "reduce:acc"
    assert report.sync_points[0].layer_type == "temporal_accumulator"
    assert report.sync_points[0].sync_wait_cycles == 1
