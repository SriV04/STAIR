from types import SimpleNamespace

from IR.sched_ir.correctness.scheduled_trace import build_scheduled_trace
from IR.sched_ir.scheduling.task_ir import (
    ResourceInstance,
    ScheduledTask,
    Task,
    TaskGraph,
    TaskSchedule,
)


class FakeGraph:
    vertices = [0]
    edges = []
    pmap = {
        0: {
            "backend_trace_id": "trace:0",
            "evaluated_output_shapes": [(8, 4)],
            "evaluated_output_qints": [
                ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"]
            ],
            "temporal_steps_T": 2,
            "parallelism_N": 8,
            "lanes_P": 4,
            "fold_axes": [0],
            "fold_group": 2,
        }
    }


class FakeSymbolicValue:
    def __init__(self, label, qint):
        self.label = label
        self.qint = qint

    def __add__(self, other):
        return FakeSymbolicValue(
            f"({self.label}+{other.label})",
            f"({self.qint}+{other.qint})",
        )


def test_build_scheduled_trace_slices_folded_outputs_by_temporal_step():
    artifact = SimpleNamespace(symbolic_outputs=["whole-output"])
    evaluation = SimpleNamespace(
        graph=FakeGraph(),
        runtime_artifacts={"trace:0": artifact},
    )
    task_graph = TaskGraph(
        tasks={
            "node:0:t0": Task(
                "node:0:t0",
                0,
                0,
                "node:0",
                ["input:0:t0"],
                ["0:out:t0"],
                2,
                1,
            ),
            "node:0:t1": Task(
                "node:0:t1",
                0,
                1,
                "node:0",
                ["input:0:t1"],
                ["0:out:t1"],
                2,
                1,
            ),
        },
        resources={
            "node:0": ResourceInstance(
                "node:0",
                0,
                {"latency_cycles": 2},
                2,
                1,
            )
        },
        initial_tokens={"input:0:t0", "input:0:t1"},
    )
    schedule = TaskSchedule(
        tasks={
            "node:0:t0": ScheduledTask(task_graph.tasks["node:0:t0"], 0, 2),
            "node:0:t1": ScheduledTask(task_graph.tasks["node:0:t1"], 1, 3),
        },
        token_ready={"0:out:t0": 2, "0:out:t1": 3},
        resource_next_issue={"node:0": 2},
    )

    trace = build_scheduled_trace(evaluation, task_graph, schedule)

    assert trace.output_tokens == ["0:out:t0", "0:out:t1"]
    assert trace.token_values["0:out:t0"].logical_slice == (
        slice(0, 4),
        slice(None),
    )
    assert trace.token_values["0:out:t1"].logical_slice == (
        slice(4, 8),
        slice(None),
    )
    assert trace.token_values["0:out:t1"].qints == ["q4", "q5", "q6", "q7"]
    assert trace.token_values["0:out:t1"].ready_cycle == 3
    assert trace.token_values["0:out:t1"].fold_group == 2
    assert trace.output_qints == ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"]


def test_build_scheduled_trace_uses_logical_unfolded_shape_for_slices():
    graph = FakeGraph()
    graph.pmap = {
        0: {
            **FakeGraph.pmap[0],
            "op_params": {"output_shapes": (None, 8, 4)},
            "evaluated_output_shapes": [(4, 4)],
            "evaluated_output_qints": [["q0", "q1", "q2", "q3"]],
            "fold_axes": [1],
        }
    }
    evaluation = SimpleNamespace(
        graph=graph,
        runtime_artifacts={"trace:0": SimpleNamespace(symbolic_outputs=["out"])},
    )
    task_graph = TaskGraph(
        tasks={
            "node:0:t1": Task(
                "node:0:t1",
                0,
                1,
                "node:0",
                ["input:0:t1"],
                ["0:out:t1"],
                2,
                1,
            )
        },
        resources={"node:0": ResourceInstance("node:0", 0, {}, 2, 1)},
        initial_tokens={"input:0:t1"},
    )
    schedule = TaskSchedule(
        tasks={"node:0:t1": ScheduledTask(task_graph.tasks["node:0:t1"], 0, 2)},
        token_ready={"0:out:t1": 2},
        resource_next_issue={"node:0": 1},
    )

    trace = build_scheduled_trace(evaluation, task_graph, schedule)

    assert trace.token_values["0:out:t1"].logical_slice == (
        slice(4, 8),
        slice(None),
    )


def test_build_scheduled_trace_pads_unused_lanes_with_neutral_qints():
    graph = FakeGraph()
    graph.pmap = {
        0: {
            **FakeGraph.pmap[0],
            "evaluated_output_shapes": [(6, 4)],
            "evaluated_output_qints": [["q0", "q1", "q2", "q3", "q4", "q5"]],
        }
    }
    evaluation = SimpleNamespace(
        graph=graph,
        runtime_artifacts={"trace:0": SimpleNamespace(symbolic_outputs=["out"])},
    )
    task_graph = TaskGraph(
        tasks={
            "node:0:t1": Task(
                "node:0:t1",
                0,
                1,
                "node:0",
                ["input:0:t1"],
                ["0:out:t1"],
                2,
                1,
            )
        },
        resources={"node:0": ResourceInstance("node:0", 0, {}, 2, 1)},
        initial_tokens={"input:0:t1"},
    )
    schedule = TaskSchedule(
        tasks={"node:0:t1": ScheduledTask(task_graph.tasks["node:0:t1"], 0, 2)},
        token_ready={"0:out:t1": 2},
        resource_next_issue={"node:0": 1},
    )

    trace = build_scheduled_trace(evaluation, task_graph, schedule)

    assert trace.token_values["0:out:t1"].qints[:2] == ["q4", "q5"]
    assert [
        getattr(qint, "neutral", False)
        for qint in trace.token_values["0:out:t1"].qints[2:]
    ] == [True, True]
    assert trace.output_qints == ["q4", "q5"]


def test_temporal_accumulator_task_combines_symbolic_values_and_qints():
    graph = FakeGraph()
    graph.pmap = {
        0: {
            **FakeGraph.pmap[0],
            "op": "reduce",
            "reduce_mode": "hybrid",
            "evaluated_output_shapes": [(2,)],
            "evaluated_output_qints": [["partial-q0", "partial-q1"]],
        }
    }
    evaluation = SimpleNamespace(
        graph=graph,
        runtime_artifacts={
            "trace:0": SimpleNamespace(
                symbolic_outputs=[
                    FakeSymbolicValue("s0", "q0"),
                    FakeSymbolicValue("s1", "q1"),
                ]
            )
        },
    )
    task_graph = TaskGraph(
        tasks={
            "node:0:t0": Task(
                "node:0:t0",
                0,
                0,
                "node:0",
                ["input:0:t0"],
                ["0:partial:t0"],
                2,
                1,
            ),
            "node:0:t1": Task(
                "node:0:t1",
                0,
                1,
                "node:0",
                ["input:0:t1"],
                ["0:partial:t1"],
                2,
                1,
            ),
            "node:0:acc": Task(
                "node:0:acc",
                0,
                None,
                "node:0:acc",
                ["0:partial:t0", "0:partial:t1"],
                ["0:out"],
                1,
                1,
                task_kind="temporal_accumulator",
            ),
        },
        resources={
            "node:0": ResourceInstance("node:0", 0, {}, 2, 1),
            "node:0:acc": ResourceInstance("node:0:acc", 0, {}, 1, 1),
        },
        initial_tokens={"input:0:t0", "input:0:t1"},
    )
    schedule = TaskSchedule(
        tasks={
            "node:0:t0": ScheduledTask(task_graph.tasks["node:0:t0"], 0, 2),
            "node:0:t1": ScheduledTask(task_graph.tasks["node:0:t1"], 1, 3),
            "node:0:acc": ScheduledTask(task_graph.tasks["node:0:acc"], 3, 4),
        },
        token_ready={"0:partial:t0": 2, "0:partial:t1": 3, "0:out": 4},
        resource_next_issue={"node:0": 2, "node:0:acc": 4},
    )

    trace = build_scheduled_trace(evaluation, task_graph, schedule)

    assert trace.token_values["0:partial:t1"].logical_slice is None
    assert trace.output_tokens == ["0:out"]
    assert trace.token_values["0:out"].value.label == "(s0+s1)"
    assert trace.token_values["0:out"].qints == ["(q0+q1)"]
    assert trace.output_qints == ["(q0+q1)"]
