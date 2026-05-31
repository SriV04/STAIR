from types import SimpleNamespace

from IR.sched_ir.correctness.checker import check_symbolic_correctness
from IR.sched_ir.correctness.da4ml_reference import trace_unfolded_reference


class FakeModel:
    inputs = ["keras-input"]
    outputs = ["keras-output"]
    name = "fake_model"


def test_trace_unfolded_reference_uses_da4ml_trace_model_and_comb_trace(monkeypatch):
    calls = {}
    model = FakeModel()

    def fake_trace_model(traced_model, verbose=False):
        calls["trace_model"] = (traced_model, verbose)
        return ["symbolic-input"], ["symbolic-output"]

    def fake_comb_trace(inputs, outputs):
        calls["comb_trace"] = (inputs, outputs)
        return SimpleNamespace(out_qint=["qout"])

    monkeypatch.setattr(
        "IR.sched_ir.correctness.da4ml_reference._trace_model",
        fake_trace_model,
    )
    monkeypatch.setattr(
        "IR.sched_ir.correctness.da4ml_reference._comb_trace",
        fake_comb_trace,
    )

    trace = trace_unfolded_reference(model, config={"verbose": True})

    assert calls["trace_model"] == (model, True)
    assert calls["comb_trace"] == (["symbolic-input"], ["symbolic-output"])
    assert trace.symbolic_inputs == ["symbolic-input"]
    assert trace.symbolic_outputs == ["symbolic-output"]
    assert trace.output_qints == ["qout"]
    assert trace.metadata["model_name"] == "fake_model"


def test_check_symbolic_correctness_traces_reference_scheduled_and_compares(monkeypatch):
    calls = {}
    design = SimpleNamespace(
        evaluation="evaluation",
        task_ir="task-ir",
        task_schedule="schedule",
        fold_plan=SimpleNamespace(groups=(SimpleNamespace(group_id=0),)),
    )
    reference = SimpleNamespace(
        symbolic_outputs=["ref"],
        output_qints=["rq"],
        metadata={"model_name": "fake"},
    )
    scheduled = SimpleNamespace(
        symbolic_outputs=["sched"],
        output_qints=["sq"],
        output_tokens=["out:t0"],
        token_values={
            "out:t0": SimpleNamespace(
                source_node=1,
                temporal_step=0,
                ready_cycle=7,
                logical_slice=None,
            )
        },
        metadata={"task_count": 1},
    )

    monkeypatch.setattr(
        "IR.sched_ir.correctness.checker.trace_unfolded_reference",
        lambda model, config=None: reference,
    )
    monkeypatch.setattr(
        "IR.sched_ir.correctness.checker.build_scheduled_trace",
        lambda evaluation, task_ir, task_schedule: scheduled,
    )

    def fake_compare(*, reference_qints, scheduled_qints, provenance):
        calls["compare"] = (reference_qints, scheduled_qints, provenance)
        return SimpleNamespace(passed=True, metadata={})

    monkeypatch.setattr(
        "IR.sched_ir.correctness.checker.compare_output_widths",
        fake_compare,
    )

    report = check_symbolic_correctness(
        design,
        model="model",
        config={"verbose": False},
    )

    assert report.passed is True
    assert report.metadata["fold_group_count"] == 1
    assert calls["compare"][0] == ["rq"]
    assert calls["compare"][1] == ["sq"]
    assert calls["compare"][2][0]["token_id"] == "out:t0"
