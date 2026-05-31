from importlib import import_module


def test_nn_ir_package_imports():
    builder = import_module("IR.nn_ir.builder")
    schema = import_module("IR.nn_ir.schema")
    styling = import_module("IR.nn_ir.styling")
    assert builder is not None
    assert schema is not None
    assert styling is not None


def test_sched_ir_package_imports():
    modules = [
        "IR.sched_ir.api",
        "IR.sched_ir.schema",
        "IR.sched_ir.lowering.decomposer",
        "IR.sched_ir.planning.fold_plan",
        "IR.sched_ir.backends.da4ml.plugin",
        "IR.sched_ir.scheduling.task_ir",
        "IR.sched_ir.scheduling.expand",
        "IR.sched_ir.scheduling.scheduler",
        "IR.sched_ir.scheduling.metrics",
        "IR.sched_ir.graphing.styling",
        "IR.sched_ir.graphing.tasks",
    ]
    for name in modules:
        assert import_module(name) is not None
