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
        "IR.sched_ir.schema",
        "IR.sched_ir.decomposer",
        "IR.sched_ir.binder",
        "IR.sched_ir.precision",
        "IR.sched_ir.folding.folder",
        "IR.sched_ir.folding.fold_precision",
        "IR.sched_ir.scheduling.scheduler_p3",
        "IR.sched_ir.scheduling.infrastructure",
        "IR.sched_ir.graphing.styling",
        "IR.sched_ir.graphing.gantt",
    ]
    for name in modules:
        assert import_module(name) is not None
