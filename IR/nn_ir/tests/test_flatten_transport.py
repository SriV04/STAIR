from __future__ import annotations

from pathlib import Path

import pytest

from IR.nn_ir.builder import build_nn_ir
from IR.sched_ir.lowering.decomposer import decompose_nn_to_sched
from IR.sched_ir.planning.fold_plan import make_fold_plan
from IR.sched_ir.scheduling.expand import expand_tasks


pytest.importorskip("keras")
pytest.importorskip("hgq")


MODEL_DIR = Path(__file__).resolve().parents[3] / "official_models" / "linformers"


def _full_linformer_model():
    import keras
    import hgq  # noqa: F401 - registers HGQ custom Keras objects

    return keras.models.load_model(MODEL_DIR / "lin8part.keras", compile=False)


def _sched_by_name(sched):
    return {sched.pmap[vx]["nn_layer_name"]: (vx, sched.pmap[vx]) for vx in sched.vertices}


def test_flatten_is_lowered_as_zero_cost_transport_in_full_linformer_model():
    nn_graph = build_nn_ir(_full_linformer_model())
    flatten_vx = next(vx for vx in nn_graph.vertices if nn_graph.pmap[vx]["layer_name"] == "flatten_2")
    flatten = nn_graph.pmap[flatten_vx]

    assert flatten["op_kind"] == "flatten"
    assert flatten["in_shapes"] == [(None, 8, 16)]
    assert flatten["out_shapes"] == [(None, 128)]
    assert flatten["foldable"] is False
    assert flatten["cost_mode"] == "synthetic_zero"

    sched = decompose_nn_to_sched(nn_graph)
    nodes = _sched_by_name(sched)
    flatten_sv, flatten_sp = nodes["flatten_2"]

    assert flatten_sp["op"] == "transport"
    assert flatten_sp["op_params"]["mode"] == "flatten"
    assert flatten_sp["op_params"]["input_shape"] == (None, 8, 16)
    assert flatten_sp["op_params"]["output_shape"] == (None, 128)
    assert flatten_sp["foldable"] is False

    incoming = [edge for edge in sched.edges if edge[1] == flatten_sv]
    outgoing = [edge for edge in sched.edges if edge[0] == flatten_sv]
    assert incoming
    assert outgoing
    assert sched.pmap[incoming[0]]["consume_mode"] == "all"
    assert sched.pmap[outgoing[0]]["tensor_shape"] == (None, 128)


def test_flatten_transport_barrier_consumes_all_folded_tokens():
    nn_graph = build_nn_ir(_full_linformer_model())
    sched = decompose_nn_to_sched(nn_graph)
    nodes = _sched_by_name(sched)
    producer_vx = nodes["q_add_8"][0]
    flatten_vx = nodes["flatten_2"][0]

    make_fold_plan(sched, factor=2)
    sched.pmap[producer_vx]["temporal_steps_T"] = 2
    for vx in sched.vertices:
        sched.pmap[vx]["cost"] = {"lut": 0, "ff": 0, "latency_cycles": 1, "ii": 1}
    for edge in sched.edges:
        sched.pmap[edge]["value_id"] = "producer:out" if edge[0] == producer_vx else f"{edge[0]}:out"

    task_graph = expand_tasks(sched)
    flatten_task = task_graph.tasks[f"node:{flatten_vx}"]

    assert "producer:out:t0" in flatten_task.input_tokens
    assert "producer:out:t1" in flatten_task.input_tokens
