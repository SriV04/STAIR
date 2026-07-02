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


def _attention_submodel(model_name: str):
    import keras
    import hgq  # noqa: F401 - registers HGQ custom Keras objects

    model = keras.models.load_model(MODEL_DIR / model_name, compile=False)
    return keras.Model(model.inputs, model.get_layer("mha1").output)


def _by_name(graph):
    return {graph.pmap[vx]["layer_name"]: (vx, graph.pmap[vx]) for vx in graph.vertices}


def test_linformer_attention_expands_to_precision_annotated_nn_ir_nodes():
    graph = build_nn_ir(_attention_submodel("lin8part.keras"))
    nodes = _by_name(graph)

    expected = {
        "mha1_lin_k_proj": ("einsum_dense", [(None, 8, 16)], [(None, 2, 16)]),
        "mha1_lin_v_proj": ("einsum_dense", [(None, 8, 16)], [(None, 2, 16)]),
        "mha1_query": ("einsum_dense", [(None, 8, 16)], [(None, 8, 1, 16)]),
        "mha1_key": ("einsum_dense", [(None, 2, 16)], [(None, 2, 1, 16)]),
        "mha1_value": ("einsum_dense", [(None, 2, 16)], [(None, 2, 1, 16)]),
        "mha1_qk": ("einsum", [(None, 2, 1, 16), (None, 8, 1, 16)], [(None, 1, 8, 2)]),
        "mha1_softmax": ("softmax", [(None, 1, 8, 2)], [(None, 1, 8, 2)]),
        "mha1_av": ("einsum", [(None, 1, 8, 2), (None, 2, 1, 16)], [(None, 8, 1, 16)]),
        "mha1_attention_output": ("einsum_dense", [(None, 8, 1, 16)], [(None, 8, 16)]),
    }

    assert not any(p["layer_class"] == "QLinformerAttention" for _, p in nodes.values())
    for name, (op_kind, in_shapes, out_shapes) in expected.items():
        assert nodes[name][1]["op_kind"] == op_kind
        assert nodes[name][1]["in_shapes"] == in_shapes
        assert nodes[name][1]["out_shapes"] == out_shapes

    query_vx, query = nodes["mha1_query"]
    qk_vx, qk = nodes["mha1_qk"]
    softmax_vx, softmax = nodes["mha1_softmax"]
    output = nodes["mha1_attention_output"][1]

    assert query["oq_qint"] is not None
    assert query["oq_kif"] is not None
    assert qk["input_qints"][1] == query["oq_qint"]
    assert qk["input_kifs"][1] == query["oq_kif"]
    assert softmax["oq_qint"] is not None
    assert softmax["oq_kif"] is not None
    assert output["iq_qint"] is not None
    assert output["iq_kif"] is not None

    query_to_qk = graph.pmap[(query_vx, qk_vx)]
    assert query_to_qk["tensor_shape"] == (None, 8, 1, 16)
    assert query_to_qk["src_qint"] == query["oq_qint"]
    assert query_to_qk["src_kif"] == query["oq_kif"]

    qk_to_softmax = graph.pmap[(qk_vx, softmax_vx)]
    assert qk_to_softmax["tensor_shape"] == (None, 1, 8, 2)
    assert qk_to_softmax["dst_qint"] == softmax["iq_qint"]
    assert qk_to_softmax["dst_kif"] == softmax["iq_kif"]


def test_mha_attention_expands_without_linformer_projection_nodes():
    graph = build_nn_ir(_attention_submodel("mha16part.keras"))
    names = set(_by_name(graph))

    assert "mha1_lin_k_proj" not in names
    assert "mha1_lin_v_proj" not in names
    assert {"mha1_query", "mha1_key", "mha1_value", "mha1_qk", "mha1_softmax", "mha1_av", "mha1_attention_output"} <= names
    assert _by_name(graph)["mha1_qk"][1]["out_shapes"] == [(None, 1, 16, 16)]


def test_attention_sched_ir_marks_dynamic_core_as_non_folded_barriers():
    nn_graph = build_nn_ir(_attention_submodel("lin8part.keras"))
    sched = decompose_nn_to_sched(nn_graph)
    nodes = {sched.pmap[vx]["nn_layer_name"]: (vx, sched.pmap[vx]) for vx in sched.vertices}

    qk_vx, qk = nodes["mha1_qk"]
    softmax_vx, softmax = nodes["mha1_softmax"]
    av_vx, av = nodes["mha1_av"]

    assert qk["op"] == "einsum"
    assert qk["op_params"]["input_shapes"] == [(None, 2, 1, 16), (None, 8, 1, 16)]
    assert qk["op_params"]["output_shape"] == (None, 1, 8, 2)
    assert qk["op_params"]["input_qints"][1] is not None
    assert qk["foldable"] is False
    assert softmax["op"] == "softmax"
    assert softmax["op_params"]["output_shape"] == (None, 1, 8, 2)
    assert softmax["foldable"] is False
    assert av["foldable"] is False

    make_fold_plan(sched, factor=2)
    assert sched.pmap[qk_vx]["fold_group"] is None
    assert sched.pmap[softmax_vx]["temporal_steps_T"] == 1
    assert sched.pmap[av_vx]["lanes_P"] == 1

    for edge in sched.edges:
        if edge[1] in {qk_vx, softmax_vx, av_vx}:
            assert sched.pmap[edge]["consume_mode"] == "all"


def test_attention_barrier_edges_consume_all_temporal_tokens():
    nn_graph = build_nn_ir(_attention_submodel("lin8part.keras"))
    sched = decompose_nn_to_sched(nn_graph)
    nodes = {sched.pmap[vx]["nn_layer_name"]: (vx, sched.pmap[vx]) for vx in sched.vertices}
    query_vx = nodes["mha1_query"][0]
    qk_vx = nodes["mha1_qk"][0]

    make_fold_plan(sched, factor=2)
    for vx in sched.vertices:
        sched.pmap[vx]["cost"] = {"lut": 0, "ff": 0, "latency_cycles": 1, "ii": 1}
    for edge in sched.edges:
        if edge[0] == query_vx:
            sched.pmap[edge]["value_id"] = "query:out"
        elif edge[0] == qk_vx:
            sched.pmap[edge]["value_id"] = "qk:out"
        else:
            sched.pmap[edge]["value_id"] = f"{edge[0]}:out"

    # The current fold plan keeps the attention core non-folded (T=1), so force
    # the query producer to be temporally folded to exercise the barrier
    # semantics: a consume_mode="all" edge must consume every temporal token.
    sched.pmap[query_vx]["temporal_steps_T"] = 2

    task_graph = expand_tasks(sched)
    qk_task = task_graph.tasks[f"node:{qk_vx}"]

    assert "query:out:t0" in qk_task.input_tokens
    assert "query:out:t1" in qk_task.input_tokens
