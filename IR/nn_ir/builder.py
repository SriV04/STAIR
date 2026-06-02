"""NN-IR builder: turn a Keras/HGQ model into a populated `HGraph`."""

from __future__ import annotations

from math import prod
from typing import Any
import warnings

from heterograph import HGraph
from . import schema as _schema
from .hgq2.attention_lowering import AttentionEdgeSpec, is_attention_layer, lower_attention_layer
from .hgq2 import hgq_extractor as _hgq

vinit_nn = _schema.vinit_nn
einit_nn = _schema.einit_nn
ginit_nn = _schema.ginit_nn


_OP_KIND = {
    "InputLayer": "input",
    "QEinsumDenseBatchnorm": "einsum_dense_bn",
    "QEinsumDense": "einsum_dense",
    "QDense": "dense",
    "QSum": "qsum",
    "QAdd": "qadd",
    "Activation": "activation",
    "Flatten": "flatten",
}


def _classify(layer) -> str:
    cls = type(layer).__name__
    if cls not in _OP_KIND:
        raise NotImplementedError(
            f"NN-IR builder does not know how to lower Keras layer class "
            f"{cls!r} (layer name: {getattr(layer, 'name', '?')!r})."
        )
    return _OP_KIND[cls]


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _layer_shapes(layer) -> tuple[list, list]:
    in_shapes: list = []
    out_shapes: list = []
    for node in getattr(layer, "_inbound_nodes", []) or []:
        for t in _as_list(node.input_tensors):
            in_shapes.append(tuple(t.shape))
        for t in _as_list(node.output_tensors):
            out_shapes.append(tuple(t.shape))
    return in_shapes, out_shapes


def _layer_inbound(layer) -> list[str]:
    inbound: list[str] = []
    for node in getattr(layer, "_inbound_nodes", []) or []:
        for t in _as_list(node.input_tensors):
            hist = getattr(t, "_keras_history", None)
            if hist is not None:
                inbound.append(hist[0].name)
    return inbound


def _non_batch_volume(shape) -> int | None:
    dims = [d for d in shape[1:] if d is not None] if shape else []
    if len(dims) == 0:
        return None
    return int(prod(dims))


_DENSE_SPATIAL_LABELS = "nmpqrstuvwxyzadefghijklo"


def _first_shape(shapes: list) -> tuple | None:
    return tuple(shapes[0]) if shapes else None


def _qdense_equation(in_shapes: list, out_shapes: list, kernel_shape: tuple | None) -> str | None:
    """Represent a Keras Dense/QDense last-axis contraction as an einsum."""
    in_shape = _first_shape(in_shapes)
    out_shape = _first_shape(out_shapes)
    if in_shape is None or out_shape is None or kernel_shape is None:
        return None
    if len(in_shape) < 2 or len(out_shape) != len(in_shape) or len(kernel_shape) != 2:
        return None

    in_features, units = kernel_shape
    if in_shape[-1] is not None and in_features is not None and int(in_shape[-1]) != int(in_features):
        return None
    if out_shape[-1] is not None and units is not None and int(out_shape[-1]) != int(units):
        return None

    spatial_rank = len(in_shape) - 2
    if spatial_rank > len(_DENSE_SPATIAL_LABELS):
        return "...c,cC->...C"

    spatial = _DENSE_SPATIAL_LABELS[:spatial_rank]
    return f"b{spatial}c,cC->b{spatial}C"


def _quant_summary_fields(summary: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    stats = {
        "avg": None,
        "max": None,
        "min": None,
        "shape": None,
    }
    if summary and summary.get("kif"):
        bits = summary["kif"].get("bits")
        if bits is not None:
            arr = _hgq.safe_array(bits)
            if arr is not None and arr.size > 0:
                stats = {
                    "avg": float(arr.mean()),
                    "max": float(arr.max()),
                    "min": float(arr.min()),
                    "shape": tuple(arr.shape),
                }

    return {
        prefix: summary,
        f"{prefix}_kif": summary.get("kif") if summary else None,
        f"{prefix}_qint": summary.get("qint") if summary else None,
        f"{prefix}_bw_avg": stats["avg"],
        f"{prefix}_bw_max": stats["max"],
        f"{prefix}_bw_min": stats["min"],
        f"{prefix}_bw_shape": stats["shape"],
        f"{prefix}_overflow_mode": summary.get("overflow_mode") if summary else None,
        f"{prefix}_round_mode": summary.get("round_mode") if summary else None,
    }


def _dense_geometry(layer_class: str, in_shapes: list, out_shapes: list, kernel_shape: tuple | None) -> dict[str, Any]:
    if layer_class != "QDense":
        return {
            "dense_contract_axis": None,
            "feature_axis": None,
            "units": None,
            "equation_synthesized": False,
        }
    units = None
    if kernel_shape is not None and len(kernel_shape) == 2 and kernel_shape[-1] is not None:
        units = int(kernel_shape[-1])
    return {
        "dense_contract_axis": -1,
        "feature_axis": -1,
        "units": units,
        "equation_synthesized": True,
        "equation": _qdense_equation(in_shapes, out_shapes, kernel_shape),
    }


def _reduction_metadata(layer_class: str, layer) -> dict[str, Any]:
    if layer_class != "QSum":
        return {
            "axis": None,
            "scale": None,
            "keepdims": None,
        }
    axes = getattr(layer, "axes", None)
    if axes is not None and not isinstance(axes, tuple):
        axes = tuple(axes) if isinstance(axes, (list, set)) else (axes,)
    return {
        "axis": axes,
        "scale": getattr(layer, "scale", None),
        "keepdims": getattr(layer, "keepdims", None),
    }


def _extract_record(
    layer,
    idx: int,
    *,
    include_values: bool,
    include_histograms: bool,
    value_storage: str,
) -> dict[str, Any]:
    in_shapes, out_shapes = _layer_shapes(layer)
    equation = getattr(layer, "equation", None)
    act = getattr(layer, "activation", None)
    activation = getattr(act, "__name__", None) if act is not None else None

    try:
        num_params = int(layer.count_params())
    except Exception:
        num_params = None

    qmeta = _hgq.extract_all_quantizers(layer)
    values = _hgq.extract_layer_values(layer) if include_values else {
        "kernel_values": None,
        "kernel_float_values": None,
        "bias_values": None,
        "batchnorm_values": None,
        "qkernel_values": None,
        "qbias_values": None,
        "uses_qkernel": False,
    }
    stats = _hgq.weight_stats(values["kernel_values"], include_histogram=include_histograms)
    kernel_values = values["kernel_values"]
    kernel_shape = tuple(kernel_values.shape) if kernel_values is not None else None
    layer_class = type(layer).__name__
    dense_geometry = _dense_geometry(layer_class, in_shapes, out_shapes, kernel_shape)
    reduction_metadata = _reduction_metadata(layer_class, layer)
    equation = equation or dense_geometry.get("equation")

    record = {
        "layer_name": layer.name,
        "layer_class": layer_class,
        "layer_idx": idx,
        "op_kind": _classify(layer),
        "equation": equation,
        "activation": activation,
        "kernel_shape": kernel_shape,
        "axis": reduction_metadata["axis"],
        "scale": reduction_metadata["scale"],
        "keepdims": reduction_metadata["keepdims"],
        "dense_contract_axis": dense_geometry["dense_contract_axis"],
        "feature_axis": dense_geometry["feature_axis"],
        "units": dense_geometry["units"],
        "equation_synthesized": bool(dense_geometry["equation_synthesized"] and equation is not None),
        "foldable": layer_class != "Flatten",
        "cost_mode": "synthetic_zero" if layer_class == "Flatten" else None,
        "has_bn": layer_class == "QEinsumDenseBatchnorm",
        "bn_folded_into_qkernel": bool(values["qkernel_values"] is not None and layer_class == "QEinsumDenseBatchnorm"),
        "in_shapes": in_shapes,
        "out_shapes": out_shapes,
        "kernel_values": values["kernel_values"] if value_storage == "inline" else None,
        "kernel_float_values": values["kernel_float_values"] if value_storage == "inline" else None,
        "bias_values": values["bias_values"] if value_storage == "inline" else None,
        "batchnorm_values": values["batchnorm_values"] if value_storage == "inline" else None,
        "qkernel_values": values["qkernel_values"] if value_storage == "inline" else None,
        "qbias_values": values["qbias_values"] if value_storage == "inline" else None,
        "uses_qkernel": values["uses_qkernel"],
        "num_params": num_params,
        "quantizer_granularity": qmeta["iq"].get("granularity") if qmeta["iq"] else None,
        "quantizer_place": "layer",
        "quantizer_source": "HGQ",
        "kernel_sparsity": stats["sparsity"],
        "kernel_nonzero_count": stats["nonzero_count"],
        "kernel_zero_count": stats["zero_count"],
        "kernel_unique_values": stats["unique_values"] if include_values and value_storage == "inline" else None,
        "kernel_unique_count": stats["unique_count"],
        "kernel_value_histogram": stats["value_histogram"],
        "kernel_min": stats["min"],
        "kernel_max": stats["max"],
        "kernel_dtype": stats["dtype"],
    }
    record.update(_quant_summary_fields(qmeta["iq"], "iq"))
    record.update(_quant_summary_fields(qmeta["kq"], "kq"))
    record.update(_quant_summary_fields(qmeta["bq"], "bq"))
    record.update(_quant_summary_fields(qmeta["oq"], "oq"))
    record["aq"] = qmeta["aq"]

    # Legacy aliases.
    record["iq_bw"] = record["iq_bw_avg"]
    record["kq_bw"] = record["kq_bw_avg"]
    record["bq_bw"] = record["bq_bw_avg"]
    record["iq_bw_per_param"] = (
        qmeta["iq"]["kif"].get("bits") if qmeta["iq"] and qmeta["iq"].get("kif") else None
    )
    record["kq_bw_per_param"] = (
        qmeta["kq"]["kif"].get("bits") if qmeta["kq"] and qmeta["kq"].get("kif") else None
    )
    record["sparsity"] = record["kernel_sparsity"]
    record["weights"] = record["kernel_values"]
    record["biases"] = record["bias_values"]
    return record


def _edge_kif(record: dict[str, Any], side: str) -> dict[str, Any] | None:
    if side == "src":
        return record.get("oq_kif") or (
            record.get("aq", {}).get("kif") if isinstance(record.get("aq"), dict) else None
        )
    return record.get("iq_kif")


def _edge_qint(record: dict[str, Any], side: str) -> dict[str, Any] | None:
    if side == "src":
        return record.get("oq_qint") or (
            record.get("aq", {}).get("qint") if isinstance(record.get("aq"), dict) else None
        )
    return record.get("iq_qint")


_REQUIRED_VX_FIELDS = ("layer_name", "op_kind", "in_shapes", "out_shapes")
_COMPUTE_OPS = {"dense", "einsum_dense", "einsum_dense_bn"}


def _validate(g: HGraph) -> None:
    for vx in g.vertices:
        p = g.pmap[vx]
        for key in _REQUIRED_VX_FIELDS:
            if p.get(key) is None:
                raise ValueError(
                    f"NN-IR vertex {vx} ({p.get('layer_name')!r}) is missing required field {key!r}"
                )
        if p.get("op_kind") in _COMPUTE_OPS and p.get("kernel_values") is None and p.get("qkernel_values") is None:
            raise ValueError(
                f"NN-IR vertex {vx} ({p.get('layer_name')!r}) is missing kernel values"
            )
        if p.get("op_kind") in _COMPUTE_OPS and p.get("kq") is None:
            warnings.warn(
                f"NN-IR vertex {p.get('layer_name')!r} has no kernel quantizer summary",
                stacklevel=2,
            )
        if p.get("op_kind") != "input" and p.get("iq") is None:
            warnings.warn(
                f"NN-IR vertex {p.get('layer_name')!r} has no input quantizer summary",
                stacklevel=2,
            )

    for edge in g.edges:
        p = g.pmap[edge]
        if p.get("tensor_shape") is None:
            raise ValueError(f"NN-IR edge {edge} is missing tensor_shape")
        if not any(
            p.get(key) is not None
            for key in ("dst_kif", "dst_qint", "src_kif", "src_qint", "bitwidth_dst", "bitwidth_src")
        ):
            warnings.warn(f"NN-IR edge {edge} has no precision metadata", stacklevel=2)


def _override_record_from_attention_spec(record: dict[str, Any], spec, parent_name: str) -> dict[str, Any]:
    record.update(
        {
            "layer_name": spec.name,
            "layer_idx": record.get("layer_idx"),
            "op_kind": spec.op_kind,
            "equation": spec.equation or record.get("equation"),
            "in_shapes": spec.in_shapes,
            "out_shapes": spec.out_shapes,
            "foldable": bool(spec.foldable),
            "dynamic_weight": bool(spec.dynamic_weight),
            "cost_mode": spec.cost_mode,
            "compound_parent": parent_name,
            "attention_role": spec.role,
            "keras_layer": spec.source_layer,
        }
    )
    return record


def _nonparam_attention_record(spec, layer_idx: int, parent_name: str) -> dict[str, Any]:
    qmeta = _hgq.extract_all_quantizers(spec.source_layer) if spec.source_layer is not None else {
        "iq": None,
        "kq": None,
        "bq": None,
        "oq": None,
        "aq": None,
    }
    record = {
        "layer_name": spec.name,
        "layer_class": type(spec.source_layer).__name__ if spec.source_layer is not None else f"Synthetic{spec.op_kind.title()}",
        "layer_idx": layer_idx,
        "op_kind": spec.op_kind,
        "equation": spec.equation,
        "activation": None,
        "kernel_shape": None,
        "axis": spec.axis,
        "stable": spec.stable,
        "input_qints": None,
        "input_kifs": None,
        "output_qint": None,
        "output_kif": None,
        "foldable": bool(spec.foldable),
        "dynamic_weight": bool(spec.dynamic_weight),
        "cost_mode": spec.cost_mode,
        "compound_parent": parent_name,
        "attention_role": spec.role,
        "keras_layer": spec.source_layer,
        "dense_contract_axis": None,
        "feature_axis": None,
        "units": None,
        "equation_synthesized": False,
        "has_bn": False,
        "bn_folded_into_qkernel": False,
        "in_shapes": spec.in_shapes,
        "out_shapes": spec.out_shapes,
        "kernel_values": None,
        "kernel_float_values": None,
        "bias_values": None,
        "batchnorm_values": None,
        "qkernel_values": None,
        "qbias_values": None,
        "uses_qkernel": False,
        "num_params": 0,
        "quantizer_granularity": None,
        "quantizer_place": "layer",
        "quantizer_source": "HGQ" if spec.source_layer is not None else "synthetic",
        "kernel_sparsity": None,
        "kernel_nonzero_count": None,
        "kernel_zero_count": None,
        "kernel_unique_values": None,
        "kernel_unique_count": None,
        "kernel_value_histogram": None,
        "kernel_min": None,
        "kernel_max": None,
        "kernel_dtype": None,
    }
    record.update(_quant_summary_fields(qmeta["iq"], "iq"))
    record.update(_quant_summary_fields(qmeta["kq"], "kq"))
    record.update(_quant_summary_fields(qmeta["bq"], "bq"))
    record.update(_quant_summary_fields(qmeta["oq"], "oq"))
    record["aq"] = qmeta["aq"]
    record["iq_bw"] = record["iq_bw_avg"]
    record["kq_bw"] = record["kq_bw_avg"]
    record["bq_bw"] = record["bq_bw_avg"]
    record["iq_bw_per_param"] = None
    record["kq_bw_per_param"] = None
    record["sparsity"] = None
    record["weights"] = None
    record["biases"] = None
    return record


def _output_qint(record: dict[str, Any]):
    return _edge_qint(record, "src") or record.get("output_qint") or record.get("oq_qint")


def _output_kif(record: dict[str, Any]):
    return _edge_kif(record, "src") or record.get("output_kif") or record.get("oq_kif")


def _input_qint(record: dict[str, Any]):
    return _edge_qint(record, "dst")


def _input_kif(record: dict[str, Any]):
    return _edge_kif(record, "dst")


def _stamp_attention_precision(records_by_name: dict[str, dict[str, Any]]) -> None:
    by_parent_role = {
        (record.get("compound_parent"), record.get("attention_role")): record
        for record in records_by_name.values()
        if record.get("compound_parent") is not None
    }
    parents = {parent for parent, _ in by_parent_role}
    for parent in parents:
        query = by_parent_role.get((parent, "query"))
        key = by_parent_role.get((parent, "key"))
        value = by_parent_role.get((parent, "value"))
        qk = by_parent_role.get((parent, "qk"))
        softmax = by_parent_role.get((parent, "softmax"))
        av = by_parent_role.get((parent, "av"))
        output = by_parent_role.get((parent, "attention_output"))
        if all(x is not None for x in (query, key, qk)):
            qk["input_qints"] = [_output_qint(key), _output_qint(query)]
            qk["input_kifs"] = [_output_kif(key), _output_kif(query)]
            qk["iq_qint"] = next((q for q in qk["input_qints"] if q is not None), None)
            qk["iq_kif"] = next((k for k in qk["input_kifs"] if k is not None), None)
            qk["iq_bw"] = _hgq.bitwidth_from_kif(qk["iq_kif"])
        if qk is not None and softmax is not None:
            qk["output_qint"] = _input_qint(softmax)
            qk["output_kif"] = _input_kif(softmax)
            qk["oq_qint"] = qk["output_qint"]
            qk["oq_kif"] = qk["output_kif"]
            qk["oq_bw"] = _hgq.bitwidth_from_kif(qk["output_kif"])
            softmax["input_qints"] = [qk["output_qint"]]
            softmax["input_kifs"] = [qk["output_kif"]]
        if softmax is not None:
            softmax["output_qint"] = _output_qint(softmax)
            softmax["output_kif"] = _output_kif(softmax)
        if all(x is not None for x in (softmax, value, av)):
            av["input_qints"] = [_output_qint(softmax), _output_qint(value)]
            av["input_kifs"] = [_output_kif(softmax), _output_kif(value)]
            av["iq_qint"] = next((q for q in av["input_qints"] if q is not None), None)
            av["iq_kif"] = next((k for k in av["input_kifs"] if k is not None), None)
            av["iq_bw"] = _hgq.bitwidth_from_kif(av["iq_kif"])
        if av is not None and output is not None:
            av["output_qint"] = _input_qint(output)
            av["output_kif"] = _input_kif(output)
            av["oq_qint"] = av["output_qint"]
            av["oq_kif"] = av["output_kif"]
            av["oq_bw"] = _hgq.bitwidth_from_kif(av["output_kif"])


def _edge_shape(src_rec: dict[str, Any]) -> tuple | None:
    return tuple(src_rec["out_shapes"][0]) if src_rec.get("out_shapes") else None


def _add_edge_with_metadata(
    g: HGraph,
    rec_by_name: dict[str, dict[str, Any]],
    name2vx: dict[str, int],
    src_name: str,
    dst_name: str,
    *,
    consume_mode: str = "stepwise",
) -> None:
    if src_name not in name2vx or dst_name not in name2vx:
        return
    src_vx = name2vx[src_name]
    dst_vx = name2vx[dst_name]
    if src_vx == dst_vx:
        return

    created = g.add_edge(src_vx, dst_vx)
    if not created:
        return

    src_rec = rec_by_name[src_name]
    dst_rec = rec_by_name[dst_name]
    edge = created[0]
    shape = _edge_shape(src_rec)
    src_kif = _edge_kif(src_rec, "src")
    dst_kif = _edge_kif(dst_rec, "dst")
    src_qint = _edge_qint(src_rec, "src")
    dst_qint = _edge_qint(dst_rec, "dst")
    bw_src = _hgq.bitwidth_from_kif(src_kif)
    bw_dst = _hgq.bitwidth_from_kif(dst_kif)
    element_bw = bw_src if bw_src is not None else bw_dst
    tensor_width = None
    if shape is not None and element_bw is not None:
        nbv = _non_batch_volume(shape)
        if nbv is not None:
            tensor_width = float(nbv * element_bw)

    boundary = False
    if src_kif is not None and dst_kif is not None:
        boundary = repr(src_kif) != repr(dst_kif)

    g.pmap[edge].update(
        {
            "tensor_shape": shape,
            "src_qint": src_qint,
            "src_kif": src_kif,
            "src_bitwidth_bits": bw_src,
            "dst_qint": dst_qint,
            "dst_kif": dst_kif,
            "dst_bitwidth_bits": bw_dst,
            "element_bitwidth_bits": element_bw,
            "element_kif": src_kif if src_kif is not None else dst_kif,
            "element_qint": src_qint if src_qint is not None else dst_qint,
            "tensor_width_bits": tensor_width,
            "volume_bits_exact": tensor_width,
            "has_quantization_boundary": boundary,
            "producer_quantizer": src_rec.get("oq") or src_rec.get("aq"),
            "consumer_quantizer": dst_rec.get("iq"),
            "consume_mode": consume_mode,
            "needs_cast": boundary,
            "cast_mode": "requantize" if boundary else None,
            # Legacy aliases.
            "bitwidth_src": bw_src,
            "bitwidth_dst": bw_dst,
            "volume_bits": tensor_width,
        }
    )


def build_nn_ir(
    model,
    name: str | None = None,
    validate: bool = True,
    *,
    include_values: bool = True,
    include_histograms: bool = True,
    value_storage: str = "inline",
) -> HGraph:
    g = HGraph(vinit=vinit_nn, einit=einit_nn, ginit=ginit_nn)

    g.pmap["name"] = name or model.name
    g.pmap["model_source"] = "keras_hgq"
    try:
        g.pmap["n_features"] = int(model.input_shape[-1])
    except Exception:
        g.pmap["n_features"] = None
    try:
        g.pmap["n_classes"] = int(model.output_shape[-1])
    except Exception:
        g.pmap["n_classes"] = None

    records: list[dict[str, Any]] = []
    synthetic_edges: list[AttentionEdgeSpec] = []
    output_alias: dict[str, str] = {}
    synthetic_idx = 0

    for i, layer in enumerate(model.layers):
        if is_attention_layer(layer):
            lowering = lower_attention_layer(
                layer,
                _layer_inbound(layer),
                _layer_shapes(layer)[0],
                _layer_shapes(layer)[1],
            )
            output_alias[layer.name] = lowering.output_name
            synthetic_edges.extend(lowering.edges)
            for spec in lowering.records:
                layer_idx = i * 100 + synthetic_idx
                synthetic_idx += 1
                if spec.source_layer is not None and spec.op_kind == "einsum_dense":
                    record = _extract_record(
                        spec.source_layer,
                        layer_idx,
                        include_values=include_values,
                        include_histograms=include_histograms,
                        value_storage=value_storage,
                    )
                    record = _override_record_from_attention_spec(record, spec, layer.name)
                else:
                    record = _nonparam_attention_record(spec, layer_idx, layer.name)
                records.append(record)
            continue

        records.append(
            _extract_record(
                layer,
                i,
                include_values=include_values,
                include_histograms=include_histograms,
                value_storage=value_storage,
            )
        )

    _stamp_attention_precision({record["layer_name"]: record for record in records})
    rec_by_name = {record["layer_name"]: record for record in records}

    name2vx: dict[str, int] = {}
    for record in records:
        vx = g.add_vx()
        name2vx[record["layer_name"]] = vx
        g.pmap[vx].update(record)

    for edge_spec in synthetic_edges:
        _add_edge_with_metadata(
            g,
            rec_by_name,
            name2vx,
            output_alias.get(edge_spec.source, edge_spec.source),
            output_alias.get(edge_spec.destination, edge_spec.destination),
            consume_mode=edge_spec.consume_mode,
        )

    for layer in model.layers:
        if is_attention_layer(layer):
            continue
        dst_name = layer.name

        for src_name in _layer_inbound(layer):
            _add_edge_with_metadata(
                g,
                rec_by_name,
                name2vx,
                output_alias.get(src_name, src_name),
                output_alias.get(dst_name, dst_name),
            )

    if validate:
        _validate(g)
    return g
