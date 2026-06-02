"""HGQ2 attention lowering helpers for NN-IR construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ATTENTION_LAYER_CLASSES = {"QMultiHeadAttention", "QLinformerAttention"}


@dataclass(frozen=True)
class AttentionNodeSpec:
    name: str
    op_kind: str
    source_layer: Any | None
    in_shapes: list[tuple]
    out_shapes: list[tuple]
    equation: str | None = None
    role: str | None = None
    foldable: bool = True
    cost_mode: str = "da4ml_trace"
    dynamic_weight: bool = False
    axis: tuple[int, ...] | None = None
    stable: bool | None = None


@dataclass(frozen=True)
class AttentionEdgeSpec:
    source: str
    destination: str
    consume_mode: str = "stepwise"


@dataclass(frozen=True)
class AttentionLowering:
    records: tuple[AttentionNodeSpec, ...]
    edges: tuple[AttentionEdgeSpec, ...]
    output_name: str


def is_attention_layer(layer) -> bool:
    return type(layer).__name__ in ATTENTION_LAYER_CLASSES


def _as_tuple_shape(shape) -> tuple:
    return tuple(None if dim is None else int(dim) for dim in tuple(shape))


def _first_shape(shapes: list[tuple] | None) -> tuple | None:
    return tuple(shapes[0]) if shapes else None


def _kernel_shape(layer) -> tuple | None:
    for attr in ("qkernel", "kernel"):
        value = getattr(layer, attr, None)
        if value is None:
            continue
        arr = np.array(value.value if hasattr(value, "value") else value)
        return tuple(int(dim) for dim in arr.shape)
    return None


def _sub_layer(layer, public_name: str, private_name: str):
    return getattr(layer, public_name, None) or getattr(layer, private_name)


def _infer_einsum_shape(equation: str | None, input_shapes: list[tuple]) -> tuple | None:
    if not equation or "->" not in equation:
        return None
    lhs, rhs = equation.replace(" ", "").split("->", 1)
    terms = lhs.split(",")
    if len(terms) != len(input_shapes):
        return None

    dims_by_label: dict[str, int | None] = {}
    for term, shape in zip(terms, input_shapes, strict=True):
        if "..." in term or len(term) != len(shape):
            return None
        for label, dim in zip(term, shape, strict=True):
            dim = None if dim is None else int(dim)
            previous = dims_by_label.get(label)
            if previous is None:
                dims_by_label[label] = dim
            elif dim is not None and previous != dim:
                return None

    if "..." in rhs:
        return None
    return tuple(dims_by_label.get(label) for label in rhs)


def _dense_output_shape(layer, input_shape: tuple) -> tuple:
    output_shape = _infer_einsum_shape(
        getattr(layer, "equation", None),
        [input_shape, _kernel_shape(layer)],
    )
    if output_shape is None:
        raise ValueError(
            f"Cannot infer output shape for attention sublayer "
            f"{getattr(layer, 'name', '?')!r} with equation {getattr(layer, 'equation', None)!r}"
        )
    return output_shape


def _lower_dense(prefix: str, role: str, layer, input_shape: tuple, *, foldable: bool = True) -> AttentionNodeSpec:
    output_shape = _dense_output_shape(layer, input_shape)
    return AttentionNodeSpec(
        name=f"{prefix}_{role}",
        op_kind="einsum_dense",
        source_layer=layer,
        in_shapes=[input_shape],
        out_shapes=[output_shape],
        equation=getattr(layer, "equation", None),
        role=role,
        foldable=foldable,
        cost_mode="da4ml_trace" if foldable else "synthetic_zero",
        dynamic_weight=False,
    )


def lower_attention_layer(layer, inbound_names: list[str], in_shapes: list[tuple], out_shapes: list[tuple]) -> AttentionLowering:
    """Expand one HGQ2 QMultiHeadAttention/QLinformerAttention layer."""
    layer_class = type(layer).__name__
    if layer_class not in ATTENTION_LAYER_CLASSES:
        raise TypeError(f"Cannot lower non-attention layer class {layer_class!r}")
    if not inbound_names or not in_shapes:
        raise ValueError(f"Attention layer {layer.name!r} has no visible Keras inputs")

    prefix = layer.name
    query_source = inbound_names[0]
    key_source = inbound_names[1] if len(inbound_names) > 1 else query_source
    value_source = inbound_names[2] if len(inbound_names) > 2 else key_source
    query_input_shape = _as_tuple_shape(in_shapes[0])
    key_input_shape = _as_tuple_shape(in_shapes[1] if len(in_shapes) > 1 else in_shapes[0])
    value_input_shape = _as_tuple_shape(in_shapes[2] if len(in_shapes) > 2 else key_input_shape)

    query_dense = _sub_layer(layer, "query_dense", "_query_dense")
    key_dense = _sub_layer(layer, "key_dense", "_key_dense")
    value_dense = _sub_layer(layer, "value_dense", "_value_dense")
    output_dense = _sub_layer(layer, "output_dense", "_output_dense")

    records: list[AttentionNodeSpec] = []
    edges: list[AttentionEdgeSpec] = []

    key_projection_source = key_source
    value_projection_source = value_source
    key_dense_input_shape = key_input_shape
    value_dense_input_shape = value_input_shape

    if layer_class == "QLinformerAttention":
        lin_k = _lower_dense(prefix, "lin_k_proj", getattr(layer, "_lin_k_proj"), key_input_shape, foldable=False)
        lin_v = _lower_dense(prefix, "lin_v_proj", getattr(layer, "_lin_v_proj"), value_input_shape, foldable=False)
        records.extend([lin_k, lin_v])
        edges.append(AttentionEdgeSpec(key_source, lin_k.name))
        edges.append(AttentionEdgeSpec(value_source, lin_v.name))
        key_projection_source = lin_k.name
        value_projection_source = lin_v.name
        key_dense_input_shape = lin_k.out_shapes[0]
        value_dense_input_shape = lin_v.out_shapes[0]

    query = _lower_dense(prefix, "query", query_dense, query_input_shape)
    key = _lower_dense(prefix, "key", key_dense, key_dense_input_shape)
    value = _lower_dense(prefix, "value", value_dense, value_dense_input_shape)
    records.extend([query, key, value])
    edges.extend(
        [
            AttentionEdgeSpec(query_source, query.name),
            AttentionEdgeSpec(key_projection_source, key.name),
            AttentionEdgeSpec(value_projection_source, value.name),
        ]
    )

    qk_shape = _infer_einsum_shape(
        getattr(layer, "_dot_product_equation", None),
        [key.out_shapes[0], query.out_shapes[0]],
    )
    if qk_shape is None:
        raise ValueError(f"Cannot infer QK shape for attention layer {layer.name!r}")
    qk = AttentionNodeSpec(
        name=f"{prefix}_qk",
        op_kind="einsum",
        source_layer=None,
        in_shapes=[key.out_shapes[0], query.out_shapes[0]],
        out_shapes=[qk_shape],
        equation=getattr(layer, "_dot_product_equation", None),
        role="qk",
        foldable=False,
        cost_mode="synthetic_zero",
        dynamic_weight=True,
    )

    softmax_layer = getattr(layer, "_softmax", None)
    softmax = AttentionNodeSpec(
        name=f"{prefix}_softmax",
        op_kind="softmax",
        source_layer=softmax_layer,
        in_shapes=[qk_shape],
        out_shapes=[qk_shape],
        role="softmax",
        foldable=False,
        cost_mode="synthetic_zero",
        dynamic_weight=True,
        axis=tuple(getattr(softmax_layer, "axes", ()) or ()),
        stable=bool(getattr(softmax_layer, "stable", True)),
    )

    av_shape = _infer_einsum_shape(
        getattr(layer, "_combine_equation", None),
        [softmax.out_shapes[0], value.out_shapes[0]],
    )
    if av_shape is None:
        raise ValueError(f"Cannot infer attention-value shape for attention layer {layer.name!r}")
    av = AttentionNodeSpec(
        name=f"{prefix}_av",
        op_kind="einsum",
        source_layer=None,
        in_shapes=[softmax.out_shapes[0], value.out_shapes[0]],
        out_shapes=[av_shape],
        equation=getattr(layer, "_combine_equation", None),
        role="av",
        foldable=False,
        cost_mode="synthetic_zero",
        dynamic_weight=True,
    )
    output = _lower_dense(prefix, "attention_output", output_dense, av_shape)
    if out_shapes:
        output = AttentionNodeSpec(
            **{
                **output.__dict__,
                "out_shapes": [_as_tuple_shape(out_shapes[0])],
            }
        )

    records.extend([qk, softmax, av, output])
    edges.extend(
        [
            AttentionEdgeSpec(key.name, qk.name, "all"),
            AttentionEdgeSpec(query.name, qk.name, "all"),
            AttentionEdgeSpec(qk.name, softmax.name, "all"),
            AttentionEdgeSpec(softmax.name, av.name, "all"),
            AttentionEdgeSpec(value.name, av.name, "all"),
            AttentionEdgeSpec(av.name, output.name, "all"),
        ]
    )

    return AttentionLowering(tuple(records), tuple(edges), output.name)
