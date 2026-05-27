"""Notebook-equivalent HGQ layer reconstruction helpers for folded traces."""

from __future__ import annotations

from typing import Any

import numpy as np


def fold_spatial_max(value: np.ndarray, fold_factor: int) -> np.ndarray:
    """Collapse per-position precision metadata after spatial folding."""
    if value.ndim != 3:
        raise ValueError(f"fold_spatial_max expects a 3D array, got {value.shape}")
    batch, spatial, channels = value.shape
    if spatial % fold_factor != 0:
        raise ValueError(f"Cannot fold spatial dimension {spatial} by factor {fold_factor}")
    return value.reshape(batch, spatial // fold_factor, fold_factor, channels).max(axis=2)


def _compatible_folded_weight(
    source_value: np.ndarray,
    folded_value: np.ndarray,
    fold_factor: int,
) -> tuple[np.ndarray | None, str | None]:
    if source_value.shape == folded_value.shape:
        return source_value, "copy"
    if (
        source_value.ndim == 3
        and folded_value.ndim == 3
        and source_value.shape[0] == folded_value.shape[0]
        and source_value.shape[2] == folded_value.shape[2]
        and source_value.shape[1] == folded_value.shape[1] * fold_factor
    ):
        return fold_spatial_max(source_value, fold_factor), "fold_spatial_max"
    return None, None


def _relative_weight_path(variable, owner_layer) -> str | None:
    path = getattr(variable, "path", None)
    if path is None:
        return None
    parts = str(path).split("/")
    if owner_layer.name not in parts:
        return None
    return "/".join(parts[parts.index(owner_layer.name) + 1 :])


def copy_folded_layer_weights(
    source_layer,
    folded_layer,
    *,
    fold_factor: int,
) -> list[dict[str, Any]]:
    """Copy weights by quantizer scope, raising rather than guessing collisions."""
    source_vars = list(source_layer.weights)
    folded_vars = list(folded_layer.weights)
    source_values = [np.asarray(variable.numpy()) for variable in source_vars]
    folded_values = [np.asarray(variable.numpy()) for variable in folded_vars]
    used_source: set[int] = set()
    copied: list[np.ndarray] = []
    log: list[dict[str, Any]] = []

    for folded_index, (folded_var, folded_value) in enumerate(zip(folded_vars, folded_values)):
        candidates = []
        for source_index, (source_var, source_value) in enumerate(zip(source_vars, source_values)):
            if source_index in used_source or source_var.name != folded_var.name:
                continue
            transformed, transform = _compatible_folded_weight(source_value, folded_value, fold_factor)
            if transformed is not None:
                candidates.append((source_index, source_var, transformed, transform))

        folded_path = _relative_weight_path(folded_var, folded_layer)
        path_matches = [
            candidate
            for candidate in candidates
            if folded_path is not None
            and _relative_weight_path(candidate[1], source_layer) == folded_path
        ]
        if len(path_matches) == 1:
            selected = path_matches[0]
        elif len(path_matches) > 1 or len(candidates) > 1:
            raise ValueError(
                f"Ambiguous folded weight copy for {folded_var.name}:{folded_value.shape}"
            )
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            copied.append(folded_value)
            log.append(
                {
                    "new_index": folded_index,
                    "new_name": folded_var.name,
                    "new_shape": tuple(folded_value.shape),
                    "old_index": None,
                    "transform": "keep_new_init",
                }
            )
            continue

        source_index, source_var, transformed, transform = selected
        used_source.add(source_index)
        copied.append(transformed)
        log.append(
            {
                "new_index": folded_index,
                "new_name": folded_var.name,
                "new_shape": tuple(folded_value.shape),
                "old_index": source_index,
                "old_name": source_var.name,
                "old_shape": tuple(source_values[source_index].shape),
                "transform": transform,
            }
        )

    folded_layer.set_weights(copied)
    return log


def build_folded_entry_layer_model(source_layer, *, fold_factor: int):
    """Rebuild the notebook's folded first HGQ einsum layer as a Keras model."""
    import keras
    import hgq  # noqa: F401  # Registers HGQ custom layers for reconstruction.

    if fold_factor < 1:
        raise ValueError(f"fold_factor must be positive, got {fold_factor}")
    input_shape = tuple(source_layer.input.shape[1:])
    output_shape = tuple(source_layer.output.shape[1:])
    if len(input_shape) != 2 or len(output_shape) != 2:
        raise ValueError(
            f"Expected folded entry layer shapes (N, C), got {input_shape} -> {output_shape}"
        )
    n_old, channels_in = input_shape
    if n_old % fold_factor != 0:
        raise ValueError(f"Input lane count {n_old} is not divisible by {fold_factor}")
    n_folded = int(n_old) // fold_factor
    kwargs = {
        "name": f"{source_layer.name}_folded_f{fold_factor}",
    }
    if hasattr(source_layer, "bias_axes"):
        kwargs["bias_axes"] = source_layer.bias_axes
    if hasattr(source_layer, "activation"):
        kwargs["activation"] = source_layer.activation
    folded_input = keras.Input(
        shape=(n_folded, channels_in),
        name=f"folded_input_f{fold_factor}",
    )
    folded_layer = source_layer.__class__(
        source_layer.equation,
        (n_folded, output_shape[-1]),
        **kwargs,
    )
    folded_output = folded_layer(folded_input)
    folded_model = keras.Model(
        inputs=folded_input,
        outputs=folded_output,
        name=f"folded_entry_f{fold_factor}",
    )
    weight_copy_log = copy_folded_layer_weights(
        source_layer,
        folded_layer,
        fold_factor=fold_factor,
    )
    return folded_model, weight_copy_log
