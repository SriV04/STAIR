"""Sandbox for replaying the DA4ML trace on the JEDI-linear reference model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "JEDI-linear" / "src"))
sys.path.insert(0, str(REPO / "heterograph"))

_CACHE_ROOT = REPO / ".cache" / "da4ml_sandbox"
_NUMBA_CACHE = _CACHE_ROOT / "numba"
_MPL_CACHE = _CACHE_ROOT / "mpl"
_XDG_CACHE = _CACHE_ROOT / "xdg"

for cache_dir in (_NUMBA_CACHE, _MPL_CACHE, _XDG_CACHE):
    cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))


N_CONSTITUENTS = 8
USE_PERMINV = True
LOAD_TRAINED_WEIGHTS = True


def _runtime() -> SimpleNamespace:
    try:
        import keras
    except ModuleNotFoundError:
        try:
            from tensorflow import keras
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "da4ml_sandbox requires either the 'keras' package or TensorFlow's bundled "
                "Keras runtime. Activate the project ML environment before running it."
            ) from exc

    import hgq  # noqa: F401
    from da4ml.trace import FixedVariableArrayInput, HWConfig, comb_trace
    from da4ml.converter.hgq2 import parser as hgq2_parser
    from model import get_gnn

    return SimpleNamespace(
        keras=keras,
        get_gnn=get_gnn,
        FixedVariableArrayInput=FixedVariableArrayInput,
        HWConfig=HWConfig,
        comb_trace=comb_trace,
        flatten_arr=hgq2_parser._flatten_arr,
        parse_model=hgq2_parser.parse_model,
        replace_tensors=hgq2_parser.replace_tensors,
        registry=hgq2_parser._registry,
    )


def _checkpoint_glob() -> Path:
    variant_dir = "3-feature-perminv" if USE_PERMINV else "3-feature"
    return (
        REPO / "official_models" / variant_dir / f"jet_classifier_large_{N_CONSTITUENTS}" / "models"
    )


def _load_model(rt: SimpleNamespace):
    if LOAD_TRAINED_WEIGHTS:
        ckpts = sorted(_checkpoint_glob().glob("*.keras"))
        if ckpts:
            return rt.keras.models.load_model(ckpts[0]), ckpts[0].name
    conf = SimpleNamespace(n_constituents=N_CONSTITUENTS, pt_eta_phi=True)
    return rt.get_gnn(conf, uq1=USE_PERMINV), "fresh"


def describe_da_tensor(x, name: str = "") -> None:
    arr = np.ravel(x)
    print(f"{name} type={type(x)} shape={getattr(x, 'shape', None)} flat_len={len(arr)}")
    for j, v in enumerate(arr[:16]):
        print(
            f"  [{j}] type={type(v).__name__} "
            f"opr={getattr(v, 'opr', None)} "
            f"qint={getattr(getattr(v, 'unscaled', v), 'qint', None)} "
            f"latency={getattr(v, 'latency', None)} "
            f"factor={getattr(v, '_factor', None)}"
        )
    if len(arr) > 16:
        print(f"  ... {len(arr) - 16} more")


def inspect_dais_replay(
    model,
    hwconf=None,
    inputs_kif=None,
    rt: SimpleNamespace | None = None,
) -> dict[str, Any]:
    rt = rt or _runtime()
    hwconf = hwconf or rt.HWConfig(1, -1, -1)

    input_shapes = [inp.shape[1:] for inp in model.inputs]
    if any(None in shape or -1 in shape for shape in input_shapes):
        raise ValueError(f"Dynamic input shape found: {input_shapes}. Provide explicit symbolic inputs.")

    if inputs_kif is None:
        da_inputs = tuple(rt.FixedVariableArrayInput(shape, hwconf) for shape in input_shapes)
    else:
        raise NotImplementedError("Add FixedVariableArray.from_kif here if you need explicit input KIFs.")

    print("\n=== MODEL INPUTS ===")
    for kt, da in zip(model.inputs, da_inputs):
        print(f"Keras input name={kt.name}, keras_shape={kt.shape}")
        describe_da_tensor(da, "DA input")

    tensor_map = {
        keras_tensor: da_tensor
        for keras_tensor, da_tensor in zip(model.inputs, da_inputs)
    }

    flat_inputs = rt.flatten_arr(da_inputs)
    print("\n=== FLATTENED INPUTS TO comb_trace ===")
    describe_da_tensor(flat_inputs, "flat_inputs")

    trace = {}
    maybe_counts = {}

    print("\n=== OPERATION REPLAY ===")
    for depth_idx, ops_at_depth in enumerate(rt.parse_model(model)):
        print(f"\n--- graph_depth_order_index={depth_idx}, n_ops={len(ops_at_depth)} ---")

        for op_idx, op in enumerate(ops_at_depth):
            cls = op.operation.__class__
            name = op.operation.name

            print(f"\n[{op_idx}] {name} ({cls.__module__}.{cls.__name__})")
            print("  requires:")
            for tensor in op.requires:
                print(
                    f"    {tensor.name}: keras_shape={tensor.shape}, "
                    f"in_tensor_map={tensor in tensor_map}"
                )

            print("  produces:")
            for tensor in op.produces:
                print(f"    {tensor.name}: keras_shape={tensor.shape}")

            if cls is rt.keras.layers.InputLayer:
                print("  skipped InputLayer")
                continue

            args = rt.replace_tensors(tensor_map, op.args)
            kwargs = rt.replace_tensors(tensor_map, op.kwargs)

            print("  replaced args:")
            for arg_index, arg in enumerate(args):
                if hasattr(arg, "shape"):
                    print(
                        f"    arg[{arg_index}] type={type(arg).__name__}, "
                        f"shape={getattr(arg, 'shape', None)}"
                    )
                else:
                    print(f"    arg[{arg_index}] type={type(arg).__name__}, value={arg}")

            print("  registry lookup:")
            if cls not in rt.registry:
                print(f"    MISSING: {rt.registry.keys()}")
                raise KeyError(f"No HGQ2 DAIS mirror op registered for {cls}")

            mirror_cls_or_factory = rt.registry[cls]
            print(f"    _registry[{cls.__name__}] -> {mirror_cls_or_factory}")

            if isinstance(op.operation, rt.keras.Model):
                raise NotImplementedError("Nested model detected. Add recursive handling if needed.")

            mirror_op = mirror_cls_or_factory(op.operation)
            print(f"    mirror_op instance: {mirror_op}")

            raw_result = mirror_op(*args, **kwargs)
            if isinstance(raw_result, dict):
                dumped = raw_result
                final = dumped["final"]
                if not isinstance(final, tuple):
                    final = (final,)
            else:
                final = raw_result if isinstance(raw_result, tuple) else (raw_result,)
                dumped = {"final": final}

            print("  mirror op returned keys:", list(dumped.keys()))

            for key, value in dumped.items():
                if isinstance(value, tuple):
                    print(f"    dump[{key}] tuple len={len(value)}")
                    for idx, item in enumerate(value):
                        describe_da_tensor(item, f"dump[{key}][{idx}]")
                else:
                    describe_da_tensor(value, f"dump[{key}]")

            for keras_tensor, da_tensor in zip(op.produces, final):
                tensor_map[keras_tensor] = da_tensor
                print(
                    f"  tensor_map update: {keras_tensor.name} "
                    f"-> DA tensor shape={getattr(da_tensor, 'shape', None)}"
                )

            maybe_counts[name] = maybe_counts.get(name, 0) + 1
            unique_name = name if maybe_counts[name] == 1 else f"{name}#{maybe_counts[name] - 1}"
            for key, value in dumped.items():
                trace[f"/{unique_name}/{key}"] = value

            current_outputs = rt.flatten_arr(final)
            partial_comb = rt.comb_trace(flat_inputs, current_outputs)
            print(
                f"  partial comb_trace: shape={partial_comb.shape}, "
                f"ops={len(partial_comb.ops)}, cost={partial_comb.cost}, "
                f"latency={partial_comb.latency}"
            )

    final_outputs = tuple(tensor_map[keras_tensor] for keras_tensor in model.outputs)
    trace["final"] = final_outputs

    flat_outputs = rt.flatten_arr(final_outputs)

    print("\n=== FINAL OUTPUTS TO comb_trace ===")
    for kt, da in zip(model.outputs, final_outputs):
        print(f"Keras output name={kt.name}, keras_shape={kt.shape}")
        describe_da_tensor(da, "DA output")

    print("\n=== FLATTENED OUTPUTS TO comb_trace ===")
    describe_da_tensor(flat_outputs, "flat_outputs")

    comb = rt.comb_trace(flat_inputs, flat_outputs)

    print("\n=== FINAL comb_trace RESULT ===")
    print("CombLogic shape:", comb.shape)
    print("n_ops:", len(comb.ops))
    print("n_inputs:", comb.shape[0])
    print("n_outputs:", comb.shape[1])
    print("out_idxs:", comb.out_idxs[:20], "..." if len(comb.out_idxs) > 20 else "")
    print("out_shifts:", comb.out_shifts)
    print("out_negs:", comb.out_negs)
    print("cost:", comb.cost)
    print("latency:", comb.latency)

    print("\n=== FIRST 20 OPS ===")
    for index, op in enumerate(comb.ops[:20]):
        print(index, op)

    return {
        "da_inputs": da_inputs,
        "flat_inputs": flat_inputs,
        "final_outputs": final_outputs,
        "flat_outputs": flat_outputs,
        "trace": trace,
        "comb": comb,
        "tensor_map": tensor_map,
    }


def main() -> dict[str, Any]:
    rt = _runtime()
    model, model_label = _load_model(rt)
    print(f"Loaded model variant: {model_label}")
    return inspect_dais_replay(model, rt=rt)


if __name__ == "__main__":
    main()
