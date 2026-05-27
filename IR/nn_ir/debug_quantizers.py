from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "JEDI-linear" / "src"))
sys.path.insert(0, str(REPO / "heterograph"))
from IR.nn_ir import builder

from model import get_gnn  # noqa: E402


def _fmt_bw(value):
    return "?" if value is None else f"{value:.2f}"


def _fmt_shape(shape):
    if shape is None:
        return "?"
    return "x".join("?" if dim is None else str(dim) for dim in shape)


def _fmt_kif(kif):
    if not kif:
        return "?"
    bits = kif.get("bits")
    shape = kif.get("shape")
    if bits is None:
        return f"shape={shape}"
    if hasattr(bits, "shape"):
        return f"shape={shape} max={float(bits.max()):.2f}"
    return str(bits)


def main() -> None:
    conf = SimpleNamespace(n_constituents=8, pt_eta_phi=True)
    model = get_gnn(conf)
    graph = builder.build_nn_ir(model, name="jedi_gnn")

    print(
        "layer | op | iq_kif | kq_kif | qkernel_shape | sparsity | unique_values"
    )
    print("-" * 100)
    for vx in graph.vertices:
        p = graph.pmap[vx]
        print(
            f"{p['layer_name']} | {p['op_kind']} | "
            f"{_fmt_kif(p.get('iq_kif'))} | {_fmt_kif(p.get('kq_kif'))} | "
            f"{_fmt_shape(p.get('kernel_shape'))} | {_fmt_bw(p.get('kernel_sparsity'))} | "
            f"{p.get('kernel_unique_count')}"
        )


if __name__ == "__main__":
    main()
