"""NN-IR package."""

__all__ = ["build_nn_ir"]


def build_nn_ir(*args, **kwargs):
    from .builder import build_nn_ir as _build_nn_ir

    return _build_nn_ir(*args, **kwargs)
