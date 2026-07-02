"""Per-op parameter records attached to Sched-IR compute/transport nodes."""

from .activation import default_activation_params
from .dense import default_dense_params
from .elementwise import default_elementwise_params
from .einsum import default_einsum_params
from .reduce import default_reduce_params
from .softmax import default_softmax_params
from .transport import default_transport_params

__all__ = [
    "default_dense_params",
    "default_reduce_params",
    "default_elementwise_params",
    "default_einsum_params",
    "default_activation_params",
    "default_softmax_params",
    "default_transport_params",
]
