from __future__ import annotations

from .types.edges import default_edge_properties
from .types.enums import (
    ACTIVATION_FUNCS,
    BUFFER_KINDS,
    EDGE_KINDS,
    ELEMENTWISE_OPS,
    INSERTED_BY,
    MUX_KINDS,
    OP_PRIMITIVES,
    PRECISION_SOURCES,
    REDUCE_IMPL_MODES,
    REDUCE_MODES,
    REDUCE_OPS,
)
from .types.graph import default_graph_properties
from .types.nodes import default_node_properties


def vinit_sched(g, vx):
    g.pmap[vx] = default_node_properties()


def einit_sched(g, e):
    g.pmap[e] = default_edge_properties()


def ginit_sched(g):
    g.pmap.update(default_graph_properties())
