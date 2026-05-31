from __future__ import annotations

from .types.edges import default_edge_properties
from .types.graph import default_graph_properties
from .types.nodes import default_node_properties


def vinit_sched(g, vx):
    g.pmap[vx] = default_node_properties()


def einit_sched(g, e):
    g.pmap[e] = default_edge_properties()


def ginit_sched(g):
    g.pmap.update(default_graph_properties())
