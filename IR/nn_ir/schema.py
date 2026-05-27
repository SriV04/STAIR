from __future__ import annotations

from .types.edges import default_edge_properties
from .types.nodes import default_node_properties


def vinit_nn(g, vx):
    g.pmap[vx] = default_node_properties()


def einit_nn(g, e):
    g.pmap[e] = default_edge_properties()


def ginit_nn(g):
    g.pmap["name"] = None
    g.pmap["model_source"] = None
    g.pmap["n_features"] = None
    g.pmap["n_classes"] = None
