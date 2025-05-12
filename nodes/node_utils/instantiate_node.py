from __future__ import annotations
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_registry import NODE_REGISTRY

def instantiate_node(obj) -> BaseNode:
    for node_definition in NODE_REGISTRY:
        if isinstance(obj, node_definition.model):
            return node_definition.node(obj)

    raise ValueError(f"Unknown model type: {type(obj)}")
