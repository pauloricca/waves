from __future__ import annotations
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_registry import NODE_REGISTRY
from nodes.node_utils.node_state_registry import get_state_registry
from nodes.wavable_value import WavableValue, WavableValueModel, WavableValueNode
from random import random


def instantiate_node(node_model_or_value: WavableValue, parent_id: str, attribute_name: str, attribute_index: int | None = None) -> BaseNode:
    """
    Instantiate a node from a model.
    
    Args:
        node_model: The node model to instantiate.
        parent_id: The parent node's ID for runtime ID generation.
        attribute_name: The attribute name in the parent on which this node is defined.
        attribute_index: Index in the attribute if it's a list, else None.
    
    Returns:
        An instantiated BaseNode.
    
    State management:
    - ALL nodes get an ID (explicit, auto-generated, or runtime-generated)
    - ALL nodes get state from the global registry
    - States are preserved across hot reloads
    - States accumulate over time for now (not cleaned up)
    """
    # Get or generate effective ID for this node
    if isinstance(node_model_or_value, (int, float, str, list)):
        node_id = random().hex()
    elif node_model_or_value.id is not None:
        node_id = node_model_or_value.id
    else:
        node_id = f"{parent_id}.{attribute_name}"
        if attribute_index is not None:
            node_id += f".{attribute_index}"
        node_id += f".{node_model_or_value.__class__.__name__}"
    
    # Get state from global registry - ALL nodes get state now
    state_registry = get_state_registry()
    
    state = state_registry.get_state(node_id)

    if state is None:
        do_initialise_state = True
        state = state_registry.create_state(node_id)
    else:
        do_initialise_state = False
    
    node: BaseNode | None = None

    if isinstance(node_model_or_value, BaseNodeModel):
        for node_definition in NODE_REGISTRY:
            if isinstance(node_model_or_value, node_definition.model):
                node = node_definition.node(node_model_or_value, node_id=node_id, state=state, do_initialise_state=do_initialise_state)
    else:
        node = WavableValueNode(WavableValueModel(value=node_model_or_value), node_id=node_id, state=state, do_initialise_state=do_initialise_state)
    
    if node is None:
        raise ValueError(f"Unknown model type: {type(node_model_or_value)}")

    return node

    
