from __future__ import annotations
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_registry import NODE_REGISTRY
from nodes.node_utils.node_state_registry import get_state_registry
from nodes.wavable_value import WavableValue, WavableValueModel, WavableValueNode
from random import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


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


def collect_all_node_ids(node_model: BaseModel, parent_id: str, attribute_name: str, attribute_index: int | None = None) -> set[str]:
    """Recursively collect all node IDs from a node tree.
    
    This traverses the entire node tree and collects all explicit and auto-generated IDs,
    enabling initialization of placeholders for forward references.
    
    Args:
        node_model: The root node model to traverse
        parent_id: Parent node ID for ID generation
        attribute_name: Attribute name for ID generation 
        attribute_index: Index for list attributes
        
    Returns:
        Set of all node IDs found in the tree
    """
    ids = set()
    
    # Handle primitive values - they get random IDs but we don't need to track them
    if isinstance(node_model, (int, float, str, list)):
        return ids
    
    # Generate ID for this node using same logic as instantiate_node
    if hasattr(node_model, 'id') and node_model.id is not None:
        node_id = node_model.id
    else:
        node_id = f"{parent_id}.{attribute_name}"
        if attribute_index is not None:
            node_id += f".{attribute_index}"
        node_id += f".{node_model.__class__.__name__}"
    
    ids.add(node_id)
    
    # Recursively collect IDs from all child nodes
    if hasattr(node_model, '__dict__'):
        for attr_name, attr_value in node_model.__dict__.items():
            if attr_name.startswith('_'):  # Skip private attributes
                continue
                
            if isinstance(attr_value, BaseNodeModel):
                # Single child node
                child_ids = collect_all_node_ids(attr_value, node_id, attr_name)
                ids.update(child_ids)
            elif isinstance(attr_value, list):
                # List of child nodes
                for i, item in enumerate(attr_value):
                    if isinstance(item, BaseNodeModel):
                        child_ids = collect_all_node_ids(item, node_id, attr_name, i)
                        ids.update(child_ids)
            elif hasattr(attr_value, '__dict__'):  # Check for nested models
                # Handle nested models (like ConfigDict with extra fields)
                for nested_name, nested_value in attr_value.__dict__.items():
                    if isinstance(nested_value, BaseNodeModel):
                        child_ids = collect_all_node_ids(nested_value, node_id, nested_name)
                        ids.update(child_ids)
                    elif isinstance(nested_value, list):
                        for i, item in enumerate(nested_value):
                            if isinstance(item, BaseNodeModel):
                                child_ids = collect_all_node_ids(item, node_id, nested_name, i)
                                ids.update(child_ids)
    
    # Handle special cases like ConfigDict with extra fields (for mix, expression nodes)
    if hasattr(node_model, '__pydantic_extra__') and node_model.__pydantic_extra__ is not None:
        for attr_name, attr_value in node_model.__pydantic_extra__.items():
            if isinstance(attr_value, BaseNodeModel):
                child_ids = collect_all_node_ids(attr_value, node_id, attr_name)
                ids.update(child_ids)
            elif isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    if isinstance(item, BaseNodeModel):
                        child_ids = collect_all_node_ids(item, node_id, attr_name, i)
                        ids.update(child_ids)
    
    return ids

    

    
