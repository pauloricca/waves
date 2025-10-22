from __future__ import annotations
from typing import Set, Optional
from types import SimpleNamespace
from nodes.node_utils.base_node import BaseNode
from nodes.node_utils.node_registry import NODE_REGISTRY
from nodes.node_utils.auto_id_generator import AutoIDGenerator

def instantiate_node(obj, hot_reload: bool = False, previous_ids: Optional[Set[str]] = None) -> BaseNode:
    """
    Instantiate a node from a model.
    
    Args:
        obj: The node model to instantiate.
        hot_reload: If True, this is a hot reload - pass to node constructors.
        previous_ids: Set of node IDs that existed before the reload (for hot reload context).
    
    Returns:
        An instantiated BaseNode.
    
    The hot_reload flag and previous_ids are used to determine whether a node's
    state should be initialized fresh or preserved from a previous instance.
    When a node's ID was in previous_ids, hot_reload=True is passed to the node.
    
    IDs are determined in priority order:
    1. Explicit 'id' field in the model
    2. Auto-generated hierarchical ID (e.g., "parent.0.child")
    """
    if previous_ids is None:
        previous_ids = set()
    
    # Get the effective ID (explicit or auto-generated)
    effective_id = AutoIDGenerator.get_effective_id(obj)
    
    # Determine if this specific node should initialize state
    # A node gets hot_reload=True if its id existed in the previous tree
    node_hot_reload = hot_reload and (effective_id in previous_ids if effective_id else False)
    
    # Create a state object for the node if it has an id (explicit or auto-generated)
    # The state will be initialized by the node itself (only if not hot_reload)
    state = SimpleNamespace() if effective_id else None
    
    for node_definition in NODE_REGISTRY:
        if isinstance(obj, node_definition.model):
            # Check if the node class supports state/hot_reload parameters
            # by looking at its __init__ signature
            try:
                # Try calling with state and hot_reload parameters first (new hot-reload pattern)
                return node_definition.node(obj, state=state, hot_reload=node_hot_reload)
            except TypeError:
                # Fall back to old pattern without state/hot_reload
                return node_definition.node(obj)

    raise ValueError(f"Unknown model type: {type(obj)}")


def instantiate_node_tree(model, hot_reload: bool = False, previous_ids: Optional[Set[str]] = None) -> BaseNode:
    """
    Instantiate a complete tree of nodes with hot reload support.
    
    This is a convenience wrapper around instantiate_node that properly handles
    the hot_reload flag and previous_ids set for tree instantiation.
    
    Args:
        model: The root node model to instantiate.
        hot_reload: If True, this is a hot reload scenario.
        previous_ids: Set of node IDs from the previous tree (for matching nodes to restore state).
    
    Returns:
        The root node of the newly instantiated tree.
    """
    return instantiate_node(model, hot_reload=hot_reload, previous_ids=previous_ids or set())
