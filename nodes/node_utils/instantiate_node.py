from __future__ import annotations
from nodes.node_utils.base_node import BaseNode
from nodes.node_utils.node_registry import NODE_REGISTRY
from nodes.node_utils.auto_id_generator import AutoIDGenerator
from nodes.node_utils.node_state_registry import get_state_registry

# Runtime ID generation context (thread-local would be better for multi-threading)
_runtime_id_counter = 0
_runtime_id_prefix = None

def set_runtime_id_prefix(prefix: str = None):
    """Set the prefix for runtime ID generation. Used when creating nodes dynamically."""
    global _runtime_id_prefix, _runtime_id_counter
    _runtime_id_prefix = prefix
    _runtime_id_counter = 0

def get_next_runtime_id() -> str:
    """Generate the next runtime ID based on current prefix and counter."""
    global _runtime_id_counter
    if _runtime_id_prefix:
        runtime_id = f"{_runtime_id_prefix}.runtime_{_runtime_id_counter}"
    else:
        runtime_id = f"runtime_{_runtime_id_counter}"
    _runtime_id_counter += 1
    return runtime_id

def instantiate_node(obj, hot_reload: bool = False) -> BaseNode:
    """
    Instantiate a node from a model.
    
    Args:
        obj: The node model to instantiate.
        hot_reload: If True, this is a hot reload scenario.
    
    Returns:
        An instantiated BaseNode.
    
    State management:
    - ALL nodes get an ID (explicit, auto-generated, or runtime-generated)
    - ALL nodes get state from the global registry
    - States are preserved across hot reloads
    - States accumulate over time (not cleaned up)
    """
    # Get or generate effective ID for this node
    effective_id = AutoIDGenerator.get_effective_id(obj)
    
    # If no explicit or auto ID, generate a runtime ID
    if effective_id is None:
        effective_id = get_next_runtime_id()
        # Store it as auto_id for future reference
        obj.__auto_id__ = effective_id
    else:
        # If there's a runtime prefix, prepend it to the ID
        # This ensures sequencer-created sounds get unique IDs
        if _runtime_id_prefix:
            effective_id = f"{_runtime_id_prefix}.{effective_id}"
    
    # Get state from global registry - ALL nodes get state now
    state_registry = get_state_registry()
    old_state_existed = effective_id in state_registry._states
    state = state_registry.get_or_create_state(effective_id, hot_reload=hot_reload)
    
    # Node should hot_reload only if it had state before
    node_hot_reload = hot_reload and old_state_existed
    
    for node_definition in NODE_REGISTRY:
        if isinstance(obj, node_definition.model):
            # All nodes now accept state and hot_reload parameters
            return node_definition.node(obj, state=state, hot_reload=node_hot_reload)

    raise ValueError(f"Unknown model type: {type(obj)}")


def instantiate_node_tree(model, hot_reload: bool = False) -> BaseNode:
    """
    Instantiate a complete tree of nodes with hot reload support.
    
    Args:
        model: The root node model to instantiate.
        hot_reload: If True, this is a hot reload scenario.
    
    Returns:
        The root node of the newly instantiated tree.
    """
    # Simply instantiate the tree - states are managed by the global registry
    return instantiate_node(model, hot_reload=hot_reload)
