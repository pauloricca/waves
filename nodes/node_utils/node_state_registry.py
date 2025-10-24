"""
Global registry for node states.

This module provides a centralized state management system for nodes:
- States are stored globally, indexed by node ID
- During hot reload, states are preserved for nodes whose IDs still exist
- States are never automatically cleaned up (to avoid interfering with active sounds)
"""

from types import SimpleNamespace
from typing import Dict, Optional


class NodeStateRegistry:
    """Global registry for node states indexed by node ID."""
    
    def __init__(self):
        self._states: Dict[str, SimpleNamespace] = {}
    
    def get_or_create_state(self, node_id: str, hot_reload: bool = False) -> SimpleNamespace:
        """
        Get existing state for a node ID, or create a new one.
        
        Args:
            node_id: The node's unique ID
            hot_reload: If True, we're in a hot reload scenario
        
        Returns:
            The state object for this node (existing or newly created)
        """
        if node_id in self._states:
            # State exists - reuse it
            return self._states[node_id]
        else:
            # Create new state
            state = SimpleNamespace()
            self._states[node_id] = state
            return state
    
    def clear(self) -> None:
        """Clear all states (useful for testing or manual cleanup)."""
        self._states.clear()


# Global singleton instance
_global_state_registry = NodeStateRegistry()


def get_state_registry() -> NodeStateRegistry:
    """Get the global state registry instance."""
    return _global_state_registry
