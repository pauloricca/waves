"""
Global registry for node states.

This module provides a centralized state management system for nodes:
- States are stored globally, indexed by node ID
- During hot reload, states are preserved for nodes whose IDs still exist
- States are never automatically cleaned up (to avoid interfering with active sounds)
"""

from types import SimpleNamespace
from typing import Dict


class NodeStateRegistry:
    """Global registry for node states indexed by node ID."""
    
    def __init__(self):
        self._states: Dict[str, SimpleNamespace] = {}
    
    def get_state(self, node_id: str) -> SimpleNamespace | None:
        if node_id in self._states:
            return self._states[node_id]
        else:
            return None
    
    def create_state(self, node_id: str) -> None:
        state = SimpleNamespace()
        self._states[node_id] = state
        return state
    
    def clear(self) -> None:
        """Clear all states (useful for testing or manual cleanup)."""
        self._states.clear()


# Global singleton instance
_global_state_registry = NodeStateRegistry()


def get_state_registry() -> NodeStateRegistry:
    return _global_state_registry
