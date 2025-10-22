"""
HotReloadManager: Captures and restores node state across YAML reloads.

This manager enables live coding by preserving the playback state of nodes
when the YAML definition changes. It works in two phases:

1. capture_state(node): Recursively traverses the node tree and captures
   all persistent state (state attributes) from nodes with ids (explicit or auto-generated).

2. restore_state(node, state_dict): Recursively traverses a newly instantiated
   node tree and restores captured state to matching nodes (matched by id).

This allows continuous playback while the sound definition evolves.

IDs can be:
- Explicit: Manually provided in YAML (e.g., "my_node")
- Auto-generated: Based on tree hierarchy (e.g., "0.1.2" or "parent.0.child")
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Set
from nodes.node_utils.base_node import BaseNode
from nodes.node_utils.auto_id_generator import AutoIDGenerator


class HotReloadManager:
    """Manages state capture and restoration for hot reloading nodes."""
    
    def capture_state(self, node: BaseNode) -> Dict[str, Any]:
        """
        Recursively capture state from all nodes in the tree.
        
        Returns a dict mapping node IDs to their state objects.
        Only nodes with an id field (explicit or auto-generated) are captured.
        
        Args:
            node: The root node to start capturing from.
        
        Returns:
            A dictionary mapping node IDs (str) to their state objects (SimpleNamespace).
        """
        state_dict = {}
        self._capture_state_recursive(node, state_dict)
        return state_dict
    
    def _capture_state_recursive(self, node: BaseNode, state_dict: Dict[str, Any]) -> None:
        """Helper to recursively capture state from node and all children."""
        # Get the effective ID (explicit or auto-generated)
        # node.node_id is already a string set in BaseNode.__init__
        node_id = node.node_id if hasattr(node, 'node_id') else None
        
        # Capture this node's state if it has an id
        if node_id:
            # Store a shallow copy of the state object (SimpleNamespace)
            # We store it directly so it can be reused in the new node
            if hasattr(node, 'state') and node.state is not None:
                state_dict[node_id] = node.state
        
        # Find all child nodes by checking for BaseNode attributes
        for attr_name in dir(node):
            # Skip special attributes and methods
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(node, attr_name)
                # Skip methods and other non-node attributes
                if callable(attr_value) or not isinstance(attr_value, BaseNode):
                    continue
                # Recursively capture state from child node
                self._capture_state_recursive(attr_value, state_dict)
            except (AttributeError, TypeError):
                # Some attributes might not be accessible or gettable, skip them
                continue
    
    def restore_state(self, node: BaseNode, state_dict: Dict[str, Any]) -> None:
        """
        Recursively restore captured state to nodes in a newly instantiated tree.
        
        Matches nodes by their node_id (explicit or auto-generated) and restores their state object.
        
        Args:
            node: The root node of the newly instantiated tree.
            state_dict: The state dictionary from capture_state().
        """
        self._restore_state_recursive(node, state_dict)
    
    def _restore_state_recursive(self, node: BaseNode, state_dict: Dict[str, Any]) -> None:
        """Helper to recursively restore state to node and all children."""
        # Get the effective ID (explicit or auto-generated)
        # node.node_id is already a string set in BaseNode.__init__
        node_id = node.node_id if hasattr(node, 'node_id') else None
        
        # Restore this node's state if it has an id
        if node_id and node_id in state_dict:
            node.state = state_dict[node_id]
        
        # Find all child nodes by checking for BaseNode attributes
        for attr_name in dir(node):
            # Skip special attributes and methods
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(node, attr_name)
                # Skip methods and other non-node attributes
                if callable(attr_value) or not isinstance(attr_value, BaseNode):
                    continue
                # Recursively restore state to child node
                self._restore_state_recursive(attr_value, state_dict)
            except (AttributeError, TypeError):
                # Some attributes might not be accessible or gettable, skip them
                continue
    
    def get_all_node_ids(self, node: BaseNode) -> Set[str]:
        """
        Get all node IDs present in the tree (for checking which nodes existed before reload).
        
        Includes both explicit and auto-generated IDs.
        
        Args:
            node: The root node to scan.
        
        Returns:
            A set of all node IDs found in the tree.
        """
        node_ids = set()
        self._collect_node_ids(node, node_ids)
        return node_ids
    
    def _collect_node_ids(self, node: BaseNode, node_ids: Set[str]) -> None:
        """Helper to recursively collect all node IDs from the tree."""
        # Get the effective ID (explicit or auto-generated)
        # node.node_id is already a string set in BaseNode.__init__
        node_id = node.node_id if hasattr(node, 'node_id') else None
        
        if node_id:
            node_ids.add(node_id)
        
        # Find all child nodes
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(node, attr_name)
                if callable(attr_value) or not isinstance(attr_value, BaseNode):
                    continue
                self._collect_node_ids(attr_value, node_ids)
            except (AttributeError, TypeError):
                continue
