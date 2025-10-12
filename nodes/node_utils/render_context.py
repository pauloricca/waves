from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np
from config import MAX_RECURSION_DEPTH

if TYPE_CHECKING:
    from nodes.node_utils.base_node import BaseNode


class RenderContext:
    """
    Manages shared state during rendering, including node outputs and recursion tracking.
    
    This context object is passed through the entire render tree, enabling:
    - Nodes to cache their outputs for reference by other nodes
    - Detection and handling of feedback loops via recursion depth tracking
    - Shared state management across the render pass
    """
    
    def __init__(self, max_recursion: int = MAX_RECURSION_DEPTH):
        self.node_outputs: dict[str, np.ndarray] = {}  # id -> cached output
        self.node_instances: dict[str, BaseNode] = {}  # id -> node instance
        self.recursion_depth: dict[str, int] = {}  # id -> current recursion depth
        self.max_recursion: int = max_recursion  # Max recursion before returning zeros
        self.is_realtime: bool = True
        self.current_chunk: int = 0
    
    def store_output(self, node_id: str, wave: np.ndarray):
        """Store a node's output for reference by other nodes"""
        self.node_outputs[node_id] = wave
    
    def store_node(self, node_id: str, node: BaseNode):
        """Store a node instance for reference by other nodes"""
        self.node_instances[node_id] = node
    
    def get_output(self, node_id: str) -> Optional[np.ndarray]:
        """Retrieve a stored node output, or None if not available"""
        return self.node_outputs.get(node_id)
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Retrieve a stored node instance, or None if not available"""
        return self.node_instances.get(node_id)
    
    def increment_recursion(self, node_id: str):
        """Increment recursion depth for a node"""
        self.recursion_depth[node_id] = self.recursion_depth.get(node_id, 0) + 1
    
    def decrement_recursion(self, node_id: str):
        """Decrement recursion depth for a node"""
        if node_id in self.recursion_depth:
            self.recursion_depth[node_id] -= 1
            if self.recursion_depth[node_id] <= 0:
                del self.recursion_depth[node_id]
    
    def get_recursion_depth(self, node_id: str) -> int:
        """Get current recursion depth for a node"""
        return self.recursion_depth.get(node_id, 0)
    
    def clear_chunk(self):
        """Clear outputs for the next chunk (in realtime mode)"""
        self.node_outputs.clear()
        self.recursion_depth.clear()
        self.current_chunk += 1
