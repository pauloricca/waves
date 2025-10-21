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
        self.node_outputs: dict[int, np.ndarray] = {}  # node instance id (id()) -> cached output
        self.node_outputs_by_id: dict[str, np.ndarray] = {}  # node string id -> cached output for references
        self.node_instances: dict[str, BaseNode] = {}  # id -> node instance
        self.recursion_depth: dict[int, int] = {}  # node instance id (id()) -> current recursion depth
        self.max_recursion: int = max_recursion  # Max recursion before returning zeros
        self.is_realtime: bool = True
        self.current_chunk: int = 0
    
    def store_output(self, node_instance_id: int, node_string_id: str, wave: np.ndarray):
        """Store a node's output both by instance (for caching) and by ID (for references)"""
        self.node_outputs[node_instance_id] = wave
        if node_string_id:
            self.node_outputs_by_id[node_string_id] = wave
    
    def store_node(self, node_id: str, node: BaseNode):
        """Store a node instance for reference by other nodes"""
        self.node_instances[node_id] = node
    
    def get_output(self, node_instance_id: int) -> Optional[np.ndarray]:
        """Retrieve a stored node output by instance, or None if not available"""
        return self.node_outputs.get(node_instance_id)
    
    def get_output_by_id(self, node_id: str) -> Optional[np.ndarray]:
        """Retrieve a stored node output by string ID (for reference nodes), or None if not available"""
        return self.node_outputs_by_id.get(node_id)
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Retrieve a stored node instance, or None if not available"""
        return self.node_instances.get(node_id)
    
    def increment_recursion(self, node_instance_id: int):
        """Increment recursion depth for a node instance"""
        self.recursion_depth[node_instance_id] = self.recursion_depth.get(node_instance_id, 0) + 1
    
    def decrement_recursion(self, node_instance_id: int):
        """Decrement recursion depth for a node instance"""
        if node_instance_id in self.recursion_depth:
            self.recursion_depth[node_instance_id] -= 1
            if self.recursion_depth[node_instance_id] <= 0:
                del self.recursion_depth[node_instance_id]
    
    def get_recursion_depth(self, node_instance_id: int) -> int:
        """Get current recursion depth for a node instance"""
        return self.recursion_depth.get(node_instance_id, 0)
    
    def clear_chunk(self):
        """Clear outputs for the next chunk (in realtime mode)"""
        self.node_outputs.clear()
        self.node_outputs_by_id.clear()
        self.recursion_depth.clear()
        self.current_chunk += 1
