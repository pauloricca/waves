from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition


# Select node: A conditional selector that chooses between different signal paths based on a test signal.
# The test signal is evaluated continuously, and when it changes mid-chunk, the output switches accordingly.
# 
# Boolean values (True/False) are converted to strings "true"/"false" (case-insensitive).
# If a path doesn't exist, zeros are rendered.
#
# Examples:
# 1. Boolean select:
#    select:
#      test: "a > b"
#      "true": osc: ...
#      "false": sequencer: ...
#
# 2. Multi-value select:
#    select:
#      test: snap: ...
#      "-1": ...
#      "0": ...
#      "1": ...
class SelectNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Allow arbitrary path keys
    test: BaseNodeModel  # The test signal that determines which path to select


class SelectNode(BaseNode):
    def __init__(self, model: SelectNodeModel, node_id: str, state=None, hot_reload=False):
        from nodes.wavable_value import WavableValueNode, WavableValueModel
        super().__init__(model, node_id, state, hot_reload)
        
        # Get this node's effective ID to build stable child IDs
        my_id = AutoIDGenerator.get_effective_id(model)
        if my_id is None:
            my_id = "select"  # Fallback if no ID
        
        # Instantiate the test node
        # Always pass True for hot_reload to allow child nodes to check their own state
        self.test_node = self.instantiate_child_node(model.test, "test")
        
        # Store all path nodes (arbitrary named arguments from extra fields)
        self.path_nodes: Dict[str, Optional[BaseNode]] = {}
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            for field_name, field_value in model.__pydantic_extra__.items():
                self.path_nodes[field_name] = self.instantiate_child_node(field_value, field_name, hot_reload=True)
    
    def _normalize_key(self, value) -> str:
        """Convert a value to a normalized string key for path lookup."""
        # Handle boolean values
        if isinstance(value, (bool, np.bool_)):
            return "true" if value else "false"
        
        # Handle numpy arrays (take first element if array)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return "0"
            value = value.flat[0]
        
        # For numeric values, try to convert to int if it's a whole number
        if isinstance(value, (int, float, np.number)):
            # Check if it's a whole number (like -1.0, 0.0, 1.0)
            if float(value) == int(value):
                return str(int(value))
            else:
                return str(float(value))
        
        # Convert to string
        key = str(value)
        
        # Normalize "true"/"false" to lowercase for case-insensitive matching
        if key.lower() in ("true", "false"):
            return key.lower()
        
        return key
    
    def _get_path_node(self, key: str) -> Optional[BaseNode]:
        """Get a path node, trying case-insensitive lookup for true/false."""
        # Direct lookup
        if key in self.path_nodes:
            return self.path_nodes[key]
        
        # Case-insensitive lookup for true/false
        if key.lower() in ("true", "false"):
            key_lower = key.lower()
            for path_key, node in self.path_nodes.items():
                if path_key.lower() == key_lower:
                    return node
        
        return None
    
    def _do_render(self, num_samples=None, context=None, **params):
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Select node requires explicit duration")
        
        # Render the test signal
        test_signal = self.test_node.render(num_samples, context, 
                                           **self.get_params_for_children(params))
        
        # Handle case where test signal ended early
        if len(test_signal) == 0:
            return np.array([], dtype=np.float32)
        
        actual_samples = len(test_signal)
        
        # Build output buffer
        output = np.zeros(actual_samples, dtype=np.float32)
        
        # Track current position
        pos = 0
        
        # Find segments where the test value is constant
        if actual_samples == 1:
            # Single sample - simple case
            segments = [(0, 1, test_signal[0])]
        else:
            segments = []
            current_key = self._normalize_key(test_signal[0])
            segment_start = 0
            
            for i in range(1, actual_samples):
                key = self._normalize_key(test_signal[i])
                if key != current_key:
                    # Value changed - close current segment
                    segments.append((segment_start, i, current_key))
                    segment_start = i
                    current_key = key
            
            # Close final segment
            segments.append((segment_start, actual_samples, current_key))
        
        # Render each segment with the appropriate path
        for start_idx, end_idx, key in segments:
            segment_length = end_idx - start_idx
            path_node = self._get_path_node(key)
            
            if path_node is not None:
                # Render the path for this segment
                segment_output = path_node.render(segment_length, context, 
                                                 **self.get_params_for_children(params))
                
                # Handle case where path returned fewer samples than requested
                if len(segment_output) < segment_length:
                    # Fill remaining with zeros
                    padded = np.zeros(segment_length, dtype=np.float32)
                    padded[:len(segment_output)] = segment_output
                    segment_output = padded
                elif len(segment_output) > segment_length:
                    # Truncate if too long
                    segment_output = segment_output[:segment_length]
                
                output[start_idx:end_idx] = segment_output
            # else: path doesn't exist, leave zeros
        
        return output


SELECT_DEFINITION = NodeDefinition(
    name="select",
    model=SelectNodeModel,
    node=SelectNode
)
