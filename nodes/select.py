from __future__ import annotations
import numpy as np
import math
from typing import Dict, Optional, Any
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from utils import empty_mono


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
#
# Path keys are also evaluated against the global expression context when possible,
# so constants like C0 can be used as aliases for their numeric value.
class SelectNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Allow arbitrary path keys
    test: Any  # The test signal that determines which path to select (actual type from parse_node)


class SelectNode(BaseNode):
    def __init__(self, model: SelectNodeModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        
        # Instantiate the test node
        # Always pass True for hot_reload to allow child nodes to check their own state
        self.test_node = self.instantiate_child_node(model.test, "test")
        
        # Store all path nodes (arbitrary named arguments from extra fields)
        self.path_nodes: Dict[str, Optional[BaseNode]] = {}
        self.evaluated_path_keys: Dict[str, Any] = {}
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            for field_name, field_value in model.__pydantic_extra__.items():
                self.path_nodes[field_name] = self.instantiate_child_node(field_value, field_name)
                evaluated_key = self._evaluate_path_key(field_name)
                if evaluated_key is not None:
                    self.evaluated_path_keys[field_name] = evaluated_key

    def _evaluate_path_key(self, key: str):
        """Evaluate a select path key as a scalar expression if possible."""
        try:
            from expression_globals import evaluate_expression, get_expression_context

            eval_context = get_expression_context({}, 0, 1)
            result = evaluate_expression(key, eval_context, num_samples=None)
            if isinstance(result, np.ndarray):
                if result.size != 1:
                    return None
                result = result.flat[0]

            if isinstance(result, (bool, np.bool_, int, float, np.number, str)):
                return result
        except Exception:
            return None

        return None
    
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

    def _values_match(self, expected, actual) -> bool:
        """Compare evaluated path keys against test values with numeric tolerance."""
        if isinstance(actual, np.ndarray):
            if actual.size == 0:
                return False
            actual = actual.flat[0]

        if isinstance(expected, (bool, np.bool_)):
            expected_value = 1.0 if bool(expected) else 0.0
            if isinstance(actual, (int, float, np.number)):
                return math.isclose(float(actual), expected_value, rel_tol=0.0, abs_tol=1e-9)
            return bool(expected) == bool(actual)

        if isinstance(actual, (bool, np.bool_)):
            actual_value = 1.0 if bool(actual) else 0.0
            if isinstance(expected, (int, float, np.number)):
                return math.isclose(float(expected), actual_value, rel_tol=0.0, abs_tol=1e-9)
            return bool(expected) == bool(actual)

        if isinstance(expected, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
            return math.isclose(
                float(expected),
                float(actual),
                rel_tol=1e-6,
                abs_tol=1e-4,
            )

        return self._normalize_key(expected) == self._normalize_key(actual)
    
    def _get_path_node(self, key: str, value=None) -> Optional[BaseNode]:
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

        if value is not None:
            for path_key, evaluated_key in self.evaluated_path_keys.items():
                if self._values_match(evaluated_key, value):
                    return self.path_nodes[path_key]
        
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
            return empty_mono()
        
        actual_samples = len(test_signal)
        
        # Build output buffer
        output = np.zeros(actual_samples, dtype=np.float32)
        
        # Track current position
        pos = 0
        
        # Find segments where the test value is constant
        if actual_samples == 1:
            # Single sample - simple case
            segments = [(0, 1, self._normalize_key(test_signal[0]), test_signal[0])]
        else:
            segments = []
            current_key = self._normalize_key(test_signal[0])
            current_value = test_signal[0]
            segment_start = 0
            
            for i in range(1, actual_samples):
                key = self._normalize_key(test_signal[i])
                if key != current_key:
                    # Value changed - close current segment
                    segments.append((segment_start, i, current_key, current_value))
                    segment_start = i
                    current_key = key
                    current_value = test_signal[i]
            
            # Close final segment
            segments.append((segment_start, actual_samples, current_key, current_value))
        
        # Render each segment with the appropriate path
        for start_idx, end_idx, key, value in segments:
            segment_length = end_idx - start_idx
            path_node = self._get_path_node(key, value)
            
            if path_node is not None:
                # Render the path for this segment
                segment_output = path_node.render(segment_length, context, 
                                                 **self.get_params_for_children(params))
                
                # Handle case where path returned empty (finished)
                if len(segment_output) == 0:
                    # Path has finished - return empty to signal completion
                    return empty_mono()
                
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
