"""
Utility for parsing node instantiation strings with parameters.

This module provides functionality to parse strings like "node_name paramVALUE paramVALUE"
and instantiate nodes with those parameters applied.

Examples:
    "my_sound f440 a0.5" - instantiate my_sound with freq=440 and amp=0.5
    "kick t2" - instantiate kick with t=2
"""
from __future__ import annotations
import re
from typing import Dict, Any, Tuple, Optional
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.instantiate_node import instantiate_node


def parse_params_from_string(param_string: str) -> Dict[str, float]:
    """
    Parse parameter key-value pairs from a string.
    
    Parameters are expected in the format: paramNAMEVALUE (e.g., "f440", "amp0.5", "t2")
    
    Args:
        param_string: String containing space-separated parameters
        
    Returns:
        Dictionary of parameter names to values
        
    Examples:
        >>> parse_params_from_string("f440 a0.5 t2")
        {'f': 440.0, 'a': 0.5, 't': 2.0}
    """
    parts = param_string.split()
    params = {}
    
    for param in parts:
        # Use regex to separate alphabetic prefix from numeric suffix
        # Matches patterns like: f440, amp0.5, t2, freq440.5
        match = re.match(r'^([a-zA-Z_]+)([-+]?[0-9]*\.?[0-9]+)$', param)
        if match:
            param_name = match.group(1)
            param_value = float(match.group(2))
            params[param_name] = param_value
    
    return params


def parse_node_string(node_string: str) -> Tuple[str, Dict[str, float]]:
    """
    Parse a node string into node name and parameters.
    
    Args:
        node_string: String in format "node_name param1VALUE param2VALUE"
        
    Returns:
        Tuple of (node_name, parameters_dict)
        
    Examples:
        >>> parse_node_string("kick f440 a0.5")
        ('kick', {'f': 440.0, 'a': 0.5})
    """
    parts = node_string.split()
    if not parts:
        raise ValueError("Empty node string")
    
    node_name = parts[0]
    param_string = ' '.join(parts[1:])
    params = parse_params_from_string(param_string)
    
    return node_name, params


def apply_params_to_model(model: BaseNodeModel, params: Dict[str, Any]) -> BaseNodeModel:
    """
    Apply parameters to a node model by creating a deep copy and setting attributes.
    
    Args:
        model: The node model to apply parameters to
        params: Dictionary of parameter names to values
        
    Returns:
        A new model instance with parameters applied
    """
    if model is None:
        return None
    
    # Create a deep copy to avoid modifying the original
    model_copy = model.model_copy(deep=True)
    
    # Apply each parameter that exists in the model
    for param_name, param_value in params.items():
        if hasattr(model_copy, param_name):
            setattr(model_copy, param_name, param_value)
    
    return model_copy


def instantiate_node_from_string(
        node_string: str,
        parent_id: str, 
        attribute_name: str,
        attribute_index: str,
        model: Optional[BaseNodeModel] = None,
    ) -> Tuple[BaseNode, Dict[str, float]]:
    """
    Instantiate a node from a string specification with parameters.
    
    This function parses a string like "node_name f440 a0.5" and creates a node instance
    with those parameters applied. If a model is provided, parameters are applied to it.
    If no model is provided, the node_name is looked up in the sound library.
    
    Args:
        node_string: String in format "node_name param1VALUE param2VALUE"
        model: Optional pre-loaded node model. If None, will be looked up by node_name.
        
    Returns:
        Tuple of (instantiated_node, parameters_dict)
        
    Examples:
        >>> node, params = instantiate_node_from_string("kick f440 a0.5")
        >>> node, params = instantiate_node_from_string("kick f440", kick_model)
    """
    from sound_library import get_sound_model
    
    node_name, params = parse_node_string(node_string)
    
    # Get model if not provided
    if model is None:
        model = get_sound_model(node_name)
    
    # Apply parameters to model
    model_with_params = apply_params_to_model(model, params)
    
    # Instantiate node
    node = instantiate_node(model_with_params, parent_id, attribute_name, attribute_index)
    
    return node, params
