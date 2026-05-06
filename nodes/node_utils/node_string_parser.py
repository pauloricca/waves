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
from utils import ensure_array


# Matches "s=note" or "freq=note*2".
ASSIGNMENT_PARAM_PATTERN = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)=(.+)$')

# Matches "s0.5" or "freq440".
COMPACT_NUMERIC_PARAM_PATTERN = re.compile(r'^([a-zA-Z_]+)([-+]?[0-9]*\.?[0-9]+)$')

# Matches "-0.5", ".25", "440", or "1e-3".
NUMERIC_VALUE_PATTERN = re.compile(r'^[-+]?(?:[0-9]*\.[0-9]+|[0-9]+\.?)(?:[eE][-+]?[0-9]+)?$')


def _parse_param_value(value_string: str) -> float | bool | str:
    """Parse assignment values as numbers/booleans when possible, otherwise expressions."""
    if NUMERIC_VALUE_PATTERN.match(value_string):
        return float(value_string)
    lowered = value_string.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value_string


def parse_params_from_string(param_string: str) -> Dict[str, Any]:
    """
    Parse parameter key-value pairs from a string.
    
    Parameters are expected in one of these formats:
    - paramNAMEVALUE (e.g., "f440", "amp0.5", "t2")
    - param=EXPRESSION (e.g., "f=note", "amp=velocity*0.5")
    
    Args:
        param_string: String containing space-separated parameters
        
    Returns:
        Dictionary of parameter names to numeric values or expression strings
        
    Examples:
        >>> parse_params_from_string("f440 a0.5 t2 s=note")
        {'f': 440.0, 'a': 0.5, 't': 2.0, 's': 'note'}
    """
    parts = param_string.split()
    params = {}
    
    for param in parts:
        assignment_match = ASSIGNMENT_PARAM_PATTERN.match(param)
        if assignment_match:
            param_name = assignment_match.group(1)
            param_value = _parse_param_value(assignment_match.group(2))
            params[param_name] = param_value
            continue

        # Use regex to separate alphabetic prefix from numeric suffix
        # Matches patterns like: f440, amp0.5, t2, freq440.5
        match = COMPACT_NUMERIC_PARAM_PATTERN.match(param)
        if match:
            param_name = match.group(1)
            param_value = float(match.group(2))
            params[param_name] = param_value
    
    return params


def parse_node_string(node_string: str) -> Tuple[str, Dict[str, Any]]:
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


def split_special_params_from_string(value_string: str, special_param_names: set[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Split reserved string-notation params from a free-form value expression.

    This is useful for contexts like automation where the main string is an
    expression ("C4", "note * 2") rather than a node name, but we still want
    trailing metadata such as "prob=0.5".
    """
    tokens = value_string.split()
    if not tokens:
        return value_string, {}

    value_tokens = []
    special_params = {}

    for token in tokens:
        assignment_match = ASSIGNMENT_PARAM_PATTERN.match(token)
        if assignment_match and assignment_match.group(1) in special_param_names:
            special_params[assignment_match.group(1)] = _parse_param_value(assignment_match.group(2))
            continue

        compact_match = COMPACT_NUMERIC_PARAM_PATTERN.match(token)
        if compact_match and compact_match.group(1) in special_param_names:
            special_params[compact_match.group(1)] = float(compact_match.group(2))
            continue

        value_tokens.append(token)

    return " ".join(value_tokens), special_params


def resolve_render_params(
    render_args: Dict[str, Any],
    inherited_params: Dict[str, Any],
    time: float,
    num_samples: int,
    context=None,
) -> Dict[str, Any]:
    """
    Resolve expression-valued render args against the current render params.

    This lets sequencer string notation pass live control variables, e.g.
    "play s=note", while keeping numeric params as cheap scalar values.
    """
    if not render_args:
        return {}

    resolved_params = {}
    expression_context = None

    for param_name, param_value in render_args.items():
        if isinstance(param_value, str):
            if expression_context is None:
                from expression_globals import get_expression_context
                expression_context = get_expression_context(inherited_params, time, num_samples, context)
            expression_context.update(resolved_params)
            from expression_globals import evaluate_expression
            resolved_params[param_name] = ensure_array(
                evaluate_expression(param_value, expression_context, num_samples),
                num_samples,
            )
            expression_context[param_name] = resolved_params[param_name]
        else:
            resolved_params[param_name] = param_value
            if expression_context is not None:
                expression_context[param_name] = param_value

    return resolved_params


def apply_params_to_model(model: BaseNodeModel, params: Dict[str, Any]) -> BaseNodeModel:
    """
    Apply parameters to a node model by creating a deep copy and setting attributes.
    
    Special handling for sub-patches:
    - If params contains 'signal' and model has 'input_signal' (even if it also has 'signal'),
      PREFER mapping 'signal' to 'input_signal' for sub-patch compatibility.
    - This allows sub-patches to use 'input_signal' internally while accepting 'signal' externally.
    
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
        # Special case: map 'signal' to 'input_signal' if input_signal exists
        # This enables the pattern where sub-patches use input_signal internally
        # but accept signal externally (common pattern for reusable components)
        if param_name == 'signal' and hasattr(model_copy, 'input_signal'):
            setattr(model_copy, 'input_signal', param_value)
        elif hasattr(model_copy, param_name):
            setattr(model_copy, param_name, param_value)
    
    return model_copy


def instantiate_node_from_string(
        node_string: str,
        parent_id: str, 
        attribute_name: str,
        attribute_index: str,
        model: Optional[BaseNodeModel] = None,
    ) -> Tuple[BaseNode, Dict[str, Any]]:
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
