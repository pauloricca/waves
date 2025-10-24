"""
Automatic ID generation for nodes based on parameter path.

IDs are generated based on the parameter path where a node is connected:
- Manual IDs (provided in YAML): "my_node"
- Auto IDs (no manual ID): "root.freq", "root.mix.signals.0", etc.

This creates stable, unique IDs that survive tree structure changes because:
1. IDs are tied to WHERE a node is used (parameter name), not WHERE it is positioned
2. Adding/removing sibling nodes doesn't affect other nodes' IDs
3. Swapping different node types preserves state only if types match (safer)

Example:
  test-ids-2:
    mix:                          <- ID: root
      signals:
        - osc:                    <- ID: root.mix.signals.0
            freq: 200
        - osc:                    <- ID: root.mix.signals.1
            freq: 400
  
  Reordering the oscs just swaps which gets .signals.0 vs .signals.1
  Their state moves with them because it's based on parameter path.
"""

from __future__ import annotations
from typing import Optional, Any


class AutoIDGenerator:
    """Generates parameter-path-based IDs for nodes that don't have explicit IDs."""
    
    @staticmethod
    def generate_ids(obj: Any, param_path: str = "root") -> None:
        """
        Recursively traverse an object and generate parameter-path-based IDs for all nodes
        that don't already have explicit IDs.
        
        Modifies the object in-place by setting __auto_id__ attribute.
        
        Args:
            obj: The object to process (typically a model, dict, or list)
            param_path: The parameter path to reach this object (e.g., "root.freq.osc")
        """
        # Skip None, primitives, and already processed objects
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return
        
        # Handle dicts - these are typically node models
        if isinstance(obj, dict):
            # Extract the node type and parameters
            for node_type, params in obj.items():
                if isinstance(params, dict):
                    AutoIDGenerator._process_node_dict(node_type, params, param_path)
                elif isinstance(params, list):
                    for i, item in enumerate(params):
                        AutoIDGenerator.generate_ids(item, f"{param_path}.{i}")
        
        # Handle lists - these are typically lists of nodes
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                AutoIDGenerator.generate_ids(item, f"{param_path}.{i}")
        
        # Handle Pydantic models
        elif hasattr(obj, '__pydantic_extra__'):
            # This is a Pydantic model with extra fields (like expression node)
            AutoIDGenerator._process_pydantic_model(obj, param_path)
        
        # Handle objects with model_dump or __dict__
        elif hasattr(obj, '__dict__'):
            AutoIDGenerator._process_pydantic_model(obj, param_path)
    
    @staticmethod
    def _process_node_dict(node_type: str, params: dict, param_path: str) -> None:
        """Process a node definition dict and generate IDs for its parameters."""
        # Recursively process nested nodes in parameters
        for param_name, param_value in params.items():
            if param_name == 'id':
                # Skip the id field itself
                continue
            
            # Build the param path for this parameter
            child_path = f"{param_path}.{param_name}"
            
            if isinstance(param_value, dict) and len(param_value) == 1:
                # This looks like a nested node (single key dict with node type)
                child_node_type, child_params = next(iter(param_value.items()))
                if isinstance(child_params, dict):
                    AutoIDGenerator._process_node_dict(child_node_type, child_params, child_path)
            
            elif isinstance(param_value, list):
                # List of nodes or values
                for i, item in enumerate(param_value):
                    item_path = f"{child_path}.{i}"
                    if isinstance(item, dict) and len(item) == 1:
                        child_node_type, child_params = next(iter(item.items()))
                        if isinstance(child_params, dict):
                            AutoIDGenerator._process_node_dict(child_node_type, child_params, item_path)
                    else:
                        AutoIDGenerator.generate_ids(item, item_path)
            else:
                AutoIDGenerator.generate_ids(param_value, child_path)
    
    @staticmethod
    def _process_pydantic_model(model: Any, param_path: str) -> None:
        """Process a Pydantic model and generate IDs for nested nodes."""
        # Check if model already has an explicit ID
        explicit_id = getattr(model, 'id', None)
        
        # Determine the current node's ID
        if explicit_id:
            # Use explicit ID, don't modify param_path for children
            # (explicit IDs are "roots" of their own subtree)
            current_path = explicit_id
        else:
            # Use the parameter path as the auto ID
            current_path = param_path
            # Store the auto-generated ID
            model.__auto_id__ = current_path
        
        # Process all fields of the model
        for field_name in dir(model):
            if field_name.startswith('_') or field_name == 'id':
                continue
            
            try:
                field_value = getattr(model, field_name)
                
                # Skip methods and callable attributes
                if callable(field_value):
                    continue
                
                # Build the path for this field
                field_path = f"{current_path}.{field_name}"
                
                # Recursively process nested structures
                if isinstance(field_value, dict):
                    AutoIDGenerator._process_dict_recursively(field_value, field_path)
                elif isinstance(field_value, list):
                    for i, item in enumerate(field_value):
                        AutoIDGenerator.generate_ids(item, f"{field_path}.{i}")
                elif hasattr(field_value, '__dict__') and not isinstance(field_value, (str, int, float, bool)):
                    # Recursively process nested models
                    AutoIDGenerator.generate_ids(field_value, field_path)
            
            except (AttributeError, TypeError):
                # Some attributes might not be accessible, skip them
                continue
    
    @staticmethod
    def _process_dict_recursively(d: dict, param_path: str) -> None:
        """Recursively process a dictionary structure."""
        for key, value in d.items():
            child_path = f"{param_path}.{key}"
            if isinstance(value, dict) and len(value) == 1:
                # Might be a node definition
                node_type, params = next(iter(value.items()))
                if isinstance(params, dict):
                    AutoIDGenerator._process_node_dict(node_type, params, child_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    AutoIDGenerator.generate_ids(item, f"{child_path}.{i}")
            elif hasattr(value, '__dict__'):
                AutoIDGenerator.generate_ids(value, child_path)
    
    @staticmethod
    def get_auto_id(model: Any) -> Optional[str]:
        """
        Get the auto-generated ID for a model if it exists.
        
        Returns:
            The auto-generated ID, or None if the model has no auto ID
        """
        return getattr(model, '__auto_id__', None)
    
    @staticmethod
    def get_effective_id(model: Any) -> Optional[str]:
        """
        Get the effective ID for a model (explicit or auto-generated).
        
        Checks explicit 'id' field first, falls back to auto-generated __auto_id__.
        
        Returns:
            The ID (explicit or auto-generated), or None if neither exists
        """
        explicit_id = getattr(model, 'id', None)
        if explicit_id:
            return explicit_id
        return AutoIDGenerator.get_auto_id(model)
