from __future__ import annotations
from typing import Dict
from pydantic import RootModel, model_validator, ValidationError
import yaml
import glob
import os

from nodes.node_utils.base_node import BaseNodeModel


# Dict mapping filename -> SoundLibraryModel
sound_libraries: Dict[str, SoundLibraryModel] = {}
# Flat dict of all sound names -> (filename, model) for quick lookup
sound_index: Dict[str, tuple[str, BaseNodeModel]] = {}
# Temporary storage of raw YAML data during multi-file loading (for cross-file reference resolution)
_raw_data_cache: Dict[str, dict] = {}


def format_validation_error(error: ValidationError, filename: str, raw_data: dict = None) -> str:
    """
    Format a Pydantic ValidationError into a clean, readable error message.
    
    Args:
        error: The ValidationError from Pydantic
        filename: The YAML file where the error occurred
        raw_data: The raw YAML data (to extract sound and node names)
    
    Returns:
        A formatted error string
    """
    lines = [f"\n{'='*60}"]
    lines.append(f"Validation Error in {filename}")
    lines.append('='*60)
    
    # Group errors by location path to avoid showing duplicate errors for Union types
    # For Union types, Pydantic creates one error per union member with type names in the path
    # We need to filter out type names like 'str', 'int', 'float', etc.
    errors_by_location = {}
    union_type_names = {'str', 'int', 'float', 'list', 'dict', 'bool', 'none', 'NoneType'}
    
    for err in error.errors():
        # Create a location key excluding union type names
        loc_key = tuple(loc for loc in err['loc'] if str(loc) not in union_type_names)
        
        # Only keep the first error for each unique location
        if loc_key not in errors_by_location:
            errors_by_location[loc_key] = err
    
    for i, (loc_key, err) in enumerate(errors_by_location.items()):
        if i > 0:
            lines.append('-'*60)
            
        location = err['loc']
        msg = err['msg']
        input_value = err.get('input', '')
        
        # Try to find which sound this error belongs to
        sound_name = None
        if raw_data and location:
            # Check if first location element is a sound name
            first = str(location[0])
            if first in raw_data:
                sound_name = first
            else:
                # Try to find by traversing the input value in raw_data
                for snd_name, snd_data in raw_data.items():
                    if _value_exists_in_dict(snd_data, input_value):
                        sound_name = snd_name
                        break
        
        # Extract path (excluding sound name, special keys, and union type names)
        path_parts = []
        for loc in location:
            loc_str = str(loc)
            if loc_str not in ['__root__', 'root'] and loc_str != sound_name and loc_str not in union_type_names:
                path_parts.append(loc_str)
        
        # Display the error
        if sound_name:
            lines.append(f"Sound: '{sound_name}'")
        
        if path_parts:
            lines.append(f"Location: {' → '.join(path_parts)}")
        
        # Simplify and improve error messages
        if 'Input should be a valid dictionary or instance of BaseNodeModel' in msg:
            clean_msg = "Expected a node definition (e.g., {'node_type': {...}})"
            if isinstance(input_value, str) and input_value.startswith('$'):
                clean_msg += f"\nNote: ${input_value[1:]} syntax only works in expressions, not as direct node parameters"
                clean_msg += "\nTo reference a node output, use: reference: {ref: " + input_value[1:] + "}"
        else:
            clean_msg = msg
        
        lines.append(f"Error: {clean_msg}")
        
        # Show the problematic value
        if input_value is not None and len(str(input_value)) < 200:
            lines.append(f"Got: {repr(input_value)}")
    
    lines.append('='*60)
    return '\n'.join(lines)


def _value_exists_in_dict(d, value, max_depth=10):
    """Helper to check if a value exists somewhere in a nested dict"""
    if max_depth <= 0:
        return False
    if not isinstance(d, dict):
        return d == value
    for v in d.values():
        if v == value:
            return True
        if isinstance(v, dict) and _value_exists_in_dict(v, value, max_depth - 1):
            return True
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and _value_exists_in_dict(item, value, max_depth - 1):
                    return True
    return False


def parse_node(data, available_sound_names=None, raw_sound_data=None, all_sound_names=None) -> BaseNodeModel:
    from nodes.node_utils.node_registry import NODE_REGISTRY
    from nodes.node_utils.node_string_parser import apply_params_to_model
    
    # Helper to check if a dict looks like a standard node definition
    def looks_like_standard_node(v):
        return isinstance(v, dict) and len(v) == 1 and isinstance(next(iter(v.values())), dict)
    
    # Helper to restructure dict for nodes with extra='allow' (select, context, expression, etc.)
    def restructure_if_needed(v):
        """
        If dict has multiple keys but one matches a node type, restructure it.
        Input:  {'select': {'test': 'note'}, '55': {...}, '57': {...}}
        Output: {'select': {'test': 'note', '55': {...}, '57': {...}}}
        """
        if not isinstance(v, dict) or len(v) <= 1:
            return v
        
        # Find node type keys
        node_type_keys = [k for k in v.keys() if any(n.name == k for n in NODE_REGISTRY)]
        sound_type_keys = [k for k in v.keys() if (available_sound_names and k in available_sound_names) or (all_sound_names and k in all_sound_names)]
        
        # If exactly one key is a node type, restructure
        if len(node_type_keys) == 1:
            node_type_key = node_type_keys[0]
            node_params = v[node_type_key].copy() if isinstance(v[node_type_key], dict) else {}
            # Add all other keys as extra params
            for k in v.keys():
                if k != node_type_key:
                    node_params[k] = v[k]
            return {node_type_key: node_params}
        elif len(sound_type_keys) == 1:
            sound_type_key = sound_type_keys[0]
            sound_params = v[sound_type_key].copy() if isinstance(v[sound_type_key], dict) else {}
            # Add all other keys as extra params
            for k in v.keys():
                if k != sound_type_key:
                    sound_params[k] = v[k]
            return {sound_type_key: sound_params}
        
        return v
    
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Each node must have exactly one node_type key: {data}")
    
    node_type, params = next(iter(data.items()))
    
    # Recursively parse params that look like nodes
    for k, v in params.items():
        # First restructure if needed
        v_restructured = restructure_if_needed(v)
        
        if looks_like_standard_node(v_restructured):
            params[k] = parse_node(v_restructured, available_sound_names, raw_sound_data, all_sound_names)
        elif isinstance(v, list):
            params[k] = [
                parse_node(restructure_if_needed(i), available_sound_names, raw_sound_data, all_sound_names) 
                if looks_like_standard_node(restructure_if_needed(i)) else i
                for i in v
            ]
    
    node_definition = next((n for n in NODE_REGISTRY if n.name == node_type), None)

    if not node_definition:
        # Check if this is a sub-patch reference - first check local file, then all files
        if available_sound_names and node_type in available_sound_names and raw_sound_data:
            # Get the raw sound data and parse it
            sound_data = raw_sound_data[node_type]
            # Parse the sound to get its base model
            base_model = parse_node(sound_data, available_sound_names, raw_sound_data, all_sound_names)
            # Apply parameters to the model
            return apply_params_to_model(base_model, params)
        elif all_sound_names and node_type in all_sound_names:
            # Reference to sound in another file - get it from the sound index
            _, sound_model = sound_index[node_type]
            
            # If the model is still None (not yet parsed), we need to parse it from raw data
            if sound_model is None:
                # Find the raw data for this sound in the cache
                for filename, raw_data in _raw_data_cache.items():
                    if node_type in raw_data:
                        # Parse it now
                        try:
                            sound_data = {node_type: raw_data[node_type]}
                            temp_library = SoundLibraryModel.model_validate(sound_data)
                            sound_model = temp_library.root[node_type]
                            # Update the index with the parsed model
                            sound_index[node_type] = (filename, sound_model)
                            break
                        except ValidationError as e:
                            print(format_validation_error(e, filename, sound_data))
                            raise
                
                if sound_model is None:
                    raise ValueError(f"Could not find or parse sound '{node_type}' from cross-file reference")
            
            # Apply parameters to the model
            return apply_params_to_model(sound_model, params)
        else:
            raise ValueError(f"Node type '{node_type}' not recognised and not found in sound library")

    model_cls = node_definition.model

    return model_cls(**params)


class SoundLibraryModel(RootModel[Dict[str, BaseNodeModel]]):
    @model_validator(mode="before")
    @classmethod
    def parse_graph(cls, data):
        # Get list of all root-level sound names for sub-patch recognition
        available_sound_names = set(data.keys())
        # Get all sound names across all files for cross-file references
        all_sound_names = set(sound_index.keys())
        
        # Parse all sounds with knowledge of available sound names and raw data
        return {k: parse_node(v, available_sound_names, data, all_sound_names) for k, v in data.items()}

    def __getitem__(self, key):
        return self.root[key]

    def keys(self):
        return self.root.keys()


def load_yaml_file(file_path: str) -> SoundLibraryModel:
    """Load and parse a single YAML file."""
    with open(file_path) as file:
        # Use the C-based LibYAML loader if available (much faster for hot reload)
        # Falls back to SafeLoader if LibYAML is not installed
        try:
            Loader = yaml.CSafeLoader
        except AttributeError:
            Loader = yaml.SafeLoader
        raw_data = yaml.load(file, Loader=Loader)
    
    if raw_data is None:
        raw_data = {}
    
    # Extract and set user variables if present (only from main waves.yaml)
    if os.path.basename(file_path) == 'waves.yaml':
        user_vars = raw_data.pop('vars', None)
        if user_vars:
            from expression_globals import set_user_variables
            set_user_variables(user_vars)
    
    try:
        library = SoundLibraryModel.model_validate(raw_data)
        
        return library
    
    except ValidationError as e:
        print(format_validation_error(e, os.path.basename(file_path), raw_data))
        raise
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def load_all_sound_libraries(directory: str = ".") -> Dict[str, SoundLibraryModel]:
    """Load all .yaml files in the directory into the sound library."""
    global sound_libraries, sound_index, _raw_data_cache
    
    # Find all .yaml files in the directory
    yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
    
    if not yaml_files:
        raise ValueError(f"No .yaml files found in {directory}")
    
    # Clear existing libraries and index
    sound_libraries.clear()
    sound_index.clear()
    _raw_data_cache.clear()
    
    # Use C-based LibYAML loader if available (much faster)
    try:
        Loader = yaml.CSafeLoader
    except AttributeError:
        Loader = yaml.SafeLoader
    
    # FIRST PASS: Load all raw YAML data and extract user variables
    raw_data_by_file = {}
    for yaml_file in yaml_files:
        filename = os.path.basename(yaml_file)
        try:
            with open(yaml_file) as file:
                raw_data = yaml.load(file, Loader=Loader)
            
            if raw_data is None:
                raw_data = {}
            
            # Extract and set user variables if present (only from main waves.yaml)
            if filename == 'waves.yaml':
                user_vars = raw_data.pop('vars', None)
                if user_vars:
                    from expression_globals import set_user_variables
                    set_user_variables(user_vars)
            
            raw_data_by_file[filename] = raw_data
            
            # Store in cache for cross-file reference resolution
            _raw_data_cache[filename] = raw_data
            
            # Pre-populate sound_index with all sound names (models will be None for now)
            for sound_name in raw_data.keys():
                if sound_name in sound_index:
                    print(f"Warning: Sound '{sound_name}' defined in multiple files. Using definition from {filename}")
                sound_index[sound_name] = (filename, None)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            raise
    
    # SECOND PASS: Parse all YAML data now that sound_index has all sound names
    for filename, raw_data in raw_data_by_file.items():
        try:
            library = SoundLibraryModel.model_validate(raw_data)
            sound_libraries[filename] = library
            
            # Update the sound index with actual models IMMEDIATELY so they're available for subsequent files
            for sound_name, sound_model in library.root.items():
                sound_index[sound_name] = (filename, sound_model)
            
            print(f"Loaded {len(library.root)} sound(s) from {filename}")
        except ValidationError as e:
            print(format_validation_error(e, filename, raw_data))
            return False
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return False
    
    # Clear the cache after loading is complete
    _raw_data_cache.clear()
    
    return sound_libraries


def reload_sound_library(filename: str, directory: str = ".") -> bool:
    """Reload a single YAML file and update the sound library."""
    global sound_libraries, sound_index, _raw_data_cache
    
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        print(f"File {filename} not found")
        return False
    
    # Preserve old state in case reload fails
    old_library = sound_libraries.get(filename)
    old_sound_index_entries = {}
    if old_library:
        for sound_name in old_library.root.keys():
            if sound_name in sound_index and sound_index[sound_name][0] == filename:
                old_sound_index_entries[sound_name] = sound_index[sound_name]
    
    try:
        # Use C-based LibYAML loader if available
        try:
            Loader = yaml.CSafeLoader
        except AttributeError:
            Loader = yaml.SafeLoader
        
        # Load raw YAML data
        with open(file_path) as file:
            raw_data = yaml.load(file, Loader=Loader)
        
        if raw_data is None:
            raw_data = {}
        
        # Extract and set user variables if present (only from main waves.yaml)
        if filename == 'waves.yaml':
            user_vars = raw_data.pop('vars', None)
            if user_vars:
                from expression_globals import set_user_variables
                set_user_variables(user_vars)
        
        # Pre-populate sound_index with sound names from this file (models will be None for now)
        temp_sound_index = {}
        for sound_name in raw_data.keys():
            temp_sound_index[sound_name] = (filename, None)
        
        # Store in cache for cross-file reference resolution
        _raw_data_cache[filename] = raw_data
        
        # Now parse the file with full knowledge of all sound names
        try:
            library = SoundLibraryModel.model_validate(raw_data)
            
            # Only update global state if validation succeeded
            # Remove old sounds from this file from the index
            for sound_name in old_sound_index_entries.keys():
                del sound_index[sound_name]
            
            # Update with new library and index
            sound_libraries[filename] = library
            
            # Update the sound index with actual models
            for sound_name, sound_model in library.root.items():
                sound_index[sound_name] = (filename, sound_model)
        except ValidationError as e:
            print(format_validation_error(e, filename, raw_data))
            raise
        
        # Clear the cache after loading is complete
        _raw_data_cache.clear()
        
        return True
    
    except Exception as e:
        print(f"Error reloading {filename}: {e}")
        # Clear the cache on error
        _raw_data_cache.clear()
        # Old state is already preserved (we didn't modify sound_index or sound_libraries)
        return False


def get_sound_model(sound_name: str) -> BaseNodeModel:
    """Get a sound model by name from any loaded library."""
    if not sound_index:
        raise ValueError("Sound library not loaded. Please load the sound library first.")
    
    if sound_name not in sound_index:
        raise ValueError(f"Sound '{sound_name}' not found in any loaded YAML file.")
    
    _, sound_model = sound_index[sound_name]
    return sound_model


def get_sound_filename(sound_name: str) -> str:
    """Get the filename that contains a sound definition."""
    if sound_name not in sound_index:
        raise ValueError(f"Sound '{sound_name}' not found in any loaded YAML file.")
    
    filename, _ = sound_index[sound_name]
    return filename