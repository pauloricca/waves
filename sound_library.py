from __future__ import annotations
from typing import Dict
from pydantic import RootModel, model_validator
import yaml
import glob
import os

from nodes.node_utils.base_node import BaseNodeModel


# Dict mapping filename -> SoundLibraryModel
sound_libraries: Dict[str, SoundLibraryModel] = {}
# Flat dict of all sound names -> (filename, model) for quick lookup
sound_index: Dict[str, tuple[str, BaseNodeModel]] = {}


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
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def load_all_sound_libraries(directory: str = ".") -> Dict[str, SoundLibraryModel]:
    """Load all .yaml files in the directory into the sound library."""
    global sound_libraries, sound_index
    
    # Find all .yaml files in the directory
    yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
    
    if not yaml_files:
        raise ValueError(f"No .yaml files found in {directory}")
    
    # Clear existing libraries and index
    sound_libraries.clear()
    sound_index.clear()
    
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
            
            # Update the sound index with actual models
            for sound_name, sound_model in library.root.items():
                sound_index[sound_name] = (filename, sound_model)
            
            print(f"Loaded {len(library.root)} sound(s) from {filename}")
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            raise
    
    return sound_libraries


def reload_sound_library(filename: str, directory: str = ".") -> bool:
    """Reload a single YAML file and update the sound library."""
    global sound_libraries, sound_index
    
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        print(f"File {filename} not found")
        return False
    
    try:
        # Remove old sounds from this file from the index
        if filename in sound_libraries:
            old_library = sound_libraries[filename]
            for sound_name in old_library.root.keys():
                if sound_name in sound_index and sound_index[sound_name][0] == filename:
                    del sound_index[sound_name]
        
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
        for sound_name in raw_data.keys():
            sound_index[sound_name] = (filename, None)
        
        # Now parse the file with full knowledge of all sound names
        library = SoundLibraryModel.model_validate(raw_data)
        sound_libraries[filename] = library
        
        # Update the sound index with actual models
        for sound_name, sound_model in library.root.items():
            sound_index[sound_name] = (filename, sound_model)
        
        return True
    
    except Exception as e:
        print(f"Error reloading {filename}: {e}")
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