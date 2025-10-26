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
    
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Each node must have exactly one node_type key: {data}")
    
    node_type, params = next(iter(data.items()))
    
    # Recursively parse params that look like nodes
    for k, v in params.items():
        if isinstance(v, dict) and len(v) == 1 and isinstance(next(iter(v.values())), dict):
            params[k] = parse_node(v, available_sound_names, raw_sound_data, all_sound_names)
        elif isinstance(v, list):
            params[k] = [
                parse_node(i, available_sound_names, raw_sound_data, all_sound_names) if isinstance(i, dict) and len(i) == 1 and isinstance(next(iter(i.values())), dict) else i
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
    
    # Load each YAML file
    for yaml_file in yaml_files:
        filename = os.path.basename(yaml_file)
        try:
            library = load_yaml_file(yaml_file)
            sound_libraries[filename] = library
            
            # Update the sound index
            for sound_name, sound_model in library.root.items():
                if sound_name in sound_index:
                    print(f"Warning: Sound '{sound_name}' defined in multiple files. Using definition from {filename}")
                sound_index[sound_name] = (filename, sound_model)
            
            print(f"Loaded {len(library.root)} sound(s) from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
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
        
        # Load the updated file
        library = load_yaml_file(file_path)
        sound_libraries[filename] = library
        
        # Update the sound index
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