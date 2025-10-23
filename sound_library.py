from __future__ import annotations
from typing import Dict
from pydantic import RootModel, model_validator
import yaml

from nodes.node_utils.base_node import BaseNodeModel


sound_library: SoundLibraryModel = None


def parse_node(data, available_sound_names=None, raw_sound_data=None) -> BaseNodeModel:
    from nodes.node_utils.node_registry import NODE_REGISTRY
    from nodes.node_utils.node_string_parser import apply_params_to_model
    
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Each node must have exactly one node_type key: {data}")
    
    node_type, params = next(iter(data.items()))
    
    # Recursively parse params that look like nodes
    for k, v in params.items():
        if isinstance(v, dict) and len(v) == 1 and isinstance(next(iter(v.values())), dict):
            params[k] = parse_node(v, available_sound_names, raw_sound_data)
        elif isinstance(v, list):
            params[k] = [
                parse_node(i, available_sound_names, raw_sound_data) if isinstance(i, dict) and len(i) == 1 and isinstance(next(iter(i.values())), dict) else i
                for i in v
            ]
    
    node_definition = next((n for n in NODE_REGISTRY if n.name == node_type), None)

    if not node_definition:
        # Check if this is a sub-patch reference
        if available_sound_names and node_type in available_sound_names and raw_sound_data:
            # Get the raw sound data and parse it
            sound_data = raw_sound_data[node_type]
            # Parse the sound to get its base model
            base_model = parse_node(sound_data, available_sound_names, raw_sound_data)
            # Apply parameters to the model
            return apply_params_to_model(base_model, params)
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
        
        # Parse all sounds with knowledge of available sound names and raw data
        return {k: parse_node(v, available_sound_names, data) for k, v in data.items()}

    def __getitem__(self, key):
        return self.root[key]

    def keys(self):
        return self.root.keys()


def load_sound_library(file_path: str) -> SoundLibraryModel:
    global sound_library
    with open(file_path) as file:
        # Use the C-based LibYAML loader if available (much faster for hot reload)
        # Falls back to SafeLoader if LibYAML is not installed
        try:
            Loader = yaml.CSafeLoader
        except AttributeError:
            Loader = yaml.SafeLoader
        raw_data = yaml.load(file, Loader=Loader)
    
    # Extract and set user variables if present
    user_vars = raw_data.pop('vars', None)
    if user_vars:
        from expression_globals import set_user_variables
        set_user_variables(user_vars)
    
    try:
        sound_library = SoundLibraryModel.model_validate(raw_data)
        
        # Generate automatic hierarchical IDs for all nodes
        # IMPORTANT: We generate IDs for each root sound starting with that sound's name as the base
        # This ensures that dynamically instantiated sounds don't get colliding IDs
        from nodes.node_utils.auto_id_generator import AutoIDGenerator
        for sound_name, sound_model in sound_library.root.items():
            # Generate IDs starting from the sound name, not from "root"
            # This makes root-level techno-kick different from other root sounds
            AutoIDGenerator.generate_ids(sound_model, param_path=sound_name)
    
    except Exception as e:
        print(f"Error loading sound library: {e}")
    return sound_library

def get_sound_model(sound_name: str):
    if sound_library is None:
        raise ValueError("Sound library not loaded. Please load the sound library first.")
    
    if sound_name not in sound_library.keys():
        raise ValueError(f"Sound '{sound_name}' not found in the sound library.")
    
    return sound_library[sound_name]