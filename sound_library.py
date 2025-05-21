from __future__ import annotations
from typing import Dict
from pydantic import RootModel, model_validator
import yaml

from nodes.node_utils.base_node import BaseNodeModel


sound_library: SoundLibraryModel = None


def parse_node(data) -> BaseNodeModel:
    from nodes.node_utils.node_registry import NODE_REGISTRY
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Each node must have exactly one node_type key: {data}")
    
    node_type, params = next(iter(data.items()))
    
    # Recursively parse params that look like nodes
    for k, v in params.items():
        if isinstance(v, dict) and len(v) == 1 and isinstance(next(iter(v.values())), dict):
            params[k] = parse_node(v)
        elif isinstance(v, list):
            params[k] = [
                parse_node(i) if isinstance(i, dict) and len(i) == 1 and isinstance(next(iter(i.values())), dict) else i
                for i in v
            ]
    
    node_definition = next((n for n in NODE_REGISTRY if n.name == node_type), None)

    if not node_definition:
        raise ValueError(f"Node type '{node_type}' not recognised")

    model_cls = node_definition.model

    return model_cls(**params)


class SoundLibraryModel(RootModel[Dict[str, BaseNodeModel]]):
    @model_validator(mode="before")
    @classmethod
    def parse_graph(cls, data):
        return {k: parse_node(v) for k, v in data.items()}

    def __getitem__(self, key):
        return self.root[key]

    def keys(self):
        return self.root.keys()


def load_sound_library(file_path: str) -> SoundLibraryModel:
    global sound_library
    with open(file_path) as file:
        raw_data = yaml.safe_load(file)
    try:
        sound_library = SoundLibraryModel.model_validate(raw_data)
    except Exception as e:
        print(f"Error loading sound library: {e}")
    return sound_library

def get_sound_model(sound_name: str):
    if sound_library is None:
        raise ValueError("Sound library not loaded. Please load the sound library first.")
    
    if sound_name not in sound_library.keys():
        raise ValueError(f"Sound '{sound_name}' not found in the sound library.")
    
    return sound_library[sound_name]