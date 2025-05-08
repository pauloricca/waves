from typing import Dict
from pydantic import RootModel, model_validator

from models.models import BaseNodeModel, DelayModel, OscillatorModel, SequencerModel


NODE_TYPE_TO_MODEL = {
    "osc": OscillatorModel,
    "delay": DelayModel,
    "sequencer": SequencerModel,
}


def parse_node(data) -> BaseNodeModel:
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

    model_cls = NODE_TYPE_TO_MODEL.get(node_type)
    if not model_cls:
        raise ValueError(f"Unknown node type: {node_type}")

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