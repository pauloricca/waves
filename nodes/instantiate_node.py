from __future__ import annotations
from models import SequenceModel, WaveModel
from nodes.base_node import BaseNode


def instantiate_node(obj) -> BaseNode:
    from nodes.wave_node import WaveNode
    if isinstance(obj, SequenceModel):
        return None
    elif isinstance(obj, WaveModel):
        return WaveNode(obj)