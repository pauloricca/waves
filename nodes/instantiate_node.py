from __future__ import annotations
from models import SequenceModel, OscillatorModel
from nodes.base_node import BaseNode


def instantiate_node(obj) -> BaseNode:
    from nodes.oscillator_node import OscillatorNode
    if isinstance(obj, SequenceModel):
        return None
    elif isinstance(obj, OscillatorModel):
        return OscillatorNode(obj)