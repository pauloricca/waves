from __future__ import annotations
from models import SequenceModel, OscillatorModel
from nodes.base_node import BaseNode


def instantiate_node(obj) -> BaseNode:
    from nodes.oscillator_node import OscillatorNode
    from nodes.sequence_node import SequenceNode
    if isinstance(obj, SequenceModel):
        return SequenceNode(obj)
    elif isinstance(obj, OscillatorModel):
        return OscillatorNode(obj)