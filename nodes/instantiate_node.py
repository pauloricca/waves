from __future__ import annotations
from models import *
from models.models import DelayModel, OscillatorModel, SequencerModel
from nodes.base_node import BaseNode

def instantiate_node(obj) -> BaseNode:
    from nodes.oscillator_node import OscillatorNode
    from nodes.sequence_node import SequencerNode
    from nodes.delay_node import DelayNode

    MODEL_TO_NODE = {
        OscillatorModel: OscillatorNode,
        DelayModel: DelayNode,
        SequencerModel: SequencerNode,
    }

    for model_cls, node_cls in MODEL_TO_NODE.items():
        if isinstance(obj, model_cls):
            return node_cls(obj)

    raise ValueError(f"Unknown model type: {type(obj)}")
