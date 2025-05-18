from __future__ import annotations
import numpy as np
from config import SAMPLE_RATE, ENVELOPE_TYPE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition


class EnvelopeModel(BaseNodeModel):
    attack: float = 0
    release: float = 0
    signal: BaseNodeModel = None


class EnvelopeNode(BaseNode):
    def __init__(self, envelope_model: EnvelopeModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.attack = envelope_model.attack
        self.release = envelope_model.release
        self.signal_node = instantiate_node(envelope_model.signal)

    def render(self, num_samples, **kwargs):
        wave = self.signal_node.render(num_samples, **kwargs)
        attack_len = int(self.attack * SAMPLE_RATE)
        release_len = int(self.release * SAMPLE_RATE)

        if attack_len > 0:
            if ENVELOPE_TYPE == "linear":
                fade_in = np.linspace(0, 1, attack_len)
            else:
                fade_in = 1 - np.exp(-np.linspace(0, 5, attack_len))
            wave[:attack_len] *= fade_in

        if release_len > 0:
            if ENVELOPE_TYPE == "linear":
                fade_out = np.linspace(1, 0, release_len)
            else:
                fade_out = np.exp(-np.linspace(0, 5, release_len))
            wave[-release_len:] *= fade_out

        return wave


ENVELOPE_DEFINITION = NodeDefinition("envelope", EnvelopeNode, EnvelopeModel)
