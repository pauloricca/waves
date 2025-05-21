from __future__ import annotations
import numpy as np
from config import SAMPLE_RATE, ENVELOPE_TYPE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition


class EnvelopeModel(BaseNodeModel):
    attack: float = 0 # length of attack in seconds
    release: float = 0 # length of release in seconds
    signal: BaseNodeModel = None


class EnvelopeNode(BaseNode):
    def __init__(self, model: EnvelopeModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)

    def render(self, num_samples, **kwargs):
        super().render(num_samples)
        signal_wave = self.signal_node.render(num_samples, **self.get_kwargs_for_children(kwargs))
        attack_len = int(self.model.attack * SAMPLE_RATE)
        release_len = int(self.model.release * SAMPLE_RATE)

        if attack_len > 0 and len(signal_wave) > 0:
            # Ensure we don't try to apply a fade_in longer than the wave itself
            actual_attack_len = min(attack_len, len(signal_wave))
            if actual_attack_len > 0:  # Additional check to ensure we have samples to process
                if ENVELOPE_TYPE == "linear":
                    fade_in = np.linspace(0, 1, actual_attack_len)
                else:
                    fade_in = 1 - np.exp(-np.linspace(0, 5, actual_attack_len))
                signal_wave[:actual_attack_len] *= fade_in

        if release_len > 0 and len(signal_wave) > 0:
            # Ensure we don't try to apply a fade_out longer than the wave itself
            actual_release_len = min(release_len, len(signal_wave))
            if actual_release_len > 0:  # Additional check to ensure we have samples to process
                if ENVELOPE_TYPE == "linear":
                    fade_out = np.linspace(1, 0, actual_release_len)
                else:
                    fade_out = np.exp(-np.linspace(0, 5, actual_release_len))
                signal_wave[-actual_release_len:] *= fade_out

        return signal_wave


ENVELOPE_DEFINITION = NodeDefinition("envelope", EnvelopeNode, EnvelopeModel)
