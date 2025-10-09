    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from utils import add_waves

class DelayModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    time: float = 0.1
    repeats: int = 3
    feedback: float = 0.3
    signal: BaseNodeModel = None

class DelayNode(BaseNode):
    def __init__(self, model: DelayModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)
        self.carry_over: np.ndarray = []

    def render(self, num_samples, **params):
        super().render(num_samples)
        signal_wave = self.signal_node.render(num_samples, **self.get_params_for_children(params))
        n_delay_time_samples = int(SAMPLE_RATE * self.model.time)
        delayed_wave = np.zeros(len(signal_wave) + n_delay_time_samples * self.model.repeats)

        # Add delays
        for i in range(self.model.repeats):
            delayed_wave[i * n_delay_time_samples : i * n_delay_time_samples + len(signal_wave)] += signal_wave * (self.model.feedback ** i)
        
        # Add carry over from previous render
        if len(self.carry_over) > 0:
            delayed_wave = add_waves(delayed_wave, self.carry_over[:len(delayed_wave)])

        # Return n_samples and save the rest as carry over for the next render
        part_to_return = delayed_wave[:num_samples]
        self.carry_over = delayed_wave[num_samples:]
        
        return part_to_return

DELAY_DEFINITION = NodeDefinition("delay", DelayNode, DelayModel)
