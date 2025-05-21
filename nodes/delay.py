    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition

class DelayModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    time: float = 0.1
    repeats: int = 3
    feedback: float = 0.3
    do_trim: bool = False
    signal: BaseNodeModel = None

class DelayNode(BaseNode):
    def __init__(self, model: DelayModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)

    def render(self, num_samples, **kwargs):
        super().render(num_samples)
        signal_wave = self.signal_node.render(num_samples, **self.get_kwargs_for_children(kwargs))
        n_delay_time_samples = int(SAMPLE_RATE * self.model.time)
        delayed_wave = np.zeros(len(signal_wave) + n_delay_time_samples * self.model.repeats)

        for i in range(self.model.repeats):
            delayed_wave[i * n_delay_time_samples : i * n_delay_time_samples + len(signal_wave)] += signal_wave * (self.model.feedback ** i)

        return delayed_wave[: len(signal_wave)] if self.model.do_trim else delayed_wave # Trim to original length

DELAY_DEFINITION = NodeDefinition("delay", DelayNode, DelayModel)
