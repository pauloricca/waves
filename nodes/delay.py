    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition

class DelayModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    time: float = 0.1
    repeats: int = 3
    feedback: float = 0.3
    do_trim: bool = False
    signal: BaseNodeModel = None

class DelayNode(BaseNode):
    def __init__(self, delay_model: DelayModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.delay_model = delay_model
        self.signal_node = instantiate_node(delay_model.signal)

    def render(self, num_samples, **kwargs):
        wave = self.signal_node.render(num_samples, **kwargs)
        n_delay_time_samples = int(SAMPLE_RATE * self.delay_model.time)
        delayed_wave = np.zeros(len(wave) + n_delay_time_samples * self.delay_model.repeats)

        for i in range(self.delay_model.repeats):
            delayed_wave[i * n_delay_time_samples : i * n_delay_time_samples + len(wave)] += wave * (self.delay_model.feedback ** i)

        return delayed_wave[: len(wave)] if self.delay_model.do_trim else delayed_wave # Trim to original length

DELAY_DEFINITION = NodeDefinition("delay", DelayNode, DelayModel)
