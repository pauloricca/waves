    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition

class SmoothModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    step: float = 0.01
    signal: BaseNodeModel = None

class SmoothNode(BaseNode):
    def __init__(self, smooth_model: SmoothModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.smooth_model = smooth_model
        self.signal_node = instantiate_node(smooth_model.signal)

    def render(self, num_samples, **kwargs):
        wave = self.signal_node.render(num_samples, **kwargs)
        smoothed_wave = np.copy(wave)
        for i in range(1, len(smoothed_wave)):
            diff = smoothed_wave[i] - smoothed_wave[i - 1]
            if abs(diff) > self.smooth_model.step:
                smoothed_wave[i] = smoothed_wave[i - 1] + np.sign(diff) * self.smooth_model.step
        return smoothed_wave


SMOOTH_DEFINITION = NodeDefinition("smooth", SmoothNode, SmoothModel)
