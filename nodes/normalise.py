    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition

class NormaliseModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel = None

class NormaliseNode(BaseNode):
    def __init__(self, normalise_model: NormaliseModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.normalise_model = normalise_model
        self.signal_node = instantiate_node(normalise_model.signal)

    def render(self, num_samples, **kwargs):
        wave = self.signal_node.render(num_samples, **kwargs)
        # Normalize the wave to the range [-1, 1]
        peak = np.max(np.abs(wave))
        if peak > 1:
            wave /= peak
        return wave


NORMALISE_DEFINITION = NodeDefinition("normalise", NormaliseNode, NormaliseModel)
