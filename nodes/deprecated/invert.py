    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition

class InvertModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel = None

class InvertNode(BaseNode):
    def __init__(self, model: InvertModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.signal_node = instantiate_node(model.signal)

    def render(self, num_samples, **params):
        super().render(num_samples)
        signal_wave = self.signal_node.render(num_samples, **self.get_params_for_children(params))
        return signal_wave[::-1]


INVERT_DEFINITION = NodeDefinition("invert", InvertNode, InvertModel)
