from __future__ import annotations
import numpy as np
from scipy.signal import butter, lfilter
from pydantic import ConfigDict
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition

class FilterModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    cutoff: float = 1000.0  # Cutoff frequency in Hz
    type: str = "lowpass"  # "lowpass" or "highpass"
    order: int = 2  # Filter order
    signal: BaseNodeModel = None

class FilterNode(BaseNode):
    def __init__(self, filter_model: FilterModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.filter_model = filter_model
        self.signal_node = instantiate_node(filter_model.signal)

    def render(self, num_samples, **kwargs):
        wave = self.signal_node.render(num_samples, **kwargs)
        nyquist = SAMPLE_RATE / 2.0
        normalized_cutoff = self.filter_model.cutoff / nyquist

        # Design filter
        b, a = butter(
            self.filter_model.order,
            normalized_cutoff,
            btype=self.filter_model.type,
            analog=False,
        )

        # Apply filter
        filtered_wave = lfilter(b, a, wave)
        return filtered_wave

FILTER_DEFINITION = NodeDefinition("filter", FilterNode, FilterModel)
