    
from __future__ import annotations
from enum import Enum
from typing import List, Union
import numpy as np
from models.models import BaseNodeModel
from nodes.base import BaseNode
from utils import interpolate_values


class InterpolationTypes(str, Enum):
    LINEAR = "LINEAR"
    SMOOTH = "SMOOTH"
    STEP = "STEP"


WavableValue = Union[float, List[Union[float, List[float]]], BaseNodeModel]


class WavableValueNode(BaseNode):
    def __init__(self, value: WavableValue, interpolation_type: InterpolationTypes = "LINEAR"):
        from nodes.node_utils.instantiate_node import instantiate_node
        from nodes.oscillator import OscillatorModel
        self.value = value
        self.interpolation_type = interpolation_type
        self.wave_node = instantiate_node(value) if isinstance(value, OscillatorModel) else None

    def render(self, num_samples):
        if self.wave_node:
            return self.wave_node.render(num_samples)
        elif isinstance(self.value, (float, int)):
            return np.array([self.value])
        if isinstance(self.value, list):
            return interpolate_values(self.value, num_samples, self.interpolation_type)