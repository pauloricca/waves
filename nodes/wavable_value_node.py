    
from __future__ import annotations
import numpy as np
from models import InterpolationTypes, WavableValue, WaveModel
from nodes.base_node import BaseNode
from nodes.instantiate_node import instantiate_node
from utils import interpolate_values


class WavableValueNode(BaseNode):
    def __init__(self, value: WavableValue, interpolation_type: InterpolationTypes = "LINEAR"):
        self.value = value
        self.interpolation_type = interpolation_type
        self.wave_node = instantiate_node(value) if isinstance(value, WaveModel) else None

    def render(self, num_samples):
        if self.wave_node:
            return self.wave_node.render(num_samples)
        elif isinstance(self.value, (float, int)):
            return np.array([self.value])
        if isinstance(self.value, list):
            return interpolate_values(self.value, num_samples, self.interpolation_type)