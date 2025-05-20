    
from __future__ import annotations
from typing import Optional
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, WavableValueNode

class NormaliseModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel = None
    min: Optional[WavableValue] = None
    max: Optional[WavableValue] = None
    source_min: Optional[float] = None
    source_max: Optional[float] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

class NormaliseNode(BaseNode):
    def __init__(self, model: NormaliseModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.model = model
        self.signal_node = instantiate_node(model.signal)
        
        if self.model.source_min is None and self.model.source_max is not None:
            self.source_min = -self.model.source_max
            self.source_max = self.model.source_max
        else:
            self.source_min = self.model.source_min
            self.source_max = self.model.source_max

        # If no min and max are provided, set them to the default range [-1, 1]
        if self.model.min is None and self.model.max is not None:
            self.min = WavableValueNode(-self.model.max)
            self.max = WavableValueNode(self.model.max)
        else:
            self.min = WavableValueNode(self.model.min or -1)
            self.max = WavableValueNode(self.model.max or 1)

    def render(self, num_samples, **kwargs):
        signal_wave = self.signal_node.render(num_samples, **kwargs)
        self.min = self.min.render(num_samples)
        self.max = self.max.render(num_samples)
        
        # Ensure min and max have the same length as wave
        if isinstance(self.min, np.ndarray) and len(self.min) != len(signal_wave):
            self.min = np.interp(np.linspace(0, 1, len(signal_wave)), np.linspace(0, 1, len(self.min)), self.min)
        if isinstance(self.max, np.ndarray) and len(self.max) != len(signal_wave):
            self.max = np.interp(np.linspace(0, 1, len(signal_wave)), np.linspace(0, 1, len(self.max)), self.max)

        if self.source_min is None:
            peak = np.max(np.abs(signal_wave))
            self.source_min = -peak
            self.source_max = peak

        # Normalize the wave from [source_min, source_max] the range [min, max]
        if(isinstance(self.min, np.ndarray) and isinstance(self.max, np.ndarray)):
            # Min and max are arrays, apply vectorized normalization
            if self.source_max != self.source_min:
                # Vectorized normalization using NumPy's broadcasting
                normalized_scale = (signal_wave - self.source_min) / (self.source_max - self.source_min)
                signal_wave = self.min + (self.max - self.min) * normalized_scale
            else:
                # Handle case where source_min equals source_max to avoid division by zero
                signal_wave = (self.min + self.max) / 2
        else:
            # Min and max are scalar values, apply normalization uniformly
            if self.source_max != self.source_min:
                signal_wave = self.min + (self.max - self.min) * \
                        (signal_wave - self.source_min) / (self.source_max - self.source_min)
            else:
                # Handle case where source_min equals source_max to avoid division by zero
                signal_wave = np.full_like(signal_wave, (self.min + self.max) / 2)

        # Clip the wave to the specified range
        if self.model.clip_min is not None:
            signal_wave = np.clip(signal_wave, self.model.clip_min, None)
        if self.model.clip_max is not None:
            signal_wave = np.clip(signal_wave, None, self.model.clip_max)

        return signal_wave


NORMALISE_DEFINITION = NodeDefinition("normalise", NormaliseNode, NormaliseModel)
