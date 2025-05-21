    
from __future__ import annotations
from typing import Optional
import numpy as np
from pydantic import ConfigDict
from config import *
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.oscillator import OSCILLATOR_RENDER_ARGS
from nodes.wavable_value import WavableValue, wavable_value_node_factory

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
        super().__init__(model)
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
            self.min = wavable_value_node_factory(-self.model.max)
            self.max = wavable_value_node_factory(self.model.max)
        else:
            self.min = wavable_value_node_factory(self.model.min or -1)
            self.max = wavable_value_node_factory(self.model.max or 1)

    def render(self, num_samples, **kwargs):
        super().render(num_samples)
        signal_wave = self.signal_node.render(num_samples, **self.get_kwargs_for_children(kwargs))
        min_wave = self.min.render(num_samples, **self.get_kwargs_for_children(kwargs, OSCILLATOR_RENDER_ARGS))
        max_wave = self.max.render(num_samples, **self.get_kwargs_for_children(kwargs, OSCILLATOR_RENDER_ARGS))
        
        # Ensure min and max have the same length as wave
        if isinstance(min_wave, np.ndarray) and len(min_wave) != len(signal_wave):
            min_wave = np.interp(np.linspace(0, 1, len(signal_wave)), np.linspace(0, 1, len(min_wave)), min_wave)
        if isinstance(max_wave, np.ndarray) and len(max_wave) != len(signal_wave):
            max_wave = np.interp(np.linspace(0, 1, len(signal_wave)), np.linspace(0, 1, len(max_wave)), max_wave)

        if self.source_min is None:
            peak = np.max(np.abs(signal_wave))
            self.source_min = -peak
            self.source_max = peak

        # Normalize the wave from [source_min, source_max] the range [min, max]
        if(isinstance(min_wave, np.ndarray) and isinstance(max_wave, np.ndarray)):
            # Min and max are arrays, apply vectorized normalization
            if self.source_max != self.source_min:
                # Vectorized normalization using NumPy's broadcasting
                normalized_scale = (signal_wave - self.source_min) / (self.source_max - self.source_min)
                signal_wave = min_wave + (max_wave - min_wave) * normalized_scale
            else:
                # Handle case where source_min equals source_max to avoid division by zero
                signal_wave = (min_wave + max_wave) / 2
        else:
            # Min and max are scalar values, apply normalization uniformly
            if self.source_max != self.source_min:
                signal_wave = min_wave + (max_wave - min_wave) * \
                        (signal_wave - self.source_min) / (self.source_max - self.source_min)
            else:
                # Handle case where source_min equals source_max to avoid division by zero
                signal_wave = np.full_like(signal_wave, (min_wave + max_wave) / 2)

        # Clip the wave to the specified range
        if self.model.clip_min is not None:
            signal_wave = np.clip(signal_wave, self.model.clip_min, None)
        if self.model.clip_max is not None:
            signal_wave = np.clip(signal_wave, None, self.model.clip_max)

        return signal_wave


NORMALISE_DEFINITION = NodeDefinition("normalise", NormaliseNode, NormaliseModel)
