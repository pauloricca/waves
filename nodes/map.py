
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from pydantic import ConfigDict, field_validator, model_validator
from config import *
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, wavable_value_node_factory

"""
Map Node

Maps values from one range to another, with optional clipping.

Parameters:
- signal: The input signal to map (required)
- from: Tuple of [min, max] defining the source range (optional, defaults to [0, 1])
- to: Tuple of [min, max] defining the target range (required)
- clip: Tuple of [min, max] to clip the output (optional)

All tuple parameters can be specified in YAML using either:
1. Bracket syntax: [value1, value2]
2. List syntax:
   - value1
   - value2

Each value can be a scalar or a WavableValue (dynamic value).

Examples:

# Basic mapping from default [0, 1] to [-1, 1]
map:
  signal:
    osc:
      type: sin
      freq: 440
  to: [-1, 1]

# Map from [-1, 1] to [0, 1] (normalize)
map:
  signal:
    osc:
      type: sin
      freq: 440
  from: [-1, 1]
  to: [0, 1]

# Map with clipping
map:
  signal:
    osc:
      type: sin
      freq: 440
  to: [-2, 2]
  clip: [-0.5, 0.5]

# Using list syntax with dynamic values
map:
  signal:
    osc:
      type: sin
      freq: 440
  to:
    - osc:
        type: sin
        freq: 0.5
        min: -1
        max: 0
    - osc:
        type: sin
        freq: 0.5
        min: 0
        max: 1
"""

class MapModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel = None
    from_: Optional[Tuple[WavableValue, WavableValue]] = None  # Using trailing underscore to avoid keyword conflict
    to: Tuple[WavableValue, WavableValue]
    clip: Optional[Tuple[WavableValue, WavableValue]] = None
    
    @field_validator('from_', 'to', 'clip', mode='before')
    @classmethod
    def validate_tuple(cls, v, info):
        """Convert various input formats to a 2-element tuple"""
        if v is None:
            return None
        
        # If it's already a tuple or list, validate it has exactly 2 elements
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"{info.field_name} must have exactly 2 values, got {len(v)}")
            return tuple(v)  # Convert to tuple for consistency
        
        raise ValueError(f"{info.field_name} must be a list or tuple with exactly 2 values")
    
    @model_validator(mode='before')
    @classmethod
    def handle_from_keyword(cls, data):
        """Handle 'from' keyword by converting it to 'from_'"""
        if isinstance(data, dict) and 'from' in data:
            data['from_'] = data.pop('from')
        return data

class MapNode(BaseNode):
    def __init__(self, model: MapModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)
        
        # Initialize from range (default to [0, 1])
        if self.model.from_ is None:
            self.from_min = wavable_value_node_factory(0)
            self.from_max = wavable_value_node_factory(1)
        else:
            self.from_min = wavable_value_node_factory(self.model.from_[0])
            self.from_max = wavable_value_node_factory(self.model.from_[1])
        
        # Initialize to range (required)
        self.to_min = wavable_value_node_factory(self.model.to[0])
        self.to_max = wavable_value_node_factory(self.model.to[1])
        
        # Initialize clip range (optional)
        if self.model.clip is not None:
            self.clip_min = self.model.clip[0]
            self.clip_max = self.model.clip[1]
        else:
            self.clip_min = None
            self.clip_max = None

    def _do_render(self, num_samples=None, context=None, **params):
        # If num_samples is None, get the full child signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # For map nodes, we need the full child signal to calculate proper mapping
                signal_wave = self.render_full_child_signal(self.signal_node, context, **self.get_params_for_children(params))
                if len(signal_wave) == 0:
                    return np.array([])
                
                num_samples = len(signal_wave)
                return self._apply_mapping(signal_wave, num_samples, context, params)
        
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # If signal is done, we're done
        if len(signal_wave) == 0:
            return np.array([], dtype=np.float32)
        
        return self._apply_mapping(signal_wave, num_samples, context, params)
    
    def _apply_mapping(self, signal_wave, num_samples, context, params):
        """Apply mapping to the signal wave"""
        child_params = self.get_params_for_children(params)
        
        from_min_wave = self.from_min.render(num_samples, context, **child_params)
        from_max_wave = self.from_max.render(num_samples, context, **child_params)
        to_min_wave = self.to_min.render(num_samples, context, **child_params)
        to_max_wave = self.to_max.render(num_samples, context, **child_params)
        
        # Ensure all waves have the same length as signal_wave
        for wave_name, wave in [('from_min', from_min_wave), ('from_max', from_max_wave), 
                                 ('to_min', to_min_wave), ('to_max', to_max_wave)]:
            if isinstance(wave, np.ndarray) and len(wave) != len(signal_wave):
                if wave_name == 'from_min':
                    from_min_wave = np.interp(np.linspace(0, 1, len(signal_wave)), 
                                              np.linspace(0, 1, len(wave)), wave)
                elif wave_name == 'from_max':
                    from_max_wave = np.interp(np.linspace(0, 1, len(signal_wave)), 
                                              np.linspace(0, 1, len(wave)), wave)
                elif wave_name == 'to_min':
                    to_min_wave = np.interp(np.linspace(0, 1, len(signal_wave)), 
                                            np.linspace(0, 1, len(wave)), wave)
                elif wave_name == 'to_max':
                    to_max_wave = np.interp(np.linspace(0, 1, len(signal_wave)), 
                                            np.linspace(0, 1, len(wave)), wave)

        # Map the wave from [from_min, from_max] to [to_min, to_max]
        # Vectorized normalization using NumPy's broadcasting
        from_range = from_max_wave - from_min_wave
        
        # Check if we have arrays or scalars
        is_array_from = isinstance(from_min_wave, np.ndarray) and isinstance(from_max_wave, np.ndarray)
        is_array_to = isinstance(to_min_wave, np.ndarray) and isinstance(to_max_wave, np.ndarray)
        
        if is_array_from or is_array_to:
            # Handle division by zero: where from_range is 0, use midpoint of to range
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = np.where(from_range != 0, 
                                     (signal_wave - from_min_wave) / from_range,
                                     0.5)  # Use 0.5 (middle) when range is 0
            
            # Apply to target range
            signal_wave = to_min_wave + (to_max_wave - to_min_wave) * normalized
        else:
            # Both are scalars
            if from_range != 0:
                normalized = (signal_wave - from_min_wave) / from_range
                signal_wave = to_min_wave + (to_max_wave - to_min_wave) * normalized
            else:
                # Handle case where from_min equals from_max to avoid division by zero
                signal_wave = np.full_like(signal_wave, (to_min_wave + to_max_wave) / 2)

        # Clip the wave to the specified range if provided
        if self.clip_min is not None or self.clip_max is not None:
            signal_wave = np.clip(signal_wave, self.clip_min, self.clip_max)

        return signal_wave


MAP_DEFINITION = NodeDefinition("map", MapNode, MapModel)
