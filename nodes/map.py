from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from pydantic import ConfigDict, field_validator, model_validator
from config import *
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.range_mapper import RangeMapper
from nodes.wavable_value import WavableValue

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from nodes.node_utils.range_mapper import RangeMapper

"""
Map node: maps values from one range to another

This node takes an input signal and maps it from an input range to an output range.
Optionally applies clipping to keep values within the target range.

Parameters:
- signal: Input signal
- from_range: Input range [min, max] (default: [0, 1])
- range: Output range [min, max] (default: [0, 1])
- clip: Whether to clip output to range (default: True)

Example:
map:
  signal:
    osc:
      type: sin
      freq: 1
  from_range: [-1, 1]    # sine wave naturally in [-1, 1]
  range: [200, 800]      # map to frequency range
  clip: true
"""

class MapModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: WavableValue = None
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
    def __init__(self, model: MapModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.is_stereo = True  # Map node is a pass-through, supports stereo
        self.model = model
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Create range mapper with optional from range (defaults to [0, 1])
        from_range = model.from_ if model.from_ is not None else (0.0, 1.0)
        self.range_mapper = RangeMapper.from_model_range(
            self, model.to, "to",
            from_range=from_range, from_attribute_name="from"
        )
        
        # Initialize clip range (optional)
        if self.model.clip is not None:
            self.clip_min = self.model.clip[0]
            self.clip_max = self.model.clip[1]
        else:
            self.clip_min = None
            self.clip_max = None

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # If num_samples is None, get the full child signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # For map nodes, we need the full child signal to calculate proper mapping
                signal_wave = self.render_full_child_signal(self.signal_node, context, num_channels, **self.get_params_for_children(params))
                if len(signal_wave) == 0:
                    return np.array([])
                
                num_samples = len(signal_wave)
                return self._apply_mapping(signal_wave, num_samples, context, params)
        
        signal_wave = self.signal_node.render(num_samples, context, num_channels, **self.get_params_for_children(params))
        
        # If signal is done, we're done
        if len(signal_wave) == 0:
            return np.array([], dtype=np.float32)
        
        return self._apply_mapping(signal_wave, num_samples, context, params)
    
    def _apply_mapping(self, signal_wave, num_samples, context, params):
        """Apply mapping to the signal wave"""
        # Apply range mapping using RangeMapper
        mapped_wave = self.range_mapper.map(signal_wave, num_samples, context, **params)

        # Clip the wave to the specified range if provided
        if self.clip_min is not None or self.clip_max is not None:
            mapped_wave = np.clip(mapped_wave, self.clip_min, self.clip_max)

        return mapped_wave


MAP_DEFINITION = NodeDefinition("map", MapNode, MapModel)
