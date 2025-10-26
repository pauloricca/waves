from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from pydantic import ConfigDict, field_validator

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.midi_utils import MidiInputManager, MIDI_DEBUG


class MidiCCModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    channel: int = 0  # MIDI channel to listen to (0-15)
    cc: int  # CC number to listen to (0-127)
    initial: float = 0.5  # Initial value in the range
    range: Tuple[float, float] = (0.0, 1.0)  # Output range [min, max]
    duration: float = math.inf  # MIDI CC nodes run indefinitely
    
    @field_validator('range', mode='before')
    @classmethod
    def validate_range(cls, v):
        """Convert various input formats to a 2-element tuple"""
        if v is None:
            return (0.0, 1.0)
        
        # If it's already a tuple or list, validate it has exactly 2 elements
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"range must have exactly 2 values, got {len(v)}")
            return tuple(v)  # Convert to tuple for consistency
        
        raise ValueError(f"range must be a list or tuple with exactly 2 values")


class MidiCCNode(BaseNode):
    def __init__(self, model: MIDICCNodeModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.channel = model.channel
        self.cc_number = model.cc
        self.min_value, self.max_value = model.range
        
        # Persistent state for CC tracking (survives hot reload)
        if do_initialise_state:
            # Convert initial value from min-max range to normalized (0-1)
            value_range = self.max_value - self.min_value
            if value_range != 0:
                self.state.current_normalized_value = (model.initial - self.min_value) / value_range
            else:
                self.state.current_normalized_value = 0.5
            
            # Track the last output value for smooth interpolation
            self.state.last_output_value = self.min_value + (self.state.current_normalized_value * (self.max_value - self.min_value))
        
        # Get the shared MIDI input manager
        self.midi_manager = MidiInputManager()
    
    def _process_midi_messages(self):
        """Check for updated MIDI CC value"""
        # Get the latest CC value for this channel and CC number
        cc_value = self.midi_manager.get_cc_value(self.channel, self.cc_number)
        
        if cc_value is not None:
            # Normalize CC value (0-127) to (0.0-1.0)
            new_normalized_value = cc_value / 127.0
            
            # Only update and log if the value changed
            if new_normalized_value != self.state.current_normalized_value:
                self.state.current_normalized_value = new_normalized_value
                if MIDI_DEBUG:
                    mapped_value = self.min_value + (self.state.current_normalized_value * (self.max_value - self.min_value))
                    print(f"CC {self.cc_number} on channel {self.channel}: {cc_value} -> {self.state.current_normalized_value:.3f} -> {mapped_value:.3f}")
    
    def _do_render(self, num_samples=None, context=None, **params):
        # MIDI CC node never finishes, so if num_samples is None, use a default buffer size
        if num_samples is None:
            from config import BUFFER_SIZE
            # For realtime mode, use buffer size
            # MIDI CC node continues indefinitely - never returns empty array
            num_samples = BUFFER_SIZE
            self._last_chunk_samples = num_samples
        
        # Process any pending MIDI messages
        self._process_midi_messages()
        
        # Map the normalized value (0-1) to the min-max range
        target_value = self.min_value + (self.state.current_normalized_value * (self.max_value - self.min_value))
        
        # When rendering a single sample, return the target value directly without smoothing
        if num_samples == 1:
            output_wave = np.array([target_value], dtype=np.float32)
        else:
            # Create smooth interpolation from last value to target value
            output_wave = np.linspace(self.state.last_output_value, target_value, num_samples, dtype=np.float32)
        
        # Store the last value for the next chunk
        self.state.last_output_value = target_value
        
        return output_wave


MIDI_CC_DEFINITION = NodeDefinition("midi_cc", MidiCCNode, MidiCCModel)
