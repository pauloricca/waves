from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from pydantic import ConfigDict, field_validator

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.midi_utils import MidiInputManager, MIDI_DEBUG
from nodes.wavable_value import WavableValue


class MidiCCModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    channel: int = 0  # MIDI channel to listen to (0-15)
    cc: WavableValue = 0  # CC number to listen to (0-127) - can be dynamic
    initial: float = 0.5  # Initial value in the range
    range: Tuple[WavableValue, WavableValue] = (0.0, 1.0)  # Output range [min, max] - both can be dynamic
    device: str | None = None  # Optional device key from config, None = use default
    duration: float = math.inf  # MIDI CC nodes run indefinitely
    
    @field_validator('range', mode='before')
    @classmethod
    def validate_range(cls, v):
        """Convert various input formats to a 2-element tuple of WavableValues"""
        if v is None:
            return (0.0, 1.0)
        
        # If it's already a tuple or list, validate it has exactly 2 elements
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"range must have exactly 2 values, got {len(v)}")
            return tuple(v)  # Convert to tuple for consistency
        
        raise ValueError(f"range must be a list or tuple with exactly 2 values")


class MidiCCNode(BaseNode):
    def __init__(self, model: MidiCCModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.channel = model.channel
        self.device_key = model.device  # Store the device key
        
        # Instantiate child nodes for dynamic parameters
        self.cc_node = self.instantiate_child_node(model.cc, "cc")
        # Extract min and max from range tuple
        self.min_node = self.instantiate_child_node(model.range[0], "range_0")
        self.max_node = self.instantiate_child_node(model.range[1], "range_1")
        
        # Persistent state for CC tracking (survives hot reload)
        if do_initialise_state:
            # Store normalized value (0-1) from the initial parameter
            self.state.current_normalized_value = model.initial
            
            # Track the last output value for smooth interpolation
            self.state.last_output_value = None  # Will be set on first render
            
            # Track last CC number to detect when it changes
            self.state.last_cc_number = None
        
        # Get the shared MIDI input manager
        self.midi_manager = MidiInputManager()
    
    def _process_midi_messages(self, cc_number: int):
        """Check for updated MIDI CC value
        
        Args:
            cc_number: The CC number to listen to (can change dynamically)
        """
        # Round to nearest integer for CC number
        cc_number = int(round(cc_number))
        cc_number = np.clip(cc_number, 0, 127)
        
        # If CC number changed, reset the state to avoid using stale values
        if self.state.last_cc_number is not None and cc_number != self.state.last_cc_number:
            # CC number changed - get the current value for the new CC
            cc_value = self.midi_manager.get_cc_value(self.channel, cc_number, self.device_key)
            if cc_value is not None:
                self.state.current_normalized_value = cc_value / 127.0
            if MIDI_DEBUG:
                print(f"CC number changed from {self.state.last_cc_number} to {cc_number}")
        
        self.state.last_cc_number = cc_number
        
        # Get the latest CC value for this channel and CC number
        cc_value = self.midi_manager.get_cc_value(self.channel, cc_number, self.device_key)
        
        if cc_value is not None:
            # Normalize CC value (0-127) to (0.0-1.0)
            new_normalized_value = cc_value / 127.0
            
            # Only update and log if the value changed
            if new_normalized_value != self.state.current_normalized_value:
                self.state.current_normalized_value = new_normalized_value
                if MIDI_DEBUG:
                    print(f"CC {cc_number} on channel {self.channel}: {cc_value} -> {self.state.current_normalized_value:.3f}")
    
    def _do_render(self, num_samples=None, context=None, **params):
        # MIDI CC node never finishes, so if num_samples is None, use a default buffer size
        if num_samples is None:
            from config import BUFFER_SIZE
            # For realtime mode, use buffer size
            # MIDI CC node continues indefinitely - never returns empty array
            num_samples = BUFFER_SIZE
            self._last_chunk_samples = num_samples
        
        # Render the dynamic parameters
        params_for_children = self.get_params_for_children(params)
        
        # Render CC number (can be dynamic)
        cc_wave = self.cc_node.render(num_samples, context, **params_for_children)
        # For CC number, we'll use the first value (or mean if varying)
        # since CC messages are discrete, not per-sample
        if isinstance(cc_wave, np.ndarray):
            cc_number = float(cc_wave[0]) if len(cc_wave) > 0 else 0.0
        else:
            cc_number = float(cc_wave)
        
        # Render min and max values (can be dynamic per-sample)
        min_wave = self.min_node.render(num_samples, context, **params_for_children)
        max_wave = self.max_node.render(num_samples, context, **params_for_children)
        
        # Ensure they're arrays of the right length
        if not isinstance(min_wave, np.ndarray):
            min_wave = np.full(num_samples, min_wave, dtype=np.float32)
        elif len(min_wave) == 1:
            min_wave = np.full(num_samples, min_wave[0], dtype=np.float32)
            
        if not isinstance(max_wave, np.ndarray):
            max_wave = np.full(num_samples, max_wave, dtype=np.float32)
        elif len(max_wave) == 1:
            max_wave = np.full(num_samples, max_wave[0], dtype=np.float32)
        
        # Process any pending MIDI messages using the current CC number
        self._process_midi_messages(cc_number)
        
        # Map the normalized value (0-1) to the min-max range for each sample
        # This allows the range to change over time while the MIDI value stays constant
        target_wave = min_wave + (self.state.current_normalized_value * (max_wave - min_wave))
        
        # Initialize last_output_value if this is the first render
        if self.state.last_output_value is None:
            self.state.last_output_value = target_wave[0] if num_samples > 0 else 0.0
        
        # When rendering a single sample, return the target value directly without smoothing
        if num_samples == 1:
            output_wave = target_wave.astype(np.float32)
        else:
            # Create smooth interpolation from last value to the first sample of target_wave
            # to avoid discontinuities between chunks
            interpolation_factor = np.linspace(0, 1, num_samples, dtype=np.float32)
            output_wave = self.state.last_output_value * (1 - interpolation_factor) + target_wave * interpolation_factor
        
        # Store the last value for the next chunk
        self.state.last_output_value = output_wave[-1] if num_samples > 0 else self.state.last_output_value
        
        return output_wave


MIDI_CC_DEFINITION = NodeDefinition("midi_cc", MidiCCNode, MidiCCModel)
