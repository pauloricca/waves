from __future__ import annotations
import math
import numpy as np
from pydantic import ConfigDict

from constants import RenderArgs
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.midi_utils import MidiInputManager, MIDI_DEBUG


class MidiCCModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    channel: int = 0  # MIDI channel to listen to (0-15)
    cc: int  # CC number to listen to (0-127)
    initial: float = 0.5  # Initial value in the min-max range
    min: float = 0.0  # Minimum output value
    max: float = 1.0  # Maximum output value
    duration: float = math.inf  # MIDI CC nodes run indefinitely


class MidiCCNode(BaseNode):
    def __init__(self, model: MidiCCModel):
        super().__init__(model)
        self.channel = model.channel
        self.cc_number = model.cc
        self.min_value = model.min
        self.max_value = model.max
        
        # Convert initial value from min-max range to normalized (0-1)
        value_range = self.max_value - self.min_value
        if value_range != 0:
            self.current_normalized_value = (model.initial - self.min_value) / value_range
        else:
            self.current_normalized_value = 0.5
        
        # Get the shared MIDI input manager
        self.midi_manager = MidiInputManager()
    
    def _process_midi_messages(self):
        """Process all pending MIDI messages and update current value"""
        messages = self.midi_manager.get_messages()
        
        for message in messages:
            # Only process CC messages for our channel and CC number
            if (hasattr(message, 'channel') and message.channel == self.channel and
                message.type == 'control_change' and message.control == self.cc_number):
                # Normalize CC value (0-127) to (0.0-1.0)
                self.current_normalized_value = message.value / 127.0
                if MIDI_DEBUG:
                    mapped_value = self.min_value + (self.current_normalized_value * (self.max_value - self.min_value))
                    print(f"CC {self.cc_number} on channel {self.channel}: {message.value} -> {self.current_normalized_value:.3f} -> {mapped_value:.3f}")
    
    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
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
        mapped_value = self.min_value + (self.current_normalized_value * (self.max_value - self.min_value))
        
        # Return an array filled with the mapped value
        output_wave = np.full(num_samples, mapped_value, dtype=np.float32)
        
        return output_wave


MIDI_CC_DEFINITION = NodeDefinition("midi_cc", MidiCCNode, MidiCCModel)
