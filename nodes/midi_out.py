from __future__ import annotations
import numpy as np
from typing import Optional
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from constants import RenderArgs


# MIDI Output node: Sends MIDI notes or CC messages to external devices
#
# For notes:
#   midi_out:
#     device: "My MIDI Device"  # Optional, defaults to MIDI_OUTPUT_DEVICE from config
#     channel: 0  # MIDI channel (0-15)
#     note: 60  # MIDI note number (0-127) or WavableValue
#     velocity: 100  # Note velocity (0-127) or WavableValue (optional, defaults to 100)
#     duration: 1  # Note duration in seconds (optional, uses gate if not provided)
#     gate: 0.5  # Gate signal (>= 0.5 = note on, < 0.5 = note off)
#
# For CC:
#   midi_out:
#     device: "My MIDI Device"
#     channel: 0
#     cc: 74  # CC number (0-127)
#     value: 64  # CC value (0-127) or WavableValue
#
# The node passes through the input signal unchanged (or generates silence if no signal provided)
class MidiOutNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    device: Optional[str] = None  # MIDI device name (from config or explicit)
    channel: int = 0  # MIDI channel (0-15)
    signal: Optional[WavableValue] = None  # Optional signal to pass through
    
    # Note-specific parameters
    note: Optional[WavableValue] = None  # MIDI note number (0-127)
    velocity: Optional[WavableValue] = None  # Note velocity (0-127), default 100
    gate: Optional[WavableValue] = None  # Gate signal for note on/off
    
    # CC-specific parameters
    cc: Optional[int] = None  # CC number (0-127)
    value: Optional[WavableValue] = None  # CC value (0-127)
    
    is_pass_through: bool = True


class MidiOutNode(BaseNode):
    def __init__(self, model: MidiOutNodeModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.is_stereo = True  # Pass-through node
        
        self.device_name = model.device
        self.channel = model.channel
        
        # Infer type based on which parameters are provided
        if model.note is not None:
            self.msg_type = "note"
        elif model.cc is not None:
            self.msg_type = "cc"
        else:
            raise ValueError("MIDI out node requires either 'note' or 'cc' parameter")
        
        # Validate channel
        if not 0 <= self.channel <= 15:
            raise ValueError(f"MIDI channel must be 0-15, got {self.channel}")
        
        # Initialize signal node if provided
        if model.signal is not None:
            self.signal_node = self.instantiate_child_node(model.signal, "signal")
        else:
            self.signal_node = None
        
        # Note-specific initialization
        if self.msg_type == "note":
            self.note_node = self.instantiate_child_node(model.note, "note")
            
            # Velocity defaults to 100 if not provided
            if model.velocity is not None:
                self.velocity_node = self.instantiate_child_node(model.velocity, "velocity")
            else:
                self.velocity_node = None
            
            # Gate for note on/off (optional if duration is set)
            if model.gate is not None:
                self.gate_node = self.instantiate_child_node(model.gate, "gate")
            else:
                self.gate_node = None
            
            self.note_duration = model.duration
            
            # Validate that we have either duration or gate
            if self.note_duration is None and self.gate_node is None:
                raise ValueError("MIDI out type 'note' requires either 'duration' or 'gate' parameter")
        
        # CC-specific initialization
        if self.msg_type == "cc":
            self.cc_number = model.cc
            
            # Validate CC number
            if not 0 <= self.cc_number <= 127:
                raise ValueError(f"MIDI CC number must be 0-127, got {self.cc_number}")
            
            self.value_node = self.instantiate_child_node(model.value, "value")
        
        # Get MIDI output manager
        from nodes.node_utils.midi_utils import MidiOutputManager
        self.midi_manager = MidiOutputManager()
        
        # State for note tracking
        if do_initialise_state:
            self.state.note_active = False  # Is a note currently on?
            self.state.active_note = None  # Which note is active
            self.state.last_gate = 0.0  # Last gate value (for edge detection)
            self.state.note_start_time = None  # When did the note start (for duration-based notes)
            self.state.last_cc_value = None  # Last CC value sent (to avoid redundant messages)
    
    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Resolve chunk length
        num_samples_resolved = self.resolve_num_samples(num_samples)
        if num_samples_resolved is None:
            raise ValueError("MIDI out node requires explicit duration")
        
        # Get current time for duration-based notes
        current_time = params.get(RenderArgs.TIME, 0)
        
        if self.msg_type == "note":
            self._handle_note_output(num_samples_resolved, context, current_time, **params)
        elif self.msg_type == "cc":
            self._handle_cc_output(num_samples_resolved, context, **params)
        
        # Pass through signal or return silence
        if self.signal_node is not None:
            return self.signal_node.render(num_samples, context, num_channels, **params)
        else:
            # Return silence
            if num_channels == 2:
                return np.zeros((num_samples_resolved, 2), dtype=np.float32)
            else:
                return np.zeros(num_samples_resolved, dtype=np.float32)
    
    def _handle_note_output(self, num_samples: int, context, current_time: float, **params):
        """Handle MIDI note on/off messages"""
        from utils import samples_to_time
        
        # Get note number (use first sample value)
        note_wave = self.note_node.render(num_samples, context, **self.get_params_for_children(params))
        if len(note_wave) == 0:
            return
        note_number = int(np.clip(note_wave[0], 0, 127))
        
        # Get velocity
        if self.velocity_node is not None:
            velocity_wave = self.velocity_node.render(num_samples, context, **self.get_params_for_children(params))
            velocity = int(np.clip(velocity_wave[0], 0, 127)) if len(velocity_wave) > 0 else 100
        else:
            velocity = 100
        
        # Determine if note should be on or off
        should_be_on = False
        
        if self.note_duration is not None:
            # Duration-based: note on at start, off after duration
            if self.state.note_start_time is None:
                # First chunk - start the note
                self.state.note_start_time = current_time
                should_be_on = True
            else:
                # Check if duration has elapsed
                elapsed = current_time - self.state.note_start_time
                should_be_on = elapsed < self.note_duration
        else:
            # Gate-based: note on when gate >= 0.5
            gate_wave = self.gate_node.render(num_samples, context, **self.get_params_for_children(params))
            if len(gate_wave) > 0:
                current_gate = gate_wave[0]
                should_be_on = current_gate >= 0.5
        
        # Send note on/off based on state changes
        if should_be_on and not self.state.note_active:
            # Note on
            self.midi_manager.send_note(self.device_name, self.channel, note_number, velocity, True)
            self.state.note_active = True
            self.state.active_note = note_number
        elif not should_be_on and self.state.note_active:
            # Note off
            self.midi_manager.send_note(self.device_name, self.channel, self.state.active_note, 0, False)
            self.state.note_active = False
            self.state.active_note = None
        elif should_be_on and self.state.note_active and note_number != self.state.active_note:
            # Note changed while gate is still on - send off for old note, on for new note
            self.midi_manager.send_note(self.device_name, self.channel, self.state.active_note, 0, False)
            self.midi_manager.send_note(self.device_name, self.channel, note_number, velocity, True)
            self.state.active_note = note_number
    
    def _handle_cc_output(self, num_samples: int, context, **params):
        """Handle MIDI CC messages"""
        # Get CC value (use mean of the chunk)
        value_wave = self.value_node.render(num_samples, context, **self.get_params_for_children(params))
        if len(value_wave) == 0:
            return
        
        cc_value = int(np.clip(np.mean(value_wave), 0, 127))
        
        # Only send if value changed (avoid flooding)
        if cc_value != self.state.last_cc_value:
            self.midi_manager.send_cc(self.device_name, self.channel, self.cc_number, cc_value)
            self.state.last_cc_value = cc_value


MIDI_OUT_DEFINITION = NodeDefinition("midi_out", MidiOutNode, MidiOutNodeModel)
