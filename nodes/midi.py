from __future__ import annotations
from typing import Optional
import math
import numpy as np
from pydantic import ConfigDict

from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.midi_utils import (
    MidiInputManager, 
    midi_note_to_frequency, 
    midi_velocity_to_amplitude,
    MIDI_DEBUG
)


class MidiModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    channel: int = 0  # MIDI channel to listen to (0-15)
    signal: BaseNodeModel  # The sound/signal to play when a note is triggered
    voices: int = 16  # Maximum number of simultaneous voices (polyphony limit)


class MidiNode(BaseNode):
    def __init__(self, model: MidiModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.channel = model.channel
        self.signal_model = model.signal
        self.max_voices = model.voices
        
        # Persistent state for active notes (survives hot reload)
        if do_initialise_state:
            self.state.active_notes = {}
            self.state.note_id_counter = 0
            self.state.note_number_to_ids = {}  # {note_number: [id1, id2, ...]}
        
        # Get the shared MIDI input manager
        self.midi_manager = MidiInputManager()
    
    def _process_midi_messages(self, **params):
        """Process all pending MIDI messages from the queue"""
        messages = self.midi_manager.get_messages()
        
        for message in messages:
            # Only process messages for our channel
            if hasattr(message, 'channel') and message.channel == self.channel:
                if message.type == 'note_on' and message.velocity > 0:
                    self._handle_note_on(message.note, message.velocity, **params)
                elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                    self._handle_note_off(message.note)
    
    def _handle_note_on(self, note_number, velocity, **params):
        """Handle MIDI note on event"""
        from nodes.node_utils.instantiate_node import instantiate_node
        
        # Convert MIDI note number to frequency
        frequency = midi_note_to_frequency(note_number)
        amplitude = midi_velocity_to_amplitude(velocity)
        
        # Voice stealing: if we've reached max voices, remove the oldest note
        if len(self.state.active_notes) >= self.max_voices:
            # Find the oldest note (lowest note_id)
            oldest_note_id = min(self.state.active_notes.keys())
            oldest_note_data = self.state.active_notes[oldest_note_id]
            oldest_note_number = oldest_note_data['note_number']
            
            # Remove it
            del self.state.active_notes[oldest_note_id]
            
            # Clean up note_number_to_ids mapping
            if oldest_note_number in self.state.note_number_to_ids:
                self.state.note_number_to_ids[oldest_note_number].discard(oldest_note_id)
                if not self.state.note_number_to_ids[oldest_note_number]:
                    del self.state.note_number_to_ids[oldest_note_number]
            
            if MIDI_DEBUG:
                print(f"Voice stealing: removed note {oldest_note_number} (id {oldest_note_id}) to make room")
        
        # Generate unique ID for this note instance
        note_index = self.state.note_id_counter
        self.state.note_id_counter += 1

        # Create a new instance of the signal node for this note
        sound_node = self.instantiate_child_node(self.signal_model, "sounds", note_index)
        
        # Setup render args - pass frequency and amplitude with aliases
        render_args = {
            'freq': frequency,
            'f': frequency,  # Alias
            'amp': amplitude,
            'a': amplitude,  # Alias
        }
        
        # Store the active note with unique ID
        self.state.active_notes[note_index] = {
            'note_number': note_number,
            'node': sound_node,
            'render_args': render_args,
            'samples_rendered': 0,
            'is_in_sustain': True
        }
        
        # Track note ID for this note number (for note off handling)
        if note_number not in self.state.note_number_to_ids:
            self.state.note_number_to_ids[note_number] = []
        self.state.note_number_to_ids[note_number].append(note_index)
        
        if MIDI_DEBUG:
            print(f"Note ON: {note_number} (freq: {frequency:.2f}Hz, vel: {velocity}, id: {note_index})")
    
    def _handle_note_off(self, note_number):
        """Handle MIDI note off event"""
        # Mark all instances of this note as no longer in sustain, triggering release phase
        if note_number in self.state.note_number_to_ids:
            for note_id in self.state.note_number_to_ids[note_number]:
                if note_id in self.state.active_notes:
                    self.state.active_notes[note_id]['is_in_sustain'] = False
                    if MIDI_DEBUG:
                        print(f"Note OFF: {note_number} (id: {note_id}, starting release)")
    
    def _do_render(self, num_samples=None, context=None, **params):
        # MIDI node never finishes, so if num_samples is None, use a default buffer size
        if num_samples is None:
            from config import BUFFER_SIZE
            # For realtime mode, use buffer size
            # MIDI node continues indefinitely - never returns empty array
            num_samples = BUFFER_SIZE
            self._last_chunk_samples = num_samples
        
        # Process any pending MIDI messages
        self._process_midi_messages(**params)
        
        # Create output buffer
        output_wave = np.zeros(num_samples, dtype=np.float32)
        
        # Render all active notes and mix them together
        notes_to_remove = []
        
        for note_id, note_data in self.state.active_notes.items():
            note_number = note_data['note_number']
            sound_node = note_data['node']
            render_args = note_data['render_args']
            is_in_sustain = note_data['is_in_sustain']
            
            # Merge render_args with params
            merged_params = self.get_params_for_children(params)
            merged_params.update(render_args)
            
            # Add gate signal (1.0 if sustaining, 0.0 if released)
            merged_params['gate'] = 1.0 if is_in_sustain else 0.0
            
            # Render the note
            try:
                note_chunk = sound_node.render(num_samples, context, **merged_params)
                
                # If the note returns an empty array, it has finished (e.g., envelope end=True)
                if len(note_chunk) == 0:
                    notes_to_remove.append((note_id, note_number))
                    continue
                
                # Update samples rendered counter
                note_data['samples_rendered'] += len(note_chunk)
                
                # Mix into output (pad if needed)
                if len(note_chunk) < len(output_wave):
                    note_chunk = np.pad(note_chunk, (0, len(output_wave) - len(note_chunk)))
                
                output_wave[:len(note_chunk)] += note_chunk
            except Exception as e:
                print(f"Error rendering note {note_number} (id: {note_id}): {e}")
                notes_to_remove.append((note_id, note_number))
        
        # Remove finished notes
        for note_id, note_number in notes_to_remove:
            if note_id in self.state.active_notes:
                del self.state.active_notes[note_id]
            
            # Clean up note_number_to_ids mapping
            if note_number in self.state.note_number_to_ids:
                if note_id in self.state.note_number_to_ids[note_number]:
                    self.state.note_number_to_ids[note_number].remove(note_id)
                
                # Remove the note_number key if no more instances exist
                if not self.state.note_number_to_ids[note_number]:
                    del self.state.note_number_to_ids[note_number]
        
        return output_wave


MIDI_DEFINITION = NodeDefinition("midi", MidiNode, MidiModel)
