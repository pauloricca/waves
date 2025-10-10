from __future__ import annotations
from typing import Optional
import math
import numpy as np
from pydantic import ConfigDict
import mido
import threading
import queue

from config import SAMPLE_RATE, MIDI_INPUT_DEVICE_NAME
from constants import RenderArgs
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition

# Debug settings
MIDI_DEBUG = False  # Set to True to see MIDI note on/off messages


class MidiModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    channel: int = 0  # MIDI channel to listen to (0-15)
    signal: BaseNodeModel  # The sound/signal to play when a note is triggered
    duration: float = math.inf  # MIDI nodes run indefinitely


class MidiNode(BaseNode):
    # Class-level MIDI input port (shared across all instances)
    _midi_input = None
    _midi_thread = None
    _midi_queue = queue.Queue()
    _midi_lock = threading.Lock()
    
    def __init__(self, model: MidiModel):
        super().__init__(model)
        self.channel = model.channel
        self.signal_model = model.signal
        
        # Track active notes: dict of {unique_id: (note_number, sound_node, render_args, samples_rendered, is_in_sustain)}
        self.active_notes = {}
        
        # Counter for generating unique note IDs (allows multiple instances of same note)
        self.note_id_counter = 0
        
        # Track which note IDs correspond to which note numbers for note off handling
        self.note_number_to_ids = {}  # {note_number: [id1, id2, ...]}
        
        # Initialize MIDI input if not already done
        self._ensure_midi_input()
    
    @classmethod
    def _ensure_midi_input(cls):
        """Ensure MIDI input is initialized (class-level, shared across instances)"""
        with cls._midi_lock:
            if cls._midi_input is None:
                try:
                    available_ports = mido.get_input_names()
                    if MIDI_DEBUG:
                        print(f"Available MIDI ports: {available_ports}")
                    
                    selected_port = None
                    
                    # First, try to use the configured device name
                    if MIDI_INPUT_DEVICE_NAME:
                        if MIDI_INPUT_DEVICE_NAME in available_ports:
                            selected_port = MIDI_INPUT_DEVICE_NAME
                            if MIDI_DEBUG:
                                print(f"Using configured MIDI device: {selected_port}")
                        else:
                            print(f"Warning: Configured MIDI device '{MIDI_INPUT_DEVICE_NAME}' not found")
                    
                    # If no configured device or it wasn't found, try to find an external controller
                    if selected_port is None and available_ports:
                        # Try to find a non-IAC driver port (external controller)
                        for port in available_ports:
                            if 'IAC' not in port:
                                selected_port = port
                                if MIDI_DEBUG:
                                    print(f"Auto-detected external MIDI controller: {selected_port}")
                                break
                    
                    # Fall back to IAC Driver if available
                    if selected_port is None and available_ports:
                        for port in available_ports:
                            if 'IAC' in port:
                                selected_port = port
                                if MIDI_DEBUG:
                                    print(f"Using IAC Driver: {selected_port}")
                                break
                    
                    # Use first available port if still nothing selected
                    if selected_port is None and available_ports:
                        selected_port = available_ports[0]
                        if MIDI_DEBUG:
                            print(f"Using first available port: {selected_port}")
                    
                    # Open the selected port or create virtual port
                    if selected_port:
                        cls._midi_input = mido.open_input(selected_port, callback=cls._midi_callback)
                        print(f"MIDI input opened: {selected_port}")
                    else:
                        if MIDI_DEBUG:
                            print("Warning: No MIDI input ports available")
                        # Create a virtual port as fallback
                        try:
                            cls._midi_input = mido.open_input('waves_virtual', virtual=True, callback=cls._midi_callback)
                            if MIDI_DEBUG:
                                print("Created virtual MIDI port: waves_virtual")
                        except:
                            if MIDI_DEBUG:
                                print("Warning: Could not create virtual MIDI port")
                except Exception as e:
                    if MIDI_DEBUG:
                        print(f"Warning: Could not open MIDI input: {e}")
    
    @classmethod
    def _midi_callback(cls, message):
        """Callback for MIDI messages (runs in MIDI thread)"""
        cls._midi_queue.put(message)
    
    def _process_midi_messages(self, **params):
        """Process all pending MIDI messages from the queue"""
        while not self._midi_queue.empty():
            try:
                message = self._midi_queue.get_nowait()
                
                # Only process messages for our channel
                if hasattr(message, 'channel') and message.channel == self.channel:
                    if message.type == 'note_on' and message.velocity > 0:
                        self._handle_note_on(message.note, message.velocity, **params)
                    elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                        self._handle_note_off(message.note)
            except queue.Empty:
                break
    
    def _handle_note_on(self, note_number, velocity, **params):
        """Handle MIDI note on event"""
        from nodes.node_utils.instantiate_node import instantiate_node
        
        # Convert MIDI note number to frequency (A4 = 440Hz = MIDI note 69)
        frequency = 440.0 * (2.0 ** ((note_number - 69) / 12.0))
        
        # Create a new instance of the signal node for this note
        sound_node = instantiate_node(self.signal_model)
        
        # Setup render args with frequency
        render_args = {
            RenderArgs.FREQUENCY: frequency,
            RenderArgs.AMPLITUDE_MULTIPLIER: velocity / 127.0  # Normalize velocity to 0-1
        }
        
        # Generate unique ID for this note instance
        note_id = self.note_id_counter
        self.note_id_counter += 1
        
        # Store the active note with unique ID
        self.active_notes[note_id] = {
            'note_number': note_number,
            'node': sound_node,
            'render_args': render_args,
            'samples_rendered': 0,
            'is_in_sustain': True
        }
        
        # Track note ID for this note number (for note off handling)
        if note_number not in self.note_number_to_ids:
            self.note_number_to_ids[note_number] = []
        self.note_number_to_ids[note_number].append(note_id)
        
        if MIDI_DEBUG:
            print(f"Note ON: {note_number} (freq: {frequency:.2f}Hz, vel: {velocity}, id: {note_id})")
    
    def _handle_note_off(self, note_number):
        """Handle MIDI note off event"""
        # Mark all instances of this note as no longer in sustain, triggering release phase
        if note_number in self.note_number_to_ids:
            for note_id in self.note_number_to_ids[note_number]:
                if note_id in self.active_notes:
                    self.active_notes[note_id]['is_in_sustain'] = False
                    if MIDI_DEBUG:
                        print(f"Note OFF: {note_number} (id: {note_id}, starting release)")
    
    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
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
        
        for note_id, note_data in self.active_notes.items():
            note_number = note_data['note_number']
            sound_node = note_data['node']
            render_args = note_data['render_args']
            is_in_sustain = note_data['is_in_sustain']
            
            # Merge render_args with params
            merged_params = self.get_params_for_children(params)
            merged_params.update(render_args)
            
            # Add sustain state to params
            merged_params[RenderArgs.IS_IN_SUSTAIN] = is_in_sustain
            
            # Render the note
            try:
                note_chunk = sound_node.render(num_samples, **merged_params)
                
                # If the sound returns empty, the note is finished (e.g., envelope release complete)
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
            if note_id in self.active_notes:
                del self.active_notes[note_id]
            
            # Clean up note_number_to_ids mapping
            if note_number in self.note_number_to_ids:
                if note_id in self.note_number_to_ids[note_number]:
                    self.note_number_to_ids[note_number].remove(note_id)
                
                # Remove the note_number key if no more instances exist
                if not self.note_number_to_ids[note_number]:
                    del self.note_number_to_ids[note_number]
        
        return output_wave
    
    def __del__(self):
        """Cleanup when node is destroyed"""
        # Note: We keep the MIDI input open as it's shared across instances
        pass


MIDI_DEFINITION = NodeDefinition("midi", MidiNode, MidiModel)
