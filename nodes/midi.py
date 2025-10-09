from __future__ import annotations
from typing import Optional
import math
import numpy as np
from pydantic import ConfigDict
import mido
import threading
import queue

from config import SAMPLE_RATE
from constants import RenderArgs
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from utils import look_for_duration

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
        
        # Track active notes: dict of {note_number: (sound_node, render_args, sound_duration, samples_rendered)}
        self.active_notes = {}
        
        # Initialize MIDI input if not already done
        self._ensure_midi_input()
    
    @classmethod
    def _ensure_midi_input(cls):
        """Ensure MIDI input is initialized (class-level, shared across instances)"""
        with cls._midi_lock:
            if cls._midi_input is None:
                try:
                    # Try to open the first available MIDI input port
                    available_ports = mido.get_input_names()
                    if available_ports:
                        if MIDI_DEBUG:
                            print(f"Available MIDI ports: {available_ports}")
                        cls._midi_input = mido.open_input(available_ports[0], callback=cls._midi_callback)
                        if MIDI_DEBUG:
                            print(f"Opened MIDI input: {available_ports[0]}")
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
        
        # Store the active note
        self.active_notes[note_number] = {
            'node': sound_node,
            'render_args': render_args,
            'samples_rendered': 0,  # Track for potential future use
            'is_in_sustain': True  # Note is being held
        }
        
        if MIDI_DEBUG:
            print(f"Note ON: {note_number} (freq: {frequency:.2f}Hz, vel: {velocity})")
    
    def _handle_note_off(self, note_number):
        """Handle MIDI note off event"""
        # Mark the note as no longer in sustain, triggering release phase
        if note_number in self.active_notes:
            self.active_notes[note_number]['is_in_sustain'] = False
            if MIDI_DEBUG:
                print(f"Note OFF: {note_number} (starting release)")
    
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
        
        for note_number, note_data in self.active_notes.items():
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
                    notes_to_remove.append(note_number)
                    continue
                
                # Update samples rendered counter
                note_data['samples_rendered'] += len(note_chunk)
                
                # Mix into output (pad if needed)
                if len(note_chunk) < len(output_wave):
                    note_chunk = np.pad(note_chunk, (0, len(output_wave) - len(note_chunk)))
                
                output_wave[:len(note_chunk)] += note_chunk
            except Exception as e:
                print(f"Error rendering note {note_number}: {e}")
                notes_to_remove.append(note_number)
        
        # Remove finished notes
        for note_number in notes_to_remove:
            if note_number in self.active_notes:
                del self.active_notes[note_number]
        
        return output_wave
    
    def __del__(self):
        """Cleanup when node is destroyed"""
        # Note: We keep the MIDI input open as it's shared across instances
        pass


MIDI_DEFINITION = NodeDefinition("midi", MidiNode, MidiModel)
