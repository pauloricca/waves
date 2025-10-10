"""
Shared MIDI utilities for MIDI input handling across different MIDI nodes.
"""
from __future__ import annotations
import mido
import threading
import queue

from config import MIDI_INPUT_DEVICE_NAME


# Debug settings
MIDI_DEBUG = False  # Set to True to see MIDI messages


class MidiInputManager:
    """Singleton class managing shared MIDI input across all MIDI nodes"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._midi_input = None
        self._midi_queue = queue.Queue()
        self._ensure_midi_input()
    
    def _ensure_midi_input(self):
        """Ensure MIDI input is initialized"""
        with self._lock:
            if self._midi_input is None:
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
                        self._midi_input = mido.open_input(selected_port, callback=self._midi_callback)
                        print(f"MIDI input opened: {selected_port}")
                    else:
                        if MIDI_DEBUG:
                            print("Warning: No MIDI input ports available")
                        # Create a virtual port as fallback
                        try:
                            self._midi_input = mido.open_input('waves_virtual', virtual=True, callback=self._midi_callback)
                            if MIDI_DEBUG:
                                print("Created virtual MIDI port: waves_virtual")
                        except:
                            if MIDI_DEBUG:
                                print("Warning: Could not create virtual MIDI port")
                except Exception as e:
                    if MIDI_DEBUG:
                        print(f"Warning: Could not open MIDI input: {e}")
    
    def _midi_callback(self, message):
        """Callback for MIDI messages (runs in MIDI thread)"""
        self._midi_queue.put(message)
    
    def get_messages(self):
        """Get all pending MIDI messages from the queue"""
        messages = []
        while not self._midi_queue.empty():
            try:
                messages.append(self._midi_queue.get_nowait())
            except queue.Empty:
                break
        return messages
    
    @property
    def queue(self):
        """Access to the MIDI message queue"""
        return self._midi_queue


def midi_note_to_frequency(note_number: int) -> float:
    """Convert MIDI note number to frequency (A4 = 440Hz = MIDI note 69)"""
    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))


def midi_velocity_to_amplitude(velocity: int) -> float:
    """Convert MIDI velocity (0-127) to amplitude (0.0-1.0)"""
    return velocity / 127.0
