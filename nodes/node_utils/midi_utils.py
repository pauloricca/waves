"""
Shared MIDI utilities for MIDI input handling across different MIDI nodes.
"""
from __future__ import annotations
import mido
import threading
import queue
import json
import os
import time

from config import MIDI_INPUT_DEVICE_NAME, DO_PERSIST_MIDI_CC_VALUES, MIDI_CC_SAVE_INTERVAL


# Debug settings
MIDI_DEBUG = False  # Set to True to see MIDI messages

# Path to store persistent MIDI CC state
MIDI_STATE_FILE = ".midi_state.json"


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
        # Store the latest CC value for each (channel, cc_number) pair
        self._cc_values = {}  # Key: (channel, cc_number), Value: midi value (0-127)
        self._cc_values_dirty = False  # Track if values changed since last save
        self._save_thread = None
        self._stop_save_thread = threading.Event()
        
        if DO_PERSIST_MIDI_CC_VALUES:
            self._load_cc_values()  # Load persisted values on startup
            self._start_save_thread()  # Start periodic save thread
        
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
    
    def _load_cc_values(self):
        """Load persisted CC values from file"""
        if os.path.exists(MIDI_STATE_FILE):
            try:
                with open(MIDI_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to tuples (channel, cc_number)
                    self._cc_values = {
                        tuple(map(int, key.split(','))): value 
                        for key, value in data.items()
                    }
                    if MIDI_DEBUG:
                        print(f"Loaded {len(self._cc_values)} MIDI CC values from {MIDI_STATE_FILE}")
            except Exception as e:
                if MIDI_DEBUG:
                    print(f"Warning: Could not load MIDI state: {e}")
                self._cc_values = {}
    
    def _save_cc_values(self):
        """Save current CC values to file"""
        if not self._cc_values_dirty:
            return  # Nothing changed, skip save
            
        try:
            # Convert tuple keys to strings for JSON serialization
            data = {f"{channel},{cc}": value for (channel, cc), value in self._cc_values.items()}
            with open(MIDI_STATE_FILE, 'w') as f:
                json.dump(data, f)
            self._cc_values_dirty = False
            if MIDI_DEBUG:
                print(f"Saved {len(self._cc_values)} MIDI CC values to {MIDI_STATE_FILE}")
        except Exception as e:
            if MIDI_DEBUG:
                print(f"Warning: Could not save MIDI state: {e}")
    
    def _save_thread_worker(self):
        """Background thread that periodically saves CC values"""
        while not self._stop_save_thread.wait(MIDI_CC_SAVE_INTERVAL):
            if self._cc_values_dirty:
                self._save_cc_values()
    
    def _start_save_thread(self):
        """Start the background save thread"""
        if self._save_thread is None or not self._save_thread.is_alive():
            self._stop_save_thread.clear()
            self._save_thread = threading.Thread(target=self._save_thread_worker, daemon=True)
            self._save_thread.start()
            if MIDI_DEBUG:
                print(f"Started MIDI CC save thread (interval: {MIDI_CC_SAVE_INTERVAL}s)")
    
    def shutdown(self):
        """Clean shutdown: save values one final time and stop threads"""
        if DO_PERSIST_MIDI_CC_VALUES:
            self._stop_save_thread.set()
            if self._save_thread and self._save_thread.is_alive():
                self._save_thread.join(timeout=1.0)
            # Final save before shutdown
            if self._cc_values_dirty:
                self._save_cc_values()
    
    def _midi_callback(self, message):
        """Callback for MIDI messages (runs in MIDI thread)"""
        # Store CC values in a dict for efficient lookup
        if message.type == 'control_change':
            key = (message.channel, message.control)
            self._cc_values[key] = message.value
            self._cc_values_dirty = True  # Mark as needing save
        # Also put in queue for backwards compatibility with nodes that want all messages
        self._midi_queue.put(message)
    
    def get_cc_value(self, channel, cc_number):
        """Get the current value for a specific CC on a specific channel.
        
        Args:
            channel: MIDI channel (0-15)
            cc_number: CC number (0-127)
            
        Returns:
            The current CC value (0-127) or None if no value has been received yet
        """
        key = (channel, cc_number)
        return self._cc_values.get(key)
    
    def get_messages(self, channel=None, cc_number=None):
        """Get all pending MIDI messages from the queue, optionally filtered by channel and CC number.
        
        This drains the queue, so messages are only retrieved once. For CC values, prefer using
        get_cc_value() which provides the latest value without consuming messages.
        
        Args:
            channel: Optional MIDI channel to filter (0-15)
            cc_number: Optional CC number to filter (0-127)
        """
        messages = []
        
        # Get all messages from the queue
        while not self._midi_queue.empty():
            try:
                msg = self._midi_queue.get_nowait()
                
                # Filter messages if channel/CC specified
                if channel is not None or cc_number is not None:
                    if hasattr(msg, 'channel') and msg.type == 'control_change':
                        if (channel is None or msg.channel == channel) and \
                           (cc_number is None or msg.control == cc_number):
                            messages.append(msg)
                else:
                    messages.append(msg)
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
