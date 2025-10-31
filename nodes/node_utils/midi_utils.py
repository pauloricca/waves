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

from config import MIDI_INPUT_DEVICES, MIDI_DEFAULT_DEVICE_KEY, DO_PERSIST_MIDI_CC_VALUES, MIDI_CC_SAVE_INTERVAL, MIDI_OUTPUT_DEVICE, MIDI_CLOCK_ENABLED


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
        # Dictionary of device_key -> {input, queue, cc_values, cc_values_dirty, device_name}
        self._devices = {}
        self._save_thread = None
        self._stop_save_thread = threading.Event()
        # Track last MIDI message for display: (device_key, message_type, channel, data1, data2)
        self._last_message = None
        
        # MIDI Clock tracking (per device)
        # device_key -> {bpm, last_clock_time, clock_count, clock_times}
        self._clock_state = {}
        
        self._initialize_devices()
        
        if DO_PERSIST_MIDI_CC_VALUES:
            self._load_all_cc_values()  # Load persisted values on startup
            self._start_save_thread()  # Start periodic save thread
    
    def _initialize_devices(self):
        """Initialize all configured MIDI devices"""
        available_ports = mido.get_input_names()
        available_ports_str = ", ".join(available_ports)
        print(f"Available MIDI input devices: {available_ports_str}")
        
        if not MIDI_INPUT_DEVICES:
            # No devices configured, use auto-detect for default device
            device_key = "default"
            selected_port = self._auto_detect_port(available_ports)
            if selected_port:
                self._open_device(device_key, selected_port)
        else:
            # Open all configured devices
            print(f"Configured MIDI devices:")
            for device_key, device_name in MIDI_INPUT_DEVICES.items():
                if device_name in available_ports:
                    self._open_device(device_key, device_name)
                    print(f"  ✓ {device_key}: {device_name}")
                else:
                    print(f"  ✗ {device_key}: {device_name} (not found)")
            
            # If default device key is specified but not found, try auto-detect
            if MIDI_DEFAULT_DEVICE_KEY and MIDI_DEFAULT_DEVICE_KEY not in self._devices:
                print(f"Warning: Default device '{MIDI_DEFAULT_DEVICE_KEY}' not available")
                selected_port = self._auto_detect_port(available_ports)
                if selected_port:
                    print(f"Auto-detected fallback device: {selected_port}")
                    self._open_device("default", selected_port)
        
        if not self._devices:
            print("Warning: No MIDI devices opened")
    
    def _auto_detect_port(self, available_ports):
        """Auto-detect a suitable MIDI port"""
        if not available_ports:
            return None
        
        # Try to find a non-IAC driver port (external controller)
        for port in available_ports:
            if 'IAC' not in port:
                return port
        
        # Fall back to IAC Driver if available
        for port in available_ports:
            if 'IAC' in port:
                return port
        
        # Use first available port
        return available_ports[0]
    
    def _open_device(self, device_key, device_name):
        """Open a MIDI device and set up its callback"""
        try:
            midi_input = mido.open_input(device_name, callback=lambda msg: self._midi_callback(device_key, msg))
            self._devices[device_key] = {
                'input': midi_input,
                'queue': queue.Queue(),
                'cc_values': {},  # (channel, cc_number) -> value
                'cc_values_dirty': False,
                'device_name': device_name
            }
            # Initialize clock state for this device
            self._clock_state[device_key] = {
                'bpm': None,
                'last_clock_time': None,
                'clock_count': 0,
                'clock_times': []  # Rolling buffer for BPM calculation
            }
            if MIDI_DEBUG:
                print(f"Opened MIDI device '{device_key}': {device_name}")
        except Exception as e:
            print(f"Error opening MIDI device '{device_key}' ({device_name}): {e}")
    
    def _get_device(self, device_key=None):
        """Get device info for a specific key, or default device"""
        if device_key and device_key in self._devices:
            return self._devices[device_key]
        
        # Try default device key from config
        if MIDI_DEFAULT_DEVICE_KEY and MIDI_DEFAULT_DEVICE_KEY in self._devices:
            return self._devices[MIDI_DEFAULT_DEVICE_KEY]
        
        # Try "default" key (auto-detected)
        if "default" in self._devices:
            return self._devices["default"]
        
        # Return first available device
        if self._devices:
            return next(iter(self._devices.values()))
        
        return None
    
    def _load_all_cc_values(self):
        """Load persisted CC values from file for all devices"""
        if os.path.exists(MIDI_STATE_FILE):
            try:
                with open(MIDI_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    
                    # Load CC values for each device
                    for device_key, device_info in self._devices.items():
                        device_name = device_info['device_name']
                        device_data = data.get(device_name)
                        
                        if device_data:
                            # Convert string keys back to tuples (channel, cc_number)
                            cc_values = device_data.get('cc_values', {})
                            device_info['cc_values'] = {
                                tuple(map(int, key.split(','))): value 
                                for key, value in cc_values.items()
                            }
                            if MIDI_DEBUG:
                                print(f"Loaded {len(device_info['cc_values'])} MIDI CC values for device '{device_key}' ({device_name})")
            except Exception as e:
                if MIDI_DEBUG:
                    print(f"Warning: Could not load MIDI state: {e}")
    
    def _save_all_cc_values(self):
        """Save current CC values to file for all devices"""
        # Check if any device has dirty values
        has_dirty = any(device_info['cc_values_dirty'] for device_info in self._devices.values())
        if not has_dirty:
            return  # Nothing changed, skip save
            
        try:
            # Build data structure: device_name -> {cc_values: {...}}
            data = {}
            for device_key, device_info in self._devices.items():
                if device_info['cc_values_dirty']:
                    device_name = device_info['device_name']
                    # Convert tuple keys to strings for JSON serialization
                    cc_values = {
                        f"{channel},{cc}": value 
                        for (channel, cc), value in device_info['cc_values'].items()
                    }
                    data[device_name] = {'cc_values': cc_values}
                    device_info['cc_values_dirty'] = False
            
            # Load existing data to merge
            if os.path.exists(MIDI_STATE_FILE):
                with open(MIDI_STATE_FILE, 'r') as f:
                    existing_data = json.load(f)
                    # Merge with existing data (don't lose other devices)
                    existing_data.update(data)
                    data = existing_data
            
            with open(MIDI_STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            if MIDI_DEBUG:
                print(f"Saved MIDI CC values for {len(data)} devices to {MIDI_STATE_FILE}")
        except Exception as e:
            if MIDI_DEBUG:
                print(f"Warning: Could not save MIDI state: {e}")
    
    def _save_thread_worker(self):
        """Background thread that periodically saves CC values"""
        while not self._stop_save_thread.wait(MIDI_CC_SAVE_INTERVAL):
            self._save_all_cc_values()
    
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
            self._save_all_cc_values()
    
    def _midi_callback(self, device_key, message):
        """Callback for MIDI messages (runs in MIDI thread)"""
        device_info = self._devices.get(device_key)
        if not device_info:
            return
        
        # Handle MIDI clock messages for tempo sync
        if message.type == 'clock':
            self._handle_clock_message(device_key)
        elif message.type == 'start':
            # Reset clock on MIDI start
            clock_state = self._clock_state.get(device_key)
            if clock_state:
                clock_state['clock_count'] = 0
                clock_state['clock_times'] = []
                clock_state['last_clock_time'] = None
        elif message.type == 'stop':
            # Could handle stop if needed
            pass
            
        # Store CC values in device-specific dict for efficient lookup
        if message.type == 'control_change':
            key = (message.channel, message.control)
            device_info['cc_values'][key] = message.value
            device_info['cc_values_dirty'] = True  # Mark as needing save
            # Track last CC message for display
            self._last_message = (device_key, 'cc', message.channel, message.control, message.value)
        elif message.type == 'note_on' and message.velocity > 0:
            # Track last note_on message for display
            self._last_message = (device_key, 'note_on', message.channel, message.note, message.velocity)
        
        # Also put in queue for backwards compatibility with nodes that want all messages
        device_info['queue'].put(message)
    
    def _handle_clock_message(self, device_key):
        """Handle MIDI clock message and calculate BPM.
        
        MIDI sends 24 clock pulses per quarter note (beat).
        By measuring the time between clock pulses, we can calculate BPM.
        """
        clock_state = self._clock_state.get(device_key)
        if not clock_state:
            return
        
        current_time = time.time()
        
        # Track clock times for averaging
        if clock_state['last_clock_time'] is not None:
            clock_state['clock_times'].append(current_time - clock_state['last_clock_time'])
            
            # Keep only the last 24 clock intervals (one beat) for averaging
            if len(clock_state['clock_times']) > 24:
                clock_state['clock_times'].pop(0)
            
            # Calculate BPM from average clock interval
            # 24 clocks per beat, so 1 beat = average_interval * 24
            if len(clock_state['clock_times']) >= 12:  # Need at least half a beat of data
                avg_interval = sum(clock_state['clock_times']) / len(clock_state['clock_times'])
                beat_duration = avg_interval * 24  # seconds per beat
                bpm = 60.0 / beat_duration  # beats per minute
                clock_state['bpm'] = bpm
                
                if MIDI_DEBUG:
                    print(f"MIDI Clock BPM: {bpm:.1f}")
        
        clock_state['last_clock_time'] = current_time
        clock_state['clock_count'] += 1
    
    def get_midi_clock_bpm(self, device_key=None):
        """Get the current BPM from MIDI clock messages.
        
        Args:
            device_key: Optional device key (from config). If None, uses default device.
            
        Returns:
            The current BPM as a float, or None if no clock has been received yet
        """
        device_info = self._get_device(device_key)
        if not device_info:
            return None
        
        # Get the actual device_key that was selected
        actual_device_key = None
        for key, info in self._devices.items():
            if info == device_info:
                actual_device_key = key
                break
        
        if actual_device_key is None:
            return None
        
        clock_state = self._clock_state.get(actual_device_key)
        if clock_state:
            return clock_state['bpm']
        return None

    
    def get_cc_value(self, channel, cc_number, device_key=None):
        """Get the current value for a specific CC on a specific channel.
        
        Args:
            channel: MIDI channel (0-15)
            cc_number: CC number (0-127)
            device_key: Optional device key (from config). If None, uses default device.
            
        Returns:
            The current CC value (0-127) or None if no value has been received yet
        """
        device_info = self._get_device(device_key)
        if not device_info:
            return None
            
        key = (channel, cc_number)
        return device_info['cc_values'].get(key)
    
    def get_messages(self, channel=None, cc_number=None, device_key=None):
        """Get all pending MIDI messages from the queue, optionally filtered by channel and CC number.
        
        This drains the queue, so messages are only retrieved once. For CC values, prefer using
        get_cc_value() which provides the latest value without consuming messages.
        
        Args:
            channel: Optional MIDI channel to filter (0-15)
            cc_number: Optional CC number to filter (0-127)
            device_key: Optional device key (from config). If None, uses default device.
        """
        device_info = self._get_device(device_key)
        if not device_info:
            return []
            
        messages = []
        
        # Get all messages from the device's queue
        while not device_info['queue'].empty():
            try:
                msg = device_info['queue'].get_nowait()
                
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
    
    def get_queue(self, device_key=None):
        """Get the message queue for a specific device.
        
        Args:
            device_key: Optional device key (from config). If None, uses default device.
        """
        device_info = self._get_device(device_key)
        if device_info:
            return device_info['queue']
        return None


class MidiOutputManager:
    """Singleton class for MIDI output, primarily for MIDI clock sync"""
    
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
        self._outputs = []  # List of output ports
        self._current_bpm = None
        self._clock_enabled = False
        self._samples_per_clock = None
        self._sample_counter = 0
        self._is_playing = False
        
        if MIDI_CLOCK_ENABLED:
            self._initialize_outputs()
    
    def _initialize_outputs(self):
        """Initialize all MIDI output devices"""
        available_ports = mido.get_output_names()
        
        if not available_ports:
            if MIDI_DEBUG:
                print("Warning: No MIDI output devices available")
            return
        
        # Open all available MIDI output ports
        for port_name in available_ports:
            try:
                output = mido.open_output(port_name)
                self._outputs.append(output)
                if MIDI_DEBUG:
                    print(f"Opened MIDI output: {port_name}")
            except Exception as e:
                if MIDI_DEBUG:
                    print(f"Error opening MIDI output '{port_name}': {e}")
    
    def enable_clock(self, bpm: float):
        """Enable MIDI clock output at the specified BPM"""
        from config import SAMPLE_RATE
        
        if not self._outputs or not MIDI_CLOCK_ENABLED:
            return
        
        self._current_bpm = bpm
        self._clock_enabled = True
        
        # MIDI clock: 24 pulses per quarter note (PPQN)
        # At 120 BPM: 120 beats/min = 2 beats/sec = 48 clocks/sec = 0.02083 sec/clock
        # samples_per_clock = SAMPLE_RATE / (BPM / 60 * 24)
        clocks_per_second = (bpm / 60.0) * 24.0
        self._samples_per_clock = SAMPLE_RATE / clocks_per_second
        self._sample_counter = 0
        
        # Send MIDI start message to all outputs
        if not self._is_playing:
            for output in self._outputs:
                try:
                    output.send(mido.Message('start'))
                except Exception as e:
                    if MIDI_DEBUG:
                        print(f"Error sending MIDI start to {output.name}: {e}")
            self._is_playing = True
            if MIDI_DEBUG:
                print(f"MIDI clock started at {bpm} BPM on {len(self._outputs)} device(s)")
    
    def disable_clock(self):
        """Disable MIDI clock output"""
        if not self._outputs:
            return
        
        self._clock_enabled = False
        
        # Send MIDI stop message to all outputs
        if self._is_playing:
            for output in self._outputs:
                try:
                    output.send(mido.Message('stop'))
                except Exception as e:
                    if MIDI_DEBUG:
                        print(f"Error sending MIDI stop to {output.name}: {e}")
            self._is_playing = False
            if MIDI_DEBUG:
                print(f"MIDI clock stopped on {len(self._outputs)} device(s)")
    
    def update_bpm(self, bpm: float):
        """Update BPM for MIDI clock (recalculates timing without stopping/starting)"""
        from config import SAMPLE_RATE
        
        if not self._clock_enabled or not self._outputs:
            return
        
        self._current_bpm = bpm
        clocks_per_second = (bpm / 60.0) * 24.0
        self._samples_per_clock = SAMPLE_RATE / clocks_per_second
    
    def process_samples(self, num_samples: int):
        """Process a chunk of samples and send MIDI clock messages as needed"""
        if not self._clock_enabled or not self._outputs or self._samples_per_clock is None:
            return
        
        # Track how many samples we've processed and send clock messages accordingly
        self._sample_counter += num_samples
        
        # Send clock messages for each full clock period that has passed
        while self._sample_counter >= self._samples_per_clock:
            for output in self._outputs:
                try:
                    output.send(mido.Message('clock'))
                except Exception as e:
                    if MIDI_DEBUG:
                        print(f"Error sending MIDI clock to {output.name}: {e}")
            
            self._sample_counter -= self._samples_per_clock
    
    def shutdown(self):
        """Clean shutdown: stop clock and close all outputs"""
        self.disable_clock()
        for output in self._outputs:
            output.close()
        self._outputs = []
    
    def _get_output(self, device_name: str = None):
        """Get a specific output device by name, or use configured default"""
        if device_name:
            # Find output by name
            for output in self._outputs:
                if output.name == device_name:
                    return output
            if MIDI_DEBUG:
                print(f"Warning: MIDI output device '{device_name}' not found")
            return None
        
        # Use configured default device if specified
        if MIDI_OUTPUT_DEVICE:
            for output in self._outputs:
                if output.name == MIDI_OUTPUT_DEVICE:
                    return output
        
        # Fall back to first available output
        if self._outputs:
            return self._outputs[0]
        
        return None
    
    def send_note(self, device_name: str, channel: int, note: int, velocity: int, is_on: bool):
        """Send a MIDI note on or note off message
        
        Args:
            device_name: Name of MIDI device (None for default)
            channel: MIDI channel (0-15)
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            is_on: True for note on, False for note off
        """
        output = self._get_output(device_name)
        if not output:
            return
        
        try:
            msg_type = 'note_on' if is_on else 'note_off'
            msg = mido.Message(msg_type, channel=channel, note=note, velocity=velocity)
            output.send(msg)
            if MIDI_DEBUG:
                print(f"MIDI {msg_type}: ch={channel} note={note} vel={velocity} -> {output.name}")
        except Exception as e:
            if MIDI_DEBUG:
                print(f"Error sending MIDI note to {output.name}: {e}")
    
    def send_cc(self, device_name: str, channel: int, control: int, value: int):
        """Send a MIDI CC (control change) message
        
        Args:
            device_name: Name of MIDI device (None for default)
            channel: MIDI channel (0-15)
            control: CC number (0-127)
            value: CC value (0-127)
        """
        output = self._get_output(device_name)
        if not output:
            return
        
        try:
            msg = mido.Message('control_change', channel=channel, control=control, value=value)
            output.send(msg)
            if MIDI_DEBUG:
                print(f"MIDI CC: ch={channel} cc={control} val={value} -> {output.name}")
        except Exception as e:
            if MIDI_DEBUG:
                print(f"Error sending MIDI CC to {output.name}: {e}")


def midi_note_to_frequency(note_number: int) -> float:
    """Convert MIDI note number to frequency (A4 = 440Hz = MIDI note 69)"""
    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))


def midi_velocity_to_amplitude(velocity: int) -> float:
    """Convert MIDI velocity (0-127) to amplitude (0.0-1.0)"""
    return velocity / 127.0


def midi_note_to_name(note_number: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note_name = note_names[note_number % 12]
    return f"{note_name}{octave}"


def get_last_midi_message_display() -> str | None:
    """
    Get a formatted display string for the most recent MIDI message across all devices.
    
    Returns:
        Formatted string like "DEVICE_KEY ch: 0 cc: 23  v: 127" or "DEVICE_KEY ch: 0 note: C4 (60)  v: 80"
        or None if no messages have been received
    """
    manager = MidiInputManager()
    
    if not manager._last_message:
        return None
    
    # Unpack the last message
    device_key, message_type, channel, data1, data2 = manager._last_message
    
    if message_type == 'cc':
        # data1 = cc number, data2 = value
        return f"{device_key} ch: {channel}  cc: {data1}  v: {data2}"
    elif message_type == 'note_on':
        # data1 = note number, data2 = velocity
        note_name = midi_note_to_name(data1)
        return f"{device_key} ch: {channel}  note: {note_name} ({data1})  v: {data2}"
    
    return None

