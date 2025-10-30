from __future__ import annotations
from enum import Enum
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono, time_to_samples

class DelayMode(str, Enum):
    DIGITAL = "DIGITAL"
    TAPE = "TAPE"
    
    @classmethod
    def _missing_(cls, value):
        """Make enum case-insensitive"""
        if isinstance(value, str):
            value = value.upper()
            for member in cls:
                if member.value.upper() == value:
                    return member
        return None

class DelayModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    time: WavableValue  # Delay time in seconds
    mode: DelayMode = DelayMode.DIGITAL  # DIGITAL or TAPE mode
    signal: BaseNodeModel = None

class DelayNode(BaseNode):
    def __init__(self, model: DelayModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.time_node = self.instantiate_child_node(model.time, "time")
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Circular buffer for storing delayed samples
        # Calculate buffer size based on the delay time parameter
        # For scalar delays, use that value. For dynamic delays, use a safe maximum.
        if isinstance(model.time, (int, float)):
            # Scalar delay time - allocate exactly what we need (plus some headroom)
            max_delay_time = float(model.time) * 1.1  # 10% headroom
        else:
            # Dynamic delay (node or list) - use a generous default
            max_delay_time = 30.0  # 30 seconds should be enough for most cases
        
        max_delay_samples = time_to_samples(max_delay_time )
        self.buffer_size = max_delay_samples
        
        # Persistent state for buffer and positions (survives hot reload)
        if do_initialise_state:
            self.state.buffer = np.zeros(self.buffer_size, dtype=np.float32)
            self.state.write_position = 0
            self.state.read_position = 0.0  # Can be fractional for tape speed changes
            self.state.previous_delay_time = None  # Track delay time changes for tape mode
            self.state.input_finished = False
            self.state.samples_since_input_finished = 0
            # Tape mode only
            self.state.tape_head_distance = None  # For TAPE mode

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # If num_samples is None, get the full child signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Need to get full signal from child
                signal_wave = self.render_full_child_signal(self.signal_node, context, **self.get_params_for_children(params))
                if len(signal_wave) == 0:
                    return np.array([])
                
                num_samples = len(signal_wave)
                # Get delay times for the full signal
                delay_times = self.time_node.render(num_samples, context, **self.get_params_for_children(params))
                return self._apply_delay(signal_wave, delay_times, num_samples)
        
        # CRITICAL: Snapshot buffer state BEFORE rendering input signal
        # This ensures all recursion depths see the same initial buffer state
        buffer_snapshot = self.state.buffer.copy()
        write_position_snapshot = self.state.write_position
        read_position_snapshot = self.state.read_position
        previous_delay_time_snapshot = self.state.previous_delay_time
        
        # Now render the input signal (which may trigger recursive calls)
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # Restore buffer state for our read/write operations
        self.state.buffer = buffer_snapshot
        self.state.write_position = write_position_snapshot
        self.state.read_position = read_position_snapshot
        self.state.previous_delay_time = previous_delay_time_snapshot
        
        # Track when input signal finishes
        input_is_active = len(signal_wave) > 0
        
        # If signal returned fewer samples than requested, pad it with zeros
        # This allows the delay to continue reading from the buffer for the full chunk
        # Even if the input signal has finished, we want to output the delayed content
        if len(signal_wave) > 0 and len(signal_wave) < num_samples:
            signal_wave = np.pad(signal_wave, (0, num_samples - len(signal_wave)), mode='constant', constant_values=0)
        elif len(signal_wave) == 0:
            # Input signal is completely finished
            if not self.state.input_finished:
                self.state.input_finished = True
                self.state.samples_since_input_finished = 0
            
            # Check if we should stop (after delay time has passed since input finished)
            # Get the current delay time to know how long to continue
            delay_time_check = self.time_node.render(1, context, **self.get_params_for_children(params))[0]
            max_tail_samples = time_to_samples(delay_time_check )
            
            if self.state.samples_since_input_finished >= max_tail_samples:
                # We've output enough tail, stop now
                return empty_mono()
            
            # Create a silent input signal (all zeros) to allow reading from the buffer
            signal_wave = np.zeros(num_samples, dtype=np.float32)
            self.state.samples_since_input_finished += num_samples
        
        # Get delay times
        delay_times = self.time_node.render(num_samples, context, **self.get_params_for_children(params))
        
        return self._apply_delay(signal_wave, delay_times, num_samples)
    
    def _apply_delay(self, signal_wave, delay_times, num_samples):
        """Apply delay to the signal wave using a circular buffer"""
        
        if self.model.mode == DelayMode.DIGITAL:
            return self._apply_digital_delay(signal_wave, delay_times, num_samples)
        else:  # TAPE mode
            return self._apply_tape_delay(signal_wave, delay_times, num_samples)
    
    def _apply_digital_delay(self, signal_wave, delay_times, num_samples):
        """Digital delay mode - delay time changes instantly without pitch shift"""
        output = np.zeros(num_samples, dtype=np.float32)
        
        # Handle both constant and time-varying delay times
        if len(delay_times) == 1:
            # Constant delay time - vectorized operations
            delay_samples = time_to_samples(delay_times[0] )
            delay_samples = np.clip(delay_samples, 0, self.buffer_size - 1)
            
            # Calculate all positions at once
            write_positions = (self.state.write_position + np.arange(num_samples)) % self.buffer_size
            read_positions = (write_positions - delay_samples) % self.buffer_size
            
            # CRITICAL: Read BEFORE write (like physical hardware)
            output = self.state.buffer[read_positions].copy()
            
            # Now write the input
            self.state.buffer[write_positions] = signal_wave
            
            # Update write position for next render
            self.state.write_position = (self.state.write_position + num_samples) % self.buffer_size
        else:
            # Time-varying delay - requires interpolation
            delay_samples_array = delay_times * SAMPLE_RATE
            delay_samples_array = np.clip(delay_samples_array, 0, self.buffer_size - 1)
            
            # Calculate write positions
            write_positions = (self.state.write_position + np.arange(num_samples)) % self.buffer_size
            
            # For fractional delays, use linear interpolation
            delay_samples_int = delay_samples_array.astype(int)
            delay_samples_frac = delay_samples_array - delay_samples_int
            
            # Calculate read positions for interpolation
            read_positions_1 = (write_positions - delay_samples_int) % self.buffer_size
            read_positions_2 = (write_positions - delay_samples_int - 1) % self.buffer_size
            
            # CRITICAL: Read BEFORE write
            sample_1 = self.state.buffer[read_positions_1]
            sample_2 = self.state.buffer[read_positions_2]
            output = sample_1 * (1 - delay_samples_frac) + sample_2 * delay_samples_frac
            
            # Now write the input
            self.state.buffer[write_positions] = signal_wave
            
            # Update write position for next render
            self.state.write_position = (self.state.write_position + num_samples) % self.buffer_size
        
        return output
    
    def _apply_tape_delay(self, signal_wave, delay_times, num_samples):
        """Tape delay mode - simulates changing tape speed, affecting both read and write
        
        In a real tape delay:
        - The physical distance between record and playback heads is FIXED
        - Changing the delay time means changing the TAPE SPEED
        - Both read and write happen at the same speed (they're on the same tape)
        - Slower tape = longer delay, lower pitch
        - Faster tape = shorter delay, higher pitch
        """
        
        # Get the target delay time
        target_delay_time = delay_times[0] if len(delay_times) == 1 else np.mean(delay_times)
        target_delay_samples = np.clip(target_delay_time * SAMPLE_RATE, 1, self.buffer_size - 1)
        
        # Initialize if first call
        if self.state.previous_delay_time is None:
            self.state.previous_delay_time = target_delay_time
            # Fixed "physical" distance between heads (in samples)
            # This represents the actual physical distance on the tape
            self.state.tape_head_distance = target_delay_samples
            self.state.read_position = float((self.state.write_position - self.state.tape_head_distance) % self.buffer_size)
        
        # Calculate tape speed based on delay time change
        # If delay increases, tape must slow down (speed < 1.0)
        # If delay decreases, tape must speed up (speed > 1.0)
        # The relationship: delay_time = head_distance / tape_speed
        # Therefore: tape_speed = head_distance / delay_time
        tape_speed = self.state.tape_head_distance / target_delay_samples
        
        # No clamping - allow extreme values for experimental effects
        
        # BOTH read and write at the tape speed
        # Generate positions for reading
        read_increments = np.full(num_samples, tape_speed)
        read_positions = self.state.read_position + np.concatenate([[0], np.cumsum(read_increments[:-1])])
        
        # Wrap and interpolate for reading
        read_positions_wrapped = read_positions % self.buffer_size
        read_positions_int = read_positions_wrapped.astype(int)
        read_positions_frac = read_positions_wrapped - read_positions_int
        
        # Cubic (Hermite) interpolation for better quality at variable speeds
        # Get 4 points: previous, current, next, next+1
        pos_prev = (read_positions_int - 1) % self.buffer_size
        pos_curr = read_positions_int
        pos_next = (read_positions_int + 1) % self.buffer_size
        pos_next2 = (read_positions_int + 2) % self.buffer_size
        
        y0 = self.state.buffer[pos_prev]
        y1 = self.state.buffer[pos_curr]
        y2 = self.state.buffer[pos_next]
        y3 = self.state.buffer[pos_next2]
        
        # Hermite interpolation (4-point, 3rd order)
        # Provides smooth interpolation with no overshoot
        t = read_positions_frac
        c0 = y1
        c1 = 0.5 * (y2 - y0)
        c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
        
        output = c0 + c1 * t + c2 * t * t + c3 * t * t * t
        
        # Write at the same tape speed
        # When tape is slow (speed < 1), we write fewer samples to the buffer
        # When tape is fast (speed > 1), we write more samples (with interpolation)
        write_increments = np.full(num_samples, tape_speed)
        write_positions = self.state.write_position + np.concatenate([[0], np.cumsum(write_increments[:-1])])
        write_positions_wrapped = write_positions % self.buffer_size
        write_positions_int = write_positions_wrapped.astype(int)
        
        # Write input signal to buffer at tape speed
        self.state.buffer[write_positions_int] = signal_wave
        
        # Update positions for next render
        total_tape_advance = num_samples * tape_speed
        self.state.write_position = (self.state.write_position + total_tape_advance) % self.buffer_size
        self.state.read_position = (self.state.read_position + total_tape_advance) % self.buffer_size
        self.state.previous_delay_time = target_delay_time
        
        return output

DELAY_DEFINITION = NodeDefinition("delay", DelayNode, DelayModel)
