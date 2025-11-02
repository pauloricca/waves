from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import load_wav_file, empty_mono, time_to_samples

# Buffer Node - Flexible read/write buffer with multiple access modes
#
# The buffer node provides a flexible circular buffer that can be written to and read from
# in various ways, similar to a combination of the delay and sample nodes but with more flexibility.
#
# KEY FEATURES:
# - Named buffers: Multiple buffer nodes can share the same buffer via the 'name' parameter
# - Two read modes: offset-based (delay-like) or position-based (sample-like)
# - File loading: Can load audio files into the buffer
# - Speed control: Read/write speed can be modulated
# - Global buffer registry: Buffers persist across node instances
#
# PARAMETERS:
# - name: Buffer identifier (defaults to node_id). Nodes with the same name share a buffer.
# - length: Buffer length in seconds
# - file: Optional audio file to load into buffer
# - signal: Optional input signal to write to buffer
#
# READ MODE 1 - Offset-based (delay-like):
# - offset: How far back (in seconds) to read from the write head
# - Works like a delay line - good for echo effects, feedback, etc.
#
# READ MODE 2 - Position-based (sample-like):
# - start/end: Normalized positions (0.0-1.0) in the buffer
# - loop: Whether to loop when reaching the end
# - overlap: Crossfade amount for looping (0.0-1.0)
# - base_freq/freq: Frequency modulation (modulates speed)
# - Works like a sample player - good for granular effects, scrubbing, etc.
#
# SPEED CONTROL (both modes):
# - speed: Playback/write speed multiplier
#   - speed > 1: Faster (compacted)
#   - speed < 1: Slower (interpolated)
#   - speed = 1: Normal speed
#
# USAGE EXAMPLES:
#   Simple delay:
#     buffer:
#       offset: 0.3
#       signal:
#         osc:
#           type: sin
#           freq: 440
#
#   Shared buffer (write and read separately):
#     mix:
#       signals:
#         - buffer:           # Writer
#             name: A
#             signal:
#               osc: {...}
#         - buffer:           # Reader 1
#             name: A
#             offset: 0.1
#         - buffer:           # Reader 2
#             name: A
#             offset: 0.5
#
#   Sample playback from buffer:
#     buffer:
#       file: samples/kick.wav
#       start: 0.0
#       end: 0.5
#       loop: true
#       speed: 1.5

# Global registry of shared buffers
# Key: buffer name (str), Value: dict with 'data' (numpy array) and 'write_head' (int)
_GLOBAL_BUFFERS = {}


def get_or_create_buffer(buffer_name: str, length_samples: int):
    """Get an existing buffer or create a new one with the specified length"""
    if buffer_name not in _GLOBAL_BUFFERS:
        _GLOBAL_BUFFERS[buffer_name] = {
            'data': np.zeros(length_samples, dtype=np.float32),
            'write_head': 0
        }
    return _GLOBAL_BUFFERS[buffer_name]


class BufferModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    
    # Buffer identification
    name: str = None  # Buffer name (defaults to node_id if not specified)
    
    # Buffer initialization
    length: float = 10.0  # Buffer length in seconds
    file: str = None  # Optional: load audio file into buffer
    
    # Write parameters
    signal: BaseNodeModel = None  # Optional: signal to write to buffer
    
    # Read parameters - two modes:
    # Mode 1: Offset-based (relative to write head)
    offset: WavableValue = 0.0  # seconds - how far back from write head to read
    
    # Mode 2: Absolute position-based (like sample node)
    start: WavableValue = None  # 0.0-1.0 (normalized position in buffer)
    end: WavableValue = None  # 0.0-1.0 (normalized position in buffer)
    loop: bool = False
    overlap: float = 0.0  # 0.0-1.0 (normalized fraction of buffer length)
    base_freq: WavableValue = None  # Hz (base frequency for freq parameter)
    freq: WavableValue = None  # Hz (target frequency - modulates speed based on base_freq)
    
    # Speed control (applies to both read and write)
    speed: WavableValue = 1.0  # Playback/write speed multiplier


class BufferNode(BaseNode):
    def __init__(self, model: BufferModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        
        # Determine buffer name (use explicit name or fall back to node_id)
        self.buffer_name = model.name if model.name is not None else node_id
        
        # Check if this node will be writing to the buffer
        is_writer = model.signal is not None or model.file is not None
        
        if is_writer:
            # Writer node: create buffer with specified length if it doesn't exist
            requested_buffer_length = time_to_samples(model.length)
            self.buffer_ref = get_or_create_buffer(self.buffer_name, requested_buffer_length)
        else:
            # Reader node: use existing buffer (or create a minimal one if it doesn't exist yet)
            # This allows read-only nodes to work without needing to specify length
            if self.buffer_name in _GLOBAL_BUFFERS:
                self.buffer_ref = _GLOBAL_BUFFERS[self.buffer_name]
            else:
                # Buffer doesn't exist yet - create a default one
                # (This shouldn't normally happen, but handle gracefully)
                default_length = time_to_samples(model.length)
                self.buffer_ref = get_or_create_buffer(self.buffer_name, default_length)
        
        # IMPORTANT: Always use the actual buffer size from the shared buffer
        self.buffer_length_samples = len(self.buffer_ref['data'])
        
        # If we have a file, load it all at once into the buffer
        # and set write_head to start position (ready to "play" from the beginning)
        if model.file is not None:
            audio_data = load_wav_file(model.file)
            file_length = len(audio_data)
            
            # Copy file data into buffer (with wrapping if needed)
            for i in range(min(file_length, self.buffer_length_samples)):
                self.buffer_ref['data'][i] = audio_data[i]
            
            # Set write_head to 0 (start of buffer)
            # The first read with offset=0 will advance it by num_samples
            self.buffer_ref['write_head'] = 0
        
        # Instantiate child nodes
        self.signal_node = self.instantiate_child_node(model.signal, "signal") if model.signal is not None else None
        self.speed_node = self.instantiate_child_node(model.speed, "speed")
        self.offset_node = self.instantiate_child_node(model.offset, "offset")
        
        # Position-based mode nodes
        self.start_node = self.instantiate_child_node(model.start, "start") if model.start is not None else None
        self.end_node = self.instantiate_child_node(model.end, "end") if model.end is not None else None
        self.freq_node = self.instantiate_child_node(model.freq, "freq") if model.freq is not None else None
        self.base_freq_node = self.instantiate_child_node(model.base_freq, "base_freq") if model.base_freq is not None else None
        
        # Determine read mode
        self.is_position_mode = self.start_node is not None or self.end_node is not None
        
        # Persistent state for playback (only for position mode)
        if do_initialise_state:
            self.state.last_playhead_position = 0.0
            self.state.total_samples_rendered = 0
    
    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Check if we've exceeded duration limit
        if self.model.duration is not None:
            max_total_samples = time_to_samples(self.model.duration)
            if self.state.total_samples_rendered >= max_total_samples:
                return empty_mono()
        
        # If num_samples is None, resolve it based on mode
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                if self.is_position_mode and not self.model.loop:
                    # Non-looping position mode: determine length from start/end
                    start_vals = self.start_node.render(1, context, **self.get_params_for_children(params)) if self.start_node else np.array([0.0])
                    end_vals = self.end_node.render(1, context, **self.get_params_for_children(params)) if self.end_node else np.array([1.0])
                    start = int(start_vals[0] * self.buffer_length_samples)
                    end = int(end_vals[0] * self.buffer_length_samples)
                    end = min(end, self.buffer_length_samples)
                    start = max(start, 0)
                    buffer_segment_length = end - start
                    if buffer_segment_length <= 0:
                        return empty_mono()
                    num_samples = buffer_segment_length
                else:
                    raise ValueError("Cannot render full signal: buffer requires duration to be specified in loop or offset mode")
        
        # Clip num_samples if it would exceed duration
        if self.model.duration is not None:
            max_total_samples = time_to_samples(self.model.duration)
            samples_remaining = max_total_samples - self.state.total_samples_rendered
            if num_samples > samples_remaining:
                num_samples = samples_remaining
        
        # Snapshot buffer state before any operations (for recursion safety)
        buffer_snapshot = self.buffer_ref['data'].copy()
        write_head_snapshot = self.buffer_ref['write_head']
        
        # Write to buffer if signal is provided
        # IMPORTANT: Write head always advances, even if signal ends
        if self.signal_node is not None:
            signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
            
            # Restore buffer state after rendering signal (in case of recursion)
            self.buffer_ref['data'] = buffer_snapshot.copy()
            self.buffer_ref['write_head'] = write_head_snapshot
            
            # Pad signal with zeros if it ended early (so write head still advances)
            if len(signal_wave) < num_samples:
                signal_wave = np.pad(signal_wave, (0, num_samples - len(signal_wave)), mode='constant', constant_values=0)
            
            # Write signal to buffer (always writes num_samples, advancing write head)
            self._write_to_buffer(signal_wave, num_samples, context, params)
            # Update snapshot after write
            buffer_snapshot = self.buffer_ref['data'].copy()
            write_head_snapshot = self.buffer_ref['write_head']
        
        # Read from buffer
        if self.is_position_mode:
            result = self._read_position_mode(num_samples, context, params)
        else:
            # For offset mode without a signal input, advance write_head BEFORE reading
            # This ensures file-loaded buffers start from the beginning
            if self.signal_node is None:
                speed = self.speed_node.render(num_samples, context, **self.get_params_for_children(params))
                if np.isscalar(speed):
                    advance_amount = speed * num_samples
                else:
                    advance_amount = np.sum(speed[:num_samples])
                self.buffer_ref['write_head'] = (self.buffer_ref['write_head'] + advance_amount) % self.buffer_length_samples
            
            result = self._read_offset_mode(num_samples, context, params)
        
        # Update total samples rendered
        self.state.total_samples_rendered += len(result)


        
        return result
    
    def _write_to_buffer(self, signal_wave, num_samples, context, params):
        """Write signal to buffer at current write head, with speed control"""
        speed = self.speed_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # Ensure speed is an array
        if np.isscalar(speed):
            speed = np.full(num_samples, speed)
        
        # Pad signal if needed
        if len(signal_wave) < num_samples:
            signal_wave = np.pad(signal_wave, (0, num_samples - len(signal_wave)), mode='constant', constant_values=0)
        
        # Calculate write positions based on speed
        # Speed > 1: compact signal (write faster, fewer samples stored)
        # Speed < 1: expand signal (write slower, interpolate)
        # Speed = 1: normal write
        
        write_head = self.buffer_ref['write_head']
        
        if np.allclose(speed, 1.0):
            # Simple case: speed is 1, direct write
            for i, sample_value in enumerate(signal_wave):
                write_pos = int((write_head + i) % self.buffer_length_samples)
                self.buffer_ref['data'][write_pos] = sample_value
            self.buffer_ref['write_head'] = (write_head + len(signal_wave)) % self.buffer_length_samples
        else:
            # Variable speed write
            # Calculate cumulative positions
            position_deltas = speed  # Each output sample advances the write position by speed
            cumulative_positions = np.cumsum(position_deltas)
            write_positions = (write_head + cumulative_positions) % self.buffer_length_samples
            
            # For each integer position in the buffer, we need to write a value
            # When speed > 1, multiple input samples map to the same buffer position (downsampling)
            # When speed < 1, input samples are spread across multiple buffer positions (interpolation)
            
            # Calculate which buffer positions we're writing to
            start_pos_int = int(write_head)
            end_pos_int = int(write_head + cumulative_positions[-1])
            
            # Handle wrapping
            if end_pos_int >= self.buffer_length_samples:
                # We wrap around - handle in two parts
                num_positions = int(cumulative_positions[-1])
                positions_to_write = [(write_head + i) % self.buffer_length_samples for i in range(num_positions)]
            else:
                positions_to_write = range(start_pos_int, end_pos_int + 1)
            
            # For each buffer position, interpolate the signal value
            for buf_pos in positions_to_write:
                # Find where this buffer position falls in the cumulative_positions
                relative_pos = (buf_pos - write_head) % self.buffer_length_samples
                # Find the signal sample index that corresponds to this buffer position
                signal_idx = np.searchsorted(cumulative_positions, relative_pos)
                if signal_idx >= len(signal_wave):
                    signal_idx = len(signal_wave) - 1
                # Simple nearest-neighbor for now (could use interpolation)
                self.buffer_ref['data'][int(buf_pos)] = signal_wave[signal_idx]
            
            # Update write head
            self.buffer_ref['write_head'] = (write_head + cumulative_positions[-1]) % self.buffer_length_samples
    
    def _read_offset_mode(self, num_samples, context, params):
        """Read from buffer using offset (relative to write head)
        
        The offset represents how far BACK in time to read from the buffer.
        offset=0 means "read the most recently written data", which is the data
        that's already in the buffer at the current write_head position (from previous chunks).
        
        Important: We read from write_head position, which points to where the NEXT write
        will happen. So reading at write_head gives us the oldest data in a circular buffer,
        or the data written num_samples ago in the previous cycle.
        """
        offset = self.offset_node.render(num_samples, context, **self.get_params_for_children(params))
        speed = self.speed_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # Ensure they are arrays
        if np.isscalar(offset):
            offset = np.full(num_samples, offset)
        if np.isscalar(speed):
            speed = np.full(num_samples, speed)
        
        # Convert offset from seconds to samples
        offset_samples = offset * SAMPLE_RATE
        
        # Clip offset to valid range (can't read further back than buffer length)
        offset_samples = np.clip(offset_samples, 0, self.buffer_length_samples - 1)
        
        # Current write head position
        write_head = self.buffer_ref['write_head']
        
        output = np.zeros(num_samples, dtype=np.float32)
        
        # For offset mode with speed, we calculate read positions similar to delay node
        # We're reading backwards from the write head by the offset amount
        
        if np.allclose(speed, 1.0) and np.allclose(offset_samples, offset_samples[0]):
            # Constant speed and offset - vectorized approach
            delay_samples = int(offset_samples[0])
            
            # Read from write_head going backwards by (offset + num_samples)
            # write_head points to where we'll write next
            # offset=0 means read the chunk that was just written (write_head - num_samples to write_head)
            # offset=1s means read 1 second further back
            read_start = (write_head - delay_samples - num_samples) % self.buffer_length_samples
            read_positions = ((read_start + np.arange(num_samples)) % self.buffer_length_samples).astype(int)
            
            # Read from buffer
            output = self.buffer_ref['data'][read_positions].copy()
        else:
            # Variable speed or offset - per-sample processing
            for i in range(num_samples):
                # Calculate how many samples we've "virtually" advanced based on speed
                if i == 0:
                    virtual_advance = 0
                else:
                    virtual_advance = np.sum(speed[:i])
                
                # Read position formula:
                # Start from write_head - num_samples (beginning of chunk that was just written)
                # Go back by offset_samples
                # Advance forward by virtual_advance (for speed control)
                read_pos = write_head - num_samples - offset_samples[i] + virtual_advance
                
                # Wrap around buffer
                read_pos = read_pos % self.buffer_length_samples
                
                # Interpolate
                read_pos_int = int(read_pos)
                read_pos_frac = read_pos - read_pos_int
                
                # Linear interpolation
                sample1 = self.buffer_ref['data'][read_pos_int]
                sample2 = self.buffer_ref['data'][(read_pos_int + 1) % self.buffer_length_samples]
                output[i] = sample1 * (1 - read_pos_frac) + sample2 * read_pos_frac
        
        return output
    
    def _read_position_mode(self, num_samples, context, params):
        """Read from buffer using start/end positions (like sample node)"""
        # Get start/end values
        if self.start_node is not None:
            start_vals = self.start_node.render(num_samples, context, **self.get_params_for_children(params))
        else:
            start_vals = np.zeros(num_samples)
        
        if self.end_node is not None:
            end_vals = self.end_node.render(num_samples, context, **self.get_params_for_children(params))
        else:
            end_vals = np.ones(num_samples)
        
        # Ensure they are arrays
        if np.isscalar(start_vals):
            start_vals = np.full(num_samples, start_vals)
        if np.isscalar(end_vals):
            end_vals = np.full(num_samples, end_vals)
        
        # Convert to buffer indices
        start_indices = np.clip(start_vals * self.buffer_length_samples, 0, self.buffer_length_samples - 1).astype(int)
        end_indices = np.clip(end_vals * self.buffer_length_samples, 0, self.buffer_length_samples).astype(int)
        
        # Initialize playhead on first render
        if self.number_of_chunks_rendered == 0 and self.state.last_playhead_position == 0:
            self.state.last_playhead_position = float(start_indices[0])
        
        # Get speed
        speed = self.speed_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # Ensure speed is an array
        if np.isscalar(speed):
            speed = np.full(num_samples, speed)
        
        # If freq is set and base_freq is set, modulate speed by frequency ratio
        if self.freq_node is not None and self.base_freq_node is not None:
            freq = self.freq_node.render(num_samples, context, **self.get_params_for_children(params))
            base_freq = self.base_freq_node.render(num_samples, context, **self.get_params_for_children(params))
            # Ensure both are arrays
            if np.isscalar(freq):
                freq = np.full(num_samples, freq)
            if np.isscalar(base_freq):
                base_freq = np.full(num_samples, base_freq)
            # Avoid division by zero
            base_freq = np.maximum(base_freq, 1e-6)
            # Multiply speed by the frequency ratio
            speed = speed * (freq / base_freq)
        
        # Calculate playhead movement
        abs_speed = np.abs(speed)
        abs_speed = np.maximum(abs_speed, 1e-6)  # avoid zero speed
        sign = np.sign(speed)
        dt = 1.0 / SAMPLE_RATE
        
        # Calculate window lengths
        window_lengths = end_indices - start_indices
        window_lengths = np.maximum(window_lengths, 1)
        
        # Integrate speed to get playhead position
        playhead_delta = abs_speed * sign * dt * SAMPLE_RATE
        playhead_absolute = self.state.last_playhead_position + np.cumsum(playhead_delta)
        
        if not self.model.loop:
            # Non-looping: check bounds
            outside_bounds = (playhead_absolute >= end_indices) | (playhead_absolute < start_indices)
            if np.any(outside_bounds):
                first_outside = np.where(outside_bounds)[0][0]
                if first_outside == 0:
                    return empty_mono()
                # Truncate
                num_samples = first_outside
                playhead_absolute = playhead_absolute[:num_samples]
                start_indices = start_indices[:num_samples]
                end_indices = end_indices[:num_samples]
            buffer_indices = playhead_absolute
        else:
            # Looping: wrap around
            outside_high = playhead_absolute >= end_indices
            outside_low = playhead_absolute < start_indices
            outside = outside_high | outside_low
            if np.any(outside):
                offset_from_start = playhead_absolute - start_indices
                wrapped_offset = np.mod(offset_from_start, window_lengths)
                playhead_absolute = np.where(outside, start_indices + wrapped_offset, playhead_absolute)
            buffer_indices = playhead_absolute
        
        if num_samples == 0:
            return empty_mono()
        
        # Clip to valid buffer range
        buffer_indices = np.clip(buffer_indices, 0, self.buffer_length_samples - 1)
        
        # Interpolate from buffer
        wave = np.interp(buffer_indices, np.arange(self.buffer_length_samples), self.buffer_ref['data'])
        
        # Update state
        self.state.last_playhead_position = playhead_absolute[-1] if len(playhead_absolute) > 0 else self.state.last_playhead_position
        self.state.total_samples_rendered += len(wave)
        
        return wave


BUFFER_DEFINITION = NodeDefinition("buffer", BufferNode, BufferModel)
