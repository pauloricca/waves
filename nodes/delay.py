from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.oscillator import OSCILLATOR_RENDER_ARGS
from nodes.wavable_value import WavableValue, wavable_value_node_factory

class DelayModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    time: WavableValue  # Delay time in seconds
    signal: BaseNodeModel = None

class DelayNode(BaseNode):
    def __init__(self, model: DelayModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.time_node = wavable_value_node_factory(model.time)
        self.signal_node = instantiate_node(model.signal)
        
        # Circular buffer for storing delayed samples
        # We allocate enough space for the maximum delay time we might need
        # Using 10 seconds as a reasonable maximum
        max_delay_samples = int(10 * SAMPLE_RATE)
        self.buffer_size = max_delay_samples
        
        # Single buffer and write position (like a physical delay unit)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_position = 0

    def _do_render(self, num_samples=None, context=None, **params):
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
                delay_times = self.time_node.render(num_samples, context, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
                return self._apply_delay(signal_wave, delay_times, num_samples)
        
        # CRITICAL: Snapshot buffer state BEFORE rendering input signal
        # This ensures all recursion depths see the same initial buffer state
        buffer_snapshot = self.buffer.copy()
        write_position_snapshot = self.write_position
        
        # Now render the input signal (which may trigger recursive calls)
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # Restore buffer state for our read/write operations
        self.buffer = buffer_snapshot
        self.write_position = write_position_snapshot
        
        # If signal is done, we're done (no tail for pure delay)
        if len(signal_wave) == 0:
            return np.array([], dtype=np.float32)
        
        # Get delay times
        delay_times = self.time_node.render(num_samples, context, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
        
        return self._apply_delay(signal_wave, delay_times, num_samples)
    
    def _apply_delay(self, signal_wave, delay_times, num_samples):
        """Apply delay to the signal wave using a circular buffer"""
        output = np.zeros(num_samples, dtype=np.float32)
        
        # Handle both constant and time-varying delay times
        if len(delay_times) == 1:
            # Constant delay time - vectorized operations
            delay_samples = int(delay_times[0] * SAMPLE_RATE)
            delay_samples = np.clip(delay_samples, 0, self.buffer_size - 1)
            
            # Calculate all positions at once
            write_positions = (self.write_position + np.arange(num_samples)) % self.buffer_size
            read_positions = (write_positions - delay_samples) % self.buffer_size
            
            # CRITICAL: Read BEFORE write (like physical hardware)
            output = self.buffer[read_positions].copy()
            
            # Now write the input
            self.buffer[write_positions] = signal_wave
            
            # Update write position for next render
            self.write_position = (self.write_position + num_samples) % self.buffer_size
        else:
            # Time-varying delay - requires interpolation
            delay_samples_array = delay_times * SAMPLE_RATE
            delay_samples_array = np.clip(delay_samples_array, 0, self.buffer_size - 1)
            
            # Calculate write positions
            write_positions = (self.write_position + np.arange(num_samples)) % self.buffer_size
            
            # For fractional delays, use linear interpolation
            delay_samples_int = delay_samples_array.astype(int)
            delay_samples_frac = delay_samples_array - delay_samples_int
            
            # Calculate read positions for interpolation
            read_positions_1 = (write_positions - delay_samples_int) % self.buffer_size
            read_positions_2 = (write_positions - delay_samples_int - 1) % self.buffer_size
            
            # CRITICAL: Read BEFORE write
            sample_1 = self.buffer[read_positions_1]
            sample_2 = self.buffer[read_positions_2]
            output = sample_1 * (1 - delay_samples_frac) + sample_2 * delay_samples_frac
            
            # Now write the input
            self.buffer[write_positions] = signal_wave
            
            # Update write position for next render
            self.write_position = (self.write_position + num_samples) % self.buffer_size
        
        return output

DELAY_DEFINITION = NodeDefinition("delay", DelayNode, DelayModel)
