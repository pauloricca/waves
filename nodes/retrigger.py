from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.panning import apply_panning
from nodes.wavable_value import WavableValue
from utils import add_waves, empty_mono, empty_stereo, to_stereo

"""
Retrigger Node

Re-triggers a child signal at a specified interval. Each retrigger starts the child signal
from the beginning, overlaying it on top of previous triggers still playing out.

Parameters:
- signal: The signal to retrigger
- interval: Time in seconds between retriggers (can be a WavableValue for dynamic timing)
- duration: Total duration of the retrigger node output in seconds

Example:
retrigger:
  signal:
    osc:
      type: sin
      freq: 440
  interval: 0.5    # Retrigger every 0.5 seconds
  duration: 4.0    # Run for 4 seconds total
"""

class RetriggerModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    time: float = 0.1
    repeat: int = 3
    feedback: float = 0.3
    spread: float = 0.0  # Stereo spread: 0 = all centered, 1 = full stereo spread
    movement: float = 0.0  # How quickly retriggers move in stereo field (cycles per second)
    signal: WavableValue = None

class RetriggerNode(BaseNode):
    def __init__(self, model: RetriggerModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Persistent state for carry over samples (survives hot reload)
        if do_initialise_state:
            self.state.carry_over = np.array([], dtype=np.float32)
            self.state.carry_over_start_time = 0.0  # Track when carry_over samples should play

    def _do_render(self, num_samples=None, context=None, **params):
        """
        Render retriggered signal with optional stereo spread and movement.
        
        When spread > 0 or movement > 0:
        - Each retrigger is positioned in the stereo field
        - Position is determined by spread (0-1) and movement (rotation speed)
        - Returns 2D array of shape (num_samples, 2)
        
        Otherwise returns mono (1D array).
        """
        # Check if we need stereo output (based on whether stereo features are enabled)
        do_stereo_output = (self.model.spread > 0 or self.model.movement > 0)
        
        # If num_samples is None, we need to render the full signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # For retrigger nodes, we need to get the full child signal first
                child_signal = self.render_full_child_signal(self.signal_node, context,  **self.get_params_for_children(params))
                if len(child_signal) == 0:
                    if do_stereo_output:
                        return empty_stereo()
                    return empty_mono()
                
                # Calculate total retrigger time and set num_samples accordingly
                n_delay_time_samples = int(SAMPLE_RATE * self.model.time)
                total_length = len(child_signal) + n_delay_time_samples * self.model.repeat
                self._last_chunk_samples = total_length
                
                # Process the full signal at once
                if do_stereo_output:
                    delayed_wave = np.zeros((total_length, 2), dtype=np.float32)
                    for i in range(self.model.repeat):
                        delay_time = i * self.model.time
                        start_idx = i * n_delay_time_samples
                        end_idx = start_idx + len(child_signal)
                        
                        # For movement, calculate pan for each sample
                        if self.model.movement > 0:
                            # Create time array for each sample in this retrigger
                            sample_times = delay_time + np.arange(len(child_signal)) / SAMPLE_RATE
                            
                            # Vectorized pan calculation
                            if self.model.repeat == 1:
                                base_pan = 0.0
                            else:
                                normalized_pos = (i / (self.model.repeat - 1)) * 2 - 1
                                base_pan = normalized_pos * self.model.spread
                            
                            # Add movement rotation
                            rotation = np.sin(2 * np.pi * self.model.movement * sample_times)
                            pans = np.clip(base_pan + rotation * self.model.spread, -1, 1)
                            
                            # Apply time-varying panning
                            attenuated_signal = child_signal * (self.model.feedback ** i)
                            left_gains = np.cos((pans + 1) * np.pi / 4)
                            right_gains = np.sin((pans + 1) * np.pi / 4)
                            stereo_signal = np.column_stack([
                                attenuated_signal * left_gains,
                                attenuated_signal * right_gains
                            ])
                        else:
                            # Static pan
                            pan = self._calculate_pan_for_repeat(i, delay_time)
                            stereo_signal = apply_panning(child_signal * (self.model.feedback ** i), pan)
                        
                        delayed_wave[start_idx:end_idx] += stereo_signal
                else:
                    delayed_wave = np.zeros(total_length)
                    for i in range(self.model.repeat):
                        delayed_wave[i * n_delay_time_samples : i * n_delay_time_samples + len(child_signal)] += child_signal * (self.model.feedback ** i)
                return delayed_wave
        
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # If signal is done and we have no carry over, we're done
        if len(signal_wave) == 0 and len(self.state.carry_over) == 0:
            if do_stereo_output:
                return empty_stereo()
            return empty_mono()
        
        n_delay_time_samples = int(SAMPLE_RATE * self.model.time)
        
        if do_stereo_output:
            delayed_wave = np.zeros((len(signal_wave) + n_delay_time_samples * self.model.repeat, 2), dtype=np.float32)
            
            # Add delays with panning (only if we have signal)
            if len(signal_wave) > 0:
                for i in range(self.model.repeat):
                    # Calculate pan position for each sample based on when it will be heard
                    # Each retrigger starts at time_since_start + (i * delay_time)
                    delay_time = i * self.model.time
                    
                    # For movement, we need to calculate pan for each sample individually
                    if self.model.movement > 0:
                        # Create time array for each sample in this retrigger
                        sample_times = self.time_since_start + delay_time + np.arange(len(signal_wave)) / SAMPLE_RATE
                        
                        # Vectorized pan calculation
                        if self.model.repeat == 1:
                            base_pan = 0.0
                        else:
                            normalized_pos = (i / (self.model.repeat - 1)) * 2 - 1
                            base_pan = normalized_pos * self.model.spread
                        
                        # Add movement rotation
                        rotation = np.sin(2 * np.pi * self.model.movement * sample_times)
                        pans = np.clip(base_pan + rotation * self.model.spread, -1, 1)
                        
                        # Apply time-varying panning using vectorized operations
                        attenuated_signal = signal_wave * (self.model.feedback ** i)
                        # Calculate left and right channels with vectorized panning
                        left_gains = np.cos((pans + 1) * np.pi / 4)
                        right_gains = np.sin((pans + 1) * np.pi / 4)
                        stereo_signal = np.column_stack([
                            attenuated_signal * left_gains,
                            attenuated_signal * right_gains
                        ])
                    else:
                        # Static pan - calculate once for the whole retrigger
                        pan = self._calculate_pan_for_repeat(i, self.time_since_start + delay_time)
                        stereo_signal = apply_panning(signal_wave * (self.model.feedback ** i), pan)
                    
                    start_idx = i * n_delay_time_samples
                    end_idx = start_idx + len(signal_wave)
                    delayed_wave[start_idx:end_idx] += stereo_signal
        else:
            delayed_wave = np.zeros(len(signal_wave) + n_delay_time_samples * self.model.repeat)
            
            # Add delays (only if we have signal)
            if len(signal_wave) > 0:
                for i in range(self.model.repeat):
                    delayed_wave[i * n_delay_time_samples : i * n_delay_time_samples + len(signal_wave)] += signal_wave * (self.model.feedback ** i)
        
        # Add carry over from previous render
        if len(self.state.carry_over) > 0:
            if do_stereo_output:
                # Ensure carry_over is stereo
                self.state.carry_over = to_stereo(self.state.carry_over)
                delayed_wave[:len(self.state.carry_over)] += self.state.carry_over[:len(delayed_wave)]
            else:
                delayed_wave = add_waves(delayed_wave, self.state.carry_over[:len(delayed_wave)])

        # Return n_samples and save the rest as carry over for the next render
        part_to_return = delayed_wave[:num_samples]
        self.state.carry_over = delayed_wave[num_samples:]
        
        return part_to_return
    
    def _calculate_pan_for_repeat(self, repeat_index: int, time: float) -> float:
        """
        Calculate pan position for a given retrigger repeat.
        
        Args:
            repeat_index: Index of the repeat (0, 1, 2, ...)
            time: Current time in seconds (for movement)
            
        Returns:
            Pan value from -1 to 1
        """
        if self.model.spread == 0 and self.model.movement == 0:
            return 0.0
        
        # Base position: spread repeats evenly across stereo field
        # For even distribution: map repeat_index to range based on spread
        if self.model.repeat == 1:
            base_pan = 0.0
        else:
            # Map repeat index to [-1, 1] range, scaled by spread
            normalized_pos = (repeat_index / (self.model.repeat - 1)) * 2 - 1  # -1 to 1
            base_pan = normalized_pos * self.model.spread
        
        # Add movement: rotate position over time
        if self.model.movement > 0:
            # Movement creates a circular rotation in the stereo field
            rotation = np.sin(2 * np.pi * self.model.movement * time)
            base_pan = np.clip(base_pan + rotation * self.model.spread, -1, 1)
        
        return base_pan

RETRIGGER_DEFINITION = NodeDefinition("retrigger", RetriggerNode, RetriggerModel)
