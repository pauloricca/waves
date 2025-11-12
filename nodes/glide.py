from __future__ import annotations

import numpy as np
from pydantic import ConfigDict

from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono, is_stereo


class GlideModel(BaseNodeModel):
    model_config = ConfigDict(extra="forbid")
    signal: WavableValue
    time: WavableValue = 0.0


class GlideNode(BaseNode):
    def __init__(self, model: GlideModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.signal_node = self.instantiate_child_node(model.signal, "signal")

        self.time_is_static = self._is_static_value(model.time)
        if self.time_is_static:
            self.time_node = None
            self.static_time = float(model.time)
        else:
            self.time_node = self.instantiate_child_node(model.time, "time")
            self.static_time = 0.0

        if do_initialise_state:
            self.state.current_value = None
            self.state.target_value = None
            self.state.samples_remaining = 0
            self.state.increment = None

    def _is_static_value(self, value: WavableValue) -> bool:
        return isinstance(value, (int, float))

    def _do_render(self, num_samples=None, context=None, **params):
        num_samples = self.resolve_num_samples(num_samples)
        child_params = self.get_params_for_children(params)

        if num_samples is None:
            signal_wave = self.render_full_child_signal(self.signal_node, context, **child_params)
            if len(signal_wave) == 0:
                return empty_mono()
            num_samples = len(signal_wave) if not is_stereo(signal_wave) else signal_wave.shape[0]
        else:
            signal_wave = self.signal_node.render(num_samples, context, **child_params)
            if len(signal_wave) == 0:
                return empty_mono()

        actual_num_samples = signal_wave.shape[0] if is_stereo(signal_wave) else len(signal_wave)

        if not self.time_is_static:
            time_wave = self.time_node.render(actual_num_samples, context, **child_params)
            if len(time_wave) == 0:
                time_values = np.zeros(actual_num_samples, dtype=np.float32)
            else:
                time_values = np.asarray(time_wave, dtype=np.float32)
                if time_values.ndim > 1:
                    time_values = time_values.reshape(-1)
                if time_values.size < actual_num_samples:
                    last_val = time_values[-1] if time_values.size > 0 else 0.0
                    pad_width = actual_num_samples - time_values.size
                    time_values = np.pad(time_values, (0, pad_width), mode="constant", constant_values=last_val)
        else:
            time_values = None

        signal_array = np.asarray(signal_wave, dtype=np.float32)
        if not is_stereo(signal_array):
            signal_array = signal_array.reshape(actual_num_samples, 1)
            mono_input = True
        else:
            mono_input = False

        if self.state.current_value is None:
            first_value = signal_array[0].copy()
            self.state.current_value = first_value.copy()
            self.state.target_value = first_value.copy()
            self.state.samples_remaining = 0
            self.state.increment = np.zeros_like(first_value, dtype=np.float32)

        current_value = self.state.current_value.astype(np.float32)
        target_value = self.state.target_value.astype(np.float32)
        increment = self.state.increment.astype(np.float32)
        samples_remaining = int(self.state.samples_remaining)

        output = np.empty_like(signal_array, dtype=np.float32)
        
        # Vectorized approach: process chunks where the target doesn't change
        i = 0
        while i < actual_num_samples:
            desired = signal_array[i]
            glide_time = self.static_time if self.time_is_static else float(time_values[i])
            if glide_time < 0:
                glide_time = 0.0

            # Check if we need to start a new glide
            if glide_time == 0 or glide_time <= (1.0 / SAMPLE_RATE):
                # Instant change - find how many consecutive samples have the same target and glide time
                if self.time_is_static:
                    # Find next change in signal
                    change_mask = np.any(signal_array[i:] != desired, axis=1) if signal_array.ndim > 1 else signal_array[i:] != desired
                    change_indices = np.where(change_mask)[0]
                    chunk_size = change_indices[0] if len(change_indices) > 0 else actual_num_samples - i
                else:
                    # For dynamic time, find consecutive samples with zero/instant glide time and same signal
                    remaining = actual_num_samples - i
                    zero_time_mask = (time_values[i:] <= (1.0 / SAMPLE_RATE))
                    signal_same_mask = np.all(signal_array[i:] == desired, axis=1) if signal_array.ndim > 1 else signal_array[i:] == desired
                    combined_mask = zero_time_mask & signal_same_mask
                    change_indices = np.where(~combined_mask)[0]
                    chunk_size = change_indices[0] if len(change_indices) > 0 else remaining
                
                output[i:i+chunk_size] = desired
                current_value = desired.copy()
                target_value = desired.copy()
                samples_remaining = 0
                increment = np.zeros_like(desired, dtype=np.float32)
                i += chunk_size
            else:
                # Check if we're starting a new glide
                if not np.allclose(desired, target_value, atol=1e-9):
                    target_value = desired.copy()
                    diff = target_value - current_value
                    steps = max(1, int(round(glide_time * SAMPLE_RATE)))
                    samples_remaining = steps
                    increment = diff / steps

                # Process glide in vectorized chunks
                if samples_remaining > 0:
                    # Find how many samples until signal changes or glide completes
                    samples_to_process = min(samples_remaining, actual_num_samples - i)
                    
                    # Check for signal changes in this chunk
                    change_mask = np.any(signal_array[i:i+samples_to_process] != desired, axis=1) if signal_array.ndim > 1 else signal_array[i:i+samples_to_process] != desired
                    change_indices = np.where(change_mask)[0]
                    if len(change_indices) > 0:
                        samples_to_process = min(samples_to_process, change_indices[0])
                    
                    # Generate ramp for this chunk
                    ramp = np.arange(samples_to_process, dtype=np.float32)
                    if signal_array.ndim > 1:
                        ramp = ramp[:, np.newaxis]
                    output[i:i+samples_to_process] = current_value + increment * (ramp + 1)
                    
                    current_value = output[i+samples_to_process-1].copy()
                    samples_remaining -= samples_to_process
                    
                    if samples_remaining == 0:
                        current_value = target_value.copy()
                    
                    i += samples_to_process
                else:
                    # Holding at target - find next change
                    if self.time_is_static:
                        change_mask = np.any(signal_array[i:] != desired, axis=1) if signal_array.ndim > 1 else signal_array[i:] != desired
                        change_indices = np.where(change_mask)[0]
                        chunk_size = change_indices[0] if len(change_indices) > 0 else actual_num_samples - i
                    else:
                        # For dynamic time, find consecutive samples with same signal value
                        signal_same_mask = np.all(signal_array[i:] == desired, axis=1) if signal_array.ndim > 1 else signal_array[i:] == desired
                        change_indices = np.where(~signal_same_mask)[0]
                        chunk_size = change_indices[0] if len(change_indices) > 0 else actual_num_samples - i
                    
                    output[i:i+chunk_size] = target_value
                    current_value = target_value.copy()
                    i += chunk_size

        self.state.current_value = current_value.copy()
        self.state.target_value = target_value.copy()
        self.state.samples_remaining = samples_remaining
        self.state.increment = increment.copy()

        if mono_input:
            return output.reshape(actual_num_samples)
        return output


GLIDE_DEFINITION = NodeDefinition("glide", GlideNode, GlideModel)
