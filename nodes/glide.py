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

        # Get glide time - use last value if dynamic
        if not self.time_is_static:
            time_wave = self.time_node.render(actual_num_samples, context, **child_params)
            if len(time_wave) == 0:
                glide_time = 0.0
            else:
                time_values = np.asarray(time_wave, dtype=np.float32)
                if time_values.ndim > 1:
                    time_values = time_values.reshape(-1)
                # Use the last value as the glide time for this chunk
                glide_time = float(time_values[-1]) if len(time_values) > 0 else 0.0
        else:
            glide_time = self.static_time

        if glide_time < 0:
            glide_time = 0.0

        signal_array = np.asarray(signal_wave, dtype=np.float32)
        if not is_stereo(signal_array):
            signal_array = signal_array.reshape(actual_num_samples, 1)
            mono_input = True
        else:
            mono_input = False

        # OPTIMIZATION: Use only the last value of the signal chunk as target
        # This dramatically improves performance when target changes frequently
        chunk_target = signal_array[-1].copy()

        if self.state.current_value is None:
            # Initialize with first value
            first_value = signal_array[0].copy()
            self.state.current_value = first_value.copy()
            self.state.target_value = first_value.copy()
            self.state.samples_remaining = 0
            self.state.increment = np.zeros_like(first_value, dtype=np.float32)

        current_value = self.state.current_value.astype(np.float32)
        samples_remaining = int(self.state.samples_remaining)
        
        output = np.empty_like(signal_array, dtype=np.float32)

        # Check if we need to start a new glide to the chunk target
        if not np.allclose(chunk_target, self.state.target_value, atol=1e-9):
            # New target detected
            if glide_time == 0 or glide_time <= (1.0 / SAMPLE_RATE):
                # Instant change - jump directly to target
                output[:] = chunk_target
                current_value = chunk_target.copy()
                self.state.target_value = chunk_target.copy()
                samples_remaining = 0
            else:
                # Start new glide to chunk target
                self.state.target_value = chunk_target.copy()
                diff = chunk_target - current_value
                steps = max(1, int(round(glide_time * SAMPLE_RATE)))
                samples_remaining = steps
                increment = diff / steps
                
                # Generate glide for this chunk
                samples_to_process = min(samples_remaining, actual_num_samples)
                ramp = np.arange(samples_to_process, dtype=np.float32)
                if signal_array.ndim > 1:
                    ramp = ramp[:, np.newaxis]
                
                output[:samples_to_process] = current_value + increment * (ramp + 1)
                
                # Fill remaining samples if chunk is larger than glide
                if samples_to_process < actual_num_samples:
                    output[samples_to_process:] = chunk_target
                
                current_value = output[-1].copy()
                samples_remaining = max(0, samples_remaining - actual_num_samples)
                
                if samples_remaining == 0:
                    current_value = chunk_target.copy()
        else:
            # Continue existing glide or hold current value
            if samples_remaining > 0:
                # Continue existing glide
                target_value = self.state.target_value.astype(np.float32)
                diff = target_value - current_value
                if samples_remaining >= actual_num_samples:
                    # More samples remaining than this chunk
                    steps_in_chunk = actual_num_samples
                else:
                    # Glide will complete within this chunk
                    steps_in_chunk = samples_remaining
                
                increment = diff / max(1, samples_remaining)
                ramp = np.arange(steps_in_chunk, dtype=np.float32)
                if signal_array.ndim > 1:
                    ramp = ramp[:, np.newaxis]
                
                output[:steps_in_chunk] = current_value + increment * (ramp + 1)
                
                # Fill remaining samples if glide completed
                if steps_in_chunk < actual_num_samples:
                    output[steps_in_chunk:] = target_value
                
                current_value = output[-1].copy()
                samples_remaining = max(0, samples_remaining - actual_num_samples)
                
                if samples_remaining == 0:
                    current_value = target_value.copy()
            else:
                # Hold at current value (no active glide)
                output[:] = current_value

        self.state.current_value = current_value.copy()
        self.state.samples_remaining = samples_remaining

        if mono_input:
            return output.reshape(actual_num_samples)
        return output


GLIDE_DEFINITION = NodeDefinition("glide", GlideNode, GlideModel)
