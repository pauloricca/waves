from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, wavable_value_node_factory
from utils import load_wav_file


class SampleModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    file: str = None
    start: WavableValue = 0.0
    end: WavableValue = 1.0
    loop: bool = False
    overlap: float = 0.0
    speed: WavableValue = 1.0
    duration: float = None
    base_freq: float = None


class SampleNode(BaseNode):
    def __init__(self, model: SampleModel):
        super().__init__(model)
        self.model = model
        self.audio = load_wav_file(model.file)
        self.speed_node = wavable_value_node_factory(model.speed)
        self.start_node = wavable_value_node_factory(model.start)
        self.end_node = wavable_value_node_factory(model.end)
        self.last_playhead_position = 0
        self.total_samples_rendered = 0

    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
        # If num_samples is None, resolve it from duration
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Render start/end for a single sample to get their initial values
                start_vals = self.start_node.render(1, **self.get_params_for_children(params))
                end_vals = self.end_node.render(1, **self.get_params_for_children(params))
                start = int(start_vals[0] * len(self.audio))
                end = int(end_vals[0] * len(self.audio))
                end = min(end, len(self.audio))
                start = max(start, 0)
                sample_length = end - start
                
                if sample_length <= 0:
                    return np.array([])
                
                # For non-looping samples, length is just the sample length
                if not self.model.loop:
                    num_samples = sample_length
                    self._last_chunk_samples = num_samples
                else:
                    raise ValueError("Cannot render full signal: looping sample requires duration to be specified")
        
        # Render start and end values for this chunk
        start_vals = self.start_node.render(num_samples, **self.get_params_for_children(params))
        end_vals = self.end_node.render(num_samples, **self.get_params_for_children(params))
        
        # Ensure they are arrays
        if np.isscalar(start_vals):
            start_vals = np.full(num_samples, start_vals)
        if np.isscalar(end_vals):
            end_vals = np.full(num_samples, end_vals)
        
        # Convert to sample indices
        start_indices = np.clip(start_vals * len(self.audio), 0, len(self.audio) - 1).astype(int)
        end_indices = np.clip(end_vals * len(self.audio), 0, len(self.audio)).astype(int)
        
        # Check if all start/end pairs are valid
        if np.any(end_indices <= start_indices):
            # For simplicity, when ranges are invalid, return zeros for those samples
            # In a more sophisticated implementation, we could handle this per-sample
            pass

        # Check if we have a duration limit and if we've exceeded it
        if self.model.duration is not None:
            max_total_samples = int(self.model.duration * SAMPLE_RATE)
            if self.total_samples_rendered >= max_total_samples:
                # Already rendered full duration
                return np.array([], dtype=np.float32)
            
            # Check if this chunk would exceed the duration
            samples_remaining = max_total_samples - self.total_samples_rendered
            if num_samples > samples_remaining:
                num_samples = samples_remaining
                # Truncate start/end arrays as well
                start_indices = start_indices[:num_samples]
                end_indices = end_indices[:num_samples]

        speed = self.speed_node.render(num_samples, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
        
        # Ensure speed is an array
        if np.isscalar(speed):
            speed = np.full(num_samples, speed)
        
        # If base_freq is set, modulate speed by frequency parameter
        if self.model.base_freq is not None:
            frequency = params.get('frequency', None)
            if frequency is not None:
                # Ensure frequency is an array
                if np.isscalar(frequency):
                    frequency = np.full(num_samples, frequency)
                # Multiply speed by the frequency ratio (current_freq / base_freq)
                speed = speed * (frequency / self.model.base_freq)

        # Use absolute speed for rate calculation but keep sign for direction
        abs_speed = np.abs(speed)
        abs_speed = np.maximum(abs_speed, 1e-6)  # avoid zero speed
        sign = np.sign(speed)
        dt = 1.0 / SAMPLE_RATE
        
        # Calculate the window lengths for each sample
        window_lengths = end_indices - start_indices
        window_lengths = np.maximum(window_lengths, 1)  # Ensure at least 1 sample
        
        # Integrate speed to get playhead position (in samples within the window)
        playhead_relative = self.last_playhead_position + np.cumsum(abs_speed * sign * dt * SAMPLE_RATE)
        
        # Check if we've reached the end (only if not looping)
        if not self.model.loop:
            # Check if playhead has gone past the end of the first window
            if self.last_playhead_position >= window_lengths[0] - 1:
                # We're done
                return np.array([], dtype=np.float32)
            
            # Check if playhead exceeds end during this chunk
            # Since window_lengths can change, we need to check against each window
            playhead_relative = np.minimum(playhead_relative, window_lengths - 1)
            playhead_relative = np.maximum(playhead_relative, 0)
            
            # Find where playhead reaches the end of its respective window
            end_idx = np.where(playhead_relative >= window_lengths - 1)[0]
            if len(end_idx) > 0:
                # Return only up to the end
                truncate_at = end_idx[0] + 1
                playhead_relative = playhead_relative[:truncate_at]
                start_indices = start_indices[:truncate_at]
                end_indices = end_indices[:truncate_at]
                window_lengths = window_lengths[:truncate_at]
                num_samples = truncate_at
        else:
            # For looping, wrap playhead within each window
            playhead_relative = np.mod(playhead_relative, window_lengths)
            # Negative values need to be adjusted to proper position in loop
            playhead_relative = np.where(playhead_relative < 0, playhead_relative + window_lengths, playhead_relative)
            playhead_relative = np.clip(playhead_relative, 0, window_lengths - 1)
        
        # Convert relative playhead positions to actual audio buffer indices
        audio_indices = start_indices + playhead_relative
        audio_indices = np.clip(audio_indices, 0, len(self.audio) - 1)
        
        # Interpolate from the audio buffer
        wave = np.interp(audio_indices, np.arange(len(self.audio)), self.audio)

        self.last_playhead_position = playhead_relative[-1]
        self.total_samples_rendered += len(wave)

        return wave

        # Uncomment the following lines if you want to implement looping with overlap, but this doesn't work with variable speed
        # if self.model.loop:
        #     result = np.zeros(num_samples)
        #     overlap_samples = int(final_len * self.model.overlap)
        #     effective_length = final_len - overlap_samples if overlap_samples > 0 else final_len
        #     position = 0

        #     while position < num_samples:
        #         remaining = num_samples - position
        #         to_copy = min(final_len, remaining)

        #         segment = wave[:to_copy].copy()
        #         if overlap_samples > 0 and to_copy > overlap_samples:
        #             fade_out = np.linspace(1, 0, overlap_samples)
        #             fade_in = np.linspace(0, 1, overlap_samples)
        #             segment[-overlap_samples:] *= fade_out
        #             if position > 0:
        #                 result[position:position+overlap_samples] += wave[:overlap_samples] * fade_in
        #         result[position:position+to_copy] += segment
        #         position += effective_length
        #     return result
        # else:
        #     return np.pad(wave, (0, num_samples - final_len), 'constant')


SAMPLE_DEFINITION = NodeDefinition("sample", SampleNode, SampleModel)
