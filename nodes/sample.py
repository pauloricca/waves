from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from constants import RenderArgs
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.oscillator import OSCILLATOR_RENDER_ARGS
from nodes.wavable_value import WavableValue, wavable_value_node_factory
from utils import load_wav_file


# Sample node with offset parameter:
# offset allows you to shift the playhead position by a certain amount (in samples).
# This can be used with an LFO to create warble, tape slow-down/speed-up effects, etc.
# Positive values shift forward, negative values shift backward.

class SampleModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    file: str = None
    start: WavableValue = 0.0  # 0.0-1.0 (normalized position in sample)
    end: WavableValue = 1.0  # 0.0-1.0 (normalized position in sample)
    loop: bool = False
    overlap: float = 0.0  # 0.0-1.0 (normalized fraction of sample length)
    speed: WavableValue = 1.0  # playback speed multiplier (1.0 = normal speed)
    duration: float = None  # seconds
    base_freq: float = None  # Hz (for frequency-based speed modulation)
    offset: WavableValue = 0.0  # samples (shift playhead position)


class SampleNode(BaseNode):
    def __init__(self, model: SampleModel):
        super().__init__(model)
        self.model = model
        self.audio = load_wav_file(model.file)
        self.speed_node = wavable_value_node_factory(model.speed)
        self.start_node = wavable_value_node_factory(model.start)
        self.end_node = wavable_value_node_factory(model.end)
        self.offset_node = wavable_value_node_factory(model.offset)
        self.last_playhead_position = 0
        self.total_samples_rendered = 0

    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
        # If num_samples is None, resolve it from duration
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Render start/end for a single sample to get their initial values
                start_vals = self.start_node.render(1, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
                end_vals = self.end_node.render(1, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
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
        start_vals = self.start_node.render(num_samples, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
        end_vals = self.end_node.render(num_samples, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
        
        # Ensure they are arrays
        if np.isscalar(start_vals):
            start_vals = np.full(num_samples, start_vals)
        if np.isscalar(end_vals):
            end_vals = np.full(num_samples, end_vals)
        
        # Convert to sample indices
        start_indices = np.clip(start_vals * len(self.audio), 0, len(self.audio) - 1).astype(int)
        end_indices = np.clip(end_vals * len(self.audio), 0, len(self.audio)).astype(int)
        
        # Initialize playhead to start position on first render
        if self.number_of_chunks_rendered == 0 and self.last_playhead_position == 0:
            self.last_playhead_position = float(start_indices[0])
        
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
            frequency = params.get(RenderArgs.FREQUENCY, None)
            if frequency is not None:
                # Ensure frequency is an array
                if np.isscalar(frequency):
                    frequency = np.full(num_samples, frequency)
                # Multiply speed by the frequency ratio (current_freq / base_freq)
                speed = speed * (frequency / self.model.base_freq)

        # Render offset
        offset = self.offset_node.render(num_samples, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))
        
        # Ensure offset is an array
        if np.isscalar(offset):
            offset = np.full(num_samples, offset)

        # Use absolute speed for rate calculation but keep sign for direction
        abs_speed = np.abs(speed)
        abs_speed = np.maximum(abs_speed, 1e-6)  # avoid zero speed
        sign = np.sign(speed)
        dt = 1.0 / SAMPLE_RATE
        
        # Calculate the window lengths for each sample
        window_lengths = end_indices - start_indices
        window_lengths = np.maximum(window_lengths, 1)  # Ensure at least 1 sample
        
        # Integrate speed to get absolute playhead position in the audio buffer
        # The playhead moves through the actual audio buffer, not relative to the window
        playhead_delta = abs_speed * sign * dt * SAMPLE_RATE
        playhead_absolute = self.last_playhead_position + np.cumsum(playhead_delta)
        
        # Apply offset (in samples) - this is applied as a displacement, not accumulated
        # Save the unmodified playhead for updating last_playhead_position
        playhead_with_offset = playhead_absolute + offset
        
        if not self.model.loop:
            # Non-looping: find where we exceed the window bounds
            outside_bounds = (playhead_with_offset >= end_indices) | (playhead_with_offset < start_indices)
            if np.any(outside_bounds):
                # Find first sample that's outside bounds
                first_outside = np.where(outside_bounds)[0][0]
                if first_outside == 0:
                    # Already outside on first sample, we're done
                    return np.array([], dtype=np.float32)
                # Truncate to just before going outside
                num_samples = first_outside
                playhead_with_offset = playhead_with_offset[:num_samples]
                playhead_absolute = playhead_absolute[:num_samples]
                start_indices = start_indices[:num_samples]
                end_indices = end_indices[:num_samples]
            
            audio_indices = playhead_with_offset
        else:
            # Looping: wrap around when outside the window
            # Check which samples are outside their window
            outside_high = playhead_with_offset >= end_indices
            outside_low = playhead_with_offset < start_indices
            outside = outside_high | outside_low
            
            if np.any(outside):
                # For samples outside the window, wrap them back
                # Calculate offset from start of window
                offset_from_start = playhead_with_offset - start_indices
                # Wrap within window length
                wrapped_offset = np.mod(offset_from_start, window_lengths)
                # Update playhead for wrapped samples
                playhead_with_offset = np.where(outside, start_indices + wrapped_offset, playhead_with_offset)
            
            audio_indices = playhead_with_offset
        
        # Handle the case where we truncated early (non-looping)
        if num_samples == 0:
            return np.array([], dtype=np.float32)
        
        # Clip to valid audio buffer range
        audio_indices = np.clip(audio_indices, 0, len(self.audio) - 1)
        
        # Interpolate from the audio buffer
        wave = np.interp(audio_indices, np.arange(len(self.audio)), self.audio)

        self.last_playhead_position = playhead_absolute[-1] if len(playhead_absolute) > 0 else self.last_playhead_position
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
