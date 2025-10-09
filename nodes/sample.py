from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.oscillator import OSCILLATOR_RENDER_ARGS
from nodes.wavable_value import WavableValue, wavable_value_node_factory
from utils import load_wav_file


class SampleModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    file: str = None
    start: float = 0.0
    end: float = 1.0
    loop: bool = False
    overlap: float = 0.0
    speed: WavableValue = 1.0
    duration: float = None


class SampleNode(BaseNode):
    def __init__(self, model: SampleModel):
        super().__init__(model)
        self.model = model
        self.audio = load_wav_file(model.file)
        self.speed_node = wavable_value_node_factory(model.speed)
        self.last_playhead_position = 0

    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
        # If num_samples is None, render the entire sample
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Calculate the full length of the sample based on start/end
                start = int(self.model.start * len(self.audio))
                end = int(self.model.end * len(self.audio))
                end = min(end, len(self.audio))
                start = max(start, 0)
                sample_length = end - start
                
                if sample_length <= 0:
                    return np.array([])
                
                # For non-looping samples, length is just the sample length
                # For looping samples, we need a duration to be specified
                if not self.model.loop:
                    num_samples = sample_length
                    self._last_chunk_samples = num_samples
                else:
                    raise ValueError("Cannot render full signal: looping sample has no duration specified")
        
        start = int(self.model.start * len(self.audio))
        end = int(self.model.end * len(self.audio))
        end = min(end, len(self.audio))
        start = max(start, 0)
        if end <= start:
            return np.zeros(num_samples)

        wave = self.audio[start:end]
        base_len = len(wave)

        speed = self.speed_node.render(num_samples, **self.get_params_for_children(params, OSCILLATOR_RENDER_ARGS))

        # Use absolute speed for rate calculation but keep sign for direction
        abs_speed = np.abs(speed)
        abs_speed = np.maximum(abs_speed, 1e-6)  # avoid zero speed
        sign = np.sign(speed)
        dt = 1.0 / SAMPLE_RATE
        
        # Integrate speed to get playhead position
        playhead = self.last_playhead_position + np.cumsum(abs_speed * sign * dt * SAMPLE_RATE)
        
        # Check if we've reached the end (only if not looping)
        if not self.model.loop:
            # Check if playhead has gone past the end of the sample
            if self.last_playhead_position >= base_len - 1:
                # We're done
                return np.array([], dtype=np.float32)
            
            # Check if playhead exceeds end during this chunk
            playhead = np.clip(playhead, 0, base_len - 1)
            # Find where playhead reaches the end
            end_indices = np.where(playhead >= base_len - 1)[0]
            if len(end_indices) > 0:
                # Return only up to the end
                truncate_at = end_indices[0] + 1
                playhead = playhead[:truncate_at]
                num_samples = truncate_at
        else:
            playhead = np.mod(playhead, base_len)
            # Negative values need to be adjusted to proper position in loop
            playhead = np.where(playhead < 0, playhead + base_len, playhead)
            playhead = np.clip(playhead, 0, base_len - 1)
    
        wave = np.interp(playhead, np.arange(base_len), wave)

        self.last_playhead_position = playhead[-1]

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
