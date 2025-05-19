from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, WavableValueNode
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
    def __init__(self, sample_model: SampleModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.model = sample_model
        self.audio = load_wav_file(sample_model.file)
        self.speed_node = WavableValueNode(sample_model.speed)

    def render(self, num_samples, **kwargs):
        start = int(self.model.start * len(self.audio))
        end = int(self.model.end * len(self.audio))
        end = min(end, len(self.audio))
        start = max(start, 0)
        if end <= start:
            return np.zeros(num_samples)

        wave = self.audio[start:end]
        base_len = len(wave)

        speed = self.speed_node.render(num_samples, **kwargs)
        is_modulated = isinstance(speed, np.ndarray) and len(speed) > 1

        if is_modulated:
            # Use absolute speed for rate calculation but keep sign for direction
            abs_speed = np.abs(speed)
            abs_speed = np.maximum(abs_speed, 1e-6)  # avoid zero speed
            sign = np.sign(speed)
            dt = 1.0 / SAMPLE_RATE
            
            # Integrate speed to get playhead position
            playhead = np.cumsum(abs_speed * sign * dt * SAMPLE_RATE)
            
            if self.model.loop:
                # Handle both forward and backward looping
                playhead = np.mod(playhead, base_len)
                # Negative values need to be adjusted to proper position in loop
                playhead = np.where(playhead < 0, playhead + base_len, playhead)
            else:
                # Clip to valid range for non-looping
                playhead = np.clip(playhead, 0, base_len - 1)
            
            wave = np.interp(playhead, np.arange(base_len), wave)
        else:
            speed_value = float(speed[0]) if isinstance(speed, np.ndarray) else float(speed)
            abs_speed = abs(speed_value)
            
            if abs_speed < 1e-6 or np.isnan(speed_value):
                return np.zeros(num_samples)
            
            # For backward playback, reverse the calculation
            if speed_value < 0:
                indices = np.arange(0, num_samples * abs_speed, abs_speed)
                indices = base_len - 1 - indices  # Start from end, move backward
            else:
                indices = np.arange(0, num_samples * abs_speed, abs_speed)
            
            if self.model.loop:
                indices = np.mod(indices, base_len)
            else:
                if speed_value < 0:
                    indices = indices[indices >= 0]  # Keep only valid negative direction indices
                else:
                    indices = indices[indices < base_len]  # Keep only valid positive direction indices
                
            wave = np.interp(indices, np.arange(base_len), wave)

        return wave[:num_samples]

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
