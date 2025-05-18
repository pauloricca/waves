from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition
from utils import load_wav_file

class SampleModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    file: str = None  # Path to the sample file
    start: float = 0.0  # A value between 0 and 1 that translates to a position in the sample
    end: float = 1.0  # A value between 0 and 1 that translates to a position in the sample
    loop: bool = False
    overlap: float = 0.0  # Used when looking, 0 to 1 where 1 is the length of the sample between start and end
    speed: float = 1.0  # Speed of playback, 1 is normal speed, 2 is double speed, etc.


class SampleNode(BaseNode):
    def __init__(self, sample_model: SampleModel):
        self.sample_model = sample_model
        self.audio = load_wav_file(sample_model.file)

    def render(self, num_samples, **kwargs):
        start = int(self.sample_model.start * len(self.audio))
        end = int(self.sample_model.end * len(self.audio))
        if end > len(self.audio):
            end = len(self.audio)
        if start < 0:
            start = 0
        if end <= start:
            return np.zeros(num_samples)

        wave = self.audio[start:end]
        sample_length = len(wave)

        # Apply speed: resample the wave to stretch/shrink playback
        if self.sample_model.speed != 1.0 and sample_length > 1:
            # Calculate new indices for resampling
            indices = np.arange(0, sample_length, self.sample_model.speed)
            indices = indices[indices < sample_length]
            wave = np.interp(indices, np.arange(sample_length), wave)
            sample_length = len(wave)

        if sample_length >= num_samples:
            return wave[:num_samples]

        if self.sample_model.loop:
            result = np.zeros(num_samples)
            overlap_samples = int(sample_length * self.sample_model.overlap)
            effective_length = sample_length - overlap_samples if overlap_samples > 0 else sample_length
            position = 0

            while position < num_samples:
                remaining = num_samples - position
                to_copy = min(sample_length, remaining)

                if overlap_samples > 0:
                    current_segment = wave[:to_copy].copy()
                    if to_copy > overlap_samples:
                        fade_out_start = to_copy - overlap_samples
                        fade_out_window = np.linspace(1.0, 0.0, overlap_samples)
                        current_segment[fade_out_start:to_copy] *= fade_out_window
                    if position > 0 and overlap_samples > 0 and to_copy > 0:
                        fade_in_length = min(overlap_samples, to_copy)
                        fade_in_window = np.linspace(0.0, 1.0, fade_in_length)
                        current_segment[:fade_in_length] *= fade_in_window
                        result[position:position + to_copy] += current_segment
                    else:
                        result[position:position + to_copy] = current_segment
                else:
                    result[position:position + to_copy] = wave[:to_copy]

                position += effective_length

            return result
        else:
            return np.pad(wave, (0, num_samples - sample_length), 'constant')


SAMPLE_DEFINITION = NodeDefinition("sample", SampleNode, SampleModel)
