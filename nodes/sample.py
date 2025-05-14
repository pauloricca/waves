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
        
        if sample_length >= num_samples:
            # If the sample is longer than requested, just return the first segment
            return wave[:num_samples]
        
        # If the sample is shorter than requested
        if self.sample_model.loop:
            # Create the result array
            result = np.zeros(num_samples)
            
            # Get overlap as fraction of sample length
            overlap_samples = int(sample_length * self.sample_model.overlap)
            
            # Calculate effective sample length (considering overlap)
            effective_length = sample_length - overlap_samples if overlap_samples > 0 else sample_length
            
            # Initialize position
            position = 0
            
            while position < num_samples:
                # Calculate how much we can copy in this iteration
                remaining = num_samples - position
                to_copy = min(sample_length, remaining)
                
                # Copy the segment with appropriate amplitude
                if overlap_samples > 0:
                    # Create a copy of the wave segment to modify
                    current_segment = wave[:to_copy].copy()
                    
                    # Apply fade-out to the end portion of the segment
                    if to_copy > overlap_samples:
                        fade_out_start = to_copy - overlap_samples
                        fade_out_window = np.linspace(1.0, 0.0, overlap_samples)
                        current_segment[fade_out_start:to_copy] *= fade_out_window
                        
                    # Apply fade-in if this is not the first segment
                    if position > 0 and overlap_samples > 0 and to_copy > 0:
                        fade_in_length = min(overlap_samples, to_copy)
                        fade_in_window = np.linspace(0.0, 1.0, fade_in_length)
                        current_segment[:fade_in_length] *= fade_in_window
                        
                        # Add the segment to the result (with crossfading)
                        result[position:position + to_copy] += current_segment
                    else:
                        # First segment, just copy
                        result[position:position + to_copy] = current_segment
                else:
                    # No overlap, just copy
                    result[position:position + to_copy] = wave[:to_copy]
                
                # Move position forward by effective length to account for overlap
                position += effective_length
            
            return result
        else:
            # No looping, just pad with zeros
            return np.pad(wave, (0, num_samples - sample_length), 'constant')


SAMPLE_DEFINITION = NodeDefinition("sample", SampleNode, SampleModel)
