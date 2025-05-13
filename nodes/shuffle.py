    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from models.models import BaseNodeModel
from nodes.node_utils.base import BaseNode
from nodes.node_utils.node_definition_type import NodeDefinition

class ShuffleModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    size: float = 0.3 # length of chunks to shuffle in seconds
    invert: float = 0.1 # percentage of the chunk to invert
    signal: BaseNodeModel = None

class ShuffleNode(BaseNode):
    def __init__(self, shuffle_model: ShuffleModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        self.shuffle_model = shuffle_model
        self.signal_node = instantiate_node(shuffle_model.signal)

    def render(self, num_samples, **kwargs):
        wave = self.signal_node.render(num_samples, **kwargs)
        chunk_size = int(SAMPLE_RATE * self.shuffle_model.size)
        num_chunks = len(wave) // chunk_size

                # Split the wave into chunks
        chunks = [wave[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

        # Wave to small to shuffle, return the original wave
        if len(chunks) < 2:
            return wave

        # Shuffle the chunks
        np.random.shuffle(chunks)
        
        num_chunks_to_invert = int(self.shuffle_model.invert * num_chunks)

        chunks_to_invert = np.random.choice(num_chunks, num_chunks_to_invert, replace=False)

        for idx in chunks_to_invert:
            chunks[idx] = chunks[idx][::-1]  # Invert the chunk

        # Recombine the shuffled chunks
        combined_wave_shuffled = np.concatenate(chunks)

        return combined_wave_shuffled


SHUFFLE_DEFINITION = NodeDefinition("shuffle", ShuffleNode, ShuffleModel)
