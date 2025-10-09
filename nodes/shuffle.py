from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition

class ShuffleModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    size: float = 0.3 # length of chunks to shuffle in seconds
    chunks: int | None = None  # Number of chunks to split the signal into
    invert: float = 0.1 # percentage of the chunk to invert
    signal: BaseNodeModel = None
    seed: int | None = None  # Add seed parameter

class ShuffleNode(BaseNode):
    def __init__(self, model: ShuffleModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)
        self.rng = np.random.default_rng(model.seed)  # Use a random generator with seed
        
        # State for chunked rendering
        self._shuffled_buffer = None
        self._total_samples_rendered = 0
        self._is_pre_rendered = False

    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
        # If we haven't pre-rendered the full signal yet, do it now
        if not self._is_pre_rendered:
            self._pre_render_full_signal(**params)
        
        # If num_samples is None, return the entire shuffled buffer
        if num_samples is None:
            self._last_chunk_samples = len(self._shuffled_buffer)
            return self._shuffled_buffer.copy()
        
        # Return the appropriate chunk from the pre-shuffled buffer
        start_idx = self._total_samples_rendered
        end_idx = start_idx + num_samples
        
        # Handle case where we're asked for more samples than available
        if start_idx >= len(self._shuffled_buffer):
            # We're past the end - signal completion by returning empty array
            return np.array([], dtype=np.float32)
        elif end_idx > len(self._shuffled_buffer):
            # Partial chunk at the end
            available_samples = len(self._shuffled_buffer) - start_idx
            result = self._shuffled_buffer[start_idx:start_idx + available_samples]
            self._total_samples_rendered += available_samples
        else:
            # Normal case - return the requested chunk
            result = self._shuffled_buffer[start_idx:end_idx]
            self._total_samples_rendered += num_samples
        
        return result
    
    def _pre_render_full_signal(self, **params):
        """Pre-render the entire signal from the child node and shuffle it"""
        # Request the child to render its entire duration by not specifying num_samples
        signal_wave = self.signal_node.render(**self.get_params_for_children(params))
        
        # Handle empty signal
        if len(signal_wave) == 0:
            self._shuffled_buffer = np.array([])
            self._is_pre_rendered = True
            return
        
        # Determine chunk parameters for shuffling
        if self.model.chunks:
            num_chunks = self.model.chunks
            chunk_size = len(signal_wave) // num_chunks
        else:
            chunk_size = int(SAMPLE_RATE * self.model.size)
            num_chunks = len(signal_wave) // chunk_size

        # Split the wave into chunks
        chunks = [signal_wave[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

        # Wave too small to shuffle, store the original wave
        if len(chunks) < 2:
            self._shuffled_buffer = signal_wave
            self._is_pre_rendered = True
            return

        # Shuffle the chunks
        self.rng.shuffle(chunks)
        
        # Invert some chunks if specified
        num_chunks_to_invert = int(self.model.invert * num_chunks)
        if num_chunks_to_invert > 0:
            chunks_to_invert = self.rng.choice(num_chunks, num_chunks_to_invert, replace=False)
            for idx in chunks_to_invert:
                chunks[idx] = chunks[idx][::-1]  # Invert the chunk

        # Recombine the shuffled chunks
        self._shuffled_buffer = np.concatenate(chunks)
        self._is_pre_rendered = True

SHUFFLE_DEFINITION = NodeDefinition("shuffle", ShuffleNode, ShuffleModel)