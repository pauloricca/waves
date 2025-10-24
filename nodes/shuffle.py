from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, wavable_value_node_factory

# Shuffle node: Randomly rearranges chunks of audio to create glitchy effects.
# The crossfade parameter smoothly transitions between chunks to avoid clicks and pops.
# Crossfading reduces the output length by (num_chunks - 1) * crossfade_samples.

class ShuffleModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    size: float = 0.3 # length of chunks to shuffle in seconds
    chunks: int | None = None  # Number of chunks to split the signal into
    invert: float = 0.1 # percentage of the chunk to invert
    crossfade: WavableValue = 0.0  # crossfade time in seconds between chunks
    signal: BaseNodeModel = None
    seed: int | None = None  # Add seed parameter

class ShuffleNode(BaseNode):
    def __init__(self, model: ShuffleModel, state=None, hot_reload=False):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model, state, hot_reload)
        self.model = model
        self.signal_node = instantiate_node(model.signal, hot_reload=hot_reload)
        self.crossfade_node = wavable_value_node_factory(model.crossfade)
        self.rng = np.random.default_rng(model.seed)  # Use a random generator with seed
        
        # State for chunked rendering
        self._shuffled_buffer = None
        self._total_samples_rendered = 0
        self._is_pre_rendered = False

    def _do_render(self, num_samples=None, context=None, **params):
        # If we haven't pre-rendered the full signal yet, do it now
        if not self._is_pre_rendered:
            self._pre_render_full_signal(context, **params)
        
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
    
    def _pre_render_full_signal(self, context, **params):
        """Pre-render the entire signal from the child node and shuffle it"""
        # Request the child to render its entire duration by not specifying num_samples
        signal_wave = self.signal_node.render(context=context, **self.get_params_for_children(params))
        
        # Handle empty signal
        if len(signal_wave) == 0:
            self._shuffled_buffer = np.array([])
            self._is_pre_rendered = True
            return
        
        # Get crossfade time (render a single sample to get the value)
        crossfade_value = self.crossfade_node.render(1, context, **self.get_params_for_children(params))
        crossfade_time = crossfade_value[0] if len(crossfade_value) > 0 else 0.0
        crossfade_samples = int(crossfade_time * SAMPLE_RATE)
        
        # Determine chunk parameters for shuffling
        if self.model.chunks:
            num_chunks = self.model.chunks
            chunk_size = len(signal_wave) // num_chunks
        else:
            chunk_size = int(SAMPLE_RATE * self.model.size)
            num_chunks = len(signal_wave) // chunk_size

        # Ensure crossfade doesn't exceed chunk size
        crossfade_samples = min(crossfade_samples, chunk_size // 2)

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

        # Recombine the shuffled chunks with crossfading
        if crossfade_samples > 0:
            self._shuffled_buffer = self._crossfade_chunks(chunks, crossfade_samples)
        else:
            # No crossfade - just concatenate
            self._shuffled_buffer = np.concatenate(chunks)
        
        self._is_pre_rendered = True
    
    def _crossfade_chunks(self, chunks, crossfade_samples):
        """Crossfade between chunks to avoid clicks and pops.
        
        Each chunk overlaps with the next by crossfade_samples:
        - The end of chunk[i] fades out
        - The beginning of chunk[i+1] fades in
        - They are mixed during the overlap period
        
        The output length is: (len(chunks) * chunk_size) - ((len(chunks) - 1) * crossfade_samples)
        """
        if len(chunks) == 0:
            return np.array([], dtype=np.float32)
        
        if len(chunks) == 1:
            return chunks[0]
        
        chunk_size = len(chunks[0])
        
        # Create fade curves (equal power crossfade for smoother transitions)
        fade_out = np.sqrt(np.linspace(1, 0, crossfade_samples, dtype=np.float32))
        fade_in = np.sqrt(np.linspace(0, 1, crossfade_samples, dtype=np.float32))
        
        # Calculate output size
        output_size = (len(chunks) * chunk_size) - ((len(chunks) - 1) * crossfade_samples)
        result = np.zeros(output_size, dtype=np.float32)
        
        # Process each chunk
        write_pos = 0
        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk)
            
            if i == 0:
                # First chunk: write entirely, no fade-in needed
                result[write_pos:write_pos + chunk_len] = chunk
                write_pos += chunk_len - crossfade_samples
            elif i == len(chunks) - 1:
                # Last chunk: fade in at the start, no fade-out needed
                # The fade-in overlaps with the previous chunk's fade-out
                chunk_faded = chunk.copy()
                chunk_faded[:crossfade_samples] *= fade_in
                
                # Mix the fade-in part with what's already there (previous chunk's fade-out)
                result[write_pos:write_pos + crossfade_samples] += chunk_faded[:crossfade_samples]
                # Write the rest
                result[write_pos + crossfade_samples:write_pos + chunk_len] = chunk[crossfade_samples:]
                write_pos += chunk_len
            else:
                # Middle chunks: fade in at start, fade out at end
                chunk_faded = chunk.copy()
                chunk_faded[:crossfade_samples] *= fade_in
                chunk_faded[-crossfade_samples:] *= fade_out
                
                # Mix the fade-in part
                result[write_pos:write_pos + crossfade_samples] += chunk_faded[:crossfade_samples]
                # Write the middle (no overlap)
                result[write_pos + crossfade_samples:write_pos + chunk_len - crossfade_samples] = \
                    chunk[crossfade_samples:-crossfade_samples]
                # Write the fade-out part (will be mixed with next chunk's fade-in)
                result[write_pos + chunk_len - crossfade_samples:write_pos + chunk_len] = \
                    chunk_faded[-crossfade_samples:]
                
                write_pos += chunk_len - crossfade_samples
        
        return result

SHUFFLE_DEFINITION = NodeDefinition("shuffle", ShuffleNode, ShuffleModel)