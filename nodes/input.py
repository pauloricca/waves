from __future__ import annotations
import numpy as np
from pydantic import ConfigDict

from audio_interfaces import get_audio_input_registry, normalise_channel_mapping
from config import BUFFER_SIZE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono, time_to_samples, match_length, to_mono


class InputModel(BaseNodeModel):
    """
    Captures live audio from an external input (microphone, line in, etc.).
    
    The input node maintains a buffer of incoming audio and serves it during render calls.
    Audio is captured continuously in a background thread when the node is instantiated.
    
    Parameters:
    - device: Optional device index or name (None = default input device)
    - amp: Amplitude/gain multiplier for the input signal
    """
    model_config = ConfigDict(extra='forbid')
    device: int | str | None = None  # Input device alias or index (None = default)
    channels: int | list[int] | tuple[int, ...] | None = 1  # Channel numbers (1-indexed)
    amp: WavableValue = 1.0  # Gain/amplitude multiplier


class InputNode(BaseNode):
    def __init__(self, model: InputModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model

        # Child nodes
        self.amp_node = self.instantiate_child_node(model.amp, "amp")

        self.channel_mapping = normalise_channel_mapping(model.channels, (1,))
        self.channel_count = len(self.channel_mapping)
        self.stream_entry = get_audio_input_registry().get_stream(model.device, self.channel_mapping)

        # Persistent state (survives hot reload)
        if do_initialise_state:
            self.state.total_samples_rendered = 0

    def _do_render(self, num_samples=None, context=None, **params):
        # Resolve num_samples from duration if None
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            # No duration specified - use requested num_samples or buffer size
            num_samples = num_samples or BUFFER_SIZE
        
        # Check if we've reached the end of our duration
        if self.duration is not None:
            total_duration_samples = time_to_samples(self.duration )
            if self.state.total_samples_rendered >= total_duration_samples:
                # We're done, return empty array
                return self._empty_output()
            
            # Limit num_samples to not exceed duration
            remaining = total_duration_samples - self.state.total_samples_rendered
            num_samples = min(num_samples, remaining)
        
        # Get audio from the shared buffer
        audio_data = self.stream_entry.read(num_samples)

        # Apply amplitude/gain
        amp = self.amp_node.render(num_samples, context, **params)
        amp_array = np.asarray(amp, dtype=np.float32)
        if amp_array.ndim > 1:
            amp_array = to_mono(amp_array)

        if amp_array.ndim == 0:
            amp_array = np.full(num_samples, float(amp_array), dtype=np.float32)
        else:
            amp_array = match_length(amp_array, num_samples)

        if self.channel_count == 1:
            audio_data = audio_data * amp_array
        else:
            audio_data = audio_data * amp_array[:, np.newaxis]

        # Update state
        self.state.total_samples_rendered += num_samples

        return audio_data

    def _empty_output(self):
        if self.channel_count == 1:
            return empty_mono()
        return np.zeros((0, self.channel_count), dtype=np.float32)


# Node definition for registry
INPUT_DEFINITION = NodeDefinition(
    name="input",
    model=InputModel,
    node=InputNode
)
