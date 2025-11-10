from __future__ import annotations
from threading import Lock
from collections import deque
import numpy as np
import sounddevice as sd
from pydantic import ConfigDict

from config import SAMPLE_RATE, BUFFER_SIZE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono, time_to_samples


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
    device: int | str | None = None  # Input device index or name (None = default)
    amp: WavableValue = 1.0  # Gain/amplitude multiplier


# Global shared audio input stream and buffer
# This ensures we only have one input stream active, shared across all input nodes
_input_stream = None
_input_buffer = None
_input_lock = Lock()
_input_device = None
_stream_active = False


def _audio_input_callback(indata, frames, time_info, status):
    """Callback for sounddevice input stream - captures audio into buffer."""
    global _input_buffer
    if status:
        print(f"Input stream status: {status}")
    if _input_buffer is not None:
        # Copy incoming audio to buffer (already mono)
        _input_buffer.extend(indata[:, 0].copy())


def _start_input_stream(device=None):
    """Start the shared audio input stream if not already running."""
    global _input_stream, _input_buffer, _input_device, _stream_active
    
    with _input_lock:
        # If stream is already running with the same device, nothing to do
        if _stream_active and _input_device == device:
            return
        
        # Stop existing stream if device changed
        if _stream_active:
            _stop_input_stream()
        
        # Initialize buffer
        _input_buffer = deque(maxlen=SAMPLE_RATE * 10)  # 10 second buffer max
        _input_device = device
        
        # Start new input stream
        try:
            _input_stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BUFFER_SIZE,
                callback=_audio_input_callback
            )
            _input_stream.start()
            _stream_active = True
            print(f"Audio input stream started (device: {device if device is not None else 'default'})")
        except Exception as e:
            print(f"Error starting audio input stream: {e}")
            _stream_active = False


def _stop_input_stream():
    """Stop the shared audio input stream."""
    global _input_stream, _input_buffer, _stream_active
    
    with _input_lock:
        if _input_stream is not None:
            _input_stream.stop()
            _input_stream.close()
            _input_stream = None
        _input_buffer = None
        _stream_active = False


class InputNode(BaseNode):
    def __init__(self, model: InputModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        
        # Child nodes
        self.amp_node = self.instantiate_child_node(model.amp, "amp")
        
        # Persistent state (survives hot reload)
        if do_initialise_state:
            self.state.total_samples_rendered = 0
        
        # Start the shared input stream
        _start_input_stream(model.device)
    
    def _do_render(self, num_samples=None, context=None, **params):
        global _input_buffer
        
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
                return empty_mono()
            
            # Limit num_samples to not exceed duration
            remaining = total_duration_samples - self.state.total_samples_rendered
            num_samples = min(num_samples, remaining)
        
        # Get audio from the shared buffer
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        with _input_lock:
            if _input_buffer is not None and len(_input_buffer) > 0:
                # Get as much as we can from the buffer
                available = min(num_samples, len(_input_buffer))
                for i in range(available):
                    audio_data[i] = _input_buffer.popleft()
        
        # Apply amplitude/gain
        amp = self.amp_node.render(num_samples, context, **params)
        audio_data = audio_data * amp
        
        # Update state
        self.state.total_samples_rendered += num_samples
        
        return audio_data


# Node definition for registry
INPUT_DEFINITION = NodeDefinition(
    name="input",
    model=InputModel,
    node=InputNode
)
