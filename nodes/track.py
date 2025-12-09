from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from nodes.node_utils.panning import apply_panning
from utils import match_length, empty_mono, empty_stereo, to_mono, is_stereo

"""
Track Node

Applies panning and volume control to signals.
Optimally handles mono/stereo conversion - only outputs stereo when necessary.

Parameters:
- signal: Input signal (mono or stereo)
- pan: Pan position (-1 = left, 0 = center, 1 = right). Can be static or dynamic (WavableValue).
       Uses equal-power panning law for natural stereo imaging.
- volume: Volume multiplier (default: 1.0). Can be static or dynamic (WavableValue).

Output behavior:
- If input is stereo: always outputs stereo (2D array)
- If input is mono and has panning (pan ≠ 0): outputs stereo (2D array) 
- If input is mono and no panning (pan = 0): outputs mono (1D array)

This optimization avoids unnecessary stereo conversion when no panning is applied.

Examples:

# No panning - stays mono for efficiency
track:
  signal:
    osc:
      type: sin
      freq: 440
  volume: 0.7

# With panning - converts mono to stereo
track:
  signal:
    osc:
      type: sin
      freq: 440
  pan: -0.5
  volume: 0.7

# Dynamic panning with an LFO
track:
  signal:
    osc:
      type: sin
      freq: 440
  pan:
    osc:
      type: sin
      freq: 0.5
      range: [-1, 1]
"""

class TrackNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: WavableValue = None
    pan: WavableValue = 0.0  # Pan position: -1 (left) to 1 (right), default center
    volume: WavableValue = 1.0  # Volume multiplier, default unity gain


class TrackNode(BaseNode):
    def __init__(self, model: TrackNodeModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        self.pan_node = self.instantiate_child_node(model.pan, "pan")
        self.volume_node = self.instantiate_child_node(model.volume, "volume")

    def _do_render(self, num_samples=None, context=None, **params):
        """
        Render signal with panning and volume.
        
        Output format:
        - If input is stereo: always output stereo
        - If input is mono and has panning (pan != 0): output stereo  
        - If input is mono and no panning (pan == 0): output mono
        """
        # Render the child signal (may be mono or stereo)
        signal = self.signal_node.render(num_samples, context, **params)
        
        # If signal is empty, return empty in same format as input would be
        if len(signal) == 0:
            return empty_stereo() if is_stereo(signal) else empty_mono()
        
        # Get pan value (static or dynamic)
        pan_value = self.pan_node.render(len(signal), context, **params)
        
        # Ensure pan_value matches signal length (and convert to mono if needed)
        if is_stereo(pan_value):
            pan_value = to_mono(pan_value)
        pan_value = match_length(pan_value, len(signal))
        
        # Get volume value (static or dynamic)
        volume_value = self.volume_node.render(len(signal), context, **params)
        
        # Ensure volume_value matches signal length (and convert to mono if needed)
        if is_stereo(volume_value):
            volume_value = to_mono(volume_value)
        volume_value = match_length(volume_value, len(signal))
        
        # Check if we need panning
        needs_panning = np.any(np.abs(pan_value) > 1e-6)  # Small threshold for floating point comparison
        
        if is_stereo(signal):
            # Input is stereo - apply volume and panning to stereo signal
            if needs_panning:
                # Convert to mono first, then pan back to stereo
                mono_signal = to_mono(signal)
                result = apply_panning(mono_signal, pan_value)
            else:
                # No panning needed, just apply volume to stereo signal
                result = signal.copy()
            
            # Apply volume to both channels
            result *= volume_value[:, np.newaxis]
            
        else:
            # Input is mono
            if needs_panning:
                # Apply panning to create stereo
                result = apply_panning(signal, pan_value)
                # Apply volume to both channels  
                result *= volume_value[:, np.newaxis]
            else:
                # No panning needed - stay mono, just apply volume
                result = signal * volume_value
        
        return result


TRACK_DEFINITION = NodeDefinition("track", TrackNode, TrackNodeModel)
