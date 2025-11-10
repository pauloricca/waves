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

Converts a mono signal to stereo with panning and volume control.
This is the fundamental stereo signal generator in the system.

Parameters:
- signal: Input mono signal
- pan: Pan position (-1 = left, 0 = center, 1 = right). Can be static or dynamic (WavableValue).
       Uses equal-power panning law for natural stereo imaging.
- volume: Volume multiplier (default: 1.0). Can be static or dynamic (WavableValue).

Output: 2D array of shape (num_samples, 2) with [left, right] channels.

Examples:

# Pan a sine wave to the left with reduced volume
track:
  signal:
    osc:
      type: sin
      freq: 440
  pan: -1
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

# Dynamic volume envelope
track:
  signal:
    sample:
      file: kick.wav
  volume:
    envelope:
      attack: 0.01
      release: 0.3
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
        Render signal with panning and volume to create stereo output.
        Always outputs stereo (2D array).
        """
        # Render the child signal (may be mono or stereo)
        signal = self.signal_node.render(num_samples, context, **params)
        
        # If signal is empty, return empty stereo
        if len(signal) == 0:
            return empty_stereo()
        
        # Convert to mono if stereo (panning works on mono signals)
        if is_stereo(signal):
            signal = to_mono(signal)
        
        # Get pan value (static or dynamic)
        pan_value = self.pan_node.render(len(signal), context, **params)
        
        # Ensure pan_value matches signal length (and convert to mono if needed)
        if is_stereo(pan_value):
            pan_value = to_mono(pan_value)
        pan_value = match_length(pan_value, len(signal))
        
        # Apply panning to create stereo
        stereo_signal = apply_panning(signal, pan_value)
        
        # Get volume value (static or dynamic)
        volume_value = self.volume_node.render(len(signal), context, **params)
        
        # Ensure volume_value matches signal length (and convert to mono if needed)
        if is_stereo(volume_value):
            volume_value = to_mono(volume_value)
        volume_value = match_length(volume_value, len(signal))
        
        # Apply volume to both channels
        stereo_signal *= volume_value[:, np.newaxis]
        
        return stereo_signal


TRACK_DEFINITION = NodeDefinition("track", TrackNode, TrackNodeModel)
