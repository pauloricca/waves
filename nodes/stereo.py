"""
Stereo node - converts mono signal to stereo with panning.

Takes a mono signal and applies panning to create stereo output.
The pan parameter can be static or dynamic (WavableValue).

Example:
  stereo:
    pan: -0.5  # Pan 50% left
    signal:
      osc:
        type: sin
        freq: 440

  # Dynamic panning
  stereo:
    pan:
      osc:
        type: sin
        freq: 2
        range: [-1, 1]
    signal:
      osc:
        type: sin
        freq: 440
"""

from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.panning import apply_panning
from nodes.wavable_value import WavableValue
from utils import match_length, empty_mono, empty_stereo


class StereoNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel = None
    pan: WavableValue = 0.0  # Pan position: -1 (left) to 1 (right), default center


class StereoNode(BaseNode):
    def __init__(self, model: StereoNodeModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.is_stereo = True  # StereoNode outputs stereo
        self.model = model
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        self.pan_node = self.instantiate_child_node(model.pan, "pan")

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        """
        Render mono signal with panning to create stereo output.
        
        When num_channels=2: Returns stereo output with panning applied
        When num_channels=1: Returns mono (center-panned mix of stereo)
        """
        # Always render the child signal as mono (num_channels=1)
        # Even if child is stereo-capable, we want mono for panning
        mono_signal = self.signal_node.render(num_samples, context, num_channels=1, **params)
        
        # If signal is empty, return appropriate empty array
        if len(mono_signal) == 0:
            if num_channels == 2:
                return empty_stereo()
            return empty_mono()
        
        # Get pan value (static or dynamic)
        pan_value = self.pan_node.render(len(mono_signal), context, **params)
        
        # Ensure pan_value matches signal length
        pan_value = match_length(pan_value, len(mono_signal))
        
        # Apply panning to create stereo
        stereo_signal = apply_panning(mono_signal, pan_value)
        
        # Return based on requested channels
        if num_channels == 2:
            return stereo_signal
        else:
            # Return mono (average of left and right)
            return (stereo_signal[:, 0] + stereo_signal[:, 1]) / 2


STEREO_DEFINITION = NodeDefinition("stereo", StereoNode, StereoNodeModel)
