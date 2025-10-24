from __future__ import annotations
from typing import Tuple, Union
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, wavable_value_node_factory

"""
Follow Node (Envelope Follower)

Tracks the amplitude envelope of an input signal and outputs a control signal
that follows the loudness of the input. The output follows the absolute value
of the input with configurable attack (rise) and release (fall) times.

Parameters:
- signal: The input signal to follow (assumed to be in range [-1, 1])
- range: Output range [min, max] (defaults to [0, 1])
- attack: Attack time in seconds - how quickly the follower rises (defaults to 0.01)
- release: Release time in seconds - how quickly the follower falls (defaults to 0.1)

The input signal is clipped to [-1, 1] before calculating the envelope.
The follower outputs a smoothed control signal that tracks the amplitude,
mapped to the specified output range.

Examples:

# Basic envelope follower
follow:
  signal:
    osc:
      type: sin
      freq: 440
  attack: 0.01
  release: 0.1

# Use as modulation source for another parameter
osc:
  type: sin
  freq: 440
  amp:
    follow:
      signal:
        sample:
          file: drums.wav
      range: [0.5, 1.0]
      attack: 0.005
      release: 0.05
"""

class FollowModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: BaseNodeModel = None
    range: Tuple[WavableValue, WavableValue] = (0.0, 1.0)
    attack: Union[float, str] = 0.01  # seconds (or expression)
    release: Union[float, str] = 0.1  # seconds (or expression)


class FollowNode(BaseNode):
    def __init__(self, model: FollowModel, state=None, hot_reload=False):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.state = state if state is not None else type('obj', (object,), {})()
        self.signal_node = instantiate_node(model.signal)
        self.range_min_node = wavable_value_node_factory(model.range[0])
        self.range_max_node = wavable_value_node_factory(model.range[1])
        
        # Persistent state for realtime rendering (survives hot reload)
        if not hot_reload:
            self.state.envelope_value = 0.0  # Current envelope follower value

    def _do_render(self, num_samples=None, context=None, **params):
        # Evaluate expression parameters
        attack = self.eval_scalar(self.model.attack, context, **params)
        release = self.eval_scalar(self.model.release, context, **params)
        
        # Get the input signal
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # If signal is done, we're done
        if len(signal_wave) == 0:
            return np.array([], dtype=np.float32)
        
        actual_num_samples = len(signal_wave)
        
        # Get the output range
        range_min = self.range_min_node.render(actual_num_samples, context, **self.get_params_for_children(params))
        range_max = self.range_max_node.render(actual_num_samples, context, **self.get_params_for_children(params))
        
        # Ensure range values are arrays
        if not isinstance(range_min, np.ndarray):
            range_min = np.full(actual_num_samples, range_min, dtype=np.float32)
        if not isinstance(range_max, np.ndarray):
            range_max = np.full(actual_num_samples, range_max, dtype=np.float32)
        
        # Calculate attack and release coefficients
        # These determine how much of the new value vs old value to use
        # coefficient = 1 - exp(-1 / (time_constant * sample_rate))
        # This gives an exponential smoothing filter
        attack_coeff = 1.0 - np.exp(-1.0 / (attack * SAMPLE_RATE)) if attack > 0 else 1.0
        release_coeff = 1.0 - np.exp(-1.0 / (release * SAMPLE_RATE)) if release > 0 else 1.0
        
        # Clip input signal to [-1, 1] and get absolute value (the envelope)
        clipped_signal = np.clip(signal_wave, -1.0, 1.0)
        target_envelope = np.abs(clipped_signal)
        
        # Apply envelope following with attack/release
        envelope_output = np.zeros(actual_num_samples, dtype=np.float32)
        current_value = self.state.envelope_value
        
        for i in range(actual_num_samples):
            target = target_envelope[i]
            
            # Use attack coefficient if rising, release coefficient if falling
            if target > current_value:
                coeff = attack_coeff
            else:
                coeff = release_coeff
            
            # Exponential smoothing: output = output + coeff * (target - output)
            current_value = current_value + coeff * (target - current_value)
            envelope_output[i] = current_value
        
        # Store the last value for the next chunk
        self.state.envelope_value = current_value
        
        # Map from [0, 1] to the output range
        output_wave = range_min + envelope_output * (range_max - range_min)
        
        return output_wave


FOLLOW_DEFINITION = NodeDefinition("follow", FollowNode, FollowModel)
