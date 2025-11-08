"""
Mix Node - Combines multiple input signals together

This node acts like a mixer, combining the outputs of multiple signal sources.
Each signal is rendered and added together to create the final mixed output.

Example usage in YAML:
    mixed_sound:
      mix:
        signals:
          - osc:
              type: sin
              freq: 440
              amp: 0.5
          - osc:
              type: sin
              freq: 550
              amp: 0.5
          - retrigger:
              time: 0.1
              repeats: 3
              signal:
                osc:
                  type: tri
                  freq: 220

The mix node supports both realtime and non-realtime rendering modes.
When signals have different durations, the mix continues until all signals
have completed.
"""

from __future__ import annotations
import numpy as np
from typing import List
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import add_waves, empty_mono


class MixModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signals: List[WavableValue] = []


class MixNode(BaseNode):
    def __init__(self, model: MixModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.is_stereo = True  # Mix node supports stereo - can mix stereo and mono signals
        self.model = model
        self.signal_nodes = [self.instantiate_child_node(signal, "signals", i) for i, signal in enumerate(model.signals)]
        
        # Track total samples rendered for duration checking
        if do_initialise_state:
            self.state.total_samples_rendered = 0

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Check if we've already rendered the full duration
        max_samples = None
        if self.duration is not None:
            from config import SAMPLE_RATE
            max_samples = int(self.duration * SAMPLE_RATE)
            
            # If we've already rendered everything, return empty
            if self.state.total_samples_rendered >= max_samples:
                return empty_mono()
            
            # If num_samples would exceed duration, limit it
            if num_samples is not None:
                remaining_samples = max_samples - self.state.total_samples_rendered
                if remaining_samples <= 0:
                    return empty_mono()
                num_samples = min(num_samples, remaining_samples)
        
        # If num_samples is None, we need to render the full signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # For mix nodes, we need to get all child signals and combine them
                mixed_wave = np.array([], dtype=np.float32)
                
                for signal_node in self.signal_nodes:
                    child_signal = self.render_full_child_signal(signal_node, context, num_channels, **self.get_params_for_children(params))
                    if len(child_signal) > 0:
                        mixed_wave = add_waves(mixed_wave, child_signal)
                
                self._last_chunk_samples = len(mixed_wave)
                self.state.total_samples_rendered += len(mixed_wave)
                return mixed_wave
        
        # Realtime rendering: render each signal for num_samples and combine
        mixed_wave = None
        any_signal_active = False
        
        for signal_node in self.signal_nodes:
            signal_wave = signal_node.render(num_samples, context, num_channels, **self.get_params_for_children(params))
            if len(signal_wave) > 0:
                any_signal_active = True
                # Add this signal to the mix
                if mixed_wave is None:
                    mixed_wave = signal_wave.copy()
                else:
                    mixed_wave = add_waves(mixed_wave, signal_wave)
        
        # If no signals are active, return empty array to signal completion
        if not any_signal_active:
            from utils import empty_stereo
            return empty_stereo() if num_channels == 2 else empty_mono()
        
        # Track how many samples we've rendered (mixed_wave should not be None here)
        if mixed_wave is not None:
            samples_rendered = len(mixed_wave)
            self.state.total_samples_rendered += samples_rendered
        
        return mixed_wave if mixed_wave is not None else np.zeros(num_samples, dtype=np.float32)


MIX_DEFINITION = NodeDefinition("mix", MixNode, MixModel)
