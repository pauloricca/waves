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
from utils import add_waves


class MixModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signals: List[BaseNodeModel] = []


class MixNode(BaseNode):
    def __init__(self, model: MixModel, node_id: str, state=None, hot_reload=False):
        super().__init__(model, node_id, state, hot_reload)
        self.model = model
        self.signal_nodes = [self.instantiate_child_node(signal, "signals", i) for i, signal in enumerate(model.signals)]

    def _do_render(self, num_samples=None, context=None, **params):
        # If num_samples is None, we need to render the full signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # For mix nodes, we need to get all child signals and combine them
                mixed_wave = np.array([], dtype=np.float32)
                
                for signal_node in self.signal_nodes:
                    child_signal = self.render_full_child_signal(signal_node, context, **self.get_params_for_children(params))
                    if len(child_signal) > 0:
                        mixed_wave = add_waves(mixed_wave, child_signal)
                
                self._last_chunk_samples = len(mixed_wave)
                return mixed_wave
        
        # Realtime rendering: render each signal for num_samples and combine
        mixed_wave = None
        any_signal_active = False
        
        for signal_node in self.signal_nodes:
            signal_wave = signal_node.render(num_samples, context, **self.get_params_for_children(params))
            if len(signal_wave) > 0:
                any_signal_active = True
                # Add this signal to the mix
                if mixed_wave is None:
                    mixed_wave = signal_wave.copy()
                else:
                    mixed_wave = add_waves(mixed_wave, signal_wave)
        
        # If no signals are active, return empty array to signal completion
        if not any_signal_active:
            return np.array([], dtype=np.float32)
        
        return mixed_wave


MIX_DEFINITION = NodeDefinition("mix", MixNode, MixModel)
