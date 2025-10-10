    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition

class SmoothModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    step: float = 0.01
    signal: BaseNodeModel = None

class SmoothNode(BaseNode):
    def __init__(self, model: SmoothModel):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)

    def render(self, num_samples=None, **params):
        super().render(num_samples)
        
        # If num_samples is None, get the full child signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Need to get full signal from child
                signal_wave = self.render_full_child_signal(self.signal_node, **self.get_params_for_children(params))
                if len(signal_wave) == 0:
                    return np.array([])
                
                # Apply smoothing to the full signal
                return self._apply_smoothing(signal_wave)
        
        signal_wave = self.signal_node.render(num_samples, **self.get_params_for_children(params))
        
        # If signal is done, we're done
        if len(signal_wave) == 0:
            return np.array([], dtype=np.float32)
        
        return self._apply_smoothing(signal_wave)
    
    def _apply_smoothing(self, signal_wave):
        """Apply smoothing to the signal wave"""
        smoothed_wave = np.copy(signal_wave)
        for i in range(1, len(smoothed_wave)):
            diff = smoothed_wave[i] - smoothed_wave[i - 1]
            if abs(diff) > self.model.step:
                smoothed_wave[i] = smoothed_wave[i - 1] + np.sign(diff) * self.model.step
        return smoothed_wave


SMOOTH_DEFINITION = NodeDefinition("smooth", SmoothNode, SmoothModel)
