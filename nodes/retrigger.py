    
from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from utils import add_waves

class RetriggerModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    time: float = 0.1
    repeats: int = 3
    feedback: float = 0.3
    signal: BaseNodeModel = None

class RetriggerNode(BaseNode):
    def __init__(self, model: RetriggerModel, state, hot_reload=False):
        from nodes.node_utils.instantiate_node import instantiate_node
        super().__init__(model)
        self.model = model
        self.signal_node = instantiate_node(model.signal)
        self.state = state
        
        # Persistent state for carry over samples (survives hot reload)
        if not hot_reload:
            self.state.carry_over = np.array([], dtype=np.float32)

    def _do_render(self, num_samples=None, context=None, **params):
        # If num_samples is None, we need to render the full signal
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # For retrigger nodes, we need to get the full child signal first
                child_signal = self.render_full_child_signal(self.signal_node, context, **self.get_params_for_children(params))
                if len(child_signal) == 0:
                    return np.array([])
                
                # Calculate total retrigger time and set num_samples accordingly
                n_delay_time_samples = int(SAMPLE_RATE * self.model.time)
                total_length = len(child_signal) + n_delay_time_samples * self.model.repeats
                self._last_chunk_samples = total_length
                
                # Process the full signal at once
                delayed_wave = np.zeros(total_length)
                for i in range(self.model.repeats):
                    delayed_wave[i * n_delay_time_samples : i * n_delay_time_samples + len(child_signal)] += child_signal * (self.model.feedback ** i)
                return delayed_wave
        
        signal_wave = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # If signal is done and we have no carry over, we're done
        if len(signal_wave) == 0 and len(self.state.carry_over) == 0:
            return np.array([], dtype=np.float32)
        
        n_delay_time_samples = int(SAMPLE_RATE * self.model.time)
        delayed_wave = np.zeros(len(signal_wave) + n_delay_time_samples * self.model.repeats)

        # Add delays (only if we have signal)
        if len(signal_wave) > 0:
            for i in range(self.model.repeats):
                delayed_wave[i * n_delay_time_samples : i * n_delay_time_samples + len(signal_wave)] += signal_wave * (self.model.feedback ** i)
        
        # Add carry over from previous render
        if len(self.state.carry_over) > 0:
            delayed_wave = add_waves(delayed_wave, self.state.carry_over[:len(delayed_wave)])

        # Return n_samples and save the rest as carry over for the next render
        part_to_return = delayed_wave[:num_samples]
        self.state.carry_over = delayed_wave[num_samples:]
        
        return part_to_return

RETRIGGER_DEFINITION = NodeDefinition("retrigger", RetriggerNode, RetriggerModel)
