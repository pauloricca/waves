from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono


# Context node: Creates a scope where additional variables are added to render params.
# Takes a signal plus any number of named WavableValue arguments. These arguments are
# rendered first and added as variables to the render params before rendering the signal.
class ContextNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='allow')  # Allow arbitrary named parameters
    signal: WavableValue
    is_pass_through: bool = True


class ContextNode(BaseNode):
    def __init__(self, model: ContextNodeModel, node_id: str, state, do_initialise_state=True):
        from nodes.wavable_value import WavableValueNode, WavableValueModel
        super().__init__(model, node_id, state, do_initialise_state)
        self.is_stereo = True  # Context is a pass-through node, supports stereo
        self.model = model
        
        # Instantiate the signal node
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        
        # Store all extra arguments as WavableValue nodes
        self.context_args = {}
        if hasattr(model, '__pydantic_extra__') and model.__pydantic_extra__:
            for field_name, field_value in model.__pydantic_extra__.items():
                self.context_args[field_name] = self.instantiate_child_node(field_value, field_name)
        
        # Track total samples rendered for duration checking
        if do_initialise_state:
            self.state.total_samples_rendered = 0
    
    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Resolve num_samples first (handles None case)
        num_samples = self.resolve_num_samples(num_samples)
        if num_samples is None:
            raise ValueError("Context node requires explicit duration")
        
        # Check if we've already rendered the full duration
        if self.duration is not None:
            from config import SAMPLE_RATE
            max_samples = int(self.duration * SAMPLE_RATE)
            
            # If we've already rendered everything, return empty
            if self.state.total_samples_rendered >= max_samples:
                return empty_mono()
            
            # If num_samples would exceed duration, limit it
            remaining_samples = max_samples - self.state.total_samples_rendered
            if remaining_samples <= 0:
                return empty_mono()
            num_samples = min(num_samples, remaining_samples)
        
        # Render all context arguments and add to params
        extended_params = params.copy()
        
        for name, node in self.context_args.items():
            # Pass num_channels to context args so they match the expected output format
            wave = node.render(num_samples, context, num_channels, **self.get_params_for_children(params))
            # If child returned empty, we're done
            if len(wave) == 0:
                return empty_mono()
            extended_params[name] = wave
        
        # Now render the signal with the extended params
        result = self.signal_node.render(num_samples, context, num_channels, **extended_params)
        
        # Track samples rendered
        self.state.total_samples_rendered += len(result)
        
        return result


CONTEXT_DEFINITION = NodeDefinition("context", ContextNode, ContextNodeModel)
